"""Analyze a native SGLang benchmark and opt-in batch CUDA timing traces.

The residual wall time is an estimate of decode-only service, not a measured PD
decode worker. Never subtract summed request TTFT: requests share prefill batches
and TTFT also includes queueing, tokenization, and transport.
"""

import argparse
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def _validate_session(session):
    if session.get("type") != "session" or session.get("schema_version") != 1:
        raise ValueError("Expected a version 1 CUDA trace session")
    for name in ("tp_size", "pp_size", "dp_size"):
        if session.get(name) != 1:
            raise ValueError(f"Decode-only subtraction requires {name}=1")
    if session.get("disaggregation_mode") != "null":
        raise ValueError("Decode-only subtraction requires NoPD serving")
    for name in ("enable_mixed_chunk", "enable_two_batch_overlap"):
        if session.get(name) is not False:
            raise ValueError(f"Decode-only subtraction does not support {name}")
    if session.get("speculative_algorithm") is not None:
        raise ValueError("Decode-only subtraction does not support speculation")
    if session.get("disable_radix_cache") is not True:
        raise ValueError("Decode-only accounting requires disabled radix cache")
    if not isinstance(session.get("disable_overlap_schedule"), bool):
        raise ValueError("Trace must declare the overlap scheduler setting")


def _analyze_single(result, traces):
    """Return E2E metrics and a separately labelled prefill-subtraction estimate.

    ``traces`` is a list of complete parsed JSONL traces. Exactly one scheduler
    session must contain the benchmark request IDs; other sessions are ignored.
    Missing chunks, duplicate events, partial traces, mixed workloads/streams,
    prefix-cache hits and recomputations fail closed rather than inflate throughput.
    """
    request_ids = result.get("request_ids", [])
    if not request_ids or any(
        not isinstance(rid, str) or not rid for rid in request_ids
    ):
        raise ValueError("Native benchmark --output-details must include request_ids")
    expected = set(request_ids)
    if len(expected) != len(request_ids):
        raise ValueError("Benchmark contains duplicate request IDs")
    if result.get("completed") != len(request_ids):
        raise ValueError("All benchmark requests must complete")
    inputs, outputs, errors = (
        result.get(name, []) for name in ("input_lens", "output_lens", "errors")
    )
    if any(len(values) != len(request_ids) for values in (inputs, outputs, errors)):
        raise ValueError("Benchmark token accounting is incomplete")
    if any(errors) or any(type(n) is not int or n < 1 for n in inputs):
        raise ValueError("Benchmark has errors or invalid prompt lengths")
    if any(type(n) is not int or n < 2 for n in outputs):
        raise ValueError("At least two output tokens per request are required")
    if result.get("total_output_tokens") != sum(outputs):
        raise ValueError("Benchmark output token totals disagree")
    duration = _number(result.get("duration"), "duration", positive=True)
    for key in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms"):
        _number(result.get(key), key, positive=True)
    if not math.isclose(
        result["output_throughput"], sum(outputs) / duration, rel_tol=1e-6
    ):
        raise ValueError("Benchmark output throughput disagrees with tokens / duration")

    matches = []
    for trace in traces:
        records = [record for record in trace if record.get("type") == "forward"]
        selected = [
            record
            for record in records
            if expected.intersection(record.get("request_ids", []))
        ]
        if selected:
            matches.append((trace, records, selected))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one compute scheduler trace for this workload"
        )
    trace, records, selected = matches[0]
    _validate_session(trace[0])
    if sum(record.get("type") == "session" for record in trace) != 1:
        raise ValueError("A trace must contain one scheduler lifetime")
    sequences = [record.get("sequence") for record in records]
    if sequences != list(range(len(sequences))):
        raise ValueError(
            "Trace contains missing, reordered, or duplicate forward events"
        )
    if (
        trace[-1].get("type") != "idle_flush"
        or trace[-1].get("last_sequence") != sequences[-1]
    ):
        raise ValueError("Trace has not finished flushing at scheduler idle")
    streams = {record.get("stream_id") for record in selected}
    if len(streams) != 1 or None in streams:
        raise ValueError("Prefill subtraction requires one compute stream")
    first_sequence, last_sequence = selected[0]["sequence"], selected[-1]["sequence"]
    if any(
        first_sequence <= record["sequence"] <= last_sequence
        and not expected.intersection(record.get("request_ids", []))
        for record in records
    ):
        raise ValueError("Unrelated forwards overlap the benchmark request interval")

    prefill_tokens, decode_forwards = Counter(), Counter()
    prefill_seconds = decode_seconds = 0.0
    prefill_batches = decode_batches = 0
    for record in selected:
        ids = record["request_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= expected:
            raise ValueError(
                "A batch mixes benchmark and unrelated or duplicate requests"
            )
        if record.get("success") is not True:
            raise ValueError("Trace contains a failed model forward")
        elapsed = _number(record.get("cuda_seconds"), "cuda_seconds", positive=True)
        if record.get("forward_mode") == "extend":
            lengths = record.get("extend_lens")
            if not isinstance(lengths, list) or len(lengths) != len(ids):
                raise ValueError("Prefill trace is missing per-request chunk lengths")
            if any(type(length) is not int or length < 1 for length in lengths):
                raise ValueError("Prefill chunk lengths must be positive integers")
            prefill_tokens.update(dict(zip(ids, lengths)))
            prefill_seconds += elapsed
            prefill_batches += 1
        elif record.get("forward_mode") == "decode":
            decode_forwards.update(ids)
            decode_seconds += elapsed
            decode_batches += 1
        else:
            raise ValueError(f"Unsupported forward mode: {record.get('forward_mode')}")
    if dict(prefill_tokens) != dict(zip(request_ids, inputs)):
        raise ValueError(
            "Trace prefill lengths differ from prompts: missing chunks, prefix hits, or recomputation"
        )
    extra_decode_forwards = 0
    overlap = not trace[0]["disable_overlap_schedule"]
    for rid, output_len in zip(request_ids, outputs):
        extra = decode_forwards[rid] - (output_len - 1)
        if extra < 0 or extra > int(overlap):
            raise ValueError(
                "Trace decode forwards disagree with completed output lengths"
            )
        extra_decode_forwards += extra
    residual = duration - prefill_seconds
    if residual <= 0:
        raise ValueError("Prefill CUDA time must be smaller than benchmark wall time")
    decode_tokens = sum(outputs) - len(outputs)
    return {
        "status": "analyzed",
        "request_count": len(request_ids),
        "end_to_end": {
            "duration_seconds": duration,
            "output_throughput_tokens_per_second": result["output_throughput"],
            "mean_ttft_ms": result["mean_ttft_ms"],
            "mean_tpot_ms": result["mean_tpot_ms"],
        },
        "cuda_trace": {
            "prefill_forward_seconds": prefill_seconds,
            "decode_forward_seconds": decode_seconds,
            "prefill_batches": prefill_batches,
            "decode_batches": decode_batches,
            "extra_overlap_decode_forwards": extra_decode_forwards,
        },
        "decode_only_estimate": {
            "method": "wall_time_minus_prefill_cuda_batches",
            "residual_seconds": residual,
            "decode_output_tokens": decode_tokens,
            "decode_throughput_tokens_per_second": decode_tokens / residual,
            "all_output_throughput_tokens_per_second": sum(outputs) / residual,
            "limitations": [
                "NoPD counterfactual estimate; not an executed decode-only or PD workload.",
                "TTFT and TPOT remain measured end-to-end metrics and are never subtracted.",
                "Residual retains queueing, CPU/transport gaps, and staging outside the timed prefill forward.",
                "CUDA event tracing adds overhead; compare identically instrumented runs.",
                "Decode token count excludes the first token produced by each prefill; all-output rate is also retained.",
            ],
        },
    }


def _tp_session(session):
    if session.get("tp_size") not in (1, 4):
        raise ValueError("Stage timing supports TP1 or TP4")
    rank = session.get("tp_rank")
    if type(rank) is not int or not 0 <= rank < session["tp_size"]:
        raise ValueError("Invalid TP rank in timing session")
    _validate_session(session | {"tp_size": 1})


def _selected(trace, expected):
    return [
        record
        for record in trace
        if record.get("type") == "forward"
        and expected.intersection(record.get("request_ids", []))
    ]


def analyze(result, traces):
    """Use the maximum rank CUDA duration of each aligned TP batch, never a rank sum.

    Events have no synchronized wall clock across GPUs. The sum of per-batch
    maxima is an explicit critical-path approximation, not an exact distributed
    prefill duration. Every participating rank must report the same batch IDs,
    mode, request order and chunk sizes, and must flush its complete trace.
    """
    expected = set(result.get("request_ids", []))
    matches = [trace for trace in traces if trace and _selected(trace, expected)]
    if not matches:
        raise ValueError("No complete rank traces contain these benchmark requests")
    sessions = [trace[0] for trace in matches]
    for session in sessions:
        _tp_session(session)
    tp_size = sessions[0]["tp_size"]
    if any(session["tp_size"] != tp_size for session in sessions):
        raise ValueError("Rank sessions disagree on TP size")
    if sorted(session["tp_rank"] for session in sessions) != list(range(tp_size)):
        raise ValueError("Need exactly one complete trace for every TP rank")
    matches.sort(key=lambda trace: trace[0]["tp_rank"])
    per_rank, selected = [], []
    signatures = []
    for trace in matches:
        local = copy.deepcopy(trace)
        local[0]["tp_size"] = 1
        per_rank.append(_analyze_single(result, [local]))
        records = _selected(trace, expected)
        selected.append(records)
        signatures.append(
            [
                (record["forward_mode"], record["request_ids"], record["extend_lens"])
                for record in records
            ]
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("TP ranks disagree on batch/request/chunk alignment")
    flags = ("disable_overlap_schedule", "disable_radix_cache", "disaggregation_mode")
    if any(
        any(trace[0][flag] != matches[0][0][flag] for flag in flags)
        for trace in matches
    ):
        raise ValueError("TP ranks disagree on scheduler settings")
    merged = copy.deepcopy(matches[0])
    merged[0]["tp_size"] = 1
    merged_records = _selected(merged, expected)
    for index, record in enumerate(merged_records):
        record["cuda_seconds"] = max(
            records[index]["cuda_seconds"] for records in selected
        )
    report = _analyze_single(result, [merged])
    report["cuda_trace"].update(
        tp_size=tp_size,
        aggregation="sum_of_max_rank_cuda_duration_per_aligned_batch",
        per_rank=[
            {"tp_rank": rank, **item["cuda_trace"]}
            for rank, item in enumerate(per_rank)
        ],
    )
    report["decode_only_estimate"][
        "method"
    ] = "wall_time_minus_aligned_tp_prefill_batch_maxima"
    report["decode_only_estimate"]["limitations"].append(
        "TP duration is a sum of per-batch maximum rank CUDA times; unsynchronized starts and collective overlap prevent interpreting it as an exact distributed critical path."
    )
    return report


def read_capacities(traces, tp_size, request_ids=None, *, require_idle=False):
    """Return current per-rank snapshots, optionally after identified requests.

    Request-scoped checks reject a stale startup zero snapshot. They require
    each request to appear in every TP trace and a capacity record emitted after
    those forwards at the current final idle marker.
    """
    expected = set(request_ids or [])
    matches = [
        trace
        for trace in traces
        if trace and (not expected or _selected(trace, expected))
    ]
    by_rank = {}
    for trace in matches:
        _tp_session(trace[0])
        if trace[0]["tp_size"] != tp_size:
            raise ValueError("Capacity trace TP size differs from this stage")
        rank = trace[0]["tp_rank"]
        if rank in by_rank:
            raise ValueError("Multiple scheduler lifetimes for one TP rank")
        records = _selected(trace, expected)
        seen = {rid for record in records for rid in record["request_ids"]}
        if not expected <= seen:
            raise ValueError("Not all request IDs are visible on every TP rank")
        if trace[-1].get("type") != "idle_flush":
            raise ValueError("Capacity must be sampled at scheduler idle")
        sequence = trace[-1]["last_sequence"]
        if records and sequence < max(record["sequence"] for record in records):
            raise ValueError("Idle marker precedes the requested workload")
        capacities = [record for record in trace if record.get("type") == "capacity"]
        if not capacities or capacities[-1].get("after_sequence") != sequence:
            raise ValueError("Missing capacity snapshot after the current workload")
        capacity = capacities[-1]["capacity"]
        if capacity.get("tp_rank") != rank or capacity.get("tp_size") != tp_size:
            raise ValueError("Capacity rank metadata disagrees with timer session")
        if require_idle and any(
            capacity.get(key) != 0 for key in ("reserved_slots", "used_slots")
        ):
            raise ValueError("Backing KV allocation or reservation remains at idle")
        by_rank[rank] = capacity
    if sorted(by_rank) != list(range(tp_size)):
        raise ValueError("Need a capacity snapshot for every TP rank")
    return [by_rank[rank] for rank in range(tp_size)]


def load_traces(trace_dir):
    return [
        read_jsonl(path) for path in sorted(Path(trace_dir).glob("forward-*.jsonl"))
    ]


def analyze_when_ready(result, trace_dir, timeout=10.0):
    """Wait briefly for the final scheduler idle flush, then validate all data."""
    _number(timeout, "timeout", positive=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            traces = load_traces(trace_dir)
            return analyze(result, traces)
        except (ValueError, OSError) as error:
            if time.monotonic() >= deadline:
                raise ValueError(f"CUDA trace did not become valid: {error}") from error
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=10.0)
    args = parser.parse_args()
    results = read_jsonl(args.result)
    if len(results) != 1:
        raise ValueError("Supply exactly one benchmark case per result file")
    report = analyze_when_ready(results[0], args.trace_dir, args.wait_seconds)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
