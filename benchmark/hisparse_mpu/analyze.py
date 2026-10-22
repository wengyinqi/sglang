"""Analyze a native SGLang benchmark and opt-in batch CUDA timing traces.

The residual wall time is an estimate of decode-only service, not a measured PD
decode worker. Never subtract summed request TTFT: requests share prefill batches
and TTFT also includes queueing, tokenization, and transport.
"""

import argparse
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


def analyze(result, traces):
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


def analyze_when_ready(result, trace_dir, timeout=10.0):
    """Wait briefly for the final scheduler idle flush, then validate all data."""
    _number(timeout, "timeout", positive=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            traces = [
                read_jsonl(path)
                for path in sorted(Path(trace_dir).glob("forward-*.jsonl"))
            ]
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
