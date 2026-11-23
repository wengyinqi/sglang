"""Analyze constant-request decode intervals and raw HTTP-frame tail latency.

Launch-to-launch intervals are scheduler/host-clock measurements. CUDA event
durations remain separately labelled; neither is an executed PD decode service.
Raw multi-token frame gaps are not converted into fictional token timestamps.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def load_analyze():
    spec = importlib.util.spec_from_file_location(
        "steady_native_analyze", Path(__file__).with_name("analyze.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return (
        values[lo] * (hi - position) + values[hi] * (position - lo)
        if lo != hi
        else values[lo]
    )


def statistics(values):
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values) if values else None,
    }


def positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Invalid positive {name}")
    return value


def request_latency_details(result):
    count = len(result["request_ids"])
    names = (
        "request_latencies",
        "ttfts",
        "request_start_monotonic_ns",
        "request_completion_monotonic_ns",
        "stream_frames",
    )
    if any(len(result.get(name, [])) != count for name in names):
        raise ValueError("Missing complete per-request latency/frame evidence")
    rows, all_chunk_gaps, amortized, strict_gaps = [], [], [], []
    strict_all = True
    for i, rid in enumerate(result["request_ids"]):
        latency = positive_number(result["request_latencies"][i], "request latency")
        ttft = positive_number(result["ttfts"][i], "TTFT")
        if latency < ttft:
            raise ValueError("Request completion latency precedes TTFT")
        start, end = (
            result["request_start_monotonic_ns"][i],
            result["request_completion_monotonic_ns"][i],
        )
        if type(start) is not int or type(end) is not int or not 0 < start < end:
            raise ValueError("Missing ordered monotonic request boundaries")
        previous_timestamp, previous_count, positive_frames, gaps = start, 0, [], []
        frames = result["stream_frames"][i]
        if not frames:
            raise ValueError("Missing stream frames")
        for frame in frames:
            stamp, cumulative = frame.get("received_monotonic_ns"), frame.get(
                "cumulative_completion_tokens"
            )
            if type(stamp) is not int or not previous_timestamp <= stamp <= end:
                raise ValueError(
                    "Frame timestamps are missing, unordered or out of request bounds"
                )
            if type(cumulative) is not int or cumulative < previous_count:
                raise ValueError("Frame cumulative tokens decreased")
            delta = cumulative - previous_count
            if frame.get("new_tokens") != delta:
                raise ValueError("Frame delta disagrees with cumulative counts")
            if delta:
                if positive_frames:
                    gap = (stamp - positive_frames[-1][0]) / 1e6
                    gaps.append(gap)
                    all_chunk_gaps.append(gap)
                    amortized.extend([gap / delta] * delta)
                positive_frames.append((stamp, delta))
            previous_timestamp, previous_count = stamp, cumulative
        if (
            previous_count != result["output_lens"][i]
            or frames[-1].get("finish_reason") is None
        ):
            raise ValueError(
                "Stream frames do not contain complete final token accounting"
            )
        single = all(delta == 1 for _, delta in positive_frames)
        strict_all = strict_all and single
        if single:
            strict_gaps.extend(gaps)
        rows.append(
            {
                "request_id": rid,
                "input_tokens": result["input_lens"][i],
                "output_tokens": previous_count,
                "request_latency_ms": latency * 1000,
                "ttft_ms": ttft * 1000,
                "request_average_tpot_ms": (latency - ttft)
                * 1000
                / (previous_count - 1),
                "start_monotonic_ns": start,
                "completion_monotonic_ns": end,
                "token_positive_frames": len(positive_frames),
                "all_positive_frames_single_token": single,
                "raw_chunk_gap_ms": statistics(gaps),
            }
        )
    return {
        "requests": rows,
        "request_latency_ms": statistics([r["request_latency_ms"] for r in rows]),
        "request_average_tpot_ms": statistics(
            [r["request_average_tpot_ms"] for r in rows]
        ),
        "ttft_ms": statistics([r["ttft_ms"] for r in rows]),
        "raw_token_positive_frame_gap_ms": statistics(all_chunk_gaps),
        "amortized_chunk_gap_per_token_ms": statistics(amortized),
        "all_frames_single_token": strict_all,
        "single_token_arrival_gap_ms": statistics(strict_gaps) if strict_all else None,
        "timing_scope": "Client HTTP frame arrival intervals; multi-token frames have no observed internal token timestamps. Native amortized ITL is listed separately.",
    }


def analyze(result, traces, *, min_steps=16, edge_trim_steps=4):
    if (
        type(min_steps) is not int
        or min_steps < 2
        or type(edge_trim_steps) is not int
        or edge_trim_steps < 0
    ):
        raise ValueError(
            "At least two stable steps and nonnegative edge trim are required"
        )
    native = load_analyze()
    native.analyze(result, traces)  # Full completeness/token/NoPD audit first.
    if any(trace and trace[0].get("attn_dp_size", 1) > 1 for trace in traces):
        spec = importlib.util.spec_from_file_location(
            "steady_multirank", Path(__file__).with_name("steady_multirank.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.analyze(
            result, traces, min_steps=min_steps, edge_trim_steps=edge_trim_steps
        )
    expected = set(result["request_ids"])
    matches = [trace for trace in traces if native._selected(trace, expected)]
    if len(matches) != 1 or matches[0][0].get("tp_size") != 1:
        raise ValueError(
            "Stable host-clock analysis currently requires a single compute scheduler"
        )
    if matches[0][0].get("disable_overlap_schedule") is not True:
        raise ValueError(
            "Stable launch-interval accounting requires a non-overlap scheduler"
        )
    selected = native._selected(matches[0], expected)
    launches = [r.get("launch_monotonic_ns") for r in selected]
    if any(type(t) is not int or t <= 0 for t in launches) or any(
        a >= b for a, b in zip(launches, launches[1:])
    ):
        raise ValueError("Forward launch monotonic timestamps are absent or unordered")
    runs, current = [], []
    for record in selected:
        ids = tuple(sorted(record["request_ids"]))
        if record["forward_mode"] != "decode":
            if current:
                runs.append(current)
                current = []
            continue
        if current and ids != tuple(sorted(current[-1]["request_ids"])):
            runs.append(current)
            current = []
        current.append(record)
    if current:
        runs.append(current)
    windows = []
    for run in runs:
        window = run[
            edge_trim_steps : (
                len(run) - edge_trim_steps if edge_trim_steps else len(run)
            )
        ]
        if len(window) < min_steps:
            continue
        first, last = window[0], window[-1]
        duration = (last["launch_monotonic_ns"] - first["launch_monotonic_ns"]) / 1e9
        intervals = window[
            :-1
        ]  # Last launch is the right boundary, not a completed step.
        batch = len(first["request_ids"])
        gaps = [
            (b["launch_monotonic_ns"] - a["launch_monotonic_ns"]) / 1e6
            for a, b in zip(window, window[1:])
        ]
        windows.append(
            {
                "first_sequence": first["sequence"],
                "right_boundary_sequence": last["sequence"],
                "request_ids": sorted(first["request_ids"]),
                "actual_batch_size": batch,
                "forward_records": len(window),
                "measured_launch_intervals": len(intervals),
                "start_monotonic_ns": first["launch_monotonic_ns"],
                "end_monotonic_ns": last["launch_monotonic_ns"],
                "scheduler_launch_span_seconds": duration,
                "decode_tokens_over_launch_intervals": batch * len(intervals),
                "decode_tokens_per_scheduler_launch_second": batch
                * len(intervals)
                / duration,
                "cuda_forward_seconds_over_same_intervals": sum(
                    r["cuda_seconds"] for r in intervals
                ),
                "scheduler_launch_gap_ms": statistics(gaps),
                "edge_trim_steps_per_side": edge_trim_steps,
            }
        )
    first_decode = next(
        (r["sequence"] for r in selected if r["forward_mode"] == "decode"), None
    )
    prefill_insertions = [
        r
        for r in selected
        if r["forward_mode"] == "extend"
        and first_decode is not None
        and r["sequence"] > first_decode
    ]
    final_decode_sequence = {
        rid: max(
            r["sequence"]
            for r in selected
            if r["forward_mode"] == "decode" and rid in r["request_ids"]
        )
        for rid in expected
    }
    return {
        "schema_version": 1,
        "status": "analyzed",
        "stable_window_observed": bool(windows),
        "stable_definition": "At least min_steps contiguous decode records with the same actual request-ID set, one stream and no prefill insertion; optional symmetric edge trim.",
        "min_steps": min_steps,
        "edge_trim_steps": edge_trim_steps,
        "windows": windows,
        "prefill_insertions": [
            {
                "sequence": r["sequence"],
                "request_ids": r["request_ids"],
                "cuda_seconds": r["cuda_seconds"],
            }
            for r in prefill_insertions
        ],
        "last_decode_sequence_per_request": final_decode_sequence,
        "actual_decode_batch_distribution": {
            str(size): sum(
                len(r["request_ids"]) == size
                for r in selected
                if r["forward_mode"] == "decode"
            )
            for size in sorted(
                {
                    len(r["request_ids"])
                    for r in selected
                    if r["forward_mode"] == "decode"
                }
            )
        },
        "client": request_latency_details(result),
        "limitations": [
            "Scheduler launch-span throughput is a bounded observed steady interval, not complete request throughput or PD throughput.",
            "CUDA event sums and scheduler launch intervals are separate clocks/scopes and are not added or subtracted to claim CPU time.",
            "A same-batch window does not establish cache-hit counts or pure KV-transfer time.",
            "Request IDs and completion ordering are measured; queue/admission time is not separately instrumented.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=16)
    parser.add_argument("--edge-trim-steps", type=int, default=4)
    args = parser.parse_args()
    native = load_analyze()
    results = native.read_jsonl(args.result)
    if len(results) != 1 or args.output.exists():
        raise ValueError("Require one result and a new output path")
    report = analyze(
        results[0],
        native.load_traces(args.trace_dir),
        min_steps=args.min_steps,
        edge_trim_steps=args.edge_trim_steps,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
