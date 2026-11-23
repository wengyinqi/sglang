"""Stable global decode windows for single-host Attention-DP4 + EP4."""

import importlib.util
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        "steady_dp_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(result, traces, *, min_steps, edge_trim_steps):
    common, steady = load("multirank"), load("steady_decode")
    expected = set(result["request_ids"])
    traces, owners, groups, streams = common.aligned_steps(traces, expected)
    selected_sequences = [
        row["sequence"]
        for group in groups
        for row in group
        if expected.intersection(row["request_ids"])
    ]
    begin, end = min(selected_sequences), max(selected_sequences)
    runs, current, insertions, distribution = [], [], [], {}
    first_decode = next(
        (
            group[0]["sequence"]
            for group in groups[begin : end + 1]
            if "decode" in group[0]["dp_forward_modes"]
        ),
        None,
    )
    for group in groups[begin : end + 1]:
        modes = group[0]["dp_forward_modes"]
        active = set(modes) - {"idle"}
        if active != {"decode"}:
            if current:
                runs.append(current)
                current = []
            if first_decode is not None and group[0]["sequence"] > first_decode:
                insertions.append(
                    {
                        "sequence": group[0]["sequence"],
                        "dp_forward_modes": modes,
                        "per_rank_request_ids": [row["request_ids"] for row in group],
                    }
                )
            continue
        signature = tuple(tuple(sorted(row["request_ids"])) for row in group)
        batch = sum(len(ids) for ids in signature)
        distribution[str(batch)] = distribution.get(str(batch), 0) + 1
        if current and signature != current[-1][0]:
            runs.append(current)
            current = []
        current.append((signature, group))
    if current:
        runs.append(current)
    windows = []
    for run in runs:
        trimmed = run[
            edge_trim_steps : (
                len(run) - edge_trim_steps if edge_trim_steps else len(run)
            )
        ]
        if len(trimmed) < min_steps:
            continue
        signature = trimmed[0][0]
        rows = [group for _, group in trimmed]
        per_rank = []
        for rank in range(4):
            stamps = [group[rank]["launch_monotonic_ns"] for group in rows]
            per_rank.append(
                {
                    "attn_dp_rank": rank,
                    "stream_id": streams[rank],
                    "request_ids": list(signature[rank]),
                    "actual_batch_size": len(signature[rank]),
                    "start_monotonic_ns": stamps[0],
                    "end_monotonic_ns": stamps[-1],
                    "scheduler_launch_span_seconds": (stamps[-1] - stamps[0]) / 1e9,
                    "scheduler_launch_gap_ms": steady.statistics(
                        [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
                    ),
                    "cuda_forward_seconds_over_same_intervals": sum(
                        group[rank]["cuda_seconds"] for group in rows[:-1]
                    ),
                }
            )
        # Latest rank launch represents the same global EP step boundary. Rank
        # skew is retained below; no summed duration is treated as wall-clock.
        boundaries = [
            max(row["launch_monotonic_ns"] for row in group) for group in rows
        ]
        duration = (boundaries[-1] - boundaries[0]) / 1e9
        batch = sum(len(ids) for ids in signature)
        windows.append(
            {
                "first_sequence": rows[0][0]["sequence"],
                "right_boundary_sequence": rows[-1][0]["sequence"],
                "request_ids": sorted(rid for ids in signature for rid in ids),
                "request_owners": {
                    rid: rank for rank, ids in enumerate(signature) for rid in ids
                },
                "actual_batch_size": batch,
                "actual_local_batch_sizes": [len(ids) for ids in signature],
                "forward_records": len(rows),
                "measured_launch_intervals": len(rows) - 1,
                "start_monotonic_ns": boundaries[0],
                "end_monotonic_ns": boundaries[-1],
                "scheduler_launch_span_seconds": duration,
                "decode_tokens_over_launch_intervals": batch * (len(rows) - 1),
                "decode_tokens_per_scheduler_launch_second": batch
                * (len(rows) - 1)
                / duration,
                "cuda_forward_seconds_over_same_intervals": sum(
                    max(row["cuda_seconds"] for row in group) for group in rows[:-1]
                ),
                "cuda_aggregation": "sum_of_max_rank_duration_per_aligned_EP_step_approximation",
                "scheduler_launch_gap_ms": steady.statistics(
                    [(b - a) / 1e6 for a, b in zip(boundaries, boundaries[1:])]
                ),
                "rank_launch_skew_ms": steady.statistics(
                    [
                        (
                            max(row["launch_monotonic_ns"] for row in group)
                            - min(row["launch_monotonic_ns"] for row in group)
                        )
                        / 1e6
                        for group in rows
                    ]
                ),
                "per_rank": per_rank,
                "edge_trim_steps_per_side": edge_trim_steps,
            }
        )
    return {
        "schema_version": 1,
        "status": "analyzed",
        "parallel_strategy": "attention_dp4_ep4",
        "stable_window_observed": bool(windows),
        "min_steps": min_steps,
        "edge_trim_steps": edge_trim_steps,
        "windows": windows,
        "request_owners": owners,
        "prefill_insertions": insertions,
        "stable_definition": "Contiguous aligned pure-decode EP steps with unchanged per-rank request ownership; idle EP participants retained; one stream per rank",
        "actual_decode_batch_distribution": distribution,
        "actual_local_decode_batch_distribution": {
            str(rank): {
                str(size): sum(
                    row[rank]["forward_mode"] == "decode"
                    and len(row[rank]["request_ids"]) == size
                    for row in groups[begin : end + 1]
                )
                for size in sorted(
                    {
                        len(row[rank]["request_ids"])
                        for row in groups[begin : end + 1]
                        if row[rank]["forward_mode"] == "decode"
                    }
                )
            }
            for rank in range(4)
        },
        "last_decode_sequence_per_request": {
            rid: max(
                group[owners[rid]]["sequence"]
                for group in groups
                if group[owners[rid]]["forward_mode"] == "decode"
                and rid in group[owners[rid]]["request_ids"]
            )
            for rid in expected
        },
        "client": steady.request_latency_details(result),
        "limitations": [
            "Single-host monotonic EP boundaries include scheduler and collective waits; this is not pure KV transfer time or a PD service benchmark.",
            "Rank CUDA event durations have independent starts; maxima are an approximation and four rank durations are never summed to wall-clock.",
            "Stable actual batch and owner identity do not remove prior admission and queue differences from whole-request latency.",
        ],
    }
