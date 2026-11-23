"""Strict single-host Attention-DP4/EP4 control-plane evidence contracts.

Rank CUDA durations have no common timestamp and must never be summed to wall
time. Only existing scheduler CPU metadata is consumed by these helpers.
"""

import importlib.util
import time
from pathlib import Path


def native():
    spec = importlib.util.spec_from_file_location(
        "multirank_native", Path(__file__).with_name("analyze.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_progress(states, *, now_ns=None, max_age_ns=5_000_000_000):
    """Require four aligned live schedulers, including idle EP participants."""
    if len(states) != 4:
        raise ValueError("Need all four live Attention-DP/EP schedulers")
    ranks = {}
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    for state in states:
        row = state.get("mpu_forward_progress")
        if not isinstance(row, dict):
            raise ValueError("Missing live rank forward progress")
        rank = row.get("attn_dp_rank")
        if type(rank) is not int or rank not in range(4) or rank in ranks:
            raise ValueError("Missing or duplicated live Attention-DP rank")
        if row.get("tp_rank") != rank or row.get("attn_dp_size") != 4:
            raise ValueError("Live topology differs from Attention-DP4/EP4")
        stamp = row.get("launch_monotonic_ns")
        if type(stamp) is not int or not 0 <= now_ns - stamp <= max_age_ns:
            raise ValueError("Stale or invalid live forward timestamp")
        ranks[rank] = row
    rows = [ranks[rank] for rank in range(4)]
    reference = rows[0]
    sequence = reference.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("Invalid live global sequence")
    modes, counts = reference.get("dp_forward_modes"), reference.get("dp_num_tokens")
    if (
        not isinstance(modes, list)
        or len(modes) != 4
        or not isinstance(counts, list)
        or len(counts) != 4
    ):
        raise ValueError("Missing actual live DP gather vectors")
    owners = {}
    for rank, row in enumerate(rows):
        if (
            row.get("sequence") != sequence
            or row.get("dp_forward_modes") != modes
            or row.get("dp_num_tokens") != counts
        ):
            raise ValueError("Live EP steps are not aligned")
        ids = row.get("request_ids")
        if (
            not isinstance(ids, list)
            or len(set(ids)) != len(ids)
            or any(not isinstance(rid, str) or not rid for rid in ids)
        ):
            raise ValueError("Invalid or duplicated live request IDs")
        if (
            row.get("success") is not True
            or row.get("forward_mode") != modes[rank]
            or modes[rank] not in ("decode", "idle")
        ):
            raise ValueError("Live stable EP step contains prefill or failed forward")
        if (
            (modes[rank] == "idle") != (not ids)
            or type(counts[rank]) is not int
            or counts[rank] != len(ids)
            or row.get("batch_size") != len(ids)
        ):
            raise ValueError("Live local batch differs from DP gather")
        for rid in ids:
            if rid in owners:
                raise ValueError("Duplicate live request owner")
            owners[rid] = rank
    if not owners:
        raise ValueError("All-idle steps are not model decode")
    return {
        "sequence": sequence,
        "forward_mode": "decode",
        "success": True,
        "request_ids": sorted(owners),
        "request_owners": owners,
        "batch_size": len(owners),
        "actual_local_batch_sizes": counts,
        "consecutive_stable_decode_steps": min(
            row.get("consecutive_stable_decode_steps", 0) for row in rows
        ),
        "per_rank": rows,
        "launch_monotonic_ns": max(row["launch_monotonic_ns"] for row in rows),
        "timing_scope": "Aligned single-host EP step; each rank retains its own stream and launch timestamp",
    }


def aligned_steps(traces, expected):
    analyzer = native()
    traces = analyzer._dp_traces(traces)
    owners = analyzer._request_owners(traces, expected)
    steps = [[row for row in trace if row.get("type") == "forward"] for trace in traces]
    for group in zip(*steps):
        modes, counts = group[0].get("dp_forward_modes"), group[0].get("dp_num_tokens")
        if (
            not isinstance(modes, list)
            or len(modes) != 4
            or not isinstance(counts, list)
            or len(counts) != 4
        ):
            raise ValueError("Missing aligned DP gather vectors")
        for rank, row in enumerate(group):
            ids, mode = row.get("request_ids"), row.get("forward_mode")
            if (
                not isinstance(ids, list)
                or len(set(ids)) != len(ids)
                or any(not isinstance(rid, str) or not rid for rid in ids)
            ):
                raise ValueError("Malformed request IDs in aligned EP step")
            tokens = sum(row.get("extend_lens") or []) if mode == "extend" else len(ids)
            if (
                row.get("success") is not True
                or mode not in ("decode", "extend", "idle")
                or row.get("dp_forward_modes") != modes
                or row.get("dp_num_tokens") != counts
                or mode != modes[rank]
                or type(counts[rank]) is not int
                or counts[rank] != tokens
                or (mode == "idle") != (not ids)
            ):
                raise ValueError("Failed or mismatched aligned EP mode/token step")
        if set(modes) == {"idle"}:
            raise ValueError("All-idle step cannot be model forward")
    # Native completeness includes vector agreement, owner and local token audit.
    streams = []
    for rank, rows in enumerate(steps):
        stamps = [row.get("launch_monotonic_ns") for row in rows]
        if any(type(stamp) is not int or stamp <= 0 for stamp in stamps) or any(
            a >= b for a, b in zip(stamps, stamps[1:])
        ):
            raise ValueError("Missing or reordered per-rank launch timestamps")
        rank_streams = {row.get("stream_id") for row in rows}
        if len(rank_streams) != 1 or None in rank_streams:
            raise ValueError("Rank changed its forward stream")
        streams.append(next(iter(rank_streams)))
    return traces, owners, list(zip(*steps)), streams
