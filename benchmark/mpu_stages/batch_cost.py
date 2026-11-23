"""Compare contiguous decode windows with matched real requests and history.

This is a host-side evidence analyzer, not a GPU benchmark. It keeps CUDA-event
forward durations separate from scheduler launch spans. The legacy single-rank
trace can reconstruct logical history lengths from complete prefill/decode
coverage, but cannot retrospectively observe sequence lengths or graph replay.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CONTROL_FIELDS = (
    "source_commit",
    "model_revision",
    "runtime_image",
    "gpu_uuid",
    "family",
    "input_snapshot_sha256",
    "input_order_contract",
    "seed",
    "dtype",
    "kv_dtype",
    "selector",
    "top_k",
    "page_size",
    "allocated_compute_kv_bytes",
    "preallocated_request_slots",
    "max_context_len",
    "input_len",
    "output_len",
    "client_concurrency",
    "cuda_graph_enabled",
    "diagnostic",
    "serving_config_sha256",
)
VARYING_FIELDS = {"hot_buffer", "backing_ratio", "variant"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def statistics(values):
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def positive_integer(value, label):
    require(type(value) is int and value > 0, "Invalid positive integer: " + label)
    return value


def positive_number(value, label):
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "Invalid positive number: " + label,
    )
    return value


def execution_scope(record, identity):
    if "actual_cuda_graph" in record:
        require(
            type(record["actual_cuda_graph"]) is bool, "Invalid graph execution flag"
        )
        require(
            not record["actual_cuda_graph"] or identity["cuda_graph_enabled"],
            "Graph replay contradicts disabled policy",
        )
        return "graph_replay" if record["actual_cuda_graph"] else "eager"
    # Eligibility is not proof of execution. Legacy traces expose neither.
    return (
        "unobserved_graph_enabled_policy"
        if identity["cuda_graph_enabled"]
        else "unobserved_graph_disabled_policy"
    )


def _validate_identity(identity):
    require(isinstance(identity, dict), "Missing experiment identity")
    for field in CONTROL_FIELDS:
        require(
            field in identity and identity[field] is not None,
            "Missing control: " + field,
        )
    require(
        identity["family"] == "qwen",
        "This analyzer currently covers single-compute Qwen",
    )
    require(
        identity["input_order_contract"] == "frozen_snapshot_row_order",
        "Unknown input-row mapping",
    )
    require(type(identity["cuda_graph_enabled"]) is bool, "Missing graph policy")
    require(type(identity["diagnostic"]) is bool, "Missing diagnostic scope")
    for field in ("source_commit", "model_revision"):
        require(
            isinstance(identity[field], str)
            and len(identity[field]) == 40
            and all(c in "0123456789abcdef" for c in identity[field]),
            "Invalid source/model revision",
        )
    require(
        isinstance(identity["input_snapshot_sha256"], str)
        and len(identity["input_snapshot_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in identity["input_snapshot_sha256"]),
        "Invalid input snapshot SHA",
    )
    require(
        isinstance(identity["serving_config_sha256"], str)
        and len(identity["serving_config_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in identity["serving_config_sha256"]),
        "Invalid serving config SHA",
    )
    for field in ("runtime_image", "gpu_uuid", "dtype", "kv_dtype", "selector"):
        require(
            isinstance(identity[field], str) and identity[field],
            "Empty control: " + field,
        )
    for field in (
        "seed",
        "top_k",
        "page_size",
        "allocated_compute_kv_bytes",
        "preallocated_request_slots",
        "max_context_len",
        "input_len",
        "output_len",
        "client_concurrency",
    ):
        positive_integer(identity[field], field)
    require(
        isinstance(identity.get("case_id"), str) and identity["case_id"],
        "Missing case ID",
    )
    for field in VARYING_FIELDS:
        require(
            field in identity and identity[field] is not None,
            "Missing varying control: " + field,
        )
    positive_integer(identity["hot_buffer"], "hot_buffer")
    positive_number(identity["backing_ratio"], "backing_ratio")
    require(identity["variant"] in ("cpu", "hbm", "mpu"), "Unknown backing variant")


def _window(rows, identity, edge_trim_steps, min_steps):
    trimmed = rows[
        edge_trim_steps : len(rows) - edge_trim_steps if edge_trim_steps else None
    ]
    if len(trimmed) < min_steps:
        return None
    first, boundary = trimmed[0], trimmed[-1]
    batch = len(first["request_ids"])
    logical = first["logical_request_rows"]
    start_history = first["history_lengths"]
    for offset, row in enumerate(trimmed):
        require(
            row["logical_request_rows"] == logical, "Window logical requests changed"
        )
        require(
            row["history_lengths"] == [n + offset for n in start_history],
            "History did not advance one token per decode",
        )
    intervals = trimmed[:-1]
    span = (boundary["launch_monotonic_ns"] - first["launch_monotonic_ns"]) / 1e9
    require(span > 0, "Nonpositive window span")
    gaps = [
        (b["launch_monotonic_ns"] - a["launch_monotonic_ns"]) / 1e6
        for a, b in zip(trimmed, trimmed[1:])
    ]
    return {
        "first_sequence": first["sequence"],
        "right_boundary_sequence": boundary["sequence"],
        "request_ids": first["request_ids"],
        "logical_request_rows": logical,
        "actual_batch_size": batch,
        "history_start": start_history,
        "history_right_boundary": boundary["history_lengths"],
        "history_scope": first["history_scope"],
        "actual_execution_scope": first["actual_execution_scope"],
        "measured_intervals": len(intervals),
        "scheduler_launch_span_seconds": span,
        "decode_tokens_over_launch_intervals": batch * len(intervals),
        "decode_tokens_per_scheduler_launch_second": batch * len(intervals) / span,
        "scheduler_launch_gap_ms": statistics(gaps),
        "cuda_forward_ms": statistics([r["cuda_seconds"] * 1000 for r in intervals]),
        "cuda_forward_seconds_sum": sum(r["cuda_seconds"] for r in intervals),
        "edge_trim_steps_per_side": edge_trim_steps,
        # Kept in memory for exact alignment, omitted from default saved summaries.
        "_records": trimmed,
    }


def analyze_case(result, trace, identity, *, min_steps=16, edge_trim_steps=4):
    """Validate a complete case and preserve only contiguous stable windows.

    Request UUIDs are observed within each run. Logical row ordinals map separate
    runs to the same frozen input rows; they are not substituted for UUIDs in the
    raw evidence. Generated text equality is checked later, not token-ID equality.
    """
    _validate_identity(identity)
    require(type(min_steps) is int and min_steps >= 2, "Require at least two steps")
    require(type(edge_trim_steps) is int and edge_trim_steps >= 0, "Invalid edge trim")
    require(isinstance(trace, list) and trace, "Missing trace")
    sessions = [r for r in trace if r.get("type") == "session"]
    require(
        len(sessions) == 1 and trace[0] is sessions[0], "Require one complete session"
    )
    session = sessions[0]
    require(
        session.get("tp_size") == session.get("pp_size") == session.get("dp_size") == 1,
        "Requires one compute scheduler; distributed ranks need owner-aware analysis",
    )
    require(
        session.get("disable_overlap_schedule") is True,
        "Overlap scheduling is unsupported",
    )
    require(
        session.get("disaggregation_mode") == "null"
        and session.get("speculative_algorithm") is None,
        "PD/speculative decode cannot use this history reconstruction",
    )
    require(
        session.get("enable_mixed_chunk") is False
        and session.get("enable_two_batch_overlap") is False,
        "Mixed/overlap forwards are unsupported",
    )
    require(session.get("disable_radix_cache") is True, "Prefix reuse is unsupported")
    ids = result.get("request_ids")
    require(
        isinstance(ids, list) and ids and all(isinstance(r, str) and r for r in ids),
        "Missing request IDs",
    )
    require(len(set(ids)) == len(ids), "Duplicate real request IDs / padding aliases")
    require(
        type(result.get("completed")) is int and result["completed"] == len(ids),
        "Incomplete requests",
    )
    require(
        result.get("dataset_name") == "frozen-ids", "Frozen input rows are required"
    )
    for field in ("input_lens", "output_lens", "errors", "generated_texts"):
        require(
            isinstance(result.get(field), list) and len(result[field]) == len(ids),
            "Missing result field: " + field,
        )
    require(not any(result["errors"]), "Request errors present")
    require(
        all(
            type(n) is int and n == identity["input_len"] for n in result["input_lens"]
        ),
        "Input length differs",
    )
    require(
        all(
            type(n) is int and n == identity["output_len"]
            for n in result["output_lens"]
        ),
        "Output length differs",
    )
    require(
        result.get("total_output_tokens") == sum(result["output_lens"]),
        "Output token total differs",
    )
    require(
        type(result.get("max_concurrency")) is int
        and result["max_concurrency"] == identity["client_concurrency"],
        "Client concurrency differs",
    )
    require(
        all(isinstance(t, str) for t in result["generated_texts"]),
        "Missing output text",
    )
    mapping = {rid: i for i, rid in enumerate(ids)}
    expected = set(ids)
    prefill, decode, distribution = Counter(), Counter(), Counter()
    sequence, last_launch, current, windows = 0, 0, [], []
    first_selected, last_selected = None, None
    for raw in trace:
        if raw.get("type") != "forward":
            continue
        require(
            type(raw.get("sequence")) is int and raw["sequence"] == sequence,
            "Incomplete/reordered forward sequence",
        )
        sequence += 1
        stamp = positive_integer(raw.get("launch_monotonic_ns"), "launch timestamp")
        require(stamp > last_launch, "Unordered forward launch timestamps")
        last_launch = stamp
        rids = raw.get("request_ids")
        require(
            isinstance(rids, list) and all(isinstance(r, str) and r for r in rids),
            "Malformed forward requests",
        )
        require(
            len(set(rids)) == len(rids),
            "Duplicate forward requests / graph padding aliases",
        )
        if not expected.intersection(rids):
            if current:
                candidate = _window(current, identity, edge_trim_steps, min_steps)
                if candidate:
                    windows.append(candidate)
                current = []
            continue
        require(set(rids) <= expected, "Foreign request mixed into formal case")
        require(raw.get("success") is True, "Failed formal forward")
        positive_number(raw.get("cuda_seconds"), "CUDA forward duration")
        require(
            type(raw.get("stream_id")) is int and raw["stream_id"] >= 0,
            "Invalid CUDA stream",
        )
        if "actual_batch_size" in raw:
            require(
                raw["actual_batch_size"] == len(rids),
                "Actual batch includes padding rows",
            )
        first_selected = raw["sequence"] if first_selected is None else first_selected
        last_selected = raw["sequence"]
        if raw.get("forward_mode") == "extend":
            lengths = raw.get("extend_lens")
            require(
                isinstance(lengths, list) and len(lengths) == len(rids),
                "Invalid prefill chunks",
            )
            for rid, length in zip(rids, lengths):
                positive_integer(length, "prefill length")
                require(decode[rid] == 0, "Prefill inserted after request decode")
                prefill[rid] += length
            if current:
                candidate = _window(current, identity, edge_trim_steps, min_steps)
                if candidate:
                    windows.append(candidate)
                current = []
            continue
        require(raw.get("forward_mode") == "decode", "Unknown forward mode")
        history = {}
        for rid in rids:
            require(
                prefill[rid] == identity["input_len"], "Decode before complete prefill"
            )
            decode[rid] += 1
            history[rid] = prefill[rid] + decode[rid]
        observed = raw.get("seq_lens")
        if observed is not None:
            require(
                isinstance(observed, list) and len(observed) == len(rids),
                "Malformed observed sequence lengths",
            )
            require(
                all(type(n) is int and n > 0 for n in observed),
                "Invalid observed sequence lengths",
            )
            require(
                dict(zip(rids, observed)) == history,
                "Observed history conflicts with complete decode coverage",
            )
            history_scope = "observed_seq_lens_verified_against_complete_coverage"
        else:
            history_scope = "derived_input_plus_complete_decode_ordinal"
        ordered = sorted(rids, key=mapping.__getitem__)
        row = {
            "sequence": raw["sequence"],
            "launch_monotonic_ns": stamp,
            "cuda_seconds": raw["cuda_seconds"],
            "stream_id": raw["stream_id"],
            "request_ids": ordered,
            "logical_request_rows": [mapping[r] for r in ordered],
            "history_lengths": [history[r] for r in ordered],
            "history_scope": history_scope,
            "actual_execution_scope": execution_scope(raw, identity),
        }
        distribution[len(ordered)] += 1
        if current and (
            row["request_ids"] != current[-1]["request_ids"]
            or row["stream_id"] != current[-1]["stream_id"]
            or row["sequence"] != current[-1]["sequence"] + 1
            or row["history_scope"] != current[-1]["history_scope"]
            or row["actual_execution_scope"] != current[-1]["actual_execution_scope"]
        ):
            candidate = _window(current, identity, edge_trim_steps, min_steps)
            if candidate:
                windows.append(candidate)
            current = []
        current.append(row)
    if current:
        candidate = _window(current, identity, edge_trim_steps, min_steps)
        if candidate:
            windows.append(candidate)
    require(
        dict(prefill) == dict.fromkeys(ids, identity["input_len"]),
        "Incomplete prefill coverage",
    )
    require(
        dict(decode) == dict.fromkeys(ids, identity["output_len"] - 1),
        "Incomplete decode coverage",
    )
    require(
        trace[-1].get("type") == "idle_flush"
        and trace[-1].get("last_sequence") == sequence - 1,
        "Trace lacks complete final idle flush",
    )
    return {
        "schema_version": 1,
        "kind": "matched_batch_case_analysis",
        "status": "analyzed",
        "identity": dict(identity),
        "request_count": len(ids),
        "request_uuid_by_input_row": ids,
        "generated_text_sha256_by_input_row": [
            hashlib.sha256(t.encode("utf-8")).hexdigest()
            for t in result["generated_texts"]
        ],
        "generated_token_ids_verified": False,
        "first_case_sequence": first_selected,
        "last_case_sequence": last_selected,
        "actual_decode_batch_distribution": {
            str(k): v for k, v in sorted(distribution.items())
        },
        "windows": windows,
        "timing_scope": "CUDA-event forward durations and scheduler launch spans are separate measurements; neither is pure KV transfer or per-request TPOT",
        "new_gpu_execution": False,
    }


def compact_analysis(analysis):
    """Serializable window boundaries suffice to replay against hash-bound inputs."""
    return {
        **analysis,
        "windows": [
            {k: v for k, v in w.items() if not k.startswith("_")}
            for w in analysis["windows"]
        ],
    }


def _window_key(window):
    history = window["history_start"]
    return (
        tuple(window["logical_request_rows"]),
        tuple(n - history[0] for n in history),
        window["history_scope"],
        window["actual_execution_scope"],
    )


def _aligned_summary(window, start, end):
    offset = start - window["history_start"][0]
    count = end - start
    rows = window["_records"][offset : offset + count + 1]
    require(len(rows) == count + 1, "Alignment exceeds contiguous window")
    require(
        rows[0]["history_lengths"][0] == start
        and rows[-1]["history_lengths"][0] == end,
        "History boundaries differ",
    )
    intervals, boundary = rows[:-1], rows[-1]
    span = (boundary["launch_monotonic_ns"] - rows[0]["launch_monotonic_ns"]) / 1e9
    batch = window["actual_batch_size"]
    gaps = [
        (b["launch_monotonic_ns"] - a["launch_monotonic_ns"]) / 1e6
        for a, b in zip(rows, rows[1:])
    ]
    return {
        "first_sequence": rows[0]["sequence"],
        "right_boundary_sequence": boundary["sequence"],
        "request_ids": rows[0]["request_ids"],
        "actual_batch_size": batch,
        "history_start": rows[0]["history_lengths"],
        "history_right_boundary": boundary["history_lengths"],
        "start_monotonic_ns": rows[0]["launch_monotonic_ns"],
        "end_monotonic_ns": boundary["launch_monotonic_ns"],
        "measured_launch_intervals": count,
        "scheduler_launch_span_seconds": span,
        "decode_tokens_over_launch_intervals": batch * count,
        "decode_tokens_per_scheduler_launch_second": batch * count / span,
        "scheduler_launch_gap_ms": statistics(gaps),
        "cuda_forward_ms": statistics([r["cuda_seconds"] * 1000 for r in intervals]),
        "cuda_forward_seconds_sum": sum(r["cuda_seconds"] for r in intervals),
        "history_scope": window["history_scope"],
        "actual_execution_scope": window["actual_execution_scope"],
    }


def match_windows(
    analyses,
    *,
    varying_fields=("hot_buffer",),
    min_intervals=16,
    actual_batch_size=None
):
    """Match exact logical request rows and history vectors, never merge runs.

    The same ordered frozen input snapshot and generated text are required.
    This is conservative text-and-length matching, not observed token-ID matching.
    """
    require(
        isinstance(analyses, list) and len(analyses) >= 2, "Require at least two cases"
    )
    require(
        type(min_intervals) is int and min_intervals >= 1, "Invalid minimum intervals"
    )
    if actual_batch_size is not None:
        positive_integer(actual_batch_size, "requested actual batch size")
    require(set(varying_fields) <= VARYING_FIELDS, "Unsupported varying controls")
    case_ids = [a["identity"]["case_id"] for a in analyses]
    require(len(set(case_ids)) == len(case_ids), "Duplicate case in comparison")
    for analysis in analyses:
        _validate_identity(analysis["identity"])
        require(analysis.get("status") == "analyzed", "Case has not passed analysis")
        require(
            analysis["identity"]["diagnostic"] is False,
            "Counter diagnostics cannot enter formal performance comparison",
        )
        require(
            analysis["request_count"] == analyses[0]["request_count"],
            "Request count differs",
        )
        for field in CONTROL_FIELDS + tuple(VARYING_FIELDS - set(varying_fields)):
            require(
                analysis["identity"].get(field) == analyses[0]["identity"].get(field),
                "Controlled field differs: " + field,
            )
    indexed = []
    for analysis in analyses:
        mapping = defaultdict(list)
        for window in analysis["windows"]:
            if (
                actual_batch_size is not None
                and window["actual_batch_size"] != actual_batch_size
            ):
                continue
            mapping[_window_key(window)].append(window)
        indexed.append(mapping)
    common = set(indexed[0])
    for mapping in indexed[1:]:
        common &= set(mapping)
    paired, missing = [], []
    if not common:
        missing.append({"reason": "no_common_contiguous_request_history_key"})
    for key in sorted(common):
        groups = [mapping[key] for mapping in indexed]
        # One logical set should not reappear in separate disjoint runs. Reject
        # ambiguous matching rather than constructing a favorable stitched span.
        if any(len(group) != 1 for group in groups):
            missing.append(
                {
                    "logical_request_rows": list(key[0]),
                    "reason": "ambiguous_multiple_contiguous_runs",
                }
            )
            continue
        windows = [group[0] for group in groups]
        rows = list(key[0])
        if any(
            [a["generated_text_sha256_by_input_row"][i] for i in rows]
            != [analyses[0]["generated_text_sha256_by_input_row"][i] for i in rows]
            for a in analyses[1:]
        ):
            missing.append(
                {
                    "logical_request_rows": rows,
                    "reason": "generated_text_differs_token_history_unverified",
                }
            )
            continue
        start = max(w["history_start"][0] for w in windows)
        end = min(w["history_right_boundary"][0] for w in windows)
        if end - start < min_intervals:
            missing.append(
                {
                    "logical_request_rows": rows,
                    "reason": "insufficient_common_contiguous_history_range",
                }
            )
            continue
        paired.append(
            {
                "logical_request_rows": rows,
                "actual_batch_size": len(rows),
                "history_scope": key[2],
                "actual_execution_scope": key[3],
                "generated_text_equal": True,
                "generated_token_ids_verified": False,
                "cases": [
                    {
                        "case_id": analysis["identity"]["case_id"],
                        "varying_controls": {
                            k: analysis["identity"][k] for k in varying_fields
                        },
                        **_aligned_summary(window, start, end),
                    }
                    for analysis, window in zip(analyses, windows)
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "matched_batch_comparison",
        "status": "matched" if paired else "no_matched_windows",
        "case_ids": case_ids,
        "varying_fields": list(varying_fields),
        "requested_actual_batch_size": actual_batch_size,
        "controlled_identity": {k: analyses[0]["identity"][k] for k in CONTROL_FIELDS},
        "paired_windows": paired,
        "unmatched_reasons": missing,
        "no_stitching": True,
        "new_gpu_execution": False,
        "limitations": [
            "Only contiguous identical logical-request sets and history-length vectors are aligned; maximum batch is not used as a proxy.",
            "Legacy history is derived from complete prefill/decode coverage; legacy graph-enabled policy does not observe per-forward graph replay.",
            "Frozen input rows and generated text match; generated token-ID histories are not recorded by these legacy performance results.",
            "Scheduler launch spans and CUDA-event sums are separate clocks/scopes, not pure transfer time, request TPOT, DDR bandwidth or MPU evidence.",
            "A closed-loop client concurrency limit is not a fixed external arrival QPS.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON with cases/result/trace/identity and varying_fields",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new output file")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    analyses, inputs = [], []
    for case in spec["cases"]:
        if "controlled_server_arguments" in case:
            encoded = json.dumps(
                case["controlled_server_arguments"], separators=(",", ":")
            )
            require(
                hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                == case["identity"]["serving_config_sha256"],
                "Controlled server arguments conflict with identity",
            )
        if "manifest" in case:
            manifest_path = Path(case["manifest"])
            digest = sha(manifest_path)
            require(digest == case["manifest_sha256"], "Input SHA differs: manifest")
            inputs.append({"path": str(manifest_path), "sha256": digest})
        result_path, trace_path = Path(case["result"]), Path(case["trace"])
        for name, path in (("result", result_path), ("trace", trace_path)):
            digest = sha(path)
            require(digest == case[name + "_sha256"], "Input SHA differs: " + name)
            inputs.append({"path": str(path), "sha256": digest})
        result = read_jsonl(result_path)
        require(len(result) == 1, "Require one complete result")
        analyses.append(
            analyze_case(result[0], read_jsonl(trace_path), case["identity"])
        )
    comparison = match_windows(
        analyses,
        varying_fields=tuple(spec["varying_fields"]),
        actual_batch_size=spec.get("actual_batch_size"),
    )
    comparison["inputs"] = inputs
    comparison["spec_sha256"] = sha(args.spec)
    comparison["analyzer_sha256"] = sha(Path(__file__))
    comparison["case_analyses"] = [compact_analysis(a) for a in analyses]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": comparison["status"],
                "paired_windows": len(comparison["paired_windows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
