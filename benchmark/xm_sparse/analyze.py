"""Replay XM/Qwen evidence and separate fixed-batch cost from capacity gains.

This module never runs CUDA or a model. Raw serving results and scheduler traces
are replayed with the existing strict NoPD analyzers. Receipt hashes are observed
at analysis time; they do not replace an earlier evidence-custody receipt.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "mpu_stages"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "xm_replay_" + name, TOOLS / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve(strict=True)
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            hashed.update(data)
    return {
        "path": str(path),
        "sha256": hashed.hexdigest(),
        "bytes": path.stat().st_size,
    }


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equal(left, right):
    """A numeric replay tolerance, not a tolerance for missing fields/records."""
    if type(left) is dict and type(right) is dict:
        return left.keys() == right.keys() and all(
            equal(left[k], right[k]) for k in left
        )
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def summary(values):
    values = list(values)
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "Non-finite sample",
    )
    if not values:
        return {"samples": 0, "mean": None, "min": None, "max": None}
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def paired_statistics(deltas):
    result = summary(deltas)
    result["confidence_interval_95"] = None
    result["interval_method"] = (
        "Two-sided Student t interval across paired repeat means; independent-repeat assumption"
    )
    if len(deltas) >= 2:
        critical = (
            12.7062,
            4.30265,
            3.18245,
            2.77645,
            2.57058,
            2.44691,
            2.36462,
            2.30600,
            2.26216,
            2.22814,
            2.20099,
            2.17881,
            2.16037,
            2.14479,
            2.13145,
            2.11991,
            2.10982,
            2.10092,
            2.09302,
            2.08596,
            2.07961,
            2.07387,
            2.06866,
            2.06390,
            2.05954,
            2.05553,
            2.05183,
            2.04841,
            2.04523,
            2.04227,
        )
        df = len(deltas) - 1
        t = critical[df - 1] if df <= len(critical) else 1.96
        half = t * statistics.stdev(deltas) / math.sqrt(len(deltas))
        result["confidence_interval_95"] = [
            result["mean"] - half,
            result["mean"] + half,
        ]
    return result


def stable_intervals(result, trace, steady):
    """Join signatures by frozen input row and each request's decode ordinal.

    This is stronger than matching batch size alone. Each interval belongs to a
    validated constant-request window, and the right launch is its boundary.
    """
    index = {rid: i for i, rid in enumerate(result["request_ids"])}
    ordinals = dict.fromkeys(index, 0)
    records, signatures = {}, {}
    for record in trace:
        if record.get("type") != "forward" or record.get("forward_mode") != "decode":
            continue
        ids = record["request_ids"]
        if not set(ids) <= index.keys():
            continue
        for rid in ids:
            ordinals[rid] += 1
        ordering = sorted(ids, key=index.__getitem__)
        records[record["sequence"]] = record
        signatures[record["sequence"]] = (
            tuple(index[rid] for rid in ordering),
            tuple(ordinals[rid] for rid in ordering),
        )
    intervals = {}
    for window in steady["windows"]:
        require(
            window["actual_batch_size"] == len(window["request_ids"]),
            "Window batch/request mismatch",
        )
        for sequence in range(
            window["first_sequence"], window["right_boundary_sequence"]
        ):
            current, following = records.get(sequence), records.get(sequence + 1)
            require(
                current is not None and following is not None,
                "Incomplete stable interval",
            )
            require(
                set(current["request_ids"]) == set(following["request_ids"]),
                "Membership changes inside window",
            )
            key = signatures[sequence]
            gap = (
                following["launch_monotonic_ns"] - current["launch_monotonic_ns"]
            ) / 1e6
            require(
                gap > 0 and key not in intervals, "Invalid or duplicate stable interval"
            )
            intervals[key] = {
                "gap_ms": gap,
                "cuda_forward_ms": current["cuda_seconds"] * 1000,
                "actual_batch_size": len(key[0]),
            }
    return intervals


def allocation_category(alias_names):
    """Classify actual retained owner names, with aliases sharing one category.

    These are persistent named allocations, not estimates of temporary selection
    workspace. The first matching owner determines the category for a storage;
    e.g. a fused-plan provider alias must not become a second hot-slot map.
    """
    names = [name.lower() for name in alias_names]
    leaves = {name.rsplit(".", 1)[-1] for name in names}
    if any("xm_fused_plan" in name or name.startswith("plan.") for name in names):
        return "fused_plan_bytes"
    if "stage_bounds" in leaves:
        return "quest_layer_staging_bytes"
    if leaves & {"page_k_bounds", "page_k_min", "page_k_max"}:
        return "quest_resident_summary_bytes"
    if leaves & {"running_k_min", "running_k_max"}:
        return "quest_running_bounds_bytes"
    if leaves & {"running_token_count", "running_page_idx"}:
        return "quest_running_metadata_bytes"
    if any("bounds_tier." in name for name in names) and leaves & {
        "backing_uva_ptr",
        "ready_keys",
        "issued_keys",
        "step_epoch",
        "prefetch_gate",
        "counters",
    }:
        return "quest_prefetch_control_bytes"
    if leaves & {"lru_slots", "_lru_init"}:
        return "hisparse_lru_bytes"
    if "req_to_host_pool" in leaves:
        return "hisparse_host_mapping_bytes"
    if "fixed.request_map" in names:
        return "request_token_mapping_bytes"
    if leaves & {
        "logical_to_physical",
        "allocator_mapping_alias",
        "full_to_hisparse_device_index_mapping",
    }:
        return "hisparse_logical_to_physical_bytes"
    if "req_to_device_buffer" in leaves:
        return "hisparse_hot_request_slots_bytes"
    if "req_device_buffer_tokens" in leaves:
        return "hisparse_hot_token_ids_bytes"
    if "req_device_buffer_token_locs" in leaves:
        return "hisparse_hot_token_locs_bytes"
    if "top_k_device_locs_buffer" in leaves:
        return "hisparse_selection_output_bytes"
    if leaves & {
        "_step_last_valid_buf",
        "_step_recent_positions_buf",
        "_step_short_layout_buf",
        "_step_is_short_buf",
        "_step_actual_lens_buf",
        "_recent_offsets_const",
        "_all_positions_const",
        "_page_offsets_const",
        "_page_idx_const",
    }:
        return "quest_selection_state_bytes"
    if "num_real_reqs" in leaves:
        return "hisparse_batch_state_bytes"
    if "cache_metrics" in leaves:
        return "hisparse_diagnostic_counter_bytes"
    if any(name.startswith("fixed.kv_descriptors.") for name in names):
        return "compute_kv_descriptors_bytes"
    if any(name.startswith("backend_specific.") for name in names) and leaves & {
        "k_data_ptrs",
        "v_data_ptrs",
        "data_ptrs",
    }:
        return "backing_kv_descriptors_bytes"
    for owner in ("physical", "logical", "outer"):
        if any(name.startswith(f"allocator_dynamic.{owner}.") for name in names):
            return f"allocator_{owner}_worklists_bytes"
    if "split_partial_workspace" in leaves:
        return "fused_split_workspace_bytes"
    if "attention_output" in leaves:
        return "fused_output_bytes"
    if leaves & {"workspace_buffer", "flashinfer_workspace"}:
        return "separate_attention_workspace_bytes"
    if any(name.startswith("kv.k") or name.startswith("kv.v") for name in names):
        return "physical_compute_kv_bytes"
    return "other_named_gpu_bytes"


def allocation_records(inventory):
    """Deduplicate aliases and retain the names supporting each classification."""
    groups = defaultdict(list)
    for name, entry in inventory.get("retained_gpu_pool_shapes", {}).items():
        group = entry.get("allocation_group")
        require(
            type(group) is int and type(entry.get("storage_bytes")) is int,
            "Invalid named allocation",
        )
        groups[group].append((name, entry))
    result = []
    for group, items in sorted(groups.items()):
        sizes = {entry["storage_bytes"] for _, entry in items}
        require(len(sizes) == 1, "Alias storage sizes disagree")
        aliases = sorted(name for name, _ in items)
        result.append(
            {
                "allocation_group": group,
                "alias_names": aliases,
                "storage_bytes": next(iter(sizes)),
                "category": allocation_category(aliases),
            }
        )
    total = sum(record["storage_bytes"] for record in result)
    require(
        total == inventory.get("retained_gpu_pool_bytes", total),
        "Named allocation total disagrees",
    )
    return result


def allocation_components(inventory):
    """Assign every named storage to exactly one persistent allocation group."""
    result = defaultdict(int)
    for record in allocation_records(inventory):
        result[record["category"]] += record["storage_bytes"]
    return dict(result)


def memory_snapshot(capacity):
    ledger = capacity.get("gpu_memory_ledger", {})
    require(ledger.get("contract") == "qwen_quest_v1", "Missing Qwen memory ledger")
    cache_inventory = {
        "retained_gpu_pool_bytes": ledger["named_compute_bytes"],
        "retained_gpu_pool_shapes": ledger["named_compute_shapes"],
    }
    named = allocation_components(cache_inventory)
    cache_records = allocation_records(cache_inventory)
    backend = capacity.get(
        "xm_fused_backend_memory", capacity.get("xm_attention_backend_memory", {})
    )
    workspace = allocation_components(backend) if backend else {}
    backend_records = allocation_records(backend) if backend else []
    # The plan is already included in the cache ledger. The backend inventory
    # has separate local allocation-group IDs, so never add its plan twice.
    workspace.pop("fused_plan_bytes", None)
    return {
        "allocation_accounting_schema": "xm_qwen_named_allocations_v2",
        "cache_named_gpu_bytes": ledger["named_compute_bytes"],
        "cache_components": named,
        "cache_allocation_groups": cache_records,
        "backend_components": workspace,
        "backend_allocation_groups": [
            record
            for record in backend_records
            if record["category"] != "fused_plan_bytes"
        ],
        "backend_plan_aliases_already_in_cache": [
            record
            for record in backend_records
            if record["category"] == "fused_plan_bytes"
        ],
        "backend_inventory_observed": bool(backend),
        "cpu_full_kv_backing_bytes": capacity.get("allocated_backing_kv_bytes"),
        "cpu_quest_summary_backing_bytes": capacity.get(
            "allocated_backing_quest_summary_bytes", 0
        ),
        "allocator": capacity.get("xm_allocator_snapshot"),
        "torch_named_total_including_observed_backend_bytes": ledger[
            "named_compute_bytes"
        ]
        + sum(workspace.values()),
        "unobserved": [
            "model weights",
            "activations",
            "graph driver/private pools",
            "NVML sampling",
        ],
    }


def counter_delta(before, after):
    left = before.get("xm_quest_summary_offload")
    right = after.get("xm_quest_summary_offload")
    if left is None and right is None:
        return None
    require(
        left is not None and right is not None, "Missing a summary-counter boundary"
    )
    a, b = left["counters"], right["counters"]
    require(a.keys() == b.keys(), "Counter schemas differ")
    result = {}
    for key in a:
        require(
            type(a[key]) is int and type(b[key]) is int and 0 <= a[key] <= b[key],
            "Counters decreased",
        )
        result[key] = b[key] - a[key]
    require(
        result.get("device_launch_failures", 0) == result.get("wait_failures", 0) == 0,
        "GPU copy failure recorded",
    )
    return {
        "delta": result,
        "mechanism": right.get("mechanism"),
        "scope": "Between idle snapshots; includes bench_serving warmup unless a separate post-warmup snapshot is supplied. Not strict measured-request payload or hardware traffic.",
        "hardware_copy_engine_bytes_measured": False,
    }


def _local(base, declared):
    """Native paths may be container paths; artifacts remain within case evidence."""
    path = base / Path(str(declared)).name
    require(path.is_file() or path.is_dir(), f"Missing evidence artifact: {path}")
    return path


def collect_campaign(path, *, evidence_root=None, min_steps=16, trim=4):
    path = Path(path).resolve(strict=True)
    campaign = read(path)
    require(
        campaign.get("schema_version") == 1
        and campaign.get("phase") in ("performance", "capacity"),
        "Expected performance/capacity campaign",
    )
    require(
        re.fullmatch(
            r"[0-9a-f]{64}", campaign.get("source_fingerprint", {}).get("sha256", "")
        ),
        "Implementation fingerprint is missing",
    )
    root = Path(evidence_root).resolve() if evidence_root else path.parent
    native, steady_module = load("analyze"), load("steady_decode")
    samples, artifacts, excluded, source_commits = [], [identity(path)], [], set()
    for outer in campaign.get("cases", []):
        require(
            isinstance(outer.get("id"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", outer["id"]),
            "Invalid case evidence directory",
        )
        if outer.get("status") != "success":
            excluded.append(
                {"case": outer.get("id"), "reason": "Case did not finish successfully"}
            )
            continue
        base = root / outer["id"] / "evidence"
        manifest_path = base / "manifest.json"
        report = read(manifest_path)
        artifacts.append(identity(manifest_path))
        require(
            report.get("status") == "success" and report.get("kind") == "benchmark",
            "Native benchmark incomplete",
        )
        require(
            report.get("stage") == outer["stage"],
            "Native stage differs from explicit ablation recipe",
        )
        require(
            report.get("settings", {}).get("execute") is True,
            "Native benchmark was only planned",
        )
        source_commits.add(report.get("runtime", {}).get("git_commit"))
        expected_workload = campaign.get("workload_identity", {}).get("sha256")
        require(
            expected_workload
            and report["settings"].get("workload_sha256") == expected_workload,
            "Frozen workload fingerprint differs",
        )
        for group in report["groups"]:
            require(
                group.get("status") == "success" and group["variant"] == "cpu",
                "Unexpected/incomplete backing variant",
            )
            trace_dir = _local(base, group["trace_dir"])
            traces = native.load_traces(trace_dir)
            require(len(traces) == 1, "One compute scheduler is required")
            artifacts.extend(
                identity(p) for p in sorted(trace_dir.glob("forward-*.jsonl"))
            )
            prior = group["capacity"][0]
            for case in group["cases"]:
                require(
                    case.get("status") == "success", "Native benchmark case incomplete"
                )
                result_path = _local(base, case["result_path"])
                results = native.read_jsonl(result_path)
                require(len(results) == 1, "Expected one raw benchmark result")
                result = results[0]
                require(
                    result["completed"] == report["settings"]["num_prompts"],
                    "Requested prompt count differs",
                )
                require(
                    result["input_lens"] == [group["input_len"]] * result["completed"],
                    "Measured input bucket differs",
                )
                require(
                    result["output_lens"]
                    == [report["settings"]["output_len"]] * result["completed"],
                    "Measured output length differs",
                )
                command = case.get("command", [])
                require(
                    "--dataset-name" in command
                    and command[command.index("--dataset-name") + 1] == "frozen-ids"
                    and "--tokenize-prompt" in command,
                    "Input-row matching requires fixed frozen IDs",
                )
                bucket_hash = (
                    report.get("workload", {})
                    .get("files", {})
                    .get(str(group["input_len"]), {})
                    .get("sha256")
                )
                require(
                    bucket_hash
                    and "--frozen-ids-sha256" in command
                    and command[command.index("--frozen-ids-sha256") + 1]
                    == bucket_hash,
                    "Frozen bucket hash differs from command",
                )
                replay = native.analyze(result, traces)
                require(
                    equal(replay, case["analysis"]),
                    "Saved timing analysis differs from raw replay",
                )
                saved_steady = read(base / (case["id"] + "-steady.json"))
                original = steady_module.analyze(
                    result,
                    traces,
                    min_steps=saved_steady["min_steps"],
                    edge_trim_steps=saved_steady["edge_trim_steps"],
                )
                require(
                    equal(original, saved_steady)
                    and equal(original, case["steady_decode"]),
                    "Saved steady analysis differs from raw replay",
                )
                selected = steady_module.analyze(
                    result, traces, min_steps=min_steps, edge_trim_steps=trim
                )
                intervals = stable_intervals(result, traces[0], selected)
                artifacts.extend(
                    identity(p)
                    for p in (result_path, base / (case["id"] + "-steady.json"))
                )
                timing_path = base / (case["id"] + "-timing.json")
                require(
                    equal(read(timing_path), replay),
                    "Saved timing file differs from manifest/raw replay",
                )
                artifacts.append(identity(timing_path))
                current = case["idle_capacity"][0]
                repeat = outer.get(
                    "global_repeat",
                    outer.get("paired_repeat", outer.get("repeat", case["repeat"])),
                )
                samples.append(
                    {
                        "arm": outer["arm"],
                        "case_id": outer["id"],
                        "repeat": repeat,
                        "input_len": group["input_len"],
                        "client_concurrency": case["concurrency"],
                        "graph_enabled": group["decode_cuda_graph_enabled"],
                        "request_average_tpot_ms": result["mean_tpot_ms"],
                        "client_ttft_ms": result["mean_ttft_ms"],
                        "client_timing": selected["client"],
                        "complete_request_output_tokens_per_second": result[
                            "output_throughput"
                        ],
                        "actual_decode_batch_distribution": selected[
                            "actual_decode_batch_distribution"
                        ],
                        "memory_before": memory_snapshot(prior),
                        "memory_after": memory_snapshot(current),
                        "summary_copy_counter_delta": counter_delta(prior, current),
                        "eligible_stable_intervals": len(intervals),
                        "_intervals": intervals,
                    }
                )
                prior = current
    require(
        all(
            isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)
            for commit in source_commits
        )
        and len(source_commits) <= 1,
        "Native source revisions differ or are missing",
    )
    return campaign, samples, artifacts, excluded, sorted(source_commits)


def flat_rows(samples, *, min_intervals=16):
    rows = []
    for sample in samples:
        batches = sorted(
            {value["actual_batch_size"] for value in sample["_intervals"].values()}
        ) or [None]
        memory = sample["memory_after"]
        client = sample.get("client_timing", {})
        strict_arrival = client.get("single_token_arrival_gap_ms")
        strict_arrival = (
            strict_arrival if client.get("all_frames_single_token") is True else None
        )
        workspace = (
            sum(memory["backend_components"].values())
            if memory["backend_inventory_observed"]
            else None
        )
        for batch in batches:
            intervals = [
                value
                for value in sample["_intervals"].values()
                if value["actual_batch_size"] == batch
            ]
            gap = statistics.mean(v["gap_ms"] for v in intervals) if intervals else None
            rows.append(
                {
                    "arm": sample["arm"],
                    "input_len": sample["input_len"],
                    "client_concurrency": sample["client_concurrency"],
                    "actual_batch": batch,
                    "repeat": sample["repeat"],
                    "eligible": len(intervals) >= min_intervals,
                    "measured_intervals": len(intervals),
                    "mean_tbt_ms": gap,
                    "stable_tokens_per_s": batch * 1000 / gap if gap else None,
                    "mean_tpot_ms": sample["request_average_tpot_ms"],
                    "ttft_ms": sample["client_ttft_ms"],
                    "strict_client_mean_tbt_ms": (
                        strict_arrival["mean"] if strict_arrival else None
                    ),
                    "strict_client_p99_tbt_ms": (
                        strict_arrival["p99"] if strict_arrival else None
                    ),
                    "strict_client_tbt_samples": (
                        strict_arrival["samples"] if strict_arrival else 0
                    ),
                    "client_all_frames_single_token": client.get(
                        "all_frames_single_token"
                    ),
                    "raw_client_mean_frame_gap_ms": client.get(
                        "raw_token_positive_frame_gap_ms", {}
                    ).get("mean"),
                    "client_tbt_metric_scope": (
                        "HTTP single-token arrival gaps across complete measured requests, excluding first token; "
                        "not restricted to matched stable-batch windows. Null when any positive frame carries multiple tokens."
                    ),
                    "native_cuda_ms": (
                        statistics.mean(v["cuda_forward_ms"] for v in intervals)
                        if intervals
                        else None
                    ),
                    "named_cache_hbm_bytes": memory["cache_named_gpu_bytes"],
                    "workspace_excluding_plan_bytes": workspace,
                    "summary_backing_bytes": memory["cpu_quest_summary_backing_bytes"],
                    "mean_tbt_metric_scope": "Native stable scheduler decode launch interval; not strict client token-arrival TBT",
                }
            )
    return rows


def same_batch_pairs(samples, *, min_intervals=16):
    comparisons = (
        ("baseline", "fused"),
        ("bounds-on-demand", "bounds-prefetch"),
        ("fused", "combined"),
        ("combined", "combined-protected"),
        ("fused", "combined-protected"),
    )
    grouped = defaultdict(dict)
    for sample in samples:
        key = (
            sample["input_len"],
            sample["client_concurrency"],
            sample["repeat"],
            sample["graph_enabled"],
        )
        require(sample["arm"] not in grouped[key], "Duplicate arm in one paired repeat")
        grouped[key][sample["arm"]] = sample
    rows = []
    for key, arms in sorted(grouped.items()):
        for baseline, changed in comparisons:
            if baseline not in arms or changed not in arms:
                continue
            left, right = arms[baseline], arms[changed]
            common = left["_intervals"].keys() & right["_intervals"].keys()
            batches = sorted({len(signature[0]) for signature in common})
            for batch in batches:
                matched = [k for k in common if len(k[0]) == batch]
                eligible = len(matched) >= min_intervals
                a = statistics.mean(left["_intervals"][k]["gap_ms"] for k in matched)
                b = statistics.mean(right["_intervals"][k]["gap_ms"] for k in matched)
                rows.append(
                    {
                        "baseline": baseline,
                        "changed": changed,
                        "input_len": key[0],
                        "client_concurrency": key[1],
                        "repeat": key[2],
                        "graph_enabled": key[3],
                        "actual_batch_size": batch,
                        "matched_intervals": len(matched),
                        "eligible": eligible,
                        "baseline_mean_launch_gap_ms": a,
                        "changed_mean_launch_gap_ms": b,
                        "launch_gap_reduction_percent": (a - b) / a * 100,
                        "baseline_stable_tokens_per_second": batch * 1000 / a,
                        "changed_stable_tokens_per_second": batch * 1000 / b,
                        "matching": "Same frozen input-row set and per-request decode ordinals in constant-request windows",
                    }
                )
    aggregates = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            key = tuple(
                row[k]
                for k in (
                    "baseline",
                    "changed",
                    "input_len",
                    "client_concurrency",
                    "actual_batch_size",
                    "graph_enabled",
                )
            )
            aggregates[key].append(row)
    summaries = []
    for key, values in sorted(aggregates.items()):
        require(
            len({v["repeat"] for v in values}) == len(values),
            "Duplicate repeat in paired summary",
        )
        summaries.append(
            dict(
                zip(
                    (
                        "baseline",
                        "changed",
                        "input_len",
                        "client_concurrency",
                        "actual_batch_size",
                        "graph_enabled",
                    ),
                    key,
                ),
                paired_repeats=len(values),
                baseline_gap_ms=summary(
                    v["baseline_mean_launch_gap_ms"] for v in values
                ),
                changed_gap_ms=summary(v["changed_mean_launch_gap_ms"] for v in values),
                launch_gap_reduction_percent=paired_statistics(
                    [v["launch_gap_reduction_percent"] for v in values]
                ),
            )
        )
    return rows, summaries


def capacity_rows(samples, *, min_intervals=16):
    """Highest *observed* stable batches, without equating concurrency to batch."""
    grouped = defaultdict(list)
    for sample in samples:
        batches = sorted(
            {value["actual_batch_size"] for value in sample["_intervals"].values()}
        )
        for batch in batches:
            intervals = [
                v
                for v in sample["_intervals"].values()
                if v["actual_batch_size"] == batch
            ]
            if len(intervals) < min_intervals:
                continue
            gap = statistics.mean(v["gap_ms"] for v in intervals)
            key = (
                sample["arm"],
                sample["input_len"],
                sample["client_concurrency"],
                batch,
                sample["graph_enabled"],
            )
            grouped[key].append(
                {
                    "repeat": sample["repeat"],
                    "launch_gap_ms": gap,
                    "stable_tokens_per_second": batch * 1000 / gap,
                    "observed_intervals": len(intervals),
                }
            )
    rows = []
    for key, values in sorted(grouped.items()):
        require(
            len({v["repeat"] for v in values}) == len(values),
            "Duplicate repeat in capacity point",
        )
        rows.append(
            dict(
                zip(
                    (
                        "arm",
                        "input_len",
                        "client_concurrency",
                        "actual_batch_size",
                        "graph_enabled",
                    ),
                    key,
                ),
                observed_repeats=len(values),
                repeat_ids=[v["repeat"] for v in values],
                mean_launch_gap_ms=summary(v["launch_gap_ms"] for v in values),
                stable_tokens_per_second=summary(
                    v["stable_tokens_per_second"] for v in values
                ),
                scope="Observed constant-request decode intervals; this is not full-request throughput or an established hardware maximum",
            )
        )
    return rows


def union_ns(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (right - left if left is not None else 0)


def _kind(name):
    low = name.lower()
    if "quest_score" in low:
        return "score"
    if "load_cache_to_device_buffer" in low or "plan_cache" in low:
        return "cache_plan_or_swap"
    if "quest_scatter_pack" in low:
        return "packing"
    if (
        "xm_hisparse_tma::attention_kernel" in low
        or "xm_hisparse_tma::merge_kernel" in low
    ):
        return "attention"
    if "flashinfer::" in low and any(
        k in low for k in ("batchdecode", "batchprefill", "mergestates", "merge_states")
    ):
        return "attention"
    return "other"


def attention_profile(database, scope, *, layer_index=23):
    """A.2-style spans only from a passed, hashed stable decode scope receipt."""
    database = Path(database).resolve(strict=True)
    require(
        scope.get("status") == "passed"
        and scope.get("actual_decode_scope_verified") is True,
        "Passed decode scope receipt required",
    )
    require(
        identity(database)["sha256"] == scope.get("sqlite_sha256"),
        "SQLite differs from verified scope",
    )
    layers, graph, count = (
        scope.get("calls_per_graph"),
        scope.get("graph_id"),
        scope.get("actual_decode_graph_launches"),
    )
    require(
        type(layers) is int and layers > 0 and 0 <= layer_index < layers,
        "Invalid layer geometry",
    )
    require(
        type(graph) is int and graph > 0 and type(count) is int and count > 0,
        "Invalid decode graph identity",
    )
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        kernels = list(
            connection.execute(
                "SELECT k.start,k.end,k.deviceId,k.streamId,k.correlationId,k.graphId,s.value "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
            )
        )
    finally:
        connection.close()
    selected = [row for row in kernels if row[5] == graph and row[4] > 0]
    by_launch = defaultdict(list)
    for start, end, device, stream, correlation, graph_id, name in selected:
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and isinstance(name, str),
            "Invalid kernel activity",
        )
        by_launch[correlation].append(
            {
                "start": start,
                "end": end,
                "device": device,
                "stream": stream,
                "name": name,
                "kind": _kind(name),
            }
        )
    require(
        len(by_launch) == count,
        "Observed main graph launches differ from verified scope",
    )
    spans, named = [], defaultdict(list)
    for correlation, rows in sorted(by_launch.items()):
        scores = [row for row in rows if row["kind"] == "score"]
        require(
            len(scores) == layers, "Cannot establish complete per-layer score sequence"
        )
        start = scores[layer_index]["start"]
        stop = (
            scores[layer_index + 1]["start"]
            if layer_index + 1 < layers
            else max(r["end"] for r in rows) + 1
        )
        segment = [r for r in rows if start <= r["start"] < stop]
        plan = [r for r in segment if r["kind"] == "cache_plan_or_swap"]
        attention = [r for r in segment if r["kind"] == "attention"]
        pack = [r for r in segment if r["kind"] == "packing"]
        require(
            len(plan) == 1 and attention,
            "Ambiguous/missing cache plan or attention in selected layer",
        )
        require(
            len({r["device"] for r in segment}) == 1, "Layer contains multiple devices"
        )
        first_plan, first_attention, end = (
            plan[0]["start"],
            min(r["start"] for r in attention),
            max(r["end"] for r in attention),
        )
        require(
            start <= first_plan <= first_attention < end < stop,
            "Non-monotonic layer path boundaries",
        )
        boundaries = [("selection", start, first_plan)]
        if pack:
            first_pack = min(r["start"] for r in pack)
            require(
                first_plan <= first_pack <= first_attention, "Ambiguous pack ordering"
            )
            boundaries.extend(
                (
                    ("cache_plan_or_swap", first_plan, first_pack),
                    ("packing", first_pack, first_attention),
                )
            )
        else:
            boundaries.append(("cache_plan_or_swap", first_plan, first_attention))
        boundaries.append(("attention_including_merge", first_attention, end))
        stages = {}
        for label, left, right in boundaries:
            active = [
                (max(left, r["start"]), min(right, r["end"]))
                for r in segment
                if r["start"] < right and r["end"] > left
            ]
            occupied = union_ns(active)
            stages[label] = {
                "span_us": (right - left) / 1000,
                "kernel_active_union_us": occupied / 1000,
                "internal_idle_gap_us": ((right - left) - occupied) / 1000,
            }
        for row in segment:
            if row["start"] <= end:
                named[row["name"]].append((row["end"] - row["start"]) / 1000)
        spans.append(
            {
                "correlation_id": correlation,
                "layer_index": layer_index,
                "total_path_span_us": (end - start) / 1000,
                "stages": stages,
            }
        )
    labels = sorted({key for span in spans for key in span["stages"]})
    return {
        "schema_version": 1,
        "status": "analyzed",
        "sqlite": identity(database),
        "actual_batch_size": scope.get("actual_batch_size"),
        "layer_index": layer_index,
        "observed_decode_steps": len(spans),
        "total_path_span_us": summary(s["total_path_span_us"] for s in spans),
        "stage_span_us": {
            label: summary(s["stages"].get(label, {}).get("span_us", 0) for s in spans)
            for label in labels
        },
        "kernel_mean_duration_us": {
            name: summary(values) for name, values in sorted(named.items())
        },
        "per_step": spans,
        "scope": "First Quest score to last attention/split-merge end; contiguous stage spans include internal idle gaps. Kernel sums are not critical-path time.",
        "limitations": [
            "Requires independent passed source/trace decode-scope and device-role audits.",
            "Prefetch dependency wait before first score is outside this path.",
            "Missing child memcpy events do not establish copy-engine cost or absence.",
        ],
    }


def analyze_campaign(path, **kwargs):
    campaign, samples, artifacts, excluded, commits = collect_campaign(path, **kwargs)
    paired, aggregates = same_batch_pairs(samples)
    public = [
        {key: value for key, value in sample.items() if key != "_intervals"}
        for sample in samples
    ]
    flattened = flat_rows(samples)
    return {
        "schema_version": 1,
        "status": (
            "analyzed"
            if any(r["eligible"] for r in flattened)
            else "no_eligible_samples"
        ),
        "phase": campaign["phase"],
        "source_fingerprint": campaign.get("source_fingerprint"),
        "native_source_commits": commits,
        "rows": flattened,
        "workload_identity": campaign.get("workload_identity"),
        "samples": public,
        "paired_repeat_rows": paired,
        "same_actual_batch_comparisons": aggregates,
        "excluded_cases": excluded,
        "artifacts": artifacts,
        "observed_stable_capacity_points": capacity_rows(samples),
        "cache_hbm_budget_bytes": campaign.get("settings", {}).get(
            "cache_budget_bytes"
        ),
        "appendix": {
            "status": "pending_independent_profile",
            "measured_path_breakdown": False,
        },
        "hash_scope": "Content hashes recorded at this replay; saved derived analyses verified against raw replay. Preserve earlier custody receipts separately.",
        "interpretation": "Qwen Quest-summary adaptation on one H800/CPU DDR. Capacity primary comparison fused→combined-protected; baseline→combined also changes the attention workspace.",
        "limitations": [
            "Client concurrency is not actual decode batch.",
            "No cost/SLO thresholds supplied; no economic optimum is inferred.",
            "TTFT and request-average TPOT are client metrics; stable launch gaps use scheduler time.",
            "Small-repeat confidence intervals rely on independent paired repeats; execution-order balance must be checked separately.",
        ],
    }


def write_outputs(report, directory, *, plot=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = report["paired_repeat_rows"]
    if rows:
        with (directory / "paired-repeats.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if report["rows"]:
        with (directory / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = report["same_actual_batch_comparisons"]
        if data:
            figure, axis = plt.subplots(figsize=(10, max(3, len(data) * 0.55)))
            labels = [
                f"{r['baseline']} → {r['changed']} | {r['input_len'] // 1024}K, B={r['actual_batch_size']}"
                for r in data
            ]
            values = [r["launch_gap_reduction_percent"]["mean"] for r in data]
            axis.barh(
                range(len(data)),
                values,
                color=["#357d9e" if v >= 0 else "#be5959" for v in values],
            )
            axis.axvline(0, color="#555555", linewidth=0.8)
            axis.set_yticks(range(len(data)), labels)
            axis.set_xlabel("Matched stable decode launch-gap reduction (%)")
            axis.set_title("Qwen/Quest adaptation on H800 PCIe + CPU DDR")
            figure.tight_layout()
            for extension in ("png", "svg"):
                figure.savefig(directory / ("same-batch-effects." + extension), dpi=180)
            plt.close(figure)
        points = report["observed_stable_capacity_points"]
        if points:
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            series = defaultdict(list)
            for row in points:
                series[(row["arm"], row["input_len"])].append(row)
            for (arm, length), rows in sorted(series.items()):
                rows.sort(
                    key=lambda row: (
                        row["actual_batch_size"],
                        row["client_concurrency"],
                    )
                )
                axis.plot(
                    [r["actual_batch_size"] for r in rows],
                    [r["stable_tokens_per_second"]["mean"] for r in rows],
                    "o-",
                    label=f"{arm}, {length // 1024}K",
                )
            axis.set_xlabel("Observed stable decode batch (requests)")
            axis.set_ylabel("Stable scheduler-span throughput (tokens/s)")
            axis.set_title(
                "Measured scan points; client concurrency is recorded separately"
            )
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
            figure.tight_layout()
            for extension in ("png", "svg"):
                figure.savefig(directory / ("observed-capacity." + extension), dpi=180)
            plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("campaign")
    campaign.add_argument("--campaign", type=Path, required=True)
    campaign.add_argument("--evidence-root", type=Path)
    campaign.add_argument("--output-dir", type=Path, required=True)
    campaign.add_argument("--plot", action="store_true")
    profile = commands.add_parser("profile")
    profile.add_argument("--sqlite", type=Path, required=True)
    profile.add_argument("--decode-scope", type=Path, required=True)
    profile.add_argument("--layer-index", type=int, default=23)
    profile.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "campaign":
        report = analyze_campaign(args.campaign, evidence_root=args.evidence_root)
        write_outputs(report, args.output_dir, plot=args.plot)
    else:
        try:
            report = attention_profile(
                args.sqlite, read(args.decode_scope), layer_index=args.layer_index
            )
        except (ValueError, sqlite3.Error) as error:
            report = {
                "status": "pending_scope_verification",
                "reason": str(error),
                "measured_path_breakdown": False,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
