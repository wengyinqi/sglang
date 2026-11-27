"""Render audited XM/Qwen ablations, keeping observed and missing evidence apart.

This command consumes analysis JSONs; it never runs a model or invents a scan
point. Performance, capacity, independent profiles, and isolated kernel probes
retain their own scopes. The original report can be appended without changing
its pages. All input/output hashes and the preservation check are written to a
sidecar receipt. ReportLab, pypdf, and matplotlib are report-only dependencies.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

ARMS = (
    "baseline",
    "fused",
    "bounds-on-demand",
    "bounds-prefetch",
    "combined",
    "combined-protected",
)
ARM_LABELS = {
    "baseline": "HiSparse 基线",
    "fused": "融合 TMA",
    "bounds-on-demand": "摘要按需",
    "bounds-prefetch": "摘要预取",
    "combined": "融合＋预取",
    "combined-protected": "融合＋保护预取",
}
MIB = 2**20
GIB = 2**30
DEFAULT_FONT = Path("C:/Windows/Fonts/simhei.ttf")
CATEGORY_LABELS = {
    "hisparse_lru_bytes": "HiSparse LRU 槽位／初始化",
    "hisparse_host_mapping_bytes": "请求→DDR KV 映射",
    "request_token_mapping_bytes": "请求 token 映射",
    "hisparse_logical_to_physical_bytes": "逻辑→物理地址映射",
    "hisparse_hot_request_slots_bytes": "请求→hot 槽位表",
    "hisparse_hot_token_ids_bytes": "hot token ID",
    "hisparse_hot_token_locs_bytes": "hot token location",
    "hisparse_selection_output_bytes": "选择结果 device locs",
    "quest_selection_state_bytes": "Quest 选择状态／常量",
    "quest_running_metadata_bytes": "running 页号／token count",
    "quest_prefetch_control_bytes": "预取 epoch／gate／ready／计数",
    "hisparse_batch_state_bytes": "实际 batch 状态",
    "hisparse_diagnostic_counter_bytes": "cache 诊断计数器",
    "compute_kv_descriptors_bytes": "计算端 KV 地址 descriptors",
    "backing_kv_descriptors_bytes": "DDR KV 地址 descriptors",
    "allocator_physical_worklists_bytes": "物理 allocator worklists",
    "allocator_logical_worklists_bytes": "逻辑 allocator worklists",
    "allocator_outer_worklists_bytes": "外层 allocator worklists",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cohort_stage_enabled(stage):
    options = stage.get("server_options", {})
    require(isinstance(options, dict), "Malformed cohort stage server_options")
    enabled = options.get("xm_correctness_cohort_barrier", False)
    require(type(enabled) is bool, "Cohort barrier recipe must be a boolean")
    return enabled


def verify_cohort_command(stage, group, info):
    """Bind the declared control to actual argv and observed server settings."""
    enabled = cohort_stage_enabled(stage)
    command = group.get("server_command", [])
    require(isinstance(command, list), "Missing actual cohort server command")
    flag = "--xm-correctness-cohort-barrier"
    require(
        command.count(flag) == int(enabled),
        "Actual cohort server command differs from its declared recipe",
    )
    require(
        info.get("xm_correctness_cohort_barrier", False) is enabled,
        "Observed cohort barrier mode differs from the actual recipe",
    )
    if enabled:
        timeout = stage["server_options"].get("xm_correctness_cohort_timeout_seconds")
        number(timeout, positive=True)
        timeout_flag = "--xm-correctness-cohort-timeout-seconds"
        require(
            command.count(timeout_flag) == 1
            and command.index(timeout_flag) + 1 < len(command)
            and command[command.index(timeout_flag) + 1] == str(timeout)
            and info.get("xm_correctness_cohort_timeout_seconds") == timeout,
            "Observed cohort timeout differs from its finite declared control",
        )
    return enabled


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def number(value, *, positive=False):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "Expected a finite numeric measurement",
    )
    require(value > 0 if positive else value >= 0, "Negative/zero measurement")
    return value


def fmt(value, digits=2):
    return "未观测" if value is None else f"{value:,.{digits}f}"


def ranges(values, divisor=1, digits=2):
    values = [value / divisor for value in values if value is not None]
    if not values:
        return "未观测"
    low, high = min(values), max(values)
    return (
        fmt(low, digits) if low == high else f"{fmt(low, digits)} - {fmt(high, digits)}"
    )


def validate_analyses(paths, *, internal_fixture=False):
    result, fingerprints, workloads, commits = [], set(), set(), set()
    for path in paths:
        data = read(path)
        require(data.get("schema_version") == 1, "Unsupported analysis schema")
        require(
            data.get("phase") in ("performance", "capacity"),
            "Expected performance/capacity analysis",
        )
        require(
            data.get("status") in ("analyzed", "no_eligible_samples"),
            "Analysis did not finish",
        )
        require(
            not data.get("internal_layout_fixture") or internal_fixture,
            "Layout fixture cannot be used in a real report",
        )
        source = data.get("source_fingerprint", {}).get("sha256", "")
        workload = data.get("workload_identity", {}).get("sha256", "")
        require(
            re.fullmatch("[0-9a-f]{64}", source) is not None,
            "Missing source fingerprint",
        )
        require(
            re.fullmatch("[0-9a-f]{64}", workload) is not None,
            "Missing frozen workload fingerprint",
        )
        fingerprints.add(source)
        workloads.add(workload)
        for commit in data.get("native_source_commits", []):
            require(
                re.fullmatch("[0-9a-f]{40}", commit) is not None,
                "Missing native source revision",
            )
            commits.add(commit)
        seen = set()
        for row in data.get("rows", []):
            require(row.get("arm") in ARMS, "Unknown experimental arm")
            key = tuple(
                row.get(k)
                for k in (
                    "arm",
                    "input_len",
                    "client_concurrency",
                    "actual_batch",
                    "repeat",
                )
            )
            require(key not in seen, "Duplicate flat sample")
            seen.add(key)
            require(type(row.get("eligible")) is bool, "Eligibility is missing")
            if row["eligible"]:
                number(row.get("mean_tbt_ms"), positive=True)
                number(row.get("stable_tokens_per_s"), positive=True)
                require(
                    type(row.get("actual_batch")) is int and row["actual_batch"] > 0,
                    "Missing actual batch",
                )
            if "client_all_frames_single_token" in row:
                strict = row.get("strict_client_mean_tbt_ms")
                if row["client_all_frames_single_token"] is not True:
                    require(
                        strict is None and row.get("strict_client_p99_tbt_ms") is None,
                        "Multi-token client frames cannot supply strict TBT",
                    )
                elif strict is not None:
                    number(strict, positive=True)
                    number(row.get("strict_client_p99_tbt_ms"), positive=True)
        for sample in data.get("samples", []):
            require(sample.get("arm") in ARMS, "Unknown sample arm")
            client = sample.get("client_timing", {})
            strict = client.get("single_token_arrival_gap_ms")
            if client.get("all_frames_single_token") is not True:
                require(
                    strict is None,
                    "Multi-token frames contain fabricated strict arrival gaps",
                )
            elif strict is not None:
                number(strict.get("mean"), positive=True)
                number(strict.get("p99"), positive=True)
            for boundary in ("memory_before", "memory_after"):
                memory = sample.get(boundary, {})
                number(memory.get("cache_named_gpu_bytes"))
                components = memory.get("cache_components", {})
                backend = memory.get("backend_components", {})
                require(
                    sum(components.values()) == memory["cache_named_gpu_bytes"],
                    "Cache component total differs",
                )
                require(
                    "fused_plan_bytes" not in backend,
                    "Backend plan duplicates the cache ledger plan",
                )
                for value in list(components.values()) + list(backend.values()):
                    number(value)
                require(
                    memory.get("torch_named_total_including_observed_backend_bytes")
                    == memory["cache_named_gpu_bytes"] + sum(backend.values()),
                    "Combined named total differs",
                )
                if (
                    memory.get("allocation_accounting_schema")
                    == "xm_qwen_named_allocations_v2"
                ):
                    for owner in ("cache", "backend"):
                        grouped = defaultdict(int)
                        ids = set()
                        for record in memory.get(owner + "_allocation_groups", []):
                            require(
                                type(record.get("allocation_group")) is int
                                and record["allocation_group"] not in ids,
                                "Repeated or missing allocation group",
                            )
                            require(
                                record.get("alias_names"),
                                "Allocation owner names missing",
                            )
                            ids.add(record["allocation_group"])
                            number(record.get("storage_bytes"))
                            grouped[record["category"]] += record["storage_bytes"]
                        require(
                            dict(grouped) == memory[owner + "_components"],
                            "Named owner records differ from component totals",
                        )
        for row in data.get("same_actual_batch_comparisons", []):
            require(
                row.get("baseline") in ARMS and row.get("changed") in ARMS,
                "Unknown comparison arm",
            )
            require(
                type(row.get("paired_repeats")) is int and row["paired_repeats"] > 0,
                "Comparison repeats are missing",
            )
            number(row.get("baseline_gap_ms", {}).get("mean"), positive=True)
            number(row.get("changed_gap_ms", {}).get("mean"), positive=True)
            effect = row.get("launch_gap_reduction_percent", {})
            require(
                type(effect.get("mean")) in (int, float)
                and math.isfinite(effect["mean"]),
                "Comparison effect is not finite",
            )
            interval = effect.get("confidence_interval_95")
            if interval is not None:
                require(
                    row["paired_repeats"] >= 2
                    and len(interval) == 2
                    and all(
                        type(v) in (int, float) and math.isfinite(v) for v in interval
                    )
                    and interval[0] <= effect["mean"] <= interval[1],
                    "Invalid repeat confidence interval",
                )
        result.append((Path(path).resolve(), data))
    require(
        len(fingerprints) == 1 and len(workloads) == 1,
        "Report cannot combine different implementation/workload fingerprints",
    )
    require(len(commits) <= 1, "Report cannot combine native source revisions")
    return result


def report_source(analyses, evidence):
    if analyses:
        return analyses[0][1]
    for _, record in evidence:
        if record.get("phase") == "correctness":
            stages = [case.get("stage") for case in record.get("cases", [])]
            commits = {
                native.get("runtime", {}).get("git_commit")
                for _, native in evidence
                if native.get("kind") == "correctness"
                and native.get("stage") in stages
                and native.get("runtime", {})
                .get("archived_source", {})
                .get("source_commit")
                == native.get("runtime", {}).get("git_commit")
            }
            commits = {
                commit
                for commit in commits
                if isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit)
            }
            return dict(
                record,
                native_source_commits=sorted(commits) if len(commits) == 1 else [],
            )
    return {}


def report_tool(name):
    spec = importlib.util.spec_from_file_location(
        "xm_report_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_binding_api():
    return SimpleNamespace(
        identity=identity,
        report_tool=report_tool,
        _native_correctness=_native_correctness,
        measurement_graph=measurement_graph,
    )


def _native_correctness(proof_path, case, evidence):
    declared = Path(case.get("stage_path", "")).parent / "evidence/manifest.json"
    if declared.is_file():
        return declared, read(declared)
    candidates = [
        (path, record)
        for path, record in evidence
        if record.get("kind") == "correctness"
        and record.get("stage") == case.get("stage")
    ]
    require(
        len(candidates) == 1,
        f"Missing unambiguous native correctness for {case.get('arm')}",
    )
    return candidates[0]


def measurement_graph(data, row, *, arm=None):
    if type(row.get("graph_enabled")) is bool:
        return row["graph_enabled"]
    arm = arm or row.get("arm")
    matches = {
        sample.get("graph_enabled")
        for sample in data.get("samples", [])
        if sample.get("arm") == arm
        and all(
            sample.get(key) == row.get(key)
            for key in ("input_len", "client_concurrency", "repeat")
        )
    }
    require(
        len(matches) == 1 and all(type(value) is bool for value in matches),
        "Missing unambiguous actual measurement Graph mode",
    )
    return matches.pop()


def measurement_scope(data):
    arms, lengths, graph = set(), set(), False
    rows = data.get("rows", []) + data.get("samples", [])
    observed = bool(
        data.get("samples")
        or data.get("same_actual_batch_comparisons")
        or data.get("observed_stable_capacity_points")
    ) or any(row.get("eligible") is True for row in rows)
    for row in rows:
        arms.add(row["arm"])
        lengths.add(row["input_len"])
        graph |= measurement_graph(data, row)
    for row in data.get("same_actual_batch_comparisons", []):
        arms.update((row["baseline"], row["changed"]))
        lengths.add(row["input_len"])
        graph |= measurement_graph(data, row, arm=row["baseline"])
    for row in data.get("observed_stable_capacity_points", []):
        arms.add(row["arm"])
        lengths.add(row["input_len"])
        graph |= measurement_graph(data, row)
    return observed, arms, lengths, graph


def replay_correctness_execution(native_path, native, *, required_batches=()):
    """Replay raw single-compute rounds; requested concurrency is not batch proof.

    ``required_batches`` contains ``(context_length, actual_batch, graph_enabled)``.
    B1 can be covered by a member of a mixed-context cohort. B>1 requires the
    same-context complete cohort in both long rounds, including 16 consecutive
    decode forwards. Graph evidence is server mode/captured shape/cohort only.
    """
    tool = report_tool("summary_batch_correctness")
    compare, run = tool.CORRECTNESS.COMPARE, tool.CORRECTNESS.RUN
    settings = native.get("settings", {})
    output_len, atol = settings.get("output_len"), settings.get("logprob_atol")
    require(
        type(output_len) is int and output_len >= 2, "Missing correctness output length"
    )
    number(atol)
    require(atol <= 0.02, "Correctness replay cannot loosen frozen logprob tolerance")
    groups = native.get("groups", [])
    require(
        groups and groups[0].get("variant") == "cpu",
        "Missing first CPU correctness reference",
    )
    require(
        len({g.get("id") for g in groups}) == len(groups),
        "Duplicate correctness group identity",
    )
    for _, _, graph in required_batches:
        require(
            {g.get("variant") for g in groups if g.get("eager") is not graph}
            >= {"cpu", "hbm"},
            "Required actual correctness Graph/eager CPU/HBM groups are missing",
        )
    reference, reference_inputs, receipts, used_ids = None, None, [], set()
    for group in groups:
        require(
            group.get("status") == "success",
            "Cannot replay failed correctness group as passed",
        )
        gid = group.get("id")
        require(
            isinstance(gid, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", gid),
            "Missing safe correctness group identity",
        )
        root = Path(native_path).parent
        declared = Path(group.get("trace_dir", ""))
        trace_dir = declared if declared.is_dir() else root / (gid + "-trace")
        paths = sorted(trace_dir.glob("*.jsonl"))
        require(
            len(paths) == 1, "Require one complete single-compute correctness trace"
        )
        trace = [
            json.loads(line)
            for line in paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(
            trace and trace[-1].get("type") == "idle_flush",
            "Missing final correctness idle flush",
        )
        run.ANALYZE._validate_session(trace[0])
        require(
            trace[0].get("disable_overlap_schedule") is True,
            "Correctness replay requires disabled overlap scheduler",
        )
        require(
            sum(row.get("type") == "session" for row in trace) == 1,
            "Correctness trace has multiple scheduler lifetimes",
        )
        forwards = [row for row in trace if row.get("type") == "forward"]
        require(
            forwards
            and [row.get("sequence") for row in forwards] == list(range(len(forwards)))
            and trace[-1].get("last_sequence") == len(forwards) - 1,
            "Incomplete or reordered correctness forward trace",
        )
        info_path = root / f"{gid}-server-info.json"
        info = read(info_path)
        cohort_barrier = verify_cohort_command(native.get("stage", {}), group, info)
        enabled = run.decode_graph_enabled(info)
        require(
            type(group.get("eager")) is bool and enabled is not group["eager"],
            "Observed server Graph mode differs from correctness group",
        )
        states = info.get("internal_states", [])
        require(len(states) == 1, "Missing single-compute correctness server state")
        shapes = states[0].get("cuda_graph_bs", [])
        rounds = group.get("rounds", [])
        require(
            len(rounds) == 3 and rounds[0] == rounds[2],
            "Correctness requires complete long-short-long rounds",
        )
        require(
            set(settings.get("input_lens", [])) <= set(rounds[0]),
            "Correctness long round misses declared contexts",
        )
        normalized, inputs, round_receipts = [], [], []
        for index, lengths in enumerate(rounds):
            response_path = root / f"{gid}-round{index}.json"
            payload = read(response_path)
            outputs = payload.get("responses")
            records = compare.normalize_outputs(outputs, lengths, output_len)
            prompt_ids = payload.get("input_ids")
            require(
                isinstance(prompt_ids, list)
                and len(prompt_ids) == len(lengths)
                and all(
                    isinstance(ids, list)
                    and len(ids) == n
                    and all(type(token) is int for token in ids)
                    for ids, n in zip(prompt_ids, lengths)
                ),
                "Raw correctness inputs differ from requested prompt lengths",
            )
            require(
                all(
                    output.get("output_ids") == record["tokens"]
                    for output, record in zip(outputs, records)
                ),
                "Raw output token IDs differ from logprob token IDs",
            )
            ids = [output["meta_info"].get("id") for output in outputs]
            require(
                all(isinstance(rid, str) and rid for rid in ids)
                and len(set(ids)) == len(ids)
                and not used_ids.intersection((gid, rid) for rid in ids),
                "Missing or reused correctness request identities",
            )
            used_ids.update((gid, rid) for rid in ids)
            members = set(ids)
            rows = [
                row
                for row in forwards
                if members.intersection(row.get("request_ids", []))
            ]
            require(
                rows
                and all(
                    row.get("success") is True
                    and set(row.get("request_ids", [])) <= members
                    and len(row["request_ids"]) == len(set(row["request_ids"]))
                    for row in rows
                ),
                "Mixed, duplicate, or failed correctness round forwards",
            )
            streams = {row.get("stream_id") for row in rows}
            require(
                len(streams) == 1 and None not in streams,
                "Correctness round requires one actual compute stream",
            )
            require(
                not any(
                    rows[0]["sequence"] <= row["sequence"] <= rows[-1]["sequence"]
                    and not members.intersection(row.get("request_ids", []))
                    for row in forwards
                ),
                "Unrelated forwards overlap correctness round",
            )
            prefill, decode = dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
            for row in rows:
                if row.get("forward_mode") == "extend":
                    chunks = row.get("extend_lens")
                    require(
                        isinstance(chunks, list)
                        and len(chunks) == len(row["request_ids"])
                        and all(type(n) is int and n > 0 for n in chunks),
                        "Missing actual correctness prefill chunks",
                    )
                    require(
                        len(chunks) <= settings.get("prefill_max_requests", 0)
                        and sum(chunks) <= settings.get("chunked_prefill_size", 0),
                        "Correctness prefill exceeded guarded request/chunk configuration",
                    )
                    for rid, count in zip(row["request_ids"], chunks):
                        prefill[rid] += count
                elif row.get("forward_mode") == "decode":
                    if cohort_barrier:
                        require(
                            set(row["request_ids"]) == members
                            and prefill == dict(zip(ids, lengths))
                            and len(set(decode.values())) == 1,
                            "Controlled correctness decode released a partial or ordinal-misaligned cohort",
                        )
                    for rid in row["request_ids"]:
                        decode[rid] += 1
                else:
                    raise ValueError("Unknown correctness forward mode")
            require(
                prefill == dict(zip(ids, lengths))
                and decode == dict.fromkeys(ids, output_len - 1),
                "Actual correctness prefill/decode counts differ from complete responses",
            )
            full = [
                row["sequence"]
                for row in rows
                if row["forward_mode"] == "decode"
                and set(row["request_ids"]) == members
            ]
            streak = longest = 0
            previous = None
            for sequence in full:
                streak = (
                    streak + 1
                    if previous is not None and sequence == previous + 1
                    else 1
                )
                longest = max(longest, streak)
                previous = sequence
            round_receipts.append(
                {
                    "round": index,
                    "context_lengths": lengths,
                    "request_ids": ids,
                    "response": identity(response_path),
                    "observed_actual_batches": sorted(
                        {
                            len(row["request_ids"])
                            for row in rows
                            if row["forward_mode"] == "decode"
                        }
                    ),
                    "full_actual_decode_sequences": full,
                    "max_consecutive_full_forwards": longest,
                    "complete_request_decode_forwards": decode,
                    "request_observed_actual_batches": {
                        rid: sorted(
                            {
                                len(row["request_ids"])
                                for row in rows
                                if row["forward_mode"] == "decode"
                                and rid in row["request_ids"]
                            }
                        )
                        for rid in ids
                    },
                }
            )
            normalized.append(records)
            inputs.append(prompt_ids)
        require(inputs[0] == inputs[2], "Long-round slot reuse inputs changed")
        if reference is None:
            reference, reference_inputs = normalized, inputs
        require(inputs == reference_inputs, "CPU/HBM correctness prompts differ")
        recomputed = compare.compare(reference, normalized, atol)
        reuse = compare.compare([normalized[0]], [normalized[2]], atol)
        require(
            recomputed == group.get("comparison_to_cpu_hisparse")
            and reuse == group.get("slot_reuse_comparison"),
            "Correctness token/logprob or slot-reuse receipt differs from raw responses",
        )
        for length, batch, graph in required_batches:
            if enabled != graph:
                continue
            for item in (round_receipts[0], round_receipts[2]):
                require(
                    length in item["context_lengths"],
                    "Measured context lacks complete actual correctness execution",
                )
                if graph:
                    require(
                        all(
                            set(
                                item["request_observed_actual_batches"][rid]
                            ).intersection(shapes)
                            for rid, n in zip(
                                item["request_ids"], item["context_lengths"]
                            )
                            if n == length
                        ),
                        "Actual correctness cohort shape is absent from Graph capture configuration",
                    )
                if batch > 1:
                    require(
                        item["context_lengths"] == [length] * batch
                        and item["max_consecutive_full_forwards"] >= 16,
                        f"Actual B={batch}, input={length} lacks 16 consecutive complete cohort decode forwards in both long correctness rounds",
                    )
                    require(
                        not graph or batch in shapes,
                        "Actual correctness batch shape is absent from Graph capture configuration",
                    )
        receipts.append(
            {
                "id": gid,
                "variant": group.get("variant"),
                "eager": group["eager"],
                "actual_graph_enabled": enabled,
                "captured_shapes": shapes,
                "cohort_barrier": cohort_barrier,
                "trace": identity(paths[0]),
                "server_info": identity(info_path),
                "rounds": round_receipts,
                "comparison_to_cpu_hisparse": recomputed,
                "slot_reuse_comparison": reuse,
            }
        )
    return {
        "kind": "cpu_replay_actual_correctness_batch",
        "status": "passed",
        "graph_scope": "Actual server mode, captured shapes and request cohort; this observer emits no per-forward Graph replay flag",
        "groups": receipts,
    }


def _required_correctness_batches(data, arm):
    if data.get("phase") != "performance":
        return []
    required = set()
    for row in data.get("rows", []):
        if row.get("arm") == arm and row.get("eligible") is True:
            required.add(
                (row["input_len"], row["actual_batch"], measurement_graph(data, row))
            )
    for row in data.get("same_actual_batch_comparisons", []):
        if arm in (row.get("baseline"), row.get("changed")):
            required.add(
                (
                    row["input_len"],
                    row["actual_batch_size"],
                    measurement_graph(data, row, arm=arm),
                )
            )
    return sorted(required)


def verify_model_receipt(proof_path, proof, evidence, data, benchmark_models):
    _, arms, lengths, graph = measurement_scope(data)
    require(
        proof.get("status") == "success",
        "Bound model correctness campaign did not pass",
    )
    require(
        proof.get("source_fingerprint") == data.get("source_fingerprint")
        and (proof.get("workload_identity") or {}).get("sha256")
        == (data.get("workload_identity") or {}).get("sha256"),
        "Bound model proof has another implementation/workload",
    )
    require(
        benchmark_models == {proof.get("settings", {}).get("model")},
        "Measured and correctness model identities differ or native benchmark manifests are missing",
    )
    require(
        lengths <= set(proof.get("settings", {}).get("input_lens", [])),
        "Measured contexts were not covered by model correctness",
    )
    require(
        not graph or not proof.get("settings", {}).get("disable_cuda_graph"),
        "Measured Graph mode was not covered by correctness",
    )
    by_arm = {case.get("arm"): case for case in proof.get("cases", [])}
    require(
        len(by_arm) == len(proof.get("cases", [])),
        "Duplicate model correctness arm receipt",
    )
    accepted = []
    for arm in sorted(arms):
        case = by_arm.get(arm, {})
        require(
            case.get("status") == "success" and case.get("returncode") == 0,
            f"No passed model correctness for {arm}",
        )
        native_path, native = _native_correctness(proof_path, case, evidence)
        budget = (
            case.get("stage", {})
            .get("variants", {})
            .get("cpu", {})
            .get("hisparse_config", {})
            .get("xm_cache_hbm_budget_bytes")
        )
        require(
            case.get("stage")
            == report_tool("campaign").recipe(
                arm,
                budget=budget,
                cohort_barrier=cohort_stage_enabled(case.get("stage", {})),
            )
            and cohort_stage_enabled(case.get("stage", {}))
            is proof.get("settings", {}).get("xm_correctness_cohort_barrier", False),
            "Correctness case is not the exact declared arm recipe",
        )
        settings, runtime = native.get("settings", {}), native.get("runtime", {})
        require(
            native.get("status") == "success"
            and native.get("kind") == "correctness"
            and native.get("stage") == case.get("stage")
            and settings.get("execute") is True,
            "Native correctness was not an actual successful exact-stage execution",
        )
        require(
            settings.get("model") == proof.get("settings", {}).get("model"),
            "Correctness model differs between campaign and native receipt",
        )
        require(
            settings.get("workload_sha256")
            == data.get("workload_identity", {}).get("sha256"),
            "Native correctness workload differs",
        )
        require(
            runtime.get("git_commit") in data["native_source_commits"],
            "Native correctness revision differs from measured results",
        )
        marker = runtime.get("archived_source", {})
        require(
            marker.get("provenance") == "git_archive"
            and marker.get("source_commit") == runtime.get("git_commit")
            and re.fullmatch("[0-9a-f]{64}", marker.get("source_archive_sha256", "")),
            "Missing exact archived native source provenance",
        )
        modes = {True, False} if graph else {True}
        groups = native.get("groups", [])
        require(
            {(g.get("variant"), g.get("eager")) for g in groups}
            >= {(variant, eager) for variant in ("cpu", "hbm") for eager in modes},
            "Missing independent CPU/HBM or eager/Graph correctness groups",
        )
        for group in groups:
            require(
                group.get("status") == "success",
                "Native correctness contains failed groups",
            )
            for key in ("comparison_to_cpu_hisparse", "slot_reuse_comparison"):
                comparison = group.get(key, {})
                require(
                    comparison.get("exact_greedy_token_match") is True,
                    "Missing exact token/slot reuse comparison",
                )
                atol, error = comparison.get("logprob_atol"), comparison.get(
                    "max_abs_logprob_error"
                )
                number(atol)
                number(error)
                require(
                    atol <= 0.02 and error <= atol,
                    "Model logprob comparison did not pass its frozen tolerance",
                )
            lifecycle = {
                item.get("case")
                for item in group.get("lifecycle_results", [])
                if item.get("status") == "passed"
            }
            require(
                group.get("lifecycle_required") is True
                and {"abort_prefill", "abort_decode", "retract_decode"} <= lifecycle,
                "Missing actual abort/retract recovery proof",
            )
        actual_execution = replay_correctness_execution(
            native_path,
            native,
            required_batches=_required_correctness_batches(data, arm),
        )
        hbm_groups = [
            group
            for group in actual_execution["groups"]
            if group["variant"] == "hbm" and group["eager"] in modes
        ]
        hbm_coverage, hbm_single_coverage = [], []
        for group in hbm_groups:
            round_pairs, round_contexts = [], []
            for item in (group["rounds"][0], group["rounds"][2]):
                size = len(item["request_ids"])
                round_pairs.append(
                    {(item["context_lengths"][0], size)}
                    if len(set(item["context_lengths"])) == 1
                    and item["full_actual_decode_sequences"]
                    and (group["eager"] or size in group["captured_shapes"])
                    else set()
                )
                round_contexts.append(
                    {
                        length
                        for length in set(item["context_lengths"])
                        if group["eager"]
                        or all(
                            set(
                                item["request_observed_actual_batches"][rid]
                            ).intersection(group["captured_shapes"])
                            for rid, n in zip(
                                item["request_ids"], item["context_lengths"]
                            )
                            if n == length
                        )
                    }
                )
            hbm_coverage.append(set.intersection(*round_pairs))
            hbm_single_coverage.append(set.intersection(*round_contexts))
        accepted.append(
            {
                "arm": arm,
                "campaign": identity(proof_path),
                "native": identity(native_path),
                "context_lengths": settings.get("input_lens"),
                "correctness_round_sizes": sorted(
                    {len(r) for g in groups for r in g.get("rounds", [])}
                ),
                "hbm_context_batch_sizes": sorted(set.intersection(*hbm_coverage)),
                "hbm_batch_coverage_modes": ["eager", "graph"] if graph else ["eager"],
                "hbm_single_request_context_lengths": sorted(
                    set.intersection(*hbm_single_coverage)
                ),
                "actual_execution": actual_execution,
            }
        )
    return accepted


def _summary_lifecycle(tool, args, group, trace, artifacts):
    """Replay fault phase and recovery from raw files, without using a server."""
    compare, lifecycle = tool.CORRECTNESS.COMPARE, tool.CORRECTNESS.LIFECYCLE
    length = max(args.input_lens)
    prefix = args.output_dir / (group["id"] + "-lifecycle")
    baseline_path = prefix.with_name(prefix.name + "-baseline.json")
    baseline = read(baseline_path)
    expected = compare.normalize_outputs([baseline], [length], args.output_len)
    baseline_id = baseline["meta_info"]["id"]
    baseline_facts = lifecycle.forward_facts([trace], baseline_id, 1)
    require(
        len(baseline_facts) == 1
        and baseline_facts[0]["prefill_tokens"] == length
        and baseline_facts[0]["decode_forwards"] == args.output_len - 1,
        "Missing complete lifecycle baseline forwards",
    )
    artifacts.append(identity(baseline_path))
    require(group.get("lifecycle_required") is True, "Missing large-batch lifecycle")
    for case in ("abort_prefill", "abort_decode", "retract_decode"):
        passed = [
            item
            for item in group.get("lifecycle_results", [])
            if item.get("case") == case and item.get("status") == "passed"
        ]
        require(len(passed) == 1, "Missing unambiguous large-batch " + case)
        item = passed[0]
        path = prefix.with_name(f"{prefix.name}-{case}-{item['attempt']}.json")
        raw = read(path)
        require(raw == item, "Large-batch lifecycle receipt differs from raw file")
        require(
            raw["response"]["meta_info"]["id"] == raw["request_id"],
            "Lifecycle fault response has another request ID",
        )
        artifacts.append(identity(path))
        facts = lifecycle.forward_facts([trace], raw["request_id"], 1)
        require(facts == raw.get("forward_facts"), "Lifecycle fault trace changed")
        lifecycle.validate_fault(case, raw["response"], facts, length, args.output_len)
        recovered = compare.normalize_outputs(
            [raw["recovery_response"]], [length], args.output_len
        )
        actual = compare.compare([expected], [recovered], args.logprob_atol)
        require(actual == raw.get("recovery_comparison"), "Recovery output changed")
        if case == "retract_decode":
            resumed = compare.normalize_outputs(
                [raw["response"]], [length], args.output_len
            )
            require(
                compare.compare([expected], [resumed], args.logprob_atol)
                == raw.get("resumed_comparison"),
                "Retracted output changed",
            )
        recovery_id = raw["recovery_response"]["meta_info"]["id"]
        require(
            len({baseline_id, raw["request_id"], recovery_id}) == 3,
            "Lifecycle baseline/fault/recovery IDs are not independent",
        )
        recovery = lifecycle.forward_facts([trace], recovery_id, 1)
        require(
            len(recovery) == 1
            and recovery[0]["prefill_tokens"] == length
            and recovery[0]["decode_forwards"] == args.output_len - 1,
            "Missing complete lifecycle recovery forwards",
        )


def verify_large_summary_control(
    path, control, data, proof_path, proof, evidence, batch, length
):
    """Audit the CPU-DDR resident/offloaded summary control, never full-HBM B."""
    tool, replay = report_tool("summary_batch_correctness"), report_tool("analyze")
    settings, gate = control.get("settings", {}), control.get("small_hbm_gate", {})
    require(
        control.get("schema_version") == 1
        and control.get("phase") == "large_batch_summary_equivalence"
        and control.get("status") in ("success", "failed")
        and settings.get("execute") is True
        and control.get("source_fingerprint") == data.get("source_fingerprint")
        and settings.get("model") == proof.get("settings", {}).get("model")
        and settings.get("input_len") == length
        and batch in settings.get("batch_sizes", []),
        "Large-batch summary control has another source/model/context or was not executed",
    )
    require(
        gate.get("status") == "passed"
        and {key: gate.get(key) for key in ("sha256", "bytes")}
        == {key: identity(proof_path)[key] for key in ("sha256", "bytes")},
        "Large-batch summary precursor is not the original bound correctness proof",
    )
    require(
        identity(gate["path"])["sha256"] == gate["sha256"],
        "Summary precursor path changed",
    )
    small_data = dict(
        data,
        rows=[
            {"arm": arm, "input_len": length, "graph_enabled": True}
            for arm in tool.ARMS
        ],
        samples=[],
        same_actual_batch_comparisons=[],
        observed_stable_capacity_points=[],
    )
    verify_model_receipt(proof_path, proof, evidence, small_data, {settings["model"]})
    artifacts, natives, arguments = [identity(path)], {}, {}
    markers = set()
    for arm in tool.ARMS:
        small_case = next(case for case in proof["cases"] if case["arm"] == arm)
        small_path, small_native = _native_correctness(proof_path, small_case, evidence)
        small_identity = identity(small_path)
        require(
            len(
                [
                    item
                    for item in gate.get("native_reports", [])
                    if item.get("arm") == arm
                    and item.get("sha256") == small_identity["sha256"]
                    and item.get("bytes") == small_identity["bytes"]
                ]
            )
            == 1,
            "Missing hash-bound small native summary precursor",
        )
        cases = [
            case
            for case in control.get("cases", [])
            if case.get("arm") == arm and case.get("batch") == batch
        ]
        require(
            len(cases) == 1 and cases[0].get("status") == "success",
            "Missing successful actual summary batch/arm",
        )
        case = cases[0]
        native_path = Path(case["native_manifest"])
        native = read(native_path)
        stage, native_settings, runtime = (
            native.get("stage"),
            native.get("settings", {}),
            native.get("runtime", {}),
        )
        require(
            native.get("kind") == "correctness"
            and native.get("status") == "success"
            and stage
            == case.get("stage")
            == tool.CAMPAIGN.recipe(arm, cohort_barrier=cohort_stage_enabled(stage))
            and cohort_stage_enabled(stage)
            is cohort_stage_enabled(small_case.get("stage", {}))
            and native_settings == case.get("settings")
            and native_settings.get("model") == settings["model"]
            and native_settings.get("workload") == settings.get("workload")
            and native_settings.get("workload_sha256")
            == data.get("workload_identity", {}).get("sha256")
            and native_settings.get("input_lens") == [length]
            and native_settings.get("output_len") == settings.get("output_len")
            and 0 <= native_settings.get("logprob_atol", -1) <= 0.02
            and runtime.get("git_commit") in data.get("native_source_commits", [])
            and runtime.get("archived_source")
            == small_native.get("runtime", {}).get("archived_source"),
            "Large native control recipe/model/workload/archive differs",
        )
        markers.add(json.dumps(runtime["archived_source"], sort_keys=True))
        args = SimpleNamespace(**native_settings)
        args.output_dir = native_path.parent
        require(
            native_settings.get("output_dir") == str(args.output_dir),
            "Large native output path differs",
        )
        groups = native.get("groups", [])
        require(
            len(groups) == 2
            and {(g.get("variant"), g.get("eager")) for g in groups}
            == {("cpu", True), ("cpu", False)},
            "Large summary control requires CPU eager and Graph, not HBM",
        )
        need = tool.allocation_requirements(stage, args, batch)
        normalized = []
        artifacts.append(identity(native_path))
        require(
            native.get("cpu_ram_preflight")
            and all(
                item.get("status") == "passed" for item in native["cpu_ram_preflight"]
            ),
            "Large CPU RAM preflight missing or failed",
        )
        for group in groups:
            require(
                group.get("status") == "success"
                and group.get("rounds") == need["rounds"],
                "Missing complete long-short-long group",
            )
            tool.check_actual_capacity(group.get("capacity", []), need)
            verify_cohort_command(
                stage,
                group,
                read(args.output_dir / f"{group['id']}-server-info.json"),
            )
            fresh, outputs = tool.verify_round_trace(group, args, batch)
            require(
                replay.equal(fresh, group.get("large_batch_trace_equivalence")),
                "Large actual-batch trace/response hashes changed",
            )
            artifacts.extend(
                [fresh["trace"]] + [r["response"] for r in fresh["rounds"]]
            )
            artifacts.append(
                identity(args.output_dir / (group["id"] + "-server-info.json"))
            )
            comparison = tool.CORRECTNESS.COMPARE.compare(
                normalized[0] if normalized else outputs, outputs, args.logprob_atol
            )
            reuse = tool.CORRECTNESS.COMPARE.compare(
                [outputs[0]], [outputs[2]], args.logprob_atol
            )
            require(
                replay.equal(comparison, group.get("comparison_to_cpu_hisparse"))
                and replay.equal(reuse, group.get("slot_reuse_comparison")),
                "Large summary token/logprob or slot reuse receipt changed",
            )
            normalized.append(outputs)
            trace = [
                json.loads(line)
                for line in Path(fresh["trace"]["path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            _summary_lifecycle(tool, args, group, trace, artifacts)
        natives[arm], arguments[arm] = native, args
    require(len(markers) == 1, "Large summary arms use different archived sources")
    comparisons = [
        item for item in control.get("comparisons", []) if item.get("batch") == batch
    ]
    require(
        len(comparisons) == 1 and comparisons[0].get("status") == "passed",
        "Missing passed large summary comparison",
    )
    actual = tool.compare_arms(
        natives[tool.ARMS[0]],
        natives[tool.ARMS[1]],
        arguments[tool.ARMS[0]],
        arguments[tool.ARMS[1]],
        batch,
    )
    require(
        replay.equal(actual, comparisons[0].get("groups")),
        "Large summary comparison differs from raw replay",
    )
    return {
        "status": "passed",
        "batch": batch,
        "input_len": length,
        "modes": ["CPU eager", "CPU Graph"],
        "summary": identity(path),
        "summary_campaign_status": control["status"],
        "precursor": identity(proof_path),
        "artifacts": artifacts,
        "scope": "Only this actual batch is admitted. CPU-DDR full KV: resident versus offloaded Quest summaries; no large-batch full-HBM proof. Graph evidence is actual server/capture configuration and full-batch forwards, not a per-forward replay flag.",
    }


def audit_large_capacity_controls(data, proof_path, proof, receipts, evidence):
    if data.get("phase") != "capacity":
        return []
    covered = {
        item["arm"]: item.get("hbm_context_batch_sizes", []) for item in receipts
    }
    single_requests = {
        item["arm"]: item.get("hbm_single_request_context_lengths", [])
        for item in receipts
    }
    required = set()
    rows = [row for row in data.get("rows", []) if row.get("eligible") is True]
    rows += data.get("observed_stable_capacity_points", [])
    for row in rows:
        arm, length = row["arm"], row["input_len"]
        batch = row.get("actual_batch", row.get("actual_batch_size"))
        if arm in (
            "bounds-on-demand",
            "bounds-prefetch",
            "combined",
            "combined-protected",
        ):
            small = max(
                (b for n, b in covered.get(arm, []) if n == length),
                default=int(length in single_requests.get(arm, [])),
            )
            if batch > small:
                require(
                    arm == "combined-protected",
                    "No large CPU summary control for capacity arm " + arm,
                )
                required.add((batch, length))
    accepted = []
    for batch, length in sorted(required):
        candidates = [
            (path, record)
            for path, record in evidence
            if record.get("phase") == "large_batch_summary_equivalence"
            and record.get("source_fingerprint") == data.get("source_fingerprint")
            and record.get("settings", {}).get("input_len") == length
            and batch in record.get("settings", {}).get("batch_sizes", [])
        ]
        require(
            len(candidates) == 1,
            f"Missing unambiguous CPU summary control for actual B={batch}, input={length}; small HBM proof does not cover this capacity sample",
        )
        path, record = candidates[0]
        accepted.append(
            verify_large_summary_control(
                path, record, data, proof_path, proof, evidence, batch, length
            )
        )
    return accepted


def audit_report_evidence(analyses, evidence, *, diagnostic_only=False):
    """Each analysis must bind its original campaign's exact model proof hash."""
    failures, accepted, bindings = [], [], []
    observed = any(measurement_scope(data)[0] for _, data in analyses)
    if diagnostic_only:
        failures.append("Explicit diagnostic-only report: no model benefit claims")
    if not observed:
        failures.append(
            "No eligible model measurements supplied; report contains scoped diagnostics only"
        )
    if observed and not diagnostic_only:
        for path, data in analyses:
            if not measurement_scope(data)[0]:
                continue
            try:
                require(
                    data.get("native_source_commits"),
                    "Missing native source commit for measured results",
                )
                require(data.get("artifacts"), "Missing raw replay artifact identities")
                campaign = read(data["artifacts"][0]["path"])
                require(
                    campaign.get("phase") == data["phase"], "Raw campaign phase differs"
                )
                controlled = campaign.get("settings", {}).get(
                    "xm_correctness_cohort_barrier", False
                )
                require(type(controlled) is bool, "Invalid measured cohort mode")
                require(
                    not controlled or data["phase"] == "performance",
                    "Capacity must retain ordinary admission without a cohort barrier",
                )
                benchmark_models = set()
                for artifact in data["artifacts"]:
                    actual = identity(artifact["path"])
                    require(
                        actual["sha256"] == artifact.get("sha256")
                        and actual["bytes"] == artifact.get("bytes"),
                        "Raw replay artifact changed or is missing",
                    )
                    if Path(artifact["path"]).name == "manifest.json":
                        raw = read(artifact["path"])
                        if raw.get("kind") == "benchmark":
                            benchmark_models.add(raw.get("settings", {}).get("model"))
                            stage = raw.get("stage", {})
                            require(
                                cohort_stage_enabled(stage) is controlled,
                                "Measured cohort mode differs from its bound campaign",
                            )
                            for group in raw.get("groups", []):
                                verify_cohort_command(
                                    stage,
                                    group,
                                    read(
                                        Path(artifact["path"]).parent
                                        / f"{group['id']}-server-info.json"
                                    ),
                                )
                            if data["phase"] == "capacity":
                                require(
                                    raw.get("stage", {})
                                    .get("variants", {})
                                    .get("cpu", {})
                                    .get("hisparse_config", {})
                                    .get("xm_cache_hbm_budget_bytes")
                                    == data.get("cache_hbm_budget_bytes")
                                    and type(data.get("cache_hbm_budget_bytes")) is int
                                    and data["cache_hbm_budget_bytes"] > 0,
                                    "Capacity report has no same configured budget in its actual native stage",
                                )
                gate = campaign.get("correctness_gate", {})
                require(
                    re.fullmatch("[0-9a-f]{64}", gate.get("sha256", ""))
                    and type(gate.get("bytes")) is int
                    and gate["bytes"] > 0
                    and isinstance(gate.get("path"), str),
                    "Analysis is not bound to its original campaign correctness proof identity",
                )
                candidates = [
                    (proof_path, proof)
                    for proof_path, proof in evidence
                    if proof.get("phase") == "correctness"
                    and isinstance(proof.get("cases"), list)
                    and identity(proof_path)["sha256"] == gate["sha256"]
                    and identity(proof_path)["bytes"] == gate["bytes"]
                ]
                declared = Path(gate["path"])
                if declared.is_file():
                    require(
                        identity(declared)["sha256"] == gate["sha256"]
                        and identity(declared)["bytes"] == gate["bytes"],
                        "Original campaign proof path now contains another proof",
                    )
                    direct = [
                        (p, proof)
                        for p, proof in candidates
                        if p.resolve() == declared.resolve()
                    ]
                    if direct:
                        candidates = direct
                require(
                    len(candidates) == 1,
                    "Missing or ambiguous proof bound to this analysis campaign",
                )
                proof_path, proof = candidates[0]
                require(
                    not controlled
                    or proof.get("settings", {}).get("xm_correctness_cohort_barrier")
                    is True,
                    "Controlled same-cohort timing lacks a bound cohort model proof",
                )
                replay_module = report_tool("analyze")
                replay = replay_module.analyze_campaign(data["artifacts"][0]["path"])
                for key in (
                    "rows",
                    "samples",
                    "paired_repeat_rows",
                    "same_actual_batch_comparisons",
                    "observed_stable_capacity_points",
                    "native_source_commits",
                    "source_fingerprint",
                    "workload_identity",
                    "cache_hbm_budget_bytes",
                ):
                    require(
                        replay_module.equal(data.get(key), replay.get(key)),
                        "Report analysis differs from a fresh raw replay: " + key,
                    )
                receipts = verify_model_receipt(
                    proof_path, proof, evidence, data, benchmark_models
                )
                source017_binding = report_tool("report_control_binding")
                explicit_precursor, point_gate, http_effects = None, None, None
                if source017_binding.source017(data):
                    explicit_precursor = source017_binding.verify_precursor(
                        proof_path, proof, evidence, control_binding_api()
                    )
                    if data["phase"] == "capacity":
                        point_gate = source017_binding.capacity_point_gate(
                            data,
                            proof_path,
                            proof,
                            receipts,
                            evidence,
                            control_binding_api(),
                        )
                        controls = []
                    else:
                        controls = audit_large_capacity_controls(
                            data, proof_path, proof, receipts, evidence
                        )
                        http_effects = source017_binding.verify_http_effects(
                            path, data, evidence, control_binding_api()
                        )
                else:
                    controls = audit_large_capacity_controls(
                        data, proof_path, proof, receipts, evidence
                    )
                accepted.extend(dict(item, analysis=str(path)) for item in receipts)
                bindings.append(
                    {
                        "analysis": identity(path),
                        "original_gate": gate,
                        "proof": identity(proof_path),
                        "status": "passed",
                        "cohort_barrier": controlled,
                        "ttft_scope": (
                            "Controlled cohort TTFT includes prefill/staging barrier; not ordinary NoPD serving TTFT."
                            if controlled
                            else "Ordinary NoPD admission without cohort barrier."
                        ),
                        "actual_measurement_batches": sorted(
                            {
                                row.get("actual_batch")
                                for row in data.get("rows", [])
                                if type(row.get("actual_batch")) is int
                            }
                        ),
                        "model_receipts": receipts,
                        "large_batch_summary_controls": controls,
                        "explicit_precursor": explicit_precursor,
                        "capacity_point_gate": point_gate,
                        "client_paired_effects": http_effects,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                failures.append(str(path) + ": " + str(error))
                bindings.append(
                    {
                        "analysis": str(path),
                        "phase": data.get("phase"),
                        "status": "blocked",
                        "reason": str(error),
                    }
                )
    passed = any(binding["status"] == "passed" for binding in bindings)
    pending_capacity = [
        point
        for binding in bindings
        for point in (binding.get("capacity_point_gate") or {}).get("pending", [])
    ]
    return {
        "status": (
            ("partial" if failures or pending_capacity else "passed")
            if passed
            else "blocked"
        ),
        "model_result_claims_allowed": passed and not diagnostic_only,
        "failures": failures,
        "accepted_model_receipts": accepted,
        "analysis_model_proof_bindings": bindings,
        "pending_capacity_points": pending_capacity,
        "scope": "Each analysis is independently admitted or suppressed. Its raw campaign proof path/hash, archived source/workload, CPU/HBM, modes, tokens/logprobs, reuse and lifecycle are rechecked. Capacity summary-offload batches beyond the small HBM context/batch coverage require an exact actual-batch CPU resident/offloaded summary control. That control is not a large full-HBM reference.",
    }


def display_analyses(analyses, audit):
    if audit.get("status") == "internal_layout_fixture":
        return analyses
    verdicts = {
        (
            str(binding["analysis"].get("path"))
            if isinstance(binding.get("analysis"), dict)
            else str(binding.get("analysis"))
        ): binding
        for binding in audit.get("analysis_model_proof_bindings", [])
    }
    result = []
    for path, original in analyses:
        binding = verdicts.get(str(path), {})
        if audit["model_result_claims_allowed"] and binding.get("status") == "passed":
            gate = binding.get("capacity_point_gate")
            shown = (
                report_tool("report_control_binding").filter_capacity(original, gate)
                if gate
                else original
            )
            result.append((path, shown))
            continue
        data = copy.deepcopy(original)
        data["suppressed_model_samples"] = len(data.get("samples", []))
        for key in (
            "rows",
            "samples",
            "paired_repeat_rows",
            "same_actual_batch_comparisons",
            "observed_stable_capacity_points",
        ):
            data[key] = []
        data["status"] = "no_eligible_samples"
        data.setdefault("excluded_cases", []).append(
            {"case": "report model-claim gate", "reason": "; ".join(audit["failures"])}
        )
        result.append((path, data))
    return result


def mechanism_identity(source, evidence):
    """MMA is a selected model path only with same-source TMA+MMA SASS proof."""
    result = {
        "attention_implementation": "unverified",
        "status": "missing_source_bound_native_identity",
        "unverified_reason": "No source-bound selected scalar/MMA model-backend identity supplied",
        "scope": "Missing implementation identity is not evidence of scalar or deployed MMA",
    }
    candidates = [
        record
        for _, record in evidence
        if record.get("kind") == "xm_mechanism_identity"
        and record.get("source_fingerprint") == source.get("source_fingerprint")
        and record.get("native_source_commit")
        in source.get("native_source_commits", [])
    ]
    selected = {
        record.get("selected_model_backend")
        for record in candidates
        if record.get("selected_model_backend")
        == record.get("attention_implementation")
        and record.get("selected_model_backend") in ("scalar", "mma")
    }
    if len(selected) > 1:
        result["unverified_reason"] = (
            "Conflicting source-bound selected scalar/MMA identities"
        )
        return result
    for record in candidates:
        if (
            record.get("selected_model_backend")
            == record.get("attention_implementation")
            == "scalar"
        ):
            return {
                "attention_implementation": "scalar",
                "status": "source_bound_selected_scalar",
                "scope": "Explicit same-source selected scalar model backend; native instruction and model result proofs remain separate",
            }
        if (
            record.get("selected_model_backend") != "mma"
            or record.get("attention_implementation") != "mma"
        ):
            continue
        sass = record.get("native_sass", {})
        try:
            require(
                sass.get("tma_instruction_observed") is True
                and sass.get("mma_instruction_observed") is True,
                "MMA model identity needs both native TMA and MMA instructions",
            )
            artifact = sass.get("artifact", {})
            actual = identity(artifact["path"])
            require(
                actual["sha256"] == artifact.get("sha256")
                and actual["bytes"] == artifact.get("bytes"),
                "Native SASS artifact changed",
            )
            disassembly = Path(artifact["path"]).read_text(encoding="utf-8")
            symbol = sass.get("kernel_symbol")
            require(
                isinstance(symbol, str) and symbol,
                "Missing selected fused-attention native function symbol",
            )
            functions = list(
                re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", disassembly, re.MULTILINE)
            )
            matches = [
                index
                for index, match in enumerate(functions)
                if match.group(1) == symbol
            ]
            require(
                len(matches) == 1,
                "Missing unambiguous selected fused-attention SASS function section",
            )
            index = matches[0]
            body = disassembly[
                functions[index].end() : (
                    functions[index + 1].start()
                    if index + 1 < len(functions)
                    else len(disassembly)
                )
            ]
            require(
                re.search(r"\b(?:UBLKCP|UTMA[A-Z0-9_]*)\b", body)
                and re.search(r"\b(?:HMMA|HGMMA)\b", body),
                "Selected fused-attention SASS function does not contain both TMA and TensorCore MMA instructions",
            )
            result = {
                "attention_implementation": "mma",
                "status": "native_selected_mma_verified",
                "native_sass": actual,
                "kernel_symbol": symbol,
                "scope": "Selected model backend with same-source TMA (UBLKCP/UTMA) and MMA in the same native fused-attention function; model correctness remains a separate gate",
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["unverified_reason"] = str(error)
    return result


def attention_description(identity):
    if identity.get("attention_implementation") == "mma":
        return "选中模型后端的同源 fused Attention 函数 SASS 已同时观测 TMA 与 MMA；当前融合 Attention 按 TensorCore QK/PV/rowsum 适配记录。"
    if identity.get("attention_implementation") == "scalar":
        return "同源选中模型后端的显式身份为 scalar；独立 MMA 候选不能当作此模型运行的后端。"
    return "当前模型融合 Attention 的实际 scalar/MMA 身份尚未提供可验证的同源选中后端证据；不指定实现，也不把独立候选视为已部署。"


def capacity_groups(data):
    grouped = defaultdict(list)
    for row in data.get("observed_stable_capacity_points", []):
        grouped[
            (row["input_len"], row["client_concurrency"], row.get("graph_enabled"))
        ].append(row)
    return grouped


def audit_profile_identities(profiles, source):
    result = []
    for arm, path, original in profiles:
        data = copy.deepcopy(original)
        if data.get("status") == "analyzed":
            try:
                scope_path = path.parent / "decode-scope.json"
                scope = read(scope_path)
                require(
                    scope.get("status") == "passed"
                    and scope.get("actual_decode_scope_verified") is True
                    and scope.get("arm") == arm,
                    "Missing passed actual decode-scope receipt for this arm",
                )
                require(
                    scope.get("sqlite_sha256") == data.get("sqlite", {}).get("sha256"),
                    "Profile SQLite differs from passed scope",
                )
                provenance = scope.get("provenance", {})
                require(
                    provenance.get("source_fingerprint")
                    == source.get("source_fingerprint"),
                    "Independent profile belongs to another implementation",
                )
                if source.get("native_source_commits"):
                    require(
                        provenance.get("source_commit")
                        in source["native_source_commits"],
                        "Independent profile source revision differs",
                    )
                require(
                    scope.get("actual_batch_size") == data.get("actual_batch_size")
                    and scope.get("actual_decode_graph_launches")
                    == data.get("observed_decode_steps")
                    and len(data.get("per_step", []))
                    == data.get("observed_decode_steps"),
                    "Independent profile batch/step coverage differs from scope",
                )
                inventory = data.get("main_graph_kernel_inventory")
                if inventory:
                    require(
                        inventory.get("graph_id") == scope.get("graph_id")
                        and inventory.get("device_id")
                        == scope.get("device_role", {}).get("trace_device_id"),
                        "Parent kernel inventory graph/device differs from scope",
                    )
                data["report_scope_receipt"] = identity(scope_path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                data.update(status="unverified_report_scope", reason=str(error))
        result.append((arm, path, data))
    return result


def validate_profile(spec):
    arm, separator, declared = spec.partition("=")
    require(separator and arm in ARMS, "Profile must use ARM=PATH")
    path = Path(declared).resolve(strict=True)
    data = read(path)
    if data.get("status") == "analyzed":
        require(data.get("schema_version") == 1, "Unsupported profile schema")
        require(
            re.fullmatch("[0-9a-f]{64}", data.get("sqlite", {}).get("sha256", "")),
            "Independent profile has no SQLite fingerprint",
        )
        number(data.get("actual_batch_size"), positive=True)
        number(data.get("observed_decode_steps"), positive=True)
        for step in data.get("per_step", []):
            number(step.get("total_path_span_us"))
            for stage in step.get("stages", {}).values():
                for key in (
                    "span_us",
                    "kernel_active_union_us",
                    "internal_idle_gap_us",
                ):
                    number(stage.get(key))
        main_graph_profile_summary(data)
    return arm, path, data


def main_graph_profile_summary(profile):
    """Summarize observed parent kernels, keeping child copies out of the scope."""
    inventory = profile.get("main_graph_kernel_inventory")
    if not inventory:
        return None
    steps = inventory.get("per_step", [])
    count = profile.get("observed_decode_steps")
    require(
        inventory.get("observed_decode_steps") == count and len(steps) == count,
        "Parent kernel inventory decode-step coverage differs from profile",
    )
    means = {
        key: statistics.mean(number(step.get(key)) for step in steps)
        for key in (
            "parent_kernel_envelope_us",
            "parent_kernel_active_union_us",
            "parent_kernel_internal_idle_us",
        )
    }
    kernels = inventory.get("kernels", {})
    for stats in kernels.values():
        for key in (
            "duration_per_invocation_us",
            "invocations_per_decode_step",
            "duration_sum_per_decode_step_us",
        ):
            number(stats.get(key, {}).get("mean"))
    means["parent_kernel_duration_sum_us"] = sum(
        stats["duration_sum_per_decode_step_us"]["mean"] for stats in kernels.values()
    )
    return means


def paired_decode_conclusion(row, label):
    """Describe one actual-batch cohort; a cross-zero CI stays inconclusive."""
    effect = row["launch_gap_reduction_percent"]
    mean, interval = effect["mean"], effect.get("confidence_interval_95")
    direction = "下降" if mean >= 0 else "增加"
    if interval is None:
        uncertainty = "95%区间未估计；只描述观测均值。"
    else:
        uncertainty = f"95%区间 [{interval[0]:.2f}, {interval[1]:.2f}]%。"
        if interval[0] <= 0 <= interval[1]:
            uncertainty += "区间跨0，本轮未能确认延迟收益或代价。"
    return (
        f"{label}：同请求与 decode 序号、客户端 C={row['client_concurrency']}、"
        f"实际 batch={row['actual_batch_size']}、输入 {row['input_len']:,} tokens，"
        f"稳定 decode 调度间隔 {direction} {abs(mean):.2f}%"
        f"（{row['baseline_gap_ms']['mean']:.2f} → {row['changed_gap_ms']['mean']:.2f} ms；"
        f"配对重复 n={row['paired_repeats']}）。{uncertainty}"
        "该间隔不是 HTTP token 到达 TBT。"
    )


def validate_client_effect_bindings(bindings):
    """Do not overwrite or mix complete HTTP cohorts across analyses."""
    seen_metrics, seen_effects = set(), set()
    source = None
    for binding in bindings:
        current = (
            binding.get("source_fingerprint"),
            binding.get("native_source_commits"),
            binding.get("workload_identity"),
        )
        require(
            all(value for value in current),
            "Complete HTTP plots need explicit source/revision/workload identity",
        )
        require(
            source is None or current == source,
            "Complete HTTP effects belong to different sources or workloads",
        )
        source = current
        for row in binding["metrics"]:
            key = tuple(
                row[name]
                for name in (
                    "arm",
                    "input_len",
                    "client_concurrency",
                    "graph_enabled",
                )
            )
            require(
                key not in seen_metrics,
                "Duplicate complete HTTP arm/input/C/Graph across admitted analyses; do not merge repeat cohorts",
            )
            seen_metrics.add(key)
        for row in binding["effects"]:
            key = tuple(
                row[name]
                for name in (
                    "baseline",
                    "changed",
                    "metric",
                    "input_len",
                    "client_concurrency",
                    "graph_enabled",
                )
            )
            require(
                key not in seen_effects,
                "Duplicate complete HTTP paired effect across admitted analyses",
            )
            seen_effects.add(key)
    return bindings


def client_effect_bindings(audit):
    if not audit.get("model_result_claims_allowed"):
        return []
    return validate_client_effect_bindings(
        [
            binding["client_paired_effects"]
            for binding in audit.get("analysis_model_proof_bindings", [])
            if binding.get("status") == "passed"
            and (binding.get("client_paired_effects") or {}).get("status") == "passed"
        ]
    )


def client_effect_conclusion(row):
    value = row["paired_repeat_reduction_percent"]
    interval = row["confidence_interval_95_percent"]
    metric = (
        "完整 HTTP mean TBT" if row["metric"] == "http_mean_tbt_ms" else "请求 TPOT"
    )
    uncertainty = (
        "区间跨0，本轮无法确认收益或代价。" if interval[0] <= 0 <= interval[1] else ""
    )
    return (
        f"{row['baseline']}→{row['changed']}：输入 {row['input_len']:,} tokens、客户端 C={row['client_concurrency']}、"
        f"Graph={row['graph_enabled']}，{metric} 的配对重复下降率 {value:.2f}%，"
        f"95% t 区间 [{interval[0]:.2f}, {interval[1]:.2f}]%，n={row['repeats']}，df={row['df']}。{uncertainty}"
    )


def draw_plots(analyses, profiles, directory, *, client_effects=()):
    validate_client_effect_bindings(client_effects)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    colors = dict(
        zip(ARMS, ("#59636e", "#2875a2", "#c19b51", "#47867c", "#855f9b", "#b05c58"))
    )
    client_rows = [row for binding in client_effects for row in binding["metrics"]]
    client_contexts = sorted(
        {(row["client_concurrency"], row["graph_enabled"]) for row in client_rows}
    )
    for concurrency, graph in client_contexts:
        selected = [
            row
            for row in client_rows
            if row["client_concurrency"] == concurrency
            and row["graph_enabled"] == graph
        ]
        lengths = sorted({row["input_len"] for row in selected})
        for metric, label in (
            ("http_mean_tbt_ms", "Complete HTTP mean TBT (ms)"),
            ("request_tpot_ms", "Request TPOT (ms)"),
        ):
            figure, axis = plt.subplots(figsize=(8, 4.4))
            bar_width = 0.8 / max(1, len(lengths))
            for offset, length in enumerate(lengths):
                values = {
                    row["arm"]: row[metric]
                    for row in selected
                    if row["input_len"] == length
                }
                arms = [arm for arm in ARMS if arm in values]
                axis.bar(
                    [
                        ARMS.index(arm) - 0.4 + bar_width * (offset + 0.5)
                        for arm in arms
                    ],
                    [values[arm] for arm in arms],
                    width=bar_width,
                    label=f"{length // 1024}K input",
                )
            axis.set_xticks(range(len(ARMS)), ARMS, rotation=20, ha="right", fontsize=8)
            axis.set_ylabel(label)
            axis.set_title(
                f"Three-repeat complete request means, C={concurrency}, Graph={graph}"
            )
            axis.legend()
            axis.grid(axis="y", alpha=0.2)
            figure.tight_layout()
            base = directory / f"http-c{concurrency}-graph{int(graph)}-{metric}"
            for extension in ("png", "svg"):
                figure.savefig(base.with_suffix("." + extension), dpi=180)
            plt.close(figure)
            images[("http", concurrency, graph, metric)] = base.with_suffix(".png")
    for index, (_, analysis) in enumerate(analyses):
        comparisons = analysis.get("same_actual_batch_comparisons", [])
        if comparisons:
            height = max(3.0, min(10.0, len(comparisons) * 0.4))
            figure, axis = plt.subplots(figsize=(9, height))
            for position, row in enumerate(comparisons):
                stats = row["launch_gap_reduction_percent"]
                value, interval = stats["mean"], stats.get("confidence_interval_95")
                axis.barh(
                    position,
                    value,
                    color="#2875a2" if value >= 0 else "#b05c58",
                    alpha=0.8,
                )
                if interval:
                    axis.errorbar(
                        value,
                        position,
                        xerr=[[value - interval[0]], [interval[1] - value]],
                        color="#222222",
                        capsize=3,
                        fmt="none",
                    )
            labels = [
                f"{r['baseline']} -> {r['changed']} | L={r['input_len']}, B={r['actual_batch_size']}, n={r['paired_repeats']}"
                for r in comparisons
            ]
            axis.set_yticks(range(len(labels)), labels, fontsize=8)
            axis.axvline(0, color="#444444", linewidth=0.8)
            axis.set_xlabel("Matched scheduler decode-interval reduction (%)")
            axis.set_title(
                "Same requests and decode ordinals\nPaired-repeat 95% t intervals"
            )
            axis.grid(axis="x", alpha=0.18)
            figure.tight_layout()
            base = directory / f"same-batch-{index + 1:02d}"
            for extension in ("png", "svg"):
                figure.savefig(base.with_suffix("." + extension), dpi=180)
            plt.close(figure)
            images[(index, "same_batch")] = base.with_suffix(".png")
        if analysis["phase"] == "capacity":
            points = analysis.get("observed_stable_capacity_points", [])
            if points:
                figure, axis = plt.subplots(figsize=(8, 4.4))
                grouped = defaultdict(list)
                for point in points:
                    grouped[
                        (
                            point["arm"],
                            point["input_len"],
                            str(point.get("graph_enabled")),
                        )
                    ].append(point)
                for (arm, length, graph), rows in sorted(grouped.items()):
                    rows.sort(
                        key=lambda row: (
                            row["actual_batch_size"],
                            row["client_concurrency"],
                        )
                    )
                    axis.plot(
                        [row["actual_batch_size"] for row in rows],
                        [row["stable_tokens_per_second"]["mean"] for row in rows],
                        "o-",
                        color=colors[arm],
                        label=f"{arm}, L={length}, graph={graph}",
                    )
                axis.set_xlabel("Observed stable actual decode batch (requests)")
                axis.set_ylabel("Stable scheduler-span output rate (tokens/s)")
                axis.set_title(
                    f"Configured cache budget: {fmt(analysis.get('cache_hbm_budget_bytes', 0) / GIB)} GiB"
                )
                axis.legend(fontsize=7.5)
                axis.grid(alpha=0.2)
                figure.tight_layout()
                base = directory / f"capacity-{index + 1:02d}"
                for extension in ("png", "svg"):
                    figure.savefig(base.with_suffix("." + extension), dpi=180)
                plt.close(figure)
                images[(index, "capacity")] = base.with_suffix(".png")
    passed = [
        (arm, data) for arm, _, data in profiles if data.get("status") == "analyzed"
    ]
    if passed:
        figure, axis = plt.subplots(figsize=(8, 4.3))
        labels = (
            "selection",
            "cache_plan_or_swap",
            "packing",
            "attention_including_merge",
        )
        left = [0.0] * len(passed)
        for label, color in zip(labels, ("#c19b51", "#59636e", "#47867c", "#2875a2")):
            values = [
                data["stage_span_us"].get(label, {}).get("mean", 0)
                for _, data in passed
            ]
            axis.barh(range(len(passed)), values, left=left, label=label, color=color)
            left = [a + b for a, b in zip(left, values)]
        axis.set_yticks(
            range(len(passed)),
            [
                f"{arm}, B={data['actual_batch_size']}, layer={data['layer_index']}"
                for arm, data in passed
            ],
            fontsize=8,
        )
        axis.set_xlabel("One-layer contiguous path span (microseconds)")
        axis.set_title("Independent decode profile: spans include internal idle gaps")
        axis.legend(fontsize=7.5)
        figure.tight_layout()
        base = directory / "attention-path-breakdown"
        for extension in ("png", "svg"):
            figure.savefig(base.with_suffix("." + extension), dpi=180)
        plt.close(figure)
        images["profile"] = base.with_suffix(".png")
    return images


def render_report(
    analyses,
    profiles,
    evidence,
    output,
    *,
    font_path,
    internal_fixture=False,
    images=None,
    report_audit=None,
    attention_identity=None,
):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )

    pdfmetrics.registerFont(TTFont("ReportChinese", str(font_path)))
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="ReportChinese",
            fontSize=21,
            leading=29,
            textColor=colors.HexColor("#16374c"),
            spaceAfter=13,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="ReportChinese",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#16374c"),
            spaceAfter=10,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="ReportChinese",
            fontSize=9.4,
            leading=15,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            fontName="ReportChinese",
            fontSize=7.5,
            leading=10.8,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "cell": ParagraphStyle(
            "cell",
            fontName="ReportChinese",
            fontSize=7.5,
            leading=10.5,
            wordWrap="CJK",
            alignment=TA_LEFT,
        ),
    }
    width = A4[0] - 88
    story, images = [], images or {}

    def paragraph(text, style="body"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def add(text, style="body"):
        story.append(paragraph(text, style))

    def section(title):
        story.append(PageBreak())
        add(title, "heading")

    def table(headers, rows, weights=None):
        weights = weights or [1] * len(headers)
        widths = [width * weight / sum(weights) for weight in weights]
        values = [
            [paragraph(value, "cell") for value in row] for row in [headers] + rows
        ]
        item = LongTable(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef2")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f7f8")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#a4b3bc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(item)
        story.append(Spacer(1, 9))

    def figure(path, caption):
        item = Image(str(path))
        scale = min(width / item.imageWidth, 305 / item.imageHeight)
        item.drawWidth, item.drawHeight = (
            item.imageWidth * scale,
            item.imageHeight * scale,
        )
        story.append(item)
        add(caption, "small")

    add("Qwen＋Quest：XM-Sparse 软件机制适配实验", "title")
    add("单 H800 PCIe · CPU DDR 全量 KV backing · NoPD", "heading")
    add(
        "本报告将融合检索与 Attention、Quest 历史 min/max 摘要卸载与 GPU 发起预取分开归因。实验沿用 1＋0 的 Qwen＋Quest＋HiSparse；第二项是经授权的 Qwen 摘要适配，不是 DeepSeek/GLM 的训练型 K-index 复现。"
    )
    if internal_fixture:
        add(
            "内部排版验证：下列数值为布局 fixture，不是实验结果，禁止用于汇报。",
            "heading",
        )
    add("结果判断", "heading")
    report_audit = report_audit or {
        "status": "blocked",
        "model_result_claims_allowed": False,
        "failures": ["No report evidence audit supplied"],
    }
    attention_identity = attention_identity or {
        "attention_implementation": "unverified",
        "status": "missing_source_bound_native_identity",
    }
    add("报告模型证据审核：" + report_audit["status"], "heading")
    if not report_audit.get("model_result_claims_allowed"):
        add(
            "本次为正式审查说明：模型结果 gate 未通过或尚未取得合格测量；不报告软件机制的模型加速、容量或吞吐收益。原始失败记录和独立诊断仍保留。"
        )
        for reason in report_audit.get("failures", []):
            add("审核原因：" + reason, "small")
    else:
        add(
            "同源模型正确性、独立 HBM/CPU、模式、token/logprob、slot reuse 与 lifecycle 证据已审核。HBM 正确性覆盖的 round batch 大小与当前扫描实际 batch 应分别阅读；小批 HBM 前置证明不代表大批 full-HBM 对照。",
            "small",
        )
        if report_audit["status"] == "partial":
            add(
                "部分分析缺证：只展示逐项通过审核的测量；被阻止的容量分析不会抹掉合格的延迟结果。"
            )
            for reason in report_audit.get("failures", []):
                add("未通过的分析：" + reason, "small")
    comparisons = [
        row
        for _, data in analyses
        if data["phase"] == "performance"
        for row in data.get("same_actual_batch_comparisons", [])
    ]
    http_bindings = client_effect_bindings(report_audit)
    http_effects = [row for binding in http_bindings for row in binding["effects"]]
    for row in http_effects:
        if row["metric"] == "http_mean_tbt_ms" and (
            row["baseline"],
            row["changed"],
        ) in (
            ("baseline", "fused"),
            ("bounds-on-demand", "bounds-prefetch"),
            ("combined", "combined-protected"),
            ("baseline", "combined-protected"),
        ):
            add(client_effect_conclusion(row))
    if not http_bindings:
        add(
            "Table 3 主指标的完整 HTTP mean TBT／TPOT 配对重放尚未准入；稳定 scheduler 间隔仅作为补充诊断，不代替该主指标。",
            "small",
        )
    for baseline, changed, label in (
        ("baseline", "fused", "融合 TMA"),
        ("bounds-on-demand", "bounds-prefetch", "跨层 Quest 摘要预取"),
        ("combined", "combined-protected", "预取保护"),
    ):
        selected = [
            row
            for row in comparisons
            if row["baseline"] == baseline and row["changed"] == changed
        ]
        if selected:
            for row in selected:
                add(paired_decode_conclusion(row, "稳定窗口补充／" + label))
        else:
            add(
                f"尚无满足稳定窗口与请求匹配要求的 {baseline}→{changed} 数据；"
                f"{label}的端到端结论保留待测。"
            )
    capacity = [
        (data, row)
        for _, data in analyses
        if data["phase"] == "capacity"
        for row in data.get("observed_stable_capacity_points", [])
    ]
    if capacity:
        add(
            "容量扫描已经取得稳定实际 batch 的观测点。表 2 保留原 baseline、fused 和 combined-protected 三条路径，"
            "fused 与 combined-protected 使用同一融合 Attention workspace；baseline 的原 FlashInfer 384 MiB workspace 单独列出。"
            "观测上界不等于硬件最大容量或最优性价比。"
        )
    else:
        add("尚未取得同 cache HBM 预算的容量扫描；摘要卸载的并发与吞吐收益保留待验证。")
    add("证据边界", "heading")
    add(
        attention_description(attention_identity)
        + "预取通过 GPU 发起 device-graph memcpy；正确性与物理 CE 利用率、"
        "带宽、重叠程度是不同证据。缺失 child memcpy 活动、受限 GPU counters 都不能被写成 CE 性能结论。"
    )
    add("Attention 机制身份：" + attention_identity["status"], "small")
    add(
        "native-only 机制 receipt 保留其采集时的 model pending／false 范围；它不等于模型失败，也不单独授权模型收益。实际模型范围由同源的独立 proof/raw 审核决定，原 native receipt 不改写。",
        "small",
    )
    add(
        "SASS 中 UBLKCP／HMMA 等计数是静态指令条目，不能解释为每 decode step 的动态执行次数、搬运流量或有效带宽。",
        "small",
    )
    if attention_identity.get("unverified_reason"):
        add("缺证原因：" + attention_identity["unverified_reason"], "small")

    section("1　实验设计、六条路径与指标")
    table(
        ["实验臂", "历史 Quest 摘要", "decode 检索／Attention"],
        [
            ["baseline", "GPU 常驻 min/max", "HiSparse swap-in → packing → FlashInfer"],
            [
                "fused",
                "GPU 常驻 min/max",
                "plan-only → TMA 检索＋Attention；保留 miss refill",
            ],
            [
                "bounds-on-demand",
                "CPU DDR；GPU 分层 stage",
                "当前层按需 GPU 发起 copy；独立 Attention",
            ],
            [
                "bounds-prefetch",
                "CPU DDR；GPU 分层 stage",
                "预取后续层；独立 Attention",
            ],
            ["combined", "CPU DDR；GPU 分层 stage", "TMA 融合＋摘要预取"],
            [
                "combined-protected",
                "CPU DDR；GPU 分层 stage",
                "融合＋保护型预取；容量比较的主要改动臂",
            ],
        ],
        [1.1, 1.25, 2.65],
    )
    add(
        "摘要机制的实际范围：全历史 min/max 卸载到 pinned CPU DDR，复用固定两个 HBM layer stages。"
        "当前 child graph copy 覆盖全部预分配的 R/P 行页范围，不是仅复制活动请求/页。"
        "实现了 on-demand、层间 lookahead 与 Attention gate；保护检查点是公开 CUDA 适配，"
        "已经在途的数据传输可继续完成。"
    )
    add(
        "尚未实现论文按预算逐页 offload、逐步恢复、测量 transfer-budget 的准入，"
        "也未实现跨 decode step 在末尾提前预取下一步 early-layer。单卡没有跨 rank MoE collective 流量；"
        "因此保护结果只覆盖当前单卡 Attention 路径。表 1、表 2 与 A.2 借鉴论文的问题分解方式，"
        "不代表完整作者软件复现。"
    )
    add(
        "on-demand 与 lookahead 都使用 GPU 发起的 child-graph DMA，因此两者对照只隔离跨层 lookahead。"
        "当前没有 CPU 提交 DMA 或 SM 搬运摘要的卸载对照臂，不能独立归因 GPU 发起相对 CPU 提交的收益，"
        "也不构成论文 Figure 10 的复现。"
    )
    add(
        "本次融合内核在每个 32-token tile 只激活一个 KV consumer warp，并使用 CTA 同步；"
        "下一 tile 的 TMA 可与当前计算流水。它保留原 selected-token 数学顺序，"
        "没有论文的 resident-first 遍历，也不是作者优化的 FlashMLA 调度。"
        "性能改善或回退都只说明该 Qwen BF16 MMA＋TMA 实现与 H800 PCIe 的组合，不能据此否定论文机制。"
    )
    table(
        ["指标", "本报告的定义与限制"],
        [
            ["客户端并发 C", "HTTP 客户端同时在途请求数；不等于 GPU 实际 batch。"],
            [
                "实际 batch B",
                "从 scheduler trace 的恒定请求集合窗口取得真实 decode 请求数。",
            ],
            [
                "稳定 decode 间隔",
                "mean_tbt_ms 字段是相邻 scheduler decode launch 的时间间隔，包含 scheduler 间隙；不是严格客户端逐 token 到达 TBT。",
            ],
            [
                "严格 HTTP TBT",
                "仅全部 token-positive HTTP 帧均为单 token 时观测相邻 token 到达间隔。覆盖完整请求的 decode（排除首 token），包含非稳定 batch 初尾；不从多 token 帧的摊销间隔恢复内部时间戳。",
            ],
            [
                "稳定输出率",
                "B×1000／稳定间隔 ms；覆盖恒定请求集合窗口，不是完整请求吞吐。",
            ],
            [
                "TTFT／请求 TPOT",
                "TTFT 含自身 prefill、准入和排队；请求 TPOT=(请求时延-TTFT)/(输出 tokens-1)，已扣自身 TTFT，但仍含 decode 调度及交错 prefill 的干扰。",
            ],
            [
                "误差／重复",
                "同 batch 对照按相同冻结输入行集合及每请求 decode 序号匹配；95% t 区间基于配对重复均值。单次重复不给区间。",
            ],
        ],
        [1, 3.7],
    )
    add(
        "扣除 prefill 时间不能消除准入与排队差异。NoPD 用于排除分离 prefill/decode 通信路径；"
        "模型、固定 token workload、sampling、Graph 与 hot/backing 配置应随原始 manifest 一起审查。"
    )
    if any(
        binding.get("status") == "passed" and binding.get("cohort_barrier") is True
        for binding in report_audit.get("analysis_model_proof_bindings", [])
    ):
        add(
            "本部分固定 batch 性能启用了同 cohort 控制：全部 pending prefill 与 HiSparse staging 就绪后统一放行 decode，"
            "所有软件臂使用相同设置，并逐份核对实际 server argv、启用状态和超时。"
            "其 TTFT 是包含 cohort 前 prefill/staging 屏障的控制值，不能解释为普通 NoPD 服务的 TTFT 收益；"
            "稳定 decode 间隔和 TPOT 仍分别按原始 scheduler／HTTP 定义统计。容量扫描关闭该屏障，保留普通准入行为。"
        )
    add(
        "baseline 是本部分同一冻结模型源码关闭两机制的 HiSparse 对照，继承 01 的 Quest／hot-LRU／swap／FlashInfer 与 NoPD 语义。"
        "新增 guard、idle ledger 与 kernel 参数可能引起重新编译，因此不宣称与历史 b61 基点的机器码或墙钟耗时逐字相同。"
        "正式六臂使用同一冻结模型源码与统一实验工具；历史基点与实际运行 revision 分开记录。"
    )
    source = report_source(analyses, evidence)
    add(
        "实现指纹 SHA256："
        + source.get("source_fingerprint", {}).get("sha256", "未记录"),
        "small",
    )
    add(
        "固定 workload SHA256："
        + (source.get("workload_identity") or {}).get("sha256", "未记录"),
        "small",
    )
    add(
        "真实模型运行 git commit："
        + (
            ", ".join(source.get("native_source_commits", []))
            or "未记录；仅存在未完成样本"
        ),
        "small",
    )

    section("2　表 1A：完整 HTTP 指标（Table 3 主口径的 Qwen 适配）")
    if http_bindings:
        metrics = [row for binding in http_bindings for row in binding["metrics"]]
        table(
            [
                "臂／输入",
                "C／Graph",
                "重复",
                "TTFT ms",
                "mean TBT ms",
                "TPOT ms",
                "完整 tokens/s",
            ],
            [
                [
                    f"{row['arm']}\n{row['input_len']:,}",
                    f"{row['client_concurrency']} / {row['graph_enabled']}",
                    row["repeats"],
                    fmt(row["client_ttft_ms"]),
                    fmt(row["http_mean_tbt_ms"]),
                    fmt(row["request_tpot_ms"]),
                    fmt(row["complete_output_tokens_per_s"], 1),
                ]
                for row in metrics
            ],
            [1.7, 0.7, 0.45, 0.8, 0.8, 0.8, 0.95],
        )
        selected = [
            row
            for row in http_effects
            if (row["baseline"], row["changed"])
            in (
                ("baseline", "fused"),
                ("bounds-on-demand", "bounds-prefetch"),
                ("combined", "combined-protected"),
                ("baseline", "combined-protected"),
            )
        ]
        table(
            ["对照／输入／C", "指标", "下降率 %", "95% t 区间 %", "n／df"],
            [
                [
                    f"{row['baseline']}→{row['changed']}\n{row['input_len']:,} / C={row['client_concurrency']}",
                    (
                        "HTTP mean TBT"
                        if row["metric"] == "http_mean_tbt_ms"
                        else "请求 TPOT"
                    ),
                    fmt(row["paired_repeat_reduction_percent"]),
                    ranges(row["confidence_interval_95_percent"]),
                    f"{row['repeats']} / {row['df']}",
                ]
                for row in selected
            ],
            [2.1, 1.1, 0.8, 1.2, 0.5],
        )
        for binding in http_bindings:
            add(binding["metric_scope"], "small")
            add(binding["ci_scope"], "small")
            outputs = sorted(
                {sample["output_tokens_per_request"] for sample in binding["samples"]}
            )
            add(
                "真实 HTTP 输出长度："
                + ", ".join(map(str, outputs))
                + " tokens。128-token 数值前置证明、512-token 正式 timing 与256-token 独立 profile 的作用和数据分开；不同 context columns／Quest P 不视为同预算 allocation。",
                "small",
            )
        for key, image_path in images.items():
            if key[0] == "http":
                metric = (
                    "完整 HTTP mean TBT"
                    if key[3] == "http_mean_tbt_ms"
                    else "请求 TPOT"
                )
                figure(
                    image_path,
                    f"图 1A　{metric}，客户端 C={key[1]}、Graph={key[2]}；三重复完整请求均值，配对下降率与区间见表。",
                )
    else:
        add(
            "待真实同源模型门、完整 HTTP 单 token 帧重放与三次配对重复准入；本表保持待测。"
        )

    section("2B　表 1B：同实际 batch 的稳定调度间隔补充")
    add(
        "下降率=(原路径间隔-新路径间隔)／原路径间隔；负值表示回退。"
        "只纳入 analysis 中满足稳定步数、恒定请求集合与 decode 序号配对的样本。"
        "当前三次随机顺序重复未做位置平衡，六组配对 CI 未作多重比较校正；"
        "CI 将配对重复均值视为独立样本，不把 token 间隔视为独立重复。"
    )
    all_pairs = []
    for index, (_, data) in enumerate(analyses):
        for row in data.get("same_actual_batch_comparisons", []):
            effect = row["launch_gap_reduction_percent"]
            ci = effect.get("confidence_interval_95")
            all_pairs.append(
                [
                    f"{row['baseline']}\n→ {row['changed']}\n{data['phase']}",
                    f"{row['input_len']:,}\nC={row['client_concurrency']}, B={row['actual_batch_size']}",
                    str(row["paired_repeats"]),
                    fmt(row["baseline_gap_ms"]["mean"]),
                    fmt(row["changed_gap_ms"]["mean"]),
                    f"{effect['mean']:.2f}%",
                    "未估计" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]%",
                ]
            )
    if all_pairs:
        table(
            [
                "对照",
                "输入／C／B",
                "重复",
                "原路径\nms",
                "新路径\nms",
                "间隔下降",
                "95%区间",
            ],
            all_pairs,
            [1.55, 1.45, 0.5, 0.7, 0.7, 0.8, 1],
        )
    else:
        add("未取得合格配对样本。")
    for index, (_, _) in enumerate(analyses):
        if (index, "same_batch") in images:
            figure(
                images[(index, "same_batch")],
                "图 1　同请求匹配的 decode 间隔变化；误差条为配对重复的 95% t 区间。",
            )

    section("3　表 2：同 cache HBM 预算的容量与吞吐")
    add(
        "保留原 baseline→fused、fused→combined-protected 和 baseline→combined-protected：分别观察融合、摘要卸载及双机制的总效果。"
        "cache-only 预算内不包含原 FlashInfer 384 MiB workspace 或融合临时 workspace，不能把该预算下的差异解释为已回收 workspace 的同总显存容量收益。"
        "配置的 cache HBM budget 不是模型、activation、Graph、驱动的总显存预算。"
        "cold NoPD 准入同时需要完整 prefill KV 与已准入/短暂共存的 hot；chunked prefill 不会逐 chunk 释放历史 KV。"
        "max-total-tokens、请求槽位、CPU backing 和准入策略也可能成为上限。已分配 GPU KV 池含 prefill/free reserve，不能全部解释为 live decode hot。"
    )
    add(
        "pinned-memory 分配域以实际 NUMA receipt 为准。full32K B4 正确性前置的 CPU ratio4／8 保留原 phase/seal/argv；大 CPU 控制、性能与容量仍使用 ratio8。"
        "这些配置可行性调整不能计作软件机制收益。"
        "ratio4只改变CPU backing与逻辑映射容量，不定义 Quest P；Quest R/P 来自实际 request-map 行数/context columns，不能从客户端C或ratio猜测。"
        "计划 backing 上界与各容量点的实测 allocation 分开记录，实际允许的 NUMA 节点、启动前检查与分配结果以提供的 NUMA receipt 和原始 capacity 证据为准。"
    )
    if capacity:
        for _, data in analyses:
            if data["phase"] != "capacity":
                continue
            budget = data.get("cache_hbm_budget_bytes")
            add(
                "此轮配置 cache HBM budget："
                + (
                    fmt(budget / GIB) + " GiB"
                    if budget
                    else "未记录，不能宣称同预算收益"
                )
            )
            rows = data.get("observed_stable_capacity_points", [])
            table(
                [
                    "臂",
                    "输入 tokens",
                    "客户端 C",
                    "准入限／Quest R",
                    "实际 B",
                    "Graph",
                    "重复",
                    "稳定间隔 ms",
                    "稳定 tokens/s",
                ],
                [
                    [
                        r["arm"],
                        f"{r['input_len']:,}",
                        r["client_concurrency"],
                        f"{r.get('request_capacity_R', '未观测')} / {r.get('quest_rows_R', '未观测')}",
                        r["actual_batch_size"],
                        str(r.get("graph_enabled", "未记录")),
                        r["observed_repeats"],
                        fmt(r["mean_launch_gap_ms"]["mean"]),
                        fmt(r["stable_tokens_per_second"]["mean"], 1),
                    ]
                    for r in rows
                ],
                [1.4, 0.8, 0.45, 0.95, 0.45, 0.55, 0.4, 0.8, 0.95],
            )
            for (length, concurrency, graph), points in capacity_groups(data).items():
                maxima = {}
                for row in points:
                    if (
                        row["arm"] not in maxima
                        or row["actual_batch_size"]
                        > maxima[row["arm"]]["actual_batch_size"]
                    ):
                        maxima[row["arm"]] = row
                left, right = maxima.get("fused"), maxima.get("combined-protected")
                same_repeats = (
                    left
                    and right
                    and set(left.get("repeat_ids", []))
                    == set(right.get("repeat_ids", []))
                    and bool(left.get("repeat_ids"))
                )
                if same_repeats and budget:
                    add(
                        f"输入 {length:,} tokens、同客户端 C={concurrency}、Graph={graph}、同重复集合：此扫描点的稳定 batch 观测上界为 "
                        f"fused {left['actual_batch_size']} → combined-protected {right['actual_batch_size']}；"
                        f"对应稳定输出率 {left['stable_tokens_per_second']['mean']:.1f} → "
                        f"{right['stable_tokens_per_second']['mean']:.1f} tokens/s。"
                        "此句比较的是各自已观测容量点，不是同 batch 单步加速。"
                    )
                elif left and right:
                    add(
                        f"输入 {length:,}、C={concurrency}、Graph={graph}：重复覆盖或预算证据不足，仅保留各自观测点，不形成容量收益判断。",
                        "small",
                    )
        for index, (_, _) in enumerate(analyses):
            if (index, "capacity") in images:
                figure(
                    images[(index, "capacity")],
                    "图 2　实际稳定 batch 与输出率；连接线仅连接实测扫描点，不代表未测容量。",
                )
    else:
        add(
            "尚无容量扫描结果。保留待验证，不从单 batch 的内存节省外推出最大并发或吞吐。"
        )
    unqualified = [
        point
        for binding in report_audit.get("analysis_model_proof_bindings", [])
        for point in (binding.get("capacity_point_gate") or {}).get("pending", [])
    ]
    if unqualified:
        add("保留的原始容量窗口诊断／尚未取得匹配模型控制", "heading")
        table(
            ["臂／输入", "C／准入限／Quest R／实际 B", "未准入原因"],
            [
                [
                    f"{point['arm']} / {point['input_len']:,}",
                    f"{point['client_concurrency']} / {point.get('request_capacity_R', '未观测')} / {point.get('quest_rows_R', '未观测')} / {point['actual_batch_size']}",
                    point["reason"],
                ]
                for point in unqualified
            ],
            [1.5, 1.2, 3.0],
        )
        add(
            "这些窗口保留原始 C、实际准入上限、Quest tensor R/P 和 B；没有模型门的窗口不进入容量收益曲线。缺 C12/B12 或匹配的 B11/R 控制不等于已测最大 B8；边界细化需新实测，不能用理论11或C11/R11替代C12/R12的运行。",
            "small",
        )
    add("失败或不合格样本", "heading")
    excluded = []
    for _, data in analyses:
        excluded.extend(
            [e.get("case", "未知"), e.get("reason", "未说明")]
            for e in data.get("excluded_cases", [])
        )
        excluded.extend(
            [
                f"{r['arm']} / L={r['input_len']}, C={r['client_concurrency']}, B={r['actual_batch']}, r={r['repeat']}",
                f"稳定间隔数 {r['measured_intervals']}；不足 eligibility 门槛",
            ]
            for r in data.get("rows", [])
            if not r["eligible"]
        )
    if excluded:
        table(["样本", "排除原因"], excluded, [2, 3])
    else:
        add("分析文件中未记录排除样本。未扫描的配置不等于已经通过。")

    section("4　完整六臂结果与重复不确定性")
    for _, data in analyses:
        grouped = defaultdict(list)
        for row in data.get("rows", []):
            if row["eligible"]:
                grouped[
                    (
                        row["arm"],
                        row["input_len"],
                        row["client_concurrency"],
                        row["actual_batch"],
                    )
                ].append(row)
        add(
            "阶段："
            + ("同 batch 性能" if data["phase"] == "performance" else "同预算容量"),
            "heading",
        )
        if grouped:
            rows = []
            for (arm, length, concurrency, batch), samples in sorted(grouped.items()):
                gaps = [s["mean_tbt_ms"] for s in samples]
                rates = [s["stable_tokens_per_s"] for s in samples]
                rows.append(
                    [
                        arm,
                        f"{length:,}\nC={concurrency}, B={batch}",
                        len(samples),
                        fmt(statistics.mean(gaps)),
                        ranges(gaps),
                        fmt(statistics.mean(rates), 1),
                        fmt(statistics.mean(s["mean_tpot_ms"] for s in samples)),
                        fmt(statistics.mean(s["ttft_ms"] for s in samples)),
                    ]
                )
            table(
                [
                    "臂",
                    "输入／C／B",
                    "重复",
                    "间隔均值\nms",
                    "重复间范围\nms",
                    "稳定\ntokens/s",
                    "请求\nTPOT ms",
                    "TTFT\nms",
                ],
                rows,
                [1.5, 1.25, 0.5, 0.8, 1.1, 0.8, 0.75, 0.8],
            )
        else:
            add("无合格稳定样本。")
        add("完整请求 HTTP 到达时延（独立口径）", "heading")
        client_groups = defaultdict(list)
        for sample in data.get("samples", []):
            client_groups[
                (sample["arm"], sample["input_len"], sample["client_concurrency"])
            ].append(sample)
        client_rows = []
        for (arm, length, concurrency), samples in sorted(client_groups.items()):
            require(
                len({sample["repeat"] for sample in samples}) == len(samples),
                "A client request sample would be counted more than once",
            )
            stricts, frame_means = [], []
            for sample in samples:
                client = sample.get("client_timing", {})
                strict = (
                    client.get("single_token_arrival_gap_ms")
                    if client.get("all_frames_single_token") is True
                    else None
                )
                if strict is not None:
                    stricts.append(strict)
                frame = client.get("raw_token_positive_frame_gap_ms", {}).get("mean")
                if frame is not None:
                    frame_means.append(frame)
            unavailable = (
                "不可严格定义"
                if any(
                    sample.get("client_timing", {}).get("all_frames_single_token")
                    is False
                    for sample in samples
                )
                else "未观测"
            )
            client_rows.append(
                [
                    arm,
                    f"{length:,}\nC={concurrency}",
                    f"{len(stricts)}/{len(samples)}",
                    sum(stat["samples"] for stat in stricts) if stricts else "未观测",
                    (
                        fmt(statistics.mean(stat["mean"] for stat in stricts))
                        if stricts
                        else unavailable
                    ),
                    ranges([stat["p99"] for stat in stricts]),
                    fmt(statistics.mean(frame_means)) if frame_means else "未观测",
                ]
            )
        if client_rows:
            table(
                [
                    "臂",
                    "输入／C",
                    "strict\n重复数",
                    "单 token\n间隔数",
                    "HTTP TBT\n重复均值 ms",
                    "每次 P99\n范围 ms",
                    "HTTP frame\n间隔均值 ms",
                ],
                client_rows,
                [1.45, 1.15, 0.7, 0.8, 1.1, 1.1, 1.1],
            )
        else:
            add("未提供 HTTP 帧证据。")
        add(
            "HTTP 每个原始样本按 arm／输入／客户端 C／repeat 只计一次，不按多个实际 batch 窗口重复计数。"
            "严格 TBT 列只汇总 strict 重复；P99 显示各重复实测 P99 的范围，不是合并分布的 P99。"
            "HTTP frame gap 为多 token 帧也可观测的原始帧间隔，不能替代单 token TBT。",
            "small",
        )
    add(
        "重复间范围展示实测分散程度，不是置信区间。请求 TPOT 与 TTFT 来自完整请求，"
        "可受预填充和排队影响；不能用这些字段替代同请求匹配的稳定 decode 间隔。"
    )

    section("5　具名 HBM／DDR 分解与 384 MiB workspace 归因")
    add(
        "所有 MiB 按 2^20 bytes 换算。每个 allocation_group 的别名只计一次；"
        "plan 同时存在于 cache ledger 和 backend inventory 时只计 cache ledger。"
        "backend workspace 只显示独立、已经观测的分配；未观测不写作零。"
    )
    all_samples = [
        dict(
            sample,
            _report_phase=data["phase"],
            _report_budget=data.get("cache_hbm_budget_bytes"),
        )
        for _, data in analyses
        for sample in data.get("samples", [])
    ]
    grouped_memory = defaultdict(list)
    for sample in all_samples:
        grouped_memory[
            (
                sample["_report_phase"],
                sample["_report_budget"],
                sample["input_len"],
                sample["client_concurrency"],
            )
        ].append(sample)
    metrics = [
        ("GPU 历史 min/max 摘要", "cache", "quest_resident_summary_bytes"),
        ("GPU 分层摘要 stage", "cache", "quest_layer_staging_bytes"),
        ("GPU running bounds", "cache", "quest_running_bounds_bytes"),
        (
            "GPU 已分配 KV 池（含 prefill/free reserve）",
            "cache",
            "physical_compute_kv_bytes",
        ),
        ("GPU plan 数组", "cache", "fused_plan_bytes"),
        ("GPU 其他 cache 管理（合并）", "cache", "other_named_gpu_bytes"),
        ("Attention split partial", "backend", "fused_split_workspace_bytes"),
        ("Attention output", "backend", "fused_output_bytes"),
        ("FlashInfer workspace", "backend", "separate_attention_workspace_bytes"),
        (
            "具名 cache＋backend 合计",
            "direct",
            "torch_named_total_including_observed_backend_bytes",
        ),
        ("CPU DDR 全量 KV", "direct", "cpu_full_kv_backing_bytes"),
        ("CPU DDR 历史摘要", "direct", "cpu_quest_summary_backing_bytes"),
    ]
    for (phase, budget, length, concurrency), samples in sorted(
        grouped_memory.items(), key=lambda item: str(item[0])
    ):
        add(
            f"阶段 {phase}，输入 {length:,} tokens，客户端 C={concurrency}；"
            f"cache budget {fmt(budget / GIB) + ' GiB' if budget else '未记录'}；"
            "每臂为实测 idle 快照范围（MiB）",
            "small",
        )
        arms = [arm for arm in ARMS if any(s["arm"] == arm for s in samples)]
        rows = []
        known = {key for _, owner, key in metrics if owner == "cache"}
        extra = sorted(
            {
                key
                for sample in samples
                for key in sample["memory_after"]["cache_components"]
            }
            - known
        )
        local_metrics = metrics + [
            ("GPU " + CATEGORY_LABELS.get(key, key), "cache", key) for key in extra
        ]
        for label, owner, key in local_metrics:
            row = [label]
            for arm in arms:
                values = []
                for sample in samples:
                    if sample["arm"] != arm:
                        continue
                    memory = sample["memory_after"]
                    if owner == "direct":
                        value = memory.get(key)
                    elif owner == "cache":
                        value = memory["cache_components"].get(key, 0)
                    elif memory["backend_inventory_observed"]:
                        value = memory["backend_components"].get(key, 0)
                    else:
                        value = None
                    values.append(value)
                row.append(ranges(values, MIB))
            rows.append(row)
        table(["分配归属"] + arms, rows, [1.75] + [1] * len(arms))
    if not grouped_memory:
        add("尚无真实模型具名分配快照。")
    add(
        "融合可能移除原路径实际保留的 384 MiB FlashInfer workspace，但这是 Attention 后端的工作区，"
        "不是全量 KV，也不是 Quest 历史摘要。只有快照确实观察到该分配时，才把这个大小作为实测。"
        "摘要卸载独立减少历史 min/max 的 GPU 常驻空间，并引入 GPU stage 与 CPU backing。"
    )
    allocator_groups = defaultdict(list)
    for sample in all_samples:
        before, after = sample["memory_before"].get("allocator"), sample[
            "memory_after"
        ].get("allocator")
        if before is not None or after is not None:
            allocator_groups[
                (
                    sample["_report_phase"],
                    sample["arm"],
                    sample["input_len"],
                    sample["client_concurrency"],
                )
            ].append(sample)
    allocator_rows = []
    for (phase, arm, length, concurrency), samples in sorted(allocator_groups.items()):
        fields = []
        for boundary, key in (
            ("memory_before", "allocated_bytes"),
            ("memory_after", "allocated_bytes"),
            ("memory_before", "reserved_bytes"),
            ("memory_after", "reserved_bytes"),
            ("memory_after", "peak_allocated_bytes"),
            ("memory_after", "peak_reserved_bytes"),
        ):
            fields.append(
                ranges(
                    [
                        (sample[boundary].get("allocator") or {}).get(key)
                        for sample in samples
                    ],
                    MIB,
                    1,
                )
            )
        allocator_rows.append(
            [f"{arm}\n{phase}", f"L={length:,}\nC={concurrency}"] + fields
        )
    if allocator_rows:
        add("Allocator 观测范围（MiB）", "heading")
        table(
            [
                "臂／阶段",
                "样本",
                "allocated\n前 idle",
                "allocated\n后 idle",
                "reserved\n前 idle",
                "reserved\n后 idle",
                "累计 peak\nallocated",
                "累计 peak\nreserved",
            ],
            allocator_rows,
            [1.55, 1.15, 1, 1, 1, 1, 1, 1],
        )
        add(
            "peak 是该 allocator 观测时的累计峰值，未按 prefill／decode 或 Graph 单独 reset；"
            "不将它归因为特定阶段。范围来自重复实测快照，原始 bytes 与 scope 保存在 analysis.json。",
            "small",
        )
    else:
        add("allocated/reserved 未提供可解析原始观测，保留未观测。")
    add(
        "此轮 inventory 不含模型权重、activation、驱动/私有 Graph pool、完整 NVML 采样，"
        "也未由 idle 前后快照证明 prefill／decode 临时峰值。因此不得把具名合计写成进程总 HBM 或峰值。"
    )
    add(
        "v2 inventory 的逐 storage allocation_group、alias_names 与 category 保存在 analysis 与报告随附"
        " named-allocations.csv；以此审查 LRU、mapping、allocator worklists 和 plan 的独立归属。",
        "small",
    )

    section("6　Appendix A.2 式独立 decode 诊断")
    add(
        "按一个选定层的首个 Quest score 到末个 Attention／merge 的连续时间段分解。"
        "分段 span 含内部 idle gap；kernel 活跃区间取并集，不能把跨流 kernel 耗时简单相加当作关键路径。"
        "首个 score 前的预取依赖等待位于本分解范围之外。"
    )
    passed = [
        (arm, data) for arm, _, data in profiles if data.get("status") == "analyzed"
    ]
    if passed:
        add(
            "完整父 CUDA Graph 归因：下表覆盖已核验主图内的所有可见 kernel，包含首个 Quest score 前的等待。"
            "envelope 是父图首个 kernel 开始到末个 kernel 结束的区间，active union 是活跃区间并集，"
            "idle 为 envelope 减 active union。kernel duration sum 是各 kernel 耗时之和，"
            "不是关键路径或 CE 耗时；child 图、memcpy、图外 CPU／采样／HTTP 均未覆盖。"
        )
        parent_rows = []
        for arm, profile in passed:
            means = main_graph_profile_summary(profile)
            if means is None:
                parent_rows.append([arm, "未提供完整父图 inventory", "-", "-", "-"])
            else:
                parent_rows.append(
                    [
                        arm,
                        fmt(means["parent_kernel_envelope_us"]),
                        fmt(means["parent_kernel_active_union_us"]),
                        fmt(means["parent_kernel_internal_idle_us"]),
                        fmt(means["parent_kernel_duration_sum_us"]),
                    ]
                )
        table(
            [
                "实验臂",
                "父图 envelope μs",
                "active union μs",
                "内部 idle μs",
                "kernel sum μs",
            ],
            parent_rows,
            [1.7, 1.1, 1.1, 1.05, 1.05],
        )
        stages = (
            "selection",
            "cache_plan_or_swap",
            "packing",
            "attention_including_merge",
        )
        for arm, profile in passed:
            add(
                f"{arm}：实际 B={profile['actual_batch_size']}，层 {profile['layer_index']}，"
                f"独立 decode steps={profile['observed_decode_steps']}",
                "heading",
            )
            rows = []
            for stage in stages:
                stats = profile.get("stage_span_us", {}).get(stage)
                if stats is None:
                    rows.append([stage, "未出现此分段", "—", "—"])
                    continue
                active = [
                    s["stages"].get(stage, {}).get("kernel_active_union_us", 0)
                    for s in profile["per_step"]
                ]
                idle = [
                    s["stages"].get(stage, {}).get("internal_idle_gap_us", 0)
                    for s in profile["per_step"]
                ]
                rows.append(
                    [
                        stage,
                        fmt(stats["mean"]),
                        fmt(statistics.mean(active)),
                        fmt(statistics.mean(idle)),
                    ]
                )
            rows.append(
                [
                    "总路径",
                    fmt(profile["total_path_span_us"]["mean"]),
                    "见分段",
                    "见分段",
                ]
            )
            table(
                ["分段", "span μs", "kernel 活跃并集 μs", "内部 idle gap μs"],
                rows,
                [2, 1, 1.25, 1.25],
            )
            kernels = sorted(
                profile.get("kernel_mean_duration_us", {}).items(),
                key=lambda item: -item[1]["mean"],
            )
            table(
                ["Kernel", "次数", "均值 μs", "范围 μs"],
                [
                    [
                        name,
                        stats["samples"],
                        fmt(stats["mean"]),
                        f"{fmt(stats['min'])} - {fmt(stats['max'])}",
                    ]
                    for name, stats in kernels
                ],
                [3.8, 0.5, 0.7, 1],
            )
            inventory = profile.get("main_graph_kernel_inventory")
            if inventory:
                all_kernels = sorted(
                    inventory["kernels"].items(),
                    key=lambda item: -item[1]["duration_sum_per_decode_step_us"][
                        "mean"
                    ],
                )

                def graph_kernel_row(name, stats):
                    return [
                        name,
                        fmt(stats["duration_per_invocation_us"]["mean"]),
                        fmt(stats["invocations_per_decode_step"]["mean"]),
                        fmt(stats["duration_sum_per_decode_step_us"]["mean"]),
                    ]

                mechanism = [
                    (name, stats)
                    for name, stats in all_kernels
                    if re.search(
                        r"(?:^|::)(?:launch_kernel|wait_kernel|checkpoint_kernel|gate_kernel)\b",
                        name,
                    )
                ]
                add(arm + "：父图中的摘要传输控制 kernel", "heading")
                if mechanism:
                    table(
                        ["Kernel", "每次均值 μs", "每 decode 次数", "每 decode sum μs"],
                        [graph_kernel_row(name, stats) for name, stats in mechanism],
                        [3.7, 0.75, 0.8, 0.95],
                    )
                else:
                    add(
                        "本父图未观测到上述 launch／wait／checkpoint／gate kernel；"
                        "这不是 child 图控制或 memcpy 耗时为零的证明。",
                        "small",
                    )
                add(arm + "：完整父图具名 kernel 清单", "heading")
                add(
                    "按每 decode duration sum 降序；每次均值来自实际 kernel 活动，"
                    "每 decode 次数来自本次独立 capture 的父图调用。"
                    "这些次数不是 SASS 静态指令计数；完整机器可读清单另存 main-graph-kernels.csv。",
                    "small",
                )
                table(
                    ["Kernel", "每次均值 μs", "每 decode 次数", "每 decode sum μs"],
                    [graph_kernel_row(name, stats) for name, stats in all_kernels],
                    [3.7, 0.75, 0.8, 0.95],
                )
            add("SQLite SHA256：" + profile["sqlite"]["sha256"], "small")
        if "profile" in images:
            figure(
                images["profile"],
                "图 3　独立诊断的一层路径分解。各臂不同 batch／层时仅描述各自观测，不构成同条件因果对照。",
            )
    else:
        add(
            "没有通过 decode-scope 审核的独立 profile；该部分保留待测，禁止根据模型总耗时填充 kernel 分段。"
        )
    for arm, _, profile in profiles:
        if profile.get("status") != "analyzed":
            add(
                f"{arm} profile：{profile.get('status', 'unknown')}；{profile.get('reason', '未取得可用独立证据')}。"
            )
    add(
        "device child memcpy 活动缺失时，不把 memcpy cost 填为零。软件 copy counters 的 idle 前后差值"
        "可能包含 bench_serving warmup；它不是硬件 PCIe／CE 传输量，不能用于有效带宽计算。"
    )

    section("7　正确性、机制证据与可复核边界")
    bindings = report_audit.get("analysis_model_proof_bindings", [])
    if bindings:
        table(
            ["分析", "审核", "范围／缺证"],
            [
                [
                    (
                        Path(binding["analysis"].get("path", "")).name
                        if isinstance(binding.get("analysis"), dict)
                        else Path(binding.get("analysis", "")).name
                    ),
                    binding["status"],
                    binding.get("reason", "原始测量与模型证明逐项通过"),
                ]
                for binding in bindings
            ],
            [1.5, 0.65, 3.35],
        )
    controls = [
        item
        for binding in bindings
        for item in binding.get("large_batch_summary_controls", [])
    ]
    if controls:
        table(
            ["输入／实际 B", "审核", "独立对照", "HBM 证明范围"],
            [
                [
                    f"{item['input_len']:,} / B={item['batch']}",
                    item["status"] + "；campaign=" + item["summary_campaign_status"],
                    "CPU DDR 全量 KV；常驻／卸载摘要；eager＋Graph；三轮／reuse／lifecycle",
                    "仅绑定的小批 HBM precursor；此大 batch 控制不是 full-HBM",
                ]
                for item in controls
            ],
            [1.1, 0.55, 2.5, 2.35],
        )
    precursors = [
        binding["explicit_precursor"]
        for binding in bindings
        if binding.get("explicit_precursor")
    ]
    if precursors:
        table(
            ["原始前置证明", "CPU ratio", "实际完整上下文范围", "作用"],
            [
                [
                    Path(item["original_campaign"]["path"]).parent.name,
                    item["precursor_capacity_ratio"],
                    "32K／B4；physical132608；full4context131584；CPU/HBM eager＋Graph",
                    "原phase/seal/argv与raw重放；不作为ratio8同预算证据",
                ]
                for item in precursors
            ],
            [1.5, 0.55, 2.1, 1.85],
        )
    baseline_controls = [
        item
        for binding in bindings
        for item in (binding.get("capacity_point_gate") or {}).get(
            "external_baseline_controls", []
        )
    ]
    if baseline_controls:
        table(
            [
                "原始 CPU baseline 控制",
                "B／准入限／Quest R",
                "前置／大控制 ratio",
                "证明范围",
            ],
            [
                [
                    Path(item["control"]["path"]).name,
                    f"{item['batch']} / {item['request_capacity_R']} / {item['quest_rows_R']}",
                    f"{item['precursor_capacity_ratio']} / {item['large_control_capacity_ratio']}",
                    "FI baseline→fused／combined-protected；同输入/greedy/LP；eager/Graph/reuse/lifecycle；无大HBM/容量证明",
                ]
                for item in baseline_controls
            ],
            [1.4, 1.1, 0.7, 2.8],
        )
        for item in baseline_controls:
            custody = item["execution_custody"]
            add(
                "外部控制源码："
                + custody["controller"]["path"]
                + "；SHA256="
                + custody["controller"]["sha256"],
                "small",
            )
            add(
                "执行 runner："
                + custody["runner"]["path"]
                + "；SHA256="
                + custody["runner"]["sha256"],
                "small",
            )
            add(
                "独立执行 custody："
                + custody["custody"]["path"]
                + "；SHA256="
                + custody["custody"]["sha256"],
                "small",
            )
    for path, record in evidence:
        arm = (
            record.get("stage", {}).get("xm_adaptation", {}).get("arm")
            if isinstance(record.get("stage"), dict)
            else None
        )
        add(path.name + (" / " + arm if arm else ""), "heading")
        add(
            "原始 receipt 类型："
            + str(record.get("kind", record.get("phase", "未提供")))
            + "；状态："
            + str(record.get("status", "未记录")),
            "small",
        )
        if record.get("scope"):
            add("原始 scope：" + str(record["scope"]), "small")
        gpu_tests = record.get("gpu_tests")
        if isinstance(gpu_tests, dict):
            table(
                ["观测", "值"],
                [
                    [
                        "独立 GPU kernel tests",
                        str(gpu_tests.get("pytest_summary", "未提供")),
                    ],
                    [
                        "FP32 reference max abs error",
                        str(gpu_tests.get("fp32_reference_max_abs_error", "未提供")),
                    ],
                    [
                        "refill tolerance",
                        json.dumps(gpu_tests.get("exact_refill_tolerance", {})),
                    ],
                    ["原始 workload", str(record.get("workload", "未说明"))],
                ],
                [1.3, 3.5],
            )
        if record.get("kind") == "correctness":
            executed = record.get("settings", {}).get("execute") is True
            add(
                "真实模型执行："
                + str(executed)
                + "；manifest 状态："
                + record.get("status", "未记录"),
                "small",
            )
            groups = record.get("groups", [])
            if executed and groups:
                rows = []
                for group in groups:
                    comparison = group.get("comparison_to_cpu_hisparse", {})
                    reuse = group.get("slot_reuse_comparison", {})
                    lifecycle = group.get("lifecycle_results", [])
                    rows.append(
                        [
                            group.get("variant", "未知"),
                            "eager" if group.get("eager") else "Graph",
                            group.get("status", "未记录"),
                            str(comparison.get("exact_greedy_token_match", "未记录")),
                            fmt(comparison.get("max_abs_logprob_error"), 6),
                            fmt(comparison.get("logprob_atol"), 6),
                            str(reuse.get("exact_greedy_token_match", "未记录")),
                            "; ".join(
                                f"{item.get('case', item.get('kind', 'case'))}: {item.get('status', 'unknown')}"
                                for item in lifecycle
                            )
                            or "未提供",
                        ]
                    )
                table(
                    [
                        "KV backing",
                        "执行",
                        "状态",
                        "token exact",
                        "logprob 最大\n绝对误差",
                        "容差",
                        "slot reuse\nexact",
                        "lifecycle",
                    ],
                    rows,
                    [0.8, 0.7, 0.75, 0.8, 1, 0.8, 0.8, 1.75],
                )
                add(
                    "比较方向是当前 CPU DDR HiSparse 臂与独立 HBM Quest，首个 CPU 参考自比较不构成独立证明；"
                    "HBM 组和 eager/Graph、slot reuse/lifecycle 共同限定已验证的路径。",
                    "small",
                )
            else:
                add("未执行的计划不计作模型正确性。", "small")
        if record.get("phase") == "correctness" and isinstance(
            record.get("cases"), list
        ):
            table(
                ["模型实验臂", "case 状态", "退出码"],
                [
                    [
                        case.get("arm", "未知"),
                        case.get("status", "未记录"),
                        case.get("returncode", "未记录"),
                    ]
                    for case in record["cases"]
                ],
                [2, 1.5, 1],
            )
        if record.get("native_mechanism"):
            mechanism = record["native_mechanism"]
            table(
                ["机制", "观测"],
                [[key, str(value)] for key, value in mechanism.items()],
                [1.8, 3],
            )
        if record.get("gpu"):
            add(
                "设备原始观测：" + json.dumps(record["gpu"], ensure_ascii=False),
                "small",
            )
        if record.get("nsight"):
            nsight = record["nsight"]
            add(
                "设备 child graph 已观测："
                + str(nsight.get("graph_child_observed", "未记录"))
                + "；CE profile 已证明："
                + str(nsight.get("ce_dma_verified_by_profile", "未记录")),
                "small",
            )
            add(
                str(nsight.get("gpu_metrics_limitation", "未提供 GPU counter 证据")),
                "small",
            )
        if record.get("frozen_linux_gpu_suite"):
            suite = record["frozen_linux_gpu_suite"]
            table(
                ["冻结 Linux 源码验收", "记录"],
                [
                    [
                        "tests / passed",
                        f"{suite.get('tests', '未记录')} / {suite.get('passed', '未记录')}",
                    ],
                    [
                        "failures / errors / skipped",
                        f"{suite.get('failures', '未记录')} / {suite.get('errors', '未记录')} / {suite.get('skipped', '未记录')}",
                    ],
                    ["XML SHA256", suite.get("xml", {}).get("sha256", "未记录")],
                ],
                [1.5, 3.5],
            )
            normalized = record.get("isolated_overlay_sources", [])
            add(
                f"关联 receipt 覆盖 {len(normalized)} 个早期 overlay 文件；"
                f"其中行尾归一后同源 {sum(item.get('same_after_line_ending_normalization') is True for item in normalized)} 个。"
                "这是源码关联与冻结源测试证据，不是模型输出或性能证据；002 二进制 receipt 不表示所有最终 LF 构建的 raw hash 相同。",
                "small",
            )
        for key in (
            "status",
            "source_state",
            "source_commit",
            "baseline_commit",
            "git_commit",
            "result",
            "adaptation",
            "error",
            "failure_phase",
        ):
            if key in record:
                add(f"{key}：{record[key]}", "small")
        if record.get("not_established_by_these_tests"):
            add(
                "这些测试没有建立："
                + "；".join(record["not_established_by_these_tests"]),
                "small",
            )
        add("证据 JSON SHA256：" + identity(path)["sha256"], "small")
    if not evidence:
        add(
            "本次生成未提供额外正确性／native mechanism JSON；只有数值 analysis，不能扩展为完整模型正确性结论。"
        )
    add(
        "源码关联范围：早期 isolated probe/overlay 与最终 LF 部署源码可能有不同 raw 文件 hash，"
        "包括 Windows CRLF 行尾差异。模型性能统一以上 analysis 的实现指纹与 native git commit；"
        "早期 probe 的通过不表示其所有 raw source bytes 或二进制与最终模型部署完全相同。"
        "最终冻结源验收应作为独立 receipt 保留。",
        "small",
    )
    add("本次适配能回答什么", "heading")
    add(
        "同请求、同实际 batch 的稳定 decode 成本；同配置 cache HBM budget 下已经观测的容量与输出率；"
        "具名 KV／摘要／管理／workspace 的分配差异。结果无论改善或回退均保留。"
    )
    add("仍不能外推什么", "heading")
    add(
        "论文 B300 scale-up fabric 的性能；训练型 K-index 的预取收益；4＋0/4＋4 的 Attention-DP＋EP 模型；"
        "真实 MPU 的 1＋1 跨卡通路；物理 CE 利用率或重叠；未给定 SLO 和小时成本时的性价比甜点区。"
    )

    section("8　数据来源与复核")
    rows = []
    for path, data in analyses:
        rows.append([path.name, data["phase"], identity(path)["sha256"]])
    for arm, path, _ in profiles:
        rows.append(
            [path.name, f"independent profile / {arm}", identity(path)["sha256"]]
        )
    for path, _ in evidence:
        rows.append([path.name, "separate scoped evidence", identity(path)["sha256"]])
    table(["文件", "角色", "SHA256"], rows, [1.6, 1.4, 3.4])
    add(
        "analysis 的原始 artifact 列表包含 serving result、scheduler trace、manifest 与重放 timing/steady 文件的内容 hash。"
        "本 PDF 的 sidecar receipt 另记录实际输入与输出 hash。派生 hash 证明本次使用的内容身份；"
        "它不替代更早的证据传输与验收记录。"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c9d2d8"))
        canvas.line(44, 42, A4[0] - 44, 42)
        canvas.setFont("ReportChinese", 7.5)
        canvas.setFillColor(colors.HexColor("#516571"))
        text = (
            "内部布局 fixture，非实验数据"
            if internal_fixture
            else "Qwen/Quest on one H800 PCIe + CPU DDR | XM-Sparse adaptation"
        )
        canvas.drawString(44, 29, text)
        canvas.drawRightString(A4[0] - 44, 29, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=44,
        rightMargin=44,
        topMargin=42,
        bottomMargin=56,
        title="Qwen Quest XM-Sparse 软件机制适配实验",
        author="MPU inference experiments",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def append_pdf(base, addition, output):
    from pypdf import PdfReader, PdfWriter

    require(
        output.resolve() not in (base.resolve(), addition.resolve()),
        "Merged PDF must use a new filename",
    )
    original_hash = identity(base)["sha256"]
    original = PdfReader(base)
    texts = [page.extract_text() for page in original.pages]
    writer = PdfWriter()
    writer.clone_document_from_reader(original)
    writer.append(str(addition))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)
    merged = PdfReader(output)
    require(
        [page.extract_text() for page in merged.pages[: len(texts)]] == texts,
        "Original PDF page text changed",
    )
    require(identity(base)["sha256"] == original_hash, "Original PDF changed")
    return {
        "original_pages": len(texts),
        "appended_pages": len(merged.pages) - len(texts),
        "original_page_texts_exactly_preserved": True,
        "original_file_unchanged": True,
    }


def write_named_allocations(analyses, output):
    rows = []
    for _, analysis in analyses:
        for sample in analysis.get("samples", []):
            for boundary in ("memory_before", "memory_after"):
                memory = sample[boundary]
                for owner in ("cache", "backend"):
                    for record in memory.get(owner + "_allocation_groups", []):
                        rows.append(
                            {
                                "phase": analysis["phase"],
                                "arm": sample["arm"],
                                "input_len": sample["input_len"],
                                "client_concurrency": sample["client_concurrency"],
                                "repeat": sample["repeat"],
                                "boundary": boundary,
                                "owner": owner,
                                "allocation_group": record["allocation_group"],
                                "category": record["category"],
                                "storage_bytes": record["storage_bytes"],
                                "alias_names": json.dumps(
                                    record["alias_names"], ensure_ascii=False
                                ),
                            }
                        )
    if not rows:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", default=[])
    parser.add_argument("--profile", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-pdf", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--plot-packages", type=Path)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--internal-layout-fixture", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    require(
        bool(args.base_pdf) == bool(args.merged_output),
        "--base-pdf and --merged-output must be supplied together",
    )
    require(
        not args.internal_layout_fixture or args.base_pdf is None,
        "Internal fixture cannot be appended to a real report",
    )
    require(
        args.analysis or (args.diagnostic_only and args.evidence),
        "Supply audited analyses, or --diagnostic-only with failure evidence",
    )
    require(args.font.is_file(), "Chinese TrueType font not found; pass --font")
    require(not args.output.exists(), "Refusing to overwrite existing PDF")
    if args.merged_output:
        require(not args.merged_output.exists(), "Refusing to overwrite merged PDF")
    if args.plot_packages:
        sys.path.insert(0, str(args.plot_packages.resolve(strict=True)))
    analyses = (
        validate_analyses(args.analysis, internal_fixture=args.internal_layout_fixture)
        if args.analysis
        else []
    )
    profiles = [validate_profile(spec) for spec in args.profile]
    evidence = [(path.resolve(strict=True), read(path)) for path in args.evidence]
    source = report_source(analyses, evidence)
    audit = audit_report_evidence(
        analyses, evidence, diagnostic_only=args.diagnostic_only
    )
    if args.internal_layout_fixture:
        audit = {
            "status": "internal_layout_fixture",
            "model_result_claims_allowed": True,
            "failures": [],
            "accepted_model_receipts": [],
        }
    shown = display_analyses(analyses, audit)
    profiles = audit_profile_identities(profiles, source)
    attention = mechanism_identity(source, evidence)
    audit["attention_identity_status"] = attention["status"]
    if attention.get("attention_implementation") == "unverified":
        audit["mechanism_identity_limitation"] = attention["unverified_reason"]
    images = (
        {}
        if args.skip_plots
        else draw_plots(
            shown,
            profiles,
            args.output.parent / (args.output.stem + "-figures"),
            client_effects=client_effect_bindings(audit),
        )
    )
    render_report(
        shown,
        profiles,
        evidence,
        args.output,
        font_path=args.font,
        internal_fixture=args.internal_layout_fixture,
        images=images,
        report_audit=audit,
        attention_identity=attention,
    )
    named_csv = write_named_allocations(
        shown, args.output.with_name(args.output.stem + "-named-allocations.csv")
    )
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_evidence_audit": audit,
        "attention_mechanism_identity": attention,
        "scope": (
            "internal layout fixture"
            if args.internal_layout_fixture
            else (
                "rendered audited model results"
                if audit["model_result_claims_allowed"]
                else "formal failure/diagnostic review; model benefit claims suppressed"
            )
        ),
        "inputs": [identity(path) for path in args.analysis + args.evidence]
        + [identity(path) for _, path, _ in profiles],
        "standalone_pdf": identity(args.output),
        "figures": [identity(path) for path in sorted(set(images.values()))],
        "visual_review": "pending_external_render_and_inspection",
    }
    receipt["verified_profile_scopes"] = [
        data["report_scope_receipt"]
        for _, _, data in profiles
        if data.get("report_scope_receipt")
    ]
    if named_csv:
        receipt["named_allocations_csv"] = identity(named_csv)
    if args.base_pdf:
        receipt["base_pdf"] = identity(args.base_pdf)
        receipt["preservation"] = append_pdf(
            args.base_pdf, args.output, args.merged_output
        )
        receipt["merged_pdf"] = identity(args.merged_output)
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pdf": str(args.output),
                "receipt": str(receipt_path),
                "merged_pdf": str(args.merged_output) if args.merged_output else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
