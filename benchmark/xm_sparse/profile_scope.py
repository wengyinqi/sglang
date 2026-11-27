"""Audit XM/Qwen decode scope without relabeling scalar TMA as TensorCore.

A passed scope binds raw trace membership, the bounded trigger, all 48 layers,
main graph runtime correlations, archived core source and actual correctness,
and the trace device UUID to the captured compute role. It is an independent
Nsight diagnostic, not a performance sample or a hardware traffic measurement.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load(Path(__file__).with_name("campaign.py"), "xm_scope_campaign")
BASE_SCOPE = load(ROOT / "benchmark/mpu_stages/decode_scope.py", "xm_scope_baseline")
ACCEPTANCE = load(ROOT / "benchmark/mpu_stages/acceptance.py", "xm_scope_acceptance")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matches_identity(path, identity):
    return (
        isinstance(identity, dict)
        and identity.get("sha256") == sha(path)
        and identity.get("bytes") == Path(path).stat().st_size
    )


def core_fingerprint(source_root):
    source_root = Path(source_root)
    files = sorted(
        p
        for p in (source_root / "python/sglang").rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc")
    )
    require(files, "No deployed core sources to fingerprint")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"files": len(files), "sha256": digest.hexdigest()}


def verify_trace(profile, trace, *, layers=48):
    # The audited native verifier's complete request/trigger guards are reused
    # verbatim. Its TensorCore-specific semantics and kernel gate are not used.
    request = profile.get("profile_request", {})
    steps = profile.get("configured_decode_steps")
    require(
        type(steps) is int and 1 <= steps <= 512 and request.get("num_steps") == steps,
        "Configured profile steps differ",
    )
    require(
        request.get("activities") == ["CUDA_PROFILER"]
        and request.get("profile_by_stage") is True
        and request.get("profile_stages") == ["decode"],
        "Require dedicated decode trigger",
    )
    command = profile.get("server_command", [])
    require(
        "--capture-range=cudaProfilerApi" in command
        and "--cuda-graph-trace=node" in command,
        "Missing bounded CUDA graph node capture",
    )
    ids = profile.get("profiled_request_ids", [])
    count = profile.get("workload", {}).get("concurrency")
    require(
        type(count) is int
        and count > 0
        and len(ids) == count
        and len(set(ids)) == count
        and all(isinstance(rid, str) and rid for rid in ids),
        "Profile workload has missing or duplicate request identities",
    )
    require(
        profile.get("completion_tokens") == count * profile["workload"]["output_len"],
        "Incomplete profiled output tokens",
    )
    require(
        type(layers) is int and layers == 48,
        "This verifier supports the pinned 48-layer Qwen model",
    )
    session = trace[0] if trace else {}
    require(
        session.get("type") == "session"
        and session.get("tp_size") == 1
        and session.get("pp_size") == 1
        and session.get("dp_size") == 1
        and session.get("disable_overlap_schedule") is True,
        "Stable capture requires one non-overlap compute scheduler",
    )
    require(
        session.get("disaggregation_mode") == "null"
        and session.get("disable_radix_cache") is True
        and session.get("speculative_algorithm") is None
        and session.get("enable_mixed_chunk") is False,
        "Unexpected PD/prefix/speculative/mixed-chunk capture",
    )
    forwards = [r for r in trace if r.get("type") == "forward"]
    require(
        [r.get("sequence") for r in forwards] == list(range(len(forwards))),
        "Incomplete or reordered forward trace",
    )
    require(
        trace[-1].get("type") == "idle_flush"
        and trace[-1].get("last_sequence") == len(forwards) - 1,
        "Profile trace did not flush completely at idle",
    )
    selected = [r for r in forwards if set(ids).intersection(r.get("request_ids", []))]
    prefill, decode = Counter(), Counter()
    for row in selected:
        require(
            row.get("success") is True
            and set(row["request_ids"]) <= set(ids)
            and len(set(row["request_ids"])) == len(row["request_ids"]),
            "Failed, mixed or duplicate profile forward",
        )
        if row["forward_mode"] == "extend":
            lengths = row.get("extend_lens")
            require(
                isinstance(lengths, list)
                and len(lengths) == len(row["request_ids"])
                and all(type(n) is int and n > 0 for n in lengths),
                "Invalid profile prefill chunks",
            )
            prefill.update(dict(zip(row["request_ids"], lengths)))
        elif row["forward_mode"] == "decode":
            decode.update(row["request_ids"])
        else:
            raise ValueError("Unknown profile forward mode")
    require(
        dict(prefill) == dict.fromkeys(ids, profile["workload"]["input_len"]),
        "Profile input coverage differs",
    )
    require(
        dict(decode) == dict.fromkeys(ids, profile["workload"]["output_len"] - 1),
        "Profile decode output coverage differs",
    )
    before, after = profile.get("trigger_progress_before"), profile.get(
        "trigger_progress_after_ack"
    )
    require(
        isinstance(before, dict) and isinstance(after, dict),
        "Missing observed stable trigger boundaries",
    )
    stable_ids = set(before.get("request_ids", []))
    minimum = profile.get("stable_trigger", {}).get("min_consecutive_decode_steps", 0)
    require(
        type(minimum) is int
        and minimum >= 16
        and before.get("consecutive_stable_decode_steps", 0) >= minimum,
        "Stable trigger did not observe at least 16 actual decode steps",
    )
    for snapshot in (before, after):
        require(
            snapshot.get("success") is True
            and snapshot.get("forward_mode") == "decode"
            and set(snapshot.get("request_ids", [])) == stable_ids
            and stable_ids <= set(ids)
            and snapshot.get("batch_size") == len(stable_ids)
            and stable_ids,
            "Trigger crossed a request-set change or prefill",
        )
    first, last = before.get("sequence"), after.get("sequence")
    require(
        type(first) is int and type(last) is int and 0 <= first <= last,
        "Invalid profile trigger sequence bounds",
    )
    required_batch = profile.get("stable_trigger", {}).get("required_actual_batch_size")
    require(
        required_batch is None
        or (type(required_batch) is int and required_batch == len(stable_ids)),
        "Observed batch differs from required profile batch size",
    )
    history = [r for r in forwards if first - minimum + 1 <= r["sequence"] <= first]
    require(
        len(history) == minimum
        and all(
            r.get("success") is True
            and r["forward_mode"] == "decode"
            and set(r["request_ids"]) == stable_ids
            for r in history
        )
        and len({r.get("stream_id") for r in history}) == 1
        and history[0].get("stream_id") is not None,
        "Stable trigger counter is not confirmed by actual consecutive decode forwards",
    )
    # Poll/control requests can span several forwards. Reject any prefill or
    # request-set change across the whole possible capture interval, including
    # a conservative extra two closing steps. Do not claim an exact event↔host
    # timestamp correspondence that the exporter did not record.
    stop = last + steps + 2
    guard = [r for r in forwards if first <= r["sequence"] <= stop]
    require(len(guard) == stop - first + 1, "Incomplete conservative capture interval")
    require(
        all(
            r.get("success") is True
            and r["forward_mode"] == "decode"
            and set(r["request_ids"]) == stable_ids
            for r in guard
        ),
        "Prefill or changed request membership in conservative capture interval",
    )
    require(
        len({r.get("stream_id") for r in guard}) == 1
        and guard[0].get("stream_id") is not None,
        "Capture interval crossed compute streams",
    )
    require(
        guard[0]["stream_id"] == history[0]["stream_id"],
        "Capture interval differs from the stable trigger stream",
    )
    for snapshot in (before, after):
        row = forwards[snapshot["sequence"]]
        require(
            snapshot.get("stream_id") == row["stream_id"]
            and snapshot.get("launch_monotonic_ns") == row.get("launch_monotonic_ns"),
            "Trigger metadata does not match its actual forward sequence",
        )
    require(
        all(
            type(r.get("launch_monotonic_ns")) is int and r["launch_monotonic_ns"] > 0
            for r in forwards
        ),
        "Missing actual monotonic forward launches",
    )
    require(
        all(
            a["launch_monotonic_ns"] < b["launch_monotonic_ns"]
            for a, b in zip(forwards, forwards[1:])
        ),
        "Non-monotonic forward launch sequence",
    )
    # Current raw exporter records request_ids, not a separate batch_size.
    # Trigger progress records batch_size and is checked above against the
    # same raw membership; validate any explicit raw size if one is present.
    require(
        all(
            r.get("batch_size", len(r["request_ids"])) == len(r["request_ids"])
            for r in selected
        ),
        "Actual batch differs from request membership",
    )
    return {
        "configured_decode_steps": steps,
        "stable_request_ids": sorted(stable_ids),
        "actual_batch_size": len(stable_ids),
        "actual_batch_source": "Complete raw request membership joined to observed trigger progress batch_size",
        "conservative_trace_first_sequence": first,
        "conservative_trace_last_sequence": stop,
        "no_prefill_or_membership_change_in_conservative_interval": True,
        "whole_request_input_tokens_verified": sum(prefill.values()),
        "whole_request_decode_tokens_verified": sum(decode.values()),
    }


def verify_provenance(
    profile, proof_path, acceptance_path, native_path, marker_path, source_root
):
    proof, receipt, native, marker = map(
        read, (proof_path, acceptance_path, native_path, marker_path)
    )
    require(
        matches_identity(acceptance_path, profile.get("acceptance")),
        "Profile acceptance receipt identity differs",
    )
    require(
        matches_identity(native_path, profile.get("correctness_artifact")),
        "Profile native correctness artifact identity differs",
    )
    require(
        matches_identity(proof_path, receipt.get("correctness_campaign")),
        "Adapted receipt references another correctness campaign",
    )
    fingerprint = core_fingerprint(source_root)
    require(
        proof.get("phase") == "correctness"
        and proof.get("status") == "success"
        and proof.get("source_fingerprint") == fingerprint,
        "Missing successful correctness for the actual core fingerprint",
    )
    stage = profile.get("stage", {})
    arm = stage.get("xm_adaptation", {}).get("arm")
    require(arm in CAMPAIGN.ARMS, "Missing known XM adaptation arm")
    config = stage.get("variants", {}).get("cpu", {}).get("hisparse_config", {})
    expected_stage = CAMPAIGN.recipe(
        arm, budget=config.get("xm_cache_hbm_budget_bytes")
    )
    require(
        stage == expected_stage
        and receipt.get("stage") == stage
        and native.get("stage") == stage,
        "Profile/correctness stage differs from the exact arm recipe",
    )
    cases = [case for case in proof.get("cases", []) if case.get("arm") == arm]
    require(
        len(cases) == 1
        and cases[0].get("status") == "success"
        and cases[0].get("returncode") == 0
        and cases[0].get("stage") == stage,
        "No exact successful native correctness case for this arm",
    )
    require(
        receipt.get("status") == "passed_correctness_only"
        and receipt.get("evidence_scope", {}).get("target_model_correctness_executed")
        is True,
        "Missing actual model correctness-only receipt",
    )
    steps = [
        step for step in receipt.get("steps", []) if step.get("id") == "correctness"
    ]
    require(
        len(steps) == 1
        and steps[0].get("status") == "passed"
        and len(steps[0].get("results", [])) == 1
        and steps[0]["results"][0].get("status") == "passed"
        and steps[0].get("expected_model_stage") == stage,
        "Invalid actual correctness acceptance step",
    )
    jobs = steps[0].get("commands", [])
    require(len(jobs) == 1, "Correctness receipt has ambiguous native jobs")
    job = dict(jobs[0], report=str(native_path))
    actual = ACCEPTANCE.external_result(steps[0], job, cases[0]["returncode"])
    require(actual.get("status") == "passed", "Raw native correctness no longer passes")
    groups = native.get("groups", [])
    require(
        {(g.get("variant"), g.get("eager")) for g in groups}
        >= {("cpu", True), ("cpu", False), ("hbm", True), ("hbm", False)},
        "Missing independent HBM Quest or eager/Graph correctness coverage",
    )
    require(
        marker.get("provenance") == "git_archive"
        and re.fullmatch(r"[0-9a-f]{40}", marker.get("source_commit", ""))
        and re.fullmatch(r"[0-9a-f]{64}", marker.get("source_archive_sha256", "")),
        "Missing exact deployed archive build marker",
    )
    require(
        (Path(source_root) / ".mpu-source.json").is_file()
        and sha(Path(source_root) / ".mpu-source.json") == sha(marker_path),
        "Audited core source directory has another archive marker",
    )
    runtimes = [
        profile.get("runtime", {}),
        native.get("runtime", {}),
        receipt.get("runtime", {}),
    ]
    require(
        all(
            r.get("archived_source") == marker
            and r.get("git_commit") == marker["source_commit"]
            and r.get("git_commit_source") == "archive_build_marker"
            and r.get("git_status") == "unavailable: archived source has no Git index"
            and r.get("sglang_source") == runtimes[0].get("sglang_source")
            for r in runtimes
        )
        and isinstance(runtimes[0].get("sglang_source"), str)
        and runtimes[0]["sglang_source"].endswith("/python/sglang/__init__.py"),
        "Runtime source/commit/archive identity differs from real correctness",
    )
    workload = profile.get("workload", {})
    settings = native.get("settings", {})
    require(
        profile.get("profile_variant") == "cpu"
        and workload.get("manifest_sha256")
        == proof.get("workload_identity", {}).get("sha256")
        and workload.get("manifest_sha256") == settings.get("workload_sha256")
        and workload.get("manifest_sha256") is not None
        and workload.get("manifest") == settings.get("workload")
        and settings.get("model") == proof.get("settings", {}).get("model")
        and workload.get("input_len") in proof.get("settings", {}).get("input_lens", [])
        and settings.get("disable_cuda_graph") is False,
        "Profile model/context/workload/Graph differs from actual correctness",
    )
    command = profile.get("server_command", [])
    require(
        command.count("--hisparse-config") == 1,
        "Require one actually executed HiSparse server configuration",
    )
    executed = json.loads(command[command.index("--hisparse-config") + 1])
    require(
        all(
            executed.get(key) == config.get(key)
            for key in (
                "xm_fused_retrieval_attention",
                "xm_quest_bounds_offload",
                "xm_quest_prefetch_mode",
                "xm_prefetch_protect",
                "xm_cache_hbm_budget_bytes",
            )
        )
        and executed.get("backing_store", "cpu") == "cpu"
        and executed.get("algorithm") == "quest",
        "Executed server command differs from declared arm mechanism",
    )
    return {
        "arm": arm,
        "source_fingerprint": fingerprint,
        "source_commit": marker["source_commit"],
        "source_archive_sha256": marker["source_archive_sha256"],
        "correctness_campaign_sha256": sha(proof_path),
        "native_correctness_sha256": sha(native_path),
        "acceptance_sha256": sha(acceptance_path),
        "archive_marker_sha256": sha(marker_path),
        "scope": "One archived core fingerprint and exact Qwen/Quest arm; real CPU/HBM eager/Graph correctness and lifecycle receipts revalidated",
    }


def uuid(value):
    require(isinstance(value, str), "Missing actual device UUID")
    value = value.lower().removeprefix("gpu-")
    require(
        re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value),
        "Invalid device UUID",
    )
    return value


def verify_device_role(conn, profile):
    captured = profile.get("devices", [])
    require(
        len(captured) == 1
        and captured[0].get("role") == "compute"
        and type(captured[0].get("visible_ordinal")) is int,
        "This Qwen 1+0 verifier requires exactly one captured compute GPU",
    )
    expected = uuid(captured[0].get("uuid"))
    columns = {r[1] for r in conn.execute("PRAGMA table_info(TARGET_INFO_GPU)")}
    require(
        {"id", "uuid", "name", "busLocation", "computeMajor", "computeMinor"}
        <= columns,
        "Nsight lacks actual UUID/architecture target evidence",
    )
    rows = list(
        conn.execute(
            "SELECT id,uuid,name,busLocation,computeMajor,computeMinor FROM TARGET_INFO_GPU"
        )
    )
    matched = [row for row in rows if uuid(row[1]) == expected]
    require(
        len(matched) == 1, "Trace UUID does not map uniquely to captured compute role"
    )
    device, raw_uuid, name, bus, major, minor = matched[0]
    require(
        type(device) is int
        and device >= 0
        and (major, minor) == (9, 0)
        and isinstance(name, str)
        and "H800" in name,
        "Require the actually captured SM90 H800 target",
    )
    active = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    }
    require(active == {device}, "Unmapped or other-role device executed CUDA kernels")
    inventory = list(csv.reader(profile.get("physical_gpu_inventory", "").splitlines()))
    require(
        len(inventory) >= 2
        and [x.strip() for x in inventory[0]]
        == ["index", "uuid", "pci.bus_id", "name"],
        "Missing captured physical GPU UUID inventory",
    )
    physical = [
        r for r in inventory[1:] if len(r) == 4 and uuid(r[1].strip()) == expected
    ]
    require(
        len(physical) == 1
        and physical[0][3].strip() == name
        and physical[0][2].strip().lower().lstrip("0") == bus.lower().lstrip("0"),
        "Physical inventory differs from actual Nsight target UUID/name/bus",
    )
    runtime_rows = list(
        csv.reader(profile.get("runtime", {}).get("gpu", "").splitlines())
    )
    require(
        any(
            len(r) == 4 and uuid(r[1].strip()) == expected and r[0].strip() == name
            for r in runtime_rows
        ),
        "Runtime GPU identity differs from Nsight target",
    )
    return {
        "trace_device_id": device,
        "uuid": raw_uuid,
        "role": "compute",
        "name": name,
        "pci_bus_id": bus,
        "compute_capability": [major, minor],
        "mapping_source": "Nsight TARGET_INFO_GPU UUID joined to captured PyTorch role, physical inventory and runtime nvidia-smi UUID",
        "mapping_verified_automatically": True,
    }


def kernel_kind(name, fused):
    if name == "_quest_score_kernel":
        return "score"
    if fused:
        if "xm_hisparse_tma::attention_kernel<" in name:
            return "attention"
        if "xm_hisparse_tma::merge_kernel<" in name:
            return "merge"
    elif name == BASE_SCOPE.DECODE_ATTENTION_KERNEL:
        return "attention"
    return "other"


def verify_graphs(conn, profile, arm, compute_device, *, layers=48):
    fused = CAMPAIGN.ARMS[arm][0]
    runtime = list(
        conn.execute(
            "SELECT r.start,r.end,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE 'cudaGraphLaunch%' ORDER BY r.start"
        )
    )
    steps = profile["configured_decode_steps"]
    require(
        len(runtime) in (steps, steps + 1),
        "Actual main graph runtime count differs from bounded decode trigger",
    )
    launch_ids = {r[2] for r in runtime}
    require(
        len(launch_ids) == len(runtime)
        and all(type(c) is int and c > 0 for c in launch_ids)
        and all(
            type(a) is int and type(b) is int and 0 <= a <= b for a, b, _, _ in runtime
        ),
        "Duplicate or invalid main graph runtime correlation",
    )
    rows = list(
        conn.execute(
            "SELECT k.start,k.end,k.deviceId,k.streamId,k.correlationId,k.graphId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start"
        )
    )
    require(
        rows
        and all(
            type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and device == compute_device
            and isinstance(name, str)
            for start, end, device, _, _, _, name in rows
        ),
        "Invalid CUDA kernel activity",
    )
    allowed_prefill = BASE_SCOPE.DECODE_ATTENTION_KERNEL if not fused else None
    require(
        not any("prefill" in r[6].lower() and r[6] != allowed_prefill for r in rows),
        "Unknown or genuine prefill kernel in decode capture",
    )
    relevant = [r for r in rows if kernel_kind(r[6], fused) != "other"]
    require(
        relevant and all(r[4] in launch_ids for r in relevant),
        "Score/attention associated with an unobserved main runtime launch",
    )
    graph_ids = {r[5] for r in relevant}
    streams = {r[3] for r in relevant}
    require(
        len(graph_ids) == 1
        and all(type(g) is int and g > 0 for g in graph_ids)
        and len(streams) == 1
        and None not in streams,
        "Score/attention/merge are not on one stable main graph and stream",
    )
    kinds = ("score", "attention", "merge") if fused else ("score", "attention")
    counts = {}
    for kind in kinds:
        per_launch = Counter(r[4] for r in relevant if kernel_kind(r[6], fused) == kind)
        require(
            set(per_launch) == launch_ids and set(per_launch.values()) == {layers},
            "Missing layer or wrong main-launch association in " + kind,
        )
        counts[kind] = sum(per_launch.values())
    # Child prefetch graphs are allowed, but must never supply the main graph's
    # score/attention calls or be folded into an attention path span.
    names = sorted({r[6] for r in relevant})
    for correlation in launch_ids:
        per_launch = [r for r in relevant if r[4] == correlation]
        scores = [r for r in per_launch if kernel_kind(r[6], fused) == "score"]
        for index, score in enumerate(scores):
            stop = (
                scores[index + 1][0]
                if index + 1 < layers
                else max(r[1] for r in per_launch) + 1
            )
            segment = [r for r in per_launch if score[0] <= r[0] < stop]
            require(
                sum(kernel_kind(r[6], fused) == "attention" for r in segment) == 1
                and (
                    not fused
                    or sum(kernel_kind(r[6], fused) == "merge" for r in segment) == 1
                ),
                "Cannot assign exactly one attention/merge to each layer score",
            )
            attention = next(
                r for r in segment if kernel_kind(r[6], fused) == "attention"
            )
            require(
                score[1] <= attention[0],
                "Attention ran before its layer selection completed",
            )
            if fused:
                merge = next(r for r in segment if kernel_kind(r[6], fused) == "merge")
                require(
                    attention[1] <= merge[0],
                    "Merge ran before fused attention completed",
                )
    return {
        "actual_decode_graph_launches": len(runtime),
        "boundary_step_included": len(runtime) == steps + 1,
        "calls_per_graph": layers,
        "graph_id": next(iter(graph_ids)),
        "main_graph_stream_id": next(iter(streams)),
        "main_runtime_correlation_ids": sorted(launch_ids),
        "quest_score_calls": counts["score"],
        "decode_attention_calls": counts["attention"],
        "fused_merge_calls": counts.get("merge", 0),
        "observed_attention_kernel_names": names,
        "attention_path": (
            "scalar FP32 BF16 GQA + cp.async.bulk TMA + split merge"
            if fused
            else "known FlashInfer TensorCore decode specialization using paged-prefill module; one query row/request"
        ),
        "uses_author_flashmla_implementation": False,
    }


def verify(
    database,
    profile_path,
    trace_path,
    *,
    correctness_campaign,
    acceptance,
    native_correctness,
    archive_marker,
    source_root=ROOT
):
    profile = read(profile_path)
    require(
        profile.get("status") == "captured_pending_role_analysis",
        "Require an actual completed profiler capture",
    )
    sqlite_ids = [
        a
        for a in profile.get("artifacts", [])
        if Path(a.get("path", "")).suffix == ".sqlite"
    ]
    require(
        len(sqlite_ids) == 1 and matches_identity(database, sqlite_ids[0]),
        "SQLite identity differs from captured evidence",
    )
    provenance = verify_provenance(
        profile,
        correctness_campaign,
        acceptance,
        native_correctness,
        archive_marker,
        source_root,
    )
    trace = [
        json.loads(line)
        for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trace_scope = verify_trace(profile, trace)
    with closing(
        sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        role = verify_device_role(conn, profile)
        graph = verify_graphs(conn, profile, provenance["arm"], role["trace_device_id"])
    return {
        "schema_version": 1,
        "status": "passed",
        "actual_decode_scope_verified": True,
        "sqlite_sha256": sha(database),
        "profile_json_sha256": sha(profile_path),
        "forward_trace_sha256": sha(trace_path),
        "provenance": provenance,
        "arm": provenance["arm"],
        "device_role": role,
        **trace_scope,
        **graph,
        "performance_result": False,
        "hardware_traffic_measured": False,
        "scope_limitations": "Request-set and compute stream are bound through complete monotonic forwards and conservative trigger bounds; no exact host-to-Nsight timestamp join or paper FlashMLA/K-index reproduction is asserted",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "sqlite",
        "profile",
        "trace",
        "correctness-campaign",
        "acceptance",
        "native-correctness",
        "archive-marker",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Require a new scope output path")
    try:
        result = verify(
            args.sqlite,
            args.profile,
            args.trace,
            correctness_campaign=args.correctness_campaign,
            acceptance=args.acceptance,
            native_correctness=args.native_correctness,
            archive_marker=args.archive_marker,
            source_root=args.source_root,
        )
    except (ValueError, sqlite3.Error, KeyError, TypeError, OSError) as error:
        result = {
            "schema_version": 1,
            "status": "failed_scope_verification",
            "actual_decode_scope_verified": False,
            "reason": str(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(args.output)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
