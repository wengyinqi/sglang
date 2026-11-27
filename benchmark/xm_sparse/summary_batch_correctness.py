"""Large actual-batch CPU-DDR Quest summary equivalence, with a small HBM gate.

This control compares resident versus offloaded summaries using the same fused
numerical kernel and CPU-DDR full KV. It does not assert a large-batch full-HBM
reference or use the historical-KV footprint as a fake GPU allocation budget.
"""

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("fused", "combined-protected")
QWEN_KV_SHAPE = {"layers": 48, "kv_heads": 4, "head_dim": 128, "element_bytes": 2}
QWEN_KV_BYTES_PER_TOKEN = 2 * math.prod(QWEN_KV_SHAPE.values())
HOST_MEMORY_RESERVE_BYTES = 10 << 30  # Same reserve as memory_pool_host.py.


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load(Path(__file__).with_name("campaign.py"), "xm_summary_batch_campaign")
CORRECTNESS = load(
    ROOT / "benchmark/mpu_stages/correctness.py", "xm_summary_batch_native"
)
RUN = CORRECTNESS.RUN
ACCEPTANCE = RUN.load("acceptance")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def allocation_requirements(stage, args, batch):
    config = RUN.STAGE.config_for(stage, args, "cpu", args.capacity_ratios[0])
    page = stage["page_size"]
    hot = config["device_buffer_size"]

    def rounded(n):
        return math.ceil(n / page) * page

    long = [args.input_lens[0]] * batch
    short = ([page + 1, hot + 1] * math.ceil(batch / 2))[:batch]
    rounds = [long, short, long]
    # Chunked prefill does not evict historical KV: complete prefill K/V stays
    # physical until staging/Quest bounds finish and collect_ready_reqs frees
    # surplus pages. Conservatively allow the new hot buffer to coexist with
    # its full prefill, alongside every other admitted request's hot buffer.
    # One prefill request per forward is a control setting, not proof that
    # staging has completed; actual full-batch forwards remain a separate gate.
    steady_hot = max(
        sum(min(rounded(n + args.output_len), hot + page) for n in lengths)
        for lengths in rounds
    )
    full_prefill = max(
        rounded(n + args.output_len) for lengths in rounds for n in lengths
    )
    compute = steady_hot + full_prefill
    backing = max(
        sum(rounded(n + args.output_len) for n in lengths) for lengths in rounds
    )
    allocated_backing_slots = (
        args.max_total_tokens * args.cpu_capacity_ratio // page + 1
    ) * page
    return {
        "rounds": rounds,
        "requested_actual_batch": batch,
        "historical_backing_slots_required": backing,
        "gpu_cold_nopd_admission_slots_required": compute,
        "gpu_steady_decode_hot_slots_required": steady_hot,
        "gpu_full_prefill_slots_required": full_prefill,
        "hot_slots_per_long_request_including_decode_page": hot + page,
        "cpu_capacity_ratio": args.cpu_capacity_ratio,
        "planned_cpu_backing_allocation_slots": allocated_backing_slots,
        "planned_cpu_backing_allocation_bytes": allocated_backing_slots
        * QWEN_KV_BYTES_PER_TOKEN,
        "kv_bytes_per_token": QWEN_KV_BYTES_PER_TOKEN,
        "kv_shape": QWEN_KV_SHAPE,
        "scope": "Cold NoPD admission reserves full prefill plus all hot buffers, including transient new-hot coexistence. Steady decode hot and full CPU logical contexts are separate constraints; allocated GPU KV also includes prefill/free reserve.",
    }


def make_groups(stage, args, batch):
    require(
        stage == CAMPAIGN.recipe(stage["xm_adaptation"]["arm"]),
        "Use the exact CPU summary control recipe",
    )
    require(
        stage["tp_size"] == 1 and stage["variant"] == "cpu",
        "One compute GPU with CPU-DDR KV required",
    )
    require(
        args.prefill_max_requests == 1,
        "The cold admission control requires one prefill request per forward",
    )
    need = allocation_requirements(stage, args, batch)
    require(
        min(args.input_lens)
        > need["hot_slots_per_long_request_including_decode_page"] - stage["page_size"],
        "Long control contexts must exceed the hot KV buffer",
    )
    require(
        args.max_total_tokens >= need["gpu_cold_nopd_admission_slots_required"],
        "Requested GPU full-prefill plus hot slot budget cannot admit the target batch",
    )
    require(
        args.max_total_tokens * args.cpu_capacity_ratio
        >= need["historical_backing_slots_required"],
        "Requested logical CPU backing cannot retain all complete contexts",
    )
    groups = []
    for eager in (True, False):
        gid = (
            "cpu-"
            + ("eager" if eager else "graph")
            + f"-ratio{args.cpu_capacity_ratio}"
        )
        groups.append(
            {
                "id": gid,
                "variant": "cpu",
                "ratio": args.cpu_capacity_ratio,
                "eager": eager,
                "status": "planned",
                "rounds": need["rounds"],
                "lifecycle_required": True,
                "lifecycle_cases": ["abort_prefill", "abort_decode", "retract_decode"],
                "planned_peak_original_token_slots": need[
                    "historical_backing_slots_required"
                ],
                "large_batch_allocation_requirements": need,
                "trace_dir": str(args.output_dir / (gid + "-trace")),
                "server_command": RUN.STAGE.server_command(
                    stage,
                    args,
                    "cpu",
                    args.input_lens[0],
                    args.cpu_capacity_ratio,
                    eager=eager,
                ),
            }
        )
    return groups


def check_actual_capacity(capacities, need):
    require(len(capacities) == 1, "Missing single-compute capacity report")
    capacity = capacities[0]
    require(
        capacity.get("backing_store") == "cpu"
        and not capacity.get("storage_device_ids"),
        "This equivalence control requires CPU-DDR KV, with no MPU device",
    )
    physical = capacity.get("physical_compute_slots")
    backing = capacity.get("backing_slots")
    padding = capacity.get("padding_slots", 0)
    require(
        type(physical) is int
        and physical >= need["gpu_cold_nopd_admission_slots_required"],
        "Actual GPU full-prefill plus hot capacity cannot admit target batch",
    )
    require(
        type(backing) is int
        and type(padding) is int
        and padding >= 0
        and backing - padding >= need["historical_backing_slots_required"],
        "Actual usable CPU backing cannot retain complete paged contexts",
    )
    require(
        backing - padding == physical * need["cpu_capacity_ratio"],
        "Actual CPU backing ratio differs from the separate logical allocation",
    )
    require(
        capacity.get("backing_kv_bytes") == backing * need["kv_bytes_per_token"]
        and capacity.get("compute_kv_bytes") == physical * need["kv_bytes_per_token"],
        "Measured CPU/GPU KV bytes disagree with the complete Qwen BF16 KV shape",
    )
    return {
        "status": "passed",
        "physical_compute_slots": physical,
        "actual_cpu_backing_slots": backing,
        "actual_usable_cpu_backing_slots": backing - padding,
        "requirements": need,
        "scope": "Actual allocation satisfies separate cold full-prefill/hot and full CPU context constraints; complete actual decode batch still requires raw trace evidence. Allocated GPU KV includes prefill/free reserve and is not live hot bytes.",
    }


def observe_cpu_ram():
    """Read host/cgroup memory before server launch; never allocate or use CUDA."""
    import psutil

    memory = psutil.virtual_memory()
    observed = {
        "host_total_bytes": int(memory.total),
        "host_available_bytes": int(memory.available),
        "effective_available_bytes": int(memory.available),
        "source": "psutil.virtual_memory.available before server launch",
    }
    for limit_path, current_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        limit_file, current_file = Path(limit_path), Path(current_path)
        if not limit_file.is_file() or not current_file.is_file():
            continue
        limit, current = (
            limit_file.read_text().strip(),
            current_file.read_text().strip(),
        )
        if limit == "max":
            observed["cgroup_memory_limit"] = "max"
            break
        require(limit.isdigit() and current.isdigit(), "Invalid cgroup RAM metadata")
        limit, current = int(limit), int(current)
        remaining = max(0, limit - current)
        observed.update(
            cgroup_memory_limit_bytes=limit,
            cgroup_memory_current_bytes=current,
            cgroup_memory_remaining_bytes=remaining,
            effective_available_bytes=min(memory.available, remaining),
        )
        break
    return observed


def cpu_ram_preflight(stage, args, batch):
    need = allocation_requirements(stage, args, batch)
    config_path = Path(args.model) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shape = {
        "layers": config.get("num_hidden_layers"),
        "kv_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "element_bytes": 2,
    }
    require(
        shape == QWEN_KV_SHAPE, "RAM preflight requires the actual Qwen BF16 KV shape"
    )
    observed = observe_cpu_ram()
    required = need["planned_cpu_backing_allocation_bytes"] + HOST_MEMORY_RESERVE_BYTES
    return {
        "status": (
            "passed" if observed["effective_available_bytes"] >= required else "failed"
        ),
        "model_config": {"path": str(config_path), **CAMPAIGN.identity(config_path)},
        "kv_shape": shape,
        "full_cpu_kv_allocation_bytes": need["planned_cpu_backing_allocation_bytes"],
        "host_reserve_bytes": HOST_MEMORY_RESERVE_BYTES,
        "minimum_available_bytes_required": required,
        **observed,
        "scope": "Minimum full pinned CPU KV allocation plus the runtime 10 GiB host reserve, checked before each server launch. This is not a complete process/summary/weight memory estimate or proof that pinning succeeds; actual allocation remains independently gated.",
    }


def small_hbm_gate(control):
    proof = control.correctness_campaign.resolve()
    accepted = json.loads(proof.read_text(encoding="utf-8"))
    CAMPAIGN.correctness_gate(
        SimpleNamespace(
            correctness_campaign=proof,
            model=control.model,
            input_lens=[control.input_len],
            disable_cuda_graph=False,
            arms=list(ARMS),
        ),
        {
            "source_fingerprint": CAMPAIGN.source_fingerprint(),
            "workload_identity": CAMPAIGN.identity(control.workload.resolve()),
        },
    )
    marker_path = ROOT / ".mpu-source.json"
    require(
        marker_path.is_file(), "Execute only against the exact deployed archived source"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(
        marker.get("provenance") == "git_archive"
        and re.fullmatch(r"[0-9a-f]{40}", marker.get("source_commit", ""))
        and re.fullmatch(r"[0-9a-f]{64}", marker.get("source_archive_sha256", "")),
        "Invalid deployed archive provenance marker",
    )
    reports = []
    for arm in ARMS:
        case = next(case for case in accepted["cases"] if case["arm"] == arm)
        require(
            case.get("returncode") == 0 and case.get("stage") == CAMPAIGN.recipe(arm),
            "Missing actual successful small-batch arm recipe",
        )
        path = Path(case["stage_path"]).parent / "evidence/manifest.json"
        native = json.loads(path.read_text(encoding="utf-8"))
        runtime = native.get("runtime", {})
        require(
            runtime.get("archived_source") == marker
            and runtime.get("git_commit") == marker.get("source_commit")
            and runtime.get("git_commit_source") == "archive_build_marker"
            and runtime.get("git_status")
            == "unavailable: archived source has no Git index"
            and runtime.get("sglang_source") == str(ROOT / "python/sglang/__init__.py"),
            "Small native correctness did not execute this archived source",
        )
        require(
            {(g.get("variant"), g.get("eager")) for g in native["groups"]}
            >= {("cpu", True), ("cpu", False), ("hbm", True), ("hbm", False)},
            "Small proof must include independent HBM and CPU eager/Graph",
        )
        job = {"report": str(path)}
        step = {
            "id": "correctness",
            "kind": "model",
            "expected_model_stage": case["stage"],
        }
        require(
            ACCEPTANCE.external_result(step, job, case["returncode"])["status"]
            == "passed",
            "Small native HBM/lifecycle proof no longer passes",
        )
        require(
            control.logprob_atol <= native.get("settings", {}).get("logprob_atol", -1),
            "Large control cannot loosen the small correctness tolerance",
        )
        reports.append({"arm": arm, "path": str(path), **CAMPAIGN.identity(path)})
    return {
        "path": str(proof),
        **CAMPAIGN.identity(proof),
        "native_reports": reports,
        "scope": "Independent HBM is proven only in this small-batch precursor, not the large CPU-only comparison",
    }


def verify_round_trace(group, args, batch):
    paths = sorted(Path(group["trace_dir"]).glob("*.jsonl"))
    require(len(paths) == 1, "Require one complete single-compute trace")
    trace = [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forwards = [row for row in trace if row.get("type") == "forward"]
    require(
        trace
        and trace[0].get("type") == "session"
        and trace[-1].get("type") == "idle_flush",
        "Missing complete native session/idle trace",
    )
    session = trace[0]
    require(
        session.get("tp_size") == session.get("pp_size") == session.get("dp_size") == 1
        and session.get("disable_overlap_schedule") is True
        and session.get("disaggregation_mode") == "null"
        and session.get("disable_radix_cache") is True
        and session.get("speculative_algorithm") is None
        and session.get("enable_mixed_chunk") is False,
        "Require actual single-compute NoPD, no prefix/speculative/mixed decode",
    )
    server_info = json.loads(
        (args.output_dir / f"{group['id']}-server-info.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        RUN.decode_graph_enabled(server_info) is not group["eager"],
        "Observed server Graph mode differs from the control",
    )
    states = server_info.get("internal_states", [])
    require(len(states) == 1, "Missing single-compute actual server state")
    if not group["eager"]:
        require(
            batch in states[0].get("cuda_graph_bs", []),
            "Target actual batch shape is absent from the completed Graph capture configuration",
        )
    require(
        [row.get("sequence") for row in forwards] == list(range(len(forwards)))
        and trace[-1].get("last_sequence") == len(forwards) - 1,
        "Incomplete or reordered native forward trace",
    )
    evidence, normalized = [], []
    for index, lengths in enumerate(group["rounds"]):
        response_path = args.output_dir / f"{group['id']}-round{index}.json"
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        outputs = payload["responses"]
        normalized.append(
            CORRECTNESS.COMPARE.normalize_outputs(outputs, lengths, args.output_len)
        )
        ids = [output["meta_info"]["id"] for output in outputs]
        require(
            len(ids) == batch and len(set(ids)) == batch,
            "Missing large-batch request identities",
        )
        members = set(ids)
        rows = [
            row for row in forwards if members.intersection(row.get("request_ids", []))
        ]
        require(
            rows
            and all(
                row.get("success") is True and set(row["request_ids"]) <= members
                for row in rows
            ),
            "Mixed or failed round forwards",
        )
        prefill, decode = dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
        for row in rows:
            require(
                len(row["request_ids"]) == len(set(row["request_ids"])),
                "Duplicate forward request membership",
            )
            if row["forward_mode"] == "extend":
                chunks = row.get("extend_lens", [])
                require(
                    len(chunks) == len(row["request_ids"])
                    and all(type(n) is int and n > 0 for n in chunks),
                    "Missing actual prefill chunk counts",
                )
                require(
                    len(row["request_ids"]) <= args.prefill_max_requests
                    and sum(chunks) <= args.chunked_prefill_size,
                    "Actual prefill exceeded the guarded one-request/chunk slot configuration",
                )
                for rid, count in zip(row["request_ids"], chunks):
                    prefill[rid] += count
            elif row["forward_mode"] == "decode":
                for rid in row["request_ids"]:
                    decode[rid] += 1
            else:
                raise ValueError("Unknown large-batch forward mode")
        require(
            prefill == dict(zip(ids, lengths))
            and decode == dict.fromkeys(ids, args.output_len - 1),
            "Actual round input/decode counts differ from complete responses",
        )
        full = [
            row["sequence"]
            for row in rows
            if row["forward_mode"] == "decode" and set(row["request_ids"]) == members
        ]
        require(
            full,
            "Requested client batch was never observed as the complete actual decode batch",
        )
        evidence.append(
            {
                "round": index,
                "actual_batch_size": batch,
                "full_batch_decode_forwards": len(full),
                "sequences": full,
                "request_ids": ids,
                "response": {
                    "path": str(response_path),
                    **CAMPAIGN.identity(response_path),
                },
            }
        )
    return {
        "status": "passed",
        "graph_scope": "Actual server mode and captured batch-size configuration plus complete decode forwards; per-forward replay flag is not emitted by this native observer",
        "trace": {"path": str(paths[0]), **CAMPAIGN.identity(paths[0])},
        "rounds": evidence,
    }, normalized


def execute_native(stage, args, groups, batch):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = RUN.new_manifest(stage, args, groups, "correctness")
    manifest["large_batch_reference_scope"] = (
        "CPU-DDR resident/offloaded summaries with the same fused kernel; no large-batch HBM reference"
    )
    path = args.output_dir / "manifest.json"
    original_wait = RUN.wait_capacities
    original_start = RUN.COMMON.start_server
    gates, ram_gates = [], []
    need = allocation_requirements(stage, args, batch)

    def guarded_wait(*positional, **keyword):
        capacities = original_wait(*positional, **keyword)
        try:
            checked = check_actual_capacity(capacities, need)
        except ValueError as error:
            gates.append(
                {
                    "status": "failed",
                    "error": str(error),
                    "capacities": capacities,
                    "requirements": need,
                }
            )
            raise
        gates.append(checked)
        return capacities

    def guarded_start(*positional, **keyword):
        try:
            checked = cpu_ram_preflight(stage, args, batch)
        except (ImportError, RuntimeError, ValueError, OSError) as error:
            checked = {"status": "failed", "error": str(error)}
        ram_gates.append(checked)
        manifest["cpu_ram_preflight"] = ram_gates
        RUN.COMMON.write_manifest(path, manifest)
        require(
            checked["status"] == "passed",
            "CPU RAM preflight failed before server launch",
        )
        return original_start(*positional, **keyword)

    RUN.wait_capacities = guarded_wait
    RUN.COMMON.start_server = guarded_start
    try:
        manifest["status"] = (
            "success" if CORRECTNESS.execute(stage, args, manifest, path) else "failed"
        )
        if manifest["status"] == "success":
            for group in groups:
                group["large_batch_trace_equivalence"], _ = verify_round_trace(
                    group, args, batch
                )
    except (ImportError, RuntimeError, ValueError, OSError, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=str(error))
    finally:
        RUN.wait_capacities = original_wait
        RUN.COMMON.start_server = original_start
        manifest["actual_allocation_gates"] = gates
        manifest["cpu_ram_preflight"] = ram_gates
        RUN.COMMON.write_manifest(path, manifest)
    require(
        manifest["status"] == "success",
        "Native large-batch correctness failed; inspect saved raw evidence",
    )
    return manifest


def compare_arms(left, right, left_args, right_args, batch):
    result = []
    require(
        left["stage"] == CAMPAIGN.recipe("fused")
        and right["stage"] == CAMPAIGN.recipe("combined-protected"),
        "Different summary comparison recipes",
    )
    require(
        len(left["groups"]) == len(right["groups"]) == 2
        and {g["eager"] for g in left["groups"]} == {True, False},
        "Require complete eager and Graph control groups",
    )
    for a, b in zip(left["groups"], right["groups"]):
        require(
            a.get("status") == b.get("status") == "success",
            "Native control groups have not actually passed",
        )
        require(
            a["eager"] == b["eager"] and a["rounds"] == b["rounds"],
            "Different actual round/mode controls",
        )
        require(
            len(a.get("capacity", [])) == len(b.get("capacity", [])) == 1
            and all(
                a["capacity"][0].get(key) == b["capacity"][0].get(key)
                for key in (
                    "physical_compute_slots",
                    "compute_kv_bytes",
                    "backing_slots",
                    "backing_kv_bytes",
                )
            ),
            "Resident/offloaded controls used different full KV allocations",
        )
        for index in range(3):
            x = json.loads(
                (left_args.output_dir / f"{a['id']}-round{index}.json").read_text()
            )
            y = json.loads(
                (right_args.output_dir / f"{b['id']}-round{index}.json").read_text()
            )
            require(
                x["input_ids"] == y["input_ids"],
                "Resident/offloaded summaries received different input token sets",
            )
        _, expected = verify_round_trace(a, left_args, batch)
        _, observed = verify_round_trace(b, right_args, batch)
        comparison = CORRECTNESS.COMPARE.compare(
            expected, observed, left_args.logprob_atol
        )
        result.append({"eager": a["eager"], "actual_batch_size": batch, **comparison})
    return result


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--correctness-campaign", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--output-len", type=int, default=80)
    parser.add_argument("--max-total-tokens", type=int, default=98304)
    parser.add_argument("--capacity-ratio", type=int, default=16)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--logprob-atol", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    control = parser.parse_args(argv)
    if (
        min(
            control.batch_sizes
            + [control.input_len, control.max_total_tokens, control.capacity_ratio]
        )
        <= 0
        or control.output_len < 80
    ):
        parser.error(
            "Positive capacities/batches and at least 80 decode tokens are required"
        )
    if not math.isfinite(control.logprob_atol) or not 0 <= control.logprob_atol <= 0.02:
        parser.error(
            "Finite nonnegative logprob tolerance no looser than 0.02 required"
        )
    if len(set(control.batch_sizes)) != len(control.batch_sizes):
        parser.error("Batch controls must not repeat an evidence directory")
    control.output_dir = control.output_dir.resolve()
    return control


def native_settings(control, batch, arm):
    case = control.output_dir / f"b{batch}-{arm}"
    stage = CAMPAIGN.recipe(arm)
    stage_path = case / "stage.json"
    CAMPAIGN.write(stage_path, stage)
    stage, args = RUN.STAGE.parse_args(
        [
            "--stage",
            str(stage_path),
            "--model",
            control.model,
            "--workload",
            str(control.workload.resolve()),
            "--input-lens",
            str(control.input_len),
            "--concurrencies",
            str(batch),
            "--num-prompts",
            str(batch),
            "--output-len",
            str(control.output_len),
            "--max-total-tokens",
            str(control.max_total_tokens),
            "--capacity-ratios",
            str(control.capacity_ratio),
            "--cpu-capacity-ratio",
            str(control.capacity_ratio),
            "--chunked-prefill-size",
            "4096",
            "--prefill-max-requests",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--seed",
            "1729",
            "--port",
            str(control.port),
            "--logprob-atol",
            str(control.logprob_atol),
            "--lifecycle",
            "--output-dir",
            str(case / "evidence"),
        ],
        correctness=True,
    )
    return stage, args, make_groups(stage, args, batch)


def main(argv=None):
    control = parse(argv)
    require(not control.output_dir.exists(), "Use a new large-batch evidence directory")
    control.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "phase": "large_batch_summary_equivalence",
        "status": "planned",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(control).items()
        },
        "source_fingerprint": CAMPAIGN.source_fingerprint(),
        "small_hbm_gate": {
            "status": "pending_execution_gate",
            "path": str(control.correctness_campaign.resolve()),
        },
        "cases": [],
        "comparisons": [],
        "reference_scope": "Large batch compares CPU-DDR full KV with resident vs offloaded Quest summaries using the same fused numerical kernel. Only small-batch HBM is independently proven.",
        "auxiliary_input_scope": "Long/reuse rounds use the same frozen real documents; short/lifecycle rounds reuse the existing native synthetic correctness fixtures",
        "allocation_scope": "Cold NoPD requires complete prefill KV plus admitted/transient hot buffers. Steady decode hot, logical full CPU backing, and allocated GPU KV pool are reported separately.",
    }
    destination = control.output_dir / "summary-batch-correctness.json"
    try:
        CAMPAIGN.write(destination, manifest)
        if control.execute:
            try:
                manifest["small_hbm_gate"] = {
                    "status": "passed",
                    **small_hbm_gate(control),
                }
            except (ImportError, RuntimeError, ValueError, OSError) as error:
                manifest.update(
                    status="blocked", failure_phase="small_hbm_gate", error=str(error)
                )
                manifest["small_hbm_gate"] = {
                    "status": "failed",
                    "error": str(error),
                    "path": str(control.correctness_campaign.resolve()),
                }
                raise
            CAMPAIGN.write(destination, manifest)
        for batch in control.batch_sizes:
            measured, settings = {}, {}
            for arm in ARMS:
                stage, args, groups = native_settings(control, batch, arm)
                case = {
                    "arm": arm,
                    "batch": batch,
                    "stage": stage,
                    "settings": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "groups": groups,
                    "status": "planned",
                    "native_manifest": str(args.output_dir / "manifest.json"),
                }
                manifest["cases"].append(case)
                CAMPAIGN.write(destination, manifest)
                if control.execute:
                    case["status"] = "running"
                    CAMPAIGN.write(destination, manifest)
                    measured[arm] = execute_native(stage, args, groups, batch)
                    settings[arm] = args
                    case["status"] = "success"
                    CAMPAIGN.write(destination, manifest)
            if control.execute:
                result = compare_arms(
                    measured[ARMS[0]],
                    measured[ARMS[1]],
                    settings[ARMS[0]],
                    settings[ARMS[1]],
                    batch,
                )
                manifest["comparisons"].append(
                    {"batch": batch, "status": "passed", "groups": result}
                )
        if control.execute:
            manifest["status"] = "success"
    except (ImportError, RuntimeError, ValueError, OSError, KeyboardInterrupt) as error:
        if manifest["status"] != "blocked":
            manifest.update(status="failed", error=str(error))
        for case in manifest["cases"]:
            if case["status"] == "running":
                case["status"] = "failed"
    finally:
        CAMPAIGN.write(destination, manifest)
    print(destination)
    return 2 if manifest["status"] in ("failed", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
