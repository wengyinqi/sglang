"""CPU report bindings for source017 controls; never launch a GPU or rewrite raw.

The external baseline controller remains source017-bound. This module reuses
its original raw replay rather than trusting its final status or inventing a
weaker numerical gate. Report tools and executed model sources stay separate.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

SOURCE_COMMIT = "3ca21f0089b30e13b254491f2b086f073ad11f10"
CORE_SHA256 = "d763476a8bc735c8197a9c0208a49493e0c4a66eb9c34974b46ccd5615fcade0"
CONTROLLER_SHA256 = "a41abf59102c04969c2a7fa3dc833ad024aaad03387d84ecc36dbad52a358bcd"
RUNNER_SHA256 = "f55afa019e2156e4061be099d7ef309006e6914fec01971d72813231700c01a2"
RUNTIME_IMAGE = (
    "sha256:2e5dc318b02a18a16579bb6a9a25119db101108603bbe9a4074ac6e0b6090425"
)
PRECURSOR_KIND = "source017_explicit_full32k_B4_precursor_binding"
BASELINE_KIND = "large_CPU_baseline_kernel_equivalence"
HTTP_KIND = "cpu_raw_replayed_complete_HTTP_paired_effects"
BASELINE_CUSTODY_KIND = "external_baseline017_actual_evidence_custody"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(left, right):
    return all(left.get(key) == right.get(key) for key in ("bytes", "sha256"))


def source017(data):
    return (
        data.get("native_source_commits") == [SOURCE_COMMIT]
        and data.get("source_fingerprint", {}).get("sha256") == CORE_SHA256
    )


def controller(path, report):
    require(
        report.identity(path)["sha256"] == CONTROLLER_SHA256,
        "External baseline controller is not the independently reviewed source017 bytes",
    )
    return load(path, "report_source017_external_controller")


def derive_precursor(settings, controller_path, report):
    tool = controller(controller_path, report)
    runtime, source = tool.runtime_for(settings)
    replay, _ = tool.precursor_gate(runtime, settings)
    return {
        "schema_version": 1,
        "kind": PRECURSOR_KIND,
        "status": "passed_CPU_raw_replay",
        "source_fingerprint": source["core_source_fingerprint"],
        "native_source_commits": [SOURCE_COMMIT],
        "settings": copy.deepcopy(settings),
        "controller": report.identity(controller_path),
        "original_campaign": report.identity(settings["correctness_campaign"]),
        "original_acceptance": report.identity(settings["full32k_acceptance"]),
        "original_seal": report.identity(settings["correctness_seal"]),
        "precursor_capacity_ratio": settings["precursor_capacity_ratio"],
        "large_control_capacity_ratio": 8,
        "replay": replay,
        "GPU_execution": False,
        "capacity_budget_equivalence_proven": False,
        "scope": "Original explicit ratio4/8 phase, seal, argv, actual physical132608/full4context131584 and CPU/HBM raw execution; this precursor is model correctness, not the large ratio8 capacity budget.",
    }


def verify_precursor(proof_path, proof, evidence, report):
    if proof.get("settings", {}).get("input_lens") != [32768] * 4:
        return None
    candidates = [
        (path, record)
        for path, record in evidence
        if record.get("kind") == PRECURSOR_KIND
        and same(record.get("original_campaign", {}), report.identity(proof_path))
    ]
    require(
        len(candidates) == 1,
        "Missing unambiguous explicit source017 full32K B4 precursor binding",
    )
    path, record = candidates[0]
    require(
        record.get("status") == "passed_CPU_raw_replay",
        "Explicit precursor binding did not pass",
    )
    fresh = derive_precursor(record["settings"], record["controller"]["path"], report)
    require(
        fresh == record,
        "Explicit precursor binding differs from original raw/phase/seal replay",
    )
    require(
        fresh["precursor_capacity_ratio"] == proof["settings"].get("capacity_ratio"),
        "Precursor ratio changed or was normalized to the large-control ratio",
    )
    return {"binding": report.identity(path), **fresh}


def request_capacity(command, info):
    flag = "--max-running-requests"
    require(
        command.count(flag) == 1 and command.index(flag) + 1 < len(command),
        "Missing unambiguous requested request-slot limit",
    )
    text = command[command.index(flag) + 1]
    require(
        isinstance(text, str) and text.isdigit(), "Invalid requested request-slot limit"
    )
    requested = int(text)
    states = info.get("internal_states", [])
    require(
        len(states) == 1, "Request-slot audit requires one observed compute scheduler"
    )
    effective = states[0].get("effective_max_running_requests_per_dp")
    require(
        type(effective) is int and 0 < effective <= requested,
        "Missing/invalid observed effective request-slot capacity",
    )
    return requested, effective


def quest_geometry(capacity, page_size=64):
    shape = (
        capacity.get("gpu_memory_ledger", {})
        .get("named_compute_shapes", {})
        .get("fixed.request_map", {})
        .get("shape")
    )
    require(
        isinstance(shape, list)
        and len(shape) == 2
        and all(type(value) is int and value > 0 for value in shape),
        "Missing actual Quest request-map R/context geometry",
    )
    require(type(page_size) is int and page_size > 0, "Invalid actual Quest page size")
    return {
        "quest_rows_R": shape[0],
        "quest_context_columns": shape[1],
        "quest_pages_P": (shape[1] + page_size - 1) // page_size,
        "geometry_scope": "Actual req_to_token tensor rows/context columns; Quest R is rows, P is ceil(max context columns/page). CPU backing ratio does not define P.",
    }


def capacity_layouts(data, report):
    campaign = read(data["artifacts"][0]["path"])
    outer = {case["id"]: case for case in campaign["cases"]}
    layouts = []
    for artifact in data["artifacts"]:
        path = Path(artifact["path"])
        if path.name != "manifest.json":
            continue
        native = read(path)
        if native.get("kind") != "benchmark":
            continue
        case = outer[path.parent.parent.name]
        for group in native["groups"]:
            info_path = path.parent / (group["id"] + "-server-info.json")
            requested, effective = request_capacity(
                group["server_command"], read(info_path)
            )
            require(
                len(group.get("capacity", [])) == 1,
                "Missing actual single-compute Quest allocation geometry",
            )
            geometry = quest_geometry(
                group["capacity"][0], native["stage"]["page_size"]
            )
            for measured in group["cases"]:
                layouts.append(
                    {
                        "arm": case["arm"],
                        "input_len": group["input_len"],
                        "client_concurrency": measured["concurrency"],
                        "repeat": case.get("global_repeat", measured["repeat"]),
                        "graph_enabled": group["decode_cuda_graph_enabled"],
                        "requested_max_running_requests": requested,
                        "request_capacity_R": effective,
                        "native": report.identity(path),
                        "server_info": report.identity(info_path),
                        **geometry,
                        "R_scope": "Observed effective scheduler admission/request-slot limit; not an assertion of the underlying tensor's padded row count.",
                    }
                )
    return layouts


def external_baseline_custody(control_path, control, evidence, report):
    candidates = [
        (path, record)
        for path, record in evidence
        if record.get("kind") == BASELINE_CUSTODY_KIND
        and same(record.get("execution_receipt", {}), report.identity(control_path))
    ]
    require(
        len(candidates) == 1,
        "Missing dedicated external-baseline execution custody; campaign seals cannot substitute",
    )
    path, custody = candidates[0]
    require(
        custody.get("status") == "sealed_actual_external_baseline_evidence",
        "External baseline execution custody did not pass",
    )
    require(
        custody.get("execution_status") == "success"
        and all(
            custody.get(key) is False
            for key in (
                "model_correctness_result",
                "large_hbm_model_correctness_proven",
                "capacity_proof",
                "performance_result",
            )
        ),
        "External custody is not successful byte preservation with separate proof gates",
    )
    require(
        custody.get("source_marker") == control.get("source_marker")
        and custody.get("core_source_fingerprint")
        == control.get("core_source_fingerprint"),
        "External custody source identity differs",
    )
    root = Path(path).resolve(strict=True).parent
    inventory = {}
    for item in custody.get("files", []):
        relative = PurePosixPath(item.get("path", ""))
        require(
            relative.parts
            and not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in str(relative),
            "Unsafe external custody relative path",
        )
        target = root.joinpath(*relative.parts).resolve(strict=True)
        target.relative_to(root)
        require(
            target not in inventory and same(report.identity(target), item),
            "Duplicate/changed external custody artifact",
        )
        inventory[target] = item
    require(inventory, "Empty external baseline custody inventory")

    def bind(item):
        value = str(item["path"])
        declared = Path(value)
        try:
            target = declared.resolve(strict=True)
            target.relative_to(root)
        except (OSError, ValueError):
            original = PurePosixPath(value)
            prefix = (
                PurePosixPath("/results")
                if value.startswith("/results/")
                else PurePosixPath(custody["evidence_root"])
            )
            relative = original.relative_to(prefix)
            require(
                ".." not in relative.parts, "Unsafe mapped external custody artifact"
            )
            target = root.joinpath(*relative.parts).resolve(strict=True)
            target.relative_to(root)
        require(
            target in inventory and same(report.identity(target), item),
            "External receipt artifact not bound by dedicated custody",
        )
        return report.identity(target)

    execution = bind(custody["execution_receipt"])
    require(
        same(execution, report.identity(control_path)),
        "Dedicated custody binds another final receipt",
    )
    plan = bind(custody["reviewed_plan"])
    require(
        same(plan, control["plan"]), "Dedicated custody binds another reviewed plan"
    )
    contract = custody["runner_contract"]
    contract_manifest = bind(contract["manifest"])
    controller_id, runner_id = bind(contract["controller"]), bind(contract["runner"])
    require(
        controller_id["sha256"] == CONTROLLER_SHA256
        and runner_id["sha256"] == RUNNER_SHA256,
        "External controller/runner differs from the report-tool delivery contract",
    )
    job = custody.get("job", {})
    job_file = bind(custody["job_file"])
    require(
        read(job_file["path"]) == job,
        "Dedicated custody embedded job differs from original producer file",
    )
    require(
        same(job.get("contract", {}), contract_manifest)
        and read(contract_manifest["path"]).get("gpu_execution_allowed") is True,
        "Dedicated custody is not bound to its authorized actual runner contract",
    )
    command = job.get("command", [])
    require(
        isinstance(command, list) and command.count(RUNTIME_IMAGE) == 1,
        "Missing actual pinned-image baseline execution command",
    )
    argv = command[command.index(RUNTIME_IMAGE) + 1 :]
    require(
        argv == custody.get("actual_execution_argv")
        and argv[:2] == ["/results/tools/large_cpu_baseline_control017.py", "execute"],
        "Dedicated custody did not preserve actual controller execution argv",
    )
    flag = "--reviewed-plan-sha256"
    require(
        isinstance(argv, list)
        and job.get("kind") == "external_baseline017_dedicated_runner_job"
        and job.get("mode") == "execute"
        and job.get("gpu_requested") is True
        and job.get("status") == "finished"
        and job.get("returncode") == 0
        and argv.count("execute") == 1
        and argv.count(flag) == 1
        and argv.index(flag) + 1 < len(argv)
        and argv[argv.index(flag) + 1] == plan["sha256"],
        "Dedicated custody is not a finished hash-bound baseline execution job",
    )
    native_reports = []
    for item in custody.get("native_reports", []):
        require(
            type(item.get("batch")) is int
            and item["batch"] > 0
            and item.get("arm") == "baseline"
            and item.get("native_status") == "success",
            "Dedicated custody lacks a successful original per-B baseline manifest",
        )
        native_reports.append(bind(item["native"]))
    require(native_reports, "Dedicated custody lacks original baseline native reports")
    return {
        "custody": report.identity(path),
        "execution_receipt": execution,
        "reviewed_plan": plan,
        "controller": controller_id,
        "runner": runner_id,
        "runner_contract": contract_manifest,
        "job_file": job_file,
        "native_reports": native_reports,
        "verified_inventory": [report.identity(target) for target in inventory],
    }


def verify_external_baseline(
    path, record, batch, request_limit, quest_rows, report, evidence=()
):
    require(
        record.get("kind") == BASELINE_KIND and record.get("status") == "success",
        "External baseline control did not actually finish",
    )
    require(
        record.get("large_hbm_model_correctness_proven") is False
        and record.get("capacity_proof") is False
        and record.get("performance_result") is False,
        "CPU model control cannot grant capacity/performance or large-HBM proof",
    )
    plan_path = record["plan"]["path"]
    require(
        same(report.identity(plan_path), record["plan"]),
        "Baseline reviewed plan changed",
    )
    plan = read(plan_path)
    require(
        plan.get("kind") == BASELINE_KIND
        and plan.get("status") == "ready"
        and plan.get("settings") == record.get("settings"),
        "Baseline result differs from its ready reviewed plan",
    )
    tool = controller(plan["controller"]["path"], report)
    require(
        same(report.identity(plan["controller"]["path"]), plan["controller"]),
        "Baseline controller identity differs from reviewed plan",
    )
    settings = record["settings"]
    require(
        plan.get("bound_inputs") == tool.bound_inputs(settings),
        "Baseline source/model/workload/proof changed after review",
    )
    runtime, gates, references = tool.all_gates(settings)
    require(
        gates == plan.get("prerequisites") == record.get("prerequisites"),
        "Baseline prerequisite replay changed",
    )
    cases = [case for case in record.get("cases", []) if case.get("batch") == batch]
    require(
        len(cases) == 1 and cases[0].get("status") == "success",
        "Missing successful original baseline at this exact actual batch",
    )
    case = cases[0]
    require(
        case.get("request_slot_capacity_R") == request_limit,
        "Actual batch control has another R; B11/R11 cannot cover C12/R12/B11",
    )
    custody = external_baseline_custody(path, record, evidence, report)
    native_path = Path(case["native_manifest"])
    native = read(native_path)
    require(
        same(report.identity(native_path), case["native"]),
        "Original baseline native artifact changed",
    )
    require(
        any(
            same(report.identity(native_path), item)
            for item in custody["native_reports"]
        ),
        "Original baseline native report is absent from dedicated custody",
    )
    replay = tool.validate_native(
        runtime, native, native_path, "baseline", settings, batch
    )
    require(replay == case.get("raw_replay"), "Original baseline raw execution differs")
    lifecycle = tool.replay_lifecycle(runtime, native, native_path)
    require(
        lifecycle == case.get("lifecycle_artifacts"),
        "Original baseline lifecycle replay differs",
    )
    artifacts = [*lifecycle, report.identity(native_path)]
    for group in replay["groups"]:
        artifacts.extend(
            [
                group["trace"],
                group["server_info"],
                *[row["response"] for row in group["rounds"]],
            ]
        )
    for item in artifacts:
        require(
            any(
                Path(stored["path"]).resolve() == Path(item["path"]).resolve()
                and same(stored, item)
                for stored in custody["verified_inventory"]
            ),
            "Baseline replay read an artifact absent from dedicated custody",
        )
    for group in native["groups"]:
        info = read(native_path.parent / (group["id"] + "-server-info.json"))
        requested, effective = request_capacity(group["server_command"], info)
        require(
            requested == effective == request_limit,
            "Original baseline actual request-slot capacity differs",
        )
        require(
            len(group.get("capacity", [])) == 1
            and quest_geometry(group["capacity"][0])["quest_rows_R"] == quest_rows,
            "Original baseline actual Quest row capacity differs",
        )
    comparisons = []
    for arm in tool.ARMS:
        reference, reference_path = references[(batch, arm)]
        actual = tool.compare_native(
            runtime,
            native,
            reference,
            native_path,
            reference_path,
            settings["logprob_atol"],
        )
        saved = [
            item
            for item in record.get("comparisons", [])
            if item.get("batch") == batch and item.get("optimized_arm") == arm
        ]
        require(
            len(saved) == 1
            and saved[0].get("status") == "passed"
            and saved[0].get("reference_arm") == "baseline"
            and saved[0].get("groups") == actual,
            "Original baseline/optimized full-response comparison changed",
        )
        comparisons.append({"arm": arm, "groups": actual})
    return {
        "status": "passed",
        "input_len": settings["input_len"],
        "batch": batch,
        "request_capacity_R": request_limit,
        "quest_rows_R": quest_rows,
        "control": report.identity(path),
        "controller": plan["controller"],
        "reviewed_plan": report.identity(plan_path),
        "execution_custody": custody,
        "precursor_capacity_ratio": settings["precursor_capacity_ratio"],
        "large_control_capacity_ratio": settings["capacity_ratio"],
        "comparisons": comparisons,
        "large_hbm_model_correctness_proven": False,
        "capacity_proof": False,
        "scope": "Original FI CPU-DDR baseline versus fused resident and combined-protected at the exact actualB/R; raw long-short-long,eager/Graph,lifecycle,inputs/greedy/LP replayed. This is a model-control binding, not the measured8GiB capacity result.",
    }


def capacity_point_gate(data, proof_path, proof, receipts, evidence, report):
    layouts = capacity_layouts(data, report)
    points = {
        (
            row["arm"],
            row["input_len"],
            row["client_concurrency"],
            row.get("actual_batch", row.get("actual_batch_size")),
            report.measurement_graph(data, row),
        )
        for row in [
            *data.get("rows", []),
            *data.get("observed_stable_capacity_points", []),
        ]
        if row.get("eligible", True) is True
        and type(row.get("actual_batch", row.get("actual_batch_size"))) is int
    }
    accepted, pending, controls = [], [], []
    verified_controls = {}
    covered = {
        item["arm"]: item.get("hbm_context_batch_sizes", []) for item in receipts
    }
    for arm, length, concurrency, batch, graph in sorted(points):
        point = {
            "arm": arm,
            "input_len": length,
            "client_concurrency": concurrency,
            "actual_batch_size": batch,
            "graph_enabled": graph,
        }
        try:
            matched = [
                item
                for item in layouts
                if all(
                    item[key] == point[key]
                    for key in (
                        "arm",
                        "input_len",
                        "client_concurrency",
                        "graph_enabled",
                    )
                )
            ]
            limits = {item["request_capacity_R"] for item in matched}
            require(
                matched and len(limits) == 1,
                "Observed R differs across repeats or is missing",
            )
            request_limit = next(iter(limits))
            point["request_capacity_R"] = request_limit
            geometries = {
                (
                    item["quest_rows_R"],
                    item["quest_context_columns"],
                    item["quest_pages_P"],
                )
                for item in matched
            }
            require(
                len(geometries) == 1, "Actual Quest R/P geometry differs across repeats"
            )
            quest_rows, context_columns, pages = next(iter(geometries))
            point.update(
                quest_rows_R=quest_rows,
                quest_context_columns=context_columns,
                quest_pages_P=pages,
            )
            small = max(
                (size for context, size in covered.get(arm, []) if context == length),
                default=0,
            )
            small_case = next(case for case in proof["cases"] if case["arm"] == arm)
            small_path, small_native = report._native_correctness(
                proof_path, small_case, evidence
            )
            small_limits = {
                request_capacity(
                    group["server_command"],
                    read(small_path.parent / (group["id"] + "-server-info.json")),
                )[1]
                for group in small_native["groups"]
            }
            small_rows = {
                quest_geometry(group["capacity"][0])["quest_rows_R"]
                for group in small_native["groups"]
                if group["variant"] == "cpu" and len(group.get("capacity", [])) == 1
            }
            if (
                batch <= small
                and small_limits == {request_limit}
                and small_rows == {quest_rows}
            ):
                point.update(
                    status="passed",
                    basis="ExactR independent CPU/HBM precursor actualB coverage",
                )
            else:
                require(
                    arm in ("baseline", "fused", "combined-protected"),
                    "No reviewed large original-baseline control for this capacity arm",
                )
                candidates = [
                    (path, record)
                    for path, record in evidence
                    if record.get("kind") == BASELINE_KIND
                    and record.get("source_marker", {}).get("source_commit")
                    == SOURCE_COMMIT
                    and record.get("core_source_fingerprint", {}).get("sha256")
                    == CORE_SHA256
                    and record.get("settings", {}).get("input_len") == length
                    and any(
                        case.get("batch") == batch for case in record.get("cases", [])
                    )
                ]
                require(
                    len(candidates) == 1,
                    "Missing unambiguous original baseline CPU control for this actualB/R",
                )
                path, record = candidates[0]
                key = (
                    report.identity(path)["sha256"],
                    batch,
                    request_limit,
                    quest_rows,
                )
                if key not in verified_controls:
                    verified_controls[key] = verify_external_baseline(
                        path, record, batch, request_limit, quest_rows, report, evidence
                    )
                    controls.append(verified_controls[key])
                point.update(
                    status="passed",
                    basis="ExactB/R original CPU baseline plus both optimized controls",
                )
            accepted.append(point)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            point.update(status="pending_model_control", reason=str(error))
            pending.append(point)
    return {
        "status": (
            "partial" if pending and accepted else ("passed" if accepted else "pending")
        ),
        "accepted": accepted,
        "pending": pending,
        "request_layouts": layouts,
        "external_baseline_controls": controls,
        "scope": "Raw C/R/actualB observations remain intact; only individually covered points may support capacity/model-benefit claims. Missing C12/R12/B12 or B11 control is not a measured maximumB8.",
    }


def filter_capacity(data, gate):
    result = copy.deepcopy(data)
    allowed = {
        (
            p["arm"],
            p["input_len"],
            p["client_concurrency"],
            p["actual_batch_size"],
            p["graph_enabled"],
        )
        for p in gate["accepted"]
    }
    lookup = {
        key: p
        for p in gate["accepted"]
        for key in [
            (
                p["arm"],
                p["input_len"],
                p["client_concurrency"],
                p["actual_batch_size"],
                p["graph_enabled"],
            )
        ]
    }
    for name in ("rows", "observed_stable_capacity_points"):
        shown = []
        for row in result.get(name, []):
            key = (
                row["arm"],
                row["input_len"],
                row["client_concurrency"],
                row.get("actual_batch", row.get("actual_batch_size")),
                row.get("graph_enabled"),
            )
            if key in allowed:
                row.update(
                    {
                        name: lookup[key][name]
                        for name in (
                            "request_capacity_R",
                            "quest_rows_R",
                            "quest_context_columns",
                            "quest_pages_P",
                        )
                    }
                )
                shown.append(row)
        result[name] = shown
    keep_cases = {
        (p["arm"], p["input_len"], p["client_concurrency"], p["graph_enabled"])
        for p in gate["accepted"]
    } - {
        (p["arm"], p["input_len"], p["client_concurrency"], p["graph_enabled"])
        for p in gate["pending"]
    }
    result["samples"] = [
        sample
        for sample in result.get("samples", [])
        if tuple(
            sample[k]
            for k in ("arm", "input_len", "client_concurrency", "graph_enabled")
        )
        in keep_cases
    ]
    for name in ("paired_repeat_rows", "same_actual_batch_comparisons"):
        result[name] = [
            row
            for row in result.get(name, [])
            if all(
                (
                    arm,
                    row["input_len"],
                    row["client_concurrency"],
                    row["actual_batch_size"],
                    row.get("graph_enabled"),
                )
                in allowed
                for arm in (row["baseline"], row["changed"])
            )
        ]
    result["unqualified_capacity_points"] = copy.deepcopy(gate["pending"])
    result["request_layouts"] = copy.deepcopy(gate["request_layouts"])
    result["capacity_display_is_derived"] = True
    result.setdefault("excluded_cases", []).extend(
        {
            "case": f"{p['arm']}/L={p['input_len']}/C={p['client_concurrency']}/R={p.get('request_capacity_R', 'unknown')}/B={p['actual_batch_size']}",
            "reason": p["reason"],
        }
        for p in gate["pending"]
    )
    if not result.get("rows") and not result.get("observed_stable_capacity_points"):
        result["status"] = "no_eligible_samples"
    return result


def verify_http_effects(analysis_path, data, evidence, report):
    if data.get("phase") != "performance":
        return None
    candidates = [
        (path, record)
        for path, record in evidence
        if record.get("kind") == HTTP_KIND
        and any(
            same(item, report.identity(analysis_path))
            for item in record.get("input_artifacts", [])
        )
    ]
    if not candidates:
        return {
            "status": "pending_complete_HTTP_effects",
            "scope": "Stable scheduler intervals do not replace Table3 HTTP meanTBT/TPOT.",
        }
    require(len(candidates) == 1, "Ambiguous complete HTTP effects for this analysis")
    path, record = candidates[0]
    require(
        record.get("status") == "passed"
        and record.get("model_correctness_proof") is False,
        "HTTP metric derivation cannot grant model correctness",
    )
    tool = report.report_tool("client_paired_effects")
    derivation = record["derivation_tool"]
    require(
        same(report.identity(tool.__file__), derivation),
        "HTTP derivation helper changed",
    )
    for item in [
        *record["input_artifacts"],
        *record["raw_replay_artifacts"],
        *record["raw_HTTP_artifacts"],
    ]:
        require(
            same(report.identity(item["path"]), item),
            "HTTP derivation raw/tool/analysis input changed",
        )
    campaign_path = data["artifacts"][0]["path"]
    campaign = read(campaign_path)
    outputs = {
        int(case["command"][case["command"].index("--output-len") + 1])
        for case in campaign["cases"]
    }
    capacities = {sample["client_concurrency"] for sample in data["samples"]}
    require(
        len(outputs) == len(capacities) == 1,
        "HTTP effect recipe has ambiguous output/C",
    )
    analyzer = record["input_artifacts"][2]["path"]
    args = SimpleNamespace(
        tool_root=Path(analyzer).parents[2],
        campaign=campaign_path,
        evidence_root=None,
        expected_native_source=data["native_source_commits"][0],
        expected_core_sha256=data["source_fingerprint"]["sha256"],
        saved_analysis=analysis_path,
        input_lens=sorted({s["input_len"] for s in data["samples"]}),
        output_len=next(iter(outputs)),
        client_concurrency=next(iter(capacities)),
    )
    fresh = tool.derive(args)
    require(
        all(
            report.report_tool("analyze").equal(record.get(key), value)
            for key, value in fresh.items()
        ),
        "HTTP effects differ from independent complete frame/raw replay",
    )
    return {"status": "passed", "evidence": report.identity(path), **fresh}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "source-archive",
        "evidence-root",
        "correctness-campaign",
        "correctness-seal",
        "full32k-acceptance",
        "workload",
        "controller",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--precursor-capacity-ratio", type=int, choices=(4, 8), required=True
    )
    parser.add_argument("--logprob-atol", type=float, default=0.02)
    args = parser.parse_args(argv)
    report = load(Path(__file__).with_name("report.py"), "precursor_binding_report")
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("controller", "output")
    }
    result = derive_precursor(settings, args.controller, report)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
