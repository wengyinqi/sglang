"""Synthetic plans and failure reports only; no GPU or model results."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "benchmark/mpu_stages"
spec = importlib.util.spec_from_file_location(
    "experiment_matrix", HERE / "experiment_matrix.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture(tmp_path, family="qwen", variant="mpu", topology="default"):
    matrix = M.read_json(HERE / "experiment-matrix.json")
    selected = matrix["families"][family]
    tp = 1 if family == "qwen" else 4
    config = {
        "top_k": selected["top_k"],
        "device_buffer_size": selected["device_buffer_size"],
    }
    stage = {
        "schema_version": 1,
        "id": f"synthetic-{family}-{variant}",
        "family": family,
        "tp_size": tp,
        "attn_dp_size": tp,
        "ep_size": tp,
        "variant": variant,
        "default_model": "synthetic/model",
        "default_revision": "a" * 40,
        "selector": selected["selector"],
        "hbm_preserves_selector": True,
        "frozen_host_budget": tp == 4,
        "physical_budget_exact": True,
        "sparse_context_multiplier": 4 if family == "deepseek-v4-flash" else 1,
        "page_size": 256 if family == "deepseek-v4-flash" else 64,
        "variants": {
            "cpu": {
                "server_args": [],
                "hisparse_config": {**config, "backing_store": "cpu"},
            },
            "mpu": {
                "server_args": [],
                "hisparse_config": {
                    **config,
                    "backing_store": "mpu",
                    "mpu_device_ids": [1] if tp == 1 else [4, 5, 6, 7],
                },
            },
            "hbm": {"server_args": [], "hisparse_config": {}},
        },
    }
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    args = SimpleNamespace(
        stage=stage_path,
        model=None,
        model_manifest=None,
        topology=topology,
        profiles=None,
        port=30000,
        output_dir=tmp_path / "out",
        job_timeout=10,
    )
    return matrix, stage, M.CAP.load_policy(HERE / "experiment-policy.json"), args


@pytest.mark.parametrize("family", ["qwen", "deepseek-v4-flash", "glm-5.3-flash"])
@pytest.mark.parametrize("variant", ["cpu", "mpu"])
def test_six_stage_workloads_and_same_capacity_baselines(tmp_path, family, variant):
    matrix, stage, policy, args = fixture(tmp_path, family, variant)
    plan = M.make_plan(matrix, stage, policy, args)
    assert plan["status"] == "planned"
    assert plan["decision_inputs_pending"] == [
        "p99_ttft_ms",
        "p99_tpot_ms",
        "total_configuration_costs",
    ]
    assert [j["id"] for j in plan["jobs"]] == (
        ["baseline", "capacity"] if variant == "mpu" else ["baseline"]
    )
    for job in plan["jobs"]:
        assert {g["eager"] for g in job["correctness_groups"]} == {True, False}
        assert all(g["lifecycle_required"] for g in job["correctness_groups"])
        assert (
            job["gate_command"][job["gate_command"].index("--through") + 1]
            == "correctness"
        )
        for group in job["performance_groups"]:
            assert len(group["cases"]) == 25
            command = group["cases"][0]["command"]
            assert command[command.index("--warmup-requests") + 1] == str(
                matrix["families"][family]["warmup_requests"]
            )
            assert {case["repeat"] for case in group["cases"]} == set(range(5))
            server = group["server_command"]
            assert server[server.index("--dp-size") + 1] == "1"
            if family != "qwen":
                assert server[server.index("--attn-dp-size") + 1] == "4"
                assert server[server.index("--ep-size") + 1] == "4"
                assert server[server.index("--moe-dense-tp-size") + 1] == "1"
        ratios = {(g["variant"], g["ratio"]) for g in job["performance_groups"]}
        if job["id"] == "baseline":
            assert ratios == (
                {("cpu", 2), ("mpu", 2)} if variant == "mpu" else {("cpu", 2)}
            )
        else:
            assert ratios == {
                ("cpu", 8),
                ("mpu", 1),
                ("mpu", 2),
                ("mpu", 4),
                ("mpu", 8),
            }


def test_qwen_four_storage_cards_are_a_separate_topology(tmp_path):
    values = fixture(tmp_path, topology="qwen-1plus4")
    plan = M.make_plan(*values)
    assert plan["topology"]["storage_device_ids"] == [1, 2, 3, 4]
    assert plan["topology"]["compute_gpus"] == 1
    for job in plan["jobs"]:
        assert job["performance_settings"]["mpu_device_ids"] == [1, 2, 3, 4]
    for family, variant in (("qwen", "cpu"), ("glm-5.3-flash", "mpu")):
        with pytest.raises(ValueError, match="stage 02"):
            M.make_plan(*fixture(tmp_path, family, variant, "qwen-1plus4"))


@pytest.mark.parametrize(
    "change",
    [
        "low_repeats",
        "low_requests",
        "low_warmup",
        "bad_topology",
        "capacity_overflow",
        "mismatched_baseline",
        "typo",
    ],
)
def test_matrix_rejects_underpowered_or_confounded_inputs(tmp_path, change):
    matrix, stage, policy, args = fixture(tmp_path)
    if change == "low_repeats":
        matrix["repeats"] = 3
    elif change == "low_requests":
        matrix["families"]["qwen"]["num_prompts"] = 32
    elif change == "low_warmup":
        matrix["families"]["qwen"]["warmup_requests"] = 1
    elif change == "bad_topology":
        stage["ep_size"] = 4
    elif change == "capacity_overflow":
        matrix["profiles"]["capacity"]["cpu_capacity_ratio"] = 2
    elif change == "mismatched_baseline":
        matrix["profiles"]["baseline"]["capacity_ratios"] = [1]
    else:
        matrix["repeatz"] = 5
    with pytest.raises(ValueError):
        M.make_plan(matrix, stage, policy, args)


def test_simulator_cost_does_not_scale_with_allocated_ratio(tmp_path):
    matrix, stage, policy, args = fixture(tmp_path)
    policy["costs"] = [{"variant": "mpu", "ratio": n, "hourly_cost": n} for n in (1, 2)]
    with pytest.raises(ValueError, match="one total cost"):
        M.make_plan(matrix, stage, policy, args)
    policy["cost_basis"] = "hypothetical_mpu"
    policy["topology_scope"] = M.DECISION.scope(stage, [1])
    assert M.make_plan(matrix, stage, policy, args)["status"] == "planned"


def test_actual_branch_recipe_plans_without_loading_gpu(tmp_path):
    matrix, _, policy, args = fixture(tmp_path)
    args.stage = HERE / "stage.json"
    stage = M.STAGE.load_stage(args.stage)
    assert M.make_plan(matrix, stage, policy, args)["jobs"]


def test_missing_or_portable_only_gate_cannot_allow_performance(tmp_path):
    _, stage, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="Correctness acceptance"):
        M.verify_gate(
            {"status": "passed", "stage": stage, "requested_through": "portable"},
            stage,
            {},
        )
    with pytest.raises(ValueError, match="Missing required"):
        M.verify_gate(
            {
                "status": "passed",
                "stage": stage,
                "requested_through": "correctness",
                "steps": [],
            },
            stage,
            {},
        )


def test_failed_gate_stops_performance_and_later_profiles(tmp_path):
    values = fixture(tmp_path)
    plan, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    calls = []

    def fail(command, env, log, timeout):
        calls.append(command)
        return {"returncode": 1}

    with (
        patch.object(M.sys, "platform", "linux"),
        patch.object(M.CHECK, "identity", return_value={"synthetic": True}),
    ):
        assert not M.execute(plan, args, runner=fail)
    assert len(calls) == 1
    assert plan["status"] == "failed"
    assert plan["jobs"][1]["status"] == "not_run_after_failure"


def test_non_linux_execution_is_blocked_not_passed(tmp_path):
    values = fixture(tmp_path)
    plan, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    with patch.object(M.sys, "platform", "win32"):
        assert not M.execute(
            plan, args, runner=lambda *args: pytest.fail("must not launch")
        )
    assert plan["status"] == "blocked"


def test_configuration_drift_is_rejected_before_metric_analysis(tmp_path):
    values = fixture(tmp_path)
    job = M.make_plan(*values)["jobs"][0]
    report = {
        "status": "success",
        "stage": values[1],
        "kind": "benchmark",
        "settings": copy.deepcopy(job["performance_settings"]),
    }
    report["settings"]["max_total_tokens"] *= 2
    with pytest.raises(ValueError, match="max_total_tokens"):
        M.verify_performance(report, values[1], job)


def test_plateau_requires_two_adjacent_edges_and_tracks_actual_bytes():
    rows = [
        {
            "variant": "mpu",
            "input_len": 8192,
            "concurrency": 4,
            "ratio": ratio,
            "previous_ratio": ratio / 2,
            "decode_plateau_evidence": state,
            "backing_to_compute_kv_byte_ratios_by_rank": [ratio + 0.01],
            "slo": {"status": "not_specified"},
        }
        for ratio, state in (
            (1, None),
            (2, "candidate_below_threshold"),
            (4, "candidate_below_threshold"),
        )
    ]
    report = M.decisions({"rows": rows, "best_observed_cost_points": []}, 2)
    candidate = report["plateau_candidates"][0]
    assert candidate["candidate_requested_ratio"] == 1
    assert candidate["tested_through_requested_ratio"] == 4
    assert candidate["actual_byte_ratio_by_point"][0]["per_rank"] == [1.01]
    assert (
        not candidate["latency_eligible"] and not candidate["economic_optimum_proven"]
    )
    rows[2]["decode_plateau_evidence"] = "inconclusive"
    assert not M.decisions({"rows": rows, "best_observed_cost_points": []}, 2)[
        "plateau_candidates"
    ]


def test_later_rebound_invalidates_an_early_plateau():
    rows = [
        {
            "variant": "mpu",
            "input_len": 8192,
            "concurrency": 4,
            "ratio": ratio,
            "previous_ratio": ratio / 2,
            "decode_plateau_evidence": state,
        }
        for ratio, state in (
            (2, "candidate_below_threshold"),
            (4, "candidate_below_threshold"),
            (8, "gain_above_threshold"),
        )
    ]
    assert not M.decisions({"rows": rows, "best_observed_cost_points": []}, 2)[
        "plateau_candidates"
    ]


def test_gate_cannot_skip_graph_even_with_successful_lifecycle_reports(tmp_path):
    values = fixture(tmp_path)
    _, stage, _, _ = values
    job = M.make_plan(*values)["jobs"][0]
    groups = copy.deepcopy(job["correctness_groups"])
    for group in groups:
        group["status"] = "success"
        group["lifecycle_results"] = [
            {"case": case, "status": "passed"}
            for case in ("abort_prefill", "abort_decode", "retract_decode")
        ]
    correctness = {
        "status": "success",
        "stage": stage,
        "groups": groups,
        "settings": job["performance_settings"],
    }
    report_path = Path(job["gate_report"]).parent / "correctness/manifest.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(correctness), encoding="utf-8")
    acceptance = {
        "status": "passed",
        "stage": stage,
        "requested_through": "correctness",
        "steps": [
            {"id": step, "status": "passed"}
            for step in (
                "portable",
                "compile",
                "jit",
                "imports",
                "device",
                "kernels",
                "model_identity",
                "correctness",
            )
        ],
    }
    M.verify_gate(acceptance, stage, job)
    correctness["groups"] = [g for g in groups if g["eager"]]
    report_path.write_text(json.dumps(correctness), encoding="utf-8")
    with pytest.raises(ValueError, match="eager/graph"):
        M.verify_gate(acceptance, stage, job)


def test_benchmark_graph_fallback_blocks_all_requests(tmp_path):
    matrix, stage, _, args = fixture(tmp_path, variant="cpu")
    _, parsed = M.STAGE.parse_args(
        ["--stage", str(args.stage), "--output-dir", str(args.output_dir)]
    )
    parsed.output_dir.mkdir()
    manifest = {"groups": M.RUN.make_plan(stage, parsed)}
    with (
        patch.object(M.RUN, "require_cuda"),
        patch.object(M.RUN.COMMON, "start_server", return_value=object()),
        patch.object(M.RUN.COMMON, "wait_ready"),
        patch.object(M.RUN.COMMON, "stop_server"),
        patch.object(
            M.RUN.COMMON, "request_json", return_value={"disable_cuda_graph": True}
        ),
        patch.object(M.RUN.subprocess, "run") as benchmark,
    ):
        assert not M.RUN.execute(
            stage, parsed, manifest, args.output_dir / "manifest.json"
        )
        benchmark.assert_not_called()
    assert all(g["status"] == "failed" for g in manifest["groups"])


def test_performance_verifier_accepts_complete_plan_but_rejects_slot_leaks(tmp_path):
    values = fixture(tmp_path, variant="cpu")
    job = M.make_plan(*values)["jobs"][0]
    groups = copy.deepcopy(job["performance_groups"])
    slots = job["performance_settings"]["max_total_tokens"]
    capacity = {
        "tp_rank": 0,
        "tp_size": 1,
        "backing_store": "cpu",
        "physical_compute_slots": slots,
        "compute_kv_bytes": slots * 16,
        "backing_slots": slots * 2,
        "backing_kv_bytes": slots * 32,
        "requested_capacity_ratio": 2,
        "effective_capacity_ratio": 2,
        "used_slots": 0,
        "reserved_slots": 0,
    }
    for group in groups:
        group.update(
            status="success",
            decode_cuda_graph_enabled=True,
            capacity=[copy.deepcopy(capacity)],
        )
        for case in group["cases"]:
            case.update(status="success", idle_capacity=[copy.deepcopy(capacity)])
    report = {
        "kind": "benchmark",
        "stage": values[1],
        "status": "success",
        "groups": groups,
        "settings": copy.deepcopy(job["performance_settings"]),
    }
    M.verify_performance(report, values[1], job)
    groups[-1]["cases"][-1]["idle_capacity"][0]["used_slots"] = 1
    with pytest.raises(ValueError, match="leaked"):
        M.verify_performance(report, values[1], job)


def test_partial_p99_policy_is_not_formal_latency_acceptance(tmp_path):
    matrix, stage, policy, args = fixture(tmp_path)
    policy["slo"] = {"p99_tpot_ms": 50}
    with pytest.raises(ValueError, match="both P99"):
        M.make_plan(matrix, stage, policy, args)


def test_identity_failure_records_blocked_before_any_process(tmp_path):
    values = fixture(tmp_path)
    plan, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    with (
        patch.object(M.sys, "platform", "linux"),
        patch.object(
            M.CHECK, "identity", side_effect=ValueError("model hash mismatch")
        ),
    ):
        assert not M.execute(
            plan, args, runner=lambda *a: pytest.fail("must not launch")
        )
    saved = M.read_json(args.output_dir / "matrix.json")
    assert saved["status"] == "blocked" and "model hash" in saved["reason"]


def test_resume_reuses_sealed_gate_and_retries_failed_performance(tmp_path):
    values = fixture(tmp_path, variant="cpu")
    plan, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    job = plan["jobs"][0]
    calls = []

    def runner(command, env, log, timeout):
        calls.append(command)
        gates = "--through" in command
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        M.RUN.COMMON.write_manifest(
            output / ("acceptance.json" if gates else "manifest.json"), {"groups": []}
        )
        return {"returncode": 0 if gates or "--resume" in command else 1}

    with (
        patch.object(M.sys, "platform", "linux"),
        patch.object(M.CHECK, "identity", return_value={"synthetic": True}),
        patch.object(M, "verify_gate"),
        patch.object(M, "verify_performance"),
        patch.object(
            M.RUN,
            "summarize",
            return_value={"rows": [], "best_observed_cost_points": []},
        ),
    ):
        assert not M.execute(plan, args, runner)
        gate_bytes = Path(job["gate_report"]).read_bytes()
        assert M.execute(plan, args, runner)
        assert len(calls) == 3 and "--resume" in calls[-1]
        assert Path(job["gate_report"]).read_bytes() == gate_bytes
        assert job["gates_process"]["reused_sealed_gate"]
        Path(job["gate_report"]).write_text("changed", encoding="utf-8")
        assert not M.execute(plan, args, runner)
        assert len(calls) == 3  # A changed gate cannot trigger any benchmark.


def test_failed_gate_uses_fresh_directory_on_resume(tmp_path):
    values = fixture(tmp_path, variant="cpu")
    plan, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    attempts = []

    def fail(command, env, log, timeout):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        attempts.append(output)
        (output / "partial.json").write_text("{}", encoding="utf-8")
        return {"returncode": 1}

    with (
        patch.object(M.sys, "platform", "linux"),
        patch.object(M.CHECK, "identity", return_value={"synthetic": True}),
    ):
        assert not M.execute(plan, args, fail)
        assert not M.execute(plan, args, fail)
    assert attempts[0].name == "gates"
    assert attempts[1].name == "gates-attempt-0002"
    assert (attempts[0] / "partial.json").exists()


@pytest.mark.parametrize(
    "field", ["gate_command", "performance_command", "performance_settings", "topology"]
)
def test_resume_rejects_plan_mutation(tmp_path, field):
    values = fixture(tmp_path)
    expected, args = M.make_plan(*values), values[-1]
    args.output_dir.mkdir()
    M.RUN.COMMON.write_manifest(args.output_dir / "policy.json", values[2])
    saved = copy.deepcopy(expected)
    M.validate_resume(saved, expected, args)
    if field == "topology":
        saved[field]["storage_device_ids"] = [1, 2, 3, 4]
    elif field == "performance_settings":
        saved["jobs"][0][field]["seed"] += 1
    else:
        saved["jobs"][0][field].append("--extra")
    with pytest.raises(ValueError, match="changed"):
        M.validate_resume(saved, expected, args)


def test_main_plans_and_resumes_without_cuda(tmp_path):
    output = tmp_path / "matrix"
    command = ["--output-dir", str(output)]
    assert M.main(command) == 0
    assert M.main(command + ["--resume"]) == 0
    assert M.read_json(output / "decision-inputs.json")["slo_ms"]["p99_ttft_ms"] is None
