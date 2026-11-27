"""Portable source017 report-binding fixtures; no real model/GPU claims."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module(name):
    spec = importlib.util.spec_from_file_location(
        "portable_" + name, ROOT / ("benchmark/xm_sparse/" + name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BIND, REPORT = module("report_control_binding"), module("report")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_requested_effective_and_Quest_R_are_distinct():
    configured, effective = BIND.request_capacity(
        ["--max-running-requests", "12"],
        {"internal_states": [{"effective_max_running_requests_per_dp": 12}]},
    )
    geometry = BIND.quest_geometry(
        {
            "gpu_memory_ledger": {
                "named_compute_shapes": {"fixed.request_map": {"shape": [13, 33286]}}
            }
        }
    )
    assert configured == effective == 12
    assert geometry["quest_rows_R"] == 13
    assert geometry["quest_pages_P"] == 521


@pytest.mark.parametrize("value", [None, True, 0, 13, "12"])
def test_missing_or_invalid_observed_R_is_not_guessed_from_C(value):
    with pytest.raises(ValueError):
        BIND.request_capacity(
            ["--max-running-requests", "12"],
            {"internal_states": [{"effective_max_running_requests_per_dp": value}]},
        )


@pytest.mark.parametrize("shape", [None, [], [12], [True, 128], [12, 0]])
def test_missing_or_invalid_Quest_geometry_rejected(shape):
    with pytest.raises(ValueError):
        BIND.quest_geometry(
            {
                "gpu_memory_ledger": {
                    "named_compute_shapes": {"fixed.request_map": {"shape": shape}}
                }
            }
        )


def test_capacity_filter_retains_uncovered_C12_R12_B11_without_maxB8_claim():
    rows = [
        {
            "arm": arm,
            "input_len": 32768,
            "client_concurrency": c,
            "actual_batch": b,
            "graph_enabled": True,
            "eligible": True,
        }
        for arm in ("baseline", "fused")
        for c, b in ((4, 4), (12, 11))
    ]
    original = {
        "phase": "capacity",
        "status": "analyzed",
        "rows": rows,
        "samples": [],
        "observed_stable_capacity_points": [],
        "paired_repeat_rows": [],
        "same_actual_batch_comparisons": [],
    }
    before = copy.deepcopy(original)
    accepted = [
        {
            "arm": arm,
            "input_len": 32768,
            "client_concurrency": 4,
            "actual_batch_size": 4,
            "graph_enabled": True,
            "request_capacity_R": 4,
            "quest_rows_R": 5,
            "quest_context_columns": 33286,
            "quest_pages_P": 521,
        }
        for arm in ("baseline", "fused")
    ]
    pending = [
        {
            "arm": arm,
            "input_len": 32768,
            "client_concurrency": 12,
            "actual_batch_size": 11,
            "graph_enabled": True,
            "request_capacity_R": 12,
            "quest_rows_R": 13,
            "reason": "ExactB/R control pending",
        }
        for arm in ("baseline", "fused")
    ]
    shown = BIND.filter_capacity(
        original, {"accepted": accepted, "pending": pending, "request_layouts": []}
    )
    assert original == before
    assert len(shown["rows"]) == 2 and all(
        row["client_concurrency"] == 4 for row in shown["rows"]
    )
    assert shown["unqualified_capacity_points"] == pending
    assert "maximum_batch" not in shown


def test_HTTP_CPU_metrics_alone_cannot_supply_model_claims():
    audit = {
        "model_result_claims_allowed": False,
        "analysis_model_proof_bindings": [
            {
                "status": "passed",
                "client_paired_effects": {"status": "passed", "metrics": [1]},
            }
        ],
    }
    assert REPORT.client_effect_bindings(audit) == []


def test_cross_zero_HTTP_interval_stays_inconclusive():
    row = {
        "metric": "http_mean_tbt_ms",
        "baseline": "baseline",
        "changed": "fused",
        "input_len": 16384,
        "client_concurrency": 1,
        "graph_enabled": True,
        "repeats": 3,
        "df": 2,
        "paired_repeat_reduction_percent": 1,
        "confidence_interval_95_percent": [-1, 3],
    }
    assert "区间跨0" in REPORT.client_effect_conclusion(row)


def test_ratio4_precursor_is_never_normalized_to8(tmp_path, monkeypatch):
    proof_path = write(
        tmp_path / "campaign.json",
        {"settings": {"input_lens": [32768] * 4, "capacity_ratio": 4}},
    )
    record = {
        "kind": BIND.PRECURSOR_KIND,
        "status": "passed_CPU_raw_replay",
        "original_campaign": REPORT.identity(proof_path),
        "precursor_capacity_ratio": 8,
        "settings": {},
        "controller": {"path": "fixture-only"},
    }
    path = write(tmp_path / "binding.json", record)
    monkeypatch.setattr(BIND, "derive_precursor", lambda *args: record)
    with pytest.raises(ValueError, match="normalized"):
        BIND.verify_precursor(
            proof_path,
            BIND.read(proof_path),
            [(path, record)],
            REPORT.control_binding_api(),
        )


@pytest.mark.parametrize("status", ["running", "failed", "pending"])
def test_external_baseline_pending_status_cannot_bind_capacity(status):
    with pytest.raises(ValueError):
        BIND.verify_external_baseline(
            "unused",
            {"kind": BIND.BASELINE_KIND, "status": status},
            11,
            12,
            13,
            REPORT.control_binding_api(),
        )


def test_old_B11_R11_control_cannot_cover_C12_R12_B11(tmp_path, monkeypatch):
    controller = write(tmp_path / "controller.py", {})
    settings = {"batch_sizes": [11]}
    plan = {
        "kind": BIND.BASELINE_KIND,
        "status": "ready",
        "settings": settings,
        "controller": REPORT.identity(controller),
        "bound_inputs": [],
        "prerequisites": {},
    }
    plan_path = write(tmp_path / "plan.json", plan)
    record = {
        "kind": BIND.BASELINE_KIND,
        "status": "success",
        "settings": settings,
        "plan": REPORT.identity(plan_path),
        "prerequisites": {},
        "large_hbm_model_correctness_proven": False,
        "capacity_proof": False,
        "performance_result": False,
        "cases": [{"batch": 11, "status": "success", "request_slot_capacity_R": 11}],
    }
    monkeypatch.setattr(
        BIND,
        "controller",
        lambda *args: SimpleNamespace(
            bound_inputs=lambda _: [], all_gates=lambda _: (None, {}, {})
        ),
    )
    with pytest.raises(ValueError, match="B11/R11"):
        BIND.verify_external_baseline(
            "unused", record, 11, 12, 13, REPORT.control_binding_api()
        )


@pytest.fixture
def dedicated_custody(tmp_path, monkeypatch):
    controller = write(tmp_path / "controller.py", {"synthetic": True})
    runner = write(tmp_path / "runner.py", {"synthetic": True})
    plan = write(tmp_path / "plan.json", {"status": "ready"})
    native = write(tmp_path / "native.json", {"synthetic": True})
    control = {
        "source_marker": {"source_commit": BIND.SOURCE_COMMIT},
        "core_source_fingerprint": {"sha256": BIND.CORE_SHA256},
        "plan": REPORT.identity(plan),
    }
    result = write(tmp_path / "control.json", control)
    contract = {
        "controller": REPORT.identity(controller),
        "runner": REPORT.identity(runner),
        "gpu_execution_allowed": True,
    }
    contract_path = write(tmp_path / "contract.json", contract)
    job = {
        "kind": "external_baseline017_dedicated_runner_job",
        "mode": "execute",
        "gpu_requested": True,
        "status": "finished",
        "returncode": 0,
        "contract": REPORT.identity(contract_path),
        "command": [
            "docker",
            BIND.RUNTIME_IMAGE,
            "/results/tools/large_cpu_baseline_control017.py",
            "execute",
            "--plan",
            str(plan),
            "--reviewed-plan-sha256",
            REPORT.identity(plan)["sha256"],
        ],
    }
    job_path = write(tmp_path / "job.json", job)
    monkeypatch.setattr(BIND, "CONTROLLER_SHA256", contract["controller"]["sha256"])
    monkeypatch.setattr(BIND, "RUNNER_SHA256", contract["runner"]["sha256"])
    custody = {
        "kind": BIND.BASELINE_CUSTODY_KIND,
        "status": "sealed_actual_external_baseline_evidence",
        "source_marker": control["source_marker"],
        "core_source_fingerprint": control["core_source_fingerprint"],
        "evidence_root": "/results",
        "execution_receipt": REPORT.identity(result),
        "reviewed_plan": REPORT.identity(plan),
        "runner_contract": {**contract, "manifest": REPORT.identity(contract_path)},
        "native_reports": [
            {
                "batch": 8,
                "arm": "baseline",
                "native": REPORT.identity(native),
                "native_status": "success",
                "request_admission_limit": 8,
            }
        ],
        "job": job,
        "job_file": REPORT.identity(job_path),
        "actual_execution_argv": job["command"][2:],
        "execution_status": "success",
        "model_correctness_result": False,
        "large_hbm_model_correctness_proven": False,
        "capacity_proof": False,
        "performance_result": False,
        "files": [
            {**REPORT.identity(path), "path": path.name}
            for path in (
                controller,
                runner,
                plan,
                native,
                result,
                contract_path,
                job_path,
            )
        ],
    }
    seal = write(tmp_path / "seal.json", custody)
    return result, control, seal, custody


def test_dedicated_custody_binds_existing_absolute_paths_before_results_mapping(
    dedicated_custody,
):
    path, control, seal, custody = dedicated_custody
    bound = BIND.external_baseline_custody(
        path, control, [(seal, custody)], REPORT.control_binding_api()
    )
    assert bound["execution_receipt"] == REPORT.identity(path)
    assert len(bound["verified_inventory"]) == 7


@pytest.mark.parametrize(
    "change",
    [
        "old_campaign_kind",
        "failed_job",
        "unbound_native",
        "path_traversal",
        "truncated_argv",
        "plan_sha",
        "nested_native_status",
        "proof_relabel",
        "execution_status",
        "unbound_contract",
    ],
)
def test_incomplete_or_untrusted_external_execution_custody_rejected(
    dedicated_custody, change
):
    path, control, seal, custody = dedicated_custody
    if change == "old_campaign_kind":
        custody["kind"] = "sealed_actual_phase_evidence"
    elif change == "failed_job":
        custody["job"]["returncode"] = 1
    elif change == "unbound_native":
        custody["files"] = [
            item for item in custody["files"] if item["path"] != "native.json"
        ]
    elif change == "path_traversal":
        custody["files"][0]["path"] = "../controller.py"
    elif change == "truncated_argv":
        custody["actual_execution_argv"] = ["execute", "--reviewed-plan-sha256"]
    elif change == "nested_native_status":
        custody["native_reports"][0]["native_status"] = "pending"
    elif change == "proof_relabel":
        custody["capacity_proof"] = True
    elif change == "execution_status":
        custody["execution_status"] = "failed"
    elif change == "unbound_contract":
        custody["job"]["contract"]["sha256"] = "0" * 64
    else:
        custody["actual_execution_argv"][-1] = "0" * 64
    with pytest.raises(ValueError):
        BIND.external_baseline_custody(
            path, control, [(seal, custody)], REPORT.control_binding_api()
        )


def http_binding(input_len=16384):
    return {
        "source_fingerprint": {"sha256": BIND.CORE_SHA256},
        "native_source_commits": [BIND.SOURCE_COMMIT],
        "workload_identity": {"sha256": "synthetic-workload"},
        "metrics": [
            {
                "arm": "baseline",
                "input_len": input_len,
                "client_concurrency": 1,
                "graph_enabled": True,
            }
        ],
        "effects": [
            {
                "baseline": "baseline",
                "changed": "fused",
                "metric": "http_mean_tbt_ms",
                "input_len": input_len,
                "client_concurrency": 1,
                "graph_enabled": True,
            }
        ],
    }


def test_disjoint_http_input_cohorts_same_source_can_share_report():
    bindings = [http_binding(), http_binding(32768)]
    assert REPORT.validate_client_effect_bindings(bindings) == bindings


@pytest.mark.parametrize(
    "change", ["duplicate", "source", "revision", "workload", "missing_identity"]
)
def test_http_plot_cohorts_never_overwrite_or_mix_sources(change):
    first, second = http_binding(), http_binding(32768)
    if change == "duplicate":
        second = http_binding()
    elif change == "source":
        second["source_fingerprint"] = {"sha256": "another-source"}
    elif change == "revision":
        second["native_source_commits"] = ["another-revision"]
    elif change == "workload":
        second["workload_identity"] = {"sha256": "another-workload"}
    else:
        second.pop("source_fingerprint")
    with pytest.raises(ValueError):
        REPORT.validate_client_effect_bindings([first, second])
