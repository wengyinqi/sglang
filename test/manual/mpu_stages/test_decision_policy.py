"""Policy fixtures are synthetic; they contain no real hardware prices or SLOs."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load("decision_policy", "benchmark/mpu_stages/decision_policy.py")
M = load("matrix_fixture", "test/manual/mpu_stages/test_experiment_matrix.py")


def plan(tmp_path):
    return M.M.make_plan(*M.fixture(tmp_path))


def test_pending_values_remain_unknown_not_zero_prices_or_passes(tmp_path):
    matrix = plan(tmp_path)
    inputs = D.template(matrix)
    policy = D.compile_policy(inputs, matrix)
    assert policy["slo"] == {} and policy["costs"] == []
    assert inputs["cpu_hourly_total"] is None and inputs["mpu_hourly_total"] is None
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert D.load("capacity_analysis").load_policy(path)["costs"] == []


@pytest.mark.parametrize(
    "change", ["partial_slo", "zero", "nan", "source", "topology", "repeat", "typo"]
)
def test_incomplete_or_mismatched_decision_inputs_are_rejected(tmp_path, change):
    matrix = plan(tmp_path)
    inputs = D.template(matrix)
    if change == "partial_slo":
        inputs["slo_ms"]["p99_ttft_ms"] = 100
    elif change == "zero":
        inputs["cpu_hourly_total"] = 0
    elif change == "nan":
        inputs["mpu_hourly_total"] = float("nan")
    elif change == "source":
        inputs["cpu_hourly_total"] = 1
    elif change == "topology":
        inputs["topology_scope"]["storage_device_ids"] = [1, 2, 3, 4]
    elif change == "repeat":
        inputs["minimum_repeats"] = 3
    else:
        inputs["hour_price"] = 1
    with pytest.raises(ValueError):
        D.compile_policy(inputs, matrix)


def test_configuration_totals_are_constant_across_capacity_allocations(tmp_path):
    matrix = plan(tmp_path)
    inputs = D.template(matrix)
    inputs.update(
        currency="synthetic_units",
        source="test fixture only",
        cpu_hourly_total=1,
        mpu_hourly_total=2,
    )
    inputs["slo_ms"] = {"p99_ttft_ms": 100, "p99_tpot_ms": 50}
    policy = D.compile_policy(inputs, matrix)
    assert {p["hourly_cost"] for p in policy["costs"] if p["variant"] == "mpu"} == {2}
    assert {p["ratio"] for p in policy["costs"] if p["variant"] == "mpu"} == {
        1,
        2,
        4,
        8,
    }
    with pytest.raises(ValueError, match="physical topology"):
        D.verify_scope(policy, matrix["stage"], [1, 2, 3, 4])


def test_review_of_missing_measurements_stays_incomplete_and_does_not_modify_plan(
    tmp_path,
):
    matrix = plan(tmp_path)
    plan_path, inputs_path, output = (
        tmp_path / "matrix.json",
        tmp_path / "inputs.json",
        tmp_path / "review.json",
    )
    plan_path.write_text(json.dumps(matrix), encoding="utf-8")
    inputs_path.write_text(json.dumps(D.template(matrix)), encoding="utf-8")
    before = plan_path.read_bytes()
    D.main(
        [
            "review",
            "--matrix-report",
            str(plan_path),
            "--inputs",
            str(inputs_path),
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert not result["measurements_rerun"]
    assert all(p["status"] == "missing_measurements" for p in result["profiles"])
    assert before == plan_path.read_bytes()


def test_standalone_capacity_policy_rejects_wrong_storage_topology(tmp_path):
    matrix = plan(tmp_path)
    policy = D.compile_policy(D.template(matrix), matrix)
    measured = {"stage": matrix["stage"], "settings": {"mpu_device_ids": [1]}}
    cap = D.load("capacity_analysis")
    cap.verify_manifest_scope(policy, measured)
    measured["settings"]["mpu_device_ids"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="physical topology"):
        cap.verify_manifest_scope(policy, measured)


def test_review_rejects_success_label_with_wrong_workload(tmp_path):
    import copy

    matrix = plan(tmp_path)
    job = matrix["jobs"][0]
    measured = {
        "kind": "benchmark",
        "status": "success",
        "stage": matrix["stage"],
        "settings": copy.deepcopy(job["performance_settings"]),
    }
    measured["settings"]["seed"] += 1
    path = Path(job["performance_report"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(measured), encoding="utf-8")
    report, inputs, output = (
        tmp_path / n for n in ("matrix.json", "inputs.json", "review.json")
    )
    report.write_text(json.dumps(matrix), encoding="utf-8")
    inputs.write_text(json.dumps(D.template(matrix)), encoding="utf-8")
    D.main(
        [
            "review",
            "--matrix-report",
            str(report),
            "--inputs",
            str(inputs),
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["profiles"][0]["status"] == "invalid_measurements"
    assert "seed" in result["profiles"][0]["reason"]
