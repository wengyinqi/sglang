"""Portable rejection checks for actual-correctness-only profile adaptation."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "xm_adapter_test_" + name, ROOT / "benchmark/xm_sparse" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = load("profile")
TELEMETRY = load("telemetry")


@pytest.fixture
def actual_proof(tmp_path, monkeypatch):
    workload = tmp_path / "workload.json"
    workload.write_text("{}")
    stage = PROFILE.CAMPAIGN.recipe("fused")
    stage_path = tmp_path / "fused/stage.json"
    PROFILE.CAMPAIGN.write(stage_path, stage)
    native_path = stage_path.parent / "evidence/manifest.json"
    groups = [
        {
            "id": variant + "-" + mode,
            "variant": variant,
            "eager": mode == "eager",
            "status": "success",
            "lifecycle_required": True,
            "lifecycle_results": [
                {"case": case, "status": "passed"}
                for case in ("abort_prefill", "abort_decode", "retract_decode")
            ],
        }
        for mode in ("eager", "graph")
        for variant in ("cpu", "hbm")
    ]
    native = {
        "status": "success",
        "stage": stage,
        "groups": groups,
        "runtime": {"git_commit": "a" * 40, "git_status": ""},
        "settings": {
            "model": "/models/qwen",
            "workload": str(workload),
            "num_prompts": 16,
            "capacity_ratios": [8],
            "disable_cuda_graph": False,
        },
    }
    PROFILE.CAMPAIGN.write(native_path, native)
    fingerprint = {"files": 1, "sha256": "source-test-fixture"}
    monkeypatch.setattr(PROFILE.CAMPAIGN, "source_fingerprint", lambda: fingerprint)
    proof = tmp_path / "campaign.json"
    accepted = {
        "status": "success",
        "phase": "correctness",
        "workload": str(workload),
        "settings": {
            "model": "/models/qwen",
            "input_lens": [16384],
            "disable_cuda_graph": False,
        },
        "source_fingerprint": fingerprint,
        "workload_identity": PROFILE.CAMPAIGN.identity(workload),
        "cases": [
            {
                "arm": "fused",
                "status": "success",
                "returncode": 0,
                "stage_path": str(stage_path),
                "command": ["fixture"],
            }
        ],
    }
    PROFILE.CAMPAIGN.write(proof, accepted)
    return proof, native_path, accepted, native


def test_receipt_asserts_only_actually_executed_correctness(actual_proof):
    proof, _, _, _ = actual_proof
    receipt, _ = PROFILE.receipt(proof, "fused", 16384)
    assert receipt["status"] == "passed_correctness_only"
    assert [step["id"] for step in receipt["steps"]] == ["correctness"]
    assert receipt["steps"][0]["results"][0]["status"] == "passed"
    assert "compile" not in receipt["evidence_scope"]


@pytest.mark.parametrize(
    "mutation", ["unfinished", "source", "missing_hbm", "missing_lifecycle", "exit"]
)
def test_rejects_missing_or_changed_real_evidence(actual_proof, mutation):
    proof, native_path, accepted, native = actual_proof
    accepted, native = copy.deepcopy(accepted), copy.deepcopy(native)
    if mutation == "unfinished":
        accepted["status"] = "running"
    elif mutation == "source":
        accepted["source_fingerprint"]["sha256"] = "changed"
    elif mutation == "missing_hbm":
        native["groups"] = [g for g in native["groups"] if g["variant"] == "cpu"]
    elif mutation == "missing_lifecycle":
        native["groups"][0]["lifecycle_results"].pop()
    elif mutation == "exit":
        accepted["cases"][0]["returncode"] = 2
    PROFILE.CAMPAIGN.write(proof, accepted)
    PROFILE.CAMPAIGN.write(native_path, native)
    with pytest.raises(ValueError):
        PROFILE.receipt(proof, "fused", 16384)


def test_native_args_preserves_storage_and_memory_settings(actual_proof):
    _, native_path, _, native = actual_proof
    native["settings"].update(
        max_total_tokens=65536,
        cpu_capacity_ratio=8,
        chunked_prefill_size=4096,
        prefill_max_requests=1,
        mem_fraction_static=0.9,
    )
    args = PROFILE.native_args(
        native,
        native_path,
        native_path.parent,
        input_len=16384,
        batch=4,
        output_len=512,
    )
    assert args[args.index("--max-total-tokens") + 1] == "65536"
    assert args[args.index("--cpu-capacity-ratio") + 1] == "8"
    assert args[args.index("--concurrencies") + 1] == "4"


@pytest.mark.parametrize("stderr", ["driver unavailable", ""])
def test_nvml_failure_is_explicit_and_observation_is_not_execution_phase(
    monkeypatch, stderr
):
    monkeypatch.setattr(
        TELEMETRY.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=stderr),
    )
    monkeypatch.setattr(
        TELEMETRY, "scheduler_observation", lambda _: {"status": "not_observed"}
    )
    item = TELEMETRY.sample(30000)
    assert item["query_status"] == "failed"
    assert item["gpus"] == []
    assert "phase" not in item


def test_nvml_missing_binary_and_malformed_csv_fail_closed(monkeypatch):
    monkeypatch.setattr(TELEMETRY, "scheduler_observation", lambda _: {})

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(TELEMETRY.subprocess, "run", missing)
    assert TELEMETRY.sample(30000)["query_status"] == "failed"
    monkeypatch.setattr(
        TELEMETRY.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="0, short", stderr=""),
    )
    assert TELEMETRY.sample(30000)["query_status"] == "failed"


def test_diagnostics_run_new_scope_before_attention_numbers(tmp_path, monkeypatch):
    trace = tmp_path / "capture/forward-trace/forward-tp0-pp0-dpNone-pid1.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("fixture")
    calls = []

    def scope(*args, **kwargs):
        calls.append(("xm_scope", args, kwargs))
        return {"status": "passed", "actual_decode_scope_verified": True}

    def analysis(database, verified):
        assert calls[0][0] == "xm_scope"
        assert verified["actual_decode_scope_verified"] is True
        calls.append(("attention", database))
        return {"status": "analyzed"}

    monkeypatch.setattr(PROFILE.SCOPE, "verify", scope)
    monkeypatch.setattr(PROFILE.ANALYZE, "attention_profile", analysis)
    assert (
        PROFILE.diagnose(tmp_path, tmp_path / "proof.json", tmp_path / "native.json")
        == 0
    )
    assert [row[0] for row in calls] == ["xm_scope", "attention"]
    assert (tmp_path / "decode-scope.json").is_file()
    assert (tmp_path / "attention-path.json").is_file()


@pytest.mark.parametrize("failure", ["scope", "ambiguous_trace"])
def test_failed_scope_never_derives_attention_results(tmp_path, monkeypatch, failure):
    directory = tmp_path / "capture/forward-trace"
    directory.mkdir(parents=True)
    (directory / "forward-a.jsonl").write_text("fixture")
    if failure == "ambiguous_trace":
        (directory / "forward-b.jsonl").write_text("fixture")

    def reject(*args, **kwargs):
        raise ValueError("actual scope rejected")

    def never(*args, **kwargs):
        pytest.fail("Attention analysis must not run without passed scope")

    monkeypatch.setattr(PROFILE.SCOPE, "verify", reject)
    monkeypatch.setattr(PROFILE.ANALYZE, "attention_profile", never)
    assert (
        PROFILE.diagnose(tmp_path, tmp_path / "proof.json", tmp_path / "native.json")
        == 2
    )
    assert not (tmp_path / "attention-path.json").exists()
