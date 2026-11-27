"""Portable report evidence contracts, not model or GPU validation."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "xm_report_test", ROOT / "benchmark/xm_sparse/report.py"
)
REPORT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REPORT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def contracts(tmp_path, monkeypatch):
    campaign_tool = REPORT.report_tool("campaign")
    stage = campaign_tool.recipe("fused")
    fingerprint, workload, commit = {"sha256": "a" * 64}, {"sha256": "b" * 64}, "c" * 40
    campaign = write(tmp_path / "campaign.json", {"phase": "performance"})
    benchmark = write(
        tmp_path / "case/evidence/manifest.json",
        {"kind": "benchmark", "stage": stage, "settings": {"model": "/models/qwen"}},
    )
    data = {
        "schema_version": 1,
        "phase": "performance",
        "status": "analyzed",
        "source_fingerprint": fingerprint,
        "workload_identity": workload,
        "native_source_commits": [commit],
        "rows": [
            {
                "arm": "fused",
                "input_len": 32768,
                "client_concurrency": 4,
                "actual_batch": 4,
                "repeat": 0,
                "eligible": True,
                "mean_tbt_ms": 10,
                "stable_tokens_per_s": 400,
            }
        ],
        "samples": [],
        "paired_repeat_rows": [],
        "same_actual_batch_comparisons": [],
        "observed_stable_capacity_points": [],
        "cache_hbm_budget_bytes": None,
        "artifacts": [REPORT.identity(campaign), REPORT.identity(benchmark)],
    }
    replay = copy.deepcopy(data)
    monkeypatch.setattr(
        REPORT,
        "report_tool",
        lambda name: (
            SimpleNamespace(
                analyze_campaign=lambda path: copy.deepcopy(replay),
                equal=lambda a, b: a == b,
            )
            if name == "analyze"
            else campaign_tool
        ),
    )
    comparison = {
        "exact_greedy_token_match": True,
        "max_abs_logprob_error": 0.001,
        "logprob_atol": 0.02,
    }
    native = {
        "kind": "correctness",
        "status": "success",
        "stage": stage,
        "settings": {
            "execute": True,
            "model": "/models/qwen",
            "input_lens": [32768],
            "workload_sha256": workload["sha256"],
        },
        "runtime": {
            "git_commit": commit,
            "archived_source": {
                "provenance": "git_archive",
                "source_commit": commit,
                "source_archive_sha256": "d" * 64,
            },
        },
        "groups": [
            {
                "variant": variant,
                "eager": eager,
                "status": "success",
                "comparison_to_cpu_hisparse": copy.deepcopy(comparison),
                "slot_reuse_comparison": copy.deepcopy(comparison),
                "rounds": [[32768, 32768]],
                "lifecycle_required": True,
                "lifecycle_results": [
                    {"case": case, "status": "passed"}
                    for case in ("abort_prefill", "abort_decode", "retract_decode")
                ],
            }
            for variant in ("cpu", "hbm")
            for eager in (True, False)
        ],
    }
    native_path = write(tmp_path / "native-correctness.json", native)
    proof = {
        "phase": "correctness",
        "status": "success",
        "source_fingerprint": fingerprint,
        "workload_identity": workload,
        "settings": {
            "model": "/models/qwen",
            "input_lens": [32768],
            "disable_cuda_graph": False,
        },
        "cases": [
            {
                "arm": "fused",
                "stage": stage,
                "stage_path": str(tmp_path / "unavailable/stage.json"),
                "status": "success",
                "returncode": 0,
            }
        ],
    }
    proof_path = write(tmp_path / "correctness.json", proof)
    write(
        campaign,
        {"phase": "performance", "correctness_gate": REPORT.identity(proof_path)},
    )
    data["artifacts"][0] = REPORT.identity(campaign)
    replay.clear()
    replay.update(copy.deepcopy(data))
    analysis_path = write(tmp_path / "analysis.json", data)
    return SimpleNamespace(
        data=data,
        replay=replay,
        analyses=[(analysis_path, data)],
        proof=proof,
        native=native,
        evidence=[(proof_path, proof), (native_path, native)],
        benchmark=benchmark,
        source=data,
    )


def test_complete_same_source_model_gate_allows_measurements(contracts):
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "passed" and audit["model_result_claims_allowed"]
    assert audit["accepted_model_receipts"][0]["correctness_round_sizes"] == [2]
    assert "small HBM" in audit["scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "source",
        "workload",
        "native_plan",
        "revision",
        "archive",
        "missing_hbm",
        "tokens",
        "logprobs",
        "tolerance",
        "reuse",
        "lifecycle",
        "wrong_arm",
    ],
)
def test_failed_or_incomplete_model_proofs_never_allow_benefits(contracts, mutation):
    proof, native = contracts.proof, contracts.native
    if mutation == "failed":
        proof["status"] = "failed"
    elif mutation == "source":
        proof["source_fingerprint"] = {"sha256": "e" * 64}
    elif mutation == "workload":
        proof["workload_identity"] = {"sha256": "e" * 64}
    elif mutation == "native_plan":
        native["settings"]["execute"] = False
    elif mutation == "revision":
        native["runtime"]["git_commit"] = "e" * 40
    elif mutation == "archive":
        native["runtime"]["archived_source"] = {}
    elif mutation == "missing_hbm":
        native["groups"] = [g for g in native["groups"] if g["variant"] == "cpu"]
    elif mutation == "tokens":
        native["groups"][0]["comparison_to_cpu_hisparse"][
            "exact_greedy_token_match"
        ] = False
    elif mutation == "logprobs":
        native["groups"][0]["comparison_to_cpu_hisparse"][
            "max_abs_logprob_error"
        ] = 0.03
    elif mutation == "tolerance":
        native["groups"][0]["comparison_to_cpu_hisparse"]["logprob_atol"] = 0.03
    elif mutation == "reuse":
        native["groups"][0]["slot_reuse_comparison"] = {}
    elif mutation == "lifecycle":
        native["groups"][0]["lifecycle_results"].pop()
    else:
        proof["cases"][0]["stage"] = copy.deepcopy(proof["cases"][0]["stage"])
        proof["cases"][0]["stage"]["xm_adaptation"]["arm"] = "baseline"
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked" and not audit["model_result_claims_allowed"]
    shown = REPORT.display_analyses(contracts.analyses, audit)[0][1]
    assert (
        shown["rows"]
        == shown["samples"]
        == shown["observed_stable_capacity_points"]
        == []
    )
    assert contracts.data["rows"]  # Input evidence is preserved.


def test_derived_analysis_cannot_diverge_from_fresh_raw_replay(contracts):
    contracts.data["rows"][0]["mean_tbt_ms"] = 0.1
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked" and "fresh raw replay" in audit["failures"][0]


def test_changed_raw_artifact_is_not_accepted(contracts):
    contracts.benchmark.write_text("{}")
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked" and "artifact" in audit["failures"][0]


def test_capacity_comparison_groups_cannot_mix_client_load_or_graph():
    points = [
        {
            "arm": "fused",
            "input_len": 32768,
            "client_concurrency": 4,
            "graph_enabled": True,
        },
        {
            "arm": "combined-protected",
            "input_len": 32768,
            "client_concurrency": 12,
            "graph_enabled": True,
        },
        {
            "arm": "combined-protected",
            "input_len": 32768,
            "client_concurrency": 4,
            "graph_enabled": False,
        },
    ]
    grouped = REPORT.capacity_groups({"observed_stable_capacity_points": points})
    assert len(grouped) == 3 and all(len(rows) == 1 for rows in grouped.values())


@pytest.mark.parametrize(
    "mutation", [None, "unselected", "missing_mma", "source", "artifact"]
)
def test_mma_candidate_needs_selected_backend_and_native_instruction_evidence(
    tmp_path, contracts, mutation
):
    artifact = tmp_path / "native.sass"
    artifact.write_text(
        "Function : fused_mma\nUBLKCP\nHMMA.16816.F32.BF16\nFunction : merge\nNOP\n"
    )
    record = {
        "kind": "xm_mechanism_identity",
        "attention_implementation": "mma",
        "selected_model_backend": "mma",
        "source_fingerprint": contracts.source["source_fingerprint"],
        "native_source_commit": "c" * 40,
        "native_sass": {
            "tma_instruction_observed": True,
            "mma_instruction_observed": True,
            "kernel_symbol": "fused_mma",
            "artifact": REPORT.identity(artifact),
        },
    }
    if mutation == "unselected":
        record["selected_model_backend"] = "scalar"
    elif mutation == "missing_mma":
        record["native_sass"]["mma_instruction_observed"] = False
    elif mutation == "source":
        record["source_fingerprint"] = {"sha256": "e" * 64}
    elif mutation == "artifact":
        artifact.write_text("changed")
    identity = REPORT.mechanism_identity(
        contracts.source, [(tmp_path / "receipt.json", record)]
    )
    assert identity["attention_implementation"] == (
        "mma" if mutation is None else "unverified"
    )


@pytest.mark.parametrize("tma", ["UBLKCP", "UTMALDG"])
def test_tma_and_mma_must_be_in_the_same_selected_function(tmp_path, contracts, tma):
    artifact = tmp_path / "split.sass"
    artifact.write_text(
        f"Function : old_tma\n{tma}\nFunction : fused_mma\nHMMA.16816.F32.BF16\n"
    )
    record = {
        "kind": "xm_mechanism_identity",
        "attention_implementation": "mma",
        "selected_model_backend": "mma",
        "source_fingerprint": contracts.source["source_fingerprint"],
        "native_source_commit": "c" * 40,
        "native_sass": {
            "tma_instruction_observed": True,
            "mma_instruction_observed": True,
            "kernel_symbol": "fused_mma",
            "artifact": REPORT.identity(artifact),
        },
    }
    identity = REPORT.mechanism_identity(
        contracts.source, [(tmp_path / "receipt.json", record)]
    )
    assert identity["attention_implementation"] == "unverified"
    artifact.write_text(f"Function : fused_mma\n{tma}\nHMMA.16816.F32.BF16\n")
    record["native_sass"]["artifact"] = REPORT.identity(artifact)
    assert (
        REPORT.mechanism_identity(
            contracts.source, [(tmp_path / "receipt.json", record)]
        )["attention_implementation"]
        == "mma"
    )


def test_missing_identity_is_unverified_and_scalar_requires_explicit_source_bound_selection(
    contracts,
):
    missing = REPORT.mechanism_identity(contracts.source, [])
    assert missing["attention_implementation"] == "unverified"
    assert "不指定实现" in REPORT.attention_description(missing)
    scalar = {
        "kind": "xm_mechanism_identity",
        "attention_implementation": "scalar",
        "selected_model_backend": "scalar",
        "source_fingerprint": contracts.source["source_fingerprint"],
        "native_source_commit": "c" * 40,
    }
    assert (
        REPORT.mechanism_identity(contracts.source, [(Path("scalar.json"), scalar)])[
            "attention_implementation"
        ]
        == "scalar"
    )


def test_two_analyses_can_bind_different_same_source_correctness_proofs(
    contracts, tmp_path, monkeypatch
):
    first = contracts.data
    second = copy.deepcopy(first)
    second["rows"][0]["actual_batch"] = 8
    proof = copy.deepcopy(contracts.proof)
    proof["settings"]["control_batch"] = 4
    proof_path = write(tmp_path / "correctness-b4.json", proof)
    campaign = write(
        tmp_path / "campaign-b4.json",
        {"phase": "performance", "correctness_gate": REPORT.identity(proof_path)},
    )
    second["artifacts"][0] = REPORT.identity(campaign)
    path = write(tmp_path / "analysis-b4.json", second)
    campaign_tool = REPORT.report_tool("campaign")
    replays = {
        first["artifacts"][0]["path"]: first,
        second["artifacts"][0]["path"]: second,
    }
    monkeypatch.setattr(
        REPORT,
        "report_tool",
        lambda name: (
            SimpleNamespace(
                analyze_campaign=lambda path: copy.deepcopy(replays[str(path)]),
                equal=lambda a, b: a == b,
            )
            if name == "analyze"
            else campaign_tool
        ),
    )
    audit = REPORT.audit_report_evidence(
        contracts.analyses + [(path, second)],
        contracts.evidence + [(proof_path, proof)],
    )
    assert audit["status"] == "passed"
    bindings = audit["analysis_model_proof_bindings"]
    assert (
        len(bindings) == 2
        and bindings[0]["proof"]["sha256"] != bindings[1]["proof"]["sha256"]
    )
    assert [b["actual_measurement_batches"] for b in bindings] == [[4], [8]]


def test_unbound_original_campaign_cannot_borrow_another_passed_proof(contracts):
    path = Path(contracts.data["artifacts"][0]["path"])
    write(path, {"phase": "performance"})
    contracts.data["artifacts"][0] = REPORT.identity(path)
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked" and "not bound" in audit["failures"][0]


@pytest.mark.parametrize("mutation", [None, "failed", "source", "sqlite", "batch"])
def test_independent_profile_requires_its_bound_decode_scope(
    tmp_path, contracts, mutation
):
    path = tmp_path / "attention-path.json"
    data = {
        "status": "analyzed",
        "sqlite": {"sha256": "f" * 64},
        "actual_batch_size": 4,
        "observed_decode_steps": 16,
        "per_step": [{}] * 16,
    }
    scope = {
        "status": "passed",
        "actual_decode_scope_verified": True,
        "arm": "fused",
        "sqlite_sha256": "f" * 64,
        "actual_batch_size": 4,
        "actual_decode_graph_launches": 16,
        "provenance": {
            "source_fingerprint": contracts.source["source_fingerprint"],
            "source_commit": "c" * 40,
        },
    }
    if mutation == "failed":
        scope["status"] = "failed"
    elif mutation == "source":
        scope["provenance"]["source_fingerprint"] = {"sha256": "e" * 64}
    elif mutation == "sqlite":
        scope["sqlite_sha256"] = "e" * 64
    elif mutation == "batch":
        scope["actual_batch_size"] = 8
    write(tmp_path / "decode-scope.json", scope)
    checked = REPORT.audit_profile_identities(
        [("fused", path, data)], contracts.source
    )[0][2]
    assert checked["status"] == (
        "analyzed" if mutation is None else "unverified_report_scope"
    )


def test_formal_failure_review_can_be_generated_without_performance_numbers(
    tmp_path, monkeypatch
):
    evidence = write(
        tmp_path / "failed.json",
        {
            "phase": "correctness",
            "status": "failed",
            "error": "token divergence",
            "source_fingerprint": {"sha256": "a" * 64},
            "workload_identity": {"sha256": "b" * 64},
            "cases": [],
        },
    )
    font, output = tmp_path / "fake.ttf", tmp_path / "review.pdf"
    font.write_bytes(b"portable-test-font-placeholder")
    captured = {}

    def render(analyses, profiles, evidence, output, **kwargs):
        captured.update(kwargs)
        assert analyses == []
        output.write_bytes(b"portable-report-contract-placeholder")

    monkeypatch.setattr(REPORT, "render_report", render)
    REPORT.main(
        [
            "--diagnostic-only",
            "--evidence",
            str(evidence),
            "--output",
            str(output),
            "--font",
            str(font),
            "--skip-plots",
        ]
    )
    receipt = json.loads(output.with_suffix(".receipt.json").read_text())
    assert captured["report_audit"]["status"] == "blocked"
    assert receipt["model_evidence_audit"]["model_result_claims_allowed"] is False
    assert "claims suppressed" in receipt["scope"]
