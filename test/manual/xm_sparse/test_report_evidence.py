"""Portable report evidence contracts, not model or GPU validation."""

import copy
import importlib.util
import json
import sys
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


def populate_raw_correctness(root, native, *, long_round=None):
    """Synthetic complete raw execution, with no claim of GPU/model validation."""
    output_len = 20
    settings = native["settings"]
    settings.update(
        output_len=output_len,
        logprob_atol=0.02,
        chunked_prefill_size=4096,
        prefill_max_requests=1,
        output_dir=str(root),
    )
    long_round = list(long_round or settings["input_lens"])
    comparison = {
        "exact_greedy_token_match": True,
        "max_abs_logprob_error": 0.0,
        "logprob_atol": 0.02,
    }
    for group in native["groups"]:
        gid = group["variant"] + ("-eager" if group["eager"] else "-graph")
        group.update(
            id=gid,
            rounds=[long_round, [65, 4097], long_round],
            trace_dir=str(root / (gid + "-trace")),
            comparison_to_cpu_hisparse=copy.deepcopy(comparison),
            slot_reuse_comparison=copy.deepcopy(comparison),
        )
        rows = [
            {
                "type": "session",
                "schema_version": 1,
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 1,
                "disaggregation_mode": "null",
                "enable_mixed_chunk": False,
                "enable_two_batch_overlap": False,
                "speculative_algorithm": None,
                "disable_radix_cache": True,
                "disable_overlap_schedule": True,
            }
        ]

        def forward(mode, ids, chunks=None):
            rows.append(
                {
                    "type": "forward",
                    "sequence": sum(row.get("type") == "forward" for row in rows),
                    "forward_mode": mode,
                    "request_ids": ids,
                    "extend_lens": chunks,
                    "success": True,
                    "stream_id": 1,
                }
            )

        for index, lengths in enumerate(group["rounds"]):
            ids = [f"{gid}-round{index}-r{i}" for i in range(len(lengths))]
            write(
                root / f"{gid}-round{index}.json",
                {
                    "input_ids": [[1] * n for n in lengths],
                    "responses": [
                        {
                            "output_ids": [1] * output_len,
                            "meta_info": {
                                "id": rid,
                                "prompt_tokens": n,
                                "completion_tokens": output_len,
                                "output_token_logprobs": [[-0.1, 1, None]] * output_len,
                            },
                        }
                        for rid, n in zip(ids, lengths)
                    ],
                },
            )
            for rid, length in zip(ids, lengths):
                while length:
                    chunk = min(length, 4096)
                    forward("extend", [rid], [chunk])
                    length -= chunk
            for _ in range(output_len - 1):
                forward("decode", ids)
        rows.append(
            {
                "type": "idle_flush",
                "last_sequence": sum(row.get("type") == "forward" for row in rows) - 1,
            }
        )
        trace_path = Path(group["trace_dir"]) / "forward.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        write(
            root / (gid + "-server-info.json"),
            {
                "disable_cuda_graph": group["eager"],
                "internal_states": [{"cuda_graph_bs": [1, 2, len(long_round)]}],
            },
        )
    return native


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
                "client_concurrency": 1,
                "actual_batch": 1,
                "graph_enabled": True,
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
    real_tool = REPORT.report_tool
    monkeypatch.setattr(
        REPORT,
        "report_tool",
        lambda name: (
            SimpleNamespace(
                analyze_campaign=lambda path: copy.deepcopy(replay),
                equal=lambda a, b: a == b,
            )
            if name == "analyze"
            else real_tool(name)
        ),
    )
    comparison = {
        "exact_greedy_token_match": True,
        "max_abs_logprob_error": 0.0,
        "logprob_atol": 0.02,
    }
    native = {
        "kind": "correctness",
        "status": "success",
        "stage": stage,
        "settings": {
            "execute": True,
            "model": "/models/qwen",
            "input_lens": [16384, 32768],
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
    populate_raw_correctness(tmp_path, native)
    native_path = write(tmp_path / "native-correctness.json", native)
    proof = {
        "phase": "correctness",
        "status": "success",
        "source_fingerprint": fingerprint,
        "workload_identity": workload,
        "settings": {
            "model": "/models/qwen",
            "input_lens": [16384, 32768],
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
    receipt = audit["accepted_model_receipts"][0]
    assert receipt["hbm_context_batch_sizes"] == []
    assert receipt["hbm_single_request_context_lengths"] == [16384, 32768]
    replay = receipt["actual_execution"]
    assert replay["status"] == "passed"
    assert replay["groups"][0]["rounds"][0]["observed_actual_batches"] == [2]
    assert "no per-forward" in replay["graph_scope"]


def test_requested_mixed_batch_cannot_prove_same_context_main_performance(contracts):
    contracts.data["rows"][0]["actual_batch"] = 2
    contracts.replay.update(copy.deepcopy(contracts.data))
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked"
    assert "16 consecutive complete cohort" in audit["failures"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "output_ids",
        "tokens",
        "logprobs",
        "input_length",
        "reuse_input",
        "missing_decode",
        "duplicate_member",
        "foreign_member",
        "missing_sequence",
        "missing_flush",
        "mixed_stream",
        "graph_mode",
        "missing_graph_shape",
    ],
)
def test_raw_execution_is_replayed_instead_of_trusting_success(contracts, mutation):
    native = contracts.native
    root = Path(contracts.evidence[1][0]).parent
    group = native["groups"][0]
    path = root / (group["id"] + "-round0.json")
    payload = REPORT.read(path)
    if mutation in ("output_ids", "tokens", "logprobs", "input_length", "reuse_input"):
        if mutation == "output_ids":
            payload["responses"][0]["output_ids"][0] = 9
        elif mutation == "tokens":
            payload["responses"][0]["output_ids"][0] = 9
            payload["responses"][0]["meta_info"]["output_token_logprobs"][0][1] = 9
        elif mutation == "logprobs":
            payload["responses"][0]["meta_info"]["output_token_logprobs"][0][0] += 0.001
        elif mutation == "input_length":
            payload["input_ids"][0].pop()
        else:
            payload["input_ids"][0][0] = 9
        write(path, payload)
    elif mutation in ("graph_mode", "missing_graph_shape"):
        graph = native["groups"][1]
        path = root / (graph["id"] + "-server-info.json")
        payload = REPORT.read(path)
        if mutation == "graph_mode":
            payload["cuda_graph_config"] = {"decode": {"backend": "disabled"}}
        else:
            payload["internal_states"][0]["cuda_graph_bs"] = [1]
        write(path, payload)
    else:
        path = Path(group["trace_dir"]) / "forward.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = next(row for row in rows if row.get("forward_mode") == "decode")
        if mutation == "missing_decode":
            rows.remove(selected)
            forwards = [row for row in rows if row.get("type") == "forward"]
            for i, row in enumerate(forwards):
                row["sequence"] = i
            rows[-1]["last_sequence"] = len(forwards) - 1
        elif mutation == "duplicate_member":
            selected["request_ids"].append(selected["request_ids"][0])
        elif mutation == "foreign_member":
            selected["request_ids"].append("unrelated-request")
        elif mutation == "missing_sequence":
            selected["sequence"] += 1
        elif mutation == "missing_flush":
            rows.pop()
        else:
            selected["stream_id"] = 2
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked", audit
    assert not audit["model_result_claims_allowed"]


def test_both_long_rounds_need_16_consecutive_complete_cohort_forwards(contracts):
    native = contracts.native
    root = Path(contracts.evidence[1][0]).parent
    native["settings"]["input_lens"] = [32768]
    populate_raw_correctness(root, native, long_round=[32768] * 2)
    args = {"required_batches": [(32768, 2, True)]}
    good = REPORT.replay_correctness_execution(contracts.evidence[1][0], native, **args)
    assert all(
        item["rounds"][2]["max_consecutive_full_forwards"] == 19
        for item in good["groups"]
    )
    group = native["groups"][1]
    path = Path(group["trace_dir"]) / "forward.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target_ids = set(
        REPORT.read(root / (group["id"] + "-round2.json"))["responses"][i]["meta_info"][
            "id"
        ]
        for i in range(2)
    )
    changed, full = [], 0
    for row in rows:
        if (
            row.get("forward_mode") == "decode"
            and set(row["request_ids"]) == target_ids
        ):
            full += 1
            if full > 15:
                changed.extend(
                    dict(row, request_ids=[rid]) for rid in row["request_ids"]
                )
                continue
        changed.append(row)
    forwards = [row for row in changed if row.get("type") == "forward"]
    for i, row in enumerate(forwards):
        row["sequence"] = i
    changed[-1]["last_sequence"] = len(forwards) - 1
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="16 consecutive complete cohort"):
        REPORT.replay_correctness_execution(contracts.evidence[1][0], native, **args)


def test_relocated_raw_correctness_uses_native_manifest_directory(contracts):
    for group in contracts.native["groups"]:
        group["trace_dir"] = "/results/unavailable/" + group["id"] + "-trace"
    receipt = REPORT.replay_correctness_execution(
        contracts.evidence[1][0], contracts.native, required_batches=[(32768, 1, True)]
    )
    assert receipt["status"] == "passed"


def test_mixed_b2_member_graph_coverage_uses_actual_cohort_capture(contracts):
    root = Path(contracts.evidence[1][0]).parent
    for group in contracts.native["groups"]:
        if not group["eager"]:
            path = root / (group["id"] + "-server-info.json")
            payload = REPORT.read(path)
            payload["internal_states"][0]["cuda_graph_bs"] = [2]
            write(path, payload)
    replay = REPORT.replay_correctness_execution(
        contracts.evidence[1][0], contracts.native, required_batches=[(32768, 1, True)]
    )
    assert replay["status"] == "passed"
    assert replay["groups"][1]["rounds"][0]["observed_actual_batches"] == [2]


def test_legacy_flat_row_recovers_actual_eager_mode_from_matching_sample(contracts):
    data = copy.deepcopy(contracts.data)
    data["rows"][0].pop("graph_enabled")
    data["samples"] = [dict(data["rows"][0], graph_enabled=False)]
    assert REPORT._required_correctness_batches(data, "fused") == [(32768, 1, False)]
    assert REPORT.measurement_scope(data)[3] is False
    data["samples"] = []
    with pytest.raises(ValueError, match="actual measurement Graph mode"):
        REPORT.measurement_scope(data)


@pytest.mark.parametrize("measurement_graph", [True, False])
@pytest.mark.parametrize("missing", ["cohort", "second_long_round", "capture"])
def test_small_hbm_capacity_coverage_intersects_required_modes_and_long_rounds(
    contracts, measurement_graph, missing
):
    native = contracts.native
    root = Path(contracts.evidence[1][0]).parent
    native["settings"]["input_lens"] = [32768]
    populate_raw_correctness(root, native, long_round=[32768] * 2)
    graph = next(
        group
        for group in native["groups"]
        if group["variant"] == "hbm" and not group["eager"]
    )
    if missing == "capture":
        path = root / (graph["id"] + "-server-info.json")
        payload = REPORT.read(path)
        payload["internal_states"][0]["cuda_graph_bs"] = [1]
        write(path, payload)
    else:
        path = Path(graph["trace_dir"]) / "forward.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        targets = {
            response["meta_info"]["id"]
            for response in REPORT.read(root / (graph["id"] + "-round2.json"))[
                "responses"
            ]
        }
        changed = []
        for row in rows:
            if row.get("forward_mode") == "decode" and (
                missing == "cohort" or set(row["request_ids"]) == targets
            ):
                changed.extend(
                    dict(row, request_ids=[rid]) for rid in row["request_ids"]
                )
            else:
                changed.append(row)
        forwards = [row for row in changed if row.get("type") == "forward"]
        for i, row in enumerate(forwards):
            row["sequence"] = i
        changed[-1]["last_sequence"] = len(forwards) - 1
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8"
        )
    data = copy.deepcopy(contracts.data)
    data["phase"] = "capacity"
    data["rows"][0].update(actual_batch=2, graph_enabled=measurement_graph)
    receipts = REPORT.verify_model_receipt(
        contracts.evidence[0][0],
        contracts.proof,
        contracts.evidence,
        data,
        {"/models/qwen"},
    )
    assert receipts[0]["hbm_context_batch_sizes"] == (
        [] if measurement_graph else [(32768, 2)]
    )
    assert receipts[0]["hbm_batch_coverage_modes"] == (
        ["eager", "graph"] if measurement_graph else ["eager"]
    )
    assert receipts[0]["hbm_single_request_context_lengths"] == (
        [] if measurement_graph and missing == "capture" else [32768]
    )


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
    second["rows"][0]["actual_batch"] = 2
    proof = copy.deepcopy(contracts.proof)
    proof["settings"].update(control_batch=2, input_lens=[32768])
    second_native = copy.deepcopy(contracts.native)
    second_native["settings"]["input_lens"] = [32768]
    native_root = tmp_path / "second-proof" / "evidence"
    populate_raw_correctness(native_root, second_native, long_round=[32768] * 2)
    write(native_root / "manifest.json", second_native)
    proof["cases"][0]["stage_path"] = str(native_root.parent / "stage.json")
    proof_path = write(tmp_path / "correctness-b2.json", proof)
    campaign = write(
        tmp_path / "campaign-b4.json",
        {"phase": "performance", "correctness_gate": REPORT.identity(proof_path)},
    )
    second["artifacts"][0] = REPORT.identity(campaign)
    path = write(tmp_path / "analysis-b4.json", second)
    real_tool = REPORT.report_tool
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
            else real_tool(name)
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
    assert [b["actual_measurement_batches"] for b in bindings] == [[1], [2]]


def test_unbound_original_campaign_cannot_borrow_another_passed_proof(contracts):
    path = Path(contracts.data["artifacts"][0]["path"])
    write(path, {"phase": "performance"})
    contracts.data["artifacts"][0] = REPORT.identity(path)
    audit = REPORT.audit_report_evidence(contracts.analyses, contracts.evidence)
    assert audit["status"] == "blocked" and "not bound" in audit["failures"][0]


@pytest.mark.parametrize(
    "mutation", [None, "failed", "source", "sqlite", "batch", "graph", "device"]
)
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
        "main_graph_kernel_inventory": {"graph_id": 2, "device_id": 0},
    }
    scope = {
        "status": "passed",
        "actual_decode_scope_verified": True,
        "arm": "fused",
        "sqlite_sha256": "f" * 64,
        "actual_batch_size": 4,
        "actual_decode_graph_launches": 16,
        "graph_id": 2,
        "device_role": {"trace_device_id": 0},
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
    elif mutation == "graph":
        data["main_graph_kernel_inventory"]["graph_id"] = 7
    elif mutation == "device":
        data["main_graph_kernel_inventory"]["device_id"] = 1
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


@pytest.fixture
def large_contracts(contracts, tmp_path):
    """Complete synthetic raw CPU contract; no GPU/model validation is claimed."""
    tool = REPORT.report_tool("summary_batch_correctness")
    batch, length, output_len = 4, 5000, 80
    workload_path = write(tmp_path / "workload.json", {"frozen": "contract-fixture"})
    workload = {key: REPORT.identity(workload_path)[key] for key in ("sha256", "bytes")}
    data, proof = contracts.data, contracts.proof
    proof["workload_identity"] = data["workload_identity"] = workload
    proof["settings"]["input_lens"] = [length]
    small_natives = {}
    proof["cases"] = []
    evidence = []
    for arm in tool.ARMS:
        small = copy.deepcopy(contracts.native)
        small["stage"] = tool.CAMPAIGN.recipe(arm)
        small["settings"].update(
            input_lens=[length], workload_sha256=workload["sha256"]
        )
        raw_root = tmp_path / ("small-" + arm) / "evidence"
        populate_raw_correctness(raw_root, small, long_round=[length] * 2)
        path = write(raw_root / "manifest.json", small)
        small_natives[arm] = (path, small)
        evidence.append((path, small))
        proof["cases"].append(
            {
                "arm": arm,
                "stage": small["stage"],
                "stage_path": str(tmp_path / "unavailable" / arm / "stage.json"),
                "status": "success",
                "returncode": 0,
            }
        )
    proof_path = write(tmp_path / "small-proof.json", proof)
    evidence.append((proof_path, proof))
    budget = 8 << 30
    data.update(phase="capacity", cache_hbm_budget_bytes=budget)
    data["rows"] = [
        dict(data["rows"][0], arm=arm, input_len=length, actual_batch=batch)
        for arm in tool.ARMS
    ]
    campaign = write(
        tmp_path / "capacity-campaign.json",
        {"phase": "capacity", "correctness_gate": REPORT.identity(proof_path)},
    )
    benchmark = write(
        contracts.benchmark,
        {
            "kind": "benchmark",
            "stage": tool.CAMPAIGN.recipe("fused", budget=budget),
            "settings": {"model": "/models/qwen"},
        },
    )
    data["artifacts"] = [REPORT.identity(campaign), REPORT.identity(benchmark)]
    contracts.replay.clear()
    contracts.replay.update(copy.deepcopy(data))
    control = {
        "schema_version": 1,
        "phase": "large_batch_summary_equivalence",
        "status": "success",
        "source_fingerprint": data["source_fingerprint"],
        "settings": {
            "execute": True,
            "model": "/models/qwen",
            "workload": str(workload_path),
            "input_len": length,
            "batch_sizes": [batch],
            "output_len": output_len,
        },
        "small_hbm_gate": {
            "status": "passed",
            **REPORT.identity(proof_path),
            "native_reports": [
                {"arm": arm, **REPORT.identity(path)}
                for arm, (path, _) in small_natives.items()
            ],
        },
        "cases": [],
        "comparisons": [],
    }
    natives, arguments = {}, {}
    for arm in tool.ARMS:
        stage = tool.CAMPAIGN.recipe(arm)
        output_dir = tmp_path / ("large-" + arm) / "evidence"
        args = SimpleNamespace(
            input_lens=[length],
            output_len=output_len,
            max_total_tokens=98304,
            capacity_ratios=[16],
            cpu_capacity_ratio=16,
            chunked_prefill_size=4096,
            prefill_max_requests=1,
            top_k=None,
            device_buffer_size=None,
            output_dir=output_dir,
            logprob_atol=0.02,
            concurrencies=[batch],
            num_prompts=batch,
            execute=False,
            model="/models/qwen",
            workload=str(workload_path),
            workload_sha256=workload["sha256"],
        )
        settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        need = tool.allocation_requirements(stage, args, batch)
        capacity = {
            "backing_store": "cpu",
            "storage_device_ids": [],
            "physical_compute_slots": 98304,
            "backing_slots": 1572928,
            "padding_slots": 64,
            "compute_kv_bytes": 98304 * 98304,
            "backing_kv_bytes": 1572928 * 98304,
        }
        native = {
            "kind": "correctness",
            "status": "success",
            "stage": stage,
            "settings": settings,
            "runtime": copy.deepcopy(contracts.native["runtime"]),
            "cpu_ram_preflight": [{"status": "passed"}],
            "groups": [],
        }
        reference = None
        for eager in (True, False):
            gid = "cpu-" + ("eager" if eager else "graph") + "-ratio16"
            group = {
                "id": gid,
                "variant": "cpu",
                "eager": eager,
                "status": "success",
                "ratio": 16,
                "rounds": need["rounds"],
                "trace_dir": str(output_dir / (gid + "-trace")),
                "capacity": [capacity],
                "lifecycle_required": True,
                "lifecycle_results": [],
            }
            rows = [
                {
                    "type": "session",
                    "tp_size": 1,
                    "tp_rank": 0,
                    "pp_size": 1,
                    "dp_size": 1,
                    "disable_overlap_schedule": True,
                    "disaggregation_mode": "null",
                    "disable_radix_cache": True,
                    "speculative_algorithm": None,
                    "enable_mixed_chunk": False,
                }
            ]

            def extend(rid, count):
                while count:
                    chunk = min(count, 4096)
                    rows.append(
                        {
                            "type": "forward",
                            "sequence": len(rows) - 1,
                            "forward_mode": "extend",
                            "request_ids": [rid],
                            "extend_lens": [chunk],
                            "success": True,
                        }
                    )
                    count -= chunk

            def decode(ids, count):
                for _ in range(count):
                    rows.append(
                        {
                            "type": "forward",
                            "sequence": len(rows) - 1,
                            "forward_mode": "decode",
                            "request_ids": ids,
                            "extend_lens": None,
                            "success": True,
                        }
                    )

            def response(rid, n=length, count=output_len):
                return {
                    "meta_info": {
                        "id": rid,
                        "prompt_tokens": n,
                        "completion_tokens": count,
                        "output_token_logprobs": [[-0.1, 1, None]] * count,
                    }
                }

            for index, lengths in enumerate(group["rounds"]):
                ids = [f"round{index}-r{i}" for i in range(batch)]
                write(
                    output_dir / f"{gid}-round{index}.json",
                    {
                        "input_ids": [[1] * n for n in lengths],
                        "responses": [response(rid, n) for rid, n in zip(ids, lengths)],
                    },
                )
                for rid, n in zip(ids, lengths):
                    extend(rid, n)
                decode(ids, output_len - 1)
            baseline = response("baseline")
            extend("baseline", length)
            decode(["baseline"], output_len - 1)
            prefix = output_dir / (gid + "-lifecycle")
            write(prefix.with_name(prefix.name + "-baseline.json"), baseline)
            for case in ("abort_prefill", "abort_decode", "retract_decode"):
                rid, recovery_id = "fault-" + case, "recovery-" + case
                fault = response(
                    rid,
                    count=(
                        0
                        if case == "abort_prefill"
                        else (2 if case == "abort_decode" else output_len)
                    ),
                )
                if case == "abort_prefill":
                    extend(rid, 2048)
                else:
                    extend(rid, length)
                    decode([rid], 1)
                    if case == "retract_decode":
                        extend(rid, 4096)
                        decode([rid], output_len - 2)
                        fault["meta_info"]["total_retractions"] = 1
                if case != "retract_decode":
                    fault["meta_info"]["finish_reason"] = {"type": "abort"}
                extend(recovery_id, length)
                decode([recovery_id], output_len - 1)
                item = {
                    "case": case,
                    "attempt": 0,
                    "status": "passed",
                    "request_id": rid,
                    "response": fault,
                    "recovery_response": response(recovery_id),
                    "forward_facts": tool.CORRECTNESS.LIFECYCLE.forward_facts(
                        [rows], rid, 1
                    ),
                    "recovery_comparison": {
                        "exact_greedy_token_match": True,
                        "max_abs_logprob_error": 0.0,
                        "logprob_atol": 0.02,
                    },
                }
                if case == "retract_decode":
                    item["resumed_comparison"] = copy.deepcopy(
                        item["recovery_comparison"]
                    )
                write(prefix.with_name(f"{prefix.name}-{case}-0.json"), item)
                group["lifecycle_results"].append(item)
            rows.append({"type": "idle_flush", "last_sequence": len(rows) - 2})
            trace_path = Path(group["trace_dir"]) / "forward.jsonl"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            write(
                output_dir / (gid + "-server-info.json"),
                {
                    "disable_cuda_graph": eager,
                    "internal_states": [
                        {"disable_cuda_graph": eager, "cuda_graph_bs": [1, 2, batch]}
                    ],
                },
            )
            fresh, outputs = tool.verify_round_trace(group, args, batch)
            group["large_batch_trace_equivalence"] = fresh
            reference = outputs if reference is None else reference
            group["comparison_to_cpu_hisparse"] = tool.CORRECTNESS.COMPARE.compare(
                reference, outputs, 0.02
            )
            group["slot_reuse_comparison"] = tool.CORRECTNESS.COMPARE.compare(
                [outputs[0]], [outputs[2]], 0.02
            )
            native["groups"].append(group)
        native_path = write(output_dir / "manifest.json", native)
        control["cases"].append(
            {
                "arm": arm,
                "batch": batch,
                "stage": stage,
                "settings": settings,
                "status": "success",
                "native_manifest": str(native_path),
            }
        )
        natives[arm], arguments[arm] = native, args
    comparisons = tool.compare_arms(
        natives[tool.ARMS[0]],
        natives[tool.ARMS[1]],
        arguments[tool.ARMS[0]],
        arguments[tool.ARMS[1]],
        batch,
    )
    control["comparisons"] = [
        {"batch": batch, "status": "passed", "groups": comparisons}
    ]
    control_path = write(tmp_path / "summary-batch-correctness.json", control)
    evidence.append((control_path, control))
    return SimpleNamespace(
        contracts=contracts,
        data=data,
        evidence=evidence,
        control=control,
        control_path=control_path,
        natives=natives,
        arguments=arguments,
        proof_path=proof_path,
    )


def test_actual_large_cpu_summary_control_admits_capacity_without_fake_hbm(
    large_contracts,
):
    fixture = large_contracts
    audit = REPORT.audit_report_evidence(fixture.contracts.analyses, fixture.evidence)
    assert audit["status"] == "passed", audit["failures"]
    control = audit["analysis_model_proof_bindings"][0]["large_batch_summary_controls"][
        0
    ]
    assert control["batch"] == 4 and "no large-batch full-HBM" in control["scope"]
    assert all(
        native["settings"]["execute"] is False for native in fixture.natives.values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "planned",
        "source",
        "precursor",
        "small_native",
        "batch",
        "workload",
        "archive",
        "recipe",
        "trace",
        "graph",
        "reuse",
        "lifecycle",
        "recovery",
        "comparison",
    ],
)
def test_incomplete_large_control_blocks_capacity_only(large_contracts, mutation):
    fixture = large_contracts
    if mutation == "missing":
        fixture.evidence = [
            (path, record)
            for path, record in fixture.evidence
            if record.get("phase") != "large_batch_summary_equivalence"
        ]
    elif mutation == "planned":
        fixture.control["settings"]["execute"] = False
    elif mutation == "source":
        fixture.control["source_fingerprint"] = {"sha256": "e" * 64}
    elif mutation == "precursor":
        fixture.control["small_hbm_gate"]["sha256"] = "e" * 64
    elif mutation == "small_native":
        fixture.control["small_hbm_gate"]["native_reports"][0]["bytes"] += 1
    elif mutation == "batch":
        fixture.control["cases"][0]["batch"] = 8
    elif mutation == "comparison":
        fixture.control["comparisons"][0]["groups"][0]["max_abs_logprob_error"] = 0.01
    else:
        native = fixture.natives["combined-protected"]
        group = native["groups"][0]
        if mutation == "workload":
            native["settings"]["workload_sha256"] = "e" * 64
        elif mutation == "archive":
            native["runtime"]["archived_source"]["source_archive_sha256"] = "e" * 64
        elif mutation == "recipe":
            native["stage"]["xm_adaptation"]["arm"] = "combined"
        elif mutation == "trace":
            Path(group["large_batch_trace_equivalence"]["trace"]["path"]).write_text(
                "changed", encoding="utf-8"
            )
        elif mutation == "graph":
            graph = native["groups"][1]
            write(
                fixture.arguments["combined-protected"].output_dir
                / (graph["id"] + "-server-info.json"),
                {
                    "disable_cuda_graph": True,
                    "internal_states": [
                        {"disable_cuda_graph": True, "cuda_graph_bs": [4]}
                    ],
                },
            )
        elif mutation == "reuse":
            group["slot_reuse_comparison"]["exact_greedy_token_match"] = False
        elif mutation == "lifecycle":
            group["lifecycle_results"].pop()
        else:
            prefix = fixture.arguments["combined-protected"].output_dir / (
                group["id"] + "-lifecycle"
            )
            path = prefix.with_name(prefix.name + "-abort_decode-0.json")
            receipt = json.loads(path.read_text())
            receipt["recovery_response"]["meta_info"]["output_token_logprobs"][0][1] = 9
            write(path, receipt)
            group["lifecycle_results"][1] = receipt
        write(
            fixture.arguments["combined-protected"].output_dir / "manifest.json", native
        )
    audit = REPORT.audit_report_evidence(fixture.contracts.analyses, fixture.evidence)
    assert audit["status"] == "blocked", audit
    assert not REPORT.display_analyses(fixture.contracts.analyses, audit)[0][1]["rows"]


def test_missing_capacity_control_preserves_valid_latency(
    large_contracts, tmp_path, monkeypatch
):
    fixture = large_contracts
    performance = copy.deepcopy(fixture.data)
    performance.update(phase="performance", cache_hbm_budget_bytes=None)
    performance["rows"] = [
        dict(performance["rows"][0], actual_batch=1, client_concurrency=1)
    ]
    campaign = write(
        tmp_path / "performance-campaign.json",
        {
            "phase": "performance",
            "correctness_gate": REPORT.identity(fixture.proof_path),
        },
    )
    performance["artifacts"][0] = REPORT.identity(campaign)
    performance_path = write(tmp_path / "performance-analysis.json", performance)
    real_tool = REPORT.report_tool
    replays = {
        performance["artifacts"][0]["path"]: performance,
        fixture.data["artifacts"][0]["path"]: fixture.data,
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
            else real_tool(name)
        ),
    )
    analyses = [(performance_path, performance)] + fixture.contracts.analyses
    evidence = [
        (path, record)
        for path, record in fixture.evidence
        if record.get("phase") != "large_batch_summary_equivalence"
    ]
    audit = REPORT.audit_report_evidence(analyses, evidence)
    assert audit["status"] == "partial" and audit["model_result_claims_allowed"]
    shown = REPORT.display_analyses(analyses, audit)
    assert shown[0][1]["rows"] and shown[1][1]["rows"] == []


def test_completed_batch_in_failed_later_control_remains_scoped(large_contracts):
    fixture = large_contracts
    fixture.control["status"] = "failed"
    fixture.control["settings"]["batch_sizes"].append(8)
    fixture.control["cases"].append({"arm": "fused", "batch": 8, "status": "failed"})
    audit = REPORT.audit_report_evidence(fixture.contracts.analyses, fixture.evidence)
    assert audit["status"] == "passed", audit["failures"]
    control = audit["analysis_model_proof_bindings"][0]["large_batch_summary_controls"][
        0
    ]
    assert control["batch"] == 4 and control["summary_campaign_status"] == "failed"


def parent_graph_profile_fixture():
    """Synthetic capture: the pre-score wait is absent from the layer-only list."""
    return {
        "schema_version": 1,
        "status": "analyzed",
        "actual_batch_size": 1,
        "layer_index": 23,
        "observed_decode_steps": 2,
        "total_path_span_us": {"mean": 10},
        "stage_span_us": {"selection": {"mean": 10}},
        "per_step": [
            {
                "stages": {
                    "selection": {
                        "kernel_active_union_us": 8,
                        "internal_idle_gap_us": 2,
                    }
                }
            },
            {
                "stages": {
                    "selection": {
                        "kernel_active_union_us": 8,
                        "internal_idle_gap_us": 2,
                    }
                }
            },
        ],
        "kernel_mean_duration_us": {
            "_quest_score_kernel": {"samples": 2, "mean": 8, "min": 8, "max": 8}
        },
        "sqlite": {"sha256": "0" * 64},
        "main_graph_kernel_inventory": {
            "graph_id": 2,
            "device_id": 0,
            "observed_decode_steps": 2,
            "per_step": [
                {
                    "parent_kernel_envelope_us": 40,
                    "parent_kernel_active_union_us": 31,
                    "parent_kernel_internal_idle_us": 9,
                },
                {
                    "parent_kernel_envelope_us": 44,
                    "parent_kernel_active_union_us": 33,
                    "parent_kernel_internal_idle_us": 11,
                },
            ],
            "kernels": {
                "<unnamed>::wait_kernel(const unsigned long long *)": {
                    "duration_per_invocation_us": {"mean": 24},
                    "invocations_per_decode_step": {"mean": 1},
                    "duration_sum_per_decode_step_us": {"mean": 24},
                },
                "_quest_score_kernel": {
                    "duration_per_invocation_us": {"mean": 8},
                    "invocations_per_decode_step": {"mean": 1},
                    "duration_sum_per_decode_step_us": {"mean": 8},
                },
            },
        },
    }


def test_parent_inventory_rejects_decode_coverage_mismatch():
    profile = parent_graph_profile_fixture()
    profile["main_graph_kernel_inventory"]["per_step"].pop()
    with pytest.raises(ValueError, match="decode-step coverage"):
        REPORT.main_graph_profile_summary(profile)


@pytest.mark.parametrize(
    "effect,interval,expected",
    [
        (4.5, [3.6, 5.4], "下降 4.50%"),
        (-24.5, [-26, -23], "增加 24.50%"),
        (-0.9, [-1.9, 0.1], "区间跨0，本轮未能确认延迟收益或代价"),
        (1.2, None, "95%区间未估计"),
    ],
)
def test_front_page_conclusion_preserves_actual_batch_and_uncertainty(
    effect, interval, expected
):
    row = {
        "launch_gap_reduction_percent": {
            "mean": effect,
            "confidence_interval_95": interval,
        },
        "client_concurrency": 8,
        "actual_batch_size": 4,
        "input_len": 32768,
        "paired_repeats": 3,
        "baseline_gap_ms": {"mean": 10},
        "changed_gap_ms": {"mean": 10 * (1 - effect / 100)},
    }
    text = REPORT.paired_decode_conclusion(row, "Quest 摘要预取")
    assert expected in text
    assert "客户端 C=8" in text and "实际 batch=4" in text
    assert "32,768 tokens" in text and "配对重复 n=3" in text
    assert "不是 HTTP token 到达 TBT" in text


def test_rendered_metric_numa_and_parent_graph_scope_preserve_evidence_boundaries(
    tmp_path, monkeypatch
):
    """Capture actual paragraph inputs; no PDF library or GPU is needed."""
    paragraphs = []

    class Block:
        def __init__(self, *args, **kwargs):
            pass

        def setStyle(self, style):
            pass

        def build(self, story, **kwargs):
            assert story

    def paragraph(text, style):
        paragraphs.append(text)
        return Block()

    colors = SimpleNamespace(HexColor=lambda value: value, white="white")
    modules = {
        "reportlab": SimpleNamespace(),
        "reportlab.lib": SimpleNamespace(colors=colors),
        "reportlab.lib.enums": SimpleNamespace(TA_LEFT=0),
        "reportlab.lib.pagesizes": SimpleNamespace(A4=(595, 842)),
        "reportlab.lib.styles": SimpleNamespace(ParagraphStyle=Block),
        "reportlab.pdfbase": SimpleNamespace(
            pdfmetrics=SimpleNamespace(registerFont=lambda font: None)
        ),
        "reportlab.pdfbase.ttfonts": SimpleNamespace(TTFont=Block),
        "reportlab.platypus": SimpleNamespace(
            Image=Block,
            LongTable=Block,
            PageBreak=Block,
            Paragraph=paragraph,
            SimpleDocTemplate=Block,
            Spacer=Block,
            TableStyle=Block,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    records = [
        {
            "kind": "unexecuted_cli_numa_capacity_adjustment",
            "status": "planned",
            "scope": "Synthetic test receipt: configuration only, not model evidence",
        },
        {
            "kind": "numa_placement_preflight",
            "status": "passed",
            "scope": "Synthetic test receipt: not actual deployment evidence",
        },
    ]
    evidence = [
        (write(tmp_path / f"numa-{index}.json", record), record)
        for index, record in enumerate(records)
    ]
    profile = parent_graph_profile_fixture()
    REPORT.render_report(
        [],
        [
            (
                "bounds-on-demand",
                write(tmp_path / "profile.json", profile),
                profile,
            )
        ],
        evidence,
        tmp_path / "unused.pdf",
        font_path=tmp_path / "unused.ttf",
        report_audit={
            "status": "blocked",
            "model_result_claims_allowed": False,
            "failures": [],
        },
    )
    text = "\n".join(paragraphs)
    assert "TPOT=(请求时延-TTFT)/(输出 tokens-1)" in text
    assert "已扣自身 TTFT" in text and "交错 prefill" in text
    assert "NUMA node 0" in text and "CPU backing ratio 8" in text
    assert "不能计作软件机制收益" in text
    assert "计划 backing 上界与各容量点的实测 allocation 分开记录" in text
    assert "NUMA receipt 和原始 capacity 证据" in text
    assert "unexecuted_cli_numa_capacity_adjustment；状态：planned" in text
    assert "numa_placement_preflight；状态：passed" in text
    assert "完整父 CUDA Graph 归因" in text
    assert "42.00" in text and "32.00" in text and "10.00" in text
    assert "&lt;unnamed&gt;::wait_kernel" in text and "24.00" in text
    assert "每 decode 次数" in text and "每 decode sum μs" in text
    assert "child 图、memcpy、图外 CPU／采样／HTTP 均未覆盖" in text
    assert "不是关键路径或 CE 耗时" in text
    assert "这些次数不是 SASS 静态指令计数" in text
