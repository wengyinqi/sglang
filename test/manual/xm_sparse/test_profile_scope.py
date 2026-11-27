"""Synthetic scope fixtures exercise rejection boundaries, never real GPU proof."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "xm_profile_scope_test", ROOT / "benchmark/xm_sparse/profile_scope.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

GPU_UUID = "318d5f6b-cc4d-6350-e58f-2c93e73b02cf"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def identity(path):
    return {"path": str(path), "sha256": S.sha(path), "bytes": path.stat().st_size}


@pytest.fixture
def scope_fixture(tmp_path):
    def create(arm="fused"):
        source_root = tmp_path / arm / "source"
        source = source_root / "python/sglang/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_text("# Synthetic source identity only\n")
        marker_path = source_root / ".mpu-source.json"
        marker = {
            "provenance": "git_archive",
            "source_commit": "c" * 40,
            "source_archive_sha256": "a" * 64,
        }
        write(marker_path, marker)
        fingerprint = S.core_fingerprint(source_root)
        stage = S.CAMPAIGN.recipe(arm)
        runtime = {
            "git_commit": "c" * 40,
            "archived_source": marker,
            "git_commit_source": "archive_build_marker",
            "git_status": "unavailable: archived source has no Git index",
            "sglang_source": "/workspace/python/sglang/__init__.py",
            "gpu": "NVIDIA H800 PCIe, GPU-" + GPU_UUID + ", 81085 MiB, 580.0",
        }
        native_path = source_root.parent / "native-correctness.json"
        groups = [
            {
                "variant": variant,
                "eager": eager,
                "status": "success",
                "lifecycle_required": True,
                "lifecycle_results": [
                    {"case": name, "status": "passed"}
                    for name in ("abort_prefill", "abort_decode", "retract_decode")
                ],
            }
            for variant in ("cpu", "hbm")
            for eager in (True, False)
        ]
        native = {
            "status": "success",
            "stage": stage,
            "runtime": runtime,
            "groups": groups,
            "settings": {
                "model": "/models/qwen",
                "workload": "/results/workload.json",
                "workload_sha256": "b" * 64,
                "disable_cuda_graph": False,
            },
        }
        write(native_path, native)
        proof_path = source_root.parent / "correctness-campaign.json"
        proof = {
            "phase": "correctness",
            "status": "success",
            "source_fingerprint": fingerprint,
            "workload_identity": {"sha256": "b" * 64},
            "settings": {"model": "/models/qwen", "input_lens": [16384]},
            "cases": [
                {"arm": arm, "stage": stage, "status": "success", "returncode": 0}
            ],
        }
        write(proof_path, proof)
        step = {
            "id": "correctness",
            "kind": "model",
            "status": "passed",
            "expected_model_stage": stage,
            "commands": [{"report": str(native_path)}],
            "results": [{"status": "passed"}],
        }
        acceptance_path = source_root.parent / "acceptance.json"
        receipt = {
            "status": "passed_correctness_only",
            "stage": stage,
            "runtime": runtime,
            "steps": [step],
            "correctness_campaign": identity(proof_path),
            "evidence_scope": {"target_model_correctness_executed": True},
        }
        write(acceptance_path, receipt)
        path = source_root.parent / "decode.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,correlationId INTEGER,graphId INTEGER,demangledName INTEGER);
            CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT,name TEXT,busLocation TEXT,computeMajor INTEGER,computeMinor INTEGER);
        """)
        fused = S.CAMPAIGN.ARMS[arm][0]
        names = {
            1: "cudaGraphLaunch_v11000",
            2: "_quest_score_kernel",
            3: (
                "void xm_hisparse_tma::attention_kernel<2048, 32, 8>()"
                if fused
                else S.BASE_SCOPE.DECODE_ATTENTION_KERNEL
            ),
            4: "void xm_hisparse_tma::merge_kernel<32>()",
            5: "plan_cache_kernel",
            6: "child_checkpoint_kernel",
        }
        conn.executemany("INSERT INTO StringIds VALUES (?,?)", names.items())
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU VALUES (3,?,'NVIDIA H800 PCIe','0000:59:00.0',9,0)",
            (GPU_UUID,),
        )
        for correlation in (1, 2):
            conn.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,1)",
                (correlation * 10000, correlation * 10000 + 1, correlation),
            )
            for layer in range(48):
                base = correlation * 10000 + layer * 100
                for name, offset, duration in [(2, 0, 10), (5, 20, 10), (3, 40, 20)] + (
                    [(4, 70, 10)] if fused else []
                ):
                    conn.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                        (
                            base + offset,
                            base + offset + duration,
                            3,
                            7,
                            correlation,
                            11,
                            name,
                        ),
                    )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10001,20005,3,9,0,12,6)"
        )
        conn.commit()
        conn.close()
        ids = ["request-a", "request-b"]
        trace = [
            {
                "type": "session",
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 1,
                "disable_overlap_schedule": True,
                "disaggregation_mode": "null",
                "disable_radix_cache": True,
                "speculative_algorithm": None,
                "enable_mixed_chunk": False,
            }
        ]
        for sequence in range(101):
            trace.append(
                {
                    "type": "forward",
                    "sequence": sequence,
                    "forward_mode": "extend" if sequence == 0 else "decode",
                    "extend_lens": [16384, 16384] if sequence == 0 else None,
                    "success": True,
                    "request_ids": ids,
                    "stream_id": 111,
                    "launch_monotonic_ns": (sequence + 1) * 1000000,
                }
            )
        trace.append({"type": "idle_flush", "last_sequence": 100})
        trace_path = source_root.parent / "forward-0.jsonl"
        trace_path.write_text(
            "\n".join(json.dumps(row) for row in trace), encoding="utf-8"
        )
        profile = {
            "status": "captured_pending_role_analysis",
            "stage": stage,
            "runtime": runtime,
            "profile_variant": "cpu",
            "acceptance": identity(acceptance_path),
            "correctness_artifact": identity(native_path),
            "artifacts": [identity(path)],
            "configured_decode_steps": 2,
            "profile_request": {
                "activities": ["CUDA_PROFILER"],
                "num_steps": 2,
                "profile_by_stage": True,
                "profile_stages": ["decode"],
            },
            "server_command": [
                "--capture-range=cudaProfilerApi",
                "--cuda-graph-trace=node",
                "--hisparse-config",
                json.dumps(stage["variants"]["cpu"]["hisparse_config"]),
            ],
            "workload": {
                "input_len": 16384,
                "output_len": 101,
                "concurrency": 2,
                "manifest": "/results/workload.json",
                "manifest_sha256": "b" * 64,
            },
            "completion_tokens": 202,
            "profiled_request_ids": ids,
            "stable_trigger": {
                "min_consecutive_decode_steps": 16,
                "required_actual_batch_size": 2,
            },
            "devices": [
                {"visible_ordinal": 0, "uuid": "GPU-" + GPU_UUID, "role": "compute"}
            ],
            "physical_gpu_inventory": "index, uuid, pci.bus_id, name\n0, GPU-"
            + GPU_UUID
            + ", 00000000:59:00.0, NVIDIA H800 PCIe",
        }
        for field, seq in (
            ("trigger_progress_before", 16),
            ("trigger_progress_after_ack", 17),
        ):
            profile[field] = dict(
                trace[seq + 1], consecutive_stable_decode_steps=seq, batch_size=len(ids)
            )
        profile_path = source_root.parent / "profile.json"
        write(profile_path, profile)
        return {
            "database": path,
            "profile_path": profile_path,
            "trace_path": trace_path,
            "correctness_campaign": proof_path,
            "acceptance": acceptance_path,
            "native_correctness": native_path,
            "archive_marker": marker_path,
            "source_root": source_root,
        }

    return create


@pytest.mark.parametrize("arm", ["baseline", "fused", "combined-protected"])
def test_fixture_binds_main_graph_uuid_and_exact_arm(scope_fixture, arm):
    args = scope_fixture(arm)
    report = S.verify(**args)
    assert report["status"] == "passed"
    assert report["calls_per_graph"] == 48
    assert report["actual_batch_size"] == 2
    assert report["actual_decode_graph_launches"] == 2
    assert report["device_role"]["trace_device_id"] == 3
    assert report["device_role"]["mapping_verified_automatically"] is True
    assert report["graph_id"] == 11
    assert report["performance_result"] is False
    assert report["uses_author_flashmla_implementation"] is False
    if arm != "baseline":
        assert "scalar FP32" in report["attention_path"]
        assert report["fused_merge_calls"] == 96


def reseal_sqlite_fixture(args):
    profile = S.read(args["profile_path"])
    profile["artifacts"] = [identity(args["database"])]
    write(args["profile_path"], profile)


def test_mma_symbols_identify_path_without_claiming_native_instructions(scope_fixture):
    args = scope_fixture()
    with sqlite3.connect(args["database"]) as conn:
        conn.execute(
            "UPDATE StringIds SET value=replace(value, 'xm_hisparse_tma::', 'xm_hisparse_tma_mma::')"
        )
    reseal_sqlite_fixture(args)
    report = S.verify(**args)
    assert report["observed_fused_implementation"] == "bf16_mma"
    assert report["native_instruction_validation_included"] is False
    assert "native TMA/MMA instructions require separate" in report["attention_path"]


def test_mixed_fusion_implementations_are_rejected(scope_fixture):
    args = scope_fixture()
    with sqlite3.connect(args["database"]) as conn:
        conn.execute(
            "UPDATE StringIds SET value=replace(value, 'xm_hisparse_tma::', 'xm_hisparse_tma_mma::') WHERE id=3"
        )
    reseal_sqlite_fixture(args)
    with pytest.raises(ValueError, match="Mixed or missing fused"):
        S.verify(**args)


def test_fused_profile_cannot_hide_an_extra_reference_attention(scope_fixture):
    args = scope_fixture()
    with sqlite3.connect(args["database"]) as conn:
        conn.execute(
            "INSERT INTO StringIds VALUES (20, ?)",
            (S.BASE_SCOPE.DECODE_ATTENTION_KERNEL,),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10050,10060,3,7,1,11,20)"
        )
    reseal_sqlite_fixture(args)
    with pytest.raises(ValueError, match="prefill kernel|reference/fallback"):
        S.verify(**args)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("source", "actual core fingerprint"),
        ("marker", "archive identity"),
        ("role", "exactly one captured compute"),
        ("uuid", "map uniquely"),
        ("recipe", "exact arm recipe"),
        ("command", "declared arm mechanism"),
        ("pending", "actual completed"),
        ("trigger", "metadata does not match"),
        ("native", "native correctness artifact identity"),
    ],
)
def test_unbound_source_role_recipe_or_trigger_never_passes(
    scope_fixture, mutation, pattern
):
    args = scope_fixture()
    profile = S.read(args["profile_path"])
    if mutation == "source":
        (args["source_root"] / "python/sglang/fixture.py").write_text("changed")
    elif mutation == "marker":
        profile["runtime"]["archived_source"]["source_archive_sha256"] = "d" * 64
    elif mutation == "role":
        profile["devices"][0]["role"] = "storage"
    elif mutation == "uuid":
        profile["devices"][0]["uuid"] = (
            "d" * 8 + "-" + "d" * 4 + "-" + "d" * 4 + "-" + "d" * 4 + "-" + "d" * 12
        )
    elif mutation == "recipe":
        profile["stage"]["xm_adaptation"]["arm"] = "baseline"
    elif mutation == "command":
        config = json.loads(profile["server_command"][-1])
        config["xm_fused_retrieval_attention"] = False
        profile["server_command"][-1] = json.dumps(config)
    elif mutation == "pending":
        profile["status"] = "planned"
    elif mutation == "trigger":
        profile["trigger_progress_before"]["launch_monotonic_ns"] += 1
    elif mutation == "native":
        native = S.read(args["native_correctness"])
        native["status"] = "running"
        write(args["native_correctness"], native)
    write(args["profile_path"], profile)
    with pytest.raises(ValueError, match=pattern):
        S.verify(**args)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("merge", "Missing layer"),
        ("child_attention", "unobserved main runtime"),
        ("duplicate_runtime", "runtime count"),
        ("device", "other-role device"),
        ("prefill", "genuine prefill"),
        ("layer_order", "exactly one attention"),
    ],
)
def test_incomplete_or_wrong_graph_activity_never_passes(
    scope_fixture, mutation, pattern
):
    args = scope_fixture()
    with sqlite3.connect(args["database"]) as conn:
        if mutation == "merge":
            conn.execute(
                "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=4 AND start=12370"
            )
        elif mutation == "child_attention":
            conn.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=0,graphId=12 WHERE demangledName=3 AND start=12340"
            )
        elif mutation == "duplicate_runtime":
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,2,1,1)")
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(3,4,1,1)")
        elif mutation == "device":
            conn.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=0 WHERE demangledName=3"
            )
        elif mutation == "prefill":
            conn.execute(
                "UPDATE StringIds SET value='genuine_prefill_kernel' WHERE id=6"
            )
        elif mutation == "layer_order":
            conn.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=12445,end=12465 WHERE demangledName=3 AND start=12340"
            )
    profile = S.read(args["profile_path"])
    profile["artifacts"] = [identity(args["database"])]
    write(args["profile_path"], profile)
    with pytest.raises(ValueError, match=pattern):
        S.verify(**args)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing", "Incomplete or reordered"),
        ("nonmonotonic", "Non-monotonic"),
        ("membership", "Profile decode output coverage"),
        ("batch", "Actual batch"),
    ],
)
def test_raw_forward_trace_boundaries_are_required(scope_fixture, mutation, pattern):
    args = scope_fixture()
    trace = [json.loads(line) for line in args["trace_path"].read_text().splitlines()]
    if mutation == "missing":
        del trace[22]
    elif mutation == "nonmonotonic":
        trace[50]["launch_monotonic_ns"] = trace[49]["launch_monotonic_ns"]
    elif mutation == "membership":
        trace[20]["request_ids"] = ["request-a"]
    elif mutation == "batch":
        trace[50]["batch_size"] = 3
    args["trace_path"].write_text("\n".join(json.dumps(row) for row in trace))
    with pytest.raises(ValueError, match=pattern):
        S.verify(**args)


def test_rejected_cli_scope_emits_failure_and_never_pass(scope_fixture, tmp_path):
    args = scope_fixture()
    profile = S.read(args["profile_path"])
    profile["status"] = "planned"
    write(args["profile_path"], profile)
    output = tmp_path / "failed-scope.json"
    argv = []
    for field, option in (
        ("database", "sqlite"),
        ("profile_path", "profile"),
        ("trace_path", "trace"),
        ("correctness_campaign", "correctness-campaign"),
        ("acceptance", "acceptance"),
        ("native_correctness", "native-correctness"),
        ("archive_marker", "archive-marker"),
        ("source_root", "source-root"),
    ):
        argv += ["--" + option, str(args[field])]
    assert S.main(argv + ["--output", str(output)]) == 2
    report = S.read(output)
    assert report["status"] == "failed_scope_verification"
    assert report["actual_decode_scope_verified"] is False
