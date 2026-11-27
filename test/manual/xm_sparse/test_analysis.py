"""Evidence-replay fixtures test accounting guards, never CUDA performance."""

import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("xm_analysis_under_test", "benchmark/xm_sparse/analyze.py")
FIXTURES = load(
    "xm_native_steady_fixtures", "test/manual/mpu_stages/test_steady_decode.py"
)
NATIVE, STEADY = A.load("analyze"), A.load("steady_decode")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def capacity():
    return {
        "gpu_memory_ledger": {
            "contract": "qwen_quest_v1",
            "named_compute_bytes": 192,
            "named_compute_shapes": {
                "fixed.quest.page_k_bounds": {
                    "allocation_group": 0,
                    "storage_bytes": 128,
                },
                "fixed.quest.page_k_min": {"allocation_group": 0, "storage_bytes": 128},
                "kv.k[0]": {"allocation_group": 1, "storage_bytes": 64},
            },
        },
        "allocated_backing_kv_bytes": 512,
        "allocated_backing_quest_summary_bytes": 0,
    }


def campaign_fixture(tmp_path):
    base = tmp_path / "baseline-r0" / "evidence"
    raw_result = FIXTURES.result()
    trace = FIXTURES.trace()
    raw_analysis, raw_steady = (
        NATIVE.analyze(raw_result, [trace]),
        STEADY.analyze(raw_result, [trace]),
    )
    result_path = base / "cpu-in4-ratio2-c2-r0.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(raw_result) + "\n", encoding="utf-8")
    trace_dir = base / "cpu-in4-ratio2-trace"
    trace_dir.mkdir()
    (trace_dir / "forward-0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trace) + "\n", encoding="utf-8"
    )
    cid = "cpu-in4-ratio2-c2-r0"
    case = {
        "id": cid,
        "concurrency": 2,
        "repeat": 0,
        "status": "success",
        "command": [
            "--dataset-name",
            "frozen-ids",
            "--tokenize-prompt",
            "--frozen-ids-sha256",
            "d" * 64,
        ],
        "result_path": "/results/whatever/" + result_path.name,
        "analysis": raw_analysis,
        "steady_decode": raw_steady,
        "idle_capacity": [capacity()],
    }
    stage = {"xm_adaptation": {"arm": "baseline"}}
    native = {
        "status": "success",
        "kind": "benchmark",
        "stage": stage,
        "runtime": {"git_commit": "c" * 40},
        "settings": {
            "execute": True,
            "workload_sha256": "b" * 64,
            "num_prompts": 2,
            "output_len": 41,
        },
        "workload": {"files": {"4": {"sha256": "d" * 64}}},
        "groups": [
            {
                "status": "success",
                "variant": "cpu",
                "input_len": 4,
                "decode_cuda_graph_enabled": True,
                "trace_dir": "/results/whatever/" + trace_dir.name,
                "capacity": [capacity()],
                "cases": [case],
            }
        ],
    }
    write(base / "manifest.json", native)
    write(base / (cid + "-steady.json"), raw_steady)
    write(base / (cid + "-timing.json"), raw_analysis)
    campaign = {
        "schema_version": 1,
        "status": "success",
        "phase": "performance",
        "source_fingerprint": {"files": 1, "sha256": "a" * 64},
        "workload_identity": {"sha256": "b" * 64, "bytes": 1},
        "settings": {},
        "cases": [
            {
                "id": "baseline-r0",
                "arm": "baseline",
                "global_repeat": 0,
                "status": "success",
                "stage": stage,
            }
        ],
    }
    write(tmp_path / "campaign.json", campaign)
    return tmp_path / "campaign.json", base, cid


def test_complete_raw_replay_records_hashes_and_observed_batch(tmp_path):
    campaign, _, _ = campaign_fixture(tmp_path)
    report = A.analyze_campaign(campaign)
    assert report["status"] == "analyzed"
    assert report["samples"][0]["eligible_stable_intervals"] == 31
    assert report["observed_stable_capacity_points"][0]["actual_batch_size"] == 2
    assert len(report["artifacts"]) >= 5
    assert all(len(a["sha256"]) == 64 for a in report["artifacts"])
    assert report["native_source_commits"] == ["c" * 40]
    assert report["rows"][0]["actual_batch"] == 2
    assert report["rows"][0]["mean_tbt_ms"] == pytest.approx(10)


def test_saved_derived_timing_cannot_override_raw_evidence(tmp_path):
    campaign, base, cid = campaign_fixture(tmp_path)
    path = base / (cid + "-steady.json")
    report = A.read(path)
    report["windows"][0]["scheduler_launch_gap_ms"]["mean"] *= 0.1
    write(path, report)
    with pytest.raises(ValueError, match="differs from raw replay"):
        A.analyze_campaign(campaign)


def test_missing_raw_forward_is_rejected(tmp_path):
    campaign, base, _ = campaign_fixture(tmp_path)
    path = base / "cpu-in4-ratio2-trace" / "forward-0.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [r for r in rows if r.get("sequence") != 5]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="missing, reordered"):
        A.analyze_campaign(campaign)


def sample(arm, *, row_ids=(0, 1), intervals=20, gap=10, client=8):
    return {
        "arm": arm,
        "input_len": 32768,
        "client_concurrency": client,
        "repeat": 0,
        "graph_enabled": True,
        "_intervals": {
            (tuple(row_ids), tuple([i + 1] * len(row_ids))): {
                "gap_ms": gap,
                "actual_batch_size": len(row_ids),
            }
            for i in range(intervals)
        },
    }


def test_same_batch_comparison_matches_requests_and_decode_ordinals():
    rows, aggregates = A.same_batch_pairs([sample("baseline"), sample("fused", gap=8)])
    assert rows[0]["actual_batch_size"] == 2
    assert rows[0]["client_concurrency"] == 8
    assert rows[0]["launch_gap_reduction_percent"] == 20
    assert aggregates[0]["paired_repeats"] == 1
    assert (
        aggregates[0]["launch_gap_reduction_percent"]["confidence_interval_95"] is None
    )
    rows, aggregates = A.same_batch_pairs(
        [sample("baseline"), sample("fused", row_ids=(4, 5))]
    )
    assert rows == aggregates == []


def test_too_short_matched_windows_do_not_form_a_gain_claim():
    rows, aggregates = A.same_batch_pairs(
        [sample("baseline", intervals=10), sample("fused", intervals=10)]
    )
    assert rows[0]["eligible"] is False
    assert aggregates == []


def test_alias_and_workspace_storage_are_not_double_counted():
    cap = capacity()
    cap["xm_fused_backend_memory"] = {
        "retained_gpu_pool_bytes": 112,
        "retained_gpu_pool_shapes": {
            "plan.device_locs": {"allocation_group": 0, "storage_bytes": 16},
            "split_partial_workspace": {"allocation_group": 1, "storage_bytes": 64},
            "attention_output": {"allocation_group": 2, "storage_bytes": 32},
        },
    }
    result = A.memory_snapshot(cap)
    assert result["cache_components"]["quest_resident_summary_bytes"] == 128
    assert result["torch_named_total_including_observed_backend_bytes"] == 192 + 64 + 32
    assert "fused_plan_bytes" not in result["backend_components"]
    cap["xm_fused_backend_memory"] = {
        "retained_gpu_pool_bytes": 384 << 20,
        "retained_gpu_pool_shapes": {
            "flashinfer_workspace": {"allocation_group": 0, "storage_bytes": 384 << 20}
        },
    }
    result = A.memory_snapshot(cap)
    assert (
        result["backend_components"]["separate_attention_workspace_bytes"] == 384 << 20
    )


def test_counter_deltas_are_diagnostic_and_never_hardware_payload():
    def snapshot(count):
        return {
            "xm_quest_summary_offload": {
                "mechanism": "gpu_graph",
                "counters": {
                    "completed_bytes": count,
                    "wait_failures": 0,
                    "device_launch_failures": 0,
                },
            }
        }

    result = A.counter_delta(snapshot(100), snapshot(150))
    assert result["delta"]["completed_bytes"] == 50
    assert result["hardware_copy_engine_bytes_measured"] is False
    assert "includes bench_serving warmup" in result["scope"]
    with pytest.raises(ValueError, match="decreased"):
        A.counter_delta(snapshot(150), snapshot(100))


def profile_fixture(tmp_path, *, missing_attention=False):
    path = tmp_path / "decode.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,correlationId INTEGER,graphId INTEGER,demangledName INTEGER);"
    )
    names = {
        1: "_quest_score_kernel",
        2: "selection_aux",
        3: "load_cache_to_device_buffer_kernel",
        4: "xm_hisparse_tma::attention_kernel",
        5: "xm_hisparse_tma::merge_kernel",
        6: "child_checkpoint_kernel",
    }
    conn.executemany("INSERT INTO StringIds VALUES (?,?)", names.items())
    for correlation in (1, 2):
        for layer in range(48):
            start = correlation * 10000 + layer * 100
            values = [(0, 10, 1), (20, 30, 2), (40, 50, 3), (60, 80, 4), (85, 90, 5)]
            for left, right, name in values:
                if missing_attention and layer == 23 and name in (4, 5):
                    continue
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                    (start + left, start + right, 0, 1, correlation, 7, name),
                )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        (10000, 40000, 0, 9, 0, 8, 6),
    )
    conn.commit()
    conn.close()
    scope = {
        "status": "passed",
        "actual_decode_scope_verified": True,
        "sqlite_sha256": A.identity(path)["sha256"],
        "calls_per_graph": 48,
        "graph_id": 7,
        "actual_decode_graph_launches": 2,
        "actual_batch_size": 1,
    }
    return path, scope


def test_attention_path_includes_idle_gaps_and_excludes_unrelated_child_graph(tmp_path):
    path, scope = profile_fixture(tmp_path)
    result = A.attention_profile(path, scope)
    assert result["total_path_span_us"]["mean"] == pytest.approx(0.09)
    assert result["stage_span_us"]["attention_including_merge"][
        "mean"
    ] == pytest.approx(0.03)
    assert result["per_step"][0]["stages"]["attention_including_merge"][
        "internal_idle_gap_us"
    ] == pytest.approx(0.005)
    assert "child_checkpoint_kernel" not in result["kernel_mean_duration_us"]


def test_incomplete_profile_does_not_manufacture_an_attention_breakdown(tmp_path):
    path, scope = profile_fixture(tmp_path, missing_attention=True)
    with pytest.raises(ValueError, match="missing cache plan or attention"):
        A.attention_profile(path, scope)
    with pytest.raises(ValueError, match="Passed decode scope"):
        A.attention_profile(path, {**scope, "status": "planned"})


def test_profile_checksum_guard(tmp_path):
    path, scope = profile_fixture(tmp_path)
    with pytest.raises(ValueError, match="differs from verified scope"):
        A.attention_profile(path, {**scope, "sqlite_sha256": "0" * 64})
