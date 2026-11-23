"""Four-rank CPU evidence fixtures; never a GPU/DeepEP/model acceptance claim."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = load("four_rank_base", "test/manual/mpu_stages/test_steady_decode.py")
DP = load("four_rank_parallel", "test/manual/mpu_stages/test_attention_dp_ep.py")
M = load("four_rank_control", "benchmark/mpu_stages/multirank.py")
IDLE = load("four_rank_idle", "benchmark/mpu_stages/multirank_stability.py")
MIX = load("four_rank_mixed", "benchmark/mpu_stages/mixed_stability.py")
QUALITY = load("four_rank_quality", "benchmark/mpu_stages/quality_multirank.py")
TASK = load("four_rank_task", "benchmark/mpu_stages/task_quality.py")
CAMPAIGN = load(
    "four_rank_campaign", "benchmark/mpu_stages/real_workload_experiment.py"
)


def fixture():
    result = F.result(("a", "b", "c", "d"), (41,) * 4)
    traces = []
    for rank, rid in enumerate(result["request_ids"]):
        session = F.session() | DP.layout(rank) | {"tp_size": 4, "tp_rank": rank}
        rows = [session]
        for sequence in range(41):
            mode = "extend" if sequence == 0 else "decode"
            record = F.forward(sequence, mode, [rid], [4] if sequence == 0 else None)
            record.update(
                dp_forward_modes=[mode] * 4,
                dp_num_tokens=[4 if sequence == 0 else 1] * 4,
                stream_id=42 + rank,
                launch_monotonic_ns=record["launch_monotonic_ns"] + rank * 1000,
                cuda_seconds=0.005 + rank * 0.001,
                seq_lens=[4 + sequence],
                actual_cuda_graph=sequence > 0,
            )
            rows.append(record)
        rows += [
            {
                "type": "capacity",
                "after_sequence": 40,
                "capacity": {
                    "tp_rank": rank,
                    "tp_size": 4,
                    "used_slots": 0,
                    "reserved_slots": 0,
                }
                | DP.layout(rank),
            },
            {"type": "idle_flush", "last_sequence": 40},
        ]
        traces.append(rows)
    return result, traces


def live(traces):
    return [
        {
            "mpu_forward_progress": row[-3]
            | {
                "attn_dp_rank": rank,
                "tp_rank": rank,
                "attn_dp_size": 4,
                "batch_size": 1,
                "consecutive_stable_decode_steps": 40,
            }
        }
        for rank, row in enumerate(traces)
    ]


def test_four_rank_steady_uses_actual_owner_sets_and_maxima():
    result, traces = fixture()
    report = F.STEADY.analyze(result, traces)
    window = report["windows"][0]
    assert window["actual_batch_size"] == 4
    assert window["actual_local_batch_sizes"] == [1] * 4
    assert window["request_owners"] == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert window["cuda_forward_seconds_over_same_intervals"] == pytest.approx(
        31 * 0.008
    )
    assert len(window["per_rank"]) == 4
    assert window["decode_tokens_over_launch_intervals"] == 124


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate_rank",
        "duplicate_owner",
        "missing_sequence",
        "vector",
        "stream",
    ],
)
def test_incomplete_steady_evidence_rejected(fault):
    result, traces = fixture()
    if fault == "missing_rank":
        traces.pop()
    elif fault == "duplicate_rank":
        traces[3][0]["tp_rank"] = 0
    elif fault == "duplicate_owner":
        traces[3][4]["request_ids"] = ["a"]
    elif fault == "missing_sequence":
        traces[3].pop(4)
    elif fault == "vector":
        traces[3][4]["dp_num_tokens"] = [9] * 4
    else:
        traces[3][4]["stream_id"] = 90
    with pytest.raises(ValueError):
        F.STEADY.analyze(result, traces)


def test_live_rank_identity_and_freshness():
    _, traces = fixture()
    states = live(traces)
    now = states[3]["mpu_forward_progress"]["launch_monotonic_ns"] + 10_000
    observed = M.live_progress(states, now_ns=now)
    assert observed["actual_local_batch_sizes"] == [1] * 4
    assert observed["consecutive_stable_decode_steps"] == 40
    for fault in ("missing", "rank", "sequence", "owner", "stale", "prefill"):
        bad = copy.deepcopy(states)
        if fault == "missing":
            bad.pop()
        elif fault == "rank":
            bad[3]["mpu_forward_progress"]["attn_dp_rank"] = 0
        elif fault == "sequence":
            bad[3]["mpu_forward_progress"]["sequence"] -= 1
        elif fault == "owner":
            bad[3]["mpu_forward_progress"]["request_ids"] = ["a"]
        elif fault == "stale":
            bad[3]["mpu_forward_progress"]["launch_monotonic_ns"] -= 6_000_000_000
        else:
            bad[3]["mpu_forward_progress"]["forward_mode"] = "extend"
        with pytest.raises(ValueError):
            M.live_progress(bad, now_ns=now)


def write_traces(root, traces):
    paths = []
    for rank, rows in enumerate(traces):
        path = root / f"forward-tp{rank}-pp0-dp0-pid{100 + rank}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        paths.append(path)
    return paths


def test_all_rank_idle_recovery_and_ownerless_queued_cancel(tmp_path):
    result, traces = fixture()
    identity = IDLE.trace_identity(write_traces(tmp_path, traces))
    report = IDLE.scan_idle(
        identity, set(result["request_ids"]) | {"queued"}, MIX.scan_idle_trace
    )
    assert len(report["capacity"]["per_rank"]) == 4
    assert report["request_facts"]["queued"]["no_forward_observed"] is True
    assert "queued" not in report["request_owners"]
    traces[3][-2]["capacity"]["used_slots"] = 1
    write_traces(tmp_path, traces)
    with pytest.raises(ValueError, match="remains at idle"):
        IDLE.scan_idle(identity, set(result["request_ids"]), MIX.scan_idle_trace)


@pytest.mark.parametrize("fault", ["migration", "vector", "sequence"])
def test_idle_rejects_partial_or_migrating_ep_evidence(tmp_path, fault):
    result, traces = fixture()
    if fault == "migration":
        traces[3][4]["request_ids"] = ["a"]
    elif fault == "vector":
        traces[3][4]["dp_num_tokens"] = [1, 1, 1, 9]
    else:
        traces[3].pop(4)
    identity = IDLE.trace_identity(write_traces(tmp_path, traces))
    with pytest.raises(ValueError):
        IDLE.scan_idle(identity, set(result["request_ids"]), MIX.scan_idle_trace)


def test_four_compute_and_four_storage_uuid_roles():
    info = {
        "attn_dp_size": 4,
        "base_gpu_id": 0,
        "gpu_id_step": 1,
        "enable_hisparse": True,
        "hisparse_config": {"backing_store": "mpu", "mpu_device_ids": [4, 5, 6, 7]},
    }
    rows = MIX.resolve_gpu_roles(info, lambda i: NS(uuid=f"GPU-fixture-{i}"))
    assert [row["role"] for row in rows] == ["compute"] * 4 + ["storage"] * 4
    assert len(MIX.gpu_identity(rows)) == 8
    info["hisparse_config"]["mpu_device_ids"] = [1, 4, 5, 6]
    with pytest.raises(ValueError):
        MIX.resolve_gpu_roles(info, lambda i: NS(uuid=f"GPU-fixture-{i}"))


def quality_fixture(family, variant):
    stage = json.loads(
        (ROOT / "benchmark/mpu_stages/stage.json").read_text(encoding="utf-8")
    )
    stage = copy.deepcopy(stage)
    stage.update(
        family="deepseek-v4-flash", page_size=256, selector="deepseek-v4-c4-indexer"
    )
    stage["variants"]["cpu"]["hisparse_config"].update(
        top_k=512, device_buffer_size=1024
    )
    if family == "glm-5.3-flash":
        stage.update(family=family, page_size=64, selector="glm53-dsa-kpool-tail")
        stage["variants"]["cpu"]["hisparse_config"].update(
            top_k=2048, device_buffer_size=4096
        )
    config = (
        copy.deepcopy(stage["variants"]["cpu"]["hisparse_config"])
        if variant != "hbm"
        else {}
    )
    if variant == "mpu":
        config.update(
            backing_store="mpu",
            mpu_capacity_ratio=2,
            mpu_device_ids=[4, 5, 6, 7],
            mpu_memory_reserve_bytes=1 << 30,
        )
    states = []
    for rank in range(4):
        states.append(
            dict(
                model_path="/models/native",
                tp_size=4,
                attn_dp_size=4,
                ep_size=4,
                dp_size=1,
                pp_size=1,
                attn_cp_size=1,
                dcp_size=1,
                moe_dp_size=1,
                moe_dense_tp_size=1,
                enable_dp_lm_head=True,
                nnodes=1,
                moe_a2a_backend="deepep",
                disable_overlap_schedule=True,
                enable_prefill_cp=False,
                enable_two_batch_overlap=False,
                enable_cp_tp_group_sharing=False,
                disable_radix_cache=True,
                enable_hierarchical_cache=False,
                allow_auto_truncate=False,
                disaggregation_mode="null",
                speculative_algorithm=None,
                enable_mixed_chunk=False,
                page_size=stage["page_size"],
                chunked_prefill_size=4096,
                max_total_tokens=131072,
                prefill_max_requests=1,
                dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                attention_backend="dsv4",
                dsv4_attn_backend="flashmla",
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_sparse",
                base_gpu_id=0,
                hisparse_config=config,
                enable_hisparse=variant != "hbm",
                mpu_parallel_metadata=DP.layout(rank),
                mpu_selector_runtime={
                    "index_topk": config.get(
                        "top_k", stage["variants"]["cpu"]["hisparse_config"]["top_k"]
                    )
                },
            )
        )
    return stage, {"internal_states": states, "server_args": states[0]}


@pytest.mark.parametrize("family", ["deepseek-v4-flash", "glm-5.3-flash"])
def test_model_native_quality_preserves_indexer_and_rank_audit(family):
    controls = []
    for variant in ("hbm", "cpu", "mpu"):
        stage, info = quality_fixture(family, variant)
        checked = QUALITY.policy(
            stage,
            info,
            variant,
            {"max_total_tokens": 131072},
            {"model_path": "/models/native"},
            capacity_ratio=2,
            top_k=stage["variants"]["cpu"]["hisparse_config"]["top_k"],
            mpu_device_ids=[4, 5, 6, 7],
            mpu_memory_reserve_bytes=1 << 30,
            compute_device_id=0,
        )
        controls.append(QUALITY.controls(checked, variant))
    assert controls[0]["selector"] == controls[1]["selector"] == controls[2]["selector"]
    assert controls[0]["common"] == controls[1]["common"] == controls[2]["common"]
    info["internal_states"].pop()
    with pytest.raises(ValueError):
        QUALITY.policy(
            stage,
            info,
            "dense",
            {"max_total_tokens": 131072},
            {"model_path": "/models/native"},
            capacity_ratio=2,
            top_k=512,
            mpu_device_ids=[4, 5, 6, 7],
            mpu_memory_reserve_bytes=1 << 30,
            compute_device_id=0,
        )


def test_four_rank_campaign_profiles_all_attention_owners(tmp_path):
    stage, args = CAMPAIGN.run.STAGE.parse_args(["--output-dir", str(tmp_path)])
    args.workload, args.workload_sha256 = tmp_path / "frozen.json", "0" * 64
    jobs = CAMPAIGN.plan(stage, args, tmp_path / "model.json", True)
    profiles = [job for job in jobs if job["id"].startswith("profile-")]
    assert profiles
    assert all(
        CAMPAIGN.option_values(job["command"], "--profile-concurrency") == ["4"]
        for job in profiles
    )
    assert all(
        CAMPAIGN.option_values(job["command"], "--profile-stable-steps") == ["16"]
        for job in profiles
    )


def test_runtime_records_cpu_seq_lengths_and_actual_worker_graph_result():
    timer = (ROOT / "python/sglang/srt/observability/mpu_stage_timing.py").read_text(
        encoding="utf-8"
    )
    scheduler = (ROOT / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "batch.seq_lens_cpu.tolist()" in timer
    assert scheduler.count('mpu_forward_record["actual_cuda_graph"] = bool(') == 2
    wrapper = (ROOT / "python/sglang/bench_serving.py").read_text(encoding="utf-8")
    assert "sglang.benchmark.serving" in wrapper
    ast.parse(scheduler)
