"""CPU regression for disjoint DP request ownership and synchronized EP steps."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIX = load("dp_fixture", "test/manual/mpu_stages/test_stage_tools.py")
ANALYZE = FIX.RUN.ANALYZE
PARALLEL = load("mpu_parallel_cpu", "python/sglang/srt/observability/mpu_parallel.py")


def layout(rank):
    return dict(
        attn_dp_size=4,
        attn_dp_rank=rank,
        attn_tp_size=1,
        ep_size=4,
        moe_tp_size=1,
        parallel_strategy="attention_dp4_ep4",
        tensor_parallel_audit={"status": "passed", "checked_modules": ["lm_head"]},
    )


def dp_fixture():
    # Rank 0 owns a, rank 1 owns b; ranks 2/3 have no requests but execute EP.
    modes = [
        ["extend", "idle", "idle", "idle"],
        ["decode", "extend", "idle", "idle"],
        ["decode", "decode", "idle", "idle"],
        ["idle", "decode", "idle", "idle"],
    ]
    tokens = [[5, 0, 0, 0], [1, 3, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]]
    traces = []
    for rank in range(4):
        trace = [FIX.session(rank, 4) | layout(rank)]
        for step, vector in enumerate(modes):
            mode = vector[rank]
            ids = [] if mode == "idle" else ["a" if rank == 0 else "b"]
            trace.append(
                dict(
                    type="forward",
                    sequence=step,
                    forward_mode=mode,
                    request_ids=ids,
                    extend_lens=[tokens[step][rank]] if mode == "extend" else None,
                    cuda_seconds=1.0 + rank / 10,
                    stream_id=rank + 1,
                    success=True,
                    dp_forward_modes=vector[:],
                    dp_num_tokens=tokens[step][:],
                )
            )
        trace += [
            {
                "type": "capacity",
                "after_sequence": 3,
                "capacity": FIX.capacity(rank, 4) | layout(rank),
            },
            {"type": "idle_flush", "last_sequence": 3},
        ]
        traces.append(trace)
    return FIX.benchmark(), traces


def test_disjoint_owners_and_idle_ep_participants():
    result, traces = dp_fixture()
    report = ANALYZE.analyze(result, traces)
    assert report["cuda_trace"]["request_owners"] == {"a": 0, "b": 1}
    assert report["cuda_trace"]["prefill_forward_seconds"] == pytest.approx(1.3)
    assert report["cuda_trace"]["mixed_forward_seconds"] == pytest.approx(1.3)
    assert report["decode_only_estimate"]["residual_seconds"] == pytest.approx(18.7)
    assert len(ANALYZE.read_capacities(traces, 4, ["a", "b"], require_idle=True)) == 4


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate_owner",
        "missing_step",
        "wrong_vector",
        "stale_idle",
        "tp",
        "tokens",
        "mixed_ids",
    ],
)
def test_bad_dp_evidence_is_rejected(fault):
    result, traces = dp_fixture()
    if fault == "missing_rank":
        traces.pop()
    elif fault == "duplicate_owner":
        traces[2][1]["request_ids"] = ["a"]
    elif fault == "missing_step":
        traces[2].pop(2)
    elif fault == "wrong_vector":
        traces[2][2]["dp_forward_modes"][0] = "extend"
    elif fault == "stale_idle":
        traces[2][-1]["last_sequence"] = -1
    elif fault == "tp":
        traces[2][0]["attn_tp_size"] = 4
    elif fault == "tokens":
        traces[0][1]["extend_lens"] = [4]
    else:
        traces[0][1]["request_ids"].append("unrelated")
    with pytest.raises(ValueError):
        ANALYZE.analyze(result, traces)


def test_old_tp_capacity_cannot_satisfy_dp_recipe():
    stage = FIX.recipe(tp=4) | {"attn_dp_size": 4, "ep_size": 4}
    args = NS(
        max_total_tokens=4096,
        cpu_capacity_ratio=2,
        top_k=None,
        device_buffer_size=None,
        mpu_device_ids=None,
        mpu_memory_reserve_bytes=None,
    )
    with pytest.raises(ValueError, match="topology"):
        FIX.RUN.verify_capacities(
            [FIX.capacity(i, 4) for i in range(4)], stage, args, "mpu", 2
        )


def test_dp_stage_commands_and_coverage():
    stage = FIX.recipe(tp=4) | {
        "attn_dp_size": 4,
        "ep_size": 4,
        "graph_max_bs_flag": "--cuda-graph-max-bs-decode",
    }
    _, args = FIX.TestStagePlanning().parse(stage, correctness=True)
    args.max_total_tokens = 16384
    groups = FIX.CORRECT.make_plan(stage, args)
    for group in groups:
        command = group["server_command"]
        for key, value in (
            ("--attn-dp-size", "4"),
            ("--ep-size", "4"),
            ("--dp-size", "1"),
            ("--moe-a2a-backend", "deepep"),
            ("--moe-dense-tp-size", "1"),
        ):
            assert command[command.index(key) + 1] == value
        assert "--enable-dp-lm-head" in command
        assert all(len(row) >= 4 for row in group["rounds"])


def test_partition_audit_catches_hidden_lm_head_tp():
    VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})
    head = VocabParallelEmbedding()
    model = NS(named_modules=lambda: [("lm_head", head)])
    head.tp_size = 4
    with pytest.raises(ValueError, match="lm_head"):
        PARALLEL.audit_model_tensor_parallelism(model)
    head.tp_size = 1
    assert PARALLEL.audit_model_tensor_parallelism(model)["status"] == "passed"


def test_single_compute_parallelism_is_unchanged():
    result = PARALLEL.parallel_metadata(NS(tp_size=1), 0)
    assert result["parallel_strategy"] == "single_compute"
    assert result["attn_tp_size"] == result["ep_size"] == 1


def test_resolved_dp_ep_contract_rejects_residual_tp_and_overlap():
    cfg = NS(
        tp_size=4,
        attn_dp_size=4,
        ep_size=4,
        dp_size=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        moe_dense_tp_size=1,
        nnodes=1,
        enable_dp_lm_head=True,
        disable_overlap_schedule=True,
        enable_prefill_cp=False,
        enable_two_batch_overlap=False,
        enable_cp_tp_group_sharing=False,
        moe_a2a_backend="deepep",
    )
    assert PARALLEL.parallel_metadata(cfg, 3)["attn_dp_rank"] == 3
    for name, value in (
        ("attn_dp_size", 1),
        ("ep_size", 1),
        ("moe_dense_tp_size", 4),
        ("enable_dp_lm_head", False),
        ("enable_cp_tp_group_sharing", True),
        ("disable_overlap_schedule", False),
        ("moe_a2a_backend", "none"),
    ):
        with pytest.raises(ValueError):
            PARALLEL.parallel_metadata(NS(**(vars(cfg) | {name: value})), 0)


def test_corrected_repo_recipe_does_not_silently_use_tp():
    import json

    stage = json.loads((ROOT / "benchmark/mpu_stages/stage.json").read_text())
    if stage["tp_size"] == 4:
        assert stage["attn_dp_size"] == stage["ep_size"] == 4
        assert "marlin" not in stage["server_args"]
        assert "deep_gemm" in stage["server_args"]
    else:
        assert stage.get("attn_dp_size", 1) == stage.get("ep_size", 1) == 1


def test_fault_follows_one_owner_and_recovery_can_pin_that_owner():
    lifecycle = load("dp_lifecycle", "benchmark/mpu_stages/lifecycle.py")
    _, traces = dp_fixture()
    facts = lifecycle.forward_facts(traces, "a", 4, attn_dp_size=4)
    assert len(facts) == 1 and facts[0]["rank"] == 0
    assert lifecycle.request_payload([1], 4, "recovery", owner=2)["routed_dp_rank"] == 2
    traces[2][1] = copy.deepcopy(traces[0][1])
    with pytest.raises(ValueError, match="exactly one"):
        lifecycle.forward_facts(traces, "a", 4, attn_dp_size=4)


def test_timer_keeps_gather_metadata_for_idle_ep_participant(tmp_path):
    event = NS(
        record=lambda stream: None,
        query=lambda: True,
        synchronize=lambda: None,
        elapsed_time=lambda other: 10,
    )
    timer = FIX.TIMER.MPUStageForwardTimer(
        tmp_path,
        FIX.session(2, 4) | layout(2),
        event_factory=lambda: event,
        stream_getter=lambda: NS(cuda_stream=42),
        capacity_provider=lambda: FIX.capacity(2, 4),
    )
    batch = NS(
        forward_mode=NS(name="IDLE"),
        reqs=[],
        mpu_dp_forward_modes=["decode", "extend", "idle", "idle"],
        mpu_dp_num_tokens=[1, 8, 0, 0],
    )
    with timer.wrap(batch):
        pass
    timer.flush_idle()
    rows = ANALYZE.read_jsonl(timer.path)
    assert rows[1]["dp_forward_modes"] == batch.mpu_dp_forward_modes
    assert rows[1]["dp_num_tokens"] == [1, 8, 0, 0]
    assert rows[1]["request_ids"] == []
    assert rows[-2]["capacity"]["attn_dp_rank"] == 2
    del batch.mpu_dp_num_tokens
    with pytest.raises(ValueError, match="actual DP scheduler gather"):
        with timer.wrap(batch):
            pass
