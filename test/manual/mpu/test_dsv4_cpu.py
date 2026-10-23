"""CPU control/ABI tests; deliberately do not claim CUDA or model validation."""

import ast
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


peer = load("test_c4_peer", "python/sglang/srt/mem_cache/mpu_peer.py")
config = load("test_c4_config", "python/sglang/srt/mem_cache/mpu_c4_config.py")
admission = load(
    "test_c4_admission", "python/sglang/srt/mem_cache/hisparse_admission.py"
)


def extract_class(relative, name, methods=None, **namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    if methods is not None:
        cls.body = [
            x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in methods
        ]
    prefix = ast.parse("from __future__ import annotations").body
    tree = ast.Module(body=prefix + [cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(relative), "exec"), namespace)
    return namespace[name]


C4Layout = extract_class(
    "python/sglang/srt/mem_cache/mpu_c4_pool.py", "C4Layout", dataclass=dataclass
)


def test_c4_value_scale_page_boundaries():
    layout = C4Layout(2, 3)
    assert layout.shape == (3, 2, 37440)
    assert layout.offsets(0) == (0, 36864)
    assert layout.offsets(63) == (36288, 37368)
    assert layout.offsets(64) == (37440, 74304)
    addresses = set()
    for slot in range(128):
        value, scale = layout.offsets(slot)
        payload = set(range(value, value + 576)) | set(range(scale, scale + 8))
        assert len(payload) == 584 and not addresses.intersection(payload)
        addresses.update(payload)
    assert len(addresses) == 128 * 584
    for slot in (-1, 128, True):
        with pytest.raises(ValueError):
            layout.offsets(slot)


def test_paged_allocator_reuse_and_invalid_frees():
    pool = peer.MPUSlotAllocator(192, 64)
    first = pool.alloc(1)
    second = pool.alloc(65)
    assert first.tolist() == list(range(64))
    assert second.tolist() == list(range(64, 192))
    assert first.device.type == "cpu" and pool.available_size() == 0
    assert pool.alloc(1) is None
    with pytest.raises(ValueError):
        pool.free(first[:1])
    assert pool.available_size() == 0
    assert pool.free(first) == 64
    with pytest.raises(ValueError):
        pool.free(first)
    assert pool.alloc(1).tolist() == first.tolist()
    pool.free(second)
    assert pool.available_size() == 128


@pytest.mark.parametrize(
    "prompt,output,slots", [(1, 1, 64), (255, 1, 64), (256, 1, 128), (1025, 257, 384)]
)
def test_lifetime_budget_uses_compressed_page_units(prompt, output, slots):
    budget = admission.HiSparseBackingBudget(1024, 64, 4)
    assert budget.slots_for(prompt, output) == slots
    assert budget.reserve("a", slots) and budget.reserve("a", slots)
    assert budget.reserved == slots
    assert not budget.reserve("b", 1024)
    budget.release("a")
    budget.release("a")
    assert budget.reserved == 0 and budget.reserve("b", 1024)


@pytest.mark.parametrize(
    "override",
    [
        {"mpu_capacity_ratio": math.nan},
        {"mpu_capacity_ratio": 0},
        {"mpu_device_ids": [0, 1, 2, 3]},
        {"mpu_memory_reserve_bytes": -1},
    ],
)
def test_invalid_mpu_options(override):
    with pytest.raises(ValueError):
        config.c4_backing_options({"backing_store": "mpu", **override})


def test_strict_tp4_nopd_launch():
    cfg = SimpleNamespace(disable_radix_cache=True)
    config.validate_c4_mpu_launch(cfg, {"backing_store": "mpu"})
    for key, value in (
        ("tp_size", 8),
        ("attn_cp_size", 2),
        ("disaggregation_mode", "decode"),
        ("enable_hierarchical_cache", True),
        ("enable_memory_saver", True),
        ("speculative_algorithm", "EAGLE"),
    ):
        with pytest.raises(ValueError):
            config.validate_c4_mpu_launch(
                SimpleNamespace(**vars(cfg), **{key: value}), {"backing_store": "mpu"}
            )
    with pytest.raises(ValueError):
        config.c4_backing_options({"mpu_device_ids": [4, 5, 6, 7]})


def test_allocation_snapshot_rejects_vmm():
    shard = SimpleNamespace(
        data_ptr=lambda: 128,
        numel=lambda: 64,
        element_size=lambda: 1,
        device=SimpleNamespace(index=4),
    )
    segment = dict(device=4, address=0, total_size=1024, is_expandable=False)
    with patch.object(torch.cuda, "memory_snapshot", return_value=[segment]):
        peer.validate_native_peer_allocations([shard])
    with patch.object(
        torch.cuda, "memory_snapshot", return_value=[dict(segment, is_expandable=True)]
    ):
        with pytest.raises(RuntimeError):
            peer.validate_native_peer_allocations([shard])


def test_peer_runtime_rejects_cpu_environment():
    with patch.object(peer.sys, "platform", "win32"):
        with pytest.raises(RuntimeError):
            peer.validate_mpu_runtime(0, [4])


def coordinator(methods, **namespace):
    cls = extract_class(
        "python/sglang/srt/managers/hisparse_coordinator.py",
        "HiSparseCoordinator",
        methods,
        torch=torch,
        **namespace
    )
    return cls()


def test_mpu_plan_then_replay_never_dispatches_storage_tensor():
    calls = []

    def planner(**kw):
        calls.append(("plan", kw))

    def replay(**kw):
        calls.append(("replay", kw))

    c = coordinator(
        {"_run_swap_in_kernel"},
        _is_xpu=False,
        load_cache_to_device_buffer_dsv4_mla=planner,
        load_cache_to_device_buffer_mla=planner,
    )
    c.is_mpu = c.is_dsv4_hisparse = True
    c.skip_io = False
    c.top_k = 2
    c.device_buffer_size = 4
    c.swap_in_block_size = 256
    c.item_size_bytes = 584
    c._mpu_miss_src = torch.empty((2, 2), dtype=torch.int64)
    c._mpu_miss_dst = torch.empty((2, 2), dtype=torch.int32)
    c._mpu_miss_count = torch.zeros(2, dtype=torch.int32)
    c.top_k_device_locs_buffer = torch.empty((2, 2), dtype=torch.int32)
    c.req_device_buffer_tokens = torch.zeros((1, 2, 5), dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.zeros_like(c.req_device_buffer_tokens)
    c.req_to_host_pool = torch.zeros((2, 10), dtype=torch.int64)
    c.lru_slots = torch.zeros((1, 2, 4), dtype=torch.int16)
    c.num_real_reqs = torch.tensor([1], dtype=torch.int32)
    local = torch.empty((1, 37440), dtype=torch.uint8)
    c.mem_pool_device = SimpleNamespace(kv_buffer=[local])
    c.mem_pool_host = SimpleNamespace(
        replay_misses=replay
    )  # no storage tensor accessor
    c._run_swap_in_kernel(
        torch.tensor([0]), torch.tensor([8]), torch.tensor([[1, 7]]), 0
    )
    assert [call[0] for call in calls] == ["plan", "replay"]
    assert calls[0][1]["skip_io"] is True and calls[0][1]["host_cache"] is local
    assert calls[1][1]["miss_src"].data_ptr() == c._mpu_miss_src.data_ptr()


def test_capture_initializes_real_pointer_maps_and_empty_finish_releases():
    c = coordinator(
        {"prepare_for_graph_capture", "release_backing", "request_finished"}
    )
    c.is_dsv4_hisparse = True
    c.mem_pool_host = SimpleNamespace(size=128, available_size=lambda: 128)
    c.backing_budget = admission.HiSparseBackingBudget(128, 64, 4)
    for name in (
        "req_to_device_buffer",
        "req_device_buffer_token_locs",
        "req_to_host_pool",
    ):
        setattr(c, name, torch.full((2, 10), -1))
    c.prepare_for_graph_capture()
    assert torch.all(c.req_device_buffer_token_locs == 0)
    c.backing_budget = admission.HiSparseBackingBudget(128, 64, 4)
    c.backing_budget.reserve("a", 64)
    c.request_finished(SimpleNamespace(rid="a", kv=SimpleNamespace(req_pool_idx=None)))
    assert c.backing_budget.reserved == 0


@pytest.mark.parametrize(
    "occupied,leased", [(True, False), (False, True), (True, True)]
)
def test_live_capture_rejected_without_overwriting_maps(occupied, leased):
    c = coordinator({"prepare_for_graph_capture"})
    c.is_dsv4_hisparse = True
    c.mem_pool_host = SimpleNamespace(
        size=128, available_size=lambda: 64 if occupied else 128
    )
    c.backing_budget = admission.HiSparseBackingBudget(128, 64, 4)
    if leased:
        c.backing_budget.reserve("active", 64)
    names = ("req_to_device_buffer", "req_device_buffer_token_locs", "req_to_host_pool")
    for name in names:
        setattr(c, name, torch.full((2, 10), 7))
    with pytest.raises(RuntimeError, match="idle backing pool"):
        c.prepare_for_graph_capture()
    for name in names:
        assert torch.all(getattr(c, name) == 7)


def test_retained_signature_collects_compute_pools_and_preserves_aliases():
    from types import ModuleType

    tensors = [torch.empty((i + 1, 2), dtype=torch.uint8) for i in range(7)]

    def state(t):
        return SimpleNamespace(kv_score_buffer=SimpleNamespace(kv_score=t))

    full = SimpleNamespace(
        c4_logical_size=128,
        swa_kv_pool=SimpleNamespace(kv_buffer=[tensors[0]]),
        c128_kv_pool=SimpleNamespace(kv_buffer=[tensors[1]]),
        index_pools={
            4: SimpleNamespace(index_k_with_scale_buffer=[tensors[2]]),
            2: SimpleNamespace(
                index_k_with_scale_buffer=None,
                index_k_payload_buffer=[tensors[3]],
                index_k_scale_buffer=[tensors[4]],
            ),
        },
        compress_state_pools=[None, state(tensors[5]), state(tensors[5])],
        indexer_compress_state_pools=[state(tensors[6]), None],
    )
    c = coordinator({"_retained_gpu_pool_tensors", "get_stage_capacity_info"})
    c.is_dsv4_hisparse = True
    c.is_mpu = True
    c.page_size = 64
    c.requested_backing_ratio = 2
    c.token_to_kv_pool_allocator = SimpleNamespace(
        get_kvcache=lambda: full,
        size_full=512,
        hisparse_attn_allocator=SimpleNamespace(size=64),
    )
    c.mem_pool_device = SimpleNamespace(
        kv_buffer=[torch.empty((2, 37440), dtype=torch.uint8)]
    )
    c.mem_pool_host = SimpleNamespace(
        size=128,
        available_size=lambda: 128,
        kv_buffer=[torch.empty((2, 37440), dtype=torch.uint8)],
        storage_device_ids=[4],
    )
    c.backing_budget = admission.HiSparseBackingBudget(128, 64, 4)
    seen = []

    def describe(tree):
        seen.append(tree)
        return {
            "retained_gpu_pool_bytes": 56,
            "retained_gpu_pool_shapes": {"fixture": True},
        }

    timer = ModuleType("sglang.srt.observability.mpu_stage_timing")
    timer.describe_retained_gpu_pools = describe
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_schedule = lambda: SimpleNamespace(max_total_tokens=512)
    with patch.dict(sys.modules, {timer.__name__: timer, runtime.__name__: runtime}):
        info = c.get_stage_capacity_info()
    assert info["retained_gpu_pool_bytes"] == 56
    assert "c4_hot" in info["retained_gpu_pool_scope"]
    assert "must not be added" in info["retained_gpu_pool_scope"]
    assert info["retained_gpu_pool_shapes"] == {"fixture": True}
    tree = seen[0]
    assert set(tree) == {
        "c4_hot",
        "swa",
        "c128",
        "indexer",
        "compress_state",
        "indexer_compress_state",
    }
    assert tree["c4_hot"] is c.mem_pool_device.kv_buffer
    assert tree["swa"][0] is tensors[0] and tree["c128"][0] is tensors[1]
    assert tree["indexer"]["4"]["key_with_scale"][0] is tensors[2]
    assert tree["indexer"]["2"]["payload"][0] is tensors[3]
    assert tree["indexer"]["2"]["scale"][0] is tensors[4]
    assert tree["compress_state"]["1"] is tree["compress_state"]["2"] is tensors[5]
    assert tree["indexer_compress_state"]["0"] is tensors[6]
