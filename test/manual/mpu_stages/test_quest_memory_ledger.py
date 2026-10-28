"""CUDA metadata spies, not GPU allocation or KV movement tests."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TIMER = load("python/sglang/srt/observability/mpu_stage_timing.py", "ledger_timer")
with patch.dict(sys.modules, {"sglang.srt.observability.mpu_stage_timing": TIMER}):
    LEDGER = load(
        "python/sglang/srt/observability/quest_memory_ledger.py", "quest_ledger"
    )


class Device:
    def __init__(self, kind="cuda", ordinal=0):
        self.type, self.ordinal = kind, ordinal

    def __str__(self):
        return f"{self.type}:{self.ordinal}"


class Tensor:
    def __init__(self, pointer, size=64, shape=(8,), kind="cuda", ordinal=0):
        self.pointer, self.size, self.shape = pointer, size, shape
        self.device, self.dtype = Device(kind, ordinal), "torch.int64"

    def stride(self):
        return (1,) * len(self.shape)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.size, data_ptr=lambda: self.pointer)

    def cpu(self):
        raise AssertionError("Must not copy payload")

    def item(self):
        raise AssertionError("Must not read values")


def fixture(variant="mpu", mapping_size=512):
    bounds = Tensor(1, 128)
    selector = SimpleNamespace(
        page_k_bounds=bounds,
        page_k_min=bounds,
        page_k_max=bounds,
        running_k_min=Tensor(2),
        running_k_max=Tensor(3),
    )
    mapping = Tensor(4, mapping_size, (mapping_size // 8,))
    pool = SimpleNamespace(
        k_buffer=[Tensor(5, 1088)],
        v_buffer=[Tensor(6, 1088)],
        k_data_ptrs=Tensor(7),
        v_data_ptrs=Tensor(8),
        full_to_hisparse_device_index_mapping=mapping,
    )
    backing = (
        SimpleNamespace(shards=[Tensor(9, 8192, ordinal=1)], peer_ptrs=Tensor(10))
        if variant == "mpu"
        else SimpleNamespace(
            k_buffer=Tensor(9, 4096, kind="cpu"),
            v_buffer=Tensor(10, 4096, kind="cpu"),
            k_data_ptrs=Tensor(11),
            v_data_ptrs=Tensor(12),
        )
    )
    return SimpleNamespace(
        quest_algorithm=selector,
        mem_pool_device=pool,
        mem_pool_host=backing,
        backing_store=variant,
        req_to_token_pool=SimpleNamespace(req_to_token=Tensor(13)),
        token_to_kv_pool_allocator=SimpleNamespace(
            full_to_hisparse_device_index_mapping=mapping,
            hisparse_attn_allocator=SimpleNamespace(free_pages=Tensor(14)),
        ),
        req_to_device_buffer=Tensor(15),
        num_real_reqs=Tensor(16),
    )


@pytest.mark.parametrize("variant", ["cpu", "mpu"])
def test_aliases_padding_and_storage_device_payload_are_accounted_separately(variant):
    result = LEDGER.collect_quest_memory(fixture(variant))
    assert result["allocated_compute_kv_bytes"] == 2176
    assert result["allocated_backing_kv_bytes"] == 8192
    groups = result["gpu_memory_ledger"]["groups"]
    assert groups["capacity_dependent"]["retained_gpu_pool_bytes"] == 512
    shapes = groups["fixed"]["retained_gpu_pool_shapes"]
    assert (
        shapes["quest.page_k_min"]["allocation_group"]
        == shapes["quest.page_k_max"]["allocation_group"]
    )
    assert (
        sum(g["retained_gpu_pool_bytes"] for g in groups.values())
        == result["gpu_memory_ledger"]["named_compute_bytes"]
    )
    assert "8192" not in str(result["gpu_memory_ledger"]["named_compute_shapes"])


def test_capacity_mapping_growth_does_not_change_fixed_selector_budget():
    first = LEDGER.collect_quest_memory(fixture(mapping_size=512))
    second = LEDGER.collect_quest_memory(fixture(mapping_size=1024))
    assert first["retained_gpu_pool_shapes"] == second["retained_gpu_pool_shapes"]
    assert (
        second["gpu_memory_ledger"]["named_compute_bytes"]
        - first["gpu_memory_ledger"]["named_compute_bytes"]
        == 512
    )


def test_missing_cuda_selector_storage_is_an_error():
    value = fixture()
    value.quest_algorithm.page_k_min = Tensor(99, kind="cpu")
    with pytest.raises(ValueError, match="initialized CUDA"):
        LEDGER.collect_quest_memory(value)


def test_every_persistent_selector_tensor_is_inventoried_including_future_fields():
    value = fixture()
    value.quest_algorithm.new_scratch = Tensor(99)
    assert (
        "quest.new_scratch"
        in LEDGER.collect_quest_memory(value)["retained_gpu_pool_shapes"]
    )
    tree = ast.parse(
        (ROOT / "python/sglang/srt/managers/hisparse_coordinator.py").read_text()
    )
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_stage_capacity_info"
    )
    assert "self._quest_memory_ledger()" in ast.unparse(method)
