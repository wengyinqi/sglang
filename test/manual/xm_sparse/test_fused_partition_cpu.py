"""FA2 partition arithmetic and Graph metadata contracts, without CUDA."""

import ast
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/jit_kernel/fused_hisparse_tma.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTION = next(
    n
    for n in TREE.body
    if getattr(n, "name", None) == "flashinfer_compatible_kv_chunk_size"
)
NAMESPACE = {"Sequence": Sequence, "NUM_KV_HEADS": 4}
exec(
    compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"),
    NAMESPACE,
)
partition = NAMESPACE["flashinfer_compatible_kv_chunk_size"]


@pytest.mark.parametrize(
    "batch,expected", [(1, 128), (2, 128), (3, 128), (4, 147), (8, 293), (12, 512)]
)
def test_h800_full_context_partition(batch, expected):
    assert partition([2048] * batch, 114) == expected


def test_short_and_padding_are_counted_like_fa2():
    assert partition([0, 1, 65, 145], 114) == 128
    assert partition([2048] * 3 + [1] * 9, 114) == 128
    assert partition([2048] * 4 + [1] * 8, 114) == 171


def test_partition_matches_exhaustive_minimum_with_device_capacity():
    rng = random.Random(715)
    for _ in range(100):
        sms = rng.choice([80, 114, 132, 148])
        lengths = [rng.randint(0, 2048) for _ in range(rng.randint(1, 16))]
        capacity = 2 * sms // 4
        high = max(128, *lengths)
        expected = next(
            (
                chunk
                for chunk in range(128, high + 1)
                if sum((max(length, 1) + chunk - 1) // chunk for length in lengths)
                <= capacity
            ),
            high,
        )
        assert partition(lengths, sms) == expected


@pytest.mark.parametrize(
    "lengths,sms",
    [([], 114), ([-1], 114), ([1.0], 114), ([True], 114), ([1], True), ([1], 0)],
)
def test_invalid_partition_metadata_is_rejected(lengths, sms):
    with pytest.raises(ValueError):
        partition(lengths, sms)


def test_partition_scalar_is_reserved_once_in_auxiliary_budget():
    path = ROOT / "python/sglang/srt/mem_cache/sparsity/quest_memory.py"
    spec = importlib.util.spec_from_file_location("quest_partition_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dimensions = dict(
        top_k=2048,
        quest_page_size=64,
        start_layer=0,
        end_layer=48,
        max_reqs=12,
        max_context_len=32768,
        kv_heads=4,
        head_dim=128,
        device_buffer_size=2048,
        device_token_capacity=12 * 2048,
    )
    ordinary = module.estimate_quest_auxiliary_bytes(**dimensions)
    fused = module.estimate_quest_auxiliary_bytes(
        **dimensions, xm_fused_retrieval_attention=True
    )
    assert fused - ordinary == 12 * 12 * 2048 + 4


def test_partition_owner_updates_only_outside_capture_and_counts_four_bytes(
    monkeypatch,
):
    owner = next(
        n for n in TREE.body if getattr(n, "name", None) == "FusedHiSparseTMAWorkspace"
    )
    names = {"plan_partition", "memory_usage"}
    methods = [
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {"torch": torch, "flashinfer_compatible_kv_chunk_size": partition}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    device_scalar = torch.tensor([128], dtype=torch.int32)
    state = SimpleNamespace(
        max_bs=12,
        top_k=2048,
        num_sms=114,
        kv_chunk_size=device_scalar,
        miss_host_locs=torch.empty((12, 2048), dtype=torch.int64),
        device_locs=torch.empty((12, 2048), dtype=torch.int32),
        partial=torch.empty((12, 32, 16, 130)),
        output=torch.empty((12, 32, 128)),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pointer = device_scalar.data_ptr()
    assert (
        namespace["plan_partition"](state, torch.full((8,), 2048, dtype=torch.int32))
        == 293
    )
    assert device_scalar.item() == 293 and device_scalar.data_ptr() == pointer
    assert namespace["memory_usage"](state)["plan_kv_chunk_size_bytes"] == 4
    with pytest.raises(ValueError):
        namespace["plan_partition"](state, torch.tensor([2049], dtype=torch.int32))
    with pytest.raises(ValueError):
        namespace["plan_partition"](state, torch.tensor([-1], dtype=torch.int32))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside Graph capture"):
        namespace["plan_partition"](state, torch.tensor([145], dtype=torch.int32))
