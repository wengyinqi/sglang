"""GPU counters must preserve actual K/V, LRU and CUDA Graph padding semantics."""

import pytest
import torch

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mha
from sglang.jit_kernel.tests.test_hisparse import ITEM_SIZE_BYTES, _long_case

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires NVIDIA CUDA GPU"
)


def make_call(metrics=None, padded=False, seq_dtype=torch.int32):
    state = _long_case()
    host_v = torch.empty_like(state["host_cache"], pin_memory=True)
    host_v.copy_(state["host_cache"] + 10000)
    device_v = state["device_buffer"].clone() + 10000
    top = torch.tensor(
        [[4, 6, 7]] + ([[0, 0, 0]] if padded else []), dtype=torch.int32, device="cuda"
    )
    locations = torch.full_like(top, -1)
    kwargs = dict(
        top_k_tokens=top,
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_k_cache=state["host_cache"],
        host_v_cache=host_v,
        device_k_buffer=state["device_buffer"],
        device_v_buffer=device_v,
        top_k_device_locs=locations,
        req_pool_indices=torch.tensor(
            [0] + ([-1] if padded else []), dtype=torch.int64, device="cuda"
        ),
        seq_lens=torch.tensor(
            [8] + ([0] if padded else []), dtype=seq_dtype, device="cuda"
        ),
        lru_slots=state["lru_slots"],
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=3,
        hot_buffer_size=4,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device="cuda"),
        cache_metrics=metrics,
    )
    return kwargs, state, device_v, locations


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_metrics_match_data_and_lru(seq_dtype):
    baseline, old, old_v, old_locs = make_call(seq_dtype=seq_dtype)
    metrics = torch.zeros(5, dtype=torch.int64, device="cuda")
    measured, state, v, locs = make_call(metrics=metrics, seq_dtype=seq_dtype)
    load_cache_to_device_buffer_mha(**baseline)
    load_cache_to_device_buffer_mha(**measured)
    torch.cuda.synchronize()
    assert metrics.cpu().tolist() == [1, 3, 1, 1, 1]
    for a, b in (
        (old_locs, locs),
        (old_v, v),
        (old["device_buffer"], state["device_buffer"]),
        (old["lru_slots"], state["lru_slots"]),
    ):
        assert torch.equal(a, b)
    assert torch.equal(state["device_buffer"][locs[0, 1]].cpu(), state["host_cache"][6])
    assert torch.equal(v[locs[0, 1]].cpu(), measured["host_v_cache"][6])
    load_cache_to_device_buffer_mha(**measured)
    torch.cuda.synchronize()
    assert metrics.cpu().tolist() == [2, 6, 3, 2, 1]


def test_graph_replay_metrics_ignore_padded_block():
    metrics = torch.zeros(5, dtype=torch.int64, device="cuda")
    kwargs, state, _, locs = make_call(metrics=metrics, padded=True)
    load_cache_to_device_buffer_mha(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mha(**kwargs)
    metrics.zero_()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert metrics.cpu().tolist() == [2, 6, 4, 2, 0]
    assert locs[1].cpu().tolist() == [-1, -1, -1]


def test_short_sequence_metrics_count_only_valid_entries():
    metrics = torch.zeros(5, dtype=torch.int64, device="cuda")
    kwargs, _, _, _ = make_call(metrics=metrics)
    kwargs["seq_lens"].fill_(3)
    kwargs["top_k_tokens"].copy_(
        torch.tensor([[2, -1, 0]], dtype=torch.int32, device="cuda")
    )
    load_cache_to_device_buffer_mha(**kwargs)
    torch.cuda.synchronize()
    assert metrics.cpu().tolist() == [1, 2, 2, 0, 0]
