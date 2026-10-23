"""Run on Linux with two peer-accessible CUDA GPUs; no model validation implied."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="Requires Linux and two peer-accessible CUDA GPUs",
)


def pool(size=128, layers=2):
    from sglang.srt.mem_cache.mpu_c4_pool import DeepSeekV4MPUPagedPool

    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("Compute 0 cannot peer-access storage 1")
    local = SimpleNamespace(
        device="cuda:0",
        size=size,
        layer_num=layers,
        page_size=64,
        bytes_per_page_padded=37440,
    )
    local.kv_buffer = [
        torch.arange((size // 64 + 1) * 37440, device="cuda:0", dtype=torch.int32)
        .remainder(251)
        .to(torch.uint8)
        .view(-1, 37440)
        for _ in range(layers)
    ]
    return local, DeepSeekV4MPUPagedPool(
        local, 1, capacity_ratio=2, memory_reserve_bytes=0
    )


def record(buffer, slot):
    page, offset = divmod(slot, 64)
    return torch.cat(
        [
            buffer[page, offset * 576 : (offset + 1) * 576],
            buffer[page, 36864 + offset * 8 : 36864 + (offset + 1) * 8],
        ]
    )


def test_c4_value_scale_peer_roundtrip_and_graph_replay():
    local, backing = pool()
    try:
        src = torch.tensor([0, 63, 64, 127], device="cuda:0", dtype=torch.int64)
        peer = torch.tensor([64, 127, 0, 63], device="cuda:0", dtype=torch.int64)
        dst = torch.tensor([128, 129, 130, 131], device="cuda:0", dtype=torch.int64)

        def roundtrip():
            backing.backup_from_device_all_layer(local, peer, src)
            for layer in range(local.layer_num):
                backing.load_to_device_per_layer(local, peer, dst, layer)

        roundtrip()
        torch.cuda.synchronize(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            roundtrip()
        for value in (17, 31, 63):
            for buf in local.kv_buffer:
                # Only compute buffers receive tensor operations.
                buf[:2].fill_(value)
            graph.replay()
            torch.cuda.synchronize(0)
            for buf in local.kv_buffer:
                for slot in dst.tolist():
                    assert torch.all(record(buf, slot) == value)
    finally:
        backing.destroy()


@pytest.mark.parametrize("block_size", [256, 960, 1024])
def test_c4_large_miss_plan_matches_original_records(block_size):
    from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_dsv4_mla

    local, backing = pool(8192, 1)
    try:
        device = "cuda:0"
        hot = 4096
        topk = 2048
        seq = 7299
        ids = torch.arange(8192, device=device, dtype=torch.int64)
        backing.backup_from_device_all_layer(local, ids, ids)
        chosen = (
            torch.cat([torch.arange(1000, 2000), torch.arange(5000, 6048)])
            .to(device, torch.int32)
            .view(1, -1)
        )
        expected = torch.stack(
            [record(local.kv_buffer[0], int(i)) for i in chosen[0].tolist()]
        )
        buffer_tokens = torch.arange(hot + 1, device=device, dtype=torch.int32).view(
            1, -1
        )
        buffer_tokens[0, hot] = seq - 1
        buffer_locs = torch.arange(hot + 1, device=device, dtype=torch.int32).view(
            1, -1
        )
        host_locs = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)
        lru = torch.arange(hot, device=device, dtype=torch.int16).view(1, -1)
        out = torch.empty((1, topk), device=device, dtype=torch.int32)
        miss_src = torch.empty((1, topk), device=device, dtype=torch.int64)
        miss_dst = torch.empty((1, topk), device=device, dtype=torch.int32)
        miss_count = torch.zeros(1, device=device, dtype=torch.int32)
        real = torch.ones(1, device=device, dtype=torch.int32)
        load_cache_to_device_buffer_dsv4_mla(
            top_k_tokens=chosen,
            device_buffer_tokens=buffer_tokens,
            host_cache_locs=host_locs,
            device_buffer_locs=buffer_locs,
            host_cache=local.kv_buffer[0],
            device_buffer=local.kv_buffer[0],
            top_k_device_locs=out,
            req_pool_indices=torch.zeros(1, device=device, dtype=torch.int64),
            seq_lens=torch.tensor([seq], device=device, dtype=torch.int32),
            lru_slots=lru,
            item_size_bytes=584,
            num_top_k=topk,
            hot_buffer_size=hot,
            page_size=1,
            block_size=block_size,
            num_real_reqs=real,
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
            skip_io=True,
        )
        backing.replay_misses(miss_src, miss_dst, miss_count, real, 0)
        torch.cuda.synchronize(0)
        actual = torch.stack(
            [record(local.kv_buffer[0], int(i)) for i in out[0].tolist()]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert miss_count.item() == 1048
    finally:
        backing.destroy()
