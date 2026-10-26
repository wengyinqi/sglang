"""Real KPool selection -> coordinator resolve -> MPU replay, without a model.

Synthetic scores enter the production selector; no miss table is supplied by
the test. Run through acceptance.py on the pinned GLM environment.
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_GLM53_MPU") != "1",
    reason="Requires explicit eight-GPU GLM MPU acceptance",
)


def test_kpool_resolve_peer_hits_eviction_tail_padding_and_graph():
    assert sys.platform == "linux" and torch.cuda.device_count() >= 8
    from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
        fast_kpool_topk_transform_fused,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.mem_cache.mpu_glm import HiSparseMLAMPUPool

    torch.cuda.set_device(0)
    device, slots, hot, width = "cuda:0", 8192, 4096, 2051
    ids = torch.arange(slots, device=device, dtype=torch.int64)
    channels = torch.arange(512, device=device)

    def pattern(tokens, layer):
        # Exact BF16 integers, varying by token, channel and layer.
        values = ((tokens[:, None] * 13 + channels + layer * 7) % 251).to(
            torch.bfloat16
        )[:, None, :]
        values[:, 0, 0] = tokens % 128
        values[:, 0, 1] = tokens // 128
        return values

    buffers = [
        torch.zeros((slots + 64, 1, 512), device=device, dtype=torch.bfloat16)
        for _ in range(11)
    ]
    local = NS(
        device=device,
        page_size=64,
        layer_num=11,
        dtype=torch.bfloat16,
        kv_cache_dim=512,
        start_layer=0,
        index_kpool=4,
        size=slots,
        kv_buffer=buffers,
    )
    pool = HiSparseMLAMPUPool(
        local, storage_device_ids=[4, 5, 6, 7], capacity_ratio=1, memory_reserve_bytes=0
    )
    graph = None
    try:
        host_map = (ids + 64) % slots  # Non-identity page mapping, including wrap.
        for layer, buf in enumerate(buffers):
            buf[:slots].copy_(pattern(ids, layer))
        pool.backup_from_device_all_layer(local, host_map, ids, "kernel")
        c = HiSparseCoordinator.__new__(HiSparseCoordinator)
        c.is_mpu, c.is_dsv4_hisparse, c.skip_io = True, False, False
        c.top_k, c.num_tail_slots = width, 3
        c.device_buffer_size, c.swap_in_block_size = hot, 256
        c.item_size_bytes, c.mem_pool_device, c.mem_pool_host = 1024, local, pool
        c.num_real_reqs = torch.ones(1, device=device, dtype=torch.int32)
        c.req_to_host_pool = host_map.view(1, -1)
        c.req_device_buffer_tokens = torch.empty(
            (11, 1, hot + 1), device=device, dtype=torch.int32
        )
        c.req_device_buffer_token_locs = (
            torch.arange(hot + 1, device=device, dtype=torch.int32)
            .view(1, 1, -1)
            .repeat(11, 1, 1)
        )
        c.lru_slots = (
            torch.arange(hot, device=device, dtype=torch.int16)
            .view(1, 1, -1)
            .repeat(11, 1, 1)
        )
        c.top_k_device_locs_buffer = torch.empty(
            (2, width), device=device, dtype=torch.int32
        )
        c._miss_src = torch.full(
            (2, width), slots + 123, device=device, dtype=torch.int64
        )
        c._miss_dst = torch.full(
            (2, width), slots + 123, device=device, dtype=torch.int32
        )
        c._miss_count = torch.full((2,), width, device=device, dtype=torch.int32)
        # A padded request must never dereference its invalid request ID.
        reqs = torch.tensor([0, 123456], device=device, dtype=torch.int64)
        seqs = torch.empty(2, device=device, dtype=torch.int32)
        lengths = torch.empty_like(seqs)
        scores = torch.empty((2, slots // 4), device=device, dtype=torch.float32)

        def prepare(seq, band, reset):
            seqs.fill_(seq)
            lengths.fill_(seq // 4)
            scores.copy_(-torch.arange(slots // 4, device=device).float().repeat(2, 1))
            # 256 groups initially cached, 256 remote; band 1 replaces them.
            for start in ((250, 1250) if band == 0 else (600, 1500)):
                scores[:, start : start + 256] += 10000
            if reset:
                for layer, buf in enumerate(buffers):
                    buf[:hot].copy_(pattern(ids[:hot], layer))
                    buf[hot : hot + 1].copy_(pattern(ids[seq - 1 : seq], layer))
                c.req_device_buffer_tokens.copy_(
                    torch.arange(hot + 1, device=device, dtype=torch.int32).view(
                        1, 1, -1
                    )
                )
                c.req_device_buffer_tokens[:, :, hot] = seq - 1
                c.lru_slots.copy_(
                    torch.arange(hot, device=device, dtype=torch.int16).view(1, 1, -1)
                )

        def pipeline():
            chosen = fast_kpool_topk_transform_fused(
                scores, lengths, kpool=4, topk=2048, seq_lens=seqs
            )
            out = c._run_swap_in_kernel(reqs, seqs, chosen, 0)
            # Production shared-index copy path must use the same miss plan.
            c._run_copy_only_kernel(2, 10)
            return chosen, out

        def check(chosen, out, seq, expected_misses=None):
            torch.cuda.synchronize(0)
            valid = chosen[0] >= 0
            tokens, locations = chosen[0, valid].long(), out[0, valid].long()
            assert chosen.shape == (2, width)
            assert (out[0, ~valid] == -1).all() and (out[1] == -1).all()
            assert ((locations >= 0) & (locations <= hot)).all()
            tail = list(range(seq // 4 * 4, seq)) + [-1] * (3 - seq % 4)
            assert chosen[0, 2048:].tolist() == tail
            # Selector oracle: compare groups independently of radix output order.
            count = min(seq // 4, 512)
            best = torch.topk(scores[0, : seq // 4], count).indices
            expected_tokens = (
                best[:, None] * 4 + torch.arange(4, device=device)
            ).flatten()
            actual_history = chosen[0, :2048]
            torch.testing.assert_close(
                actual_history[actual_history >= 0].long().sort().values,
                expected_tokens.sort().values,
                rtol=0,
                atol=0,
            )
            for layer in (0, 10):
                torch.testing.assert_close(
                    buffers[layer][locations], pattern(tokens, layer), rtol=0, atol=0
                )
            if expected_misses is not None:
                assert c._miss_count[0].item() == expected_misses
            count = c._miss_count[0].item()
            if count:
                src, dst = c._miss_src[0, :count], c._miss_dst[0, :count].long()
                loaded_tokens = c.req_device_buffer_tokens[0, 0, dst].long()
                torch.testing.assert_close(src, host_map[loaded_tokens], rtol=0, atol=0)

        # Real misses, then all hits, then eviction of previously cached tokens.
        prepare(7299, 0, True)
        chosen, out = pipeline()
        check(chosen, out, 7299, 1026)
        chosen, out = pipeline()
        check(chosen, out, 7299, 0)
        prepare(7299, 1, False)
        chosen, out = pipeline()
        check(chosen, out, 7299)
        assert c._miss_count[0].item() > 0
        # Warm every JIT path before capture; captured selector reads live inputs.
        prepare(7299, 0, True)
        pipeline()
        torch.cuda.synchronize(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_chosen, captured_out = pipeline()
        for seq, band in ((7299, 0), (7299, 1), (4095, 0), (67, 0), (64, 0)):
            prepare(seq, band, True)
            graph.replay()
            check(captured_chosen, captured_out, seq, 0 if seq <= hot else None)
        before = [buffers[layer].clone() for layer in (0, 10)]
        c.num_real_reqs.zero_()
        graph.replay()
        torch.cuda.synchronize(0)
        assert (captured_out == -1).all()
        for saved, layer in zip(before, (0, 10)):
            torch.testing.assert_close(buffers[layer], saved, rtol=0, atol=0)
    finally:
        if graph is not None:
            del graph
        pool.destroy()
