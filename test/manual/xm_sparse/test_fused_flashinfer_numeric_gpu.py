"""FlashInfer BF16 compatibility, independent of selection and Resolve.

The model acceptance remains exact greedy IDs and <=0.02 logprob difference.
This kernel regression catches the much larger FP32-P arithmetic mismatch.
"""

import os
import sys
import unittest

import torch


def qk_norm_rope(shape, positions, gain):
    values = torch.randn(shape, dtype=torch.float32)
    values *= values.square().mean(-1, keepdim=True).rsqrt() * gain
    frequency = 1.0 / (1_000_000.0 ** (torch.arange(64).float() / 64))
    angle = positions[:, None] * frequency[None, :]
    c, s = angle.cos()[:, None, :], angle.sin()[:, None, :]
    a, b = values[..., :64], values[..., 64:]
    return torch.cat((a * c - b * s, b * c + a * s), -1).to(torch.bfloat16)


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_XM_TMA_GPU_TESTS") == "1",
    "Set SGLANG_RUN_XM_TMA_GPU_TESTS=1 on a Linux SM90 host",
)
class TestFusedFlashInferNumericGPU(unittest.TestCase):
    def test_same_bf16_qkv_short_partial_full_and_backing(self):
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (9, 0)
        ):
            self.skipTest("Linux SM90 required")
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        from sglang.jit_kernel.fused_hisparse_tma_mma import (
            FusedHiSparseTMAMMAWorkspace,
        )

        torch.manual_seed(418)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda", torch.cuda.current_device())
        topk = 2048
        host_k = torch.empty((topk, 4, 128), dtype=torch.bfloat16, pin_memory=True)
        host_v = torch.empty_like(host_k, pin_memory=True)
        hot_k = torch.empty_like(host_k, device=device)
        hot_v = torch.empty_like(host_k, device=device)
        tma = FusedHiSparseTMAMMAWorkspace(
            device=device,
            max_bs=1,
            top_k=topk,
            hot_buffer_size=topk,
            item_size_bytes=1024,
            host_k_layers=[host_k],
            host_v_layers=[host_v],
        )
        actual = torch.ones(1, device=device, dtype=torch.int32)
        real = torch.ones_like(actual)
        indptr = torch.zeros(2, device=device, dtype=torch.int32)
        indices = torch.arange(topk, device=device, dtype=torch.int32)
        tma.device_locs[0].copy_(indices)
        last_page = torch.ones(1, device=device, dtype=torch.int32)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(384 << 20, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page,
        )
        for gain in (1.0, 4.0):
            host_k.copy_(qk_norm_rope(host_k.shape, torch.arange(topk), 1.0))
            host_v.copy_(torch.randn(host_v.shape).to(torch.bfloat16))
            hot_k.copy_(host_k)
            hot_v.copy_(host_v)
            for count in (65, 145, 2048):
                with self.subTest(count=count, q_gain=gain):
                    q = qk_norm_rope((1, 32, 128), torch.tensor([count - 1]), gain).to(
                        device
                    )
                    actual.fill_(count)
                    tma.plan_partition(torch.tensor([count], dtype=torch.int32))
                    indptr[1] = count
                    wrapper.plan(
                        indptr=indptr,
                        indices=indices,
                        last_page_len=last_page,
                        num_qo_heads=32,
                        num_kv_heads=4,
                        head_dim=128,
                        page_size=1,
                        sm_scale=128**-0.5,
                        data_type=torch.bfloat16,
                        q_data_type=torch.bfloat16,
                        non_blocking=False,
                    )
                    expected = wrapper.run(q, (hot_k.unsqueeze(1), hot_v.unsqueeze(1)))
                    tma.miss_host_locs.fill_(-1)
                    output = tma.run(
                        q,
                        hot_k=hot_k,
                        hot_v=hot_v,
                        layer_id=0,
                        actual_lens=actual,
                        num_real_reqs=real,
                        scaling=128**-0.5,
                    ).clone()
                    difference = output.float() - expected.float()
                    # Small scalar-QK/tensor-core reduction differences can
                    # cross a BF16 rounding midpoint. Preserve stringent RMS
                    # and differing-element limits as well as absolute error.
                    self.assertLessEqual(float(difference.abs().max().item()), 0.001)
                    self.assertLessEqual(
                        float(difference.square().mean().sqrt().item()), 0.00002
                    )
                    self.assertLessEqual(int((output != expected).sum().item()), 16)
                    for backing in ("mapped", "mixed"):
                        if backing == "mapped":
                            tma.miss_host_locs[0].copy_(indices.long())
                        else:
                            tma.miss_host_locs[0].copy_(
                                torch.where(indices % 2 == 0, indices.long(), -1)
                            )
                        hot_k_before, hot_v_before = hot_k.clone(), hot_v.clone()
                        mapped = tma.run(
                            q,
                            hot_k=hot_k,
                            hot_v=hot_v,
                            layer_id=0,
                            actual_lens=actual,
                            num_real_reqs=real,
                            scaling=128**-0.5,
                        ).clone()
                        torch.testing.assert_close(mapped, output, rtol=0, atol=0)
                        torch.testing.assert_close(hot_k, hot_k_before, rtol=0, atol=0)
                        torch.testing.assert_close(hot_v, hot_v_before, rtol=0, atol=0)
                    torch.cuda.synchronize()

    def test_dynamic_partition_matches_fa2_batch_and_graph_padding(self):
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (9, 0)
        ):
            self.skipTest("Linux SM90 required")
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        from sglang.jit_kernel.fused_hisparse_tma_mma import (
            FusedHiSparseTMAMMAWorkspace,
        )

        torch.manual_seed(715)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda", torch.cuda.current_device())
        batch, topk = 12, 2048
        rows = batch * topk
        host_k = torch.empty((rows, 4, 128), dtype=torch.bfloat16, pin_memory=True)
        host_v = torch.empty_like(host_k, pin_memory=True)
        host_k.copy_(qk_norm_rope(host_k.shape, torch.arange(rows), 1.0))
        host_v.copy_(torch.randn(host_v.shape).to(torch.bfloat16))
        hot_k, hot_v = host_k.to(device), host_v.to(device)
        tma = FusedHiSparseTMAMMAWorkspace(
            device=device,
            max_bs=batch,
            top_k=topk,
            hot_buffer_size=topk,
            item_size_bytes=1024,
            host_k_layers=[host_k],
            host_v_layers=[host_v],
        )
        physical = torch.arange(rows, device=device, dtype=torch.int32).view(
            batch, topk
        )
        tma.device_locs.copy_(physical)
        tma.miss_host_locs.copy_(torch.where(physical % 2 == 0, physical.long(), -1))
        actual = torch.full((batch,), topk, device=device, dtype=torch.int32)
        real = torch.tensor([batch], device=device, dtype=torch.int32)
        q = qk_norm_rope((batch, 32, 128), torch.full((batch,), topk - 1), 1.0).to(
            device
        )
        fi_workspace = torch.empty(384 << 20, dtype=torch.uint8, device=device)
        receipts = []
        for size in (1, 2, 4, 8, 12):
            indptr = torch.empty(size + 1, device=device, dtype=torch.int32)
            indices = torch.empty(size * topk, device=device, dtype=torch.int32)
            last_page = torch.ones(size, device=device, dtype=torch.int32)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                fi_workspace,
                kv_layout="NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last_page,
            )
            cases = {
                "full": [topk] * size,
                "short": [65] * size,
                "mixed": [topk if request % 2 == 0 else 145 for request in range(size)],
                "padding": [topk] * max(1, size // 2)
                + [1] * (size - max(1, size // 2)),
            }
            for name, lengths in cases.items():
                with self.subTest(batch=size, case=name):
                    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
                    pointer_cpu = torch.cat(
                        (
                            torch.zeros(1, dtype=torch.int32),
                            lengths_cpu.cumsum(0).to(torch.int32),
                        )
                    )
                    indptr.copy_(pointer_cpu)
                    packed = torch.cat(
                        [
                            physical[request, :length]
                            for request, length in enumerate(lengths)
                        ]
                    )
                    indices.fill_(-1)
                    indices[: packed.numel()].copy_(packed)
                    wrapper.plan(
                        indptr=indptr,
                        indices=indices,
                        last_page_len=last_page,
                        num_qo_heads=32,
                        num_kv_heads=4,
                        head_dim=128,
                        page_size=1,
                        sm_scale=128**-0.5,
                        data_type=torch.bfloat16,
                        q_data_type=torch.bfloat16,
                        non_blocking=False,
                    )
                    info = wrapper._plan_info
                    actual_chunk = int(
                        wrapper._int_workspace_buffer[info[9] : info[9] + 4]
                        .view(torch.int32)
                        .item()
                    )
                    predicted_chunk = tma.plan_partition(lengths_cpu)
                    self.assertEqual(actual_chunk, predicted_chunk)
                    self.assertEqual(info[3], 16)  # CTA Q rows, hence 4 KV warps
                    actual[:size].copy_(lengths_cpu)
                    real.fill_(size)
                    expected = wrapper.run(
                        q[:size], (hot_k.unsqueeze(1), hot_v.unsqueeze(1))
                    )
                    output = tma.run(
                        q[:size],
                        hot_k=hot_k,
                        hot_v=hot_v,
                        layer_id=0,
                        actual_lens=actual[:size],
                        num_real_reqs=real,
                        scaling=128**-0.5,
                    ).clone()
                    difference = output.float() - expected.float()
                    self.assertLessEqual(float(difference.abs().max().item()), 0.001)
                    self.assertLessEqual(
                        float(difference.square().mean().sqrt().item()), 0.00002
                    )
                    receipts.append(
                        (
                            size,
                            name,
                            actual_chunk,
                            float(difference.abs().max().item()),
                            float(difference.square().mean().sqrt().item()),
                        )
                    )
        # Capture a fixed B12 graph, then change chunk/length/real-batch values
        # outside capture. The output and partition scalar addresses stay fixed.
        actual.fill_(topk)
        real.fill_(batch)
        tma.plan_partition(torch.full((batch,), topk, dtype=torch.int32))
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                tma.run(
                    q,
                    hot_k=hot_k,
                    hot_v=hot_v,
                    layer_id=0,
                    actual_lens=actual,
                    num_real_reqs=real,
                    scaling=128**-0.5,
                )
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = tma.run(
                q,
                hot_k=hot_k,
                hot_v=hot_v,
                layer_id=0,
                actual_lens=actual,
                num_real_reqs=real,
                scaling=128**-0.5,
            )
        pointers = (tma.kv_chunk_size.data_ptr(), captured_output.data_ptr())
        for count in (12, 8, 4, 1):
            lengths_cpu = torch.tensor(
                [topk] * count + [1] * (batch - count), dtype=torch.int32
            )
            actual.copy_(lengths_cpu)
            real.fill_(count)
            tma.plan_partition(lengths_cpu)
            expected = tma.run(
                q,
                hot_k=hot_k,
                hot_v=hot_v,
                layer_id=0,
                actual_lens=actual,
                num_real_reqs=real,
                scaling=128**-0.5,
            ).clone()
            graph.replay()
            torch.testing.assert_close(captured_output, expected, rtol=0, atol=0)
            self.assertTrue(torch.all(captured_output[count:] == 0).item())
            self.assertEqual(
                pointers, (tma.kv_chunk_size.data_ptr(), captured_output.data_ptr())
            )
        torch.cuda.synchronize()
        print(f"FA2 dynamic partition receipts={receipts}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
