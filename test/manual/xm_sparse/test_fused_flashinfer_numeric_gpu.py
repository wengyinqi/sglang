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
        if sys.platform != "linux" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("Linux SM90 required")
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        from sglang.jit_kernel.fused_hisparse_tma import FusedHiSparseTMAWorkspace

        torch.manual_seed(418)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda", torch.cuda.current_device())
        topk = 2048
        host_k = torch.empty((topk, 4, 128), dtype=torch.bfloat16, pin_memory=True)
        host_v = torch.empty_like(host_k, pin_memory=True)
        hot_k = torch.empty_like(host_k, device=device)
        hot_v = torch.empty_like(host_k, device=device)
        tma = FusedHiSparseTMAWorkspace(
            device=device, max_bs=1, top_k=topk, hot_buffer_size=topk,
            item_size_bytes=1024, host_k_layers=[host_k], host_v_layers=[host_v],
        )
        actual = torch.ones(1, device=device, dtype=torch.int32)
        real = torch.ones_like(actual)
        indptr = torch.zeros(2, device=device, dtype=torch.int32)
        indices = torch.arange(topk, device=device, dtype=torch.int32)
        tma.device_locs[0].copy_(indices)
        last_page = torch.ones(1, device=device, dtype=torch.int32)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(384 << 20, dtype=torch.uint8, device=device),
            kv_layout="NHD", use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr, paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page,
        )
        for gain in (1.0, 4.0):
            host_k.copy_(qk_norm_rope(host_k.shape, torch.arange(topk), 1.0))
            host_v.copy_(torch.randn(host_v.shape).to(torch.bfloat16))
            hot_k.copy_(host_k)
            hot_v.copy_(host_v)
            for count in (65, 145, 2048):
                with self.subTest(count=count, q_gain=gain):
                    q = qk_norm_rope((1, 32, 128), torch.tensor([count - 1]), gain).to(device)
                    actual.fill_(count)
                    indptr[1] = count
                    wrapper.plan(
                        indptr=indptr, indices=indices, last_page_len=last_page,
                        num_qo_heads=32, num_kv_heads=4, head_dim=128, page_size=1,
                        sm_scale=128 ** -0.5, data_type=torch.bfloat16,
                        q_data_type=torch.bfloat16, non_blocking=False,
                    )
                    expected = wrapper.run(q, (hot_k.unsqueeze(1), hot_v.unsqueeze(1)))
                    tma.miss_host_locs.fill_(-1)
                    output = tma.run(
                        q, hot_k=hot_k, hot_v=hot_v, layer_id=0,
                        actual_lens=actual, num_real_reqs=real, scaling=128 ** -0.5,
                    ).clone()
                    difference = output.float() - expected.float()
                    # Small scalar-QK/tensor-core reduction differences can
                    # cross a BF16 rounding midpoint. Preserve stringent RMS
                    # and differing-element limits as well as absolute error.
                    self.assertLessEqual(float(difference.abs().max().item()), 0.001)
                    self.assertLessEqual(float(difference.square().mean().sqrt().item()), 0.00002)
                    self.assertLessEqual(int((output != expected).sum().item()), 16)
                    for backing in ("mapped", "mixed"):
                        if backing == "mapped":
                            tma.miss_host_locs[0].copy_(indices.long())
                        else:
                            tma.miss_host_locs[0].copy_(torch.where(indices % 2 == 0, indices.long(), -1))
                        hot_k_before, hot_v_before = hot_k.clone(), hot_v.clone()
                        mapped = tma.run(
                            q, hot_k=hot_k, hot_v=hot_v, layer_id=0,
                            actual_lens=actual, num_real_reqs=real, scaling=128 ** -0.5,
                        ).clone()
                        torch.testing.assert_close(mapped, output, rtol=0, atol=0)
                        torch.testing.assert_close(hot_k, hot_k_before, rtol=0, atol=0)
                        torch.testing.assert_close(hot_v, hot_v_before, rtol=0, atol=0)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
