"""Real SM90 mapped-DDR PlanOnly/TMA attention/refill/Graph acceptance.

SGLANG_RUN_XM_TMA_GPU_TESTS=1 PYTHONPATH=python python -m unittest \
    discover -s test/manual/xm_sparse -p test_fused_hisparse_tma_gpu.py -v

Qwen BF16 GQA Q32/KV4/D128, top_k=2048. Thirty-two-row TMA tiles
reuse both mbarrier phases. Raw refill and FP32 reference checks are retained.
No model required.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_XM_TMA_GPU_TESTS") == "1",
    "Set SGLANG_RUN_XM_TMA_GPU_TESTS=1 on a Linux SM90 host",
)
class TestFusedHiSparseTMAGPU(unittest.TestCase):
    batch = 3
    requests_count = 2
    topk = 2048
    hot = 2048
    rows = 8001
    seq = 6000

    @classmethod
    def setUpClass(cls):
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or torch.version.hip
            or torch.cuda.get_device_capability() != (9, 0)
        ):
            raise unittest.SkipTest("Linux NVIDIA SM90 CUDA host required")
        from sglang.jit_kernel.fused_hisparse_tma import FusedHiSparseTMAWorkspace
        from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mha

        cls.workspace_cls = FusedHiSparseTMAWorkspace
        cls.baseline_loader = staticmethod(load_cache_to_device_buffer_mha)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def setUp(self):
        torch.manual_seed(418)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.host = torch.empty(
            (2, 1, self.rows, 4, 128), dtype=torch.bfloat16, pin_memory=True
        )
        self.host.copy_(torch.randn(self.host.shape, dtype=torch.float32).to(torch.bfloat16))
        self.hot_buffer = torch.zeros(
            (2, 1, self.requests_count * (self.hot + 1), 4, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.reference_buffer = torch.empty_like(self.hot_buffer)
        self.mapping = (
            torch.arange(self.requests_count * self.seq, dtype=torch.int64, device=self.device)
            .mul(7)
            .remainder(self.rows)
            .reshape(self.requests_count, self.seq)
        )
        self.locations = torch.arange(
            self.requests_count * (self.hot + 1), dtype=torch.int32, device=self.device
        ).reshape(self.requests_count, self.hot + 1)
        self.tokens = torch.empty(
            (1, self.requests_count, self.hot + 1), dtype=torch.int32, device=self.device
        )
        self.lru = torch.empty(
            (1, self.requests_count, self.hot), dtype=torch.int16, device=self.device
        )
        self.reference_tokens = torch.empty_like(self.tokens)
        self.reference_lru = torch.empty_like(self.lru)
        self.reference_locations = torch.full(
            (self.batch, self.topk), -1, dtype=torch.int32, device=self.device
        )
        self.metrics = torch.zeros((1, 5), dtype=torch.int64, device=self.device)
        self.reference_metrics = torch.zeros(5, dtype=torch.int64, device=self.device)
        self.requests = torch.tensor([0, 1, 99999], dtype=torch.int64, device=self.device)
        self.lengths = torch.tensor(
            [self.seq, self.seq, 99999], dtype=torch.int32, device=self.device
        )
        self.real = torch.tensor([2], dtype=torch.int32, device=self.device)
        self.actual = torch.tensor([self.topk, self.topk, 1], dtype=torch.int32, device=self.device)
        self.top = torch.zeros((self.batch, self.topk), dtype=torch.int32, device=self.device)
        self.q = torch.randn((self.batch, 32, 128), device=self.device, dtype=torch.bfloat16)
        self.workspace = self.workspace_cls(
            device=self.device,
            max_bs=self.batch,
            top_k=self.topk,
            hot_buffer_size=self.hot,
            item_size_bytes=1024,
            host_k_layers=tuple(self.host[0]),
            host_v_layers=tuple(self.host[1]),
            collect_metrics=True,
        )
        self.coord = SimpleNamespace(
            req_device_buffer_tokens=self.tokens,
            req_to_host_pool=self.mapping,
            req_device_buffer_token_locs=self.locations.unsqueeze(0),
            mem_pool_device=SimpleNamespace(
                k_buffer=self.hot_buffer[0], v_buffer=self.hot_buffer[1]
            ),
            lru_slots=self.lru,
            num_real_reqs=self.real,
            cache_metrics=self.metrics,
        )
        self.select("mixed")
        self.reset_hot()

    def tearDown(self):
        # Retain mapped host allocations through completion of every TMA.
        torch.cuda.synchronize(self.device)

    def select(self, case):
        if case == "hits":
            selected = list(range(self.topk - 1)) + [self.seq - 1]
        elif case == "misses":
            selected = list(range(2500, 2500 + self.topk - 1)) + [self.seq - 1]
        elif case == "mixed":
            selected = list(range(512)) + list(range(2500, 2500 + self.topk - 513)) + [self.seq - 1]
        elif case == "short":
            selected = list(range(17)) + [-1] * (self.topk - 17)
            self.lengths[:2] = 17
            self.actual[:2] = 17
        else:
            raise ValueError(case)
        self.top[:2] = torch.tensor(selected, dtype=torch.int32, device=self.device)

    def reset_hot(self):
        self.tokens[:, :, :self.hot] = torch.arange(self.hot, dtype=torch.int32, device=self.device)
        self.tokens[:, :, self.hot] = self.seq - 1
        self.lru[:] = torch.arange(self.hot, dtype=torch.int16, device=self.device)
        for request in range(self.requests_count):
            selected = torch.cat((self.mapping[request, :self.hot], self.mapping[request, -1:]))
            source = self.host[:, :, selected.cpu()].to(self.device)
            start = request * (self.hot + 1)
            self.hot_buffer[:, :, start:start + self.hot + 1].copy_(source)
        self.reference_buffer.copy_(self.hot_buffer)
        self.reference_tokens.copy_(self.tokens)
        self.reference_lru.copy_(self.lru)
        self.metrics.zero_()
        self.reference_metrics.zero_()
        self.reference_locations.fill_(-1)

    def plan(self):
        self.workspace.resolve(
            self.coord,
            top_k_tokens=self.top,
            req_pool_indices=self.requests,
            seq_lens=self.lengths,
            layer_id=0,
        )

    def fused(self, cap=0.0):
        return self.workspace.run(
            self.q,
            hot_k=self.hot_buffer[0, 0],
            hot_v=self.hot_buffer[1, 0],
            layer_id=0,
            actual_lens=self.actual,
            num_real_reqs=self.real,
            scaling=128 ** -0.5,
            logits_soft_cap=cap,
        )

    def baseline(self):
        self.baseline_loader(
            self.top,
            self.reference_tokens[0],
            self.mapping,
            self.locations,
            self.host[0, 0],
            self.host[1, 0],
            self.reference_buffer[0, 0],
            self.reference_buffer[1, 0],
            self.reference_locations,
            self.requests,
            self.lengths,
            self.reference_lru[0],
            item_size_bytes=1024,
            num_top_k=self.topk,
            hot_buffer_size=self.hot,
            block_size=1024,
            num_real_reqs=self.real,
            cache_metrics=self.reference_metrics,
        )

    def check(self, output, cap=0.0):
        self.baseline()
        for actual, expected in (
            (self.workspace.device_locs, self.reference_locations),
            (self.tokens, self.reference_tokens),
            (self.lru, self.reference_lru),
            (self.hot_buffer, self.reference_buffer),
            (self.metrics[0], self.reference_metrics),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        max_error = 0.0
        for request in range(int(self.real.item())):
            count = int(self.actual[request].item())
            physical = self.reference_locations[request, :count].long()
            keys = self.reference_buffer[0, 0, physical].float().repeat_interleave(8, dim=1)
            values = self.reference_buffer[1, 0, physical].float().repeat_interleave(8, dim=1)
            scores = torch.einsum("hd,khd->hk", self.q[request].float(), keys) * (128 ** -0.5)
            if cap:
                scores = cap * torch.tanh(scores / cap)
            expected = torch.einsum("hk,khd->hd", scores.softmax(-1), values)
            max_error = max(max_error, float((output[request].float() - expected).abs().max().item()))
            torch.testing.assert_close(output[request].float(), expected, atol=0.003, rtol=0.02)
        self.assertTrue(torch.all(output[int(self.real.item()):] == 0).item())
        print(f"FP32 reference max_abs_error={max_error:.8f}", flush=True)

    def test_all_hits_preserve_residency(self):
        self.select("hits")
        self.plan()
        self.check(self.fused())
        self.assertEqual(self.metrics[0, 4].item(), 0)

    def test_mapped_misses_phase_reuse_and_exact_refill(self):
        self.select("misses")
        before = self.hot_buffer.clone()
        self.plan()
        # Resolve has updated metadata but has performed zero payload I/O.
        torch.testing.assert_close(self.hot_buffer, before, rtol=0, atol=0)
        self.check(self.fused())
        self.assertEqual(self.metrics[0, 4].item(), 2 * (self.topk - 1))

    def test_mixed_newest_softcap_and_repeated_hits(self):
        self.plan()
        self.check(self.fused(cap=3.0), cap=3.0)
        self.metrics.zero_()
        self.reference_metrics.zero_()
        self.plan()
        self.check(self.fused())
        self.assertEqual(self.metrics[0, 4].item(), 0)
        torch.testing.assert_close(self.workspace.device_locs[:2, -1], self.locations[:, self.hot], rtol=0, atol=0)

    def test_short_sequence_and_invalid_padded_request(self):
        self.select("short")
        self.lengths = self.lengths.to(torch.int64)
        self.plan()
        self.check(self.fused())
        self.assertTrue(torch.all(self.workspace.device_locs[:2, 17:] == -1).item())

    def test_compute_gpu_stores_to_mapped_ddr_are_visible_to_tma(self):
        # Keep optional CUDA dependencies out of module import on CPU hosts,
        # but expose tl to Triton's source compiler through module globals.
        global triton, tl
        import triton
        import triton.language as tl

        @triton.jit
        def store_mapped(source, destination_ptrs, side_elements: tl.constexpr):
            offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
            side = tl.program_id(1)
            destination = tl.load(destination_ptrs + side).to(tl.pointer_type(tl.bfloat16))
            values = tl.load(source + side * side_elements + offsets, offsets < side_elements, 0)
            tl.store(destination + offsets, values, offsets < side_elements)

        self.select("misses")
        self.reset_hot()
        fresh = self.host.to(self.device).add_(0.5)
        destinations = torch.tensor(
            list(self.workspace._sources[0][:2]), dtype=torch.int64, device=self.device
        )
        # A compute-GPU global store writes the mapped host backing, then TMA
        # consumes it on the same stream. There is no CPU data copy or host
        # synchronize between the producer and the fused consumer.
        side_elements = self.rows * 4 * 128
        store_mapped[(triton.cdiv(side_elements, 256), 2)](
            fresh, destinations, side_elements
        )
        self.plan()
        self.check(self.fused())
        for request in range(2):
            physical = self.workspace.device_locs[request, :-1].long()
            backing_rows = self.mapping[request, self.top[request, :-1].long()]
            torch.testing.assert_close(
                self.hot_buffer[:, 0, physical], fresh[:, 0, backing_rows], rtol=0, atol=0
            )

    def test_graph_replay_changed_sources_selection_lengths_and_real_batch(self):
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup):
            for _ in range(3):
                self.plan()
                self.fused()
        torch.cuda.current_stream(self.device).wait_stream(warmup)
        torch.cuda.synchronize(self.device)
        self.reset_hot()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.plan()
            output = self.fused()
        pointers = (self.workspace.device_locs.data_ptr(), self.workspace.miss_host_locs.data_ptr(), output.data_ptr())
        for case, real in (("misses", 2), ("hits", 1), ("short", 2), ("mixed", 2)):
            torch.cuda.synchronize(self.device)
            # CPU producer changes the retained pinned source only after the
            # preceding TMA is complete. Tests phase reuse and fresh bytes.
            self.host.add_(0.125)
            self.lengths[:2] = self.seq
            self.actual[:2] = self.topk
            self.real[0] = real
            self.select(case)
            self.reset_hot()
            graph.replay()
            self.check(output)
            self.assertEqual(pointers, (self.workspace.device_locs.data_ptr(), self.workspace.miss_host_locs.data_ptr(), output.data_ptr()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
