"""Opt-in peer-UVA fused HiSparse acceptance, without a model download.

SGLANG_RUN_MPU_GPU_TESTS=1 PYTHONPATH=python python -m pytest -q \
    test/manual/mpu/test_mpu_resolve_gpu.py

Only the compute GPU runs tensor/kernel operations. Storage GPUs receive
torch.empty allocations; initialization, write-through and reads execute on
the compute GPU. Select 1..4 storage ordinals with
SGLANG_MPU_TEST_STORAGE_DEVICES=1,2,3,4 to include cross-shard resolution.
"""

import os
import sys
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_MPU_GPU_TESTS") == "1",
    "Set SGLANG_RUN_MPU_GPU_TESTS=1 on a Linux peer-GPU host",
)
class TestMPUResolveGPU(unittest.TestCase):
    dimensions = (2, 601, 64, 32, 300)

    @classmethod
    def setUpClass(cls):
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or torch.version.hip
        ):
            raise unittest.SkipTest("Linux NVIDIA CUDA host required")
        cls.compute = torch.device(
            "cuda", int(os.environ.get("SGLANG_MPU_TEST_COMPUTE_DEVICE", "0"))
        )
        cls.storage = [
            int(value)
            for value in os.environ.get("SGLANG_MPU_TEST_STORAGE_DEVICES", "1").split(
                ","
            )
        ]
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise unittest.SkipTest(
                "Peer allocation test requires the native CUDA allocator"
            )
        if "expandable_segments:True" in os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ).replace(" ", ""):
            raise unittest.SkipTest("expandable_segments allocations are unsupported")
        from sglang.jit_kernel.hisparse import load_cache_from_mpu_to_device_buffer_mha
        from sglang.jit_kernel.mpu_io import (
            enable_peer_access,
            transfer_peer_kv,
            zero_peer_kv,
        )

        cls.resolve_kernel = staticmethod(load_cache_from_mpu_to_device_buffer_mha)
        cls.transfer = staticmethod(transfer_peer_kv)
        cls.zero = staticmethod(zero_peer_kv)
        enable_peer_access(cls.compute.index, cls.storage)

    def setUp(self):
        torch.cuda.set_device(self.compute)
        self.layers, self.rows, self.hot, self.topk, self.seq = self.dimensions
        self.shard_rows = (self.rows + len(self.storage) - 1) // len(self.storage)
        self.slabs = [
            torch.empty(
                (2, self.layers, self.shard_rows, 2, 8),
                dtype=torch.bfloat16,
                device=f"cuda:{device}",
            )
            for device in self.storage
        ]
        self.peer_ptrs = torch.tensor(
            [
                [
                    [slab[side, layer].data_ptr() for layer in range(self.layers)]
                    for side in range(2)
                ]
                for slab in self.slabs
            ],
            dtype=torch.int64,
            device=self.compute,
        )
        self.zero(self.peer_ptrs, self.shard_rows, 32)
        self.source = (
            torch.arange(
                2 * self.layers * self.rows * 16,
                dtype=torch.float32,
                device=self.compute,
            )
            .remainder(127)
            .to(torch.bfloat16)
            .reshape(2, self.layers, self.rows, 2, 8)
        )
        self.source_ptrs = torch.tensor(
            [
                [self.source[side, layer].data_ptr() for layer in range(self.layers)]
                for side in range(2)
            ],
            dtype=torch.int64,
            device=self.compute,
        )
        self.all_rows = torch.arange(self.rows, dtype=torch.int64, device=self.compute)
        self.buffer = torch.empty(
            (2, self.layers, 2 * (self.hot + 1), 2, 8),
            dtype=torch.bfloat16,
            device=self.compute,
        )
        # Map token positions to nontrivial global backing slots. The modulus
        # in both fixtures is coprime with 7; misses span peer shard edges.
        self.mapping = (
            (torch.arange(2 * self.seq, device=self.compute, dtype=torch.int64) * 7)
            % self.rows
        ).reshape(2, self.seq)
        self.locations = torch.arange(
            2 * (self.hot + 1), dtype=torch.int32, device=self.compute
        ).reshape(2, self.hot + 1)
        self.tokens = torch.empty(
            (self.layers, 2, self.hot + 1), dtype=torch.int32, device=self.compute
        )
        self.lru = torch.empty(
            (self.layers, 2, self.hot), dtype=torch.int16, device=self.compute
        )
        hit_count = self.topk // 4
        miss_start = self.hot + 36
        self.top = torch.tensor(
            [
                list(range(hit_count))
                + list(range(miss_start, miss_start + self.topk - hit_count - 1))
                + [self.seq - 1]
            ]
            * 3,
            dtype=torch.int32,
            device=self.compute,
        )
        self.output = torch.full(
            (self.layers, 3, self.topk), -99, dtype=torch.int32, device=self.compute
        )
        # Third request is graph padding and must not access its invalid row.
        self.requests = torch.tensor(
            [0, 1, 9999], dtype=torch.int64, device=self.compute
        )
        self.lengths = torch.tensor(
            [self.seq, self.seq, 9999], dtype=torch.int32, device=self.compute
        )
        self.real_requests = torch.tensor([2], dtype=torch.int32, device=self.compute)
        self.reset_hot()
        self.publish()

    def tearDown(self):
        # Pointer-table ownership extends until all compute work has completed.
        torch.cuda.synchronize(self.compute)

    def publish(self):
        self.transfer(
            self.peer_ptrs,
            self.source_ptrs,
            self.all_rows,
            self.all_rows,
            self.shard_rows,
            32,
            True,
            local_rows=self.rows,
            peer_rows=self.rows,
        )

    def reset_hot(self):
        self.tokens[:, :, : self.hot] = torch.arange(
            self.hot, dtype=torch.int32, device=self.compute
        )
        self.tokens[:, :, self.hot] = self.seq - 1
        self.lru[:] = torch.arange(self.hot, dtype=torch.int16, device=self.compute)
        self.buffer.zero_()
        for request in range(2):
            start = request * (self.hot + 1)
            selected = torch.cat(
                (self.mapping[request, : self.hot], self.mapping[request, -1:])
            )
            self.buffer[:, :, start : start + self.hot + 1] = self.source[
                :, :, selected
            ]
        self.output.fill_(-99)

    def resolve(self):
        for layer in range(self.layers):
            self.resolve_kernel(
                self.top,
                self.tokens[layer],
                self.mapping,
                self.locations,
                self.peer_ptrs,
                self.buffer[0, layer],
                self.buffer[1, layer],
                self.output[layer],
                self.requests,
                self.lengths,
                self.lru[layer],
                item_size_bytes=32,
                num_top_k=self.topk,
                hot_buffer_size=self.hot,
                shard_rows=self.shard_rows,
                layer_id=layer,
                peer_rows=self.rows,
                num_real_reqs=self.real_requests,
            )

    def assert_payload(self, count=None):
        if count is None:
            count = self.topk
        for layer in range(self.layers):
            for request in range(2):
                logical = self.top[request, :count].long()
                physical = self.output[layer, request, :count].long()
                expected = self.source[:, layer, self.mapping[request, logical]]
                actual = self.buffer[:, layer, physical]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    self.lru[layer, request].sort().values,
                    torch.arange(self.hot, dtype=torch.int16, device=self.compute),
                    rtol=0,
                    atol=0,
                )
        self.assertTrue(torch.all(self.output[:, 2] == -99).item())

    def test_misses_hits_evictions_newest_and_int64_lengths(self):
        self.resolve()
        self.assert_payload()
        first_locations = self.output.clone()
        first_lru = self.lru.clone()
        self.resolve()  # All selected tokens now hit, including the newest slot.
        self.assert_payload()
        torch.testing.assert_close(self.output, first_locations, rtol=0, atol=0)
        torch.testing.assert_close(self.lru, first_lru, rtol=0, atol=0)
        self.top[:2] = torch.arange(
            self.seq - self.topk, self.seq, dtype=torch.int32, device=self.compute
        )
        self.lengths = self.lengths.to(torch.int64)
        self.resolve()  # Evict previous selections while retaining newest slot.
        self.assert_payload()
        for request in range(2):
            self.assertTrue(
                torch.all(
                    self.output[:, request, -1] == self.locations[request, self.hot]
                ).item()
            )

    def test_short_sequence_uses_resident_cache(self):
        self.lengths[:2] = 15
        self.top[:2].fill_(-1)
        self.top[:2, :15] = torch.arange(15, dtype=torch.int32, device=self.compute)
        self.resolve()
        self.assert_payload(15)
        self.assertTrue(torch.all(self.output[:, :2, 15:] == -99).item())

    def test_graph_replay_after_peer_write_through(self):
        warmup = torch.cuda.Stream(device=self.compute)
        warmup.wait_stream(torch.cuda.current_stream(self.compute))
        with torch.cuda.stream(warmup):
            self.resolve()  # JIT compilation must happen before capture.
        torch.cuda.current_stream(self.compute).wait_stream(warmup)
        self.reset_hot()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.resolve()
        for _ in range(3):
            self.source.add_(1)
            self.publish()
            self.reset_hot()
            graph.replay()
            self.assert_payload()
            graph.replay()  # Replaying again takes the all-hit path.
            self.assert_payload()


class TestMPUResolvePaperBudgetGPU(TestMPUResolveGPU):
    """Exercise many scan iterations at paper-sized top-k and larger hot cache.

    The small fixture fits its chunk counters in one scan iteration. This one
    catches corrupted LRU/miss compaction that occurs only after iteration 1.
    """

    dimensions = (2, 20001, 4096, 2048, 10000)


class TestMPUResolveLargeHotCacheGPU(TestMPUResolveGPU):
    """Include the opt-in dynamic shared-memory (>48 KiB) launch path."""

    dimensions = (2, 28001, 8192, 2048, 14000)


if __name__ == "__main__":
    unittest.main()
