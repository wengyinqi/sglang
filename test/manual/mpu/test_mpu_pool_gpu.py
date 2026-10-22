"""Opt-in Linux peer-GPU pool checks; no model download is required.

SGLANG_RUN_MPU_GPU_TESTS=1 PYTHONPATH=python python -m pytest -q \
    test/manual/mpu/test_mpu_pool_gpu.py

Defaults to compute ordinal 0, storage ordinal 1. Set
SGLANG_MPU_TEST_STORAGE_DEVICES=1,2,3,4 for the future 1+4 topology.
Every KV initialization/copy/check is issued on the compute GPU.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_MPU_GPU_TESTS") == "1",
    "Set SGLANG_RUN_MPU_GPU_TESTS=1 on a Linux peer-GPU host",
)
class TestMPUPeerPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or torch.version.cuda is None
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
        from sglang.srt.mem_cache.mpu_memory_pool import MPUMHATokenToKVPool

        cls.pool_type = MPUMHATokenToKVPool

    def setUp(self):
        torch.cuda.set_device(self.compute)
        rows, heads, dim, layers = 32, 2, 8, 3
        self.device_pool = SimpleNamespace(
            device=str(self.compute),
            size=rows,
            head_num=heads,
            head_dim=dim,
            layer_num=layers,
            start_layer=0,
            end_layer=layers - 1,
            store_dtype=torch.bfloat16,
            k_buffer=[
                torch.zeros(
                    (rows, heads, dim), dtype=torch.bfloat16, device=self.compute
                )
                for _ in range(layers)
            ],
            v_buffer=[
                torch.zeros(
                    (rows, heads, dim), dtype=torch.bfloat16, device=self.compute
                )
                for _ in range(layers)
            ],
        )
        self.pool = self.pool_type(
            self.device_pool,
            self.storage,
            host_to_device_ratio=2,
            memory_reserve_bytes=0,
        )
        self.addCleanup(self.pool.close)

    def boundary_slots(self):
        slots = {0, self.pool.size - 1}
        for shard in range(1, len(self.storage)):
            slots.update(
                (shard * self.pool.shard_rows - 1, shard * self.pool.shard_rows)
            )
        return sorted(slots)

    def test_peer_zero_and_shard_pointer_layout(self):
        host = torch.tensor(self.boundary_slots(), device=self.compute)
        target = torch.arange(host.numel(), device=self.compute)
        for layer in range(self.device_pool.layer_num):
            self.device_pool.k_buffer[layer].fill_(3)
            self.device_pool.v_buffer[layer].fill_(-3)
            self.pool.load_to_device_per_layer(self.device_pool, host, target, layer)
        torch.cuda.synchronize(self.compute)
        for buffer in self.device_pool.k_buffer + self.device_pool.v_buffer:
            torch.testing.assert_close(
                buffer[target], torch.zeros_like(buffer[target]), rtol=0, atol=0
            )
        ptrs = self.pool.peer_ptrs.cpu()
        layout = self.pool.layout_descriptor
        for shard_index, shard in enumerate(self.pool.shards):
            self.assertEqual(tuple(shard.shape), layout.shape)
            for side in range(2):
                for layer in range(layout.layers):
                    self.assertEqual(
                        ptrs[shard_index, side, layer].item(),
                        shard.data_ptr() + layout.byte_offset(side, layer, 0),
                    )

    def test_all_layer_store_and_per_layer_load_are_bit_exact(self):
        host = torch.tensor(self.boundary_slots(), device=self.compute)
        source = torch.arange(host.numel(), device=self.compute) + 3
        target = torch.arange(host.numel(), device=self.compute).flip(0) + 16
        expected = []
        for index, buffer in enumerate(
            self.device_pool.k_buffer + self.device_pool.v_buffer
        ):
            values = torch.arange(buffer.numel(), device=self.compute).reshape(
                buffer.shape
            )
            buffer.copy_(((values % 127) - 63 + index * 2).to(torch.bfloat16))
            expected.append(buffer[source].clone())
        stream = torch.cuda.Stream(device=self.compute)
        stream.wait_stream(torch.cuda.current_stream(self.compute))
        with torch.cuda.stream(stream):
            self.pool.backup_from_device_all_layer(self.device_pool, host, source)
            for buffer in self.device_pool.k_buffer + self.device_pool.v_buffer:
                buffer.zero_()
            for layer in range(self.device_pool.layer_num):
                self.pool.load_to_device_per_layer(
                    self.device_pool, host, target, layer
                )
        # close must join the nondefault stream before releasing peer storage.
        self.pool.close()
        for buffer, values in zip(
            self.device_pool.k_buffer + self.device_pool.v_buffer, expected
        ):
            torch.testing.assert_close(buffer[target], values, rtol=0, atol=0)
            torch.testing.assert_close(
                buffer[0], torch.zeros_like(buffer[0]), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
