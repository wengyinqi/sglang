"""Opt-in transfer/graph test; does not load GLM or measure model performance.

Run on the eight-GPU Linux host:
  SGLANG_TEST_GLM53_MPU=1 python -m unittest discover -s test/manual/mpu_stages -p test_glm53_gpu.py
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_GLM53_MPU") == "1",
    "GPU acceptance requires explicit opt-in",
)
class TestGLMPeerGPU(unittest.TestCase):
    def test_bf16_record_roundtrip_miss_replay_padding_and_cuda_graph(self):
        self.assertEqual(sys.platform, "linux")
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
        from sglang.srt.mem_cache.mpu_glm import HiSparseMLAMPUPool

        with torch.cuda.device(0):
            local = [
                (
                    (torch.arange(192 * 512, device="cuda:0").reshape(192, 1, 512) % 23)
                    + i
                ).to(torch.bfloat16)
                for i in range(11)
            ]
            device_pool = NS(
                device="cuda:0",
                page_size=64,
                layer_num=11,
                dtype=torch.bfloat16,
                kv_cache_dim=512,
                start_layer=0,
                index_kpool=4,
                size=128,
                kv_buffer=local,
            )
            pool = HiSparseMLAMPUPool(
                device_pool, storage_device_ids=[4, 5, 6, 7], capacity_ratio=1
            )
            try:
                host = torch.tensor(
                    [0, 63, 64, 127], dtype=torch.int64, device="cuda:0"
                )
                loc = torch.tensor(
                    [64, 127, 128, 191], dtype=torch.int64, device="cuda:0"
                )
                expected = [buf[loc].clone() for buf in local]
                pool.backup_from_device_all_layer(device_pool, host, loc, "kernel")
                for i, buf in enumerate(local):
                    buf[loc] = 0
                    pool.load_to_device_per_layer(device_pool, host, loc, i, "kernel")
                    torch.testing.assert_close(buf[loc], expected[i], rtol=0, atol=0)
                src = torch.full(
                    (2, 2051), pool.size + 1, dtype=torch.int64, device="cuda:0"
                )
                dst = torch.full(
                    (2, 2051), pool.local_slots + 1, dtype=torch.int32, device="cuda:0"
                )
                src[0, :4] = host
                dst[0, :4] = loc.to(torch.int32)
                count = torch.tensor([4, 2051], dtype=torch.int32, device="cuda:0")
                real = torch.ones(1, dtype=torch.int32, device="cuda:0")
                # The padded row deliberately contains invalid addresses.
                pool.replay_misses(src, dst, count, real, 10)
                torch.cuda.synchronize(0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pool.replay_misses(src, dst, count, real, 10)
                local[10][loc] = 0
                graph.replay()
                torch.testing.assert_close(local[10][loc], expected[10], rtol=0, atol=0)
                real.zero_()
                local[10][loc] = 0
                graph.replay()
                torch.testing.assert_close(
                    local[10][loc], torch.zeros_like(expected[10]), rtol=0, atol=0
                )
                del graph
            finally:
                pool.destroy()


if __name__ == "__main__":
    unittest.main()
