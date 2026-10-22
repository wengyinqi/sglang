"""CPU checks for MPU layout, allocation, and fail-closed runtime contracts."""

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

POOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/mpu_memory_pool.py"
)
SPEC = importlib.util.spec_from_file_location("mpu_pool_cpu_test_module", POOL_PATH)
POOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POOL
SPEC.loader.exec_module(POOL)


class TestMPULayout(unittest.TestCase):
    def test_layout_matches_contiguous_host_abi(self):
        layout = POOL.MPUKVLayout(layers=3, shard_rows=7, heads=2, head_dim=8)
        tensor = torch.empty(layout.shape, dtype=torch.bfloat16)
        self.assertEqual(layout.token_stride_bytes, 32)
        self.assertEqual(layout.bytes_per_slot, 192)
        self.assertEqual(layout.shard_bytes, tensor.numel() * tensor.element_size())
        for side in range(2):
            for layer in range(3):
                for row in range(7):
                    self.assertEqual(
                        layout.byte_offset(side, layer, row),
                        tensor[side, layer, row].data_ptr() - tensor.data_ptr(),
                    )

    def test_all_slots_resolve_across_one_to_four_shards(self):
        for shards in range(1, 5):
            with self.subTest(shards=shards):
                capacity = POOL.plan_mpu_capacity(11, 2, shards)
                self.assertEqual(capacity.size, 22)
                self.assertGreaterEqual(capacity.allocated_slots, capacity.size)
                self.assertLess(capacity.allocated_slots - capacity.size, shards)
                locations = [capacity.split_slot(slot) for slot in range(capacity.size)]
                self.assertEqual(len(set(locations)), capacity.size)
                for slot, (shard, row) in enumerate(locations):
                    self.assertLess(shard, shards)
                    self.assertLess(row, capacity.shard_rows)
                    self.assertEqual(shard * capacity.shard_rows + row, slot)
                for invalid in (-1, capacity.size, capacity.allocated_slots, True):
                    with self.assertRaises(ValueError):
                        capacity.split_slot(invalid)

    def test_fractional_ratio_rounds_up_without_exposing_padding(self):
        capacity = POOL.plan_mpu_capacity(9, 1.5, 4)
        self.assertEqual(capacity.size, 14)
        self.assertEqual(capacity.shard_rows, 4)
        self.assertEqual(capacity.allocated_slots, 16)
        self.assertEqual(capacity.actual_ratio, 14 / 9)
        allocator = POOL.MPUSlotAllocator(capacity.size)
        self.assertEqual(allocator.alloc(capacity.size).tolist(), list(range(14)))
        self.assertIsNone(allocator.alloc(1))

    def test_invalid_dimensions_and_capacity(self):
        for args in (
            (0, 2, 1),
            (8, 0, 1),
            (8, -1, 1),
            (8, float("inf"), 1),
            (8, float("nan"), 1),
            (8, True, 1),
            (8, 2, 0),
            (8, 2, 5),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                POOL.plan_mpu_capacity(*args)
        with self.assertRaises(ValueError):
            POOL.MPUKVLayout(1, 0, 4, 128)
        layout = POOL.MPUKVLayout(2, 3, 4, 128)
        for args in ((2, 0, 0), (0, 2, 0), (0, 0, 3), (0, 0, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                layout.byte_offset(*args)


class TestMPUAllocator(unittest.TestCase):
    def test_exhaustion_reuse_and_metadata_ownership(self):
        allocator = POOL.MPUSlotAllocator(7)
        first = allocator.alloc(3)
        self.assertEqual(first.tolist(), [0, 1, 2])
        self.assertEqual(allocator.available_size(), 4)
        self.assertIsNone(allocator.alloc(5))
        second = allocator.alloc(4)
        self.assertEqual(second.tolist(), [3, 4, 5, 6])
        self.assertEqual(allocator.available_size(), 0)
        self.assertEqual(allocator.free(first[[2, 0]]), 2)
        self.assertEqual(allocator.alloc(2).tolist(), [2, 0])
        self.assertEqual(allocator.alloc(0).numel(), 0)
        self.assertEqual(allocator.free(torch.empty(0, dtype=torch.int64)), 0)
        # Returned indices do not alias the allocator's future free queue.
        allocator.clear()
        selected = allocator.alloc(2)
        selected.fill_(6)
        self.assertEqual(allocator.alloc(5).tolist(), [2, 3, 4, 5, 6])

    def test_failed_free_does_not_mutate_allocator(self):
        allocator = POOL.MPUSlotAllocator(5)
        allocator.alloc(3)
        for indices in (
            torch.tensor([-1]),
            torch.tensor([5]),
            torch.tensor([3]),
            torch.tensor([0, 0]),
            torch.tensor([0, 4]),
            torch.tensor([[0]]),
            torch.tensor([0.0]),
        ):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                allocator.free(indices)
            self.assertEqual(allocator.available_size(), 2)
        allocator.free(torch.tensor([0, 1, 2], dtype=torch.int32))
        self.assertEqual(allocator.available_size(), 5)
        with self.assertRaises(ValueError):
            allocator.free(torch.tensor([0]))

    def test_invalid_requests(self):
        allocator = POOL.MPUSlotAllocator(5)
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                allocator.alloc(count)
        self.assertEqual(allocator.available_size(), 5)


class TestMPURuntime(unittest.TestCase):
    def fake_device_pool(self):
        buffer = SimpleNamespace(
            device=torch.device("cuda", 0),
            dtype=torch.bfloat16,
            ndim=3,
            shape=(32, 2, 8),
            is_contiguous=lambda: True,
        )
        return SimpleNamespace(
            device="cuda:0",
            size=32,
            head_num=2,
            head_dim=8,
            layer_num=3,
            start_layer=0,
            end_layer=2,
            store_dtype=torch.bfloat16,
            k_buffer=[buffer] * 3,
            v_buffer=[buffer] * 3,
        )

    def runtime(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(POOL.sys, "platform", "linux"))
        stack.enter_context(mock.patch.object(POOL.torch.version, "cuda", "12.8"))
        stack.enter_context(
            mock.patch.object(POOL.torch.cuda, "is_available", return_value=True)
        )
        stack.enter_context(
            mock.patch.object(POOL.torch.cuda, "device_count", return_value=5)
        )
        stack.enter_context(
            mock.patch.object(
                POOL.torch.cuda.memory, "get_allocator_backend", return_value="native"
            )
        )
        peer = stack.enter_context(
            mock.patch.object(
                POOL.torch.cuda, "can_device_access_peer", return_value=True
            )
        )
        stack.enter_context(
            mock.patch.dict(
                POOL.os.environ,
                {"PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": ""},
            )
        )
        return peer

    def test_directional_access_only(self):
        peer = self.runtime()
        POOL.validate_mpu_runtime(0, [1, 2, 3, 4])
        self.assertEqual(
            peer.call_args_list, [mock.call(0, device) for device in (1, 2, 3, 4)]
        )

    def test_bad_topology_ids_fail_before_peer_probe(self):
        peer = self.runtime()
        for devices in ([], [0], [1, 1], [1, 2, 3, 4, 5], [-1], [True], [5]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                POOL.validate_mpu_runtime(0, devices)
        peer.assert_not_called()

    def test_cpu_windows_async_vmm_and_missing_peer_fail_closed(self):
        self.runtime()
        failures = (
            (mock.patch.object(POOL.sys, "platform", "win32"), "Linux"),
            (mock.patch.object(POOL.torch.version, "cuda", None), "NVIDIA CUDA"),
            (
                mock.patch.object(POOL.torch.cuda, "is_available", return_value=False),
                "NVIDIA CUDA",
            ),
            (
                mock.patch.object(
                    POOL.torch.cuda.memory,
                    "get_allocator_backend",
                    return_value="cudaMallocAsync",
                ),
                "native",
            ),
            (
                mock.patch.dict(
                    POOL.os.environ, {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
                ),
                "expandable_segments",
            ),
            (
                mock.patch.dict(
                    POOL.os.environ,
                    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments : true"},
                ),
                "expandable_segments",
            ),
            (
                mock.patch.object(
                    POOL.torch.cuda, "can_device_access_peer", return_value=False
                ),
                "cannot peer-access",
            ),
        )
        for context, message in failures:
            with self.subTest(message=message), context, self.assertRaisesRegex(
                RuntimeError, message
            ):
                POOL.validate_mpu_runtime(0, [1])

    def test_cpu_pool_rejected_without_payload_allocation(self):
        with mock.patch.object(POOL.torch, "empty") as allocate, self.assertRaisesRegex(
            ValueError, "CUDA GPU"
        ):
            POOL.MPUMHATokenToKVPool(SimpleNamespace(device="cpu"), [1])
        allocate.assert_not_called()

    def test_deterministic_empty_fill_is_rejected_before_storage_allocation(self):
        self.runtime()
        with mock.patch.object(
            POOL.torch, "are_deterministic_algorithms_enabled", return_value=True
        ), mock.patch.object(
            POOL.torch.utils,
            "deterministic",
            SimpleNamespace(fill_uninitialized_memory=True),
        ), mock.patch.object(
            POOL.torch, "empty"
        ) as allocate:
            with self.assertRaisesRegex(RuntimeError, "storage-GPU initialization"):
                POOL.MPUMHATokenToKVPool(self.fake_device_pool(), [1])
            allocate.assert_not_called()
        with mock.patch.object(
            POOL.torch, "are_deterministic_algorithms_enabled", return_value=True
        ), mock.patch.object(
            POOL.torch.utils,
            "deterministic",
            SimpleNamespace(fill_uninitialized_memory=False),
        ):
            POOL.validate_mpu_runtime(0, [1])

    def test_low_capacity_shard_fails_before_any_payload_allocation(self):
        self.runtime()
        # 64 slots / 2 shards * (K+V) * 3 layers * 2 heads * 8 dims * BF16.
        needed = 32 * 2 * 3 * 2 * 8 * 2
        with mock.patch.object(
            POOL.torch.cuda,
            "mem_get_info",
            side_effect=[(needed + 16, needed + 16), (needed + 15, needed + 15)],
        ) as info:
            with mock.patch.object(
                POOL.torch, "empty"
            ) as allocate, self.assertRaisesRegex(ValueError, "storage GPU 2"):
                POOL.MPUMHATokenToKVPool(
                    self.fake_device_pool(), [1, 2], memory_reserve_bytes=16
                )
        self.assertEqual(info.call_args_list, [mock.call(1), mock.call(2)])
        allocate.assert_not_called()

    def test_non_mha_or_invalid_layout_fails_before_memory_probe(self):
        self.runtime()
        for dtype in (torch.float32, torch.float16, torch.uint8):
            device_pool = self.fake_device_pool()
            device_pool.store_dtype = dtype
            with self.subTest(dtype=dtype), mock.patch.object(
                POOL.torch.cuda, "mem_get_info"
            ) as info, self.assertRaisesRegex(ValueError, "BF16"):
                POOL.MPUMHATokenToKVPool(device_pool, [1])
            info.assert_not_called()
        device_pool = self.fake_device_pool()
        device_pool.v_buffer[0].shape = (32, 2, 16)
        with mock.patch.object(
            POOL.torch.cuda, "mem_get_info"
        ) as info, self.assertRaisesRegex(ValueError, "contiguous BF16 MHA/GQA"):
            POOL.MPUMHATokenToKVPool(device_pool, [1])
        info.assert_not_called()
        for kwargs in ({"page_size": 16}, {"layout": "page_first"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "page_size=1"
            ):
                POOL.MPUMHATokenToKVPool(self.fake_device_pool(), [1], **kwargs)

    def test_actual_allocation_must_be_native_nonexpandable(self):
        shard = SimpleNamespace(
            device=torch.device("cuda", 1),
            data_ptr=lambda: 1024,
            numel=lambda: 128,
            element_size=lambda: 2,
        )
        native = {
            "device": 1,
            "address": 1000,
            "total_size": 300,
            "is_expandable": False,
        }
        with mock.patch.object(
            POOL.torch.cuda, "memory_snapshot", return_value=[native]
        ):
            POOL.validate_native_peer_allocations([shard])
        for segment in (
            {**native, "is_expandable": True},
            {**native, "total_size": 128},
            {**native, "device": 2},
            {key: value for key, value in native.items() if key != "is_expandable"},
        ):
            with self.subTest(segment=segment), mock.patch.object(
                POOL.torch.cuda, "memory_snapshot", return_value=[segment]
            ), self.assertRaisesRegex(RuntimeError, "non-expandable"):
                POOL.validate_native_peer_allocations([shard])

    def test_failed_close_keeps_peer_allocations_owned(self):
        pool = object.__new__(POOL.MPUMHATokenToKVPool)
        pool._closed = False
        pool.compute_device = torch.device("cuda", 0)
        pool.shards = [object()]
        pool.peer_ptrs = object()
        pool.local_ptrs = object()
        with mock.patch.object(
            POOL.torch.cuda, "synchronize", side_effect=RuntimeError("not drained")
        ):
            with self.assertRaisesRegex(RuntimeError, "not drained"):
                pool.close()
        self.assertFalse(pool._closed)
        self.assertEqual(len(pool.shards), 1)
        self.assertIsNotNone(pool.peer_ptrs)
        with mock.patch.object(POOL.torch.cuda, "synchronize"):
            pool.close()

    def test_close_drains_compute_before_releasing_pointers(self):
        pool = object.__new__(POOL.MPUMHATokenToKVPool)
        pool._closed = False
        pool.compute_device = torch.device("cuda", 0)
        pool.shards = [object(), object()]
        pool.peer_ptrs = object()
        pool.local_ptrs = object()

        def drain(device):
            self.assertEqual(device, pool.compute_device)
            self.assertEqual(len(pool.shards), 2)
            self.assertIsNotNone(pool.peer_ptrs)
            self.assertIsNotNone(pool.local_ptrs)

        with mock.patch.object(
            POOL.torch.cuda, "synchronize", side_effect=drain
        ) as sync:
            pool.close()
            pool.close()
            sync.assert_called_once()
        self.assertFalse(pool.shards)
        self.assertIsNone(pool.peer_ptrs)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.alloc(1)


if __name__ == "__main__":
    unittest.main()
