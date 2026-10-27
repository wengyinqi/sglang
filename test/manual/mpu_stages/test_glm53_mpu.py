"""GLM MPU production contracts using CPU tensors and mocked CUDA boundaries."""

import math
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from test_glm53_cpu import (
    COORDINATOR,
    definition,
    real_paged_allocator,
    DevicePool,
    Hybrid,
)

POOL = "python/sglang/srt/mem_cache/mpu_glm.py"
PEER = "python/sglang/srt/mem_cache/mpu_peer.py"


def options():
    return definition(POOL, "glm_backing_options", {"math": math})


class TestMPUConfig(unittest.TestCase):
    def test_payload_or_float_tensors_are_not_accepted_as_indices(self):
        validate = definition(POOL, "validate_metadata_indices")
        validate(torch.tensor([1], dtype=torch.int32), torch.device("cuda:0"))
        for invalid in (
            torch.ones(1),
            torch.ones((1, 1), dtype=torch.int64),
            NS(ndim=1, dtype=torch.int64, device=torch.device("cuda:4")),
        ):
            with self.assertRaisesRegex(ValueError, "integer vectors"):
                validate(invalid, torch.device("cuda:0"))

    def test_bare_cuda_resolves_current_compute_ordinal_and_checks_rank(self):
        resolve = definition(POOL, "resolve_compute_ordinal")
        with patch.object(torch.cuda, "current_device", return_value=2):
            self.assertEqual(resolve("cuda", tp_rank=2), 2)
            self.assertEqual(resolve("cuda:1", tp_rank=1), 1)
            with self.assertRaisesRegex(ValueError, "process rank"):
                resolve("cuda", tp_rank=0)
            with self.assertRaisesRegex(ValueError, "CUDA"):
                resolve("cpu")

    def test_defaults_and_exact_pairing(self):
        parse = options()
        self.assertEqual(parse({}, tp_size=4, compute_device=0), ("cpu", [], None, 0))
        self.assertEqual(
            parse({"backing_store": "mpu"}, tp_size=4, compute_device=3),
            ("mpu", [4, 5, 6, 7], 2.0, 1 << 30),
        )

    def test_invalid_configuration_has_no_fallback(self):
        parse = options()
        for extra, tp, ordinal in [
            ({"backing_store": "hbm"}, 4, 0),
            ({"mpu_capacity_ratio": 2}, 4, 0),
            ({"backing_store": "mpu"}, 8, 0),
            ({"backing_store": "mpu"}, 4, 4),
            ({"backing_store": "mpu", "mpu_device_ids": [0, 5, 6, 7]}, 4, 0),
            ({"backing_store": "mpu", "mpu_capacity_ratio": 0}, 4, 0),
            ({"backing_store": "mpu", "mpu_capacity_ratio": math.inf}, 4, 0),
            ({"backing_store": "mpu", "mpu_memory_reserve_bytes": True}, 4, 0),
            ({"backing_store": "mpu", "mpu_capacity_gb": 16}, 4, 0),
        ]:
            with self.subTest(extra=extra, tp=tp, ordinal=ordinal), self.assertRaises(
                ValueError
            ):
                parse(extra, tp_size=tp, compute_device=ordinal)


class TestBackingCapacity(unittest.TestCase):
    def test_retained_metadata_covers_index_tails_and_all_kda_state_without_copy(self):
        get_tensors = definition(
            COORDINATOR,
            "get_retained_gpu_pool_tensors",
            {"HybridLinearKVPool": Hybrid},
            "HiSparseCoordinator",
        )
        wrapper = Hybrid()
        index, tail, score, conv, state, ring, pos = [torch.zeros(4) for _ in range(7)]
        wrapper.full_kv_pool = NS(
            kv_buffer=[torch.zeros((8,1,512),dtype=torch.bfloat16)],
            index_k_with_scale_buffer=[index],
            _compress_tail_k=[tail],
            _compress_tail_score=[score],
        )
        wrapper.mamba_pool = NS(
            mamba_cache=NS(
                conv=[conv], temporal=state, replayssm_d=ring, replayssm_g=None
            ),
            replayssm_write_pos=pos,
        )
        result = get_tensors(
            NS(token_to_kv_pool_allocator=NS(get_kvcache=lambda: wrapper))
        )
        self.assertIs(result["dsa.index_k_with_scale_buffer"][0], index)
        self.assertIs(result["dsa.compress_tail_score"][0], score)
        self.assertIs(result["kda.state"]["conv"][0], conv)
        self.assertIs(result["kda.state"]["temporal"], state)
        self.assertIs(result["kda.state"]["replayssm_d"], ring)
        self.assertIs(result["kda.slot_metadata"]["replayssm_write_pos"], pos)

    def test_serving_capacity_follows_logical_limit(self):
        get_capacity = definition(
            "python/sglang/srt/model_executor/model_runner.py",
            "logical_max_total_num_tokens",
            owner="ModelRunner",
        )
        runner = NS(
            enable_hisparse=True,
            model_config=NS(hf_config=NS(model_type="glm5_next")),
            req_to_token_pool=NS(schedulable_token_capacity=lambda x: x),
            kv_cache_configurator=NS(
                logical_token_capacity=lambda **kw: kw["max_total_num_tokens"]
            ),
            token_to_kv_pool_allocator=NS(size_full=64),
            max_total_num_tokens=128,
        )
        self.assertEqual(get_capacity.fget(runner), 64)
        runner.token_to_kv_pool_allocator.size_full = 256
        self.assertEqual(get_capacity.fget(runner), 128)

    def test_smaller_backing_caps_history_without_resizing_index_addresses(self):
        alloc = real_paged_allocator()(
            128, 64, torch.bfloat16, "cpu", DevicePool(), True, 2
        )
        original = alloc.full_to_hisparse_device_index_mapping
        alloc.limit_logical_capacity(64)
        self.assertIs(alloc.full_to_hisparse_device_index_mapping, original)
        self.assertEqual(original.numel(), 321)
        self.assertEqual(alloc.size_full, 64)
        self.assertEqual(alloc.hisparse_attn_allocator.size, 128)
        zero, length = torch.tensor([0]), torch.tensor([64])
        indices = alloc.alloc_extend(zero, zero, length, length, torch.tensor([-1]), 64)
        self.assertIsNotNone(indices)
        with self.assertRaisesRegex(RuntimeError, "live requests"):
            alloc.limit_logical_capacity(128)
        self.assertIsNone(
            alloc.alloc_decode(torch.tensor([65]), torch.tensor([65]), indices[-1:])
        )
        alloc.free(indices)
        self.assertEqual(alloc.logical_attn_allocator.available_size(), 64)

    def test_larger_backing_cannot_invent_index_address_capacity(self):
        alloc = real_paged_allocator()(
            128, 64, torch.bfloat16, "cpu", DevicePool(), False, 2
        )
        alloc.limit_logical_capacity(512)
        self.assertEqual(alloc.size_full, 256)


class TestPeerAdapter(unittest.TestCase):
    def make_pool(self):
        # Fake only allocation/device boundaries; metadata page allocation,
        # adapter layout, pointer table and transfer calls execute production code.
        cuda = NS(synchronize=Mock(), current_device=Mock(return_value=1))
        tensor = Mock(
            side_effect=lambda data, **kw: torch.tensor(
                data, dtype=kw["dtype"], device="cpu"
            )
        )
        torch_boundary = NS(
            device=torch.device,
            bfloat16=torch.bfloat16,
            int64=torch.int64,
            tensor=tensor,
            cuda=cuda,
        )
        slots = definition(PEER, "MPUSlotAllocator", {"threading": threading})
        mixin = definition(
            "python/sglang/srt/mem_cache/pool_host/hisparse.py",
            "HiSparseHostPoolMixin",
            {"logger": Mock()},
        )
        peer = Mock(
            side_effect=lambda c, s, shape, dtype, reserve: NS(
                tensor=torch.zeros(shape, dtype=dtype), close=Mock()
            )
        )
        transfer, replay = Mock(), Mock()
        pool_class = definition(
            POOL,
            "HiSparseMLAMPUPool",
            {
                "torch": torch_boundary,
                "math": math,
                "glm_backing_options": options(),
                "validate_metadata_indices": definition(
                    POOL, "validate_metadata_indices"
                ),
                "resolve_compute_ordinal": definition(
                    POOL, "resolve_compute_ordinal", {"torch": torch_boundary}
                ),
                "HiSparseHostPoolMixin": mixin,
                "MPUSlotAllocator": slots,
                "PeerAllocation": peer,
                "transfer_records": transfer,
                "replay_misses": replay,
            },
        )
        buffers = [
            NS(
                device=torch.device("cuda:1"),
                dtype=torch.bfloat16,
                shape=(192, 1, 512),
                is_contiguous=lambda: True,
                data_ptr=lambda i=i: 8192 + i * 192 * 1024,
            )
            for i in range(11)
        ]
        device_pool = NS(
            device="cuda",
            page_size=64,
            layer_num=11,
            dtype=torch.bfloat16,
            kv_cache_dim=512,
            start_layer=0,
            index_kpool=4,
            size=128,
            kv_buffer=buffers,
        )
        pool = pool_class(
            device_pool, storage_device_ids=[4, 5, 6, 7], capacity_ratio=2.0
        )
        return pool, device_pool, peer, transfer, replay, cuda, tensor

    def test_layout_allocation_and_compute_pointer_tables(self):
        pool, _, peer, _, _, _, metadata = self.make_pool()
        self.assertEqual(
            peer.call_args.args, (1, 5, (11, 256, 1, 512), torch.bfloat16, 1 << 30)
        )
        self.assertEqual(pool.kv_buffer.shape, (11, 256, 1, 512))
        self.assertEqual(pool.token_stride_size, 1024)
        self.assertEqual(pool.storage_device_ids, [5])
        self.assertEqual(metadata.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["device"] == torch.device("cuda:1")
                for call in metadata.call_args_list
            )
        )

    def test_chunk_append_allocator_exhaustion_free_and_reuse(self):
        pool, *_ = self.make_pool()
        mapping = torch.full((2, 320), -1, dtype=torch.int64)
        allocated = torch.zeros(2, dtype=torch.int64)
        first = pool.alloc_paged_token_slots(mapping, allocated, 0, 0, 3).clone()
        pool.alloc_paged_token_slots(mapping, allocated, 0, 3, 127)
        self.assertEqual(allocated[0].item(), 192)
        torch.testing.assert_close(first, mapping[0, :3])
        self.assertEqual(pool.available_size(), 64)
        pool.alloc_paged_token_slots(mapping, allocated, 1, 0, 64)
        self.assertIsNone(pool.alloc(64))
        owned = pool.allocated_host_indices(mapping, 0, allocated[0])
        pool.free(owned)
        with self.assertRaises(ValueError):
            pool.free(owned)
        pool.free(pool.allocated_host_indices(mapping, 1, allocated[1]))
        self.assertEqual(pool.available_size(), 256)
        self.assertEqual(pool.alloc(256).numel(), 256)

    def test_transfers_use_compute_metadata_and_close_waits_before_release(self):
        pool, device_pool, peer, transfer, replay, cuda, _ = self.make_pool()
        # Adapter metadata is CPU-backed in this portable boundary test.
        pool.device = torch.device("cpu")
        indices = torch.tensor([2, 4], dtype=torch.int64)
        pool.backup_from_device_all_layer(device_pool, indices, indices + 64, "kernel")
        self.assertEqual(transfer.call_args.kwargs["layer_id"], -1)
        self.assertTrue(transfer.call_args.kwargs["to_peer"])
        pool.load_to_device_per_layer(device_pool, indices, indices + 64, 10, "kernel")
        self.assertFalse(transfer.call_args.kwargs["to_peer"])
        self.assertEqual(transfer.call_args.kwargs["peer_slots"], 256)
        self.assertEqual(transfer.call_args.kwargs["local_slots"], 192)
        pool.replay_misses(
            torch.zeros((1, 2051), dtype=torch.int64),
            torch.zeros((1, 2051), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            10,
        )
        self.assertEqual(replay.call_args.kwargs["layer_id"], 10)
        events = []
        cuda.synchronize.side_effect = lambda *a: events.append("sync")
        allocation = pool._peer
        allocation.close.side_effect = lambda: events.append("close")
        pool.destroy()
        pool.destroy()
        self.assertEqual(events, ["sync", "close"])
        self.assertIsNone(pool.kv_buffer)
        with self.assertRaises(RuntimeError):
            pool.alloc(64)


class TestCoordinatorPeerPlan(unittest.TestCase):
    def test_startup_constructs_only_peer_backing_and_caps_logical_slots(self):
        wrapper = Hybrid()
        wrapper.full_kv_pool = DevicePool()
        wrapper.full_attention_layer_id_mapping = {
            layer: i for i, layer in enumerate(range(3, 44, 4))
        }
        child = wrapper.full_kv_pool
        child.page_size = 64
        child.layer_num = 11
        child.index_kpool = 4
        allocator_cls = real_paged_allocator()
        allocator = allocator_cls(128, 64, torch.bfloat16, "cpu", wrapper, False, 2)
        host_constructor = Mock(
            side_effect=AssertionError("CPU KV allocation is forbidden")
        )
        peer_constructor = Mock(return_value=NS(size=128, token_stride_size=1024))
        coordinator = NS(
            enable_prefetch=False,
            _init_shared_index_prefetch=Mock(),
            _allocate_miss_plan=Mock(),
        )
        init = definition(
            COORDINATOR,
            "__init__",
            {
                "envs": NS(SGLANG_DEBUG_HISPARSE_SKIP_IO=NS(get=lambda: False)),
                "_is_xpu": False,
                "get_parallel": lambda: NS(tp_size=4),
                "HybridLinearKVPool": Hybrid,
                "DeepSeekV4HiSparseTokenToKVPoolAllocator": type("DSV4", (), {}),
                "MiniMaxSparseKVPool": type("M3", (), {}),
                "HiSparseTokenToKVPoolAllocator": allocator_cls,
                "MLATokenToKVPoolHost": host_constructor,
                "device_module": NS(Stream=Mock(), Event=Mock()),
                "HiSparseSpecSwapManager": lambda *a, **kw: NS(scratch_capacity=0),
            },
            "HiSparseCoordinator",
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.mpu_glm": NS(
                    glm_backing_options=lambda *a, **kw: (
                        "mpu",
                        [4, 5, 6, 7],
                        1.0,
                        1 << 30,
                    ),
                    HiSparseMLAMPUPool=peer_constructor,
                    resolve_compute_ordinal=lambda *a, **kw: 0,
                ),
                "sglang.srt.distributed": NS(get_tensor_model_parallel_rank=lambda: 0),
                "sglang.srt.mem_cache.sparsity": NS(
                    parse_hisparse_config=lambda: NS(
                        sparse_extra_config={"backing_store": "mpu"}
                    )
                ),
            },
        ), patch.object(torch.distributed, "get_world_size", return_value=4):
            init(
                coordinator,
                NS(req_to_token=torch.zeros((2, 8192)), max_context_len=8192),
                allocator,
                2048,
                4096,
                "cpu",
                object(),
            )
        host_constructor.assert_not_called()
        peer_constructor.assert_called_once()
        self.assertTrue(coordinator.is_mpu)
        self.assertEqual(coordinator.top_k, 2051)
        self.assertEqual(allocator.size_full, 128)
        self.assertEqual(coordinator.logical_address_capacity, 256)
        coordinator._allocate_miss_plan.assert_called_once_with(2)

    def coordinator(self):
        return NS(
            is_mpu=True,
            is_dsv4_hisparse=False,
            skip_io=False,
            top_k=2051,
            num_tail_slots=3,
            mem_pool_host=NS(replay_misses=Mock()),
            mem_pool_device=NS(kv_buffer=[object()] * 11),
            _miss_src=torch.zeros((2, 2051), dtype=torch.int64),
            _miss_dst=torch.zeros((2, 2051), dtype=torch.int32),
            _miss_count=torch.zeros(2, dtype=torch.int32),
            num_real_reqs=torch.ones(1, dtype=torch.int32),
            req_device_buffer_tokens=[object()] * 11,
            req_device_buffer_token_locs=[object()] * 11,
            req_to_host_pool=object(),
            top_k_device_locs_buffer=torch.full((2, 2051), -1, dtype=torch.int32),
            lru_slots=[object()] * 11,
            item_size_bytes=1024,
            device_buffer_size=4096,
            swap_in_block_size=960,
        )

    def test_swap_only_plans_then_replays_peer_misses(self):
        c = self.coordinator()
        planner = Mock()
        swap = definition(
            COORDINATOR,
            "_run_swap_in_kernel",
            {"_is_xpu": False, "load_cache_to_device_buffer_mla": planner},
            "HiSparseCoordinator",
        )
        result = swap(
            c,
            torch.tensor([0]),
            torch.tensor([5000]),
            torch.zeros((1, 2051), dtype=torch.int32),
            10,
        )
        self.assertEqual(result.shape, (1, 2051))
        self.assertTrue(planner.call_args.kwargs["skip_io"])
        self.assertIs(
            planner.call_args.kwargs["host_cache"], c.mem_pool_device.kv_buffer[10]
        )
        self.assertEqual(planner.call_args.kwargs["num_tail_slots"], 3)
        self.assertEqual(c.mem_pool_host.replay_misses.call_args.args[-1], 10)

    def test_prefetch_uses_peer_replay(self):
        c = self.coordinator()
        host_copy = Mock()
        copy = definition(
            COORDINATOR,
            "_run_copy_only_kernel",
            {"copy_cache_planned_mla": host_copy},
            "HiSparseCoordinator",
        )
        copy(c, 1, 9)
        host_copy.assert_not_called()
        self.assertEqual(c.mem_pool_host.replay_misses.call_args.args[-1], 9)

    def test_capture_dummy_slots_are_valid_and_live_backing_rejected(self):
        prepare = definition(
            COORDINATOR, "prepare_for_graph_capture", owner="HiSparseCoordinator"
        )
        c = NS(
            full_attention_layer_id_mapping={3: 0},
            mem_pool_host=NS(size=256, available_size=lambda: 256),
            req_to_device_buffer=torch.full((1, 64), -1),
            req_device_buffer_token_locs=torch.full((11, 1, 64), -1),
            req_to_host_pool=torch.full((1, 256), -1),
        )
        prepare(c)
        self.assertTrue(torch.all(c.req_to_host_pool == 0))
        self.assertTrue(torch.all(c.req_device_buffer_token_locs == 0))
        c.mem_pool_host.available_size = lambda: 192
        with self.assertRaisesRegex(RuntimeError, "idle"):
            prepare(c)


if __name__ == "__main__":
    unittest.main()
