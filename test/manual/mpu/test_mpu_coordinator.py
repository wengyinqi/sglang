"""CPU execution of production MPU dispatch and capacity lifecycle methods.

The coordinator class is compiled without CUDA-only imports. Kernels/storage
are spies, so these tests verify integration, not CUDA or peer-memory behavior.
"""

import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/managers/hisparse_coordinator.py"
ADMISSION = ROOT / "python/sglang/srt/mem_cache/mpu_admission.py"
spec = importlib.util.spec_from_file_location("mpu_admission_test", ADMISSION)
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


class TestMPUAdmission(unittest.TestCase):
    def test_prefill_and_decode_hold_full_lease_until_release(self):
        budget = admission.MPUAdmissionBudget(100)
        self.assertTrue(budget.reserve("first", 70))
        self.assertTrue(budget.reserve("first", 70))  # another prefill chunk
        self.assertEqual(budget.reserved, 70)
        self.assertFalse(budget.reserve("second", 31))
        self.assertTrue(budget.reserve("smaller", 30))
        budget.release("first")
        budget.release("first")  # abort/finish cleanup is idempotent
        self.assertTrue(budget.reserve("second", 31))
        self.assertEqual(budget.reserved, 61)

    def test_rejects_mutation_and_oversized_request(self):
        budget = admission.MPUAdmissionBudget(100)
        self.assertFalse(budget.reserve("too-large", 101))
        self.assertTrue(budget.reserve("r", 70))
        with self.assertRaises(ValueError):
            budget.reserve("r", 80)
        for count in (0, -1, 0.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                budget.reserve("r2", count)
        self.assertEqual(budget.reserved, 70)


class _DevicePool:
    def __init__(self):
        self.size = 128
        self.page_size = 64
        self.layer_num = 1
        self.k_buffer = [torch.zeros(192, 1, 4, dtype=torch.bfloat16)]
        self.v_buffer = [torch.zeros_like(self.k_buffer[0])]
        self.full_to_hisparse_device_index_mapping = torch.zeros(257, dtype=torch.int64)


class _Backing:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.size = 256
        self.shard_rows = 256
        self.token_stride_size = 8
        self.peer_ptrs = object()
        self.k_buffer = [object()]
        self.v_buffer = [object()]
        self.free = Mock()
        self.close = Mock()


def coordinator_type():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HiSparseCoordinator"
    )
    names = {
        "__init__",
        "swap_in_selected_pages",
        "mpu_request_error",
        "reserve_mpu_request",
        "release_mpu_request",
        "request_finished",
        "close",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    cpu_host = Mock(side_effect=_Backing)
    stream = SimpleNamespace(wait_stream=Mock())
    namespace = dict(
        torch=torch,
        logger=logging.getLogger(__name__),
        MODE_QUEST="quest",
        MODE_DSA_NATIVE="dsa_native",
        _VALID_MODES=("quest", "dsa_native"),
        device_module=SimpleNamespace(
            Stream=Mock(), Event=Mock(), current_stream=lambda: stream
        ),
        HiSparseMHATokenToKVPool=_DevicePool,
        HiSparseNSATokenToKVPool=type("NSA", (), {}),
        MHATokenToKVPoolHost=cpu_host,
        MLATokenToKVPoolHost=Mock(),
        load_cache_from_mpu_to_device_buffer_mha=Mock(),
        load_cache_to_device_buffer_mha=Mock(),
        load_cache_to_device_buffer_mla=Mock(),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["HiSparseCoordinator"], namespace


class TestMPUCoordinator(unittest.TestCase):
    def fixture(self, backing="mpu"):
        cls, ns = coordinator_type()
        device = _DevicePool()
        alloc = SimpleNamespace(
            get_kvcache=lambda: device,
            free_hisparse_indices=Mock(),
            full_to_hisparse_device_index_mapping=device.full_to_hisparse_device_index_mapping,
        )
        pool = SimpleNamespace(
            req_to_token=torch.zeros(2, 512, dtype=torch.int64), max_context_len=512
        )
        quest = SimpleNamespace(
            page_k_min=object(), top_k=64, invalidate_request=Mock()
        )
        mpu_factory = Mock(side_effect=_Backing)
        modules = {
            "sglang.srt.mem_cache.mpu_memory_pool": SimpleNamespace(
                MPUMHATokenToKVPool=mpu_factory
            ),
            "sglang.srt.mem_cache.mpu_admission": admission,
        }
        with patch.dict(sys.modules, modules), patch.object(
            torch.distributed, "get_world_size", return_value=1
        ):
            coord = cls(
                pool,
                alloc,
                64,
                64,
                "cpu",
                None,
                mode="quest",
                quest_algorithm=quest,
                backing_store=backing,
                mpu_device_ids=[1],
                mpu_capacity_ratio=2.0,
            )
        return coord, ns, mpu_factory

    def test_mpu_constructor_never_allocates_host_kv(self):
        coord, ns, mpu = self.fixture()
        mpu.assert_called_once()
        ns["MHATokenToKVPoolHost"].assert_not_called()
        ns["MLATokenToKVPoolHost"].assert_not_called()
        self.assertEqual(coord.mpu_budget.capacity, 256)
        coord.close()
        coord.mem_pool_host.close.assert_called_once()

    def test_resolve_dispatch_uses_peer_table_and_global_backing_slots(self):
        coord, ns, _ = self.fixture()
        selected = torch.zeros(1, 64, dtype=torch.int32)
        result = coord.swap_in_selected_pages(
            torch.tensor([0]), torch.tensor([129]), selected, 0
        )
        call = ns["load_cache_from_mpu_to_device_buffer_mha"].call_args.kwargs
        self.assertIs(call["peer_ptrs"], coord.mem_pool_host.peer_ptrs)
        self.assertIs(call["mpu_cache_locs"], coord.req_to_host_pool)
        self.assertEqual(call["peer_rows"], 256)
        self.assertNotIn("host_cache_locs", call)
        self.assertEqual(result.shape, (1, 64))
        ns["load_cache_to_device_buffer_mha"].assert_not_called()

    def test_cpu_constructor_and_dispatch_remain_host_backed(self):
        coord, ns, mpu = self.fixture("cpu")
        mpu.assert_not_called()
        self.assertIsNone(coord.mpu_budget)
        coord.swap_in_selected_pages(
            torch.tensor([0]),
            torch.tensor([129]),
            torch.zeros(1, 64, dtype=torch.int32),
            0,
        )
        ns["load_cache_to_device_buffer_mha"].assert_called_once()
        ns["load_cache_from_mpu_to_device_buffer_mha"].assert_not_called()

    def test_finish_releases_backing_slots_and_request_lease(self):
        coord, _, _ = self.fixture()
        req = SimpleNamespace(
            rid="r",
            origin_input_ids=[0] * 129,
            sampling_params=SimpleNamespace(max_new_tokens=16),
            req_pool_idx=0,
            kv_allocated_len=132,
        )
        self.assertIsNone(coord.mpu_request_error(req))
        self.assertTrue(coord.reserve_mpu_request(req))
        coord.req_to_host_pool[0, :130] = torch.arange(130)
        coord.req_device_buffer_size[0] = 128
        coord.req_to_device_buffer[0, :128] = torch.arange(64, 192)
        coord.request_finished(req)
        self.assertEqual(coord.mpu_budget.reserved, 0)
        freed = coord.mem_pool_host.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.arange(130)))
        self.assertTrue(torch.all(coord.req_to_host_pool[0] == -1))
        self.assertEqual(coord.req_device_buffer_size[0], 0)

    def test_oversized_and_session_requests_are_rejected(self):
        coord, _, _ = self.fixture()
        req = SimpleNamespace(
            rid="r",
            origin_input_ids=[0] * 250,
            sampling_params=SimpleNamespace(max_new_tokens=16),
        )
        self.assertIn("cannot hold", coord.mpu_request_error(req))
        req.session = object()
        self.assertIn("non-session", coord.mpu_request_error(req))

    def test_rejected_prompt_dummy_prefill_never_reserves_kv(self):
        coord, _, _ = self.fixture()
        req = SimpleNamespace(
            rid="invalid",
            origin_input_ids=[0],
            sampling_params=SimpleNamespace(max_new_tokens=-64),
            to_finish=object(),
        )
        self.assertTrue(coord.reserve_mpu_request(req))
        self.assertEqual(coord.mpu_budget.reserved, 0)


if __name__ == "__main__":
    unittest.main()
