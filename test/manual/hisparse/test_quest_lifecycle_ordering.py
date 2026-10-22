"""CPU regression tests for staging event dependencies and metadata sizing.

The production coordinator methods execute with a deterministic stream/event
simulator. This exercises the release decision when D2H completes before the
bounds reader, without importing the CUDA-only serving stack. GPU end-to-end
tests are still required for the actual transfer and bounds kernels.
"""

import ast
import importlib.util
import logging
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
COORDINATOR = ROOT / "python/sglang/srt/managers/hisparse_coordinator.py"
MEMORY = ROOT / "python/sglang/srt/mem_cache/sparsity/quest_memory.py"
MIXIN = ROOT / "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"


class _Stream:
    def __init__(self, device):
        self.device = device
        self.ready_at = 0

    def synchronize(self):
        self.device.clock = max(self.device.clock, self.ready_at)


class _Event:
    def __init__(self, device):
        self.device = device
        self.ready_at = 0

    def record(self):
        self.ready_at = self.device.current.ready_at

    def wait(self, stream):
        stream.ready_at = max(stream.ready_at, self.ready_at)

    def query(self):
        return self.device.clock >= self.ready_at

    def synchronize(self):
        self.device.clock = max(self.device.clock, self.ready_at)


class _StreamContext:
    def __init__(self, device, stream):
        self.device, self.stream = device, stream

    def __enter__(self):
        self.previous = self.device.current
        self.device.current = self.stream

    def __exit__(self, *args):
        self.device.current = self.previous


class _Device:
    def __init__(self):
        self.clock = 0
        self.current = _Stream(self)

    def Event(self):
        return _Event(self)

    def stream(self, stream):
        return _StreamContext(self, stream)


def _coordinator_type(device):
    # Compile the real orchestration methods, omitting GPU import side effects.
    source = ast.parse(COORDINATOR.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "HiSparseCoordinator"
    )
    names = {
        "admit_request_into_staging",
        "collect_ready_reqs",
        "abort_staging_request",
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
    namespace = {
        "torch": torch,
        "device_module": device,
        "HiSparseAct": namedtuple("HiSparseAct", "start_event finish_event req"),
        "MODE_QUEST": "quest",
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(COORDINATOR), "exec"), namespace
    )
    return namespace["HiSparseCoordinator"]


class TestQuestStagingOrdering(unittest.TestCase):
    def _fixture(self, mode="quest"):
        device = _Device()
        coord = _coordinator_type(device)()
        coord.device = "cpu"
        coord.mode = mode
        coord.tp_world_size = 1
        coord.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64).view(1, -1)
        )
        coord.mem_pool_device = SimpleNamespace(
            _translate_loc_to_hisparse_device=lambda x: x
        )
        coord.req_to_host_pool = torch.full((1, 64), -1, dtype=torch.int64)
        coord.write_staging_stream = _Stream(device)
        coord.quest_bounds_stream = _Stream(device)
        coord._pending_bounds_events = {}
        coord.ack_staging_queue = []
        coord._skip_first_backup = [False]
        coord.released = []
        coord.invalidated = []
        coord.mem_pool_host = SimpleNamespace(
            alloc=lambda n: torch.arange(n),
            backup_from_device_all_layer=lambda *a, **kw: setattr(
                device.current, "ready_at", device.current.ready_at + 2
            ),
            free=lambda x: coord.released.append(("host", device.clock)),
        )
        coord._update_quest_prefill_representations = lambda *a: setattr(
            device.current, "ready_at", device.current.ready_at + 10
        )
        coord.alloc_device_buffer = lambda req: coord.released.append(
            ("device", device.clock)
        )
        coord._early_free_prefill_logical_slots = lambda req: None
        coord.quest_algorithm = SimpleNamespace(
            invalidate_request=lambda idx: coord.invalidated.append((idx, device.clock))
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            fill_ids=list(range(64)),
            kv_allocated_len=64,
            rid="r",
            hisparse_staging=False,
        )
        return device, coord, req

    def test_device_slots_wait_for_bounds_even_after_host_copy_finishes(self):
        device, coord, req = self._fixture()
        coord.admit_request_into_staging(req)
        device.clock = 2
        self.assertEqual(coord.collect_ready_reqs(), [])
        self.assertEqual(coord.released, [])
        device.clock = 10
        self.assertEqual(coord.collect_ready_reqs(), [req])
        self.assertEqual(coord.released, [("device", 10)])
        self.assertFalse(req.hisparse_staging)

    def test_non_quest_admission_waits_only_for_host_copy(self):
        device, coord, req = self._fixture(mode="dsa_native")
        coord.admit_request_into_staging(req)
        device.clock = 2
        self.assertEqual(coord.collect_ready_reqs(), [req])
        self.assertEqual(coord.released, [("device", 2)])

    def test_abort_drains_readers_then_invalidates_reused_slot(self):
        _, coord, req = self._fixture()
        coord.admit_request_into_staging(req)
        coord.abort_staging_request(req)
        self.assertEqual(coord.invalidated, [(0, 10)])
        self.assertEqual(coord.released, [("host", 10)])
        self.assertEqual(coord._pending_bounds_events, {})
        self.assertEqual(coord.ack_staging_queue, [])
        self.assertTrue(torch.all(coord.req_to_host_pool == -1))


class TestQuestMetadataBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("quest_memory_under_test", MEMORY)
        cls.memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.memory)

    def _kwargs(self, requests=8):
        return dict(
            top_k=2048,
            quest_page_size=64,
            start_layer=0,
            end_layer=48,
            max_reqs=requests,
            max_context_len=204804,
            kv_heads=4,
            head_dim=128,
        )

    def test_paper_scale_request_cap_controls_multigib_bounds(self):
        small = self.memory.estimate_quest_auxiliary_bytes(**self._kwargs(8))
        default_slots = self.memory.estimate_quest_auxiliary_bytes(**self._kwargs(2048))
        self.assertGreater(small, 2 * (1 << 30))
        self.assertLess(small, 3 * (1 << 30))
        self.assertGreater(default_slots, 590 * (1 << 30))

    def test_host_ratio_only_changes_gpu_address_map_in_estimate(self):
        kwargs = dict(
            **self._kwargs(),
            device_buffer_size=4096,
            device_token_capacity=262144,
            kv_page_size=64
        )
        ratio2 = self.memory.estimate_quest_auxiliary_bytes(
            **kwargs, host_to_device_ratio=2
        )
        ratio8 = self.memory.estimate_quest_auxiliary_bytes(
            **kwargs, host_to_device_ratio=8
        )
        self.assertEqual(ratio8 - ratio2, 262144 * 6 * 8)
        self.assertGreater(
            ratio2, self.memory.estimate_quest_auxiliary_bytes(**self._kwargs())
        )


class TestQuestPoolSizing(unittest.TestCase):
    def _resolve(self, reserved_bytes, token_limit=100):
        source = ast.parse(MIXIN.read_text(encoding="utf-8"))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_resolve_memory_pool_config"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        namespace = {"logger": logging.getLogger(__name__)}
        exec(compile(ast.fix_missing_locations(module), str(MIXIN), "exec"), namespace)
        budgets = []

        def calculate(available_bytes, page_size):
            budgets.append(available_bytes)
            return SimpleNamespace(max_total_num_tokens=available_bytes // 10)

        configurator = SimpleNamespace(
            calculate_pool_sizes=calculate,
            calculate_pool_sizes_from_max_tokens=lambda tokens, page_size: SimpleNamespace(
                max_total_num_tokens=tokens
            ),
        )
        runner = SimpleNamespace(
            server_args=SimpleNamespace(page_size=1, mem_fraction_static=0.8),
            _profile_available_bytes=lambda _: 1000,
            _apply_token_constraints=lambda tokens: min(tokens, token_limit),
            _resolve_max_num_reqs=lambda tokens: min(16, tokens // 2),
            _quest_auxiliary_memory_bytes=lambda reqs, tokens: reserved_bytes,
        )
        replacement = SimpleNamespace(
            create_memory_pool_configurator=lambda _: configurator
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.model_executor.pool_configurator": replacement,
            },
        ):
            result = namespace["_resolve_memory_pool_config"](runner, 0)
        return budgets, result

    def test_metadata_reservation_reduces_actual_kv_budget(self):
        budgets, result = self._resolve(600)
        self.assertEqual(budgets, [1000, 400])
        self.assertEqual(result.max_total_num_tokens, 40)

    def test_user_token_cap_survives_metadata_reprofile(self):
        _, result = self._resolve(600, token_limit=30)
        self.assertEqual(result.max_total_num_tokens, 30)

    def test_excessive_metadata_fails_before_kv_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "Reduce --max-running-requests"):
            self._resolve(1000)

    def test_other_backends_keep_original_budget(self):
        budgets, result = self._resolve(0)
        self.assertEqual(budgets, [1000])
        self.assertEqual(result.max_total_num_tokens, 100)


if __name__ == "__main__":
    unittest.main()
