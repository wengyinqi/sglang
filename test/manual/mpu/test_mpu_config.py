"""CPU regressions for MPU configuration and paged logical capacity.

Execute production definitions via AST to avoid Linux/CUDA serving imports.
"""

import ast
import importlib.util
import json
import math
import sys
import unittest
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_definition(path, name, namespace, class_name=None):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(
            node for node in body if getattr(node, "name", None) == class_name
        ).body
    definition = next(node for node in body if getattr(node, "name", None) == name)
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


CONFIG_NAMESPACE = dict(dataclass=dataclass, field=field, Optional=Optional, math=math)
SparseConfig = load_definition(
    "python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py",
    "SparseConfig",
    CONFIG_NAMESPACE,
)
parse_config = load_definition(
    "python/sglang/srt/mem_cache/sparsity/factory.py",
    "_parse_sparse_config",
    dict(json=json, math=math, SparseConfig=SparseConfig),
)


class TestMPUConfig(unittest.TestCase):
    def parse(self, config=None, enabled=True, compute=0):
        return parse_config(
            SimpleNamespace(
                hisparse_config=json.dumps(config) if config is not None else None,
                enable_hisparse=enabled,
                base_gpu_id=compute,
            )
        )

    def mpu(self, **updates):
        return dict(algorithm="quest", backing_store="mpu", **updates)

    def test_cpu_defaults_and_integer_capacity_are_unchanged(self):
        for config in (None, {"host_to_device_ratio": 3}):
            cfg = self.parse(config, enabled=False)
            ratio = 2 if config is None else 3
            self.assertEqual(cfg.backing_store, "cpu")
            self.assertEqual(cfg.backing_capacity_ratio, ratio)
            self.assertEqual(cfg.logical_token_capacity(128, 64), 128 * ratio)
            self.assertEqual(cfg.sparse_extra_config, {})

    def test_mpu_defaults_and_explicit_storage_ordinals(self):
        cfg = self.parse(self.mpu())
        self.assertEqual(cfg.mpu_device_ids, [1])
        self.assertEqual(cfg.mpu_capacity_ratio, 2.0)
        self.assertEqual(cfg.mpu_memory_reserve_bytes, 1 << 30)
        self.assertEqual(cfg.backing_capacity_ratio, 2.0)
        self.assertEqual(cfg.quest_page_size, 64)
        cfg = self.parse(
            self.mpu(mpu_device_ids=[0, 2, 4, 5], mpu_memory_reserve_bytes=0),
            compute=1,
        )
        self.assertEqual(cfg.mpu_device_ids, [0, 2, 4, 5])
        self.assertEqual(cfg.sparse_extra_config, {})

    def test_capacity_ratio_is_total_across_shards_and_page_aligned_down(self):
        for ratio in (0.1, 0.5, 1.0, 1.51, 2.0, 3.2):
            for count in range(1, 5):
                cfg = self.parse(
                    self.mpu(
                        mpu_device_ids=list(range(1, count + 1)),
                        mpu_capacity_ratio=ratio,
                    )
                )
                capacity = cfg.logical_token_capacity(128, 64)
                backing_slots = math.ceil(128 * ratio)
                self.assertIsInstance(capacity, int)
                self.assertEqual(capacity % 64, 0)
                self.assertLessEqual(capacity, backing_slots)
                self.assertLess(backing_slots - capacity, 64)

    def test_mpu_requires_hisparse_and_quest(self):
        with self.assertRaisesRegex(ValueError, "enable-hisparse"):
            self.parse(self.mpu(), enabled=False)
        for algorithm in (None, "deepseek_nsa", "Quest"):
            with self.assertRaisesRegex(ValueError, "algorithm='quest'"):
                self.parse(dict(backing_store="mpu", algorithm=algorithm))

    def test_cpu_rejects_mpu_only_knobs(self):
        for key, value in (
            ("mpu_device_ids", [1]),
            ("mpu_capacity_ratio", 2),
            ("mpu_memory_reserve_bytes", 0),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "MPU options"
            ):
                self.parse({key: value})

    def test_invalid_storage_ids_fail_before_any_cuda_access(self):
        for ids in ([], [0], [1, 1], [1, 2, 3, 4, 5], [-1], [True], [1.5], "1", None):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.parse(self.mpu(mpu_device_ids=ids))

    def test_invalid_numbers_fail_closed(self):
        for ratio in (0, -1, True, "2", None, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(
                ValueError, "mpu_capacity_ratio"
            ):
                self.parse(self.mpu(mpu_capacity_ratio=ratio))
        for reserve in (-1, 0.5, True, "0", None):
            with self.subTest(reserve=reserve), self.assertRaisesRegex(
                ValueError, "mpu_memory_reserve_bytes"
            ):
                self.parse(self.mpu(mpu_memory_reserve_bytes=reserve))
        for ratio in (0, -1, 1.5, True, "2"):
            with self.subTest(host_ratio=ratio), self.assertRaisesRegex(
                ValueError, "host_to_device_ratio"
            ):
                self.parse({"host_to_device_ratio": ratio})

    def test_malformed_config_is_not_silently_ignored(self):
        for config in ([], "mpu", 2, {"backing_store": "other"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.parse(config)


class TestMPUServingScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(
            load_definition(
                "python/sglang/srt/server_args.py",
                "_check_mpu_server_args",
                dict(
                    get_linear_attn_spec_by_arch=lambda arch: (
                        "hybrid" if arch == "Hybrid" else None
                    )
                ),
                "ServerArgs",
            )
        )

    def setUp(self):
        self.model = SimpleNamespace(
            attention_arch="MHA",
            is_hybrid_swa=False,
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
        )
        self.args = SimpleNamespace(
            enable_hisparse=True,
            device="cuda",
            tp_size=1,
            dp_size=1,
            pp_size=1,
            ep_size=1,
            nnodes=1,
            speculative_algorithm=None,
            disaggregation_mode="null",
            kv_cache_dtype="bfloat16",
            cpu_offload_gb=0,
            offload_group_size=-1,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            get_model_config=lambda: self.model,
        )
        patcher = patch.dict(
            sys.modules,
            {
                "sglang.srt.configs.model_config": SimpleNamespace(
                    AttentionArch=SimpleNamespace(MHA="MHA")
                )
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_colocated_bf16_single_compute_gpu(self):
        self.validate(self.args)

    def test_unsupported_serving_modes(self):
        for field_name, value in (
            ("enable_hisparse", False),
            ("device", "cpu"),
            ("tp_size", 2),
            ("dp_size", 2),
            ("pp_size", 2),
            ("ep_size", 2),
            ("nnodes", 2),
            ("speculative_algorithm", "EAGLE"),
            ("disaggregation_mode", "decode"),
            ("kv_cache_dtype", "float16"),
            ("cpu_offload_gb", 1),
            ("offload_group_size", 1),
            ("enable_hierarchical_cache", True),
            ("enable_lmcache", True),
            ("disaggregation_decode_enable_offload_kvcache", True),
        ):
            with self.subTest(field=field_name):
                previous = getattr(self.args, field_name)
                setattr(self.args, field_name, value)
                with self.assertRaises(ValueError):
                    self.validate(self.args)
                setattr(self.args, field_name, previous)

    def test_mla_and_hybrid_pools_are_rejected(self):
        for name, value in (
            ("attention_arch", "MLA"),
            ("is_hybrid_swa", True),
            ("hf_config", SimpleNamespace(architectures=["Hybrid"])),
        ):
            with self.subTest(field=name):
                previous = getattr(self.model, name)
                setattr(self.model, name, value)
                with self.assertRaisesRegex(ValueError, "MHA/GQA"):
                    self.validate(self.args)
                setattr(self.model, name, previous)


class TestMPULogicalMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "mpu_quest_memory_test",
            ROOT / "python/sglang/srt/mem_cache/sparsity/quest_memory.py",
        )
        cls.memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.memory)
        cls.init_allocator = staticmethod(
            load_definition(
                "python/sglang/srt/mem_cache/hisparse_memory_pool.py",
                "__init__",
                dict(
                    torch=torch,
                    Optional=Optional,
                    NSATokenToKVPool=object,
                    weakref=weakref,
                    PagedTokenToKVPoolAllocator=lambda size, *args: SimpleNamespace(
                        size=size
                    ),
                ),
                "HiSparseTokenToKVPoolAllocator",
            )
        )

    def test_mapping_tensor_and_estimator_use_the_same_integer_capacity(self):
        cfg = SparseConfig(backing_store="mpu", mpu_capacity_ratio=1.51)
        capacity = cfg.logical_token_capacity(128, 64)
        pool = Mock()
        allocator = SimpleNamespace(clear=lambda: None)
        self.init_allocator(
            allocator,
            128,
            64,
            torch.bfloat16,
            "cpu",
            pool,
            False,
            logical_token_capacity=capacity,
        )
        self.assertEqual(allocator._size_full, 192)
        self.assertEqual(allocator.logical_attn_allocator.size, 192)
        self.assertEqual(allocator.hisparse_attn_allocator.size, 128)
        mapping = allocator.full_to_hisparse_device_index_mapping
        self.assertEqual(mapping.numel(), capacity + 64 + 1)
        self.assertEqual(mapping[-1].item(), -1)
        kwargs = dict(
            top_k=128,
            quest_page_size=64,
            start_layer=0,
            end_layer=2,
            max_reqs=2,
            max_context_len=1024,
            kv_heads=4,
            head_dim=128,
            device_buffer_size=256,
            device_token_capacity=128,
            kv_page_size=64,
        )
        baseline = self.memory.estimate_quest_auxiliary_bytes(
            **kwargs, logical_token_capacity=0
        )
        actual = self.memory.estimate_quest_auxiliary_bytes(
            **kwargs, logical_token_capacity=capacity
        )
        self.assertIsInstance(actual, int)
        self.assertEqual(actual - baseline, capacity * mapping.element_size())

    def test_capacity_smaller_than_one_page_fails_before_allocating_mapping(self):
        for capacity in (0, -64, 65, 128.0, True):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(
                ValueError, "whole number of pages"
            ):
                self.init_allocator(
                    SimpleNamespace(),
                    128,
                    64,
                    torch.bfloat16,
                    "cpu",
                    Mock(),
                    False,
                    logical_token_capacity=capacity,
                )


if __name__ == "__main__":
    unittest.main()
