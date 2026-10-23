"""Portable contracts for GLM-5.3-Flash NoPD HiSparse.

Production AST definitions run with CPU tensors and injected runtime boundaries.
These tests do not claim CUDA kernel compilation or model-output equivalence.
"""

import ast
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def definition(path, name, namespace=None, owner=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if owner:
        body = next(n for n in body if getattr(n, "name", None) == owner).body
    node = next(n for n in body if getattr(n, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ns = dict(torch=torch, weakref=weakref)
    ns.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns[name]


ALLOCATOR = "python/sglang/srt/mem_cache/allocator/hisparse.py"
COORDINATOR = "python/sglang/srt/managers/hisparse_coordinator.py"
CONFIGURATOR = "python/sglang/srt/mem_cache/kv_cache_configurator.py"


class Hybrid:
    pass


class DevicePool:
    def register_mapping(self, mapping):
        self.mapping = mapping

    def _translate_loc_to_hisparse_device(self, locations):
        return self.mapping[locations]


def real_paged_allocator():
    # Only platform dispatch is mocked: alloc/free/page arithmetic are production.
    base_init = definition(
        "python/sglang/srt/mem_cache/allocator/base.py",
        "__init__",
        {"abc": NS(abstractmethod=lambda f: f)},
        "BaseTokenToKVPoolAllocator",
    )
    base = type("Base", (), {"__init__": base_init})
    path = "python/sglang/srt/mem_cache/allocator/paged.py"
    get_pages = definition("python/sglang/srt/utils/common.py", "get_num_new_pages")
    paged = definition(
        path,
        "PagedTokenToKVPoolAllocator",
        {
            "BaseTokenToKVPoolAllocator": base,
            "get_bool_env_var": lambda _: True,
            "current_platform": NS(capabilities=NS(supports_triton=False)),
            "_is_hip": False,
            "get_num_new_pages": get_pages,
            "alloc_extend_naive": definition(path, "alloc_extend_naive"),
            "alloc_decode_naive": definition(path, "alloc_decode_naive"),
        },
    )
    return definition(
        ALLOCATOR,
        "HiSparseTokenToKVPoolAllocator",
        {
            "BaseTokenToKVPoolAllocator": base,
            "HybridLinearKVPool": Hybrid,
            "HiSparseDSATokenToKVPool": DevicePool,
            "current_platform": NS(get_paged_allocator_cls=lambda: None),
            "PagedTokenToKVPoolAllocator": paged,
            "get_num_new_pages": get_pages,
        },
    )


class TestLogicalLifetime(unittest.TestCase):
    def test_hot_buffer_release_keeps_history_reserved_until_request_release(self):
        allocator = real_paged_allocator()(
            128, 64, torch.bfloat16, "cpu", DevicePool(), True, 2
        )
        zero, length = torch.tensor([0]), torch.tensor([128])
        prefill = allocator.alloc_extend(
            zero, zero, length, length, torch.tensor([-1]), 128
        )
        self.assertEqual(allocator.logical_attn_allocator.available_size(), 128)
        hot = allocator.alloc_device_buffer(prefill, 64)
        self.assertTrue(
            torch.all(allocator.full_to_hisparse_device_index_mapping[prefill] == 0)
        )
        self.assertEqual(allocator.logical_attn_allocator.available_size(), 128)
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), 64)
        history = [prefill]
        last = prefill[-1:]
        for n in range(129, 257):
            length = torch.tensor([n])
            last = allocator.alloc_decode(length, length, last)
            self.assertIsNotNone(last)
            history.append(last)
        self.assertEqual(allocator.logical_attn_allocator.available_size(), 0)
        self.assertIsNone(
            allocator.alloc_decode(torch.tensor([257]), torch.tensor([257]), last)
        )
        # Unmapped logical pages do not free the separately owned hot buffer.
        allocator.free(torch.cat(history))
        self.assertEqual(allocator.logical_attn_allocator.available_size(), 256)
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), 64)
        allocator.free_hisparse_indices(hot)
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), 128)
        self.assertEqual(allocator.full_to_hisparse_device_index_mapping[-1].item(), -1)


class TestHybridAllocator(unittest.TestCase):
    def initialize(self, pool):
        allocator = NS(clear=lambda: None)
        init = definition(
            ALLOCATOR,
            "__init__",
            {
                "HybridLinearKVPool": Hybrid,
                "HiSparseDSATokenToKVPool": DevicePool,
                "current_platform": NS(get_paged_allocator_cls=lambda: None),
                "PagedTokenToKVPoolAllocator": lambda size, *args: NS(size=size),
            },
            "HiSparseTokenToKVPoolAllocator",
        )
        init(allocator, 128, 64, torch.bfloat16, "cpu", pool, False, 2)
        return allocator

    def test_wrapper_keeps_kda_but_only_child_gets_token_mapping(self):
        pool = Hybrid()
        pool.full_kv_pool = DevicePool()
        pool.mamba_pool = object()
        allocator = self.initialize(pool)
        self.assertIs(allocator._kvcache, pool)
        self.assertIs(allocator.hisparse_kvcache, pool.full_kv_pool)
        self.assertFalse(hasattr(pool, "mapping"))
        self.assertEqual(pool.full_kv_pool.mapping.numel(), 256 + 64 + 1)
        self.assertEqual(pool.full_kv_pool.mapping[-1].item(), -1)
        self.assertEqual(allocator.hisparse_attn_allocator.size, 128)

    def test_plain_dsa_path_is_preserved(self):
        pool = DevicePool()
        allocator = self.initialize(pool)
        self.assertIs(allocator.hisparse_kvcache, pool)
        self.assertEqual(allocator.logical_attn_allocator.size, 256)

    def test_hybrid_with_ordinary_dsa_child_fails_early(self):
        pool = Hybrid()
        pool.full_kv_pool = object()
        with self.assertRaisesRegex(TypeError, "HiSparse DSA"):
            self.initialize(pool)

    def test_wrapper_forward_does_not_touch_kda(self):
        translate = definition(
            "python/sglang/srt/mem_cache/memory_pool.py",
            "translate_loc_to_hisparse_device",
            owner="HybridLinearKVPool",
        )
        pool = NS(
            full_kv_pool=NS(translate_loc_to_hisparse_device=lambda x: x + 64),
            mamba_pool=Mock(),
        )
        torch.testing.assert_close(
            translate(pool, torch.tensor([1, 2])), torch.tensor([65, 66])
        )
        pool.mamba_pool.assert_not_called()


class TestHybridPoolConstruction(unittest.TestCase):
    def test_compact_dsa_layers_and_original_topk_mask(self):
        layers = list(range(3, 44, 4))
        pool_class = Mock(return_value=object())
        build = definition(
            CONFIGURATOR,
            "_build_dsa_kv_pool",
            {
                "get_memory": lambda: NS(enable_hisparse=True),
                "reject_out_of_tree_path": lambda *a, **k: None,
                "current_platform": object(),
                "HiSparseDSATokenToKVPool": pool_class,
                "_should_elide_dsa_index_k": lambda **k: True,
                "dsa_layer_skips_topk": lambda config, layer: layer == 11,
                "calculate_mla_kv_cache_dim": lambda **k: 512,
                "get_exec": lambda: NS(features=NS(enable_memory_saver=False)),
                "get_dsa_index_head_dim": lambda config: 128,
                "get_dsa_index_kpool": lambda config: 4,
                "get_dsa_index_kpool_compress": lambda config: True,
                "max_speculative_num_draft_tokens": lambda: None,
            },
            "KVCacheConfigurator",
        )
        owner = NS(
            is_draft_worker=False,
            loc_space_scale=1,
            layer_info=NS(start_layer=0, end_layer=45, num_effective_layers=45),
            model_config=NS(kv_lora_rank=512, qk_rope_head_dim=0, hf_config=object()),
            pool_page_size=64,
            kv_cache_dtype=torch.bfloat16,
            device="cpu",
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.cp.utils": NS(
                    get_glm_dsa_cp_layer_shard_info=lambda x: (None, None)
                ),
                "sglang.srt.mem_cache.sparsity": NS(
                    parse_hisparse_config=lambda: NS(host_to_device_ratio=2)
                ),
            },
        ):
            build(
                owner,
                max_total_num_tokens=128,
                max_running_requests=2,
                dsa_pool_class=object,
                full_attention_layer_ids=layers,
            )
        kwargs = pool_class.call_args.kwargs
        self.assertEqual(
            (kwargs["start_layer"], kwargs["end_layer"], kwargs["layer_num"]),
            (0, 11, 11),
        )
        self.assertEqual(kwargs["skip_topk_layers"], [False, False, True] + [False] * 8)
        self.assertEqual(kwargs["index_kpool"], 4)
        self.assertEqual(kwargs["host_to_device_ratio"], 2)


class TestHybridSelection(unittest.TestCase):
    def coordinator(self):
        return NS(
            is_dsv4_hisparse=False,
            top_k=2051,
            num_tail_slots=3,
            full_attention_layer_id_mapping={
                layer: i for i, layer in enumerate(range(3, 44, 4))
            },
            device="cpu",
            device_buffer_size=4096,
            req_to_device_buffer=torch.arange(4097).view(1, -1) + 64,
            req_to_host_pool=torch.arange(8192).view(1, -1),
            mem_pool_host=NS(load_to_device_per_layer=Mock()),
            mem_pool_device=object(),
        )

    def test_short_sequence_tail_columns_keep_their_original_positions(self):
        coordinator = self.coordinator()
        naive = definition(COORDINATOR, "naive_load_topk", owner="HiSparseCoordinator")
        picks = torch.full((1, 2051), -1, dtype=torch.int32)
        picks[0, :3] = torch.tensor([0, 1, 2])
        picks[0, 2048:] = torch.tensor([0, 1, 2])
        actual = naive(coordinator, torch.tensor([0]), torch.tensor([3]), picks, 43)
        torch.testing.assert_close(
            actual[0, 2048:], torch.tensor([64, 65, 66], dtype=torch.int32)
        )
        self.assertTrue(torch.all(actual[0, 3:2048] == -1))
        coordinator.mem_pool_host.load_to_device_per_layer.assert_not_called()

    def test_long_sequence_loads_dense_child_layer_and_reserves_latest_slot(self):
        coordinator = self.coordinator()
        naive = definition(COORDINATOR, "naive_load_topk", owner="HiSparseCoordinator")
        picks = torch.full((1, 2051), -1, dtype=torch.int32)
        picks[0, :2] = torch.tensor([10, 4999])
        picks[0, 2048:] = torch.tensor([4997, 4998, -1])
        actual = naive(coordinator, torch.tensor([0]), torch.tensor([5000]), picks, 43)
        call = coordinator.mem_pool_host.load_to_device_per_layer.call_args
        self.assertEqual(call.args[3], 10)
        torch.testing.assert_close(call.args[1], torch.tensor([10, 4997, 4998]))
        self.assertEqual(actual[0, 1].item(), 4160)
        self.assertEqual(actual[0, 2050].item(), -1)

    def test_runtime_swap_translates_global_layer_once(self):
        swap = definition(
            COORDINATOR, "swap_in_selected_pages", owner="HiSparseCoordinator"
        )
        coordinator = self.coordinator()
        coordinator.enable_prefetch = False
        coordinator._run_swap_in_kernel = Mock(return_value="done")
        self.assertEqual(
            swap(
                coordinator,
                torch.tensor([0]),
                torch.tensor([5000]),
                torch.zeros((1, 2051)),
                43,
            ),
            "done",
        )
        self.assertEqual(coordinator._run_swap_in_kernel.call_args.args[3], 10)

    def test_kernel_wrapper_enforces_2051_columns_and_passes_tail_contract(self):
        module = NS(load_cache_to_device_buffer=Mock())
        jit = Mock(return_value=module)
        load = definition(
            "python/sglang/kernels/ops/kvcache/hisparse.py",
            "_load_cache_to_device_buffer_mla",
            {"_jit_sparse_module": jit},
        )
        kwargs = dict(
            is_dsv4_layout=False,
            top_k_tokens=torch.full((1, 2051), -1, dtype=torch.int32),
            top_k_device_locs=torch.empty((1, 2051), dtype=torch.int32),
            device_buffer_tokens=torch.empty((1, 4097), dtype=torch.int32),
            host_cache_locs=torch.empty((1, 8192), dtype=torch.int64),
            device_buffer_locs=torch.empty((1, 4097), dtype=torch.int32),
            host_cache=torch.empty((8192, 1, 8)),
            device_buffer=torch.empty((4097, 1, 8)),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([5000]),
            lru_slots=torch.empty((1, 4096), dtype=torch.int16),
            item_size_bytes=32,
            num_top_k=2051,
            hot_buffer_size=4096,
            page_size=1,
            block_size=960,
            num_real_reqs=torch.tensor([1], dtype=torch.int32),
            miss_src=None,
            miss_dst=None,
            miss_count=None,
            skip_io=False,
            num_tail_slots=3,
        )
        load(**kwargs)
        self.assertEqual(jit.call_args.kwargs["num_tail_slots"], 3)
        module.load_cache_to_device_buffer.assert_called_once()
        with self.assertRaisesRegex(ValueError, "int32"):
            load(
                **{**kwargs, "top_k_tokens": torch.empty((1, 2048), dtype=torch.int32)}
            )
        with self.assertRaisesRegex(ValueError, "tail slots"):
            load(**{**kwargs, "num_tail_slots": 2051})


class TestScopeAndCapacity(unittest.TestCase):
    def config(self):
        return NS(
            disaggregation_mode="null",
            speculative_algorithm=None,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            moe_dp_size=1,
            nnodes=1,
            tp_size=4,
            enable_prefill_cp=False,
            enable_pdmux=False,
            enable_two_batch_overlap=False,
            kv_cache_dtype="bfloat16",
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            hisparse_config=None,
        )

    def test_supported_cpu_ddr_scope(self):
        validate = definition(
            "python/sglang/srt/arg_groups/hisparse_hook.py",
            "validate_glm_hisparse_scope",
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.mem_cache.mpu_glm": NS(glm_backing_options=Mock())},
        ):
            validate(self.config())

    def test_unvalidated_modes_fail_closed(self):
        validate = definition(
            "python/sglang/srt/arg_groups/hisparse_hook.py",
            "validate_glm_hisparse_scope",
        )
        for field, value in (
            ("disaggregation_mode", "decode"),
            ("speculative_algorithm", "EAGLE"),
            ("attn_dp_size", 2),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("pp_size", 2),
            ("kv_cache_dtype", "fp8_e4m3"),
            ("enable_hierarchical_cache", True),
            ("enable_prefill_cp", True),
        ):
            cfg = self.config()
            setattr(cfg, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError), patch.dict(
                sys.modules,
                {"sglang.srt.mem_cache.mpu_glm": NS(glm_backing_options=Mock())},
            ):
                validate(cfg)

    def test_capacity_reports_payload_bytes_and_actual_occupancy(self):
        info = definition(
            COORDINATOR, "get_stage_capacity_info", owner="HiSparseCoordinator"
        )
        coordinator = NS(
            backing_store="cpu",
            backing_capacity_ratio=2,
            is_mpu=False,
            logical_address_capacity=256,
            get_retained_gpu_pool_tensors=lambda: {"test": []},
            token_to_kv_pool_allocator=NS(
                hisparse_attn_allocator=NS(size=128), size_full=256
            ),
            mem_pool_device=NS(
                page_size=64,
                kv_buffer=[torch.empty((192, 1, 512), dtype=torch.bfloat16)] * 11,
            ),
            mem_pool_host=NS(
                size=320,
                kv_buffer=torch.empty((11, 320, 1, 512), dtype=torch.bfloat16),
                available_size=lambda: 256,
            ),
        )
        describe = Mock(
            return_value={"retained_gpu_pool_bytes": {}, "retained_gpu_pool_shapes": {}}
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.observability.mpu_stage_timing": NS(
                    describe_retained_gpu_pools=describe
                )
            },
        ):
            result = info(coordinator)
        describe.assert_called_once_with({"test": []})
        self.assertEqual(result["compute_kv_bytes"], 11 * 192 * 1024)
        self.assertEqual(result["backing_kv_bytes"], 11 * 320 * 1024)
        self.assertEqual(result["used_slots"], 64)
        self.assertEqual(result["storage_device_ids"], [])
        self.assertEqual(result["effective_capacity_ratio"], 2.5)
        self.assertEqual(result["reserved_slots"], 0)
        self.assertEqual(result["logical_admission_slots"], 256)
        self.assertEqual(result["padding_slots"], {"compute": 64, "backing": 64})


if __name__ == "__main__":
    unittest.main()
