"""Metadata-only contract tests; none constitutes CUDA/P2P/model evidence."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEMORY = load(
    "python/sglang/srt/observability/experiment_memory.py", "experiment_memory_test"
)
CONTRACT = load("benchmark/mpu_stages/memory_contract.py", "memory_contract_test")


class Device:
    def __init__(self, kind="cuda", ordinal=0):
        self.type, self.ordinal = kind, ordinal

    def __str__(self):
        return f"{self.type}:{self.ordinal}"


class Storage:
    def __init__(self, identity, size):
        self._cdata, self.size = identity, size

    def nbytes(self):
        return self.size

    def data_ptr(self):
        raise AssertionError(
            "Underlying storage identity is sufficient; never expose pointers"
        )


class Tensor:
    def __init__(
        self,
        identity,
        size=64,
        shape=(8,),
        *,
        element_bytes=8,
        kind="cuda",
        ordinal=0,
        offset=0,
        stride=None,
        storage=None,
    ):
        self.shape, self.device = shape, Device(kind, ordinal)
        self.dtype = f"torch.test_{element_bytes}byte"
        self._element_bytes, self.offset = element_bytes, offset
        self.storage = storage or Storage(identity, size)
        self._stride = stride if stride is not None else self.contiguous_stride(shape)

    @staticmethod
    def contiguous_stride(shape):
        result, step = [], 1
        for size in reversed(shape):
            result.append(step)
            step *= max(1, size)
        return tuple(reversed(result))

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def element_size(self):
        return self._element_bytes

    def untyped_storage(self):
        return self.storage

    def cpu(self):
        raise AssertionError("Accounting must not copy values")

    def item(self):
        raise AssertionError("Accounting must not read values")

    def to(self, *args, **kwargs):
        raise AssertionError("Accounting must not move or cast tensors")

    def __getitem__(self, key):
        raise AssertionError("Packed accounting must not construct views")


def context(**overrides):
    return {
        "snapshot_id": "s0",
        "capture_group_id": "g0",
        "phase": "stable_decode",
        "family": "qwen",
        "rank": 0,
        "world_size": 1,
        "process_id": 11,
        "device_uuid": "GPU-0",
        "source_revision": "abc123",
        "layers": 2,
        "preallocated_request_slots": 4,
        "max_context_len": 128,
        "live_history_tokens": 100,
        **overrides,
    }


def snapshot(amount=64, **overrides):
    return MEMORY.capture_memory_snapshot(
        context(**overrides), {"compute_kv": Tensor(10, amount, (amount // 8,))}
    )


def test_real_cpu_views_and_transpose_share_one_storage():
    torch = pytest.importorskip("torch")
    base = torch.arange(24, dtype=torch.int64).reshape(4, 6)
    result = MEMORY.capture_memory_snapshot(
        context(),
        {"base": base, "views": {"transpose": base.T, "slice": base[1:, 2:]}},
        device_types=None,
    )
    assert result["named_storage_bytes"] == 192
    assert result["tensors"]["views.slice"]["storage_offset"] == 8
    assert result["tensors"]["views.transpose"]["stride"] == [1, 6]
    assert result["exclusive_component_bytes"] == {"shared_components": 192}
    assert sum(result["inclusive_component_bytes"].values()) == 384
    CONTRACT.validate_snapshot(result)


def test_alias_storage_devices_are_distinct_and_cpu_filtered():
    shared = Storage(101, 64)
    result = MEMORY.describe_tensor_storages(
        {
            "a": Tensor(1, storage=shared),
            "b": Tensor(1, storage=shared),
            "peer": Tensor(1, ordinal=1, storage=shared),
            "host": Tensor(1, kind="cpu", storage=shared),
        }
    )
    assert result["named_storage_bytes"] == 128
    assert len(result["storages"]) == 2
    assert result["filtered_tensors"] == [
        {"name": "host", "device": "cpu:0", "reason": "device_type_filtered"}
    ]
    assert "101" not in json.dumps(result)


def test_zero_placeholders_do_not_merge_with_null_pointer():
    first = Tensor(101, 0, (0, 132), element_bytes=1)
    second = Tensor(102, 0, (0, 132), element_bytes=1)
    result = MEMORY.describe_tensor_storages({"a": first, "b": second, "alias": first})
    assert result["named_storage_bytes"] == 0
    assert len(result["storages"]) == 2
    assert (
        result["tensors"]["a"]["allocation_group"]
        == result["tensors"]["alias"]["allocation_group"]
    )


@pytest.mark.parametrize(
    "tensor",
    [
        Tensor(1, 64, (9,)),
        Tensor(1, 64, (8,), offset=1),
        Tensor(1, 64, (8,), stride=(-1,)),
        Tensor(1, 64, (2, 4), stride=(1,)),
    ],
)
def test_malformed_view_extent_or_stride_is_rejected(tensor):
    with pytest.raises(ValueError):
        MEMORY.describe_tensor_storages({"bad": tensor})


def test_overlapping_component_partition_keeps_all_aliases():
    source = Tensor(1)
    result = MEMORY.describe_tensor_storages(
        {"kv": {"main": source}, "registry": [source], "unique": Tensor(2)}
    )
    assert result["storages"][0]["aliases"] == ["kv.main", "registry[0]"]
    assert result["inclusive_component_bytes"] == {
        "kv": 64,
        "registry": 64,
        "unique": 64,
    }
    assert result["exclusive_component_bytes"] == {
        "shared_components": 64,
        "unique": 64,
    }


def test_allocator_nvm_graph_scopes_are_not_extra_named_memory():
    result = MEMORY.capture_memory_snapshot(
        context(),
        {"kv": Tensor(1)},
        allocator_metrics={
            "allocated_bytes": 100,
            "reserved_bytes": 300,
            "peak_scope": "phase",
        },
        nvml_samples=[
            {"timestamp_seconds": 0, "device_uuid": "GPU-0", "used_bytes": 1000},
            {"timestamp_seconds": 0.5, "device_uuid": "GPU-0", "used_bytes": 1200},
        ],
        graph_pool_summary={
            "status": "measured",
            "graph_private_pool_allocated_bytes": 32,
            "graph_private_pool_reserved_bytes": 256,
        },
    )
    assert result["named_storage_bytes"] == 64
    assert result["nvml"]["sampled_used_peak_bytes"] == 1200
    assert result["nvml"]["sampling_interval_seconds"] == [0.5]
    assert result["nvml"]["sampled_peak_is_exact_peak"] is False
    assert result["allocator"]["active_bytes"] is None
    assert (
        result["normalization"]["kv"]["per_preallocated_history_token_bytes"]
        == 64 / 512
    )
    assert result["normalization"]["kv"]["per_live_history_token_bytes"] == 0.64


@pytest.mark.parametrize(
    "overrides",
    [
        {"snapshot_id": ""},
        {"phase": "unknown"},
        {"family": "crossmodel"},
        {"rank": 2, "world_size": 2},
        {"rank": True},
        {"world_size": 0},
        {"layers": 0},
        {"preallocated_request_slots": -1},
        {"max_context_len": float("nan")},
    ],
)
def test_bad_context_is_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        snapshot(**overrides)


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"timestamp_seconds": float("nan"), "used_bytes": 1, "device_uuid": "g"}],
        [{"timestamp_seconds": 1, "used_bytes": True, "device_uuid": "g"}],
        [{"timestamp_seconds": 1, "used_bytes": 1}],
        [
            {"timestamp_seconds": 1, "used_bytes": 1, "device_uuid": "a"},
            {"timestamp_seconds": 2, "used_bytes": 1, "device_uuid": "b"},
        ],
        [
            {"timestamp_seconds": 2, "used_bytes": 1, "device_uuid": "a"},
            {"timestamp_seconds": 1, "used_bytes": 1, "device_uuid": "a"},
        ],
    ],
)
def test_malformed_or_mixed_nvml_samples_are_rejected(samples):
    with pytest.raises((ValueError, TypeError)):
        MEMORY.summarize_nvml_samples(samples)


def test_unmeasured_empty_inventory_preserves_null_scopes():
    result = MEMORY.capture_memory_snapshot(context(), {})
    assert result["named_storage_bytes"] == 0
    assert result["allocator"]["allocated_bytes"] is None
    assert result["nvml"]["sampled_used_peak_bytes"] is None
    assert result["graph_pools"]["graph_private_pool_allocated_bytes"] is None
    assert result["rules"]["absent_owner"] == "not_evidence_of_zero_memory"


def test_zero_live_tokens_and_no_indexer_have_undefined_normalization():
    result = MEMORY.capture_memory_snapshot(
        context(live_history_tokens=0, indexer_layers=0), {"compute_kv": Tensor(1)}
    )
    assert result["normalization"]["compute_kv"]["per_live_history_token_bytes"] is None
    ledger = CONTRACT.indexer_cache_accounting(
        result,
        component_coverage={
            "indexer_k_vectors": "not_applicable",
            "indexer_k_scales": "not_applicable",
        },
    )
    assert (
        ledger["categories"]["indexer_k_vectors"]["normalized_content_bytes"][
            "per_indexer_layer"
        ]
        is None
    )


def test_bad_allocator_float_and_other_device_nvml_are_rejected():
    with pytest.raises(ValueError):
        MEMORY.capture_memory_snapshot(
            context(), {}, allocator_metrics={"allocated_bytes": 1.5}
        )
    with pytest.raises(ValueError):
        MEMORY.capture_memory_snapshot(
            context(),
            {},
            nvml_samples=[
                {"timestamp_seconds": 0, "device_uuid": "GPU-1", "used_bytes": 1}
            ],
        )


def test_cycle_selection_is_bounded_and_raw_cycle_tree_rejected():
    cyclic = {"tensor": Tensor(1)}
    cyclic["self"] = cyclic
    result = MEMORY.tensor_attributes(SimpleNamespace(cyclic=cyclic))
    assert "tensor" in result["cyclic"]
    with pytest.raises(ValueError, match="Cyclic"):
        MEMORY.describe_tensor_storages(cyclic)


def test_repeated_graph_output_dataclass_preserves_owner_aliases():
    logits = SimpleNamespace(logits=Tensor(1))
    graph = SimpleNamespace(output_buffers={1: logits, 2: logits})
    inventory = MEMORY.describe_tensor_storages(
        MEMORY.collect_model_tensor_trees("qwen", graph_runner=graph)
    )
    assert inventory["storages"][0]["aliases"] == [
        "graph_live_owned.output_buffers.1.logits",
        "graph_live_owned.output_buffers.2.logits",
    ]


def test_qwen_hbm_quest_backend_and_graph_dataclass_are_collected():
    quest = SimpleNamespace(page_k_bounds=Tensor(1), running_k_min=Tensor(2))
    wrapper = SimpleNamespace(
        float_workspace_buffer=Tensor(3), int_workspace_buffer=Tensor(4)
    )
    backend = SimpleNamespace(
        quest=quest, decode_wrappers=[wrapper], workspace_buffer=Tensor(5)
    )
    graph = SimpleNamespace(
        buffers=SimpleNamespace(input_ids=Tensor(6)),
        output_buffers={1: SimpleNamespace(logits=Tensor(7))},
    )
    result = MEMORY.collect_model_tensor_trees(
        "qwen", backend=backend, graph_runner=graph
    )
    inventory = MEMORY.describe_tensor_storages(result)
    assert "quest_persistent.page_k_bounds" in inventory["tensors"]
    assert any("float_workspace_buffer" in name for name in inventory["tensors"])
    assert "graph_live_owned.input_buffers.input_ids" in inventory["tensors"]
    assert "graph_live_owned.output_buffers.1.logits" in inventory["tensors"]
    assert MEMORY.collect_model_logical_regions("qwen", backend=backend) == []


def indexer_pool(fp4=False, index_kpool=1, *, skip=False):
    page_size, dim = 4, 128
    tokens = page_size // index_kpool
    row_bytes = tokens * (68 if fp4 else 132)
    return SimpleNamespace(
        index_k_with_scale_buffer=[
            Tensor(
                21,
                (0 if skip else 2) * row_bytes,
                ((0 if skip else 2), row_bytes),
                element_bytes=1,
            )
        ],
        page_size=page_size,
        index_head_dim=dim,
        quant_block_size=128,
        use_fp4_indexer=fp4,
        start_layer=3,
        index_kpool=index_kpool,
        index_page_size=tokens,
    )


@pytest.mark.parametrize("fp4", [False, True])
def test_deepseek_packed_k_and_scale_are_contents_not_extra_allocations(fp4):
    indexer = indexer_pool(fp4)
    pool = SimpleNamespace(index_pools={4: indexer})
    trees = MEMORY.collect_model_tensor_trees("deepseek-v4-flash", kv_pool=pool)
    regions = MEMORY.collect_model_logical_regions("deepseek-v4-flash", kv_pool=pool)
    result = MEMORY.capture_memory_snapshot(
        context(family="deepseek-v4-flash"), trees, regions=regions
    )
    assert result["named_storage_bytes"] == 8 * (68 if fp4 else 132)
    assert (
        sum(region["logical_content_bytes"] for region in regions)
        == result["named_storage_bytes"]
    )
    assert len(result["storages"]) == 1
    assert all(
        region["allocation_group"] == 0 and region["adds_physical_allocation"] is False
        for region in result["logical_regions"]
    )
    assert result["logical_regions"][0]["layer_id"] == 3
    assert result["logical_regions"][0]["sequence_compression_ratio"] == 4
    assert result["logical_regions"][0]["quantization_block_elements"] == (
        32 if fp4 else 128
    )
    ledger = CONTRACT.indexer_cache_accounting(result)
    assert ledger["categories"]["indexer_k_vectors"]["content_bytes"] == 8 * (
        64 if fp4 else 128
    )
    assert ledger["categories"]["indexer_k_scales"]["content_bytes"] == 32
    assert ledger["categories"]["indexer_mapping_lru"]["content_bytes"] is None
    assert ledger["runtime_counters"]["hits"] is None
    CONTRACT.validate_snapshot(result)


@pytest.mark.parametrize("skip", [False, True])
def test_glm_kpool_placeholder_tail_and_linear_state_are_separate(skip):
    dsa = indexer_pool(index_kpool=2, skip=skip)
    dsa.kv_buffer = [Tensor(30)]
    dsa._compress_tail_k = [Tensor(31)]
    dsa._compress_tail_score = [Tensor(32)]
    mamba = SimpleNamespace(
        mamba_cache=SimpleNamespace(conv=[Tensor(33)], temporal=Tensor(34)),
        request_map=Tensor(35),
    )
    pool = SimpleNamespace(full_kv_pool=dsa, mamba_pool=mamba)
    trees = MEMORY.collect_model_tensor_trees("glm-5.3-flash", kv_pool=pool)
    regions = MEMORY.collect_model_logical_regions("glm-5.3-flash", kv_pool=pool)
    result = MEMORY.capture_memory_snapshot(
        context(family="glm-5.3-flash"), trees, regions=regions
    )
    assert result["inclusive_component_bytes"]["linear_state"] == 128
    assert result["inclusive_component_bytes"]["indexer_k_tail"] == 64
    assert result["logical_regions"][0]["index_kpool"] == 2
    assert result["logical_regions"][0]["logical_content_bytes"] == (0 if skip else 512)
    ledger = CONTRACT.indexer_cache_accounting(result)
    assert ledger["categories"]["indexer_k_vectors"]["status"] == "measured"
    assert ledger["categories"]["indexer_k_vectors"]["content_bytes"] == (
        0 if skip else 512
    )


def test_separate_fp4_aiter_allocations_and_scales_remain_physical_allocations():
    indexer = indexer_pool(True)
    indexer.index_k_with_scale_buffer = None
    indexer.index_k_payload_buffer = [
        Tensor(41, 512, (2, 1, 4, 4, 16), element_bytes=1)
    ]
    indexer.index_k_scale_buffer = [Tensor(42, 32, (2, 1, 4, 4), element_bytes=1)]
    indexer.uses_aiter_fp4_layout = True
    pool = SimpleNamespace(index_pools={4: indexer})
    result = MEMORY.capture_memory_snapshot(
        context(family="deepseek-v4-flash"),
        MEMORY.collect_model_tensor_trees("deepseek-v4-flash", kv_pool=pool),
        regions=MEMORY.collect_model_logical_regions("deepseek-v4-flash", kv_pool=pool),
    )
    assert len(result["storages"]) == 2
    assert result["named_storage_bytes"] == 544
    ledger = CONTRACT.indexer_cache_accounting(result)
    assert ledger["categories"]["indexer_k_vectors"]["content_bytes"] == 512
    assert ledger["categories"]["indexer_k_scales"]["content_bytes"] == 32


def test_deepseek_registry_aliases_and_compression_state_are_preserved():
    shared = Tensor(44)
    child = SimpleNamespace(kv_buffer=[shared])
    state = SimpleNamespace(
        kv_score_buffer=SimpleNamespace(kv_score=Tensor(45)), ring_rows=Tensor(46)
    )
    pool = SimpleNamespace(
        c4_kv_pool=child, kv_pools={4: child}, compress_state_pools=[None, state]
    )
    result = MEMORY.capture_memory_snapshot(
        context(family="deepseek-v4-flash"),
        MEMORY.collect_model_tensor_trees("deepseek-v4-flash", kv_pool=pool),
    )
    assert result["named_storage_bytes"] == 192
    assert len(result["storages"][0]["aliases"]) == 2
    assert result["inclusive_component_bytes"]["compression_state"] == 128


@pytest.mark.parametrize(
    "mutation", ["missing_tensor", "oversize", "overlap", "duplicate_region", "bad_row"]
)
def test_bad_packed_regions_are_rejected(mutation):
    pool = SimpleNamespace(index_pools={4: indexer_pool()})
    trees = MEMORY.collect_model_tensor_trees("deepseek-v4-flash", kv_pool=pool)
    regions = MEMORY.collect_model_logical_regions("deepseek-v4-flash", kv_pool=pool)
    if mutation == "missing_tensor":
        regions[0]["tensor_name"] = "absent"
    elif mutation == "oversize":
        regions[0]["logical_content_bytes"] += 1
    elif mutation == "overlap":
        regions[1]["geometry"]["region_offset_within_row_bytes"] = 0
    elif mutation == "duplicate_region":
        regions.append(copy.deepcopy(regions[0]))
    else:
        regions[0]["geometry"]["row_stride_bytes"] += 1
    with pytest.raises(ValueError):
        MEMORY.capture_memory_snapshot(
            context(family="deepseek-v4-flash"), trees, regions=regions
        )


def controls(**overrides):
    result = {
        name: {"dense": "same", "candidate": "same"}
        for name in CONTRACT.COMPARISON_CONTROLS
    }
    result.update(overrides)
    return result


def test_same_model_net_delta_and_unknown_component_do_not_become_zero():
    dense, candidate = snapshot(), snapshot(128, snapshot_id="s1")
    candidate = MEMORY.capture_memory_snapshot(
        candidate["context"],
        {"compute_kv": Tensor(10, 128, (16,)), "quest_persistent": Tensor(11)},
    )
    result = CONTRACT.compare_to_dense(dense, candidate, controls=controls())
    assert result["named_storage_net_delta_bytes"] == 128
    assert result["inclusive_component_deltas"]["compute_kv"]["net_delta_bytes"] == 64
    assert (
        result["inclusive_component_deltas"]["quest_persistent"]["net_delta_bytes"]
        is None
    )
    assert result["allocator_deltas"]["allocated_bytes"] is None


@pytest.mark.parametrize(
    "candidate_change, control_change",
    [
        ({"family": "glm-5.3-flash"}, {}),
        ({"phase": "prefill"}, {}),
        ({"device_uuid": "GPU-1"}, {}),
        ({"source_revision": "other"}, {}),
        ({}, {"actual_batch": {"dense": 1, "candidate": 2}}),
        ({}, {"request_set_fingerprint": {"dense": None, "candidate": None}}),
    ],
)
def test_unmatched_model_device_batch_or_request_sets_are_rejected(
    candidate_change, control_change
):
    with pytest.raises(ValueError):
        CONTRACT.compare_to_dense(
            snapshot(),
            snapshot(**candidate_change),
            controls=controls(**control_change),
        )


def test_rank_local_storage_ids_never_merge_and_device_nvml_not_summed():
    first = snapshot(world_size=4)
    second = snapshot(
        world_size=4, rank=1, process_id=12, snapshot_id="s1", device_uuid="GPU-1"
    )
    assert (
        first["tensors"]["compute_kv"]["allocation_group"]
        == second["tensors"]["compute_kv"]["allocation_group"]
    )
    result = CONTRACT.aggregate_rank_snapshots([first, second])
    assert result["named_process_owned_storage_bytes_sum"] == 128
    assert result["captured_ranks"] == [0, 1]
    assert result["all_ranks_captured"] is False
    assert result["cross_process_shared_storage_deduplication"] == "unmeasured"
    assert len(result["nvml_series_by_physical_device"]) == 2
    assert "nvml_peak_sum" not in result


def test_mixed_capture_group_and_duplicate_ranks_are_rejected():
    with pytest.raises(ValueError):
        CONTRACT.aggregate_rank_snapshots([snapshot(), snapshot(snapshot_id="s1")])
    with pytest.raises(ValueError):
        CONTRACT.aggregate_rank_snapshots(
            [
                snapshot(world_size=2),
                snapshot(world_size=2, rank=1, capture_group_id="g1"),
            ]
        )


def test_explicit_future_k_cache_mapping_workspace_and_counter_scopes():
    result = MEMORY.capture_memory_snapshot(
        context(indexer_layers=2),
        {"indexer_mapping_lru": Tensor(1), "indexer_workspace": Tensor(2)},
    )
    ledger = CONTRACT.indexer_cache_accounting(
        result,
        component_coverage={
            "indexer_k_vectors": "not_applicable",
            "indexer_k_scales": "not_applicable",
        },
        counters={"lookups": 10, "hits": 7, "misses": 3},
        counter_scope={"phase": "stable_decode", "reset_interval": "steps_1_to_100"},
    )
    assert (
        ledger["categories"]["indexer_mapping_lru"]["normalized_content_bytes"][
            "per_indexer_layer"
        ]
        == 32
    )
    assert ledger["categories"]["indexer_k_vectors"]["content_bytes"] == 0
    assert ledger["runtime_counters"]["hit_rate"] == 0.7
    assert ledger["runtime_counters"]["evictions"] is None
    with pytest.raises(ValueError):
        CONTRACT.indexer_cache_accounting(result, counters={"hits": 1})
    with pytest.raises(ValueError):
        CONTRACT.indexer_cache_accounting(
            result,
            counters={"lookups": 10, "hits": 7, "misses": 4},
            counter_scope={"phase": "stable_decode", "reset_interval": "steps"},
        )


@pytest.mark.parametrize(
    "field",
    [
        "named_storage_bytes",
        "named_storage_bytes_by_device",
        "exclusive_component_bytes",
    ],
)
def test_tampered_physical_accounting_is_rejected(field):
    value = snapshot()
    value[field] = 0 if field == "named_storage_bytes" else {}
    with pytest.raises(ValueError):
        CONTRACT.validate_snapshot(value)


def test_tampered_component_owner_and_packed_geometry_are_rejected():
    value = snapshot()
    value["inclusive_component_bytes"] = {"invented": 64}
    with pytest.raises(ValueError, match="Component totals"):
        CONTRACT.validate_snapshot(value)
    pool = SimpleNamespace(index_pools={4: indexer_pool()})
    value = MEMORY.capture_memory_snapshot(
        context(family="deepseek-v4-flash"),
        MEMORY.collect_model_tensor_trees("deepseek-v4-flash", kv_pool=pool),
        regions=MEMORY.collect_model_logical_regions("deepseek-v4-flash", kv_pool=pool),
    )
    value["logical_regions"][0]["geometry"]["region_bytes_per_row"] += 1
    with pytest.raises(ValueError, match="geometry"):
        CONTRACT.validate_snapshot(value)


def test_unknown_or_mismatched_allocator_peak_intervals_suppress_peak_delta():
    dense = MEMORY.capture_memory_snapshot(
        context(),
        {"compute_kv": Tensor(1)},
        allocator_metrics={"peak_allocated_bytes": 100, "peak_scope": "unmeasured"},
    )
    candidate = MEMORY.capture_memory_snapshot(
        context(snapshot_id="s1"),
        {"compute_kv": Tensor(1)},
        allocator_metrics={"peak_allocated_bytes": 200, "peak_scope": "phase_1"},
    )
    result = CONTRACT.compare_to_dense(dense, candidate, controls=controls())
    assert result["allocator_deltas"]["peak_allocated_bytes"] is None
    assert result["allocator_peak_scopes_matched"] is False
