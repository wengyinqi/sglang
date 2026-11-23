"""Metadata-only memory inventories for the MPU experiment stages.

The inventory reads shapes, strides and storage identities, never tensor values.
Storage IDs are local to one snapshot; they cannot identify allocations across
snapshots, processes or devices. Allocator, graph-pool and NVML observations are
separate overlapping measurements, not extra summands of named tensor bytes.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

CONTRACT = "experiment_memory_v1"
PHASES = {
    "initialization",
    "initial",
    "initialized",
    "graph_warmup",
    "prefill",
    "decode",
    "stable_decode",
    "idle",
}
FAMILIES = {"qwen", "deepseek-v4-flash", "glm-5.3-flash"}
HOT_NAMES = {
    "_lru_init",
    "lru_slots",
    "req_device_buffer_token_locs",
    "req_device_buffer_tokens",
    "req_to_device_buffer",
}
ALLOCATOR_FIELDS = (
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "inactive_split_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(
            f"{label} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return value


def _is_tensor(value: Any) -> bool:
    return callable(getattr(value, "untyped_storage", None)) and hasattr(value, "shape")


def _selected(value: Any, ancestors=None):
    if _is_tensor(value):
        return value
    ancestors = set() if ancestors is None else ancestors
    if not isinstance(value, (tuple, list, dict)) or id(value) in ancestors:
        return None
    ancestors = {*ancestors, id(value)}
    if isinstance(value, (tuple, list)):
        result = {
            str(i): chosen
            for i, item in enumerate(value)
            if (chosen := _selected(item, ancestors)) is not None
        }
        return result or None
    if isinstance(value, dict):
        result = {
            str(k): chosen
            for k, item in value.items()
            if (chosen := _selected(item, ancestors)) is not None
        }
        return result or None
    return None


def _owner_fields(owner):
    if owner is None:
        return {}
    try:
        return vars(owner)
    except TypeError:
        # Native model metadata may be msgspec.Struct/dataclass slots. These
        # are declared data fields, not arbitrary properties or getter kernels.
        names = getattr(type(owner), "__struct_fields__", None)
        if names is None:
            names = getattr(type(owner), "__dataclass_fields__", None)
        if names is None:
            names = getattr(type(owner), "__slots__", ())
        if isinstance(names, str):
            names = (names,)
        return {
            name: getattr(owner, name)
            for name in names
            if not name.startswith("__") and hasattr(owner, name)
        }


def tensor_attributes(owner: Any, *, exclude=()) -> dict:
    """Only immediate tensor/list/dict attributes; do not evaluate properties."""
    if owner is None:
        return {}
    attributes = _owner_fields(owner)
    return {
        name: chosen
        for name, value in attributes.items()
        if name not in exclude and (chosen := _selected(value)) is not None
    }


def describe_tensor_storages(named_trees: dict, *, device_types=("cuda",)) -> dict:
    """Deduplicate selected storage and retain every view and owner component.

    Cross-component aliases are kept as shared allocations. The inclusive
    component totals overlap; only the exclusive/shared partition sums to the
    unique total. Raw pointers and Python storage identities are never emitted.
    ``device_types=None`` permits real CPU metadata tests and host ledgers.
    """
    if not isinstance(named_trees, dict):
        raise TypeError("Named tensor trees must be a dictionary")
    if isinstance(device_types, str):
        raise TypeError("device_types must be a sequence of device types, not a string")
    allowed = None if device_types is None else set(device_types)
    if allowed is not None and any(
        not isinstance(item, str) or not item for item in allowed
    ):
        raise ValueError("Device type filters must be nonempty strings")
    tensors, allocations, ignored, kept_storage_refs = {}, {}, [], []

    def visit(path, value, component, ancestors=None):
        if value is None:
            return
        ancestors = set() if ancestors is None else ancestors
        if isinstance(value, (dict, tuple, list)):
            if id(value) in ancestors:
                raise ValueError(f"Cyclic tensor tree at {path}")
            ancestors = {*ancestors, id(value)}
        if isinstance(value, dict):
            for name in sorted(value):
                if not isinstance(name, str) or not name:
                    raise TypeError("Tensor tree names must be nonempty strings")
                visit(
                    f"{path}.{name}" if path else name,
                    value[name],
                    component or name,
                    ancestors,
                )
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item, component, ancestors)
            return
        if not _is_tensor(value):
            raise TypeError(f"Named owner {path} is not a tensor tree")
        device = str(value.device)
        device_type = getattr(value.device, "type", device.split(":", 1)[0])
        if allowed is not None and device_type not in allowed:
            ignored.append(
                {"name": path, "device": device, "reason": "device_type_filtered"}
            )
            return
        if path in tensors:
            raise ValueError(f"Duplicate tensor path {path}")
        shape = [_integer(size, path + ".shape") for size in value.shape]
        stride = [_integer(size, path + ".stride") for size in value.stride()]
        if len(shape) != len(stride):
            raise ValueError(f"Shape/stride rank differs for {path}")
        for number in shape + stride:
            _integer(number, path + ".shape/stride")
        offset = _integer(value.storage_offset(), path + ".storage_offset")
        element_bytes = _integer(
            value.element_size(), path + ".element_bytes", positive=True
        )
        storage = value.untyped_storage()
        storage_bytes = _integer(storage.nbytes(), path + ".storage_bytes")
        extent = (
            offset
            if any(size == 0 for size in shape)
            else offset
            + sum((size - 1) * step for size, step in zip(shape, stride))
            + 1
        )
        if extent * element_bytes > storage_bytes:
            raise ValueError(f"Tensor view exceeds storage for {path}")
        # _cdata distinguishes real zero-sized placeholders even when data_ptr=0.
        # Keep fallback storage wrappers alive so Python IDs cannot be recycled.
        storage_identity = getattr(storage, "_cdata", None)
        if storage_identity is None:
            pointer = int(storage.data_ptr())
            storage_identity = (
                ("pointer", pointer) if pointer else ("empty_wrapper", id(storage))
            )
        kept_storage_refs.append(storage)
        key = (device, storage_identity)
        record = allocations.setdefault(
            key,
            {
                "allocation_group": len(allocations),
                "device": device,
                "storage_bytes": storage_bytes,
                "aliases": [],
                "components": [],
            },
        )
        if record["storage_bytes"] != storage_bytes:
            raise ValueError(f"Aliased storage byte counts differ for {path}")
        record["aliases"].append(path)
        if component not in record["components"]:
            record["components"].append(component)
        tensors[path] = {
            "device": device,
            "dtype": str(value.dtype),
            "shape": shape,
            "stride": stride,
            "storage_offset": offset,
            "element_bytes": element_bytes,
            "logical_tensor_bytes": math.prod(shape) * element_bytes,
            "storage_bytes": storage_bytes,
            "allocation_group": record["allocation_group"],
            "component": component,
        }

    visit("", named_trees, None)
    records = list(allocations.values())
    inclusive, exclusive, by_device = {}, {}, {}
    for record in records:
        components = sorted(record["components"])
        record["components"] = components
        record["aliases"].sort()
        record["exclusive_component"] = (
            components[0] if len(components) == 1 else "shared_components"
        )
        amount = record["storage_bytes"]
        for component in components:
            inclusive[component] = inclusive.get(component, 0) + amount
        category = record["exclusive_component"]
        exclusive[category] = exclusive.get(category, 0) + amount
        by_device[record["device"]] = by_device.get(record["device"], 0) + amount
    return {
        "named_storage_bytes": sum(row["storage_bytes"] for row in records),
        "named_storage_bytes_by_device": by_device,
        "tensors": tensors,
        "storages": records,
        "inclusive_component_bytes": inclusive,
        "exclusive_component_bytes": exclusive,
        "filtered_tensors": ignored,
        "allocation_identity_scope": "snapshot_process_device",
        "inclusive_components_are_additive": False,
        "exclusive_components_are_additive": True,
    }


def _backend_trees(backend, seen=None, depth=0):
    if backend is None or depth > 3:
        return {}
    seen = set() if seen is None else seen
    if id(backend) in seen:
        return {}
    seen.add(id(backend))
    excluded = {
        "coord",
        "coordinator",
        "model_runner",
        "model",
        "quest_algorithm",
        "sparse_algorithm",
        "token_to_kv_pool",
        "req_to_token_pool",
    }
    trees = tensor_attributes(backend, exclude=excluded)
    attributes = _owner_fields(backend)
    for name, value in attributes.items():
        if name in excluded:
            continue
        if any(
            token in name for token in ("metadata", "indexer", "candidates", "buffer")
        ):
            selected = _structured_tensors(value)
            if selected is not None:
                trees[name] = selected
            continue
        if _selected(value) is not None:
            continue
        if "wrapper" not in name and "backend" not in name:
            continue
        children = (
            value.items()
            if isinstance(value, dict)
            else (
                enumerate(value)
                if isinstance(value, (list, tuple))
                else [(name, value)]
            )
        )
        nested = {
            str(key): item
            for key, child in children
            if (item := _backend_trees(child, seen, depth + 1))
        }
        if nested:
            trees[name] = nested
    return trees


def _structured_tensors(value, depth=0, seen=None):
    """Graph output/dataclass fields only; bounded traversal, never deepcopy."""
    if value is None or depth > 4:
        return None
    if _is_tensor(value):
        return value
    seen = set() if seen is None else seen
    if id(value) in seen:
        return None
    seen = {*seen, id(value)}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        items = _owner_fields(value).items()
    result = {
        str(name): child
        for name, item in items
        if name
        not in {
            "model_runner",
            "_model_runner",
            "model",
            "_cuda_graph_runner",
            "token_to_kv_pool",
            "req_to_token_pool",
            "coordinator",
            "hisparse_coordinator",
        }
        and (child := _structured_tensors(item, depth + 1, seen)) is not None
    }
    return result or None


def _pool_payload(pool):
    return {
        name: value
        for name in ("k_buffer", "v_buffer", "kv_buffer", "kv_buffer_rope")
        if (value := getattr(pool, name, None)) is not None
    }


def _add_indexer(trees, pool, label):
    packed = getattr(pool, "index_k_with_scale_buffer", None)
    if packed is not None:
        trees.setdefault("indexer_packed_keys_scales", {})[label] = {"buffers": packed}
    for attr, component in (
        ("index_k_payload_buffer", "indexer_k_vectors"),
        ("index_k_scale_buffer", "indexer_k_scales"),
        ("_compress_tail_k", "indexer_k_tail"),
        ("_compress_tail_score", "indexer_score_tail"),
    ):
        value = getattr(pool, attr, None)
        if value is not None:
            trees.setdefault(component, {})[label] = {attr: value}


def _base_owners(
    model_runner, coordinator, kv_pool, allocator, request_pool, backend, graph_runner
):
    if model_runner is not None:
        coordinator = coordinator or getattr(model_runner, "hisparse_coordinator", None)
        kv_pool = kv_pool or getattr(model_runner, "token_to_kv_pool", None)
        allocator = allocator or getattr(
            model_runner, "token_to_kv_pool_allocator", None
        )
        request_pool = request_pool or getattr(model_runner, "req_to_token_pool", None)
        backend = backend or getattr(model_runner, "attn_backend", None)
        if graph_runner is None:
            graph_runner = getattr(model_runner, "graph_runner", None)
        if graph_runner is None:
            graph_runner = {
                label: runner
                for label, attr in (
                    ("decode", "decode_cuda_graph_runner"),
                    ("prefill", "prefill_cuda_graph_runner"),
                )
                if (runner := getattr(model_runner, attr, None)) is not None
            } or None
    if kv_pool is None and allocator is not None:
        getter = getattr(allocator, "get_kvcache", None)
        kv_pool = getter() if callable(getter) else None
    return coordinator, kv_pool, allocator, request_pool, backend, graph_runner


def _graph_trees(runner):
    backend = getattr(runner, "backend", None)
    return {
        "direct": tensor_attributes(runner),
        "input_buffers": _structured_tensors(getattr(runner, "buffers", None)),
        "output_buffers": _structured_tensors(getattr(runner, "output_buffers", None)),
        "registry": _structured_tensors(
            getattr(getattr(runner, "buffer_registry", None), "_slots", None)
        ),
        "backend_outputs": {
            name: _structured_tensors(getattr(backend, name, None))
            for name in (
                "_outputs",
                "_output_buffer",
                "_shared_output_buffer",
                "_capture_inputs",
            )
        },
        "captured_attention_metadata": _structured_tensors(
            getattr(runner, "attn_metadata_buffers", None)
        ),
    }


def collect_model_tensor_trees(
    family,
    *,
    model_runner=None,
    model=None,
    coordinator=None,
    kv_pool=None,
    allocator=None,
    request_pool=None,
    backend=None,
    graph_runner=None,
):
    """Select the concrete model owners without copying KV or evaluating kernels.

    Supported adapters are Qwen GQA/Quest, DeepSeek V4 compressed/SWA/indexer,
    and GLM DSA/KPool plus linear-state pools. Missing owners remain absent;
    an absent component is not evidence that its memory usage is zero.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unsupported experiment model family {family}")
    coordinator, kv_pool, allocator, request_pool, backend, graph_runner = _base_owners(
        model_runner,
        coordinator,
        kv_pool,
        allocator,
        request_pool,
        backend,
        graph_runner,
    )
    model = model or getattr(model_runner, "model", None)
    trees = {}
    for method, component in (
        ("named_parameters", "model_weights"),
        ("named_buffers", "model_buffers"),
    ):
        getter = getattr(model, method, None)
        if callable(getter):
            try:
                values = dict(getter(remove_duplicate=False))
            except TypeError:
                values = dict(getter())
            if values:
                trees[component] = values
    if request_pool is not None:
        trees["request_mapping"] = tensor_attributes(request_pool)
    if allocator is not None:
        trees["allocator_worklists"] = {"outer": tensor_attributes(allocator)}
        for name in (
            "hisparse_attn_allocator",
            "logical_attn_allocator",
            "swa_allocator",
            "full_allocator",
            "allocator",
            "attn_allocator",
        ):
            child = getattr(allocator, name, None)
            if child is not None:
                trees["allocator_worklists"][name] = tensor_attributes(child)
    if coordinator is not None:
        trees["hisparse_hot_lru"] = {
            name: getattr(coordinator, name)
            for name in HOT_NAMES
            if getattr(coordinator, name, None) is not None
        }
        trees["hisparse_management_other"] = tensor_attributes(
            coordinator, exclude=HOT_NAMES
        )
        backing = getattr(coordinator, "mem_pool_host", None)
        if backing is not None:
            payload = _pool_payload(backing)
            if getattr(backing, "shards", None) is not None:
                payload["shards"] = backing.shards
            trees["backing_kv_payload"] = payload
            trees["backing_descriptors"] = tensor_attributes(
                backing, exclude={*payload, "_local_refs"}
            )
    if family == "qwen":
        pool = getattr(coordinator, "mem_pool_device", None) or kv_pool
        if pool is not None:
            trees["compute_kv"] = _pool_payload(pool)
            trees["compute_pool_metadata"] = tensor_attributes(
                pool, exclude={"k_buffer", "v_buffer", "kv_buffer", "kv_buffer_rope"}
            )
        selector = (
            getattr(coordinator, "quest_algorithm", None)
            or getattr(backend, "quest_algorithm", None)
            or getattr(backend, "quest", None)
        )
        if selector is not None:
            trees["quest_persistent"] = tensor_attributes(selector)
    elif family == "deepseek-v4-flash" and kv_pool is not None:
        pool = kv_pool
        hot = getattr(coordinator, "mem_pool_device", None)
        if hot is not None:
            trees["compute_c4_kv"] = _pool_payload(hot)
        for attr, component in (
            ("swa_kv_pool", "compute_swa_kv"),
            ("c4_kv_pool", "compute_c4_kv"),
            ("c128_kv_pool", "compute_c128_kv"),
            ("unified_kv_pool", "compute_unified_kv"),
        ):
            child = getattr(pool, attr, None)
            if child is not None:
                trees.setdefault(component, {})[attr] = _pool_payload(child)
        for ratio, child in sorted(getattr(pool, "kv_pools", {}).items()):
            trees.setdefault("compressed_kv_registry", {})[str(ratio)] = _pool_payload(
                child
            )
        for ratio, child in sorted(getattr(pool, "index_pools", {}).items()):
            _add_indexer(trees, child, "ratio_" + str(ratio))
        for attr, component in (
            ("compress_state_pools", "compression_state"),
            ("indexer_compress_state_pools", "indexer_compression_state"),
        ):
            trees[component] = {
                str(index): {
                    "kv_score": getattr(
                        getattr(child, "kv_score_buffer", None), "kv_score", None
                    ),
                    "metadata": tensor_attributes(child),
                }
                for index, child in enumerate(getattr(pool, attr, []) or [])
                if child is not None
            }
        window = getattr(pool, "request_window", None)
        if window is not None:
            trees["request_swa_window"] = {
                "metadata": tensor_attributes(window),
                "state": _pool_payload(getattr(window, "state", None)),
            }
        trees["compute_pool_metadata"] = tensor_attributes(pool)
    elif family == "glm-5.3-flash" and kv_pool is not None:
        dsa = getattr(kv_pool, "full_kv_pool", None) or kv_pool
        trees["compute_kv"] = _pool_payload(dsa)
        _add_indexer(trees, dsa, "dsa")
        trees["compute_pool_metadata"] = tensor_attributes(
            dsa,
            exclude={
                "kv_buffer",
                "index_k_with_scale_buffer",
                "index_k_payload_buffer",
                "index_k_scale_buffer",
                "_compress_tail_k",
                "_compress_tail_score",
            },
        )
        mamba = getattr(kv_pool, "mamba_pool", None)
        if mamba is not None:
            state = getattr(mamba, "mamba_cache", None)
            trees["linear_state"] = tensor_attributes(state)
            trees["linear_state_metadata"] = tensor_attributes(mamba)
        trees["hybrid_wrapper_metadata"] = tensor_attributes(kv_pool)
    if backend is not None:
        trees["attention_backend_owned"] = _backend_trees(backend)
    if model_runner is not None:
        trees["decode_attention_backend_owned"] = {
            "active": _backend_trees(
                getattr(model_runner, "decode_attn_backend", None)
            ),
            "group": {
                str(index): _backend_trees(child)
                for index, child in enumerate(
                    getattr(model_runner, "decode_attn_backend_group", ()) or ()
                )
            },
        }
    if graph_runner is not None:
        trees["graph_live_owned"] = (
            {label: _graph_trees(runner) for label, runner in graph_runner.items()}
            if isinstance(graph_runner, dict)
            else _graph_trees(graph_runner)
        )
    return {
        name: selected
        for name, value in trees.items()
        if (selected := _selected(value)) is not None
    }


def normalization(component_bytes, context):
    for name, amount in component_bytes.items():
        _integer(amount, "component_bytes." + name)
    denominators = {
        "per_layer": context.get("layers"),
        "per_preallocated_request_slot": context.get("preallocated_request_slots"),
        "per_live_history_token": context.get("live_history_tokens"),
    }
    slots, length = context.get("preallocated_request_slots"), context.get(
        "max_context_len"
    )
    denominators["per_preallocated_history_token"] = (
        slots * length if slots is not None and length is not None else None
    )
    for label, value in denominators.items():
        if value is not None:
            _integer(
                value,
                label + ".denominator",
                positive=label != "per_live_history_token",
            )
    return {
        name: {
            label + "_bytes": amount / denominator if denominator else None
            for label, denominator in denominators.items()
        }
        for name, amount in component_bytes.items()
    }


def _indexer_regions(pool, label, *, family, compression_ratio=None):
    """Describe pool content without creating K/scale views or calling kernels."""
    packed = getattr(pool, "index_k_with_scale_buffer", None)
    payload = getattr(pool, "index_k_payload_buffer", None)
    scales = getattr(pool, "index_k_scale_buffer", None)
    if packed is None and payload is None and scales is None:
        return []
    head_dim = _integer(
        getattr(pool, "index_head_dim", None), "index_head_dim", positive=True
    )
    page_size = _integer(getattr(pool, "page_size", None), "page_size", positive=True)
    index_kpool = _integer(
        getattr(pool, "index_kpool", 1), "index_kpool", positive=True
    )
    if page_size % index_kpool:
        raise ValueError("index_kpool must divide the physical page size")
    tokens_per_row = page_size // index_kpool
    fp4 = bool(getattr(pool, "use_fp4_indexer", False))
    quant_block = (
        32
        if fp4
        else _integer(
            getattr(pool, "quant_block_size", 128), "quant_block_size", positive=True
        )
    )
    if head_dim % quant_block or (fp4 and head_dim % 2):
        raise ValueError("Indexer head dimension does not fit its quantization blocks")
    key_bytes_per_token = head_dim // 2 if fp4 else head_dim
    scales_per_token = head_dim // quant_block
    scale_element_bytes = 1 if fp4 else 4
    start_layer = getattr(pool, "start_layer", 0)
    start_layer = 0 if start_layer is None else _integer(start_layer, "start_layer")
    common = {
        "family": family,
        "index_head_dim": head_dim,
        "quantization_block_elements": quant_block,
        "sequence_compression_ratio": compression_ratio,
        "index_kpool": index_kpool,
        "tokens_per_physical_row": tokens_per_row,
        "key_element_bits": 4 if fp4 else 8,
        "scale_element_bytes": scale_element_bytes,
        "numeric_key_format": "fp4_e2m1" if fp4 else "fp8_e4m3fn",
        "numeric_scale_format": "ue8m0" if fp4 else "float32",
        "index_page_size": getattr(pool, "index_page_size", tokens_per_row),
        "preshuffle_layout": (
            "aiter_separate_page8"
            if getattr(pool, "uses_aiter_fp4_layout", False)
            else (
                "page8"
                if getattr(pool, "index_k_page8", False)
                else "runtime_backend_dependent"
            )
        ),
    }
    regions = []
    if packed is not None:
        for index, tensor in enumerate(packed):
            rows, row_bytes = map(int, tensor.shape)
            key_row_bytes = tokens_per_row * key_bytes_per_token
            scale_row_bytes = tokens_per_row * scales_per_token * scale_element_bytes
            if (
                int(tensor.element_size()) != 1
                or row_bytes != key_row_bytes + scale_row_bytes
            ):
                raise ValueError(
                    "Packed indexer row shape differs from its K/scale layout"
                )
            for content, offset, per_row, logical_shape in (
                (
                    "indexer_k_vectors",
                    0,
                    key_row_bytes,
                    [rows, tokens_per_row, head_dim],
                ),
                (
                    "indexer_k_scales",
                    key_row_bytes,
                    scale_row_bytes,
                    [rows, tokens_per_row, scales_per_token],
                ),
            ):
                regions.append(
                    {
                        "region_id": f"{label}.layer_{index}.{content}",
                        "tensor_name": f"indexer_packed_keys_scales.{label}.buffers.{index}",
                        "content": content,
                        "logical_content_bytes": rows * per_row,
                        "logical_shape": logical_shape,
                        "geometry": {
                            "layout": "page_keys_then_scales",
                            "rows": rows,
                            "row_stride_bytes": row_bytes,
                            "region_offset_within_row_bytes": offset,
                            "region_bytes_per_row": per_row,
                        },
                        "layer_id": start_layer + index,
                        "layer_id_domain": "pool_declared_layer",
                        "pool_local_layer_id": index,
                        "model_layer_ids": None,
                        **common,
                    }
                )
    else:
        for values, attribute, content in (
            (payload, "index_k_payload_buffer", "indexer_k_vectors"),
            (scales, "index_k_scale_buffer", "indexer_k_scales"),
        ):
            for index, tensor in enumerate(values or []):
                logical_bytes = math.prod(tensor.shape) * int(tensor.element_size())
                regions.append(
                    {
                        "region_id": f"{label}.layer_{index}.{content}",
                        "tensor_name": f"{content}.{label}.{attribute}.{index}",
                        "content": content,
                        "logical_content_bytes": logical_bytes,
                        "logical_shape": list(tensor.shape),
                        "geometry": {"layout": "whole_tensor"},
                        "layer_id": start_layer + index,
                        "layer_id_domain": "pool_declared_layer",
                        "pool_local_layer_id": index,
                        "model_layer_ids": None,
                        **common,
                    }
                )
    return regions


def collect_model_logical_regions(
    family,
    *,
    model_runner=None,
    model=None,
    coordinator=None,
    kv_pool=None,
    allocator=None,
    request_pool=None,
    backend=None,
    graph_runner=None,
):
    """K/scale contents of packed allocations, separate from physical bytes.

    Owner arguments match ``collect_model_tensor_trees``. Qwen has no trained
    indexer K cache; its Quest bounds are tensor allocations in quest_persistent.
    This API does not implement an indexer K cache or measure runtime hit rates.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unsupported experiment model family {family}")
    _, kv_pool, _, _, _, _ = _base_owners(
        model_runner,
        coordinator,
        kv_pool,
        allocator,
        request_pool,
        backend,
        graph_runner,
    )
    if kv_pool is None or family == "qwen":
        return []
    if family == "glm-5.3-flash":
        dsa = getattr(kv_pool, "full_kv_pool", None) or kv_pool
        regions = _indexer_regions(dsa, "dsa", family=family)
        mapping = getattr(kv_pool, "full_attention_layer_id_mapping", {})
        for region in regions:
            ids = [
                global_id
                for global_id, local_id in mapping.items()
                if local_id == region["pool_local_layer_id"]
            ]
            region["model_layer_ids"] = sorted(ids) if ids else None
        return regions
    regions = [
        region
        for ratio, pool in sorted(getattr(kv_pool, "index_pools", {}).items())
        for region in _indexer_regions(
            pool, "ratio_" + str(ratio), family=family, compression_ratio=ratio
        )
    ]
    mapping = getattr(kv_pool, "layer_mapping", ()) or ()
    for region in regions:
        ratio = region["sequence_compression_ratio"]
        local = region["pool_local_layer_id"]
        ids = [
            global_id
            for global_id, item in enumerate(mapping)
            if item is not None
            and getattr(item, "compress_ratio", item[0]) == ratio
            and getattr(item, "compress_layer_id", item[1]) == local
        ]
        region["model_layer_ids"] = ids if ids else None
    return regions


def bind_logical_regions(regions, inventory):
    """Validate region extents and bind contents to snapshot-local storages."""
    if regions is None:
        return []
    if not isinstance(regions, list):
        raise TypeError("Logical regions must be a list")
    result, seen, geometries = [], set(), {}
    for raw in regions:
        if not isinstance(raw, dict):
            raise TypeError("A logical region must be a dictionary")
        region = copy.deepcopy(raw)
        region_id, path = region.get("region_id"), region.get("tensor_name")
        if not isinstance(region_id, str) or not region_id or region_id in seen:
            raise ValueError("Logical region IDs must be nonempty and unique")
        seen.add(region_id)
        if path not in inventory["tensors"]:
            raise ValueError(f"Logical region tensor is absent from inventory: {path}")
        tensor = inventory["tensors"][path]
        amount = _integer(region.get("logical_content_bytes"), "logical_content_bytes")
        geometry = region.get("geometry", {})
        if geometry.get("layout") == "whole_tensor":
            if amount != tensor["logical_tensor_bytes"]:
                raise ValueError(
                    "Whole-tensor region bytes must equal its tensor bytes"
                )
            interval = (0, tensor["logical_tensor_bytes"])
        elif geometry.get("layout") == "page_keys_then_scales":
            rows = _integer(geometry.get("rows"), "region.rows")
            stride = _integer(
                geometry.get("row_stride_bytes"),
                "region.row_stride_bytes",
                positive=True,
            )
            offset = _integer(
                geometry.get("region_offset_within_row_bytes"), "region.offset"
            )
            size = _integer(
                geometry.get("region_bytes_per_row"), "region.bytes_per_row"
            )
            if (
                len(tensor["shape"]) != 2
                or rows != tensor["shape"][0]
                or stride != tensor["shape"][1] * tensor["element_bytes"]
                or tensor["stride"] != [tensor["shape"][1], 1]
                or offset + size > stride
                or amount != rows * size
            ):
                raise ValueError("Packed region does not fit its contiguous page rows")
            interval = (offset, offset + size)
        else:
            raise ValueError("Unknown logical region layout")
        for previous in geometries.setdefault(path, []):
            if max(previous[0], interval[0]) < min(previous[1], interval[1]):
                raise ValueError("Logical content regions overlap")
        geometries[path].append(interval)
        region["allocation_group"] = tensor["allocation_group"]
        region["device"] = tensor["device"]
        region["physical_storage_bytes"] = tensor["storage_bytes"]
        region["adds_physical_allocation"] = False
        result.append(region)
    return result


def capture_memory_snapshot(
    context,
    named_trees,
    *,
    allocator_metrics=None,
    nvml_samples=None,
    graph_pool_summary=None,
    regions=None,
    device_types=("cuda",),
):
    """Capture host metadata only; callers own stream/phase synchronization."""
    if not isinstance(context, dict) or context.get("phase") not in PHASES:
        raise ValueError("A named experiment memory phase is required")
    if not isinstance(context.get("snapshot_id"), str) or not context["snapshot_id"]:
        raise ValueError("snapshot_id must be a nonempty string")
    if context.get("family") is not None and context["family"] not in FAMILIES:
        raise ValueError("Unknown experiment model family")
    for name in ("rank", "world_size"):
        if context.get(name) is not None:
            _integer(context[name], name, positive=name == "world_size")
    if (
        context.get("rank") is not None
        and context.get("world_size") is not None
        and context["rank"] >= context["world_size"]
    ):
        raise ValueError("rank must be inside world_size")
    json.dumps(context, allow_nan=False)
    inventory = describe_tensor_storages(named_trees, device_types=device_types)
    allocator = {name: None for name in ALLOCATOR_FIELDS}
    if allocator_metrics is not None:
        if not isinstance(allocator_metrics, dict):
            raise TypeError("Allocator metrics must be a dictionary")
        allocator.update(copy.deepcopy(allocator_metrics))
        for name in ALLOCATOR_FIELDS:
            if allocator.get(name) is not None:
                _integer(allocator[name], "allocator." + name)
    context = copy.deepcopy(context)
    context.setdefault("process_id", os.getpid())
    nvml = summarize_nvml_samples(nvml_samples)
    if (
        context.get("device_uuid") is not None
        and nvml.get("device_uuid") is not None
        and context["device_uuid"] != nvml["device_uuid"]
    ):
        raise ValueError("NVML sample device differs from snapshot compute device")
    result = {
        "schema_version": 1,
        "contract": CONTRACT,
        "context": context,
        "inventory_scope": "explicit_named_live_tensor_owners",
        **inventory,
        "normalization": normalization(inventory["inclusive_component_bytes"], context),
        "allocator": allocator,
        "allocator_peak_scope": allocator.get(
            "peak_scope",
            (
                "unmeasured"
                if allocator_metrics is None
                else "caller_must_define_reset_interval"
            ),
        ),
        "nvml": nvml,
        "graph_pools": (
            copy.deepcopy(graph_pool_summary)
            if graph_pool_summary is not None
            else {
                "status": "unmeasured",
                "graph_private_pool_reserved_bytes": None,
                "graph_private_pool_allocated_bytes": None,
            }
        ),
        "logical_regions": bind_logical_regions(regions, inventory),
        "rules": {
            "unique_storage_ids": "local_to_one_snapshot_process_and_device",
            "named_storage_vs_allocator_nvml": "overlapping_scopes_do_not_add",
            "logical_regions_vs_physical_storage": "regions_describe_contents_not_additional_allocations",
            "graph_live_owned_vs_graph_pool": "overlap_possible_do_not_add",
            "absent_owner": "not_evidence_of_zero_memory",
            "unknown_values": "null_not_zero",
            "preallocated_history_denominator": "slots_times_max_context_not_observed_live_tokens",
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def summarize_nvml_samples(samples):
    """Summarize caller-supplied NVML samples; maximum is only a sampled peak."""
    if samples is None:
        return {
            "status": "unmeasured",
            "sample_count": 0,
            "sampled_used_peak_bytes": None,
            "sampling_interval_seconds": None,
            "samples": None,
        }
    if not isinstance(samples, list) or not samples:
        raise ValueError("Measured NVML samples must be a nonempty list")
    result, identities, last_time = [], set(), None
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError("Malformed NVML sample")
        time = sample.get("timestamp_seconds")
        if (
            type(time) not in (int, float)
            or not math.isfinite(time)
            or (last_time is not None and time < last_time)
        ):
            raise ValueError("NVML timestamps must be finite and monotonic")
        last_time = time
        _integer(sample.get("used_bytes"), "nvml.used_bytes")
        identity = sample.get("device_uuid")
        if not isinstance(identity, str) or not identity:
            raise ValueError("NVML samples require device_uuid")
        identities.add(identity)
        result.append(copy.deepcopy(sample))
    if len(identities) != 1:
        raise ValueError("One NVML sample series must identify exactly one device")
    times = [sample["timestamp_seconds"] for sample in result]
    return {
        "status": "measured",
        "sample_count": len(result),
        "device_uuid": next(iter(identities)),
        "sampled_used_peak_bytes": max(sample["used_bytes"] for sample in result),
        "first_used_bytes": result[0]["used_bytes"],
        "last_used_bytes": result[-1]["used_bytes"],
        "sampling_interval_seconds": [
            times[i] - times[i - 1] for i in range(1, len(times))
        ],
        "sampled_peak_is_exact_peak": False,
        "samples": result,
    }


def allocator_metrics_from_cuda(cuda, device=None, *, peak_scope=None):
    """Read CUDA allocator counters only; do not reset peaks or synchronize."""
    stats = cuda.memory_stats(device)
    names = {
        "allocated_bytes": "allocated_bytes.all.current",
        "reserved_bytes": "reserved_bytes.all.current",
        "active_bytes": "active_bytes.all.current",
        "inactive_split_bytes": "inactive_split_bytes.all.current",
        "peak_allocated_bytes": "allocated_bytes.all.peak",
        "peak_reserved_bytes": "reserved_bytes.all.peak",
    }
    result = {
        name: int(stats[key]) if key in stats else None for name, key in names.items()
    }
    result["peak_scope"] = peak_scope or "caller_must_define_reset_interval"
    return result
