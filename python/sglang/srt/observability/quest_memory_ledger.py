"""Metadata-only accounting for persistent Quest/HiSparse allocations.

No tensor values, KV copies, kernels, device synchronization or raw addresses
are emitted. This is an inventory of the named owners, not total CUDA memory.
"""

from sglang.srt.observability.mpu_stage_timing import describe_retained_gpu_pools


def cuda_attributes(owner, exclude=()):
    def select(value):
        # CUDA streams/events also have a .device; they are not tensors.
        if callable(getattr(value, "untyped_storage", None)):
            return (
                value
                if getattr(getattr(value, "device", None), "type", None) == "cuda"
                else None
            )
        if isinstance(value, (list, tuple)):
            items = {
                str(i): selected
                for i, item in enumerate(value)
                if (selected := select(item)) is not None
            }
            return items or None
        if isinstance(value, dict):
            items = {
                str(k): selected
                for k, item in value.items()
                if (selected := select(item)) is not None
            }
            return items or None
        return None

    return {
        name: selected
        for name, value in vars(owner).items()
        if name not in exclude and (selected := select(value)) is not None
    }


def allocation_bytes(tensors):
    """Include allocator padding; deduplicate views using host-side metadata."""
    storages = {}

    def visit(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = int(storage.nbytes())

    visit(tensors)
    return sum(storages.values())


def collect_quest_memory(coordinator):
    c = coordinator
    pool, backing = c.mem_pool_device, c.mem_pool_host
    selector = cuda_attributes(c.quest_algorithm)
    if (
        not {
            "page_k_bounds",
            "page_k_min",
            "page_k_max",
            "running_k_min",
            "running_k_max",
        }
        <= selector.keys()
    ):
        raise ValueError(
            "Quest memory ledger requires initialized CUDA selector storage"
        )
    trees = {
        "kv": {"k": pool.k_buffer, "v": pool.v_buffer},
        "fixed": {
            "quest": selector,
            "coordinator": cuda_attributes(c),
            "request_map": c.req_to_token_pool.req_to_token,
            "kv_descriptors": {
                name: getattr(pool, name) for name in ("k_data_ptrs", "v_data_ptrs")
            },
        },
        "capacity_dependent": {
            "logical_to_physical": pool.full_to_hisparse_device_index_mapping,
            "allocator_mapping_alias": c.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
        },
        "backend_specific": cuda_attributes(backing, exclude=("shards", "_local_refs")),
        "allocator_dynamic": {
            "physical": cuda_attributes(
                c.token_to_kv_pool_allocator.hisparse_attn_allocator
            ),
            "logical": cuda_attributes(
                c.token_to_kv_pool_allocator.logical_attn_allocator
            ),
            "outer": cuda_attributes(
                c.token_to_kv_pool_allocator,
                exclude=("full_to_hisparse_device_index_mapping",),
            ),
        },
    }
    groups = {key: describe_retained_gpu_pools(value) for key, value in trees.items()}
    combined = describe_retained_gpu_pools(trees)
    backing_bytes = allocation_bytes(
        backing.shards
        if c.backing_store == "mpu"
        else [backing.k_buffer, backing.v_buffer]
    )
    return {
        **groups["fixed"],
        "allocated_compute_kv_bytes": groups["kv"]["retained_gpu_pool_bytes"],
        "allocated_backing_kv_bytes": backing_bytes,
        "gpu_memory_ledger": {
            "schema_version": 1,
            "contract": "qwen_quest_v1",
            "groups": groups,
            "named_compute_bytes": combined["retained_gpu_pool_bytes"],
            "named_compute_shapes": combined["retained_gpu_pool_shapes"],
            "rules": {
                "kv": "fixed across variants, ratios and idle snapshots within an input length",
                "fixed": "fixed across variants, ratios and idle snapshots within an input length",
                "capacity_dependent": "logical address map; report growth, require stable within a capacity point",
                "backend_specific": "CPU/MPU descriptors differ; require stable within the same backend and topology",
                "allocator_dynamic": "allocator worklists may change after requests; report each idle snapshot",
            },
            "excludes": [
                "model weights",
                "activations",
                "attention backend workspaces",
                "CUDA graph private pools",
                "CUDA caching allocator reserved but unused memory",
            ],
        },
    }
