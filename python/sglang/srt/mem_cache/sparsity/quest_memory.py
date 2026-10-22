"""Persistent Quest metadata sizing, without importing the GPU runtime."""


def estimate_quest_storage_bytes(
    top_k: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    max_reqs: int,
    max_context_len: int,
    kv_heads: int,
    head_dim: int,
) -> int:
    """Bytes of persistent tensors allocated by QuestAlgorithm.init_storage.

    Views do not add storage. This excludes allocator rounding and temporary
    tensors used by page scoring, prefill and graph capture.
    """
    layers = end_layer - start_layer
    if (
        min(top_k, page_size, layers, max_reqs, max_context_len, kv_heads, head_dim)
        <= 0
    ):
        raise ValueError("Quest storage dimensions must all be positive")
    if top_k % page_size:
        raise ValueError("Quest top_k must be divisible by page_size")
    pages = (max_context_len + page_size - 1) // page_size
    bounds = 4 * layers * max_reqs * (pages + 1) * kv_heads * head_dim
    step_and_counters = max_reqs * (4 * page_size + 4 * top_k + 21)
    constants = 8 * (2 * page_size + top_k + pages)
    return bounds + step_and_counters + constants


def estimate_quest_auxiliary_bytes(
    *,
    top_k: int,
    quest_page_size: int,
    start_layer: int,
    end_layer: int,
    max_reqs: int,
    max_context_len: int,
    kv_heads: int,
    head_dim: int,
    device_buffer_size: int = 0,
    device_token_capacity: int = 0,
    host_to_device_ratio: int = 1,
    kv_page_size: int = 64,
    logical_token_capacity: int | None = None,
) -> int:
    """Reserve Quest tensors and request/address tables outside the KV pool.

    A positive device_buffer_size includes the HiSparse coordinator and its
    logical-to-physical mapping. Host K/V bytes are not GPU memory. Prefill
    workspaces, allocator rounding and graph/kernel temporaries are excluded;
    they still require the headroom controlled by mem_fraction_static.
    """
    total = estimate_quest_storage_bytes(
        top_k,
        quest_page_size,
        start_layer,
        end_layer,
        max_reqs,
        max_context_len,
        kv_heads,
        head_dim,
    )
    # ReqToTokenPool plus the sparse backend's packed indices and lengths.
    total += 4 * max_reqs * max_context_len
    total += 4 * (max_reqs * top_k + 4 * max_reqs + top_k + 3)
    if device_buffer_size:
        if (
            min(
                device_buffer_size,
                device_token_capacity,
                host_to_device_ratio,
                kv_page_size,
            )
            <= 0
        ):
            raise ValueError("HiSparse memory dimensions must all be positive")
        layers = end_layer - start_layer
        padded = device_buffer_size + kv_page_size
        total += 8 * max_reqs * max_context_len  # req_to_host_pool
        total += 8 * max_reqs * padded  # req_to_device_buffer
        total += 8 * layers * max_reqs * padded  # token IDs and locations
        total += 2 * (layers * max_reqs + 1) * device_buffer_size  # LRU
        total += 4 * (max_reqs * top_k + 1)  # swap-in output and real BS
        # The address map scales with logical capacity; K/V tensors do not.
        if logical_token_capacity is None:
            logical_token_capacity = device_token_capacity * host_to_device_ratio
        if type(logical_token_capacity) is not int or logical_token_capacity < 0:
            raise ValueError(
                "HiSparse logical token capacity must be a nonnegative integer"
            )
        total += 8 * (logical_token_capacity + kv_page_size + 1)
    else:
        total += max_reqs  # Quest-only first-decode update mask (bool)
    return total
