from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
    )


def load_cache_to_device_buffer_mha(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_k_cache: torch.Tensor,
    host_v_cache: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """MHA/GQA variant of :func:`load_cache_to_device_buffer_mla`.

    Differs from the MLA variant in that both K and V tensors are passed
    through (``IsMLA=false`` in ``hisparse.cuh``), so each miss copies both
    K and V. ``item_size_bytes`` must be the per-token stride of a single
    side (K or V), i.e. ``head_num * head_dim * dtype.itemsize``.
    """
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=False
    )

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_k_cache,
        host_v_cache,
        device_k_buffer,
        device_v_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
    )


@functools.cache
def _jit_sparse_mpu_module(
    item_size_bytes: int, block_size: int, num_top_k: int, hot_buffer_size: int
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size)
    return load_jit(
        "sparse_mpu_cache",
        *make_cpp_args(item_size_bytes, block_size, num_top_k, hot_buffer_size),
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_from_mpu_to_device_buffer",
                f"load_cache_from_mpu_to_device_buffer<{template_args}>",
            )
        ],
    )


def load_cache_from_mpu_to_device_buffer_mha(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    mpu_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    peer_ptrs: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    shard_rows: int,
    layer_id: int,
    peer_rows: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """Resolve HiSparse hits/misses using compute-issued peer loads for misses.

    LRU, newest-token handling and the short-sequence fast path are shared with
    the pinned-host MHA kernel. ``peer_ptrs`` holds [shards, 2, layers] base
    addresses on the compute GPU; no storage-GPU tensor is passed to dispatch.
    ``peer_rows`` excludes padding in the last shard. The owner must order
    backing stores before this call and retain peer allocations until it ends.
    """
    from sglang.jit_kernel.mpu_io import validate_peer_table

    validate_peer_table(peer_ptrs, shard_rows, item_size_bytes)
    if (
        type(num_top_k) is not int
        or type(hot_buffer_size) is not int
        or not 0 < num_top_k <= hot_buffer_size <= 32767
    ):
        raise ValueError("MPU requires 0 < num_top_k <= hot_buffer_size <= 32767")
    if (
        type(block_size) is not int
        or not 0 < block_size <= 1024
        or block_size % 32
        or type(page_size) is not int
        or page_size != 1
    ):
        raise ValueError("MPU requires a warp-aligned block size and page_size=1")
    if type(layer_id) is not int or not 0 <= layer_id < peer_ptrs.shape[2]:
        raise ValueError("MPU layer_id is out of range")
    if (
        type(peer_rows) is not int
        or not 0 < peer_rows <= shard_rows * peer_ptrs.shape[0]
    ):
        raise ValueError("peer_rows must exclude padding and fit in the peer shards")
    matrices = (
        (top_k_tokens, torch.int32),
        (device_buffer_tokens, torch.int32),
        (mpu_cache_locs, torch.int64),
        (device_buffer_locs, torch.int32),
        (top_k_device_locs, torch.int32),
        (lru_slots, torch.int16),
    )
    for tensor, dtype in matrices:
        if tensor.ndim != 2 or tensor.dtype != dtype or tensor.stride(1) != 1:
            raise ValueError(
                "MPU resolve matrices require the expected dtype and contiguous rows"
            )
    if device_buffer_tokens.shape != device_buffer_locs.shape:
        raise ValueError("MPU buffer token and location shapes differ")
    if device_buffer_tokens.stride(0) != device_buffer_locs.stride(0):
        raise ValueError("MPU buffer token and location row strides differ")
    if device_buffer_tokens.shape[1] < hot_buffer_size + page_size:
        raise ValueError("MPU device buffer must include the newest-token page")
    if lru_slots.shape != (device_buffer_tokens.shape[0], hot_buffer_size):
        raise ValueError("MPU LRU shape does not match the request buffer")
    if mpu_cache_locs.shape[0] != device_buffer_tokens.shape[0]:
        raise ValueError("MPU backing map request count does not match the buffer")
    batch_size = top_k_tokens.shape[0]
    if (
        top_k_tokens.shape[1] != num_top_k
        or top_k_device_locs.shape != top_k_tokens.shape
    ):
        raise ValueError(
            "MPU top-k input and output shapes must match [batch, num_top_k]"
        )
    if req_pool_indices.shape != (batch_size,) or req_pool_indices.dtype != torch.int64:
        raise ValueError("MPU request indices must be int64 [batch]")
    if seq_lens.shape != (batch_size,) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("MPU sequence lengths must be int32 or int64 [batch]")
    for tensor in (device_k_buffer, device_v_buffer):
        if (
            tensor.ndim < 2
            or not tensor.shape[0]
            or not tensor.is_contiguous()
            or tensor.dtype != torch.bfloat16
        ):
            raise ValueError("MPU local K/V slabs must be contiguous [rows, ...]")
        if tensor[0].numel() * tensor.element_size() != item_size_bytes:
            raise ValueError("MPU local K/V row size does not match item_size_bytes")
    if (
        device_k_buffer.shape != device_v_buffer.shape
        or device_k_buffer.dtype != device_v_buffer.dtype
    ):
        raise ValueError("MPU local K/V shapes and dtypes must match")
    tensors = [tensor for tensor, _ in matrices] + [
        req_pool_indices,
        seq_lens,
        device_k_buffer,
        device_v_buffer,
    ]
    if any(tensor.device != peer_ptrs.device for tensor in tensors):
        raise ValueError("All MPU resolve tensors must be on the compute GPU")
    if not req_pool_indices.is_contiguous() or not seq_lens.is_contiguous():
        raise ValueError("MPU resolve vectors must be contiguous")
    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [batch_size], dtype=torch.int32, device=peer_ptrs.device
        )
    if (
        num_real_reqs.shape != (1,)
        or num_real_reqs.dtype != torch.int32
        or num_real_reqs.device != peer_ptrs.device
        or not num_real_reqs.is_contiguous()
    ):
        raise ValueError("num_real_reqs must be int32 [1] on the compute GPU")
    if not batch_size:
        return
    with torch.cuda.device(peer_ptrs.device):
        _jit_sparse_mpu_module(
            item_size_bytes, block_size, num_top_k, hot_buffer_size
        ).load_cache_from_mpu_to_device_buffer(
            top_k_tokens,
            device_buffer_tokens,
            mpu_cache_locs,
            device_buffer_locs,
            peer_ptrs,
            device_k_buffer,
            device_v_buffer,
            top_k_device_locs,
            req_pool_indices,
            seq_lens,
            lru_slots,
            num_real_reqs,
            page_size,
            item_size_bytes,
            shard_rows,
            layer_id,
            peer_rows,
        )
