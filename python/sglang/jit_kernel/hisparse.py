from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import get_jit_cuda_arch, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    collect_metrics: bool = False,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla, collect_metrics
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                (
                    "load_cache_to_device_buffer_with_metrics"
                    if collect_metrics
                    else "load_cache_to_device_buffer"
                ),
                (
                    f"load_cache_to_device_buffer_with_metrics<{template_args}>"
                    if collect_metrics
                    else f"load_cache_to_device_buffer<{template_args}>"
                ),
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
    cache_metrics: torch.Tensor | None = None,
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

    if cache_metrics is not None and (
        cache_metrics.dtype != torch.int64
        or cache_metrics.device != top_k_tokens.device
        or cache_metrics.shape != (5,)
        or not cache_metrics.is_contiguous()
    ):
        raise ValueError(
            "Cache metrics must be contiguous int64[5] on the compute device"
        )

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=False,
        collect_metrics=cache_metrics is not None,
    )

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    entry = (
        module.load_cache_to_device_buffer
        if cache_metrics is None
        else module.load_cache_to_device_buffer_with_metrics
    )
    extra = [] if cache_metrics is None else [cache_metrics]
    entry(
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
        *extra,
    )


@functools.cache
def _jit_plan_mha_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    collect_metrics: bool,
) -> Module:
    args = make_cpp_args(block_size, num_top_k, hot_buffer_size, collect_metrics)
    return load_jit(
        "hisparse_mha_plan_only",
        *make_cpp_args(
            item_size_bytes, block_size, num_top_k, hot_buffer_size, collect_metrics
        ),
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            ("plan_cache_to_device_buffer_mha", f"plan_cache_to_device_buffer_mha<{args}>")
        ],
        build_directory=_xm_build_directory(
            "hisparse_mha_plan_only", "hisparse.cuh", *args
        ),
    )


def _xm_build_directory(name: str, source_name: str, *args) -> str | None:
    """Persist only the experimental JITs when an explicit cache is set."""
    cache_root = os.environ.get("SGLANG_JIT_CACHE_DIR")
    if not cache_root:
        return None
    source = Path(__file__).parent / "csrc" / source_name
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    architecture = get_jit_cuda_arch().target_name
    directory = Path(cache_root) / (
        f"{name}_{'_'.join(str(arg) for arg in args)}_{architecture}_{source_hash}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


@functools.cache
def _empty_plan_metrics() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64)


def prepare_plan_cache_to_device_buffer_mha(
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int = 1024,
    collect_metrics: bool = False,
) -> None:
    """Compile before capture; the default cache-copy ABI remains untouched."""
    _jit_plan_mha_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, collect_metrics
    )
    _empty_plan_metrics()


def plan_cache_to_device_buffer_mha(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    miss_host_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    num_real_reqs: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int = 1024,
    cache_metrics: torch.Tensor | None = None,
) -> None:
    """Resolve without payload I/O, for the following fused attention kernel.

    ``miss_host_locs`` uses original selection order: -1 means hot/newest,
    otherwise it is the backing row. ``top_k_device_locs`` names the hot hit
    or refill destination. The caller must complete refill on the same stream
    before another resolver or cache consumer can observe the updated mapping.
    """
    if not 0 < num_top_k <= hot_buffer_size <= 32767:
        raise ValueError("Plan-only resolve requires 0 < top_k <= hot <= 32767")
    if block_size % 32 or not 0 < block_size <= 1024:
        raise ValueError("Plan-only resolve requires a warp-aligned CUDA block")
    if top_k_tokens.ndim != 2 or top_k_tokens.shape[1] != num_top_k:
        raise ValueError("Plan-only selection must be [batch, top_k]")
    if top_k_tokens.device.type != "cuda":
        raise ValueError("Plan-only resolve requires the CUDA compute device")
    batch_size = top_k_tokens.shape[0]
    if (
        device_buffer_tokens.ndim != 2
        or device_buffer_locs.shape != device_buffer_tokens.shape
        or device_buffer_tokens.shape[1] < hot_buffer_size + 1
        or host_cache_locs.ndim != 2
        or host_cache_locs.shape[0] != device_buffer_tokens.shape[0]
        or lru_slots.shape != (device_buffer_tokens.shape[0], hot_buffer_size)
        or req_pool_indices.shape != (batch_size,)
        or seq_lens.shape != (batch_size,)
        or num_real_reqs.shape != (1,)
    ):
        raise ValueError("Plan-only request, cache, LRU or length shapes disagree")
    tensors = (
        (top_k_tokens, torch.int32),
        (top_k_device_locs, torch.int32),
        (miss_host_locs, torch.int64),
        (device_buffer_tokens, torch.int32),
        (device_buffer_locs, torch.int32),
        (host_cache_locs, torch.int64),
        (lru_slots, torch.int16),
        (req_pool_indices, torch.int64),
        (num_real_reqs, torch.int32),
    )
    for tensor, dtype in tensors:
        if tensor.dtype != dtype or tensor.device != top_k_tokens.device:
            raise ValueError("Plan-only tensors require their declared dtype/device")
        if not tensor.is_contiguous():
            raise ValueError("Plan-only tensors must be contiguous")
    if top_k_device_locs.shape != top_k_tokens.shape or miss_host_locs.shape != top_k_tokens.shape:
        raise ValueError("Plan outputs must match the complete selection shape")
    if seq_lens.dtype not in (torch.int32, torch.int64) or seq_lens.device != top_k_tokens.device:
        raise ValueError("Plan-only sequence lengths must be CUDA int32/int64")
    if not seq_lens.is_contiguous():
        raise ValueError("Plan-only sequence lengths must be contiguous")
    for tensor in (device_k_buffer, device_v_buffer):
        if (
            tensor.device != top_k_tokens.device
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != 3
            or tensor.shape[0] <= 0
            or not tensor.is_contiguous()
            or tensor.shape[1] * tensor.shape[2] * tensor.element_size()
            != item_size_bytes
        ):
            raise ValueError("Plan-only KV buffers require matching CUDA BF16 rows")
    if device_k_buffer.shape != device_v_buffer.shape:
        raise ValueError("Plan-only K/V hot-buffer shapes disagree")
    if cache_metrics is not None and (
        cache_metrics.dtype != torch.int64
        or cache_metrics.shape != (5,)
        or cache_metrics.device != top_k_tokens.device
        or not cache_metrics.is_contiguous()
    ):
        raise ValueError("Plan metrics must be contiguous CUDA int64[5]")
    top_k_device_locs.fill_(-1)
    miss_host_locs.fill_(-1)
    _jit_plan_mha_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        cache_metrics is not None,
    ).plan_cache_to_device_buffer_mha(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        device_k_buffer,
        device_v_buffer,
        top_k_device_locs,
        miss_host_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        1,
        item_size_bytes,
        cache_metrics if cache_metrics is not None else _empty_plan_metrics(),
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
