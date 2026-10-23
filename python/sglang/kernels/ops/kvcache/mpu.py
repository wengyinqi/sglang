"""Compute-GPU-only load/store for linear MLA and paged C4 peer payloads."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "mpu_peer_records",
        cuda_files=["kvcacheio/mpu.cuh"],
        cuda_wrappers=[
            ("enable_peer_access", "mpu_enable_peer_access"),
            ("zero", "mpu_zero"),
            ("transfer", "mpu_transfer"),
            ("replay", "mpu_replay"),
        ],
    )


def _metadata(tensor, device, dtype, ndim):
    if (
        tensor.device != device
        or not tensor.is_cuda
        or tensor.dtype != dtype
        or tensor.ndim != ndim
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "MPU metadata must be contiguous with its declared dtype on the compute GPU"
        )


def _contract(
    peer_ptrs, local_ptrs, peer_slots, local_slots, item_bytes, layer_id, c4_layout
):
    _metadata(peer_ptrs, peer_ptrs.device, torch.int64, 1)
    _metadata(local_ptrs, peer_ptrs.device, torch.int64, 1)
    if peer_ptrs.shape != local_ptrs.shape or not peer_ptrs.numel():
        raise ValueError("MPU pointers require one peer/local address per layer")
    if any(type(v) is not int or v <= 0 for v in (peer_slots, local_slots, item_bytes)):
        raise ValueError("MPU capacities and record bytes must be positive integers")
    if item_bytes % 8 or type(c4_layout) is not bool:
        raise ValueError("MPU requires 8-byte records and an explicit layout")
    if type(layer_id) is not int or not -1 <= layer_id < peer_ptrs.numel():
        raise ValueError("MPU layer is outside pointer table")
    if c4_layout and (item_bytes != 584 or peer_slots % 64 or local_slots % 64):
        raise ValueError("C4 requires 64 slots/page and 576 value + 8 scale bytes")


def enable_peer_access(compute_device, storage_device):
    if any(type(v) is not int or v < 0 for v in (compute_device, storage_device)):
        raise ValueError("MPU ordinals must be nonnegative integers")
    if compute_device == storage_device:
        raise ValueError("Compute and storage devices must be different")
    with torch.cuda.device(compute_device):
        _module().enable_peer_access(compute_device, storage_device)


def zero_peer(peer_ptrs, byte_count):
    _metadata(peer_ptrs, peer_ptrs.device, torch.int64, 1)
    if type(byte_count) is not int or byte_count <= 0 or byte_count % 8:
        raise ValueError("MPU zero requires an aligned positive byte count")
    with torch.cuda.device(peer_ptrs.device):
        _module().zero(peer_ptrs, byte_count)
        peer_ptrs.record_stream(torch.cuda.current_stream(peer_ptrs.device))


def transfer_records(
    peer_ptrs,
    local_ptrs,
    peer_indices,
    local_indices,
    *,
    peer_slots,
    local_slots,
    item_bytes,
    layer_id=-1,
    to_peer,
    c4_layout=False,
):
    _contract(
        peer_ptrs, local_ptrs, peer_slots, local_slots, item_bytes, layer_id, c4_layout
    )
    for indices in (peer_indices, local_indices):
        _metadata(indices, peer_ptrs.device, torch.int64, 1)
    if peer_indices.shape != local_indices.shape or type(to_peer) is not bool:
        raise ValueError("MPU index counts must agree and direction must be explicit")
    if not peer_indices.numel():
        return
    with torch.cuda.device(peer_ptrs.device):
        _module().transfer(
            peer_ptrs,
            local_ptrs,
            peer_indices,
            local_indices,
            peer_slots,
            local_slots,
            item_bytes,
            layer_id,
            to_peer,
            c4_layout,
        )
        stream = torch.cuda.current_stream(peer_ptrs.device)
        for t in (peer_ptrs, local_ptrs, peer_indices, local_indices):
            t.record_stream(stream)


def replay_misses(
    peer_ptrs,
    local_ptrs,
    miss_src,
    miss_dst,
    miss_count,
    num_real_reqs,
    *,
    peer_slots,
    local_slots,
    item_bytes,
    layer_id,
    c4_layout=False,
):
    _contract(
        peer_ptrs, local_ptrs, peer_slots, local_slots, item_bytes, layer_id, c4_layout
    )
    if layer_id < 0:
        raise ValueError("Miss replay requires a selected layer")
    for tensor, dtype, ndim in (
        (miss_src, torch.int64, 2),
        (miss_dst, torch.int32, 2),
        (miss_count, torch.int32, 1),
        (num_real_reqs, torch.int32, 1),
    ):
        _metadata(tensor, peer_ptrs.device, dtype, ndim)
    if (
        miss_src.shape != miss_dst.shape
        or miss_count.shape != (miss_src.shape[0],)
        or num_real_reqs.shape != (1,)
    ):
        raise ValueError("Invalid fixed-size MPU miss plan")
    with torch.cuda.device(peer_ptrs.device):
        _module().replay(
            peer_ptrs,
            local_ptrs,
            miss_src,
            miss_dst,
            miss_count,
            num_real_reqs,
            peer_slots,
            local_slots,
            item_bytes,
            layer_id,
            c4_layout,
        )
        stream = torch.cuda.current_stream(peer_ptrs.device)
        for t in (peer_ptrs, local_ptrs, miss_src, miss_dst, miss_count, num_real_reqs):
            t.record_stream(stream)
