"""Compute-GPU kernels for a load/store-only peer KV backing store.

Only metadata pointer tables cross the CPU. Remote K/V tensors are never
passed to a dispatching tensor operation: every kernel launches on the device
owning ``peer_ptrs`` and dereferences peer UVA addresses from there.
"""

from __future__ import annotations

import functools

import torch

from sglang.jit_kernel.utils import load_jit


@functools.cache
def _module():
    return load_jit(
        "mpu_load_store_io",
        cuda_files=["mpu_io.cuh"],
        cuda_wrappers=[
            ("enable_peer_access", "mpu_enable_peer_access"),
            ("zero_peer_kv", "mpu_zero_peer_kv"),
            ("transfer_peer_kv", "mpu_transfer_peer_kv"),
        ],
    )


def enable_peer_access(compute_device: int, storage_devices: list[int]) -> None:
    """Enable only compute -> storage access; fail instead of host fallback."""
    if type(compute_device) is not int or compute_device < 0:
        raise ValueError("compute_device must be a nonnegative CUDA device index")
    if not 1 <= len(storage_devices) <= 4:
        raise ValueError("MPU requires between one and four storage devices")
    if any(type(device) is not int or device < 0 for device in storage_devices):
        raise ValueError("storage_devices must contain nonnegative CUDA device indices")
    if (
        len(set(storage_devices)) != len(storage_devices)
        or compute_device in storage_devices
    ):
        raise ValueError("compute and storage devices must be distinct")
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("MPU load/store requires an NVIDIA CUDA runtime")
    count = torch.cuda.device_count()
    if max(compute_device, *storage_devices) >= count:
        raise ValueError("MPU device index exceeds the visible CUDA device count")
    for device in storage_devices:
        if not torch.cuda.can_device_access_peer(compute_device, device):
            raise RuntimeError(
                f"CUDA peer access {compute_device} -> {device} is unavailable; "
                "CPU KV staging is not supported by the MPU backend"
            )
    # The C++ helper restores the caller's current device after each setup call.
    with torch.cuda.device(compute_device):
        module = _module()
        for device in storage_devices:
            module.enable_peer_access(compute_device, device)


def validate_peer_table(
    peer_ptrs: torch.Tensor, shard_rows: int, item_size_bytes: int
) -> None:
    if (
        not peer_ptrs.is_cuda
        or peer_ptrs.dtype != torch.int64
        or not peer_ptrs.is_contiguous()
    ):
        raise ValueError("peer_ptrs must be a contiguous CUDA int64 metadata tensor")
    if (
        peer_ptrs.ndim != 3
        or not 1 <= peer_ptrs.shape[0] <= 4
        or peer_ptrs.shape[1] != 2
    ):
        raise ValueError("peer_ptrs must have shape [1..4 shards, 2, layers]")
    if peer_ptrs.shape[2] <= 0 or type(shard_rows) is not int or shard_rows <= 0:
        raise ValueError("MPU layers and shard_rows must be positive")
    if type(item_size_bytes) is not int or item_size_bytes <= 0 or item_size_bytes % 8:
        raise ValueError("MPU K/V record bytes must be a positive multiple of 8")


def zero_peer_kv(
    peer_ptrs: torch.Tensor, shard_rows: int, item_size_bytes: int
) -> None:
    """Initialize peer payload with stores issued only by the compute GPU."""
    validate_peer_table(peer_ptrs, shard_rows, item_size_bytes)
    with torch.cuda.device(peer_ptrs.device):
        _module().zero_peer_kv(peer_ptrs, shard_rows, item_size_bytes)


def transfer_peer_kv(
    peer_ptrs: torch.Tensor,
    local_ptrs: torch.Tensor,
    peer_indices: torch.Tensor,
    local_indices: torch.Tensor,
    shard_rows: int,
    item_size_bytes: int,
    to_peer: bool,
    layer_id: int = -1,
    *,
    local_rows: int,
    peer_rows: int,
) -> None:
    """Store or load records, on the caller's current compute-device stream.

    ``peer_ptrs`` is int64 [shards, 2, layers], ``local_ptrs`` is int64
    [2, layers], and both index vectors are int64 [tokens] on that same GPU.
    ``layer_id=-1`` transfers all layers; otherwise only that local layer.
    The owner keeps all pointed-to allocations alive until this stream drains.
    Row bounds describe the real allocations, excluding last-shard padding.
    """
    validate_peer_table(peer_ptrs, shard_rows, item_size_bytes)
    layers = peer_ptrs.shape[2]
    if local_ptrs.shape != (2, layers):
        raise ValueError("local_ptrs must have shape [2, layers]")
    if peer_indices.ndim != 1 or local_indices.shape != peer_indices.shape:
        raise ValueError("peer_indices and local_indices must be equal-length vectors")
    for tensor in (local_ptrs, peer_indices, local_indices):
        if (
            tensor.device != peer_ptrs.device
            or tensor.dtype != torch.int64
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "MPU transfer metadata must be contiguous int64 on the compute GPU"
            )
    if (
        type(to_peer) is not bool
        or type(layer_id) is not int
        or not -1 <= layer_id < layers
    ):
        raise ValueError("invalid MPU transfer direction or layer_id")
    if (
        type(local_rows) is not int
        or type(peer_rows) is not int
        or local_rows <= 0
        or not 0 < peer_rows <= shard_rows * peer_ptrs.shape[0]
    ):
        raise ValueError(
            "MPU transfer row bounds must be positive and fit peer capacity"
        )
    if not peer_indices.numel():
        return
    with torch.cuda.device(peer_ptrs.device):
        _module().transfer_peer_kv(
            peer_ptrs,
            local_ptrs,
            peer_indices,
            local_indices,
            shard_rows,
            item_size_bytes,
            to_peer,
            layer_id,
            local_rows,
            peer_rows,
        )
