"""Native CUDA peer allocations; CPU owns control metadata only.

Storage devices never launch KV initialization/copy/compute kernels. The CUDA
simulator contract is intentionally separate from a future physical MPU driver.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading

import torch


def validate_mpu_devices(compute_device: int, storage_device_ids: list[int]) -> None:
    ids = [compute_device, *storage_device_ids]
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("MPU device IDs must be nonnegative CUDA-visible ordinals")
    if not 1 <= len(storage_device_ids) <= 4:
        raise ValueError("MPU simulation requires 1 to 4 storage GPUs")
    if len(set(ids)) != len(ids):
        raise ValueError("Compute GPU and all MPU storage GPUs must be distinct")


def validate_mpu_runtime(compute_device: int, storage_device_ids: list[int]) -> None:
    """Reject unsupported allocators/topologies before allocating any payload."""
    validate_mpu_devices(compute_device, storage_device_ids)
    if (
        sys.platform != "linux"
        or torch.version.cuda is None
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("MPU peer storage requires Linux with NVIDIA CUDA")
    if max(compute_device, *storage_device_ids) >= torch.cuda.device_count():
        raise ValueError("An MPU device ID is outside CUDA_VISIBLE_DEVICES")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("MPU peer storage requires the native CUDA allocator")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        # torch.empty would otherwise initialize its payload on the storage
        # device. MPU payload initialization must run on the compute GPU.
        raise RuntimeError(
            "MPU storage requires deterministic fill_uninitialized_memory=False "
            "to avoid a storage-GPU initialization kernel"
        )
    # VMM expandable segments need allocation-scoped access registration that
    # cudaDeviceEnablePeerAccess alone does not provide. Fail closed for now.
    alloc_conf = ",".join(
        os.environ.get(key, "")
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    )
    if re.search(r"expandable_segments\s*:\s*true", alloc_conf, re.IGNORECASE):
        raise RuntimeError("MPU peer storage does not support expandable_segments")
    for storage in storage_device_ids:
        if not torch.cuda.can_device_access_peer(compute_device, storage):
            raise RuntimeError(
                f"Compute GPU {compute_device} cannot peer-access storage GPU {storage}"
            )


def validate_native_peer_allocations(shards: list[torch.Tensor]) -> None:
    """Check the actual allocations, including programmatic allocator changes.

    Native allocator snapshots are control metadata. Requiring a matching
    non-VMM segment also rejects a custom allocation hidden behind a native
    global allocator setting; no payload is read back from a storage GPU.
    """
    segments = torch.cuda.memory_snapshot()
    for shard in shards:
        begin = shard.data_ptr()
        end = begin + shard.numel() * shard.element_size()
        matches = [
            segment
            for segment in segments
            if segment["device"] == shard.device.index
            and segment["address"] <= begin
            and end <= segment["address"] + segment["total_size"]
        ]
        if len(matches) != 1 or matches[0].get("is_expandable") is not False:
            raise RuntimeError(
                "MPU peer allocation must belong to a verifiable native "
                "non-expandable CUDA segment"
            )


class PeerAllocation:
    """One native allocation, accessed and initialized from the compute GPU."""

    def __init__(
        self,
        compute_device: int,
        storage_device: int,
        shape,
        dtype,
        memory_reserve_bytes: int = 1 << 30,
    ):
        validate_mpu_runtime(compute_device, [storage_device])
        if (
            type(memory_reserve_bytes) is not int
            or memory_reserve_bytes < 0
            or not shape
            or any(type(x) is not int or x <= 0 for x in shape)
        ):
            raise ValueError("Invalid MPU shape or reserve byte count")
        byte_count = math.prod(shape) * dtype.itemsize
        if byte_count % 8:
            raise ValueError("MPU allocations must be a multiple of eight bytes")
        free, total = torch.cuda.mem_get_info(storage_device)
        if byte_count + memory_reserve_bytes > free:
            raise ValueError(
                f"MPU {storage_device}: {byte_count} payload + "
                f"{memory_reserve_bytes} reserve exceeds {free} free bytes"
            )
        from sglang.kernels.ops.kvcache import mpu

        self.compute_device = torch.device("cuda", compute_device)
        self.tensor = None
        self._closed = False
        mpu.enable_peer_access(compute_device, storage_device)
        try:
            self.tensor = torch.empty(
                shape, dtype=dtype, device=f"cuda:{storage_device}"
            )
            validate_native_peer_allocations([self.tensor])
            ptr = torch.tensor(
                [self.tensor.data_ptr()], dtype=torch.int64, device=self.compute_device
            )
            mpu.zero_peer(ptr, byte_count)
            torch.cuda.synchronize(self.compute_device)
        except BaseException:
            self.close()
            raise
        self.info = dict(
            compute_device=compute_device,
            storage_device=storage_device,
            allocated_bytes=byte_count,
            storage_total_bytes=total,
            memory_reserve_bytes=memory_reserve_bytes,
        )

    def close(self):
        if not self._closed:
            torch.cuda.synchronize(self.compute_device)
            self.tensor = None
            self._closed = True


class MPUSlotAllocator:
    """Whole-page CPU metadata allocator; no KV payload or storage operations."""

    def __init__(self, size: int, page_size: int = 1):
        if (
            type(size) is not int
            or type(page_size) is not int
            or size <= 0
            or page_size <= 0
            or size % page_size
        ):
            raise ValueError(
                "MPU size must contain a positive integral number of pages"
            )
        self.size = size
        self.page_size = page_size
        self.lock = threading.RLock()
        self.clear()

    def clear(self):
        with self.lock:
            self._free_pages = torch.arange(
                self.size // self.page_size, dtype=torch.int64, device="cpu"
            )
            self._allocated = torch.zeros(
                self.size // self.page_size, dtype=torch.bool, device="cpu"
            )

    def available_size(self):
        with self.lock:
            return self._free_pages.numel() * self.page_size

    def alloc(self, need_size):
        if type(need_size) is not int or need_size < 0:
            raise ValueError("MPU slot count must be a nonnegative integer")
        pages = (need_size + self.page_size - 1) // self.page_size
        with self.lock:
            if pages > self._free_pages.numel():
                return None
            selected = self._free_pages[:pages].clone()
            self._free_pages = self._free_pages[pages:]
            self._allocated[selected] = True
            return (
                selected[:, None] * self.page_size
                + torch.arange(self.page_size, dtype=torch.int64, device="cpu")
            ).flatten()

    def free(self, indices):
        if indices.ndim != 1 or indices.dtype not in (torch.int64, torch.int32):
            raise ValueError("MPU free requires an integer vector")
        slots = indices.to(device="cpu", dtype=torch.int64)
        if slots.numel() == 0:
            return 0
        with self.lock:
            if slots.min() < 0 or slots.max() >= self.size:
                raise ValueError("MPU free is outside allocation")
            pages, counts = torch.unique(slots // self.page_size, return_counts=True)
            if (
                slots.unique().numel() != slots.numel()
                or not torch.all(counts == self.page_size)
                or not self._allocated[pages].all()
            ):
                raise ValueError(
                    "MPU free must contain complete allocated pages exactly once"
                )
            self._allocated[pages] = False
            self._free_pages = torch.cat([self._free_pages, pages])
            return slots.numel()
