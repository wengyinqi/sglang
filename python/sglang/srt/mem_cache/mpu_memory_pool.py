"""MPU simulator storage for HiSparse's ordinary MHA/GQA KV layout.

Only the compute GPU launches KV kernels. Storage GPUs own allocations in the
same ``[2, layers, slots, kv_heads, head_dim]`` layout as the layer-first host
pool. CPU tensors in this module contain allocation or pointer metadata only.
There is deliberately no inheritance from HostKVCache and no host-KV fallback.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MPUKVLayout:
    """Byte-address ABI of one contiguous storage shard."""

    layers: int
    shard_rows: int
    heads: int
    head_dim: int
    element_size: int = 2

    def __post_init__(self):
        for name in ("layers", "shard_rows", "heads", "head_dim", "element_size"):
            _positive_int(getattr(self, name), name)

    @property
    def shape(self) -> tuple[int, ...]:
        return (2, self.layers, self.shard_rows, self.heads, self.head_dim)

    @property
    def token_stride_bytes(self) -> int:
        return self.heads * self.head_dim * self.element_size

    @property
    def bytes_per_slot(self) -> int:
        return 2 * self.layers * self.token_stride_bytes

    @property
    def shard_bytes(self) -> int:
        return self.shard_rows * self.bytes_per_slot

    def byte_offset(self, side: int, layer: int, row: int) -> int:
        for value, bound, name in (
            (side, 2, "side"),
            (layer, self.layers, "layer"),
            (row, self.shard_rows, "row"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < bound
            ):
                raise ValueError(f"{name} is outside this shard")
        return (
            (side * self.layers + layer) * self.shard_rows + row
        ) * self.token_stride_bytes


@dataclass(frozen=True)
class MPUCapacity:
    """Usable logical slots are separate from uniform-shard padding."""

    size: int
    shard_rows: int
    num_shards: int
    device_slots: int
    requested_ratio: float

    @property
    def allocated_slots(self) -> int:
        return self.shard_rows * self.num_shards

    @property
    def actual_ratio(self) -> float:
        return self.size / self.device_slots

    def split_slot(self, slot: int) -> tuple[int, int]:
        if (
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or not 0 <= slot < self.size
        ):
            raise ValueError("MPU slot is outside the usable capacity")
        return divmod(slot, self.shard_rows)


def plan_mpu_capacity(device_slots: int, ratio: float, num_shards: int) -> MPUCapacity:
    _positive_int(device_slots, "device_slots")
    if (
        isinstance(num_shards, bool)
        or not isinstance(num_shards, int)
        or not 1 <= num_shards <= 4
    ):
        raise ValueError("MPU simulation requires 1 to 4 storage GPUs")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio <= 0
    ):
        raise ValueError("MPU capacity ratio must be finite and positive")
    size = math.ceil(device_slots * ratio)
    return MPUCapacity(
        size=size,
        shard_rows=(size + num_shards - 1) // num_shards,
        num_shards=num_shards,
        device_slots=device_slots,
        requested_ratio=ratio,
    )


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


class MPUSlotAllocator:
    """Thread-safe CPU control metadata; contains no KV payload."""

    def __init__(self, size: int):
        _positive_int(size, "size")
        self.size = size
        self.lock = threading.RLock()
        self.clear()

    def clear(self) -> None:
        with self.lock:
            self._free_slots = torch.arange(self.size, dtype=torch.int64, device="cpu")
            self._allocated = torch.zeros(self.size, dtype=torch.bool, device="cpu")

    def available_size(self) -> int:
        with self.lock:
            return self._free_slots.numel()

    def alloc(self, need_size: int) -> torch.Tensor | None:
        if (
            isinstance(need_size, bool)
            or not isinstance(need_size, int)
            or need_size < 0
        ):
            raise ValueError("Requested MPU slot count must be a nonnegative integer")
        with self.lock:
            if need_size > self._free_slots.numel():
                return None
            selected = self._free_slots[:need_size].clone()
            self._free_slots = self._free_slots[need_size:]
            self._allocated[selected] = True
            return selected

    def free(self, indices: torch.Tensor) -> int:
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("MPU free expects a one-dimensional integer slot tensor")
        # This is allocation metadata, never a KV transfer through the CPU.
        slots = indices.to(device="cpu", dtype=torch.int64)
        with self.lock:
            if slots.numel() == 0:
                return 0
            if slots.min().item() < 0 or slots.max().item() >= self.size:
                raise ValueError("Cannot free an out-of-range MPU slot")
            if (
                slots.unique().numel() != slots.numel()
                or not self._allocated[slots].all().item()
            ):
                raise ValueError("Cannot free duplicate or unallocated MPU slots")
            self._allocated[slots] = False
            self._free_slots = torch.cat((self._free_slots, slots))
            return slots.numel()


class MPUMHATokenToKVPool:
    """Host-pool-compatible backing store with compute-side peer loads/stores.

    ``storage_device_ids`` uses ordinals after CUDA_VISIBLE_DEVICES filtering.
    The ratio applies to total usable slots across all storage GPUs. Callers
    must finish KV accesses before freeing slots, just as with the host pool.
    ``close()`` drains compute work before releasing raw-pointer allocations.
    """

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        storage_device_ids: list[int],
        host_to_device_ratio: float = 2.0,
        *,
        page_size: int = 1,
        layout: str = "layer_first",
        memory_reserve_bytes: int = 1 << 30,
    ):
        self._closed = True
        self.shards: list[torch.Tensor] = []
        self._local_refs: list[torch.Tensor] = []
        self.peer_ptrs = None
        self.local_ptrs = None
        self.device_pool = device_pool
        self.compute_device = torch.device(device_pool.device)
        if self.compute_device.type != "cuda":
            raise ValueError("MPU compute pool must be on a CUDA GPU")
        if self.compute_device.index is None:
            self.compute_device = torch.device("cuda", torch.cuda.current_device())
        self.storage_device_ids = list(storage_device_ids)
        validate_mpu_runtime(self.compute_device.index, self.storage_device_ids)
        if page_size != 1 or layout != "layer_first":
            raise ValueError("MPU HiSparse requires page_size=1 and layer_first layout")
        if (
            isinstance(memory_reserve_bytes, bool)
            or not isinstance(memory_reserve_bytes, int)
            or memory_reserve_bytes < 0
        ):
            raise ValueError("MPU memory reserve must be a nonnegative byte count")
        if device_pool.store_dtype != torch.bfloat16:
            raise ValueError("MPU Quest currently requires BF16 MHA/GQA KV storage")
        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.capacity = plan_mpu_capacity(
            device_pool.size, host_to_device_ratio, len(self.storage_device_ids)
        )
        self.size = self.capacity.size
        self.shard_rows = self.capacity.shard_rows
        self.layout_descriptor = MPUKVLayout(
            self.layer_num,
            self.shard_rows,
            self.head_num,
            self.head_dim,
            self.dtype.itemsize,
        )
        self.token_stride_size = self.layout_descriptor.token_stride_bytes
        if self.token_stride_size % 8:
            raise ValueError("MPU K/V record bytes must be a multiple of 8")
        self.size_per_token = self.layout_descriptor.bytes_per_slot
        self.layout = layout
        self.page_size = page_size
        self.page_num = self.size
        self.device = self.compute_device
        self.pin_memory = False
        self._validate_local_buffers()
        self._local_refs = [*device_pool.k_buffer, *device_pool.v_buffer]
        for storage in self.storage_device_ids:
            free_bytes, _ = torch.cuda.mem_get_info(storage)
            if self.layout_descriptor.shard_bytes + memory_reserve_bytes > free_bytes:
                raise ValueError(
                    f"MPU storage GPU {storage} requires "
                    f"{self.layout_descriptor.shard_bytes} KV bytes plus "
                    f"{memory_reserve_bytes} reserve bytes, but has {free_bytes} free bytes"
                )

        from sglang.jit_kernel import mpu_io

        self._io = mpu_io
        self._allocator = MPUSlotAllocator(self.size)
        # Enable only compute -> storage access. There is no storage-side
        # reader, model, worker, or KV kernel.
        self._io.enable_peer_access(self.compute_device.index, self.storage_device_ids)
        self._closed = False
        try:
            for storage in self.storage_device_ids:
                self.shards.append(
                    torch.empty(
                        self.layout_descriptor.shape,
                        dtype=self.dtype,
                        device=torch.device("cuda", storage),
                    )
                )
            validate_native_peer_allocations(self.shards)
            ptrs = [
                [
                    [
                        shard.data_ptr()
                        + self.layout_descriptor.byte_offset(side, layer, 0)
                        for layer in range(self.layer_num)
                    ]
                    for side in range(2)
                ]
                for shard in self.shards
            ]
            self.peer_ptrs = torch.tensor(
                ptrs, dtype=torch.int64, device=self.compute_device
            )
            self.local_ptrs = torch.tensor(
                [
                    [buf.data_ptr() for buf in buffers]
                    for buffers in (device_pool.k_buffer, device_pool.v_buffer)
                ],
                dtype=torch.int64,
                device=self.compute_device,
            )
            with torch.cuda.device(self.compute_device):
                self._io.zero_peer_kv(
                    self.peer_ptrs, self.shard_rows, self.token_stride_size
                )
            # All future staging/decode streams must observe initialization.
            torch.cuda.synchronize(self.compute_device)
        except BaseException:
            self.close()
            raise
        logger.info("Allocated MPU KV storage: %s", self.capacity_info)

    def _validate_local_buffers(self) -> None:
        rows = None
        for buffers in (self.device_pool.k_buffer, self.device_pool.v_buffer):
            if len(buffers) != self.layer_num:
                raise ValueError("MPU requires one K and V buffer for each layer")
            for buffer in buffers:
                if (
                    buffer.device != self.compute_device
                    or buffer.dtype != self.dtype
                    or buffer.ndim != 3
                    or tuple(buffer.shape[1:]) != (self.head_num, self.head_dim)
                    or not buffer.is_contiguous()
                ):
                    raise ValueError(
                        "MPU requires contiguous BF16 MHA/GQA compute buffers"
                    )
                if rows is not None and rows != buffer.shape[0]:
                    raise ValueError(
                        "All compute KV buffers must have the same row count"
                    )
                rows = buffer.shape[0]
        self.local_rows = rows

    @property
    def capacity_info(self) -> dict:
        return {
            **asdict(self.capacity),
            "actual_ratio": self.capacity.actual_ratio,
            "allocated_slots": self.capacity.allocated_slots,
            "padding_slots": self.capacity.allocated_slots - self.size,
            "bytes_per_slot": self.size_per_token,
            "usable_bytes": self.size * self.size_per_token,
            "allocated_bytes": len(self.storage_device_ids)
            * self.layout_descriptor.shard_bytes,
            "bytes_per_shard": self.layout_descriptor.shard_bytes,
            "storage_device_ids": self.storage_device_ids,
            "layout": self.layout,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MPU KV pool is closed")

    def get_size_per_token(self) -> int:
        return self.size_per_token

    def available_size(self) -> int:
        self._require_open()
        return self._allocator.available_size()

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self._require_open()
        return self._allocator.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        self._require_open()
        return self._allocator.free(indices)

    def clear(self) -> None:
        self._require_open()
        torch.cuda.synchronize(self.compute_device)
        self._allocator.clear()

    def _transfer(
        self,
        device_pool,
        host_indices,
        device_indices,
        *,
        to_peer,
        layer_id,
        io_backend,
    ):
        self._require_open()
        if device_pool is not self.device_pool:
            raise ValueError(
                "MPU transfers must use the compute pool bound at construction"
            )
        if io_backend != "kernel":
            raise ValueError("MPU supports only compute-GPU peer load/store kernels")
        if layer_id != -1 and not 0 <= layer_id < self.layer_num:
            raise ValueError("MPU transfer layer_id is outside the pool")
        indices = []
        for tensor in (host_indices, device_indices):
            if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "MPU transfers require one-dimensional integer indices"
                )
            if tensor.device.type != "cpu" and tensor.device != self.compute_device:
                raise ValueError("MPU index metadata must be on CPU or the compute GPU")
            indices.append(
                tensor.to(device=self.compute_device, dtype=torch.int64).contiguous()
            )
        if indices[0].numel() != indices[1].numel():
            raise ValueError("MPU source and destination index counts must match")
        with torch.cuda.device(self.compute_device):
            self._io.transfer_peer_kv(
                self.peer_ptrs,
                self.local_ptrs,
                indices[0],
                indices[1],
                self.shard_rows,
                self.token_stride_size,
                to_peer=to_peer,
                layer_id=layer_id,
                local_rows=self.local_rows,
                peer_rows=self.size,
            )
            # The raw-pointer kernel does not itself register tensor lifetime
            # with PyTorch's allocator when launched on a staging stream.
            stream = torch.cuda.current_stream(self.compute_device)
            for tensor in indices:
                tensor.record_stream(stream)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ) -> None:
        self._transfer(
            device_pool,
            host_indices,
            device_indices,
            to_peer=True,
            layer_id=-1,
            io_backend=io_backend,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend="kernel"
    ) -> None:
        self._transfer(
            device_pool,
            host_indices,
            device_indices,
            to_peer=False,
            layer_id=layer_id,
            io_backend=io_backend,
        )

    def close(self) -> None:
        if self._closed:
            return
        # Includes staging, decode backup, and normal compute streams, even
        # when the peer tensors were allocated under a storage-device context.
        torch.cuda.synchronize(self.compute_device)
        self.peer_ptrs = None
        self.local_ptrs = None
        self.shards.clear()
        self._local_refs = []
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                # Explicit close reports CUDA failures. Interpreter shutdown
                # may have already dismantled torch.cuda and cannot recover.
                pass
