"""DeepSeek-V4 C4 backing in peer HBM, with the existing CPU C4 byte ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.mpu_peer import MPUSlotAllocator, PeerAllocation
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


@dataclass(frozen=True)
class C4Layout:
    pages: int
    layers: int
    page_size: int = 64
    page_bytes: int = 37440

    def __post_init__(self):
        if (
            type(self.pages) is not int
            or type(self.layers) is not int
            or self.pages <= 0
            or self.layers <= 0
            or self.page_size != 64
            or self.page_bytes != 37440
        ):
            raise ValueError("C4 MPU requires layers of 64-slot, 37440-byte pages")

    @property
    def shape(self):
        return (self.layers, self.pages, self.page_bytes)

    def offsets(self, slot):
        if type(slot) is not int or not 0 <= slot < self.pages * self.page_size:
            raise ValueError("C4 slot out of bounds")
        page, offset = divmod(slot, 64)
        return page * 37440 + offset * 576, page * 37440 + 36864 + offset * 8


class DeepSeekV4MPUPagedPool(HiSparseHostPoolMixin):
    def __init__(
        self,
        device_pool,
        storage_device,
        capacity_ratio=2.0,
        memory_reserve_bytes=1 << 30,
    ):
        self._closed = True
        if (
            type(capacity_ratio) not in (int, float)
            or not math.isfinite(capacity_ratio)
            or capacity_ratio <= 0
        ):
            raise ValueError("MPU capacity ratio must be finite and positive")
        self.device_pool = device_pool
        self.compute_device = torch.device(device_pool.device)
        if self.compute_device.type != "cuda":
            raise ValueError("C4 MPU requires a CUDA compute pool")
        if self.compute_device.index is None:
            self.compute_device = torch.device("cuda", torch.cuda.current_device())
        if device_pool.page_size != 64 or device_pool.bytes_per_page_padded != 37440:
            raise ValueError("MPU requires the original DeepSeek-V4 C4 page layout")
        self.page_size = self.slot_page_size = 64
        self.layer_num = device_pool.layer_num
        self.num_host_pages = math.ceil(device_pool.size * capacity_ratio / 64)
        self.size = self.num_host_pages * 64
        self.layout_descriptor = C4Layout(self.num_host_pages, self.layer_num)
        self.item_bytes = 37440
        self.layout = "layer_first"
        self.pin_memory = False
        self.requested_capacity_ratio = capacity_ratio
        self.storage_device_ids = [storage_device]
        self.local_slots = device_pool.kv_buffer[0].shape[0] * 64
        for t in device_pool.kv_buffer:
            if (
                t.device != self.compute_device
                or t.dtype != torch.uint8
                or tuple(t.shape) != (self.local_slots // 64, 37440)
                or not t.is_contiguous()
            ):
                raise ValueError(
                    "C4 compute pool must have contiguous uint8 paged buffers"
                )
        self._allocator = MPUSlotAllocator(self.size, 64)
        self._allocation = PeerAllocation(
            self.compute_device.index,
            storage_device,
            self.layout_descriptor.shape,
            torch.uint8,
            memory_reserve_bytes,
        )
        self.kv_buffer = list(self._allocation.tensor.unbind(0))
        self._local_refs = list(device_pool.kv_buffer)
        self.peer_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.int64,
            device=self.compute_device,
        )
        self.local_ptrs = torch.tensor(
            [x.data_ptr() for x in self._local_refs],
            dtype=torch.int64,
            device=self.compute_device,
        )
        self._closed = False

    def available_size(self):
        return self._allocator.available_size()

    def alloc(self, need_size):
        return self._allocator.alloc(need_size)

    def free(self, indices):
        return self._allocator.free(indices)

    def clear(self):
        torch.cuda.synchronize(self.compute_device)
        self._allocator.clear()

    def _transfer(
        self, device_pool, host_indices, device_indices, to_peer, layer_id, io_backend
    ):
        if (
            self._closed
            or device_pool is not self.device_pool
            or io_backend != "kernel"
        ):
            raise ValueError(
                "C4 MPU transfer requires its bound pool and compute kernel"
            )
        from sglang.kernels.ops.kvcache.mpu import transfer_records

        ids = []
        for t in (host_indices, device_indices):
            if t.ndim != 1 or t.dtype not in (torch.int32, torch.int64):
                raise ValueError("C4 transfer expects integer index vectors")
            if t.device.type != "cpu" and t.device != self.compute_device:
                raise ValueError("C4 index metadata must reside on compute or CPU")
            ids.append(t.to(device=self.compute_device, dtype=torch.int64).contiguous())
        transfer_records(
            self.peer_ptrs,
            self.local_ptrs,
            *ids,
            peer_slots=self.size,
            local_slots=self.local_slots,
            item_bytes=584,
            layer_id=layer_id,
            to_peer=to_peer,
            c4_layout=True,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        self._transfer(device_pool, host_indices, device_indices, True, -1, io_backend)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend="kernel"
    ):
        self._transfer(
            device_pool, host_indices, device_indices, False, layer_id, io_backend
        )

    def replay_misses(self, miss_src, miss_dst, miss_count, num_real_reqs, layer_id):
        from sglang.kernels.ops.kvcache.mpu import replay_misses

        replay_misses(
            self.peer_ptrs,
            self.local_ptrs,
            miss_src,
            miss_dst,
            miss_count,
            num_real_reqs,
            peer_slots=self.size,
            local_slots=self.local_slots,
            item_bytes=584,
            layer_id=layer_id,
            c4_layout=True,
        )

    def destroy(self):
        if self._closed:
            return
        torch.cuda.synchronize(self.compute_device)
        self.peer_ptrs = self.local_ptrs = None
        self.kv_buffer = []
        self._local_refs = []
        self._allocation.close()
        self._closed = True
