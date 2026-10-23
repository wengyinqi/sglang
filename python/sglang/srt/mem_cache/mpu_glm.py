"""GLM hybrid DSA main-KV backing on one peer MPU per compute rank.

KDA state and index K remain in their original compute-GPU pools. This class
has no HostKVCache parent and never allocates a CPU KV payload.
"""

from __future__ import annotations

import math

import torch

from sglang.kernels.ops.kvcache.mpu import replay_misses, transfer_records
from sglang.srt.mem_cache.mpu_peer import MPUSlotAllocator, PeerAllocation
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


def resolve_compute_ordinal(device, *, tp_rank=None):
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("GLM MPU requires a CUDA compute device")
    ordinal = resolved.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    if tp_rank is not None and ordinal != tp_rank:
        raise ValueError("GLM MPU requires compute CUDA ordinal to equal TP rank")
    return ordinal


def validate_metadata_indices(indices, compute_device):
    if (
        indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or (indices.device.type != "cpu" and indices.device != compute_device)
    ):
        raise ValueError(
            "GLM MPU indices must be integer vectors on CPU or the compute GPU"
        )


def glm_backing_options(extra, *, tp_size, compute_device):
    """Validate the explicitly scoped 4-compute / 4-storage topology."""
    mode = extra.get("backing_store", "cpu")
    if mode not in ("cpu", "mpu"):
        raise ValueError("GLM HiSparse backing_store must be cpu or mpu")
    if mode == "cpu":
        if any(key.startswith("mpu_") for key in extra):
            raise ValueError("MPU options require backing_store=mpu")
        return mode, [], None, 0
    ids = extra.get("mpu_device_ids", [4, 5, 6, 7])
    if (
        tp_size != 4
        or type(compute_device) is not int
        or compute_device not in range(4)
        or not isinstance(ids, list)
        or any(type(i) is not int for i in ids)
        or ids != [4, 5, 6, 7]
    ):
        raise ValueError("GLM MPU requires TP4 compute ordinals 0..3 paired with 4..7")
    ratio = extra.get("mpu_capacity_ratio", 2.0)
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("mpu_capacity_ratio must be finite and positive")
    reserve = extra.get("mpu_memory_reserve_bytes", 1 << 30)
    if type(reserve) is not int or reserve < 0:
        raise ValueError("mpu_memory_reserve_bytes must be a nonnegative integer")
    known = {"mpu_device_ids", "mpu_capacity_ratio", "mpu_memory_reserve_bytes"}
    if any(key.startswith("mpu_") and key not in known for key in extra):
        raise ValueError("Unknown GLM MPU option")
    return mode, ids, float(ratio), reserve


class HiSparseMLAMPUPool(HiSparseHostPoolMixin):
    def __init__(
        self,
        device_pool,
        *,
        storage_device_ids,
        capacity_ratio,
        memory_reserve_bytes=1 << 30,
    ):
        self.device = torch.device("cuda", resolve_compute_ordinal(device_pool.device))
        _, ids, ratio, reserve = glm_backing_options(
            dict(
                backing_store="mpu",
                mpu_device_ids=storage_device_ids,
                mpu_capacity_ratio=capacity_ratio,
                mpu_memory_reserve_bytes=memory_reserve_bytes,
            ),
            tp_size=4,
            compute_device=self.device.index,
        )
        self.page_size = device_pool.page_size
        self.layer_num = device_pool.layer_num
        self.dtype = device_pool.dtype
        self.kv_cache_dim = device_pool.kv_cache_dim
        if (
            self.layer_num != 11
            or self.kv_cache_dim != 512
            or self.dtype != torch.bfloat16
            or self.page_size != 64
            or device_pool.start_layer != 0
            or device_pool.index_kpool != 4
        ):
            raise ValueError(
                "GLM MPU supports the pinned 11-layer BF16 512-wide KPool4 layout"
            )
        self.size = (
            math.floor(device_pool.size * ratio / self.page_size) * self.page_size
        )
        if self.size <= 0:
            raise ValueError("GLM MPU capacity must hold at least one page")
        self.requested_capacity_ratio = ratio
        self.storage_device_ids = [ids[self.device.index]]
        self.configured_storage_device_ids = ids
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout = "layer_first"
        self._closed = False
        self._device_pool = device_pool
        self._slots = MPUSlotAllocator(self.size, self.page_size)
        self.device_kv_buffers = tuple(device_pool.kv_buffer)
        self.local_slots = self.device_kv_buffers[0].shape[0]
        for buffer in self.device_kv_buffers:
            if (
                buffer.device != self.device
                or buffer.dtype != self.dtype
                or buffer.shape != (self.local_slots, 1, self.kv_cache_dim)
                or not buffer.is_contiguous()
            ):
                raise ValueError(
                    "GLM MPU requires contiguous per-layer compute KV rows"
                )
        self._peer = PeerAllocation(
            self.device.index,
            self.storage_device_ids[0],
            (self.layer_num, self.size, 1, self.kv_cache_dim),
            self.dtype,
            reserve,
        )
        try:
            # Views and addresses only: no operation dispatches on these tensors.
            self.kv_buffer = self._peer.tensor
            self.peer_data_ptrs = torch.tensor(
                [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)],
                dtype=torch.int64,
                device=self.device,
            )
            self.local_data_ptrs = torch.tensor(
                [buffer.data_ptr() for buffer in self.device_kv_buffers],
                dtype=torch.int64,
                device=self.device,
            )
        except BaseException:
            self.destroy()
            raise

    def available_size(self):
        return self._slots.available_size()

    def alloc(self, need_size):
        if self._closed:
            raise RuntimeError("GLM MPU pool is closed")
        return self._slots.alloc(need_size)

    def free(self, indices):
        validate_metadata_indices(indices, self.device)
        # Request cleanup is infrequent; drain every compute stream before reuse.
        # This also covers staging/prefetch, whose raw peer pointers carry no
        # storage-allocator stream tracking. No storage GPU work is launched.
        torch.cuda.synchronize(self.device)
        return self._slots.free(indices)

    def _transfer(
        self, device_pool, peer_indices, local_indices, layer_id, to_peer, io_backend
    ):
        if (
            self._closed
            or device_pool is not self._device_pool
            or io_backend != "kernel"
        ):
            raise ValueError(
                "GLM MPU requires its original compute pool and kernel I/O"
            )
        validate_metadata_indices(peer_indices, self.device)
        validate_metadata_indices(local_indices, self.device)
        transfer_records(
            self.peer_data_ptrs,
            self.local_data_ptrs,
            peer_indices.to(device=self.device, dtype=torch.int64).contiguous(),
            local_indices.to(device=self.device, dtype=torch.int64).contiguous(),
            peer_slots=self.size,
            local_slots=self.local_slots,
            item_bytes=self.token_stride_size,
            layer_id=layer_id,
            to_peer=to_peer,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self._transfer(device_pool, host_indices, device_indices, -1, True, io_backend)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        self._transfer(
            device_pool, host_indices, device_indices, layer_id, False, io_backend
        )

    def replay_misses(self, miss_src, miss_dst, miss_count, num_real_reqs, layer_id):
        if self._closed:
            raise RuntimeError("GLM MPU pool is closed")
        replay_misses(
            self.peer_data_ptrs,
            self.local_data_ptrs,
            miss_src,
            miss_dst,
            miss_count,
            num_real_reqs,
            peer_slots=self.size,
            local_slots=self.local_slots,
            item_bytes=self.token_stride_size,
            layer_id=layer_id,
        )

    def destroy(self):
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self.kv_buffer = None
        self.peer_data_ptrs = None
        self.local_data_ptrs = None
        self._peer.close()
        self._closed = True
