"""Shared SM90 TMA workspace and retained scalar arithmetic diagnostic.

This module preserves Quest selection. Resolve emits a plan without copying KV;
the CUDA kernel reads hot or host rows by TMA into shared memory, computes
attention, and refills misses. This scalar class retains the arithmetic candidate
whose real-model gate failed. The backend selects the BF16 MMA subclass.
This is an experimental adaptation rather than the XM-Sparse author's code.
"""

from __future__ import annotations

import functools
import math
from typing import Sequence

import torch

from sglang.jit_kernel.hisparse import (
    _xm_build_directory,
    plan_cache_to_device_buffer_mha,
    prepare_plan_cache_to_device_buffer_mha,
)
from sglang.jit_kernel.utils import load_jit, make_cpp_args

NUM_SPLITS = 16
TILE_ROWS = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
PARTIAL_WIDTH = HEAD_DIM + 2


def flashinfer_compatible_kv_chunk_size(
    actual_lens: Sequence[int], num_sms: int
) -> int:
    """CPU FA2 decode partition for Q32/KV4/D128, NHD page_size=1.

    Matches PrefillBinarySearchKVChunkSize/PrefillSplitQOKVIndptr in the
    frozen FlashInfer runtime scheduler.cuh (SHA256
    3fdc85c6b42d9c27fd688c614fef162bb523f3a95f095ba40dc53e3c52712c3b).
    Decode has one packed eight-head query per request, hence one CTA Q tile.
    The device SM count is queried once; H800's 114 SMs are not hard-coded.
    """
    if type(num_sms) is not int or num_sms < 2:
        raise ValueError("FA2 partition requires a valid device SM count")
    lengths = tuple(actual_lens)
    if not lengths or any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("FA2 partition lengths must be nonnegative integers")
    max_batch_size_if_split = (2 * num_sms) // NUM_KV_HEADS
    low, high = 128, max(1, *lengths)
    while low < high:
        middle = (low + high) // 2
        split_batch = sum((max(length, 1) + middle - 1) // middle for length in lengths)
        if split_batch > max_batch_size_if_split:
            low = middle + 1
        else:
            high = middle
    return low


@functools.cache
def _jit_fused_retrieval_attention(top_k: int):
    args = make_cpp_args(top_k, NUM_SPLITS, TILE_ROWS)
    return load_jit(
        "fused_hisparse_tma_bf16_gqa",
        *args,
        cuda_files=["fused_hisparse_tma.cuh"],
        cuda_wrappers=[
            ("mapped_source_pointer", "xm_hisparse_tma::mapped_source_pointer"),
            (
                "fused_retrieval_attention",
                f"xm_hisparse_tma::fused_retrieval_attention<{args}>",
            ),
        ],
        extra_cuda_cflags=["-lineinfo"],
        build_directory=_xm_build_directory(
            "fused_hisparse_tma_bf16_gqa", "fused_hisparse_tma.cuh", *args
        ),
    )


def validate_fused_tma_configuration(
    *,
    device: torch.device,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_data_type: torch.dtype,
    q_data_type: torch.dtype,
    backing_store: str,
) -> None:
    if backing_store != "cpu":
        raise ValueError("The experimental 1+0 TMA kernel requires CPU DDR backing")
    if (num_qo_heads, num_kv_heads, head_dim) != (32, 4, 128):
        raise ValueError("The experimental TMA kernel requires GQA Q32/KV4/D128")
    if kv_data_type != torch.bfloat16 or q_data_type != torch.bfloat16:
        raise ValueError("The experimental TMA kernel requires BF16 Q/K/V")
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("TMA fusion requires the NVIDIA CUDA runtime")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("This experimental kernel is validated for SM90 only")


class FusedHiSparseTMAWorkspace:
    """Scalar diagnostic with shared mapped owners and Graph-stable buffers."""

    def _load_module(self, top_k: int):
        return _jit_fused_retrieval_attention(top_k)

    def __init__(
        self,
        *,
        device: torch.device,
        max_bs: int,
        top_k: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        host_k_layers: Sequence[torch.Tensor],
        host_v_layers: Sequence[torch.Tensor],
        collect_metrics: bool = False,
    ) -> None:
        if max_bs <= 0 or not 0 < top_k <= hot_buffer_size <= 32767:
            raise ValueError("Invalid TMA batch/selection/hot-buffer capacity")
        if top_k > NUM_SPLITS * 128:
            raise ValueError("The experimental TMA partition supports top_k <= 2048")
        if item_size_bytes != NUM_KV_HEADS * HEAD_DIM * 2:
            raise ValueError("TMA BF16 GQA requires a 1024-byte K or V row stride")
        if len(host_k_layers) != len(host_v_layers) or not host_k_layers:
            raise ValueError("TMA requires complete K/V layer owner lists")
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.max_bs = max_bs
        self.top_k = top_k
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.num_sms = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.host_k_layers = tuple(host_k_layers)
        self.host_v_layers = tuple(host_v_layers)
        self._module = self._load_module(top_k)
        self._sources = []
        for k, v in zip(self.host_k_layers, self.host_v_layers):
            if (
                k.device.type != "cpu"
                or v.device.type != "cpu"
                or k.dtype != torch.bfloat16
                or v.dtype != torch.bfloat16
                or k.ndim != 3
                or k.shape != v.shape
                or tuple(k.shape[1:]) != (NUM_KV_HEADS, HEAD_DIM)
                or not k.is_contiguous()
                or not v.is_contiguous()
            ):
                raise ValueError("TMA backing must be layer-first CPU BF16[rows,4,128]")
            # This CUDA pointer-attribute query is performed once before Graph
            # capture, not in a model forward. It validates actual registration
            # and CUDA mapping even for manually cudaHostRegister'd tensors.
            k_pointer = self._module.mapped_source_pointer(k)
            v_pointer = self._module.mapped_source_pointer(v)
            self._sources.append(
                (k_pointer, v_pointer, k.shape[0], k.data_ptr(), v.data_ptr())
            )
        prepare_plan_cache_to_device_buffer_mha(
            item_size_bytes,
            top_k,
            hot_buffer_size,
            collect_metrics=collect_metrics,
        )
        self.miss_host_locs = torch.full(
            (max_bs, top_k), -1, dtype=torch.int64, device=self.device
        )
        self.device_locs = torch.full(
            (max_bs, top_k), -1, dtype=torch.int32, device=self.device
        )
        self.kv_chunk_size = torch.full(
            (1,), 128, dtype=torch.int32, device=self.device
        )
        self.partial = torch.empty(
            (max_bs, NUM_Q_HEADS, NUM_SPLITS, PARTIAL_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        self.output = torch.empty(
            (max_bs, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=self.device
        )

    def memory_usage(self) -> dict[str, int]:
        return {
            "plan_host_source_bytes": self.miss_host_locs.numel() * 8,
            "plan_device_destination_bytes": self.device_locs.numel() * 4,
            "plan_kv_chunk_size_bytes": self.kv_chunk_size.numel() * 4,
            "split_partial_workspace_bytes": self.partial.numel() * 4,
            "attention_output_bytes": self.output.numel() * 2,
        }

    def plan_partition(self, actual_lens_cpu: torch.Tensor) -> int:
        """Metadata-only update outside capture; no GPU-to-CPU readback."""
        if (
            actual_lens_cpu.device.type != "cpu"
            or actual_lens_cpu.ndim != 1
            or actual_lens_cpu.dtype not in (torch.int32, torch.int64)
            or not 0 < actual_lens_cpu.numel() <= self.max_bs
        ):
            raise ValueError("TMA partition requires a CPU integer length vector")
        lengths = actual_lens_cpu.tolist()
        if any(length > self.top_k for length in lengths):
            raise ValueError("TMA partition length exceeds selection capacity")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TMA partition metadata must be updated outside Graph capture"
            )
        chunk_size = flashinfer_compatible_kv_chunk_size(lengths, self.num_sms)
        self.kv_chunk_size.fill_(chunk_size)
        return chunk_size

    def resolve(
        self, coordinator, *, top_k_tokens, req_pool_indices, seq_lens, layer_id
    ):
        bs = top_k_tokens.shape[0]
        if not 0 < bs <= self.max_bs:
            raise ValueError("TMA resolve batch exceeds preallocated capacity")
        plan_cache_to_device_buffer_mha(
            top_k_tokens=top_k_tokens,
            device_buffer_tokens=coordinator.req_device_buffer_tokens[layer_id],
            host_cache_locs=coordinator.req_to_host_pool,
            device_buffer_locs=coordinator.req_device_buffer_token_locs[layer_id],
            device_k_buffer=coordinator.mem_pool_device.k_buffer[layer_id],
            device_v_buffer=coordinator.mem_pool_device.v_buffer[layer_id],
            top_k_device_locs=self.device_locs[:bs],
            miss_host_locs=self.miss_host_locs[:bs],
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=coordinator.lru_slots[layer_id],
            num_real_reqs=coordinator.num_real_reqs,
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.hot_buffer_size,
            cache_metrics=(
                coordinator.cache_metrics[layer_id]
                if coordinator.cache_metrics is not None
                else None
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        *,
        hot_k: torch.Tensor,
        hot_v: torch.Tensor,
        layer_id: int,
        actual_lens: torch.Tensor,
        num_real_reqs: torch.Tensor,
        scaling: float,
        logits_soft_cap: float = 0.0,
    ) -> torch.Tensor:
        bs = q.shape[0]
        if (
            q.dtype != torch.bfloat16
            or q.device != self.device
            or q.shape != (bs, NUM_Q_HEADS, HEAD_DIM)
            or not q.is_contiguous()
            or not 0 < bs <= self.max_bs
        ):
            raise ValueError("TMA query must be contiguous CUDA BF16[B,32,128]")
        if (
            not math.isfinite(scaling)
            or scaling <= 0
            or not math.isfinite(logits_soft_cap)
            or logits_soft_cap < 0
        ):
            raise ValueError("Invalid attention scale or logits soft cap")
        for tensor in (hot_k, hot_v):
            if (
                tensor.dtype != torch.bfloat16
                or tensor.device != self.device
                or tensor.ndim != 3
                or tuple(tensor.shape[1:]) != (NUM_KV_HEADS, HEAD_DIM)
                or not tensor.is_contiguous()
            ):
                raise ValueError("TMA hot buffers must be CUDA BF16[rows,4,128]")
        if hot_k.shape != hot_v.shape:
            raise ValueError("TMA K/V hot shapes differ")
        if (
            actual_lens.dtype != torch.int32
            or actual_lens.device != self.device
            or actual_lens.ndim != 1
            or actual_lens.numel() < bs
            or not actual_lens.is_contiguous()
            or num_real_reqs.dtype != torch.int32
            or num_real_reqs.device != self.device
            or num_real_reqs.shape != (1,)
            or not num_real_reqs.is_contiguous()
        ):
            raise ValueError(
                "TMA lengths/real-batch metadata require CUDA int32 vectors"
            )
        k_pointer, v_pointer, host_rows, original_k, original_v = self._sources[
            layer_id
        ]
        if (
            self.host_k_layers[layer_id].data_ptr() != original_k
            or self.host_v_layers[layer_id].data_ptr() != original_v
        ):
            raise RuntimeError("Mapped host allocation changed after TMA preparation")
        self._module.fused_retrieval_attention(
            q,
            hot_k,
            hot_v,
            k_pointer,
            v_pointer,
            host_rows,
            self.miss_host_locs[:bs],
            self.device_locs[:bs],
            actual_lens[:bs],
            num_real_reqs,
            self.kv_chunk_size,
            self.partial[:bs],
            self.output[:bs],
            scaling,
            logits_soft_cap,
        )
        return self.output[:bs]
