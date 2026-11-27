"""Opt-in SM90 BF16 MMA retrieval-attention over mapped CPU DDR KV."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from sglang.jit_kernel.fused_hisparse_tma import (
    NUM_SPLITS,
    TILE_ROWS,
    FusedHiSparseTMAWorkspace,
)
from sglang.jit_kernel.hisparse import _xm_build_directory
from sglang.jit_kernel.utils import load_jit, make_cpp_args


@functools.cache
def _jit_mma_attention(top_k: int):
    args = make_cpp_args(top_k, NUM_SPLITS, TILE_ROWS)
    shared_header = Path(__file__).parent / "csrc/fused_hisparse_tma.cuh"
    dependency_hash = hashlib.sha256(shared_header.read_bytes()).hexdigest()
    return load_jit(
        "fused_hisparse_tma_mma_bf16_gqa",
        *args,
        cuda_files=["fused_hisparse_tma_mma.cuh"],
        cuda_wrappers=[
            ("mapped_source_pointer", "xm_hisparse_tma::mapped_source_pointer"),
            (
                "fused_retrieval_attention",
                f"xm_hisparse_tma_mma::fused_retrieval_attention<{args}>",
            ),
        ],
        extra_cuda_cflags=["-lineinfo"],
        build_directory=_xm_build_directory(
            "fused_hisparse_tma_mma_bf16_gqa",
            "fused_hisparse_tma_mma.cuh",
            *args,
            dependency_hash,
        ),
    )


class FusedHiSparseTMAMMAWorkspace(FusedHiSparseTMAWorkspace):
    """Reuse validated owners/planning; forward executes only the MMA kernel.

    The parent prepares the mapped-pointer owners and plan ABI using this
    module from construction; this class never runs a scalar or
    FlashInfer fallback and never copies the selected KV before attention.
    """

    def _load_module(self, top_k: int):
        return _jit_mma_attention(top_k)
