"""Public CUDA device-graph memcpy support for the Qwen/Quest adaptation.

No SM kernel copies the payload. GPU launcher nodes initiate immutable memcpy
graphs. Checkpoint kernels implement a bounded approximation of priority gating,
not the paper's copy-engine command-stream semaphore implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from sglang.jit_kernel.utils import cache_once


@cache_once
def native_module(device_index: int):
    if sys.platform != "linux" or not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("Quest device-graph DMA requires Linux NVIDIA CUDA")
    nvcc = os.environ.get("SGLANG_XM_NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("Set SGLANG_XM_NVCC to a CUDA >= 12 compiler")
    source = Path(__file__).parent / "csrc" / "xm_bounds_prefetch.cu"
    capability = torch.cuda.get_device_capability(device_index)
    if capability < (7, 0):
        raise RuntimeError(
            "Quest DMA checkpoint kernels require compute capability >= 7.0"
        )
    version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    identity = hashlib.sha256(
        source.read_bytes() + version.encode() + str(capability).encode()
    ).hexdigest()[:20]
    cache = Path(
        os.environ.get("SGLANG_XM_NATIVE_CACHE", "~/.cache/sglang/xm-native")
    ).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"quest_bounds_{identity}.so"
    if not library.is_file():
        temporary = cache / f"quest_bounds_{identity}.{os.getpid()}.tmp.so"
        architecture = f"{capability[0]}{capability[1]}"
        command = [
            nvcc,
            str(source),
            "-o",
            str(temporary),
            "-shared",
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-rdc=true",
            "--cudart=shared",
            "-Xcompiler",
            "-fPIC",
            "-lcudadevrt",
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                "Quest GPU-init DMA compilation failed:\n"
                + result.stdout
                + result.stderr
            )
        temporary.replace(library)
    module = ctypes.CDLL(str(library))
    u64 = ctypes.c_uint64
    integer = ctypes.c_int
    module.xm_bounds_last_error.argtypes = []
    module.xm_bounds_last_error.restype = ctypes.c_char_p
    declarations = {
        "xm_bounds_mapped_pointer": [u64, integer, ctypes.POINTER(u64)],
        "xm_bounds_create_copy": [u64] * 9 + [integer] * 3 + [u64, ctypes.POINTER(u64)],
        "xm_bounds_begin": [u64, u64, integer, u64],
        "xm_bounds_launch": [u64, u64, integer],
        "xm_bounds_wait": [u64, u64],
        "xm_bounds_protect": [u64, integer, integer, u64],
        "xm_bounds_destroy": [u64],
    }
    for name, arguments in declarations.items():
        function = getattr(module, name)
        function.argtypes = arguments
        function.restype = integer
    module._xm_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    module._xm_library_path = str(library)
    return module


def checked(module, name, *arguments):
    if getattr(module, name)(*arguments):
        message = module.xm_bounds_last_error().decode("utf-8", errors="replace")
        raise RuntimeError(f"{name}: {message}")


def mapped_host_pointer(backing: torch.Tensor, device_index: int) -> int:
    if (
        backing.device.type != "cpu"
        or not backing.is_pinned()
        or not backing.is_contiguous()
    ):
        raise ValueError("Quest backing must be a contiguous pinned CPU tensor")
    module = native_module(device_index)
    pointer = ctypes.c_uint64()
    checked(
        module,
        "xm_bounds_mapped_pointer",
        backing.data_ptr(),
        device_index,
        ctypes.byref(pointer),
    )
    return pointer.value


class DeviceGraphCopy:
    """One immutable layer source/destination; tensor owners outlive the graph."""

    def __init__(self, tier, layer: int, *, chunk_bytes: int = 1024 * 1024):
        self.tier = tier
        self.layer = layer
        self.module = native_module(tier.device_index)
        handle = ctypes.c_uint64()
        layer_bytes = tier.stage_bounds[0].numel() * tier.stage_bounds.element_size()
        source = tier.backing_bounds.data_ptr() + layer * layer_bytes
        destination = tier.stage_bounds.data_ptr() + (layer % 2) * layer_bytes
        checked(
            self.module,
            "xm_bounds_create_copy",
            source,
            destination,
            layer_bytes,
            chunk_bytes,
            tier.step_epoch.data_ptr(),
            tier.ready_keys.data_ptr(),
            tier.issued_keys.data_ptr(),
            tier.prefetch_gate.data_ptr(),
            tier.counters.data_ptr(),
            tier.layers,
            layer,
            tier.device_index,
            torch.cuda.current_stream(tier.device).cuda_stream,
            ctypes.byref(handle),
        )
        self.handle = handle.value

    def launch(self, stream: torch.cuda.Stream, *, captured: bool):
        checked(
            self.module,
            "xm_bounds_launch",
            self.handle,
            stream.cuda_stream,
            int(captured),
        )

    def wait(self, stream: torch.cuda.Stream):
        checked(self.module, "xm_bounds_wait", self.handle, stream.cuda_stream)

    def close(self):
        if self.handle:
            checked(self.module, "xm_bounds_destroy", self.handle)
            self.handle = 0
