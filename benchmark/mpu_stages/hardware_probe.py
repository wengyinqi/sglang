"""Isolated, evidence-scoped environment probes; never starts a model server.

The parent imports only the Python standard library. ``compile --compile-only``
compiles the production MPU CUDA header and its real FFI exports into an object
for Hopper; it neither links nor loads that object. Default ``compile`` also
calls the production JIT factory in another process and checks its exports.
``device`` executes small compute-side load/store checks, not model correctness
or a simultaneous tensor-parallel workload. All ordinals are CUDA-visible IDs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUSES = {"passed", "failed", "blocked"}
PACKAGES = (
    "torch",
    "triton",
    "flashinfer-python",
    "sgl-kernel",
    "sglang-kernel",
    "apache-tvm-ffi",
    "tilelang",
    "transformers",
    "msgspec",
    "ninja",
    "sglang",
)
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CXX",
)
FAMILIES = {
    "qwen": {
        "kernel_module": "sglang.jit_kernel.mpu_io",
        "kernel_root": "python/sglang/jit_kernel",
        "arch_module": "sglang.jit_kernel.utils",
        "pool_module": "sglang.srt.mem_cache.mpu_memory_pool",
        "runtime_modules": ["sglang.srt.layers.attention.flashinfer_backend"],
    },
    "deepseek-v4-flash": {
        "kernel_module": "sglang.kernels.ops.kvcache.mpu",
        "kernel_root": "python/sglang/kernels/jit",
        "arch_module": "sglang.kernels.jit.utils.arch",
        "pool_module": "sglang.srt.mem_cache.mpu_peer",
        "runtime_modules": [
            "sglang.srt.mem_cache.deepseek_v4_memory_pool",
            "sglang.srt.layers.attention.deepseek_v4_backend",
        ],
    },
    "glm-5.3-flash": {
        "kernel_module": "sglang.kernels.ops.kvcache.mpu",
        "kernel_root": "python/sglang/kernels/jit",
        "arch_module": "sglang.kernels.jit.utils.arch",
        "pool_module": "sglang.srt.mem_cache.mpu_peer",
        "runtime_modules": ["sglang.srt.mem_cache.mpu_glm"],
    },
}


class Blocked(RuntimeError):
    """An absent runtime/toolchain prerequisite, rather than a passed probe."""


class ProbeInterrupted(RuntimeError):
    """An outer supervisor requested termination; unwind and kill our workers."""


def termination_handler(signum, frame):
    raise ProbeInterrupted(f"Probe received termination signal {signum}")


def check(name, status, reason=None, **evidence):
    if status not in STATUSES:
        raise ValueError(f"Invalid probe status: {status}")
    result = {"name": name, "status": status, "evidence": evidence}
    if reason:
        result["reason"] = reason
    return result


def aggregate(checks):
    statuses = {entry["status"] for entry in checks}
    if "failed" in statuses:
        return "failed"
    if not checks or "blocked" in statuses:
        return "blocked"
    return "passed"


def classify_exception(error):
    if isinstance(
        error,
        (Blocked, ModuleNotFoundError, FileNotFoundError, subprocess.TimeoutExpired),
    ):
        return "blocked"
    text = str(error).lower()
    # Keep compiler/type/ABI errors as failed. These specific errors identify
    # missing environment prerequisites, not a kernel that compiled correctly.
    missing = (
        "no nvidia driver",
        "found no nvidia driver",
        "no cuda gpus are available",
        "torch not compiled with cuda",
        "cuda driver version is insufficient",
        "libcuda.so",
        "libcudart.so",
        "cannot open shared object file",
        "requires the native cuda allocator",
        "does not support expandable_segments",
        "requires linux with nvidia cuda",
        "cannot peer-access storage gpu",
        "cuda peer access",
        "peer load/store is unavailable",
    )
    return "blocked" if any(token in text for token in missing) else "failed"


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def compare_environment_pins(pyproject, expected, actual):
    """Compare core pins as data; never import packages to establish identity."""
    source = {}
    for name, version in re.findall(
        r'["\']([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^"\' ;]+)', pyproject
    ):
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in expected:
            source.setdefault(normalized, set()).add(version)
    checks = []
    for name, wanted in expected.items():
        installed = actual.get(name)
        source_pins = sorted(source.get(name, set()))
        allowed = {wanted, wanted + "+cu130"} if name == "torch" else {wanted}
        if source_pins != [wanted]:
            status, reason = (
                "failed",
                "Source pyproject does not match the selected environment profile",
            )
        elif installed is None:
            status, reason = "blocked", "Required distribution is not installed"
        elif installed not in allowed:
            status, reason = (
                "failed",
                "Installed distribution does not match the selected core pin",
            )
        else:
            status, reason = "passed", None
        checks.append(
            check(
                name,
                status,
                reason,
                expected=wanted,
                actual=installed,
                source_pins=source_pins,
            )
        )
    return check(
        "environment_package_pins",
        aggregate(checks),
        expected_packages=expected,
        actual_packages=actual,
        package_checks=checks,
        version_policy="Exact core pins; torch additionally accepts the +cu130 local suffix",
        scope="package_metadata_and_source_pins_only_no_import_or_cuda_execution",
    )


def environment_pins(payload):
    profiles = []
    for path in sorted(
        (ROOT / "benchmark/mpu_stages/environments").glob("*-cu130.json")
    ):
        profile = json.loads(path.read_text(encoding="utf-8"))
        if payload["family"] in profile.get("families", []):
            profiles.append((path, profile))
    if len(profiles) != 1:
        raise ValueError(
            "Exactly one cu130 environment profile must match the stage family"
        )
    path, profile = profiles[0]
    expected = profile.get("expected_packages")
    if (
        profile.get("schema_version") != 1
        or not isinstance(expected, dict)
        or not expected
    ):
        raise ValueError(
            "Environment profile must declare schema_version=1 and expected_packages"
        )
    actual = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
    pyproject = ROOT / "python/pyproject.toml"
    result = compare_environment_pins(
        pyproject.read_text(encoding="utf-8"), expected, actual
    )
    result["evidence"].update(
        profile=file_identity(path), pyproject=file_identity(pyproject)
    )
    return result


def environment():
    return {
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages": package_versions(),
        "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    }


def source_metadata(stage_path, family):
    def git(*args):
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    module_path = ROOT / "python" / (family["kernel_module"].replace(".", "/") + ".py")
    return {
        "repository": str(ROOT),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain"),
        "stage": file_identity(stage_path),
        "probe": file_identity(__file__),
        "kernel_module": file_identity(module_path),
        "dependency_spec": file_identity(ROOT / "python/pyproject.toml"),
    }


def load_recipe(path):
    stage = json.loads(Path(path).read_text(encoding="utf-8"))
    if stage.get("schema_version") != 1 or stage.get("family") not in FAMILIES:
        raise ValueError("Expected stage schema_version=1 and a supported family")
    if stage.get("tp_size") not in (1, 4) or stage.get("variant") not in ("cpu", "mpu"):
        raise ValueError("Stage must use TP1/TP4 and cpu/mpu backing")
    return stage


def device_mapping(stage, compute_ids=None, mpu_ids=None, memory_reserve_bytes=None):
    flags = stage.get("server_args", [])
    base = (
        int(flags[flags.index("--base-gpu-id") + 1]) if "--base-gpu-id" in flags else 0
    )
    step = (
        int(flags[flags.index("--gpu-id-step") + 1]) if "--gpu-id-step" in flags else 1
    )
    expected = [base + rank * step for rank in range(stage["tp_size"])]
    compute = expected if compute_ids is None else compute_ids
    if len(compute) != stage["tp_size"]:
        raise ValueError("--compute-device-ids must contain exactly tp_size ordinals")
    if compute != expected:
        raise ValueError(
            "Compute ordinals must match the stage server's rank-to-device mapping"
        )
    if stage["tp_size"] == 4 and compute != list(range(4)):
        raise ValueError(
            "TP4 MPU experiments require CUDA ordinal == TP rank (0,1,2,3)"
        )
    if memory_reserve_bytes is not None and (
        type(memory_reserve_bytes) is not int or memory_reserve_bytes < 0
    ):
        raise ValueError("MPU memory reserve must be a nonnegative integer")
    reserve = None
    if stage["variant"] == "cpu":
        if mpu_ids:
            raise ValueError("CPU backing does not accept --mpu-device-ids")
        storage = []
    else:
        config = stage.get("variants", {}).get("mpu", {}).get("hisparse_config", {})
        storage = config.get("mpu_device_ids", []) if mpu_ids is None else mpu_ids
        reserve = (
            config.get("mpu_memory_reserve_bytes", 1 << 30)
            if memory_reserve_bytes is None
            else memory_reserve_bytes
        )
        if type(reserve) is not int or reserve < 0:
            raise ValueError("Stage MPU memory reserve must be a nonnegative integer")
        if (stage["tp_size"] == 1 and not 1 <= len(storage) <= 4) or (
            stage["tp_size"] == 4 and len(storage) != 4
        ):
            raise ValueError("TP1 needs 1-4 MPU IDs; TP4 needs one MPU ID per rank")
    all_ids = [*compute, *storage]
    if any(type(i) is not int or i < 0 for i in all_ids) or len(set(all_ids)) != len(
        all_ids
    ):
        raise ValueError(
            "Compute and MPU IDs must be unique nonnegative visible ordinals"
        )
    pairs = (
        [(compute[0], peer) for peer in storage]
        if stage["tp_size"] == 1
        else list(zip(compute, storage))
    )
    return {
        "compute_device_ids": compute,
        "mpu_device_ids": storage,
        "mpu_memory_reserve_bytes": reserve,
        "pairs": [{"compute": c, "storage": s} for c, s in pairs],
        "ordinal_scope": "after CUDA_VISIBLE_DEVICES filtering",
    }


def production_spec(family, root=ROOT):
    """Read literal arguments to the actual factory without importing SGLang."""
    module_file = root / "python" / (family["kernel_module"].replace(".", "/") + ".py")
    tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
    factory = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_module"
        ),
        None,
    )
    if factory is None:
        raise ValueError(f"Production module has no _module factory: {module_file}")
    calls = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_jit"
    ]
    if len(calls) != 1:
        raise ValueError("Expected exactly one literal production load_jit call")
    call = calls[0]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    if set(kwargs) != {"cuda_files", "cuda_wrappers"}:
        raise ValueError(
            "MPU JIT factory changed; review compile-only flags and sources"
        )
    sources = [
        (root / family["kernel_root"] / "csrc" / name).resolve()
        for name in kwargs["cuda_files"]
    ]
    if not sources or any(not path.is_file() for path in sources):
        raise ValueError("Production CUDA sources are missing")
    wrappers = kwargs["cuda_wrappers"]
    if not wrappers or any(len(pair) != 2 for pair in wrappers):
        raise ValueError("Production FFI exports are missing")
    return {
        "module_name": "sgl_kernel_jit_"
        + "_".join(ast.literal_eval(arg) for arg in call.args),
        "sources": [str(path) for path in sources],
        "wrappers": wrappers,
        "kernel_include": str(root / family["kernel_root"] / "include"),
        "namespace": (
            "" if family["kernel_module"] == "sglang.jit_kernel.mpu_io" else "sglang"
        ),
    }


def wrapper_source(spec):
    lines = [
        "#include <tvm/ffi/container/tensor.h>",
        "#include <tvm/ffi/dtype.h>",
        "#include <tvm/ffi/error.h>",
        "#include <tvm/ffi/extra/c_env_api.h>",
        "#include <tvm/ffi/function.h>",
    ]
    lines.extend(f'#include "{Path(path).as_posix()}"' for path in spec["sources"])
    if spec["namespace"]:
        lines.append(f'namespace {spec["namespace"]} {{')
    lines.extend(
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, ({target}));"
        for name, target in spec["wrappers"]
    )
    if spec["namespace"]:
        lines.append("}")
    return "\n".join(lines) + "\n"


def find_nvcc():
    # Match the project's CUDA_HOME/CUDA_PATH precedence, not a different PATH toolkit.
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    candidate = Path(configured) / "bin/nvcc" if configured else None
    if candidate is not None:
        return str(candidate) if candidate.is_file() else None
    return shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc"
        if Path("/usr/local/cuda/bin/nvcc").is_file()
        else None
    )


def ffi_include_paths():
    """Locate installed real headers without importing CUDA/native libraries."""
    try:
        dist = importlib.metadata.distribution("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError as error:
        raise Blocked(
            "apache-tvm-ffi must be installed for the production FFI headers"
        ) from error
    roots = []
    for suffix in ("tvm/ffi/function.h", "dlpack/dlpack.h"):
        found = []
        for entry in dist.files or ():
            if str(entry).replace("\\", "/").endswith(suffix):
                path = Path(dist.locate_file(entry)).resolve()
                if path.is_file():
                    found.append(path.parents[len(Path(suffix).parts) - 1])
        if not found:
            raise Blocked(
                f"Installed apache-tvm-ffi metadata does not locate {suffix}; a complete header installation is required"
            )
        roots.append(str(found[0]))
    return roots


def compile_command(nvcc, source, output, includes):
    command = [
        nvcc,
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        "-DSGL_CUDA_ARCH=900",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-Xcompiler",
        "-fPIC",
    ]
    if os.environ.get("CXX"):
        command.extend(["-ccbin", os.environ["CXX"]])
    command.extend(f"-I{path}" for path in includes)
    return [
        *command,
        "-MMD",
        "-MF",
        str(output) + ".d",
        "-c",
        str(source),
        "-o",
        str(output),
    ]


def compile_object(payload):
    family = FAMILIES[payload["family"]]
    spec = production_spec(family)
    nvcc = find_nvcc()
    if not nvcc:
        raise Blocked("nvcc is unavailable; source inspection is not CUDA compilation")
    work = Path(payload["work_dir"])
    source, output = work / "mpu_production.cu", work / "mpu_production.o"
    source.write_text(wrapper_source(spec), encoding="utf-8")
    includes = [spec["kernel_include"], *ffi_include_paths()]
    command = compile_command(nvcc, source, output, includes)
    (work / "compiler_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    compiler = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=False, timeout=15
    )
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=payload["timeout"],
    )
    (work / "nvcc.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (work / "nvcc.stderr.log").write_text(proc.stderr, encoding="utf-8")
    evidence = {
        "command": command,
        "compiler_version": compiler.stdout.strip(),
        "target": "sm_90a",
        "sources": [file_identity(path) for path in spec["sources"]],
        "ffi_exports": spec["wrappers"],
        "object": file_identity(output),
        "compiler_dependency_file": file_identity(str(output) + ".d"),
        "returncode": proc.returncode,
        "native_module_loaded": False,
        "gpu_executed": False,
        "scope": "production_cuda_translation_unit_object_only_no_link_or_load",
    }
    if proc.returncode:
        return check(
            "production_object_compile",
            "failed",
            "nvcc rejected the production translation unit; see compiler logs",
            **evidence,
        )
    if not output.is_file() or output.stat().st_size == 0:
        return check(
            "production_object_compile",
            "failed",
            "nvcc returned success without a nonempty object",
            **evidence,
        )
    return check("production_object_compile", "passed", **evidence)


def import_module_probe(payload):
    module = importlib.import_module(payload["module"])
    origin = getattr(module, "__file__", None)
    if payload["module"].startswith("sglang.") and (
        origin is None or not Path(origin).resolve().is_relative_to(ROOT / "python")
    ):
        raise RuntimeError("Production import resolved outside this checkout")
    return check(
        "production_import",
        "passed",
        module=payload["module"],
        origin=origin,
        scope="import_only_no_kernel_compilation_or_gpu_execution",
    )


def native_load(payload):
    family = FAMILIES[payload["family"]]
    arch = importlib.import_module(family["arch_module"])
    kernel = importlib.import_module(family["kernel_module"])
    spec = production_spec(family)
    with arch.override_jit_cuda_arch(9, 0, "a"):
        module = kernel._module()
    for export, _ in spec["wrappers"]:
        if not callable(getattr(module, export, None)):
            raise RuntimeError(f"Native module did not export callable {export}")
    return check(
        "production_native_load",
        "passed",
        factory=family["kernel_module"] + "._module",
        ffi_exports=spec["wrappers"],
        target="sm_90a",
        native_module_loaded=True,
        gpu_executed=False,
        scope="native_module_load_and_exports_no_gpu_launch",
    )


def cuda_torch():
    import torch

    if torch.version.cuda is None or torch.version.hip or not torch.cuda.is_available():
        raise Blocked(
            "A working NVIDIA CUDA PyTorch runtime and visible GPU are required"
        )
    return torch


def device_inventory(payload):
    torch = cuda_torch()
    mapping = payload["mapping"]
    count = torch.cuda.device_count()
    selected = mapping["compute_device_ids"] + mapping["mpu_device_ids"]
    devices = []
    for ordinal in range(count):
        prop = torch.cuda.get_device_properties(ordinal)
        free, total = torch.cuda.mem_get_info(ordinal)
        devices.append(
            {
                "ordinal": ordinal,
                "name": prop.name,
                "capability": [prop.major, prop.minor],
                "uuid": str(getattr(prop, "uuid", "unreported")),
                "total_memory_bytes": total,
                "free_memory_bytes": free,
            }
        )
    evidence = {
        "devices": devices,
        "mapping": mapping,
        "torch_cuda": torch.version.cuda,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
    }
    if max(selected) >= count:
        return check(
            "cuda_device_inventory",
            "blocked",
            "Requested ordinal is outside CUDA_VISIBLE_DEVICES",
            **evidence,
        )
    if any(
        devices[index]["capability"] != [9, 0]
        for index in mapping["compute_device_ids"]
    ):
        return check(
            "cuda_device_inventory",
            "blocked",
            "This stage probe targets Hopper compute GPUs (capability 9.0)",
            **evidence,
        )
    if mapping["mpu_device_ids"]:
        allocator = evidence["allocator_backend"]
        config = ",".join(
            os.environ.get(key, "")
            for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        )
        if allocator != "native" or re.search(
            r"expandable_segments\s*:\s*true", config, re.I
        ):
            return check(
                "cuda_device_inventory",
                "blocked",
                "MPU requires native, non-VMM CUDA allocations",
                **evidence,
            )
        if (
            torch.are_deterministic_algorithms_enabled()
            and torch.utils.deterministic.fill_uninitialized_memory
        ):
            return check(
                "cuda_device_inventory",
                "blocked",
                "Deterministic uninitialized fill would launch on the storage GPU",
                **evidence,
            )
    evidence["directed_peer_access"] = [
        {
            **pair,
            "can_access": torch.cuda.can_device_access_peer(
                pair["compute"], pair["storage"]
            ),
        }
        for pair in mapping["pairs"]
    ]
    if any(not pair["can_access"] for pair in evidence["directed_peer_access"]):
        return check(
            "cuda_device_inventory",
            "blocked",
            "At least one compute -> MPU P2P link is unavailable",
            **evidence,
        )
    return check("cuda_device_inventory", "passed", **evidence)


def compute_smoke(payload):
    torch = cuda_torch()
    ordinal = payload["compute"]
    with torch.cuda.device(ordinal):
        tensor = torch.arange(32, dtype=torch.int64, device=f"cuda:{ordinal}")
        actual = (tensor + 7).sum().item()
        torch.cuda.synchronize(ordinal)
    if actual != sum(range(32)) + 7 * 32:
        raise RuntimeError(
            "Compute allocation/arithmetic smoke produced incorrect data"
        )
    return check(
        "compute_smoke",
        "passed",
        compute_device=ordinal,
        gpu_executed=True,
        scope="single_rank_allocation_and_arithmetic_no_collectives",
    )


def peer_smoke(payload):
    torch = cuda_torch()
    family = FAMILIES[payload["family"]]
    compute, storage = payload["compute"], payload["storage"]
    kernel = importlib.import_module(family["kernel_module"])
    pool = importlib.import_module(family["pool_module"])
    # These production validators check native allocation ownership, including
    # programmatic allocator changes that environment-variable checks miss.
    pool.validate_mpu_runtime(compute, [storage])
    reserve = payload["reserve_bytes"]
    free, total = torch.cuda.mem_get_info(storage)
    if free < reserve + 4096:
        raise Blocked(
            "Storage GPU has insufficient free bytes after the configured reserve"
        )
    owner = None
    peer = None
    with torch.cuda.device(compute):
        try:
            if payload["family"] == "qwen":
                kernel.enable_peer_access(compute, [storage])
                # Four rows, separate K/V, one layer, eight bytes per record.
                peer = torch.empty(
                    (2, 1, 4, 1), dtype=torch.int64, device=f"cuda:{storage}"
                )
                pool.validate_native_peer_allocations([peer])
                ptrs = torch.tensor(
                    [[[peer.data_ptr()], [peer.data_ptr() + 32]]],
                    dtype=torch.int64,
                    device=f"cuda:{compute}",
                )
                local = torch.empty((2, 4), dtype=torch.int64, device=f"cuda:{compute}")
                local_ptrs = torch.tensor(
                    [[local.data_ptr()], [local.data_ptr() + 32]],
                    dtype=torch.int64,
                    device=local.device,
                )
                indices = torch.arange(4, dtype=torch.int64, device=local.device)
                kernel.zero_peer_kv(ptrs, 4, 8)

                def transfer(to_peer):
                    kernel.transfer_peer_kv(
                        ptrs,
                        local_ptrs,
                        indices,
                        indices,
                        4,
                        8,
                        to_peer,
                        local_rows=4,
                        peer_rows=4,
                    )

            else:
                owner = pool.PeerAllocation(
                    compute, storage, (4, 8), torch.int64, memory_reserve_bytes=reserve
                )
                peer = owner.tensor
                ptrs = torch.tensor(
                    [peer.data_ptr()], dtype=torch.int64, device=f"cuda:{compute}"
                )
                local = torch.empty((4, 8), dtype=torch.int64, device=f"cuda:{compute}")
                local_ptrs = torch.tensor(
                    [local.data_ptr()], dtype=torch.int64, device=local.device
                )
                indices = torch.arange(4, dtype=torch.int64, device=local.device)

                def transfer(to_peer):
                    kernel.transfer_records(
                        ptrs,
                        local_ptrs,
                        indices,
                        indices,
                        peer_slots=4,
                        local_slots=4,
                        item_bytes=64,
                        to_peer=to_peer,
                        c4_layout=False,
                    )

            transfer(False)
            torch.cuda.synchronize(compute)
            if torch.count_nonzero(local).item() != 0:
                raise RuntimeError(
                    "Production compute-side peer zero/load smoke failed"
                )
            expected = (
                torch.arange(
                    local.numel(), dtype=torch.int64, device=local.device
                ).reshape(local.shape)
                + 1701
            )
            local.copy_(expected)
            transfer(True)
            local.zero_()
            transfer(False)
            torch.cuda.synchronize(compute)
            if not torch.equal(local, expected):
                raise RuntimeError(
                    "Production compute -> peer store -> compute load mismatch"
                )
            return check(
                "directed_peer_load_store",
                "passed",
                compute_device=compute,
                storage_device=storage,
                allocated_bytes=peer.numel() * peer.element_size(),
                memory_reserve_bytes=reserve,
                storage_free_bytes_before=free,
                storage_total_bytes=total,
                gpu_executed=True,
                native_module_loaded=True,
                native_non_vmm_allocation_verified=True,
                storage_payload_operations="allocation_only_no_tensor_operations",
                storage_kernel_profile_verified=False,
                scope="production_zero_and_linear_load_store_only_no_model_layout_or_concurrent_tp_validation",
            )
        finally:
            torch.cuda.synchronize(compute)
            if owner is not None:
                owner.close()
            peer = None


WORKERS = {
    "environment_pins": environment_pins,
    "import": import_module_probe,
    "compile_object": compile_object,
    "native_load": native_load,
    "inventory": device_inventory,
    "compute": compute_smoke,
    "peer": peer_smoke,
}


def worker_main(request_path, output_path):
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    # A normal production import with the checkout first; no sys.modules stubs.
    sys.path.insert(0, str(ROOT / "python"))
    try:
        result = WORKERS[payload["kind"]](payload)
    except Exception as error:
        result = check(
            payload["kind"],
            classify_exception(error),
            str(error),
            exception_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
    result["runtime"] = environment()
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "passed" else 2


def process_result(command, returncode, output_path, stdout, stderr):
    """Do not turn a process exit, skip, or malformed report into success."""
    try:
        result = json.loads(Path(output_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return check(
            "worker_protocol",
            "failed",
            f"Child produced no valid JSON: {error}",
            command=command,
            returncode=returncode,
        )
    if (
        not isinstance(result, dict)
        or result.get("status") not in STATUSES
        or not result.get("name")
    ):
        return check(
            "worker_protocol",
            "failed",
            "Child report has an invalid status/schema",
            command=command,
            returncode=returncode,
        )
    if result["status"] == "passed" and returncode != 0:
        return check(
            "worker_protocol",
            "failed",
            "Child claimed passed but exited nonzero",
            command=command,
            returncode=returncode,
        )
    result.update(
        command=command,
        returncode=returncode,
        stdout_log=str(stdout),
        stderr_log=str(stderr),
    )
    return result


def wait_for_worker(command, *, cwd, env, stdout, stderr, timeout):
    """End the whole worker group on timeout/termination, including nvcc/ninja."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=(sys.platform == "linux"),
    )
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProbeInterrupted, KeyboardInterrupt):
        if sys.platform == "linux":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
        raise


def run_worker(args, artifacts, name, kind, **payload):
    work = artifacts / name
    work.mkdir(parents=True, exist_ok=False)
    request, output = work / "request.json", work / "result.json"
    request.write_text(
        json.dumps(
            {"kind": kind, "timeout": args.timeout, "work_dir": str(work), **payload}
        ),
        encoding="utf-8",
    )
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--_worker",
        str(request),
        str(output),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["SGLANG_JIT_CACHE_DIR"] = str(work / "jit_cache")
    env["SGLANG_JIT_FORCE_RECOMPILE"] = "1"
    env["SGLANG_CRASH_ON_JIT_COMPILE"] = "0"
    stdout, stderr = work / "stdout.log", work / "stderr.log"
    start = time.monotonic()
    try:
        # File-backed output avoids keeping large compiler/import logs in RAM.
        with stdout.open("w", encoding="utf-8") as out, stderr.open(
            "w", encoding="utf-8"
        ) as err:
            returncode = wait_for_worker(
                command, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=args.timeout
            )
        result = process_result(command, returncode, output, stdout, stderr)
    except subprocess.TimeoutExpired:
        result = check(
            name,
            "blocked",
            "Worker timed out; no successful result was recorded",
            command=command,
            stdout_log=str(stdout),
            stderr_log=str(stderr),
        )
    except OSError as error:
        result = check(name, "blocked", str(error), command=command)
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    result["worker_id"] = name
    return result


def run_probe(args):
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "blocked",
        "checks": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": environment(),
        "capability_scope": {
            "imports": "real_production_imports_only",
            "compile": (
                "object_only" if args.compile_only else "object_and_native_module_load"
            ),
            "device": "sequential_compute_and_directed_peer_smoke_no_collectives_or_model",
        }[args.mode],
    }
    try:
        stage = load_recipe(args.stage)
        family = FAMILIES[stage["family"]]
        report["stage"] = {
            key: stage[key] for key in ("id", "family", "tp_size", "variant")
        }
        report["source"] = source_metadata(args.stage, family)
        mapping = device_mapping(
            stage,
            args.compute_device_ids,
            args.mpu_device_ids,
            args.mpu_memory_reserve_bytes,
        )
        report["mapping"] = mapping
        if args.timeout <= 0 or not math.isfinite(args.timeout):
            raise ValueError("--timeout must be finite and positive")
        if args.compile_only and args.mode != "compile":
            raise ValueError("--compile-only is valid only in compile mode")
        if sys.platform != "linux":
            report["checks"].append(
                check(
                    "supported_platform",
                    "blocked",
                    "Production CUDA probes require Linux; this host is unsupported",
                    platform=sys.platform,
                )
            )
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            artifacts = args.output.parent / (args.output.stem + ".artifacts") / stamp
            artifacts.mkdir(parents=True, exist_ok=False)
            report["artifacts"] = str(artifacts)
            if args.mode == "imports":
                pins = run_worker(
                    args,
                    artifacts,
                    "environment-pins",
                    "environment_pins",
                    family=stage["family"],
                )
                report["checks"].append(pins)
                modules = [
                    "torch",
                    "triton",
                    "flashinfer",
                    "sgl_kernel",
                    "tvm_ffi",
                    "sglang.srt.server_args",
                    "sglang.srt.managers.hisparse_coordinator",
                    "sglang.srt.mem_cache.hisparse_memory_pool",
                    family["kernel_module"],
                    family["pool_module"],
                    *family["runtime_modules"],
                ]
                for index, module in enumerate(
                    dict.fromkeys(modules) if pins["status"] == "passed" else ()
                ):
                    report["checks"].append(
                        run_worker(
                            args,
                            artifacts,
                            f"import-{index:02d}",
                            "import",
                            module=module,
                        )
                    )
            elif args.mode == "compile":
                report["checks"].append(
                    run_worker(
                        args,
                        artifacts,
                        "compile-object",
                        "compile_object",
                        family=stage["family"],
                    )
                )
                report["native_module_load_requested"] = not args.compile_only
                report["gpu_execution_requested"] = False
                if not args.compile_only:
                    if report["checks"][-1]["status"] == "passed":
                        report["checks"].append(
                            run_worker(
                                args,
                                artifacts,
                                "native-load",
                                "native_load",
                                family=stage["family"],
                            )
                        )
                    else:
                        report["checks"].append(
                            check(
                                "production_native_load",
                                "blocked",
                                "Object compilation did not pass",
                            )
                        )
            else:
                inventory = run_worker(
                    args, artifacts, "inventory", "inventory", mapping=mapping
                )
                report["checks"].append(inventory)
                smi = shutil.which("nvidia-smi")
                if smi:
                    try:
                        topo = subprocess.run(
                            [smi, "topo", "-m"],
                            capture_output=True,
                            text=True,
                            timeout=min(args.timeout, 30),
                            check=False,
                        )
                        report["nvidia_smi_topology"] = {
                            "returncode": topo.returncode,
                            "stdout": topo.stdout,
                            "stderr": topo.stderr,
                            "ordinal_scope": "nvidia-smi physical indices; use UUIDs to match CUDA-visible ordinals",
                        }
                    except (OSError, subprocess.TimeoutExpired) as error:
                        report["nvidia_smi_topology"] = {"error": str(error)}
                if inventory["status"] == "passed":
                    for rank, compute in enumerate(mapping["compute_device_ids"]):
                        report["checks"].append(
                            run_worker(
                                args,
                                artifacts,
                                f"compute-rank-{rank}",
                                "compute",
                                compute=compute,
                            )
                        )
                    reserve = mapping["mpu_memory_reserve_bytes"]
                    for pair in mapping["pairs"]:
                        report["checks"].append(
                            run_worker(
                                args,
                                artifacts,
                                f"peer-{pair['compute']}-{pair['storage']}",
                                "peer",
                                family=stage["family"],
                                reserve_bytes=reserve,
                                **pair,
                            )
                        )
    except (ProbeInterrupted, KeyboardInterrupt) as error:
        report["checks"].append(check("probe_interrupted", "failed", str(error)))
    except Exception as error:
        report["checks"].append(
            check(
                "probe_configuration",
                classify_exception(error),
                str(error),
                exception_type=type(error).__name__,
            )
        )
    report["status"] = aggregate(report["checks"])
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_worker":
        return worker_main(*argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("imports", "compile", "device")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stage", type=Path, default=Path(__file__).with_name("stage.json")
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for isolated production probes",
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="Seconds per isolated check"
    )
    parser.add_argument(
        "--compute-device-ids",
        type=int,
        nargs="+",
        help="Optional explicit check of the stage's rank-to-visible-ordinal mapping",
    )
    parser.add_argument("--mpu-device-ids", type=int, nargs="+")
    parser.add_argument(
        "--mpu-memory-reserve-bytes",
        type=int,
        help="Override the stage's storage-memory reserve for peer allocation smoke",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Build a production CUDA object for sm_90a; do not link, load, or execute it",
    )
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.stage = args.stage.resolve()
    previous_handler = signal.signal(signal.SIGTERM, termination_handler)
    try:
        report = run_probe(args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"mode": args.mode, "status": report["status"], "output": str(args.output)}
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
