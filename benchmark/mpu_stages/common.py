"""Standalone process and native benchmark helpers for all six MPU stages."""

import importlib.metadata
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def benchmark_command(args, input_len, concurrency, repeat, output):
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dataset-name",
        "random-ids",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "1",
        "--tokenize-prompt",
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(args.seed + repeat),
        "--warmup-requests",
        "1",
        "--flush-cache",
        "--output-details",
        "--output-file",
        str(output),
    ]


def subprocess_env():
    env = os.environ.copy()
    # Both workers and the benchmark must execute the commit recorded below,
    # even when another SGLang installation exists in the chosen environment.
    env["PYTHONPATH"] = str(REPO / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def capture(command):
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=subprocess_env(),
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def runtime_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--porcelain"]),
        "sglang_source": capture(
            [
                sys.executable,
                "-c",
                "import importlib.util; print(importlib.util.find_spec('sglang').origin)",
            ]
        ),
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    # The distribution is sglang-kernel, while its import namespace is sgl_kernel.
    # Keep the old distribution spelling too so older environments remain visible.
    for package in (
        "torch",
        "triton",
        "flashinfer-python",
        "sglang",
        "sglang-kernel",
        "sgl-kernel",
        "tilelang",
        "apache-tvm-ffi",
    ):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    marker = REPO / ".mpu-source.json"
    if marker.is_file():
        try:
            source = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not isinstance(source, dict)
                or source.get("provenance") != "git_archive"
                or not re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit", ""))
                or not re.fullmatch(
                    r"[0-9a-f]{64}", source.get("source_archive_sha256", "")
                )
            ):
                raise ValueError("Invalid archived-source identity")
            info["archived_source"] = source
            if info["git_commit"].startswith("unavailable:"):
                info["git_commit"] = source["source_commit"]
                info["git_commit_source"] = "archive_build_marker"
                # An archive has no index to prove an unmodified working tree.
                # Its commit marker records build provenance, not a current diff.
                info["git_status"] = "unavailable: archived source has no Git index"
        except (OSError, ValueError, TypeError) as error:
            info["archived_source_error"] = str(error)
    info.setdefault("git_commit_source", "git")
    if hasattr(os, "sysconf"):
        info["host_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )
    return info


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def check_result(result, num_prompts, input_len, output_len):
    if result.get("completed") != num_prompts:
        raise ValueError("not every request completed")
    if result.get("total_output_tokens") != num_prompts * output_len:
        raise ValueError(
            "generated token count does not match the requested fixed length"
        )
    if result.get("output_lens") != [output_len] * num_prompts:
        raise ValueError(
            "per-request generation lengths differ from the requested fixed length"
        )
    if result.get("input_lens") != [input_len] * num_prompts:
        raise ValueError("measured prompt lengths differ from the planned workload")
    if any(result.get("errors", [])):
        raise ValueError("benchmark reported request errors")
    for name in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms"):
        value = result.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
            raise ValueError(f"invalid metric: {name}")


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    # Local serving requests must not be routed through a configured web proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode}; see server log"
            )
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("server startup timed out; see server log")


def stop_server(process):
    # The runner owns this newly created process group, including TP workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def request_json(port, route, payload=None, *, timeout=30):
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def start_server(args, command, log, *, trace_dir=None, env_updates=None):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"port {args.port} is already in use")
    env = subprocess_env()
    env.pop("SGLANG_MPU_STAGE_TIMING_DIR", None)
    env.pop("SGLANG_HISPARSE_TIMING_DIR", None)
    stage = (
        json.loads(Path(args.stage).read_text(encoding="utf-8"))
        if getattr(args, "stage", None)
        else {}
    )
    env.update(stage.get("server_environment", {}))
    if trace_dir:
        env["SGLANG_MPU_STAGE_TIMING_DIR"] = str(trace_dir)
    env.update(env_updates or {})
    return subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
