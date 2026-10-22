"""Plan or run the Qwen + Quest full-HBM / HiSparse comparison on Linux CUDA.

The default is a command manifest only. Execution is explicit; measured results
are accepted only when every request completes at the requested output length.
See README.md for the differences from the paper's unpublished GH200 recipe.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
REPO = Path(__file__).resolve().parents[2]
MODES = ("quest", "hisparse")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--input-lens", type=int, nargs="+", default=[4096, 32768, 200000])
    p.add_argument(
        "--output-len",
        type=int,
        required=True,
        help="Explicit research choice; Figure 5 does not disclose this value.",
    )
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tp-size", type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument("--device-buffer-size", type=int, default=4096)
    p.add_argument("--host-to-device-ratio", type=int, default=4)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--max-total-tokens", type=int)
    p.add_argument("--chunked-prefill-size", type=int, default=8192)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--disable-overlap-schedule", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--benchmark-timeout", type=int, default=14400)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    return p


def validate(args):
    for name in (
        "output_len",
        "num_prompts",
        "repeats",
        "host_to_device_ratio",
        "chunked_prefill_size",
        "startup_timeout",
        "benchmark_timeout",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if any(n <= 0 for n in args.input_lens + args.concurrencies):
        raise ValueError("input lengths and concurrency must be positive")
    if max(args.input_lens) + args.output_len > 262144:
        raise ValueError(
            "input + output exceeds the pinned Qwen context window (262144)"
        )
    if args.num_prompts < max(args.concurrencies):
        raise ValueError("num_prompts must cover the largest concurrency")
    if args.device_buffer_size < 2048 or args.device_buffer_size % 64:
        raise ValueError("device_buffer_size must be >= 2048 and divisible by 64")
    if not 0 < args.mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must be between 0 and 1")
    if args.max_total_tokens is not None and args.max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")


def server_command(args, mode, input_len):
    config = {
        "algorithm": "quest",
        "top_k": 2048,
        "quest_page_size": 64,
        "device_buffer_size": args.device_buffer_size,
        "host_to_device_ratio": args.host_to_device_ratio,
    }
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--revision",
        args.revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tp-size",
        str(args.tp_size),
        "--disable-radix-cache",
        "--page-size",
        "64",
        "--context-length",
        str(input_len + args.output_len),
        "--max-running-requests",
        str(max(args.concurrencies)),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--cuda-graph-max-bs",
        str(max(args.concurrencies)),
        "--random-seed",
        str(args.seed),
        "--hisparse-config",
        json.dumps(config),
    ]
    if mode == "quest":
        cmd += ["--attention-backend", "flashinfer_quest"]
    elif mode == "hisparse":
        cmd += [
            "--enable-hisparse",
            "--prefill-attention-backend",
            "flashinfer",
            "--decode-attention-backend",
            "flashinfer_hisparse",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if args.max_total_tokens is not None:
        cmd += ["--max-total-tokens", str(args.max_total_tokens)]
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    return cmd


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


def make_plan(args):
    groups = []
    for input_len in dict.fromkeys(args.input_lens):
        for mode in MODES:
            group_id = f"{mode}-in{input_len}"
            cases = []
            for concurrency in dict.fromkeys(args.concurrencies):
                for repeat in range(args.repeats):
                    case_id = f"{group_id}-c{concurrency}-r{repeat}"
                    result_path = args.output_dir / f"{case_id}.jsonl"
                    cases.append(
                        {
                            "id": case_id,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "status": "planned",
                            "result_path": str(result_path),
                            "command": benchmark_command(
                                args, input_len, concurrency, repeat, result_path
                            ),
                        }
                    )
            groups.append(
                {
                    "id": group_id,
                    "mode": mode,
                    "input_len": input_len,
                    "status": "planned",
                    "server_command": server_command(args, mode, input_len),
                    "cases": cases,
                }
            )
    return groups


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
    for package in ("torch", "triton", "flashinfer-python", "sglang", "sgl-kernel"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
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


def execute(args, manifest, manifest_path):
    if sys.platform != "linux":
        raise RuntimeError(
            "execution requires a Linux NVIDIA CUDA environment; use plan mode here"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no model benchmark was run")
    for group in manifest["groups"]:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"port {args.port} is already in use")
        process = None
        try:
            with (args.output_dir / f'{group["id"]}-server.log').open(
                "w", encoding="utf-8"
            ) as log:
                process = subprocess.Popen(
                    group["server_command"],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=subprocess_env(),
                )
                group["status"] = "starting"
                write_manifest(manifest_path, manifest)
                wait_ready(process, args.port, args.startup_timeout)
                group["status"] = "running"
                for case in group["cases"]:
                    try:
                        with (args.output_dir / f'{case["id"]}.log').open(
                            "w", encoding="utf-8"
                        ) as bench_log:
                            subprocess.run(
                                case["command"],
                                cwd=REPO,
                                stdout=bench_log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=args.benchmark_timeout,
                                env=subprocess_env(),
                            )
                        lines = (
                            Path(case["result_path"])
                            .read_text(encoding="utf-8")
                            .splitlines()
                        )
                        if len(lines) != 1:
                            raise ValueError("expected one fresh benchmark result")
                        result = json.loads(lines[0])
                        check_result(
                            result,
                            args.num_prompts,
                            group["input_len"],
                            args.output_len,
                        )
                        case["metrics"] = {
                            key: result[key]
                            for key in (
                                "output_throughput",
                                "mean_ttft_ms",
                                "mean_tpot_ms",
                                "completed",
                            )
                        }
                        case["status"] = "success"
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        case.update(status="failed", error=str(error))
                        # A timed-out client can leave generation running on
                        # the server. Do not measure another case against it.
                        for remaining in group["cases"]:
                            if remaining["status"] == "planned":
                                remaining["status"] = "not_run_after_failure"
                        break
                    write_manifest(manifest_path, manifest)
                group["status"] = (
                    "success"
                    if all(case["status"] == "success" for case in group["cases"])
                    else "failed"
                )
        except (OSError, RuntimeError) as error:
            group.update(status="failed", error=str(error))
            for case in group["cases"]:
                if case["status"] == "planned":
                    case["status"] = "not_run_server_failure"
        finally:
            if process is not None:
                stop_server(process)
            write_manifest(manifest_path, manifest)
    return all(group["status"] == "success" for group in manifest["groups"])


def main():
    args = parser().parse_args()
    validate(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "arXiv:2608.07009v1, sections 4.1/4.2 and Figure 5",
        "claim": "branch comparison; not an exact reproduction of the unpublished paper recipe",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime": runtime_info(),
        "groups": make_plan(args),
    }
    path = args.output_dir / "manifest.json"
    write_manifest(path, manifest)
    if args.execute:
        try:
            manifest["status"] = "running"
            success = execute(args, manifest, path)
            manifest["status"] = "success" if success else "failed"
        except (ImportError, RuntimeError, KeyboardInterrupt) as error:
            manifest.update(status="failed", error=str(error))
        finally:
            write_manifest(path, manifest)
    print(path)
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
