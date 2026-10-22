"""Plan or execute a fixed-compute-KV capacity sweep for Quest versus MPU HiSparse.

Execution requires Linux CUDA and explicitly passing --execute. No model or GPU
is needed to create the command manifest. Each case keeps its native benchmark,
server log, actual capacities and independently labelled timing analysis.
"""

import argparse
import importlib.util
import json
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("hisparse_qwen_runner", REPO / "benchmark/hisparse_qwen/run.py")
ANALYZE = load_module("hisparse_mpu_analyze", Path(__file__).with_name("analyze.py"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE.MODEL)
    p.add_argument(
        "--revision", help="Defaults to the pinned revision for the default model."
    )
    p.add_argument("--input-lens", type=int, nargs="+", default=[4096, 32768])
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--modes", nargs="+", choices=["quest", "mpu"], default=["quest", "mpu"]
    )
    p.add_argument("--mpu-device-ids", type=int, nargs="+", default=[1])
    p.add_argument("--capacity-ratios", type=float, nargs="+", default=[1, 2, 4])
    p.add_argument("--mpu-memory-reserve-bytes", type=int, default=1 << 30)
    p.add_argument("--top-k", type=int, default=2048)
    p.add_argument("--device-buffer-size", type=int, default=4096)
    p.add_argument(
        "--max-total-tokens",
        type=int,
        required=True,
        help="Fixed physical compute-GPU KV slots, excluding allocator padding.",
    )
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--chunked-prefill-size", type=int, default=8192)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--benchmark-timeout", type=int, default=14400)
    p.add_argument("--trace-timeout", type=float, default=30)
    p.add_argument("--plateau-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    return p


def validate(args):
    for key in (
        "output_len",
        "num_prompts",
        "repeats",
        "top_k",
        "device_buffer_size",
        "max_total_tokens",
        "chunked_prefill_size",
        "startup_timeout",
        "benchmark_timeout",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.output_len < 2:
        raise ValueError("Decode timing requires at least two output tokens")
    for key in (
        "top_k",
        "device_buffer_size",
        "max_total_tokens",
        "chunked_prefill_size",
    ):
        if getattr(args, key) % 64:
            raise ValueError(f"{key} must be divisible by Quest page size 64")
    if args.device_buffer_size < args.top_k:
        raise ValueError("device_buffer_size must cover top_k")
    if args.device_buffer_size > args.max_total_tokens:
        raise ValueError("device_buffer_size exceeds the physical compute KV budget")
    if any(n <= 0 for n in args.input_lens + args.concurrencies):
        raise ValueError("input lengths and concurrency must be positive")
    if args.num_prompts < max(args.concurrencies):
        raise ValueError("num_prompts must cover the largest concurrency")
    if not 1 <= len(args.mpu_device_ids) <= 4 or len(set(args.mpu_device_ids)) != len(
        args.mpu_device_ids
    ):
        raise ValueError("MPU requires 1 to 4 distinct storage device IDs")
    if any(device <= 0 for device in args.mpu_device_ids):
        raise ValueError(
            "Compute GPU is visible ordinal 0; storage ordinals must be positive"
        )
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in args.capacity_ratios):
        raise ValueError("capacity_ratios must be finite and positive")
    if args.mpu_memory_reserve_bytes < 0:
        raise ValueError("mpu_memory_reserve_bytes cannot be negative")
    if not 0 < args.mem_fraction_static < 1 or not 0 <= args.plateau_threshold < 1:
        raise ValueError("invalid mem_fraction_static or plateau_threshold")
    if not math.isfinite(args.trace_timeout) or args.trace_timeout <= 0:
        raise ValueError("trace_timeout must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("invalid port")


def server_command(args, mode, input_len, ratio=None, *, eager=None):
    config = {
        "algorithm": "quest",
        "top_k": args.top_k,
        "quest_page_size": 64,
        "device_buffer_size": args.device_buffer_size,
    }
    if mode == "mpu":
        config.update(
            backing_store="mpu",
            mpu_device_ids=args.mpu_device_ids,
            mpu_capacity_ratio=ratio,
            mpu_memory_reserve_bytes=args.mpu_memory_reserve_bytes,
        )
    elif mode != "quest":
        raise ValueError(f"unknown mode {mode}")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tp-size",
        "1",
        "--dp-size",
        "1",
        "--pp-size",
        "1",
        "--base-gpu-id",
        "0",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--page-size",
        "64",
        "--context-length",
        str(input_len + args.output_len),
        "--max-running-requests",
        str(max(args.concurrencies)),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--cuda-graph-max-bs",
        str(max(args.concurrencies)),
        "--random-seed",
        str(args.seed),
        "--hisparse-config",
        json.dumps(config),
    ]
    revision = args.revision or (BASE.REVISION if args.model == BASE.MODEL else None)
    if revision:
        cmd += ["--revision", revision]
    if mode == "mpu":
        cmd += [
            "--enable-hisparse",
            "--prefill-attention-backend",
            "flashinfer",
            "--decode-attention-backend",
            "flashinfer_hisparse",
        ]
    else:
        cmd += ["--attention-backend", "flashinfer_quest"]
    if args.disable_cuda_graph if eager is None else eager:
        cmd += ["--disable-cuda-graph"]
    return cmd


def make_plan(args):
    groups = []
    for input_len in dict.fromkeys(args.input_lens):
        for mode in dict.fromkeys(args.modes):
            for ratio in sorted(set(args.capacity_ratios)) if mode == "mpu" else [None]:
                gid = f"{mode}-in{input_len}" + (
                    f"-ratio{ratio:g}" if ratio is not None else ""
                )
                cases = []
                for concurrency in dict.fromkeys(args.concurrencies):
                    for repeat in range(args.repeats):
                        cid = f"{gid}-c{concurrency}-r{repeat}"
                        result_path = args.output_dir / f"{cid}.jsonl"
                        cases.append(
                            {
                                "id": cid,
                                "concurrency": concurrency,
                                "repeat": repeat,
                                "status": "planned",
                                "result_path": str(result_path),
                                "command": BASE.benchmark_command(
                                    args, input_len, concurrency, repeat, result_path
                                ),
                            }
                        )
                groups.append(
                    {
                        "id": gid,
                        "mode": mode,
                        "input_len": input_len,
                        "ratio": ratio,
                        "status": "planned",
                        "trace_dir": str(args.output_dir / f"{gid}-trace"),
                        "server_command": server_command(args, mode, input_len, ratio),
                        "cases": cases,
                    }
                )
    return groups


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


def verify_capacity(info, args, mode, ratio):
    states = info.get("internal_states", [])
    if len(states) != 1:
        raise ValueError("Expected exactly one compute scheduler in server info")
    state = states[0]
    if state.get("effective_max_running_requests_per_dp") != max(args.concurrencies):
        raise ValueError(
            "Actual scheduler request limit differs from the fixed sweep setting"
        )
    if mode == "quest":
        slots = state.get("max_total_num_tokens")
        capacity = {"backing_store": "full_hbm", "physical_compute_slots": slots}
    else:
        capacity = state.get("hisparse_mpu_capacity", {})
        slots = capacity.get("physical_compute_slots")
        expected = {
            "backing_store": "mpu",
            "compute_device_id": 0,
            "storage_device_ids": args.mpu_device_ids,
            "requested_capacity_ratio": ratio,
        }
        if any(capacity.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Server MPU topology/configuration differs from the planned case"
            )
        wanted_slots = math.ceil(args.max_total_tokens * ratio)
        if capacity.get("usable_mpu_slots") != wanted_slots:
            raise ValueError(
                "Server usable MPU capacity differs from the planned ratio"
            )
        for key in (
            "allocated_mpu_bytes",
            "compute_kv_bytes",
            "effective_capacity_ratio",
        ):
            if (
                not isinstance(capacity.get(key), (int, float))
                or not math.isfinite(capacity[key])
                or capacity[key] <= 0
            ):
                raise ValueError(f"Server must report positive {key}")
        if not math.isclose(
            capacity["effective_capacity_ratio"], wanted_slots / args.max_total_tokens
        ):
            raise ValueError("Server effective MPU ratio disagrees with usable slots")
    if type(slots) is not int or slots != args.max_total_tokens:
        raise ValueError(
            f"Actual compute KV slots {slots} differ from fixed budget {args.max_total_tokens}"
        )
    return capacity


def start_server(args, command, log, *, trace_dir=None):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"port {args.port} is already in use")
    env = BASE.subprocess_env()
    env.pop("SGLANG_HISPARSE_TIMING_DIR", None)
    if trace_dir:
        env["SGLANG_HISPARSE_TIMING_DIR"] = str(trace_dir)
    return subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def require_cuda(args):
    if sys.platform != "linux":
        raise RuntimeError(
            "Execution requires Linux CUDA; plan mode is available on this host"
        )
    import torch

    if (
        not torch.cuda.is_available()
        or max(args.mpu_device_ids) >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            "Compute GPU and all selected storage GPUs must be CUDA-visible"
        )


def summarize(groups, threshold):
    """Report workload-specific adjacent marginal returns, never infer hardware cost."""
    rows = []
    for group in groups:
        for concurrency in sorted({case["concurrency"] for case in group["cases"]}):
            cases = [
                case for case in group["cases"] if case["concurrency"] == concurrency
            ]
            row = {
                "mode": group["mode"],
                "input_len": group["input_len"],
                "concurrency": concurrency,
                "ratio": group["ratio"],
                "successful_repeats": sum(
                    case["status"] == "success" for case in cases
                ),
                "planned_repeats": len(cases),
                "capacity": group.get("capacity"),
            }
            if all(case["status"] == "success" for case in cases):
                rates = [
                    case["analysis"]["decode_only_estimate"][
                        "decode_throughput_tokens_per_second"
                    ]
                    for case in cases
                ]
                row.update(
                    median_decode_estimate=statistics.median(rates),
                    min_decode_estimate=min(rates),
                    max_decode_estimate=max(rates),
                    median_e2e_output_throughput=statistics.median(
                        case["analysis"]["end_to_end"][
                            "output_throughput_tokens_per_second"
                        ]
                        for case in cases
                    ),
                    median_ttft_ms=statistics.median(
                        case["analysis"]["end_to_end"]["mean_ttft_ms"] for case in cases
                    ),
                    median_tpot_ms=statistics.median(
                        case["analysis"]["end_to_end"]["mean_tpot_ms"] for case in cases
                    ),
                )
            rows.append(row)
    references = {
        (row["input_len"], row["concurrency"]): row
        for row in rows
        if row["mode"] == "quest" and "median_decode_estimate" in row
    }
    previous = {}
    for row in rows:
        if row["mode"] != "mpu":
            continue
        key = row["input_len"], row["concurrency"]
        reference = references.get(key)
        if reference and "median_decode_estimate" in row:
            row["relative_to_quest_hbm"] = {
                "decode_estimate_speedup": row["median_decode_estimate"]
                / reference["median_decode_estimate"],
                "e2e_output_throughput_speedup": row["median_e2e_output_throughput"]
                / reference["median_e2e_output_throughput"],
                "ttft_reduction_fraction": 1
                - row["median_ttft_ms"] / reference["median_ttft_ms"],
                "tpot_reduction_fraction": 1
                - row["median_tpot_ms"] / reference["median_tpot_ms"],
            }
        before = previous.get(key)
        if (
            before is not None
            and "median_decode_estimate" in row
            and "median_decode_estimate" in before
        ):
            gain = row["median_decode_estimate"] / before["median_decode_estimate"] - 1
            row.update(
                previous_ratio=before["ratio"],
                relative_decode_gain=gain,
                gain_below_threshold=gain < threshold,
            )
        previous[key] = row
    return {
        "rows": rows,
        "plateau_threshold": threshold,
        "interpretation": "Adjacent gains under this workload only; no economic optimum without hardware costs, noise analysis and capacity/SLO constraints.",
    }


def execute(args, manifest, path):
    require_cuda(args)
    for group in manifest["groups"]:
        process = None
        try:
            with (args.output_dir / f'{group["id"]}-server.log').open(
                "w", encoding="utf-8"
            ) as log:
                process = start_server(
                    args, group["server_command"], log, trace_dir=group["trace_dir"]
                )
                group["status"] = "starting"
                BASE.write_manifest(path, manifest)
                BASE.wait_ready(process, args.port, args.startup_timeout)
                info = request_json(args.port, "get_server_info")
                BASE.write_manifest(
                    args.output_dir / f'{group["id"]}-server-info.json', info
                )
                group["capacity"] = verify_capacity(
                    info, args, group["mode"], group["ratio"]
                )
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
                                env=BASE.subprocess_env(),
                            )
                        results = ANALYZE.read_jsonl(case["result_path"])
                        if len(results) != 1:
                            raise ValueError(
                                "Expected one fresh native benchmark result"
                            )
                        result = results[0]
                        BASE.check_result(
                            result,
                            args.num_prompts,
                            group["input_len"],
                            args.output_len,
                        )
                        # Complete the idle flush while the server is still alive.
                        case["analysis"] = ANALYZE.analyze_when_ready(
                            result, group["trace_dir"], args.trace_timeout
                        )
                        BASE.write_manifest(
                            args.output_dir / f'{case["id"]}-timing.json',
                            case["analysis"],
                        )
                        case["status"] = "success"
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        case.update(status="failed", error=str(error))
                        break
                    BASE.write_manifest(path, manifest)
                group["status"] = (
                    "success"
                    if all(case["status"] == "success" for case in group["cases"])
                    else "failed"
                )
        except (OSError, ValueError, RuntimeError) as error:
            group.update(status="failed", error=str(error))
        finally:
            if process is not None:
                BASE.stop_server(process)
            for case in group["cases"]:
                if case["status"] == "planned":
                    case["status"] = "not_run_after_failure"
            BASE.write_manifest(path, manifest)
    manifest["summary"] = summarize(manifest["groups"], args.plateau_threshold)
    BASE.write_manifest(args.output_dir / "capacity-curve.json", manifest["summary"])
    return all(group["status"] == "success" for group in manifest["groups"])


def main():
    args = parser().parse_args()
    validate(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = BASE.runtime_info()
    runtime.update(
        gpu_topology=BASE.capture(["nvidia-smi", "topo", "-m"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        allocator_config_new=os.environ.get("PYTORCH_ALLOC_CONF"),
    )
    manifest = {
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "MPU simulator validation; no results exist until successful execution",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime": runtime,
        "groups": make_plan(args),
    }
    path = args.output_dir / "manifest.json"
    BASE.write_manifest(path, manifest)
    if args.execute:
        try:
            manifest["status"] = "running"
            manifest["status"] = (
                "success" if execute(args, manifest, path) else "failed"
            )
        except (ImportError, RuntimeError, KeyboardInterrupt) as error:
            manifest.update(status="failed", error=str(error))
        finally:
            BASE.write_manifest(path, manifest)
    print(path)
    return int(manifest["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
