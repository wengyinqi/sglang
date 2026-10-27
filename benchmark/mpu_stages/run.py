"""Portable staged CPU-DDR/MPU HiSparse benchmarking. Default operation only plans."""

import copy
import importlib.util
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"mpu_stages_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON, STAGE, ANALYZE = load("common"), load("stage"), load("analyze")


def make_plan(stage, args):
    groups = []
    for input_len in dict.fromkeys(args.input_lens):
        for variant, ratio in STAGE.variants(stage, args):
            gid = f"{variant}-in{input_len}" + (
                f"-ratio{ratio:g}" if ratio is not None else ""
            )
            cases = []
            for concurrency in dict.fromkeys(args.concurrencies):
                for repeat in range(args.repeats):
                    cid = f"{gid}-c{concurrency}-r{repeat}"
                    output = args.output_dir / f"{cid}.jsonl"
                    cases.append(
                        {
                            "id": cid,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "status": "planned",
                            "result_path": str(output),
                            "command": COMMON.benchmark_command(
                                args, input_len, concurrency, repeat, output
                            ),
                        }
                    )
            groups.append(
                {
                    "id": gid,
                    "variant": variant,
                    "ratio": ratio,
                    "input_len": input_len,
                    "status": "planned",
                    "trace_dir": str(args.output_dir / f"{gid}-trace"),
                    "server_command": STAGE.server_command(
                        stage, args, variant, input_len, ratio
                    ),
                    "cases": cases,
                }
            )
    return groups


def require_cuda(stage, args):
    if sys.platform != "linux":
        raise RuntimeError("Execution requires Linux CUDA; no GPU model run occurred")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < stage["tp_size"]:
        raise RuntimeError("All compute TP ranks must be CUDA-visible")
    if stage["variant"] == "mpu":
        config = STAGE.config_for(stage, args, "mpu", args.capacity_ratios[0])
        devices = config.get("mpu_device_ids", [])
        if not devices or max(devices) >= torch.cuda.device_count():
            raise RuntimeError("All MPU storage device ordinals must be CUDA-visible")


def verify_capacities(capacities, stage, args, variant, ratio):
    if len(capacities) != stage["tp_size"]:
        raise ValueError("Missing per-rank capacity")
    expected_backing = "hbm" if variant == "hbm" else variant
    for rank, capacity in enumerate(capacities):
        if (
            capacity.get("tp_rank") != rank
            or capacity.get("tp_size") != stage["tp_size"]
        ):
            raise ValueError("Capacity TP metadata differs from the stage")
        if capacity.get("backing_store") != expected_backing:
            raise ValueError("Measured backing store differs from selected variant")
        if stage.get("attn_dp_size", 1) > 1:
            for key, value in {
                "attn_dp_size": 4,
                "attn_dp_rank": rank,
                "attn_tp_size": 1,
                "ep_size": 4,
                "moe_tp_size": 1,
                "parallel_strategy": "attention_dp4_ep4",
            }.items():
                if capacity.get(key) != value:
                    raise ValueError(
                        f"Runtime capacity has wrong DP+EP topology: {key}"
                    )
            audit = capacity.get("tensor_parallel_audit", {})
            if audit.get("status") != "passed" or not audit.get("checked_modules"):
                raise ValueError("Runtime model tensor-parallel audit is missing")
        for key in ("physical_compute_slots", "compute_kv_bytes"):
            if type(capacity.get(key)) is not int or capacity[key] <= 0:
                raise ValueError(f"Missing positive actual {key}")
        if variant != "hbm":
            verify_retained_pools(
                capacity, required=stage.get("frozen_host_budget", False)
            )
        if (
            variant != "hbm"
            and stage.get("physical_budget_exact", True)
            and capacity["physical_compute_slots"] != args.max_total_tokens
        ):
            raise ValueError(
                "Actual physical compute slots differ from requested fixed budget"
            )
        if variant != "hbm":
            for key in ("backing_slots", "backing_kv_bytes"):
                if type(capacity.get(key)) is not int or capacity[key] <= 0:
                    raise ValueError(f"Missing positive actual {key}")
            requested = capacity.get("requested_capacity_ratio")
            actual = capacity.get("effective_capacity_ratio")
            if not isinstance(requested, (int, float)) or not math.isclose(
                requested, ratio
            ):
                raise ValueError("Measured requested backing ratio differs from plan")
            if (
                not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not math.isclose(
                    actual,
                    capacity["backing_slots"] / capacity["physical_compute_slots"],
                )
            ):
                raise ValueError(
                    "Actual backing ratio disagrees with reported pool sizes"
                )
            # CPU pools can add allocator padding. Keep reported usable/allocated
            # sizes, ratios and scope; do not pretend the requested ratio is exact.
        if variant == "mpu":
            if (
                capacity.get("addressable_backing_slots", capacity["backing_slots"])
                < capacity["backing_slots"]
            ):
                raise ValueError(
                    "Allocated MPU capacity exceeds the addressable backing budget; refusing a false capacity plateau"
                )
            config = STAGE.config_for(stage, args, variant, ratio)
            devices = config["mpu_device_ids"]
            expected = devices if stage["tp_size"] == 1 else [devices[rank]]
            if capacity.get("storage_device_ids") != expected:
                raise ValueError("Actual per-rank MPU storage map differs from plan")
    return capacities


def verify_retained_pools(capacity, *, required=False):
    total = capacity.get("retained_gpu_pool_bytes")
    shapes = capacity.get("retained_gpu_pool_shapes")
    if total is None and shapes is None and not required:
        return
    if type(total) is not int or total < 0 or not isinstance(shapes, dict):
        raise ValueError("Missing retained GPU pool bytes/shapes capacity signature")
    if required and (total <= 0 or not shapes):
        raise ValueError("Frozen GPU budgets require nonempty retained pool evidence")
    storages = {}
    for path, item in shapes.items():
        if not isinstance(path, str) or not path or not isinstance(item, dict):
            raise ValueError("Invalid retained GPU tensor signature")
        shape, stride = item.get("shape"), item.get("stride")
        if (
            not isinstance(shape, list)
            or not isinstance(stride, list)
            or len(shape) != len(stride)
            or any(type(n) is not int or n < 0 for n in shape + stride)
            or not isinstance(item.get("dtype"), str)
            or not item["dtype"]
        ):
            raise ValueError("Invalid retained GPU tensor shape/dtype/stride")
        for key in ("storage_offset", "storage_bytes", "allocation_group"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError(f"Invalid retained GPU tensor {key}")
        group, size = item["allocation_group"], item["storage_bytes"]
        if storages.setdefault(group, size) != size:
            raise ValueError("Retained GPU storage aliases disagree on allocation size")
    if sum(storages.values()) != total:
        raise ValueError("Retained GPU total bytes disagree with deduplicated storage")


def decode_graph_enabled(server_info):
    """The new phase backend is authoritative; legacy bool is old-checkout fallback."""
    config = server_info.get("cuda_graph_config")
    if isinstance(config, dict):
        backend = config.get("decode", {}).get("backend")
        if backend not in ("disabled", "full", "breakable"):
            raise ValueError(
                "Server did not expose a recognized resolved decode CUDA graph backend"
            )
        return backend != "disabled"
    legacy = server_info.get("disable_cuda_graph")
    if type(legacy) is not bool:
        raise ValueError(
            "Server did not expose its effective decode CUDA graph configuration"
        )
    return not legacy


def wait_capacities(
    trace_dir, stage, args, variant, ratio, *, request_ids=None, require_idle=False
):
    deadline = time.monotonic() + args.trace_timeout
    while True:
        try:
            result = ANALYZE.read_capacities(
                ANALYZE.load_traces(trace_dir),
                stage["tp_size"],
                request_ids,
                require_idle=require_idle,
            )
            return verify_capacities(result, stage, args, variant, ratio)
        except (OSError, ValueError) as error:
            if time.monotonic() >= deadline:
                raise ValueError(
                    f"Per-rank capacity did not become valid: {error}"
                ) from error
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def enforce_fixed_compute(reference, capacities):
    shape = [
        (
            capacity["physical_compute_slots"],
            capacity["compute_kv_bytes"],
            capacity.get("retained_gpu_pool_bytes"),
            copy.deepcopy(capacity.get("retained_gpu_pool_shapes")),
        )
        for capacity in capacities
    ]
    if reference is not None and reference != shape:
        raise ValueError(
            f"Actual compute KV allocation drifted across variants/ratios: {reference} -> {shape}"
        )
    return shape


def summarize(groups, threshold, policy=None):
    rows, previous = [], {}
    for group in groups:
        for concurrency in sorted({case["concurrency"] for case in group["cases"]}):
            cases = [
                case for case in group["cases"] if case["concurrency"] == concurrency
            ]
            row = {
                "variant": group["variant"],
                "input_len": group["input_len"],
                "concurrency": concurrency,
                "ratio": group["ratio"],
                "successful_repeats": sum(
                    case["status"] == "success" for case in cases
                ),
                "planned_repeats": len(cases),
                "per_rank_capacity": group.get("capacity"),
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
                )
                for field in (
                    "output_throughput_tokens_per_second",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                ):
                    row[f"median_{field}"] = statistics.median(
                        case["analysis"]["end_to_end"][field] for case in cases
                    )
                key = group["variant"], group["input_len"], concurrency
                before = previous.get(key)
                if before is not None and "median_decode_estimate" in before:
                    gain = (
                        row["median_decode_estimate"] / before["median_decode_estimate"]
                        - 1
                    )
                    row.update(
                        previous_ratio=before["ratio"],
                        relative_decode_gain=gain,
                        gain_below_threshold=gain < threshold,
                    )
            previous[group["variant"], group["input_len"], concurrency] = row
            rows.append(row)
    summary = {
        "rows": rows,
        "plateau_threshold": threshold,
        "interpretation": "+0 means HiSparse CPU DDR backing. Per-workload adjacent gains are not an economic optimum without costs, uncertainty and SLO constraints.",
    }
    if policy is not None:
        return load("capacity_analysis").enrich(summary, groups, policy)
    return summary


def execute(stage, args, manifest, path):
    require_cuda(stage, args)
    physical_references = {}
    for group in manifest["groups"]:
        process = None
        try:
            with (args.output_dir / f"{group['id']}-server.log").open(
                "w", encoding="utf-8"
            ) as log:
                process = COMMON.start_server(
                    args, group["server_command"], log, trace_dir=group["trace_dir"]
                )
                group["status"] = "starting"
                COMMON.write_manifest(path, manifest)
                COMMON.wait_ready(process, args.port, args.startup_timeout)
                info = COMMON.request_json(args.port, "get_server_info")
                COMMON.write_manifest(
                    args.output_dir / f"{group['id']}-server-info.json", info
                )
                group["capacity"] = wait_capacities(
                    group["trace_dir"], stage, args, group["variant"], group["ratio"]
                )
                key = group["input_len"]
                if group["variant"] != "hbm":
                    physical_references[key] = enforce_fixed_compute(
                        physical_references.get(key), group["capacity"]
                    )
                for case in group["cases"]:
                    try:
                        with (args.output_dir / f"{case['id']}.log").open(
                            "w", encoding="utf-8"
                        ) as bench_log:
                            subprocess.run(
                                case["command"],
                                cwd=COMMON.REPO,
                                stdout=bench_log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=args.benchmark_timeout,
                                env=COMMON.subprocess_env(),
                            )
                        results = ANALYZE.read_jsonl(case["result_path"])
                        if len(results) != 1:
                            raise ValueError(
                                "Expected exactly one new native benchmark result"
                            )
                        result = results[0]
                        COMMON.check_result(
                            result,
                            args.num_prompts,
                            group["input_len"],
                            args.output_len,
                        )
                        case["analysis"] = ANALYZE.analyze_when_ready(
                            result, group["trace_dir"], args.trace_timeout
                        )
                        case["idle_capacity"] = wait_capacities(
                            group["trace_dir"],
                            stage,
                            args,
                            group["variant"],
                            group["ratio"],
                            request_ids=result["request_ids"],
                            require_idle=True,
                        )
                        if group["variant"] != "hbm":
                            enforce_fixed_compute(
                                physical_references[key], case["idle_capacity"]
                            )
                        COMMON.write_manifest(
                            args.output_dir / f"{case['id']}-timing.json",
                            case["analysis"],
                        )
                        case["status"] = "success"
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        case.update(status="failed", error=str(error))
                        break
                    COMMON.write_manifest(path, manifest)
                group["status"] = (
                    "success"
                    if all(case["status"] == "success" for case in group["cases"])
                    else "failed"
                )
        except (OSError, ValueError, RuntimeError) as error:
            group.update(status="failed", error=str(error))
        finally:
            if process is not None:
                COMMON.stop_server(process)
            for case in group["cases"]:
                if case["status"] == "planned":
                    case["status"] = "not_run_after_failure"
            COMMON.write_manifest(path, manifest)
    manifest["summary"] = summarize(
        manifest["groups"], args.plateau_threshold, manifest.get("analysis_policy")
    )
    COMMON.write_manifest(args.output_dir / "capacity-curve.json", manifest["summary"])
    return all(group["status"] == "success" for group in manifest["groups"])


def new_manifest(stage, args, groups, kind):
    runtime = COMMON.runtime_info()
    runtime.update(
        gpu_topology=COMMON.capture(["nvidia-smi", "topo", "-m"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        allocator_config_new=os.environ.get("PYTORCH_ALLOC_CONF"),
    )
    return {
        "analysis_policy": load("capacity_analysis").load_policy(
            getattr(args, "analysis_policy", None)
        ),
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "stage": stage,
        "claim": "Plans are not executed validation; +0 means CPU DDR HiSparse backing, never full-HBM.",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime": runtime,
        "groups": groups,
    }


def main():
    stage, args = STAGE.parse_args()
    groups = make_plan(stage, args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = new_manifest(stage, args, groups, "benchmark")
    path = args.output_dir / "manifest.json"
    COMMON.write_manifest(path, manifest)
    if args.execute:
        try:
            manifest["status"] = (
                "success" if execute(stage, args, manifest, path) else "failed"
            )
        except (ImportError, RuntimeError, KeyboardInterrupt) as error:
            manifest.update(status="failed", error=str(error))
        finally:
            COMMON.write_manifest(path, manifest)
    print(path)
    return int(manifest["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
