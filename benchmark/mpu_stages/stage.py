"""Versioned, model-independent stage recipes and command construction."""

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_stage(path):
    stage = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "id",
        "family",
        "tp_size",
        "default_model",
        "variant",
        "selector",
        "variants",
    }
    if stage.get("schema_version") != 1 or required - stage.keys():
        raise ValueError(
            "Stage recipe must follow schema_version 1 with all required fields"
        )
    if stage["tp_size"] not in (1, 4) or stage["variant"] not in ("cpu", "mpu"):
        raise ValueError("Stages support TP1/TP4 and CPU or MPU backing")
    if "cpu" not in stage["variants"] or stage["variant"] not in stage["variants"]:
        raise ValueError("Every stage requires its CPU HiSparse reference recipe")
    for name, variant in stage["variants"].items():
        if name not in ("cpu", "mpu", "hbm") or not isinstance(variant, dict):
            raise ValueError("Unknown stage variant")
        flags = variant.get("server_args", [])
        if not isinstance(flags, list) or any(
            not isinstance(flag, str) for flag in flags
        ):
            raise ValueError("server_args must be an exact JSON array of CLI strings")
    return stage


def parse_args(argv=None, *, correctness=False):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--stage", type=Path, default=Path(__file__).with_name("stage.json")
    )
    known, _ = pre.parse_known_args(argv)
    stage = load_stage(known.stage)
    defaults = stage.get("defaults", {})
    p = argparse.ArgumentParser(
        parents=[pre],
        description="Plan by default; --execute runs this checkout's selected HiSparse stage.",
    )
    p.add_argument("--model", default=stage["default_model"])
    p.add_argument("--revision", default=stage.get("default_revision"))
    p.add_argument(
        "--input-lens",
        nargs="+",
        type=int,
        default=defaults.get(
            "correctness_input_lens" if correctness else "input_lens",
            None if correctness else [4096],
        ),
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=defaults.get(
            "correctness_output_len" if correctness else "output_len",
            80 if correctness else 128,
        ),
    )
    p.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=defaults.get("concurrencies", [1, 2, 4]),
    )
    p.add_argument("--num-prompts", type=int, default=defaults.get("num_prompts", 32))
    p.add_argument("--repeats", type=int, default=defaults.get("repeats", 3))
    p.add_argument(
        "--max-total-tokens", type=int, default=defaults.get("max_total_tokens", 65536)
    )
    p.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=defaults.get("chunked_prefill_size", 2048),
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=defaults.get("mem_fraction_static", 0.8),
    )
    p.add_argument(
        "--capacity-ratios",
        nargs="+",
        type=float,
        default=defaults.get("capacity_ratios", [2]),
    )
    p.add_argument(
        "--cpu-capacity-ratio", type=int, default=defaults.get("cpu_capacity_ratio", 2)
    )
    p.add_argument("--top-k", type=int)
    p.add_argument("--device-buffer-size", type=int)
    p.add_argument("--mpu-device-ids", nargs="+", type=int)
    p.add_argument("--mpu-memory-reserve-bytes", type=int)
    p.add_argument(
        "--include-hbm",
        action="store_true",
        help="Optional full-HBM storage reference using the identical sparse selector; +0 always means CPU backing.",
    )
    p.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Benchmark eager only; correctness then checks eager only.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--benchmark-timeout", type=int, default=14400)
    p.add_argument("--trace-timeout", type=float, default=30)
    p.add_argument("--logprob-atol", type=float, default=0.02)
    p.add_argument("--plateau-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.input_lens is None:
        config = config_for(stage, args, stage["variant"], args.capacity_ratios[0])
        hot = config.get("device_buffer_size", 4096) * stage.get(
            "sparse_context_multiplier", 1
        )
        page = stage.get("page_size", 64)
        args.input_lens = [2 * hot + 1, 2 * hot + page, 3 * hot + 1]
    if args.model != stage["default_model"] and "--revision" not in (
        argv if argv is not None else sys.argv[1:]
    ):
        args.revision = None
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        args.output_dir = Path("results/mpu_stages") / f"{stage['id']}-{stamp}"
    args.output_dir = args.output_dir.resolve()
    validate(stage, args)
    return stage, args


def validate(stage, args):
    for key in (
        "output_len",
        "num_prompts",
        "repeats",
        "max_total_tokens",
        "chunked_prefill_size",
        "startup_timeout",
        "benchmark_timeout",
        "cpu_capacity_ratio",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.output_len < 2 or any(n <= 0 for n in args.input_lens + args.concurrencies):
        raise ValueError(
            "Positive prompt/concurrency and at least 2 output tokens required"
        )
    if args.num_prompts < max(args.concurrencies):
        raise ValueError("num_prompts must cover concurrency")
    if any(not math.isfinite(r) or r <= 0 for r in args.capacity_ratios):
        raise ValueError("Capacity ratios must be finite and positive")
    if stage["variant"] == "cpu" and any(
        not float(r).is_integer() for r in args.capacity_ratios
    ):
        raise ValueError("CPU HiSparse host_to_device_ratio must be an integer")
    for key in ("top_k", "device_buffer_size"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.mpu_device_ids is not None:
        if len(set(args.mpu_device_ids)) != len(args.mpu_device_ids) or any(
            device < stage["tp_size"] for device in args.mpu_device_ids
        ):
            raise ValueError(
                "Storage device IDs must be unique and separate from compute ranks"
            )
        if (stage["tp_size"] == 1 and not 1 <= len(args.mpu_device_ids) <= 4) or (
            stage["tp_size"] == 4 and len(args.mpu_device_ids) != 4
        ):
            raise ValueError(
                "TP1 supports 1–4 storage devices; TP4 currently uses one storage device per rank"
            )
    if args.mpu_memory_reserve_bytes is not None and args.mpu_memory_reserve_bytes < 0:
        raise ValueError("MPU memory reserve cannot be negative")
    if not 0 < args.mem_fraction_static < 1 or not 0 <= args.plateau_threshold < 1:
        raise ValueError("Invalid memory fraction or plateau threshold")
    if (
        not math.isfinite(args.trace_timeout)
        or args.trace_timeout <= 0
        or not 1 <= args.port <= 65535
    ):
        raise ValueError("Invalid timeout or port")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        raise ValueError("Invalid logprob tolerance")
    if args.include_hbm and (
        "hbm" not in stage["variants"]
        or stage.get("hbm_preserves_selector") is not True
    ):
        raise ValueError(
            "Stage must explicitly define a full-HBM recipe preserving the same sparse selector"
        )
    if stage["variant"] == "mpu":
        if (
            stage.get("frozen_host_budget", False)
            and max(args.capacity_ratios) > args.cpu_capacity_ratio
        ):
            raise ValueError(
                "MPU ratio exceeds the frozen index/address budget; increase --cpu-capacity-ratio for all CPU/MPU runs first"
            )
        cpu = config_for(stage, args, "cpu", args.cpu_capacity_ratio)
        mpu = config_for(stage, args, "mpu", args.capacity_ratios[0])
        for key in (
            "algorithm",
            "top_k",
            "device_buffer_size",
            "quest_page_size",
            "min_sparse_prompt_len",
            "swap_in_block_size",
        ):
            if cpu.get(key) != mpu.get(key):
                raise ValueError(
                    f"CPU/MPU comparison must preserve selector and hot-buffer setting {key}"
                )


def variants(stage, args):
    result = []
    if stage["variant"] == "mpu":
        result.append(("cpu", args.cpu_capacity_ratio))
    result.extend(
        (stage["variant"], ratio) for ratio in sorted(set(args.capacity_ratios))
    )
    if args.include_hbm:
        result.append(("hbm", None))
    return result


def config_for(stage, args, variant, ratio):
    recipe = stage["variants"][variant]
    config = copy.deepcopy(recipe.get("hisparse_config", {}))
    if variant in ("cpu", "mpu"):
        key = recipe.get(
            "ratio_key",
            "host_to_device_ratio" if variant == "cpu" else "mpu_capacity_ratio",
        )
        if variant == "cpu" and not float(ratio).is_integer():
            raise ValueError("CPU ratio must be integral")
        config[key] = int(ratio) if variant == "cpu" else ratio
    for name in ("top_k", "device_buffer_size"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if variant == "mpu":
        config["backing_store"] = "mpu"
        if stage.get("frozen_host_budget", False):
            config["host_to_device_ratio"] = args.cpu_capacity_ratio
        if args.mpu_device_ids is not None:
            config["mpu_device_ids"] = args.mpu_device_ids
        if args.mpu_memory_reserve_bytes is not None:
            config["mpu_memory_reserve_bytes"] = args.mpu_memory_reserve_bytes
    return config


def server_command(stage, args, variant, input_len, ratio, *, eager=None):
    recipe = stage["variants"][variant]
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *stage.get("server_args", []),
        *recipe.get("server_args", []),
    ]
    options = {
        "--model-path": args.model,
        "--host": "127.0.0.1",
        "--port": args.port,
        "--tp-size": stage["tp_size"],
        "--dp-size": 1,
        "--pp-size": 1,
        "--max-running-requests": max(args.concurrencies),
        "--max-total-tokens": args.max_total_tokens,
        "--context-length": input_len + args.output_len,
        "--chunked-prefill-size": args.chunked_prefill_size,
        "--mem-fraction-static": args.mem_fraction_static,
        "--cuda-graph-max-bs": max(args.concurrencies),
        "--random-seed": args.seed,
    }
    if args.revision:
        options["--revision"] = args.revision
    config = config_for(stage, args, variant, ratio)
    if config:
        options["--hisparse-config"] = json.dumps(config)
    for flag, value in options.items():
        if flag in cmd:
            index = cmd.index(flag)
            if cmd[index + 1] != str(value):
                raise ValueError(
                    f"Stage recipe must not override managed option {flag}"
                )
        else:
            cmd += [flag, str(value)]
    for flag in ("--disable-radix-cache", "--disable-overlap-schedule"):
        if flag not in cmd:
            cmd.append(flag)
    use_eager = args.disable_cuda_graph if eager is None else eager
    if use_eager and "--disable-cuda-graph" not in cmd:
        cmd.append("--disable-cuda-graph")
    elif not use_eager and "--disable-cuda-graph" in cmd:
        raise ValueError("Stage recipe disables requested CUDA graph acceptance")
    return cmd
