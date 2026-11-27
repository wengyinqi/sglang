"""Qwen adaptation ablations, using the frozen HiSparse experiment tools.

Planning is portable. Execution requires the H800 runtime and real frozen
document inputs. Every mechanism has its own explicit recipe and raw evidence.
"""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "mpu_stages"
BASE_COMMIT = "b61d8c7f1529ff1971026ae68dcc42af80e7a592"
ARMS = {
    "baseline": (False, False, "none", False),
    "fused": (True, False, "none", False),
    "bounds-on-demand": (False, True, "on_demand", False),
    "bounds-prefetch": (False, True, "lookahead", False),
    "combined": (True, True, "lookahead", False),
    "combined-protected": (True, True, "lookahead", True),
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def identity(path):
    path = Path(path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def source_fingerprint():
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in (ROOT / "python" / "sglang").rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc")
    )
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"files": len(files), "sha256": digest.hexdigest()}


def correctness_gate(args, manifest):
    proof = args.correctness_campaign
    if proof is None:
        raise ValueError("Execution requires --correctness-campaign from this source")
    accepted = json.loads(proof.read_text(encoding="utf-8"))
    if accepted.get("status") != "success" or accepted.get("phase") != "correctness":
        raise ValueError("Correctness campaign did not actually finish successfully")
    if accepted.get("source_fingerprint") != manifest["source_fingerprint"]:
        raise ValueError("Implementation changed after correctness acceptance")
    if accepted.get("workload_identity") != manifest["workload_identity"]:
        raise ValueError("Correctness and performance use different frozen workloads")
    if accepted.get("settings", {}).get("model") != args.model:
        raise ValueError("Correctness uses a different model")
    if not set(args.input_lens) <= set(
        accepted.get("settings", {}).get("input_lens", [])
    ):
        raise ValueError("Input context was not covered by correctness acceptance")
    if not args.disable_cuda_graph and accepted.get("settings", {}).get(
        "disable_cuda_graph"
    ):
        raise ValueError("CUDA Graph mode has not passed model correctness")
    by_arm = {case["arm"]: case for case in accepted.get("cases", [])}
    for arm in args.arms:
        case = by_arm.get(arm)
        if not case or case.get("status") != "success":
            raise ValueError(f"No passed model correctness for {arm}")
        raw_path = Path(case["stage_path"]).parent / "evidence" / "manifest.json"
        native = json.loads(raw_path.read_text(encoding="utf-8"))
        if native.get("status") != "success" or not native.get("groups"):
            raise ValueError("Missing successful raw model correctness manifest")
        if any(g.get("status") != "success" for g in native["groups"]):
            raise ValueError("Incomplete native model correctness groups")
        if not any(g["variant"] == "hbm" for g in native["groups"]):
            raise ValueError("Independent HBM Quest correctness comparison is missing")
    return {"path": str(proof.resolve()), **identity(proof)}


def recipe(arm, *, budget=None):
    if arm not in ARMS:
        raise ValueError("Unknown XM-Sparse adaptation arm")
    stage = json.loads((TOOLS / "stage.json").read_text(encoding="utf-8"))
    stage = copy.deepcopy(stage)
    fused, offload, mode, protect = ARMS[arm]
    config = stage["variants"]["cpu"]["hisparse_config"]
    config.update(
        xm_fused_retrieval_attention=fused,
        xm_quest_bounds_offload=offload,
        xm_quest_prefetch_mode=mode,
        xm_prefetch_protect=protect,
    )
    if budget is not None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("Positive cache budget bytes required")
        config["xm_cache_hbm_budget_bytes"] = budget
        stage["physical_budget_exact"] = False
    stage["xm_adaptation"] = {
        "arm": arm,
        "frozen_baseline_commit": BASE_COMMIT,
        "model": "Qwen3-30B-A3B-Thinking-2507",
        "index": "Quest BF16 page min/max summaries, not a trained K-index",
        "backing": "CPU pinned DDR; one H800 PCIe; NoPD",
        "budget_scope": "KV and named Quest/HiSparse cache metadata; weights, attention workspace and graph pools reported separately",
    }
    return stage


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    p.add_argument(
        "--phase",
        choices=("correctness", "performance", "capacity"),
        default="performance",
    )
    p.add_argument("--input-lens", nargs="+", type=int, default=[16384, 32768])
    p.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4])
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--num-prompts", type=int)
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--capacity-ratio", type=int, default=8)
    p.add_argument("--cache-budget-bytes", type=int)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--correctness-campaign", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.num_prompts is not None and args.num_prompts < max(args.concurrencies):
        p.error("num-prompts must cover every requested client concurrency")
    if (
        min(
            args.input_lens
            + args.concurrencies
            + [
                args.output_len,
                args.repeats,
                args.max_total_tokens,
                args.capacity_ratio,
            ]
        )
        <= 0
    ):
        p.error("Lengths, counts and capacities must be positive")
    if args.phase == "capacity" and not args.cache_budget_bytes:
        p.error("Capacity comparison requires an explicit cache HBM budget")
    if args.phase != "capacity" and args.cache_budget_bytes is not None:
        p.error(
            "Same-batch latency tests use a fixed physical KV allocation; budget belongs to capacity phase"
        )
    if args.phase == "correctness" and args.output_len < 80:
        p.error("Correctness must cross at least one 64-token decode page boundary")
    return args


def make_plan(args):
    root = args.output_dir.resolve()
    cases = []
    # Capacity points change the maximum admitted batch and report actual
    # stable windows. A client concurrency is never treated as an actual batch.
    points = args.concurrencies if args.phase == "capacity" else [None]
    schedule = []
    if args.phase == "correctness":
        schedule = [(None, None, arm) for arm in args.arms]
    else:
        for repeat in range(args.repeats):
            order = list(args.arms)
            random.Random(1729 + repeat).shuffle(order)
            for point in points:
                schedule.extend((point, repeat, arm) for arm in order)
    for point, repeat, arm in schedule:
        name = arm + (f"-c{point}" if point is not None else "")
        if repeat is not None:
            name += f"-r{repeat}"
        case_root = root / name
        stage_path = case_root / "stage.json"
        stage = recipe(arm, budget=args.cache_budget_bytes)
        command = [
            sys.executable,
            str(
                TOOLS / ("correctness.py" if args.phase == "correctness" else "run.py")
            ),
            "--stage",
            str(stage_path),
            "--model",
            args.model,
            "--workload",
            str(args.workload.resolve()),
            "--output-dir",
            str(case_root / "evidence"),
            "--input-lens",
            *map(str, args.input_lens),
            "--concurrencies",
            *map(str, [point] if point is not None else args.concurrencies),
            "--output-len",
            str(args.output_len),
            "--repeats",
            "1",
            "--num-prompts",
            str(
                point
                if point is not None
                else (args.num_prompts or max(16, max(args.concurrencies)))
            ),
            "--max-total-tokens",
            str(args.max_total_tokens),
            "--capacity-ratios",
            str(args.capacity_ratio),
            "--cpu-capacity-ratio",
            str(args.capacity_ratio),
            "--chunked-prefill-size",
            "4096",
            "--prefill-max-requests",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--port",
            str(args.port),
            "--warmup-requests",
            "1",
            "--seed",
            "1729",
        ]
        if args.phase == "correctness":
            # Independent all-HBM Quest reference has both mechanisms off.
            command += ["--include-hbm", "--lifecycle"]
        if args.disable_cuda_graph:
            command += ["--disable-cuda-graph"]
        if args.execute:
            command += ["--execute"]
        cases.append(
            {
                "id": name,
                "arm": arm,
                "global_repeat": repeat,
                "client_concurrency": point,
                "stage": stage,
                "stage_path": str(stage_path),
                "command": command,
                "status": "planned",
            }
        )
    return {
        "schema_version": 1,
        "status": "planned",
        "phase": args.phase,
        "baseline_commit": BASE_COMMIT,
        "workload": str(args.workload.resolve()),
        "workload_identity": identity(args.workload)
        if args.workload.is_file()
        else None,
        "source_fingerprint": source_fingerprint(),
        "settings": {
            "model": args.model,
            "input_lens": args.input_lens,
            "disable_cuda_graph": args.disable_cuda_graph,
            "max_total_tokens": args.max_total_tokens,
            "capacity_ratio": args.capacity_ratio,
            "cache_budget_bytes": args.cache_budget_bytes,
        },
        "cases": cases,
        "evidence_boundary": "Real H800 Qwen adaptation. Fixed-batch stable decode and budget capacity are separate; subtracting prefill does not remove admission or queue effects.",
    }


def main(argv=None):
    args = parse(argv)
    manifest = make_plan(args)
    if args.execute and args.phase != "correctness":
        manifest["correctness_gate"] = correctness_gate(args, manifest)
    destination = args.output_dir.resolve() / "campaign.json"
    write(destination, manifest)
    for case in manifest["cases"]:
        write(case["stage_path"], case["stage"])
    if not args.execute:
        print(destination)
        return
    # Keep interrupted/failed cases visible. Failed measurements are excluded
    # by report analysis, and an exception does not erase completed evidence.
    try:
        for case in manifest["cases"]:
            case["status"] = "running"
            write(destination, manifest)
            with (
                Path(case["stage_path"])
                .with_name("execution.log")
                .open("w", encoding="utf-8") as log
            ):
                result = subprocess.run(
                    case["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            case["returncode"] = result.returncode
            case["status"] = "success" if result.returncode == 0 else "failed"
            write(destination, manifest)
            if result.returncode:
                raise RuntimeError(
                    f"Case failed: {case['id']}; inspect saved execution/server logs"
                )
        manifest["status"] = "success"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write(destination, manifest)


if __name__ == "__main__":
    main()
