"""Counterbalanced repeated CPU/MPU measurements with atomic paired-block resume.

One block contains all storage variants for one input length and repeat ID.
An incomplete block is retried in full in a new directory; its samples are not
merged with completed pairs. Model startup and warmup remain outside timing.
"""

import argparse
import copy
import json
import random
import signal
import sys
from pathlib import Path

import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired_base", HERE / "run.py")
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)
CHECK, ACCEPT = RUN.load("checkpoint"), RUN.load("acceptance")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--order-seed", type=int)
    control, forwarded = parser.parse_known_args(argv)
    stage, args = RUN.STAGE.parse_args(forwarded)
    return stage, args, control


def schedule(groups, repeats, seed):
    rng, blocks = random.Random(seed), []
    inputs = list(dict.fromkeys(g["input_len"] for g in groups))
    rng.shuffle(inputs)
    for length in inputs:
        matching = [g for g in groups if g["input_len"] == length]
        order = [g["id"] for g in matching]
        rng.shuffle(order)
        concurrencies = sorted({c["concurrency"] for g in matching for c in g["cases"]})
        for repeat in range(repeats):
            offset = repeat % len(order)
            concurrency_order = concurrencies.copy()
            rng.shuffle(concurrency_order)
            blocks.append(
                {
                    "id": f"in{length}-r{repeat}",
                    "input_len": length,
                    "repeat": repeat,
                    "group_order": order[offset:] + order[:offset],
                    "concurrency_order": concurrency_order,
                    "status": "planned",
                    "attempts": [],
                }
            )
    return blocks


def block_plan(groups, block, directory, args):
    result = []
    by_id = {g["id"]: g for g in groups}
    for gid in block["group_order"]:
        group = copy.deepcopy(by_id[gid])
        group["trace_dir"] = str(directory / (gid + "-trace"))
        cases = {
            c["concurrency"]: c
            for c in group["cases"]
            if c["repeat"] == block["repeat"]
        }
        group["cases"] = [cases[c] for c in block["concurrency_order"]]
        for case in group["cases"]:
            output = directory / (case["id"] + ".jsonl")
            case["result_path"] = str(output)
            case["command"] = RUN.COMMON.benchmark_command(
                args, group["input_len"], case["concurrency"], case["repeat"], output
            )
        result.append(group)
    return result


def assemble(stage, args, manifest):
    """Reconstruct samples only from complete, content-verified paired blocks."""
    groups = RUN.make_plan(stage, args)
    destinations = {g["id"]: g for g in groups}
    physical, point_memory = {}, {}
    for block in manifest["paired_execution"]["blocks"]:
        if block["status"] != "success":
            continue
        attempt = block["attempts"][-1]
        directory = (
            args.output_dir
            / "blocks"
            / block["id"]
            / f"attempt-{len(block['attempts']):04d}"
        )
        CHECK.verify(directory, attempt["receipt"])
        result = json.loads((directory / "block.json").read_text(encoding="utf-8"))
        expected = block_plan(RUN.make_plan(stage, args), block, directory, args)
        if result.get("status") != "success" or len(result["groups"]) != len(expected):
            raise ValueError("Paired block is not complete")
        for actual, planned in zip(result["groups"], expected):
            if actual.get("status") != "success" or actual.get(
                "decode_cuda_graph_enabled"
            ) != (not args.disable_cuda_graph):
                raise ValueError("Paired block has a failed group or wrong graph mode")
            if any(
                actual.get(k) != planned[k]
                for k in (
                    "id",
                    "variant",
                    "ratio",
                    "input_len",
                    "server_command",
                    "trace_dir",
                )
            ):
                raise ValueError("Paired group differs from the fixed schedule")
            cap = RUN.verify_capacities(
                actual["capacity"], stage, args, actual["variant"], actual["ratio"]
            )
            if actual["variant"] != "hbm":
                physical[actual["input_len"]] = RUN.enforce_fixed_compute(
                    physical.get(actual["input_len"]), cap
                )
                point_memory[actual["id"]] = RUN.enforce_point_memory(
                    point_memory.get(actual["id"]), cap
                )
            if len(actual["cases"]) != len(planned["cases"]):
                raise ValueError("Missing paired concurrency case")
            for case, expected_case in zip(actual["cases"], planned["cases"]):
                if case.get("status") != "success" or any(
                    case.get(k) != expected_case[k]
                    for k in ("id", "repeat", "concurrency", "command", "result_path")
                ):
                    raise ValueError("Incomplete or changed paired sample")
                if actual["variant"] != "hbm":
                    RUN.verify_capacities(
                        case["idle_capacity"],
                        stage,
                        args,
                        actual["variant"],
                        actual["ratio"],
                    )
                    if any(
                        p.get("used_slots") != 0 or p.get("reserved_slots") != 0
                        for p in case["idle_capacity"]
                    ):
                        raise ValueError("Paired sample leaked backing slots")
                    RUN.enforce_fixed_compute(
                        physical[actual["input_len"]], case["idle_capacity"]
                    )
                    RUN.enforce_point_memory(
                        point_memory[actual["id"]], case["idle_capacity"]
                    )
                destination = destinations[actual["id"]]
                index = next(
                    i
                    for i, c in enumerate(destination["cases"])
                    if c["id"] == case["id"]
                )
                destination["cases"][index] = copy.deepcopy(case)
            destination = destinations[actual["id"]]
            destination.setdefault("capacity", cap)
            destination["decode_cuda_graph_enabled"] = actual[
                "decode_cuda_graph_enabled"
            ]
    for group in groups:
        group["status"] = (
            "success"
            if all(c["status"] == "success" for c in group["cases"])
            else "incomplete_paired_blocks"
        )
    return groups


def validate_schedule(manifest, expected):
    blocks = manifest.get("paired_execution", {}).get("blocks", [])
    fields = ("id", "input_len", "repeat", "group_order", "concurrency_order")
    if len(blocks) != len(expected) or any(
        any(a.get(k) != b[k] for k in fields) for a, b in zip(blocks, expected)
    ):
        raise ValueError("Resume schedule changed")


def execute(stage, args, manifest, executor=RUN.execute):
    base = RUN.make_plan(stage, args)
    path = args.output_dir / "manifest.json"
    # Validate all committed evidence before starting another process.
    manifest["groups"] = assemble(stage, args, manifest)
    for block in manifest["paired_execution"]["blocks"]:
        if block["status"] == "success":
            continue
        attempt = {"status": "running"}
        block["attempts"].append(attempt)
        block["status"] = "running"
        directory = (
            args.output_dir
            / "blocks"
            / block["id"]
            / f"attempt-{len(block['attempts']):04d}"
        )
        RUN.COMMON.write_manifest(path, manifest)
        try:
            directory.mkdir(parents=True, exist_ok=False)
            block_args = copy.copy(args)
            block_args.output_dir = directory
            report = {
                "status": "running",
                "groups": block_plan(base, block, directory, args),
            }
            block_path = directory / "block.json"
            RUN.COMMON.write_manifest(block_path, report)
            success = executor(stage, block_args, report, block_path)
            report["status"] = "success" if success else "failed"
            RUN.COMMON.write_manifest(block_path, report)
            if not success:
                raise ValueError(
                    "Incomplete paired block; retry every variant on resume"
                )
            attempt.update(status="success", receipt=CHECK.seal(directory))
            block["status"] = "success"
            manifest["groups"] = assemble(stage, args, manifest)
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            KeyboardInterrupt,
        ) as error:
            attempt.update(status="failed", error=str(error))
            block["status"] = "failed"
            manifest["status"] = "failed"
            RUN.COMMON.write_manifest(path, manifest)
            return False
        RUN.COMMON.write_manifest(path, manifest)
    manifest["summary"] = RUN.summarize(
        manifest["groups"], args.plateau_threshold, manifest["analysis_policy"]
    )
    manifest["summary"]["run_order"] = (
        "Seeded cyclic counterbalance across variants; matched repeat seeds and concurrency order; whole-block retry. Drift is reduced, not eliminated."
    )
    manifest["status"] = "success"
    RUN.COMMON.write_manifest(
        args.output_dir / "capacity-curve.json", manifest["summary"]
    )
    RUN.COMMON.write_manifest(path, manifest)
    return True


def validate_resume(manifest, stage, args, expected, seed):
    validate_schedule(manifest, expected)
    if manifest["paired_execution"]["schedule_seed"] != seed:
        raise ValueError("Resume schedule seed changed")
    current = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "execute"
    }
    CHECK.same_identity(
        {k: v for k, v in manifest["settings"].items() if k != "execute"}, current
    )
    if manifest["stage"] != stage or manifest["analysis_policy"] != RUN.load(
        "capacity_analysis"
    ).load_policy(args.analysis_policy):
        raise ValueError(
            "Stage or policy changed; reanalyze separately, never mix measurements"
        )


def main(argv=None):
    stage, args, control = parse_args(argv)
    groups = RUN.make_plan(stage, args)
    seed = args.seed if control.order_seed is None else control.order_seed
    expected = schedule(groups, args.repeats, seed)
    path = args.output_dir / "manifest.json"
    if control.resume:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_resume(manifest, stage, args, expected, seed)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = RUN.new_manifest(stage, args, groups, "benchmark")
        manifest["paired_execution"] = {
            "schema_version": 1,
            "schedule_seed": seed,
            "blocks": expected,
            "identity": None,
        }
        RUN.COMMON.write_manifest(args.output_dir / "manifest.json", manifest)
    passed = True
    if args.execute:
        with CHECK.exclusive(args.output_dir):
            if control.resume:
                manifest = json.loads(
                    (args.output_dir / "manifest.json").read_text(encoding="utf-8")
                )
                validate_resume(manifest, stage, args, expected, seed)
            try:
                RUN.require_cuda(stage, args)
                identity = CHECK.identity(
                    RUN.COMMON, ACCEPT, stage, args, control.model_manifest
                )
                old = manifest["paired_execution"]["identity"]
                if old is not None:
                    CHECK.same_identity(old, identity)
                elif control.resume and any(
                    b["attempts"] for b in manifest["paired_execution"]["blocks"]
                ):
                    raise ValueError("Executed checkpoint has no identity")
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                manifest.update(status="blocked", reason=str(error))
                RUN.COMMON.write_manifest(path, manifest)
                print(path)
                return 1
            manifest.pop("reason", None)
            manifest["paired_execution"]["identity"] = identity
            previous = None
            if sys.platform == "linux":

                def interrupt(signum, frame):
                    raise KeyboardInterrupt("Paired runner terminated")

                previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                passed = execute(stage, args, manifest)
            finally:
                if previous is not None:
                    signal.signal(signal.SIGTERM, previous)
    print(args.output_dir / "manifest.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
