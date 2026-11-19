"""Bounded Qwen real-workload storage pairing, capacity sweep and diagnostics."""

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "real_campaign_run", Path(__file__).with_name("run.py")
)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def command(root, name, entry, arguments):
    return {
        "id": name,
        "command": [
            sys.executable,
            str(Path(__file__).with_name(entry)),
            *arguments,
            "--output-dir",
            str(root / name),
            "--execute",
        ],
        "log": str(root / f"{name}.console.log"),
        "status": "planned",
    }


def common_options(args):
    options = []
    for flag, value in (
        ("--stage", args.stage),
        ("--model", args.model),
        ("--revision", args.revision),
        ("--workload", args.workload),
        ("--workload-sha256", args.workload_sha256),
        ("--max-total-tokens", args.max_total_tokens),
        ("--chunked-prefill-size", args.chunked_prefill_size),
        ("--prefill-max-requests", args.prefill_max_requests),
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--top-k", args.top_k),
        ("--device-buffer-size", args.device_buffer_size),
        ("--seed", args.seed),
        ("--port", args.port),
        ("--warmup-requests", args.warmup_requests),
        ("--cpu-capacity-ratio", getattr(args, "cpu_capacity_ratio", None)),
        ("--mpu-memory-reserve-bytes", getattr(args, "mpu_memory_reserve_bytes", None)),
        ("--startup-timeout", getattr(args, "startup_timeout", None)),
        ("--benchmark-timeout", getattr(args, "benchmark_timeout", None)),
        ("--trace-timeout", getattr(args, "trace_timeout", None)),
        ("--logprob-atol", getattr(args, "logprob_atol", None)),
        ("--plateau-threshold", getattr(args, "plateau_threshold", None)),
        ("--analysis-policy", getattr(args, "analysis_policy", None)),
    ):
        if value is not None:
            options += [flag, str(value)]
    if getattr(args, "mpu_device_ids", None) is not None:
        options += ["--mpu-device-ids", *map(str, args.mpu_device_ids)]
    if getattr(args, "disable_cuda_graph", False):
        options += ["--disable-cuda-graph"]
    return options


def campaign_variant(stage):
    if (
        stage["family"] != "qwen"
        or stage["variant"] not in ("cpu", "mpu")
        or stage["tp_size"] != 1
        or stage.get("selector") != "quest"
        or "hbm" not in stage["variants"]
        or stage.get("hbm_preserves_selector") is not True
    ):
        raise ValueError(
            "This bounded campaign requires Qwen Quest CPU/MPU with its HBM reference"
        )
    return stage["variant"]


def plan(stage, args, model_manifest, profile):
    target = campaign_variant(stage)
    if not args.workload or not model_manifest:
        raise ValueError("A frozen real workload and full model manifest are required")
    root = args.output_dir
    common = common_options(args)
    jobs = [
        command(
            root,
            "correctness",
            "acceptance.py",
            common
            + [
                "--through",
                "correctness",
                "--model-manifest",
                str(model_manifest),
                "--input-lens",
                "8193",
                "8256",
                "32768",
                "--output-len",
                "80",
                "--num-prompts",
                "32",
                "--capacity-ratios",
                "1",
                "2",
                "4",
                "8",
                "--include-hbm",
            ],
        )
    ]
    jobs += [
        command(
            root,
            "paired",
            "run.py",
            common
            + [
                "--input-lens",
                "8192",
                "32768",
                "--output-len",
                "256",
                "--num-prompts",
                "32",
                "--repeats",
                "5",
                "--concurrencies",
                "1",
                "4",
                "32",
                "--capacity-ratios",
                "2",
                "--include-hbm",
            ],
        )
    ]
    # Ratio 2 at concurrency 32 is already measured in the paired block.
    jobs += [
        command(
            root,
            "capacity-extra",
            "run.py",
            common
            + [
                "--input-lens",
                "8192",
                "32768",
                "--output-len",
                "256",
                "--num-prompts",
                "32",
                "--repeats",
                "5",
                "--concurrencies",
                "32",
                "--capacity-ratios",
                "1",
                "4",
                "8",
            ],
        )
    ]
    for length in (256, 1024):
        jobs += [
            command(
                root,
                f"decode-length-{length}",
                "run.py",
                common
                + [
                    "--input-lens",
                    "32768",
                    "--output-len",
                    str(length),
                    "--num-prompts",
                    "8",
                    "--repeats",
                    "3",
                    "--concurrencies",
                    "1",
                    "--capacity-ratios",
                    "2",
                    "--include-hbm",
                ],
            )
        ]
    if profile:
        profiles = [(target, 2), ("hbm", 2), (target, 8)]
        if target == "mpu":
            profiles.insert(0, ("cpu", args.cpu_capacity_ratio))
        for variant, ratio in profiles:
            jobs += [
                command(
                    root,
                    f"profile-{variant}-ratio{ratio}",
                    "profile_decode.py",
                    common
                    + [
                        "--acceptance",
                        str(root / "correctness/acceptance.json"),
                        "--input-lens",
                        "32768",
                        "--output-len",
                        "80",
                        "--num-prompts",
                        "32",
                        "--capacity-ratios",
                        str(2 if variant == "cpu" and target == "mpu" else ratio),
                        "--profile-variant",
                        variant,
                        "--profile-concurrency",
                        "1",
                        "--profile-steps",
                        "32",
                        "--include-hbm",
                        "--concurrencies",
                        "1",
                        "4",
                        "32",
                    ],
                )
            ]
    return jobs


def option_values(command, flag):
    start = command.index(flag) + 1
    values = []
    for value in command[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def planned_counts(stage, jobs):
    counts = dict(
        formal_performance_cases=0,
        formal_measured_requests=0,
        length_diagnostic_cases=0,
        length_diagnostic_requests=0,
    )
    for job in jobs:
        if job["id"] not in ("paired", "capacity-extra") and not job["id"].startswith(
            "decode-length-"
        ):
            continue
        command = job["command"]
        variant_count = len(set(option_values(command, "--capacity-ratios")))
        variant_count += int(stage["variant"] == "mpu")
        variant_count += int("--include-hbm" in command)
        cases = (
            variant_count
            * len(set(option_values(command, "--input-lens")))
            * len(set(option_values(command, "--concurrencies")))
            * int(option_values(command, "--repeats")[0])
        )
        requests = cases * int(option_values(command, "--num-prompts")[0])
        case_key, request_key = (
            ("formal_performance_cases", "formal_measured_requests")
            if job["id"] in ("paired", "capacity-extra")
            else ("length_diagnostic_cases", "length_diagnostic_requests")
        )
        counts[case_key] += cases
        counts[request_key] += requests
    return counts


def group_identity(group):
    return group["variant"], group["input_len"], group["ratio"]


def unique_groups(groups):
    indexed = {}
    for group in groups:
        key = group_identity(group)
        if key in indexed:
            raise ValueError(f"Duplicate campaign group: {key}")
        if group.get("status") != "success" or not group.get("cases"):
            raise ValueError(f"Incomplete campaign group: {key}")
        indexed[key] = group
    return indexed


def combine(root):
    pair = json.loads((root / "paired/manifest.json").read_text(encoding="utf-8"))
    capacity = json.loads(
        (root / "capacity-extra/manifest.json").read_text(encoding="utf-8")
    )
    if pair["status"] != "success" or capacity["status"] != "success":
        raise ValueError("Cannot combine incomplete paired/capacity evidence")
    for key in ("stage", "runtime", "workload", "analysis_policy"):
        if pair[key] != capacity[key]:
            raise ValueError(f"Campaign blocks disagree on {key}")
    excluded = {"output_dir", "capacity_ratios", "include_hbm", "concurrencies"}
    if {k: v for k, v in pair["settings"].items() if k not in excluded} != {
        k: v for k, v in capacity["settings"].items() if k not in excluded
    }:
        raise ValueError("Paired and capacity blocks changed a controlled setting")
    target = campaign_variant(pair["stage"])
    paired_groups = unique_groups(pair["groups"])
    extra_groups = unique_groups(capacity["groups"])
    groups = copy.deepcopy(capacity["groups"])
    for group in paired_groups.values():
        if group["variant"] == target:
            if group["ratio"] != 2:
                raise ValueError("Paired campaign must supply target ratio 2")
            copied = copy.deepcopy(group)
            copied["cases"] = [c for c in copied["cases"] if c["concurrency"] == 32]
            if not copied["cases"]:
                raise ValueError("Paired target ratio 2 has no concurrency-32 cases")
            groups.append(copied)
    indexed = unique_groups(groups)
    lengths = set(pair["settings"]["input_lens"])
    expected = {(target, length, ratio) for length in lengths for ratio in (1, 2, 4, 8)}
    if target == "mpu":
        cpu_ratio = pair["settings"]["cpu_capacity_ratio"]
        for length in lengths:
            key = ("cpu", length, cpu_ratio)
            expected.add(key)
            if key not in paired_groups or key not in extra_groups:
                raise ValueError("Missing matched CPU reference in a campaign block")
            reference = paired_groups[key]["capacity"]
            repeated = extra_groups[key]["capacity"]
            run.enforce_fixed_compute(
                run.enforce_fixed_compute(None, reference), repeated
            )
            run.enforce_point_memory(
                run.enforce_point_memory(None, reference), repeated
            )
            keys = (
                "backing_slots",
                "backing_kv_bytes",
                "allocated_backing_kv_bytes",
                "requested_capacity_ratio",
                "effective_capacity_ratio",
                "storage_device_ids",
            )
            if [tuple(row.get(k) for k in keys) for row in reference] != [
                tuple(row.get(k) for k in keys) for row in repeated
            ]:
                raise ValueError(
                    "CPU reference backing capacity changed between campaign blocks"
                )
    if set(indexed) != expected:
        raise ValueError(
            "Campaign capacity groups do not cover exactly the target ratios and CPU reference"
        )
    if any(
        case.get("status") != "success" or case["concurrency"] != 32
        for group in groups
        for case in group["cases"]
    ):
        raise ValueError("Combined capacity requires completed concurrency-32 cases")
    groups.sort(key=lambda g: (g["input_len"], g["variant"], g["ratio"]))
    references = {}
    for group in groups:
        key = group["input_len"]
        references[key] = run.enforce_fixed_compute(
            references.get(key), group["capacity"]
        )
    combined = dict(
        pair,
        kind="real_workload_capacity",
        groups=groups,
        status="success",
        original_manifests=[
            str(root / "paired/manifest.json"),
            str(root / "capacity-extra/manifest.json"),
        ],
    )
    combined["settings"] = dict(
        pair["settings"],
        capacity_ratios=[1, 2, 4, 8],
        include_hbm=False,
        concurrencies=[32],
    )
    combined["summary"] = run.summarize(
        groups,
        pair["settings"].get("plateau_threshold", 0.05),
        combined["analysis_policy"],
    )
    run.COMMON.write_manifest(root / "capacity-combined.json", combined)
    return combined["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--profiles", action="store_true")
    control, forwarded = parser.parse_known_args(argv)
    stage, args = run.STAGE.parse_args(forwarded)
    root = args.output_dir
    jobs = plan(stage, args, control.model_manifest, control.profiles)
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "planned",
        "stage": stage,
        "source": run.COMMON.runtime_info(),
        "workload": str(args.workload),
        "workload_sha256": args.workload_sha256,
        **planned_counts(stage, jobs),
        "case_count_scope": "Planned requests including repeated CPU reference measurements; counts are not execution evidence",
        "jobs": jobs,
    }
    output = root / "campaign.json"
    run.COMMON.write_manifest(output, report)
    if args.execute:
        report["status"] = "running"
        for job in jobs:
            job["status"] = "running"
            run.COMMON.write_manifest(output, report)
            with Path(job["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    job["command"],
                    cwd=run.COMMON.REPO,
                    env=run.COMMON.subprocess_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            job.update(
                status="success" if result.returncode == 0 else "failed",
                returncode=result.returncode,
            )
            run.COMMON.write_manifest(output, report)
            if result.returncode:
                report["status"] = "failed"
                break
            if job["id"] == "capacity-extra":
                report["capacity_summary"] = combine(root)
                run.COMMON.write_manifest(output, report)
        else:
            report["status"] = "success"
        for job in jobs:
            if job["status"] == "planned":
                job["status"] = "not_run_after_failure"
        run.COMMON.write_manifest(output, report)
    print(output)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
