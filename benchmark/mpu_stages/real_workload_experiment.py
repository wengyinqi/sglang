"""Bounded single-GPU real-workload pairing, capacity sweep and diagnostics."""

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
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--top-k", args.top_k),
        ("--device-buffer-size", args.device_buffer_size),
        ("--seed", args.seed),
        ("--port", args.port),
        ("--warmup-requests", args.warmup_requests),
    ):
        if value is not None:
            options += [flag, str(value)]
    return options


def plan(stage, args, model_manifest, profile):
    if stage["family"] != "qwen" or stage["variant"] != "cpu" or stage["tp_size"] != 1:
        raise ValueError("This bounded campaign is Qwen Quest CPU DDR 1+0 only")
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
        for variant, ratio in (("cpu", 2), ("hbm", 2), ("cpu", 8)):
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
                        str(ratio),
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


def combine(root):
    pair = json.loads((root / "paired/manifest.json").read_text())
    capacity = json.loads((root / "capacity-extra/manifest.json").read_text())
    if pair["status"] != "success" or capacity["status"] != "success":
        raise ValueError("Cannot combine incomplete paired/capacity evidence")
    for key in ("stage", "runtime", "workload"):
        if pair[key] != capacity[key]:
            raise ValueError(f"Campaign blocks disagree on {key}")
    excluded = {"output_dir", "capacity_ratios", "include_hbm", "concurrencies"}
    if {k: v for k, v in pair["settings"].items() if k not in excluded} != {
        k: v for k, v in capacity["settings"].items() if k not in excluded
    }:
        raise ValueError("Paired and capacity blocks changed a controlled setting")
    groups = copy.deepcopy(capacity["groups"])
    for group in pair["groups"]:
        if group["variant"] == "cpu":
            copied = copy.deepcopy(group)
            copied["cases"] = [c for c in copied["cases"] if c["concurrency"] == 32]
            groups.append(copied)
    groups.sort(key=lambda g: (g["input_len"], g["ratio"]))
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
    combined["summary"] = run.summarize(groups, 0.05, combined["analysis_policy"])
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
        "formal_performance_cases": 90,
        "formal_measured_requests": 2880,
        "length_diagnostic_cases": 12,
        "length_diagnostic_requests": 96,
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
