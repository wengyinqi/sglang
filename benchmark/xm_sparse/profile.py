"""Adapt a real passed XM-Qwen correctness case to bounded decode profiling.

The receipt contains only the actually executed correctness phase. It is not
a fabricated full hardware/build acceptance and makes no such claim.
"""

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load(Path(__file__).with_name("campaign.py"), "xm_profile_campaign")
PROFILE = load(ROOT / "benchmark/mpu_stages/profile_decode.py", "xm_profile_native")
ACCEPTANCE = PROFILE.load("acceptance")
SCOPE = load(Path(__file__).with_name("profile_scope.py"), "xm_profile_scope")
ANALYZE = load(Path(__file__).with_name("analyze.py"), "xm_profile_analysis")


def receipt(proof, arm, input_len):
    accepted = json.loads(proof.read_text(encoding="utf-8"))
    settings = accepted.get("settings", {})
    CAMPAIGN.correctness_gate(
        SimpleNamespace(
            correctness_campaign=proof,
            model=settings.get("model"),
            input_lens=[input_len],
            disable_cuda_graph=settings.get("disable_cuda_graph", False),
            arms=[arm],
        ),
        {
            "source_fingerprint": CAMPAIGN.source_fingerprint(),
            "workload_identity": CAMPAIGN.identity(Path(accepted["workload"])),
        },
    )
    case = next(case for case in accepted["cases"] if case["arm"] == arm)
    report_path = Path(case["stage_path"]).parent / "evidence/manifest.json"
    native = json.loads(report_path.read_text(encoding="utf-8"))
    stage = json.loads(Path(case["stage_path"]).read_text(encoding="utf-8"))
    if native.get("stage") != stage:
        raise ValueError("Native correctness ran a different stage")
    job = {"report": str(report_path), "command": case["command"]}
    step = {
        "id": "correctness",
        "kind": "model",
        "expected_model_stage": stage,
        "commands": [job],
    }
    result = ACCEPTANCE.external_result(step, job, case.get("returncode", 0))
    if result["status"] != "passed":
        raise ValueError("Actual native correctness no longer passes: " + str(result))
    step.update(status="passed", results=[result])
    return {
        "schema_version": 1,
        "status": "passed_correctness_only",
        "stage": stage,
        "runtime": native["runtime"],
        "steps": [step],
        "evidence_scope": {"target_model_correctness_executed": True},
        "correctness_campaign": PROFILE.identity(proof),
        "scope": "Only actual passed model correctness is adapted; no other acceptance phase is asserted",
    }, native


def native_args(native, stage_path, output_dir, *, input_len, batch, output_len):
    settings = native["settings"]
    result = [
        "--stage",
        str(stage_path),
        "--model",
        settings["model"],
        "--workload",
        settings["workload"],
        "--input-lens",
        str(input_len),
        "--output-len",
        str(output_len),
        "--concurrencies",
        str(batch),
        "--num-prompts",
        str(max(batch, settings["num_prompts"])),
        "--output-dir",
        str(output_dir),
        "--include-hbm",
    ]
    for name in (
        "revision",
        "top_k",
        "device_buffer_size",
        "max_total_tokens",
        "cpu_capacity_ratio",
        "mem_fraction_static",
        "chunked_prefill_size",
        "prefill_max_requests",
        "mpu_memory_reserve_bytes",
        "workload_sha256",
    ):
        if settings.get(name) is not None:
            result.extend(["--" + name.replace("_", "-"), str(settings[name])])
    result += ["--capacity-ratios", str(settings["capacity_ratios"][0])]
    if settings.get("mpu_device_ids") is not None:
        result += ["--mpu-device-ids", *map(str, settings["mpu_device_ids"])]
    if settings.get("disable_cuda_graph"):
        result.append("--disable-cuda-graph")
    return result


def diagnose(root, proof, native_path):
    """Audit actual capture before deriving A.2-style stage/kernel numbers."""
    root = Path(root)
    capture = root / "capture"
    database = capture / "decode.sqlite"
    scope_path = root / "decode-scope.json"
    analysis_path = root / "attention-path.json"
    status = {"schema_version": 1, "performance_result": False}
    step = "scope"
    try:
        traces = sorted((capture / "forward-trace").glob("*.jsonl"))
        if len(traces) != 1:
            raise ValueError("Require one complete single-compute forward trace")
        scope = SCOPE.verify(
            database,
            capture / "profile.json",
            traces[0],
            correctness_campaign=proof,
            acceptance=root / "correctness-only-receipt.json",
            native_correctness=native_path,
            archive_marker=ROOT / ".mpu-source.json",
            source_root=ROOT,
        )
        CAMPAIGN.write(scope_path, scope)
        if (
            scope.get("status") != "passed"
            or scope.get("actual_decode_scope_verified") is not True
        ):
            raise ValueError("New XM-specific scope did not actually pass")
        step = "attention_path"
        analysis = ANALYZE.attention_profile(database, scope)
        CAMPAIGN.write(analysis_path, analysis)
        status.update(
            status="diagnostic_complete",
            scope=str(scope_path),
            attention_path=str(analysis_path),
        )
    except (ValueError, sqlite3.Error, KeyError, TypeError, OSError) as error:
        status.update(
            status="failed_diagnostic_validation", step=step, reason=str(error)
        )
        if not scope_path.exists():
            CAMPAIGN.write(
                scope_path,
                {
                    "schema_version": 1,
                    "status": "failed_scope_verification",
                    "actual_decode_scope_verified": False,
                    "reason": str(error),
                },
            )
    CAMPAIGN.write(root / "diagnostic-status.json", status)
    return 0 if status["status"] == "diagnostic_complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-campaign", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(CAMPAIGN.ARMS), required=True)
    parser.add_argument("--input-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--stable-steps", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if min(args.input_len, args.batch_size, args.output_len, args.stable_steps) <= 0:
        parser.error("Lengths, batch and stable trigger must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    adapted, native = receipt(
        args.correctness_campaign.resolve(), args.arm, args.input_len
    )
    acceptance_path = root / "correctness-only-receipt.json"
    CAMPAIGN.write(acceptance_path, adapted)
    stage_path = root / "stage.json"
    CAMPAIGN.write(stage_path, adapted["stage"])
    forwarded = native_args(
        native,
        stage_path,
        root / "capture",
        input_len=args.input_len,
        batch=args.batch_size,
        output_len=args.output_len,
    ) + [
        "--acceptance",
        str(acceptance_path),
        "--profile-variant",
        "cpu",
        "--profile-concurrency",
        str(args.batch_size),
        "--profile-steps",
        str(args.profile_steps),
        "--profile-stable-steps",
        str(args.stable_steps),
        "--profile-batch-size",
        str(args.batch_size),
    ]
    if args.execute:
        forwarded.append("--execute")
    CAMPAIGN.write(
        root / "profile-adapter.json",
        {
            "schema_version": 1,
            "arm": args.arm,
            "command_args": forwarded,
            "scope": "Independent diagnostic; not a formal throughput sample",
        },
    )
    captured = PROFILE.main(forwarded)
    if captured or not args.execute:
        return captured
    return diagnose(
        root,
        args.correctness_campaign.resolve(),
        adapted["steps"][0]["commands"][0]["report"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
