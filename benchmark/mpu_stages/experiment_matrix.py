"""Versioned NoPD experiment matrix; plan without hardware, execute through gates.

Each invocation uses the current stage and one physical topology. The optional
Qwen 1+4 topology is a separate invocation on stage 02, never a capacity point.
"""

import argparse
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"matrix_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE, RUN, CORRECT, ACCEPT, CAP = [
    load(name)
    for name in ("stage", "run", "correctness", "acceptance", "capacity_analysis")
]
PAIRED, CHECK, DECISION = (
    load("paired_run"),
    load("checkpoint"),
    load("decision_policy"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Unexpected or missing {label} fields")


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Matrix requires positive integers")


def positive_list(values):
    if not isinstance(values, list) or not values or values != sorted(set(values)):
        raise ValueError("Matrix lists must be nonempty, unique and increasing")
    for value in values:
        positive_int(value)


def validate(matrix, stage, policy):
    if policy["slo"] and not {"p99_ttft_ms", "p99_tpot_ms"} <= policy["slo"].keys():
        raise ValueError(
            "Formal policy needs both P99 TTFT and TPOT SLOs, or an empty pending SLO"
        )
    exact_keys(
        matrix,
        (
            "schema_version",
            "id",
            "input_lens",
            "output_len",
            "repeats",
            "seed",
            "mem_fraction_static",
            "logprob_atol",
            "plateau_threshold",
            "plateau_consecutive_edges",
            "profiles",
            "families",
        ),
        "matrix",
    )
    if (
        matrix["schema_version"] != 1
        or not isinstance(matrix["id"], str)
        or not matrix["id"]
    ):
        raise ValueError("Invalid matrix identity")
    positive_list(matrix["input_lens"])
    for key in ("output_len", "repeats", "seed", "plateau_consecutive_edges"):
        positive_int(matrix[key])
    if (
        matrix["repeats"] < max(5, policy["minimum_repeats"])
        or matrix["output_len"] < 2
    ):
        raise ValueError(
            "Formal matrix requires at least five repeats and two output tokens"
        )
    if matrix["plateau_consecutive_edges"] < 2:
        raise ValueError("A plateau needs at least two consecutive capacity increases")
    for key in ("mem_fraction_static", "logprob_atol", "plateau_threshold"):
        CAP.number(matrix[key])
    if matrix["mem_fraction_static"] >= 1 or matrix["plateau_threshold"] >= 1:
        raise ValueError("Invalid matrix fraction")
    exact_keys(matrix["profiles"], ("baseline", "capacity"), "profiles")
    for profile in matrix["profiles"].values():
        exact_keys(profile, ("cpu_capacity_ratio", "capacity_ratios"), "profile")
        positive_int(profile["cpu_capacity_ratio"])
        positive_list(profile["capacity_ratios"])
        if max(profile["capacity_ratios"]) > profile["cpu_capacity_ratio"]:
            raise ValueError("Capacity scan exceeds the fixed CPU address budget")
    base = matrix["profiles"]["baseline"]
    if base["capacity_ratios"] != [base["cpu_capacity_ratio"]]:
        raise ValueError("Baseline must match CPU/MPU requested capacity")
    if (
        len(matrix["profiles"]["capacity"]["capacity_ratios"])
        <= matrix["plateau_consecutive_edges"]
    ):
        raise ValueError("Capacity scan is too short for the plateau rule")
    exact_keys(
        matrix["families"], ("qwen", "deepseek-v4-flash", "glm-5.3-flash"), "families"
    )
    family = matrix["families"][stage["family"]]
    exact_keys(
        family,
        (
            "selector",
            "top_k",
            "device_buffer_size",
            "max_total_tokens",
            "chunked_prefill_size",
            "concurrencies",
            "num_prompts",
            "warmup_requests",
            "correctness_input_lens",
            "correctness_output_len",
        ),
        "family",
    )
    for key in ("concurrencies", "correctness_input_lens"):
        positive_list(family[key])
    for key in set(family) - {"selector", "concurrencies", "correctness_input_lens"}:
        positive_int(family[key])
    if family["selector"] != stage["selector"]:
        raise ValueError("Matrix selector differs from the current stage")
    maximum = max(family["concurrencies"])
    if (
        family["num_prompts"] < max(256, 8 * maximum)
        or family["warmup_requests"] < maximum
    ):
        raise ValueError(
            "Require 256 requests, eight concurrency waves and one warmup wave"
        )
    if family["top_k"] > family["device_buffer_size"]:
        raise ValueError("Selector budget exceeds hot buffer")
    if max(family["correctness_input_lens"]) < max(matrix["input_lens"]):
        raise ValueError("Correctness must cover the longest performance context")
    expected = (1, 1, 1) if stage["family"] == "qwen" else (4, 4, 4)
    if (
        stage["tp_size"],
        stage.get("attn_dp_size", 1),
        stage.get("ep_size", 1),
    ) != expected:
        raise ValueError("Matrix requires Qwen single rank or Attention-DP4 + EP4")
    # Costs are totals for a fixed machine. Changing its KV allocation does not
    # change its rental/purchase price. Hypothetical MPU pricing is separate.
    if policy.get("cost_basis") == "simulator_hardware":
        for variant in ("cpu", "mpu"):
            if (
                len(
                    {
                        p["hourly_cost"]
                        for p in policy["costs"]
                        if p["variant"] == variant
                    }
                )
                > 1
            ):
                raise ValueError(
                    "Fixed simulator topology must have one total cost per variant"
                )
    return family


def options(mapping):
    result = []
    for key, value in mapping.items():
        if value is not None:
            result.append("--" + key.replace("_", "-"))
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def make_plan(matrix, stage, policy, args):
    family = validate(matrix, stage, policy)
    is_mpu = stage["variant"] == "mpu"
    if args.topology == "qwen-1plus4" and (stage["family"] != "qwen" or not is_mpu):
        raise ValueError("Qwen 1+4 requires the Qwen MPU branch (stage 02)")
    storage = (
        (
            [1, 2, 3, 4]
            if args.topology == "qwen-1plus4"
            else stage["variants"]["mpu"]["hisparse_config"]["mpu_device_ids"]
        )
        if is_mpu
        else []
    )
    profiles = args.profiles or (["baseline", "capacity"] if is_mpu else ["baseline"])
    DECISION.verify_scope(policy, stage, storage)
    if len(set(profiles)) != len(profiles) or (not is_mpu and "capacity" in profiles):
        raise ValueError(
            "Capacity scanning belongs to MPU stages; profiles cannot repeat"
        )
    jobs = []
    for name in profiles:
        profile = matrix["profiles"][name]
        root = args.output_dir / name
        shared = {
            "stage": args.stage,
            "model": args.model or stage["default_model"],
            "revision": stage.get("default_revision"),
            "port": args.port,
            "max_total_tokens": family["max_total_tokens"],
            "chunked_prefill_size": family["chunked_prefill_size"],
            "top_k": family["top_k"],
            "device_buffer_size": family["device_buffer_size"],
            "mem_fraction_static": matrix["mem_fraction_static"],
            "seed": matrix["seed"],
            "repeats": matrix["repeats"],
            "num_prompts": family["num_prompts"],
            "warmup_requests": family["warmup_requests"],
            "concurrencies": family["concurrencies"],
            "logprob_atol": matrix["logprob_atol"],
            "plateau_threshold": matrix["plateau_threshold"],
            "analysis_policy": args.output_dir / "policy.json",
            **profile,
        }
        if storage:
            shared["mpu_device_ids"] = storage
        correct = options(
            {
                **shared,
                "input_lens": family["correctness_input_lens"],
                "output_len": family["correctness_output_len"],
                "output_dir": root / "gates",
            }
        )
        perf = options(
            {
                **shared,
                "input_lens": matrix["input_lens"],
                "output_len": matrix["output_len"],
                "output_dir": root / "performance",
            }
        )
        # Validate the exact delegated arguments without CUDA or side effects.
        _, cargs = STAGE.parse_args(
            correct + ["--lifecycle", "--include-hbm"], correctness=True
        )
        cplan = CORRECT.make_plan(stage, cargs)
        _, pargs = STAGE.parse_args(perf)
        pplan = RUN.make_plan(stage, pargs)
        gate_command = [
            sys.executable,
            str(HERE / "acceptance.py"),
            "--through",
            "correctness",
            *correct,
        ]
        if args.model_manifest:
            gate_command += ["--model-manifest", str(args.model_manifest)]
        jobs.append(
            {
                "id": name,
                "status": "planned",
                "profile": profile,
                "correctness_groups": cplan,
                "performance_groups": pplan,
                "benchmark_cases": sum(len(g["cases"]) for g in pplan),
                "gate_command": gate_command + ["--execute"],
                "performance_command": [
                    sys.executable,
                    str(HERE / "paired_run.py"),
                    *perf,
                    *(
                        ["--model-manifest", str(args.model_manifest)]
                        if args.model_manifest
                        else []
                    ),
                    "--execute",
                ],
                "gate_report": str(root / "gates/acceptance.json"),
                "performance_report": str(root / "performance/manifest.json"),
                "performance_settings": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(pargs).items()
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "experiment_matrix",
        "status": "planned",
        "matrix": matrix,
        "stage": stage,
        "analysis_policy": policy,
        "topology": {
            "compute_gpus": stage["tp_size"],
            "storage_device_ids": storage,
            "storage_gpus": len(storage),
            "parallel_strategy": "single_rank"
            if stage["tp_size"] == 1
            else "attention_dp4_ep4",
        },
        "jobs": jobs,
        "claim": "Planned commands are not GPU/model/performance evidence. +0 uses CPU DDR backing.",
        "run_order": "Seeded cyclic counterbalance across CPU/MPU variants per input length and repeat; shared shuffled concurrency order. Resume retries incomplete paired blocks in full.",
        "decision_inputs_pending": [
            k
            for k, ready in (
                ("p99_ttft_ms", "p99_ttft_ms" in policy["slo"]),
                ("p99_tpot_ms", "p99_tpot_ms" in policy["slo"]),
                ("total_configuration_costs", bool(policy["costs"])),
            )
            if not ready
        ],
    }


def decisions(summary, consecutive_edges):
    """A plateau candidate is conditional on SLO and never an economic optimum."""
    candidates, chains = [], {}
    for row in summary["rows"]:
        if row["variant"] != "mpu":
            continue
        key = row["input_len"], row["concurrency"]
        chain = chains.setdefault(key, [])
        if row.get("decode_plateau_evidence") == "candidate_below_threshold":
            chain.append(row)
        else:
            chain.clear()
    for key, chain in chains.items():
        if len(chain) >= consecutive_edges:
            row = chain[-1]
            first = chain[0]["previous_ratio"]
            points = [
                r
                for r in summary["rows"]
                if r["variant"] == "mpu"
                and (r["input_len"], r["concurrency"]) == key
                and first <= r["ratio"] <= row["ratio"]
            ]
            candidates.append(
                {
                    "input_len": key[0],
                    "concurrency": key[1],
                    "candidate_requested_ratio": first,
                    "tested_through_requested_ratio": row["ratio"],
                    "actual_byte_ratio_by_point": [
                        {
                            "requested_ratio": r["ratio"],
                            "per_rank": r.get(
                                "backing_to_compute_kv_byte_ratios_by_rank"
                            ),
                        }
                        for r in points
                    ],
                    "latency_eligible": all(
                        r.get("slo", {}).get("status") == "passed" for r in points
                    ),
                    "economic_optimum_proven": False,
                }
            )
    return {
        "plateau_candidates": candidates,
        "best_observed_cost_points": summary["best_observed_cost_points"],
        "scope": "One profile and one physical topology. No cross-topology plateau or global economic optimum.",
    }


def verify_gate(report, stage, job):
    if (
        report.get("status") != "passed"
        or report.get("stage") != stage
        or report.get("requested_through") != "correctness"
    ):
        raise ValueError("Correctness acceptance did not pass the requested stage")
    expected = {
        "portable",
        "compile",
        "jit",
        "imports",
        "device",
        "kernels",
        "model_identity",
        "correctness",
    }
    steps = {s["id"]: s for s in report.get("steps", []) if s["status"] == "passed"}
    if not expected <= steps.keys():
        raise ValueError("Missing required pre-performance gates")
    correctness = read_json(
        Path(job["gate_report"]).parent / "correctness/manifest.json"
    )
    result = ACCEPT.external_result(
        {"kind": "model_run", "id": "correctness", "expected_model_stage": stage},
        {"report": str(Path(job["gate_report"]).parent / "correctness/manifest.json")},
        0,
    )
    if result["status"] != "passed":
        raise ValueError("Invalid model/lifecycle evidence")
    actual = [(g["variant"], g["ratio"], g["eager"]) for g in correctness["groups"]]
    planned = [
        (g["variant"], g["ratio"], g["eager"]) for g in job["correctness_groups"]
    ]
    if actual != planned:
        raise ValueError("Model gate omitted a required capacity or eager/graph mode")
    for actual, planned in zip(correctness["groups"], job["correctness_groups"]):
        if (
            actual.get("rounds") != planned["rounds"]
            or actual.get("server_command") != planned["server_command"]
        ):
            raise ValueError(
                "Model gate used a different workload, storage map or budget"
            )
    # The correctness runner replaces its concurrency limit with the request
    # count of the three lifecycle rounds. Other scientific settings are fixed.
    for key in (
        "model",
        "revision",
        "logprob_atol",
        "max_total_tokens",
        "chunked_prefill_size",
        "top_k",
        "device_buffer_size",
        "cpu_capacity_ratio",
        "capacity_ratios",
        "mpu_device_ids",
    ):
        if correctness.get("settings", {}).get(key) != job["performance_settings"][key]:
            raise ValueError(f"Model gate settings drifted: {key}")


def verify_performance(report, stage, job):
    if (
        report.get("status") != "success"
        or report.get("stage") != stage
        or report.get("kind") != "benchmark"
    ):
        raise ValueError("Performance runner failed or used another stage")
    for key, value in job["performance_settings"].items():
        if key != "execute" and report.get("settings", {}).get(key) != value:
            raise ValueError(f"Performance settings drifted: {key}")
    if report.get("groups") is None or len(report["groups"]) != len(
        job["performance_groups"]
    ):
        raise ValueError("Incomplete performance groups")
    references = {}
    _, args, _ = PAIRED.parse_args(job["performance_command"][2:])
    paired = report.get("paired_execution")
    if paired:
        PAIRED.validate_schedule(
            report, PAIRED.schedule(RUN.make_plan(stage, args), args.repeats, args.seed)
        )
        if PAIRED.assemble(stage, args, report) != report["groups"]:
            raise ValueError("Performance samples disagree with sealed paired blocks")
    for actual, planned in zip(report["groups"], job["performance_groups"]):
        if actual.get("decode_cuda_graph_enabled") is not True:
            raise ValueError(
                "Performance requires observed decode CUDA graph execution mode"
            )
        for key in ("id", "variant", "ratio", "input_len", "server_command"):
            if actual.get(key) != planned[key]:
                raise ValueError(f"Performance group differs: {key}")
        if actual.get("status") != "success" or len(actual["cases"]) != len(
            planned["cases"]
        ):
            raise ValueError("Incomplete performance repeats")
        capacity = RUN.verify_capacities(
            actual["capacity"], stage, args, actual["variant"], actual["ratio"]
        )
        key = actual["input_len"]
        references[key] = RUN.enforce_fixed_compute(references.get(key), capacity)
        for case, planned_case in zip(actual["cases"], planned["cases"]):
            if case.get("status") != "success" or any(
                case.get(k) != planned_case[k]
                for k in (
                    ("id", "repeat", "concurrency")
                    if paired
                    else ("id", "repeat", "concurrency", "command")
                )
            ):
                raise ValueError("Performance repeat differs from plan or failed")
            idle = RUN.verify_capacities(
                case["idle_capacity"], stage, args, actual["variant"], actual["ratio"]
            )
            RUN.enforce_fixed_compute(references[key], idle)
            if any(
                p.get("used_slots") != 0 or p.get("reserved_slots") != 0 for p in idle
            ):
                raise ValueError("Performance requests leaked backing slots")


def execute(manifest, args, runner=ACCEPT.run_command):
    path = args.output_dir / "matrix.json"
    if sys.platform != "linux":
        manifest.update(
            status="blocked",
            reason="Execution requires Linux CUDA; planning alone is not GPU evidence",
        )
        RUN.COMMON.write_manifest(path, manifest)
        return False
    try:
        identity = CHECK.identity(
            RUN.COMMON, ACCEPT, manifest["stage"], args, args.model_manifest
        )
        if manifest.get("execution_identity") is not None:
            CHECK.same_identity(manifest["execution_identity"], identity)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        manifest.update(status="blocked", reason=str(error))
        RUN.COMMON.write_manifest(path, manifest)
        return False
    manifest.pop("reason", None)
    manifest["execution_identity"] = identity
    manifest["execution_attempt"] = manifest.get("execution_attempt", 0) + 1
    stopping = False
    for job in manifest["jobs"]:
        if stopping:
            job["status"] = "not_run_after_failure"
            continue
        try:
            job["status"] = "running"
            RUN.COMMON.write_manifest(path, manifest)
            for phase, command_key, report_key, verifier in (
                ("gates", "gate_command", "gate_report", verify_gate),
                (
                    "performance",
                    "performance_command",
                    "performance_report",
                    verify_performance,
                ),
            ):
                command = job[command_key].copy()
                if phase == "gates" and job.get("gates_receipt"):
                    CHECK.verify(Path(job[report_key]).parent, job["gates_receipt"])
                    result = {"returncode": 0, "reused_sealed_gate": True}
                else:
                    if phase == "gates" and Path(job[report_key]).parent.exists():
                        new_dir = (
                            Path(job[report_key]).parent.parent
                            / f"gates-attempt-{manifest['execution_attempt']:04d}"
                        )
                        command[command.index("--output-dir") + 1] = str(new_dir)
                        job[command_key] = command.copy()
                        job[report_key] = str(new_dir / "acceptance.json")
                    if phase == "performance" and Path(job[report_key]).exists():
                        command.append("--resume")
                    result = runner(
                        command,
                        RUN.COMMON.subprocess_env(),
                        args.output_dir
                        / f"{job['id']}-{phase}-attempt-{manifest['execution_attempt']:04d}.log",
                        args.job_timeout,
                    )
                job[phase + "_process"] = result
                if result["returncode"]:
                    raise ValueError(f"{phase} process failed; later work is blocked")
                report = read_json(job[report_key])
                verifier(report, manifest["stage"], job)
                job[phase + "_sha256"] = digest(job[report_key])
                if phase == "gates":
                    job["gates_receipt"] = CHECK.seal(Path(job[report_key]).parent)
                    RUN.COMMON.write_manifest(path, manifest)
            summary = RUN.summarize(
                report["groups"],
                manifest["matrix"]["plateau_threshold"],
                manifest["analysis_policy"],
            )
            job["decisions"] = decisions(
                summary, manifest["matrix"]["plateau_consecutive_edges"]
            )
            RUN.COMMON.write_manifest(
                args.output_dir / f"{job['id']}-analysis.json", summary
            )
            job["status"] = "measured"
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as error:
            job.update(status="failed", reason=str(error))
            stopping = True
        RUN.COMMON.write_manifest(path, manifest)
    manifest["status"] = "failed" if stopping else "measured"
    manifest["claim"] = (
        "Measurement completion is separate from SLO, plateau and cost decisions; see each profile's evidence."
    )
    RUN.COMMON.write_manifest(path, manifest)
    return not stopping


def validate_resume(manifest, expected, args):
    """Check the saved plan again under the lock before executing any command."""
    for key in ("matrix", "stage", "analysis_policy", "topology"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Resume plan changed: {key}")
    if (
        read_json(args.stage) != expected["stage"]
        or read_json(args.output_dir / "policy.json") != expected["analysis_policy"]
    ):
        raise ValueError("Saved stage/policy snapshots changed")
    if len(manifest["jobs"]) != len(expected["jobs"]):
        raise ValueError("Resume profiles changed")
    for actual, planned in zip(manifest["jobs"], expected["jobs"]):
        for key in (
            "id",
            "profile",
            "correctness_groups",
            "performance_groups",
            "performance_command",
            "performance_settings",
            "performance_report",
        ):
            if actual.get(key) != planned[key]:
                raise ValueError(f"Resume workload or command changed: {key}")
        command = actual["gate_command"].copy()
        index = command.index("--output-dir") + 1
        directory = Path(command[index])
        root = Path(planned["gate_report"]).parent.parent
        allowed = {root / "gates"} | {
            root / f"gates-attempt-{n:04d}"
            for n in range(1, manifest.get("execution_attempt", 0) + 1)
        }
        if directory not in allowed or actual["gate_report"] != str(
            directory / "acceptance.json"
        ):
            raise ValueError("Resume gate evidence path changed")
        command[index] = planned["gate_command"][index]
        if command != planned["gate_command"]:
            raise ValueError("Resume gate command changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=HERE / "experiment-matrix.json")
    parser.add_argument("--policy", type=Path, default=HERE / "experiment-policy.json")
    parser.add_argument("--stage", type=Path, default=HERE / "stage.json")
    parser.add_argument(
        "--topology", choices=("default", "qwen-1plus4"), default="default"
    )
    parser.add_argument("--profiles", choices=("baseline", "capacity"), nargs="+")
    parser.add_argument("--model")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--job-timeout", type=int, default=604800)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for key in ("stage", "matrix", "policy", "output_dir", "model_manifest"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.job_timeout <= 0:
        raise ValueError("Job timeout must be positive")
    if args.model and Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())
    matrix, stage, policy = (
        read_json(args.matrix),
        STAGE.load_stage(args.stage),
        CAP.load_policy(args.policy),
    )
    args.model = args.model or stage["default_model"]
    make_plan(matrix, stage, policy, args)  # Reject bad inputs before creating output.
    inputs = {
        key: {"path": str(getattr(args, key)), "sha256": digest(getattr(args, key))}
        for key in ("stage", "matrix", "policy")
    }
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    args.stage = args.output_dir / "stage.json"
    if args.resume:
        expected = make_plan(matrix, stage, policy, args)
        manifest = read_json(args.output_dir / "matrix.json")
        validate_resume(manifest, expected, args)
    else:
        RUN.COMMON.write_manifest(args.stage, stage)
        RUN.COMMON.write_manifest(args.output_dir / "policy.json", policy)
        manifest = make_plan(matrix, stage, policy, args)
        manifest["inputs"] = inputs
        manifest["runtime"] = RUN.COMMON.runtime_info()
        RUN.COMMON.write_manifest(
            args.output_dir / "decision-inputs.json", DECISION.template(manifest)
        )
        RUN.COMMON.write_manifest(args.output_dir / "matrix.json", manifest)
    previous = None
    if sys.platform == "linux":

        def interrupt(signum, frame):
            raise KeyboardInterrupt("Matrix terminated")

        previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.execute:
            with CHECK.exclusive(args.output_dir):
                if args.resume:
                    manifest = read_json(args.output_dir / "matrix.json")
                    validate_resume(manifest, expected, args)
                passed = execute(manifest, args)
        else:
            passed = True
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    print(args.output_dir / "matrix.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
