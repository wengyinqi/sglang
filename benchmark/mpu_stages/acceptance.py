"""Plan or execute ordered acceptance using the existing six-stage tools.

A successful requested scope is not a claim that later hardware/model scopes ran.
Run --through portable --execute on the development host, or --through compile
inside the matching Linux CUDA development image before GPUs arrive.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASES = (
    "portable",
    "compile",
    "jit",
    "imports",
    "device",
    "kernels",
    "correctness",
    "performance",
)


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"acceptance_{name}", HERE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE, COMMON = load("stage"), load("common")


def portable_targets(stage):
    """Explicit CPU coverage; optional GPU tests are separate mandatory gates."""
    targets = [
        "test/manual/mpu_stages/test_stage_tools.py",
        "test/manual/mpu_stages/test_acceptance.py",
        "test/manual/mpu_stages/test_environment_packaging.py",
        "test/manual/mpu_stages/test_native_environment.py",
        "test/manual/mpu_stages/test_capacity_analysis.py",
        "test/manual/mpu_stages/test_experiment_matrix.py",
        "test/manual/mpu_stages/test_paired_run.py",
        "test/manual/mpu_stages/test_decision_policy.py",
        "test/manual/mpu_stages/test_profile_tools.py",
        "test/manual/mpu_stages/test_model_snapshot.py",
        "test/manual/mpu_stages/test_attention_dp_ep.py",
        "test/manual/mpu_stages/test_hardware_probe.py",
        "test/manual/mpu_stages/test_jit_matrix.py",
        "test/manual/mpu_stages/test_lifecycle_cpu.py",
        "test/manual/mpu_stages/test_lifecycle_tools.py",
        "test/manual/mpu_stages/test_delivery_bundle.py",
    ]
    if sys.platform == "linux":
        targets.append("test/manual/mpu_stages/test_cpu_processes_linux.py")
    if stage["family"] == "qwen":
        targets += [
            "test/manual/hisparse/test_quest_selector.py::TestQuestSelectorCPU",
            "test/manual/hisparse/test_quest_backend_contracts.py",
            "test/manual/hisparse/test_quest_lifecycle_ordering.py",
            "test/manual/hisparse/test_benchmark_stream_accounting.py",
            "test/manual/hisparse/test_mpu_timing.py::TestDecodeOnlyAccounting",
            "test/manual/test_hisparse_qwen_benchmark.py",
        ]
        targets += [
            f"test/manual/hisparse/test_mpu_timing.py::TestForwardTrace::{name}"
            for name in (
                "test_hot_path_never_syncs_or_writes_events",
                "test_failed_forward_is_recorded_and_exception_propagates",
                "test_completed_events_remain_ordered_across_idle_waves",
                "test_disabled_trace_does_not_require_cuda_or_create_files",
            )
        ]
        targets += [
            f"test/manual/mpu/test_mpu_{name}.py"
            for name in (
                "config",
                "coordinator",
                "experiment",
                "kernel_contracts",
                "pool",
            )
        ]
    elif stage["family"] == "deepseek-v4-flash":
        targets += [
            "test/manual/mpu/test_dsv4_cpu.py",
            "test/manual/mpu_stages/test_native_benchmark_stream.py",
        ]
    elif stage["family"] == "glm-5.3-flash":
        targets += [
            f"test/manual/mpu_stages/{name}.py"
            for name in (
                "test_glm53_cpu",
                "test_glm53_mpu",
                "test_native_benchmark_stream",
            )
        ]
    else:
        raise ValueError(f"Unsupported stage family: {stage['family']}")
    return targets


def visible_tokens(value):
    if value is None:
        return None
    devices = [item.strip() for item in value.split(",")]
    if (
        not devices
        or any(not item or item == "-1" for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain distinct enabled CUDA ordinals or UUIDs"
        )
    return devices


def remap_ordinals(ordinals, visibility):
    if visibility is None:
        return ",".join(map(str, ordinals))
    if max(ordinals) >= len(visibility):
        raise ValueError("Requested test ordinal is outside CUDA_VISIBLE_DEVICES")
    return ",".join(visibility[index] for index in ordinals)


def kernel_jobs(stage, args, visibility):
    """Run pair diagnostics, real GLM resolve, then simultaneous MPU data paths."""
    jobs = []
    if stage["family"] == "qwen":
        jobs.append(
            {
                "name": "quest-cuda",
                "targets": [
                    "test/manual/hisparse/test_quest_selector.py::TestQuestSelectorCUDA",
                    "test/manual/hisparse/test_mpu_timing.py::TestForwardTrace::test_real_cuda_stream_eager_and_graph_replay",
                ],
                "env": {"CUDA_VISIBLE_DEVICES": remap_ordinals([0], visibility)},
            }
        )
        if stage["variant"] == "mpu":
            config = STAGE.config_for(stage, args, "mpu", args.capacity_ratios[0])
            storage = config["mpu_device_ids"]
            jobs.append(
                {
                    "name": "mpu-shards",
                    "targets": [
                        "test/manual/mpu/test_mpu_pool_gpu.py",
                        "test/manual/mpu/test_mpu_resolve_gpu.py",
                    ],
                    "env": {
                        "CUDA_VISIBLE_DEVICES": remap_ordinals(
                            [0, *storage], visibility
                        ),
                        "SGLANG_RUN_MPU_GPU_TESTS": "1",
                        "SGLANG_MPU_TEST_COMPUTE_DEVICE": "0",
                        "SGLANG_MPU_TEST_STORAGE_DEVICES": ",".join(
                            map(str, range(1, len(storage) + 1))
                        ),
                    },
                }
            )
        return jobs
    # Select CUDA-supported tests explicitly. Several ROCm-only cases do not
    # contain "rocm" in their names, so a broad -k filter would leave mandatory
    # skips and block a healthy CUDA installation.
    cuda_cases = (
        "test_transfer_cache_dsv4_mla_copies_paged_token",
        "test_dsv4_swap_in_reads_paged_host_layout",
        "test_load_cache_to_device_buffer_fast_path",
        "test_load_cache_to_device_buffer_fast_path_overwrites_stale_output",
        "test_load_cache_to_device_buffer_hits_newest_and_updates_lru",
        "test_load_blocks_to_device_buffer_mha_handles_partial_newest_block",
        "test_load_cache_to_device_buffer_miss_uses_updated_lru_slot",
        "test_load_cache_to_device_buffer_multiple_misses_copy_all_slots",
        "test_load_cache_to_device_buffer_batched_with_padding",
        "test_load_cache_to_device_buffer_dsv4_mla_miss_copy_layout",
    )
    jobs.append(
        {
            "name": "hisparse-cuda",
            "targets": [
                f"test/registered/kernels/ops/kvcache/test_hisparse.py::{name}"
                for name in cuda_cases
            ],
            "env": {"CUDA_VISIBLE_DEVICES": remap_ordinals([0], visibility)},
        }
    )
    if stage["family"] == "glm-5.3-flash":
        jobs[-1]["targets"].append(
            "test/registered/kernels/ops/attention/test_dsa_kpool_topk_transform.py"
        )
    if stage["variant"] == "mpu":
        storage = STAGE.config_for(stage, args, "mpu", args.capacity_ratios[0])[
            "mpu_device_ids"
        ]
        for rank in range(4):
            if stage["family"] == "deepseek-v4-flash":
                mapping = [rank, storage[rank]]
                targets = ["test/manual/mpu/test_dsv4_gpu.py"]
                extra = {}
            else:
                order = [rank, *[index for index in range(4) if index != rank]]
                mapping = order + [storage[index] for index in order]
                targets = [
                    "test/manual/mpu_stages/test_glm53_gpu.py",
                    "test/manual/mpu_stages/test_glm53_resolve_gpu.py",
                ]
                extra = {"SGLANG_TEST_GLM53_MPU": "1"}
            jobs.append(
                {
                    "name": f"mpu-pair-{rank}",
                    "targets": targets,
                    "env": {
                        "CUDA_VISIBLE_DEVICES": remap_ordinals(mapping, visibility),
                        **extra,
                    },
                    "compute_ordinal": rank,
                    "storage_ordinal": storage[rank],
                }
            )
        jobs.append(
            {
                "name": "mpu-concurrent",
                "targets": ["test/manual/mpu_stages/test_mpu_concurrent_gpu.py"],
                "env": {
                    "CUDA_VISIBLE_DEVICES": remap_ordinals(
                        [0, 1, 2, 3, *storage], visibility
                    ),
                    "SGLANG_TEST_MPU_CONCURRENT": "1",
                    "SGLANG_MPU_TEST_FAMILY": stage["family"],
                    "SGLANG_MPU_CONCURRENT_REPORT": str(
                        args.output_dir.resolve() / "concurrent-ranks.json"
                    ),
                },
            }
        )
    return jobs


def pytest_command(targets, report):
    return [sys.executable, "-m", "pytest", "-q", *targets, f"--junitxml={report}"]


def junit_result(path, returncode):
    """Skipped/empty suites cannot pass a required acceptance layer."""
    try:
        root = ET.parse(path).getroot()
        if root.tag not in ("testsuite", "testsuites"):
            raise ValueError("Not a JUnit test suite")
        cases = list(root.iter("testcase"))
        errors, failures, skipped = (
            len(list(root.iter(tag))) for tag in ("error", "failure", "skipped")
        )
        result = {
            "tests": len(cases),
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
        }
        if returncode or errors or failures:
            return {"status": "failed", **result}
        if not cases or skipped:
            return {
                "status": "blocked",
                "reason": "Required suite is empty or contains skipped tests",
                **result,
            }
        return {"status": "passed", **result}
    except (OSError, ET.ParseError, ValueError) as error:
        return {
            "status": "failed",
            "reason": f"Missing or invalid JUnit report: {error}",
        }


def forwarded_args(argv, output_dir, *, correctness=False, model=None):
    clean = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in ("--output-dir", "--stage") or (
            model is not None and item == "--model"
        ):
            if index + 1 >= len(argv):
                raise ValueError(f"Missing value for {item}")
            index += 2
            continue
        if (
            item.startswith(("--output-dir=", "--stage="))
            or (model is not None and item.startswith("--model="))
            or item == "--execute"
        ):
            index += 1
            continue
        clean.append(item)
        index += 1
    if correctness and "--include-hbm" not in clean:
        clean.append("--include-hbm")
    if correctness and "--lifecycle" not in clean:
        clean.append("--lifecycle")
    if model is not None:
        clean += ["--model", model]
    return [
        *clean,
        "--stage",
        str(output_dir / "stage.json"),
        "--output-dir",
        str(output_dir / ("correctness" if correctness else "performance")),
        "--execute",
    ]


def make_steps(stage, args, control, stage_argv):
    root = args.output_dir
    steps = [
        {
            "id": "portable",
            "phase": "portable",
            "kind": "pytest",
            "commands": [
                {
                    "name": "cpu-tests",
                    "command": pytest_command(
                        portable_targets(stage), root / "portable.xml"
                    ),
                    "report": str(root / "portable.xml"),
                    "env": {"MPU_CPU_CONTROL_EVIDENCE": str(root / "cpu-processes")}
                    if sys.platform == "linux"
                    else {},
                }
            ],
        }
    ]
    for mode in ("compile", "jit", "imports", "device"):
        report = root / f"{mode}.json"
        command = [
            sys.executable,
            str(HERE / "hardware_probe.py"),
            "--stage",
            str(root / "stage.json"),
            "--mode",
            mode,
            "--output",
            str(report),
            "--timeout",
            str(control.probe_timeout),
        ]
        if mode == "compile":
            command.append("--compile-only")
        if mode == "jit":
            command += ["--jit-matrix", getattr(control, "jit_matrix", "default")]
        if args.mpu_device_ids is not None:
            command += ["--mpu-device-ids", *map(str, args.mpu_device_ids)]
        if args.mpu_memory_reserve_bytes is not None:
            command += [
                "--mpu-memory-reserve-bytes",
                str(args.mpu_memory_reserve_bytes),
            ]
        steps.append(
            {
                "id": mode,
                "phase": mode,
                "kind": "probe",
                "expected_stage": {
                    **{
                        key: stage[key]
                        for key in ("id", "family", "tp_size", "variant")
                    },
                    "attn_dp_size": stage.get("attn_dp_size", 1),
                    "ep_size": stage.get("ep_size", 1),
                },
                "expected_jit_cases": (
                    load("jit_matrix").cases_for(
                        stage,
                        load("hardware_probe").FAMILIES[stage["family"]],
                        getattr(control, "jit_matrix", "default"),
                    )
                    if mode == "jit"
                    else None
                ),
                "commands": [
                    {"name": mode, "command": command, "report": str(report), "env": {}}
                ],
            }
        )
    jobs, mapping_error = [], None
    if PHASES.index(control.through) >= PHASES.index("kernels"):
        try:
            jobs = kernel_jobs(
                stage, args, visible_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
            )
        except ValueError as error:
            # A plan can describe unmet hardware requirements. It must not stop
            # an earlier, explicitly requested portable/compile-only scope.
            mapping_error = str(error)
    steps.append(
        {
            "id": "kernels",
            "phase": "kernels",
            "kind": "pytest",
            "planning_blocker": mapping_error,
            "commands": [
                {
                    **job,
                    "command": pytest_command(
                        job["targets"], root / f"kernel-{job['name']}.xml"
                    ),
                    "report": str(root / f"kernel-{job['name']}.xml"),
                }
                for job in jobs
            ],
        }
    )
    steps.append(
        {
            "id": "model_identity",
            "phase": "correctness",
            "kind": "model_identity",
            "commands": [],
        }
    )
    for kind in ("correctness", "performance"):
        steps.append(
            {
                "id": kind,
                "phase": kind,
                "kind": "model_run",
                "expected_model_stage": stage,
                "commands": [
                    {
                        "name": kind,
                        "command": [
                            sys.executable,
                            str(
                                HERE
                                / (
                                    "correctness.py"
                                    if kind == "correctness"
                                    else "run.py"
                                )
                            ),
                            *forwarded_args(
                                stage_argv,
                                root,
                                correctness=kind == "correctness",
                                model=args.model,
                            ),
                        ],
                        "report": str(root / kind / "manifest.json"),
                        "env": {},
                    }
                ],
            }
        )
    through = PHASES.index(control.through)
    for step in steps:
        step["status"] = (
            "planned" if PHASES.index(step["phase"]) <= through else "not_requested"
        )
    return steps


def run_command(command, env, log, timeout):
    started = time.monotonic()
    with Path(log).open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=sys.platform == "linux",
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if sys.platform == "linux":
                try:
                    # The probe owns isolated child sessions. Let its TERM
                    # handler reap them before resorting to an outer SIGKILL.
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise
    return {
        "returncode": code,
        "duration_seconds": time.monotonic() - started,
        "log": str(log),
    }


def external_result(step, job, returncode):
    if step["kind"] == "pytest":
        return junit_result(job["report"], returncode)
    try:
        report = json.loads(Path(job["report"]).read_text(encoding="utf-8"))
        status = report.get("status")
        if step["kind"] == "probe":
            if (
                report.get("schema_version") != 1
                or report.get("mode") != step["id"]
                or status not in ("passed", "failed", "blocked")
            ):
                raise ValueError("Probe reported another mode or an unknown status")
            if status == "passed":
                checks = report.get("checks")
                if (
                    returncode
                    or not isinstance(checks, list)
                    or not checks
                    or any(check.get("status") != "passed" for check in checks)
                ):
                    raise ValueError(
                        "Probe exit/check evidence disagrees with passed status"
                    )
                if (
                    step.get("expected_stage") is not None
                    and report.get("stage") != step["expected_stage"]
                ):
                    raise ValueError("Probe ran a different stage")
                if step["id"] == "compile":
                    objects = [
                        check["evidence"].get("object", {})
                        for check in checks
                        if check.get("name") == "production_object_compile"
                    ]
                    if (
                        report.get("capability_scope") != "object_only"
                        or len(objects) != 1
                    ):
                        raise ValueError(
                            "Compile gate requires production object-only evidence"
                        )
                    item = objects[0]
                    artifact = Path(item.get("path", ""))
                    if (
                        not item.get("exists")
                        or not artifact.is_file()
                        or artifact.stat().st_size != item.get("bytes")
                        or item.get("bytes", 0) <= 0
                    ):
                        raise ValueError("Production CUDA object is missing or changed")
                    digest = hashlib.sha256()
                    with artifact.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != item.get("sha256"):
                        raise ValueError("Production CUDA object checksum changed")
                if step["id"] == "jit":
                    artifacts = load("jit_matrix").validate_report(
                        report, step.get("expected_jit_cases")
                    )
                    for item in artifacts:
                        artifact = Path(item.get("path", ""))
                        if (
                            not artifact.is_file()
                            or artifact.stat().st_size != item.get("bytes")
                            or hashlib.sha256(artifact.read_bytes()).hexdigest()
                            != item.get("sha256")
                        ):
                            raise ValueError(
                                "Production JIT shared library is missing or changed"
                            )
            return {"status": status, "evidence": report}
        groups = report.get("groups", [])
        if (
            step.get("expected_model_stage") is not None
            and report.get("stage") != step["expected_model_stage"]
        ):
            return {
                "status": "failed",
                "reason": "Model report has a different stage/parallelism contract",
            }
        if (
            returncode
            or status != "success"
            or not groups
            or any(group.get("status") != "success" for group in groups)
        ):
            return {
                "status": "failed",
                "reason": "Model runner did not finish all required groups",
                "runner_status": status,
            }
        if step.get("id") == "correctness":
            required = {"abort_prefill", "abort_decode", "retract_decode"}
            for group in groups:
                accepted = {
                    item.get("case")
                    for item in group.get("lifecycle_results", [])
                    if item.get("status") == "passed"
                }
                if (
                    group.get("lifecycle_required") is not True
                    or not required <= accepted
                ):
                    return {
                        "status": "failed",
                        "reason": "Missing required abort/retract recovery evidence",
                    }
                if report.get("stage", {}).get("attn_dp_size", 1) > 1:
                    owner_rounds = group.get("round_attention_owners", [])
                    if len(owner_rounds) != 3 or any(
                        set(row.values()) != set(range(4)) for row in owner_rounds
                    ):
                        return {
                            "status": "failed",
                            "reason": "Missing correctness request coverage across all Attention-DP owners",
                        }
                    for rank in range(4):
                        covered = {
                            item.get("case")
                            for item in group.get("lifecycle_results", [])
                            if item.get("status") == "passed"
                            and item.get("attention_dp_rank") == rank
                        }
                        if not required <= covered:
                            return {
                                "status": "failed",
                                "reason": f"Missing lifecycle coverage for Attention-DP rank {rank}",
                            }
        return {"status": "passed", "runner_status": status, "groups": len(groups)}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"status": "failed", "reason": f"Invalid delegated report: {error}"}


def model_identity(stage, args, manifest_path):
    if manifest_path is None or not Path(args.model).is_dir():
        return {
            "status": "blocked",
            "reason": "Target-model acceptance requires --model pointing to a local snapshot and --model-manifest created with --full-checksum",
        }
    result = load("model_snapshot").verify_manifest(
        manifest_path, args.model, stage=stage, require_full_checksum=True
    )
    if not result.get("ok"):
        return {"status": "failed", "evidence": result}
    if not result.get("upstream_identity_verified"):
        return {
            "status": "blocked",
            "reason": "The local fingerprint is intact, but target revision identity requires a matching trusted checksum reference; recreate the manifest with --reference",
            "evidence": result,
        }
    return {"status": "passed", "evidence": result}


def execute(manifest, args, control, runner=run_command):
    path = args.output_dir / "acceptance.json"
    stopping = False
    for step in manifest["steps"]:
        if step["status"] == "not_requested":
            continue
        if stopping:
            step["status"] = "not_run_after_failure"
            continue
        try:
            step["status"] = "running"
            COMMON.write_manifest(path, manifest)
            if step["kind"] == "model_identity":
                step.update(
                    model_identity(manifest["stage"], args, control.model_manifest)
                )
            else:
                step["results"] = []
                for index, job in enumerate(step["commands"]):
                    env = COMMON.subprocess_env()
                    env.update(job.get("env", {}))
                    # Prevent test selection/options inherited from a shell from
                    # silently replacing the mandatory test targets.
                    if step["kind"] == "pytest":
                        env.pop("PYTEST_ADDOPTS", None)
                    timeout = (
                        control.layer_timeout
                        if step["kind"] != "model_run"
                        else control.model_timeout
                    )
                    run = runner(
                        job["command"],
                        env,
                        args.output_dir / f"{step['id']}-{index}.log",
                        timeout,
                    )
                    result = {
                        **run,
                        **external_result(step, job, run["returncode"]),
                        "name": job["name"],
                    }
                    step["results"].append(result)
                    if result["status"] != "passed":
                        break
                step["status"] = (
                    step["results"][-1]["status"] if step["results"] else "blocked"
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            ImportError,
            subprocess.SubprocessError,
        ) as error:
            step.update(status="failed", reason=str(error))
        except KeyboardInterrupt:
            step.update(status="failed", reason="Interrupted; later gates were not run")
        stopping = step["status"] != "passed"
        COMMON.write_manifest(path, manifest)
    selected = [step for step in manifest["steps"] if step["status"] != "not_requested"]
    passed = bool(selected) and all(step["status"] == "passed" for step in selected)
    manifest["status"] = (
        "passed"
        if passed
        else (
            "failed"
            if any(step["status"] == "failed" for step in selected)
            else "blocked"
        )
    )
    completed = {step["id"] for step in manifest["steps"] if step["status"] == "passed"}
    manifest["evidence_scope"] = {
        "requested_scope_passed": passed,
        "linux_cpu_control_executed": sys.platform == "linux"
        and "portable" in completed,
        "cuda_object_compiled": "compile" in completed,
        "production_jit_modules_loaded": "jit" in completed,
        "production_python_imports": "imports" in completed,
        "device_smoke_executed": "device" in completed,
        "selected_kernel_tests_executed": "kernels" in completed,
        "target_model_correctness_executed": "correctness" in completed,
        "performance_measured": "performance" in completed,
    }
    COMMON.write_manifest(path, manifest)
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--through", choices=PHASES, default="correctness")
    parser.add_argument(
        "--jit-matrix", choices=("default", "boundaries"), default="default"
    )
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--probe-timeout", type=int, default=600)
    parser.add_argument("--layer-timeout", type=int, default=3600)
    parser.add_argument("--model-timeout", type=int, default=86400)
    control, forwarded = parser.parse_known_args(argv)
    if any(
        getattr(control, key) <= 0
        for key in ("probe_timeout", "layer_timeout", "model_timeout")
    ):
        raise ValueError("Acceptance timeouts must be positive")
    if "--help" in forwarded or "-h" in forwarded:
        print(__doc__)
        print(parser.format_help())
    stage, args = STAGE.parse_args(forwarded)
    if Path(args.model).is_dir():
        # Children run with cwd=REPO, which can differ from the caller's cwd.
        args.model = str(Path(args.model).resolve())
    steps = make_steps(stage, args, control, forwarded)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    COMMON.write_manifest(args.output_dir / "stage.json", stage)
    manifest = {
        "schema_version": 1,
        "status": "planned",
        "requested_through": control.through,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "runtime": COMMON.runtime_info(),
        "steps": steps,
        "model_manifest": (
            str(control.model_manifest.resolve()) if control.model_manifest else None
        ),
        "scope_notes": [
            "+0 is HiSparse CPU DDR backing; the correctness gate also requires the same-selector HBM reference.",
            "Compile gate builds production MPU objects only; native load and GPU execution are separate evidence.",
            "Required pytest skips block the next layer. Passing a requested subset does not pass later layers.",
            "Four-compute kernel acceptance tests concurrent data paths; it does not certify Attention-DP/DeepEP collectives or measured GPU traffic overlap.",
            "GLM MPU tests use real KPool selection and resolve-to-peer replay; model correctness also requires observed abort/retract recovery.",
            "Economic/SLO acceptance remains separate from successful data-path and model tests.",
            "Target model acceptance requires both full local fingerprints and a matching trusted fixed-revision reference.",
        ],
    }
    path = args.output_dir / "acceptance.json"
    COMMON.write_manifest(path, manifest)
    previous_term = None
    if sys.platform == "linux":

        def interrupt_on_term(signum, frame):
            raise KeyboardInterrupt("Acceptance terminated")

        previous_term = signal.signal(signal.SIGTERM, interrupt_on_term)
    try:
        passed = execute(manifest, args, control) if args.execute else True
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
    print(path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
