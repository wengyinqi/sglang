"""Collect a separate, bounded decode Nsight trace after correctness acceptance.

Uses the existing SGLANG_PROFILE_V2 / CUDA_PROFILER stage trigger. Profiling
results are diagnostics and are never merged into the throughput benchmark.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "mpu_profile_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON, STAGE, RUN = load("common"), load("stage"), load("run")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def correctness_gate(report, stage, runtime, args, model_report, variant=None):
    if (
        report.get("evidence_scope", {}).get("target_model_correctness_executed")
        is not True
        or report.get("stage") != stage
        or report.get("runtime", {}).get("git_commit") != runtime.get("git_commit")
        or model_report.get("runtime", {}).get("git_commit")
        != runtime.get("git_commit")
        or model_report.get("stage") != stage
        or report.get("runtime", {}).get("git_status")
        not in ("", "unavailable: archived source has no Git index")
        or runtime.get("git_status")
        not in ("", "unavailable: archived source has no Git index")
    ):
        raise ValueError(
            "Profiling requires correctness acceptance for this clean source and stage"
        )
    steps = [s for s in report.get("steps", []) if s.get("id") == "correctness"]
    if len(steps) != 1 or steps[0].get("status") != "passed":
        raise ValueError("Missing passed correctness step")
    results = steps[0].get("results", [])
    if len(results) != 1 or results[0].get("status") != "passed":
        raise ValueError("Missing actual correctness result")
    settings = model_report.get("settings", {})
    for name in (
        "model",
        "revision",
        "top_k",
        "device_buffer_size",
        "mpu_device_ids",
        "max_total_tokens",
        "cpu_capacity_ratio",
        "mem_fraction_static",
        "chunked_prefill_size",
        "prefill_max_requests",
        "disable_cuda_graph",
        "mpu_memory_reserve_bytes",
        "workload",
        "workload_sha256",
    ):
        actual = getattr(args, name, None)
        if isinstance(actual, Path):
            actual = str(actual)
        if settings.get(name) != actual:
            raise ValueError(
                f"Profiling configuration differs from correctness acceptance: {name}"
            )
    if args.capacity_ratios[0] not in settings.get("capacity_ratios", []):
        raise ValueError(
            "Profile capacity ratio was not covered by correctness acceptance"
        )
    if variant == "hbm" and not settings.get("include_hbm"):
        raise ValueError("HBM profiling requires the passed HBM correctness reference")


def plan(stage, args, steps, variant=None, concurrency=None):
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError("Profile steps must be between 1 and 512")
    # A further scheduler step closes the upstream profile trigger. Keep the
    # request alive past the trigger, rather than relying on an idle flush.
    output_len = max(args.output_len, steps + 4)
    variant = variant or stage["variant"]
    concurrency = stage.get("attn_dp_size", 1) if concurrency is None else concurrency
    if variant not in stage["variants"] or (
        variant == "hbm" and not stage.get("hbm_preserves_selector")
    ):
        raise ValueError("Profile variant must preserve the accepted sparse selector")
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("Profile concurrency must be positive")
    if stage.get("attn_dp_size", 1) > 1 and concurrency != 4:
        raise ValueError("Multi-rank profiling requires four Attention owners")
    root = args.output_dir
    server_args = copy.copy(args)
    server_args.output_len = output_len
    server_args.concurrencies = [max(max(args.concurrencies), concurrency)]
    server = STAGE.server_command(
        stage,
        server_args,
        variant,
        args.input_lens[0],
        None if variant == "hbm" else args.capacity_ratios[0],
    )
    wrapper = (
        "import os,pathlib,runpy,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "sys.argv=['sglang.launch_server',*sys.argv[2:]]; runpy.run_module('sglang.launch_server',run_name='__main__')"
    )
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        "--wait=all",
        "--output=" + str(root / "decode"),
        server[0],
        "-c",
        wrapper,
        str(root / "server.pid"),
        *server[3:],
    ]
    return {
        "schema_version": 1,
        "status": "planned",
        "stage": stage,
        "profile_variant": variant,
        "server_command": command,
        "environment": {"SGLANG_PROFILE_V2": "1"},
        "profile_request": {
            "activities": ["CUDA_PROFILER"],
            "profile_by_stage": True,
            "profile_stages": ["decode"],
            "num_steps": steps,
            "with_stack": False,
            "record_shapes": False,
            "merge_profiles": False,
        },
        "workload": {
            "input_len": args.input_lens[0],
            "output_len": output_len,
            "concurrency": concurrency,
            "prompt": (
                "frozen real-document input IDs; diagnostic only"
                if getattr(args, "workload", None)
                else "synthetic repeated token 42; diagnostic only"
            ),
            "manifest": str(args.workload) if getattr(args, "workload", None) else None,
            "manifest_sha256": getattr(args, "workload_sha256", None),
        },
        "configured_decode_steps": steps,
        "actual_steps_verified": False,
        "performance_result": False,
        "host_kv_payload_absence_proven": False,
        "claim": "Decode-triggered CUDA timeline only; pending real hardware and trace inspection",
    }


def post_text(port, route, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"Profile control returned HTTP {response.status}")
        return response.read().decode()


def progress(port):
    info = COMMON.request_json(port, "get_server_info", timeout=10)
    states = info.get("internal_states", [])
    if len(states) != 1:
        raise ValueError("Stable decode trigger requires one compute scheduler")
    return states[0].get("mpu_forward_progress")


def wait_stable(port, future, min_steps, batch_size, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if future.done():
            future.result()  # Propagate a failed request instead of accepting stale state.
            raise ValueError("Requests completed before a live stable decode trigger")
        observed = progress(port)
        if (
            isinstance(observed, dict)
            and observed.get("success") is True
            and observed.get("forward_mode") == "decode"
            and observed.get("consecutive_stable_decode_steps", 0) >= min_steps
            and (batch_size is None or observed.get("batch_size") == batch_size)
        ):
            ids = observed.get("request_ids", [])
            if (
                not ids
                or len(set(ids)) != len(ids)
                or any(not isinstance(rid, str) or not rid for rid in ids)
            ):
                raise ValueError("Stable trigger lacks distinct actual request IDs")
            return observed
        time.sleep(0.05)
    raise ValueError("No live same-request-set decode window before profile deadline")


def execute(report, stage, args, acceptance):
    process = None
    root = args.output_dir
    try:
        if sys.platform != "linux" or not shutil.which("nsys"):
            report.update(
                status="blocked",
                reason="Linux and Nsight Systems are required; nothing was profiled",
            )
            return False
        try:
            RUN.require_cuda(stage, args)
        except (ImportError, RuntimeError) as error:
            report.update(status="blocked", reason=str(error))
            return False
        if acceptance is None:
            raise ValueError(
                "Supply --acceptance with passed target-model correctness evidence"
            )
        receipt = json.loads(acceptance.read_text(encoding="utf-8"))
        report["runtime"] = COMMON.runtime_info()
        correctness = next(s for s in receipt["steps"] if s["id"] == "correctness")
        job = correctness["commands"][0]
        model_report = json.loads(Path(job["report"]).read_text(encoding="utf-8"))
        if (
            load("acceptance").external_result(correctness, job, 0)["status"]
            != "passed"
        ):
            raise ValueError("Correctness artifact no longer passes its gate")
        correctness_gate(
            receipt,
            stage,
            report["runtime"],
            args,
            model_report,
            report["profile_variant"],
        )
        report["acceptance"] = identity(acceptance)
        report["correctness_artifact"] = identity(job["report"])
        report["nsys_version"] = COMMON.capture(["nsys", "--version"])
        # Record visible ordinal -> UUID without launching any storage kernels.
        import torch

        mapping = load("hardware_probe").device_mapping(
            stage
            | {
                "variant": (
                    "cpu"
                    if report["profile_variant"] == "hbm"
                    else report["profile_variant"]
                )
            },
            mpu_ids=args.mpu_device_ids,
        )
        report["devices"] = [
            {
                "visible_ordinal": ordinal,
                "uuid": str(torch.cuda.get_device_properties(ordinal).uuid),
                "role": role,
            }
            for role, ordinals in (
                ("compute", mapping["compute_device_ids"]),
                ("storage", mapping["mpu_device_ids"]),
            )
            for ordinal in ordinals
        ]
        report["physical_gpu_inventory"] = COMMON.capture(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv"]
        )
        with (root / "server-nsys.log").open("w", encoding="utf-8") as log:
            process = COMMON.start_server(
                args,
                report["server_command"],
                log,
                trace_dir=str(root / "forward-trace"),
                env_updates=report["environment"],
            )
            COMMON.wait_ready(process, args.port, args.startup_timeout)
            # Setup and warmup run before the CUDA profiler trigger is armed.
            COMMON.request_json(
                args.port,
                "generate",
                {
                    "input_ids": [42] * 64,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4,
                        "ignore_eos": True,
                    },
                },
                timeout=300,
            )
            report["flush_ack"] = post_text(args.port, "flush_cache", {})
            count = report["workload"]["concurrency"]
            prompts = (
                load("workload").inputs(
                    args.workload, report["workload"]["input_len"], count
                )
                if getattr(args, "workload", None)
                else [[42] * report["workload"]["input_len"] for _ in range(count)]
            )
            payload = {
                "input_ids": prompts if count > 1 else prompts[0],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": report["workload"]["output_len"],
                    "ignore_eos": True,
                },
            }
            stable_steps = getattr(args, "profile_stable_steps", 0)
            if stable_steps:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        COMMON.request_json,
                        args.port,
                        "generate",
                        payload,
                        timeout=args.benchmark_timeout,
                    )
                    report["trigger_progress_before"] = wait_stable(
                        args.port,
                        future,
                        stable_steps,
                        getattr(args, "profile_batch_size", None),
                        args.benchmark_timeout,
                    )
                    report["profile_ack"] = post_text(
                        args.port, "start_profile", report["profile_request"]
                    )
                    report["trigger_progress_after_ack"] = progress(args.port)
                    response = future.result(timeout=args.benchmark_timeout)
            else:
                report["profile_ack"] = post_text(
                    args.port, "start_profile", report["profile_request"]
                )
                response = COMMON.request_json(
                    args.port, "generate", payload, timeout=args.benchmark_timeout
                )
            responses = response if isinstance(response, list) else [response]
            if len(responses) != report["workload"]["concurrency"] or any(
                item.get("meta_info", {}).get("completion_tokens")
                != report["workload"]["output_len"]
                for item in responses
            ):
                raise ValueError(
                    "Profiled request did not complete the intended decode length"
                )
            report["completion_tokens"] = sum(
                item["meta_info"]["completion_tokens"] for item in responses
            )
            report["profiled_request_ids"] = [
                item["meta_info"].get("id") for item in responses
            ]
            if stable_steps and not set(
                report["trigger_progress_before"]["request_ids"]
            ) <= set(report["profiled_request_ids"]):
                raise ValueError(
                    "Stable trigger request IDs are not from the profiled workload"
                )
            RUN.wait_capacities(
                root / "forward-trace",
                stage,
                args,
                report["profile_variant"],
                None if report["profile_variant"] == "hbm" else args.capacity_ratios[0],
                request_ids=report["profiled_request_ids"],
                require_idle=True,
            )
            if stage.get("attn_dp_size", 1) > 1:
                traces = RUN.ANALYZE.load_traces(root / "forward-trace")
                owners = RUN.ANALYZE._request_owners(
                    RUN.ANALYZE._dp_traces(traces), set(report["profiled_request_ids"])
                )
                if set(owners.values()) != set(range(stage["attn_dp_size"])):
                    raise ValueError(
                        "Profile workload did not cover all Attention-DP owners"
                    )
                report["attention_owners"] = owners
            # Terminate the owned server, leaving nsys alive to finalize its report.
            server_pid = int((root / "server.pid").read_text())
            os.kill(server_pid, signal.SIGTERM)
            process.wait(timeout=120)
        trace = root / "decode.nsys-rep"
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError("Nsight did not produce a nonempty trace")
        with (root / "nsys-export.log").open("w") as log:
            subprocess.run(
                [
                    "nsys",
                    "export",
                    "--type=sqlite",
                    "--output=" + str(root / "decode.sqlite"),
                    str(trace),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        report["artifacts"] = [identity(trace), identity(root / "decode.sqlite")]
        report["status"] = "captured_pending_role_analysis"
        report["scope_notes"] = [
            "The existing v2 trigger selects decode after warmup and prefill; verify actual ranges/rank coverage in the trace.",
            "Profiler steps follow the upstream scheduler trigger and may include one boundary step.",
            "A CUDA timeline cannot attribute all memory transactions to KV payload or prove absence of mapped-host loads.",
            "No correctness/reference .cpu() comparisons are executed inside the capture.",
        ]
        return True
    except KeyboardInterrupt:
        report.update(status="failed", error="Profile collection interrupted")
        return False
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        report.update(status="failed", error=str(error))
        return False
    finally:
        if process is not None and process.poll() is None:
            COMMON.stop_server(process)
        COMMON.write_manifest(root / "profile.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--profile-variant", choices=("cpu", "mpu", "hbm"))
    parser.add_argument("--profile-concurrency", type=int)
    parser.add_argument("--profile-stable-steps", type=int, default=0)
    parser.add_argument("--profile-batch-size", type=int)
    control, forwarded = parser.parse_known_args(argv)
    if control.profile_stable_steps < 0 or (
        control.profile_batch_size is not None and control.profile_batch_size < 1
    ):
        raise ValueError(
            "Stable trigger steps must be nonnegative and batch size positive"
        )
    if control.profile_batch_size is not None and not control.profile_stable_steps:
        raise ValueError(
            "Profile batch-size constraint requires a stable decode trigger"
        )
    stage, args = STAGE.parse_args(forwarded)
    args.profile_stable_steps = control.profile_stable_steps
    args.profile_batch_size = control.profile_batch_size
    if Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())
    report = plan(
        stage,
        args,
        control.profile_steps,
        control.profile_variant,
        control.profile_concurrency,
    )
    report["stable_trigger"] = {
        "min_consecutive_decode_steps": control.profile_stable_steps,
        "required_actual_batch_size": control.profile_batch_size,
        "criterion": "CPU metadata from actual consecutive forwards with the same request-ID set; no sleep-only assumption",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    COMMON.write_manifest(args.output_dir / "profile.json", report)
    ok = execute(report, stage, args, control.acceptance) if args.execute else True
    print(args.output_dir / "profile.json")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
