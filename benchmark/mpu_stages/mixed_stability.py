"""Bounded mixed-load client for one already running, traced Qwen server.

The service is never restarted. Periodic admission pauses drain requests and
allow the existing idle-only CUDA trace to flush; they are labelled separately
from load intervals. Cancellation requires a terminal abort and traced phase.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import json
import math
import random
import re
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"mixed_stability_{name}", Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKLOAD, COMPARISON = load("workload"), load("comparison")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def request(port, route, payload=None, timeout=30, *, json_response=True):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=timeout) as response:
        body = response.read()
        if not json_response:
            return {"http_status": response.status, "body": body.decode("utf-8")}
        return json.loads(body) if body else None


def payload(tokens, output_len, rid, *, stream=True, logprobs=False):
    return {
        "input_ids": tokens,
        "rid": rid,
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "return_logprob": logprobs,
    }


def server_signature(info):
    fields = (
        "model_path",
        "base_gpu_id",
        "gpu_id_step",
        "attn_dp_size",
        "ep_size",
        "moe_dense_tp_size",
        "enable_dense_mlp_attn_tp",
        "enable_dp_lm_head",
        "moe_a2a_backend",
        "tp_size",
        "pp_size",
        "dp_size",
        "disaggregation_mode",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "enable_hierarchical_cache",
        "enable_hisparse",
        "chunked_prefill_size",
        "prefill_max_requests",
        "max_running_requests",
        "hisparse_host_to_device_ratio",
        "hisparse_device_buffer_size",
        "hisparse_top_k",
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
        "hisparse_config",
        "speculative_algorithm",
        "enable_mixed_chunk",
        "enable_two_batch_overlap",
        "incremental_streaming_output",
        "version",
    )
    multirank = info.get("attn_dp_size") == 4
    if multirank:
        if not (
            info.get("tp_size") == 4
            and info.get("dp_size") == 1
            and info.get("pp_size") == 1
            and info.get("ep_size") == 4
        ):
            raise ValueError("Mixed stability requires Attention-DP4/EP4")
        required = {
            "moe_dense_tp_size": 1,
            "attn_cp_size": 1,
            "dcp_size": 1,
            "moe_dp_size": 1,
            "nnodes": 1,
            "enable_dp_lm_head": True,
            "moe_a2a_backend": "deepep",
            "enable_prefill_cp": False,
            "enable_cp_tp_group_sharing": False,
        }
        if any(info.get(key) != value for key, value in required.items()):
            raise ValueError(
                "Mixed stability found residual TP/CP or different DeepEP controls"
            )
    for field in (() if multirank else ("tp_size", "pp_size", "dp_size")):
        if info.get(field) != 1:
            raise ValueError(
                "Mixed stability client requires one compute GPU: " + field
            )
    if (
        info.get("disaggregation_mode") != "null"
        or info.get("disable_radix_cache") is not True
    ):
        raise ValueError("Mixed stability requires NoPD and disabled radix cache")
    if (
        info.get("disable_overlap_schedule") is not True
        or info.get("enable_hierarchical_cache") is not False
    ):
        raise ValueError("Mixed stability requires disabled overlap and HiCache")
    if (
        info.get("speculative_algorithm") is not None
        or info.get("enable_mixed_chunk") is not False
        or info.get("enable_two_batch_overlap") is not False
    ):
        raise ValueError(
            "Mixed stability requires ordinary unmixed, non-speculative forwards"
        )
    return {field: info.get(field) for field in fields}


def process_resource(pid, proc_root=Path("/proc")):
    directory = Path(proc_root) / str(pid)
    raw = (directory / "stat").read_text()
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("Malformed Linux process stat")
    fields = raw[closing + 2 :].split()
    start_ticks = int(fields[19])  # field 22, after pid and comm
    status = (directory / "status").read_text()
    values = {}
    for key in ("VmRSS", "VmHWM", "VmSize"):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB$", status, re.MULTILINE)
        if not match:
            raise ValueError("Missing process memory field " + key)
        values[key] = int(match.group(1)) * 1024
    return {"pid": pid, "start_ticks": start_ticks, **values}


def trace_identity(trace_dir):
    paths = sorted(Path(trace_dir).glob("forward-*.jsonl"))
    if len(paths) == 4:
        return load("multirank_stability").trace_identity(paths)
    if len(paths) != 1:
        raise ValueError("Need one or four complete traced scheduler lifetimes")
    path = paths[0]
    with path.open(encoding="utf-8") as stream:
        session = json.loads(stream.readline())
    if session.get("type") != "session" or session.get("schema_version") != 1:
        raise ValueError("Invalid CUDA trace session")
    for field in ("tp_size", "pp_size", "dp_size"):
        if session.get(field) != 1:
            raise ValueError("Trace is not a single-GPU session")
    if (
        session.get("disable_overlap_schedule") is not True
        or session.get("disable_radix_cache") is not True
    ):
        raise ValueError("Trace must disable overlap and radix cache")
    match = re.search(r"-pid(\d+)\.jsonl$", path.name)
    if not match:
        raise ValueError("Trace filename has no scheduler PID")
    return {
        "path": str(path.resolve()),
        "scheduler_pid": int(match.group(1)),
        "session": session,
    }


def cuda_uuid(value):
    """Normalize PyTorch's bare CUuuid to the physical GPU NVML identity."""
    text = str(value)
    if text.startswith(("GPU-", "MIG-")):
        return text
    try:
        return "GPU-" + str(UUID(text))
    except ValueError as error:
        raise ValueError("Invalid CUDA device UUID") from error


def resolve_gpu_roles(info, properties_getter=None):
    """Resolve configured CUDA ordinals to UUIDs; never infer NVML row order."""
    config = info.get("hisparse_config") or {}
    config = json.loads(config) if isinstance(config, str) else config
    if not isinstance(config, dict):
        raise ValueError("Invalid HiSparse configuration for GPU role mapping")
    compute = info.get("base_gpu_id", 0)
    if type(compute) is not int or compute < 0:
        raise ValueError("Compute GPU ordinal must be a nonnegative integer")
    compute_ids = (
        [compute + index * info.get("gpu_id_step", 1) for index in range(4)]
        if info.get("attn_dp_size") == 4
        else [compute]
    )
    storage = []
    if info.get("enable_hisparse") and config.get("backing_store") == "mpu":
        storage = config.get("mpu_device_ids")
        if (
            not isinstance(storage, list)
            or not 1 <= len(storage) <= 4
            or any(type(ordinal) is not int or ordinal < 0 for ordinal in storage)
            or len(set([*compute_ids, *storage])) != len(storage) + len(compute_ids)
        ):
            raise ValueError(
                "Single-compute MPU stability needs 1-4 disjoint storage ordinals"
            )
    if properties_getter is None:
        import torch

        properties_getter = torch.cuda.get_device_properties
    roles = [
        {
            "visible_ordinal": ordinal,
            "uuid": cuda_uuid(properties_getter(ordinal).uuid),
            "role": role,
        }
        for role, ordinals in (("compute", compute_ids), ("storage", storage))
        for ordinal in ordinals
    ]
    gpu_identity(roles)
    if len(compute_ids) == 4 and storage and len(storage) != 4:
        raise ValueError("Four-compute MPU experiment needs four storage devices")
    return roles


def gpu_identity(rows):
    """Freeze selected UUID/role/ordinal identity, independent of query order."""
    if not rows:
        raise ValueError("GPU resource identity is empty")
    identity = {}
    for row in rows:
        uuid = row.get("uuid")
        role = row.get("role", "compute" if len(rows) == 1 else None)
        ordinal = row.get("visible_ordinal", 0 if len(rows) == 1 else None)
        if (
            not isinstance(uuid, str)
            or not uuid.startswith(("GPU-", "MIG-"))
            or role not in ("compute", "storage")
            or type(ordinal) is not int
            or ordinal < 0
            or uuid in identity
        ):
            raise ValueError("Invalid or duplicated GPU UUID/role/ordinal identity")
        identity[uuid] = {"role": role, "visible_ordinal": ordinal}
    if (
        sum(row["role"] == "compute" for row in identity.values()) not in (1, 4)
        or (
            sum(row["role"] == "compute" for row in identity.values()) == 1
            and len(identity) > 5
        )
        or (
            sum(row["role"] == "compute" for row in identity.values()) == 4
            and len(identity) not in (4, 8)
        )
        or len({row["visible_ordinal"] for row in identity.values()}) != len(identity)
    ):
        raise ValueError(
            "GPU resource identity needs one compute and up to four storage devices"
        )
    return identity


def gpu_used_deltas(initial, final):
    identity = gpu_identity(initial)
    if gpu_identity(final) != identity:
        raise ValueError("Selected GPU UUID/role/ordinal identity changed")
    before = {row["uuid"]: row["used_bytes"] for row in initial}
    after = {row["uuid"]: row["used_bytes"] for row in final}
    if any(
        type(value) is not int or value < 0
        for value in [*before.values(), *after.values()]
    ):
        raise ValueError("GPU memory samples must be nonnegative integer bytes")
    return {
        uuid: {**identity[uuid], "used_delta_bytes": after[uuid] - before[uuid]}
        for uuid in identity
    }


def resource_snapshot(port, identity, baseline_processes, signature, gpu_roles=None):
    if (
        trace_identity(
            Path(identity.get("trace_dir", Path(identity.get("path", ".")).parent))
        )
        != identity
    ):
        raise ValueError("Scheduler lifetime changed during mixed load")
    info = request(port, "server_info")
    if server_signature(info) != signature:
        raise ValueError("Server configuration changed during mixed load")
    processes = [process_resource(p["pid"]) for p in baseline_processes]
    if any(
        p["start_ticks"] != b["start_ticks"]
        for p, b in zip(processes, baseline_processes)
    ):
        raise ValueError("Owned server/scheduler process was restarted")
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]
    roles = None if gpu_roles is None else gpu_identity(gpu_roles)
    if roles is not None:
        command += ["--id=" + ",".join(roles)]
    gpu = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    rows = []
    for line in gpu.splitlines():
        uuid, memory = line.split(",")
        rows.append(
            {"uuid": uuid.strip(), "used_bytes": int(memory.strip()) * 1024 * 1024}
        )
    if roles is None:
        if len(rows) != 1:
            raise ValueError(
                "Multi-GPU stability requires an explicit CUDA UUID/role mapping"
            )
        gpu_identity(rows)
    else:
        if len({row["uuid"] for row in rows}) != len(rows) or {
            row["uuid"] for row in rows
        } != set(roles):
            raise ValueError("Selected GPU UUID inventory changed or is incomplete")
        samples = {row["uuid"]: row for row in rows}
        rows = [{**samples[uuid], **metadata} for uuid, metadata in roles.items()]
        gpu_identity(rows)
    return {
        "monotonic_ns": time.monotonic_ns(),
        "processes": processes,
        "gpu": rows,
        "gpu_resource_scope": "Selected CUDA-visible UUIDs and compute/storage roles; other NVML-visible GPUs are not measured",
        "server_info": info,
        "load": request(port, "get_load"),
    }


def scan_idle_trace(path, expected, session):
    """Stream the growing trace without retaining its full multi-hour history."""
    facts = {rid: {"prefill_tokens": 0, "decode_forwards": 0} for rid in expected}
    sequence, last, capacity, session_count = -1, None, None, 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            last = row
            if row.get("type") == "session":
                session_count += 1
                if row != session:
                    raise ValueError("Trace session changed")
            elif row.get("type") == "forward":
                if (
                    row.get("sequence") != sequence + 1
                    or row.get("success") is not True
                ):
                    raise ValueError("Failed, missing or reordered traced forward")
                sequence += 1
                ids = row.get("request_ids", [])
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate request ID in traced batch")
                mode = row.get("forward_mode")
                if mode not in (
                    ("extend", "decode", "idle")
                    if session.get("attn_dp_size") == 4
                    else ("extend", "decode")
                ):
                    raise ValueError("Unsupported mixed-load forward mode")
                if mode == "extend":
                    lengths = row.get("extend_lens")
                    if not isinstance(lengths, list) or len(lengths) != len(ids):
                        raise ValueError("Missing per-request prefill chunk lengths")
                    if any(type(n) is not int or n <= 0 for n in lengths):
                        raise ValueError("Invalid prefill token lengths")
                if mode == "idle" and ids:
                    raise ValueError("Idle EP rank has request IDs")
                for i, rid in enumerate(ids):
                    if rid not in facts:
                        continue
                    if mode == "extend":
                        facts[rid]["prefill_tokens"] += lengths[i]
                    else:
                        facts[rid]["decode_forwards"] += 1
            elif row.get("type") == "capacity":
                capacity = row
    if (
        session_count != 1
        or not last
        or last.get("type") != "idle_flush"
        or last.get("last_sequence") != sequence
    ):
        raise ValueError("Trace has not fully flushed at scheduler idle")
    if not capacity or capacity.get("after_sequence") != sequence:
        raise ValueError("Missing current idle capacity snapshot")
    result = capacity["capacity"]
    if result.get("tp_rank") != session.get("tp_rank", 0) or result.get(
        "tp_size"
    ) != session.get("tp_size"):
        raise ValueError("Idle capacity rank identity differs")
    if any(result.get(field) != 0 for field in ("reserved_slots", "used_slots")):
        raise ValueError("KV allocation/reservation remains at idle")
    for rid in expected:
        if facts[rid]["prefill_tokens"] == facts[rid]["decode_forwards"] == 0:
            facts[rid]["no_forward_observed"] = True
    return {"last_sequence": sequence, "capacity": result, "request_facts": facts}


def validate_request(row, facts):
    if row.get("status") != "terminal":
        raise ValueError("Request failed or stream has no terminal response")
    meta = row["response"].get("meta_info", {})
    if meta.get("id") != row["request_id"]:
        raise ValueError("Response request ID changed")
    count = meta.get("completion_tokens")
    if type(count) is not int or count < 0 or count > row["output_len"]:
        raise ValueError("Invalid actual output token count")
    if row["cancel_attempt"].get("error"):
        raise ValueError("Owned cancellation control call failed")
    reason = (meta.get("finish_reason") or {}).get("type")
    if reason == "abort":
        if not row["cancel_attempt"].get("attempted") or row["cancel_attempt"].get(
            "error"
        ):
            raise ValueError("Abort lacked a successful owned cancellation call")
        if count >= row["output_len"]:
            raise ValueError("Abort arrived after normal output completion")
        if facts["decode_forwards"] > 0:
            if facts["prefill_tokens"] != row["input_len"] or count < 2:
                raise ValueError("Decode abort has inconsistent phase/token evidence")
            phase = "decode"
        elif facts["prefill_tokens"] > 0:
            if facts["prefill_tokens"] == row["input_len"] and count <= 1:
                phase = "prefill_complete_no_decode"
            elif 0 < facts["prefill_tokens"] < row["input_len"] and count == 0:
                phase = "partial_prefill"
            else:
                raise ValueError("Prefill abort lacks a partial prefill phase")
        else:
            if count:
                raise ValueError("Abort output exists without traced forwards")
            phase = "no_forward_observed"
        return {
            "terminal": "abort",
            "actual_cancel_phase": phase,
            "requested_cancel_phase": row["cancel_mode"],
            "trace": facts,
        }
    if (
        reason != "length"
        or count != row["output_len"]
        or meta.get("prompt_tokens") != row["input_len"]
    ):
        raise ValueError("Normal fixed-budget request did not complete as planned")
    if (
        facts["prefill_tokens"] != row["input_len"]
        or facts["decode_forwards"] != count - 1
    ):
        raise ValueError(
            "Normal request has missing/recomputed prefill or decode forwards"
        )
    if meta.get("cached_tokens", 0) != 0:
        raise ValueError("Radix-prefix reuse appeared in mixed-load request")
    return {
        "terminal": "length",
        "actual_cancel_phase": None,
        "cancel_window_missed": row["cancel_mode"] is not None,
        "trace": facts,
    }


def stream_request(port, tokens, case, timeout, output_dir):
    rid = case["request_id"]
    row = {
        **case,
        "status": "running",
        "start_monotonic_ns": time.monotonic_ns(),
        "chunks": [],
        "cancel_attempt": {"attempted": False},
    }
    finished, cancel_lock = threading.Event(), threading.Lock()

    def cancel():
        with cancel_lock:
            if finished.is_set() or row["cancel_attempt"]["attempted"]:
                return
            row["cancel_attempt"] = {
                "attempted": True,
                "monotonic_ns": time.monotonic_ns(),
            }
            try:
                row["cancel_attempt"]["http"] = request(
                    port, "abort_request", {"rid": rid}, timeout=30, json_response=False
                )
            except Exception as error:
                row["cancel_attempt"]["error"] = repr(error)

    timer = None
    if case["cancel_mode"] == "timed":
        timer = threading.Timer(case["cancel_delay_seconds"], cancel)
        timer.daemon = True
        timer.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(payload(tokens, case["output_len"], rid)).encode(),
            headers={"Content-Type": "application/json"},
        )
        previous = 0
        with opener.open(req, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                item = json.loads(data)
                meta = item.get("meta_info", {})
                count = meta.get("completion_tokens")
                if type(count) is not int or count < previous:
                    raise ValueError(
                        "Missing/non-monotonic cumulative streamed token counts"
                    )
                stamp = time.monotonic_ns()
                row["chunks"].append(
                    {
                        "monotonic_ns": stamp,
                        "completion_tokens": count,
                        "delta_tokens": count - previous,
                    }
                )
                if count > 0 and "first_token_monotonic_ns" not in row:
                    row["first_token_monotonic_ns"] = stamp
                previous = count
                row["response"] = item
                if (
                    case["cancel_mode"] == "decode"
                    and count >= case["cancel_after_tokens"]
                    and not meta.get("finish_reason")
                ):
                    cancel()
                if meta.get("finish_reason"):
                    row["terminal_monotonic_ns"] = stamp
                    finished.set()
        if not finished.is_set():
            raise ValueError("Stream ended without terminal finish_reason")
        row["status"] = "terminal"
    except Exception as error:
        row.update(status="failed", error=repr(error))
        cancel()
    finally:
        finished.set()
        if timer:
            timer.cancel()
            timer.join(35)
        row["end_monotonic_ns"] = time.monotonic_ns()
        write(Path(output_dir) / f"{rid}.json", row)
    return row


def plan_case(rng, number, corpus, output_lens, cancel_rate, run_id):
    length = rng.choice(sorted(corpus))
    index = rng.randrange(len(corpus[length]))
    output_len = rng.choice(output_lens)
    mode = rng.choice(("timed", "decode")) if rng.random() < cancel_rate else None
    return {
        "request_id": f"mpu-stability-{run_id}-{number:08d}",
        "request_number": number,
        "input_len": length,
        "corpus_row": index,
        "source_id": corpus[length][index]["source_id"],
        "input_sha256": corpus[length][index]["input_sha256"],
        "output_len": output_len,
        "cancel_mode": mode,
        "cancel_delay_seconds": rng.choice((0.02, 0.1, 0.5)),
        "cancel_after_tokens": 4,
    }


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def metrics(rows, elapsed):
    normal = [r for r in rows if r.get("validation", {}).get("terminal") == "length"]
    ttft, tpot, latency, chunks = [], [], [], []
    for row in normal:
        first = row.get("first_token_monotonic_ns")
        end = row.get("terminal_monotonic_ns")
        if first is None or end is None:
            raise ValueError("Normal response lacks client token/terminal timestamps")
        ttft.append((first - row["start_monotonic_ns"]) / 1e6)
        latency.append((end - row["start_monotonic_ns"]) / 1e6)
        tpot.append((end - first) / (row["output_len"] - 1) / 1e6)
        positive = [c for c in row["chunks"] if c["delta_tokens"] > 0]
        chunks.extend(
            (b["monotonic_ns"] - a["monotonic_ns"]) / 1e6
            for a, b in zip(positive, positive[1:])
        )
    observed = sum(
        r.get("response", {}).get("meta_info", {}).get("completion_tokens", 0)
        for r in rows
    )
    return {
        "request_count": len(rows),
        "normal_completed": len(normal),
        "cancelled": len(rows) - len(normal),
        "cancel_phases": dict(
            Counter(
                r["validation"].get("actual_cancel_phase")
                for r in rows
                if r["validation"]["terminal"] == "abort"
            )
        ),
        "missed_cancel_windows": sum(
            r["validation"].get("cancel_window_missed", False) for r in rows
        ),
        "duration_seconds": elapsed,
        "all_observed_output_tokens": observed,
        "all_observed_output_throughput": observed / elapsed,
        "normal_output_tokens": sum(r["output_len"] for r in normal),
        "normal_output_throughput": sum(r["output_len"] for r in normal) / elapsed,
        "mean_ttft_ms": sum(ttft) / len(ttft) if ttft else None,
        "p99_ttft_ms": percentile(ttft, 0.99),
        "mean_request_tpot_ms": sum(tpot) / len(tpot) if tpot else None,
        "p99_request_tpot_ms": percentile(tpot, 0.99),
        "p99_request_latency_ms": percentile(latency, 0.99),
        "p99_positive_chunk_gap_ms": percentile(chunks, 0.99),
        "latency_scope": "normal fixed-budget requests only; TPOT is request-average, chunk gaps are not per-token ITL",
        "throughput_scope": "epoch load plus drain elapsed; excludes explicit idle/probe pause; cancelled token work separately reported",
    }


def corpus_rows(workload, lengths):
    result = {}
    manifest = WORKLOAD.load_manifest(workload, lengths=lengths)
    for length in lengths:
        path, bucket = WORKLOAD.bucket(workload, length)
        # The existing loader validates all IDs and per-row token hashes.
        WORKLOAD.inputs(workload, length, bucket["num_requests"])
        result[length] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return manifest, result


def capacity_signature(capacity):
    """Only persistent allocation groups are invariant; free worklists may grow."""
    if "per_rank" in capacity:
        return {"per_rank": [capacity_signature(row) for row in capacity["per_rank"]]}
    fields = (
        "backing_store",
        "physical_compute_slots",
        "backing_slots",
        "compute_kv_bytes",
        "backing_kv_bytes",
        "allocated_backing_kv_bytes",
        "requested_capacity_ratio",
        "effective_capacity_ratio",
        "storage_device_ids",
        "retained_gpu_pool_bytes",
        "retained_gpu_pool_shapes",
    )
    ledger = capacity.get("gpu_memory_ledger", {}).get("groups", {})
    return {
        **{key: capacity.get(key) for key in fields},
        "persistent_ledger_groups": {
            key: ledger.get(key)
            for key in ("kv", "fixed", "capacity_dependent", "backend_specific")
        },
    }


@contextlib.contextmanager
def client_pool(args, active_ids, active_lock):
    """Abort owned in-flight requests before waiting for worker cleanup on error."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        yield pool
    except BaseException:
        with active_lock:
            ids = sorted(active_ids)
        for rid in ids:
            try:
                request(
                    args.port,
                    "abort_request",
                    {"rid": rid},
                    timeout=10,
                    json_response=False,
                )
            except Exception:
                pass  # Outer failure receipt separately records cleanup failures.
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def idle_checkpoint(
    args,
    identity,
    expected,
    processes,
    signature,
    probe_tokens,
    baseline,
    label,
    *,
    gpu_roles=None,
):
    rid = f"mpu-stability-{args.output_dir.name}-probe-{label}"
    started = time.monotonic_ns()
    response = request(
        args.port,
        "generate",
        payload(probe_tokens, args.probe_output_len, rid, stream=False, logprobs=True),
        args.request_timeout,
    )
    write(args.output_dir / f"probe-{label}.json", response)
    current = COMPARISON.normalize_outputs(
        [response], [len(probe_tokens)], args.probe_output_len
    )
    if (response.get("meta_info", {}).get("finish_reason") or {}).get(
        "type"
    ) != "length":
        raise ValueError("Recovery probe did not complete normally")
    comparison = (
        COMPARISON.compare([baseline], [current], args.logprob_atol)
        if baseline is not None
        else None
    )
    deadline = time.monotonic() + args.idle_timeout
    while True:
        try:
            trace = (
                load("multirank_stability").scan_idle(
                    identity, set(expected) | {rid}, scan_idle_trace
                )
                if "per_rank" in identity
                else scan_idle_trace(
                    identity["path"], set(expected) | {rid}, identity["session"]
                )
            )
            probe_fact = trace["request_facts"][rid]
            if (
                probe_fact["prefill_tokens"] != len(probe_tokens)
                or probe_fact["decode_forwards"] != args.probe_output_len - 1
            ):
                raise ValueError("Recovery probe forward coverage is incomplete")
            break
        except (ValueError, json.JSONDecodeError) as error:
            if time.monotonic() >= deadline:
                raise ValueError(
                    "Idle/recovery evidence did not become valid: " + str(error)
                ) from error
            time.sleep(0.25)
    return current, {
        "probe_id": rid,
        "probe_comparison": comparison,
        "trace": trace,
        "resources": resource_snapshot(
            args.port, identity, processes, signature, gpu_roles
        ),
        "pause_seconds": (time.monotonic_ns() - started) / 1e9,
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requests_dir = args.output_dir / "requests"
    requests_dir.mkdir()
    manifest, corpus = corpus_rows(args.workload, args.input_lens)
    recipe = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    receipt = {
        "schema_version": 1,
        "kind": "mpu_mixed_stability",
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": recipe,
        "workload_sha256": sha(args.workload),
        "source": manifest.get("source"),
        "sampling": "seeded burst arrivals bounded by client concurrency; public frozen input IDs are reused across epochs",
        "memory_conclusion": "resource trajectories, no universal leak/SLO threshold inferred",
        "epochs": [],
    }
    write(args.output_dir / "mixed-stability.json", receipt)
    if not args.execute:
        return receipt
    identity = trace_identity(args.trace_dir)
    server_info = request(args.port, "server_info")
    signature = server_signature(server_info)
    gpu_roles = resolve_gpu_roles(server_info)
    processes = [
        process_resource(pid)
        for pid in sorted(
            {
                args.server_pid,
                *identity.get("scheduler_pids", [identity.get("scheduler_pid")]),
            }
        )
    ]
    receipt.update(
        status="running",
        trace_identity=identity,
        server_signature=signature,
        process_lifetimes=processes,
        gpu_roles=gpu_roles,
    )
    probe_tokens = corpus[max(corpus)][0]["input_ids"]
    reference, initial = idle_checkpoint(
        args,
        identity,
        set(),
        processes,
        signature,
        probe_tokens,
        None,
        "initial",
        gpu_roles=gpu_roles,
    )
    receipt["initial_checkpoint"] = initial
    write(args.output_dir / "mixed-stability.json", receipt)
    rng, serial = random.Random(args.seed), 0
    start = time.monotonic()
    deadline = start + args.duration_seconds
    all_metrics = []
    active_ids, active_lock = set(), threading.Lock()
    last_capacity = initial["trace"]["capacity"]
    memory_signature = capacity_signature(last_capacity)
    try:
        with client_pool(args, active_ids, active_lock) as pool:
            while time.monotonic() < deadline:
                epoch = len(receipt["epochs"])
                began = time.monotonic()
                stop = min(deadline, began + args.epoch_seconds)
                future_rows, rows = {}, []
                next_burst = next_monitor = began
                offered = admitted = 0

                def submit(case):
                    with active_lock:
                        active_ids.add(case["request_id"])
                    tokens = corpus[case["input_len"]][case["corpus_row"]]["input_ids"]
                    return pool.submit(
                        stream_request,
                        args.port,
                        tokens,
                        case,
                        args.request_timeout,
                        requests_dir,
                    )

                while time.monotonic() < stop or future_rows:
                    now = time.monotonic()
                    if now < stop and now >= next_burst:
                        count = rng.choice(
                            (1, min(4, args.concurrency), args.concurrency)
                        )
                        offered += count
                        slots = args.concurrency - len(future_rows)
                        for _ in range(min(count, slots)):
                            case = plan_case(
                                rng,
                                serial,
                                corpus,
                                args.output_lens,
                                args.cancel_rate,
                                args.output_dir.name,
                            )
                            serial += 1
                            future_rows[submit(case)] = case["request_id"]
                            admitted += 1
                            append(
                                args.output_dir / "request-plan.jsonl",
                                {
                                    **case,
                                    "epoch": epoch,
                                    "offered_monotonic_ns": time.monotonic_ns(),
                                },
                            )
                        next_burst = now + args.burst_interval_seconds
                    for future in [future for future in future_rows if future.done()]:
                        rid = future_rows.pop(future)
                        with active_lock:
                            active_ids.discard(rid)
                        row = future.result()
                        rows.append(row)
                        if row["status"] != "terminal":
                            raise ValueError(
                                f"Mixed-load request failed: {rid}: {row.get('error')}"
                            )
                    if now >= next_monitor:
                        snapshot = resource_snapshot(
                            args.port, identity, processes, signature, gpu_roles
                        )
                        if gpu_identity(snapshot["gpu"]) != gpu_identity(
                            initial["resources"]["gpu"]
                        ):
                            raise ValueError("Visible GPU identity changed")
                        append(
                            args.output_dir / "resource-samples.jsonl",
                            {
                                "epoch": epoch,
                                "phase": "load" if now < stop else "drain",
                                **snapshot,
                            },
                        )
                        next_monitor = now + args.monitor_interval_seconds
                    time.sleep(0.02)
                elapsed = time.monotonic() - began
                ids = {row["request_id"] for row in rows}
                _, checkpoint = idle_checkpoint(
                    args,
                    identity,
                    ids,
                    processes,
                    signature,
                    probe_tokens,
                    reference,
                    f"epoch-{epoch:03d}",
                    gpu_roles=gpu_roles,
                )
                cap = checkpoint["trace"]["capacity"]
                if capacity_signature(cap) != memory_signature:
                    raise ValueError(
                        "Retained KV/metadata allocation signature changed at idle"
                    )
                for row in rows:
                    row["validation"] = validate_request(
                        row, checkpoint["trace"]["request_facts"][row["request_id"]]
                    )
                    write(requests_dir / f"{row['request_id']}.json", row)
                    append(
                        args.output_dir / "request-results.jsonl",
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("chunks", "response")
                        },
                    )
                stats = metrics(rows, elapsed)
                all_metrics.append(stats)
                checkpoint["trace"].pop("request_facts")
                receipt["epochs"].append(
                    {
                        "index": epoch,
                        "offered_burst_requests": offered,
                        "admitted_requests": admitted,
                        "clipped_at_client_concurrency": offered - admitted,
                        "metrics": stats,
                        "checkpoint": checkpoint,
                    }
                )
                write(args.output_dir / "mixed-stability.json", receipt)
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "elapsed_seconds": time.monotonic() - start,
                            **stats,
                        }
                    ),
                    flush=True,
                )
        receipt["final_flush_cache"] = request(
            args.port, "flush_cache", {}, timeout=30, json_response=False
        )
        _, final = idle_checkpoint(
            args,
            identity,
            set(),
            processes,
            signature,
            probe_tokens,
            reference,
            "final",
            gpu_roles=gpu_roles,
        )
        receipt["final_checkpoint"] = final
        if capacity_signature(final["trace"]["capacity"]) != memory_signature:
            raise ValueError(
                "Retained KV/storage allocation signature changed at final idle"
            )
        phases = Counter()
        for stats in all_metrics:
            phases.update(stats["cancel_phases"])
        if args.cancel_rate and phases["decode"] < 1:
            raise ValueError(
                "No terminal/traced decode cancellation was actually exercised"
            )
        total_duration = time.monotonic() - start
        if total_duration < args.duration_seconds:
            raise ValueError("Stability duration is incomplete")
        gpu_deltas = gpu_used_deltas(
            initial["resources"]["gpu"], final["resources"]["gpu"]
        )
        compute_uuid = next(
            uuid for uuid, row in gpu_deltas.items() if row["role"] == "compute"
        )
        receipt.update(
            status="passed",
            actual_seconds=total_duration,
            total_requests=serial,
            verified_cancel_phases=dict(phases),
            initial_final_selected_process_rss_delta_bytes=sum(
                p["VmRSS"] for p in final["resources"]["processes"]
            )
            - sum(p["VmRSS"] for p in initial["resources"]["processes"]),
            initial_final_gpu_used_delta_bytes=gpu_deltas[compute_uuid][
                "used_delta_bytes"
            ],
            initial_final_gpu_used_delta_bytes_by_uuid=gpu_deltas,
            gpu_delta_scope="Legacy scalar is the compute GPU delta; per-UUID deltas retain each compute/storage role separately",
            functional_scope="same service/scheduler lifetime, successful mixed requests, traced cancellations, periodic zero idle KV, identical deterministic recovery probes",
            performance_scope="empirical epoch throughput/latency/resource drift, SLO and hardware cost remain unfilled",
        )
    except BaseException as error:
        with active_lock:
            remaining = sorted(active_ids)
        cleanup = []
        for rid in remaining:
            try:
                cleanup.append(
                    {
                        "rid": rid,
                        "abort_call": request(
                            args.port,
                            "abort_request",
                            {"rid": rid},
                            timeout=10,
                            json_response=False,
                        ),
                    }
                )
            except Exception as cleanup_error:
                cleanup.append({"rid": rid, "error": repr(cleanup_error)})
        receipt.update(
            status="failed",
            error=repr(error),
            cleanup_calls=cleanup,
            actual_seconds=time.monotonic() - start,
        )
        write(args.output_dir / "mixed-stability.json", receipt)
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output_dir / "mixed-stability.json", receipt)
    return receipt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--server-pid", type=int, required=True)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-lens", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--output-lens", nargs="+", type=int, default=[80, 256, 1024])
    p.add_argument("--duration-seconds", type=float, default=10800)
    p.add_argument("--epoch-seconds", type=float, default=300)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--burst-interval-seconds", type=float, default=5)
    p.add_argument("--monitor-interval-seconds", type=float, default=15)
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--idle-timeout", type=float, default=30)
    p.add_argument("--cancel-rate", type=float, default=0.1)
    p.add_argument("--probe-output-len", type=int, default=64)
    p.add_argument("--logprob-atol", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.server_pid <= 0 or args.concurrency < 1 or args.probe_output_len < 8:
        p.error("Positive process/concurrency and probe >=8 tokens required")
    for name in (
        "duration_seconds",
        "epoch_seconds",
        "burst_interval_seconds",
        "monitor_interval_seconds",
        "request_timeout",
        "idle_timeout",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            p.error(name + " must be finite and positive")
    if not math.isfinite(args.cancel_rate) or not 0 <= args.cancel_rate <= 1:
        p.error("cancel-rate must be in [0,1]")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        p.error("logprob-atol must be finite and nonnegative")
    if any(n <= 0 for n in args.input_lens) or any(n < 8 for n in args.output_lens):
        p.error("Positive inputs and output budgets >=8 required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.output_dir.name):
        p.error("output-dir basename must be a short request-ID-safe experiment name")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    existed = arguments.output_dir.exists()
    try:
        result = run(arguments)
    except BaseException as error:
        path = arguments.output_dir / "mixed-stability.json"
        if not existed and path.is_file():
            failed = json.loads(path.read_text(encoding="utf-8"))
            failed.update(
                status="failed",
                error=repr(error),
                finished_utc=datetime.now(timezone.utc).isoformat(),
            )
            write(path, failed)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_requests": result.get("total_requests"),
                "actual_seconds": result.get("actual_seconds"),
            }
        ),
        flush=True,
    )
