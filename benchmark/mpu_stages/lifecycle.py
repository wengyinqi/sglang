"""Fault acceptance against a server owned by correctness.py.

Use the public abort/pause(retract)/continue APIs. Trace evidence must establish
the actual fault phase; an HTTP 200 alone never proves cancellation or retract.
"""

import copy
import json
import threading
import time
import urllib.request
import uuid


class MissedPrefillWindow(ValueError):
    pass


def post(port, route, payload, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else None


def request_payload(inputs, output_len, rid, stream=False, owner=None):
    return {
        **({"routed_dp_rank": owner} if owner is not None else {}),
        "input_ids": inputs,
        "rid": rid,
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "return_logprob": True,
    }


def stream_worker(port, payload, timeout, state, decode_seen, done):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                item = json.loads(data)
                state["response"] = item
                meta = item.get("meta_info", {})
                if meta.get("completion_tokens", 0) >= 2 and not meta.get(
                    "finish_reason"
                ):
                    decode_seen.set()
        if not state.get("response", {}).get("meta_info", {}).get("finish_reason"):
            raise ValueError("Stream ended without a terminal response")
    except BaseException as error:
        state["error"] = repr(error)
    finally:
        done.set()


def fault_request(args, inputs, case, delay=0):
    rid = f"mpu-{case}-{uuid.uuid4().hex}"
    state, decode_seen, done = {}, threading.Event(), threading.Event()
    payload = request_payload(
        inputs,
        args.output_len,
        rid,
        stream=True,
        owner=getattr(args, "mpu_fault_owner", None),
    )
    worker = threading.Thread(
        target=stream_worker,
        args=(args.port, payload, args.benchmark_timeout, state, decode_seen, done),
        daemon=True,
    )
    paused = False
    worker.start()
    try:
        if case == "abort_prefill":
            # The phase is confirmed retrospectively from every rank's trace.
            done.wait(delay)
        else:
            deadline = time.monotonic() + args.benchmark_timeout
            while not decode_seen.wait(0.01):
                if done.is_set() or time.monotonic() >= deadline:
                    raise ValueError(f"No live decode observed before {case}: {state}")
        if case == "retract_decode":
            paused = True  # Resume even if the pause HTTP request times out.
            post(args.port, "pause_generation", {"mode": "retract"}, 30)
            post(args.port, "continue_generation", {}, 30)
            paused = False
        else:
            post(args.port, "abort_request", {"rid": rid}, 30)
        if not done.wait(args.benchmark_timeout):
            raise TimeoutError(f"{case} request did not finish")
        if "error" in state:
            raise ValueError(state["error"])
        response = state["response"]
        if response.get("meta_info", {}).get("id") != rid:
            raise ValueError("Fault response did not preserve the submitted request ID")
        return rid, response
    finally:
        try:
            if paused:
                post(args.port, "continue_generation", {}, 30)
        finally:
            if not done.is_set():
                try:
                    post(args.port, "abort_request", {"rid": rid}, 10)
                finally:
                    worker.join(10)
            else:
                worker.join()


def forward_facts(traces, rid, tp_size, attn_dp_size=1):
    facts = []
    for trace in traces:
        rows = [
            record
            for record in trace
            if record.get("type") == "forward" and rid in record.get("request_ids", [])
        ]
        if not rows:
            continue
        session = trace[0]
        if session.get("tp_size") != tp_size or any(
            not row.get("success") for row in rows
        ):
            raise ValueError(
                "Fault traces contain failed forwards or a different TP size"
            )
        prefill = 0
        modes = []
        for row in rows:
            mode = row["forward_mode"]
            if mode not in ("extend", "decode"):
                raise ValueError(f"Unexpected fault forward mode: {mode}")
            modes.append(mode)
            if mode == "extend":
                prefill += row["extend_lens"][row["request_ids"].index(rid)]
        facts.append(
            {
                "rank": session["tp_rank"],
                "prefill_tokens": prefill,
                "decode_forwards": modes.count("decode"),
                "modes": modes,
            }
        )
    if not facts:
        return []
    if attn_dp_size > 1:
        if attn_dp_size != 4 or len(facts) != 1 or facts[0]["rank"] not in range(4):
            raise ValueError("Fault request must have exactly one Attention-DP owner")
        return facts
    if sorted(item["rank"] for item in facts) != list(range(tp_size)):
        raise ValueError("Fault request is missing from one or more TP ranks")
    signature = lambda f: (f["prefill_tokens"], f["modes"])
    if any(signature(f) != signature(facts[0]) for f in facts[1:]):
        raise ValueError("TP ranks disagree on the fault request's lifecycle")
    return sorted(facts, key=lambda f: f["rank"])


def validate_fault(case, response, facts, input_len, output_len):
    meta = response.get("meta_info", {})
    reason = meta.get("finish_reason") or {}
    if case == "abort_prefill":
        if (
            reason.get("type") != "abort"
            or not facts
            or any(
                not 0 < f["prefill_tokens"] < input_len or f["decode_forwards"]
                for f in facts
            )
        ):
            raise MissedPrefillWindow("Abort did not land inside chunked prefill")
    elif case == "abort_decode":
        if (
            reason.get("type") != "abort"
            or not facts
            or any(f["decode_forwards"] < 1 for f in facts)
            or not 2 <= meta.get("completion_tokens", 0) < output_len
        ):
            raise ValueError("Decode abort was not observed before normal completion")
    elif case == "retract_decode":
        if (
            meta.get("num_retractions", 0) < 1
            or not facts
            or any(
                f["decode_forwards"] < 1 or f["prefill_tokens"] <= input_len
                for f in facts
            )
        ):
            raise ValueError(
                "No real decode retract and prefill recomputation observed"
            )
    else:
        raise ValueError(f"Unknown lifecycle case: {case}")


def _run_owner_cases(run, compare, stage, args, group, inputs):
    """Each fault must be followed by an unchanged deterministic recovery output."""
    prefix = args.output_dir / f"{group['id']}-lifecycle"
    timeout = args.benchmark_timeout
    baseline_rid = f"mpu-baseline-{uuid.uuid4().hex}"
    baseline = post(
        args.port,
        "generate",
        request_payload(
            inputs,
            args.output_len,
            baseline_rid,
            owner=getattr(args, "mpu_fault_owner", None),
        ),
        timeout,
    )
    expected = compare.normalize_outputs([baseline], [len(inputs)], args.output_len)
    fixed_compute = run.enforce_fixed_compute(None, group["capacity"])
    run.COMMON.write_manifest(
        prefix.with_name(prefix.name + "-baseline.json"), baseline
    )
    run.wait_capacities(
        group["trace_dir"],
        stage,
        args,
        group["variant"],
        group["ratio"],
        request_ids=[baseline_rid],
        require_idle=True,
    )
    cases = group["lifecycle_results"] = []
    for case in ("abort_prefill", "abort_decode", "retract_decode"):
        attempts = (0.01, 0.05, 0.2) if case == "abort_prefill" else (0,)
        passed = False
        for index, delay in enumerate(attempts):
            receipt = {
                "case": case,
                "attempt": index,
                "abort_delay_seconds": delay,
                "status": "running",
            }
            cases.append(receipt)
            path = prefix.with_name(f"{prefix.name}-{case}-{index}.json")
            try:
                rid, response = fault_request(args, inputs, case, delay)
                receipt.update(request_id=rid, response=response)
                # A fresh recovery ID forces capacity snapshots to be newer than
                # the fault, even when abort happened before its first forward.
                recovery_rid = f"mpu-recovery-{uuid.uuid4().hex}"
                recovered = post(
                    args.port,
                    "generate",
                    request_payload(
                        inputs,
                        args.output_len,
                        recovery_rid,
                        owner=getattr(args, "mpu_fault_owner", None),
                    ),
                    timeout,
                )
                receipt["recovery_response"] = recovered
                actual = compare.normalize_outputs(
                    [recovered], [len(inputs)], args.output_len
                )
                receipt["recovery_comparison"] = compare.compare(
                    [expected], [actual], args.logprob_atol
                )
                capacities = run.wait_capacities(
                    group["trace_dir"],
                    stage,
                    args,
                    group["variant"],
                    group["ratio"],
                    request_ids=[recovery_rid],
                    require_idle=True,
                )
                run.enforce_fixed_compute(fixed_compute, capacities)
                if getattr(args, "mpu_fault_owner", None) is not None:
                    owners = run.ANALYZE._request_owners(
                        run.ANALYZE.load_traces(group["trace_dir"]),
                        {baseline_rid, recovery_rid},
                    )
                    if set(owners.values()) != {args.mpu_fault_owner}:
                        raise ValueError(
                            "Baseline and recovery must run on the faulted DP owner"
                        )
                receipt["idle_capacity"] = capacities
                facts = forward_facts(
                    run.ANALYZE.load_traces(group["trace_dir"]),
                    rid,
                    stage["tp_size"],
                    stage.get("attn_dp_size", 1),
                )
                if facts and getattr(args, "mpu_fault_owner", None) is not None:
                    if facts[0]["rank"] != args.mpu_fault_owner:
                        raise ValueError(
                            "Fault request did not execute on its requested DP owner"
                        )
                receipt["forward_facts"] = facts
                validate_fault(case, response, facts, len(inputs), args.output_len)
                if case == "retract_decode":
                    resumed = compare.normalize_outputs(
                        [response], [len(inputs)], args.output_len
                    )
                    receipt["resumed_comparison"] = compare.compare(
                        [expected], [resumed], args.logprob_atol
                    )
                receipt["status"], passed = "passed", True
                break
            except MissedPrefillWindow as error:
                receipt.update(status="window_missed", error=str(error))
            except BaseException as error:
                receipt.update(status="failed", error=repr(error))
                raise
            finally:
                run.COMMON.write_manifest(path, receipt)
        if not passed:
            raise ValueError(
                "Prefill abort window missed in all attempts; increase the long prompt or adjust chunk size. This case is not accepted."
            )


def run_cases(run, compare, stage, args, group, inputs):
    if stage.get("attn_dp_size", 1) == 1:
        return _run_owner_cases(run, compare, stage, args, group, inputs)
    group["lifecycle_by_attention_rank"] = {}
    group["lifecycle_results"] = []
    for owner in range(4):
        local_args = copy.copy(args)
        local_args.mpu_fault_owner = owner
        local_group = {**group, "id": f"{group['id']}-dp{owner}"}
        _run_owner_cases(run, compare, stage, local_args, local_group, inputs)
        group["lifecycle_by_attention_rank"][str(owner)] = local_group[
            "lifecycle_results"
        ]
        group["lifecycle_results"].extend(
            {**item, "attention_dp_rank": owner}
            for item in local_group["lifecycle_results"]
        )
