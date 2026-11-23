"""Verify a stable multi-request Qwen decode Nsight capture, conservatively.

This does not replace the device-role audit or deployment-source semantics audit.
It accepts their identities as inputs and binds complete graph calls to a
conservative same-request-set forward interval around the observed trigger.
"""

import argparse
import hashlib
import importlib.util
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

DECODE_ATTENTION_KERNEL = (
    "void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<"
    "(flashinfer::MaskMode)0, (unsigned int)16, (unsigned int)1, (unsigned int)2, "
    "(unsigned int)8, (unsigned int)8, (unsigned int)1, (unsigned int)4, "
    "(flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, "
    "float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, "
    "PagedParams>(T2)"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify(database, profile, trace, semantics, *, layers=48):
    require(
        profile.get("status") == "captured_pending_role_analysis",
        "Require an actual completed profiler capture",
    )
    identities = [
        a for a in profile.get("artifacts", []) if Path(a["path"]).suffix == ".sqlite"
    ]
    require(
        len(identities) == 1 and sha(database) == identities[0]["sha256"],
        "SQLite identity differs from captured evidence",
    )
    require(
        semantics.get("status") == "verified_from_deployed_sources"
        and isinstance(semantics.get("source_commit"), str)
        and bool(semantics["source_commit"])
        and semantics.get("query_rows_per_request") == 1
        and semantics.get("tensor_core_decode_uses_paged_prefill_module") is True
        and semantics.get("source_commit")
        == profile.get("runtime", {}).get("git_commit")
        and semantics.get("receipt_sha256"),
        "Missing matching deployment-source decode semantics",
    )
    attention_name = semantics.get("allowed_attention_kernel_name")
    require(
        attention_name == DECODE_ATTENTION_KERNEL,
        "Unknown decode attention specialization",
    )
    request = profile.get("profile_request", {})
    steps = profile.get("configured_decode_steps")
    require(
        type(steps) is int and 1 <= steps <= 512 and request.get("num_steps") == steps,
        "Configured profile steps differ",
    )
    require(
        request.get("activities") == ["CUDA_PROFILER"]
        and request.get("profile_by_stage") is True
        and request.get("profile_stages") == ["decode"],
        "Require dedicated decode trigger",
    )
    command = profile.get("server_command", [])
    require(
        "--capture-range=cudaProfilerApi" in command
        and "--cuda-graph-trace=node" in command,
        "Missing bounded CUDA graph node capture",
    )
    ids = profile.get("profiled_request_ids", [])
    count = profile.get("workload", {}).get("concurrency")
    require(
        type(count) is int
        and count > 0
        and len(ids) == count
        and len(set(ids)) == count
        and all(isinstance(rid, str) and rid for rid in ids),
        "Profile workload has missing or duplicate request identities",
    )
    require(
        profile.get("completion_tokens") == count * profile["workload"]["output_len"],
        "Incomplete profiled output tokens",
    )
    require(
        type(layers) is int and layers == 48,
        "This verifier supports the pinned 48-layer Qwen model",
    )
    session = trace[0] if trace else {}
    require(
        session.get("type") == "session"
        and session.get("tp_size") == 1
        and session.get("pp_size") == 1
        and session.get("dp_size") == 1
        and session.get("disable_overlap_schedule") is True,
        "Stable capture requires one non-overlap compute scheduler",
    )
    require(
        session.get("disaggregation_mode") == "null"
        and session.get("disable_radix_cache") is True
        and session.get("speculative_algorithm") is None
        and session.get("enable_mixed_chunk") is False,
        "Unexpected PD/prefix/speculative/mixed-chunk capture",
    )
    forwards = [r for r in trace if r.get("type") == "forward"]
    require(
        [r.get("sequence") for r in forwards] == list(range(len(forwards))),
        "Incomplete or reordered forward trace",
    )
    require(
        trace[-1].get("type") == "idle_flush"
        and trace[-1].get("last_sequence") == len(forwards) - 1,
        "Profile trace did not flush completely at idle",
    )
    selected = [r for r in forwards if set(ids).intersection(r.get("request_ids", []))]
    prefill, decode = Counter(), Counter()
    for row in selected:
        require(
            row.get("success") is True
            and set(row["request_ids"]) <= set(ids)
            and len(set(row["request_ids"])) == len(row["request_ids"]),
            "Failed, mixed or duplicate profile forward",
        )
        if row["forward_mode"] == "extend":
            lengths = row.get("extend_lens")
            require(
                isinstance(lengths, list)
                and len(lengths) == len(row["request_ids"])
                and all(type(n) is int and n > 0 for n in lengths),
                "Invalid profile prefill chunks",
            )
            prefill.update(dict(zip(row["request_ids"], lengths)))
        elif row["forward_mode"] == "decode":
            decode.update(row["request_ids"])
        else:
            raise ValueError("Unknown profile forward mode")
    require(
        dict(prefill) == dict.fromkeys(ids, profile["workload"]["input_len"]),
        "Profile input coverage differs",
    )
    require(
        dict(decode) == dict.fromkeys(ids, profile["workload"]["output_len"] - 1),
        "Profile decode output coverage differs",
    )
    before, after = profile.get("trigger_progress_before"), profile.get(
        "trigger_progress_after_ack"
    )
    require(
        isinstance(before, dict) and isinstance(after, dict),
        "Missing observed stable trigger boundaries",
    )
    stable_ids = set(before.get("request_ids", []))
    minimum = profile.get("stable_trigger", {}).get("min_consecutive_decode_steps", 0)
    require(
        type(minimum) is int
        and minimum >= 16
        and before.get("consecutive_stable_decode_steps", 0) >= minimum,
        "Stable trigger did not observe at least 16 actual decode steps",
    )
    for snapshot in (before, after):
        require(
            snapshot.get("success") is True
            and snapshot.get("forward_mode") == "decode"
            and set(snapshot.get("request_ids", [])) == stable_ids
            and stable_ids <= set(ids)
            and snapshot.get("batch_size") == len(stable_ids)
            and stable_ids,
            "Trigger crossed a request-set change or prefill",
        )
    first, last = before.get("sequence"), after.get("sequence")
    require(
        type(first) is int and type(last) is int and 0 <= first <= last,
        "Invalid profile trigger sequence bounds",
    )
    required_batch = profile.get("stable_trigger", {}).get("required_actual_batch_size")
    require(
        required_batch is None
        or (type(required_batch) is int and required_batch == len(stable_ids)),
        "Observed batch differs from required profile batch size",
    )
    history = [r for r in forwards if first - minimum + 1 <= r["sequence"] <= first]
    require(
        len(history) == minimum
        and all(
            r.get("success") is True
            and r["forward_mode"] == "decode"
            and set(r["request_ids"]) == stable_ids
            for r in history
        )
        and len({r.get("stream_id") for r in history}) == 1
        and history[0].get("stream_id") is not None,
        "Stable trigger counter is not confirmed by actual consecutive decode forwards",
    )
    # Poll/control requests can span several forwards. Reject any prefill or
    # request-set change across the whole possible capture interval, including
    # a conservative extra two closing steps. Do not claim an exact event↔host
    # timestamp correspondence that the exporter did not record.
    stop = last + steps + 2
    guard = [r for r in forwards if first <= r["sequence"] <= stop]
    require(len(guard) == stop - first + 1, "Incomplete conservative capture interval")
    require(
        all(
            r.get("success") is True
            and r["forward_mode"] == "decode"
            and set(r["request_ids"]) == stable_ids
            for r in guard
        ),
        "Prefill or changed request membership in conservative capture interval",
    )
    require(
        len({r.get("stream_id") for r in guard}) == 1
        and guard[0].get("stream_id") is not None,
        "Capture interval crossed compute streams",
    )
    require(
        guard[0]["stream_id"] == history[0]["stream_id"],
        "Capture interval differs from the stable trigger stream",
    )
    for snapshot in (before, after):
        row = forwards[snapshot["sequence"]]
        require(
            snapshot.get("stream_id") == row["stream_id"]
            and snapshot.get("launch_monotonic_ns") == row.get("launch_monotonic_ns"),
            "Trigger metadata does not match its actual forward sequence",
        )
    with closing(
        sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        launches = list(
            conn.execute(
                "SELECT r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE 'cudaGraphLaunch%' ORDER BY r.start"
            )
        )
        named = list(
            conn.execute(
                "SELECT k.correlationId,k.graphId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id"
            )
        )
    actual = len(launches)
    require(
        actual in (steps, steps + 1), "Actual graph count differs from bounded trigger"
    )
    require(
        named
        and not any(
            "prefill" in name.lower() and name != attention_name for _, _, name in named
        ),
        "Unknown or genuine prefill kernel in decode capture",
    )
    launch_ids = {correlation for correlation, in launches}
    require(len(launch_ids) == actual, "Duplicate runtime graph correlation")
    scores = Counter(
        correlation for correlation, _, name in named if name == "_quest_score_kernel"
    )
    attention = Counter(
        correlation for correlation, _, name in named if name == attention_name
    )
    require(
        set(scores) == set(attention) == launch_ids
        and set(scores.values()) == set(attention.values()) == {layers},
        "Missing layer or wrong launch association in score/attention graph",
    )
    graph_ids = {
        graph
        for _, graph, name in named
        if name in ("_quest_score_kernel", attention_name)
    }
    require(
        len(graph_ids) == 1 and all(type(g) is int and g > 0 for g in graph_ids),
        "Score/attention do not belong to one stable CUDA graph",
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "sqlite_sha256": sha(database),
        "source_semantics_receipt_sha256": semantics["receipt_sha256"],
        "configured_decode_steps": steps,
        "actual_decode_graph_launches": actual,
        "boundary_step_included": actual == steps + 1,
        "stable_request_ids": sorted(stable_ids),
        "actual_batch_size": len(stable_ids),
        "conservative_trace_first_sequence": first,
        "conservative_trace_last_sequence": stop,
        "no_prefill_or_membership_change_in_conservative_interval": True,
        "quest_score_calls": sum(scores.values()),
        "decode_attention_calls": sum(attention.values()),
        "calls_per_graph": layers,
        "graph_id": next(iter(graph_ids)),
        "whole_request_input_tokens_verified": sum(prefill.values()),
        "whole_request_decode_tokens_verified": sum(decode.values()),
        "actual_decode_scope_verified": True,
        "scope_limitations": "Request membership is bound conservatively through trigger snapshots and complete forward sequences; this is not an exact host↔Nsight timestamp join. Device-role identity must pass its separate audit.",
    }


def multirank_scope(database, profile, traces, semantics):
    """Audit trigger scope without guessing model-specific kernel semantics."""
    spec = importlib.util.spec_from_file_location(
        "scope_dp_contract", Path(__file__).with_name("multirank.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = profile.get("profiled_request_ids", [])
    require(
        profile.get("status") == "captured_pending_role_analysis"
        and ids
        and len(ids) == len(set(ids)),
        "Incomplete multirank capture",
    )
    identities = [
        item
        for item in profile.get("artifacts", [])
        if Path(item["path"]).suffix == ".sqlite"
    ]
    require(
        len(identities) == 1 and sha(database) == identities[0]["sha256"],
        "SQLite capture identity differs",
    )
    native = module.native()
    complete = native._dp_traces(traces)
    owners = native._request_owners(complete, set(ids))
    require(
        set(owners.values()) == set(range(4)),
        "Profile must cover all four Attention owners",
    )
    before, after = profile.get("trigger_progress_before", {}), profile.get(
        "trigger_progress_after_ack", {}
    )
    require(
        before.get("request_owners") == owners
        and after.get("request_owners") == owners
        and type(before.get("sequence")) is int
        and after.get("sequence", -1) >= before["sequence"],
        "Missing aligned four-rank live trigger evidence",
    )
    _, _, groups, streams = module.aligned_steps(complete, set(ids))
    pure = []
    for group in groups:
        if before["sequence"] <= group[0]["sequence"] <= after["sequence"]:
            require(
                set(group[0]["dp_forward_modes"]) == {"decode"},
                "Trigger boundary includes prefill/idle owner",
            )
            require(
                all(
                    sorted(group[rank]["request_ids"])
                    == sorted(rid for rid, owner in owners.items() if owner == rank)
                    for rank in range(4)
                ),
                "Trigger request ownership changed",
            )
            pure.append(group[0]["sequence"])
    require(pure, "No trace step contains the live trigger")
    # Native HBM DeepSeek/GLM decode kernels differ from the Qwen verifier.
    # Binding Nsight calls to a model phase needs a deployed source/rank-device
    # audit and full model-specific kernel semantics; pending remains explicit.
    return {
        "schema_version": 1,
        "status": "trigger_verified_pending_model_kernel_semantics",
        "parallel_strategy": "attention_dp4_ep4",
        "request_owners": owners,
        "trigger_sequences": pure,
        "per_rank_streams": streams,
        "model_family": profile.get("stage", {}).get("family"),
        "semantics_status": semantics.get("status"),
        "decode_kernel_scope_verified": False,
        "required_next_evidence": [
            "Deployed model-source decode kernel semantics",
            "All rank PID/CUDA device/Nsight correlation identities",
            "Bounded complete graph/eager calls at the verified trigger interval",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = sorted(args.trace_dir.glob("forward-*.jsonl"))
    require(
        len(traces) in (1, 4) and not args.output.exists(),
        "Require one or four complete traces and a new output path",
    )
    records = [
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for path in traces
    ]
    report = (multirank_scope if len(traces) == 4 else verify)(
        args.sqlite,
        json.loads(args.profile.read_text(encoding="utf-8")),
        records if len(traces) == 4 else records[0],
        json.loads(args.semantics.read_text(encoding="utf-8")),
    )
    report["semantics_json_sha256"] = sha(args.semantics)
    report["profile_json_sha256"] = sha(args.profile)
    report["forward_trace_sha256"] = (
        {path.name: sha(path) for path in traces}
        if len(traces) == 4
        else sha(traces[0])
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
