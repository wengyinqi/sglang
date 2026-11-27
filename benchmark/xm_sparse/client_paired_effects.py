"""Replay complete-request HTTP metrics without treating token gaps as repeats.

The caller supplies a saved, raw-replayed campaign analysis and exact expected
native source. This CPU derivation does not grant a model correctness gate.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = (
    "baseline",
    "fused",
    "bounds-on-demand",
    "bounds-prefetch",
    "combined",
    "combined-protected",
)
PAIRS = (
    ("baseline", "fused"),
    ("baseline", "bounds-on-demand"),
    ("bounds-on-demand", "bounds-prefetch"),
    ("fused", "combined"),
    ("combined", "combined-protected"),
    ("fused", "combined-protected"),
    ("baseline", "combined"),
    ("baseline", "combined-protected"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(left, right, message):
    require(
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9),
        message,
    )


def raw_metrics(raw, sample, expected_output):
    """Independently recompute full positive singleton-frame gaps and TPOT."""
    ids = raw.get("request_ids", [])
    count = len(ids)
    require(count > 0 and count == raw.get("completed"), "Incomplete request IDs")
    require(len(set(ids)) == count, "Duplicate request IDs")
    require(
        raw.get("input_lens") == [sample["input_len"]] * count, "Raw context mismatch"
    )
    require(
        raw.get("output_lens") == [expected_output] * count,
        "Raw output-length mismatch",
    )
    require(expected_output > 1, "TBT requires at least two output tokens")
    names = (
        "request_latencies",
        "ttfts",
        "request_start_monotonic_ns",
        "request_completion_monotonic_ns",
        "stream_frames",
        "generated_texts",
    )
    require(
        all(len(raw.get(name, [])) == count for name in names),
        "Incomplete raw request timing/text arrays",
    )
    client = sample["client_timing"]
    require(
        client.get("all_frames_single_token") is True,
        "Multi-token frames have no strict internal token timing",
    )
    require(len(client.get("requests", [])) == count, "Derived request count mismatch")
    gaps, tpots, ttfts = [], [], []
    for i, rid in enumerate(ids):
        request = client["requests"][i]
        require(request.get("request_id") == rid, "Derived request order mismatch")
        start, end = (
            raw["request_start_monotonic_ns"][i],
            raw["request_completion_monotonic_ns"][i],
        )
        require(
            type(start) is int and type(end) is int and 0 < start < end,
            "Unordered request monotonic boundaries",
        )
        previous_stamp, previous_count, stamps = start, 0, []
        frames = raw["stream_frames"][i]
        require(bool(frames), "Missing raw stream frames")
        for frame in frames:
            stamp = frame.get("received_monotonic_ns")
            cumulative = frame.get("cumulative_completion_tokens")
            require(
                type(stamp) is int and previous_stamp <= stamp <= end,
                "Unordered or out-of-bounds raw frame",
            )
            require(
                type(cumulative) is int and cumulative >= previous_count,
                "Invalid cumulative token count",
            )
            delta = cumulative - previous_count
            require(frame.get("new_tokens") == delta, "Raw token delta mismatch")
            require(delta in (0, 1), "Strict TBT cannot use multi-token frames")
            if delta:
                stamps.append(stamp)
            previous_stamp, previous_count = stamp, cumulative
        require(
            previous_count == expected_output
            and len(stamps) == expected_output
            and frames[-1].get("finish_reason") is not None,
            "Incomplete singleton token stream",
        )
        local_gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        require(all(value >= 0 for value in local_gaps), "Negative token arrival gap")
        latency, ttft = raw["request_latencies"][i] * 1000, raw["ttfts"][i] * 1000
        require(
            math.isfinite(latency) and math.isfinite(ttft) and 0 < ttft <= latency,
            "Invalid raw latency/TTFT",
        )
        tpot = (latency - ttft) / (expected_output - 1)
        close(latency, request["request_latency_ms"], "Request latency replay mismatch")
        close(ttft, request["ttft_ms"], "TTFT replay mismatch")
        close(tpot, request["request_average_tpot_ms"], "Request TPOT replay mismatch")
        close(
            statistics.mean(local_gaps),
            request["raw_chunk_gap_ms"]["mean"],
            "Per-request HTTP TBT replay mismatch",
        )
        require(
            request["raw_chunk_gap_ms"]["samples"] == expected_output - 1,
            "Per-request HTTP interval count mismatch",
        )
        gaps.extend(local_gaps)
        tpots.append(tpot)
        ttfts.append(ttft)
    mean_tbt, mean_tpot = statistics.mean(gaps), statistics.mean(tpots)
    require(math.isfinite(mean_tbt) and mean_tbt > 0, "Invalid positive HTTP mean TBT")
    close(
        mean_tbt,
        client["single_token_arrival_gap_ms"]["mean"],
        "Aggregate HTTP TBT replay mismatch",
    )
    require(
        client["single_token_arrival_gap_ms"]["samples"] == len(gaps),
        "Aggregate HTTP interval count mismatch",
    )
    close(mean_tpot, sample["request_average_tpot_ms"], "Sample TPOT replay mismatch")
    close(
        statistics.mean(ttfts), sample["client_ttft_ms"], "Sample TTFT replay mismatch"
    )
    require(
        type(raw.get("duration")) in (int, float)
        and math.isfinite(raw["duration"])
        and raw["duration"] > 0,
        "Invalid complete benchmark duration",
    )
    full_throughput = count * expected_output / raw["duration"]
    close(full_throughput, raw["output_throughput"], "Raw full throughput mismatch")
    close(
        full_throughput,
        sample["complete_request_output_tokens_per_second"],
        "Derived full throughput mismatch",
    )
    return {
        "http_mean_tbt_ms": mean_tbt,
        "request_tpot_ms": mean_tpot,
        "client_ttft_ms": statistics.mean(ttfts),
        "complete_output_tokens_per_s": full_throughput,
        "requests": count,
        "output_tokens_per_request": expected_output,
        "http_interval_count": len(gaps),
        "actual_decode_batch_distribution": sample["actual_decode_batch_distribution"],
        "request_rows_exact_text": raw["generated_texts"],
    }


def paired_effects(samples):
    grouped = defaultdict(dict)
    for row in samples:
        require(
            all(
                type(row.get(metric)) in (int, float)
                and math.isfinite(row[metric])
                and row[metric] > 0
                for metric in ("http_mean_tbt_ms", "request_tpot_ms")
            ),
            "Paired HTTP TBT/TPOT must be finite and positive",
        )
        key = (
            row["input_len"],
            row["client_concurrency"],
            row["repeat"],
            row["graph_enabled"],
        )
        require(row["arm"] not in grouped[key], "Duplicate paired sample")
        grouped[key][row["arm"]] = row
    effects, metrics = [], []
    contexts = sorted({(k[0], k[1], k[3]) for k in grouped})
    for context in contexts:
        repeats = sorted(k[2] for k in grouped if (k[0], k[1], k[3]) == context)
        require(
            repeats == [0, 1, 2],
            "Exactly three independent paired repetitions required",
        )
        cases = [grouped[context[0], context[1], r, context[2]] for r in repeats]
        require(
            all(set(case) == set(ARMS) for case in cases),
            "Incomplete six-arm paired repeat",
        )
        for case in cases:
            first = case["baseline"]["request_rows_exact_text"]
            require(
                all(row["request_rows_exact_text"] == first for row in case.values()),
                "Benchmark generated texts differ across paired arms",
            )
        for arm in ARMS:
            rows = [case[arm] for case in cases]
            metrics.append(
                {
                    "arm": arm,
                    "input_len": context[0],
                    "client_concurrency": context[1],
                    "graph_enabled": context[2],
                    "repeats": 3,
                    **{
                        name: statistics.mean(row[name] for row in rows)
                        for name in (
                            "http_mean_tbt_ms",
                            "request_tpot_ms",
                            "client_ttft_ms",
                            "complete_output_tokens_per_s",
                        )
                    },
                }
            )
        for left, right in PAIRS:
            for metric in ("http_mean_tbt_ms", "request_tpot_ms"):
                changes = [
                    (case[left][metric] - case[right][metric])
                    / case[left][metric]
                    * 100
                    for case in cases
                ]
                mean = statistics.mean(changes)
                half = 4.302652729911275 * statistics.stdev(changes) / math.sqrt(3)
                effects.append(
                    {
                        "baseline": left,
                        "changed": right,
                        "metric": metric,
                        "input_len": context[0],
                        "client_concurrency": context[1],
                        "graph_enabled": context[2],
                        "paired_repeat_reduction_percent": mean,
                        "confidence_interval_95_percent": [mean - half, mean + half],
                        "paired_repeat_values_percent": changes,
                        "repeats": 3,
                        "df": 2,
                    }
                )
    return metrics, effects


def derive(args):
    analyzer_path = Path(args.tool_root) / "benchmark/xm_sparse/analyze.py"
    spec = importlib.util.spec_from_file_location("paired_raw_analyzer", analyzer_path)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    campaign, samples, artifacts, excluded, commits = analyzer.collect_campaign(
        args.campaign, evidence_root=args.evidence_root
    )
    require(
        campaign.get("status") == "success" and campaign["phase"] == "performance",
        "Only completed formal performance campaigns are accepted",
    )
    require(not excluded, "Formal campaign has excluded cases")
    require(commits == [args.expected_native_source], "Native source mismatch")
    require(
        campaign["source_fingerprint"]["sha256"] == args.expected_core_sha256,
        "Production core fingerprint mismatch",
    )
    for case in campaign["cases"]:
        command = case.get("command", [])
        require(
            command.count("--output-len") == 1
            and command.index("--output-len") + 1 < len(command)
            and command[command.index("--output-len") + 1] == str(args.output_len),
            "Campaign command output length mismatch",
        )
    require(
        sorted(set(s["input_len"] for s in samples)) == sorted(args.input_lens),
        "Campaign input bucket mismatch",
    )
    require(
        all(s["client_concurrency"] == args.client_concurrency for s in samples),
        "Campaign concurrency mismatch",
    )
    saved = read(args.saved_analysis)
    require(
        saved.get("status") == "analyzed" and saved.get("excluded_cases") == [],
        "Saved analysis incomplete",
    )
    public = [{k: v for k, v in s.items() if k != "_intervals"} for s in samples]
    require(
        analyzer.equal(public, saved.get("samples")),
        "Saved samples differ from raw replay",
    )
    require(
        saved.get("source_fingerprint") == campaign["source_fingerprint"]
        and saved.get("native_source_commits") == commits
        and saved.get("workload_identity") == campaign["workload_identity"],
        "Saved source/workload binding mismatch",
    )
    raw_root = (
        Path(args.evidence_root) if args.evidence_root else Path(args.campaign).parent
    )
    rows, raw_identities = [], []
    for sample in samples:
        base = raw_root / sample["case_id"] / "evidence"
        native = read(base / "manifest.json")
        require(
            native.get("settings", {}).get("output_len") == args.output_len,
            "Native manifest output length mismatch",
        )
        matches = []
        for group in native["groups"]:
            if group["input_len"] != sample["input_len"]:
                continue
            for case in group["cases"]:
                if case["concurrency"] == sample["client_concurrency"]:
                    matches.append(base / Path(case["result_path"]).name)
        require(len(matches) == 1, "Ambiguous per-sample raw result")
        path = matches[0]
        lines = path.read_text(encoding="utf-8").splitlines()
        require(len(lines) == 1, "Expected exactly one raw result row")
        raw = json.loads(lines[0])
        row = {
            k: sample[k]
            for k in (
                "arm",
                "case_id",
                "input_len",
                "client_concurrency",
                "repeat",
                "graph_enabled",
            )
        }
        row.update(raw_metrics(raw, sample, args.output_len))
        rows.append(row)
        raw_identities.append(identity(path))
    metrics, effects = paired_effects(rows)
    return {
        "kind": "cpu_raw_replayed_complete_HTTP_paired_effects",
        "status": "passed",
        "source_fingerprint": campaign["source_fingerprint"],
        "native_source_commits": commits,
        "workload_identity": campaign["workload_identity"],
        "input_artifacts": [
            identity(args.campaign),
            identity(args.saved_analysis),
            identity(analyzer_path),
        ],
        "raw_replay_artifacts": artifacts,
        "raw_HTTP_artifacts": raw_identities,
        "samples": [
            {k: v for k, v in row.items() if k != "request_rows_exact_text"}
            for row in rows
        ],
        "metrics": metrics,
        "effects": effects,
        "GPU_execution": False,
        "model_correctness_proof": False,
        "metric_scope": "Complete measured HTTP positive singleton-frame arrival gaps exclude the first token, include request edges and any admission/scheduling effects. TPOT uses completion latency minus TTFT divided by output tokens minus one. Not the matched stable scheduler interval metric.",
        "ci_scope": "Three paired repeat percentages, df2 Student t interval. Token gaps are not independent repeats. Execution order is not position-balanced; no multiple-comparison correction.",
        "correctness_scope": "Generated texts match across paired benchmark arms; these timing requests do not contain complete greedy token-ID/logprob proof. Separate same-source CPU/HBM eager/Graph model gates remain required.",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "tool-root",
        "campaign",
        "saved-analysis",
        "expected-native-source",
        "expected-core-sha256",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument("--input-lens", nargs="+", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--client-concurrency", type=int, required=True)
    args = parser.parse_args()
    result = derive(args)
    result["derivation_tool"] = identity(__file__)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "samples": len(result["samples"]),
                "output": str(output),
                "effects": len(result["effects"]),
            }
        )
    )


if __name__ == "__main__":
    main()
