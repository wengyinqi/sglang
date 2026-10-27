"""Repeat uncertainty, explicit costs and SLO gates for capacity sweeps.

Only measured points are compared. Bootstrap intervals are descriptive and do
not remove run-order, hardware, workload or prefill-subtraction bias.
"""

import json
import math
import random
import statistics
from pathlib import Path


def number(value, positive=True):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(
            "Expected a finite positive number"
            if positive
            else "Expected a finite nonnegative number"
        )
    return value


def load_policy(path):
    policy = {"schema_version": 1, "minimum_repeats": 5, "slo": {}, "costs": []}
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(supplied) - {
            "schema_version",
            "minimum_repeats",
            "slo",
            "costs",
            "currency",
            "cost_basis",
            "source",
        }:
            raise ValueError("Unknown analysis policy field")
        policy.update(supplied)
    if (
        policy["schema_version"] != 1
        or type(policy["minimum_repeats"]) is not int
        or policy["minimum_repeats"] < 3
    ):
        raise ValueError(
            "Analysis requires schema_version=1 and at least three repeats"
        )
    allowed = {"p99_ttft_ms", "p99_tpot_ms", "mean_ttft_ms", "mean_tpot_ms"}
    if not isinstance(policy["slo"], dict) or set(policy["slo"]) - allowed:
        raise ValueError("Unsupported SLO; use measured TTFT/TPOT metrics")
    for value in policy["slo"].values():
        number(value)
    if policy["costs"] and (
        not policy.get("currency")
        or not policy.get("source")
        or policy.get("cost_basis") not in ("simulator_hardware", "hypothetical_mpu")
    ):
        raise ValueError(
            "Costs need currency, source and simulator_hardware/hypothetical_mpu basis"
        )
    if not isinstance(policy["costs"], list):
        raise ValueError("Costs must be a list of explicit configuration totals")
    seen = set()
    for point in policy["costs"]:
        key = (point["variant"], point["ratio"])
        if key in seen or key[0] not in ("cpu", "mpu", "hbm"):
            raise ValueError("Duplicate or invalid cost point")
        seen.add(key)
        if key[0] == "hbm":
            if key[1] is not None:
                raise ValueError("HBM cost point uses ratio=null")
        else:
            number(key[1])
        number(point["hourly_cost"])
    return policy


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        index - low
    )


def bootstrap(values, other=None):
    """Paired by repeat ID for gains; intervals over medians otherwise."""
    for value in values + (other or []):
        number(value)
    if not values or (other is not None and len(values) != len(other)):
        raise ValueError("Bootstrap requires nonempty aligned repeats")
    if len(values) < 3:
        return None
    rng = random.Random(1729)
    estimates = []
    for _ in range(2000):
        indices = [rng.randrange(len(values)) for _ in values]
        current = statistics.median(values[i] for i in indices)
        estimates.append(
            current
            if other is None
            else current / statistics.median(other[i] for i in indices) - 1
        )
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def enrich(summary, groups, policy):
    costs = {(p["variant"], p["ratio"]): p["hourly_cost"] for p in policy["costs"]}
    lookup = {(g["variant"], g["input_len"], g["ratio"]): g for g in groups}
    previous = {}
    for row in summary["rows"]:
        group = lookup[row["variant"], row["input_len"], row["ratio"]]
        cases = [c for c in group["cases"] if c["concurrency"] == row["concurrency"]]
        row["repeat_evidence"] = {
            "minimum_required": policy["minimum_repeats"],
            "status": "incomplete",
        }
        key = row["variant"], row["input_len"], row["concurrency"]
        prior = previous.get(key)
        previous[key] = None
        if not cases or any(c["status"] != "success" for c in cases):
            continue
        indexed = {c["repeat"]: c for c in cases}
        if len(indexed) != len(cases):
            raise ValueError(
                "Duplicate repeat IDs cannot be treated as independent samples"
            )
        ids = sorted(indexed)
        methods = {
            indexed[i]["analysis"]["decode_only_estimate"].get("method") for i in ids
        }
        if len(methods) > 1:
            raise ValueError(
                "Cannot mix TP and DP prefill subtraction methods in repeated samples"
            )
        rates = [
            indexed[i]["analysis"]["decode_only_estimate"][
                "decode_throughput_tokens_per_second"
            ]
            for i in ids
        ]
        actual = [
            indexed[i]["analysis"]["end_to_end"]["output_throughput_tokens_per_second"]
            for i in ids
        ]
        for value in rates + actual:
            number(value)
        enough = len(cases) >= policy["minimum_repeats"]
        row["repeat_evidence"].update(
            status="sufficient_for_policy" if enough else "insufficient_repeats",
            n=len(cases),
            decode_median_ci95=bootstrap(rates),
            actual_output_median_ci95=bootstrap(actual),
            decode_estimate_method=next(iter(methods)),
            mixed_ep_steps_by_repeat=[
                indexed[i]["analysis"].get("cuda_trace", {}).get("mixed_steps", 0)
                for i in ids
            ],
        )
        capacities = row.get("per_rank_capacity") or []
        effective = [p.get("effective_capacity_ratio") for p in capacities]
        row["effective_slot_ratios_by_rank"] = effective
        row["backing_to_compute_kv_byte_ratios_by_rank"] = [
            (
                p["backing_kv_bytes"] / p["compute_kv_bytes"]
                if p.get("backing_kv_bytes") is not None
                and p.get("compute_kv_bytes", 0) > 0
                else None
            )
            for p in capacities
        ]
        row["slo"] = {
            "status": "not_specified",
            "scope": "each successful repeat's measured request-latency metric",
        }
        if policy["slo"]:
            results = {}
            for field, limit in policy["slo"].items():
                values = [c["analysis"]["end_to_end"].get(field) for c in cases]
                if any(v is None for v in values):
                    results[field] = {"status": "missing_metric", "limit_ms": limit}
                else:
                    for value in values:
                        number(value, positive=False)
                    results[field] = {
                        "status": "passed" if max(values) <= limit else "failed",
                        "limit_ms": limit,
                        "worst_repeat_ms": max(values),
                    }
            states = {r["status"] for r in results.values()}
            row["slo"].update(
                status=(
                    "failed"
                    if "failed" in states
                    else "missing_metric"
                    if "missing_metric" in states
                    else "passed"
                ),
                metrics=results,
            )
        if prior and prior[1] == ids:
            prev_row, _, prev_rates, prev_effective = prior
            increased = len(effective) == len(prev_effective) > 0 and all(
                a is not None and b is not None and a > b
                for a, b in zip(effective, prev_effective)
            )
            interval = bootstrap(rates, prev_rates)
            row["marginal_decode_gain_ci95"] = interval
            if not increased:
                evidence = "allocation_not_increased_or_unknown"
            elif (
                not enough
                or prev_row["repeat_evidence"]["status"] != "sufficient_for_policy"
            ):
                evidence = "insufficient_repeats"
            elif interval[1] < 0:
                evidence = "regression"
            elif interval[1] < summary["plateau_threshold"]:
                evidence = "candidate_below_threshold"
            elif interval[0] >= summary["plateau_threshold"]:
                evidence = "gain_above_threshold"
            else:
                evidence = "inconclusive"
            row["decode_plateau_evidence"] = evidence
        previous[key] = row, ids, rates, effective
        cost = costs.get((row["variant"], row["ratio"]))
        row["cost"] = {"status": "not_supplied"}
        if cost is not None:
            row["cost"] = {
                "status": "calculated",
                "hourly_cost": cost,
                "currency": policy["currency"],
                "basis": policy["cost_basis"],
                "actual_output_tokens_per_currency": statistics.median(actual)
                * 3600
                / cost,
                "estimated_decode_tokens_per_currency": statistics.median(rates)
                * 3600
                / cost,
                "eligible_for_observed_comparison": enough
                and row["slo"]["status"] == "passed",
            }
    eligible = {}
    for row in summary["rows"]:
        if row.get("cost", {}).get("eligible_for_observed_comparison"):
            key = row["input_len"], row["concurrency"]
            if (
                key not in eligible
                or row["cost"]["actual_output_tokens_per_currency"]
                > eligible[key]["cost"]["actual_output_tokens_per_currency"]
            ):
                eligible[key] = row
    summary["analysis_policy"] = policy
    summary["best_observed_cost_points"] = [
        {k: row[k] for k in ("input_len", "concurrency", "variant", "ratio", "cost")}
        for row in eligible.values()
    ]
    summary["uncertainty_scope"] = {
        "method": "2000 deterministic paired-by-repeat bootstrap resamples; percentile 95% intervals",
        "winner_statistical_separation_proven": False,
        "global_or_economic_optimum_proven": False,
        "limitations": [
            "Small repeat counts give coarse intervals; defaults require five repeats for candidate conclusions.",
            "Intervals do not cover fixed run order, thermal drift, workload shifts or systematic prefill-subtraction bias.",
            "Explicit hourly costs describe the supplied physical configuration, not allocated KV ratio times a unit price.",
            "H200 simulator economics and a hypothetical MPU cost model are different inputs.",
        ],
    }
    return summary


def main(argv=None):
    import argparse
    import hashlib
    import importlib.util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("kind") != "benchmark":
        raise ValueError("Supply a capacity benchmark manifest")
    spec = importlib.util.spec_from_file_location(
        "mpu_capacity_run", Path(__file__).with_name("run.py")
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    report = run.summarize(
        manifest["groups"],
        manifest["settings"]["plateau_threshold"],
        load_policy(args.policy),
    )
    report["input_manifest"] = {
        "path": str(args.manifest.resolve()),
        "sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "status": manifest.get("status"),
        "runtime": manifest.get("runtime"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
