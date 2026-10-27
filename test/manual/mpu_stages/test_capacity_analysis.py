"""Synthetic statistical fixtures, not measured GPU/cost results."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/mpu_stages" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAP, RUN = load("capacity_analysis"), load("run")


def group(ratio, rates, effective=None):
    return {
        "variant": "mpu",
        "input_len": 8192,
        "ratio": ratio,
        "capacity": [
            {
                "effective_capacity_ratio": effective or ratio,
                "backing_kv_bytes": 100 * ratio,
                "compute_kv_bytes": 100,
            }
        ],
        "cases": [
            {
                "status": "success",
                "concurrency": 1,
                "repeat": i,
                "analysis": {
                    "decode_only_estimate": {
                        "decode_throughput_tokens_per_second": rate
                    },
                    "end_to_end": {
                        "output_throughput_tokens_per_second": rate / 2,
                        "mean_ttft_ms": 10,
                        "mean_tpot_ms": 5,
                        "p99_ttft_ms": 12,
                        "p99_tpot_ms": 6,
                    },
                },
            }
            for i, rate in enumerate(rates)
        ],
    }


class CapacityTests(unittest.TestCase):
    def test_no_costs_or_slo_never_produces_an_economic_winner(self):
        report = RUN.summarize(
            [group(1, [100] * 5), group(2, [101] * 5)], 0.05, CAP.load_policy(None)
        )
        self.assertEqual(
            report["rows"][1]["decode_plateau_evidence"], "candidate_below_threshold"
        )
        self.assertEqual(report["best_observed_cost_points"], [])
        self.assertFalse(
            report["uncertainty_scope"]["global_or_economic_optimum_proven"]
        )

    def test_small_samples_noise_and_unchanged_capacity_cannot_prove_plateau(self):
        fixtures = [
            ([group(1, [100] * 3), group(2, [101] * 3)], "insufficient_repeats"),
            (
                [group(1, [100] * 5), group(2, [101] * 5, effective=1)],
                "allocation_not_increased_or_unknown",
            ),
            ([group(1, [100] * 5), group(2, [90, 95, 100, 110, 120])], "inconclusive"),
        ]
        for groups, status in fixtures:
            report = RUN.summarize(groups, 0.05, CAP.load_policy(None))
            self.assertEqual(report["rows"][1]["decode_plateau_evidence"], status)

    def test_explicit_cost_and_all_repeat_slo_gates(self):
        policy = {
            **CAP.load_policy(None),
            "currency": "fixture_units",
            "cost_basis": "hypothetical_mpu",
            "source": "synthetic test only",
            "slo": {"p99_tpot_ms": 7},
            "costs": [
                {"variant": "mpu", "ratio": 1, "hourly_cost": 1},
                {"variant": "mpu", "ratio": 2, "hourly_cost": 2},
            ],
        }
        groups = [group(1, [100] * 5), group(2, [110] * 5)]
        report = RUN.summarize(groups, 0.05, policy)
        self.assertEqual(report["best_observed_cost_points"][0]["ratio"], 1)
        self.assertEqual(
            report["rows"][0]["cost"]["actual_output_tokens_per_currency"], 180000
        )
        groups[0]["cases"][0]["analysis"]["end_to_end"]["p99_tpot_ms"] = 8
        groups[1]["cases"][0]["analysis"]["end_to_end"].pop("p99_tpot_ms")
        report = RUN.summarize(groups, 0.05, policy)
        self.assertEqual(report["best_observed_cost_points"], [])
        self.assertEqual(report["rows"][0]["slo"]["status"], "failed")
        self.assertEqual(report["rows"][1]["slo"]["status"], "missing_metric")

    def test_duplicate_repeats_and_incomplete_runs_are_not_samples(self):
        g = group(1, [100] * 5)
        g["cases"][1]["repeat"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate repeat"):
            RUN.summarize([g], 0.05, CAP.load_policy(None))
        g = group(1, [100] * 5)
        g["cases"][1]["status"] = "failed"
        report = RUN.summarize([g], 0.05, CAP.load_policy(None))
        self.assertEqual(report["rows"][0]["repeat_evidence"]["status"], "incomplete")

    def test_policy_rejects_unlabeled_or_nonpositive_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            base = {
                "currency": "fixture",
                "source": "synthetic",
                "cost_basis": "simulator_hardware",
                "costs": [{"variant": "mpu", "ratio": 2, "hourly_cost": 1}],
            }
            for change in (
                {"cost_basis": None},
                {"source": ""},
                {"minimum_repeats": 1},
                {"costs": [{"variant": "mpu", "ratio": 2, "hourly_cost": 0}]},
            ):
                path.write_text(json.dumps({**base, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    CAP.load_policy(path)
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(CAP.load_policy(path)["cost_basis"], "simulator_hardware")


if __name__ == "__main__":
    unittest.main()
