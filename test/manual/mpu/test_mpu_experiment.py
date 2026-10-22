"""CPU checks for MPU experiment controls and conservative acceptance accounting."""

import copy
import importlib.util
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load("mpu_experiment", "benchmark/hisparse_mpu/run.py")
CHECK = load("mpu_correctness", "benchmark/hisparse_mpu/correctness.py")


def args():
    return RUN.parser().parse_args(
        [
            "--model",
            "/models/qwen",
            "--output-len",
            "80",
            "--max-total-tokens",
            "4096",
            "--top-k",
            "128",
            "--device-buffer-size",
            "256",
            "--chunked-prefill-size",
            "512",
            "--output-dir",
            "unused",
            "--capacity-ratios",
            "1",
            "2",
            "4",
        ]
    )


class TestMPUExperiment(unittest.TestCase):
    def test_sweep_fixes_compute_budget_and_scheduler_limits(self):
        settings = args()
        RUN.validate(settings)
        plan = RUN.make_plan(settings)
        self.assertEqual(len(plan), len(settings.input_lens) * 4)
        for group in plan:
            command = group["server_command"]
            self.assertEqual(command[command.index("--max-total-tokens") + 1], "4096")
            self.assertEqual(command[command.index("--max-running-requests") + 1], "4")
            self.assertIn("--disable-overlap-schedule", command)
            self.assertNotIn("--revision", command)
            self.assertIn("--output-details", group["cases"][0]["command"])
        self.assertEqual({case["concurrency"] for case in plan[0]["cases"]}, {1, 2, 4})

    def test_rejects_impossible_or_invalid_controls(self):
        for key, value in [
            ("capacity_ratios", [float("nan")]),
            ("mpu_device_ids", [0]),
            ("mpu_device_ids", [1, 1]),
            ("output_len", 1),
            ("max_total_tokens", 4095),
            ("trace_timeout", 0),
        ]:
            with self.subTest(key=key, value=value):
                settings = args()
                setattr(settings, key, value)
                with self.assertRaises(ValueError):
                    RUN.validate(settings)

    def test_capacity_requires_actual_allocation_and_topology(self):
        settings = args()
        capacity = {
            "backing_store": "mpu",
            "compute_device_id": 0,
            "storage_device_ids": [1],
            "requested_capacity_ratio": 2,
            "physical_compute_slots": 4096,
            "usable_mpu_slots": 8192,
            "allocated_mpu_bytes": 10000,
            "compute_kv_bytes": 5000,
            "effective_capacity_ratio": 2,
        }
        info = {
            "internal_states": [
                {
                    "hisparse_mpu_capacity": capacity,
                    "effective_max_running_requests_per_dp": 4,
                }
            ]
        }
        self.assertEqual(RUN.verify_capacity(info, settings, "mpu", 2), capacity)
        for key, value in [
            ("physical_compute_slots", 8192),
            ("usable_mpu_slots", 8191),
            ("allocated_mpu_bytes", 0),
            ("effective_capacity_ratio", 1),
            ("storage_device_ids", [2]),
        ]:
            with self.subTest(key=key):
                bad = copy.deepcopy(info)
                bad["internal_states"][0]["hisparse_mpu_capacity"][key] = value
                with self.assertRaises(ValueError):
                    RUN.verify_capacity(bad, settings, "mpu", 2)
        with self.assertRaises(ValueError):
            RUN.verify_capacity(
                {"internal_states": [{"max_total_num_tokens": 4000}]},
                settings,
                "quest",
                None,
            )

    def test_failed_repeat_never_produces_a_plateau_claim(self):
        def group(ratio, values):
            return {
                "mode": "mpu",
                "input_len": 4096,
                "ratio": ratio,
                "cases": [
                    {
                        "concurrency": 4,
                        "status": "success" if value else "failed",
                        "analysis": {
                            "decode_only_estimate": {
                                "decode_throughput_tokens_per_second": value
                            },
                            "end_to_end": {
                                "mean_ttft_ms": 100,
                                "mean_tpot_ms": 2,
                                "output_throughput_tokens_per_second": value,
                            },
                        },
                    }
                    for value in values
                ],
            }

        rows = RUN.summarize(
            [
                group(1, [100, 100]),
                group(2, [102, 104]),
                group(3, [None, 103]),
                group(4, [104, 104]),
            ],
            0.05,
        )["rows"]
        self.assertAlmostEqual(rows[1]["relative_decode_gain"], 0.03)
        self.assertTrue(rows[1]["gain_below_threshold"])
        self.assertNotIn("median_decode_estimate", rows[2])
        self.assertNotIn("relative_decode_gain", rows[3])

    def test_correctness_plan_covers_sparse_long_chunked_and_slot_reuse(self):
        settings = args()
        plan = CHECK.make_plan(settings)
        self.assertEqual(len(plan), 8)
        for group in plan:
            self.assertEqual(group["rounds"][0], group["rounds"][2])
            self.assertGreater(min(group["rounds"][0]), settings.device_buffer_size)
            self.assertLess(settings.chunked_prefill_size, min(group["rounds"][0]))
            self.assertLessEqual(
                group["planned_peak_page_rounded_slots"], settings.max_total_tokens
            )
        settings.max_total_tokens = 2048
        with self.assertRaises(ValueError):
            CHECK.make_plan(settings)

    def test_normalization_and_equivalence_reject_partial_or_wrong_outputs(self):
        output = [
            {
                "meta_info": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-1.0, 3, None], [-2.0, 4, None]],
                }
            }
        ]
        normalized = CHECK.normalize_outputs(output, [7], 2)
        self.assertTrue(
            CHECK.compare([normalized], [normalized], 0.02)["exact_greedy_token_match"]
        )
        with self.assertRaises(ValueError):
            CHECK.normalize_outputs(output, [7, 7], 2)
        with self.assertRaises(ValueError):
            CHECK.normalize_outputs(output, [7], 3)
        for field, value in [
            ("tokens", [3, 5]),
            ("logprobs", [-1.0, -2.03]),
            ("logprobs", [-1.0, math.nan]),
        ]:
            changed = copy.deepcopy(normalized)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                CHECK.compare([normalized], [changed], 0.02)

    def test_correctness_waits_for_complete_mpu_reservation_release(self):
        settings = args()
        busy = {"reserved_mpu_slots": 80, "used_mpu_slots": 1}
        idle = {"reserved_mpu_slots": 0, "used_mpu_slots": 0}
        with patch.object(CHECK.RUN, "request_json", return_value={}), patch.object(
            CHECK.RUN, "verify_capacity", side_effect=[busy, idle]
        ), patch.object(CHECK.time, "sleep"):
            self.assertEqual(CHECK.wait_mpu_release(settings, 1), idle)
        with patch.object(CHECK.RUN, "request_json", return_value={}), patch.object(
            CHECK.RUN, "verify_capacity", return_value=busy
        ), self.assertRaisesRegex(ValueError, "remain after request completion"):
            CHECK.wait_mpu_release(settings, 1, timeout=0)

    def test_benchmark_analyzes_trace_before_stopping_owned_server(self):
        settings = args()
        settings.modes, settings.input_lens, settings.concurrencies = (
            ["quest"],
            [4096],
            [1],
        )
        settings.repeats = 1
        events = []
        result = {
            "completed": settings.num_prompts,
            "total_output_tokens": settings.num_prompts * settings.output_len,
            "input_lens": [4096] * settings.num_prompts,
            "output_lens": [settings.output_len] * settings.num_prompts,
            "output_throughput": 100,
            "mean_ttft_ms": 10,
            "mean_tpot_ms": 2,
        }
        analysis = {
            "decode_only_estimate": {"decode_throughput_tokens_per_second": 110},
            "end_to_end": {
                "mean_ttft_ms": 10,
                "mean_tpot_ms": 2,
                "output_throughput_tokens_per_second": 100,
            },
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            settings.output_dir = Path(directory)
            manifest = {"groups": RUN.make_plan(settings)}
            stack.enter_context(patch.object(RUN, "require_cuda"))
            stack.enter_context(
                patch.object(RUN, "start_server", return_value=object())
            )
            stack.enter_context(patch.object(RUN.BASE, "wait_ready"))
            stack.enter_context(
                patch.object(
                    RUN,
                    "request_json",
                    return_value={
                        "internal_states": [
                            {
                                "max_total_num_tokens": 4096,
                                "effective_max_running_requests_per_dp": 1,
                            }
                        ]
                    },
                )
            )
            stack.enter_context(patch.object(RUN.subprocess, "run"))
            stack.enter_context(
                patch.object(RUN.ANALYZE, "read_jsonl", return_value=[result])
            )
            stack.enter_context(
                patch.object(
                    RUN.ANALYZE,
                    "analyze_when_ready",
                    side_effect=lambda *a: events.append("analyze") or analysis,
                )
            )
            stack.enter_context(
                patch.object(
                    RUN.BASE,
                    "stop_server",
                    side_effect=lambda *a: events.append("stop"),
                )
            )
            self.assertTrue(
                RUN.execute(settings, manifest, settings.output_dir / "manifest.json")
            )
        self.assertEqual(events, ["analyze", "stop"])


if __name__ == "__main__":
    unittest.main()
