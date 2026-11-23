"""CPU evidence contracts for real-batch and logical-history comparisons."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "batch_cost_test", ROOT / "benchmark/mpu_stages/batch_cost.py"
)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def identity(name="a", hot=2048):
    return {
        "case_id": name,
        "source_commit": "a" * 40,
        "model_revision": "b" * 40,
        "runtime_image": "sha256:" + "c" * 64,
        "gpu_uuid": "GPU-test",
        "family": "qwen",
        "input_snapshot_sha256": "d" * 64,
        "input_order_contract": "frozen_snapshot_row_order",
        "seed": 1729,
        "dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "selector": "quest",
        "top_k": 2048,
        "page_size": 64,
        "allocated_compute_kv_bytes": 4096,
        "preallocated_request_slots": 2,
        "max_context_len": 80,
        "input_len": 4,
        "output_len": 65,
        "client_concurrency": 2,
        "cuda_graph_enabled": True,
        "diagnostic": False,
        "serving_config_sha256": "e" * 64,
        "hot_buffer": hot,
        "backing_ratio": 4,
        "variant": "cpu",
    }


def session():
    return {
        "type": "session",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "disable_overlap_schedule": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "disable_radix_cache": True,
    }


def forward(sequence, mode, ids, *, interval_ms=10, lengths=None):
    return {
        "type": "forward",
        "sequence": sequence,
        "forward_mode": mode,
        "request_ids": list(ids),
        "extend_lens": lengths,
        "launch_monotonic_ns": 1_000_000_000 + sequence * interval_ms * 1_000_000,
        "cuda_seconds": 0.003,
        "success": True,
        "stream_id": 0,
    }


def fixture(name="a", interval_ms=10, staggered=False):
    ids = [name + "0", name + "1"]
    result = {
        "request_ids": ids,
        "completed": 2,
        "dataset_name": "frozen-ids",
        "input_lens": [4, 4],
        "output_lens": [65, 65],
        "errors": ["", ""],
        "generated_texts": ["same output zero", "same output one"],
        "total_output_tokens": 130,
        "max_concurrency": 2,
    }
    trace = [session()]
    if staggered:
        trace += [
            forward(0, "extend", ids[:1], interval_ms=interval_ms, lengths=[4]),
            forward(1, "decode", ids[:1], interval_ms=interval_ms),
            forward(2, "extend", ids[1:], interval_ms=interval_ms, lengths=[4]),
        ]
        trace += [
            forward(n + 3, "decode", ids, interval_ms=interval_ms) for n in range(63)
        ]
        trace.append(forward(66, "decode", ids[1:], interval_ms=interval_ms))
    else:
        trace.append(forward(0, "extend", ids, interval_ms=interval_ms, lengths=[4, 4]))
        trace += [
            forward(n, "decode", ids, interval_ms=interval_ms) for n in range(1, 65)
        ]
    trace.append({"type": "idle_flush", "last_sequence": trace[-1]["sequence"]})
    return result, trace


def analyze(name="a", hot=2048, interval_ms=10, trim=4, staggered=False):
    result, trace = fixture(name, interval_ms, staggered)
    return BATCH.analyze_case(result, trace, identity(name, hot), edge_trim_steps=trim)


class BatchCostContracts(unittest.TestCase):
    def test_default_stream_and_exact_right_boundary_accounting(self):
        result, trace = fixture()
        trace[61]["cuda_seconds"] = 99  # Last retained launch is the right boundary.
        data = BATCH.analyze_case(result, trace, identity())
        window = data["windows"][0]
        self.assertEqual(window["actual_batch_size"], 2)
        self.assertEqual(window["measured_intervals"], 55)
        self.assertEqual(window["decode_tokens_over_launch_intervals"], 110)
        self.assertAlmostEqual(window["scheduler_launch_span_seconds"], 0.55)
        self.assertAlmostEqual(window["cuda_forward_seconds_sum"], 55 * 0.003)
        self.assertAlmostEqual(window["decode_tokens_per_scheduler_launch_second"], 200)
        self.assertEqual(
            window["actual_execution_scope"], "unobserved_graph_enabled_policy"
        )

    def test_uuid_changes_do_not_break_frozen_row_matching(self):
        left, right = analyze(), analyze("b", 4096, interval_ms=20)
        paired = BATCH.match_windows([left, right])["paired_windows"][0]
        self.assertEqual(paired["logical_request_rows"], [0, 1])
        self.assertNotEqual(
            paired["cases"][0]["request_ids"], paired["cases"][1]["request_ids"]
        )
        self.assertTrue(paired["generated_text_equal"])
        self.assertFalse(paired["generated_token_ids_verified"])
        self.assertAlmostEqual(
            paired["cases"][0]["decode_tokens_per_scheduler_launch_second"], 200
        )
        self.assertAlmostEqual(
            paired["cases"][1]["decode_tokens_per_scheduler_launch_second"], 100
        )

    def test_common_history_intersection_not_maximum_window_length(self):
        paired = BATCH.match_windows([analyze(), analyze("b", 4096, trim=8)])[
            "paired_windows"
        ][0]
        for case in paired["cases"]:
            self.assertEqual(case["measured_launch_intervals"], 47)
            self.assertEqual(case["history_start"], [13, 13])
            self.assertEqual(case["history_right_boundary"], [60, 60])

    def test_same_max_batch_with_different_history_vector_is_not_matched(self):
        data = BATCH.match_windows([analyze(), analyze("b", 4096, staggered=True)])
        self.assertEqual(data["status"], "no_matched_windows")
        self.assertEqual(data["paired_windows"], [])

    def test_disjoint_windows_are_never_stitched(self):
        left, right = analyze(), analyze("b", 4096)
        right["windows"].append(copy.deepcopy(right["windows"][0]))
        data = BATCH.match_windows([left, right])
        self.assertEqual(data["status"], "no_matched_windows")
        self.assertEqual(
            data["unmatched_reasons"][0]["reason"], "ambiguous_multiple_contiguous_runs"
        )

    def test_generated_text_difference_prevents_unverified_history_match(self):
        left, right = analyze(), analyze("b", 4096)
        right["generated_text_sha256_by_input_row"][1] = "different"
        data = BATCH.match_windows([left, right])
        self.assertEqual(data["paired_windows"], [])

    def test_requested_batch_is_a_filter_not_a_client_concurrency_proxy(self):
        left, right = analyze(), analyze("b", 4096)
        data = BATCH.match_windows([left, right], actual_batch_size=1)
        self.assertEqual(data["status"], "no_matched_windows")
        self.assertEqual(data["requested_actual_batch_size"], 1)
        self.assertEqual(
            data["unmatched_reasons"][0]["reason"],
            "no_common_contiguous_request_history_key",
        )
        self.assertEqual(
            len(
                BATCH.match_windows([left, right], actual_batch_size=2)[
                    "paired_windows"
                ]
            ),
            1,
        )

    def test_observed_graph_and_history_are_explicit(self):
        result, trace = fixture()
        for row in trace:
            if row.get("forward_mode") == "decode":
                row["seq_lens"] = [4 + row["sequence"]] * 2
                row["actual_cuda_graph"] = True
        data = BATCH.analyze_case(result, trace, identity())
        self.assertEqual(data["windows"][0]["actual_execution_scope"], "graph_replay")
        self.assertEqual(
            data["windows"][0]["history_scope"],
            "observed_seq_lens_verified_against_complete_coverage",
        )

    def test_actual_graph_mode_changes_split_windows(self):
        result, trace = fixture()
        for row in trace:
            if row.get("forward_mode") == "decode":
                row["actual_cuda_graph"] = row["sequence"] <= 32
        data = BATCH.analyze_case(result, trace, identity())
        self.assertEqual(
            [w["actual_execution_scope"] for w in data["windows"]],
            ["graph_replay", "eager"],
        )
        self.assertEqual([w["measured_intervals"] for w in data["windows"]], [23, 23])

    def test_observed_history_conflict_rejected(self):
        result, trace = fixture()
        trace[2]["seq_lens"] = [100, 100]
        with self.assertRaisesRegex(ValueError, "Observed history conflicts"):
            BATCH.analyze_case(result, trace, identity())

    def test_controlled_differences_refused(self):
        for field, value in (
            ("source_commit", "e" * 40),
            ("input_snapshot_sha256", "f" * 64),
            ("gpu_uuid", "GPU-other"),
            ("top_k", 4096),
            ("allocated_compute_kv_bytes", 8192),
            ("client_concurrency", 4),
            ("preallocated_request_slots", 4),
        ):
            with self.subTest(field=field):
                left, right = analyze(), analyze("b", 4096)
                right["identity"][field] = value
                with self.assertRaisesRegex(ValueError, "Controlled field differs"):
                    BATCH.match_windows([left, right])

    def test_counter_diagnostic_refused_for_formal_comparison(self):
        left, right = analyze(), analyze("b", 4096)
        right["identity"]["diagnostic"] = True
        with self.assertRaisesRegex(ValueError, "Counter diagnostics"):
            BATCH.match_windows([left, right])

    def test_damaged_or_noncomparable_trace_refused(self):
        for mutation in (
            "missing_sequence",
            "unordered_clock",
            "duplicate_id",
            "padding_batch",
            "incomplete_decode",
            "incomplete_prefill",
            "foreign_request",
            "failed_forward",
            "no_flush",
            "overlap",
            "multirank",
            "graph_contradiction",
        ):
            with self.subTest(mutation=mutation):
                result, trace = fixture()
                config = identity()
                if mutation == "missing_sequence":
                    trace[3]["sequence"] = 999
                elif mutation == "unordered_clock":
                    trace[3]["launch_monotonic_ns"] = trace[2]["launch_monotonic_ns"]
                elif mutation == "duplicate_id":
                    trace[2]["request_ids"] = ["a0", "a0"]
                elif mutation == "padding_batch":
                    trace[2]["actual_batch_size"] = 4
                elif mutation == "incomplete_decode":
                    trace[2]["request_ids"] = ["a0"]
                elif mutation == "incomplete_prefill":
                    trace[1]["extend_lens"] = [3, 4]
                elif mutation == "foreign_request":
                    trace[2]["request_ids"] = ["a0", "foreign"]
                elif mutation == "failed_forward":
                    trace[2]["success"] = False
                elif mutation == "no_flush":
                    trace[-1]["last_sequence"] = -1
                elif mutation == "overlap":
                    trace[0]["disable_overlap_schedule"] = False
                elif mutation == "multirank":
                    trace[0]["tp_size"] = 4
                elif mutation == "graph_contradiction":
                    config["cuda_graph_enabled"] = False
                    trace[2]["actual_cuda_graph"] = True
                with self.assertRaises(ValueError):
                    BATCH.analyze_case(result, trace, config)

    def test_missing_and_malformed_controls_refused(self):
        for field, value in (
            ("source_commit", None),
            ("model_revision", "not-a-revision"),
            ("input_snapshot_sha256", "bad"),
            ("family", "glm"),
            ("input_order_contract", "completion_order"),
            ("cuda_graph_enabled", 1),
            ("top_k", True),
            ("backing_ratio", -1),
        ):
            with self.subTest(field=field):
                result, trace = fixture()
                config = identity()
                config[field] = value
                with self.assertRaises(ValueError):
                    BATCH.analyze_case(result, trace, config)

    def test_compact_output_preserves_source_boundaries_without_rows(self):
        analysis = analyze()
        compact = BATCH.compact_analysis(analysis)
        self.assertNotIn("_records", compact["windows"][0])
        self.assertIn("_records", analysis["windows"][0])
        self.assertEqual(compact["windows"][0]["first_sequence"], 5)


if __name__ == "__main__":
    unittest.main()
