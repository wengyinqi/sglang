"""CPU-only stage planning, TP accounting and post-request capacity checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load("stage_runner", "benchmark/mpu_stages/run.py")
CORRECT = load("stage_correctness", "benchmark/mpu_stages/correctness.py")
TIMER = load("stage_timer", "python/sglang/srt/observability/mpu_stage_timing.py")


def recipe(tp=1, variant="mpu"):
    common_config = {
        "algorithm": "quest",
        "top_k": 128,
        "device_buffer_size": 256,
        "quest_page_size": 64,
    }
    return {
        "schema_version": 1,
        "id": f"fixture-tp{tp}-{variant}",
        "family": "qwen",
        "tp_size": tp,
        "default_model": "/models/fixture",
        "variant": variant,
        "selector": "quest",
        "hbm_preserves_selector": True,
        "server_args": [
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--page-size",
            "64",
        ],
        "variants": {
            "cpu": {
                "server_args": [
                    "--enable-hisparse",
                    "--prefill-attention-backend",
                    "flashinfer",
                    "--decode-attention-backend",
                    "flashinfer_hisparse",
                ],
                "hisparse_config": common_config | {"host_to_device_ratio": 2},
            },
            "mpu": {
                "server_args": [
                    "--enable-hisparse",
                    "--prefill-attention-backend",
                    "flashinfer",
                    "--decode-attention-backend",
                    "flashinfer_hisparse",
                ],
                "hisparse_config": common_config
                | {
                    "backing_store": "mpu",
                    "mpu_device_ids": [1] if tp == 1 else [4, 5, 6, 7],
                },
            },
            "hbm": {
                "server_args": ["--attention-backend", "flashinfer_quest"],
                "hisparse_config": common_config,
            },
        },
        "defaults": {
            "input_lens": [1024],
            "output_len": 16,
            "max_total_tokens": 4096,
            "chunked_prefill_size": 256,
            "capacity_ratios": [1, 2],
            "concurrencies": [1, 2],
            "num_prompts": 4,
            "repeats": 2,
        },
    }


def session(rank=0, tp=1):
    return {
        "type": "session",
        "schema_version": 1,
        "tp_rank": rank,
        "tp_size": tp,
        "pp_rank": 0,
        "pp_size": 1,
        "dp_rank": None,
        "dp_size": 1,
        "disaggregation_mode": "null",
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "speculative_algorithm": None,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
    }


def capacity(rank=0, tp=1, variant="mpu", ratio=2):
    return {
        "tp_rank": rank,
        "tp_size": tp,
        "backing_store": variant,
        "physical_compute_slots": 4096,
        "compute_kv_bytes": 4096 * 100,
        "backing_slots": 4096 * ratio,
        "backing_kv_bytes": 4096 * ratio * 100,
        "requested_capacity_ratio": ratio,
        "effective_capacity_ratio": ratio,
        "storage_device_ids": (
            ([1] if tp == 1 else [4 + rank]) if variant == "mpu" else []
        ),
        "reserved_slots": 0,
        "used_slots": 0,
    }


def trace(rank=0, tp=1, prefill=(1.5, 0.5)):
    def forward(index, mode, ids, seconds, lengths=None):
        return {
            "type": "forward",
            "sequence": index,
            "forward_mode": mode,
            "request_ids": ids,
            "extend_lens": lengths,
            "cuda_seconds": seconds,
            "stream_id": 42 + rank,
            "success": True,
        }

    return [
        session(rank, tp),
        forward(0, "extend", ["warmup"], 1, [8]),
        forward(1, "extend", ["a", "b"], prefill[0], [3, 3]),
        forward(2, "extend", ["a"], prefill[1], [2]),
        forward(3, "decode", ["a", "b"], 0.5),
        forward(4, "decode", ["a", "b"], 0.5),
        {"type": "capacity", "after_sequence": 4, "capacity": capacity(rank, tp)},
        {"type": "idle_flush", "last_sequence": 4},
    ]


def benchmark():
    return {
        "request_ids": ["a", "b"],
        "completed": 2,
        "input_lens": [5, 3],
        "output_lens": [3, 3],
        "errors": ["", ""],
        "total_output_tokens": 6,
        "duration": 20.0,
        "output_throughput": 0.3,
        "mean_ttft_ms": 2000,
        "mean_tpot_ms": 500,
    }


class TestStagePlanning(unittest.TestCase):
    def parse(self, stage, extra=None, *, correctness=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.json"
            path.write_text(json.dumps(stage), encoding="utf-8")
            return RUN.STAGE.parse_args(
                [
                    "--stage",
                    str(path),
                    "--output-dir",
                    str(Path(directory) / "result"),
                    *(extra or []),
                ],
                correctness=correctness,
            )

    def test_plus_zero_is_cpu_backing_and_hbm_is_optional(self):
        stage, args = self.parse(recipe(variant="cpu"))
        groups = RUN.make_plan(stage, args)
        self.assertEqual({group["variant"] for group in groups}, {"cpu"})
        self.assertTrue(
            all("--enable-hisparse" in group["server_command"] for group in groups)
        )
        self.assertFalse(args.execute)
        self.assertEqual(len(groups), 2)

    def test_mpu_stage_automatically_keeps_cpu_reference(self):
        stage, args = self.parse(recipe(tp=4))
        groups = RUN.make_plan(stage, args)
        self.assertEqual([group["variant"] for group in groups], ["cpu", "mpu", "mpu"])
        for group in groups:
            cmd = group["server_command"]
            self.assertEqual(cmd[cmd.index("--tp-size") + 1], "4")
            self.assertEqual(cmd[cmd.index("--max-total-tokens") + 1], "4096")
            self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "2")
            self.assertIn("--disable-overlap-schedule", cmd)

    def test_cpu_ratio_integer_and_selector_reference_guard(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            self.parse(recipe(variant="cpu"), ["--capacity-ratios", "1.5"])
        stage, args = self.parse(recipe(), ["--capacity-ratios", "1.5"])
        self.assertEqual(RUN.STAGE.variants(stage, args), [("cpu", 2), ("mpu", 1.5)])
        invalid = recipe() | {"hbm_preserves_selector": False}
        with self.assertRaisesRegex(ValueError, "same sparse selector"):
            self.parse(invalid, ["--include-hbm"])

    def test_correctness_compares_cpu_first_and_reuses_slots(self):
        stage, args = self.parse(recipe(), correctness=True)
        groups = CORRECT.make_plan(stage, args)
        self.assertEqual(groups[0]["variant"], "cpu")
        self.assertEqual(len(groups), 6)
        for group in groups:
            self.assertEqual(group["rounds"][0], group["rounds"][2])
            self.assertGreater(min(group["rounds"][0]), 256)
            self.assertIn("--enable-hisparse", group["server_command"])

    def test_actual_capacity_padding_and_profiled_units(self):
        stage, args = self.parse(recipe())
        actual = capacity(variant="cpu")
        actual.update(backing_slots=8193, effective_capacity_ratio=8193 / 4096)
        self.assertEqual(
            RUN.verify_capacities([actual], stage, args, "cpu", 2), [actual]
        )
        actual["physical_compute_slots"] = 1024
        actual["effective_capacity_ratio"] = 8193 / 1024
        with self.assertRaisesRegex(ValueError, "fixed budget"):
            RUN.verify_capacities([actual], stage, args, "cpu", 2)
        stage["physical_budget_exact"] = False
        RUN.verify_capacities([actual], stage, args, "cpu", 2)
        with self.assertRaisesRegex(ValueError, "drifted"):
            RUN.enforce_fixed_compute([(4096, actual["compute_kv_bytes"])], [actual])

    def test_different_selectors_cannot_be_called_storage_equivalence(self):
        stage = recipe()
        stage["variants"]["mpu"]["hisparse_config"]["top_k"] = 256
        with self.assertRaisesRegex(ValueError, "preserve selector"):
            self.parse(stage)

    def test_new_runner_finishes_analysis_before_owned_server_shutdown(self):
        stage, args = self.parse(recipe(variant="cpu"))
        args.input_lens, args.concurrencies, args.num_prompts, args.output_len = (
            [5],
            [1],
            2,
            3,
        )
        args.capacity_ratios, args.repeats = [2], 1
        value = benchmark()
        value["input_lens"] = [5, 5]
        report = {
            "end_to_end": {
                "output_throughput_tokens_per_second": 0.3,
                "mean_ttft_ms": 2000,
                "mean_tpot_ms": 500,
            },
            "decode_only_estimate": {"decode_throughput_tokens_per_second": 0.25},
        }
        events = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args.output_dir = Path(directory)
            manifest = {"groups": RUN.make_plan(stage, args)}
            stack.enter_context(patch.object(RUN, "require_cuda"))
            stack.enter_context(
                patch.object(RUN.COMMON, "start_server", return_value=object())
            )
            stack.enter_context(patch.object(RUN.COMMON, "wait_ready"))
            stack.enter_context(
                patch.object(RUN.COMMON, "request_json", return_value={})
            )
            stack.enter_context(
                patch.object(
                    RUN, "wait_capacities", return_value=[capacity(variant="cpu")]
                )
            )
            stack.enter_context(patch.object(RUN.subprocess, "run"))
            stack.enter_context(
                patch.object(RUN.ANALYZE, "read_jsonl", return_value=[value])
            )
            stack.enter_context(
                patch.object(
                    RUN.ANALYZE,
                    "analyze_when_ready",
                    side_effect=lambda *a: events.append("analyze") or report,
                )
            )
            stack.enter_context(
                patch.object(
                    RUN.COMMON,
                    "stop_server",
                    side_effect=lambda *a: events.append("stop"),
                )
            )
            self.assertTrue(
                RUN.execute(stage, args, manifest, args.output_dir / "manifest.json")
            )
        self.assertEqual(events, ["analyze", "stop"])


class TestTPAccounting(unittest.TestCase):
    def test_tp4_uses_maximum_each_batch_not_sum_ranks_or_max_rank_total(self):
        traces = [
            trace(rank, 4, durations)
            for rank, durations in enumerate([(1.5, 0.5), (1, 2), (1, 1), (2, 1.5)])
        ]
        result = RUN.ANALYZE.analyze(benchmark(), traces)
        self.assertEqual(result["cuda_trace"]["prefill_forward_seconds"], 4)
        self.assertEqual(result["cuda_trace"]["tp_size"], 4)
        self.assertEqual(result["decode_only_estimate"]["residual_seconds"], 16)
        self.assertEqual(len(result["cuda_trace"]["per_rank"]), 4)

    def test_tp1_preserves_original_accounting(self):
        result = RUN.ANALYZE.analyze(benchmark(), [trace()])
        self.assertEqual(result["cuda_trace"]["prefill_forward_seconds"], 2)
        self.assertEqual(result["end_to_end"]["mean_ttft_ms"], 2000)

    def test_missing_duplicate_or_misaligned_rank_is_rejected(self):
        traces = [trace(rank, 4) for rank in range(4)]
        for bad in (traces[:3], [*traces, traces[0]]):
            with self.subTest(count=len(bad)), self.assertRaisesRegex(
                ValueError, "every TP rank"
            ):
                RUN.ANALYZE.analyze(benchmark(), bad)
        bad = copy.deepcopy(traces)
        bad[1][4]["request_ids"] = ["b", "a"]
        with self.assertRaisesRegex(ValueError, "alignment"):
            RUN.ANALYZE.analyze(benchmark(), bad)

    def test_stale_capacity_zero_cannot_mask_current_leak(self):
        good = trace()
        self.assertEqual(
            RUN.ANALYZE.read_capacities([good], 1, ["a", "b"], require_idle=True)[0][
                "used_slots"
            ],
            0,
        )
        for mutation in ("stale", "leak", "missing_request"):
            bad = copy.deepcopy(good)
            if mutation == "stale":
                bad[-2]["after_sequence"] = -1
            elif mutation == "leak":
                bad[-2]["capacity"]["reserved_slots"] = 80
            else:
                bad[2]["request_ids"] = ["a"]
                bad[4]["request_ids"] = ["a"]
                bad[5]["request_ids"] = ["a"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                RUN.ANALYZE.read_capacities([bad], 1, ["a", "b"], require_idle=True)


class TestTimerCapacity(unittest.TestCase):
    def test_capacity_helper_supports_hbm_tuple_sizes_and_rank_local_coordinators(self):
        pool = SimpleNamespace(size=4096, get_kv_size_bytes=lambda: (300, 400))
        allocator = SimpleNamespace(
            size=4096, available_size=lambda: 4094, get_kvcache=lambda: pool
        )
        actual = TIMER.collect_stage_capacity(allocator, None, 2, 4)
        self.assertEqual(actual["backing_store"], "hbm")
        self.assertEqual(actual["compute_kv_bytes"], 700)
        self.assertEqual(actual["used_slots"], 2)
        self.assertEqual(actual["tp_rank"], 2)
        coordinator = SimpleNamespace(
            get_stage_capacity_info=lambda: {"backing_store": "cpu", "tp_rank": 0}
        )
        self.assertEqual(
            TIMER.collect_stage_capacity(allocator, coordinator, 3, 4),
            {"backing_store": "cpu", "tp_rank": 3, "tp_size": 4},
        )

    def test_capacity_snapshot_follows_current_events_and_idle_only(self):
        state = capacity()
        calls, events = [], []

        def provider():
            calls.append("snapshot")
            return dict(state)

        def make_event():
            event = SimpleNamespace(ready=False)
            event.record = lambda stream: None
            event.query = lambda: event.ready

            def synchronize():
                for item in events:
                    item.ready = True

            event.synchronize = synchronize
            event.elapsed_time = lambda end: 10
            events.append(event)
            return event

        with tempfile.TemporaryDirectory() as directory:
            timer = TIMER.MPUStageForwardTimer(
                directory,
                session(),
                event_factory=make_event,
                stream_getter=lambda: SimpleNamespace(cuda_stream=42),
                capacity_provider=provider,
            )
            timer.flush_idle()
            initial = RUN.ANALYZE.read_jsonl(timer.path)
            self.assertEqual(initial[-2]["after_sequence"], -1)
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(name="EXTEND"),
                reqs=[SimpleNamespace(rid="a")],
                extend_lens=[5],
            )
            with timer.wrap(batch):
                state["used_slots"] = 5
            self.assertEqual(len(calls), 1)
            timer.flush_idle()
            rows = RUN.ANALYZE.read_jsonl(timer.path)
            self.assertEqual(rows[-2]["after_sequence"], 0)
            self.assertEqual(rows[-2]["capacity"]["used_slots"], 5)
            state["used_slots"] = 0
            timer.flush_idle()
            final = RUN.ANALYZE.read_jsonl(timer.path)
            self.assertEqual(final[-2]["capacity"]["used_slots"], 0)
            timer.flush_idle()
            self.assertEqual(RUN.ANALYZE.read_jsonl(timer.path), final)


if __name__ == "__main__":
    unittest.main()
