"""CPU tests for MPU experiment timing and conservative result accounting."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TIMING = load("hisparse_timing", "python/sglang/srt/observability/hisparse_timing.py")
ANALYZE = load("mpu_analyze", "benchmark/hisparse_mpu/analyze.py")


def session():
    return {
        "type": "session",
        "schema_version": 1,
        "tp_rank": 0,
        "pp_rank": 0,
        "dp_rank": None,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "disaggregation_mode": "null",
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "speculative_algorithm": None,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
    }


class FakeEvents:
    def __init__(self):
        self.events = []
        self.synchronizations = 0

    def create(self):
        event = SimpleNamespace(ready=False, stream=None)

        def record(stream):
            event.stream = stream

        def synchronize():
            self.synchronizations += 1
            for item in self.events:
                item.ready = True

        event.record = record
        event.query = lambda: event.ready
        event.synchronize = synchronize
        event.elapsed_time = lambda end: 25.0
        self.events.append(event)
        return event


class TestForwardTrace(unittest.TestCase):
    def make_timer(self, directory):
        events = FakeEvents()
        stream = SimpleNamespace(cuda_stream=42)
        timer = TIMING.HiSparseForwardTimer(
            directory,
            session(),
            event_factory=events.create,
            stream_getter=lambda: stream,
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")],
            extend_lens=[5, 3],
        )
        return timer, events, stream, batch

    def test_hot_path_never_syncs_or_writes_events(self):
        with tempfile.TemporaryDirectory() as directory:
            timer, events, stream, batch = self.make_timer(directory)
            with timer.wrap(batch):
                pass
            self.assertEqual(events.synchronizations, 0)
            self.assertEqual(len(ANALYZE.read_jsonl(timer.path)), 1)
            self.assertTrue(all(event.stream is stream for event in events.events))
            # Batch metadata can be reused/mutated after forward.
            batch.reqs[0].rid = "changed"
            batch.extend_lens[0] = 99
            timer.flush_idle()
            records = ANALYZE.read_jsonl(timer.path)
            self.assertEqual(records[1]["request_ids"], ["a", "b"])
            self.assertEqual(records[1]["extend_lens"], [5, 3])
            self.assertEqual(records[1]["cuda_seconds"], 0.025)
            self.assertEqual(records[-1], {"type": "idle_flush", "last_sequence": 0})
            self.assertEqual(events.synchronizations, 1)
            timer.flush_idle()
            self.assertEqual(ANALYZE.read_jsonl(timer.path), records)

    def test_failed_forward_is_recorded_and_exception_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            timer, events, stream, batch = self.make_timer(directory)
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                with timer.wrap(batch):
                    raise RuntimeError("kernel failed")
            self.assertEqual(events.synchronizations, 0)
            timer.flush_idle()
            self.assertFalse(ANALYZE.read_jsonl(timer.path)[1]["success"])

    def test_completed_events_remain_ordered_across_idle_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            timer, events, stream, batch = self.make_timer(directory)
            for _ in range(2):
                with timer.wrap(batch):
                    pass
            timer.flush_idle()
            batch.forward_mode.name = "DECODE"
            with timer.wrap(batch):
                pass
            timer.flush_idle()
            records = ANALYZE.read_jsonl(timer.path)
            forwards = [record for record in records if record["type"] == "forward"]
            self.assertEqual([record["sequence"] for record in forwards], [0, 1, 2])
            self.assertIsNone(forwards[-1]["extend_lens"])
            self.assertEqual(records[-1]["last_sequence"], 2)

    def test_disabled_trace_does_not_require_cuda_or_create_files(self):
        self.assertIsNone(TIMING.HiSparseForwardTimer.create("", None, 0, 0, None))

    def test_real_cuda_stream_eager_and_graph_replay(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA hardware is required")
        stream = torch.cuda.Stream()
        values = torch.ones((1024, 1024), device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            values.add_(1)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values.add_(1)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            reqs=[SimpleNamespace(rid="cuda-request")],
            extend_lens=[5],
        )
        for mode in ("eager", "graph"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                timer = TIMING.HiSparseForwardTimer(directory, session())
                with torch.cuda.stream(stream), timer.wrap(batch):
                    if mode == "eager":
                        values.add_(1)
                    else:
                        graph.replay()
                timer.flush_idle()
                record = ANALYZE.read_jsonl(timer.path)[1]
                self.assertEqual(record["stream_id"], stream.cuda_stream)
                self.assertEqual(record["request_ids"], ["cuda-request"])
                self.assertGreater(record["cuda_seconds"], 0)
                self.assertTrue(record["success"])


class TestDecodeOnlyAccounting(unittest.TestCase):
    def setUp(self):
        self.result = {
            "request_ids": ["a", "b"],
            "completed": 2,
            "input_lens": [5, 3],
            "output_lens": [3, 3],
            "errors": ["", ""],
            "total_output_tokens": 6,
            "duration": 10.0,
            "output_throughput": 0.6,
            "mean_ttft_ms": 2300.0,
            "mean_tpot_ms": 1000.0,
            # Deliberately overlapping request TTFTs must not be subtracted.
            "ttfts": [6.0, 7.0],
        }

        def record(sequence, mode, ids, seconds, lengths=None):
            return {
                "type": "forward",
                "sequence": sequence,
                "forward_mode": mode,
                "request_ids": ids,
                "extend_lens": lengths,
                "cuda_seconds": seconds,
                "stream_id": 42,
                "success": True,
            }

        self.trace = [
            session(),
            record(0, "extend", ["warmup"], 9.0, [20]),
            {"type": "idle_flush", "last_sequence": 0},
            record(1, "extend", ["a", "b"], 1.5, [3, 3]),
            record(2, "extend", ["a"], 0.5, [2]),
            record(3, "decode", ["a", "b"], 1.0),
            record(4, "decode", ["a", "b"], 1.0),
            {"type": "idle_flush", "last_sequence": 4},
        ]

    def test_batches_count_once_warmup_is_excluded_and_first_token_is_explicit(self):
        report = ANALYZE.analyze(self.result, [self.trace])
        self.assertEqual(report["end_to_end"]["mean_ttft_ms"], 2300.0)
        self.assertEqual(report["cuda_trace"]["prefill_forward_seconds"], 2.0)
        self.assertEqual(report["cuda_trace"]["prefill_batches"], 2)
        estimate = report["decode_only_estimate"]
        self.assertEqual(estimate["residual_seconds"], 8.0)
        self.assertEqual(estimate["decode_output_tokens"], 4)
        self.assertEqual(estimate["decode_throughput_tokens_per_second"], 0.5)
        self.assertEqual(estimate["all_output_throughput_tokens_per_second"], 0.75)

    def test_rejects_unsupported_scheduler_shapes(self):
        for change in (
            {"tp_size": 2},
            {"pp_size": 2},
            {"dp_size": 2},
            {"disaggregation_mode": "decode"},
            {"enable_mixed_chunk": True},
            {"enable_two_batch_overlap": True},
            {"speculative_algorithm": "EAGLE"},
            {"disable_radix_cache": False},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ANALYZE.analyze(
                    self.result, [[self.trace[0] | change] + self.trace[1:]]
                )

    def test_rejects_partial_duplicate_mixed_or_failed_records(self):
        for index, change in (
            (3, {"extend_lens": [4, 3]}),
            (3, {"request_ids": ["a", "foreign"]}),
            (3, {"request_ids": ["a", "a"]}),
            (4, {"sequence": 1}),
            (4, {"stream_id": 43}),
            (4, {"success": False}),
            (4, {"forward_mode": "mixed"}),
            (4, {"cuda_seconds": float("nan")}),
            (4, {"cuda_seconds": 11.0}),
            (6, {"request_ids": ["a"]}),
            (7, {"last_sequence": 3}),
        ):
            trace = copy.deepcopy(self.trace)
            trace[index].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ANALYZE.analyze(self.result, [trace])
        for traces in ([], [self.trace, self.trace], [self.trace[:-1]]):
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                ANALYZE.analyze(self.result, traces)

    def test_rejects_incomplete_benchmark_outputs(self):
        for change in (
            {"completed": 1},
            {"request_ids": [None, "b"]},
            {"request_ids": ["a", "a"]},
            {"output_lens": [3]},
            {"output_lens": [1, 3]},
            {"errors": ["OOM", ""]},
            {"total_output_tokens": 5},
            {"output_throughput": 6.0},
            {"mean_ttft_ms": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ANALYZE.analyze(self.result | change, [self.trace])

    def test_rejects_unrelated_batches_inside_workload_interval(self):
        trace = copy.deepcopy(self.trace)
        trace.insert(5, trace[1] | {"request_ids": ["foreign"]})
        sequence = 0
        for record in trace:
            if record["type"] == "forward":
                record["sequence"] = sequence
                sequence += 1
            elif record["type"] == "idle_flush":
                record["last_sequence"] = sequence - 1
        with self.assertRaisesRegex(ValueError, "Unrelated forwards"):
            ANALYZE.analyze(self.result, [trace])

    def test_overlap_extra_forward_is_visible_and_bounded(self):
        trace = copy.deepcopy(self.trace)
        trace[0]["disable_overlap_schedule"] = False
        trace.insert(-1, trace[-2] | {"sequence": 5})
        trace[-1]["last_sequence"] = 5
        report = ANALYZE.analyze(self.result, [trace])
        self.assertEqual(report["cuda_trace"]["extra_overlap_decode_forwards"], 2)
        trace.insert(-1, trace[-2] | {"sequence": 6})
        trace[-1]["last_sequence"] = 6
        with self.assertRaises(ValueError):
            ANALYZE.analyze(self.result, [trace])

    def test_file_analysis_requires_valid_final_idle_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_file = Path(directory) / "forward-fixture.jsonl"
            trace_file.write_text(
                "\n".join(json.dumps(record) for record in self.trace) + "\n",
                encoding="utf-8",
            )
            report = ANALYZE.analyze_when_ready(self.result, directory, timeout=0.01)
            self.assertEqual(report["status"], "analyzed")
            trace_file.write_text('{"type":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not become valid"):
                ANALYZE.analyze_when_ready(self.result, directory, timeout=0.01)


if __name__ == "__main__":
    unittest.main()
