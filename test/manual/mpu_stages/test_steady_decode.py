"""Synthetic timing/HTTP/SQLite fixtures verify accounting, never GPU behavior."""

import ast
import asyncio
import dataclasses
import importlib.util
import json
import sqlite3
import sys
import tempfile
import time
import traceback
import types
import typing
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEADY = load("steady_test", "benchmark/mpu_stages/steady_decode.py")
TIMER = load(
    "timer_progress_test", "python/sglang/srt/observability/mpu_stage_timing.py"
)
PROFILE = load("profile_stable_test", "benchmark/mpu_stages/profile_decode.py")
SCOPE = load("multi_scope_test", "benchmark/mpu_stages/decode_scope.py")


def session():
    return {
        "type": "session",
        "schema_version": 1,
        "tp_rank": 0,
        "tp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "dp_rank": None,
        "dp_size": 1,
        "disaggregation_mode": "null",
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "speculative_algorithm": None,
    }


def forward(sequence, mode, ids, lengths=None):
    return {
        "type": "forward",
        "sequence": sequence,
        "forward_mode": mode,
        "request_ids": ids,
        "extend_lens": lengths,
        "success": True,
        "stream_id": 42,
        "cuda_seconds": 0.005,
        "launch_monotonic_ns": 10_000_000_000 + sequence * 10_000_000,
    }


def result(ids=("a", "b"), counts=(41, 41)):
    frames = [
        [
            {
                "received_monotonic_ns": 1_000_000_000 + n * 10_000_000,
                "cumulative_completion_tokens": n,
                "new_tokens": 1,
                "finish_reason": {"type": "length"} if n == count else None,
            }
            for n in range(1, count + 1)
        ]
        for count in counts
    ]
    return {
        "request_ids": list(ids),
        "completed": len(ids),
        "input_lens": [4] * len(ids),
        "output_lens": list(counts),
        "errors": [""] * len(ids),
        "total_output_tokens": sum(counts),
        "duration": 2.0,
        "output_throughput": sum(counts) / 2,
        "mean_ttft_ms": 10,
        "mean_tpot_ms": 50,
        "request_latencies": [2.0] * len(ids),
        "ttfts": [0.01] * len(ids),
        "request_start_monotonic_ns": [1_000_000_000] * len(ids),
        "request_completion_monotonic_ns": [3_000_000_000] * len(ids),
        "stream_frames": frames,
    }


def trace():
    rows = [forward(0, "extend", ["a", "b"], [4, 4])]
    rows += [forward(i, "decode", ["a", "b"]) for i in range(1, 41)]
    return [session(), *rows, {"type": "idle_flush", "last_sequence": 40}]


class SteadyTests(unittest.TestCase):
    def test_true_constant_set_window_has_matching_launch_intervals(self):
        report = STEADY.analyze(result(), [trace()])
        self.assertTrue(report["stable_window_observed"])
        window = report["windows"][0]
        self.assertEqual(window["actual_batch_size"], 2)
        self.assertEqual(window["forward_records"], 32)
        self.assertEqual(window["measured_launch_intervals"], 31)
        self.assertEqual(window["decode_tokens_over_launch_intervals"], 62)
        self.assertAlmostEqual(window["scheduler_launch_span_seconds"], 0.31)
        self.assertAlmostEqual(window["decode_tokens_per_scheduler_launch_second"], 200)
        self.assertTrue(report["client"]["all_frames_single_token"])

    def test_prefill_insertion_and_same_size_new_membership_split_windows(self):
        rows = [forward(0, "extend", ["a"], [4])]
        rows += [forward(i, "decode", ["a"]) for i in range(1, 21)]
        rows += [forward(21, "extend", ["b"], [4])]
        rows += [forward(i, "decode", ["a", "b"]) for i in range(22, 42)]
        records = [session(), *rows, {"type": "idle_flush", "last_sequence": 41}]
        report = STEADY.analyze(result(counts=(41, 21)), [records], edge_trim_steps=0)
        self.assertEqual([w["actual_batch_size"] for w in report["windows"]], [1, 2])
        self.assertEqual([r["sequence"] for r in report["prefill_insertions"]], [21])

    def test_multitoken_frame_remains_chunk_timing_not_strict_token_timing(self):
        native = result()
        first = native["stream_frames"][0]
        first[1]["new_tokens"] = 2
        del first[0]
        report = STEADY.request_latency_details(native)
        self.assertFalse(report["all_frames_single_token"])
        self.assertIsNone(report["single_token_arrival_gap_ms"])
        self.assertGreater(report["raw_token_positive_frame_gap_ms"]["samples"], 0)

    def test_missing_or_incomplete_stream_and_clock_evidence_fails_closed(self):
        for edit in (
            lambda r: r.pop("request_latencies"),
            lambda r: r["stream_frames"][0][-1].update(finish_reason=None),
            lambda r: r["stream_frames"][0][-1].update(new_tokens=9),
            lambda r: r["request_completion_monotonic_ns"].__setitem__(0, 1),
        ):
            native = result()
            edit(native)
            with self.assertRaises(ValueError):
                STEADY.analyze(native, [trace()])
        records = trace()
        records[2].pop("launch_monotonic_ns")
        with self.assertRaisesRegex(ValueError, "timestamps"):
            STEADY.analyze(result(), [records])

    def test_overlap_or_multiple_streams_do_not_claim_stable_launch_scope(self):
        records = trace()
        records[0]["disable_overlap_schedule"] = False
        with self.assertRaisesRegex(ValueError, "non-overlap"):
            STEADY.analyze(result(), [records])
        records = trace()
        records[2]["stream_id"] = 43
        with self.assertRaisesRegex(ValueError, "one compute stream"):
            STEADY.analyze(result(), [records])


class ProgressTests(unittest.TestCase):
    def test_live_progress_is_cpu_only_copies_ids_and_resets_on_extend(self):
        def event():
            return types.SimpleNamespace(
                record=lambda stream: None, query=lambda: False
            )

        with tempfile.TemporaryDirectory() as directory:
            timer = TIMER.MPUStageForwardTimer(
                directory,
                session(),
                event_factory=event,
                stream_getter=lambda: types.SimpleNamespace(cuda_stream=42),
            )
            batch = types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(name="DECODE"),
                reqs=[types.SimpleNamespace(rid="a")],
                extend_lens=None,
            )
            with timer.wrap(batch):
                pass
            with timer.wrap(batch):
                pass
            snapshot = timer.progress_snapshot()
            self.assertEqual(snapshot["consecutive_stable_decode_steps"], 2)
            snapshot["request_ids"].append("mutated")
            self.assertEqual(timer.progress_snapshot()["request_ids"], ["a"])
            self.assertEqual(len(timer.path.read_text().splitlines()), 1)
            timer._stream_getter = lambda: types.SimpleNamespace(cuda_stream=43)
            with timer.wrap(batch):
                pass
            self.assertEqual(
                timer.progress_snapshot()["consecutive_stable_decode_steps"], 1
            )
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with timer.wrap(batch):
                    raise RuntimeError("fixture failure")
            self.assertEqual(
                timer.progress_snapshot()["consecutive_stable_decode_steps"], 0
            )
            batch.forward_mode.name = "EXTEND"
            batch.extend_lens = [4]
            with timer.wrap(batch):
                pass
            self.assertEqual(
                timer.progress_snapshot()["consecutive_stable_decode_steps"], 0
            )

    def test_profile_trigger_uses_actual_progress_and_rejects_stale_completed_request(
        self,
    ):
        future = types.SimpleNamespace(done=lambda: False)
        snapshot = {
            "success": True,
            "forward_mode": "decode",
            "request_ids": ["a", "b"],
            "batch_size": 2,
            "consecutive_stable_decode_steps": 16,
        }
        with mock.patch.object(PROFILE, "progress", return_value=snapshot):
            self.assertEqual(PROFILE.wait_stable(1, future, 16, 2, 1), snapshot)
        finished = types.SimpleNamespace(done=lambda: True, result=lambda: {})
        with self.assertRaisesRegex(ValueError, "completed"):
            PROFILE.wait_stable(1, finished, 16, 2, 1)

    def test_profile_batch_constraint_requires_stable_trigger(self):
        with self.assertRaisesRegex(ValueError, "requires a stable"):
            PROFILE.main(["--profile-batch-size", "2"])


class ScopeTests(unittest.TestCase):
    def fixture(self, root):
        database = root / "decode.sqlite"
        with closing(sqlite3.connect(database)) as conn:
            conn.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
            conn.executemany(
                "INSERT INTO StringIds VALUES(?,?)",
                [
                    (1, "cudaGraphLaunch_v3020"),
                    (2, "_quest_score_kernel"),
                    (3, SCOPE.DECODE_ATTENTION_KERNEL),
                ],
            )
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,correlationId INTEGER,nameId INTEGER)"
            )
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(correlationId INTEGER,graphId INTEGER,demangledName INTEGER)"
            )
            for i in range(4):
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,1)",
                    (i, i + 100),
                )
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,7,?)",
                    [(i + 100, kind) for kind in (2, 3) for _ in range(48)],
                )
            conn.commit()
        records = trace()
        snap = {
            "success": True,
            "forward_mode": "decode",
            "request_ids": ["a", "b"],
            "batch_size": 2,
            "stream_id": 42,
            "launch_monotonic_ns": forward(17, "decode", ["a", "b"])[
                "launch_monotonic_ns"
            ],
            "sequence": 17,
            "consecutive_stable_decode_steps": 16,
        }
        profile = {
            "status": "captured_pending_role_analysis",
            "artifacts": [{"path": str(database), "sha256": SCOPE.sha(database)}],
            "runtime": {"git_commit": "test-source"},
            "configured_decode_steps": 4,
            "profile_request": {
                "num_steps": 4,
                "activities": ["CUDA_PROFILER"],
                "profile_by_stage": True,
                "profile_stages": ["decode"],
            },
            "server_command": [
                "--capture-range=cudaProfilerApi",
                "--cuda-graph-trace=node",
            ],
            "profiled_request_ids": ["a", "b"],
            "workload": {"concurrency": 2, "input_len": 4, "output_len": 41},
            "completion_tokens": 82,
            "trigger_progress_before": snap,
            "trigger_progress_after_ack": snap
            | {
                "sequence": 18,
                "launch_monotonic_ns": forward(18, "decode", ["a", "b"])[
                    "launch_monotonic_ns"
                ],
            },
            "stable_trigger": {"min_consecutive_decode_steps": 16},
        }
        semantics = {
            "status": "verified_from_deployed_sources",
            "query_rows_per_request": 1,
            "tensor_core_decode_uses_paged_prefill_module": True,
            "source_commit": "test-source",
            "receipt_sha256": "synthetic-source-proof",
            "allowed_attention_kernel_name": SCOPE.DECODE_ATTENTION_KERNEL,
        }
        return database, profile, records, semantics

    def test_multiple_request_capture_matches_layers_and_conservative_window(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = SCOPE.verify(*args)
            self.assertEqual(report["actual_batch_size"], 2)
            self.assertEqual(report["quest_score_calls"], 192)

    def test_stable_counter_batch_and_trigger_sequence_require_matching_evidence(self):
        for fault in ("short_history", "wrong_batch", "wrong_clock", "missing_source"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                db, profile, records, semantics = self.fixture(Path(directory))
                if fault == "short_history":
                    profile["trigger_progress_before"]["sequence"] = 8
                elif fault == "wrong_batch":
                    profile["stable_trigger"]["required_actual_batch_size"] = 3
                elif fault == "wrong_clock":
                    profile["trigger_progress_after_ack"]["launch_monotonic_ns"] = 1
                else:
                    semantics.pop("source_commit")
                    profile["runtime"].pop("git_commit")
                with self.assertRaises(ValueError):
                    SCOPE.verify(db, profile, records, semantics)

    def test_prefill_membership_wrong_specialization_and_missing_layer_reject(self):
        for fault in ("prefill", "membership", "specialization", "missing_layer"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                db, profile, records, semantics = self.fixture(Path(directory))
                if fault == "prefill":
                    records[21]["forward_mode"] = "extend"
                    records[21]["extend_lens"] = [1, 1]
                elif fault == "membership":
                    profile["trigger_progress_after_ack"]["request_ids"] = ["a"]
                elif fault == "specialization":
                    semantics["allowed_attention_kernel_name"] = (
                        SCOPE.DECODE_ATTENTION_KERNEL.replace("(bool)0", "(bool)1", 1)
                    )
                else:
                    with closing(sqlite3.connect(db)) as conn:
                        conn.execute(
                            "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1"
                        )
                        conn.commit()
                    profile["artifacts"][0]["sha256"] = SCOPE.sha(db)
                with self.assertRaises(ValueError):
                    SCOPE.verify(db, profile, records, semantics)


class NativeFrameTests(unittest.TestCase):
    def module(self, chunks):
        path = ROOT / "python/sglang/benchmark/serving.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            "RequestFuncInput",
            "RequestFuncOutput",
            "async_request_sglang_generate",
            "_record_server_prompt_len",
        }
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef))
            and n.name in names
        ]
        module = types.ModuleType("bench_frame_fixture")
        module.__dict__.update(vars(typing))
        module.__dict__.update(
            dataclass=dataclasses.dataclass,
            field=dataclasses.field,
            time=time,
            sys=sys,
            traceback=traceback,
            json=json,
            orjson=types.SimpleNamespace(loads=json.loads),
            tqdm=object,
            remove_prefix=lambda text, prefix: text.removeprefix(prefix),
            get_request_headers=lambda: {},
        )
        ticks = iter(range(1_000_000_000, 2_000_000_000, 10_000))
        perf_ticks = iter(n / 1e9 for n in range(1_000_000_000, 2_000_000_000, 10_000))
        module.time = types.SimpleNamespace(
            monotonic_ns=lambda: next(ticks), perf_counter=lambda: next(perf_ticks)
        )

        class Response:
            status = 200

            @property
            def content(self):
                async def stream():
                    for item in chunks:
                        yield item

                return stream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, **kwargs):
                return Response()

        module._create_bench_client_session = Client
        module.args = types.SimpleNamespace(
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            output_details=True,
            temperature=0.0,
            top_p=1.0,
            cache_report=False,
        )
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module

    def test_generate_preserves_exact_frame_counts_and_final_latency(self):
        frames = [
            {
                "text": "x",
                "meta_info": {"id": "r", "prompt_tokens": 4, "completion_tokens": 2},
            },
            {
                "text": "xy",
                "meta_info": {
                    "id": "r",
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "finish_reason": {"type": "length"},
                },
            },
        ]
        module = self.module(
            [("data: " + json.dumps(frame)).encode() for frame in frames]
            + [b"data: [DONE]"]
        )
        request = module.RequestFuncInput(
            [42] * 4, "http://localhost/generate", 4, 3, "model", None, None, {}
        )
        output = asyncio.run(module.async_request_sglang_generate(request))
        self.assertTrue(output.success)
        self.assertEqual(
            [frame["new_tokens"] for frame in output.stream_frames], [2, 1]
        )
        self.assertEqual(output.stream_frames[-1]["finish_reason"], {"type": "length"})
        self.assertGreater(output.completion_monotonic_ns, output.start_monotonic_ns)
        self.assertGreaterEqual(output.latency, output.ttft)


if __name__ == "__main__":
    unittest.main()
