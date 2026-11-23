"""Portable tests for mixed-load evidence boundaries and loopback cancellation."""

import copy
import importlib.util
import json
import random
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "mixed_stability_tests", ROOT / "benchmark/mpu_stages/mixed_stability.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def case(number=0, cancel="decode"):
    return {
        "request_id": f"request-{number}",
        "input_len": 8,
        "output_len": 20,
        "cancel_mode": cancel,
        "cancel_delay_seconds": 0.02,
        "cancel_after_tokens": 4,
    }


def terminal(reason="abort", count=4):
    return {
        **case(),
        "status": "terminal",
        "cancel_attempt": {"attempted": True},
        "response": {
            "meta_info": {
                "id": "request-0",
                "prompt_tokens": 8,
                "completion_tokens": count,
                "finish_reason": {"type": reason},
            }
        },
    }


SESSION = {
    "type": "session",
    "schema_version": 1,
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1,
    "disable_overlap_schedule": True,
    "disable_radix_cache": True,
}


class MixedStabilityTests(unittest.TestCase):
    def test_real_torch_cuuuid_string_format_is_normalized_for_nvml(self):
        raw = "318d5f6b-cc4d-6350-e58f-2c93e73b02cf"
        expected = "GPU-" + raw
        self.assertEqual(M.cuda_uuid(raw), expected)
        self.assertEqual(M.cuda_uuid(expected), expected)
        roles = M.resolve_gpu_roles(
            {
                "base_gpu_id": 0,
                "enable_hisparse": True,
                "hisparse_config": {"backing_store": "cpu"},
            },
            lambda ordinal: SimpleNamespace(uuid=raw),
        )
        self.assertEqual(
            M.gpu_identity(roles), {expected: {"role": "compute", "visible_ordinal": 0}}
        )
        with self.assertRaisesRegex(ValueError, "Invalid CUDA device UUID"):
            M.cuda_uuid("invalid-device-identity")

    def test_mpu_gpu_roles_use_cuda_ordinal_uuids_not_nvml_row_order(self):
        info = {
            "base_gpu_id": 2,
            "enable_hisparse": True,
            "hisparse_config": {"backing_store": "mpu", "mpu_device_ids": [0, 4, 1, 3]},
        }
        uuids = {ordinal: f"GPU-cuda-{ordinal}" for ordinal in range(5)}
        roles = M.resolve_gpu_roles(
            info, lambda ordinal: SimpleNamespace(uuid=uuids[ordinal])
        )
        self.assertEqual(
            roles[0], {"visible_ordinal": 2, "uuid": "GPU-cuda-2", "role": "compute"}
        )
        self.assertEqual([row["visible_ordinal"] for row in roles[1:]], [0, 4, 1, 3])
        self.assertEqual({row["role"] for row in roles[1:]}, {"storage"})
        self.assertEqual(M.gpu_identity(roles), M.gpu_identity(list(reversed(roles))))
        for storage in ([], [2], [1, 1], [True], [0, 1, 2, 3, 4]):
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                M.resolve_gpu_roles(
                    info
                    | {
                        "hisparse_config": {
                            "backing_store": "mpu",
                            "mpu_device_ids": storage,
                        }
                    },
                    lambda ordinal: SimpleNamespace(uuid=uuids[ordinal]),
                )

    def test_cpu_gpu_roles_do_not_label_visible_unused_devices_as_mpu(self):
        info = {
            "base_gpu_id": 3,
            "enable_hisparse": True,
            "hisparse_config": {"backing_store": "cpu", "mpu_device_ids": [1, 2]},
        }
        roles = M.resolve_gpu_roles(
            info, lambda ordinal: SimpleNamespace(uuid=f"GPU-cuda-{ordinal}")
        )
        self.assertEqual(
            roles, [{"visible_ordinal": 3, "uuid": "GPU-cuda-3", "role": "compute"}]
        )

    def test_mpu_resource_snapshot_freezes_selected_uuids_and_reports_each_role(self):
        info = {
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "disaggregation_mode": "null",
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_hierarchical_cache": False,
            "speculative_algorithm": None,
            "enable_mixed_chunk": False,
            "enable_two_batch_overlap": False,
        }
        roles = [
            {"visible_ordinal": 3, "uuid": "GPU-compute", "role": "compute"},
            {"visible_ordinal": 1, "uuid": "GPU-storage", "role": "storage"},
        ]
        identity = {"path": "/results/forward-pid42.jsonl"}
        process = {"pid": 42, "start_ticks": 7}
        signature = M.server_signature(info)

        def request(port, route):
            return info if route == "server_info" else {}

        with patch.object(M, "trace_identity", return_value=identity), patch.object(
            M, "process_resource", return_value=process
        ), patch.object(M, "request", side_effect=request), patch.object(
            M.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="GPU-storage, 500\nGPU-compute, 200\n"),
        ) as query:
            snapshot = M.resource_snapshot(1, identity, [process], signature, roles)
            self.assertEqual(
                [row["uuid"] for row in snapshot["gpu"]], ["GPU-compute", "GPU-storage"]
            )
            self.assertEqual(snapshot["gpu"][1]["used_bytes"], 500 * 1024 * 1024)
            self.assertEqual(snapshot["gpu"][1]["role"], "storage")
            self.assertIn("--id=GPU-compute,GPU-storage", query.call_args.args[0])
            for raw in (
                "GPU-compute, 200\n",
                "GPU-compute, 200\nGPU-compute, 300\n",
                "GPU-compute, 200\nGPU-other, 300\n",
            ):
                query.return_value = SimpleNamespace(stdout=raw)
                with self.subTest(raw=raw), self.assertRaisesRegex(
                    ValueError, "UUID inventory"
                ):
                    M.resource_snapshot(1, identity, [process], signature, roles)

    def test_gpu_deltas_are_per_uuid_and_detect_role_or_storage_identity_changes(self):
        before = [
            {
                "uuid": "GPU-compute",
                "role": "compute",
                "visible_ordinal": 0,
                "used_bytes": 100,
            },
            {
                "uuid": "GPU-storage",
                "role": "storage",
                "visible_ordinal": 1,
                "used_bytes": 200,
            },
        ]
        after = [before[1] | {"used_bytes": 190}, before[0] | {"used_bytes": 110}]
        delta = M.gpu_used_deltas(before, after)
        self.assertEqual(delta["GPU-compute"]["used_delta_bytes"], 10)
        self.assertEqual(delta["GPU-storage"]["used_delta_bytes"], -10)
        self.assertEqual(
            M.gpu_used_deltas(
                [{"uuid": "GPU-only", "used_bytes": 10}],
                [{"uuid": "GPU-only", "used_bytes": 12}],
            )["GPU-only"]["used_delta_bytes"],
            2,
        )
        for key, value in (
            ("uuid", "GPU-replaced"),
            ("role", "compute"),
            ("visible_ordinal", 2),
        ):
            changed = copy.deepcopy(after)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.gpu_used_deltas(before, changed)

    def test_storage_allocation_identity_is_part_of_persistent_signature(self):
        before = {
            "backing_store": "mpu",
            "storage_device_ids": [1, 2, 3, 4],
            "allocated_backing_kv_bytes": 1 << 30,
        }
        for key, value in (
            ("storage_device_ids", [1, 2, 3, 5]),
            ("allocated_backing_kv_bytes", (1 << 30) + 64),
        ):
            with self.subTest(key=key):
                self.assertNotEqual(
                    M.capacity_signature(before),
                    M.capacity_signature(before | {key: value}),
                )

    def test_cancellation_requires_terminal_abort_and_real_phase(self):
        proof = M.validate_request(
            terminal(), {"prefill_tokens": 8, "decode_forwards": 3}
        )
        self.assertEqual(proof["actual_cancel_phase"], "decode")
        proof = M.validate_request(
            terminal(count=0), {"prefill_tokens": 4, "decode_forwards": 0}
        )
        self.assertEqual(proof["actual_cancel_phase"], "partial_prefill")
        proof = M.validate_request(
            terminal(count=0), {"prefill_tokens": 0, "decode_forwards": 0}
        )
        self.assertEqual(proof["actual_cancel_phase"], "no_forward_observed")
        proof = M.validate_request(
            terminal(count=1), {"prefill_tokens": 8, "decode_forwards": 0}
        )
        self.assertEqual(proof["actual_cancel_phase"], "prefill_complete_no_decode")
        for bad_row, bad_fact in (
            (terminal(count=20), {"prefill_tokens": 8, "decode_forwards": 19}),
            (terminal(), {"prefill_tokens": 4, "decode_forwards": 3}),
            (terminal(count=2), {"prefill_tokens": 8, "decode_forwards": 0}),
            (
                {**terminal(), "cancel_attempt": {"attempted": False}},
                {"prefill_tokens": 8, "decode_forwards": 3},
            ),
            (
                {**terminal(), "status": "running"},
                {"prefill_tokens": 8, "decode_forwards": 3},
            ),
        ):
            with self.subTest(row=bad_row, facts=bad_fact), self.assertRaises(
                ValueError
            ):
                M.validate_request(bad_row, bad_fact)

    def test_normal_output_requires_full_trace_even_after_late_cancel(self):
        output = terminal("length", 20)
        facts = {"prefill_tokens": 8, "decode_forwards": 19}
        self.assertTrue(M.validate_request(output, facts)["cancel_window_missed"])
        for key, value in (
            ("prefill_tokens", 7),
            ("prefill_tokens", 16),
            ("decode_forwards", 18),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                M.validate_request(output, {**facts, key: value})
        output["cancel_attempt"]["error"] = "HTTP failure"
        with self.assertRaises(ValueError):
            M.validate_request(output, facts)

    def test_idle_trace_requires_fresh_capacity_and_no_slot_leak(self):
        rows = [
            SESSION,
            {
                "type": "forward",
                "sequence": 0,
                "success": True,
                "forward_mode": "extend",
                "request_ids": ["r"],
                "extend_lens": [8],
            },
            {
                "type": "forward",
                "sequence": 1,
                "success": True,
                "forward_mode": "decode",
                "request_ids": ["r"],
                "extend_lens": None,
            },
            {
                "type": "capacity",
                "after_sequence": 1,
                "capacity": {
                    "tp_rank": 0,
                    "tp_size": 1,
                    "reserved_slots": 0,
                    "used_slots": 0,
                },
            },
            {"type": "idle_flush", "last_sequence": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"

            def put(items):
                path.write_text(
                    "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
                )

            put(rows)
            result = M.scan_idle_trace(path, {"r", "queued"}, SESSION)
            self.assertEqual(
                result["request_facts"]["r"],
                {"prefill_tokens": 8, "decode_forwards": 1},
            )
            self.assertTrue(result["request_facts"]["queued"]["no_forward_observed"])
            bad = copy.deepcopy(rows)
            bad[-2]["after_sequence"] = 0
            put(bad)
            with self.assertRaises(ValueError):
                M.scan_idle_trace(path, {"r"}, SESSION)
            bad = copy.deepcopy(rows)
            bad[-2]["capacity"]["used_slots"] = 1
            put(bad)
            with self.assertRaises(ValueError):
                M.scan_idle_trace(path, {"r"}, SESSION)
            bad = copy.deepcopy(rows)
            bad[2]["sequence"] = 2
            put(bad)
            with self.assertRaises(ValueError):
                M.scan_idle_trace(path, {"r"}, SESSION)

    def test_allocator_worklists_are_reported_but_not_persistent_shape_gates(self):
        original = {
            "physical_compute_slots": 65536,
            "retained_gpu_pool_bytes": 64,
            "gpu_memory_ledger": {
                "groups": {"fixed": {"bytes": 64}, "allocator_dynamic": {"bytes": 10}}
            },
        }
        changed = copy.deepcopy(original)
        changed["gpu_memory_ledger"]["groups"]["allocator_dynamic"]["bytes"] = 20
        self.assertEqual(M.capacity_signature(original), M.capacity_signature(changed))
        changed["gpu_memory_ledger"]["groups"]["fixed"]["bytes"] = 65
        self.assertNotEqual(
            M.capacity_signature(original), M.capacity_signature(changed)
        )

    def test_process_identity_handles_spaces_in_comm_and_records_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root / "42"
            process.mkdir()
            fields = ["S"] + ["0"] * 18 + ["999"] + ["0"] * 10
            (process / "stat").write_text(
                "42 (worker process) " + " ".join(fields), encoding="utf-8"
            )
            (process / "status").write_text(
                "VmRSS:\t12 kB\nVmHWM:\t20 kB\nVmSize:\t40 kB\n", encoding="utf-8"
            )
            result = M.process_resource(42, root)
            self.assertEqual(result["start_ticks"], 999)
            self.assertEqual(result["VmRSS"], 12 * 1024)

    def test_seeded_case_plan_is_replayable_and_never_fabricates_inputs(self):
        corpus = {
            8: [{"source_id": "short", "input_sha256": "a" * 64}],
            32: [{"source_id": "long", "input_sha256": "b" * 64}],
        }

        def make():
            rng = random.Random(1729)
            return [
                M.plan_case(rng, i, corpus, [64, 256, 1024], 0.5, "run")
                for i in range(50)
            ]

        self.assertEqual(make(), make())
        self.assertEqual({row["source_id"] for row in make()}, {"short", "long"})
        self.assertEqual(len({row["request_id"] for row in make()}), 50)

    def test_loopback_decode_cancel_uses_second_http_request_and_terminal_reason(self):
        state = {"abort": set()}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.end_headers()
                if self.path == "/abort_request":
                    with lock:
                        state["abort"].add(body["rid"])
                    return
                for count in range(1, 21):
                    with lock:
                        aborted = body["rid"] in state["abort"]
                    reason = "abort" if aborted else ("length" if count == 20 else None)
                    item = {
                        "output_ids": list(range(count)),
                        "meta_info": {
                            "id": body["rid"],
                            "prompt_tokens": 8,
                            "completion_tokens": count,
                            "finish_reason": {"type": reason} if reason else None,
                        },
                    }
                    self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
                    self.wfile.flush()
                    if reason:
                        break
                    time.sleep(0.004)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = M.stream_request(
                    server.server_port, list(range(8)), case(), 5, Path(directory)
                )
                self.assertEqual(result["status"], "terminal")
                self.assertTrue(result["cancel_attempt"]["attempted"])
                self.assertEqual(
                    result["response"]["meta_info"]["finish_reason"]["type"], "abort"
                )
                self.assertTrue(
                    all(chunk["delta_tokens"] == 1 for chunk in result["chunks"])
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_200_without_terminal_sse_is_failed(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = M.stream_request(
                    server.server_port,
                    list(range(8)),
                    case(cancel=None),
                    5,
                    Path(directory),
                )
                self.assertEqual(result["status"], "failed")
                self.assertIn("without terminal", result["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
