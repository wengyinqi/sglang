"""Synthetic trace fixtures test evidence boundaries, never GPU behavior."""

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/mpu_stages" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE, ANALYZE = load("profile_decode"), load("profile_analyze")


class ProfileTests(unittest.TestCase):
    def test_plan_uses_existing_decode_trigger_and_sufficient_request_length(self):
        with tempfile.TemporaryDirectory() as directory:
            stage, args = PROFILE.STAGE.parse_args(
                ["--input-lens", "8192", "--output-len", "4", "--output-dir", directory]
            )
            plan = PROFILE.plan(stage, args, 32)
            self.assertIn("--capture-range=cudaProfilerApi", plan["server_command"])
            self.assertEqual(plan["profile_request"]["profile_stages"], ["decode"])
            self.assertEqual(plan["environment"], {"SGLANG_PROFILE_V2": "1"})
            cmd = plan["server_command"]
            self.assertEqual(int(cmd[cmd.index("--context-length") + 1]), 8192 + 36 + 2)
            self.assertFalse(plan["performance_result"])
            self.assertEqual(plan["workload"]["output_len"], 36)

    def test_portable_or_mismatched_acceptance_cannot_unlock_capture(self):
        with self.assertRaises(ValueError):
            PROFILE.correctness_gate(
                {"evidence_scope": {"target_model_correctness_executed": False}},
                {},
                {},
                None,
                {},
            )

    def test_mpu_stage_cpu_profile_uses_reference_ratio_not_target_ratio(self):
        stage, args = PROFILE.STAGE.parse_args(
            [
                "--cpu-capacity-ratio",
                "3",
                "--capacity-ratios",
                "1",
                "2",
                "4",
                "8",
                "--mpu-device-ids",
                "1",
                "2",
                "3",
                "4",
            ]
        )
        stage["variant"] = "mpu"
        cpu = PROFILE.plan(stage, args, 32, "cpu", 1)
        mpu = PROFILE.plan(stage, args, 32, "mpu", 1)
        cpu_command, mpu_command = cpu["server_command"], mpu["server_command"]
        cpu_config = json.loads(cpu_command[cpu_command.index("--hisparse-config") + 1])
        mpu_config = json.loads(mpu_command[mpu_command.index("--hisparse-config") + 1])
        self.assertEqual(cpu_config["host_to_device_ratio"], 3)
        self.assertEqual(mpu_config["mpu_capacity_ratio"], 1)

        runtime = {"git_commit": "a" * 40, "git_status": ""}
        receipt = {
            "stage": stage,
            "runtime": runtime,
            "evidence_scope": {"target_model_correctness_executed": True},
            "steps": [
                {
                    "id": "correctness",
                    "status": "passed",
                    "results": [{"status": "passed"}],
                }
            ],
        }
        settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        model_report = {"stage": stage, "runtime": runtime, "settings": settings}
        PROFILE.correctness_gate(receipt, stage, runtime, args, model_report, "cpu")
        args.capacity_ratios = [16]
        with self.assertRaisesRegex(ValueError, "capacity ratio"):
            PROFILE.correctness_gate(receipt, stage, runtime, args, model_report, "mpu")
        settings["cpu_capacity_ratio"] = 4
        with self.assertRaisesRegex(ValueError, "cpu_capacity_ratio"):
            PROFILE.correctness_gate(receipt, stage, runtime, args, model_report, "cpu")
        settings["cpu_capacity_ratio"] = 3
        settings["mpu_device_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "mpu_device_ids"):
            PROFILE.correctness_gate(receipt, stage, runtime, args, model_report, "cpu")

    def test_mpu_campaign_reference_profiles_only_map_their_actual_compute_device(self):
        stage, args = PROFILE.STAGE.parse_args(
            ["--mpu-device-ids", "1", "2", "3", "4", "--mpu-memory-reserve-bytes", "0"]
        )
        for variant in ("cpu", "hbm"):
            with self.subTest(variant=variant):
                mapping = PROFILE.profile_device_mapping(stage, args, variant)
                self.assertEqual(mapping["compute_device_ids"], [0])
                self.assertEqual(mapping["mpu_device_ids"], [])
                self.assertEqual(mapping["pairs"], [])
                self.assertIsNone(mapping["mpu_memory_reserve_bytes"])
        mapping = PROFILE.profile_device_mapping(stage, args, "mpu")
        self.assertEqual(mapping["compute_device_ids"], [0])
        self.assertEqual(mapping["mpu_device_ids"], [1, 2, 3, 4])
        self.assertEqual(mapping["mpu_memory_reserve_bytes"], 0)
        self.assertEqual(
            mapping["pairs"],
            [{"compute": 0, "storage": device} for device in range(1, 5)],
        )

    def test_mpu_kv_kernels_are_visible_in_bottleneck_categories(self):
        bottleneck = load("bottleneck_analyze")
        self.assertEqual(
            bottleneck.category("mpu_transfer_kernel<1024>"), "kv_resolve_and_pack"
        )
        self.assertEqual(
            bottleneck.category("load_cache_from_mpu_to_device_buffer"),
            "kv_resolve_and_pack",
        )
        self.assertEqual(bottleneck.category("unknown_kernel"), "unclassified")

    def test_stable_profile_budget_keeps_trigger_and_capture_in_the_same_request(self):
        stage, args = PROFILE.STAGE.parse_args(
            [
                "--input-lens",
                "8192",
                "--output-len",
                "4",
                "--cpu-capacity-ratio",
                "3",
                "--capacity-ratios",
                "1",
            ]
        )
        args.profile_stable_steps = 16
        profile = PROFILE.plan(stage, args, 64, "cpu", 8)
        self.assertEqual(profile["workload"]["output_len"], 84)
        command = profile["server_command"]
        config = json.loads(command[command.index("--hisparse-config") + 1])
        self.assertEqual(config["host_to_device_ratio"], 3)
        self.assertGreaterEqual(
            int(command[command.index("--context-length") + 1]), 8192 + 84
        )

    def test_live_stable_trigger_rejects_wrong_batch_and_propagates_request_failure(
        self,
    ):
        snapshot = {
            "success": True,
            "forward_mode": "decode",
            "consecutive_stable_decode_steps": 16,
            "batch_size": 2,
            "request_ids": ["a", "b"],
        }
        future = Future()
        with patch.object(
            PROFILE, "progress", side_effect=[snapshot | {"batch_size": 1}, snapshot]
        ), patch.object(PROFILE.time, "sleep"):
            self.assertEqual(PROFILE.wait_stable(1, future, 16, 2, 1), snapshot)
        failed = Future()
        failed.set_exception(RuntimeError("actual request failed"))
        with self.assertRaisesRegex(RuntimeError, "actual request failed"):
            PROFILE.wait_stable(1, failed, 16, 2, 1)

    def test_live_stable_trigger_requires_distinct_real_request_ids(self):
        snapshot = {
            "success": True,
            "forward_mode": "decode",
            "consecutive_stable_decode_steps": 16,
            "batch_size": 2,
            "request_ids": ["a", "a"],
        }
        with patch.object(
            PROFILE, "progress", return_value=snapshot
        ), self.assertRaisesRegex(ValueError, "distinct actual request IDs"):
            PROFILE.wait_stable(1, Future(), 16, 2, 1)

    def fixture(self, root, kernels):
        path = root / "trace.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(deviceId INTEGER,start INTEGER,end INTEGER)"
            )
            conn.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)", kernels
            )
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(copyKind INTEGER,bytes INTEGER,start INTEGER,end INTEGER)"
            )
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1,64,0,10)")
        conn.close()
        profile = {
            "status": "captured_pending_role_analysis",
            "devices": [
                {"uuid": "GPU-compute", "role": "compute"},
                {"uuid": "GPU-storage", "role": "storage"},
            ],
            "artifacts": [{"path": str(path), "sha256": ANALYZE.sha256(path)}],
        }
        mapping = {
            "schema_version": 1,
            "mapping_source": "synthetic fixture only",
            "devices": [
                {"trace_device_id": 3, "uuid": "GPU-compute", "role": "compute"},
                {"trace_device_id": 7, "uuid": "GPU-storage", "role": "storage"},
            ],
        }
        return path, profile, mapping

    def test_role_activity_is_not_inferred_from_cuda_visible_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, mapping = self.fixture(Path(directory), [(3, 100, 200)])
            report = ANALYZE.analyze(path, mapping, profile)
            self.assertEqual(
                report["storage_kernel_check"], "no_storage_kernels_observed"
            )
            self.assertEqual(report["memcpy"][0]["bytes"], 64)
            self.assertFalse(report["host_kv_payload_absence_proven"])
            self.assertFalse(report["total_kv_traffic_measured"])
            self.assertFalse(report["mapping_verified_automatically"])

    def test_empty_unknown_device_or_storage_execution_never_passes(self):
        for rows, status in (
            ([], "blocked"),
            ([(0, 0, 10)], "blocked"),
            ([(3, 0, 10), (7, 0, 10)], "failed"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path, profile, mapping = self.fixture(Path(directory), rows)
                self.assertEqual(
                    ANALYZE.analyze(path, mapping, profile)["status"], status
                )

    def test_trace_tampering_and_uuid_role_substitution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, mapping = self.fixture(Path(directory), [(3, 0, 10)])
            mapping["devices"][0]["role"] = "storage"
            with self.assertRaisesRegex(ValueError, "bijectively"):
                ANALYZE.analyze(path, mapping, profile)
            with path.open("ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs"):
                ANALYZE.analyze(path, mapping, profile)


if __name__ == "__main__":
    unittest.main()
