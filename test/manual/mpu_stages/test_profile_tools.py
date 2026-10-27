"""Synthetic trace fixtures test evidence boundaries, never GPU behavior."""

import importlib.util
import sqlite3
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
            self.assertEqual(int(cmd[cmd.index("--context-length") + 1]), 8192 + 36)
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
