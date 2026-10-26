"""Portable acceptance-gate tests: no simulated result is hardware evidence."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "acceptance_under_test", REPO / "benchmark/mpu_stages/acceptance.py"
)
ACCEPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPT)


def recipe(family="glm-5.3-flash", variant="mpu"):
    tp = 1 if family == "qwen" else 4
    return {
        "schema_version": 1,
        "id": "test-stage",
        "family": family,
        "tp_size": tp,
        "variant": variant,
        "default_model": "org/model",
        "selector": "test",
        "hbm_preserves_selector": True,
        "variants": {
            "cpu": {"hisparse_config": {}},
            "mpu": {
                "hisparse_config": {"mpu_device_ids": [1] if tp == 1 else [4, 5, 6, 7]}
            },
            "hbm": {"hisparse_config": {}},
        },
    }


def args(root, devices=None):
    return NS(
        output_dir=root,
        mpu_device_ids=devices,
        capacity_ratios=[2],
        cpu_capacity_ratio=2,
        top_k=None,
        device_buffer_size=None,
        mpu_memory_reserve_bytes=None,
        model="org/model",
    )


class AcceptanceTests(unittest.TestCase):
    def test_model_success_without_fault_evidence_cannot_pass_correctness(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "manifest.json"
            group = {"status": "success"}
            report = {"status": "success", "groups": [group]}
            job = {"report": str(path)}
            step = {"id": "correctness", "kind": "model_run"}
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "failed")
            group.update(
                lifecycle_required=True,
                lifecycle_results=[
                    {"case": case, "status": "passed"}
                    for case in ("abort_prefill", "abort_decode", "retract_decode")
                ],
            )
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "passed")
            group["lifecycle_results"][0]["status"] = "window_missed"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "failed")

    def test_junit_skip_empty_failures_and_missing_reports_never_pass(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "report.xml"
            examples = [
                (
                    '<testsuites><testsuite><testcase name="actual"/></testsuite></testsuites>',
                    0,
                    "passed",
                ),
                (
                    '<testsuite><testcase><skipped message="no GPU"/></testcase></testsuite>',
                    0,
                    "blocked",
                ),
                ("<testsuite/>", 0, "blocked"),
                ("<testsuite><testcase><failure/></testcase></testsuite>", 0, "failed"),
                ("<testsuite><testcase/></testsuite>", 1, "failed"),
                ("invalid xml", 0, "failed"),
            ]
            for xml, code, expected in examples:
                with self.subTest(expected=expected, xml=xml):
                    path.write_text(xml, encoding="utf-8")
                    self.assertEqual(
                        ACCEPT.junit_result(path, code)["status"], expected
                    )
            self.assertEqual(
                ACCEPT.junit_result(path.with_name("absent.xml"), 0)["status"], "failed"
            )

    def test_all_four_actual_device_pairs_are_mapped_with_uuid_visibility(self):
        visibility = [f"GPU-{i}" for i in range(8)]
        for family in ("deepseek-v4-flash", "glm-5.3-flash"):
            jobs = ACCEPT.kernel_jobs(recipe(family), args(Path("results")), visibility)
            self.assertEqual(len(jobs), 6)
            pairs = jobs[1:5]
            for rank, job in enumerate(pairs):
                actual = job["env"]["CUDA_VISIBLE_DEVICES"].split(",")
                self.assertEqual(actual[0], visibility[rank])
                self.assertEqual(
                    actual[1 if family.startswith("deepseek") else 4],
                    visibility[4 + rank],
                )
                self.assertEqual(job["storage_ordinal"], 4 + rank)
                self.assertEqual(len(set(actual)), len(actual))
            concurrent = jobs[-1]
            self.assertEqual(
                concurrent["env"]["CUDA_VISIBLE_DEVICES"], ",".join(visibility)
            )
            self.assertEqual(concurrent["env"]["SGLANG_TEST_MPU_CONCURRENT"], "1")
            if family == "glm-5.3-flash":
                self.assertTrue(
                    all(
                        any("resolve_gpu" in t for t in job["targets"]) for job in pairs
                    )
                )

    def test_qwen_multishard_mapping_and_cpu_stages_do_not_require_mpu(self):
        jobs = ACCEPT.kernel_jobs(
            recipe("qwen"), args(Path("out"), [2, 4, 6, 7]), list(map(str, range(8)))
        )
        self.assertEqual(jobs[1]["env"]["CUDA_VISIBLE_DEVICES"], "0,2,4,6,7")
        self.assertEqual(jobs[1]["env"]["SGLANG_MPU_TEST_STORAGE_DEVICES"], "1,2,3,4")
        for family in ("qwen", "deepseek-v4-flash", "glm-5.3-flash"):
            baseline = ACCEPT.kernel_jobs(
                recipe(family, "cpu"), args(Path("out")), None
            )
            self.assertEqual(len(baseline), 1)
            self.assertNotIn("storage_ordinal", baseline[0])

    def test_visibility_is_never_treated_as_physical_id_list(self):
        self.assertEqual(ACCEPT.remap_ordinals([0, 2], ["5", "GPU-uuid", "3"]), "5,3")
        for invalid in ("", "0,0", "-1", "0,"):
            with self.assertRaises(ValueError):
                ACCEPT.visible_tokens(invalid)
        with self.assertRaises(ValueError):
            ACCEPT.remap_ordinals([4], ["GPU-0"])

    def test_correctness_uses_same_selector_hbm_oracle_and_is_a_performance_gate(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(name)
            control = NS(through="performance", probe_timeout=3)
            settings = args(root)
            settings.model = "local model"
            steps = ACCEPT.make_steps(
                recipe(),
                settings,
                control,
                [
                    "--stage",
                    "original.json",
                    "--output-dir=old",
                    "--execute",
                    "--model",
                    "local model",
                ],
            )
            ids = [step["id"] for step in steps]
            self.assertLess(ids.index("correctness"), ids.index("performance"))
            self.assertLess(ids.index("model_identity"), ids.index("correctness"))
            cmd = steps[-2]["commands"][0]["command"]
            self.assertEqual(cmd.count("--execute"), 1)
            self.assertIn("--include-hbm", cmd)
            self.assertIn("--lifecycle", cmd)
            self.assertNotIn("original.json", cmd)
            self.assertIn("local model", cmd)
            self.assertEqual(cmd[cmd.index("--stage") + 1], str(root / "stage.json"))
            self.assertIn("--compile-only", steps[1]["commands"][0]["command"])

    def test_portable_compile_plans_work_when_cuda_is_hidden(self):
        for through in ("portable", "compile"):
            for visibility in ("", "-1", "0"):
                with tempfile.TemporaryDirectory() as name, patch.dict(
                    os.environ, {"CUDA_VISIBLE_DEVICES": visibility}
                ):
                    steps = ACCEPT.make_steps(
                        recipe(),
                        args(Path(name)),
                        NS(through=through, probe_timeout=3),
                        [],
                    )
                    kernel_step = next(
                        step for step in steps if step["id"] == "kernels"
                    )
                    self.assertEqual(kernel_step["status"], "not_requested")
                    self.assertEqual(kernel_step["commands"], [])

    def test_cli_model_is_not_an_abbreviation_of_model_manifest(self):
        with tempfile.TemporaryDirectory() as name, patch.object(
            ACCEPT.COMMON, "runtime_info", return_value={}
        ):
            root = Path(name)
            model = root / "local model"
            model.mkdir()
            output = root / "plan"
            self.assertEqual(
                ACCEPT.main(
                    [
                        "--through",
                        "portable",
                        "--model",
                        str(model),
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            report = json.loads(
                (output / "acceptance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "planned")
            command = report["steps"][-2]["commands"][0]["command"]
            self.assertEqual(
                command[command.index("--model") + 1], str(model.resolve())
            )

    def test_blocked_layer_stops_all_later_commands_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "gpu.xml"
            report.write_text(
                "<testsuite><testcase><skipped/></testcase></testsuite>",
                encoding="utf-8",
            )
            first = {
                "id": "kernels",
                "phase": "kernels",
                "kind": "pytest",
                "status": "planned",
                "commands": [
                    {
                        "name": "gpu",
                        "command": ["never-launched"],
                        "report": str(report),
                        "env": {},
                    }
                ],
            }
            later = copy.deepcopy(first)
            later["id"] = "correctness"
            manifest = {"stage": recipe(), "steps": [first, later]}
            invocations = []

            def fake_runner(command, env, log, timeout):
                invocations.append(command)
                self.assertNotIn("PYTEST_ADDOPTS", env)
                return {"returncode": 0}

            with patch.dict(os.environ, {"PYTEST_ADDOPTS": "-k nothing"}):
                passed = ACCEPT.execute(
                    manifest,
                    args(root),
                    NS(model_manifest=None, layer_timeout=5, model_timeout=5),
                    fake_runner,
                )
            self.assertFalse(passed)
            self.assertEqual(len(invocations), 1)
            self.assertEqual(first["status"], "blocked")
            self.assertEqual(later["status"], "not_run_after_failure")
            self.assertFalse(
                manifest["evidence_scope"]["target_model_correctness_executed"]
            )

    def test_probe_mode_exit_and_artifact_must_agree(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "probe.json"
            step, job = {"kind": "probe", "id": "compile"}, {"report": str(path)}
            path.write_text(
                json.dumps(
                    {"schema_version": 1, "mode": "compile", "status": "blocked"}
                ),
                encoding="utf-8",
            )
            self.assertEqual(ACCEPT.external_result(step, job, 1)["status"], "blocked")
            path.write_text(
                json.dumps({"mode": "compile", "status": "passed"}), encoding="utf-8"
            )
            self.assertEqual(ACCEPT.external_result(step, job, 1)["status"], "failed")
            path.write_text(
                json.dumps({"mode": "device", "status": "passed"}), encoding="utf-8"
            )
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "failed")

    def test_compile_pass_requires_unchanged_nonempty_object_artifact(self):
        import hashlib

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            object_file = root / "fixture.o"
            object_file.write_bytes(b"compiler-output-fixture")
            path = root / "compile.json"
            report = {
                "schema_version": 1,
                "mode": "compile",
                "status": "passed",
                "capability_scope": "object_only",
                "checks": [
                    {
                        "name": "production_object_compile",
                        "status": "passed",
                        "evidence": {
                            "object": {
                                "path": str(object_file),
                                "exists": True,
                                "bytes": object_file.stat().st_size,
                                "sha256": hashlib.sha256(
                                    object_file.read_bytes()
                                ).hexdigest(),
                            }
                        },
                    }
                ],
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            step, job = {"kind": "probe", "id": "compile"}, {"report": str(path)}
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "passed")
            object_file.write_bytes(b"replaced")
            self.assertEqual(ACCEPT.external_result(step, job, 0)["status"], "failed")

    def test_partial_model_manifest_cannot_unlock_performance(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "manifest.json"
            job = {"report": str(path)}
            for groups in (
                [],
                [{"status": "planned"}],
                [{"status": "success"}, {"status": "failed"}],
            ):
                path.write_text(
                    json.dumps({"status": "success", "groups": groups}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    ACCEPT.external_result({"kind": "model_run"}, job, 0)["status"],
                    "failed",
                )

    def test_model_path_without_fingerprints_is_not_identity_validation(self):
        result = ACCEPT.model_identity(recipe(), NS(model="org/model"), None)
        self.assertEqual(result["status"], "blocked")

    def test_local_content_fingerprint_alone_does_not_certify_target_revision(self):
        with tempfile.TemporaryDirectory() as name:
            verifier = NS(
                verify_manifest=lambda *a, **kw: {
                    "ok": True,
                    "upstream_identity_verified": False,
                }
            )
            with patch.object(ACCEPT, "load", return_value=verifier):
                result = ACCEPT.model_identity(
                    recipe(), NS(model=name), Path(name) / "manifest.json"
                )
            self.assertEqual(result["status"], "blocked")

    def test_archived_source_and_kernel_distribution_are_recorded_without_fake_clean_status(
        self,
    ):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / ".mpu-source.json").write_text(
                json.dumps(
                    {
                        "provenance": "git_archive",
                        "source_commit": "a" * 40,
                        "source_archive_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(ACCEPT.COMMON, "REPO", root), patch.object(
                ACCEPT.COMMON, "capture", return_value="unavailable: git not installed"
            ):
                info = ACCEPT.COMMON.runtime_info()
            self.assertEqual(info["git_commit"], "a" * 40)
            self.assertEqual(info["git_commit_source"], "archive_build_marker")
            self.assertTrue(info["git_status"].startswith("unavailable:"))
            self.assertIn("sglang-kernel", info["packages"])
            self.assertIn("tilelang", info["packages"])

    def test_real_subprocess_logs_and_nonzero_exit_are_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            log = Path(name) / "process.log"
            result = ACCEPT.run_command(
                [sys.executable, "-c", "print('fixture'); raise SystemExit(7)"],
                os.environ.copy(),
                log,
                30,
            )
            self.assertEqual(result["returncode"], 7)
            self.assertIn("fixture", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
