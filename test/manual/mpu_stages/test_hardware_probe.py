"""CPU tests for probe orchestration/evidence; never simulate passing CUDA."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmark/mpu_stages/hardware_probe.py"
SPEC = importlib.util.spec_from_file_location("hardware_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def recipe(tp=4, variant="mpu"):
    return {
        "schema_version": 1,
        "id": "fixture",
        "family": "glm-5.3-flash",
        "tp_size": tp,
        "variant": variant,
        "variants": {
            "mpu": {
                "hisparse_config": {"mpu_device_ids": [4, 5, 6, 7] if tp == 4 else [1]}
            }
        },
    }


class ProbeUnitTests(unittest.TestCase):
    def test_worker_isolates_both_production_loader_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = Namespace(
                python=sys.executable,
                timeout=1,
                output=root / "probe.json",
                jit_cache_root=root / "cache",
            )
            with mock.patch.object(probe, "wait_for_worker", return_value=1) as worker:
                probe.run_worker(args, root / "reports", "fixture", "jit_factory")
            env = worker.call_args.kwargs["env"]
            cache = Path(env["SGLANG_JIT_CACHE_DIR"])
            self.assertTrue(cache.is_relative_to(root / "cache"))
            self.assertEqual(Path(env["TVM_FFI_CACHE_DIR"]), cache / "tvm-ffi")
            self.assertFalse(cache.is_relative_to(root / "reports"))

    def test_status_aggregation_never_passes_empty_or_blocked(self):
        self.assertEqual(probe.aggregate([]), "blocked")
        self.assertEqual(
            probe.aggregate([probe.check("x", "passed"), probe.check("y", "blocked")]),
            "blocked",
        )
        self.assertEqual(
            probe.aggregate([probe.check("x", "blocked"), probe.check("y", "failed")]),
            "failed",
        )

    def test_environment_and_code_failures_are_distinct(self):
        for error in (
            ModuleNotFoundError("torch"),
            probe.Blocked("no device"),
            RuntimeError("Found no NVIDIA driver"),
        ):
            self.assertEqual(probe.classify_exception(error), "blocked")
        for error in (
            SyntaxError("bad source"),
            RuntimeError("nvcc: invalid kernel argument"),
            ImportError("undefined symbol: mismatch"),
        ):
            self.assertEqual(probe.classify_exception(error), "failed")

    def test_core_pin_comparison_normalizes_names_and_accepts_torch_cu130(self):
        source = '"torch==2.9.1", "flashinfer_python[cu13]==0.6.8.post1", "sglang-kernel==0.4.1.post1"'
        expected = {
            "torch": "2.9.1",
            "flashinfer-python": "0.6.8.post1",
            "sglang-kernel": "0.4.1.post1",
        }
        actual = {
            **expected,
            "torch": "2.9.1+cu130",
            "sglang-kernel": "0.4.1.post1+cu130",
        }
        result = probe.compare_environment_pins(source, expected, actual)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["evidence"]["actual_packages"]["torch"], "2.9.1+cu130")
        for version in ("0.4.1.post1+cu128", "0.4.10+cu130"):
            self.assertEqual(
                probe.compare_environment_pins(
                    source, expected, {**actual, "sglang-kernel": version}
                )["status"],
                "failed",
            )
        self.assertEqual(
            result["evidence"]["package_checks"][0]["evidence"]["source_pins"],
            ["2.9.1"],
        )

    def test_missing_core_packages_block_but_wrong_generation_or_source_fail(self):
        source = '"torch==2.14.1", "tilelang==0.1.12"'
        expected = {"torch": "2.14.1", "tilelang": "0.1.12"}
        missing = probe.compare_environment_pins(
            source, expected, {"torch": "2.14.1", "tilelang": None}
        )
        self.assertEqual(missing["status"], "blocked")
        for torch_version in ("2.9.1+cu130", "2.14.1+cu128"):
            result = probe.compare_environment_pins(
                source, expected, {**expected, "torch": torch_version}
            )
            self.assertEqual(result["status"], "failed")
        self.assertEqual(
            probe.compare_environment_pins(
                source + ', "torch==2.9.1"', expected, expected
            )["status"],
            "failed",
        )
        self.assertEqual(
            probe.compare_environment_pins('"torch==2.14.1"', expected, expected)[
                "status"
            ],
            "failed",
        )

    def test_mapping_covers_tp4_pairs_and_qwen_fanout(self):
        mapping = probe.device_mapping(recipe())
        self.assertEqual(
            mapping["pairs"],
            [{"compute": rank, "storage": rank + 4} for rank in range(4)],
        )
        mapping = probe.device_mapping(recipe(tp=1), [0], [1, 2, 3, 4])
        self.assertEqual(
            mapping["pairs"], [{"compute": 0, "storage": peer} for peer in [1, 2, 3, 4]]
        )

    def test_cpu_variant_requires_only_compute_devices(self):
        mapping = probe.device_mapping(recipe(variant="cpu"))
        self.assertEqual(mapping["compute_device_ids"], [0, 1, 2, 3])
        self.assertEqual(mapping["mpu_device_ids"], [])
        self.assertEqual(mapping["pairs"], [])
        with self.assertRaisesRegex(ValueError, "CPU backing"):
            probe.device_mapping(recipe(variant="cpu"), mpu_ids=[4, 5, 6, 7])

    def test_memory_reserve_override_is_part_of_mapping(self):
        self.assertEqual(
            probe.device_mapping(recipe())["mpu_memory_reserve_bytes"], 1 << 30
        )
        self.assertEqual(
            probe.device_mapping(recipe(), memory_reserve_bytes=0)[
                "mpu_memory_reserve_bytes"
            ],
            0,
        )
        self.assertIsNone(
            probe.device_mapping(recipe(variant="cpu"))["mpu_memory_reserve_bytes"]
        )
        with self.assertRaises(ValueError):
            probe.device_mapping(recipe(), memory_reserve_bytes=-1)

    def test_invalid_mappings_are_rejected_before_hardware(self):
        for compute, storage in (
            ([0, 1, 2], [4, 5, 6, 7]),
            ([0, 1, 2, 3], [4]),
            ([0, 1, 2, 3], [3, 5, 6, 7]),
            ([0, 1, 2, 3], [4, 4, 6, 7]),
        ):
            with self.subTest(compute=compute, storage=storage), self.assertRaises(
                ValueError
            ):
                probe.device_mapping(recipe(), compute, storage)
        with self.assertRaises(ValueError):
            probe.device_mapping(recipe(tp=1), mpu_ids=[1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "rank-to-device"):
            probe.device_mapping(recipe(tp=1), [2], [4])

    def test_family_uses_correct_source_tree(self):
        self.assertEqual(
            probe.FAMILIES["qwen"]["kernel_module"], "sglang.jit_kernel.mpu_io"
        )
        for name in ("deepseek-v4-flash", "glm-5.3-flash"):
            self.assertEqual(
                probe.FAMILIES[name]["kernel_module"], "sglang.kernels.ops.kvcache.mpu"
            )

    def test_current_production_factory_is_literal_and_exports_all_wrappers(self):
        stage = probe.load_recipe(SCRIPT.with_name("stage.json"))
        spec = probe.production_spec(probe.FAMILIES[stage["family"]])
        source = probe.wrapper_source(spec)
        self.assertIn("TVM_FFI_DLL_EXPORT_TYPED_FUNC", source)
        self.assertEqual(
            source.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC"), len(spec["wrappers"])
        )
        for name, target in spec["wrappers"]:
            self.assertIn(f"({name}, ({target}))", source)
        for path in spec["sources"]:
            self.assertTrue(Path(path).is_file())

    def test_compile_command_is_object_only_and_explicit_hopper(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            command = probe.compile_command(
                "/cuda/bin/nvcc", "prod.cu", "prod.o", ["/real include"]
            )
        self.assertIn("-c", command)
        self.assertIn("-gencode=arch=compute_90a,code=sm_90a", command)
        self.assertIn("-DSGL_CUDA_ARCH=900", command)
        self.assertIn("-I/real include", command)
        self.assertNotIn("-shared", command)

    def test_missing_nvcc_is_blocked_before_any_hardware_import(self):
        stage = probe.load_recipe(SCRIPT.with_name("stage.json"))
        with mock.patch.object(probe, "find_nvcc", return_value=None):
            with self.assertRaisesRegex(probe.Blocked, "not CUDA compilation"):
                probe.compile_object({"family": stage["family"], "work_dir": "/unused"})

    def test_no_artifact_or_invalid_status_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "result.json"
            self.assertEqual(
                probe.process_result(["child"], 0, report, "out", "err")["status"],
                "failed",
            )
            report.write_text('{"name":"x","status":"skipped"}', encoding="utf-8")
            self.assertEqual(
                probe.process_result(["child"], 0, report, "out", "err")["status"],
                "failed",
            )
            report.write_text('{"name":"x","status":"passed"}', encoding="utf-8")
            self.assertEqual(
                probe.process_result(["child"], 7, report, "out", "err")["status"],
                "failed",
            )

    def test_real_missing_import_child_is_nonzero_and_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, report = Path(tmp) / "request.json", Path(tmp) / "result.json"
            request.write_text(
                json.dumps(
                    {"kind": "import", "module": "_mpu_probe_nonexistent_dependency_"}
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--_worker", str(request), str(report)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(
                result["evidence"]["exception_type"], "ModuleNotFoundError"
            )
            self.assertIn("packages", result["runtime"])

    def test_real_worker_timeout_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / "out").open("w") as out, (Path(tmp) / "err").open(
                "w"
            ) as err:
                with self.assertRaises(subprocess.TimeoutExpired):
                    probe.wait_for_worker(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        cwd=ROOT,
                        env=os.environ.copy(),
                        stdout=out,
                        stderr=err,
                        timeout=0.1,
                    )

    def test_termination_handler_unwinds_and_cleans_worker_group(self):
        with self.assertRaises(probe.ProbeInterrupted):
            probe.termination_handler(signal.SIGTERM, None)
        process = mock.Mock(pid=12345)
        process.wait.side_effect = [probe.ProbeInterrupted("outer termination"), 0]
        # This checks process-control logic only, never mocks CUDA availability.
        with mock.patch.object(
            probe.subprocess, "Popen", return_value=process
        ), mock.patch.object(probe.sys, "platform", "linux"), mock.patch.object(
            probe.os, "killpg", create=True
        ) as killpg, mock.patch.object(
            probe.signal, "SIGKILL", 9, create=True
        ):
            with self.assertRaises(probe.ProbeInterrupted):
                probe.wait_for_worker(
                    ["child"], cwd=ROOT, env={}, stdout=None, stderr=None, timeout=30
                )
        killpg.assert_called_once_with(12345, 9)
        self.assertEqual(process.wait.call_count, 2)

    if sys.platform == "linux":

        def test_linux_sigterm_reaps_real_worker_and_grandchild(self):
            # Defined only on Linux: Windows CPU results do not claim to have
            # executed POSIX signal/process-group behavior.
            with tempfile.TemporaryDirectory() as tmp:
                pid_file = Path(tmp) / "pids.json"
                child_code = (
                    "import json,os,pathlib,subprocess,sys,time; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                    f"pathlib.Path({str(pid_file) + '.tmp'!r}).write_text(json.dumps([os.getpid(),p.pid])); "
                    f"os.replace({str(pid_file) + '.tmp'!r},{str(pid_file)!r}); "
                    "time.sleep(30)"
                )
                child_command = [sys.executable, "-c", child_code]
                parent_code = f"""
import importlib.util,os,signal,subprocess
spec=importlib.util.spec_from_file_location('probe',{str(SCRIPT)!r})
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
signal.signal(signal.SIGTERM,probe.termination_handler)
try:
    probe.wait_for_worker({child_command!r},cwd={str(ROOT)!r},env=os.environ.copy(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
except probe.ProbeInterrupted:
    raise SystemExit(2)
"""
                parent = subprocess.Popen(
                    [sys.executable, "-c", parent_code],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                pids = []
                try:
                    deadline = time.monotonic() + 8
                    while (
                        not pid_file.exists()
                        and time.monotonic() < deadline
                        and parent.poll() is None
                    ):
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists(), "Child did not start")
                    pids = json.loads(pid_file.read_text())
                    parent.send_signal(signal.SIGTERM)
                    self.assertEqual(parent.wait(timeout=5), 2)

                    def running(pid):
                        try:
                            return (
                                Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
                            )
                        except FileNotFoundError:
                            return False

                    deadline = time.monotonic() + 3
                    while (
                        any(running(pid) for pid in pids)
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                    self.assertFalse(
                        any(running(pid) for pid in pids),
                        "Worker descendant survived SIGTERM cleanup",
                    )
                finally:
                    if parent.poll() is None:
                        parent.kill()
                        parent.wait()
                    if pids:
                        try:
                            os.killpg(pids[0], signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_unsupported_host_cli_writes_blocked_nonzero(self):
        if sys.platform == "linux":
            return  # Linux CI may own real CUDA hardware; this test never runs it.
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("imports", "compile", "device"):
                report = Path(tmp) / f"{mode}.json"
                command = [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    mode,
                    "--output",
                    str(report),
                ]
                if mode == "compile":
                    command.append("--compile-only")
                proc = subprocess.run(
                    command, capture_output=True, text=True, timeout=30, check=False
                )
                self.assertNotEqual(proc.returncode, 0)
                result = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(result["schema_version"], 1)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["checks"][0]["name"], "supported_platform")
                self.assertTrue(result["source"]["kernel_module"]["exists"])


if __name__ == "__main__":
    unittest.main()
