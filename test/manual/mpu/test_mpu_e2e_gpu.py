"""Opt-in service acceptance. No model download or server launch by default."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "mpu_runner", REPO / "benchmark/hisparse_mpu/run.py"
)
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)


@unittest.skipUnless(
    os.environ.get("SGLANG_MPU_E2E_MODEL"),
    "Set SGLANG_MPU_E2E_MODEL to a local Qwen model to opt in",
)
class TestMPUEndToEnd(unittest.TestCase):
    def test_hbm_mpu_eager_graph_and_slot_reuse(self):
        self.assertEqual(sys.platform, "linux", "MPU acceptance requires Linux CUDA")
        model = Path(os.environ["SGLANG_MPU_E2E_MODEL"])
        self.assertTrue(model.is_dir(), "Use an existing local model directory")
        devices = os.environ.get("SGLANG_MPU_E2E_DEVICES", "1").split(",")
        with tempfile.TemporaryDirectory(prefix="mpu-e2e-") as temporary:
            output = Path(temporary) / "result"
            command = [
                sys.executable,
                str(REPO / "benchmark/hisparse_mpu/correctness.py"),
                "--model",
                str(model),
                "--output-len",
                "80",
                "--top-k",
                "128",
                "--device-buffer-size",
                "256",
                "--max-total-tokens",
                "4096",
                "--chunked-prefill-size",
                "512",
                "--capacity-ratios",
                "1",
                "--mpu-device-ids",
                *devices,
                "--output-dir",
                str(output),
                "--execute",
            ]
            # Runner owns/cleans every server process group. Overall timeout is
            # configured through its per-server startup and per-request limits.
            result = subprocess.run(
                command, cwd=REPO, env=RUN.BASE.subprocess_env(), check=False
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result.returncode, 0, json.dumps(manifest, indent=2))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(len(manifest["groups"]), 4)


if __name__ == "__main__":
    unittest.main()
