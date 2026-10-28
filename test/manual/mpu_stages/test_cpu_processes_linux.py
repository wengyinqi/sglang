"""Mandatory Linux integration; missing Gloo/platform fails instead of skipping."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "cpu_control_test", ROOT / "benchmark/mpu_stages/cpu_control.py"
)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


@pytest.mark.parametrize(
    "name,world,fault",
    [
        ("single", 1, None),
        ("four", 4, None),
        ("missing-rank", 4, "timeout"),
        ("rank-exit", 4, "exit"),
    ],
)
def test_real_cpu_processes(tmp_path, name, world, fault):
    root = Path(os.environ.get("MPU_CPU_CONTROL_EVIDENCE", tmp_path))
    result = CONTROL.run_case(root / name, world, fault)
    (root / f"{name}.json").write_text(json.dumps(result, indent=2))
    assert result["status"] == "passed"
    assert result["gpu_executed"] is False and result["deepep_executed"] is False
