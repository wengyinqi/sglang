"""Actual CUDA/NVML UUID interoperability; does not exercise a peer GPU."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "mixed_stability_cuda", ROOT / "benchmark/mpu_stages/mixed_stability.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA GPU")
def test_actual_cuda_uuid_matches_nvml_identity(record_property):
    # CPU backing deliberately selects only the compute card, even when more
    # physical GPUs are NVML-visible. The stage02 MPU device gate is separate.
    roles = M.resolve_gpu_roles(
        {
            "base_gpu_id": 0,
            "enable_hisparse": True,
            "hisparse_config": {"backing_store": "cpu"},
        }
    )
    assert len(roles) == 1 and roles[0]["role"] == "compute"
    uuid = roles[0]["uuid"]
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--id=" + uuid,
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).strip()
    rows = [line.split(",") for line in raw.splitlines()]
    assert len(rows) == 1 and rows[0][0].strip() == uuid
    sample = [{**roles[0], "used_bytes": int(rows[0][1].strip()) * 1024 * 1024}]
    assert M.gpu_used_deltas(sample, sample)[uuid]["used_delta_bytes"] == 0
    record_property("actual_cuda_nvml_uuid", uuid)
    record_property("gpu_roles", json.dumps(roles))
    record_property("mpu_p2p_executed", "false")
