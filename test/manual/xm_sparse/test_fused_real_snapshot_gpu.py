"""Exact replay of externally retained real B4 model Q/K/V, not a model gate."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Sealed frozen source014 teacher-forced step0 captures. The old kernel has
# known BF16 differences here, including six elements in the worst layer36.
FIXTURE_SHA256 = {
    0: "aa662795ece9bbf0bb2da87d2521ee475ca643d34081b4010a1af47468ea815a",
    23: "4854c77dd82a791fda1abf968cc3bc45adcf064a2fb7bb3fbde16b5d2fbd2c5b",
    36: "11b0855e251215cb1ce5becd9d041a27d2c676dbf779148062045a46802ab4df",
    47: "020a1b47ee8616b20c5a762ddd7a88469bb949586d69ec8f37d2eb288ea61d61",
}


@pytest.mark.skipif(
    os.environ.get("SGLANG_RUN_XM_TMA_GPU_TESTS") != "1"
    or not os.environ.get("SGLANG_XM_REAL_SNAPSHOT_DIR"),
    reason="Requires Linux SM90 and the four externally retained real snapshots",
)
def test_real_b4_snapshot_exact_fi_hot_mapped_and_graph():
    import torch

    if sys.platform != "linux" or not torch.cuda.is_available():
        pytest.skip("Linux CUDA runtime required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    directory = Path(os.environ["SGLANG_XM_REAL_SNAPSHOT_DIR"])
    paths = sorted(directory.glob("step0-layer*.pt"))
    # Include the actual source014 worst layer, rather than allowing an
    # arbitrary easier tensor fixture to satisfy the numerical regression.
    layers = sorted(int(path.stem.split("layer")[1].split("-")[0]) for path in paths)
    assert layers == [0, 23, 36, 47]
    for path in paths:
        layer = int(path.stem.split("layer")[1].split("-")[0])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA256[layer]
        data = torch.load(path, map_location="cpu", weights_only=True)
        assert data["selected_k"].shape == (4, 2048, 4, 128)
        assert data["selected_v"].shape == (4, 2048, 4, 128)
        assert data["stats"]["selected_payload_exact"]
        assert torch.all(data["valid"]).item()
        assert torch.all(data["actual_lens"] == 2048).item()
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "xm_real_b4_snapshot_replay", root / "benchmark/xm_sparse/probe_mma_snapshot.py"
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    records = replay.replay_snapshots(directory, include_graph=True)
    assert len(records) == 4
    receipt = os.environ.get("SGLANG_XM_REAL_SNAPSHOT_RECEIPT")
    if receipt:
        path = Path(receipt)
        assert not path.exists(), "Do not overwrite a numerical evidence receipt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "exact_real_b4_same_qkv_snapshot_regression",
                    "records": records,
                    "fused_model_correctness_verified": False,
                    "performance_result": False,
                },
                indent=2,
            )
            + "\n"
        )
    for record in records:
        for name in (
            "same_dense_fi_vs_saved_fi",
            "mma_vs_fi",
            "mapped_vs_hot",
            "graph_hot_vs_fi",
            "graph_mapped_vs_fi",
        ):
            assert record[name]["torch_equal"], (record["snapshot"], name, record[name])
        for name in (
            "mapped_refill_k_exact",
            "mapped_refill_v_exact",
            "graph_refill_k_exact",
            "graph_refill_v_exact",
            "graph_addresses_stable",
        ):
            assert record[name], (record["snapshot"], name)
        assert record["fused_model_correctness_verified"] is False
        assert record["performance_result"] is False
