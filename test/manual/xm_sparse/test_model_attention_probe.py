"""Portable diagnostic provenance guards, without Torch/CUDA execution."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "xm_layer_probe_test", ROOT / "benchmark/xm_sparse/probe_model_attention.py"
)
PROBE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROBE)


@pytest.fixture
def capture(tmp_path):
    config = {
        "steps": 3,
        "snapshot_layers": [0, 23, 47],
        "actual_batch": 2,
        "tool_sha256": "fixture-tool",
        "return_reference": True,
    }
    rows = []
    for step in range(3):
        for layer in range(48):
            row = {
                "step": step,
                "layer": layer,
                "tool_sha256": "fixture-tool",
                "batch": 2,
                "selected_payload_exact": True,
                "reference_returned": True,
                "performance_result": False,
            }
            if step == 0 and layer in config["snapshot_layers"]:
                path = tmp_path / f"layer{layer}.pt"
                path.write_bytes(b"portable fixture, not a model tensor")
                row["snapshot"] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": PROBE.digest(path),
                }
            rows.append(row)
    stats = tmp_path / "layer-stats-pid1.jsonl"
    stats.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path, config, rows, stats


def test_complete_capture_never_asserts_fused_model_or_performance(capture):
    output, config, _, _ = capture
    result = PROBE.validate_capture(output, config)
    assert result["layer_step_rows"] == 144
    assert result["fused_model_correctness_verified"] is False
    assert result["performance_result"] is False


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "tool", "payload", "return_mode", "snapshot"]
)
def test_partial_or_changed_diagnostics_fail_closed(capture, mutation):
    output, config, rows, stats = capture
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "tool":
        rows[0]["tool_sha256"] = "changed"
    elif mutation == "payload":
        rows[0]["selected_payload_exact"] = False
    elif mutation == "return_mode":
        rows[0]["reference_returned"] = False
    else:
        Path(rows[0]["snapshot"]["path"]).write_bytes(b"changed")
    stats.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        PROBE.validate_capture(output, config)


def test_startup_rejects_changed_tool_before_runtime_patch(tmp_path, monkeypatch):
    class Backend:
        def _run_fused_attention(*args):
            return None

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = types.ModuleType("sglang.srt.layers.attention.flashinfer_hisparse_backend")
    module.FlashInferHiSparseDecodeBackend = Backend
    module.quest_scatter_pack = lambda *args: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tool_sha256": "changed"}))
    with pytest.raises(RuntimeError, match="changed"):
        PROBE.install_worker(path)
    assert not getattr(Backend, "_xm_model_probe_installed", False)
