"""Allocator-lifetime and graph-pool accounting without a CUDA runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PATH = (
    Path(__file__).parents[3]
    / "python/sglang/srt/observability/experiment_memory_observer.py"
)
SPEC = importlib.util.spec_from_file_location("memory_observer_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_disabled_has_no_runtime_import():
    with patch.dict("os.environ", {}, clear=True):
        assert MODULE.ExperimentMemoryObserver.create(object()) is None


def test_history_capacity_is_explicit_and_positive():
    with patch.dict("os.environ", {}, clear=True):
        assert MODULE.history_limit_from_environment() == 500000
    with patch.dict(
        "os.environ", {"SGLANG_EXPERIMENT_MEMORY_HISTORY_ENTRIES": "2000000"}
    ):
        assert MODULE.history_limit_from_environment() == 2000000
    with patch.dict("os.environ", {"SGLANG_EXPERIMENT_MEMORY_HISTORY_ENTRIES": "0"}):
        try:
            MODULE.history_limit_from_environment()
        except ValueError:
            pass
        else:
            raise AssertionError("Nonpositive history capacity must be rejected")


def test_graph_segments_are_device_scoped_and_nonadditive():
    runner = SimpleNamespace(graphs={1: SimpleNamespace(pool=lambda: (1, 2))})
    segments = [
        {
            "device": device,
            "segment_pool_id": pool,
            "total_size": size,
            "allocated_size": size // 2,
            "active_size": size // 2,
        }
        for device, pool, size in [
            (0, [1, 2], 100),
            (1, [1, 2], 200),
            (0, [0, 0], 1000),
        ]
    ]
    result = MODULE.graph_pool_accounting({"segments": segments}, runner, 0)
    assert result["reserved_bytes"] == 100
    assert result["allocated_bytes"] == 50
    assert result["segment_count"] == 1
    assert (
        MODULE.graph_pool_accounting({"segments": segments}, None, 0)["reserved_bytes"]
        == 0
    )
    assert MODULE.graph_pool_accounting({}, runner, 0)["reserved_bytes"] is None


def test_transient_overlap_cross_phase_and_reused_addresses():
    events = [
        {"action": "alloc", "addr": 9, "size": 900},
        {
            "action": "alloc",
            "addr": 1,
            "size": 10,
            "frames": [{"filename": "quest_algorithm.py"}],
        },
        {
            "action": "alloc",
            "addr": 2,
            "size": 20,
            "frames": [{"filename": "attention.py"}],
        },
        {"action": "free_requested", "addr": 1},
        {"action": "free_completed", "addr": 1},
        {"action": "alloc", "addr": 1, "size": 40},
        {"action": "free_requested", "addr": 2},
        {"action": "free_requested", "addr": 1},
        {"action": "free_requested", "addr": 9},
        {"action": "alloc", "addr": 3, "size": 100},
    ]
    result = MODULE.transient_allocation_accounting(events, 1)
    assert result["phase_local_allocation_count"] == 3
    assert result["peak_live_bytes"] == 60
    assert result["phase_crossing_or_retained_allocation_count"] == 1
    assert result["peak_live_bytes_by_python_callsite"] == {
        "quest_score_topk_bounds_code": 10,
        "attention_backend_code": 20,
        "other_or_unattributed": 40,
    }
    assert result["category_peaks_additive"] is False
    assert "addr" not in result
    assert result["raw_addresses_emitted"] is False


def test_idle_hook_is_idempotent_before_and_after_wave(tmp_path):
    observer = object.__new__(MODULE.ExperimentMemoryObserver)
    observer._phase, observer._sequence = "decode", 4
    observer.context, observer.path = {}, tmp_path / "events.jsonl"
    phases = []

    def checkpoint(final=False):
        phases.append(observer._phase)
        observer._phase = None if final else observer._phase

    def begin(phase, metadata=None):
        observer._phase = phase

    observer.checkpoint, observer.begin_phase = checkpoint, begin
    observer.finish_idle()
    observer.finish_idle()
    assert phases == ["decode", "idle"]
    assert observer.path.read_text().count("memory_wave_complete") == 1
