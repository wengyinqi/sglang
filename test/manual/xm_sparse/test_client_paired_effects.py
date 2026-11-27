"""Portable synthetic HTTP/statistics contracts, never model/GPU evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "portable_client_effects", ROOT / "benchmark/xm_sparse/client_paired_effects.py"
)
CLIENT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLIENT)


@pytest.fixture
def http_fixture():
    raw = {
        "request_ids": ["r"],
        "completed": 1,
        "input_lens": [16],
        "output_lens": [3],
        "request_latencies": [0.35],
        "ttfts": [0.1],
        "request_start_monotonic_ns": [1_000_000_000],
        "request_completion_monotonic_ns": [1_350_000_000],
        "generated_texts": ["abc"],
        "stream_frames": [
            [
                {
                    "received_monotonic_ns": 1_100_000_000 + i * 100_000_000,
                    "cumulative_completion_tokens": i + 1,
                    "new_tokens": 1,
                    "finish_reason": "length" if i == 2 else None,
                }
                for i in range(3)
            ]
        ],
        "duration": 0.35,
        "output_throughput": 3 / 0.35,
    }
    sample = {
        "input_len": 16,
        "request_average_tpot_ms": 125,
        "client_ttft_ms": 100,
        "complete_request_output_tokens_per_second": 3 / 0.35,
        "actual_decode_batch_distribution": {"1": 2},
        "client_timing": {
            "all_frames_single_token": True,
            "single_token_arrival_gap_ms": {"mean": 100, "samples": 2},
            "requests": [
                {
                    "request_id": "r",
                    "request_latency_ms": 350,
                    "ttft_ms": 100,
                    "request_average_tpot_ms": 125,
                    "raw_chunk_gap_ms": {"mean": 100, "samples": 2},
                }
            ],
        },
    }
    return raw, sample


def paired_rows():
    return [
        {
            "arm": arm,
            "input_len": 16,
            "client_concurrency": 1,
            "repeat": repeat,
            "graph_enabled": True,
            "http_mean_tbt_ms": 100 if arm != "fused" else 90 - repeat * 10,
            "request_tpot_ms": 125 if arm != "fused" else 115 - repeat * 10,
            "client_ttft_ms": 100,
            "complete_output_tokens_per_s": 8,
            "request_rows_exact_text": ["abc"],
        }
        for repeat in range(3)
        for arm in CLIENT.ARMS
    ]


def test_full_HTTP_tbt_and_completion_tpot_remain_separate(http_fixture):
    raw, sample = http_fixture
    result = CLIENT.raw_metrics(raw, sample, 3)
    assert result["http_mean_tbt_ms"] == 100
    assert result["request_tpot_ms"] == 125
    assert result["http_interval_count"] == 2


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_duration_rejected(http_fixture, duration):
    raw, sample = http_fixture
    raw["duration"] = duration
    with pytest.raises(ValueError):
        CLIENT.raw_metrics(raw, sample, 3)


@pytest.mark.parametrize(
    "change",
    [
        "multitoken",
        "missing_finish",
        "missing_token",
        "bad_order",
        "wrong_length",
        "derived_tbt",
        "zero_tbt",
    ],
)
def test_incomplete_or_unrecoverable_HTTP_timing_rejected(http_fixture, change):
    raw, sample = http_fixture
    if change == "multitoken":
        raw["stream_frames"][0][1].update(new_tokens=2, cumulative_completion_tokens=3)
    elif change == "missing_finish":
        raw["stream_frames"][0][-1]["finish_reason"] = None
    elif change == "missing_token":
        raw["stream_frames"][0].pop()
    elif change == "bad_order":
        raw["stream_frames"][0][1]["received_monotonic_ns"] = 1
    elif change == "wrong_length":
        raw["output_lens"] = [128]
    elif change == "derived_tbt":
        sample["client_timing"]["single_token_arrival_gap_ms"]["mean"] = 101
    else:
        for frame in raw["stream_frames"][0]:
            frame["received_monotonic_ns"] = 1_100_000_000
        sample["client_timing"]["requests"][0]["raw_chunk_gap_ms"]["mean"] = 0
        sample["client_timing"]["single_token_arrival_gap_ms"]["mean"] = 0
    with pytest.raises(ValueError):
        CLIENT.raw_metrics(raw, sample, 3)


def test_three_paired_repetitions_not_token_pseudoreplicates():
    metrics, effects = CLIENT.paired_effects(paired_rows())
    assert len(metrics) == 6 and len(effects) == 16
    effect = next(
        row
        for row in effects
        if row["baseline"] == "baseline"
        and row["changed"] == "fused"
        and row["metric"] == "http_mean_tbt_ms"
    )
    assert effect["paired_repeat_reduction_percent"] == 20
    assert effect["paired_repeat_values_percent"] == [10, 20, 30]
    assert effect["repeats"] == 3 and effect["df"] == 2
    assert effect["confidence_interval_95_percent"][0] < 0


@pytest.mark.parametrize(
    "metric,value",
    [
        ("http_mean_tbt_ms", 0),
        ("http_mean_tbt_ms", float("inf")),
        ("http_mean_tbt_ms", float("nan")),
        ("request_tpot_ms", 0),
        ("request_tpot_ms", True),
    ],
)
def test_paired_percentages_require_finite_positive_metrics(metric, value):
    rows = paired_rows()
    rows[0][metric] = value
    with pytest.raises(ValueError):
        CLIENT.paired_effects(rows)


@pytest.mark.parametrize(
    "change", ["missing_arm", "duplicate", "text", "missing_repeat"]
)
def test_unpaired_or_different_request_outputs_rejected(change):
    rows = paired_rows()
    if change == "missing_arm":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "text":
        rows[0]["request_rows_exact_text"] = ["changed"]
    else:
        rows = [row for row in rows if row["repeat"] != 2]
    with pytest.raises(ValueError):
        CLIENT.paired_effects(rows)
