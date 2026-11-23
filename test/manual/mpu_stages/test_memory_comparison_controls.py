"""Matched caller controls cannot override actual snapshot experiment state."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "memory_fixture", ROOT / "test/manual/mpu_stages/test_experiment_memory.py"
)
FIXTURE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FIXTURE)
CONTRACT = FIXTURE.CONTRACT


def test_equal_fake_controls_cannot_bypass_real_actual_batch_difference():
    with pytest.raises(ValueError, match="snapshot context"):
        CONTRACT.compare_to_dense(
            FIXTURE.snapshot(actual_batch=1),
            FIXTURE.snapshot(actual_batch=2),
            controls=FIXTURE.controls(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "model_revision",
        "actual_batch",
        "request_set_fingerprint",
        "output_len",
        "cuda_graph",
    ],
)
def test_missing_real_context_field_fails_closed(field):
    with pytest.raises(ValueError, match="Missing actual snapshot"):
        CONTRACT.compare_to_dense(
            FIXTURE.snapshot(**{field: None}),
            FIXTURE.snapshot(**{field: None}),
            controls=FIXTURE.controls(),
        )


@pytest.mark.parametrize("phase", ["initialization", "graph_warmup"])
def test_pre_request_not_applicable_is_limited_to_both_unknown_request_controls(phase):
    fields = {"actual_batch": None, "request_set_fingerprint": None, "output_len": None}
    controls = FIXTURE.controls(
        **{
            name: {"dense": "not_applicable", "candidate": "not_applicable"}
            for name in fields
        }
    )
    result = CONTRACT.compare_to_dense(
        FIXTURE.snapshot(phase=phase, **fields),
        FIXTURE.snapshot(phase=phase, **fields),
        controls=controls,
    )
    assert set(result["not_applicable_controls"]) == set(fields)
    assert result["controls_bound_to_snapshot_context"] is True


def test_not_applicable_cannot_hide_known_initialization_control():
    controls = FIXTURE.controls(
        actual_batch={"dense": "not_applicable", "candidate": "not_applicable"}
    )
    with pytest.raises(ValueError, match="snapshot context"):
        CONTRACT.compare_to_dense(
            FIXTURE.snapshot(phase="initialization"),
            FIXTURE.snapshot(phase="initialization"),
            controls=controls,
        )


def test_not_applicable_is_rejected_for_unknown_decode_state():
    controls = FIXTURE.controls(
        actual_batch={"dense": "not_applicable", "candidate": "not_applicable"}
    )
    with pytest.raises(ValueError, match="Missing actual snapshot"):
        CONTRACT.compare_to_dense(
            FIXTURE.snapshot(actual_batch=None),
            FIXTURE.snapshot(actual_batch=None),
            controls=controls,
        )


def test_boolean_control_is_not_interchangeable_with_integer_zero():
    controls = FIXTURE.controls(cuda_graph={"dense": 0, "candidate": 0})
    with pytest.raises(ValueError, match="snapshot context"):
        CONTRACT.compare_to_dense(
            FIXTURE.snapshot(), FIXTURE.snapshot(), controls=controls
        )
