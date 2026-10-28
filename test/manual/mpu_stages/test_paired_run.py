"""Synthetic process results test scheduling and recovery, never GPU performance."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("paired", "benchmark/mpu_stages/paired_run.py")
FIX = load("stage_fixtures", "test/manual/mpu_stages/test_stage_tools.py")


def fixture(tmp_path):
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(FIX.recipe()), encoding="utf-8")
    stage, args = P.RUN.STAGE.parse_args(
        [
            "--stage",
            str(stage_path),
            "--repeats",
            "3",
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    args.output_dir.mkdir()
    groups = P.RUN.make_plan(stage, args)
    manifest = {
        "status": "planned",
        "groups": groups,
        "analysis_policy": P.RUN.load("capacity_analysis").load_policy(None),
        "paired_execution": {"blocks": P.schedule(groups, 3, 1729)},
    }
    return stage, args, manifest


def fake_executor(stage, args, report, path):
    for group in report["groups"]:
        cap = FIX.capacity(variant=group["variant"], ratio=group["ratio"])
        group.update(
            status="success", decode_cuda_graph_enabled=True, capacity=cap and [cap]
        )
        for case in group["cases"]:
            case.update(
                status="success",
                idle_capacity=[copy.deepcopy(cap)],
                analysis={
                    "decode_only_estimate": {
                        "method": "synthetic_fixture",
                        "decode_throughput_tokens_per_second": 100,
                    },
                    "end_to_end": {
                        "output_throughput_tokens_per_second": 50,
                        "mean_ttft_ms": 10,
                        "mean_tpot_ms": 5,
                    },
                },
            )
            Path(case["result_path"]).write_text(
                '{"synthetic":true}\n', encoding="utf-8"
            )
    return True


def test_counterbalance_seeds_and_concurrency_are_paired(tmp_path):
    stage, args, manifest = fixture(tmp_path)
    blocks = manifest["paired_execution"]["blocks"]
    assert blocks == P.schedule(P.RUN.make_plan(stage, args), 3, 1729)
    for position in range(3):
        assert set(
            Counter(block["group_order"][position] for block in blocks).values()
        ) == {1}
    for block in blocks:
        groups = P.block_plan(
            P.RUN.make_plan(stage, args), block, args.output_dir, args
        )
        for group in groups:
            assert [c["concurrency"] for c in group["cases"]] == block[
                "concurrency_order"
            ]
            assert {
                c["command"][c["command"].index("--seed") + 1] for c in group["cases"]
            } == {str(args.seed + block["repeat"])}


def test_incomplete_pair_retries_all_variants_without_overwriting_evidence(tmp_path):
    stage, args, manifest = fixture(tmp_path)
    calls = []

    def interrupted(stage, args, report, path):
        calls.append(str(path))
        fake_executor(stage, args, report, path)
        return len(calls) != 2

    assert not P.execute(stage, args, manifest, interrupted)
    blocks = manifest["paired_execution"]["blocks"]
    assert [b["status"] for b in blocks] == ["success", "failed", "planned"]
    assert all(
        sum(c["status"] == "success" for c in g["cases"]) == 2
        for g in manifest["groups"]
    )
    failed_file = Path(calls[-1])
    failed_bytes = failed_file.read_bytes()
    resumed = []

    def finish(stage, args, report, path):
        resumed.append(str(path))
        return fake_executor(stage, args, report, path)

    assert P.execute(stage, args, manifest, finish)
    assert len(resumed) == 2
    assert "attempt-0002" in resumed[0]
    assert failed_file.read_bytes() == failed_bytes
    assert all(
        g["status"] == "success" and len(g["cases"]) == 6 for g in manifest["groups"]
    )
    assert P.execute(
        stage, args, manifest, lambda *a: pytest.fail("completed blocks must not rerun")
    )


@pytest.mark.parametrize("tamper", ["changed", "missing", "added"])
def test_modified_completed_artifacts_block_resume_before_launch(tmp_path, tamper):
    stage, args, manifest = fixture(tmp_path)
    assert P.execute(stage, args, manifest, fake_executor)
    artifact = Path(manifest["groups"][0]["cases"][0]["result_path"])
    if tamper == "changed":
        artifact.write_text("modified", encoding="utf-8")
    elif tamper == "missing":
        artifact.unlink()
    else:
        (artifact.parent / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        P.execute(
            stage, args, manifest, lambda *a: pytest.fail("must reject before running")
        )


def test_metadata_drift_across_repeated_blocks_is_rejected(tmp_path):
    stage, args, manifest = fixture(tmp_path)
    calls = []

    def drifting(stage, args, report, path):
        fake_executor(stage, args, report, path)
        calls.append(path)
        if len(calls) == 2:
            report["groups"][0]["capacity"][0]["retained_gpu_pool_bytes"] += 64
        return True

    assert not P.execute(stage, args, manifest, drifting)
    assert manifest["paired_execution"]["blocks"][1]["status"] == "failed"


def test_changed_schedule_and_identity_are_rejected(tmp_path):
    _, _, manifest = fixture(tmp_path)
    expected = copy.deepcopy(manifest["paired_execution"]["blocks"])
    manifest["paired_execution"]["blocks"][0]["group_order"].reverse()
    with pytest.raises(ValueError, match="schedule"):
        P.validate_schedule(manifest, expected)
    for field in ("source", "model_manifest_sha256", "gpu", "topology", "environment"):
        with pytest.raises(ValueError, match=field):
            P.CHECK.same_identity({field: "original"}, {field: "changed"})


def test_lock_prevents_two_writers_and_releases_on_exception(tmp_path):
    with P.CHECK.exclusive(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with P.CHECK.exclusive(tmp_path):
                pass
    with pytest.raises(ValueError):
        with P.CHECK.exclusive(tmp_path):
            raise ValueError("synthetic interruption")
    with P.CHECK.exclusive(tmp_path):
        pass


def test_source_fingerprint_detects_content_and_dependency_changes(tmp_path):
    source = tmp_path / "python"
    source.mkdir()
    (source / "module.py").write_text("x=1\n", encoding="utf-8")
    first = P.CHECK.source_fingerprint(tmp_path)
    (source / "runtime.txt").write_text("dependency==1\n", encoding="utf-8")
    assert P.CHECK.source_fingerprint(tmp_path) != first
