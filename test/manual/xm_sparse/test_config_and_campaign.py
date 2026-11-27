"""Portable configuration and experiment-evidence guards; no GPU claims."""

import ast
import copy
import importlib.util
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[3]


def definition(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    item = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(
        compile(ast.Module(body=[item], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


SparseConfig = definition(
    "python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py",
    "SparseConfig",
    dict(dataclass=dataclass, field=field, Optional=Optional, math=math),
)
parse_config = definition(
    "python/sglang/srt/mem_cache/sparsity/factory.py",
    "_parse_sparse_config",
    dict(json=json, math=math, SparseConfig=SparseConfig),
)
spec = importlib.util.spec_from_file_location(
    "xm_campaign", ROOT / "benchmark/xm_sparse/campaign.py"
)
CAMPAIGN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CAMPAIGN)
stage_spec = importlib.util.spec_from_file_location(
    "xm_campaign_stage", ROOT / "benchmark/mpu_stages/stage.py"
)
STAGE = importlib.util.module_from_spec(stage_spec)
stage_spec.loader.exec_module(STAGE)


def parse(config, **kwargs):
    return parse_config(
        SimpleNamespace(
            hisparse_config=json.dumps(config), enable_hisparse=True, **kwargs
        )
    )


@pytest.mark.parametrize("arm", list(CAMPAIGN.ARMS))
def test_every_explicit_ablation_has_valid_configuration(arm):
    config = CAMPAIGN.recipe(arm)["variants"]["cpu"]["hisparse_config"]
    actual = parse(config)
    assert (
        actual.xm_fused_retrieval_attention,
        actual.xm_quest_bounds_offload,
        actual.xm_quest_prefetch_mode,
        actual.xm_prefetch_protect,
    ) == CAMPAIGN.ARMS[arm]
    assert not actual.sparse_extra_config


@pytest.mark.parametrize(
    "update",
    [
        {"xm_quest_bounds_offload": True},
        {"xm_quest_prefetch_mode": "lookahead"},
        {"xm_prefetch_protect": True},
        {"xm_fused_retrieval_attention": "true"},
        {"xm_cache_hbm_budget_bytes": -1},
        {"xm_cache_hbm_budget_bytes": True},
    ],
)
def test_invalid_or_silently_ignored_mechanisms_are_rejected(update):
    with pytest.raises(ValueError):
        parse(dict(algorithm="quest", **update))


def test_cpu_budget_can_compare_mechanism_off_baseline():
    result = parse(
        CAMPAIGN.recipe("baseline", budget=8 << 30)["variants"]["cpu"][
            "hisparse_config"
        ]
    )
    assert result.xm_cache_hbm_budget_bytes == 8 << 30
    assert result.xm_quest_prefetch_mode == "none"
    with pytest.raises(ValueError, match="one compute GPU"):
        parse(dict(algorithm="quest", xm_fused_retrieval_attention=True), tp_size=4)


def test_capacity_is_separate_from_latency_and_preserves_allocation_recipe():
    argv = [
        "--model",
        "local-model",
        "--workload",
        "workload.json",
        "--output-dir",
        "results",
    ]
    with pytest.raises(SystemExit):
        CAMPAIGN.parse(argv + ["--phase", "capacity"])
    with pytest.raises(SystemExit):
        CAMPAIGN.parse(argv + ["--cache-budget-bytes", "1024"])
    args = CAMPAIGN.parse(
        argv
        + [
            "--phase",
            "capacity",
            "--cache-budget-bytes",
            "8589934592",
            "--concurrencies",
            "1",
            "4",
            "--arms",
            "baseline",
            "combined",
        ]
    )
    plan = CAMPAIGN.make_plan(args)
    assert len(plan["cases"]) == 12
    assert {c["global_repeat"] for c in plan["cases"]} == {0, 1, 2}
    assert {c["client_concurrency"] for c in plan["cases"]} == {1, 4}
    assert all(c["stage"]["physical_budget_exact"] is False for c in plan["cases"])
    assert all(c["status"] == "planned" for c in plan["cases"])
    assert all("--execute" not in c["command"] for c in plan["cases"])


def test_summary_offload_reserves_two_gpu_layers_and_keeps_running_bounds():
    spec = importlib.util.spec_from_file_location(
        "xm_memory", ROOT / "python/sglang/srt/mem_cache/sparsity/quest_memory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dims = (2048, 64, 0, 48, 9, 32768, 4, 128)
    baseline = module.estimate_quest_storage_bytes(*dims)
    offload = module.estimate_quest_storage_bytes(*dims, xm_bounds_offload=True)
    assert baseline - offload == 4 * (48 - 2) * 9 * 512 * 4 * 128 - 116


@pytest.mark.parametrize(
    "phase,concurrency,enabled",
    [
        ("correctness", 1, True),
        ("performance", 1, False),
        ("performance", 4, True),
        ("capacity", 4, False),
    ],
)
def test_cohort_control_is_explicit_and_capacity_keeps_ordinary_admission(
    tmp_path, phase, concurrency, enabled
):
    argv = [
        "--model",
        "local-model",
        "--workload",
        str(tmp_path / "workload.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--phase",
        phase,
        "--concurrencies",
        str(concurrency),
        "--arms",
        "baseline",
    ]
    if phase == "capacity":
        argv += ["--cache-budget-bytes", str(8 << 30)]
    args = CAMPAIGN.parse(argv)
    plan = CAMPAIGN.make_plan(args)
    assert plan["settings"]["xm_correctness_cohort_barrier"] is enabled
    for case in plan["cases"]:
        stage = case["stage"]
        assert (
            stage.get("server_options", {}).get("xm_correctness_cohort_barrier", False)
            is enabled
        )
        stage_path = tmp_path / "stage.json"
        stage_path.write_text(json.dumps(stage))
        parsed_stage, native = STAGE.parse_args(
            [
                "--stage",
                str(stage_path),
                "--model",
                "local-model",
                "--input-lens",
                "16384",
                "--concurrencies",
                str(concurrency),
                "--num-prompts",
                str(max(concurrency, 2)),
            ]
        )
        command = STAGE.server_command(parsed_stage, native, "cpu", 16384, 8)
        assert ("--xm-correctness-cohort-barrier" in command) is enabled
        if enabled:
            flag = "--xm-correctness-cohort-timeout-seconds"
            assert command[command.index(flag) + 1] == "30.0"
            assert "not ordinary NoPD" in plan["settings"]["ttft_scope"]
        else:
            assert "--xm-correctness-cohort-timeout-seconds" not in command


def test_full_hbm_cohort_budget_has_explicit_padding_without_silent_resize(tmp_path):
    argv = [
        "--model",
        "local-model",
        "--workload",
        str(tmp_path / "workload.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--phase",
        "correctness",
        "--input-lens",
        *(["32768"] * 4),
        "--output-len",
        "128",
        "--concurrencies",
        "4",
        "--arms",
        "baseline",
    ]
    args = CAMPAIGN.parse(argv + ["--max-total-tokens", "131584"])
    with pytest.raises(ValueError, match="explicit admission allowance"):
        CAMPAIGN.make_plan(args)
    assert args.max_total_tokens == 131584
    args = CAMPAIGN.parse(argv + ["--max-total-tokens", "132608"])
    plan = CAMPAIGN.make_plan(args)
    need = plan["settings"]["correctness_allocation_requirements"]
    assert need["full_hbm_paged_slots_required"] == 131584
    assert need["cohort_admission_allowance_slots"] == 1024
    assert need["minimum_requested_physical_slots"] == 132608
    command = plan["cases"][0]["command"]
    assert command[command.index("--max-total-tokens") + 1] == "132608"


@pytest.mark.parametrize("num_prompts", [None, 4, 3, 16])
def test_controlled_performance_submits_exactly_one_cohort(tmp_path, num_prompts):
    argv = [
        "--model",
        "local-model",
        "--workload",
        str(tmp_path / "workload.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--phase",
        "performance",
        "--concurrencies",
        "4",
        "--arms",
        "baseline",
    ]
    if num_prompts is not None:
        argv += ["--num-prompts", str(num_prompts)]
    if num_prompts in (3, 16):
        with pytest.raises(SystemExit):
            CAMPAIGN.parse(argv)
    else:
        args = CAMPAIGN.parse(argv)
        plan = CAMPAIGN.make_plan(args)
        assert args.num_prompts == plan["settings"]["num_prompts"] == 4
        assert all(
            case["command"][case["command"].index("--num-prompts") + 1] == "4"
            for case in plan["cases"]
        )


def test_controlled_performance_cannot_share_one_count_between_multiple_cohorts(
    tmp_path,
):
    with pytest.raises(SystemExit):
        CAMPAIGN.parse(
            [
                "--model",
                "local-model",
                "--workload",
                str(tmp_path / "workload.json"),
                "--output-dir",
                str(tmp_path / "out"),
                "--phase",
                "performance",
                "--concurrencies",
                "1",
                "4",
            ]
        )


@pytest.mark.parametrize(
    "options",
    [
        {"xm_correctness_cohort_barrier": "true"},
        {
            "xm_correctness_cohort_barrier": True,
            "xm_correctness_cohort_timeout_seconds": 0,
        },
        {
            "xm_correctness_cohort_barrier": True,
            "xm_correctness_cohort_timeout_seconds": float("inf"),
        },
        {
            "xm_correctness_cohort_barrier": False,
            "xm_correctness_cohort_timeout_seconds": 30,
        },
        {"unknown_ignored_option": True},
    ],
)
def test_managed_stage_options_cannot_be_malformed_or_silently_ignored(options):
    stage = copy.deepcopy(CAMPAIGN.recipe("baseline"))
    stage["server_options"] = options
    with pytest.raises(ValueError):
        STAGE.server_options(stage)


@pytest.mark.parametrize("mutation", [None, "recipe", "argv", "timeout", "observed"])
def test_execution_gate_requires_bound_cohort_recipe_command_and_settings(
    tmp_path, mutation
):
    """Synthetic proof guards only; no model acceptance is claimed by this test."""
    stage = CAMPAIGN.recipe("baseline", cohort_barrier=True)
    root = tmp_path / "baseline"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    group = {
        "id": "hbm-eager",
        "variant": "hbm",
        "status": "success",
        "server_command": [
            "--xm-correctness-cohort-barrier",
            "--xm-correctness-cohort-timeout-seconds",
            "30.0",
        ],
    }
    info = {
        "xm_correctness_cohort_barrier": True,
        "xm_correctness_cohort_timeout_seconds": 30.0,
    }
    if mutation == "argv":
        group["server_command"].remove("--xm-correctness-cohort-barrier")
    elif mutation == "timeout":
        group["server_command"][-1] = "60.0"
    elif mutation == "observed":
        info["xm_correctness_cohort_barrier"] = False
    if mutation == "recipe":
        stage = CAMPAIGN.recipe("baseline")
    CAMPAIGN.write(
        evidence / "manifest.json",
        {
            "status": "success",
            "stage": stage,
            "groups": [group],
        },
    )
    CAMPAIGN.write(evidence / "hbm-eager-server-info.json", info)
    proof_path = tmp_path / "correctness.json"
    proof = {
        "status": "success",
        "phase": "correctness",
        "source_fingerprint": {"sha256": "a" * 64},
        "workload_identity": {"sha256": "b" * 64},
        "settings": {
            "model": "local-model",
            "input_lens": [32768],
            "disable_cuda_graph": False,
            "xm_correctness_cohort_barrier": True,
        },
        "cases": [
            {
                "arm": "baseline",
                "stage": stage,
                "status": "success",
                "stage_path": str(root / "stage.json"),
            }
        ],
    }
    CAMPAIGN.write(proof_path, proof)
    args = SimpleNamespace(
        correctness_campaign=proof_path,
        model="local-model",
        input_lens=[32768],
        disable_cuda_graph=False,
        arms=["baseline"],
    )
    manifest = {
        "source_fingerprint": proof["source_fingerprint"],
        "workload_identity": proof["workload_identity"],
    }
    if mutation:
        with pytest.raises(ValueError, match="cohort"):
            CAMPAIGN.correctness_gate(args, manifest)
    else:
        assert (
            CAMPAIGN.correctness_gate(args, manifest)["sha256"]
            == CAMPAIGN.identity(proof_path)["sha256"]
        )
