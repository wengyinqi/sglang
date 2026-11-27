"""Portable configuration and experiment-evidence guards; no GPU claims."""

import ast
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
