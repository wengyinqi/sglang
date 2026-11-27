"""Portable large CPU-reference guards; these tests make no model/GPU claim."""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "xm_large_batch_test", ROOT / "benchmark/xm_sparse/summary_batch_correctness.py"
)
TOOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOOL)


def settings(tmp_path, *, batch=12, length=32768):
    return SimpleNamespace(
        input_lens=[length],
        output_len=80,
        max_total_tokens=98304,
        capacity_ratios=[16],
        cpu_capacity_ratio=16,
        chunked_prefill_size=4096,
        prefill_max_requests=1,
        top_k=None,
        device_buffer_size=None,
        output_dir=tmp_path,
        logprob_atol=0.02,
        concurrencies=[batch],
    )


def test_cold_full_prefill_hot_and_cpu_context_are_separate_budgets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        TOOL.RUN.STAGE, "server_command", lambda *a, **k: ["portable-plan"]
    )
    stage, args = TOOL.CAMPAIGN.recipe("fused"), settings(tmp_path)
    groups = TOOL.make_groups(stage, args, 12)
    need = groups[0]["large_batch_allocation_requirements"]
    assert need["historical_backing_slots_required"] == 394752
    assert need["historical_backing_slots_required"] > args.max_total_tokens
    assert need["gpu_cold_nopd_admission_slots_required"] == 82816
    assert need["gpu_steady_decode_hot_slots_required"] == 49920
    assert need["gpu_full_prefill_slots_required"] == 32896
    assert need["planned_cpu_backing_allocation_bytes"] == (144 << 30) + 64 * 98304
    assert {group["variant"] for group in groups} == {"cpu"}
    assert {group["eager"] for group in groups} == {True, False}
    assert all(len(round_) == 12 for round_ in groups[0]["rounds"])


@pytest.mark.parametrize("slots", [54080, 65536, 78656, 82815])
def test_chunk_only_or_missing_new_hot_coexistence_budget_is_rejected(
    tmp_path, monkeypatch, slots
):
    monkeypatch.setattr(TOOL.RUN.STAGE, "server_command", lambda *a, **k: [])
    args = settings(tmp_path)
    args.max_total_tokens = slots
    with pytest.raises(ValueError, match="full-prefill plus hot"):
        TOOL.make_groups(TOOL.CAMPAIGN.recipe("fused"), args, 12)


@pytest.mark.parametrize("constraint", ["gpu", "cpu", "prefill"])
def test_requested_budgets_are_separately_guarded(tmp_path, monkeypatch, constraint):
    monkeypatch.setattr(TOOL.RUN.STAGE, "server_command", lambda *a, **k: [])
    args = settings(tmp_path)
    if constraint == "gpu":
        args.max_total_tokens = 53000
    elif constraint == "cpu":
        args.cpu_capacity_ratio = 1
    else:
        args.prefill_max_requests = 2
    with pytest.raises(ValueError):
        TOOL.make_groups(TOOL.CAMPAIGN.recipe("fused"), args, 12)


def test_actual_backing_padding_cannot_count_as_usable_context(tmp_path):
    need = TOOL.allocation_requirements(
        TOOL.CAMPAIGN.recipe("fused"), settings(tmp_path), 12
    )
    capacity = {
        "backing_store": "cpu",
        "storage_device_ids": [],
        "physical_compute_slots": 98304,
        "backing_slots": 1572928,
        "padding_slots": 1572928 - 394752 + 1,
        "backing_kv_bytes": 1572928 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
        "compute_kv_bytes": 98304 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
    }
    with pytest.raises(ValueError, match="usable CPU"):
        TOOL.check_actual_capacity([capacity], need)
    capacity["padding_slots"] = 64
    assert TOOL.check_actual_capacity([capacity], need)["status"] == "passed"
    capacity["physical_compute_slots"] = 53000
    with pytest.raises(ValueError, match="GPU full-prefill"):
        TOOL.check_actual_capacity([capacity], need)


@pytest.mark.parametrize("mutation", ["ratio", "cpu_bytes", "gpu_bytes"])
def test_actual_full_kv_allocation_shape_is_independently_checked(tmp_path, mutation):
    need = TOOL.allocation_requirements(
        TOOL.CAMPAIGN.recipe("fused"), settings(tmp_path), 12
    )
    capacity = {
        "backing_store": "cpu",
        "storage_device_ids": [],
        "physical_compute_slots": 98304,
        "backing_slots": 1572928,
        "padding_slots": 64,
        "backing_kv_bytes": 1572928 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
        "compute_kv_bytes": 98304 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
    }
    if mutation == "ratio":
        capacity["padding_slots"] = 0
    elif mutation == "cpu_bytes":
        capacity["backing_kv_bytes"] -= 1
    else:
        capacity["compute_kv_bytes"] -= 1
    with pytest.raises(ValueError):
        TOOL.check_actual_capacity([capacity], need)


@pytest.fixture
def completed_rounds(tmp_path):
    batch, length = 4, 5000
    args = settings(tmp_path, batch=batch, length=length)
    stage = TOOL.CAMPAIGN.recipe("fused")
    need = TOOL.allocation_requirements(stage, args, batch)
    group = {
        "id": "cpu-eager-ratio16",
        "variant": "cpu",
        "ratio": 16,
        "status": "success",
        "eager": True,
        "rounds": need["rounds"],
        "trace_dir": str(tmp_path / "trace"),
        "capacity": [
            {
                "physical_compute_slots": 98304,
                "compute_kv_bytes": 98304 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
                "backing_slots": 1572928,
                "backing_kv_bytes": 1572928 * TOOL.QWEN_KV_BYTES_PER_TOKEN,
            }
        ],
    }
    rows = [
        {
            "type": "session",
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "disable_overlap_schedule": True,
            "disaggregation_mode": "null",
            "disable_radix_cache": True,
            "speculative_algorithm": None,
            "enable_mixed_chunk": False,
        }
    ]
    sequence = 0
    for index, lengths in enumerate(need["rounds"]):
        ids = [f"r{index}-request{i}" for i in range(batch)]
        responses = [
            {
                "meta_info": {
                    "id": rid,
                    "prompt_tokens": n,
                    "completion_tokens": 80,
                    "output_token_logprobs": [[-0.1, 1, None]] * 80,
                }
            }
            for rid, n in zip(ids, lengths)
        ]
        TOOL.CAMPAIGN.write(
            tmp_path / f"{group['id']}-round{index}.json",
            {"input_ids": [[1] * n for n in lengths], "responses": responses},
        )
        for rid, n in zip(ids, lengths):
            while n:
                chunk = min(n, 4096)
                rows.append(
                    {
                        "type": "forward",
                        "sequence": sequence,
                        "forward_mode": "extend",
                        "request_ids": [rid],
                        "extend_lens": [chunk],
                        "success": True,
                    }
                )
                sequence += 1
                n -= chunk
        for _ in range(79):
            rows.append(
                {
                    "type": "forward",
                    "sequence": sequence,
                    "forward_mode": "decode",
                    "request_ids": ids,
                    "extend_lens": None,
                    "success": True,
                }
            )
            sequence += 1
    rows.append({"type": "idle_flush", "last_sequence": sequence - 1})
    trace = tmp_path / "trace/forward-tp0-pp0-dpNone-pid1.jsonl"
    trace.parent.mkdir()
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    TOOL.CAMPAIGN.write(
        tmp_path / f"{group['id']}-server-info.json",
        {
            "disable_cuda_graph": True,
            "internal_states": [
                {"disable_cuda_graph": True, "cuda_graph_bs": [1, 2, 4]}
            ],
        },
    )
    return group, args, batch, trace, rows


def test_actual_batch_comes_from_complete_round_membership(completed_rounds):
    group, args, batch, _, _ = completed_rounds
    proof, outputs = TOOL.verify_round_trace(group, args, batch)
    assert proof["status"] == "passed"
    assert [row["actual_batch_size"] for row in proof["rounds"]] == [4, 4, 4]
    assert [row["full_batch_decode_forwards"] for row in proof["rounds"]] == [
        79,
        79,
        79,
    ]
    assert len(outputs) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "missing_full_batch",
        "prefill_slots",
        "mode",
        "missing_graph_shape",
    ],
)
def test_failed_raw_coverage_is_never_reported_as_actual_batch(
    completed_rounds, mutation
):
    group, args, batch, trace, rows = completed_rounds
    rows = copy.deepcopy(rows)
    if mutation == "incomplete":
        rows.pop()
    elif mutation == "missing_full_batch":
        for row in rows:
            if row.get("forward_mode") == "decode":
                row["request_ids"] = row["request_ids"][:-1]
    elif mutation == "prefill_slots":
        row = next(row for row in rows if row.get("forward_mode") == "extend")
        row["extend_lens"] = [8192]
    elif mutation == "mode":
        rows[0]["disaggregation_mode"] = "decode"
    else:
        group["eager"] = False
        TOOL.CAMPAIGN.write(
            args.output_dir / f"{group['id']}-server-info.json",
            {
                "disable_cuda_graph": False,
                "internal_states": [
                    {"disable_cuda_graph": False, "cuda_graph_bs": [1, 2]}
                ],
            },
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        TOOL.verify_round_trace(group, args, batch)


def test_missing_or_failed_small_hbm_proof_blocks_execution(tmp_path):
    proof = tmp_path / "failed-small.json"
    proof.write_text(json.dumps({"status": "failed", "phase": "correctness"}))
    workload = tmp_path / "workload.json"
    workload.write_text("{}")
    control = SimpleNamespace(
        correctness_campaign=proof,
        model="/models/qwen",
        input_len=32768,
        workload=workload,
        logprob_atol=0.02,
    )
    with pytest.raises(ValueError, match="did not actually finish"):
        TOOL.small_hbm_gate(control)


def test_complete_responses_at_partial_batches_do_not_prove_requested_batch(
    completed_rounds,
):
    group, args, batch, trace, rows = completed_rounds
    partial = []
    sequence = 0
    for row in rows:
        if row.get("type") == "forward":
            replacements = [copy.deepcopy(row)]
            if row["forward_mode"] == "decode":
                replacements = [
                    dict(row, request_ids=row["request_ids"][:2]),
                    dict(row, request_ids=row["request_ids"][2:]),
                ]
            for replacement in replacements:
                replacement["sequence"] = sequence
                sequence += 1
                partial.append(replacement)
        elif row.get("type") == "idle_flush":
            partial.append(dict(row, last_sequence=sequence - 1))
        else:
            partial.append(row)
    trace.write_text("".join(json.dumps(row) + "\n" for row in partial))
    with pytest.raises(ValueError, match="never observed"):
        TOOL.verify_round_trace(group, args, batch)


@pytest.mark.parametrize("mutation", [None, "inputs", "kv", "tokens"])
def test_cross_arm_comparison_requires_same_inputs_kv_and_tokens(
    completed_rounds, mutation
):
    original, args, batch, _, _ = completed_rounds
    entries = list(args.output_dir.iterdir())
    manifests, arguments = [], []
    for arm in TOOL.ARMS:
        destination = args.output_dir / arm
        destination.mkdir()
        for entry in entries:
            if entry.is_dir():
                shutil.copytree(entry, destination / entry.name)
            else:
                shutil.copy2(entry, destination / entry.name)
        groups = []
        for eager in (True, False):
            group = copy.deepcopy(original)
            group["trace_dir"] = str(destination / "trace")
            if not eager:
                group["id"] = "cpu-graph-ratio16"
                group["eager"] = False
                for index in range(3):
                    shutil.copy2(
                        destination / f"{original['id']}-round{index}.json",
                        destination / f"{group['id']}-round{index}.json",
                    )
                TOOL.CAMPAIGN.write(
                    destination / f"{group['id']}-server-info.json",
                    {
                        "disable_cuda_graph": False,
                        "internal_states": [
                            {"disable_cuda_graph": False, "cuda_graph_bs": [1, 2, 4]}
                        ],
                    },
                )
            groups.append(group)
        manifests.append({"stage": TOOL.CAMPAIGN.recipe(arm), "groups": groups})
        arguments.append(SimpleNamespace(**(vars(args) | {"output_dir": destination})))
    if mutation == "kv":
        manifests[1]["groups"][0]["capacity"][0]["compute_kv_bytes"] += 1
    elif mutation in ("inputs", "tokens"):
        path = arguments[1].output_dir / "cpu-eager-ratio16-round0.json"
        raw = json.loads(path.read_text())
        if mutation == "inputs":
            raw["input_ids"][0][0] = 2
        else:
            raw["responses"][0]["meta_info"]["output_token_logprobs"][0][1] = 2
        TOOL.CAMPAIGN.write(path, raw)
    if mutation:
        with pytest.raises(ValueError):
            TOOL.compare_arms(*manifests, *arguments, batch)
    else:
        comparisons = TOOL.compare_arms(*manifests, *arguments, batch)
        assert len(comparisons) == 2 and all(
            row["exact_greedy_token_match"] for row in comparisons
        )


@pytest.mark.parametrize(
    "arguments", [["--logprob-atol", "0.03"], ["--batch-sizes", "12", "12"]]
)
def test_cli_cannot_loosen_tolerance_or_reuse_evidence(tmp_path, arguments):
    with pytest.raises(SystemExit):
        TOOL.parse(
            [
                "--model",
                "/models/qwen",
                "--workload",
                str(tmp_path / "w"),
                "--correctness-campaign",
                str(tmp_path / "proof"),
                "--output-dir",
                str(tmp_path / "out"),
                *arguments,
            ]
        )


def control_arguments(tmp_path):
    return [
        "--model",
        str(tmp_path / "model"),
        "--workload",
        str(tmp_path / "workload.json"),
        "--correctness-campaign",
        str(tmp_path / "small-proof.json"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_cli_default_retains_complete_cold_prefill_capacity(tmp_path):
    control = TOOL.parse(control_arguments(tmp_path))
    assert control.max_total_tokens == 98304 and control.capacity_ratio == 8
    args = settings(tmp_path)
    args.max_total_tokens = control.max_total_tokens
    args.capacity_ratios = [control.capacity_ratio]
    args.cpu_capacity_ratio = control.capacity_ratio
    args.output_len = 512
    need = TOOL.allocation_requirements(TOOL.CAMPAIGN.recipe("fused"), args, 12)
    assert need["historical_backing_slots_required"] == 399360
    assert need["gpu_cold_nopd_admission_slots_required"] == 83200
    assert args.max_total_tokens >= need["gpu_cold_nopd_admission_slots_required"]
    assert (
        args.max_total_tokens * args.cpu_capacity_ratio
        >= need["historical_backing_slots_required"]
    )
    assert need["planned_cpu_backing_allocation_bytes"] == (72 << 30) + 64 * 98304


@pytest.mark.parametrize("available_delta", [-1, 0, 1])
def test_cpu_ram_preflight_uses_full_allocation_and_host_reserve(
    tmp_path, monkeypatch, available_delta
):
    args = settings(tmp_path)
    args.model = str(tmp_path / "model")
    TOOL.CAMPAIGN.write(
        Path(args.model) / "config.json",
        {"num_hidden_layers": 48, "num_key_value_heads": 4, "head_dim": 128},
    )
    stage = TOOL.CAMPAIGN.recipe("combined-protected")
    need = TOOL.allocation_requirements(stage, args, 12)
    required = need["planned_cpu_backing_allocation_bytes"] + (10 << 30)
    monkeypatch.setattr(
        TOOL,
        "observe_cpu_ram",
        lambda: {"effective_available_bytes": required + available_delta},
    )
    gate = TOOL.cpu_ram_preflight(stage, args, 12)
    assert gate["status"] == ("failed" if available_delta < 0 else "passed")
    assert gate["minimum_available_bytes_required"] == required
    assert gate["model_config"]["bytes"] > 0


def test_ram_preflight_cannot_assume_an_unrelated_model_shape(tmp_path, monkeypatch):
    args = settings(tmp_path)
    args.model = str(tmp_path / "model")
    TOOL.CAMPAIGN.write(
        Path(args.model) / "config.json",
        {"num_hidden_layers": 48, "num_key_value_heads": 8, "head_dim": 128},
    )
    monkeypatch.setattr(
        TOOL, "observe_cpu_ram", lambda: pytest.fail("Do not use assumed KV size")
    )
    with pytest.raises(ValueError, match="actual Qwen"):
        TOOL.cpu_ram_preflight(TOOL.CAMPAIGN.recipe("fused"), args, 12)


@pytest.mark.parametrize("limit", [str(200 << 30), "max", "invalid"])
def test_container_memory_can_reduce_the_host_ram_preflight(monkeypatch, limit):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=256 << 30, available=180 << 30)
        ),
    )
    contents = {
        "/sys/fs/cgroup/memory.max": limit,
        "/sys/fs/cgroup/memory.current": str(100 << 30),
    }

    class FakeFile:
        def __init__(self, name):
            self.name = name

        def is_file(self):
            return self.name in contents

        def read_text(self):
            return contents[self.name]

    monkeypatch.setattr(TOOL, "Path", FakeFile)
    if limit == "invalid":
        with pytest.raises(ValueError, match="cgroup RAM"):
            TOOL.observe_cpu_ram()
    else:
        observed = TOOL.observe_cpu_ram()
        assert observed["effective_available_bytes"] == (
            (180 if limit == "max" else 100) << 30
        )


def test_small_hbm_failure_is_saved_before_any_large_server_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TOOL.CAMPAIGN, "source_fingerprint", lambda: {"sha256": "a" * 64}
    )
    monkeypatch.setattr(
        TOOL, "execute_native", lambda *a: pytest.fail("HBM gate must precede launch")
    )
    code = TOOL.main([*control_arguments(tmp_path), "--execute"])
    assert code == 2
    manifest = json.loads(
        (tmp_path / "output/summary-batch-correctness.json").read_text()
    )
    assert manifest["status"] == "blocked"
    assert manifest["failure_phase"] == "small_hbm_gate"
    assert manifest["small_hbm_gate"]["status"] == "failed"
    assert manifest["cases"] == manifest["comparisons"] == []
    assert "small-proof.json" in manifest["error"]


def test_invalid_cold_budget_plan_saves_failed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TOOL.CAMPAIGN, "source_fingerprint", lambda: {"sha256": "a" * 64}
    )

    def native_settings(control, batch, arm):
        stage, args = TOOL.CAMPAIGN.recipe(arm), settings(tmp_path)
        args.max_total_tokens = control.max_total_tokens
        return stage, args, TOOL.make_groups(stage, args, batch)

    monkeypatch.setattr(TOOL, "native_settings", native_settings)
    code = TOOL.main(
        [
            *control_arguments(tmp_path),
            "--batch-sizes",
            "12",
            "--max-total-tokens",
            "65536",
        ]
    )
    manifest = json.loads(
        (tmp_path / "output/summary-batch-correctness.json").read_text()
    )
    assert code == 2 and manifest["status"] == "failed"
    assert "full-prefill plus hot" in manifest["error"]
    assert manifest["small_hbm_gate"]["status"] == "pending_execution_gate"


def test_prelaunch_ram_failure_saves_native_manifest_and_restores_hooks(
    tmp_path, monkeypatch
):
    args = settings(tmp_path / "native")
    args.model = str(tmp_path / "missing-model")
    stage = TOOL.CAMPAIGN.recipe("fused")
    groups = [{"id": "cpu-eager-ratio16", "status": "planned"}]
    monkeypatch.setattr(
        TOOL.RUN,
        "new_manifest",
        lambda stage, args, groups, kind: {"stage": stage, "groups": groups},
    )

    def original_start(*a, **k):
        pytest.fail("RAM gate must block launch")

    monkeypatch.setattr(TOOL.RUN.COMMON, "start_server", original_start)

    def execute(stage, args, manifest, path):
        TOOL.RUN.COMMON.start_server(args, [], None)
        return True

    monkeypatch.setattr(TOOL.CORRECTNESS, "execute", execute)
    original_wait = TOOL.RUN.wait_capacities
    with pytest.raises(ValueError, match="inspect saved raw evidence"):
        TOOL.execute_native(stage, args, groups, 12)
    native = json.loads((args.output_dir / "manifest.json").read_text())
    assert native["status"] == "failed"
    assert native["cpu_ram_preflight"][0]["status"] == "failed"
    assert "config.json" in native["cpu_ram_preflight"][0]["error"]
    assert TOOL.RUN.wait_capacities is original_wait
    assert TOOL.RUN.COMMON.start_server is original_start


def test_failed_actual_full_prefill_capacity_is_retained_as_failed_evidence(
    tmp_path, monkeypatch
):
    args, stage = settings(tmp_path / "native"), TOOL.CAMPAIGN.recipe("fused")
    groups = [{"id": "cpu-eager-ratio16", "status": "planned"}]
    monkeypatch.setattr(
        TOOL.RUN,
        "new_manifest",
        lambda stage, args, groups, kind: {"stage": stage, "groups": groups},
    )
    insufficient = [
        {
            "backing_store": "cpu",
            "storage_device_ids": [],
            "physical_compute_slots": 65536,
        }
    ]
    monkeypatch.setattr(TOOL.RUN, "wait_capacities", lambda *a, **k: insufficient)

    def execute(stage, args, manifest, path):
        TOOL.RUN.wait_capacities(None)
        return True

    monkeypatch.setattr(TOOL.CORRECTNESS, "execute", execute)
    with pytest.raises(ValueError, match="inspect saved raw evidence"):
        TOOL.execute_native(stage, args, groups, 12)
    native = json.loads((args.output_dir / "manifest.json").read_text())
    assert native["status"] == "failed"
    gate = native["actual_allocation_gates"][0]
    assert gate["status"] == "failed" and gate["capacities"] == insufficient
    assert gate["requirements"]["gpu_cold_nopd_admission_slots_required"] == 82816
