"""Qwen/Quest summary-tier correctness, not a model-performance benchmark.

Linux/H800:
    PYTHONPATH=python pytest -q test/manual/xm_sparse/test_quest_bounds_tier.py

Checks mapped-host writers, double-stage lifetime, actual GPU copy counters,
CUDA graph replay, GQA score/TopK parity, request reuse and checkpoint gating.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS = ROOT / "python/sglang/srt/mem_cache/sparsity/algorithms"
spec = importlib.util.spec_from_file_location(
    "quest_xm_configuration_under_test", ALGORITHMS / "quest_algorithm.py"
)
portable_algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable_algorithm)


@pytest.mark.parametrize(
    "flags",
    [
        dict(xm_bounds_offload=True),
        dict(xm_prefetch_mode="lookahead"),
        dict(xm_prefetch_protect=True),
        dict(xm_bounds_offload="true"),
        dict(xm_prefetch_mode="unknown"),
    ],
)
def test_algorithm_rejects_inconsistent_summary_tier_configuration(flags):
    with pytest.raises(ValueError):
        portable_algorithm.QuestAlgorithm(128, 64, torch.device("cpu"), **flags)


GPU = pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available() or torch.version.hip,
    reason="The public CUDA device-graph summary tier requires Linux/NVIDIA",
)


def make_pair(mode="lookahead", protect=False):
    from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm

    reference = QuestAlgorithm(128, 64, torch.device("cuda"))
    tiered = QuestAlgorithm(
        128,
        64,
        torch.device("cuda"),
        xm_bounds_offload=True,
        xm_prefetch_mode=mode,
        xm_prefetch_protect=protect,
    )
    for quest in (reference, tiered):
        quest.init_storage(2, 6, 3, 512, 2, 32)
    return reference, tiered


def assert_state_equal(reference, tiered):
    torch.cuda.synchronize()
    for name in (
        "page_k_bounds",
        "running_k_min",
        "running_k_max",
        "running_token_count",
        "running_page_idx",
    ):
        expected = getattr(reference, name).cpu()
        actual = getattr(tiered, name).cpu()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)


def keys_and_pointers():
    torch.manual_seed(317)
    keys = [
        torch.randn(1024, 2, 32, dtype=torch.bfloat16, device="cuda") for _ in range(8)
    ]
    pointers = torch.tensor(
        [key.data_ptr() for key in keys], dtype=torch.int64, device="cuda"
    )
    return keys, pointers


def prefill_pair(pair, keys, pointers, *, fused=True, lengths=(129, 191, 64)):
    for request, length in enumerate(lengths):
        indices = torch.arange(length, dtype=torch.int64, device="cuda")
        for quest in pair:
            if fused:
                quest.update_prefill_representations_fused(request, pointers, indices)
            else:
                for layer in range(quest.start_layer, quest.end_layer):
                    quest.update_prefill_representations(
                        layer, request, keys[layer], indices
                    )


@GPU
@pytest.mark.parametrize("fused", [True, False])
def test_mapped_bounds_writers_finalize_reuse_and_storage_limit(fused):
    reference, tiered = make_pair()
    try:
        keys, pointers = keys_and_pointers()
        prefill_pair((reference, tiered), keys, pointers, fused=fused)
        assert tiered.page_k_bounds.is_pinned()
        assert (
            tiered.page_k_bounds.data_ptr()
            == tiered.bounds_tier.backing_bounds.data_ptr()
        )
        assert tiered.bounds_tier.stage_bounds.shape[0] == 2
        assert_state_equal(reference, tiered)

        requests = torch.arange(3, dtype=torch.int64, device="cuda")
        locations = torch.tensor([400, 401, 402], dtype=torch.int64, device="cuda")
        # Includes an out-of-storage completion, which advances the scalar
        # counters without touching an unrelated page or resetting its bounds.
        for quest in (reference, tiered):
            quest.running_token_count.copy_(torch.tensor([63, 62, 63], device="cuda"))
            quest.running_page_idx.copy_(torch.tensor([2, 2, 8], device="cuda"))
        for _ in range(3):
            for quest in (reference, tiered):
                quest.update_decode_representations_fused(pointers, requests, locations)
                quest.maybe_finalize_decode_representations(requests)
            assert_state_equal(reference, tiered)

        # A shorter occupant of the same request slot must not inherit pages.
        for quest in (reference, tiered):
            quest.invalidate_request(1)
            for layer in range(quest.start_layer, quest.end_layer):
                quest.update_prefill_representations(
                    layer, 1, keys[layer], torch.arange(65, device="cuda")
                )
        assert_state_equal(reference, tiered)
        assert not tiered.page_valid[:, 1, 1:].any().item()
        # Rebuilding a page-aligned prefix clears the previous running bounds.
        for quest in (reference, tiered):
            for layer in range(quest.start_layer, quest.end_layer):
                quest.update_prefill_representations(
                    layer, 1, keys[layer], torch.arange(128, device="cuda")
                )
        assert_state_equal(reference, tiered)
    finally:
        tiered.close()


@GPU
def test_mapped_finalize_captures_and_replays_mixed_page_completions():
    reference, tiered = make_pair("on_demand")
    try:
        keys, pointers = keys_and_pointers()
        prefill_pair((reference, tiered), keys, pointers)
        requests = torch.arange(3, dtype=torch.int64, device="cuda")
        locations = torch.tensor([400, 401, 402], dtype=torch.int64, device="cuda")

        def update():
            for quest in (reference, tiered):
                quest.update_decode_representations_fused(pointers, requests, locations)
                quest.maybe_finalize_decode_representations(requests)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                update()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update()
        for quest in (reference, tiered):
            quest.running_token_count.copy_(torch.tensor([63, 62, 63], device="cuda"))
            quest.running_page_idx.copy_(torch.tensor([2, 2, 8], device="cuda"))
            quest.running_k_min.fill_(-2)
            quest.running_k_max.fill_(3)
        for _ in range(3):
            graph.replay()
            assert_state_equal(reference, tiered)
    finally:
        tiered.close()


@GPU
@pytest.mark.parametrize("mode", ["on_demand", "lookahead"])
@pytest.mark.parametrize("captured", [False, True])
def test_eager_and_graph_double_stage_scores_topk_and_byte_counters(mode, captured):
    from sglang.srt.mem_cache.sparsity.algorithms.quest_score_kernel import quest_score

    reference, tiered = make_pair(mode)
    try:
        keys, pointers = keys_and_pointers()
        prefill_pair((reference, tiered), keys, pointers, lengths=(0, 129, 321))
        requests = torch.arange(3, dtype=torch.int64, device="cuda")
        lengths = torch.tensor([0, 129, 321], dtype=torch.int32, device="cuda")
        queries = torch.randn(3, 8, 32, device="cuda")
        queries[0].zero_()
        queries[:, 1] = -queries[:, 0]

        def step():
            reference.prepare_step(lengths)
            tiered.prepare_step(lengths)
            outputs = []
            for layer in range(reference.start_layer, reference.end_layer):
                expected = reference.retrieve_topk(queries, layer, requests, lengths)
                actual = tiered.retrieve_topk(queries, layer, requests, lengths)
                slot = tiered.bounds_tier.stage_bounds[
                    (layer - reference.start_layer) % 2 : (
                        layer - reference.start_layer
                    )
                    % 2
                    + 1
                ]
                expected_score = quest_score(
                    queries.float(),
                    reference.page_k_bounds,
                    requests,
                    layer - reference.start_layer,
                )
                actual_score = quest_score(queries.float(), slot, requests, 0)
                # Clone before the parity slot is reused two layers later.
                outputs.append(
                    (expected, actual, expected_score.clone(), actual_score.clone())
                )
            return outputs

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if captured:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = step()
            torch.cuda.synchronize()

        initial = tiered.summary_offload_statistics()["counters"]
        for offset in (0, 1, 2):
            lengths.copy_(
                torch.tensor(
                    [0, 129 + offset, 321 + offset], dtype=torch.int32, device="cuda"
                )
            )
            queries.add_(0.015625)
            if captured:
                graph.replay()
            else:
                outputs = step()
            torch.cuda.synchronize()
            for expected, actual, expected_score, actual_score in outputs:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(actual_score, expected_score, atol=0, rtol=0)
            for row, length in zip(outputs[-1][1][0].tolist(), lengths.tolist()):
                if length:
                    meaningful = row[: min(length, 128)]
                    assert len(set(meaningful)) == len(meaningful)
                    assert 0 <= min(meaningful) <= max(meaningful) < length

        final = tiered.summary_offload_statistics()
        delta_copies = (
            final["counters"]["completed_layer_copies"]
            - initial["completed_layer_copies"]
        )
        assert delta_copies == 3 * reference.num_layers
        delta_bytes = final["counters"]["completed_bytes"] - initial["completed_bytes"]
        assert (
            delta_bytes == delta_copies * final["backing_bytes"] // reference.num_layers
        )
        assert final["counters"]["device_launch_failures"] == 0
        assert final["counters"]["wait_failures"] == 0
        assert final["copies_preallocated_request_and_context_rows"]
    finally:
        tiered.close()


@GPU
def test_checkpoint_gate_waits_and_resumes_on_an_independent_gpu_stream():
    reference, tiered = make_pair("on_demand", protect=True)
    try:
        keys, pointers = keys_and_pointers()
        prefill_pair((reference, tiered), keys, pointers)
        tier = tiered.bounds_tier
        tier.begin_step()
        tier.read_layer(0)
        # Lazy module loading may synchronize the CUDA context on the first
        # call. Load this control kernel before holding a gate that needs a
        # later GPU kernel to release it.
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        initial_wait = tier.statistics()["counters"]["checkpoint_wait_cycles"]
        tier.set_protected(True)
        tier.prefetch_layer(1)
        # The current stream holds the protection flag while the independent
        # prefetch stream reaches a checkpoint. This is a gate test only.
        torch.cuda._sleep(10_000_000)
        tier.set_protected(False)
        second_layer = tier.read_layer(1)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            second_layer[0].cpu(), tier.backing_bounds[1], atol=0, rtol=0
        )
        counters = tier.statistics()["counters"]
        assert counters["checkpoint_wait_cycles"] - initial_wait > 1_000_000
        assert counters["completed_layer_copies"] == 2
        assert counters["wait_failures"] == 0
    finally:
        tiered.close()
