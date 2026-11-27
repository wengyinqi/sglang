"""Actual HiSparse backend metadata-once / multiple-forward epoch contract.

The actual backend methods are extracted without importing its large dependency
tree. The CUDA cases use the real QuestAlgorithm, mapped-host writers, two-stage
device-copy graphs and score kernels; only the attention result is replaced by
selection/score evidence. This is not a full-model attention correctness test.
"""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
BACKEND_SOURCE = (
    ROOT / "python/sglang/srt/layers/attention/flashinfer_hisparse_backend.py"
)


def actual_backend_methods():
    tree = ast.parse(BACKEND_SOURCE.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FlashInferHiSparseDecodeBackend"
    )
    names = {
        "forward_decode",
        "init_forward_metadata",
        "init_forward_metadata_capture_cuda_graph",
    }
    methods = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(BACKEND_SOURCE), "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


def backend_fixture(quest, coord, attention):
    return SimpleNamespace(
        quest=quest,
        coord=coord,
        _current_bs=None,
        _wrapper=None,
        _get_or_create_wrapper=Mock(return_value=object()),
        _plan_for_step=Mock(),
        _cached_sm_scale=1.0,
        _cached_logits_soft_cap=0.0,
        _prefetch_protection_ready=set(),
        xm_fused_retrieval_attention=True,
        head_dim=32,
        _run_fused_attention=attention,
        _run_separate_attention=Mock(
            side_effect=AssertionError("unexpected separate path")
        ),
    )


def test_real_backend_metadata_once_starts_each_complete_forward_at_first_layer():
    methods = actual_backend_methods()
    quest = SimpleNamespace(
        start_layer=2,
        end_layer=6,
        bounds_tier=object(),
        xm_prefetch_protect=False,
        prepare_step=Mock(),
        begin_summary_step=Mock(),
        retrieve_topk=Mock(return_value=([0], None)),
    )
    coord = SimpleNamespace(wait_for_pending_bounds=Mock())
    backend = backend_fixture(quest, coord, Mock(return_value="attention"))
    batch = SimpleNamespace(
        batch_size=3, req_pool_indices=[0, 1, 2], seq_lens=[0, 193, 321]
    )
    methods["init_forward_metadata"](backend, batch)
    quest.prepare_step.assert_called_once_with(batch.seq_lens, begin_summary_step=False)
    quest.begin_summary_step.assert_not_called()
    for forward in range(2):
        for layer in range(2, 6):
            methods["forward_decode"](
                backend,
                None,
                None,
                None,
                SimpleNamespace(layer_id=layer),
                batch,
                save_kv_cache=False,
            )
            assert quest.begin_summary_step.call_count == forward + 1
    assert quest.prepare_step.call_count == 1
    assert quest.retrieve_topk.call_count == 8


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available() or torch.version.hip,
    reason="Real epoch/Graph contract needs Linux NVIDIA CUDA",
)
@pytest.mark.parametrize("mode", ["on_demand", "lookahead"])
@pytest.mark.parametrize("protect", [False, True])
def test_metadata_once_warmup_twice_capture_and_replay_forward_refreshes_summary_epoch(
    mode, protect
):
    from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
    from sglang.srt.mem_cache.sparsity.algorithms.quest_score_kernel import quest_score

    methods = actual_backend_methods()
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
    try:
        torch.manual_seed(843)
        keys = [
            torch.randn(1024, 2, 32, dtype=torch.bfloat16, device="cuda")
            for _ in range(8)
        ]
        pointers = torch.tensor(
            [key.data_ptr() for key in keys], dtype=torch.int64, device="cuda"
        )
        lengths = torch.tensor([0, 193, 321], dtype=torch.int32, device="cuda")
        requests = torch.arange(3, dtype=torch.int64, device="cuda")
        prefill_indices = [
            torch.arange(length, dtype=torch.int64, device="cuda")
            for length in (0, 193, 321)
        ]

        def publish_bounds():
            for request, indices in enumerate(prefill_indices):
                for quest in (reference, tiered):
                    quest.update_prefill_representations_fused(
                        request, pointers, indices
                    )

        publish_bounds()
        reference.prepare_step(lengths, begin_summary_step=False)
        query = torch.ones(3, 8, 32, device="cuda")
        query[0].zero_()
        query[:, 1].neg_()
        batch = SimpleNamespace(
            batch_size=3, req_pool_indices=requests, seq_lens=lengths
        )
        coord = SimpleNamespace(
            wait_for_pending_bounds=Mock(),
            req_to_host_pool=torch.zeros(3, 512, dtype=torch.int64, device="cuda"),
            req_to_device_buffer=torch.zeros(3, 64, dtype=torch.int64, device="cuda"),
            req_device_buffer_token_locs=torch.zeros(
                4, 3, 64, dtype=torch.int64, device="cuda"
            ),
            num_real_reqs=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )

        def score_evidence(q, layer, forward_batch, positions):
            offset = layer.layer_id - reference.start_layer
            stage = tiered.bounds_tier.stage_bounds[offset % 2 : offset % 2 + 1]
            expected_positions, _ = reference.retrieve_topk(
                q, layer.layer_id, requests, lengths
            )
            expected_score = quest_score(
                q.float(), reference.page_k_bounds, requests, offset
            )
            actual_score = quest_score(q.float(), stage, requests, 0)
            return (
                positions.clone(),
                expected_positions.clone(),
                actual_score.clone(),
                expected_score.clone(),
            )

        backend = backend_fixture(tiered, coord, score_evidence)
        # Reproduce cuda_graph_runner: metadata is initialized only once,
        # then all layers are run repeatedly for warmup and capture.
        methods["init_forward_metadata_capture_cuda_graph"](
            backend,
            3,
            3,
            requests,
            lengths,
            None,
            None,
            None,
        )
        assert tiered.bounds_tier.step_epoch.item() == 0
        assert backend._plan_for_step.call_count == 1

        def forward():
            return [
                methods["forward_decode"](
                    backend,
                    query,
                    None,
                    None,
                    SimpleNamespace(layer_id=layer),
                    batch,
                    save_kv_cache=False,
                )
                for layer in range(reference.start_layer, reference.end_layer)
            ]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert tiered.bounds_tier.step_epoch.item() == 2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward()
        torch.cuda.synchronize()
        assert tiered.bounds_tier.step_epoch.item() == 2
        initial = tiered.summary_offload_statistics()["counters"]

        previous_scores = None
        for replay in range(3):
            # Actual GPU producers publish fresh mapped-host bounds before
            # replay; an old epoch would skip their copies and expose stale data.
            for layer in range(reference.start_layer, reference.end_layer):
                keys[layer][:64].add_(1.0 + replay)
            publish_bounds()
            graph.replay()
            torch.cuda.synchronize()
            assert tiered.bounds_tier.step_epoch.item() == 3 + replay
            for positions, expected_positions, scores, expected_scores in outputs:
                torch.testing.assert_close(
                    positions, expected_positions, atol=0, rtol=0
                )
                torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
            current_scores = outputs[-1][2].clone()
            if previous_scores is not None:
                assert not torch.equal(current_scores, previous_scores)
            previous_scores = current_scores
        final = tiered.summary_offload_statistics()["counters"]
        assert final["completed_layer_copies"] - initial["completed_layer_copies"] == 12
        assert final["completed_bytes"] - initial["completed_bytes"] == 73728
        assert final["host_parent_launches"] == initial["host_parent_launches"]
        assert final["device_launch_failures"] == 0
        assert final["wait_failures"] == 0
        assert backend._plan_for_step.call_count == 1
    finally:
        tiered.close()
