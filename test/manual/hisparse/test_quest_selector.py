"""CPU selector regressions and optional CUDA kernel/capture parity checks.

Run without installing SGLang or Triton:
    python -m pytest test/manual/hisparse/test_quest_selector.py

The CUDA tests additionally need a supported PyTorch/Triton installation.
They validate kernels and selection, not full-model or paper throughput.
"""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS = ROOT / "python/sglang/srt/mem_cache/sparsity/algorithms"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


algorithm = _load("quest_algorithm_under_test", ALGORITHMS / "quest_algorithm.py")
memory = _load("quest_memory_under_test", ALGORITHMS.parent / "quest_memory.py")


def make_quest(top_k=256, page_size=64, context=2048, heads=4, dim=32, device="cpu"):
    quest = algorithm.QuestAlgorithm(top_k, page_size, torch.device(device))
    quest.init_storage(0, 2, 3, context, heads, dim)
    return quest


class TestQuestSelectorCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_invalid_configuration(self):
        for top_k, page_size in ((0, 64), (-64, 64), (65, 64), (64, 0)):
            with self.subTest(top_k=top_k, page_size=page_size):
                with self.assertRaises(ValueError):
                    algorithm.QuestAlgorithm(top_k, page_size, torch.device("cpu"))

    def test_storage_estimate_counts_distinct_tensor_storage(self):
        quest = make_quest(context=513)
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in vars(quest).values()
            if isinstance(tensor, torch.Tensor)
        }
        self.assertEqual(
            sum(storages.values()),
            memory.estimate_quest_storage_bytes(256, 64, 0, 2, 3, 513, 4, 32),
        )

    def test_full_budget_is_unique_across_page_boundaries(self):
        # Qwen's 32 query / 4 KV heads, with the paper's 2048-token budget.
        quest = make_quest(top_k=2048, context=5000, heads=4, dim=128)
        quest.page_k_min.fill_(-1)
        quest.page_k_max.copy_(
            torch.arange(quest.max_pages_per_req).view(1, 1, -1, 1, 1) + 1
        )
        query = torch.ones(1, 32, 128)
        for length in (2047, 2048, 2049, 4095, 4096, 4097, 4159, 4160, 4161):
            with self.subTest(length=length):
                seq_lens = torch.tensor([length], dtype=torch.int32)
                quest.prepare_step(seq_lens)
                positions, actual = quest.retrieve_topk(
                    query, 0, torch.tensor([0]), seq_lens
                )
                selected = positions[0, : actual[0]].tolist()
                self.assertEqual(len(selected), min(length, 2048))
                self.assertEqual(len(set(selected)), len(selected))
                self.assertGreaterEqual(min(selected), 0)
                self.assertLess(max(selected), length)
                self.assertEqual(selected[-64:], list(range(length - 64, length)))

    def test_mixed_short_long_and_zero_length_padding(self):
        quest = make_quest()
        quest.page_k_min.fill_(-1)
        quest.page_k_max.fill_(1)
        seq_lens = torch.tensor([0, 3, 257], dtype=torch.int32)
        quest.prepare_step(seq_lens)
        positions, actual = quest.retrieve_topk(
            torch.zeros(3, 32, 32), 0, torch.arange(3), seq_lens
        )
        self.assertEqual(actual.tolist(), [0, 3, 256])
        self.assertTrue(torch.all(positions[0] == 0))
        self.assertEqual(positions[1, :3].tolist(), [0, 1, 2])
        self.assertEqual(torch.unique(positions[2]).numel(), 256)

    def test_context_smaller_than_selection_budget(self):
        quest = make_quest(context=17)
        seq_lens = torch.tensor([17], dtype=torch.int32)
        quest.prepare_step(seq_lens)
        positions, actual = quest.retrieve_topk(
            torch.ones(1, 4, 32), 0, torch.tensor([0]), seq_lens
        )
        self.assertEqual(actual.tolist(), [17])
        self.assertEqual(positions[0, :17].tolist(), list(range(17)))

    def test_one_page_budget_is_recent_only(self):
        quest = make_quest(top_k=64)
        seq_lens = torch.tensor([193], dtype=torch.int32)
        quest.prepare_step(seq_lens)
        positions, actual = quest.retrieve_topk(
            torch.ones(1, 4, 32), 0, torch.tensor([0]), seq_lens
        )
        self.assertEqual(actual.tolist(), [64])
        self.assertEqual(positions[0].tolist(), list(range(129, 193)))

    def test_opposing_gqa_queries_do_not_cancel(self):
        quest = make_quest(top_k=128, heads=1, dim=1)
        quest.page_k_min.fill_(0)
        quest.page_k_max.fill_(0)
        quest.page_k_min[:, 0, 0] = -10
        quest.page_k_max[:, 0, 0] = 10
        quest.page_k_max[:, 0, 1] = 2
        query = torch.tensor([[[1.0], [-1.0]]])
        scores = algorithm.quest_score_reference(
            query, quest.page_k_bounds, torch.tensor([0]), 0
        )
        self.assertEqual(scores[0, 0].item(), 10)
        self.assertEqual(scores[0, 1].item(), 1)
        seq_lens = torch.tensor([257])
        quest.prepare_step(seq_lens)
        positions, _ = quest.retrieve_topk(query, 0, torch.tensor([0]), seq_lens)
        self.assertEqual(positions[0, :64].tolist(), list(range(64)))

    def test_gqa_score_matches_independent_per_head_upper_bounds(self):
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(2, 32, 8, generator=generator)
        keys = torch.randn(2, 3, 64, 4, 8, generator=generator).bfloat16()
        bounds = torch.stack((keys.amin(2), keys.amax(2)), dim=-1).unsqueeze(0)
        scores = algorithm.quest_score_reference(query, bounds, torch.arange(2), 0)
        expected = torch.zeros_like(scores)
        for request in range(2):
            for page in range(3):
                for head in range(32):
                    kv_head = head // 8
                    for dim in range(8):
                        coefficient = query[request, head, dim]
                        endpoint = int(coefficient >= 0)
                        expected[request, page] += (
                            coefficient
                            * bounds[0, request, page, kv_head, dim, endpoint].float()
                            / 8
                        )
        torch.testing.assert_close(scores, expected, atol=1e-4, rtol=1e-5)

    def test_invalid_bounds_have_negative_infinite_scores_for_zero_queries(self):
        quest = make_quest()
        quest.page_k_min[:, :, 0] = -1
        quest.page_k_max[:, :, 0] = 1
        scores = algorithm.quest_score_reference(
            torch.zeros(1, 32, 32), quest.page_k_bounds, torch.tensor([0]), 0
        )
        self.assertEqual(scores[0, 0].item(), 0)
        self.assertTrue(torch.isneginf(scores[0, 1:]).all())
        self.assertFalse(torch.isnan(scores).any())

    def test_finalize_mixed_page_completions_and_slot_reuse(self):
        quest = make_quest()
        quest.running_token_count[:] = torch.tensor([63, 1, 62])
        quest.running_page_idx[:] = torch.tensor([1, 2, 3])
        quest.running_k_min.fill_(-2)
        quest.running_k_max.fill_(3)
        quest.maybe_finalize_decode_representations(torch.arange(3))
        self.assertEqual(quest.running_token_count.tolist(), [0, 2, 63])
        self.assertEqual(quest.running_page_idx.tolist(), [2, 2, 3])
        self.assertTrue(torch.all(quest.page_k_min[:, 0, 1] == -2))
        self.assertFalse(quest.page_valid[:, 1].any())
        self.assertFalse(quest.page_valid[:, 2].any())
        quest.invalidate_request(0)
        self.assertFalse(quest.page_valid[:, 0].any())
        self.assertEqual(quest.running_token_count[0].item(), 0)
        self.assertEqual(quest.running_page_idx[0].item(), 0)

    def test_chunked_prefill_matches_one_shot_after_partial_page_boundary(self):
        chunked = make_quest()
        one_shot = make_quest()
        keys = torch.arange(400, dtype=torch.bfloat16).view(-1, 1, 1).expand(-1, 4, 32)
        for prefix in (50, 64, 130, 320):
            for layer in range(2):
                chunked.update_prefill_representations(
                    layer, 0, keys, torch.arange(prefix)
                )
            one_shot.invalidate_request(0)
            for layer in range(2):
                one_shot.update_prefill_representations(
                    layer, 0, keys, torch.arange(prefix)
                )
            for name in (
                "page_k_bounds",
                "running_k_min",
                "running_k_max",
                "running_token_count",
                "running_page_idx",
            ):
                with self.subTest(prefix=prefix, tensor=name):
                    torch.testing.assert_close(
                        getattr(chunked, name), getattr(one_shot, name)
                    )
            lengths = torch.tensor([prefix])
            query = torch.ones(1, 32, 32)
            selections = []
            for quest in (chunked, one_shot):
                quest.prepare_step(lengths)
                selections.append(
                    quest.retrieve_topk(query, 0, torch.tensor([0]), lengths)
                )
            torch.testing.assert_close(selections[0], selections[1])
        # A new shorter request must not inherit pages from the old occupant.
        chunked.invalidate_request(0)
        for layer in range(2):
            chunked.update_prefill_representations(layer, 0, keys, torch.arange(70))
        self.assertTrue(chunked.page_valid[:, 0, :1].all())
        self.assertFalse(chunked.page_valid[:, 0, 1:].any())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class TestQuestSelectorCUDA(unittest.TestCase):
    def test_selection_graph_replays_new_lengths_and_padding(self):
        quest = make_quest(device="cuda")
        quest.page_k_min.fill_(-1)
        quest.page_k_max.fill_(1)
        requests = torch.arange(3, device="cuda")
        lengths = torch.tensor([0, 63, 257], dtype=torch.int32, device="cuda")
        query = torch.ones(3, 32, 32, device="cuda")

        def step():
            quest.prepare_step(lengths)
            return quest.retrieve_topk(query, 0, requests, lengths)

        # Score-kernel parity is tested separately; using the reference here
        # also verifies that the debug fallback stays graph-capturable.
        with mock.patch.object(algorithm, "_USE_TRITON_SCORE", False):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                positions, actual = step()
            lengths.copy_(
                torch.tensor([256, 257, 400], dtype=torch.int32, device="cuda")
            )
            graph.replay()
        self.assertEqual(actual.tolist(), [256, 256, 256])
        for row, length in zip(positions.tolist(), (256, 257, 400)):
            self.assertEqual(len(set(row)), 256)
            self.assertLess(max(row), length)

    def test_fused_score_matches_reference_for_qwen_gqa_and_padding(self):
        fused = _load(
            "quest_score_cuda_under_test", ALGORITHMS / "quest_score_kernel.py"
        )
        quest = make_quest(heads=4, dim=128, device="cuda")
        quest.page_k_min[:, :, :4].fill_(-2)
        quest.page_k_max[:, :, :4].fill_(3)
        query = torch.randn(3, 32, 128, device="cuda")
        query[1].zero_()
        requests = torch.arange(3, device="cuda")
        expected = algorithm.quest_score_reference(
            query, quest.page_k_bounds, requests, 0
        )
        actual = fused.quest_score(query, quest.page_k_bounds, requests, 0)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)

    def test_finalize_can_capture_and_replay_mixed_page_boundaries(self):
        quest = make_quest(device="cuda")
        requests = torch.arange(3, device="cuda")
        quest.running_k_min.fill_(-2)
        quest.running_k_max.fill_(3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                quest.maybe_finalize_decode_representations(requests)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quest.maybe_finalize_decode_representations(requests)
        quest.running_token_count.copy_(torch.tensor([63, 1, 62], device="cuda"))
        graph.replay()
        self.assertEqual(quest.running_token_count.tolist(), [0, 2, 63])
        self.assertEqual(quest.running_page_idx.tolist(), [1, 0, 0])

    def test_scatter_skips_padding_and_preserves_scratch(self):
        fused = _load(
            "quest_scatter_cuda_under_test", ALGORITHMS / "quest_scatter_pack_kernel.py"
        )
        page_table = torch.arange(3 * 192, dtype=torch.int32, device="cuda").reshape(
            3, 192
        )
        actual = torch.tensor([0, 3, 192], dtype=torch.int32, device="cuda")
        indptr = torch.tensor([0, 0, 3, 195], dtype=torch.int32, device="cuda")
        result = torch.full((577,), -19, dtype=torch.int32, device="cuda")
        fused.quest_scatter_pack(page_table, actual, indptr, result, 576, 192)
        expected = torch.cat((page_table[1, :3], page_table[2]))
        torch.testing.assert_close(result[:195], expected)
        self.assertTrue(torch.all(result[195:] == -19))


if __name__ == "__main__":
    unittest.main()
