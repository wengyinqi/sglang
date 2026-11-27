"""CPU control-flow checks for cold-shape warmup and prefetch protection."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


class TestPrefetchProtectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/attention/flashinfer_hisparse_backend.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashInferHiSparseDecodeBackend")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "forward_decode")
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        cls.capturing = Mock(return_value=False)
        namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=cls.capturing))}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        cls.forward = staticmethod(namespace["forward_decode"])

    def setUp(self):
        self.capturing.return_value = False
        self.protect = Mock()
        self.backend = SimpleNamespace(
            _current_bs=2,
            _prefetch_protection_ready=set(),
            _cached_sm_scale=1.0,
            _cached_logits_soft_cap=0.0,
            xm_fused_retrieval_attention=True,
            quest=SimpleNamespace(
                xm_prefetch_protect=True,
                end_layer=2,
                retrieve_topk=Mock(return_value=([0], None)),
                set_prefetch_protected=self.protect,
            ),
            _run_fused_attention=Mock(return_value="output"),
            _run_separate_attention=Mock(return_value="output"),
        )
        self.batch = SimpleNamespace(req_pool_indices=[0, 1], seq_lens=[32, 32])

    def run_layer(self, layer_id):
        return self.forward(self.backend, None, None, None, SimpleNamespace(layer_id=layer_id), self.batch, save_kv_cache=False)

    def test_cold_shape_keeps_gate_open_until_all_layers_finish(self):
        self.run_layer(0)
        self.assertEqual(self.backend._prefetch_protection_ready, set())
        self.protect.assert_not_called()
        self.run_layer(1)
        self.assertEqual(self.backend._prefetch_protection_ready, {2})
        self.protect.assert_not_called()
        self.run_layer(0)
        self.assertEqual(self.protect.call_args_list, [call(True), call(False)])

    def test_new_shape_cannot_capture_without_complete_eager_warmup(self):
        self.backend._prefetch_protection_ready.add(1)
        self.capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "complete eager.*batch shape 2"):
            self.run_layer(0)
        self.backend._run_fused_attention.assert_not_called()
        self.protect.assert_not_called()

    def test_runtime_failure_reopens_an_armed_gate(self):
        self.backend._prefetch_protection_ready.add(2)
        self.backend._run_fused_attention.side_effect = RuntimeError("attention failed")
        with self.assertRaisesRegex(RuntimeError, "attention failed"):
            self.run_layer(0)
        self.assertEqual(self.protect.call_args_list, [call(True), call(False)])

    def test_failed_cold_last_layer_does_not_arm_shape(self):
        self.backend._run_fused_attention.side_effect = RuntimeError("attention failed")
        with self.assertRaisesRegex(RuntimeError, "attention failed"):
            self.run_layer(1)
        self.assertEqual(self.backend._prefetch_protection_ready, set())
        self.protect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
