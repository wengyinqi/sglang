"""CPU regressions for Quest configuration and decode bounds bookkeeping.

Load the actual method ASTs to isolate these contracts from SGLang's Linux,
CUDA and FlashInfer import dependencies.  No replacement implementation of
the methods under test is used.  Run with Python's unittest or pytest.
"""

import ast
import signal
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(relative_path, class_name, method_name, parent=None):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    body = method
    if parent is not None:
        # Retain a real __class__ cell for methods using zero-argument super().
        body = ast.ClassDef(
            name="IsolatedBackend",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            body,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "Parent": parent}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return (
        namespace["IsolatedBackend"] if parent is not None else namespace[method_name]
    )


class TestHiSparseCaptureState(unittest.TestCase):
    def test_staging_slots_bound_new_request_admission(self):
        available = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "get_num_allocatable_reqs",
        )
        available.__globals__["get_global_server_args"] = lambda: SimpleNamespace(
            pp_max_micro_batch_size=8
        )
        scheduler = SimpleNamespace(
            pp_size=1,
            enable_hisparse=True,
            req_to_token_pool=SimpleNamespace(available_size=lambda: 0),
        )
        self.assertEqual(available(scheduler, 0), 0)
        scheduler.req_to_token_pool.available_size = lambda: 2
        self.assertEqual(available(scheduler, 1), 2)
        scheduler.enable_hisparse = False
        self.assertEqual(available(scheduler, 1), 7)

    def test_capture_initializes_the_table_consumed_by_resolver(self):
        capture = load_method(
            "python/sglang/srt/layers/attention/flashinfer_hisparse_backend.py",
            "FlashInferHiSparseDecodeBackend",
            "init_forward_metadata_capture_cuda_graph",
        )
        coord = SimpleNamespace(
            req_to_host_pool=torch.full((3, 128), -1, dtype=torch.int64),
            req_to_device_buffer=torch.zeros(3, 65, dtype=torch.int64),
            req_device_buffer_token_locs=torch.full((2, 3, 65), -1, dtype=torch.int32),
            num_real_reqs=torch.zeros(1, dtype=torch.int32),
        )
        backend = SimpleNamespace(
            coord=coord,
            quest=SimpleNamespace(prepare_step=Mock()),
            _get_or_create_wrapper=Mock(),
            _plan_for_step=Mock(),
            _cached_sm_scale=None,
            _cached_logits_soft_cap=None,
            head_dim=128,
        )
        capture(
            backend,
            2,
            2,
            torch.tensor([0, 1]),
            torch.ones(2, dtype=torch.int32),
            None,
            None,
            None,
        )
        self.assertTrue(torch.all(coord.req_device_buffer_token_locs[:, :2] == 0))
        self.assertTrue(torch.all(coord.req_device_buffer_token_locs[:, 2] == -1))
        self.assertEqual(coord.num_real_reqs.item(), 2)


class TestQuestConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(
            load_method(
                "python/sglang/srt/server_args.py",
                "ServerArgs",
                "_check_quest_server_args",
            )
        )

    def config(self, offload=False):
        return SimpleNamespace(
            max_running_requests=4,
            enable_hisparse=offload,
            get_attention_backends=lambda: (
                ("flashinfer", "flashinfer_hisparse")
                if offload
                else ("flashinfer_quest", "flashinfer_quest")
            ),
            speculative_algorithm=None,
            enable_mixed_chunk=False,
            enable_two_batch_overlap=False,
            enable_pdmux=False,
            pp_size=1,
            disable_overlap_schedule=False,
            disaggregation_mode="null",
            enforce_piecewise_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
        )

    def test_supported_backend_pairs(self):
        for offload in (False, True):
            cfg = self.config(offload)
            self.validate(cfg)
            self.assertTrue(cfg.disable_piecewise_cuda_graph)

    def test_request_bounds_allocation_requires_explicit_limit(self):
        for limit in (None, 0, -1):
            cfg = self.config()
            cfg.max_running_requests = limit
            with self.assertRaisesRegex(ValueError, "max-running-requests"):
                self.validate(cfg)

    def test_decode_only_quest_cannot_bypass_prefill_bounds(self):
        cfg = self.config()
        cfg.get_attention_backends = lambda: ("flashinfer", "flashinfer_quest")
        with self.assertRaisesRegex(ValueError, "prefill builds the page bounds"):
            self.validate(cfg)

    def test_offload_cannot_use_full_hbm_backend(self):
        cfg = self.config(True)
        cfg.get_attention_backends = lambda: ("flashinfer_quest", "flashinfer_quest")
        with self.assertRaisesRegex(ValueError, "flashinfer_hisparse"):
            self.validate(cfg)

    def test_unsupported_lifecycle_modes_fail_early(self):
        for attr, value in (
            ("speculative_algorithm", "EAGLE"),
            ("enable_mixed_chunk", True),
            ("disaggregation_mode", "decode"),
            ("enforce_piecewise_cuda_graph", True),
            ("pp_size", 2),
        ):
            cfg = self.config()
            setattr(cfg, attr, value)
            with self.assertRaises(ValueError):
                self.validate(cfg)

    def test_multiple_backend_instances_are_rejected(self):
        for offload in (False, True):
            for attr, flag in (
                ("enable_two_batch_overlap", "enable-two-batch-overlap"),
                ("enable_pdmux", "enable-pdmux"),
            ):
                with self.subTest(offload=offload, flag=flag):
                    cfg = self.config(offload)
                    setattr(cfg, attr, True)
                    with self.assertRaisesRegex(ValueError, flag):
                        self.validate(cfg)

    def test_ordinary_overlap_schedule_setting_is_preserved(self):
        for offload in (False, True):
            for disabled in (False, True):
                with self.subTest(offload=offload, disabled=disabled):
                    cfg = self.config(offload)
                    cfg.disable_overlap_schedule = disabled
                    self.validate(cfg)
                    self.assertEqual(cfg.disable_overlap_schedule, disabled)


class TestQuestDecodePadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update = staticmethod(
            load_method(
                "python/sglang/srt/layers/attention/flashinfer_quest_backend.py",
                "FlashInferQuestDecodeBackend",
                "_do_quest_decode_step_update",
            )
        )

    def backend(self):
        return SimpleNamespace(
            _skip_first_decode_update=torch.tensor([False, True]),
            req_to_token=torch.arange(2 * 256).reshape(2, 256),
            quest=SimpleNamespace(
                update_decode_representations_fused=Mock(),
                maybe_finalize_decode_representations=Mock(),
            ),
            runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(k_data_ptrs=object())
            ),
        )

    def test_padding_neither_updates_bounds_nor_clears_live_skip_flag(self):
        backend = self.backend()
        self.update(backend, torch.tensor([0, 1, 0, 1]), torch.tensor([66, 1, 1, 1]))
        call = backend.quest.update_decode_representations_fused.call_args.kwargs
        torch.testing.assert_close(call["req_indices"], torch.tensor([0]))
        torch.testing.assert_close(call["device_locs"], torch.tensor([64]))
        self.assertTrue(backend._skip_first_decode_update[1])
        torch.testing.assert_close(
            backend.quest.maybe_finalize_decode_representations.call_args.args[0],
            torch.tensor([0]),
        )

    def test_prefill_is_skipped_once_then_previous_decode_is_added(self):
        backend = self.backend()
        self.update(backend, torch.tensor([1]), torch.tensor([66]))
        self.assertEqual(
            backend.quest.update_decode_representations_fused.call_args.kwargs[
                "req_indices"
            ].numel(),
            0,
        )
        self.assertFalse(backend._skip_first_decode_update[1])
        self.update(backend, torch.tensor([1]), torch.tensor([67]))
        call = backend.quest.update_decode_representations_fused.call_args.kwargs
        torch.testing.assert_close(call["req_indices"], torch.tensor([1]))
        torch.testing.assert_close(call["device_locs"], torch.tensor([256 + 65]))


class TestQuestFusedPrefill(unittest.TestCase):
    def test_fused_kv_write_still_builds_bounds(self):
        class DenseBackend:
            def forward_extend(self, *args):
                return "dense-attention-output"

        backend_cls = load_method(
            "python/sglang/srt/layers/attention/flashinfer_quest_backend.py",
            "FlashInferQuestDecodeBackend",
            "forward_extend",
            parent=DenseBackend,
        )
        backend = backend_cls()
        backend.quest = SimpleNamespace(end_layer=2)
        backend._update_quest_for_extend = Mock()
        batch = object()
        for save_kv_cache in (True, False):
            with self.subTest(save_kv_cache=save_kv_cache):
                backend._update_quest_for_extend.reset_mock()
                backend.forward_extend(
                    None,
                    None,
                    None,
                    SimpleNamespace(layer_id=0),
                    batch,
                    save_kv_cache=save_kv_cache,
                )
                backend._update_quest_for_extend.assert_not_called()
                result = backend.forward_extend(
                    None,
                    None,
                    None,
                    SimpleNamespace(layer_id=1),
                    batch,
                    save_kv_cache=save_kv_cache,
                )
                self.assertEqual(result, "dense-attention-output")
                backend._update_quest_for_extend.assert_called_once_with(batch)


class TestQuestAcceptanceWorkerCleanup(unittest.TestCase):
    def test_worker_process_group_cleanup_on_success_and_timeout(self):
        from test.manual.hisparse import test_quest_offload_equivalence as acceptance

        command = ["python", "acceptance-worker"]
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out):
                process = Mock(pid=12345)
                process.wait.side_effect = [
                    subprocess.TimeoutExpired(command, 1) if timed_out else 0,
                    0,
                    0,
                ]
                with patch.object(
                    acceptance.subprocess, "Popen", return_value=process
                ) as popen:
                    # killpg is Linux-specific; mock it without signaling
                    # any process so this contract also runs on Windows.
                    with patch.object(
                        acceptance.os, "killpg", create=True
                    ) as killpg, patch.object(
                        acceptance.signal, "SIGKILL", 9, create=True
                    ):
                        if timed_out:
                            with self.assertRaises(subprocess.TimeoutExpired):
                                acceptance.run_worker_process(command, timeout=1)
                        else:
                            acceptance.run_worker_process(command, timeout=1)
                        popen.assert_called_once()
                        self.assertEqual(popen.call_args.args, (command,))
                        self.assertTrue(popen.call_args.kwargs["start_new_session"])
                        self.assertEqual(popen.call_args.kwargs["cwd"], ROOT)
                        self.assertEqual(
                            popen.call_args.kwargs["env"]["PYTHONPATH"].split(
                                acceptance.os.pathsep
                            )[0],
                            str(ROOT / "python"),
                        )
                        self.assertEqual(
                            killpg.call_args_list,
                            [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)],
                        )


if __name__ == "__main__":
    unittest.main()
