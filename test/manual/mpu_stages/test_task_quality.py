"""Complete task-input, normal-EOS scoring and paired-evidence contracts."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "task_quality_cpu", ROOT / "benchmark/mpu_stages/task_quality.py"
)
QUALITY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QUALITY)


class Tokenizer:
    def __init__(self):
        self.messages, self.encode_arguments = [], []

    def __len__(self):
        return 65536

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return "<user>" + messages[0]["content"] + "</user><assistant>\n<think>\n"

    def encode(self, text, **kwargs):
        self.encode_arguments.append(kwargs)
        return list(map(ord, text))


def records():
    return [
        dict(
            _id=f"d{domain}-{i}",
            domain=f"domain-{domain}",
            difficulty="hard",
            length="short",
            context=f"UNMODIFIED DOCUMENT {domain}/{i}",
            question="Which answer?",
            answer="A",
            choice_A="correct",
            choice_B="second",
            choice_C="third",
            choice_D="fourth",
        )
        for domain in range(6)
        for i in range(4)
    ]


def selection(root, count=3):
    selected, candidates = QUALITY.select(
        records(),
        Tokenizer(),
        count=count,
        seed=1729,
        max_input_tokens=32768,
        max_new_tokens=8192,
        max_total_tokens=65536,
    )
    root.mkdir()
    model = {
        "model_id": QUALITY.MODEL_ID,
        "revision": QUALITY.MODEL_REVISION,
        "full_checksum": True,
        "upstream_identity_verified": True,
        "content_fingerprint_sha256": "1" * 64,
        "model_path": "/models/qwen",
    }
    QUALITY.write(root / "model-manifest.json", model)
    QUALITY.write(
        root / "inputs.json", {"kind": "mpu_complete_task_inputs", "rows": selected}
    )
    manifest = {
        "schema_version": 1,
        "kind": "mpu_task_quality_subset",
        "quality_evaluation": True,
        "model_id": QUALITY.MODEL_ID,
        "model_revision": QUALITY.MODEL_REVISION,
        "source": {"revision": QUALITY.DATASET_REVISION},
        "vocabulary_size": 65536,
        "inputs_file": "inputs.json",
        "inputs_sha256": QUALITY.file_sha(root / "inputs.json"),
        "num_samples": count,
        "max_input_tokens": 32768,
        "max_total_tokens": 65536,
        "model_manifest_sha256": QUALITY.file_sha(root / "model-manifest.json"),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 8192,
            "ignore_eos": False,
        },
        "candidates": candidates,
        "scope": "CPU fixture only; not a task accuracy result",
    }
    QUALITY.write(root / "selection.json", manifest)
    QUALITY.write(root / "source-identity.json", {"source_commit": "2" * 40})
    return root / "selection.json", selected


def response(row, text="reasoning about Answer: D</think>\nAnswer: A", finish="stop"):
    return {
        "text": text,
        "meta_info": {
            "prompt_tokens": row["complete_input_tokens"],
            "completion_tokens": 4,
            "finish_reason": {"type": finish, "matched": 151645},
            "output_token_logprobs": [[-0.1, t] for t in (1, 2, 3, 4)],
        },
    }


def server(variant="cpu"):
    info = {
        **QUALITY.COMMON_CONTROLS,
        "model_path": "/models/qwen",
        "tp_size": 1,
        "dp_size": 1,
        "disable_radix_cache": True,
        "enable_hierarchical_cache": False,
        "allow_auto_truncate": False,
        "max_total_tokens": 65536,
        "enable_hisparse": variant == "cpu",
        "attention_backend": "flashinfer",
        "decode_attention_backend": "flashinfer_hisparse" if variant == "cpu" else None,
        "hisparse_config": None,
    }
    if variant == "hbm":
        info["attention_backend"] = "flashinfer_quest"
    if variant != "dense":
        info["hisparse_config"] = json.dumps(
            {
                "algorithm": "quest",
                "backing_store": "cpu",
                "host_to_device_ratio": 4,
                "top_k": 2048,
                "quest_page_size": 64,
                "device_buffer_size": 4096,
            }
        )
    return info


class TestTaskQuality(unittest.TestCase):
    def test_selection_is_complete_deterministic_and_domain_balanced(self):
        tok = Tokenizer()
        rows, candidates = QUALITY.select(
            records(),
            tok,
            count=12,
            seed=1729,
            max_input_tokens=32768,
            max_new_tokens=8192,
            max_total_tokens=65536,
        )
        again, _ = QUALITY.select(
            records(),
            Tokenizer(),
            count=12,
            seed=1729,
            max_input_tokens=32768,
            max_new_tokens=8192,
            max_total_tokens=65536,
        )
        self.assertEqual(rows, again)
        self.assertEqual(
            [r["domain"] for r in rows], [f"domain-{i}" for i in range(6)] * 2
        )
        self.assertEqual(
            sum(c["selection_status"] == "selected" for c in candidates), 12
        )
        for record, message, arguments in zip(
            records(), tok.messages, tok.encode_arguments
        ):
            self.assertIn(record["context"], message[0]["content"])
            self.assertNotIn(
                "Answer: " + record["answer"] + "\n", message[0]["content"]
            )
            self.assertEqual(
                arguments, {"add_special_tokens": False, "truncation": False}
            )

    def test_overlong_source_is_excluded_whole_not_cropped(self):
        source = records()
        source[0]["context"] = "Z" * 40000
        rows, candidates = QUALITY.select(
            source,
            Tokenizer(),
            count=12,
            seed=1729,
            max_input_tokens=32768,
            max_new_tokens=8192,
            max_total_tokens=65536,
        )
        self.assertEqual(
            candidates[0]["selection_status"], "complete_input_exceeds_limit"
        )
        self.assertNotIn(source[0]["_id"], [r["source_id"] for r in rows])

    def test_physical_budget_and_duplicate_record_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            QUALITY.select(
                records(),
                Tokenizer(),
                count=24,
                seed=1729,
                max_input_tokens=32768,
                max_new_tokens=8192,
                max_total_tokens=8300,
            )
        source = records()
        source[1]["_id"] = source[0]["_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            QUALITY.select(
                source,
                Tokenizer(),
                count=12,
                seed=1729,
                max_input_tokens=32768,
                max_new_tokens=8192,
                max_total_tokens=65536,
            )

    def test_cropped_performance_manifest_cannot_enter_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "workload.json"
            QUALITY.write(
                p, {"kind": "mpu_real_document_workload", "quality_evaluation": False}
            )
            with self.assertRaisesRegex(ValueError, "cropped"):
                QUALITY.load_selection(p)

    def test_input_token_or_stop_policy_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, rows = selection(Path(temporary) / "selection")
            manifest = QUALITY.read(p)
            rows[0]["input_ids"][0] += 1
            QUALITY.write(
                p.parent / "inputs.json",
                {"kind": "mpu_complete_task_inputs", "rows": rows},
            )
            manifest["inputs_sha256"] = QUALITY.file_sha(p.parent / "inputs.json")
            QUALITY.write(p, manifest)
            with self.assertRaisesRegex(ValueError, "sequence changed"):
                QUALITY.load_selection(p)
        with tempfile.TemporaryDirectory() as temporary:
            p, _ = selection(Path(temporary) / "selection")
            manifest = QUALITY.read(p)
            manifest["sampling_params"]["ignore_eos"] = True
            QUALITY.write(p, manifest)
            with self.assertRaisesRegex(ValueError, "normal EOS"):
                QUALITY.load_selection(p)

    def test_closed_thinking_is_not_confused_with_reasoning_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, rows = selection(Path(temporary) / "selection")
            result = QUALITY.normalize_response(response(rows[0]), rows[0], 8192)
            self.assertTrue(result["correct"])
            self.assertEqual(result["predicted_answer"], "A")

    def test_truncated_output_never_counts_as_correct_even_with_letter(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, rows = selection(Path(temporary) / "selection")
            result = QUALITY.normalize_response(
                response(rows[0], finish="length"), rows[0], 8192
            )
            self.assertFalse(result["correct"])
            self.assertFalse(result["scorable"])
            self.assertEqual(result["answer_status"], "length_truncated")

    def test_incomplete_thinking_or_ambiguous_final_answer_is_unscorable(self):
        self.assertEqual(
            QUALITY.final_answer("Answer: A", opens_thinking=True),
            (None, "incomplete_thinking"),
        )
        self.assertEqual(
            QUALITY.final_answer("</think>Answer: A\nAnswer: B", opens_thinking=True),
            (None, "ambiguous_final_answer"),
        )
        self.assertEqual(
            QUALITY.final_answer("</think>\\boxed{C}", opens_thinking=True),
            ("C", "parsed_final_answer"),
        )
        self.assertEqual(
            QUALITY.final_answer(
                "</think>The document uses A and B.", opens_thinking=True
            ),
            (None, "missing_final_answer"),
        )

    def test_server_prompt_truncation_and_missing_output_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, rows = selection(Path(temporary) / "selection")
            altered = response(rows[0])
            altered["meta_info"]["prompt_tokens"] -= 1
            with self.assertRaisesRegex(ValueError, "cropped"):
                QUALITY.normalize_response(altered, rows[0], 8192)
            altered = response(rows[0])
            altered["meta_info"]["output_token_logprobs"].pop()
            with self.assertRaisesRegex(ValueError, "Missing native"):
                QUALITY.normalize_response(altered, rows[0], 8192)

    def test_actual_backend_config_is_checked_in_flat_and_nested_info(self):
        for variant in QUALITY.VARIANTS:
            QUALITY.server_policy(
                server(variant),
                variant,
                {"max_total_tokens": 65536},
                {"model_path": "/models/qwen"},
                capacity_ratio=4,
                top_k=2048,
            )
            QUALITY.server_policy(
                {"server_args": server(variant)},
                variant,
                {"max_total_tokens": 65536},
                {"model_path": "/models/qwen"},
                capacity_ratio=4,
                top_k=2048,
            )
        for key, value in (
            ("enable_hierarchical_cache", True),
            ("allow_auto_truncate", True),
            ("tp_size", 4),
        ):
            info = server()
            info[key] = value
            with self.assertRaises(ValueError):
                QUALITY.server_policy(
                    info,
                    "cpu",
                    {"max_total_tokens": 65536},
                    {"model_path": "/models/qwen"},
                    capacity_ratio=4,
                    top_k=2048,
                )
        with self.assertRaisesRegex(ValueError, "preserve Quest"):
            QUALITY.server_policy(
                server("dense"),
                "hbm",
                {"max_total_tokens": 65536},
                {"model_path": "/models/qwen"},
                capacity_ratio=4,
                top_k=2048,
            )

    def make_runs(self, root, *, fail_variant=None, cleanup_failure=False):
        p, rows = selection(root / "selection")
        paths = []
        self.last_routes = {}
        for variant in QUALITY.VARIANTS:
            args = SimpleNamespace(
                selection=p,
                model_manifest=p.parent / "model-manifest.json",
                source_identity=p.parent / "source-identity.json",
                variant=variant,
                base_url="http://fixture.invalid",
                capacity_ratio=4,
                top_k=2048,
                request_timeout=10,
                output_dir=root / variant,
            )
            cases = iter(
                [
                    (
                        TimeoutError("synthetic request timeout")
                        if variant == fail_variant
                        else response(rows[0])
                    ),
                    response(rows[1], finish="length"),
                    response(rows[2]),
                ]
            )

            routes = []

            def fake_request(base, route, data=None, timeout=30, *, json_response=True):
                routes.append(route)
                if route == "get_server_info":
                    return server(variant)
                if route == "abort_request":
                    self.assertFalse(json_response)
                    return {"http_status": 200, "body": ""}
                if route.startswith("flush_cache?timeout="):
                    self.assertFalse(json_response)
                    if cleanup_failure:
                        raise TimeoutError("synthetic scheduler drain timeout")
                    return {"http_status": 200, "body": "Cache flushed.\n"}
                self.assertEqual(data["sampling_params"]["ignore_eos"], False)
                item = next(cases)
                if isinstance(item, Exception):
                    raise item
                return item

            with patch.object(
                QUALITY, "request", fake_request
            ), contextlib.redirect_stdout(io.StringIO()):
                QUALITY.evaluate(args)
            self.last_routes[variant] = routes
            paths.append(args.output_dir)
        return p, paths

    def test_timeout_cleanup_is_explicit_and_idle_before_next_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, paths = self.make_runs(root, fail_variant="cpu")
            routes = self.last_routes["cpu"]
            self.assertEqual(
                routes[:4],
                [
                    "get_server_info",
                    "generate",
                    "abort_request",
                    "flush_cache?timeout=30",
                ],
            )
            self.assertEqual(routes.count("generate"), 3)
            run = QUALITY.read(paths[2] / "quality-run.json")
            self.assertEqual(run["status"], "completed_with_unscorable")
            rows = QUALITY.read(paths[2] / "results.json")["rows"]
            self.assertEqual(rows[0]["cleanup"]["status"], "passed")
            with contextlib.redirect_stdout(io.StringIO()):
                QUALITY.compare(paths, p, root / "comparison")
            report = QUALITY.read(root / "comparison/quality-comparison.json")
            self.assertEqual(report["variant_metrics"]["cpu"]["samples"], 3)
            self.assertEqual(report["variant_metrics"]["cpu"]["correct"], 1)

    def test_failed_cleanup_stops_generation_and_keeps_remaining_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, paths = self.make_runs(root, fail_variant="cpu", cleanup_failure=True)
            self.assertEqual(self.last_routes["cpu"].count("generate"), 1)
            run = QUALITY.read(paths[2] / "quality-run.json")
            self.assertEqual(run["status"], "failed_cleanup")
            rows = QUALITY.read(paths[2] / "results.json")["rows"]
            self.assertEqual(
                [row["answer_status"] for row in rows],
                [
                    "request_or_evidence_error",
                    "not_run_after_cleanup_failure",
                    "not_run_after_cleanup_failure",
                ],
            )
            self.assertTrue(
                all(not row["correct"] and not row["scorable"] for row in rows)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                QUALITY.compare(paths, p, root / "comparison")
            report = QUALITY.read(root / "comparison/quality-comparison.json")
            self.assertEqual(report["status"], "compared_with_cleanup_failure")
            self.assertEqual(report["variant_metrics"]["cpu"]["samples"], 3)
            self.assertEqual(report["variant_metrics"]["cpu"]["correct"], 0)

    def test_common_prefill_precision_and_quest_hot_controls_fail_closed(self):
        for key, value in (
            ("chunked_prefill_size", 4096),
            ("prefill_max_requests", 32),
            ("disable_overlap_schedule", False),
            ("page_size", 128),
            ("kv_cache_dtype", "fp8_e4m3"),
            ("speculative_algorithm", "EAGLE"),
        ):
            with self.subTest(key=key):
                info = server()
                info[key] = value
                with self.assertRaisesRegex(ValueError, "Common quality control"):
                    QUALITY.server_policy(
                        info,
                        "cpu",
                        {"max_total_tokens": 65536},
                        {"model_path": "/models/qwen"},
                        capacity_ratio=4,
                        top_k=2048,
                    )
        for key, value in (("quest_page_size", 128), ("device_buffer_size", 8192)):
            with self.subTest(key=key):
                info = server()
                config = json.loads(info["hisparse_config"])
                config[key] = value
                info["hisparse_config"] = json.dumps(config)
                with self.assertRaises(ValueError):
                    QUALITY.server_policy(
                        info,
                        "cpu",
                        {"max_total_tokens": 65536},
                        {"model_path": "/models/qwen"},
                        capacity_ratio=4,
                        top_k=2048,
                    )

    def test_compare_rechecks_saved_server_info_and_selector_consistency(self):
        for fault in ("server_hash", "changed_prefill", "selector_threshold"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                p, paths = self.make_runs(root)
                info = QUALITY.read(paths[2] / "server-info.json")
                run = QUALITY.read(paths[2] / "quality-run.json")
                if fault == "selector_threshold":
                    config = json.loads(info["hisparse_config"])
                    config["min_sparse_prompt_len"] = 512
                    info["hisparse_config"] = json.dumps(config)
                    run["server_controls"]["selector"]["min_sparse_prompt_len"] = 512
                else:
                    info["chunked_prefill_size"] = 4096
                QUALITY.write(paths[2] / "server-info.json", info)
                if fault != "server_hash":
                    run["server_info_sha256"] = QUALITY.file_sha(
                        paths[2] / "server-info.json"
                    )
                    QUALITY.write(paths[2] / "quality-run.json", run)
                with self.assertRaises(ValueError):
                    QUALITY.compare(paths, p, root / "comparison")

    def test_all_preselected_cases_remain_in_quality_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, paths = self.make_runs(root)
            with contextlib.redirect_stdout(io.StringIO()):
                QUALITY.compare(paths, p, root / "comparison")
            result = QUALITY.read(root / "comparison/quality-comparison.json")
            for variant in QUALITY.VARIANTS:
                self.assertEqual(result["variant_metrics"][variant]["samples"], 3)
                self.assertEqual(result["variant_metrics"][variant]["correct"], 2)
                self.assertEqual(result["variant_metrics"][variant]["scorable"], 2)
                self.assertEqual(
                    result["variant_metrics"][variant]["accuracy_all_selected"], 2 / 3
                )

    def test_compare_recomputes_answers_from_raw_outputs_instead_of_trusting_scores(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, paths = self.make_runs(root)
            altered = QUALITY.read(paths[2] / "results.json")
            altered["rows"][0]["predicted_answer"] = "B"
            altered["rows"][0]["correct"] = False
            QUALITY.write(paths[2] / "results.json", altered)
            (paths[2] / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in altered["rows"]) + "\n",
                encoding="utf-8",
            )
            run = QUALITY.read(paths[2] / "quality-run.json")
            run["results_sha256"] = QUALITY.file_sha(paths[2] / "results.json")
            run["results_jsonl_sha256"] = QUALITY.file_sha(paths[2] / "results.jsonl")
            run["metrics"] = QUALITY.metrics(altered["rows"])
            QUALITY.write(paths[2] / "quality-run.json", run)
            with self.assertRaisesRegex(ValueError, "independently normalized"):
                QUALITY.compare(paths, p, root / "comparison")

    def test_compare_rejects_dropped_cases_even_if_summary_hashes_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, paths = self.make_runs(root)
            altered = QUALITY.read(paths[2] / "results.json")
            altered["rows"].pop()
            QUALITY.write(paths[2] / "results.json", altered)
            run = QUALITY.read(paths[2] / "quality-run.json")
            run["results_sha256"] = QUALITY.file_sha(paths[2] / "results.json")
            run["metrics"] = QUALITY.metrics(altered["rows"])
            QUALITY.write(paths[2] / "quality-run.json", run)
            with self.assertRaisesRegex(ValueError, "dropped"):
                QUALITY.compare(paths, p, root / "comparison")


if __name__ == "__main__":
    unittest.main()
