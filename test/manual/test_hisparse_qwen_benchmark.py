"""CPU protocol checks for the future Linux/CUDA benchmark runner."""

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "benchmark/hisparse_qwen/run.py"
SPEC = importlib.util.spec_from_file_location("qwen_benchmark", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class TestBenchmarkProtocol(unittest.TestCase):
    def args(self, *extra):
        return BENCH.parser().parse_args(
            ["--output-len", "128", "--output-dir", "unused", *extra]
        )

    def test_invalid_experiments_fail_before_launch(self):
        for extra in (
            ["--output-len", "0"],
            ["--input-lens", "262144"],
            ["--num-prompts", "1"],
            ["--concurrencies", "0"],
            ["--device-buffer-size", "1024"],
            ["--host-to-device-ratio", "0"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                BENCH.validate(self.args(*extra))

    def test_both_modes_share_model_precision_selector_and_scheduler(self):
        args = self.args()
        quest = BENCH.server_command(args, "quest", 4096)
        hisparse = BENCH.server_command(args, "hisparse", 4096)
        for flag in (
            "--model-path",
            "--revision",
            "--dtype",
            "--kv-cache-dtype",
            "--tp-size",
            "--context-length",
            "--max-running-requests",
            "--hisparse-config",
            "--chunked-prefill-size",
        ):
            self.assertEqual(
                quest[quest.index(flag) + 1], hisparse[hisparse.index(flag) + 1]
            )
        self.assertNotIn("--enable-hisparse", quest)
        self.assertIn("--enable-hisparse", hisparse)
        self.assertEqual(
            quest[quest.index("--attention-backend") + 1], "flashinfer_quest"
        )

    def test_workload_seeds_match_between_modes(self):
        groups = BENCH.make_plan(self.args("--input-lens", "4096"))
        self.assertEqual(len(groups), 2)
        for baseline, offload in zip(groups[0]["cases"], groups[1]["cases"]):
            left, right = baseline["command"], offload["command"]
            for flag in (
                "--seed",
                "--random-input-len",
                "--random-output-len",
                "--max-concurrency",
                "--num-prompts",
                "--random-range-ratio",
            ):
                self.assertEqual(
                    left[left.index(flag) + 1], right[right.index(flag) + 1]
                )

    def test_fixed_lengths_use_the_dataset_implementation(self):
        command = BENCH.benchmark_command(self.args(), 4096, 1, 0, Path("unused"))
        ratio = float(command[command.index("--random-range-ratio") + 1])
        self.assertIn("--tokenize-prompt", command)
        path = BENCH.REPO / "python/sglang/benchmark/datasets/common.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_random_lens"
        )
        namespace = {"np": np, "List": list}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        for length in (128, 4096, 200000):
            self.assertEqual(
                namespace["compute_random_lens"](length, ratio, 32), [length] * 32
            )

    def test_subprocesses_load_this_checkout(self):
        self.assertEqual(
            BENCH.subprocess_env()["PYTHONPATH"].split(BENCH.os.pathsep)[0],
            str(BENCH.REPO / "python"),
        )

    def test_partial_requests_and_short_generations_are_not_successes(self):
        result = {
            "completed": 2,
            "total_output_tokens": 16,
            "input_lens": [4096, 4096],
            "output_lens": [8, 8],
            "errors": ["", ""],
            "output_throughput": 10.0,
            "mean_ttft_ms": 100.0,
            "mean_tpot_ms": 2.0,
        }
        BENCH.check_result(result, 2, 4096, 8)
        for change in (
            {"completed": 1},
            {"total_output_tokens": 15},
            {"output_lens": [7, 9]},
            {"input_lens": [4095, 4096]},
            {"errors": ["OOM", ""]},
            {"output_throughput": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                BENCH.check_result(result | change, 2, 4096, 8)

    def test_plan_does_not_claim_execution_or_overwrite_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--output-len",
                "128",
                "--input-lens",
                "4096",
                "--output-dir",
                str(output),
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertTrue(
                all(
                    case["status"] == "planned"
                    for group in manifest["groups"]
                    for case in group["cases"]
                )
            )
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
