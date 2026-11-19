"""Input provenance, direct-ID loading, and paired request identity contracts."""

import ast
import copy
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKLOAD = load("real_workload", "benchmark/mpu_stages/workload.py")
RUN = load("real_workload_run", "benchmark/mpu_stages/run.py")
PROFILE = load("real_workload_profile", "benchmark/mpu_stages/profile_decode.py")
CAMPAIGN = load(
    "real_workload_campaign", "benchmark/mpu_stages/real_workload_experiment.py"
)
BOTTLENECK = load(
    "real_workload_bottleneck", "benchmark/mpu_stages/bottleneck_analyze.py"
)


def frozen_module():
    # Isolate the production loader from serving imports on CPU-only hosts.
    path = ROOT / "python/sglang/benchmark/datasets/frozen_ids.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "sglang.benchmark.datasets.common"
        )
    ]
    module = types.ModuleType("frozen_ids_cpu_test")
    module.BaseDataset = object
    module.DatasetRow = lambda **kwargs: SimpleNamespace(**kwargs)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


FROZEN = frozen_module()


class Tokenizer:
    def __init__(self, root):
        self.name_or_path = str(root)
        (root / "tokenizer.json").write_text('{"fixture":true}', encoding="utf-8")

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        return ids[:max_length] if truncation else ids

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class TestRealWorkload(unittest.TestCase):
    def prepare(self, root, count=3):
        tokenizer = Tokenizer(root)
        records = [
            dict(
                _id=f"doc-{i}",
                context=(f"Document {i}. " * 500),
                question=f"Which document is {i}?",
                domain="fixture",
                **{f"choice_{letter}": letter for letter in "ABCD"},
            )
            for i in range(4)
        ]
        output = root / "frozen"
        WORKLOAD.freeze(
            records,
            tokenizer,
            [600, 1000],
            count,
            1729,
            output,
            {"repo": "fixture", "revision": "0" * 40},
        )
        return tokenizer, output / "workload.json"

    def test_natural_crops_preserve_question_order_and_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer, manifest_path = self.prepare(Path(tmp))
            manifest = WORKLOAD.load_manifest(
                manifest_path, lengths=[600, 1000], count=3
            )
            short, long = [
                json.loads(WORKLOAD.bucket(manifest_path, n)[0].read_text())
                for n in (600, 1000)
            ]
            self.assertEqual(
                [r["source_id"] for r in short["rows"]],
                [r["source_id"] for r in long["rows"]],
            )
            self.assertEqual(
                manifest["row_order"],
                "fixed across variants, concurrency points and repeats",
            )
            text = "".join(chr(t) for t in short["rows"][0]["input_ids"])
            self.assertIn("Question:", text)
            self.assertTrue(text.endswith("</user><assistant>"))
            for n in (600, 1000):
                path, item = WORKLOAD.bucket(manifest_path, n)
                rows = FROZEN.load_rows(
                    path, item["sha256"], n, 3, tokenizer.name_or_path, len(tokenizer)
                )
                self.assertEqual(
                    [r["input_ids"] for r in rows], WORKLOAD.inputs(manifest_path, n, 3)
                )

    def test_short_documents_are_never_repeated_to_fill_a_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dict(
                _id="short",
                context="short",
                question="Q",
                **{f"choice_{letter}": letter for letter in "ABCD"},
            )
            with self.assertRaisesRegex(ValueError, "Only 0 distinct"):
                WORKLOAD.freeze(
                    [record], Tokenizer(root), [1000], 1, 1, root / "frozen", {}
                )

    def test_tampered_tokenizer_input_and_undersized_snapshot_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer, manifest_path = self.prepare(root)
            path, item = WORKLOAD.bucket(manifest_path, 600)
            with self.assertRaisesRegex(ValueError, "Not enough"):
                FROZEN.load_rows(path, item["sha256"], 600, 4, root, len(tokenizer))
            (root / "tokenizer.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                FROZEN.load_rows(path, item["sha256"], 600, 3, root, len(tokenizer))
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                WORKLOAD.load_manifest(manifest_path)
            with self.assertRaisesRegex(ValueError, "hash changed"):
                FROZEN.load_rows(path, item["sha256"], 600, 3, root, len(tokenizer))

    def test_invalid_ids_and_duplicate_documents_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer, manifest_path = self.prepare(root)
            path, _ = WORKLOAD.bucket(manifest_path, 600)
            original = json.loads(path.read_text())
            for mutate in (
                lambda p: p["rows"][0]["input_ids"].__setitem__(0, True),
                lambda p: p["rows"][0]["input_ids"].__setitem__(0, 256),
                lambda p: p["rows"][1].__setitem__(
                    "source_id", p["rows"][0]["source_id"]
                ),
            ):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                WORKLOAD.write_json(path, payload)
                with self.assertRaises(ValueError):
                    FROZEN.load_rows(
                        path,
                        WORKLOAD.file_sha256(path),
                        600,
                        3,
                        tokenizer.name_or_path,
                        len(tokenizer),
                    )

    def test_each_variant_repeat_and_concurrency_use_the_same_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest_path = self.prepare(root)
            args = SimpleNamespace(
                model="fixture",
                port=30000,
                output_len=256,
                num_prompts=3,
                seed=1729,
                warmup_requests=1,
                workload=manifest_path,
            )
            commands = [
                RUN.COMMON.benchmark_command(args, 600, c, r, root / f"{c}-{r}.jsonl")
                for c in (1, 3)
                for r in (0, 1)
            ]
            identities = [
                (
                    cmd[cmd.index("--dataset-path") + 1],
                    cmd[cmd.index("--frozen-ids-sha256") + 1],
                )
                for cmd in commands
            ]
            self.assertEqual(len(set(identities)), 1)
            self.assertTrue(
                all(
                    cmd[cmd.index("--dataset-name") + 1] == "frozen-ids"
                    for cmd in commands
                )
            )
            self.assertEqual(
                {int(cmd[cmd.index("--random-output-len") + 1]) for cmd in commands},
                {256},
            )

    def test_frozen_loader_never_uses_tokenizer_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer, manifest_path = self.prepare(root)
            path, item = WORKLOAD.bucket(manifest_path, 600)
            tokenizer.encode = lambda *a, **k: self.fail(
                "Frozen input was re-tokenized"
            )
            dataset = FROZEN.FrozenIDsDataset(str(path), item["sha256"], 600, 256, 3)
            rows = dataset.load(tokenizer)
            self.assertEqual(
                [r.prompt for r in rows], WORKLOAD.inputs(manifest_path, 600, 3)
            )
            with self.assertRaisesRegex(ValueError, "tokenize-prompt"):
                FROZEN.FrozenIDsDataset.from_args(
                    SimpleNamespace(tokenize_prompt=False)
                )

    def test_hbm_profile_uses_hbm_recipe_and_rejects_zero_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, args = RUN.STAGE.parse_args(["--output-dir", str(root)])
            stage = RUN.STAGE.load_stage(args.stage)
            report = PROFILE.plan(stage, args, 32, "hbm", 1)
            command = report["server_command"]
            self.assertIn("flashinfer_quest", command)
            self.assertNotIn("--enable-hisparse", command)
            self.assertEqual(report["profile_variant"], "hbm")
            with self.assertRaises(ValueError):
                PROFILE.plan(stage, args, 32, "cpu", 0)

    def test_campaign_reuses_ratio2_and_preserves_fixed_gpu_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, args = RUN.STAGE.parse_args(["--output-dir", str(root)])
            args.workload = root / "workload.json"
            args.workload_sha256 = "0" * 64
            stage = RUN.STAGE.load_stage(args.stage)
            jobs = CAMPAIGN.plan(stage, args, root / "model.json", True)
            paired = next(job for job in jobs if job["id"] == "paired")["command"]
            extra = next(job for job in jobs if job["id"] == "capacity-extra")[
                "command"
            ]
            self.assertIn("--include-hbm", paired)
            self.assertNotIn("--include-hbm", extra)
            self.assertEqual(
                extra[
                    extra.index("--capacity-ratios") + 1 : extra.index("--output-dir")
                ],
                ["1", "4", "8"],
            )
            for flag in (
                "--max-total-tokens",
                "--chunked-prefill-size",
                "--prefill-max-requests",
                "--mem-fraction-static",
                "--workload-sha256",
            ):
                self.assertEqual(
                    paired[paired.index(flag) + 1], extra[extra.index(flag) + 1]
                )

    def test_prefill_control_preserves_decode_concurrency_and_rejects_zero(self):
        _, args = RUN.STAGE.parse_args(["--concurrencies", "32"])
        stage = RUN.STAGE.load_stage(args.stage)
        self.assertEqual(args.prefill_max_requests, 1)
        for variant, ratio in (("cpu", 2), ("mpu", 2), ("hbm", None)):
            command = RUN.STAGE.server_command(stage, args, variant, 32768, ratio)
            self.assertEqual(command[command.index("--prefill-max-requests") + 1], "1")
            self.assertEqual(command[command.index("--max-running-requests") + 1], "32")
        with self.assertRaises(ValueError):
            RUN.STAGE.parse_args(["--prefill-max-requests", "0"])

    def test_profile_rejects_a_different_prefill_batch_control(self):
        stage, args = RUN.STAGE.parse_args([])
        runtime = {"git_commit": "a" * 40, "git_status": ""}
        report = {
            "stage": stage,
            "runtime": runtime,
            "evidence_scope": {"target_model_correctness_executed": True},
            "steps": [
                {
                    "id": "correctness",
                    "status": "passed",
                    "results": [{"status": "passed"}],
                }
            ],
        }
        settings = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        model_report = {"stage": stage, "runtime": runtime, "settings": settings}
        PROFILE.correctness_gate(report, stage, runtime, args, model_report, "cpu")
        settings["prefill_max_requests"] = 2
        with self.assertRaisesRegex(ValueError, "prefill_max_requests"):
            PROFILE.correctness_gate(report, stage, runtime, args, model_report, "cpu")

    def test_campaign_cli_preserves_model_path_and_manifest_as_separate_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest_path = self.prepare(root)
            output = root / "campaign-plan"
            with contextlib.redirect_stdout(io.StringIO()):
                code = CAMPAIGN.main(
                    [
                        "--model",
                        "/fixture/local-qwen",
                        "--model-manifest",
                        str(root / "model-manifest.json"),
                        "--workload",
                        str(manifest_path),
                        "--input-lens",
                        "600",
                        "1000",
                        "--num-prompts",
                        "3",
                        "--concurrencies",
                        "1",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
            for job in report["jobs"]:
                command = job["command"]
                self.assertEqual(
                    command[command.index("--model") + 1], "/fixture/local-qwen"
                )
            first = report["jobs"][0]["command"]
            self.assertEqual(
                first[first.index("--model-manifest") + 1],
                str(root / "model-manifest.json"),
            )

    def test_cpu_campaign_retains_original_job_counts_and_profile_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, args = RUN.STAGE.parse_args(["--output-dir", str(root)])
            stage = copy.deepcopy(stage)
            stage.update(id="01-qwen-quest-1gpu-0mpu", variant="cpu")
            args.workload = root / "workload.json"
            args.workload_sha256 = "0" * 64
            jobs = CAMPAIGN.plan(stage, args, root / "model.json", True)
            self.assertEqual(
                CAMPAIGN.planned_counts(stage, jobs),
                dict(
                    formal_performance_cases=90,
                    formal_measured_requests=2880,
                    length_diagnostic_cases=12,
                    length_diagnostic_requests=96,
                ),
            )
            self.assertEqual(
                [j["id"] for j in jobs if j["id"].startswith("profile-")],
                ["profile-cpu-ratio2", "profile-hbm-ratio2", "profile-cpu-ratio8"],
            )

    def test_mpu_campaign_preserves_shards_reference_and_graph_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, args = RUN.STAGE.parse_args(
                [
                    "--output-dir",
                    str(root),
                    "--mpu-device-ids",
                    "1",
                    "2",
                    "3",
                    "4",
                    "--mpu-memory-reserve-bytes",
                    "4096",
                    "--cpu-capacity-ratio",
                    "3",
                    "--disable-cuda-graph",
                ]
            )
            stage = copy.deepcopy(stage)
            stage.update(id="02-qwen-quest-1gpu-1mpu", variant="mpu")
            args.workload = root / "workload.json"
            args.workload_sha256 = "0" * 64
            jobs = CAMPAIGN.plan(stage, args, root / "model.json", True)
            self.assertEqual(
                CAMPAIGN.planned_counts(stage, jobs),
                dict(
                    formal_performance_cases=130,
                    formal_measured_requests=4160,
                    length_diagnostic_cases=18,
                    length_diagnostic_requests=144,
                ),
            )
            for job in jobs:
                command = job["command"]
                self.assertEqual(
                    CAMPAIGN.option_values(command, "--mpu-device-ids"),
                    ["1", "2", "3", "4"],
                )
                self.assertEqual(
                    CAMPAIGN.option_values(command, "--cpu-capacity-ratio"), ["3"]
                )
                self.assertEqual(
                    CAMPAIGN.option_values(command, "--mpu-memory-reserve-bytes"),
                    ["4096"],
                )
                self.assertIn("--disable-cuda-graph", command)
                self.assertEqual(
                    CAMPAIGN.option_values(command, "--prefill-max-requests"), ["1"]
                )
            profiles = [job for job in jobs if job["id"].startswith("profile-")]
            self.assertEqual(
                [j["id"] for j in profiles],
                [
                    "profile-cpu-ratio3",
                    "profile-mpu-ratio2",
                    "profile-hbm-ratio2",
                    "profile-mpu-ratio8",
                ],
            )
            self.assertEqual(
                CAMPAIGN.option_values(profiles[0]["command"], "--capacity-ratios"),
                ["2"],
            )
            self.assertIn("--include-hbm", jobs[0]["command"])
            stage["family"] = "glm"
            with self.assertRaisesRegex(ValueError, "Qwen Quest"):
                CAMPAIGN.plan(stage, args, root / "model.json", False)

    def campaign_manifests(self, root, target="mpu"):
        stage, _ = RUN.STAGE.parse_args([])
        stage = copy.deepcopy(stage)
        stage.update(variant=target)
        lengths, cpu_ratio = [8192, 32768], 3

        def group(variant, length, ratio, concurrencies):
            gid = f"{variant}-{length}-{ratio}"
            return dict(
                id=gid,
                variant=variant,
                input_len=length,
                ratio=ratio,
                status="success",
                capacity=[
                    dict(
                        physical_compute_slots=65536,
                        compute_kv_bytes=6291456,
                        backing_slots=int(65536 * (ratio or 1)),
                        backing_kv_bytes=int(6291456 * (ratio or 1)),
                        allocated_backing_kv_bytes=int(6291456 * (ratio or 1)),
                        requested_capacity_ratio=ratio,
                        effective_capacity_ratio=ratio,
                        storage_device_ids=[1] if variant == "mpu" else [],
                    )
                ],
                cases=[
                    dict(id=f"{gid}-c{c}", concurrency=c, repeat=0, status="success")
                    for c in concurrencies
                ],
            )

        paired, extra = [], []
        for length in lengths:
            paired += [
                group(target, length, 2, [1, 4, 32]),
                group("hbm", length, None, [1, 4, 32]),
            ]
            extra += [group(target, length, ratio, [32]) for ratio in (1, 4, 8)]
            if target == "mpu":
                paired.append(group("cpu", length, cpu_ratio, [1, 4, 32]))
                extra.append(group("cpu", length, cpu_ratio, [32]))
        shared = dict(
            stage=stage,
            runtime={"git_commit": "a" * 40},
            workload={"sha256": "0" * 64},
            analysis_policy={},
            status="success",
        )
        settings = dict(
            input_lens=lengths,
            cpu_capacity_ratio=cpu_ratio,
            prefill_max_requests=1,
            plateau_threshold=0.05,
        )
        pair = dict(
            shared,
            groups=paired,
            settings=dict(
                settings,
                output_dir=str(root / "paired"),
                capacity_ratios=[2],
                include_hbm=True,
                concurrencies=[1, 4, 32],
            ),
        )
        capacity = dict(
            shared,
            groups=extra,
            settings=dict(
                settings,
                output_dir=str(root / "capacity-extra"),
                capacity_ratios=[1, 4, 8],
                include_hbm=False,
                concurrencies=[32],
            ),
        )
        return pair, capacity

    def write_campaign_manifests(self, root, pair, capacity):
        for name, payload in (("paired", pair), ("capacity-extra", capacity)):
            (root / name).mkdir(exist_ok=True)
            (root / name / "manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    def test_capacity_combine_selects_target_ratio2_and_one_cpu_reference(self):
        for target, expected_groups in (("cpu", 8), ("mpu", 10)):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pair, capacity = self.campaign_manifests(root, target)
                self.write_campaign_manifests(root, pair, capacity)
                with patch.object(
                    CAMPAIGN.run, "summarize", return_value={"fixture": "summary"}
                ):
                    CAMPAIGN.combine(root)
                combined = json.loads(
                    (root / "capacity-combined.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(combined["groups"]), expected_groups)
                self.assertEqual(
                    len(set(CAMPAIGN.group_identity(g) for g in combined["groups"])),
                    expected_groups,
                )
                self.assertTrue(
                    all(
                        c["concurrency"] == 32
                        for g in combined["groups"]
                        for c in g["cases"]
                    )
                )
                self.assertTrue(all(g["variant"] != "hbm" for g in combined["groups"]))
                for length in (8192, 32768):
                    self.assertEqual(
                        {
                            g["ratio"]
                            for g in combined["groups"]
                            if g["variant"] == target and g["input_len"] == length
                        },
                        {1, 2, 4, 8},
                    )

    def test_capacity_combine_rejects_identity_control_and_duplicate_reference_drift(
        self,
    ):
        mutations = (
            lambda p, c: c.__setitem__("runtime", {"git_commit": "b" * 40}),
            lambda p, c: c["settings"].__setitem__("prefill_max_requests", 2),
            lambda p, c: c.__setitem__("analysis_policy", {"changed": True}),
            lambda p, c: c["groups"].append(copy.deepcopy(c["groups"][-1])),
            lambda p, c: c["groups"].pop(2),
            lambda p, c: c["groups"][-1]["capacity"][0].__setitem__(
                "allocated_backing_kv_bytes", 1
            ),
            lambda p, c: p["groups"][0].__setitem__(
                "cases", [dict(concurrency=1, status="success")]
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pair, capacity = self.campaign_manifests(root)
                mutate(pair, capacity)
                self.write_campaign_manifests(root, pair, capacity)
                with patch.object(
                    CAMPAIGN.run, "summarize", return_value={}
                ), self.assertRaises(ValueError):
                    CAMPAIGN.combine(root)

    def test_kernel_union_does_not_double_count_overlapping_intervals(self):
        self.assertEqual(BOTTLENECK.union_ns([(0, 10), (5, 15), (20, 25)]), 20)
        self.assertEqual(
            BOTTLENECK.category("load_cache_to_device_buffer_kernel<bf16>"),
            "kv_resolve_and_pack",
        )
        self.assertEqual(
            BOTTLENECK.category("_quest_score_kernel"), "quest_selection_and_bounds"
        )
        self.assertEqual(BOTTLENECK.category("some_new_kernel"), "unclassified")


if __name__ == "__main__":
    unittest.main()
