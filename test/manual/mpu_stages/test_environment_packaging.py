"""Offline construction/verification of the two source-pinned image recipes."""

import importlib.util
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "mpu_environment", ROOT / "benchmark/mpu_stages/build_environment.py"
)
ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENV)

DOCKERFILE = """ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base
ARG BRANCH_TYPE=remote
FROM base AS framework
FROM scratch AS local_src
COPY . /src
FROM framework AS framework_final
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \\
        cp -r /tmp/local_src /sgl-workspace/sglang; \\
    else \\
        git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \\
    fi \\
    && rm -rf /tmp/local_src
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime
"""


class EnvironmentPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.run_git("init", "-q")
        self.run_git("config", "user.email", "fixture@example.invalid")
        self.run_git("config", "user.name", "Fixture")
        self.stage = {
            "schema_version": 1,
            "id": "01-fixture",
            "family": "qwen",
            "default_model": "test/fixture",
            "default_revision": "a" * 40,
        }
        files = {
            "benchmark/mpu_stages/stage.json": json.dumps(self.stage),
            "python/pyproject.toml": '[project]\nname="sglang"\ndependencies = ["torch==2.9.1", "flashinfer_python==0.6.8.post1", "sglang-kernel==0.4.1.post1"]\n',
            "python/sglang/__init__.py": "# committed source\n",
            "docker/Dockerfile": DOCKERFILE,
        }
        for name, content in files.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "Fixture")

    def run_git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    def plan(self, **kwargs):
        return ENV.prepare_environment(
            self.repo, None, "HEAD", self.root / "output", **kwargs
        )

    def test_plan_is_offline_and_archives_only_committed_source(self):
        (self.repo / "private-secret.txt").write_text("do not copy", encoding="utf-8")
        (self.repo / "python/sglang/__init__.py").write_text(
            "# dirty version\n", encoding="utf-8"
        )
        plan = self.plan()
        self.assertEqual(plan["status"], "recipe_generated")
        self.assertFalse(plan["built"])
        self.assertTrue(plan["source"]["working_tree_dirty"])
        self.assertFalse(plan["source"]["working_tree_changes_included"])
        self.assertEqual(plan["source"]["commit"], self.run_git("rev-parse", "HEAD"))
        with tarfile.open(plan["build"]["stdin_file"]) as context:
            self.assertNotIn("private-secret.txt", context.getnames())
            self.assertEqual(
                context.extractfile("python/sglang/__init__.py").read(),
                b"# committed source\n",
            )
            dockerfile = context.extractfile("_mpu_build/Dockerfile").read().decode()
            self.assertNotIn("git clone", dockerfile)
            self.assertIn("ENV PYTHONPATH=/sgl-workspace/sglang/python", dockerfile)
            self.assertIn("/sgl-workspace/sglang/.mpu-source.json", dockerfile)
            self.assertIn("ARG BRANCH_TYPE=local", dockerfile)
            marker = json.load(context.extractfile("_mpu_build/build-input.json"))
            self.assertEqual(marker["source_commit"], plan["source"]["commit"])
        self.assertIn("BRANCH_TYPE=local", plan["build"]["command"])
        self.assertIsNone(plan["image"]["id"])
        self.assertFalse(plan["reproducibility"]["all_dependencies_locked"])

    def test_source_family_cannot_select_a_different_dependency_stack(self):
        self.stage["family"] = "glm-5.3-flash"
        (self.repo / ENV.STAGE_PATH).write_text(
            json.dumps(self.stage), encoding="utf-8"
        )
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "Mismatched fixture")
        with self.assertRaisesRegex(ValueError, "pyproject mismatch"):
            self.plan()

    def test_different_stage_or_mutable_worktree_changes_are_rejected(self):
        alternate = self.root / "stage.json"
        alternate.write_text(
            json.dumps(self.stage | {"id": "02-other"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            ENV.prepare_environment(self.repo, alternate, "HEAD", self.root / "output")

    def test_two_recipes_have_pins_matching_their_families(self):
        old, _ = ENV.recipe_for("qwen")
        new, _ = ENV.recipe_for("deepseek-v4-flash")
        glm, _ = ENV.recipe_for("glm-5.3-flash")
        self.assertEqual(new, glm)
        self.assertEqual(old["expected_packages"]["torch"], "2.9.1")
        self.assertEqual(new["expected_packages"]["torch"], "2.14.1")
        self.assertEqual(new["expected_packages"]["tilelang"], "0.1.12")
        self.assertEqual(
            ENV.dependency_pins(
                '"flashinfer_python[cu13]==0.7.2"', {"flashinfer-python": "0.7.2"}
            ),
            {"flashinfer-python": "0.7.2"},
        )

    def test_custom_base_digest_is_recorded_but_not_claimed_registry_verified(self):
        base = "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:" + "a" * 64
        plan = self.plan(base_image=base, image_tag="local/mpu:fixture")
        self.assertEqual(plan["build"]["image_tag"], "local/mpu:fixture")
        self.assertEqual(
            plan["build"]["external_images"],
            [{"reference": base, "digest_pinned": True, "registry_verified": False}],
        )

    def test_context_tampering_blocks_execution_before_docker(self):
        plan = self.plan()
        with Path(plan["build"]["stdin_file"]).open("ab") as stream:
            stream.write(b"tampered")
        with patch.object(ENV.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "context changed"):
                ENV.execute_build(plan, self.root / "output")
            run.assert_not_called()

    def test_execute_records_failure_without_claiming_a_built_image(self):
        plan = self.plan()
        with patch.object(
            ENV.subprocess, "run", side_effect=FileNotFoundError("docker unavailable")
        ):
            with self.assertRaises(FileNotFoundError):
                ENV.execute_build(plan, self.root / "output")
        result = json.loads((self.root / "output/environment-build.json").read_text())
        self.assertFalse(result["built"])
        self.assertEqual(result["status"], "build_or_capture_failed")

    def test_execute_records_image_id_digest_and_complete_freeze(self):
        plan = self.plan()
        runtime = {
            "source_commit": plan["source"]["commit"],
            "source_archive_sha256": plan["source"]["archive_sha256"],
            "pip_freeze": "pip==26.0\ntorch==2.9.1+cu130\n",
        }
        details = {"Id": "sha256:" + "b" * 64, "RepoDigests": []}
        with patch.object(
            ENV.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(),
                SimpleNamespace(stdout=json.dumps([details])),
                SimpleNamespace(stdout=json.dumps(runtime)),
            ],
        ) as run:
            result = ENV.execute_build(plan, self.root / "output")
        self.assertTrue(result["built"])
        self.assertEqual(result["image"]["id"], details["Id"])
        self.assertEqual(result["image"]["repo_digests"], [])
        self.assertEqual(
            (self.root / "output/pip-freeze.txt").read_text(), runtime["pip_freeze"]
        )
        self.assertIn(details["Id"], run.call_args_list[2].args[0])


if __name__ == "__main__":
    unittest.main()
