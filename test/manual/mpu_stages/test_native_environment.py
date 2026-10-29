"""Portable build-input and artifact gates; no fake Linux/GPU success."""

import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/mpu_stages" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NATIVE, IMAGE = load("native_environment"), load("runtime_image")
CI = load("ci_native_delivery")


class NativeEnvironmentTests(unittest.TestCase):
    def test_ci_requires_the_exact_checked_out_stage_and_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            recipe = repo / "benchmark/mpu_stages/stage.json"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(json.dumps({"id": CI.STAGES[0]}))
            for args in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", "-C", str(repo), *args], check=True, capture_output=True
                )
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            CI.validate_checkout(repo, CI.STAGES[0], commit)
            with self.assertRaisesRegex(ValueError, "Checkout commit or stage"):
                CI.validate_checkout(repo, CI.STAGES[1], commit)
            with self.assertRaisesRegex(ValueError, "Checkout commit or stage"):
                CI.validate_checkout(repo, CI.STAGES[0], "0" * 40)
            with self.assertRaisesRegex(ValueError, "complete commit SHA"):
                CI.validate_checkout(repo, CI.STAGES[0], commit[:12])

    def test_ci_cleanup_rejects_a_parent_or_unowned_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            outside = Path(folder)
            marker = outside / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(ValueError):
                CI.remove_owned(outside, outside)
            with self.assertRaises(ValueError):
                CI.remove_owned(outside.parent, outside)
            self.assertEqual(marker.read_text(), "keep")

    def test_image_configuration_preserves_base_path_and_checks_launch_identity(self):
        expected = IMAGE.runtime_config(
            "/opt/mpu/fixture", "a" * 40, {"Env": ["PATH=/base/bin"]}
        )
        self.assertEqual(
            expected["Env"]["PATH"], "/opt/mpu/fixture/runtime/bin:/base/bin"
        )
        actual = dict(
            expected,
            Env=[f"{key}={value}" for key, value in expected["Env"].items()]
            + ["HOME=/root"],
        )
        IMAGE.verify_runtime_config(expected, actual)
        for key, value in (
            ("Entrypoint", ["/bin/false"]),
            ("WorkingDir", "/tmp"),
            ("Env", ["PATH=/another/runtime/bin"]),
            ("Labels", {}),
        ):
            with self.subTest(field=key):
                with self.assertRaisesRegex(ValueError, "Runtime image"):
                    IMAGE.verify_runtime_config(expected, dict(actual, **{key: value}))

    def test_native_prerequisites_accept_installed_amd64_runtime_library(self):
        self.assertEqual(
            NATIVE.check_required_os_packages(
                ["gcc-13", "libnuma1"],
                "gcc-13\tinstalled\nlibnuma1:amd64\tinstalled\n",
            ),
            {"gcc-13": "installed", "libnuma1": "installed"},
        )

    def test_missing_or_unconfigured_numa_fails_before_dependency_install(self):
        for snapshot in (
            "gcc-13\tinstalled\n",
            "libnuma1:amd64\tconfig-files\n",
            "libnuma1:amd64\thalf-configured\n",
            "libnuma1:i386\tinstalled\n",
        ):
            with self.subTest(snapshot=snapshot):
                with self.assertRaisesRegex(ValueError, "libnuma1"):
                    NATIVE.check_required_os_packages(["libnuma1"], snapshot)

    def test_source_and_locks_come_from_commit_not_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=repo, stderr=subprocess.DEVNULL
                )

            git("init", "-q")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            paths = {
                NATIVE.PROFILE: '{"id":"fixture"}\n',
                NATIVE.BUILD.STAGE_PATH: '{"family":"qwen","id":"fixture"}\n',
            }
            for name in (
                "qwen-runtime.txt",
                "qwen-toolchain.txt",
                "test-tools.txt",
                "qwen-grpc-Cargo.lock",
            ):
                paths[NATIVE.LOCKS + "/" + name] = "committed\n"
            for relative, text in paths.items():
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, newline="\n")
            git("add", ".")
            git("commit", "-qm", "Fixture")
            (repo / NATIVE.PROFILE).write_text("dirty invalid input")
            (repo / "untracked.txt").write_text("must not archive")
            plan = NATIVE.prepare(repo, "HEAD", root / "plan", "/opt/mpu/fixture")
            self.assertEqual(plan["profile"], {"id": "fixture"})
            self.assertFalse(plan["built"])
            self.assertFalse(plan["locking_scope"]["all_dependencies_locked"])
            with tarfile.open(root / "plan/source.tar") as archive:
                self.assertNotIn("untracked.txt", archive.getnames())
                for name, digest in plan["inputs"].items():
                    self.assertEqual(
                        hashlib.sha256(archive.extractfile(name).read()).hexdigest(),
                        digest,
                    )
            with self.assertRaisesRegex(ValueError, "new or empty"):
                NATIVE.prepare(repo, "HEAD", root / "plan", "/opt/mpu/fixture")

    def test_local_direct_url_substitution_requires_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pkg+cu130.whl"
            wheel.write_bytes(b"synthetic fixture")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            text = f"pkg @ https://example.invalid/pkg%2Bcu130.whl --hash=sha256:{digest}\n"
            rewritten = NATIVE.effective_lock(text, directory)
            self.assertIn(wheel.resolve().as_uri(), rewritten)
            self.assertIn(digest, rewritten)
            wheel.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                NATIVE.effective_lock(text, directory)

    def test_prefix_rejects_system_and_windows_locations(self):
        self.assertEqual(NATIVE.linux_prefix("/opt/mpu/stage01"), "/opt/mpu/stage01")
        for value in (
            "/",
            "/opt",
            "/mnt/d/stage",
            "/opt/mpu/../bad",
            "D:/stage",
            "/opt/mpu/bad name",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NATIVE.linux_prefix(value)

    def test_registry_wheelhouse_pin_is_explicit_and_rejects_wrong_platform_or_hash(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pkg-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"
            wheel.write_bytes(b"synthetic fixture")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            lock = "pkg==1.0 \\\n    --hash=sha256:" + digest + "\n"
            self.assertIn(
                wheel.resolve().as_uri(), NATIVE.effective_lock(lock, directory)
            )
            wheel.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                NATIVE.effective_lock(lock, directory)
            wheel.rename(wheel.with_name("pkg-1.0-cp312-cp312-win_amd64.whl"))
            self.assertNotIn("file:", NATIVE.effective_lock(lock, directory))

    def test_wheel_must_include_every_declared_extension_and_no_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "python").mkdir()
            (source / "python/pyproject.toml").write_text(
                '[[tool.setuptools-rust.ext-modules]]\ntarget="sglang.fixture"\n'
            )
            wheel = source / "fixture.whl"
            for entries, valid in (
                (["sglang/fixture.abi3.so"], True),
                (["sglang/other.abi3.so"], False),
                (["sglang/fixture.abi3.so", "sglang/fixture.pyc"], False),
            ):
                with zipfile.ZipFile(wheel, "w") as archive:
                    for name in entries:
                        archive.writestr(
                            name, b"synthetic fixture; not a native library"
                        )
                if valid:
                    self.assertEqual(
                        NATIVE.check_wheel(wheel, source)["declared_extensions"],
                        ["sglang.fixture"],
                    )
                else:
                    with self.assertRaises(ValueError):
                        NATIVE.check_wheel(wheel, source)

    def test_incomplete_native_receipt_cannot_package_image(self):
        for receipt in (
            {},
            {"status": "built_jit_verified", "built": False},
            {"status": "built_jit_verified", "built": True, "gpu_executed": True},
        ):
            with self.assertRaisesRegex(ValueError, "successful native"):
                IMAGE.validate_native(receipt)

    def test_rebound_local_base_tag_cannot_change_pinned_image(self):
        with patch.object(
            IMAGE.subprocess, "check_output", return_value='[{"Id":"sha256:wrong"}]'
        ):
            with self.assertRaisesRegex(ValueError, "no longer"):
                IMAGE.verify_base_reference("fixture:tag", "sha256:expected")


if __name__ == "__main__":
    unittest.main()
