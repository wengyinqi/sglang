"""Portable build-input and artifact gates; no fake Linux/GPU success."""

import hashlib
import importlib.util
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/mpu_stages" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NATIVE, IMAGE = load("native_environment"), load("runtime_image")


class NativeEnvironmentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
