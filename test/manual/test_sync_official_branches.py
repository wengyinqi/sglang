"""Exercise mirror updates against temporary local Git remotes, without network."""

import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "tools/sync_official_branches.py"
SPEC = importlib.util.spec_from_file_location("sync_official_branches", SOURCE)
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def git(root, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Mirror Test",
            "-c",
            "user.email=mirror@example.test",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class MirrorSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        git(self.upstream, "init", "-b", "main")
        git(self.upstream, "commit", "--allow-empty", "-m", "upstream base")
        self.base = git(self.upstream, "rev-parse", "HEAD")
        for name in SYNC.BRANCHES[1:]:
            git(self.upstream, "branch", name)
        self.checkout = self.root / "checkout"
        git(
            self.root, "clone", "--no-hardlinks", str(self.upstream), str(self.checkout)
        )
        git(self.checkout, "remote", "rename", "origin", "upstream")
        for name in SYNC.BRANCHES[1:]:
            git(self.checkout, "branch", name, f"upstream/{name}")
        git(self.checkout, "switch", "-c", "mpu-exp/test")
        git(self.checkout, "commit", "--allow-empty", "-m", "experiment")
        self.experiment = git(self.checkout, "rev-parse", "HEAD")
        self.fork = self.root / "fork.git"
        git(
            self.root,
            "clone",
            "--bare",
            "--no-hardlinks",
            str(self.upstream),
            str(self.fork),
        )
        git(self.checkout, "remote", "add", "origin", str(self.fork))
        git(self.upstream, "commit", "--allow-empty", "-m", "official update")
        self.updated = git(self.upstream, "rev-parse", "HEAD")

    def sync(self, *args):
        with patch.object(SYNC, "ROOT", self.checkout), patch.object(
            SYNC, "UPSTREAM", str(self.upstream)
        ), contextlib.redirect_stdout(io.StringIO()):
            SYNC.main(list(args))

    def test_plan_then_atomic_push_preserves_experiment(self):
        self.sync()
        self.assertEqual(git(self.checkout, "rev-parse", "main"), self.base)
        self.sync("--apply", "--push")
        self.assertEqual(git(self.checkout, "rev-parse", "main"), self.updated)
        self.assertEqual(git(self.fork, "rev-parse", "main"), self.updated)
        self.assertEqual(
            git(self.checkout, "rev-parse", "mpu-exp/test"), self.experiment
        )
        for name in SYNC.BRANCHES[1:]:
            self.assertEqual(git(self.checkout, "rev-parse", name), self.base)

    def test_occupied_mirror_rejected_before_any_update(self):
        git(self.checkout, "switch", "hisparse-fix")
        with self.assertRaisesRegex(RuntimeError, "worktrees"):
            self.sync("--apply")
        self.assertEqual(git(self.checkout, "rev-parse", "main"), self.base)

    def test_unpreserved_local_and_uninspected_remote_work_rejected(self):
        git(self.checkout, "switch", "hisparse_quest")
        git(self.checkout, "commit", "--allow-empty", "-m", "unpreserved work")
        divergent = git(self.checkout, "rev-parse", "HEAD")
        git(self.checkout, "switch", "mpu-exp/test")
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            self.sync("--apply")
        self.assertEqual(git(self.checkout, "rev-parse", "hisparse_quest"), divergent)
        git(self.checkout, "branch", "mpu-exp/preserved", divergent)
        git(self.checkout, "push", "origin", "mpu-exp/test:refs/heads/main")
        with self.assertRaisesRegex(RuntimeError, "uninspected"):
            self.sync("--apply", "--push")
        self.assertEqual(git(self.checkout, "rev-parse", "main"), self.base)
        self.sync("--apply")
        self.assertEqual(git(self.checkout, "rev-parse", "hisparse_quest"), self.base)
        self.assertEqual(
            git(self.checkout, "rev-parse", "mpu-exp/preserved"), divergent
        )


if __name__ == "__main__":
    unittest.main()
