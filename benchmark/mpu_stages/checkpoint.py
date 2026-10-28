"""Immutable evidence receipts and fail-closed experiment resume identities."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            if not path.is_file():
                raise ValueError("Evidence cannot contain directory or broken symlinks")
            result[path.relative_to(root).as_posix()] = {
                "target": os.readlink(path),
                "sha256": sha(path),
            }
            continue
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    if not result:
        raise ValueError("Cannot seal empty evidence")
    return result


def verify(root, receipt):
    if not receipt or seal(root) != receipt:
        raise ValueError("Completed evidence is missing, added to, or modified")


@contextmanager
def exclusive(root):
    """OS lock automatically releases after process exit/crash; keep its inode."""
    path = Path(root) / ".experiment.lock"
    with path.open("a+b") as stream:
        stream.seek(0)
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Another process owns this experiment directory"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def source_fingerprint(repo):
    """Hash actual code as well as recording Git identity, including archives."""
    digest = hashlib.sha256()
    count = 0
    suffixes = {
        ".py",
        ".json",
        ".cu",
        ".cuh",
        ".h",
        ".c",
        ".cc",
        ".cpp",
        ".rs",
        ".toml",
        ".txt",
        ".lock",
        ".yaml",
        ".yml",
    }
    for part in ("python", "sgl-kernel", "benchmark/mpu_stages"):
        root = Path(repo) / part
        for directory, folders, files in os.walk(root):
            folders[:] = sorted(
                n
                for n in folders
                if n
                not in {
                    ".git",
                    "__pycache__",
                    "build",
                    "target",
                    ".pytest_cache",
                    ".venv",
                }
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix not in suffixes:
                    continue
                if path.is_symlink():
                    raise ValueError(
                        "Source fingerprint refuses ambiguous symlinked code"
                    )
                digest.update(path.relative_to(repo).as_posix().encode() + b"\0")
                digest.update(bytes.fromhex(sha(path)))
                count += 1
    if not count:
        raise ValueError("Source tree was not found")
    return {"sha256": digest.hexdigest(), "files": count}


def identity(common, acceptance, stage, args, model_manifest):
    model = acceptance.model_identity(stage, args, model_manifest)
    if model["status"] != "passed":
        raise ValueError(
            "Resume/execution requires full trusted model identity: " + str(model)
        )
    runtime = common.runtime_info()
    dirty = runtime["git_status"]
    if dirty and not (
        dirty.startswith("unavailable:")
        and runtime.get("git_commit_source") == "archive_build_marker"
    ):
        raise ValueError(
            "Commit source changes before running or resuming an experiment"
        )
    if runtime["gpu"].startswith("unavailable:"):
        raise ValueError("GPU identity is unavailable")
    topology = common.capture(["nvidia-smi", "topo", "-m"])
    if topology.startswith("unavailable:"):
        raise ValueError("GPU topology is unavailable")
    return {
        "source_commit": runtime["git_commit"],
        "source": source_fingerprint(common.REPO),
        "model_manifest_sha256": sha(model_manifest),
        "model_content_fingerprint": model["evidence"].get(
            "content_fingerprint_sha256"
        ),
        "model_path": str(Path(args.model).resolve()),
        "packages": runtime["packages"],
        "python": runtime["python"],
        "gpu": runtime["gpu"],
        "topology": topology,
        "environment": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(
                (
                    "CUDA_",
                    "SGLANG_",
                    "QUEST_",
                    "NCCL_",
                    "NVSHMEM_",
                    "PYTORCH_",
                    "TORCH_",
                )
            )
            if k not in {"SGLANG_MPU_STAGE_TIMING_DIR", "SGLANG_HISPARSE_TIMING_DIR"}
        },
    }


def same_identity(expected, actual):
    if expected != actual:
        changed = sorted(
            k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)
        )
        raise ValueError("Resume identity changed: " + ", ".join(changed))
