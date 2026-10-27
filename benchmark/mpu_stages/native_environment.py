"""Source-pinned native Linux build, with separate runtime and CUDA toolchain.

prepare works on Windows; execute requires Ubuntu 24.04 / Python 3.12 on a
native Linux filesystem. Existing build prefixes and evidence are not replaced.
This builds the experimental server wheel, not upstream Docker BUILD_TYPE=all.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "mpu_build_environment", HERE / "build_environment.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
LOCKS = "benchmark/mpu_stages/environments/locks"
PROFILE = "benchmark/mpu_stages/environments/native-platform.json"


def identity(path):
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": BUILD.sha256(path),
    }


def linux_prefix(value):
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 4
        or ".." in path.parts
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
        or value.startswith(("/mnt/", "/proc/", "/sys/", "/dev/"))
    ):
        raise ValueError(
            "Use a dedicated native Linux prefix, for example /opt/mpu/stage01-build1"
        )
    return str(path)


def prepare(repo, source_ref, output, prefix):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    prefix = linux_prefix(prefix)
    commit = BUILD.git(repo, "rev-parse", "--verify", source_ref + "^{commit}")
    stage = json.loads(BUILD.git(repo, "show", f"{commit}:{BUILD.STAGE_PATH}"))
    profile = json.loads(BUILD.git(repo, "show", f"{commit}:{PROFILE}"))
    family = "qwen" if stage["family"] == "qwen" else "dsa"
    if stage["family"] not in ("qwen", "deepseek-v4-flash", "glm-5.3-flash"):
        raise ValueError("Unsupported family")
    files = [
        PROFILE,
        f"{LOCKS}/{family}-runtime.txt",
        f"{LOCKS}/{family}-toolchain.txt",
        f"{LOCKS}/test-tools.txt",
    ]
    if family == "qwen":
        files.append(f"{LOCKS}/qwen-grpc-Cargo.lock")
    inputs = {
        name: hashlib.sha256(
            BUILD.git(repo, "show", f"{commit}:{name}").encode()
        ).hexdigest()
        for name in files
    }
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evidence output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "source.tar"
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "archive",
            "--format=tar",
            f"--output={archive}",
            commit,
        ],
        cwd=repo,
        check=True,
    )
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "built": False,
        "source_commit": commit,
        "source_archive_sha256": BUILD.sha256(archive),
        "provenance": "git_archive",
        "stage": stage,
        "profile": profile,
        "family": family,
        "prefix": prefix,
        "inputs": inputs,
        "working_tree_changes_included": False,
        "locking_scope": {
            "python_versions_and_distribution_hashes": True,
            "cargo_locks": True,
            "os_packages": False,
            "all_dependencies_locked": False,
        },
    }
    BUILD.write_json(output / "native-plan.json", plan)
    return plan


def effective_lock(text, wheelhouse):
    """Only rewrite a direct URL to a verified local copy; retain its hash."""
    if wheelhouse is None:
        return text
    from urllib.parse import unquote, urlsplit

    pattern = r"(?m)^(\S+) @ (https://\S+) --hash=sha256:([0-9a-f]{64})$"

    def replace(match):
        name, url, expected = match.groups()
        wheel = Path(wheelhouse) / unquote(PurePosixPath(urlsplit(url).path).name)
        if not wheel.is_file() or BUILD.sha256(wheel) != expected:
            raise ValueError(f"Missing or changed direct-URL wheel: {wheel}")
        return f"{name} @ {wheel.resolve().as_uri()} --hash=sha256:{expected}"

    return re.sub(pattern, replace, text)


def native_filesystem(path):
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    kind = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(probe)], text=True
    ).strip()
    if kind in ("9p", "drvfs", "fuseblk", "ntfs", "smb2", "cifs"):
        raise ValueError(
            f"Build/cache must use a native Linux filesystem, got {kind}: {path}"
        )
    return kind


def check_wheel(wheel, source):
    import tomllib

    project = tomllib.loads((source / "python/pyproject.toml").read_text())
    declared = [
        entry["target"]
        for entry in project.get("tool", {})
        .get("setuptools-rust", {})
        .get("ext-modules", [])
    ]
    if not declared:
        for path in sorted((source / "rust").glob("*/Cargo.toml")):
            metadata = (
                tomllib.loads(path.read_text())
                .get("package", {})
                .get("metadata", {})
                .get("sglang", {})
            )
            if "python-module" in metadata:
                declared.append(metadata["python-module"])
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        extensions = [n for n in names if n.endswith(".so")]
        if (
            archive.testzip() is not None
            or any(n.endswith(".pyc") for n in names)
            or not declared
            or len(declared) != len(extensions)
            or any(
                not any(n.startswith(d.replace(".", "/") + ".") for n in extensions)
                for d in declared
            )
        ):
            raise ValueError(
                "Wheel CRC, bytecode or declared native extension coverage mismatch"
            )
    return {
        "declared_extensions": declared,
        "packaged_extensions": extensions,
        "crc_passed": True,
    }


def build_env(prefix, cache, offline):
    cuda = prefix / "toolchain/lib/python3.12/site-packages/nvidia/cu13"
    env = dict(
        os.environ,
        UV_CACHE_DIR=str(cache),
        UV_CONCURRENT_DOWNLOADS="2",
        UV_CONCURRENT_INSTALLS="1",
        MAX_JOBS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="1",
        CARGO_BUILD_JOBS="1",
        CARGO_TARGET_DIR=str(cache / "cargo-target"),
        PYTHONDONTWRITEBYTECODE="1",
        SGLANG_BUILD_RUST_EXTS="all",
        CUDA_HOME=str(cuda),
        CXX="/usr/bin/g++-13",
        CC="/usr/bin/gcc-13",
        PATH=str(prefix / "runtime/bin")
        + ":"
        + str(Path.home() / ".cargo/bin")
        + ":"
        + os.environ["PATH"],
        PYTHONPATH=str(prefix / "source/python"),
        UV_NO_PROGRESS="1",
    )
    if offline:
        env.update(UV_OFFLINE="1", CARGO_NET_OFFLINE="true")
    return env


def execute(output, cache, wheelhouse=None, offline=False):
    output, cache = Path(output).resolve(), Path(cache).resolve()
    plan = json.loads((output / "native-plan.json").read_text(encoding="utf-8"))
    prefix = Path(linux_prefix(plan["prefix"]))
    if (
        sys.platform != "linux"
        or sys.version_info[:2] != (3, 12)
        or platform.machine() != "x86_64"
    ):
        raise ValueError("Native execution requires Linux x86_64 / Python 3.12")
    if prefix.exists():
        raise ValueError(
            "Build prefix already exists; select a fresh prefix and retain previous evidence"
        )
    archive = output / "source.tar"
    if BUILD.sha256(archive) != plan["source_archive_sha256"]:
        raise ValueError("Source archive changed after planning")
    filesystem = native_filesystem(prefix)
    native_filesystem(cache)
    prefix.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    source = prefix / "source"
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    for name, digest in plan["inputs"].items():
        if BUILD.sha256(source / name) != digest:
            raise ValueError(f"Archived input hash mismatch: {name}")
    BUILD.write_json(source / ".mpu-source.json", plan)
    env = build_env(prefix, cache, offline)
    result = {
        "schema_version": 1,
        "status": "running",
        "built": False,
        "plan": plan,
        "filesystem": filesystem,
        "steps": [],
        "gpu_executed": False,
        "locking_scope": plan["locking_scope"],
    }
    result_file = output / "native-build.json"

    def run(name, command, allowed=(0,), timeout=7200):
        start = time.monotonic()
        log = output / (name + ".log")
        item = {
            "name": name,
            "command": list(map(str, command)),
            "log": str(log),
            "status": "running",
        }
        result["steps"].append(item)
        BUILD.write_json(result_file, result)
        with log.open("w") as stream:
            proc = subprocess.run(
                item["command"],
                cwd=source,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        item.update(
            returncode=proc.returncode,
            seconds=time.monotonic() - start,
            status="passed" if proc.returncode == 0 else "nonzero",
        )
        BUILD.write_json(result_file, result)
        if proc.returncode not in allowed:
            raise RuntimeError(f"{name} failed ({proc.returncode}); see {log}")
        return log

    try:
        release = platform.freedesktop_os_release()
        if (release.get("ID"), release.get("VERSION_ID")) != ("ubuntu", "24.04"):
            raise ValueError(
                "This profile targets Ubuntu 24.04; validate a new profile for other OS versions"
            )
        result["host"] = {
            "os_release": release,
            "kernel": platform.release(),
            "python": platform.python_version(),
        }
        for name, command, version in (
            ("uv", ["uv", "--version"], "0.13.5"),
            ("rustc", ["rustc", "--version"], "1.99.1"),
            ("gcc", ["gcc-13", "-dumpfullversion"], "13.3.0"),
        ):
            log = run("version-" + name, command)
            if version not in log.read_text():
                raise ValueError(
                    f"Expected {name} {version}; received {log.read_text().strip()}"
                )
        result["os_packages"] = run("os-packages", ["dpkg-query", "-W"]).read_text()
        family = plan["family"]
        for kind in ("runtime", "toolchain"):
            venv = prefix / kind
            run("venv-" + kind, ["uv", "venv", "--python", sys.executable, str(venv)])
            text = (source / LOCKS / f"{family}-{kind}.txt").read_text()
            if kind == "runtime":
                text += "\n" + (source / LOCKS / "test-tools.txt").read_text()
            effective = prefix / (kind + "-effective.txt")
            effective.write_text(effective_lock(text, wheelhouse))
            command = [
                "uv",
                "pip",
                "sync",
                "--python",
                str(venv / "bin/python"),
                "--require-hashes",
                "--only-binary",
                ":all:",
                str(effective),
                "--index",
                "https://pypi.org/simple",
                "--default-index",
                "https://download.pytorch.org/whl/cu130",
                "--index-strategy",
                "unsafe-best-match",
            ]
            if wheelhouse:
                command += ["--find-links", str(Path(wheelhouse).resolve())]
            run("sync-" + kind, command)
        cuda = Path(env["CUDA_HOME"])
        for link, target in (
            (cuda / "lib64", "lib"),
            (cuda / "lib/libcudart.so", "libcudart.so.13"),
        ):
            if not link.exists():
                link.symlink_to(target)
        if not (cuda / "lib64/libcudart.so").is_file():
            raise ValueError("Compiler CUDA runtime library is absent")
        if family == "qwen":
            shutil.copyfile(
                source / LOCKS / "qwen-grpc-Cargo.lock",
                source / "rust/sglang-grpc/Cargo.lock",
            )
        cargo_locks = {
            str(p.relative_to(source)): BUILD.sha256(p)
            for p in (source / "rust").rglob("Cargo.lock")
        }
        for name in ("README.md", "LICENSE"):
            if (source / name).is_file() and not (source / "python" / name).exists():
                shutil.copyfile(source / name, source / "python" / name)
        python = prefix / "runtime/bin/python"
        wheels = prefix / "wheels"
        run(
            "build-wheel",
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                wheels,
                source / "python",
            ],
        )
        for relative, digest in cargo_locks.items():
            if BUILD.sha256(source / relative) != digest:
                raise ValueError(f"Cargo lock changed during build: {relative}")
        built = list(wheels.glob("*.whl"))
        if len(built) != 1:
            raise ValueError("Expected exactly one built server wheel")
        wheel = built[0]
        result["wheel"] = {**identity(wheel), **check_wheel(wheel, source)}
        result["cargo_locks"] = cargo_locks
        run(
            "install-wheel",
            ["uv", "pip", "install", "--no-deps", "--python", python, wheel],
        )
        run("dependency-check", ["uv", "pip", "check", "--python", python])
        result["pip_freeze"] = run(
            "pip-freeze", ["uv", "pip", "freeze", "--python", python]
        ).read_text()
        site = prefix / "runtime/lib/python3.12/site-packages"
        for index, relative in enumerate(result["wheel"]["packaged_extensions"]):
            extension = site / relative
            code = (
                "import importlib.util,sys,torch; p=sys.argv[1]; "
                "s=importlib.util.spec_from_file_location(p.rsplit('/',1)[-1].split('.')[0],p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(p)"
            )
            run(
                f"native-extension-{index}",
                [python, "-c", code, extension],
                timeout=120,
            )
            target = source / "python" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extension, target)
        acceptance = source / "benchmark/mpu_stages/acceptance.py"
        run(
            "acceptance-jit",
            [
                python,
                acceptance,
                "--through",
                "jit",
                "--jit-matrix",
                "boundaries",
                "--output-dir",
                prefix / "acceptance",
                "--execute",
            ],
        )
        run(
            "imports",
            [
                python,
                source / "benchmark/mpu_stages/hardware_probe.py",
                "--mode",
                "imports",
                "--output",
                prefix / "imports.json",
            ],
            allowed=(0, 2),
        )
        result["imports"] = json.loads((prefix / "imports.json").read_text())
        if result["imports"]["status"] == "failed":
            raise ValueError("Production import probe has a code/ABI failure")
        result.update(
            status="built_jit_verified",
            built=True,
            validation_scope="wheel, all declared Rust extensions, dependency consistency, portable tests and production JIT; no GPU/model execution",
        )
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        BUILD.write_json(result_file, result)
        BUILD.write_json(prefix / "native-build.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    prepare_parser.add_argument("--source-ref", required=True)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--prefix", required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--output-dir", required=True, type=Path)
    execute_parser.add_argument("--cache", required=True, type=Path)
    execute_parser.add_argument("--wheelhouse", type=Path)
    execute_parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.repo, args.source_ref, args.output_dir, args.prefix)
    else:
        result = execute(args.output_dir, args.cache, args.wheelhouse, args.offline)
    print(json.dumps({"status": result["status"], "built": result["built"]}))


if __name__ == "__main__":
    main()
