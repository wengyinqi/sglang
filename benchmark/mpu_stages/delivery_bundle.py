"""Create, verify and restore an offline MPU experiment delivery.

The runtime image revision and acceptance-tool revision are separate identities.
Reuse is allowed only when every change is in the explicit test/tool allowlist.
Images, native wheels, JIT libraries, recipes and model references travel together.
Model weights and GPU acceptance are intentionally not part of this CPU delivery.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ALLOWED = (
    "benchmark/mpu_stages/",
    "test/manual/mpu_stages/",
    ".github/workflows/mpu-cpu-control.yml",
    ".github/workflows/run-cpu-ci.yml",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def capture(command, **kwargs):
    return run(command, text=True, capture_output=True, **kwargs).stdout.strip()


def git(repo, *args):
    return capture(["git", "-C", repo, *args])


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or ":" in relative
        or str(path) != relative
    ):
        raise ValueError(f"Unsafe bundle path: {relative}")
    full = Path(root).joinpath(*path.parts)
    if not full.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    return full


def validate_tool_delta(changed):
    invalid = [
        p for p in changed if not any(p.startswith(prefix) for prefix in ALLOWED)
    ]
    if invalid:
        raise ValueError(f"Runtime changed; rebuild the native image first: {invalid}")
    # A recipe/lock change is not a test-only change, even under benchmark/.
    forbidden = (
        "benchmark/mpu_stages/stage.json",
        "benchmark/mpu_stages/environments/",
    )
    if any(any(p.startswith(prefix) for prefix in forbidden) for p in changed):
        raise ValueError(
            "Recipe/dependency identity changed; rebuild or review the runtime"
        )


def inspect_image(image_id):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("An immutable Docker image ID is required")
    details = json.loads(capture(["docker", "image", "inspect", image_id]))[0]
    if (
        details["Id"] != image_id
        or details["Os"] != "linux"
        or details["Architecture"] != "amd64"
    ):
        raise ValueError("Image identity/platform mismatch")
    return details


def save_image(image_id, path):
    # One archive per stage allows a small target host to restore one at a time.
    partial = path.with_suffix(path.suffix + ".partial")
    with (
        partial.open("wb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0
        ) as zipped,
    ):
        process = subprocess.Popen(
            ["docker", "image", "save", image_id], stdout=subprocess.PIPE
        )
        try:
            shutil.copyfileobj(process.stdout, zipped, length=1024 * 1024)
            if process.wait() != 0:
                raise RuntimeError("docker save failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    partial.rename(path)


def image_metadata(image, native, native_receipt_sha256):
    """Read native-image plans while keeping their original receipt intact."""
    if "plan" not in image:
        return image
    plan = image["plan"]
    smoke = image.get("smoke", {})
    if (
        plan.get("build_route") != "verified-native-runtime"
        or plan.get("native_receipt_sha256") != native_receipt_sha256
        or plan.get("source_commit") != native["plan"]["source_commit"]
        or plan.get("prefix") != native["plan"]["prefix"]
        or plan.get("native_extensions") != native["wheel"]["packaged_extensions"]
        or smoke.get("source_commit") != plan.get("source_commit")
        or smoke.get("native_extensions_loaded")
        != len(native["wheel"]["packaged_extensions"])
        or smoke.get("gpu_executed") is not False
    ):
        raise ValueError("Native image plan, receipt or container evidence mismatch")
    # Stage and wheel identity come from the hash-bound native receipt; they
    # are not guessed from an image tag or copied into the original evidence.
    return dict(
        image,
        source_commit=plan["source_commit"],
        stage=native["plan"]["stage"],
        native_receipt_sha256=plan["native_receipt_sha256"],
        wheel_sha256=native["wheel"]["sha256"],
        runtime_prefix=plan["prefix"],
        source_prefix=plan["prefix"],
        image_tag=plan["image_tag"],
    )


def stage_artifacts(repo, spec, root):
    image, native, jit = (
        read(spec[key]) for key in ("image_receipt", "native_receipt", "jit_receipt")
    )
    if (
        image.get("status") != "built_container_smoke_verified"
        or image.get("built") is not True
        or image.get("gpu_model_validated") is not False
    ):
        raise ValueError("A CPU-verified experimental image receipt is required")
    if native.get("status") != "built_jit_verified" or native.get("built") is not True:
        raise ValueError("A verified native receipt is required")
    image = image_metadata(image, native, sha(spec["native_receipt"]))
    runtime_commit = image["source_commit"]
    if (
        native["plan"]["source_commit"] != runtime_commit
        or image["stage"] != native["plan"]["stage"]
    ):
        raise ValueError("Image/native source identities differ")
    if image["native_receipt_sha256"] != sha(spec["native_receipt"]):
        raise ValueError("Native receipt changed after the image build")
    commit = git(repo, "rev-parse", "--verify", spec["source_ref"] + "^{commit}")
    changed = git(repo, "diff", "--name-only", runtime_commit, commit).splitlines()
    validate_tool_delta(changed)
    stage = json.loads(git(repo, "show", f"{commit}:benchmark/mpu_stages/stage.json"))
    if stage != image["stage"]:
        raise ValueError("Selected stage differs from the verified runtime")
    stage_id = stage["id"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", stage_id):
        raise ValueError("Invalid stage identifier")
    target = root / "stages" / stage_id
    target.mkdir(parents=True)
    details = inspect_image(image["image_id"])
    if (
        details["Config"]["Labels"].get("org.opencontainers.image.revision")
        != runtime_commit
    ):
        raise ValueError("Docker image revision disagrees with the receipt")
    write(target / "image-inspect.json", details)
    for key in ("image_receipt", "native_receipt", "jit_receipt"):
        shutil.copyfile(spec[key], target / (key.replace("_", "-") + ".json"))
    write(target / "stage.json", stage)
    run(
        [
            "git",
            "-C",
            repo,
            "archive",
            "--format=tar.gz",
            "--output",
            target / "source.tar.gz",
            commit,
        ]
    )
    wheel = Path(native["wheel"]["path"])
    if sha(wheel) != native["wheel"]["sha256"] or sha(wheel) != image["wheel_sha256"]:
        raise ValueError("Native wheel no longer matches the receipts")
    shutil.copyfile(wheel, target / wheel.name)
    # Preserve the actual libraries, not just paths into the old host's cache.
    libraries = []
    if jit.get("status") != "passed" or not jit.get("checks"):
        raise ValueError("A successful production JIT receipt is required")
    if (
        jit["source"]["archive_provenance"]["source_commit"] != runtime_commit
        or jit["stage"]["id"] != stage_id
    ):
        raise ValueError("JIT receipt belongs to a different runtime/stage")
    seen_cases = set()
    expected_cases = {case["id"]: case for case in jit["jit_cases"]}
    for check in jit["checks"]:
        evidence = check["evidence"]
        case = evidence["case"]["id"]
        if evidence["case"] != expected_cases.get(case):
            raise ValueError(
                "JIT case arguments differ from the planned specialization"
            )
        if (
            check["status"] != "passed"
            or not evidence.get("native_module_loaded")
            or evidence.get("gpu_executed") is not False
            or case in seen_cases
        ):
            raise ValueError("Incomplete/duplicate JIT acceptance evidence")
        seen_cases.add(case)
        for source_record in evidence["sources"]:
            source_path = Path(source_record["path"])
            relative = source_path.relative_to(
                Path(native["plan"]["prefix"]) / "source"
            ).as_posix()
            original = run(
                ["git", "-C", repo, "show", f"{runtime_commit}:{relative}"],
                capture_output=True,
            ).stdout
            if hashlib.sha256(original).hexdigest() != source_record["sha256"]:
                raise ValueError("JIT source hash differs from the compiled revision")
        if not evidence["shared_libraries"] or not evidence["exports"]:
            raise ValueError("Missing JIT library or export inventory")
        for record in evidence["shared_libraries"]:
            library = Path(record["path"])
            if (
                sha(library) != record["sha256"]
                or library.stat().st_size != record["bytes"]
            ):
                raise ValueError("JIT library changed or is missing")
            relative = f"jit/{record['sha256']}/{library.name}"
            dest = safe_path(target, relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(library, dest)
            libraries.append(
                {
                    "path": relative,
                    "case": case,
                    "sha256": record["sha256"],
                    "exports": evidence["exports"],
                }
            )
    if seen_cases != expected_cases.keys():
        raise ValueError("JIT receipt is missing planned cases")
    with tempfile.TemporaryDirectory(prefix="mpu-bundle-source-") as temp:
        source = Path(temp)
        with tarfile.open(target / "source.tar.gz") as archive:
            archive.extractall(source, filter="data")
        tools = source / "benchmark/mpu_stages"
        reference_path = (
            tools / "environments/model-references" / f"{stage['family']}.json"
        )
        reference = read(reference_path)
        if (
            reference["model_id"] != stage["default_model"]
            or reference["revision"] != stage["default_revision"]
            or reference["resolved_revision"] != stage["default_revision"]
        ):
            raise ValueError("Model reference differs from the selected checkpoint")
        shutil.copyfile(reference_path, target / "model-reference.json")
        module_spec = importlib.util.spec_from_file_location(
            "bundle_stage", tools / "stage.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        _, args = module.parse_args(
            [
                "--stage",
                str(target / "stage.json"),
                "--model",
                "/models/selected",
                "--revision",
                stage["default_revision"],
            ]
        )
        command = module.server_command(
            stage, args, stage["variant"], args.input_lens[0], args.capacity_ratios[0]
        )
        command[0] = image["runtime_prefix"] + "/runtime/bin/python"
        write(
            target / "launch.json",
            {
                "command": command,
                "environment": stage.get("server_environment", {}),
                "model_path_placeholder": "/models/selected",
                "executed": False,
                "no_pd": True,
            },
        )
    print(f"Exporting {stage_id}: {image['image_id']}", flush=True)
    save_image(image["image_id"], target / "image.tar.gz")
    inspect_image(image["image_id"])
    return {
        "id": stage_id,
        "directory": str(target.relative_to(root)).replace("\\", "/"),
        "image_id": image["image_id"],
        "image_tag": image["image_tag"],
        "runtime_commit": runtime_commit,
        "tools_commit": commit,
        "tool_changes": changed,
        "runtime_prefix": image["runtime_prefix"],
        "runtime_source_prefix": image["source_prefix"] + "/source",
        "wheel": wheel.name,
        "native_extensions": native["wheel"]["packaged_extensions"],
        "jit_libraries": libraries,
        "jit_cases": len(seen_cases),
        "model": {
            "id": stage["default_model"],
            "revision": stage["default_revision"],
            "weights_included": False,
        },
    }


def create(repo, specification, output):
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise ValueError("Create on Linux with Python 3.12+ and Docker")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = read(specification)
    if not isinstance(specs, list) or not specs:
        raise ValueError("Specify a nonempty array of stage inputs")
    result = {
        "schema_version": 1,
        "status": "incomplete",
        "stages": [],
        "gpu_executed": False,
        "model_weights_included": False,
    }
    write(output / "bundle.json", result)
    for item in specs:
        result["stages"].append(stage_artifacts(repo, item, output))
        write(output / "bundle.json", result)
    shutil.copyfile(__file__, output / "restore.py")
    shutil.copyfile(HERE / "DELIVERY_BUNDLE.md", output / "README.md")
    result["files"] = {
        p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "bundle.json"
    }
    result["status"] = "complete"
    write(output / "bundle.json", result)
    (output / "bundle.sha256").write_text(sha(output / "bundle.json") + "\n")
    return {
        "bundle": str(output),
        "manifest_sha256": sha(output / "bundle.json"),
        "stages": len(result["stages"]),
    }


def verify(root, expected_sha256=None):
    root = Path(root).resolve()
    # A separately retained digest detects replacement of the entire manifest.
    expected = expected_sha256 or (root / "bundle.sha256").read_text().strip()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected)
        or sha(root / "bundle.json") != expected
    ):
        raise ValueError("Bundle manifest checksum mismatch")
    manifest = read(root / "bundle.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or not manifest.get("stages")
    ):
        raise ValueError("Incomplete/unsupported delivery bundle")
    files = manifest.get("files", {})
    required = {"restore.py"}
    seen = set()
    for stage in manifest["stages"]:
        if stage["id"] in seen:
            raise ValueError("Duplicate delivery stage")
        seen.add(stage["id"])
        base = stage["directory"]
        required.update(
            f"{base}/{name}"
            for name in (
                "source.tar.gz",
                "image.tar.gz",
                "stage.json",
                "model-reference.json",
                "launch.json",
                "image-receipt.json",
                "native-receipt.json",
                "jit-receipt.json",
                "image-inspect.json",
                stage["wheel"],
            )
        )
        if not stage.get("jit_libraries"):
            raise ValueError("JIT library inventory is required")
        required.update(f"{base}/{item['path']}" for item in stage["jit_libraries"])
    if not required.issubset(files):
        raise ValueError(
            f"Missing required artifacts: {sorted(required - files.keys())}"
        )
    for relative, identity in files.items():
        path = safe_path(root, relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or sha(path) != identity["sha256"]
        ):
            raise ValueError(f"Artifact checksum/size mismatch: {relative}")
    return manifest


def inside(root, stage_id, output):
    """Run only in the restored container; root is the read-only delivery mount."""
    import torch  # Establish native libtorch dependencies before extension loading.
    import tvm_ffi

    manifest = read(root / "bundle.json")
    stage = next(s for s in manifest["stages"] if s["id"] == stage_id)
    folder = safe_path(root, stage["directory"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "failed",
        "gpu_executed": False,
        "stage": stage_id,
        "runtime_commit": stage["runtime_commit"],
        "tools_commit": stage["tools_commit"],
    }
    try:
        source = Path("/tmp/mpu-restored-source")
        source.mkdir(exist_ok=False)
        with tarfile.open(folder / "source.tar.gz") as archive:
            archive.extractall(source, filter="data")
        # The source snapshot contains newer tests, but the production tree must
        # still match the already compiled runtime byte for byte.
        count = 0
        for base in ("python", "sgl-kernel"):
            for path in (source / base).rglob("*"):
                if path.is_file():
                    original = Path(stage["runtime_source_prefix"]) / path.relative_to(
                        source
                    )
                    if not original.is_file() or sha(path) != sha(original):
                        raise ValueError(
                            f"Runtime source differs: {path.relative_to(source)}"
                        )
                    count += 1
        receipt["production_source_files_verified"] = count
        write(
            source / ".mpu-source.json",
            {
                "source_commit": stage["tools_commit"],
                "runtime_commit": stage["runtime_commit"],
            },
        )
        site = Path(stage["runtime_prefix"]) / "runtime/lib/python3.12/site-packages"
        installed = 0
        with zipfile.ZipFile(folder / stage["wheel"]) as wheel:
            for name in wheel.namelist():
                if name.endswith((".py", ".so")):
                    if (
                        sha(safe_path(site, name))
                        != hashlib.sha256(wheel.read(name)).hexdigest()
                    ):
                        raise ValueError(f"Installed wheel file differs: {name}")
                    installed += 1
        for relative in stage["native_extensions"]:
            path = safe_path(site, relative)
            spec = importlib.util.spec_from_file_location(path.name.split(".")[0], path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        loaded = []
        for item in stage["jit_libraries"]:
            path = safe_path(folder, item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("Relocated JIT library changed")
            module = tvm_ffi.load_module(str(path))
            if not all(
                callable(getattr(module, name, None)) for name in item["exports"]
            ):
                raise ValueError("Relocated JIT library is missing exports")
            loaded.append(item["case"])
        receipt.update(
            installed_files_verified=installed,
            native_extensions_loaded=len(stage["native_extensions"]),
            relocated_jit_cases=sorted(set(loaded)),
            torch_version=torch.__version__,
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(source / "python"),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (output / "portable.log").open("w") as log:
            run(
                [
                    sys.executable,
                    source / "benchmark/mpu_stages/acceptance.py",
                    "--through",
                    "portable",
                    "--output-dir",
                    output / "portable",
                    "--execute",
                ],
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result = read(output / "portable/acceptance.json")
        if (
            result["status"] != "passed"
            or not result["evidence_scope"]["linux_cpu_control_executed"]
        ):
            raise ValueError("Restored Linux CPU acceptance did not pass")
        receipt.update(
            status="passed", linux_cpu_control_executed=True, portable="passed"
        )
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        write(output / "restore-container.json", receipt)


def restore(root, output, selected, expected_sha256=None):
    if sys.platform != "linux":
        raise ValueError("Restore requires Linux and Docker")
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = verify(root, expected_sha256)
    stages = (
        manifest["stages"]
        if not selected
        else [s for s in manifest["stages"] if s["id"] in selected]
    )
    if not stages or selected and set(selected) != {s["id"] for s in stages}:
        raise ValueError("Unknown or empty stage selection")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for stage in stages:
        folder = safe_path(root, stage["directory"])
        target = output / stage["id"]
        target.mkdir()
        with (target / "docker-load.log").open("w") as log:
            run(
                ["docker", "image", "load", "--input", folder / "image.tar.gz"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        actual = inspect_image(stage["image_id"])
        if actual["RootFS"] != read(folder / "image-inspect.json")["RootFS"]:
            raise ValueError("Restored layer identity differs")
        write(target / "image-inspect.json", actual)
        # Only delivery artifacts and a new evidence directory are mounted.
        # No original venv, CUDA cache, model directory or build checkout is used.
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={root},dst=/bundle,readonly",
            "--mount",
            f"type=bind,src={target},dst=/evidence",
            "--entrypoint",
            stage["runtime_prefix"] + "/runtime/bin/python",
            stage["image_id"],
            "/bundle/restore.py",
            "_container",
            "--bundle",
            "/bundle",
            "--stage",
            stage["id"],
            "--output",
            "/evidence",
        ]
        write(target / "container-command.json", command)
        with (target / "container.log").open("w") as log:
            run(command, stdout=log, stderr=subprocess.STDOUT)
        results.append(read(target / "restore-container.json"))
        write(
            output / "restore.json",
            {"status": "running", "results": results, "gpu_executed": False},
        )
    report = {
        "status": "passed",
        "results": results,
        "gpu_executed": False,
        "manifest_sha256": sha(root / "bundle.json"),
    }
    write(output / "restore.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--repo", type=Path, required=True)
    make.add_argument("--spec", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    for name in ("verify", "restore", "_container"):
        p = commands.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        if name != "_container":
            p.add_argument("--expected-sha256")
        if name != "verify":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument(
                "--stage",
                action="append" if name == "restore" else "store",
                required=name == "_container",
            )
    args = parser.parse_args()
    if args.command == "create":
        result = create(args.repo, args.spec, args.output)
    elif args.command == "verify":
        manifest = verify(args.bundle, args.expected_sha256)
        result = {"status": "passed", "stages": len(manifest["stages"])}
    elif args.command == "restore":
        result = restore(args.bundle, args.output, args.stage, args.expected_sha256)
    else:
        inside(args.bundle, args.stage, args.output)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
