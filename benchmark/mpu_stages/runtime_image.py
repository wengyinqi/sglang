"""Package a verified native build as an experimental Docker server image.

No package downloads occur in docker build. An immutable local Ubuntu 24.04
base image ID is required, with Python 3.12 and the native build's OS libraries.
This is separate from the upstream Dockerfile's BUILD_TYPE=all route.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "mpu_native_environment", HERE / "native_environment.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)
BUILD = NATIVE.BUILD


def validate_native(result):
    if (
        result.get("status") != "built_jit_verified"
        or result.get("built") is not True
        or result.get("gpu_executed") is not False
    ):
        raise ValueError(
            "A successful native wheel/JIT receipt is required before image packaging"
        )
    wheel = result["wheel"]
    if BUILD.sha256(wheel["path"]) != wheel["sha256"]:
        raise ValueError("Native wheel changed since validation")
    return Path(NATIVE.linux_prefix(result["plan"]["prefix"]))


def verify_base_reference(reference, expected):
    details = json.loads(
        subprocess.check_output(["docker", "image", "inspect", reference], text=True)
    )
    if len(details) != 1 or details[0].get("Id") != expected:
        raise ValueError("Local base reference no longer identifies the pinned image")
    return details[0]


def runtime_config(prefix, commit, base):
    environment = dict(item.split("=", 1) for item in base.get("Env") or [])
    return {
        "Env": {
            "PATH": prefix
            + "/runtime/bin:"
            + environment.get(
                "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "PYTHONPATH": prefix + "/source/python",
            "CUDA_HOME": prefix + "/toolchain/lib/python3.12/site-packages/nvidia/cu13",
            "CXX": "/usr/bin/g++-13",
            "MAX_JOBS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "Labels": {
            "org.opencontainers.image.revision": commit,
            "ai.sglang.mpu.build_route": "verified-native-runtime",
        },
        "WorkingDir": prefix + "/source",
        "Entrypoint": [prefix + "/runtime/bin/python", "-m", "sglang.launch_server"],
        "Cmd": ["--help"],
    }


def verify_runtime_config(expected, actual):
    for key in ("WorkingDir", "Entrypoint", "Cmd"):
        if actual.get(key) != expected[key]:
            raise ValueError(
                f"Runtime image {key} differs from the prepared configuration"
            )
    environment = dict(item.split("=", 1) for item in actual.get("Env") or [])
    for key, value in expected["Env"].items():
        if environment.get(key) != value:
            raise ValueError(f"Runtime image environment differs: {key}")
    for key, value in expected["Labels"].items():
        if (actual.get("Labels") or {}).get(key) != value:
            raise ValueError(f"Runtime image label differs: {key}")


def build_single_commit(plan, output):
    """Avoid VFS copying the entire rootfs for each Dockerfile metadata step."""
    container = None
    with (output / "docker-build.log").open("w") as log:
        try:
            created = subprocess.check_output(
                [
                    "docker",
                    "create",
                    "--name",
                    "mpu-image-" + uuid.uuid4().hex,
                    "--network",
                    "none",
                    "--entrypoint",
                    "/bin/true",
                    plan["base_reference"],
                ],
                text=True,
                stderr=log,
            ).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", created):
                raise ValueError("Docker create returned no valid container identity")
            container = created
            with (output / "environment.tar").open("rb") as archive:
                subprocess.run(
                    ["docker", "cp", "-", container + ":/"],
                    stdin=archive,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            changes = []
            config = plan["runtime_config"]
            for instruction, key in (("LABEL", "Labels"), ("ENV", "Env")):
                for name, value in config[key].items():
                    changes.extend(
                        ["--change", f"{instruction} {name}={json.dumps(value)}"]
                    )
            for instruction, key in (
                ("WORKDIR", "WorkingDir"),
                ("ENTRYPOINT", "Entrypoint"),
                ("CMD", "Cmd"),
            ):
                value = config[key]
                changes.extend(
                    [
                        "--change",
                        instruction
                        + " "
                        + (json.dumps(value) if isinstance(value, list) else value),
                    ]
                )
            subprocess.run(
                ["docker", "commit", *changes, container, plan["image_tag"]],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        finally:
            if container:
                subprocess.run(
                    ["docker", "rm", "--force", container],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )


def prepare(native_result, output, base_image_id, tag):
    if sys.platform != "linux":
        raise ValueError("Package Linux environments on the Linux build host")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_image_id):
        raise ValueError("Base must be a verified local immutable Docker image ID")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Image evidence directory must be new or empty")
    result = json.loads(Path(native_result).read_text())
    prefix = validate_native(result)
    # Keep venv shebangs, Python origin paths and extension lookup paths intact.
    details = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", base_image_id], text=True
        )
    )[0]
    if (
        details["Id"] != base_image_id
        or details.get("Architecture") != "amd64"
        or details.get("Os") != "linux"
    ):
        raise ValueError("Wrong image ID or platform")
    # BuildKit interprets FROM sha256:<id> as a registry repository. Use a
    # dedicated local tag, check its ID before/after building, and verify the
    # filesystem layer prefix against the immutable base.
    base_reference = "sglang-mpu-verified-base:" + base_image_id.split(":", 1)[1]
    subprocess.run(["docker", "tag", base_image_id, base_reference], check=True)
    verify_base_reference(base_reference, base_image_id)
    output.mkdir(parents=True, exist_ok=True)
    BUILD.write_json(output / "base-inspect.json", details)
    archive = output / "environment.tar"
    files = []

    def select(member):
        relative = Path(member.name)
        source_build = str(prefix / "source/python/build").lstrip("/")
        if (
            member.name == source_build
            or member.name.startswith(source_build + "/")
            or any(
                p in ("__pycache__", ".git") or p.endswith(".egg-info")
                for p in relative.parts
            )
        ):
            return None
        if member.isfile():
            files.append(NATIVE.identity(Path("/") / member.name))
        return member

    with tarfile.open(archive, "w") as stream:
        for name in ("runtime", "toolchain", "source", "native-build.json"):
            path = prefix / name
            stream.add(path, arcname=str(path).lstrip("/"), filter=select)
    BUILD.write_json(output / "files.json", files)
    tag = BUILD.image_reference(tag)
    if "@" in tag:
        raise ValueError("Output tag cannot be a digest reference")
    dockerfile = f"""FROM {base_reference}
LABEL org.opencontainers.image.revision="{result['plan']['source_commit']}"
LABEL ai.sglang.mpu.build_route="verified-native-runtime"
ADD environment.tar /
ENV PATH="{prefix}/runtime/bin:$PATH" \\
    PYTHONPATH="{prefix}/source/python" \\
    CUDA_HOME="{prefix}/toolchain/lib/python3.12/site-packages/nvidia/cu13" \\
    CXX="/usr/bin/g++-13" MAX_JOBS="1" PYTHONDONTWRITEBYTECODE="1"
WORKDIR {prefix}/source
ENTRYPOINT ["{prefix}/runtime/bin/python", "-m", "sglang.launch_server"]
CMD ["--help"]
"""
    (output / "Dockerfile").write_text(dockerfile)
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "built": False,
        "build_route": "verified-native-runtime",
        "base_image_id": base_image_id,
        "base_reference": base_reference,
        "base_rootfs_layers": details["RootFS"]["Layers"],
        "source_commit": result["plan"]["source_commit"],
        "prefix": str(prefix),
        "environment_archive": NATIVE.identity(archive),
        "dockerfile_sha256": BUILD.sha256(output / "Dockerfile"),
        "native_receipt_sha256": BUILD.sha256(native_result),
        "image_tag": tag,
        "native_extensions": result["wheel"]["packaged_extensions"],
        "all_dependencies_locked": False,
        "gpu_model_validated": False,
        "packaging_strategy": (
            "single_commit"
            if subprocess.check_output(
                ["docker", "info", "--format", "{{.Driver}}"], text=True
            ).strip()
            == "vfs"
            else "dockerfile"
        ),
        "runtime_config": runtime_config(
            str(prefix), result["plan"]["source_commit"], details.get("Config") or {}
        ),
    }
    BUILD.write_json(output / "image-plan.json", plan)
    return plan


def execute(output):
    output = Path(output).resolve()
    plan = json.loads((output / "image-plan.json").read_text())
    archive = plan["environment_archive"]
    if (
        BUILD.sha256(output / "environment.tar") != archive["sha256"]
        or BUILD.sha256(output / "Dockerfile") != plan["dockerfile_sha256"]
    ):
        raise ValueError("Image inputs changed after preparation")
    result = {
        "schema_version": 1,
        "status": "building",
        "built": False,
        "plan": plan,
        "gpu_model_validated": False,
    }
    try:
        base = verify_base_reference(plan["base_reference"], plan["base_image_id"])
        if "runtime_config" in plan and plan["runtime_config"] != runtime_config(
            plan["prefix"], plan["source_commit"], base.get("Config") or {}
        ):
            raise ValueError(
                "Prepared runtime configuration differs from the pinned inputs"
            )
        command = [
            "docker",
            "build",
            "--pull=false",
            "--network",
            "none",
            "--tag",
            plan["image_tag"],
            str(output),
        ]
        if plan.get("packaging_strategy") == "single_commit":
            build_single_commit(plan, output)
        else:
            with (output / "docker-build.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        details = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", plan["image_tag"]], text=True
            )
        )[0]
        image_id = details["Id"]
        if "runtime_config" in plan:
            verify_runtime_config(plan["runtime_config"], details["Config"])
        verify_base_reference(plan["base_reference"], plan["base_image_id"])
        layers = plan["base_rootfs_layers"]
        if details["RootFS"]["Layers"][: len(layers)] != layers:
            raise ValueError(
                "Built image filesystem does not derive from the pinned base"
            )
        BUILD.write_json(output / "image-inspect.json", details)
        # Exercise the actual container, all native extensions and package
        # metadata. No --gpus and no model are involved in this smoke check.
        code = """import importlib.util,json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
marker=json.loads((root/'source/.mpu-source.json').read_text())
assert marker['source_commit']==expected
receipt=json.loads((root/'native-build.json').read_text())
for relative in receipt['wheel']['packaged_extensions']:
    path=root/'runtime/lib/python3.12/site-packages'/relative
    spec=importlib.util.spec_from_file_location(path.name.split('.')[0],path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print(json.dumps({'source_commit':expected,'native_extensions_loaded':len(receipt['wheel']['packaged_extensions']),
                  'cuda_available':torch.cuda.is_available(),'gpu_executed':False}))
"""
        process = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                plan["prefix"] + "/runtime/bin/python",
                image_id,
                "-c",
                code,
                plan["prefix"],
                plan["source_commit"],
            ],
            capture_output=True,
            text=True,
        )
        (output / "container-smoke.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(
                "Container native smoke failed; inspect container-smoke.log"
            )
        smoke = json.loads(process.stdout)
        if smoke["native_extensions_loaded"] != len(plan["native_extensions"]):
            raise ValueError("Container extension coverage mismatch")
        result.update(
            status="built_container_smoke_verified",
            built=True,
            image_id=image_id,
            smoke=smoke,
            validation_scope="actual container startup, source identity and native extension load; server/GPU execution still requires NVIDIA hardware",
        )
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        BUILD.write_json(output / "image-build.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-result", type=Path)
    parser.add_argument("--base-image-id")
    parser.add_argument("--image-tag")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.native_result:
        if not args.base_image_id or not args.image_tag:
            parser.error("Preparation requires --base-image-id and --image-tag")
        result = prepare(
            args.native_result, args.output_dir, args.base_image_id, args.image_tag
        )
    if args.execute:
        result = execute(args.output_dir)
    elif not args.native_result:
        parser.error(
            "Use --native-result to prepare or --execute to build an existing plan"
        )
    print(json.dumps({"status": result["status"], "built": result["built"]}))


if __name__ == "__main__":
    main()
