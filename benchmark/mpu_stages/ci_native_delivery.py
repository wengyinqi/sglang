"""Build and independently restore one source-pinned stage on a hosted Linux runner.

No GPU or model weights are used. The original build prefix and image are
removed only after their offline bundle is verified, before the restore test.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = (
    "01-qwen-quest-1gpu-0mpu",
    "02-qwen-quest-1gpu-1mpu",
    "03-deepseek-v4-flash-4gpu-0mpu",
    "04-deepseek-v4-flash-4gpu-4mpu",
    "05-glm-5.3-flash-4gpu-0mpu",
    "06-glm-5.3-flash-4gpu-4mpu",
)
COMPAT_SHA256 = "24ac8323d11760e8d080a2d3eb3ba418987353beadedbd08121a1f4d72b8f620"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def command(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def capture(*args):
    return command(*args, capture_output=True, text=True).stdout.strip()


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_checkout(repo, stage, expected):
    if stage not in STAGES or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("A registered stage and complete commit SHA are required")
    actual = capture("git", "-C", repo, "rev-parse", "HEAD")
    recipe = read(Path(repo) / "benchmark/mpu_stages/stage.json")
    if actual != expected or recipe["id"] != stage:
        raise ValueError("Checkout commit or stage differs from the planned matrix")


def remove_owned(path, allowed):
    """Delete only a concrete directory created by this single CI stage."""
    path, allowed = Path(path), Path(allowed)
    if path != allowed or path.resolve() != allowed or path.is_symlink():
        raise ValueError("Cleanup target differs from the owned CI directory")
    if not str(path).startswith("/opt/mpu/ci-") or len(path.parts) != 4:
        raise ValueError("Cleanup target is outside a dedicated CI prefix")
    shutil.rmtree(path)


def retain_evidence(prefix, output):
    for name in ("acceptance", "imports.artifacts"):
        for path in (prefix / name).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".log", ".xml", ".cu", ".o"}:
                target = output / "native-evidence" / path.relative_to(prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    shutil.copy2(prefix / "imports.json", output / "native-evidence/imports.json")


def execute(repo, stage, commit, output, compat):
    validate_checkout(repo, stage, commit)
    if sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise ValueError("Execute only in the dedicated GitHub Linux workflow")
    if os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise ValueError(
            "This disposable build/cleanup profile requires a hosted runner"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sid = stage[:2]
    prefix = Path("/opt/mpu/ci-stage" + sid)
    cache = Path("/opt/mpu/ci-cache" + sid)
    restore_data = Path("/opt/mpu/ci-restore" + sid)
    if any(path.exists() for path in (prefix, cache, restore_data)):
        raise ValueError("CI prefixes must be fresh")
    native, image, bundle = (
        module(name)
        for name in ("native_environment", "runtime_image", "delivery_bundle")
    )
    if bundle.sha(compat) != COMPAT_SHA256:
        raise ValueError("CUDA userspace compatibility package checksum mismatch")
    result = dict(
        status="running",
        stage=stage,
        source_commit=commit,
        gpu_executed=False,
        model_executed=False,
        runner=os.environ.get("ImageVersion"),
        steps=[],
    )
    write(output / "delivery-result.json", result)
    restore_process = None
    restore_log = None
    previous_host = os.environ.pop("DOCKER_HOST", None)
    pidfile = Path("/run/mpu-ci-restore-" + sid + ".pid")

    def progress(name):
        result["steps"].append(name)
        write(output / "delivery-result.json", result)
        print("MPU DELIVERY", stage, name, flush=True)

    try:
        progress("native-build-start")
        native.prepare(repo, commit, output / "native", str(prefix))
        built = native.execute(output / "native", cache)
        if built["imports"]["status"] != "passed":
            raise ValueError("Production imports must pass before delivery")
        progress("native-build-passed")
        retain_evidence(prefix, output)
        context = output / "base"
        context.mkdir()
        shutil.copy2(compat, context / "cuda-compat.deb")
        (context / "Dockerfile").write_text("""FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev gcc-13 g++-13 pkg-config libssl-dev protobuf-compiler git ninja-build libnuma1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY cuda-compat.deb /tmp/cuda-compat.deb
RUN dpkg -i /tmp/cuda-compat.deb && rm /tmp/cuda-compat.deb
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
""")
        base_tag = "local/mpu-ci-base:" + commit
        with (output / "base-build.log").open("w") as log:
            command(
                "docker",
                "build",
                "--tag",
                base_tag,
                context,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        base_id = capture("docker", "image", "inspect", "--format", "{{.Id}}", base_tag)
        write(
            output / "base-inspect.json",
            json.loads(capture("docker", "image", "inspect", base_id)),
        )
        (context / "cuda-compat.deb").unlink()
        progress("runtime-image-start")
        image.prepare(
            output / "native/native-build.json",
            output / "image",
            base_id,
            "local/mpu-ci:" + stage + "-" + commit[:12],
        )
        image_result = image.execute(output / "image")
        if image_result["status"] != "built_container_smoke_verified":
            raise ValueError("Runtime image smoke failed")
        (output / "image/environment.tar").unlink()
        image_id = image_result["image_id"]
        with (output / "server-cli.log").open("w") as log:
            command(
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                image_id,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if "--model-path" not in (output / "server-cli.log").read_text():
            raise ValueError("Default server CLI help did not initialize")
        result["server_cli_help_passed"] = True
        specification = output / "bundle-inputs.json"
        write(
            specification,
            [
                dict(
                    source_ref=commit,
                    image_receipt=str(output / "image/image-build.json"),
                    native_receipt=str(output / "native/native-build.json"),
                    jit_receipt=str(prefix / "acceptance/jit.json"),
                )
            ],
        )
        progress("offline-export-start")
        bundle.create(repo, specification, output / "bundle")
        bundle.verify(output / "bundle")
        digest = bundle.sha(output / "bundle/bundle.json")
        # Original environments and build image are no longer available during
        # restore. Only the separately verified offline bundle is used.
        remove_owned(prefix, prefix)
        remove_owned(cache, cache)
        command("docker", "image", "rm", image_id)
        if (
            subprocess.run(
                ["docker", "image", "inspect", image_id], capture_output=True
            ).returncode
            == 0
        ):
            raise ValueError("Original build image still exists")
        result.update(
            original_build_prefix_removed=True, original_build_image_removed=True
        )
        progress("independent-restore-start")
        socket = "unix:///run/mpu-ci-restore-" + sid + ".sock"
        config = output / "restore-daemon-config.json"
        write(config, {})
        restore_log = (output / "restore-daemon.log").open("w")
        restore_process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "dockerd",
                "--config-file",
                str(config),
                "--host",
                socket,
                "--data-root",
                str(restore_data),
                "--exec-root",
                "/run/mpu-ci-restore-" + sid,
                "--pidfile",
                str(pidfile),
                "--storage-driver=overlay2",
                "--group=docker",
                "--iptables=false",
                "--ip6tables=false",
                "--bridge=none",
                "--ip-forward=false",
                "--ip-masq=false",
            ],
            stdout=restore_log,
            stderr=subprocess.STDOUT,
        )
        os.environ["DOCKER_HOST"] = socket
        for _ in range(60):
            if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
                break
            if restore_process.poll() is not None:
                raise RuntimeError("Independent Docker daemon exited")
            time.sleep(1)
        else:
            raise RuntimeError("Independent Docker daemon readiness timeout")
        if (
            subprocess.run(
                ["docker", "image", "inspect", image_id], capture_output=True
            ).returncode
            == 0
        ):
            raise ValueError("Independent image store was not empty")
        restored = bundle.restore(
            output / "bundle", output / "restored", [stage], digest
        )
        if restored["status"] != "passed" or len(restored["results"]) != 1:
            raise ValueError("Offline restore did not pass")
        result.update(
            status="passed",
            image_id=image_id,
            manifest_sha256=digest,
            separate_image_store=True,
            restoration=restored,
            cuda_compat_deb_sha256=COMPAT_SHA256,
        )
        progress("independent-restore-passed")
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if (
            restore_process is not None
            and restore_process.poll() is None
            and pidfile.exists()
        ):
            pid = int(pidfile.read_text().strip())
            args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if str(restore_data).encode() not in args:
                raise ValueError("Refuse to stop a daemon outside this CI stage")
            command("sudo", "-n", "kill", "-TERM", str(pid))
            restore_process.wait(timeout=30)
        if restore_log is not None:
            restore_log.close()
        if previous_host is None:
            os.environ.pop("DOCKER_HOST", None)
        else:
            os.environ["DOCKER_HOST"] = previous_host
        write(output / "delivery-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compat-deb", type=Path)
    args = parser.parse_args()
    if not all((args.stage, args.expected_sha, args.output, args.compat_deb)):
        parser.error("stage, expected-sha, output and compat-deb are required")
    print(
        json.dumps(
            execute(
                args.repo, args.stage, args.expected_sha, args.output, args.compat_deb
            )
        )
    )


if __name__ == "__main__":
    main()
