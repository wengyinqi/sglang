"""Generate a source-pinned Linux Docker recipe; only --execute starts Docker.

Uses only Python's standard library and Git on the planning host. The tar context
preserves Git modes/symlinks and never copies a live working directory.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE_PATH = "benchmark/mpu_stages/stage.json"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    output = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return output if args[0] == "show" else output.strip()


def recipe_for(family):
    for path in sorted((HERE / "environments").glob("*-cu130.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        if family in recipe["families"]:
            return recipe, path
    raise ValueError(f"Unsupported stage family: {family}")


def dependency_pins(pyproject, expected):
    # The relevant requirements are exact pins in both source generations. Do
    # not import packaging or require Python 3.11's tomllib on the Windows host.
    found = {}
    for name, version in re.findall(
        r'["\']([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^"\' ;]+)', pyproject
    ):
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in expected:
            found.setdefault(normalized, set()).add(version)
    for name, wanted in expected.items():
        if found.get(name) != {wanted}:
            raise ValueError(
                f"Pinned source pyproject mismatch for {name}: {found.get(name)} != {wanted}"
            )
    return {name: next(iter(found[name])) for name in expected}


def image_reference(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*", value):
        raise ValueError(f"Invalid image reference: {value!r}")
    return value


def generate_dockerfile(
    original, recipe, commit, archive_sha, base_image=None, uv_image=None
):
    original = original.replace("\r\n", "\n")
    if (
        "FROM scratch AS local_src" not in original
        or "FROM framework AS framework_final" not in original
    ):
        raise ValueError(
            "Source Dockerfile does not expose the supported local-source mechanism"
        )
    # Make the source selection unconditional as well as passing local build
    # arguments. No default or caller override can select an official release.
    start = original.find('RUN if [ "$BRANCH_TYPE" = "local" ]; then')
    end = original.find("    && rm -rf /tmp/local_src", start)
    if start < 0 or end < 0:
        raise ValueError(
            "Source Dockerfile local-copy block changed; review before building"
        )
    end += len("    && rm -rf /tmp/local_src")
    dockerfile = (
        original[:start]
        + "RUN cp -r /tmp/local_src /sgl-workspace/sglang && rm -rf /tmp/local_src"
        + original[end:]
    )
    dockerfile = dockerfile.replace("ARG BRANCH_TYPE=remote", "ARG BRANCH_TYPE=local")
    if base_image:
        image_reference(base_image)
        dockerfile, count = re.subn(
            r"(?m)^FROM nvidia/cuda:\$\{CUDA_VERSION\}-cudnn-devel-ubuntu24\.04 AS (base|runtime)$",
            lambda match: f"FROM {base_image} AS {match.group(1)}",
            dockerfile,
        )
        if count != 2:
            raise ValueError("Expected two NVIDIA base image declarations")
    if uv_image:
        image_reference(uv_image)
        dockerfile, count = re.subn(
            r"(?m)^FROM ghcr\.io/astral-sh/uv:\$\{UV_VERSION\} AS uv$",
            f"FROM {uv_image} AS uv",
            dockerfile,
        )
        if count != 1:
            raise ValueError("This source Dockerfile has no configurable uv stage")
    dockerfile += f"""

# MPU experiment provenance. Package install and runtime both use the archive.
FROM {recipe["target"]} AS mpu_experiment
LABEL org.opencontainers.image.revision="{commit}"
LABEL ai.sglang.mpu.source_archive_sha256="{archive_sha}"
ENV PYTHONPATH=/sgl-workspace/sglang/python
COPY _mpu_build/build-input.json /opt/mpu/build-input.json
COPY _mpu_build/build-input.json /sgl-workspace/sglang/.mpu-source.json
COPY _mpu_build/capture_environment.py /opt/mpu/capture_environment.py
RUN python3 /opt/mpu/capture_environment.py --expected /opt/mpu/build-input.json --output-dir /opt/mpu
WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
"""
    return dockerfile


def external_images(dockerfile, cuda_version):
    images = []
    for line in dockerfile.splitlines():
        match = re.match(r"FROM (\S+)", line)
        if match and ("/" in match.group(1) or ":" in match.group(1)):
            ref = (
                match.group(1)
                .replace("${CUDA_VERSION}", cuda_version)
                .replace("${UV_VERSION}", "latest")
            )
            images.append(
                {
                    "reference": ref,
                    "digest_pinned": bool(re.search(r"@sha256:[0-9a-f]{64}$", ref)),
                    "registry_verified": False,
                }
            )
    return list({item["reference"]: item for item in images}.values())


def make_context(source_archive, output, additions):
    # Streaming copy avoids extracting Git symlinks (which require extra Windows
    # privileges) and faithfully retains executable bits on Linux.
    with tarfile.open(source_archive, "r") as source, tarfile.open(
        output, "w"
    ) as target:
        for member in source:
            if member.name == "_mpu_build" or member.name.startswith("_mpu_build/"):
                raise ValueError("Reserved _mpu_build path exists in source archive")
            target.addfile(
                member, source.extractfile(member) if member.isfile() else None
            )
        for name, content in additions.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            item.mode = 0o644
            item.mtime = 0
            target.addfile(item, io.BytesIO(content))


def prepare_environment(
    repo,
    stage_path,
    source_ref,
    output_dir,
    image_tag=None,
    base_image=None,
    uv_image=None,
):
    repo, output_dir = Path(repo).resolve(), Path(output_dir).resolve()
    commit = git(repo, "rev-parse", "--verify", f"{source_ref}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Source ref did not resolve to a commit")
    pinned_stage = json.loads(git(repo, "show", f"{commit}:{STAGE_PATH}"))
    stage = (
        pinned_stage
        if stage_path is None
        else json.loads(Path(stage_path).read_text(encoding="utf-8"))
    )
    if stage != pinned_stage:
        raise ValueError(
            "Selected stage differs from the stage in the pinned source commit"
        )
    recipe, recipe_path = recipe_for(stage["family"])
    pyproject = git(repo, "show", f"{commit}:python/pyproject.toml")
    pins = dependency_pins(pyproject, recipe["expected_packages"])
    original = git(repo, "show", f"{commit}:{recipe['dockerfile']}")
    status = git(repo, "status", "--porcelain", "--untracked-files=normal")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Output directory must be new or empty; existing evidence is never overwritten"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "source.tar"
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
    archive_sha = sha256(archive)
    dockerfile = generate_dockerfile(
        original, recipe, commit, archive_sha, base_image, uv_image
    )
    capture_path = HERE / "environments" / "capture_environment.py"
    tag = image_reference(image_tag or f"sglang-mpu:{stage['id']}-{commit[:12]}")
    if "@" in tag:
        raise ValueError("The output image tag cannot be a digest reference")
    build_args = {
        "BRANCH_TYPE": "local",
        "USE_LATEST_SGLANG": "0",
        "CUDA_VERSION": recipe["cuda_version"],
        "BUILD_TYPE": recipe["build_type"],
        "SGL_KERNEL_VERSION": pins["sglang-kernel"],
        "FLASHINFER_VERSION": pins["flashinfer-python"],
    }
    command = [
        "docker",
        "build",
        "--platform",
        recipe["platform"],
        "--target",
        "mpu_experiment",
        "--file",
        "_mpu_build/Dockerfile",
        "--tag",
        tag,
    ]
    for name, value in build_args.items():
        command += ["--build-arg", f"{name}={value}"]
    command += ["-"]
    build_input = {
        "source_commit": commit,
        "source_archive_sha256": archive_sha,
        "provenance": "git_archive",
        "expected_packages": pins,
        "stage": stage,
        "recipe_id": recipe["id"],
    }
    additions = {
        "_mpu_build/Dockerfile": dockerfile.encode(),
        "_mpu_build/build-input.json": (
            json.dumps(build_input, indent=2) + "\n"
        ).encode(),
        "_mpu_build/capture_environment.py": capture_path.read_bytes(),
    }
    context = output_dir / "build-context.tar"
    make_context(archive, context, additions)
    (output_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    write_json(output_dir / "build-input.json", build_input)
    plan = {
        "schema_version": 1,
        "status": "recipe_generated",
        "built": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "recipe": recipe,
        "expected_packages": pins,
        "source": {
            "repository": str(repo),
            "requested_ref": source_ref,
            "commit": commit,
            "working_tree_dirty": bool(status),
            "working_tree_status": status,
            "working_tree_changes_included": False,
            "archive": str(archive),
            "archive_sha256": archive_sha,
            "archive_git_config": {"core.autocrlf": "false", "core.eol": "lf"},
            "pyproject_sha256": hashlib.sha256(pyproject.encode()).hexdigest(),
            "upstream_dockerfile_sha256": hashlib.sha256(original.encode()).hexdigest(),
        },
        "generator": {
            "script_sha256": sha256(__file__),
            "recipe_sha256": sha256(recipe_path),
            "capture_script_sha256": sha256(capture_path),
        },
        "build": {
            "command": command,
            "stdin_file": str(context),
            "context_sha256": sha256(context),
            "dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
            "image_tag": tag,
            "build_args": build_args,
            "external_images": external_images(dockerfile, recipe["cuda_version"]),
        },
        "image": {"id": None, "repo_digests": [], "pip_freeze": None},
        "reproducibility": {
            "source_locked": True,
            "all_dependencies_locked": False,
            "base_registry_verification": "not_performed",
            "note": "Recipe only. Docker/apt/transitive packages and upstream auxiliary downloads are not fully locked. No image or GPU/model result exists until executed and recorded.",
        },
    }
    write_json(output_dir / "environment-plan.json", plan)
    return plan


def execute_build(plan, output_dir):
    output_dir = Path(output_dir).resolve()
    context = Path(plan["build"]["stdin_file"])
    if sha256(context) != plan["build"]["context_sha256"]:
        raise ValueError("Build context changed after plan generation")
    result = {
        "schema_version": 1,
        "status": "build_started",
        "built": False,
        "source_commit": plan["source"]["commit"],
        "image": {"id": None, "repo_digests": []},
    }
    result_path = output_dir / "environment-build.json"
    write_json(result_path, result)
    try:
        with context.open("rb") as source, (output_dir / "docker-build.log").open(
            "w", encoding="utf-8"
        ) as log:
            subprocess.run(
                plan["build"]["command"],
                cwd=output_dir,
                stdin=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        inspect = subprocess.run(
            ["docker", "image", "inspect", plan["build"]["image_tag"]],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        images = json.loads(inspect.stdout)
        if len(images) != 1 or not images[0].get("Id"):
            raise ValueError("Docker inspect did not identify exactly one built image")
        details = images[0]
        write_json(output_dir / "image-inspect.json", details)
        # Execute by immutable local ID, never by the mutable tag just inspected.
        captured = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python3",
                details["Id"],
                "-c",
                "from pathlib import Path; print(Path('/opt/mpu/environment-runtime.json').read_text())",
            ],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = json.loads(captured.stdout)
        if (
            runtime["source_commit"] != plan["source"]["commit"]
            or runtime["source_archive_sha256"] != plan["source"]["archive_sha256"]
        ):
            raise ValueError("Built image source identity does not match this plan")
        write_json(output_dir / "environment-runtime.json", runtime)
        (output_dir / "pip-freeze.txt").write_text(
            runtime["pip_freeze"], encoding="utf-8"
        )
        result.update(
            {
                "status": "built_metadata_verified",
                "built": True,
                "runtime": runtime,
                "image": {
                    "id": details["Id"],
                    "repo_digests": details.get("RepoDigests", []),
                    "inspect_file": "image-inspect.json",
                },
                "validation_scope": "Docker build plus package/source metadata; no GPU/model execution",
                "all_dependencies_locked": False,
            }
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result.update({"status": "build_or_capture_failed", "error": str(error)})
        write_json(result_path, result)
        raise
    write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--stage",
        type=Path,
        help="Must match stage.json in the selected source commit; default: archived stage",
    )
    parser.add_argument(
        "--source-ref",
        required=True,
        help="Local committed experimental ref; resolved and recorded as full commit",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-tag")
    parser.add_argument(
        "--base-image",
        help="NVIDIA CUDA devel image compatible with recipe CUDA_VERSION; tag or verified digest supplied by operator",
    )
    parser.add_argument(
        "--uv-image", help="Optional pinned uv image for the newer Dockerfile"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start Docker build explicitly; planning otherwise performs no network access",
    )
    args = parser.parse_args(argv)
    plan = prepare_environment(
        args.repo,
        args.stage,
        args.source_ref,
        args.output_dir,
        args.image_tag,
        args.base_image,
        args.uv_image,
    )
    result = execute_build(plan, args.output_dir) if args.execute else plan
    print(
        json.dumps(
            {
                "status": result["status"],
                "built": result["built"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
