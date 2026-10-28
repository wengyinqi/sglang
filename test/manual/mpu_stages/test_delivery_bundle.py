"""Negative acceptance cases for transport integrity and runtime/tool identity."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "bundle_test", ROOT / "benchmark/mpu_stages/delivery_bundle.py"
)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def native_image_receipts():
    native = {
        "plan": {
            "source_commit": "a" * 40,
            "prefix": "/opt/mpu/native-fixture",
            "stage": {"id": "fixture", "variant": "cpu"},
        },
        "wheel": {"packaged_extensions": ["sglang/core.so"], "sha256": "b" * 64},
    }
    image = {
        "plan": {
            "build_route": "verified-native-runtime",
            "source_commit": native["plan"]["source_commit"],
            "prefix": native["plan"]["prefix"],
            "native_extensions": native["wheel"]["packaged_extensions"],
            "native_receipt_sha256": "c" * 64,
            "image_tag": "fixture:stage",
        },
        "smoke": {
            "source_commit": native["plan"]["source_commit"],
            "native_extensions_loaded": 1,
            "gpu_executed": False,
        },
    }
    return image, native


def test_native_image_builder_receipt_can_feed_delivery_without_rewriting_it():
    image, native = native_image_receipts()
    result = B.image_metadata(image, native, "c" * 64)
    assert result["stage"] == native["plan"]["stage"]
    assert result["wheel_sha256"] == native["wheel"]["sha256"]
    assert result["runtime_prefix"] == native["plan"]["prefix"]
    assert "stage" not in image and "wheel_sha256" not in image


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_receipt_sha256", "d" * 64),
        ("source_commit", "d" * 40),
        ("prefix", "/opt/mpu/another-build"),
        ("native_extensions", ["sglang/other.so"]),
    ],
)
def test_native_image_plan_cannot_substitute_another_build(key, value):
    image, native = native_image_receipts()
    image["plan"][key] = value
    with pytest.raises(ValueError, match="mismatch"):
        B.image_metadata(image, native, "c" * 64)


def test_native_image_requires_complete_container_extension_evidence():
    image, native = native_image_receipts()
    image["smoke"]["native_extensions_loaded"] = 0
    with pytest.raises(ValueError, match="mismatch"):
        B.image_metadata(image, native, "c" * 64)


def test_legacy_flat_image_receipt_remains_supported():
    image = {"source_commit": "a" * 40, "stage": {"id": "legacy"}}
    assert B.image_metadata(image, {}, "c" * 64) is image


@pytest.mark.parametrize(
    "path",
    [
        "python/sglang/srt/managers/scheduler.py",
        "sgl-kernel/csrc/foo.cu",
        "benchmark/mpu_stages/stage.json",
        "benchmark/mpu_stages/environments/locks/dsa-runtime.txt",
    ],
)
def test_runtime_or_recipe_delta_cannot_reuse_old_image(path):
    with pytest.raises(ValueError):
        B.validate_tool_delta(["test/manual/mpu_stages/test_delivery_bundle.py", path])


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "a/../../escape", "C:/Windows", "a\\..\\bad"]
)
def test_artifact_path_cannot_escape_bundle(tmp_path, name):
    with pytest.raises(ValueError):
        B.safe_path(tmp_path, name)


def bundle(root):
    directory = root / "stages/01"
    directory.mkdir(parents=True)
    files = [
        "source.tar.gz",
        "image.tar.gz",
        "stage.json",
        "model-reference.json",
        "launch.json",
        "image-receipt.json",
        "native-receipt.json",
        "jit-receipt.json",
        "image-inspect.json",
        "native.whl",
        "jit/library.so",
    ]
    for name in files:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture only")
    (root / "restore.py").write_text("# test")
    value = {
        "schema_version": 1,
        "status": "complete",
        "stages": [
            {
                "id": "01",
                "directory": "stages/01",
                "wheel": "native.whl",
                "jit_libraries": [{"path": "jit/library.so"}],
            }
        ],
        "files": {
            p.relative_to(root).as_posix(): {
                "sha256": B.sha(p),
                "bytes": p.stat().st_size,
            }
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    seal(root, value)
    return value


def seal(root, value):
    B.write(root / "bundle.json", value)
    (root / "bundle.sha256").write_text(B.sha(root / "bundle.json"))


@pytest.mark.parametrize("name", ["image.tar.gz", "jit/library.so", "source.tar.gz"])
def test_truncation_or_modified_bytes_block_restore(tmp_path, name):
    bundle(tmp_path)
    B.verify(tmp_path)
    (tmp_path / "stages/01" / name).write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        B.verify(tmp_path)


def test_removed_library_from_manifest_is_still_required(tmp_path):
    value = bundle(tmp_path)
    del value["files"]["stages/01/jit/library.so"]
    seal(tmp_path, value)
    with pytest.raises(ValueError, match="Missing required"):
        B.verify(tmp_path)


def test_external_digest_detects_resealed_manifest(tmp_path):
    value = bundle(tmp_path)
    digest = B.sha(tmp_path / "bundle.json")
    value["stages"][0]["id"] = "different-stage"
    seal(tmp_path, value)
    with pytest.raises(ValueError, match="manifest checksum"):
        B.verify(tmp_path, digest)


def test_incomplete_export_never_passes(tmp_path):
    value = bundle(tmp_path)
    value["status"] = "incomplete"
    seal(tmp_path, value)
    with pytest.raises(ValueError, match="Incomplete"):
        B.verify(tmp_path)


def test_linux_portable_requires_real_process_suite():
    spec = importlib.util.spec_from_file_location(
        "acceptance_bundle_test", ROOT / "benchmark/mpu_stages/acceptance.py"
    )
    accept = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accept)
    from unittest.mock import patch

    with patch.object(accept.sys, "platform", "linux"):
        assert (
            "test/manual/mpu_stages/test_cpu_processes_linux.py"
            in accept.portable_targets({"family": "qwen"})
        )
    with patch.object(accept.sys, "platform", "win32"):
        assert (
            "test/manual/mpu_stages/test_cpu_processes_linux.py"
            not in accept.portable_targets({"family": "qwen"})
        )
