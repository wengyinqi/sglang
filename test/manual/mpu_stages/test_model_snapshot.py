"""Small local fixtures verify model identity boundaries without model downloads."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "mpu_model_snapshot", ROOT / "benchmark/mpu_stages/model_snapshot.py"
)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class ModelSnapshotTests(unittest.TestCase):
    def test_bundled_reference_matches_this_stage_and_has_only_checksum_inventory(self):
        directory = ROOT / "benchmark/mpu_stages"
        stage = MODEL.read_json(directory / "stage.json")
        path = directory / "environments/model-references" / f"{stage['family']}.json"
        reference = MODEL.reference_identity(path, MODEL.stage_identity(stage))
        self.assertIs(reference["weight_payloads_downloaded"], False)
        self.assertIn(stage["default_revision"], reference["source_url"])
        weights = [
            entry for entry in reference["files"].values() if entry["role"] == "weight"
        ]
        self.assertTrue(weights)
        for name, entry in reference["files"].items():
            MODEL.safe_name(name)
            self.assertGreater(entry["size"], 0)
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "model"
        self.root.mkdir()
        self.stage = {
            "family": "qwen",
            "default_model": "test/fixture",
            "default_revision": "a" * 40,
        }
        data = {
            "config.json": {"model_type": "fixture"},
            "tokenizer_config.json": {"tokenizer_class": "fixture"},
            "tokenizer.json": {"version": "1.0"},
            "model.safetensors.index.json": {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            },
        }
        for name, value in data.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        (self.root / "model-00001-of-00002.safetensors").write_bytes(
            b"weight fixture 1"
        )
        (self.root / "model-00002-of-00002.safetensors").write_bytes(
            b"weight fixture 2"
        )

    def create(self, full=False, reference=None):
        return MODEL.create_manifest(self.root, self.stage, full, reference)

    def test_default_metadata_inventory_is_not_a_verified_weight_identity(self):
        manifest = self.create()
        self.assertEqual(manifest["status"], "metadata_only")
        self.assertFalse(manifest["full_checksum"])
        self.assertFalse(manifest["upstream_identity_verified"])
        self.assertTrue(MODEL.verify_manifest(manifest, self.root)["ok"])
        self.assertFalse(
            MODEL.verify_manifest(manifest, self.root, require_full_checksum=True)["ok"]
        )

    def test_full_fingerprint_is_portable_but_not_automatically_hf_verified(self):
        manifest = self.create(full=True)
        copied = self.root.parent / "copy"
        shutil.copytree(self.root, copied)
        result = MODEL.verify_manifest(
            manifest, copied, stage=self.stage, require_full_checksum=True
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "full-content-fingerprint")
        self.assertFalse(result["upstream_identity_verified"])

    def test_same_size_weight_corruption_is_detected_with_full_checksums(self):
        manifest = self.create(full=True)
        (self.root / "model-00001-of-00002.safetensors").write_bytes(
            b"weight fixture X"
        )
        result = MODEL.verify_manifest(manifest, self.root, require_full_checksum=True)
        self.assertFalse(result["ok"])
        self.assertTrue(any("Checksum changed" in error for error in result["errors"]))

    def test_missing_shard_and_lfs_pointer_are_incomplete(self):
        (self.root / "model-00001-of-00002.safetensors").unlink()
        (self.root / "model-00002-of-00002.safetensors").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:fixture\n"
        )
        manifest = self.create(full=True)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertTrue(
            any("Missing weight shard" in error for error in manifest["errors"])
        )
        self.assertTrue(any("Git LFS pointer" in error for error in manifest["errors"]))

    def test_config_tokenizer_index_and_stage_changes_fail_verification(self):
        manifest = self.create(full=True)
        (self.root / "tokenizer_config.json").write_text(
            '{"tokenizer_class": "changed"}'
        )
        result = MODEL.verify_manifest(
            manifest, self.root, stage=self.stage | {"default_revision": "b" * 40}
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("does not match selected stage" in error for error in result["errors"])
        )
        self.assertTrue(
            any("tokenizer_config.json" in error for error in result["errors"])
        )

    def test_path_traversal_in_weight_index_is_rejected(self):
        (self.root / "model.safetensors.index.json").write_text(
            '{"weight_map": {"layer": "../outside.safetensors"}}'
        )
        manifest = self.create()
        self.assertEqual(manifest["status"], "incomplete")
        self.assertTrue(
            any("Unsafe model file path" in error for error in manifest["errors"])
        )

    def test_tampered_manifest_cannot_claim_full_checksums(self):
        manifest = self.create()
        manifest["full_checksum"] = True
        result = MODEL.verify_manifest(manifest, self.root, require_full_checksum=True)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("Required checksum missing" in error for error in result["errors"])
        )

    def test_trusted_reference_requires_weight_bytes_and_matching_revision(self):
        full = self.create(full=True)
        reference = {
            "schema_version": 1,
            "kind": "hf_revision_reference",
            "model_id": "test/fixture",
            "revision": "a" * 40,
            "resolved_revision": "a" * 40,
            "files": full["files"],
        }
        self.assertFalse(self.create(reference=reference)["upstream_identity_verified"])
        self.assertTrue(
            self.create(full=True, reference=reference)["upstream_identity_verified"]
        )
        with self.assertRaisesRegex(ValueError, "revision does not match"):
            self.create(full=True, reference=reference | {"revision": "b" * 40})

    def test_new_metadata_and_missing_tokenizer_block(self):
        manifest = self.create(full=True)
        (self.root / "configuration_fixture.py").write_text("# new remote code\n")
        (self.root / "tokenizer.json").unlink()
        result = MODEL.verify_manifest(manifest, self.root)
        self.assertFalse(result["ok"])
        self.assertTrue(any("file set changed" in error for error in result["errors"]))
        self.assertTrue(any("Missing tokenizer" in error for error in result["errors"]))

    def test_reference_fetch_gets_only_metadata_and_lfs_identity(self):
        calls = []
        config = b'{"model_type":"fixture"}'
        api = {
            "sha": "a" * 40,
            "siblings": [
                {"rfilename": "config.json", "size": len(config)},
                {
                    "rfilename": "model.safetensors",
                    "size": 100000000000,
                    "lfs": {"sha256": "b" * 64, "size": 100000000000},
                },
            ],
        }

        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, limit):
                return self.content[:limit]

        class Opener:
            def open(self, request, timeout):
                calls.append(request.full_url)
                return Response(
                    json.dumps(api).encode() if "/api/" in request.full_url else config
                )

        reference = MODEL.fetch_reference(self.stage, Opener())
        self.assertEqual(len(calls), 2)
        self.assertFalse(
            any("resolve/" in url and url.endswith("safetensors") for url in calls)
        )
        self.assertEqual(reference["files"]["model.safetensors"]["sha256"], "b" * 64)
        self.assertEqual(
            reference["files"]["config.json"]["sha256"],
            hashlib.sha256(config).hexdigest(),
        )
        self.assertFalse(reference["weight_payloads_downloaded"])


if __name__ == "__main__":
    unittest.main()
