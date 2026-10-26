"""Inventory/verify a local model without loading it or downloading weights.

Default snapshots hash metadata and inventory weight sizes. --full-checksum reads
every weight byte to produce a repeatable local content identity; neither a path
nor a declared Hugging Face revision proves those bytes came from that revision.
"""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
}
VOCABULARY_NAMES = TOKENIZER_NAMES - {"merges.txt"}
MAX_METADATA_BYTES = 64 * 1024 * 1024


def read_json(value):
    result = (
        value
        if isinstance(value, dict)
        else json.loads(Path(value).read_text(encoding="utf-8"))
    )
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe model file path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("..", ".") for part in value.split("/")):
        raise ValueError(f"Unsafe model file path: {value!r}")
    return value


def metadata_role(name):
    path = PurePosixPath(name)
    if path.name.endswith(".index.json"):
        return "weight_index"
    if path.name in TOKENIZER_NAMES or "token" in path.name or path.suffix == ".jinja":
        return "tokenizer"
    if path.suffix == ".py":
        return "remote_code"
    return "config"


def metadata_name(name):
    path = PurePosixPath(name)
    # Include remote-code modules and chat-template subdirectories as well as
    # every top-level JSON config; skip download-cache bookkeeping.
    return not any(part.startswith(".") for part in path.parts) and (
        (
            len(path.parts) == 1
            and (
                path.suffix in (".json", ".model", ".jinja")
                or path.name in TOKENIZER_NAMES
            )
        )
        or path.suffix == ".py"
        or (path.parts[0] == "chat_templates" and path.suffix == ".jinja")
    )


def weight_name(name):
    path = PurePosixPath(name)
    return path.suffix == ".safetensors" or (
        path.suffix == ".bin" and path.name.startswith(("pytorch_model", "model"))
    )


def inventory(model_path):
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Local model directory is missing: {root}")
    metadata = {}
    errors = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if metadata_name(name) and path.is_file():
            metadata[name] = path
    if "config.json" not in metadata:
        errors.append("Missing config.json")
    if "tokenizer_config.json" not in metadata:
        errors.append("Missing tokenizer_config.json")
    if not VOCABULARY_NAMES.intersection(metadata):
        errors.append("Missing tokenizer vocabulary/model artifact")
    shards = set()
    index_count = 0
    for name, path in metadata.items():
        if not name.endswith(".index.json"):
            continue
        try:
            data = read_json(path)
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append(f"Invalid or empty weight_map: {name}")
                continue
            index_count += 1
            for item in weight_map.values():
                safe_name(item)
                # HF indexes use paths relative to their containing directory.
                shard = (PurePosixPath(name).parent / item).as_posix()
                safe_name(shard)
                shards.add(shard)
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"Invalid weight index {name}: {error}")
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and weight_name(path.name)
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    }
    if not index_count:
        if len(present) == 1:
            shards.update(present)
        elif not present:
            errors.append("No weight index or single weight file found")
        else:
            errors.append("Multiple weight files require a weight index")
            shards.update(present)
    else:
        # Unreferenced payloads may be another serialization, but are still
        # recorded so a claimed full fingerprint cannot silently omit them.
        shards.update(present)
    for name in sorted(shards):
        path = root / name
        if not path.is_file():
            errors.append(f"Missing weight shard: {name}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty weight shard: {name}")
        else:
            with path.open("rb") as stream:
                if stream.read(200).startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ):
                    errors.append(f"Weight shard is only a Git LFS pointer: {name}")
    return root, metadata, sorted(shards), errors


def stage_identity(stage):
    stage = read_json(stage)
    revision = stage.get("default_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Stage must declare a fixed full 40-character model revision")
    if any(
        not isinstance(stage.get(key), str) or not stage[key]
        for key in ("family", "default_model")
    ):
        raise ValueError("Stage must declare a family and model id")
    return {
        "family": stage["family"],
        "model_id": stage["default_model"],
        "revision": revision,
    }


def reference_identity(reference, identity):
    reference = read_json(reference)
    if (
        reference.get("schema_version") != 1
        or reference.get("kind") != "hf_revision_reference"
    ):
        raise ValueError("Unsupported trusted reference format")
    for key in ("model_id", "revision"):
        if reference.get(key) != identity[key]:
            raise ValueError(f"Reference {key} does not match selected stage")
    if not reference.get("resolved_revision") == identity["revision"]:
        raise ValueError(
            "Reference revision was not resolved to the selected immutable revision"
        )
    if not isinstance(reference.get("files"), dict):
        raise ValueError("Reference has no file checksums")
    for name, record in reference["files"].items():
        safe_name(name)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("size"), int)
            or record["size"] < 0
        ):
            raise ValueError(f"Reference file record is malformed: {name}")
        if record.get("sha256") is not None and (
            not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError(f"Reference checksum is malformed: {name}")
    return reference


def check_reference(files, reference, full_checksum):
    errors = []
    verified_weights = True
    for name, expected in reference["files"].items():
        safe_name(name)
        actual = files.get(name)
        if not actual:
            errors.append(f"Reference file missing: {name}")
            verified_weights = False
            continue
        if actual["size"] != expected.get("size"):
            errors.append(f"Reference size mismatch: {name}")
        if expected.get("sha256") and actual.get("sha256"):
            if actual["sha256"] != expected["sha256"]:
                errors.append(f"Reference checksum mismatch: {name}")
        elif actual["role"] != "weight":
            errors.append(f"Reference metadata lacks a comparable SHA256: {name}")
        else:
            verified_weights = False
    for name, actual in files.items():
        if name not in reference["files"]:
            errors.append(f"Local file absent from reference: {name}")
        if actual["role"] == "weight" and not reference["files"].get(name, {}).get(
            "sha256"
        ):
            verified_weights = False
    return errors, bool(full_checksum and verified_weights and not errors)


def fingerprint(files):
    # Paths, roles, lengths, and hashes form identity; mtime and local paths do
    # not, allowing identical models to be copied to another disk/machine.
    content = {
        name: {key: record.get(key) for key in ("role", "size", "sha256")}
        for name, record in sorted(files.items())
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def create_manifest(model_path, stage, full_checksum=False, reference=None):
    identity = stage_identity(stage)
    root, metadata, shards, errors = inventory(model_path)
    files = {}
    for name, path in sorted(metadata.items()):
        stat = path.stat()
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                errors.append(f"Invalid metadata JSON {name}: {error}")
        digest = hash_file(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(f"Metadata changed while hashing: {name}")
        files[name] = {
            "role": metadata_role(name),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
    for name in shards:
        path = root / name
        if path.is_file():
            stat = path.stat()
            digest = hash_file(path) if full_checksum else None
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                errors.append(f"Weight shard changed while hashing: {name}")
            files[name] = {
                "role": "weight",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
    verified = False
    reference_data = None
    if reference is not None:
        reference_data = reference_identity(reference, identity)
        mismatches, verified = check_reference(files, reference_data, full_checksum)
        errors.extend(mismatches)
    complete_checksum = bool(
        full_checksum
        and shards
        and not errors
        and all(files.get(name, {}).get("sha256") for name in shards)
    )
    return {
        "schema_version": 1,
        "kind": "mpu_model_snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
        "model_path": str(root),
        "status": (
            "incomplete"
            if errors
            else ("full-content-fingerprint" if complete_checksum else "metadata_only")
        ),
        "full_checksum": complete_checksum,
        "files": files,
        "weight_shards": shards,
        "content_fingerprint_sha256": fingerprint(files),
        "errors": errors,
        "upstream_identity_verified": verified,
        "reference": reference_data,
        "provenance": {
            "kind": "hf_checksum_reference" if reference_data else "local_snapshot",
            "declared_revision_is_proof": False,
            "scope": (
                "Full local content identity"
                if complete_checksum
                else "Metadata checksums and weight existence/size only"
            ),
            "note": "The model path/HF cache directory name is not evidence of weight origin. Without a trusted matching reference, the revision above is only declared.",
        },
    }


def verify_manifest(
    manifest_or_path,
    model_path,
    reference=None,
    stage=None,
    require_full_checksum=False,
):
    manifest = read_json(manifest_or_path)
    errors = []
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "mpu_model_snapshot"
    ):
        raise ValueError("Unsupported model manifest format")
    if not isinstance(manifest.get("full_checksum"), bool):
        raise ValueError("Manifest full_checksum must be boolean")
    if not isinstance(manifest.get("files"), dict) or not isinstance(
        manifest.get("weight_shards"), list
    ):
        raise ValueError(
            "Manifest files/weight_shards inventory is missing or malformed"
        )
    for key in ("family", "model_id", "revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Manifest {key} is missing or malformed")
    if manifest.get("errors"):
        errors.append("Original manifest was incomplete")
    identity = {key: manifest[key] for key in ("family", "model_id", "revision")}
    if stage is not None and stage_identity(stage) != identity:
        errors.append("Manifest family/model/revision does not match selected stage")
    full_checksum = bool(manifest.get("full_checksum"))
    if require_full_checksum and not full_checksum:
        errors.append(
            "Full weight checksums are required; recreate with --full-checksum"
        )
    expected_files = manifest.get("files", {})
    if not expected_files or not manifest.get("weight_shards"):
        errors.append("Manifest has no metadata/weight inventory")
    for name, record in expected_files.items():
        safe_name(name)
        if not isinstance(record, dict) or record.get("role") not in (
            "weight",
            "weight_index",
            "tokenizer",
            "config",
            "remote_code",
        ):
            raise ValueError(f"Manifest file record is malformed: {name}")
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise ValueError(f"Manifest file size is malformed: {name}")
        if record.get("role") != "weight" or full_checksum:
            if not isinstance(record.get("sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", record["sha256"]
            ):
                errors.append(f"Required checksum missing: {name}")
    source_stage = {
        "family": identity["family"],
        "default_model": identity["model_id"],
        "default_revision": identity["revision"],
    }
    current = create_manifest(
        model_path,
        source_stage,
        full_checksum=full_checksum,
        reference=reference if reference is not None else manifest.get("reference"),
    )
    errors.extend(current["errors"])
    if set(expected_files) != set(current["files"]):
        errors.append("Model metadata/weight file set changed")
    if manifest.get("weight_shards") != current["weight_shards"]:
        errors.append("Weight index/shard list changed")
    for name in expected_files.keys() & current["files"].keys():
        old, new = expected_files[name], current["files"][name]
        if old.get("role") != new["role"] or old.get("size") != new["size"]:
            errors.append(f"Role/size changed: {name}")
        if old.get("sha256") and old["sha256"] != new["sha256"]:
            errors.append(f"Checksum changed: {name}")
        if not old.get("sha256") and old.get("mtime_ns") != new["mtime_ns"]:
            errors.append(
                f"Unhashed weight timestamp changed; full checksum required: {name}"
            )
    if fingerprint(expected_files) != manifest.get("content_fingerprint_sha256"):
        errors.append("Manifest fingerprint does not match its file records")
    return {
        "schema_version": 1,
        "ok": not errors,
        "status": "failed" if errors else current["status"],
        "errors": errors,
        "full_checksum": bool(full_checksum and not errors),
        "upstream_identity_verified": bool(
            current["upstream_identity_verified"] and not errors
        ),
        **identity,
        "model_path": str(Path(model_path).resolve()),
        "content_fingerprint_sha256": current["content_fingerprint_sha256"],
        "validation_scope": current["provenance"]["scope"],
    }


def fetch_reference(stage, opener=None):
    """Fetch fixed-revision metadata and LFS SHA256 values, never weight bytes.

    Public HTTPS only. The result is an operator-reviewable reference, not a
    signature. Private repositories can supply an independently trusted JSON.
    """
    identity = stage_identity(stage)
    model_id, revision = identity["model_id"], identity["revision"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("HF reference requires a namespace/model repository id")
    opener = opener or urllib.request.build_opener()
    base = "https://huggingface.co"

    def fetch(url, limit):
        with opener.open(
            urllib.request.Request(
                url, headers={"User-Agent": "sglang-mpu-model-snapshot/1"}
            ),
            timeout=60,
        ) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError(f"Metadata response exceeds {limit} bytes")
            return body

    # blobs=true returns an exact-revision sibling inventory including LFS
    # digests, avoiding any need to fetch even the first byte of large weights.
    url = f"{base}/api/models/{model_id}/revision/{revision}?blobs=true"
    info = json.loads(fetch(url, 16 * 1024 * 1024))
    if info.get("sha") != revision:
        raise ValueError("HF did not resolve the requested immutable revision")
    files = {}
    for entry in info.get("siblings", []):
        name = safe_name(entry["rfilename"])
        if not metadata_name(name) and not weight_name(name):
            continue
        if weight_name(name):
            lfs = entry.get("lfs") or {}
            digest = lfs.get("sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
                raise ValueError(
                    f"HF provides no LFS SHA256 for {name}; supply a trusted checksum reference"
                )
            files[name] = {
                "role": "weight",
                "size": lfs.get("size", entry.get("size")),
                "sha256": digest,
            }
        else:
            if entry.get("size", 0) > MAX_METADATA_BYTES:
                raise ValueError(f"Metadata too large to fetch safely: {name}")
            data = fetch(
                f"{base}/{model_id}/resolve/{revision}/{urllib.parse.quote(name, safe='/')}",
                MAX_METADATA_BYTES,
            )
            files[name] = {
                "role": metadata_role(name),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    if "config.json" not in files or not any(
        record["role"] == "weight" for record in files.values()
    ):
        raise ValueError("HF reference lacks config or weight inventory")
    return {
        "schema_version": 1,
        "kind": "hf_revision_reference",
        **identity,
        "resolved_revision": revision,
        "files": files,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": url,
        "weight_payloads_downloaded": False,
        "trust_basis": "HTTPS Hugging Face fixed-revision API plus metadata content; operator must preserve/review this reference",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Hash metadata; --full-checksum additionally reads all weights"
    )
    create.add_argument("--stage", type=Path, default=HERE / "stage.json")
    create.add_argument("--model", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--full-checksum", action="store_true")
    create.add_argument("--reference", type=Path)
    verify = commands.add_parser(
        "verify",
        help="Re-read local metadata/weights at the manifest's checksum strength",
    )
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--stage", type=Path)
    verify.add_argument("--reference", type=Path)
    verify.add_argument("--require-full-checksum", action="store_true")
    verify.add_argument("--output", type=Path)
    fetch = commands.add_parser(
        "fetch-reference",
        help="Online: fixed-revision metadata/LFS checksums only, no weights",
    )
    fetch.add_argument("--stage", type=Path, default=HERE / "stage.json")
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        if args.model.resolve() in args.output.resolve().parents:
            raise ValueError(
                "Save the manifest outside the model directory to avoid a self-referential inventory"
            )
        result = create_manifest(
            args.model, args.stage, args.full_checksum, args.reference
        )
        write_json(args.output, result)
        ok = not result["errors"]
    elif args.command == "verify":
        result = verify_manifest(
            args.manifest,
            args.model,
            args.reference,
            args.stage,
            args.require_full_checksum,
        )
        if args.output:
            write_json(args.output, result)
        ok = result["ok"]
    else:
        result = fetch_reference(args.stage)
        write_json(args.output, result)
        ok = True
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
