"""Freeze real-document token IDs once for paired storage-path experiments."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_sha256(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_manifest(path, *, lengths=(), count=None, expected_sha256=None):
    path = Path(path).resolve()
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen workload manifest hash changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "mpu_real_document_workload"
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError("Invalid real-document workload manifest")
    for length in lengths:
        if str(length) not in manifest["files"]:
            raise ValueError(f"Frozen workload has no {length}-token bucket")
    for key, item in manifest["files"].items():
        if (
            not key.isdecimal()
            or not isinstance(item, dict)
            or item.get("input_len") != int(key)
            or type(item.get("num_requests")) is not int
            or item["num_requests"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
            or not isinstance(item.get("path"), str)
        ):
            raise ValueError("Invalid frozen workload bucket")
        target = (path.parent / item["path"]).resolve()
        if not target.is_relative_to(path.parent):
            raise ValueError("Workload bucket must stay inside the snapshot directory")
        if file_sha256(target) != item["sha256"]:
            raise ValueError(f"Frozen {key}-token input file hash changed")
        if int(key) in lengths and count is not None and item["num_requests"] < count:
            raise ValueError(
                "Not enough distinct frozen requests; repetition is forbidden"
            )
    return manifest


def bucket(path, length):
    manifest = load_manifest(path, lengths=[length])
    item = manifest["files"][str(length)]
    return Path(path).resolve().parent / item["path"], item


def inputs(path, length, count):
    source, item = bucket(path, length)
    if count > item["num_requests"]:
        raise ValueError("Not enough distinct frozen requests")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if len(rows) != item["num_requests"]:
        raise ValueError("Frozen row count disagrees with manifest")
    selected = rows[:count]
    for row in selected:
        tokens = row.get("input_ids", [])
        if (
            len(tokens) != length
            or any(type(token) is not int or token < 0 for token in tokens)
            or row.get("input_sha256") != token_sha256(tokens)
        ):
            raise ValueError("Frozen token sequence is malformed or changed")
    return [row["input_ids"] for row in selected]


def freeze(records, tokenizer, lengths, count, seed, output_dir, source):
    """Crop genuine document prefixes; retain each question and chat envelope.

    Encoding is deliberately segmented at the document boundary. The resulting
    IDs, rather than a later decode/re-encode, define the exact experimental input.
    No document is repeated or padded to manufacture a long context.
    """
    if (
        not lengths
        or any(type(n) is not int or n <= 0 for n in lengths)
        or type(count) is not int
        or count <= 0
        or not isinstance(records, list)
    ):
        raise ValueError("Positive lengths/count and a JSON record array are required")
    lengths = sorted(set(lengths))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    order = list(enumerate(records))
    random.Random(seed).shuffle(order)
    marker = "__MPU_REAL_DOCUMENT_BOUNDARY__"
    rows = {length: [] for length in lengths}
    seen = set()
    for index, record in order:
        context = record.get("context")
        if not isinstance(context, str) or not context.strip():
            continue
        if any(
            not isinstance(record.get(key), str)
            for key in ("question", "choice_A", "choice_B", "choice_C", "choice_D")
        ):
            raise ValueError("LongBench-v2 record is missing its question/options")
        source_id = str(record.get("_id", record.get("id", index)))
        if source_id in seen:
            raise ValueError("Duplicate source document ID")
        seen.add(source_id)
        question = (
            f"\n\nQuestion: {record['question']}\n"
            + "\n".join(f"{letter}. {record[f'choice_{letter}']}" for letter in "ABCD")
            + "\nAnswer the question using the document above."
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Read this document:\n" + marker + question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.count(marker) != 1:
            raise ValueError("Chat template did not preserve the document boundary")
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        budgets = {length: length - len(prefix) - len(suffix) for length in lengths}
        if min(budgets.values()) <= 0:
            raise ValueError("Input bucket cannot fit the question and chat envelope")
        document = tokenizer.encode(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=max(budgets.values()),
        )
        if len(document) < max(budgets.values()):
            continue
        for length in lengths:
            tokens = prefix + document[: budgets[length]] + suffix
            rows[length].append(
                {
                    "source_id": source_id,
                    "source_record_index": index,
                    "domain": record.get("domain"),
                    "source_context_sha256": hashlib.sha256(
                        context.encode("utf-8")
                    ).hexdigest(),
                    "context_tokens_used": budgets[length],
                    "context_crop": "prefix; no repetition or synthetic padding",
                    "input_ids": tokens,
                    "input_sha256": token_sha256(tokens),
                }
            )
        if len(rows[lengths[0]]) == count:
            break
    if len(rows[lengths[0]]) != count:
        raise ValueError(
            f"Only {len(rows[lengths[0]])} distinct documents cover every bucket; requested {count}"
        )
    tokenizer_path = Path(tokenizer.name_or_path)
    tokenizer_files = {
        name: file_sha256(tokenizer_path / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if (tokenizer_path / name).is_file()
    }
    if not tokenizer_files:
        raise ValueError(
            "Freezing requires a local tokenizer snapshot with file fingerprints"
        )
    manifest = {
        "schema_version": 1,
        "kind": "mpu_real_document_workload",
        "source": source,
        "seed": seed,
        "num_distinct_documents": count,
        "tokenizer_files": tokenizer_files,
        "vocabulary_size": len(tokenizer),
        "input_policy": "segmented chat-template encoding; natural document prefix plus preserved question/options; send frozen IDs directly",
        "row_order": "fixed across variants, concurrency points and repeats",
        "quality_evaluation": False,
        "quality_note": "Cropped contexts and fixed output budgets are performance workloads, not official LongBench-v2 accuracy evaluation",
        "files": {},
    }
    for length in lengths:
        name = f"input-{length}.json"
        write_json(
            output_dir / name,
            {
                "schema_version": 1,
                "kind": "mpu_frozen_token_ids",
                "input_len": length,
                "vocabulary_size": len(tokenizer),
                "tokenizer_files": tokenizer_files,
                "source": source,
                "rows": rows[length],
            },
        )
        manifest["files"][str(length)] = {
            "path": name,
            "input_len": length,
            "num_requests": count,
            "sha256": file_sha256(output_dir / name),
        }
    write_json(output_dir / "workload.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-lens", nargs="+", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if file_sha256(args.source_json) != args.source_sha256:
        raise ValueError("Downloaded source hash differs from the pinned receipt")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("Pin the dataset to an immutable 40-character revision")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = freeze(
        json.loads(args.source_json.read_text(encoding="utf-8")),
        tokenizer,
        args.input_lens,
        args.num_prompts,
        args.seed,
        args.output_dir,
        {
            "repo": args.source_repo,
            "revision": args.source_revision,
            "file": args.source_json.name,
            "sha256": args.source_sha256,
            "license": args.source_license,
        },
    )
    print(
        json.dumps(
            {
                "num_distinct_documents": result["num_distinct_documents"],
                "files": result["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
