"""Strict input-ID snapshots: no sampling, padding, or text re-tokenization."""

import hashlib
import json
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


def load_rows(path, expected_sha256, input_len, count, tokenizer_path, vocabulary_size):
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("Frozen IDs require their expected SHA256")
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Frozen input file hash changed")
    payload = json.loads(content)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "mpu_frozen_token_ids"
        or payload.get("input_len") != input_len
        or payload.get("vocabulary_size") != vocabulary_size
        or not isinstance(payload.get("rows"), list)
    ):
        raise ValueError("Frozen IDs have incompatible length/vocabulary/schema")
    fingerprints = payload.get("tokenizer_files")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("Frozen inputs need tokenizer fingerprints")
    for name, digest in fingerprints.items():
        if name not in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            raise ValueError("Unexpected tokenizer fingerprint file")
        if (
            hashlib.sha256((Path(tokenizer_path) / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError("Runtime tokenizer differs from the frozen tokenizer")
    if type(count) is not int or count <= 0 or len(payload["rows"]) < count:
        raise ValueError("Not enough distinct frozen rows; repetition is forbidden")
    selected = payload["rows"][:count]
    seen = set()
    for row in selected:
        ids = row.get("input_ids")
        source_id = row.get("source_id")
        if (
            not isinstance(ids, list)
            or len(ids) != input_len
            or any(
                type(token) is not int or not 0 <= token < vocabulary_size
                for token in ids
            )
            or not isinstance(source_id, str)
            or source_id in seen
        ):
            raise ValueError("Malformed or duplicate frozen request")
        seen.add(source_id)
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if row.get("input_sha256") != digest:
            raise ValueError("Frozen row token hash changed")
    return selected


@dataclass
class FrozenIDsDataset(BaseDataset):
    dataset_path: str
    expected_sha256: str
    input_len: int
    output_len: int
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace):
        if not getattr(args, "tokenize_prompt", False):
            raise ValueError("Frozen IDs require --tokenize-prompt to send input_ids")
        return cls(
            args.dataset_path,
            args.frozen_ids_sha256,
            args.random_input_len,
            args.random_output_len,
            args.num_prompts,
        )

    def load(self, tokenizer, model_id=None):
        rows = load_rows(
            self.dataset_path,
            self.expected_sha256,
            self.input_len,
            self.num_requests,
            tokenizer.name_or_path,
            len(tokenizer),
        )
        return [
            DatasetRow(
                prompt=row["input_ids"],
                prompt_len=self.input_len,
                output_len=self.output_len,
            )
            for row in rows
        ]
