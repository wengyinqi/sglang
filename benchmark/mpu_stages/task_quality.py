"""Complete-input LongBench-v2 subset: freeze, query one backend, then compare.

This is a preselected, hardware-bounded task-quality diagnostic, not the official
LongBench-v2 leaderboard. It never crops inputs, ignores EOS, or scores reasoning
text as a final answer. No serving or GPU modules are imported by the evaluator.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import time
import urllib.request
from pathlib import Path

DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
MODEL_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
VARIANTS = ("dense", "hbm", "cpu")
ALL_VARIANTS = VARIANTS + ("mpu",)
MPU_MEMORY_RESERVE_BYTES = 1 << 30
COMMON_CONTROLS = {
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1,
    "ep_size": 1,
    "disaggregation_mode": "null",
    "disable_overlap_schedule": True,
    "disable_radix_cache": True,
    "enable_hierarchical_cache": False,
    "allow_auto_truncate": False,
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "page_size": 64,
    "chunked_prefill_size": 8192,
    "prefill_max_requests": 1,
    "speculative_algorithm": None,
    "enable_mixed_chunk": False,
    "enable_two_batch_overlap": False,
}


QUALITY_STAGE = None


def configure_stage(stage):
    """CLI pins each model branch; default function API retains Qwen controls."""
    global QUALITY_STAGE, MODEL_ID, MODEL_REVISION, VARIANTS, ALL_VARIANTS
    QUALITY_STAGE = stage
    MODEL_ID, MODEL_REVISION = stage["default_model"], stage["default_revision"]
    VARIANTS = ("dense", "hbm", "cpu") if stage["family"] == "qwen" else ("hbm", "cpu")
    ALL_VARIANTS = VARIANTS + ("mpu",)


def multirank_quality():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "quality_model_native", Path(__file__).with_name("quality_multirank.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def model_identity(path):
    model = read(path)
    require(
        model.get("revision") == MODEL_REVISION and model.get("model_id") == MODEL_ID,
        "Quality diagnostic requires this stage pinned model",
    )
    require(
        model.get("full_checksum") is True
        and model.get("upstream_identity_verified") is True,
        "A fully checksummed, upstream-verified model manifest is required",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", model.get("content_fingerprint_sha256", "")),
        "Missing full model content identity",
    )
    return model


def prompt(record):
    require(
        all(
            isinstance(record.get(k), str) and record[k].strip()
            for k in (
                "context",
                "question",
                "choice_A",
                "choice_B",
                "choice_C",
                "choice_D",
            )
        ),
        "Incomplete LongBench-v2 source record",
    )
    require(
        isinstance(record.get("answer"), str)
        and record["answer"] in "ABCD"
        and len(record["answer"]) == 1,
        "Expected one source answer letter",
    )
    return (
        "Read the following complete document and answer its question.\n\n"
        + record["context"]
        + "\n\nQuestion: "
        + record["question"]
        + "\n"
        + "\n".join(f"{letter}. {record[f'choice_{letter}']}" for letter in "ABCD")
        + "\nGive your final answer in the exact form 'Answer: X', where X is A, B, C, or D."
    )


def select(
    records,
    tokenizer,
    *,
    count,
    seed,
    max_input_tokens,
    max_new_tokens,
    max_total_tokens,
):
    """Choose before inference, by seeded within-domain order and round robin."""
    require(
        isinstance(records, list) and records and count > 0,
        "Positive sample count and source records required",
    )
    require(
        max_input_tokens > 0
        and max_new_tokens > 0
        and max_total_tokens > max_new_tokens,
        "Invalid complete-input/output budgets",
    )
    # Qwen tokenizer len() can enumerate its vocabulary metadata. Query it once
    # for this read-only selection, rather than once for every encoded token.
    vocabulary_size = len(tokenizer)
    require(vocabulary_size > 0, "Tokenizer vocabulary must be nonempty")
    candidates, eligible, seen = [], collections.defaultdict(list), set()
    for index, record in enumerate(records):
        require(isinstance(record, dict), "Source records must be objects")
        source_id = str(record.get("_id", record.get("id", index)))
        require(source_id and source_id not in seen, "Duplicate source ID")
        seen.add(source_id)
        domain = record.get("domain")
        require(isinstance(domain, str) and domain, "Missing source domain")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(record)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
        require(
            tokens and all(type(t) is int and 0 <= t < vocabulary_size for t in tokens),
            "Tokenizer returned malformed complete input IDs",
        )
        reason = (
            "complete_input_exceeds_limit"
            if len(tokens) > max_input_tokens
            else (
                "input_plus_output_exceeds_physical_budget"
                if len(tokens) + max_new_tokens > max_total_tokens
                else None
            )
        )
        candidate = {
            "source_id": source_id,
            "source_record_index": index,
            "domain": domain,
            "difficulty": record.get("difficulty"),
            "source_length_bucket": record.get("length"),
            "complete_input_tokens": len(tokens),
            "eligible": reason is None,
            "selection_status": reason or "eligible_not_selected",
        }
        candidates.append(candidate)
        if reason is None:
            eligible[domain].append(
                {
                    **candidate,
                    "expected_answer": record["answer"],
                    "source_context_sha256": hashlib.sha256(
                        record["context"].encode("utf-8")
                    ).hexdigest(),
                    "input_ids": tokens,
                    "input_sha256": token_sha(tokens),
                    "generation_prompt_opens_thinking": rendered.rstrip().endswith(
                        "<think>"
                    ),
                }
            )
    require(
        sum(map(len, eligible.values())) >= count,
        "Not enough complete, eligible task samples",
    )
    rng = random.Random(seed)
    for domain in sorted(eligible):
        rng.shuffle(eligible[domain])
    selected, offsets = [], collections.Counter()
    while len(selected) < count:
        for domain in sorted(eligible):
            if offsets[domain] < len(eligible[domain]) and len(selected) < count:
                selected.append(eligible[domain][offsets[domain]])
                offsets[domain] += 1
    chosen = {r["source_id"] for r in selected}
    for c in candidates:
        if c["source_id"] in chosen:
            c["selection_status"] = "selected"
    for row in selected:
        row["selection_status"] = "selected"
    return selected, candidates


def freeze(args):
    require(
        file_sha(args.source_json) == args.source_sha256, "Source dataset hash changed"
    )
    require(
        args.source_revision == DATASET_REVISION, "Use the fixed LongBench-v2 revision"
    )
    model = model_identity(args.model_manifest)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    tokenizer_files = {
        name: file_sha(Path(args.tokenizer) / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if (Path(args.tokenizer) / name).is_file()
    }
    require(
        "tokenizer_config.json" in tokenizer_files
        and "tokenizer.json" in tokenizer_files,
        "Local model tokenizer fingerprints are required",
    )
    for name, digest in tokenizer_files.items():
        require(
            model.get("files", {}).get(name, {}).get("sha256") == digest,
            f"Tokenizer does not match the full model manifest: {name}",
        )
    rows, candidates = select(
        read(args.source_json),
        tokenizer,
        count=args.num_samples,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(
        args.output_dir / "inputs.json",
        {"kind": "mpu_complete_task_inputs", "rows": rows},
    )
    manifest = {
        "schema_version": 1,
        "kind": "mpu_task_quality_subset",
        "quality_evaluation": True,
        "official_leaderboard_evaluation": False,
        "scope": "Complete-input, preselected hardware-bounded LongBench-v2 subset; custom fixed answer-format prompt",
        "source": {
            "repo": "zai-org/LongBench-v2",
            "revision": args.source_revision,
            "license": "apache-2.0",
            "data_sha256": args.source_sha256,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": file_sha(args.model_manifest),
        "model_content_fingerprint_sha256": model["content_fingerprint_sha256"],
        "tokenizer_files": tokenizer_files,
        "vocabulary_size": len(tokenizer),
        "seed": args.seed,
        "selection_policy": "seeded within-domain shuffle, sorted-domain round robin; selected before inference",
        "input_policy": "full document + original question/options; whole chat-template encoding; no cropping, repetition, padding or re-tokenization",
        "max_input_tokens": args.max_input_tokens,
        "max_total_tokens": args.max_total_tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": False,
        },
        "num_samples": len(rows),
        "selected_by_domain": dict(collections.Counter(r["domain"] for r in rows)),
        "eligible_by_domain": dict(
            collections.Counter(c["domain"] for c in candidates if c["eligible"])
        ),
        "candidates": candidates,
        "inputs_file": "inputs.json",
        "inputs_sha256": file_sha(args.output_dir / "inputs.json"),
        "generator_sha256": file_sha(__file__),
    }
    write(args.output_dir / "selection.json", manifest)
    print(
        json.dumps(
            {
                "status": "frozen",
                "samples": len(rows),
                "selected_by_domain": manifest["selected_by_domain"],
                "selection_sha256": file_sha(args.output_dir / "selection.json"),
            },
            ensure_ascii=False,
        )
    )


def load_selection(path):
    path = Path(path).resolve()
    manifest = read(path)
    require(
        manifest.get("kind") == "mpu_task_quality_subset"
        and manifest.get("quality_evaluation") is True,
        "A complete-input quality selection is required; cropped performance workloads are forbidden",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("model_id") == MODEL_ID
        and manifest.get("model_revision") == MODEL_REVISION
        and manifest.get("source", {}).get("revision") == DATASET_REVISION,
        "Quality selection lacks fixed source/model identity",
    )
    target = (path.parent / manifest["inputs_file"]).resolve()
    require(
        target.is_relative_to(path.parent)
        and file_sha(target) == manifest["inputs_sha256"],
        "Frozen quality input file changed or escapes snapshot",
    )
    payload = read(target)
    require(
        payload.get("kind") == "mpu_complete_task_inputs",
        "Wrong frozen quality input format",
    )
    rows = payload["rows"]
    require(
        len(rows) == manifest["num_samples"]
        and len({r["source_id"] for r in rows}) == len(rows),
        "Quality rows are missing or duplicated",
    )
    for row in rows:
        tokens = row["input_ids"]
        require(
            len(tokens) == row["complete_input_tokens"]
            and token_sha(tokens) == row["input_sha256"],
            "Complete input token sequence changed",
        )
        require(
            all(
                type(t) is int and 0 <= t < manifest["vocabulary_size"] for t in tokens
            ),
            "Invalid frozen token ID",
        )
        require(
            isinstance(row.get("expected_answer"), str)
            and row["expected_answer"] in "ABCD"
            and len(row["expected_answer"]) == 1
            and type(row.get("generation_prompt_opens_thinking")) is bool,
            "Malformed task answer or thinking-template contract",
        )
        require(
            len(tokens) <= manifest["max_input_tokens"]
            and len(tokens) + manifest["sampling_params"]["max_new_tokens"]
            <= manifest["max_total_tokens"],
            "Complete input no longer fits recorded budget",
        )
    require(
        manifest["sampling_params"].get("ignore_eos") is False
        and manifest["sampling_params"].get("temperature") == 0,
        "Quality evaluation requires normal EOS and greedy sampling",
    )
    return manifest, rows


def final_answer(text, *, opens_thinking):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif opens_thinking or "<think>" in text:
        return None, "incomplete_thinking"
    marked = re.findall(
        r"\bAnswer\s*:\s*\(?([ABCD])\)?(?![A-Za-z0-9])", text, re.I
    ) + re.findall(r"\\boxed\s*\{\s*([ABCD])\s*\}", text, re.I)
    marked = {v.upper() for v in marked}
    if len(marked) == 1:
        return next(iter(marked)), "parsed_final_answer"
    if len(marked) > 1:
        return None, "ambiguous_final_answer"
    bare = re.fullmatch(r"\s*\(?([ABCD])\)?[.\s]*", text, re.I)
    return (
        (bare.group(1).upper(), "parsed_final_answer")
        if bare
        else (None, "missing_final_answer")
    )


def normalize_response(response, row, max_new_tokens):
    require(
        isinstance(response, dict) and isinstance(response.get("meta_info"), dict),
        "Missing native response metadata",
    )
    meta = response["meta_info"]
    require(
        meta.get("prompt_tokens") == row["complete_input_tokens"],
        "Server cropped or changed the complete prompt",
    )
    require(
        type(meta.get("completion_tokens")) is int
        and 0 < meta["completion_tokens"] <= max_new_tokens,
        "Invalid generated token count",
    )
    reason = meta.get("finish_reason")
    require(
        isinstance(reason, dict) and isinstance(reason.get("type"), str),
        "Missing actual finish reason",
    )
    text = response.get("text")
    require(isinstance(text, str), "Missing generated text")
    records = meta.get("output_token_logprobs")
    require(
        isinstance(records, list) and len(records) == meta["completion_tokens"],
        "Missing native output token IDs",
    )
    tokens = [
        v[1]
        for v in records
        if isinstance(v, (list, tuple))
        and len(v) >= 2
        and type(v[1]) is int
        and v[1] >= 0
        and isinstance(v[0], (float, int))
        and math.isfinite(v[0])
    ]
    require(len(tokens) == len(records), "Malformed native output token IDs")
    answer, state = final_answer(
        text, opens_thinking=row["generation_prompt_opens_thinking"]
    )
    if reason["type"] != "stop":
        state, answer = (
            "length_truncated" if reason["type"] == "length" else "not_normally_stopped"
        ), None
    return {
        "source_id": row["source_id"],
        "domain": row["domain"],
        "input_sha256": row["input_sha256"],
        "expected_answer": row["expected_answer"],
        "predicted_answer": answer,
        "answer_status": state,
        "scorable": answer is not None,
        "correct": answer == row["expected_answer"],
        "finish_reason": reason,
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "output_text": text,
        "output_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_token_ids": tokens,
        "output_token_ids_sha256": token_sha(tokens),
    }


def mpu_parameters(capacity_ratio, device_ids, memory_reserve_bytes, compute_device_id):
    require(
        type(capacity_ratio) in (int, float)
        and math.isfinite(capacity_ratio)
        and capacity_ratio > 0,
        "MPU capacity ratio must be finite and positive",
    )
    require(
        isinstance(device_ids, (list, tuple))
        and 1 <= len(device_ids) <= 4
        and all(type(device) is int and device >= 0 for device in device_ids)
        and len(set(device_ids)) == len(device_ids),
        "MPU device IDs must name one to four distinct nonnegative devices",
    )
    require(
        type(compute_device_id) is int
        and compute_device_id >= 0
        and compute_device_id not in device_ids,
        "MPU devices overlap the compute device or compute identity is invalid",
    )
    require(
        type(memory_reserve_bytes) is int and memory_reserve_bytes >= 0,
        "MPU memory reserve must be a nonnegative integer byte count",
    )
    return {
        "mpu_capacity_ratio": capacity_ratio,
        "mpu_device_ids": list(device_ids),
        "mpu_memory_reserve_bytes": memory_reserve_bytes,
        "compute_device_id": compute_device_id,
    }


def server_policy(
    info,
    variant,
    selection,
    model,
    *,
    capacity_ratio,
    top_k,
    mpu_device_ids=(1,),
    mpu_memory_reserve_bytes=MPU_MEMORY_RESERVE_BYTES,
    compute_device_id=0,
):
    if QUALITY_STAGE is not None and QUALITY_STAGE.get("attn_dp_size") == 4:
        return multirank_quality().policy(
            QUALITY_STAGE,
            info,
            variant,
            selection,
            model,
            capacity_ratio=capacity_ratio,
            top_k=top_k,
            mpu_device_ids=mpu_device_ids,
            mpu_memory_reserve_bytes=mpu_memory_reserve_bytes,
            compute_device_id=compute_device_id,
        )
    require(variant in ALL_VARIANTS, "Unknown quality variant")
    info = info.get("server_args", info)
    require(
        isinstance(info, dict) and info.get("model_path") == model["model_path"],
        "Server model path differs from verified manifest",
    )
    require(
        info.get("tp_size") == info.get("dp_size") == 1,
        "Quality comparison requires a single compute GPU",
    )
    require(
        info.get("disable_radix_cache") is True
        and info.get("enable_hierarchical_cache") is False,
        "Disable radix reuse and HiCache for this storage/algorithm diagnostic",
    )
    require(
        info.get("allow_auto_truncate") is False,
        "Server must not silently truncate complete inputs",
    )
    require(
        info.get("max_total_tokens") == selection["max_total_tokens"],
        "Physical KV budget differs",
    )
    for key, expected in COMMON_CONTROLS.items():
        require(
            key in info and info[key] == expected,
            "Common quality control differs: " + key,
        )
    effective_prefill = info.get("prefill_attention_backend") or info.get(
        "attention_backend"
    )
    require(
        effective_prefill == "flashinfer"
        or (variant == "hbm" and effective_prefill == "flashinfer_quest"),
        "Quality comparison requires dense FlashInfer prefill",
    )
    config = info.get("hisparse_config") or {}
    if isinstance(config, str):
        config = json.loads(config)
    require(isinstance(config, dict), "Invalid resolved HiSparse config")
    if variant == "cpu":
        require(
            info.get("enable_hisparse") is True
            and info.get("decode_attention_backend") == "flashinfer_hisparse",
            "CPU quality variant is not the actual HiSparse path",
        )
        require(
            config.get("algorithm") == "quest"
            and config.get("backing_store", "cpu") == "cpu"
            and config.get("host_to_device_ratio") == capacity_ratio
            and config.get("top_k") == top_k
            and config.get("device_buffer_size") == 4096,
            "Resolved CPU Quest/capacity configuration differs",
        )
    elif variant == "mpu":
        expected = mpu_parameters(
            capacity_ratio,
            mpu_device_ids,
            mpu_memory_reserve_bytes,
            compute_device_id,
        )
        require(
            info.get("enable_hisparse") is True
            and info.get("decode_attention_backend") == "flashinfer_hisparse",
            "MPU quality variant is not the actual HiSparse path",
        )
        require(
            config.get("algorithm") == "quest"
            and config.get("backing_store") == "mpu"
            and config.get("mpu_capacity_ratio") == expected["mpu_capacity_ratio"]
            and config.get("top_k") == top_k
            and config.get("device_buffer_size") == 4096,
            "Resolved MPU Quest/capacity configuration differs",
        )
        actual = mpu_parameters(
            config.get("mpu_capacity_ratio"),
            config.get("mpu_device_ids"),
            config.get("mpu_memory_reserve_bytes"),
            info.get("base_gpu_id"),
        )
        require(actual == expected, "Resolved MPU device/reserve identity differs")
    elif variant == "hbm":
        require(
            info.get("enable_hisparse") is False
            and (
                info.get("attention_backend") == "flashinfer_quest"
                or info.get("decode_attention_backend") == "flashinfer_quest"
            ),
            "HBM quality variant does not preserve Quest",
        )
        require(config.get("top_k") == top_k, "HBM Quest budget differs")
    else:
        require(
            info.get("enable_hisparse") is False
            and info.get("attention_backend") == "flashinfer"
            and info.get("decode_attention_backend") in (None, "flashinfer"),
            "Dense quality variant must use dense FlashInfer attention",
        )
    if variant != "dense":
        require(
            config.get("algorithm") == "quest" and config.get("quest_page_size") == 64,
            "Resolved Quest algorithm/page configuration differs",
        )
    return info


def server_controls(info, variant):
    if info.get("mpu_quality_controls_checked"):
        return multirank_quality().controls(info, variant)
    config = info.get("hisparse_config") or {}
    config = json.loads(config) if isinstance(config, str) else config
    controls = {
        "common": {key: info[key] for key in COMMON_CONTROLS}
        | {"max_total_tokens": info["max_total_tokens"]},
        "selector": (
            {
                key: config.get(key)
                for key in (
                    "algorithm",
                    "top_k",
                    "quest_page_size",
                    "min_sparse_prompt_len",
                )
            }
            if variant != "dense"
            else None
        ),
        "cpu_hot_buffer_size": (
            config.get("device_buffer_size") if variant == "cpu" else None
        ),
    }
    if variant == "mpu":
        controls["mpu_storage"] = {
            "backing_store": config.get("backing_store"),
            "mpu_capacity_ratio": config.get("mpu_capacity_ratio"),
            "mpu_device_ids": config.get("mpu_device_ids"),
            "mpu_memory_reserve_bytes": config.get("mpu_memory_reserve_bytes"),
            "device_buffer_size": config.get("device_buffer_size"),
            "compute_device_id": info.get("base_gpu_id"),
        }
    return controls


def request(base_url, route, data=None, timeout=30, *, json_response=True):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/" + route,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return (
            json.loads(body)
            if json_response
            else {"http_status": response.status, "body": body.decode("utf-8")}
        )


def cleanup_request(base_url, rid, timeout):
    """Explicit abort followed by the scheduler's full-idle flush acknowledgement.

    Abort HTTP 200 alone does not prove that the request stopped. The native
    deferred flush succeeds only at Scheduler.is_fully_idle(), including
    HiSparse staging, so a delayed disconnect cannot overlap the next sample.
    """
    evidence = {"request_id": rid, "status": "started"}
    try:
        evidence["abort_ack"] = request(
            base_url,
            "abort_request",
            {"rid": rid},
            timeout=timeout,
            json_response=False,
        )
        require(
            evidence["abort_ack"].get("http_status") == 200,
            "Abort was not acknowledged",
        )
        evidence["scheduler_idle_ack"] = request(
            base_url,
            f"flush_cache?timeout={timeout:g}",
            {},
            timeout=timeout + 5,
            json_response=False,
        )
        ack = evidence["scheduler_idle_ack"]
        require(
            ack.get("http_status") == 200
            and ack.get("body", "").startswith("Cache flushed."),
            "Scheduler did not confirm complete idle",
        )
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error=repr(error))
    return evidence


def metrics(rows):
    counts = collections.Counter(r["answer_status"] for r in rows)
    require(rows, "Empty quality result set")
    domains = {}
    for domain in sorted({r["domain"] for r in rows}):
        subset = [r for r in rows if r["domain"] == domain]
        domains[domain] = {
            "samples": len(subset),
            "correct": sum(r["correct"] for r in subset),
            "scorable": sum(r["scorable"] for r in subset),
            "accuracy_all_selected": sum(r["correct"] for r in subset) / len(subset),
        }
    return {
        "samples": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy_all_selected": sum(r["correct"] for r in rows) / len(rows),
        "scorable": sum(r["scorable"] for r in rows),
        "answer_status_counts": dict(counts),
        "by_domain": domains,
        "denominator_policy": "Every preselected sample; truncated/ambiguous/errors/not-run count as incorrect and remain listed",
    }


def evaluate(args):
    cleanup_timeout = getattr(args, "cleanup_timeout", 30)
    require(
        math.isfinite(args.request_timeout)
        and args.request_timeout > 0
        and math.isfinite(cleanup_timeout)
        and cleanup_timeout > 0,
        "Request/cleanup timeouts must be finite and positive",
    )
    manifest, inputs = load_selection(args.selection)
    model = model_identity(args.model_manifest)
    require(
        file_sha(args.model_manifest) == manifest["model_manifest_sha256"],
        "Frozen model manifest changed",
    )
    source = read(args.source_identity)
    require(
        re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit", "")),
        "Missing deployed source identity",
    )
    parameters = {"capacity_ratio": args.capacity_ratio, "top_k": args.top_k}
    mpu_policy = {}
    if args.variant == "mpu":
        expected_mpu = mpu_parameters(
            args.capacity_ratio,
            getattr(args, "mpu_device_ids", [1]),
            getattr(args, "mpu_memory_reserve_bytes", MPU_MEMORY_RESERVE_BYTES),
            getattr(args, "compute_device_id", 0),
        )
        parameters.update(expected_mpu)
        mpu_policy = {
            key: expected_mpu[key]
            for key in (
                "mpu_device_ids",
                "mpu_memory_reserve_bytes",
                "compute_device_id",
            )
        }
    info = request(args.base_url, "get_server_info")
    resolved = server_policy(
        info,
        args.variant,
        manifest,
        model,
        capacity_ratio=args.capacity_ratio,
        top_k=args.top_k,
        **mpu_policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "server-info.json", info)
    results = []
    cleanup_failed_source_id = None
    for index, row in enumerate(inputs):
        if cleanup_failed_source_id is not None:
            result = {
                "source_id": row["source_id"],
                "domain": row["domain"],
                "input_sha256": row["input_sha256"],
                "expected_answer": row["expected_answer"],
                "predicted_answer": None,
                "correct": False,
                "scorable": False,
                "answer_status": "not_run_after_cleanup_failure",
                "cleanup_failed_source_id": cleanup_failed_source_id,
                "client_elapsed_seconds": 0,
            }
            results.append(result)
            with (args.output_dir / "results.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            write(args.output_dir / "results.json", {"rows": results})
            continue
        rid = f"quality-{args.variant}-{row['source_id']}"
        request_payload = {
            "input_ids": row["input_ids"],
            "sampling_params": manifest["sampling_params"],
            "return_logprob": True,
            "logprob_start_len": -1,
            "stream": False,
            "rid": rid,
        }
        started = time.monotonic()
        raw = None
        try:
            response = request(
                args.base_url, "generate", request_payload, args.request_timeout
            )
            raw = args.output_dir / f"response-{index:03d}.json"
            write(raw, response)
            result = normalize_response(
                response, row, manifest["sampling_params"]["max_new_tokens"]
            )
            result.update(response_file=raw.name, response_sha256=file_sha(raw))
        except Exception as error:
            result = {
                "source_id": row["source_id"],
                "domain": row["domain"],
                "input_sha256": row["input_sha256"],
                "expected_answer": row["expected_answer"],
                "predicted_answer": None,
                "correct": False,
                "scorable": False,
                "answer_status": "request_or_evidence_error",
                "error": repr(error),
            }
            if raw is not None and raw.is_file():
                result.update(response_file=raw.name, response_sha256=file_sha(raw))
            result["cleanup"] = cleanup_request(args.base_url, rid, cleanup_timeout)
            if result["cleanup"]["status"] != "passed":
                cleanup_failed_source_id = row["source_id"]
        result["client_elapsed_seconds"] = time.monotonic() - started
        results.append(result)
        with (args.output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        write(args.output_dir / "results.json", {"rows": results})
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "done": index + 1,
                    "total": len(inputs),
                    "source_id": row["source_id"],
                    "answer_status": result["answer_status"],
                    "correct": result["correct"],
                }
            ),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "kind": "mpu_task_quality_run",
        "variant": args.variant,
        "status": (
            "failed_cleanup"
            if cleanup_failed_source_id is not None
            else (
                "completed"
                if all(r["scorable"] for r in results)
                else "completed_with_unscorable"
            )
        ),
        "selection_sha256": file_sha(args.selection),
        "selection": str(args.selection.resolve()),
        "model_manifest_sha256": file_sha(args.model_manifest),
        "model_revision": model["revision"],
        "source_identity": source,
        "source_identity_sha256": file_sha(args.source_identity),
        "server_info_sha256": file_sha(args.output_dir / "server-info.json"),
        "resolved_variant_policy_checked": True,
        "resolved_sparse_config": resolved.get("hisparse_config"),
        "verified_model_path": model["model_path"],
        "variant_parameters": parameters,
        "server_controls": server_controls(resolved, args.variant),
        "cleanup_timeout": cleanup_timeout,
        "cleanup_failed_source_id": cleanup_failed_source_id,
        "sampling_params": manifest["sampling_params"],
        "results_file": "results.json",
        "results_sha256": file_sha(args.output_dir / "results.json"),
        "results_jsonl_sha256": file_sha(args.output_dir / "results.jsonl"),
        "metrics": metrics(results),
        "generator_sha256": file_sha(__file__),
        "scope": manifest["scope"],
    }
    write(args.output_dir / "quality-run.json", result)
    print(
        json.dumps(
            {"status": result["status"], "variant": args.variant, **result["metrics"]},
            ensure_ascii=False,
        )
    )


def compare(paths, selection_path, output_dir, *, require_mpu=False):
    baseline_variant = VARIANTS[0]
    selection, selected_rows = load_selection(selection_path)
    runs = {}
    for path in paths:
        path = Path(path).resolve()
        path = path / "quality-run.json" if path.is_dir() else path
        run = read(path)
        require(
            run.get("kind") == "mpu_task_quality_run"
            and run.get("variant") in ALL_VARIANTS,
            "Invalid quality run",
        )
        require(run["variant"] not in runs, "Duplicate backend quality run")
        results_path = (path.parent / run["results_file"]).resolve()
        require(
            results_path.is_relative_to(path.parent)
            and file_sha(results_path) == run["results_sha256"],
            "Quality result file changed",
        )
        rows = read(results_path)["rows"]
        require(
            run["selection_sha256"] == file_sha(selection_path),
            "Run used a different quality selection",
        )
        require(
            run["model_manifest_sha256"] == selection["model_manifest_sha256"]
            and run["model_revision"] == selection["model_revision"]
            and run["sampling_params"] == selection["sampling_params"],
            "Run differs from frozen model/sampling contract",
        )
        server_path = path.parent / "server-info.json"
        require(
            file_sha(server_path) == run["server_info_sha256"],
            "Server configuration evidence changed",
        )
        parameters = run["variant_parameters"]
        mpu_policy = {}
        if run["variant"] == "mpu":
            require(
                all(
                    key in parameters
                    for key in (
                        "mpu_capacity_ratio",
                        "mpu_device_ids",
                        "mpu_memory_reserve_bytes",
                        "compute_device_id",
                    )
                )
                and parameters["mpu_capacity_ratio"] == parameters["capacity_ratio"],
                "Recorded MPU identity/capacity controls are incomplete or differ",
            )
            expected_mpu = mpu_parameters(
                parameters["capacity_ratio"],
                parameters["mpu_device_ids"],
                parameters["mpu_memory_reserve_bytes"],
                parameters["compute_device_id"],
            )
            mpu_policy = {
                key: expected_mpu[key]
                for key in (
                    "mpu_device_ids",
                    "mpu_memory_reserve_bytes",
                    "compute_device_id",
                )
            }
        resolved = server_policy(
            read(server_path),
            run["variant"],
            selection,
            {"model_path": run["verified_model_path"]},
            capacity_ratio=parameters["capacity_ratio"],
            top_k=parameters["top_k"],
            **mpu_policy,
        )
        require(
            server_controls(resolved, run["variant"]) == run["server_controls"],
            "Recorded quality controls differ from actual server evidence",
        )
        require(
            len(rows) == len(selected_rows), "A backend dropped preselected samples"
        )
        jsonl = path.parent / "results.jsonl"
        require(
            file_sha(jsonl) == run["results_jsonl_sha256"]
            and [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
            ]
            == rows,
            "JSON and JSONL evidence differ",
        )
        require(
            metrics(rows) == run["metrics"],
            "Quality summary differs from per-sample evidence",
        )
        for row_index, (row, expected) in enumerate(zip(rows, selected_rows)):
            require(
                all(
                    row.get(k) == expected[k]
                    for k in ("source_id", "domain", "input_sha256", "expected_answer")
                ),
                "Quality result differs from preselected task identity",
            )
            if "response_file" in row:
                raw = (path.parent / row["response_file"]).resolve()
                require(
                    raw.is_relative_to(path.parent)
                    and file_sha(raw) == row["response_sha256"],
                    "Raw model output changed",
                )
                try:
                    normalized = normalize_response(
                        read(raw),
                        expected,
                        selection["sampling_params"]["max_new_tokens"],
                    )
                except ValueError:
                    require(
                        row.get("answer_status") == "request_or_evidence_error"
                        and row.get("correct") is False
                        and row.get("scorable") is False,
                        "Invalid raw response was scored",
                    )
                else:
                    require(
                        all(row.get(k) == value for k, value in normalized.items()),
                        "Recorded quality answer differs from independently normalized raw response",
                    )
            else:
                require(
                    row.get("answer_status")
                    in ("request_or_evidence_error", "not_run_after_cleanup_failure")
                    and row.get("correct") is False
                    and row.get("scorable") is False
                    and row.get("predicted_answer") is None,
                    "A missing response was scored",
                )
            if row.get("answer_status") == "not_run_after_cleanup_failure":
                failed_id = row.get("cleanup_failed_source_id")
                earlier = rows[:row_index]
                require(
                    run.get("status") == "failed_cleanup"
                    and failed_id == run.get("cleanup_failed_source_id")
                    and any(
                        previous.get("source_id") == failed_id
                        and previous.get("answer_status") == "request_or_evidence_error"
                        and previous.get("cleanup", {}).get("status") == "failed"
                        for previous in earlier
                    ),
                    "Not-run row lacks a preceding failed cleanup",
                )
            if row.get("answer_status") == "request_or_evidence_error":
                cleanup = row.get("cleanup", {})
                require(
                    cleanup.get("status") in ("passed", "failed"),
                    "Request error lacks cleanup evidence",
                )
                if cleanup["status"] == "failed":
                    require(
                        run.get("status") == "failed_cleanup"
                        and run.get("cleanup_failed_source_id") == row["source_id"]
                        and all(
                            later.get("answer_status")
                            == "not_run_after_cleanup_failure"
                            for later in rows[row_index + 1 :]
                        ),
                        "Requests continued after failed cleanup",
                    )
        runs[run["variant"]] = (run, rows)
    require(
        set(runs) in (set(VARIANTS), set(ALL_VARIANTS)),
        "All stage baseline paths are required; same-model native sparse HBM/CPU for indexer models",
    )
    require(
        not require_mpu or set(runs) == set(ALL_VARIANTS),
        "MPU comparison requires every stage baseline plus MPU",
    )
    variants = ALL_VARIANTS if "mpu" in runs else VARIANTS
    baseline, dense_rows = runs[baseline_variant]
    for run, rows in runs.values():
        require(
            all(
                run[key] == baseline[key]
                for key in (
                    "selection_sha256",
                    "model_manifest_sha256",
                    "model_revision",
                    "sampling_params",
                    "source_identity_sha256",
                    "source_identity",
                    "verified_model_path",
                )
            ),
            "Compared task runs differ in inputs, model, sampling or source",
        )
        require(
            [(r["source_id"], r["input_sha256"], r["expected_answer"]) for r in rows]
            == [
                (r["source_id"], r["input_sha256"], r["expected_answer"])
                for r in dense_rows
            ],
            "A backend dropped, reordered or changed selected inputs",
        )
        require(
            run["server_controls"]["common"] == baseline["server_controls"]["common"],
            "Compared runs differ in common prefill/precision controls",
        )
    require(
        runs["hbm"][0]["server_controls"]["selector"]
        == runs["cpu"][0]["server_controls"]["selector"],
        "HBM/CPU Quest selector controls differ",
    )
    if "mpu" in runs:
        require(
            runs["mpu"][0]["server_controls"]["selector"]
            == runs["hbm"][0]["server_controls"]["selector"],
            "HBM/MPU Quest selector controls differ",
        )
    comparisons = []
    for i, row in enumerate(dense_rows):
        comparisons.append(
            {
                "source_id": row["source_id"],
                "domain": row["domain"],
                "expected_answer": row["expected_answer"],
                "variants": {
                    v: {
                        k: runs[v][1][i].get(k)
                        for k in (
                            "predicted_answer",
                            "correct",
                            "scorable",
                            "answer_status",
                            "completion_tokens",
                            "output_token_ids_sha256",
                            "output_text_sha256",
                        )
                    }
                    for v in variants
                },
            }
        )
    summary = {
        "schema_version": 1,
        "kind": "mpu_task_quality_comparison",
        "status": (
            "compared_with_cleanup_failure"
            if any(run[0]["status"] == "failed_cleanup" for run in runs.values())
            else "compared"
        ),
        "selection_sha256": baseline["selection_sha256"],
        "model_revision": baseline["model_revision"],
        "scope": baseline["scope"],
        "baseline_variant": baseline_variant,
        "official_leaderboard_evaluation": False,
        "variant_run_status": {v: runs[v][0]["status"] for v in variants},
        "variant_metrics": {v: runs[v][0]["metrics"] for v in variants},
        "paired_effects": {
            v: {
                (
                    "dense_correct_quest_wrong_or_unscorable"
                    if baseline_variant == "dense"
                    else "hbm_correct_storage_wrong_or_unscorable"
                ): sum(
                    runs[baseline_variant][1][i]["correct"] and not r["correct"]
                    for i, r in enumerate(runs[v][1])
                ),
                (
                    "dense_wrong_or_unscorable_quest_correct"
                    if baseline_variant == "dense"
                    else "hbm_wrong_or_unscorable_storage_correct"
                ): sum(
                    not runs[baseline_variant][1][i]["correct"] and r["correct"]
                    for i, r in enumerate(runs[v][1])
                ),
                (
                    "accuracy_delta_vs_dense"
                    if baseline_variant == "dense"
                    else "accuracy_delta_vs_hbm"
                ): runs[v][0]["metrics"]["accuracy_all_selected"]
                - baseline["metrics"]["accuracy_all_selected"],
            }
            for v in variants[1:]
        },
        "hbm_cpu_final_answer_equal": sum(
            runs["hbm"][1][i]["predicted_answer"] is not None
            and runs["hbm"][1][i]["predicted_answer"] == r["predicted_answer"]
            for i, r in enumerate(runs["cpu"][1])
        ),
        "hbm_cpu_generated_output_token_match": sum(
            runs["hbm"][1][i].get("output_token_ids_sha256") is not None
            and runs["hbm"][1][i].get("output_token_ids_sha256")
            == r.get("output_token_ids_sha256")
            for i, r in enumerate(runs["cpu"][1])
        ),
        "generated_output_match_scope": "All emitted token IDs, including outputs stopped at length; task correctness separately requires normal stop and a parsed final answer",
        "rows": comparisons,
    }
    if "mpu" in runs:
        summary["mpu_storage_controls"] = runs["mpu"][0]["server_controls"][
            "mpu_storage"
        ]
        summary["storage_capacity_ratios"] = {
            variant: runs[variant][0]["variant_parameters"]["capacity_ratio"]
            for variant in ("cpu", "mpu")
        }
        for other in ("hbm", "cpu"):
            summary[f"{other}_mpu_final_answer_equal"] = sum(
                runs[other][1][i]["predicted_answer"] is not None
                and runs[other][1][i]["predicted_answer"] == row["predicted_answer"]
                for i, row in enumerate(runs["mpu"][1])
            )
            summary[f"{other}_mpu_generated_output_token_match"] = sum(
                runs[other][1][i].get("output_token_ids_sha256") is not None
                and runs[other][1][i].get("output_token_ids_sha256")
                == row.get("output_token_ids_sha256")
                for i, row in enumerate(runs["mpu"][1])
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    write(output_dir / "quality-comparison.json", summary)
    lines = [
        "# Complete-input Quest task-quality diagnostic",
        "",
        summary["scope"],
        "",
        "Normal EOS; every preselected sample stays in the denominator. Length truncation is unscorable and counts as incorrect, not a valid final answer.",
        "",
        "| Backend | Correct / selected | Accuracy | Scorable |",
        "| --- | --- | --- | --- |",
    ]
    for v in variants:
        m = summary["variant_metrics"][v]
        lines.append(
            f"| {v} | {m['correct']} / {m['samples']} | {m['accuracy_all_selected']:.3%} | {m['scorable']} |"
        )
    lines.extend(
        [
            "",
            "| Source ID | Domain | Gold | " + " | ".join(variants) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in variants) + " |",
        ]
    )
    for row in comparisons:
        cells = [row["source_id"], row["domain"], row["expected_answer"]]
        cells.extend(
            f"{row['variants'][v]['predicted_answer'] or '-'} ({row['variants'][v]['answer_status']})"
            for v in variants
        )
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "quality-comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "variant_metrics": summary["variant_metrics"],
            },
            ensure_ascii=False,
        )
    )


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument(
        "--stage", type=Path, default=Path(__file__).with_name("stage.json")
    )
    control, argv = pre.parse_known_args(argv)
    configure_stage(read(control.stage))
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("freeze", allow_abbrev=False)
    create.add_argument("--source-json", type=Path, required=True)
    create.add_argument("--source-sha256", required=True)
    create.add_argument("--source-revision", default=DATASET_REVISION)
    create.add_argument("--tokenizer", required=True)
    create.add_argument("--model-manifest", type=Path, required=True)
    create.add_argument("--num-samples", type=int, default=24)
    create.add_argument("--seed", type=int, default=1729)
    create.add_argument("--max-input-tokens", type=int, default=32768)
    create.add_argument("--max-new-tokens", type=int, default=8192)
    create.add_argument("--max-total-tokens", type=int, default=65536)
    create.add_argument("--output-dir", type=Path, required=True)
    execute = commands.add_parser("evaluate", allow_abbrev=False)
    execute.add_argument("--selection", type=Path, required=True)
    execute.add_argument("--model-manifest", type=Path, required=True)
    execute.add_argument("--source-identity", type=Path, required=True)
    execute.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    execute.add_argument("--base-url", default="http://127.0.0.1:30000")
    execute.add_argument(
        "--capacity-ratio",
        type=int,
        default=QUALITY_STAGE["defaults"]["cpu_capacity_ratio"],
    )
    execute.add_argument(
        "--top-k",
        type=int,
        default=QUALITY_STAGE["variants"]["cpu"]["hisparse_config"]["top_k"],
    )
    execute.add_argument(
        "--mpu-device-ids",
        type=int,
        nargs="+",
        default=QUALITY_STAGE["variants"]["mpu"]["hisparse_config"].get(
            "mpu_device_ids", [1]
        ),
    )
    execute.add_argument(
        "--mpu-memory-reserve-bytes", type=int, default=MPU_MEMORY_RESERVE_BYTES
    )
    execute.add_argument("--compute-device-id", type=int, default=0)
    execute.add_argument("--request-timeout", type=float, default=900)
    execute.add_argument("--cleanup-timeout", type=float, default=30)
    execute.add_argument("--output-dir", type=Path, required=True)
    aggregate = commands.add_parser("compare", allow_abbrev=False)
    aggregate.add_argument("--runs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--require-mpu", action="store_true")
    aggregate.add_argument("--selection", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        freeze(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        compare(
            args.runs,
            args.selection,
            args.output_dir,
            require_mpu=args.require_mpu,
        )


if __name__ == "__main__":
    main()
