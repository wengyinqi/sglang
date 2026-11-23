"""Run isolated Qwen memory diagnostics against frozen token IDs.

The diagnostic deliberately synchronizes and records allocator history. Its
latencies must never be used as serving-performance evidence. Native model
families on stages03--06 use the shared observer/contract with their own launch
recipes; this runner accepts only the verified single-compute Qwen recipe.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(options):
    common, stage_module = load("common"), load("stage")
    target = options.output_dir
    target.mkdir(parents=True, exist_ok=False)
    workload = json.loads(options.workload.read_text(encoding="utf-8"))
    data = workload["files"][str(options.input_len)]
    tokens_path = options.workload.parent / data["path"]
    if sha(tokens_path) != data["sha256"]:
        raise ValueError("Frozen workload tokens differ from their manifest")
    rows = json.loads(tokens_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("requests", rows.get("rows"))
    row = rows[options.row]
    tokens = row["input_ids"] if isinstance(row, dict) else row
    if len(tokens) != options.input_len or not all(
        type(value) is int for value in tokens
    ):
        raise ValueError("Malformed frozen input IDs")
    input_sha = hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()
    source = json.loads((common.REPO / ".mpu-source.json").read_text(encoding="utf-8"))
    stage, args = stage_module.parse_args(
        [
            "--model",
            options.model,
            "--workload",
            str(options.workload),
            "--input-lens",
            str(options.input_len),
            "--output-len",
            "1024",
            "--max-total-tokens",
            "65536",
            "--concurrencies",
            "8",
            "--chunked-prefill-size",
            "8192",
            "--prefill-max-requests",
            "1",
            "--mem-fraction-static",
            "0.85",
            "--top-k",
            "2048",
            "--device-buffer-size",
            "4096",
            "--seed",
            "1729",
            "--capacity-ratios",
            "4",
            "--port",
            str(options.port),
            "--output-dir",
            str(target / "plan"),
        ]
    )
    if stage["tp_size"] != 1 or "qwen" not in stage.get("model_family", "qwen").lower():
        raise ValueError("Memory probe requires the Qwen single-compute recipe")
    variant = "hbm" if options.variant == "dense" else options.variant
    command = stage_module.server_command(
        stage,
        args,
        variant,
        options.input_len,
        4 if variant == "cpu" else None,
        eager=options.eager,
    )
    if options.variant == "dense":
        index = command.index("--attention-backend")
        if command[index + 1] != "flashinfer_quest":
            raise ValueError("Unexpected Dense reference recipe")
        command[index + 1] = "flashinfer"
        if "--hisparse-config" in command:
            index = command.index("--hisparse-config")
            del command[index : index + 2]
    context = {
        "family": "qwen",
        "variant": options.variant,
        "model_revision": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
        "weight_format": "bfloat16",
        "parallelism": "tp1_dp1_ep1",
        "source_revision": source.get("source_commit", source.get("commit")),
        "source_identity": source,
        "context_len": options.input_len + 1026,
        "request_slots": 8,
        "request_set_fingerprint": input_sha,
        "output_len": options.output_len,
        "cuda_graph": not options.eager,
        "capture_group_id": target.name,
    }
    receipt = {
        "status": "running",
        "diagnostic_only": True,
        "eligible_for_formal_performance": False,
        "context": context,
        "workload_sha256": sha(options.workload),
        "tokens_file_sha256": sha(tokens_path),
        "row": options.row,
        "command": command,
    }
    common.write_manifest(target / "receipt.json", receipt)
    server = None
    with (target / "server.log").open("x", encoding="utf-8") as stream:
        try:
            server = common.start_server(
                args,
                command,
                stream,
                env_updates={
                    "SGLANG_EXPERIMENT_MEMORY_DIR": str(target / "memory"),
                    "SGLANG_EXPERIMENT_MEMORY_CONTEXT": json.dumps(context),
                    "SGLANG_EXPERIMENT_MEMORY_DECODE_SAMPLES": "1,8,64,127",
                },
            )
            common.wait_ready(server, args.port, args.startup_timeout)
            info = common.request_json(args.port, "get_server_info")
            common.write_manifest(target / "server-info.json", info)
            response = common.request_json(
                args.port,
                "generate",
                {
                    "input_ids": tokens,
                    "rid": "memory-frozen-row" + str(options.row),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": options.output_len,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                    "stream": False,
                },
                timeout=1800,
            )
            common.write_manifest(target / "response.json", response)
            if (
                response.get("meta_info", {}).get("completion_tokens")
                != options.output_len
            ):
                raise ValueError("Generation did not complete the fixed output budget")
            path = target / "memory/memory-rank0.jsonl"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        "Diagnostic server exited before the idle receipt"
                    )
                records = [json.loads(line) for line in path.read_text().splitlines()]
                full_decode = [
                    index
                    for index, record in enumerate(records)
                    if record["type"] == "memory_snapshot"
                    and record["context"]["phase"] == "decode"
                    and record["context"]["phase_steps"] >= options.output_len - 1
                    and record["context"].get("seq_lens", [0])[0] >= options.input_len
                    and record["context"]["final_phase_snapshot"]
                ]
                if full_decode and any(
                    record["type"] == "memory_wave_complete"
                    for record in records[full_decode[-1] + 1 :]
                ):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Missing final idle/completion memory evidence")
            # Startup warmup may produce an earlier complete wave. Require the
            # target output-length decode and idle records before accepting.
            decode = [
                record
                for record in records
                if record["type"] == "memory_snapshot"
                and record["context"]["phase"] == "decode"
            ]
            if (
                not decode
                or max(record["context"]["phase_steps"] for record in decode)
                < options.output_len - 1
            ):
                raise ValueError(
                    "Memory evidence lacks the full diagnostic decode wave"
                )
            receipt.update(
                status="passed",
                response_sha256=sha(target / "response.json"),
                memory_sha256=sha(path),
                observed_decode_steps=max(
                    record["context"]["phase_steps"] for record in decode
                ),
            )
        except BaseException as error:
            receipt.update(status="failed", error=repr(error))
            raise
        finally:
            if server is not None:
                common.stop_server(server)
            common.write_manifest(target / "receipt.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--variant", choices=("dense", "hbm", "cpu"), required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--port", type=int, default=30902)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
