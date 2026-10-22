"""Opt-in end-to-end greedy-token and logprob equivalence, Quest HBM versus MPU.

Runs every mode in a fresh server, with three request rounds covering long
prefill, short/long mixtures, page boundaries, and recycled request/KV slots.
Both eager and CUDA graph configurations are compared to the eager HBM output.
"""

import importlib.util
import math
import time
from datetime import datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mpu_runner", Path(__file__).with_name("run.py")
)
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)


def prompt_rounds(args):
    buffer = args.device_buffer_size
    long = [4 * buffer + 1, 4 * buffer + 64, 6 * buffer + 1]
    return [long, [65, buffer + 1], long]


def make_plan(args):
    rounds = prompt_rounds(args)
    longest = max(max(lengths) for lengths in rounds)
    # All requests of one round must fit simultaneously, including allocator
    # page rounding; otherwise this would silently test retraction/serialization.
    peak = max(
        sum(math.ceil((n + args.output_len) / 64) * 64 for n in lengths)
        for lengths in rounds
    )
    if args.max_total_tokens < peak:
        raise ValueError(
            f"Correctness round needs at least {peak} physical compute slots"
        )
    if args.num_prompts < 3:
        raise ValueError("Correctness runs three concurrent requests")
    if args.chunked_prefill_size >= min(rounds[0]):
        raise ValueError(
            "Use chunked_prefill_size below the shortest long prompt to cover chunk boundaries"
        )
    args.concurrencies = [3]
    groups = []
    for eager in (True, False):
        for mode in ("quest", "mpu"):
            for ratio in sorted(set(args.capacity_ratios)) if mode == "mpu" else [None]:
                if (
                    ratio is not None
                    and math.ceil(args.max_total_tokens * ratio) < peak
                ):
                    raise ValueError(
                        "Correctness MPU capacity must cover all concurrent KV tokens"
                    )
                gid = f"{mode}-{'eager' if eager else 'graph'}" + (
                    f"-ratio{ratio:g}" if ratio is not None else ""
                )
                groups.append(
                    {
                        "id": gid,
                        "mode": mode,
                        "ratio": ratio,
                        "eager": eager,
                        "status": "planned",
                        "rounds": rounds,
                        "planned_peak_page_rounded_slots": peak,
                        "server_command": RUN.server_command(
                            args, mode, longest, ratio, eager=eager
                        ),
                    }
                )
    return groups


def normalize_outputs(outputs, lengths, output_len):
    if not isinstance(outputs, list) or len(outputs) != len(lengths):
        raise ValueError("Server must return exactly every request")
    normalized = []
    for output, prompt_len in zip(outputs, lengths):
        meta = output.get("meta_info") if isinstance(output, dict) else None
        if not isinstance(meta, dict):
            raise ValueError("Server response is missing request metadata")
        records = meta.get("output_token_logprobs", [])
        if (
            meta.get("prompt_tokens") != prompt_len
            or meta.get("completion_tokens") != output_len
        ):
            raise ValueError("Actual request token counts differ from planned lengths")
        if len(records) != output_len:
            raise ValueError("Missing output token logprobs")
        tokens, logprobs = [], []
        for record in records:
            if not isinstance(record, (list, tuple)) or len(record) < 2:
                raise ValueError("Malformed output token logprob record")
            logprob, token = record[:2]
            if (
                type(token) is not int
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
            ):
                raise ValueError("Non-finite logprob or malformed token ID")
            tokens.append(token)
            logprobs.append(logprob)
        normalized.append({"tokens": tokens, "logprobs": logprobs})
    return normalized


def wait_mpu_release(args, ratio, timeout=10.0):
    """HTTP completion can precede scheduler cleanup; require eventual release."""
    deadline = time.monotonic() + timeout
    while True:
        capacity = RUN.verify_capacity(
            RUN.request_json(args.port, "get_server_info"), args, "mpu", ratio
        )
        if (
            capacity.get("reserved_mpu_slots") == 0
            and capacity.get("used_mpu_slots") == 0
        ):
            return capacity
        if time.monotonic() >= deadline:
            raise ValueError(
                f"MPU allocations/reservations remain after request completion: {capacity}"
            )
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def compare(reference, actual, atol):
    if len(reference) != len(actual):
        raise ValueError("Missing correctness rounds")
    max_error = 0.0
    for round_id, (left, right) in enumerate(zip(reference, actual)):
        if len(left) != len(right):
            raise ValueError(f"Request count changed in round {round_id}")
        for request_id, (expected, observed) in enumerate(zip(left, right)):
            if expected["tokens"] != observed["tokens"]:
                divergence = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(expected["tokens"], observed["tokens"])
                        )
                        if a != b
                    ),
                    min(len(expected["tokens"]), len(observed["tokens"])),
                )
                raise ValueError(
                    f"Token divergence at round={round_id} request={request_id} token={divergence}; inspect saved raw responses"
                )
            if len(expected["logprobs"]) != len(observed["logprobs"]):
                raise ValueError("Missing comparison logprobs")
            for token_id, (a, b) in enumerate(
                zip(expected["logprobs"], observed["logprobs"])
            ):
                delta = abs(a - b)
                max_error = max(max_error, delta)
                if not math.isfinite(delta) or delta > atol:
                    raise ValueError(
                        f"Logprob mismatch at round={round_id} request={request_id} token={token_id}: abs_error={delta} > {atol}"
                    )
    return {
        "exact_greedy_token_match": True,
        "max_abs_logprob_error": max_error,
        "logprob_atol": atol,
    }


def execute(args, manifest, path):
    RUN.require_cuda(args)
    from transformers import AutoTokenizer

    revision = args.revision or (
        RUN.BASE.REVISION if args.model == RUN.BASE.MODEL else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    base = tokenizer.encode(
        "Consider these records. "
        + " ".join(f"Record {i} has value {i * 13 % 97}." for i in range(256)),
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("Tokenizer produced no prompt tokens")
    reference = None
    for group in manifest["groups"]:
        process = None
        try:
            with (args.output_dir / f'{group["id"]}-server.log').open(
                "w", encoding="utf-8"
            ) as log:
                process = RUN.start_server(args, group["server_command"], log)
                RUN.BASE.wait_ready(process, args.port, args.startup_timeout)
                info = RUN.request_json(args.port, "get_server_info")
                RUN.BASE.write_manifest(
                    args.output_dir / f'{group["id"]}-server-info.json', info
                )
                if info.get("disable_cuda_graph") is not group["eager"]:
                    raise ValueError(
                        "Server CUDA graph setting differs from the acceptance case"
                    )
                if info.get("disable_overlap_schedule") is not True:
                    raise ValueError(
                        "Correctness acceptance requires the configured serial scheduler"
                    )
                group["capacity"] = RUN.verify_capacity(
                    info, args, group["mode"], group["ratio"]
                )
                records = []
                group["completed_round_capacities"] = []
                for index, lengths in enumerate(group["rounds"]):
                    inputs = [(base * math.ceil(n / len(base)))[:n] for n in lengths]
                    output = RUN.request_json(
                        args.port,
                        "generate",
                        {
                            "input_ids": inputs,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": args.output_len,
                                "ignore_eos": True,
                            },
                            "return_logprob": True,
                            "stream": False,
                        },
                        timeout=args.benchmark_timeout,
                    )
                    RUN.BASE.write_manifest(
                        args.output_dir / f'{group["id"]}-round{index}.json',
                        {"input_ids": inputs, "responses": output},
                    )
                    records.append(normalize_outputs(output, lengths, args.output_len))
                    if group["mode"] == "mpu":
                        group["completed_round_capacities"].append(
                            wait_mpu_release(args, group["ratio"])
                        )
                if reference is None:
                    reference = records
                group["comparison"] = compare(reference, records, args.logprob_atol)
                group["slot_reuse_comparison"] = compare(
                    [records[0]], [records[2]], args.logprob_atol
                )
                group["status"] = "success"
        except (OSError, ValueError, RuntimeError) as error:
            group.update(status="failed", error=str(error))
            # Without the HBM reference, other comparisons cannot be accepted.
            if reference is None:
                break
        finally:
            if process is not None:
                RUN.BASE.stop_server(process)
            RUN.BASE.write_manifest(path, manifest)
    return all(group["status"] == "success" for group in manifest["groups"])


def main():
    parser = RUN.parser()
    parser.description = __doc__
    parser.add_argument("--logprob-atol", type=float, default=0.02)
    args = parser.parse_args()
    RUN.validate(args)
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        raise ValueError("logprob-atol must be finite and nonnegative")
    args.output_dir = args.output_dir.resolve()
    groups = make_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "Exact greedy tokens and selected-token logprobs against Quest HBM; no task-accuracy or performance claim",
        "runtime": RUN.BASE.runtime_info(),
        "groups": groups,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    path = args.output_dir / "manifest.json"
    RUN.BASE.write_manifest(path, manifest)
    if args.execute:
        try:
            manifest["status"] = (
                "success" if execute(args, manifest, path) else "failed"
            )
        except (ImportError, RuntimeError, KeyboardInterrupt) as error:
            manifest.update(status="failed", error=str(error))
        finally:
            RUN.BASE.write_manifest(path, manifest)
    print(path)
    return int(manifest["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
