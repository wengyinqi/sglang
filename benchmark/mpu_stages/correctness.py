"""CPU-DDR HiSparse reference versus MPU with the identical sparse selector."""

import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mpu_stage_runner", Path(__file__).with_name("run.py")
)
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)
COMPARE = RUN.load("comparison")
LIFECYCLE = RUN.load("lifecycle")


def make_plan(stage, args):
    if getattr(args, "lifecycle", False) and args.output_len < 8:
        raise ValueError("Lifecycle acceptance requires at least 8 output tokens")
    config = RUN.STAGE.config_for(
        stage, args, stage["variant"], args.capacity_ratios[0]
    )
    hot = config.get("device_buffer_size", 4096) * stage.get(
        "sparse_context_multiplier", 1
    )
    if min(args.input_lens) <= hot:
        raise ValueError(
            "Correctness long prompts must exceed the hot buffer in original token units"
        )
    if args.chunked_prefill_size >= min(args.input_lens):
        raise ValueError(
            "Correctness requires chunked prefill below the shortest long prompt"
        )
    page = stage.get("page_size", 64)
    short = [page + 1, min(hot + 1, max(args.input_lens))]
    rounds = [args.input_lens, short, args.input_lens]
    if stage.get("attn_dp_size", 1) > 1:
        # Round-robin assigns one independent request to each attention owner.
        rounds = [
            (lengths * math.ceil(4 / len(lengths)))[: max(4, len(lengths))]
            for lengths in rounds
        ]
    peak = max(
        sum(math.ceil((n + args.output_len) / page) * page for n in lengths)
        for lengths in rounds
    )
    if peak > args.max_total_tokens:
        raise ValueError(
            f"Simultaneous correctness round requires at least {peak} requested token slots"
        )
    args.concurrencies = [max(len(lengths) for lengths in rounds)]
    groups = []
    for eager in (True,) if args.disable_cuda_graph else (True, False):
        for variant, ratio in RUN.STAGE.variants(stage, args):
            gid = f"{variant}-{'eager' if eager else 'graph'}" + (
                f"-ratio{ratio:g}" if ratio is not None else ""
            )
            groups.append(
                {
                    "id": gid,
                    "variant": variant,
                    "ratio": ratio,
                    "eager": eager,
                    "status": "planned",
                    "rounds": rounds,
                    "lifecycle_required": getattr(args, "lifecycle", False),
                    "lifecycle_cases": (
                        ["abort_prefill", "abort_decode", "retract_decode"]
                        if getattr(args, "lifecycle", False)
                        else []
                    ),
                    "planned_peak_original_token_slots": peak,
                    "trace_dir": str(args.output_dir / f"{gid}-trace"),
                    "server_command": RUN.STAGE.server_command(
                        stage, args, variant, max(args.input_lens), ratio, eager=eager
                    ),
                }
            )
    if groups[0]["variant"] != "cpu":
        raise ValueError("The first acceptance reference must be CPU-DDR HiSparse")
    return groups


def execute(stage, args, manifest, path):
    RUN.require_cuda(stage, args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code="--trust-remote-code" in stage.get("server_args", []),
    )
    base = tokenizer.encode(
        "Consider these records. "
        + " ".join(f"Record {i} has value {i * 13 % 97}." for i in range(256)),
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("Tokenizer produced no prompt tokens")
    reference, physical_reference = None, None
    for group in manifest["groups"]:
        process = None
        try:
            with (args.output_dir / f"{group['id']}-server.log").open(
                "w", encoding="utf-8"
            ) as log:
                process = RUN.COMMON.start_server(
                    args, group["server_command"], log, trace_dir=group["trace_dir"]
                )
                RUN.COMMON.wait_ready(process, args.port, args.startup_timeout)
                info = RUN.COMMON.request_json(args.port, "get_server_info")
                RUN.COMMON.write_manifest(
                    args.output_dir / f"{group['id']}-server-info.json", info
                )
                if RUN.decode_graph_enabled(info) == group["eager"]:
                    raise ValueError(
                        "Actual CUDA graph setting differs from the acceptance case"
                    )
                group["capacity"] = RUN.wait_capacities(
                    group["trace_dir"], stage, args, group["variant"], group["ratio"]
                )
                point_memory = RUN.enforce_point_memory(None, group["capacity"])
                if group["variant"] != "hbm":
                    physical_reference = RUN.enforce_fixed_compute(
                        physical_reference, group["capacity"]
                    )
                records = []
                group["round_idle_capacity"] = []
                for index, lengths in enumerate(group["rounds"]):
                    inputs = [(base * math.ceil(n / len(base)))[:n] for n in lengths]
                    outputs = RUN.COMMON.request_json(
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
                    RUN.COMMON.write_manifest(
                        args.output_dir / f"{group['id']}-round{index}.json",
                        {"input_ids": inputs, "responses": outputs},
                    )
                    records.append(
                        COMPARE.normalize_outputs(outputs, lengths, args.output_len)
                    )
                    ids = [output["meta_info"].get("id") for output in outputs]
                    if any(not isinstance(rid, str) or not rid for rid in ids) or len(
                        set(ids)
                    ) != len(ids):
                        raise ValueError(
                            "Correctness responses require unique native request IDs"
                        )
                    group["round_idle_capacity"].append(
                        RUN.wait_capacities(
                            group["trace_dir"],
                            stage,
                            args,
                            group["variant"],
                            group["ratio"],
                            request_ids=ids,
                            require_idle=True,
                        )
                    )
                    if stage.get("attn_dp_size", 1) > 1:
                        traces = RUN.ANALYZE._dp_traces(
                            RUN.ANALYZE.load_traces(group["trace_dir"])
                        )
                        owners = RUN.ANALYZE._request_owners(traces, set(ids))
                        if set(owners.values()) != set(range(4)):
                            raise ValueError(
                                "Correctness round must exercise all four Attention-DP owners"
                            )
                        group.setdefault("round_attention_owners", []).append(owners)
                    if group["variant"] != "hbm":
                        RUN.enforce_fixed_compute(
                            physical_reference, group["round_idle_capacity"][-1]
                        )
                        RUN.enforce_point_memory(
                            point_memory, group["round_idle_capacity"][-1]
                        )
                if reference is None:
                    reference = records
                group["comparison_to_cpu_hisparse"] = COMPARE.compare(
                    reference, records, args.logprob_atol
                )
                group["slot_reuse_comparison"] = COMPARE.compare(
                    [records[0]], [records[2]], args.logprob_atol
                )
                if group["lifecycle_required"]:
                    length = max(args.input_lens)
                    inputs = (base * math.ceil(length / len(base)))[:length]
                    LIFECYCLE.run_cases(RUN, COMPARE, stage, args, group, inputs)
                group["status"] = "success"
        except (OSError, ValueError, RuntimeError) as error:
            group.update(status="failed", error=str(error))
            if reference is None:
                break
        finally:
            if process is not None:
                RUN.COMMON.stop_server(process)
            RUN.COMMON.write_manifest(path, manifest)
    for group in manifest["groups"]:
        if group["status"] == "planned":
            group["status"] = "not_run_after_reference_failure"
    return all(group["status"] == "success" for group in manifest["groups"])


def main():
    stage, args = RUN.STAGE.parse_args(correctness=True)
    groups = make_plan(stage, args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = RUN.new_manifest(stage, args, groups, "correctness")
    path = args.output_dir / "manifest.json"
    RUN.COMMON.write_manifest(path, manifest)
    if args.execute:
        try:
            manifest["status"] = (
                "success" if execute(stage, args, manifest, path) else "failed"
            )
        except (ImportError, RuntimeError, ValueError, KeyboardInterrupt) as error:
            manifest.update(status="failed", error=str(error))
        finally:
            RUN.COMMON.write_manifest(path, manifest)
    print(path)
    return int(manifest["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
