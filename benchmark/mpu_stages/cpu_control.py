"""Linux process-level control checks, with real Gloo and production definitions.

AST isolation avoids importing the CUDA model stack. Request transport uses pipes;
CUDA stream completion, model batches and KV contents are explicit test boundaries.
This does not start a model server, run DeepEP, or validate GPU/MPU data movement.
"""

import argparse
import dataclasses
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent


def fixtures():
    spec = importlib.util.spec_from_file_location(
        "mpu_cpu_fixtures", HERE / "cpu_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def controls(world):
    import torch

    f = fixtures()
    mode = f.definition(
        "python/sglang/srt/model_executor/forward_batch_info.py",
        "ForwardMode",
        {"IntEnum": IntEnum, "auto": auto},
    )
    path = "python/sglang/srt/managers/scheduler_components/dp_attn.py"
    modern = (f.ROOT / path).exists()
    if not modern:
        path = "python/sglang/srt/managers/scheduler_dp_attn_mixin.py"
    group = NS(active_ranks_cpu=torch.ones(world, dtype=torch.int64))
    scope = {
        "dataclass": dataclasses.dataclass,
        "ForwardMode": mode,
        "_ENABLE_METRICS_DP_ATTENTION": False,
        "time": time,
        "get_tp_group": lambda: group,
        "get_parallel": lambda: NS(tp_group=group),
    }
    if modern:
        scope["exchange_sync_rows"] = f.definition(
            path,
            "exchange_sync_rows",
            {
                "all_gather_single": torch.distributed.all_gather_into_tensor,
            },
        )
    cls = f.definition(path, "MLPSyncBatchInfo", scope)
    return f, mode, cls, modern


def sync(cls, mode, modern, world, tokens, phase):
    import torch

    kwargs = dict(
        tp_size=1,
        cp_size=1,
        num_tokens=tokens,
        num_tokens_for_logprob=tokens,
        is_extend_in_batch=phase == "EXTEND",
        local_can_run_tbo=False,
        local_forward_mode=mode[phase].value,
    )
    if modern:
        kwargs.update(
            num_dp_ranks=world,
            can_run_decode_cuda_graph=phase != "EXTEND",
            can_run_draft_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
        )
    else:
        kwargs.update(dp_size=world, can_cuda_graph=phase != "EXTEND")
    info = cls(**kwargs)
    info.all_gather("cpu", torch.distributed.group.WORLD)
    rows = info.tp0_info_cpu if modern else info.tp0_info
    return {
        "tokens": info.global_num_tokens,
        "modes": rows[:, 5].tolist(),
        "any_prefill": info.is_extend_in_batch,
        "decode_graph_vote": info.can_run_decode_cuda_graph
        if modern
        else info.can_cuda_graph,
    }


def worker(rank, world, rendezvous, pipe, release, output, fault, timeout):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    record = {"rank": rank, "pid": os.getpid(), "steps": [], "status": "failed"}
    initialized = False
    try:
        f, mode, cls, modern = controls(world)
        record["source_sha256"] = f.SOURCES
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=timeout),
        )
        initialized = True
        record.update(backend=dist.get_backend(), world_size=dist.get_world_size())
        pipe.send({"ready": rank})
        pipe.recv()  # Parent releases all ranks only after group initialization.
        if fault:
            if rank == world - 1:
                if fault == "exit":
                    os._exit(17)
                record["status"] = "withheld_collective"
                release.wait(timeout * 4)
            else:
                started = time.monotonic()
                try:
                    sync(cls, mode, modern, world, 1, "DECODE")
                except RuntimeError as error:
                    record.update(
                        status="collective_failed",
                        error=str(error),
                        elapsed_seconds=time.monotonic() - started,
                    )
                else:
                    raise AssertionError("Collective accepted a missing rank")
            return

        pool = f.make_pool()
        mapping = torch.full((4, 512), -1, dtype=torch.int64)
        allocated = torch.zeros(4, dtype=torch.int64)
        audit, owned = [], {}
        coordinator = f.coordinator(pool, mapping, allocated, audit)
        while True:
            command = pipe.recv()
            if command["op"] == "stop":
                break
            if command["op"] == "admit":
                rid = command["rid"]
                assert command["owner"] == rank and rid not in owned
                row = len(owned)
                assert row < 4 and allocated[row] == 0
                got = pool.alloc(64)
                assert got is not None
                mapping[row, :64], allocated[row] = got, 64
                owned[rid] = row
                continue
            if command["op"] in ("abort", "retract"):
                row = owned.pop(command["rid"])
                audit.clear()
                coordinator._has_pending_backup = True
                req = NS(
                    req_pool_idx=row,
                    kv_allocated_len=64,
                    hisparse_staging=False,
                    kv=NS(req_pool_idx=row, kv_allocated_len=64),
                )
                if command["op"] == "retract":
                    coordinator.retract_req(req)
                else:
                    coordinator.request_finished(req)
                assert audit.index("wait_producer") < audit.index("free_backing")
                assert audit.index("wait_backup") < audit.index("free_backing")
                assert (mapping[row] == -1).all()
                allocated[row] = 0  # Legacy Quest does not own this paged array.
                record.setdefault("releases", []).append(
                    {"op": command["op"], "audit": audit[:]}
                )
                continue
            assert command["op"] == "step"
            assert bool(owned) == (command["phase"] != "IDLE")
            result = sync(cls, mode, modern, world, command["tokens"], command["phase"])
            if not owned:
                assert pool.available_size() == pool.size
                full = pool.alloc(pool.size)
                assert sorted(full.tolist()) == list(range(pool.size))
                assert pool.alloc(64) is None
                pool.free(full)
            result.update(
                step=command["step"], owners=sorted(owned), free=pool.available_size()
            )
            record["steps"].append(result)
            pipe.send(result)
        assert not owned and pool.available_size() == pool.size
        record["status"] = "passed"
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        Path(output, f"rank-{rank}.json").write_text(json.dumps(record, indent=2))
        if initialized:
            dist.destroy_process_group()


class Transport:
    def __init__(self, connection, owner):
        self.connection, self.owner = connection, owner

    def send_pyobj(self, request):
        self.connection.send({"op": "admit", "rid": request.rid, "owner": self.owner})


def router(pipes):
    f = fixtures()
    path = "python/sglang/srt/managers/data_parallel_controller.py"
    scope = {
        "logger": logging.getLogger(__name__),
        "sock_send": lambda sock, req: sock.send_pyobj(req),
    }
    cls = type(
        "ProductionRouter",
        (),
        {
            name: f.definition(path, name, scope, "DataParallelController")
            for name in ("maybe_external_dp_rank_routing", "round_robin_scheduler")
        },
    )
    instance = cls()
    instance.workers = [Transport(pipe, rank) for rank, pipe in enumerate(pipes)]
    instance._active_workers = list(range(len(pipes)))
    instance.status = [True] * len(pipes)
    instance.round_robin_counter = 0
    return instance


def receive(pipe, seconds=45):
    if not pipe.poll(seconds):
        raise TimeoutError("CPU worker did not report within its deadline")
    return pipe.recv()


def validate(records, world, fault, exits, timeout):
    assert set(exits) == set(range(world))
    assert all(r["backend"] == "gloo" and r["world_size"] == world for r in records)
    if fault:
        assert len(records) == world - (fault == "exit")
        for r in records:
            if r["rank"] == world - 1:
                assert r["status"] == "withheld_collective"
            else:
                assert r["status"] == "collective_failed" and r["error"]
                assert r["elapsed_seconds"] < timeout * 4
                if fault == "timeout":
                    assert r["elapsed_seconds"] >= timeout * 0.7
        assert exits[world - 1] == (17 if fault == "exit" else 0)
        assert all(code == 0 for rank, code in exits.items() if rank != world - 1)
        return
    assert len(records) == world and len({r["pid"] for r in records}) == world
    assert all(code == 0 for code in exits.values())
    for r in records:
        assert r["status"] == "passed" and len(r["steps"]) == 6
        assert {e["op"] for e in r["releases"]} == {"abort", "retract"}
    for step in range(6):
        rows = [r["steps"][step] for r in records]
        assert (
            len(
                {
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "tokens",
                                "modes",
                                "any_prefill",
                                "decode_graph_vote",
                            )
                        },
                        sort_keys=True,
                    )
                    for row in rows
                }
            )
            == 1
        )
        owners = [rid for row in rows for rid in row["owners"]]
        assert len(owners) == len(set(owners))
        expected_tokens = (
            [64] * world,
            [1] * world,
            [0] * world,
            [64] * world,
            ([16, 1] + [0] * (world - 2)) if world > 1 else [16],
            [0] * world,
        )[step]
        assert rows[0]["tokens"] == expected_tokens
        _, mode, _, _ = controls(world)
        expected_modes = [
            mode.IDLE.value
            if n == 0
            else (mode.DECODE.value if n == 1 else mode.EXTEND.value)
            for n in expected_tokens
        ]
        assert rows[0]["modes"] == expected_modes
        assert rows[0]["any_prefill"] == (step in (0, 3, 4))
        assert rows[0]["decode_graph_vote"] == (step not in (0, 3, 4))
        for rank, row in enumerate(rows):
            expected = (
                []
                if expected_tokens[rank] == 0
                else [f"{'a' if step < 3 else 'b'}-{rank}"]
            )
            assert row["owners"] == expected
            assert row["free"] == (512 if not expected else 448)


def run_case(output, world=4, fault=None, timeout=5):
    if sys.platform != "linux":
        raise RuntimeError(
            "This mandatory process integration check requires Linux/Gloo"
        )
    import torch

    if not torch.distributed.is_gloo_available():
        raise RuntimeError("PyTorch Gloo is required; this check cannot be skipped")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # CPU-only fork shares read-only libtorch pages on the 4 GiB build host.
    # No CUDA calls or process groups are initialized before fork.
    torch.set_num_threads(1)
    ctx = mp.get_context("fork")
    pairs, release = [ctx.Pipe() for _ in range(world)], ctx.Event()
    processes = []
    with tempfile.TemporaryDirectory(prefix="mpu-gloo-") as temp:
        rendezvous = Path(temp, "store").as_uri()
        try:
            for rank in range(world):
                process = ctx.Process(
                    target=worker,
                    args=(
                        rank,
                        world,
                        rendezvous,
                        pairs[rank][1],
                        release,
                        str(output),
                        fault,
                        timeout,
                    ),
                )
                process.start()
                processes.append(process)
            pipes = [pair[0] for pair in pairs]
            for rank, pipe in enumerate(pipes):
                assert receive(pipe) == {"ready": rank}
            for pipe in pipes:
                pipe.send({"start": True})
            if not fault:
                route = router(pipes)
                for step in range(6):
                    if step in (0, 3):
                        for rank in range(world):
                            route.round_robin_scheduler(
                                NS(
                                    rid=f"{'a' if step == 0 else 'b'}-{rank}",
                                    routed_dp_rank=None if step == 0 else rank,
                                )
                            )
                    if step == 2:
                        for rank, pipe in enumerate(pipes):
                            pipe.send({"op": "abort", "rid": f"a-{rank}"})
                    if step in (4, 5):
                        ranks = range(2, world) if step == 4 else range(min(2, world))
                        for rank in ranks:
                            pipes[rank].send({"op": "retract", "rid": f"b-{rank}"})
                    for rank, pipe in enumerate(pipes):
                        phase = (
                            "EXTEND",
                            "DECODE",
                            "IDLE",
                            "EXTEND",
                            "EXTEND"
                            if rank == 0
                            else ("DECODE" if rank == 1 else "IDLE"),
                            "IDLE",
                        )[step]
                        tokens = (
                            0
                            if phase == "IDLE"
                            else (1 if phase == "DECODE" else (16 if step == 4 else 64))
                        )
                        pipe.send(
                            {
                                "op": "step",
                                "step": step,
                                "phase": phase,
                                "tokens": tokens,
                            }
                        )
                    for pipe in pipes:
                        receive(pipe)
                for pipe in pipes:
                    pipe.send({"op": "stop"})
            deadline = time.monotonic() + max(30, timeout * 4)
            for process in processes[: world - 1 if fault else world]:
                process.join(max(0, deadline - time.monotonic()))
            release.set()
            processes[-1].join(max(0, deadline - time.monotonic()))
            exits = {rank: p.exitcode for rank, p in enumerate(processes)}
            records = [
                json.loads(p.read_text()) for p in sorted(output.glob("rank-*.json"))
            ]
            validate(records, world, fault, exits, timeout)
            return {
                "status": "passed",
                "world_size": world,
                "fault": fault,
                "exit_codes": exits,
                "records": records,
                "gpu_executed": False,
                "deepep_executed": False,
                "boundaries": "Pipe request transport; synthetic batches; CUDA completion events and model compute excluded",
            }
        finally:
            release.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
            for pair in pairs:
                for pipe in pair:
                    pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "gpu_executed": False, "cases": []}
    try:
        for name, world, fault in (
            ("single", 1, None),
            ("four", 4, None),
            ("missing-rank", 4, "timeout"),
            ("rank-exit", 4, "exit"),
        ):
            result["cases"].append(run_case(args.output / name, world, fault))
        result["status"] = "passed"
    finally:
        (args.output / "cpu-control.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
