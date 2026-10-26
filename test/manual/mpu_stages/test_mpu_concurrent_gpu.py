"""Four concurrent compute processes using four disjoint production MPU pools.

This tests data-path isolation and cross-stream dependencies, not TP collectives
or proof of bandwidth overlap. Only compute GPUs execute payload operations.
"""

import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_MPU_CONCURRENT") != "1",
    reason="Requires explicit eight-GPU concurrent MPU acceptance",
)


def _worker(rank, family, barrier, connection):
    import torch

    pool = None
    try:
        assert sys.platform == "linux" and torch.cuda.device_count() >= 8
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        size = 4096
        if family == "glm-5.3-flash":
            from sglang.srt.mem_cache.mpu_glm import HiSparseMLAMPUPool

            buffers = [
                torch.zeros((size + 64, 1, 512), device=device, dtype=torch.bfloat16)
                for _ in range(11)
            ]
            local = NS(
                device=device,
                page_size=64,
                layer_num=11,
                dtype=torch.bfloat16,
                kv_cache_dim=512,
                start_layer=0,
                index_kpool=4,
                size=size,
                kv_buffer=buffers,
            )
            pool = HiSparseMLAMPUPool(
                local,
                storage_device_ids=[4, 5, 6, 7],
                capacity_ratio=1,
                memory_reserve_bytes=0,
            )
        elif family == "deepseek-v4-flash":
            from sglang.srt.mem_cache.mpu_c4_pool import DeepSeekV4MPUPagedPool

            buffers = [
                torch.zeros((size // 64 + 1, 37440), device=device, dtype=torch.uint8)
                for _ in range(2)
            ]
            local = NS(
                device=device,
                page_size=64,
                bytes_per_page_padded=37440,
                layer_num=2,
                size=size,
                kv_buffer=buffers,
            )
            pool = DeepSeekV4MPUPagedPool(
                local, rank + 4, capacity_ratio=1, memory_reserve_bytes=0
            )
        else:
            raise ValueError(f"Unsupported concurrent family: {family}")
        assert pool.storage_device_ids == [rank + 4]
        src = torch.arange(1024, device=device, dtype=torch.int64)
        dst = src + 2048
        peer = (src + 64) % size
        plan_src = peer.view(1, -1)
        plan_dst = dst.to(torch.int32).view(1, -1)
        counts = torch.tensor([1024], device=device, dtype=torch.int32)
        real = torch.ones(1, device=device, dtype=torch.int32)
        write_stream, read_stream = torch.cuda.Stream(), torch.cuda.Stream()
        ready = torch.cuda.Event()

        def fill_and_expect(iteration):
            expected = []
            for layer, buffer in enumerate(buffers):
                if family == "glm-5.3-flash":
                    values = (
                        (
                            src[:, None]
                            + torch.arange(512, device=device)
                            + rank * 37
                            + layer * 11
                            + iteration * 19
                        )
                        % 251
                    ).to(torch.bfloat16)[:, None, :]
                    # Rank/iteration/slot identity has exact representable fields.
                    values[:, 0, :4] = torch.stack(
                        (
                            src % 128,
                            src // 128,
                            torch.full_like(src, rank),
                            torch.full_like(src, iteration),
                        ),
                        dim=1,
                    )
                    buffer[src] = values
                    buffer[dst] = -1
                    expected.append(values.clone())
                else:
                    pages = buffer.view(-1, 37440)
                    pages.fill_(211)
                    # Use paged indexing: flattening strided page regions can copy.
                    payload = (
                        (
                            src[:, None]
                            + torch.arange(576, device=device)
                            + rank * 37
                            + iteration * 19
                            + layer * 11
                        )
                        % 251
                    ).to(torch.uint8)
                    payload[:, :4] = torch.stack(
                        (
                            src % 128,
                            src // 128,
                            torch.full_like(src, rank),
                            torch.full_like(src, iteration),
                        ),
                        dim=1,
                    ).to(torch.uint8)
                    scales = (
                        (
                            src[:, None]
                            + torch.arange(8, device=device)
                            + rank
                            + iteration
                        )
                        % 251
                    ).to(torch.uint8)
                    pages[
                        (src // 64)[:, None],
                        (src % 64)[:, None] * 576 + torch.arange(576, device=device),
                    ] = payload
                    pages[
                        (src // 64)[:, None],
                        36864
                        + (src % 64)[:, None] * 8
                        + torch.arange(8, device=device),
                    ] = scales
                    expected.append(torch.cat((payload, scales), dim=1))
            return expected

        def backup():
            if family == "glm-5.3-flash":
                pool.backup_from_device_all_layer(local, peer, src, "kernel")
            else:
                pool.backup_from_device_all_layer(local, peer, src)

        def replay():
            for layer in range(len(buffers)):
                pool.replay_misses(plan_src, plan_dst, counts, real, layer)

        # Compile and initialize each child before the simultaneous launch gate.
        fill_and_expect(0)
        backup()
        replay()
        torch.cuda.synchronize(rank)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay()
        rounds = []
        for iteration in range(1, 9):
            expected = fill_and_expect(iteration)
            torch.cuda.synchronize(rank)
            barrier.wait(timeout=180)
            submitted = time.monotonic_ns()
            with torch.cuda.stream(write_stream):
                backup()
                ready.record()
            with torch.cuda.stream(read_stream):
                read_stream.wait_event(ready)
                if iteration % 2:
                    replay()
                else:
                    graph.replay()
            # All four ranks submit before any host-side comparison begins.
            barrier.wait(timeout=180)
            torch.cuda.synchronize(rank)
            for layer, buffer in enumerate(buffers):
                if family == "glm-5.3-flash":
                    actual = buffer[dst]
                else:
                    values = buffer[
                        (dst // 64)[:, None],
                        (dst % 64)[:, None] * 576 + torch.arange(576, device=device),
                    ]
                    scales = buffer[
                        (dst // 64)[:, None],
                        36864
                        + (dst % 64)[:, None] * 8
                        + torch.arange(8, device=device),
                    ]
                    actual = torch.cat((values, scales), dim=1)
                    assert (
                        buffer[:, 37376:] == 211
                    ).all(), "C4 page padding was overwritten"
                torch.testing.assert_close(actual, expected[layer], rtol=0, atol=0)
            # Free waits for all compute streams, then make the same capacity reusable.
            allocated = pool.alloc(size)
            assert allocated is not None and pool.alloc(64) is None
            drained = torch.cuda.Event()
            with torch.cuda.stream(read_stream):
                replay()
                drained.record()
            pool.free(allocated)
            assert (
                drained.query()
            ), "Backing slots were reused before the read stream drained"
            assert pool.available_size() == size
            rounds.append(
                {
                    "iteration": iteration,
                    "submitted_ns": submitted,
                    "verified_ns": time.monotonic_ns(),
                    "graph": iteration % 2 == 0,
                }
            )
        del graph
        connection.send(
            {
                "rank": rank,
                "storage": rank + 4,
                "status": "passed",
                "pid": os.getpid(),
                "rounds": rounds,
            }
        )
    except BaseException:
        try:
            barrier.abort()
            connection.send(
                {"rank": rank, "status": "failed", "error": traceback.format_exc()}
            )
        finally:
            raise
    finally:
        if pool is not None:
            pool.destroy()
        connection.close()


def test_four_rank_concurrent_roundtrip_replay_and_reuse():
    family = os.environ["SGLANG_MPU_TEST_FAMILY"]
    context = mp.get_context("spawn")
    barrier = context.Barrier(4)
    children, readers, reports = [], [], []
    try:
        for rank in range(4):
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(
                target=_worker, args=(rank, family, barrier, writer)
            )
            process.start()
            writer.close()
            children.append(process)
            readers.append(reader)
        deadline = time.monotonic() + 600
        for reader in readers:
            assert reader.poll(
                max(0, deadline - time.monotonic())
            ), "MPU rank timed out"
            reports.append(reader.recv())
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0, reports
        assert sorted(
            (r["rank"], r["storage"]) for r in reports if r["status"] == "passed"
        ) == list(zip(range(4), range(4, 8))), reports
        assert len({r["pid"] for r in reports}) == 4
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(10)
            if child.is_alive():
                child.kill()
                child.join(10)
        for reader in readers:
            reader.close()
        path = os.environ.get("SGLANG_MPU_CONCURRENT_REPORT")
        if path:
            Path(path).write_text(
                json.dumps(
                    {
                        "family": family,
                        "ranks": reports,
                        "scope": "concurrent data paths; no TP collective or measured GPU-overlap claim",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
