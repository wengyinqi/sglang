"""Seeded production allocator/coordinator checks; no CUDA/model execution."""

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "mpu_cpu_fixtures", ROOT / "benchmark/mpu_stages/cpu_fixtures.py"
)
FIXTURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIXTURES
SPEC.loader.exec_module(FIXTURES)
PAGED = FIXTURES.PAGED
definition = FIXTURES.definition
make_pool = FIXTURES.make_pool
make_logical_allocator = FIXTURES.make_logical_allocator
coordinator = FIXTURES.coordinator


@pytest.mark.parametrize("seed", [0, 1, 7, 19, 41, 97, 251, 2026])
def test_seeded_append_exhaust_abort_retract_and_reuse(seed):
    rng, history = random.Random(seed), []
    pool = make_pool()
    page = 64 if PAGED else 1
    mapping = torch.full((6, 512), -1, dtype=torch.int64)
    allocated = torch.zeros(6, dtype=torch.int64)
    lengths = [0] * 6
    audit = []
    c = coordinator(pool, mapping, allocated, audit)
    failures = releases = retracts = 0
    try:
        for step in range(400):
            rid = rng.randrange(6)
            action = rng.choice(
                ("append", "append", "append", "finish", "abort", "retract")
            )
            if action == "append":
                amount = min(rng.choice((1, 3, 63, 64, 65, 129)), 512 - lengths[rid])
                end = lengths[rid] + amount
                needed = ((end + page - 1) // page * page) - int(allocated[rid])
                before = mapping.clone(), allocated.clone(), pool.available_size()
                history.append((step, rid, action, amount, needed))
                if PAGED:
                    try:
                        got = pool.alloc_paged_token_slots(
                            mapping, allocated, rid, lengths[rid], amount
                        )
                    except RuntimeError:
                        assert needed > before[2]
                        got = None
                else:
                    got = pool.alloc(needed)
                    if got is not None:
                        mapping[rid, lengths[rid] : end] = got
                        allocated[rid] = end
                if got is None:
                    failures += 1
                    assert torch.equal(mapping, before[0]) and torch.equal(
                        allocated, before[1]
                    )
                    assert pool.available_size() == before[2]
                else:
                    lengths[rid] = end
            elif lengths[rid]:
                history.append((step, rid, action, lengths[rid]))
                audit.clear()
                c._has_pending_backup = True
                req = NS(
                    req_pool_idx=rid,
                    kv_allocated_len=lengths[rid],
                    hisparse_staging=False,
                    kv=NS(req_pool_idx=rid, kv_allocated_len=lengths[rid]),
                )
                if action == "retract":
                    c.retract_req(req)
                    retracts += 1
                else:
                    c.request_finished(req)
                assert audit.index("wait_producer") < audit.index("free_backing")
                assert audit.index("wait_backup") < audit.index("free_backing")
                assert (mapping[rid] == -1).all()
                allocated[rid] = 0  # Legacy Qwen has no paged allocated-length array.
                lengths[rid] = 0
                releases += 1
            owned = []
            for row, length in enumerate(lengths):
                n = (length + page - 1) // page * page
                assert allocated[row] == n
                assert (mapping[row, n:] == -1).all()
                slots = mapping[row, :n].tolist()
                assert all(0 <= slot < pool.size for slot in slots)
                owned.extend(slots)
            assert len(owned) == len(set(owned)), "two requests own the same address"
            assert len(owned) + pool.available_size() == pool.size
        for rid, length in enumerate(lengths):
            if length:
                pool.free(mapping[rid, : int(allocated[rid])])
        assert pool.available_size() == pool.size
        # Exact refill proves no leak/duplicate page hidden by a counter.
        full = pool.alloc(pool.size)
        assert sorted(full.tolist()) == list(range(pool.size))
        assert pool.alloc(page) is None
        assert failures > 0 and releases > 0 and retracts > 0
    except BaseException as error:
        raise AssertionError(f"seed={seed}; operations={history}") from error


def test_failed_free_preserves_every_live_owner():
    pool = make_pool()
    live = pool.alloc(128)
    for bad in (
        torch.tensor([-1]),
        torch.tensor([pool.size]),
        live[:1].repeat(2),
        torch.cat((live, torch.tensor([pool.size - 1]))),
    ):
        available = pool.available_size()
        with pytest.raises(ValueError):
            pool.free(bad)
        assert pool.available_size() == available
    pool.free(live)
    assert pool.available_size() == pool.size


def test_paged_mapping_overflow_does_not_leak_allocations():
    if not PAGED:
        # Legacy Quest allocates individual slots; invalid counts are rejected.
        pool = make_pool()
        for invalid in (-1, True, 1.5):
            with pytest.raises(ValueError):
                pool.alloc(invalid)
        assert pool.available_size() == pool.size
        return
    pool = make_pool()
    mapping = torch.full((1, 64), -1, dtype=torch.int64)
    allocated = torch.zeros(1, dtype=torch.int64)
    for start, count in ((0, 65), (-1, 1), (0, -1)):
        with pytest.raises(ValueError):
            pool.alloc_paged_token_slots(mapping, allocated, 0, start, count)
        assert pool.available_size() == pool.size
        assert allocated.item() == 0 and (mapping == -1).all()


def test_admission_or_hot_buffer_failure_keeps_existing_ownership():
    if not PAGED:
        budget = definition(
            "python/sglang/srt/mem_cache/mpu_admission.py", "MPUAdmissionBudget"
        )(512)
        rng, leases = random.Random(2026), {}
        for _ in range(1000):
            rid = str(rng.randrange(12))
            if rng.randrange(3) == 0:
                budget.release(rid)
                leases.pop(rid, None)
            else:
                slots = leases.get(rid, rng.randrange(1, 200))
                should_fit = rid in leases or sum(leases.values()) + slots <= 512
                assert budget.reserve(rid, slots) == should_fit
                if should_fit:
                    leases[rid] = slots
            assert budget.reserved == sum(leases.values())
        for rid in list(leases):
            budget.release(rid)
        assert budget.reserved == 0
        # Legacy Quest has the same hot-buffer expansion ownership contract.
        pool = make_pool()
        pool.alloc(pool.size)
        holder = NS(
            page_size=64,
            device="cpu",
            hisparse_attn_allocator=pool,
            full_to_hisparse_device_index_mapping=torch.arange(1, 65),
        )
        before = holder.full_to_hisparse_device_index_mapping.clone()
        expand = definition(
            "python/sglang/srt/mem_cache/hisparse_memory_pool.py",
            "alloc_device_buffer",
            owner="HiSparseTokenToKVPoolAllocator",
        )
        with pytest.raises((AssertionError, RuntimeError)):
            expand(holder, torch.arange(64), 128)
        assert torch.equal(holder.full_to_hisparse_device_index_mapping, before)
        return
    allocator = make_logical_allocator()
    zero, length = torch.tensor([0]), torch.tensor([64])
    first = allocator.alloc_extend(zero, zero, length, length, torch.tensor([-1]), 64)
    second = allocator.alloc_extend(zero, zero, length, length, torch.tensor([-1]), 64)
    before = allocator.full_to_hisparse_device_index_mapping.clone()
    with pytest.raises((AssertionError, RuntimeError)):
        allocator.alloc_device_buffer(first, 128)
    assert torch.equal(allocator.full_to_hisparse_device_index_mapping, before)
    allocator.free(first)
    allocator.free(second)
    assert allocator.hisparse_attn_allocator.available_size() == 128
    assert allocator.logical_attn_allocator.available_size() == 512
    # C4 has a separate compressed allocator implementation; test it directly.
    pool = make_pool()
    pool.alloc(pool.size)
    holder = NS(
        hisparse_page_size=64,
        device="cpu",
        hisparse_attn_allocator=pool,
        full_to_hisparse_device_index_mapping=torch.arange(64, 128),
    )
    before = holder.full_to_hisparse_device_index_mapping.clone()
    expand = definition(
        "python/sglang/srt/mem_cache/allocator/hisparse.py",
        "alloc_device_buffer",
        owner="DeepSeekV4HiSparseTokenToKVPoolAllocator",
    )
    with pytest.raises((AssertionError, RuntimeError)):
        expand(holder, torch.arange(64), 128)
    assert torch.equal(holder.full_to_hisparse_device_index_mapping, before)
