"""Graph-safe writes to mapped pinned Quest summaries, retaining BF16 bounds."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _finalize_bounds(
    pointer_table,
    req_indices,
    counts,
    pages,
    running_min,
    running_max,
    MAX_REQS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    KVHD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    active = tl.program_id(1)
    tile = tl.program_id(2)
    req = tl.load(req_indices + active).to(tl.int64)
    page = tl.load(pages + req).to(tl.int64)
    count = tl.load(counts + req)
    complete = (count == PAGE_SIZE - 1) & (page < MAX_PAGES)
    d = tile * BLOCK + tl.arange(0, BLOCK)
    mask = d < KVHD
    source = (layer * MAX_REQS + req) * KVHD + d
    minimum = tl.load(running_min + source, mask=mask, other=0)
    maximum = tl.load(running_max + source, mask=mask, other=0)
    bounds = tl.load(pointer_table).to(tl.pointer_type(tl.bfloat16))
    destination = ((layer * MAX_REQS + req) * MAX_PAGES + page) * KVHD * 2 + d * 2
    tl.store(bounds + destination, minimum, mask=mask & complete)
    tl.store(bounds + destination + 1, maximum, mask=mask & complete)
    tl.store(
        running_min + source,
        tl.where(complete, 3.3895313892515355e38, minimum),
        mask=mask,
    )
    tl.store(
        running_max + source,
        tl.where(complete, -3.3895313892515355e38, maximum),
        mask=mask,
    )
    tl.inline_asm_elementwise(
        "fence.sc.sys; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _advance_counts(req_indices, counts, pages, PAGE_SIZE: tl.constexpr):
    req = tl.load(req_indices + tl.program_id(0)).to(tl.int64)
    count = tl.load(counts + req)
    page = tl.load(pages + req)
    tl.store(pages + req, page + (count == PAGE_SIZE - 1).to(tl.int32))
    tl.store(counts + req, (count + 1) % PAGE_SIZE)


@triton.jit
def _invalidate_bounds(
    pointer_table,
    req: tl.constexpr,
    MAX_REQS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    KVHD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    page = tl.program_id(1)
    d = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    bounds = tl.load(pointer_table).to(tl.pointer_type(tl.bfloat16))
    destination = ((layer * MAX_REQS + req) * MAX_PAGES + page) * KVHD * 2 + d * 2
    tl.store(bounds + destination, 3.3895313892515355e38, mask=d < KVHD)
    tl.store(bounds + destination + 1, -3.3895313892515355e38, mask=d < KVHD)
    tl.inline_asm_elementwise(
        "fence.sc.sys; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def finalize_bounds(
    tier, req_indices, running_min, running_max, counts, pages, page_size
):
    if not req_indices.numel():
        return
    kvhd = tier.backing_bounds.shape[3] * tier.backing_bounds.shape[4]
    block = min(128, triton.next_power_of_2(kvhd))
    _finalize_bounds[(tier.layers, req_indices.numel(), triton.cdiv(kvhd, block))](
        tier.backing_uva_ptr,
        req_indices,
        counts,
        pages,
        running_min,
        running_max,
        MAX_REQS=tier.backing_bounds.shape[1],
        MAX_PAGES=tier.backing_bounds.shape[2],
        KVHD=kvhd,
        PAGE_SIZE=page_size,
        BLOCK=block,
        num_warps=4,
    )
    # Keep counters immutable until every layer has consumed the old values.
    _advance_counts[(req_indices.numel(),)](
        req_indices, counts, pages, PAGE_SIZE=page_size
    )


def invalidate_bounds(tier, req_pool_idx: int):
    kvhd = tier.backing_bounds.shape[3] * tier.backing_bounds.shape[4]
    block = min(128, triton.next_power_of_2(kvhd))
    _invalidate_bounds[
        (tier.layers, tier.backing_bounds.shape[2], triton.cdiv(kvhd, block))
    ](
        tier.backing_uva_ptr,
        req=req_pool_idx,
        MAX_REQS=tier.backing_bounds.shape[1],
        MAX_PAGES=tier.backing_bounds.shape[2],
        KVHD=kvhd,
        BLOCK=block,
        num_warps=4,
    )
