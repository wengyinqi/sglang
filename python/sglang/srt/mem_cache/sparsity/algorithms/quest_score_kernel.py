"""Fused shared-page Quest scoring, retaining each GQA query's sign.

For each KV group, average the per-query-head bounding-box upper bounds,
then sum across groups. Separating positive/negative query coefficients
before reducing heads avoids cancellation from averaging queries first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_score_kernel(
    q_positive_ptr,
    q_negative_ptr,
    k_bounds_ptr,
    req_pool_indices_ptr,
    scores_ptr,
    layer_offset: tl.constexpr,
    MAX_REQS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    KVHD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_req = tl.program_id(0)
    pid_page = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    mask = d < KVHD
    qp = tl.load(q_positive_ptr + pid_req * KVHD + d, mask=mask, other=0)
    qn = tl.load(q_negative_ptr + pid_req * KVHD + d, mask=mask, other=0)
    req_idx = tl.load(req_pool_indices_ptr + pid_req).to(tl.int64)
    base = ((layer_offset * MAX_REQS + req_idx) * MAX_PAGES + pid_page) * KVHD * 2
    k_min = tl.load(k_bounds_ptr + base + d * 2, mask=mask, other=0).to(tl.float32)
    k_max = tl.load(k_bounds_ptr + base + d * 2 + 1, mask=mask, other=0).to(tl.float32)
    valid = tl.sum(((k_min > k_max) & mask).to(tl.int32), axis=0) == 0
    # Invalid sentinels must not participate in arithmetic, including for
    # all-zero queries. Padded dimensions contribute zero to the reduction.
    k_min = tl.where(valid, k_min, 0.0)
    k_max = tl.where(valid, k_max, 0.0)
    score = tl.sum(qp * k_max + qn * k_min, axis=0)
    tl.store(
        scores_ptr + pid_req * MAX_PAGES + pid_page,
        tl.where(valid, score, -float("inf")),
    )


def quest_score(
    q: torch.Tensor,
    page_k_bounds: torch.Tensor,
    req_pool_indices: torch.Tensor,
    layer_offset: int,
) -> torch.Tensor:
    """Compute a shared page score from all query heads without cancellation.

    q is float32 [batch, query_heads, head_dim] and bounds are BF16
    [layers, requests, pages, kv_heads, head_dim, 2]. Query heads must
    be grouped consecutively by KV head, as in Qwen GQA.
    """
    bs, q_heads, head_dim = q.shape
    _, max_reqs, max_pages, kv_heads, bound_dim, _ = page_k_bounds.shape
    assert head_dim == bound_dim and q_heads % kv_heads == 0
    assert q.dtype == torch.float32 and q.is_contiguous()
    assert page_k_bounds.is_contiguous()

    grouped = q.reshape(bs, kv_heads, q_heads // kv_heads, head_dim)
    q_positive = grouped.clamp(min=0).mean(dim=2).contiguous()
    q_negative = grouped.clamp(max=0).mean(dim=2).contiguous()
    scores = torch.empty((bs, max_pages), dtype=torch.float32, device=q.device)
    if bs == 0:
        return scores
    kvhd = kv_heads * head_dim
    _quest_score_kernel[(bs, max_pages)](
        q_positive,
        q_negative,
        page_k_bounds,
        req_pool_indices,
        scores,
        layer_offset=layer_offset,
        MAX_REQS=max_reqs,
        MAX_PAGES=max_pages,
        KVHD=kvhd,
        BLOCK=triton.next_power_of_2(kvhd),
        num_warps=4,
    )
    return scores
