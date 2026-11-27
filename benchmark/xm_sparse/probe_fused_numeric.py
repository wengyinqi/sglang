"""Same-Q/K/V numerical probe; no model, selection, or cache-state mutation.

Run through run_h800.sh. The original kernel and FlashInfer both consume the
same BF16 rows; FP32 and BF16-P references identify arithmetic differences.
This is diagnostic evidence, not a replacement for model acceptance.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def error(actual, expected):
    diff = actual.float() - expected.float()
    return {
        "max_abs": float(diff.abs().max().item()),
        "rms": float(diff.square().mean().sqrt().item()),
        "unequal": int((actual != expected).sum().item()),
    }


def rounded_probability_reference(scores, values, *, tile, span):
    """Tilewise unnormalized BF16 P, shared denominator/PV rounding.

    Independently merge FP32 split states. This deliberately does not claim
    FlashInfer's exact tensor-core reduction/partition order.
    """
    heads, count = scores.shape
    states = []
    for first in range(0, count, span):
        maximum = torch.full((heads,), -torch.inf, device=scores.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((heads, 128), device=scores.device)
        for start in range(first, min(first + span, count), tile):
            last = min(start + tile, first + span, count)
            score = scores[:, start:last]
            next_max = torch.maximum(maximum, score.max(dim=-1).values)
            scale = (maximum - next_max).exp()
            weights = (score - next_max[:, None]).exp().to(torch.bfloat16).float()
            denominator = denominator * scale + weights.sum(dim=-1)
            numerator = numerator * scale[:, None] + torch.einsum(
                "hk,khd->hd", weights, values[start:last]
            )
            maximum = next_max
        states.append((maximum, denominator, numerator))
    maximum = torch.stack([item[0] for item in states]).max(dim=0).values
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((heads, 128), device=scores.device)
    for old_max, old_sum, old_output in states:
        weight = (old_max - maximum).exp()
        denominator += old_sum * weight
        numerator += old_output * weight[:, None]
    return (numerator / denominator[:, None]).to(torch.bfloat16)


def normalized_rope(shape, positions, gain):
    value = torch.randn(shape, dtype=torch.float32)
    value = value * value.square().mean(-1, keepdim=True).rsqrt() * gain
    frequency = 1.0 / (1_000_000.0 ** (torch.arange(64).float() / 64))
    angle = positions[:, None] * frequency[None, :]
    cosine, sine = angle.cos()[:, None, :], angle.sin()[:, None, :]
    first, second = value[..., :64], value[..., 64:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(torch.bfloat16)


def flashinfer_order_reference(scores, values, *, chunk, warp_tile, round_partial, round_merge=False):
    heads, count = scores.shape
    chunks = []
    for first in range(0, count, chunk):
        states = []
        for warp in range(4):
            maximum = torch.full((heads,), -torch.inf, device=scores.device)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((heads, 128), device=scores.device)
            for start in range(first + warp * warp_tile, min(first + chunk, count), 4 * warp_tile):
                last = min(start + warp_tile, first + chunk, count)
                score = scores[:, start:last]
                next_max = torch.maximum(maximum, score.max(dim=-1).values)
                scale = (maximum - next_max).exp()
                weights = (score - next_max[:, None]).exp().to(torch.bfloat16).float()
                denominator = denominator * scale + weights.sum(dim=-1)
                numerator = numerator * scale[:, None] + torch.einsum(
                    "hk,khd->hd", weights, values[start:last]
                )
                maximum = next_max
            states.append((maximum, denominator, numerator))
        maximum = torch.stack([item[0] for item in states]).max(dim=0).values
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((heads, 128), device=scores.device)
        for old_max, old_sum, old_output in states:
            weight = (old_max - maximum).exp()
            denominator += old_sum * weight
            numerator += old_output * weight[:, None]
        if round_partial:
            numerator = (numerator / denominator[:, None]).to(torch.bfloat16).float() * denominator[:, None]
        chunks.append((maximum, denominator, numerator))
    if round_merge and len(chunks) > 1:
        groups = []
        for group in range(min(8, len(chunks))):
            selected = chunks[group::8]
            maximum = torch.stack([item[0] for item in selected]).max(dim=0).values
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((heads, 128), device=scores.device)
            for old_max, old_sum, old_output in selected:
                weight = (old_max - maximum).exp()
                denominator += old_sum * weight
                numerator += old_output * weight[:, None]
            numerator = (numerator / denominator[:, None]).to(torch.bfloat16).float() * denominator[:, None]
            groups.append((maximum, denominator, numerator))
        chunks = groups
    maximum = torch.stack([item[0] for item in chunks]).max(dim=0).values
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((heads, 128), device=scores.device)
    for old_max, old_sum, old_output in chunks:
        weight = (old_max - maximum).exp()
        denominator += old_sum * weight
        numerator += old_output * weight[:, None]
    return (numerator / denominator[:, None]).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[65, 145, 2048])
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0, 4.0])
    args = parser.parse_args()
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from sglang.jit_kernel.fused_hisparse_tma import FusedHiSparseTMAWorkspace

    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(418)
    device = torch.device("cuda", torch.cuda.current_device())
    topk = 2048
    host_k = torch.empty((topk, 4, 128), dtype=torch.bfloat16, pin_memory=True)
    host_v = torch.empty_like(host_k, pin_memory=True)
    hot_k = torch.empty_like(host_k, device=device)
    hot_v = torch.empty_like(host_k, device=device)
    tma = FusedHiSparseTMAWorkspace(
        device=device, max_bs=1, top_k=topk, hot_buffer_size=topk,
        item_size_bytes=1024, host_k_layers=[host_k], host_v_layers=[host_v],
    )
    tma.device_locs[0] = torch.arange(topk, device=device, dtype=torch.int32)
    tma.miss_host_locs.fill_(-1)
    actual = torch.ones(1, device=device, dtype=torch.int32)
    real = torch.ones_like(actual)
    indptr = torch.zeros(2, dtype=torch.int32, device=device)
    indices = torch.arange(topk, device=device, dtype=torch.int32)
    last_page = torch.ones(1, dtype=torch.int32, device=device)
    workspace = torch.empty(384 << 20, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_cuda_graph=True, use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr, paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last_page,
    )
    records = []
    for gain in args.gains:
        host_k.copy_(normalized_rope(host_k.shape, torch.arange(topk), 1.0))
        host_v.copy_(torch.randn(host_v.shape).to(torch.bfloat16))
        hot_k.copy_(host_k)
        hot_v.copy_(host_v)
        for count in args.lengths:
            q = normalized_rope((1, 32, 128), torch.tensor([count - 1]), gain).to(device)
            actual.fill_(count)
            indptr[1] = count
            wrapper.plan(
                indptr=indptr, indices=indices, last_page_len=last_page,
                num_qo_heads=32, num_kv_heads=4, head_dim=128, page_size=1,
                sm_scale=128 ** -0.5, data_type=torch.bfloat16,
                q_data_type=torch.bfloat16, non_blocking=False,
            )
            separate = wrapper.run(q, (hot_k.unsqueeze(1), hot_v.unsqueeze(1)))
            fused = tma.run(
                q, hot_k=hot_k, hot_v=hot_v, layer_id=0,
                actual_lens=actual, num_real_reqs=real, scaling=128 ** -0.5,
            ).clone()
            keys = hot_k[:count].float().repeat_interleave(8, dim=1)
            values = hot_v[:count].float().repeat_interleave(8, dim=1)
            scores = torch.einsum("hd,khd->hk", q[0].float(), keys) * (128 ** -0.5)
            fp32 = torch.einsum("hk,khd->hd", scores.softmax(-1), values)
            record = {
                "count": count, "q_gain": gain,
                "q_rms": float(q.float().square().mean().sqrt().item()),
                "k_rms": float(keys.square().mean().sqrt().item()),
                "score_abs_max": float(scores.abs().max().item()),
                "plan_info": list(getattr(wrapper, "_plan_info", [])),
                "fused_vs_flashinfer": error(fused[0], separate[0]),
                "fused_vs_fp32": error(fused[0], fp32),
                "flashinfer_vs_fp32": error(separate[0], fp32),
                "rounded_references": {},
            }
            info = record["plan_info"]
            record["kv_chunk_size"] = int(
                wrapper._int_workspace_buffer[info[9]:info[9] + 4].view(torch.int32).item()
            )
            record["shared_memory_per_sm"] = torch.cuda.get_device_properties(device).shared_memory_per_multiprocessor
            for warp_tile in (16, 32, 64, 128):
                for partial_round in (False, True):
                    reference = flashinfer_order_reference(
                        scores, values, chunk=record["kv_chunk_size"],
                        warp_tile=warp_tile, round_partial=partial_round,
                    )
                    record["rounded_references"][f"warp{warp_tile}_partial{int(partial_round)}"] = {
                        "vs_flashinfer": error(reference, separate[0]),
                        "vs_fused": error(reference, fused[0]),
                    }
            reference = flashinfer_order_reference(
                scores, values, chunk=record["kv_chunk_size"], warp_tile=32,
                round_partial=True, round_merge=True,
            )
            record["rounded_references"]["warp32_partial1_merge1"] = {
                "vs_flashinfer": error(reference, separate[0]),
                "vs_fused": error(reference, fused[0]),
            }
            for span in (256, topk):
                for tile in (64, 128, 256):
                    reference = rounded_probability_reference(scores, values, tile=tile, span=span)
                    record["rounded_references"][f"tile{tile}_span{span}"] = {
                        "vs_flashinfer": error(reference, separate[0]),
                        "vs_fused": error(reference, fused[0]),
                    }
            # Backing path cannot affect arithmetic: all-mapped misses with
            # the same rows must reproduce the all-hot attention bit for bit.
            tma.miss_host_locs[0] = indices.long()
            mapped = tma.run(
                q, hot_k=hot_k, hot_v=hot_v, layer_id=0,
                actual_lens=actual, num_real_reqs=real, scaling=128 ** -0.5,
            ).clone()
            record["mapped_vs_hot"] = error(mapped, fused)
            tma.miss_host_locs.fill_(-1)
            torch.cuda.synchronize()
            print(json.dumps(record), flush=True)
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
