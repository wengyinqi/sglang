"""Explicit MMA replay of saved same-QKV diagnostic tensors, not a model gate."""

import argparse
import json
from pathlib import Path

import torch


def difference(actual, expected):
    values = actual.float() - expected.float()
    return {
        "max_abs": float(values.abs().max().item()),
        "rms": float(values.square().mean().sqrt().item()),
        "unequal": int((actual != expected).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from sglang.jit_kernel.fused_hisparse_tma_mma import FusedHiSparseTMAMMAWorkspace

    records = []
    device = torch.device("cuda", torch.cuda.current_device())
    for path in sorted(args.snapshots.glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        keys, values = data["selected_k"], data["selected_v"]
        batch, topk = keys.shape[:2]
        assert keys.shape == values.shape == (batch, topk, 4, 128)
        assert torch.all(data["valid"]).item()
        host_k = keys.reshape(-1, 4, 128).pin_memory()
        host_v = values.reshape(-1, 4, 128).pin_memory()
        hot_k, hot_v = host_k.to(device), host_v.to(device)
        q = data["q"].view(batch, 32, 128).contiguous().to(device)
        actual_cpu = data["actual_lens"]
        actual = actual_cpu.to(device)
        real = torch.tensor([batch], device=device, dtype=torch.int32)
        workspace = FusedHiSparseTMAMMAWorkspace(
            device=device,
            max_bs=batch,
            top_k=topk,
            hot_buffer_size=topk,
            item_size_bytes=1024,
            host_k_layers=[host_k],
            host_v_layers=[host_v],
        )
        physical = torch.arange(batch * topk, device=device, dtype=torch.int32).view(
            batch, topk
        )
        workspace.device_locs.copy_(physical)
        workspace.miss_host_locs.fill_(-1)
        workspace.plan_partition(actual_cpu)
        indptr = torch.cat(
            (torch.zeros(1, dtype=torch.int32), actual_cpu.cumsum(0).to(torch.int32))
        ).to(device)
        indices = torch.cat(
            [physical[row, :length] for row, length in enumerate(actual_cpu.tolist())]
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(384 << 20, device=device, dtype=torch.uint8),
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=torch.ones(
                batch, dtype=torch.int32, device=device
            ),
        )
        scale = data["stats"]["scaling"]
        cap = data["stats"]["logits_soft_cap"]
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=torch.ones(batch, dtype=torch.int32, device=device),
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim=128,
            page_size=1,
            sm_scale=scale,
            logits_soft_cap=cap,
            data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
            non_blocking=False,
        )
        reference = wrapper.run(q, (hot_k.unsqueeze(1), hot_v.unsqueeze(1)))
        output = workspace.run(
            q,
            hot_k=hot_k,
            hot_v=hot_v,
            layer_id=0,
            actual_lens=actual,
            num_real_reqs=real,
            scaling=scale,
            logits_soft_cap=cap,
        ).clone()
        workspace.miss_host_locs.copy_(physical.long())
        mapped = workspace.run(
            q,
            hot_k=hot_k,
            hot_v=hot_v,
            layer_id=0,
            actual_lens=actual,
            num_real_reqs=real,
            scaling=scale,
            logits_soft_cap=cap,
        ).clone()
        record = {
            "snapshot": str(path),
            "original_scalar_difference": data["stats"]["max_abs_output_error"],
            "same_dense_fi_vs_saved_fi": difference(
                reference.reshape(batch, -1), data["reference_output"].to(device)
            ),
            "mma_vs_fi": difference(output, reference),
            "mapped_vs_hot": difference(mapped, output),
            "chunk_size": int(workspace.kv_chunk_size.item()),
            "scope": "Saved same-QKV attention arithmetic replay; not full-model correctness",
        }
        print(json.dumps(record), flush=True)
        records.append(record)
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
