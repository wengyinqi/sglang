# Qwen adaptation of XM-Sparse software mechanisms

Branch: `research/qwen-quest-xm-sparse-1gpu-0mpu`.
Frozen parent: `b61d8c7f1529ff1971026ae68dcc42af80e7a592` on stage 01.
The official branches and the six existing stage branches are unchanged.

This experiment uses one H800 PCIe, BF16 Qwen3-30B-A3B-Thinking-2507,
Quest page bounds, CPU pinned DDR full-KV backing, and NoPD. It adapts the
software ideas; it does not reproduce GLM-5.3/DeepSeek's trained K-index,
B300 TMA gather instructions, or the paper's scale-up memory pool.

## Mechanisms

`xm_fused_retrieval_attention` retains Quest selection and HiSparse's hot/LRU
policy. A payload-free cache plan precedes a BF16 SM90 kernel that fetches
selected K/V into double-buffered shared memory with TMA, computes online
softmax, and refills the hot slots. Split partials have a separate reduction.
The output must pass numerical, refill, repeated-step and CUDA Graph tests.

`xm_quest_bounds_offload` moves complete historical min/max summaries to a
CPU pinned owner. Running bounds remain on the GPU. Two fixed HBM layer
stages are reused in order. `on_demand` waits at each layer; `lookahead`
starts the next layer's copy earlier. Native device graph launch triggers
memcpy child graphs from the GPU. This differs from the paper's custom GPU
command-generation/CPU batch-publication library. Copy-engine execution and
overlap must be supported by observed hardware evidence, not inferred from
configuration alone.

`xm_prefetch_protect` gates future prefetch work around attention. Its chunk
checkpoints are a public-CUDA adaptation, not native copy-engine semaphore
commands. Traffic already in flight may finish after a pause. This branch
has no cross-rank MoE collective traffic to protect.

All mechanisms default off. Unsupported settings are rejected. Plain
load/store or a full pre-copy to HBM is not a TMA fallback for this ablation.

## Separate questions and evidence

1. **Same actual batch latency:** keep physical KV allocation, context,
   Quest top-K/hot size, frozen inputs, graph mode and seed identical. Report
   stable native decode launch intervals, request-average TPOT, and client
   TTFT separately. Client concurrency is not a measured batch size. A trace
   without enough constant-request windows at the target batch is ineligible.
2. **Same cache HBM budget capacity:** use `xm_cache_hbm_budget_bytes` for
   KV and named selector/cache-management tensors; report actual named bytes
   and unused budget. Weights, attention workspace, graph private pools,
   allocated/reserved and NVML are separate. Scan client concurrency but
   choose capacity/throughput from actually observed stable batches.
3. **Attention-path breakdown:** independent Nsight decode captures; list
   score/TopK, cache plan or swap-in, packing, attention and split reduction.
   Include idle gaps in first-score-to-last-attention spans; overlapping
   transfer/kernel durations cannot simply be summed into critical-path time.
4. **Memory breakdown:** full summary backing, GPU layer stages, running
   bounds, hot KV, LRU/mappings, fused plan and partial workspace, graph pools
   and allocator/NVML snapshots. Logical summary bytes are not PCIe bytes.

The paper's Table 3 and Appendix A.2 guide attribution and plot layout;
their numeric results are author measurements on different hardware/models.
Negative or statistically unresolved gains are reported as such.

## Entry points

`campaign.py` only plans unless `--execute` is present. It writes explicit
stage recipes and invokes the existing correctness, serving, timing, steady
decode, lifecycle and workload-hash validators. Six arms are available:
baseline, fused, bounds-on-demand, bounds-prefetch, combined,
combined-protected. Each arm keeps raw logs/responses/traces.

On the configured server, `run_h800.sh` uses the existing frozen offline image
and local weights. `XM_TASK_ROOT` and `XM_SOURCE_ROOT` can override paths. The
default task root is `/home/user/Workspace/mpu/xm-sparse-qwen-20261127-001`.

```bash
bash benchmark/xm_sparse/run_h800.sh benchmark/xm_sparse/campaign.py \
  --phase correctness --model /models/qwen \
  --workload /data/frozen-xm-qwen-20261127/workload.json \
  --input-lens 16384 32768 --output-len 80 \
  --output-dir /results/results/correctness --execute

bash benchmark/xm_sparse/run_h800.sh benchmark/xm_sparse/campaign.py \
  --phase performance --model /models/qwen \
  --workload /data/frozen-xm-qwen-20261127/workload.json \
  --input-lens 16384 32768 --concurrencies 1 4 --repeats 3 \
  --correctness-campaign /results/results/correctness/campaign.json \
  --output-dir /results/results/performance --execute

bash benchmark/xm_sparse/run_h800.sh benchmark/xm_sparse/campaign.py \
  --phase capacity --cache-budget-bytes 8589934592 \
  --model /models/qwen --workload /data/frozen-xm-qwen-20261127/workload.json \
  --input-lens 32768 --concurrencies 1 4 8 12 \
  --arms fused combined-protected --output-dir /results/results/capacity \
  --correctness-campaign /results/results/correctness/campaign.json \
  --execute
```

Performance entry points are run only after the same source and mechanism
configuration pass correctness. A planned command, compilation, or CPU test
is not a real model/performance result.
