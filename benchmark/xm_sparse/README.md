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
The Qwen kernel uses BF16 Tensor Core MMA for QK, PV and probability row
sums, with FP32 accumulation and the frozen FlashInfer FA2 softmax,
partition and partial/cascade rounding order. This is a Qwen GQA kernel,
not the paper's FlashMLA implementation. Its measured effect belongs to
this explicit implementation and hardware pair. The initial scalar FP32
implementation remains available only for arithmetic diagnostics; its
failed real-model comparisons are retained as separate evidence.
The numerical-compatibility implementation preserves selected-token order;
it does not retain the initial resident-first mathematical traversal. TMA
still retrieves K/V directly into double-buffered shared memory and the
kernel refills hot slots. The CPU partition planner follows the frozen FA2
scheduler using the actual device SM count and per-step lengths. It updates
a four-byte Graph-stable chunk-size tensor outside capture; no FlashInfer
planning workspace is retained. Model and larger-batch acceptance remain
separate from the isolated numerical probes.

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

Summary offload currently covers the full preallocated historical summary
tensor. Each immutable child graph copies every allocated request/page row,
not only live pages. This branch does not yet implement the paper's partial
budget-driven page eviction/restoration, measured transfer-budget admission,
or prefetching early layers at the end of the preceding decode step.

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
  --input-lens 16384 --concurrencies 1 --num-prompts 1 \
  --output-len 128 --repeats 3 \
  --correctness-campaign /results/results/correctness/campaign.json \
  --output-dir /results/results/performance --execute

bash benchmark/xm_sparse/run_h800.sh benchmark/xm_sparse/campaign.py \
  --phase capacity --cache-budget-bytes 8589934592 \
  --model /models/qwen --workload /data/frozen-xm-qwen-20261127/workload.json \
  --input-lens 32768 --concurrencies 4 8 12 --capacity-ratio 8 \
  --max-total-tokens 98304 --output-len 512 --repeats 3 \
  --arms fused combined-protected --output-dir /results/results/capacity \
  --correctness-campaign /results/results/correctness/campaign.json \
  --execute
```

Performance entry points are run only after the same source and mechanism
configuration pass correctness. A planned command, compilation, or CPU test
is not a real model/performance result.

For a four-request performance point, first run additional real model
correctness with four equal input lengths. `correctness.py` sets the requested
cohort from the number of lengths in each round, so a `16384 32768` campaign
requests two members. The trace must independently prove its actual batch.
Four 16K/32K requests with 128 output tokens require 66048/131584 paged KV
slots, plus an explicit conservative admission allowance of
`4 * maximum_round_batch * page_size`: 1024 slots at B4/page64. Use
67072/132608 requested slots for these proofs. Exact full-KV bytes alone do
not cover the ignore-EOS admission reserve. The HBM reference must fit on
the actual device; do not waive its output comparison.

Correctness, large-batch CPU summary controls, and same-batch performance
with B>1 enable the isolated `--xm-correctness-cohort-barrier`. It continues
prefill and staging collection while deferring decode until the already
submitted cohort is ready. The raw replay requires every decode forward
to contain the complete cohort with equal output ordinals, starting at
token one. Full greedy token equality and the 0.02 logprob tolerance are
unchanged. An unadmittable cohort fails after 30 seconds without prefill
progress; it is never released as a partial batch. Short rounds and
one-request lifecycle probes do not wait for a maximum batch size.

This is a validation and same-cohort control, not a serving optimization.
Its TTFT includes waiting for the cohort's prefill/staging and must not be
presented as ordinary NoPD serving TTFT. B1 timing and capacity scans leave
the barrier off. The original source006 B4 failures remain separate evidence:
CPU reuse had mismatched output ordinals, and the 32K HBM reference never
observed B4. Different schedule shapes were observed; the cause of their
logprob and greedy differences has not been established. New controlled
measurements use a new frozen source and are reported separately from
source006 B1/Nsight results.

The capacity scan uses CPU backing ratio eight to retain complete contexts
without exceeding the H800 server's pinned-memory NUMA allocation domain.
The launcher binds memory to node zero (about 125 GiB); a nine-GiB physical
KV pool with ratio sixteen would request 144 GiB and cannot fit there, even
though the host has about 251 GiB in total. Check allowed NUMA nodes as well
as host/cgroup memory. Verify the runtime's actual backing
slots can hold all requested contexts and output tokens before attributing
a plateau to HBM. An observed scan maximum is not an exhaustive maximum or
a cost/SLO optimum.
Choose a cache budget that actually creates pressure within the scanned
concurrencies. At 32K and at most twelve clients, an 8-GiB budget plus the
65536-token physical cap can leave both paths below the admission limit;
that point alone cannot establish a summary-offload capacity gain. The
98304-token cap in the example exceeds what the 8-GiB cache budget can
allocate, so the measured memory resolver determines physical capacity.
Runtime allocations and stable batch traces determine the boundary.

NoPD chunked prefill retains the current request's complete GPU KV history
until CPU backup and hot-buffer admission finish. A one-chunk estimate of
prefill staging memory is insufficient. Check full prefill KV plus all
concurrently allocated hot buffers (including the newly admitted request)
against actual physical slots. Cold NoPD admission can therefore limit
batch size before the decode-only hot-buffer limit. Subtracting prefill
time does not remove this memory/admission constraint. Report the observed
cold NoPD batch, and do not label it an isolated maximum decode capacity.
The allocated GPU KV pool includes hot payload, prefill storage and unused
reserve; it must not all be labelled live hot KV.

`analyze.py campaign` replays raw results and scheduler traces and emits
JSON, CSV and optional PNG/SVG. `profile_decode.py` adapts only an actually passed
model correctness receipt to an independent Nsight decode capture.
`report.py` renders the analyses and can append new pages to an existing
PDF without replacing it. `telemetry.py` records low-rate host NVML totals
and observed last-forward progress; it does not measure CE utilization or
assign a stale progress record to a currently active model phase.

For an independent decode diagnostic, explicitly mount the installed
Nsight tools. The ordinary correctness/performance launcher does not enable
the additional profiling container options:

```bash
XM_NSIGHT_ROOT=/home/user/Workspace/mpu/tools/nsight-2026.6.1 \
  bash benchmark/xm_sparse/run_h800.sh benchmark/xm_sparse/profile_decode.py \
  --correctness-campaign /results/results/correctness-001/campaign.json \
  --arm fused --input-len 16384 --batch-size 1 --output-len 256 \
  --profile-steps 32 --stable-steps 16 \
  --output-dir /results/results/profile-16k-b1-fused-001 --execute
```

The replay also preserves raw HTTP frame timing. A strict client mean/P99
TBT is reported only when every token-positive frame contains one token.
Multi-token frames retain their observed frame gaps and leave strict TBT
unavailable. This full-request client statistic is separate from matched
stable scheduler decode intervals and cannot be assigned to a constant
batch window without additional matching.
