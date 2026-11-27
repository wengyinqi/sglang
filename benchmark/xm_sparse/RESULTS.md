# Qwen Quest adaptation of XM-Sparse software mechanisms

This research branch extends stage 01 at
`b61d8c7f1529ff1971026ae68dcc42af80e7a592`. Experiments use one H800 PCIe,
BF16 Qwen3-30B-A3B-Thinking-2507, full historical KV in CPU pinned DDR,
and frozen LongBench-v2 document inputs. It is a Qwen adaptation: the offloaded
index is Quest page min/max bounds, not the paper's trained DeepSeek/GLM K-index.
There is no MPU fabric, P2P or expert-parallel performance evidence here.

## Mechanisms and attribution

Six arms separately enable fused retrieval/attention, summary offload, cross-layer
lookahead and a CUDA checkpoint gate. Fused attention consumes the cache plan
without gathering selected KV into a separate HBM tensor. A double-buffered TMA
pipeline retrieves hot or mapped-DDR KV to shared memory; BF16 MMA implements
QK/PV and row sums with FP32 accumulation, online softmax and split reduction.
Original selected-token order is preserved to match the FlashInfer reference.

Historical BF16 min/max summaries have layout `[layers, requests, pages, 4, 128, 2]`.
GPU running bounds and two layer stages remain resident. Both the on-demand and
lookahead arms use GPU-launched device child memcpy graphs: their difference
isolates cross-layer lookahead, not GPU submission versus CPU submission.
Copies currently cover preallocated request/page rows. They do not implement
budget-driven page eviction, cross-step early-layer prefetch, or the paper's
native CE semaphore. The checkpoint gate is a CUDA adaptation with no EP proof.

## Retained source006 B1 results

Model source `226df0df3808d529b1123d4e32b2430d4b935b40` (source006) passed
24 real-model groups: six arms, CPU-DDR and independent HBM Quest, eager and
CUDA Graph, with mixed 16K/32K submitted inputs and 80 output tokens.
Actual traces observed B2 windows on both CPU and HBM. Some CPU reuse rounds
also contained B1 forwards; this older protocol did not require a complete,
synchronized cohort from the first decode forward. Per-request output matching
and observed B2 windows do not imply that every decode forward was B2.
All greedy tokens and selected-token logprobs matched exactly; reuse,
abort and retract checks passed. This does not establish larger-batch coverage.

The separate formal timing experiment used 16K input, actual B1 / client
concurrency C1, 128 output tokens, and three paired repetitions per arm.
Complete HTTP mean TBT uses 127 single-token arrival intervals; TPOT includes
the completion endpoint tail. Stable scheduler intervals use a shorter,
explicitly selected decode window and are a separate metric.

| Arm | Complete HTTP mean TBT (ms) | TPOT (ms) | Stable scheduler interval (ms) |
|---|---:|---:|---:|
| baseline | 13.112 | 13.114 | 12.571 |
| fused | 16.103 | 16.104 | 15.652 |
| bounds-on-demand | 14.535 | 14.536 | 13.964 |
| bounds-prefetch | 13.908 | 13.909 | 13.335 |
| combined | 16.834 | 16.836 | 16.386 |
| combined-protected | 17.008 | 17.010 | 16.535 |

- Fused versus baseline: complete HTTP mean TBT increased 22.81%; the paired
  reduction interval is [-24.90%, -20.71%]. This implementation is slower at B1.
- Lookahead versus on-demand: complete HTTP mean TBT decreased 4.31%; paired
  95% interval [3.67%, 4.95%]. Offloaded lookahead still trails the original
  resident-summary baseline in this experiment.
- The checkpoint effect's confidence interval crosses zero; benefit or cost
  is unresolved. There is no measured EP traffic-protection effect.

Intervals use three paired-repeat percentages and a df=2 t interval. Token
gaps are not independent repetitions. Execution order was not position-balanced
and no multiple-comparison correction was applied. These are hardware/model-
specific measurements, not reproduction of the paper's numeric results.

Independent source006 Nsight decode captures support the timing direction.
The following activities are cross-layer aggregates averaged over parent/main
decode Graphs, not a single layer's latency: fused attention activity is
4.134 ms plus 0.204 ms split merge, versus
baseline 0.572 plus 0.134 ms. Saved plan/swap and packing activity is about
0.586 and 0.074 ms and does not offset this regression. Lookahead reduces
summary wait activity from 0.883 to 0.089 ms; its main graph envelope changes
from 13.089 to 12.429 ms. Activity sums are not a critical-path sum. Child-DMA
events and CE bandwidth counters were unavailable, so CE bandwidth is unknown.

The fixed physical GPU KV pool is 6.005859 GiB; full CPU KV backing is
48.000092 GiB. Named cache/backend allocations give the following net HBM
storage reductions, with aliases deduplicated:

| Path versus baseline | Named HBM reduction (MiB) | Added DDR summary backing (MiB) |
|---|---:|---:|
| fused | 383.714840 | 0 |
| summary offload, on-demand or lookahead | 23.269421 | 24.281250 |
| fused plus summary offload | 406.984261 | 24.281250 |

The baseline's 384 MiB backend workspace becomes 0.261719 MiB plus a 24,580-byte
plan already counted in cache inventory. These are named storage deltas,
not NVML process-total savings, a live-hot-KV measurement or demonstrated
admission/concurrency gains. Phase-specific temporary/Graph memory attribution
is not yet complete. SLO limits and hourly hardware costs remain unspecified.

## B4 numerical repair and evidence boundary

Controlled source014 (`e3e8aa08e303665f39461966c0e6c3f2e56630a4`) baseline
passed full 16K B4 CPU/HBM eager/Graph. Fusion failed the independent HBM
comparison: first greedy divergences were zero-based output indices 42, 59,
50 and 105 (generated tokens 43, 60, 51 and 106);
same-history maximum logprob errors were 0.0947, 0.0798, 0.1175 and 0.1405,
above the unchanged 0.02 gate. Both paths had complete synchronized B4 from
the first decode-forward ordinal 1 (the first generated token comes from
prefill). This was an actual numerical failure; its performance and
capacity campaigns were not run.

The repaired kernel preserves the reference FA2 empty-warp rescaling in the
last 128-row iteration: a rounded FMUL followed by FFMA and approximate EX2
is not necessarily an exact identity. The retained worst-layer36 snapshot
had six BF16 output differences before the repair; frozen source016 replay
eliminated them. Four real layer snapshots (0/23/36/47), a local-HBM selected-KV
arithmetic reference, zero-hot DDR retrieval and Graph replay all matched
bitwise. This snapshot reference is not a full HBM Quest model experiment.

Source017 (`3ca21f0089b30e13b254491f2b086f073ad11f10`) changes only an outdated
CPU profiling fixture relative to source016. Their 1,973-file production-core
fingerprint is identical:
`d763476a8bc735c8197a9c0208a49493e0c4a66eb9c34974b46ccd5615fcade0`.
The complete source017 Linux/H800 suite passed 328 tests and 26 subtests with
no skips, including the four real snapshots and B4/B8/B12 dynamic partitions.
Source016's earlier fixture failure and skipped fixture-dependent suite test
remain recorded rather than relabelled as passed.

The source017 16K B4 fused arm has completed all four CPU-DDR/independent
HBM Quest eager/Graph groups, including long/short/reuse and lifecycle checks:
all greedy token IDs match and maximum selected-token logprob error is zero.
Raw replay independently verified 127 synchronized B4 decode forwards per
long round in every group, starting at decode ordinal 1. The remaining five
arms are still pending; no six-arm B4 or source017 performance conclusion is
admitted yet. Kernel snapshots, native
instructions and synthetic batch tests do not prove full greedy generation.
Subsequent measurements must retain the exact greedy-token
and 0.02 logprob gate, record actual batch/cohort and separate source revisions.
Controlled cohort TTFT includes the prefill/staging barrier; capacity scans
use ordinary NoPD admission with the barrier off. Subtracting prefill time
does not remove admission, queue or cold-prefill memory effects.
