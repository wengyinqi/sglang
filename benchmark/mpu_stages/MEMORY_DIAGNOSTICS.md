# Isolated memory diagnostics

`memory_probe.py` runs the Qwen single-compute Dense, HBM Quest or CPU DDR
Quest reference using frozen input IDs. It fixes a 65536-token compute KV
budget, eight request slots, hot4096/topk2048/page64, configured context33794,
one active request and 128 output tokens. Prefill piecewise Graph is explicitly
disabled; `--eager` changes decode execution. Actual phase execution is recorded.

Use a fresh output directory, a committed source archive with `.mpu-source.json`,
the matching Qwen runtime, model and frozen workload. This runner is deliberately
synchronizing; never use its latency or throughput as performance evidence.

```bash
python benchmark/mpu_stages/memory_probe.py \
  --model /models/qwen --workload /data/frozen-longbench-qwen/workload.json \
  --variant hbm --eager --output-dir /results/memory-hbm-eager
```

The shared observer is disabled by default. Native DeepSeek/GLM launch recipes
can set `SGLANG_EXPERIMENT_MEMORY_DIR` and a JSON
`SGLANG_EXPERIMENT_MEMORY_CONTEXT` containing their family and source/model/rank
controls. They retain Attention-DP4+EP4. A single-rank Qwen runner cannot validate
their GPU model execution or substitute for four-rank evidence.

`SGLANG_EXPERIMENT_MEMORY_HISTORY_ENTRIES` controls the allocator-history ring
(default500000). Long eager selector loops may exhaust this ring; a truncated
transient result is null, never zero. A diagnostic-only recovery may increase
the limit (for example2000000) and must retain the source identity and setting.
`SGLANG_EXPERIMENT_MEMORY_DECODE_SAMPLES` chooses forward checkpoints; the
default is1,8,64,127. The scheduler writes a final decode and idle receipt only
after releasing request resources. Startup/health requests are separate waves.

Named physical storage, phase-local allocator peaks, Graph pool segments and
NVML device samples overlap. Compare each scope independently. Storage IDs are
snapshot-local; logical packed K/scale regions are content within an allocation.
Temporary allocation attribution counts only recorded allocations born and
logically freed within the phase. Crossing-phase scratch and Graph-retained
scratch are excluded from that subset. NVML maxima are sampled peaks.

The offline `memory_contract.py` validates snapshots, computes controlled
same-model Dense deltas and aggregates explicit rank capture groups. Unknown
owners, workspace attribution and unsupported Graph wrappers remain unavailable.
