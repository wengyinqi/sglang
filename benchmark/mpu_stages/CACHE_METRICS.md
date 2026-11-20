# CPU-backed Qwen diagnostic cache counters

`SGLANG_HISPARSE_CACHE_METRICS=1` opts into separate CPU-backed Qwen Quest diagnostics. Default is `0`: no counter allocation or updates. All formal performance sweeps keep this off.

The instrumented resolve kernel accumulates five int64 counters per attention layer: real request/layer calls, selected entries, resident hits, newest-token hits, and miss entries. CUDA Graph padding blocks return before accounting. Short sequences use the ordered resident path and count only nonnegative selected entries as resident hits. Miss counts are taken from the actual resolve/copy path, not inferred from CUDA memcpy activity or kernel names.

Idle capacity snapshots include `cache_metrics`. Counters are cumulative for the service process and include warmup/capture execution when it occurs; do not call them per-formal-case totals without explicit before/after scope evidence. Counters are copied to CPU only by the diagnostic idle snapshot provider.

`logical_miss_kv_payload_bytes` equals miss entries times the per-side token stride times two (K and V), summed across layers. This is the logical payload requested by those miss copies. It excludes prefill backup, new-token writeback and selector metadata, and does not measure hardware transactions, DDR bandwidth, DMA traffic or MPU speedup. Counter-enabled latency/throughput must not be mixed with counter-disabled formal results.

CPU accounting validation: `test/manual/hisparse/test_cache_metrics_accounting.py`.
Actual CUDA accounting/data/graph validation: `python/sglang/jit_kernel/tests/test_hisparse_cache_metrics.py`.
