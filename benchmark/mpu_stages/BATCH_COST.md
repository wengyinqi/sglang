# Fixed actual decode batch cost

`batch_cost.py` reanalyzes complete, saved single-compute Qwen traces. It does not
run a model. Keep this analysis separate from the closed-loop end-to-end and
capacity results produced by `real_workload_experiment.py`.

## Matching contract

The analyzer requires one scheduler, NoPD, no overlap/speculation/prefix reuse,
complete ordered forward coverage, all input prefill tokens and output-minus-one
decode forwards, and a final idle flush. A stable window has the same real
request UUIDs, stream, execution scope and history-length vector, with symmetric
edge trimming. Different runs map request UUIDs through the benchmark result's
fixed frozen-input row order. Matching requires the same logical input rows and
generated output text hashes; it never stitches disconnected windows.

Control the source/model/runtime/GPU, frozen input file, serving configuration,
KV allocation, preallocated request slots, context/output length, seed, client
concurrency and instrumentation. `varying_fields` can select `hot_buffer`,
`backing_ratio` or `variant`; the rest must match. The caller derives the
`serving_config_sha256` from the normalized complete server arguments, omitting
only the declared treatment fields. If supplied, the CLI checks
`controlled_server_arguments` against that digest and verifies the manifest hash.
Treat an identity as an explicit experimental contract; it must come from the
saved manifest, never from an inferred label such as `C4`.

The optional `actual_batch_size` filters actual scheduler windows. It does not
change client concurrency or pad a smaller batch. A missing common batch produces
`no_matched_windows`, rather than replacing it with each run's maximum batch.

## Replay

Create a JSON spec with `cases`, `varying_fields`, and optionally
`actual_batch_size`. Each case contains `result`, `result_sha256`, `trace`,
`trace_sha256`, and `identity`. `identity` fields are defined in `CONTROL_FIELDS`
and `VARYING_FIELDS` in the source. Preserve the observed case ID. Paths identify
the complete raw trace and its single complete benchmark-result JSON object.

```bash
python benchmark/mpu_stages/batch_cost.py \
  --spec /results/fixed-batch-spec.json \
  --output /results/new-fixed-batch-analysis.json
```

The output records the SHA256 of the spec, analyzer and inputs, exact raw sequence
boundaries, logical input rows, history ranges, execution scope and window
metrics. Use a new output filename; old evidence is never overwritten.

## Metric and evidence scope

`cuda_forward_ms` measures each CUDA-event forward. The scheduler launch span
uses the first launch through a final right-boundary launch; its token count
excludes the right-boundary forward. Its throughput is a batch-window rate, not
per-request TPOT. CUDA sums and scheduler spans are separate scopes: their
difference is not CPU time or pure KV-transfer latency.

New traces may record `seq_lens` and `actual_cuda_graph`; sequence lengths are
checked against complete decode coverage and Graph replay is explicit. Legacy
history is labeled `derived_input_plus_complete_decode_ordinal`, and an enabled
Graph policy remains `unobserved_graph_enabled_policy`. It does not retrospectively
prove replay. Generated text equality does not prove token-ID history equality.

With a fixed actual batch, a larger hot buffer can reduce forward cost while
using more of the same GPU KV pool. Report that cost result beside the separate
same-budget admission/capacity result. Subtracting prefill forward time from wall
time leaves admission, queueing and transport gaps; it does not make differing
batches comparable. Closed-loop client concurrency is also not a fixed external
arrival rate. This tool establishes neither pure DDR bandwidth nor MPU/P2P
performance.
