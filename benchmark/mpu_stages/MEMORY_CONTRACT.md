# Experimental memory accounting and indexer K-cache interface

`experiment_memory_v1` is an opt-in metadata collector for Qwen Quest,
DeepSeek-V4-Flash and GLM-5.3-Flash. It records shapes, dtype, stride, storage
offset and underlying storage bytes. It does not read tensor values, create
packed K/scale views, copy KV, execute kernels, reset CUDA peaks or synchronize.
The caller owns phase boundaries, stream ordering, CUDA allocator sampling,
NVML sampling and actual CUDA Graph pool identification.

The CPU tests validate the accounting contract and concrete owner adapters.
They do not establish GPU allocation sizes, multi-GPU output correctness,
indexer-cache performance, or the DeepSeek/GLM `4+0` model baseline. No indexer
K-cache replacement algorithm is introduced by this interface.

## Caller API

```python
from sglang.srt.observability.experiment_memory import (
    collect_model_tensor_trees,
    collect_model_logical_regions,
    capture_memory_snapshot,
)

owners = {"model_runner": runner}
trees = collect_model_tensor_trees(family, **owners)
regions = collect_model_logical_regions(family, **owners)
record = capture_memory_snapshot(
    context,
    trees,
    regions=regions,
    allocator_metrics=allocator_metrics,
    nvml_samples=nvml_samples,
    graph_pool_summary=graph_pool_summary,
)
```

`context` requires `snapshot_id` and one supported `phase`: `initialization`,
`initial`, `initialized`, `graph_warmup`, `prefill`, `decode`, `stable_decode`,
or `idle`. Include source/model revision, family, rank/world size, process ID,
physical compute-device UUID, workload/run/point identifiers, graph mode,
parallel strategy and the actual request-set fingerprint whenever known.
Every snapshot-local `allocation_group` is local to this process/device/capture;
never join numeric IDs from different snapshots or ranks.

NVML samples use `{timestamp_seconds, used_bytes, device_uuid}`. A series must
have monotonic finite times and a single UUID matching the snapshot compute
device. Its maximum is a **sampled peak**, not an exact peak. Missing data is
`null`; no CUDA/NVML/Graph sample is synthesized by a CPU test.

## What the adapters cover

| Owner | Accounting |
|---|---|
| All families | Parameters, model buffers, request mapping, allocator worklists, immediate backend/wrapper workspaces, Graph input/output/live tensors |
| HiSparse | Hot/LRU tensors, other coordinator tensors, backing payload and descriptors; storage device separates compute HBM, CPU DDR and simulated MPU HBM |
| Qwen | K/V payload and pool metadata; persistent Quest min/max/page bounds via HiSparse selector or HBM backend `.quest` |
| DeepSeek V4 | C4/C128/SWA/unified KV and registry aliases; packed or separate indexer K/scale, compression state, request-window state |
| GLM Flash | Hybrid full DSA KV, packed indexer K, KPool tail K/score, Mamba conv/temporal state and metadata |

Owner discovery is explicit and bounded. Unseen owners, native-library
allocations, transient kernels, allocator slack and hidden/custom communication
buffers are not proven absent. Allocation traces and separately sampled
allocator/Graph/NVML evidence are needed to cover those scopes.

## Byte scopes and normalization

`named_storage_bytes` deduplicates each physical storage using device plus
underlying storage identity. Views retain all tensor aliases. A physical
allocation referenced by multiple categories belongs to `shared_components`
in the exclusive partition; inclusive category totals intentionally overlap.
The exclusive/shared partition sums to the unique named total.

Never add named tensor bytes, allocator allocated/reserved, Graph private-pool
bytes and NVML device usage. They describe overlapping scopes. Likewise,
Graph-live tensors can reside inside the Graph pool, and allocator reserved
already contains allocated blocks and unused cache.

Physical KV/storage bytes include allocator/page padding. The normalization
denominator `preallocated_request_slots` should be the actual
`req_to_token_pool.size`, and `max_context_len` the actual pool width.
`ReqToTokenPool.req_to_token` is shaped `(size, max_context_len)`; the KV pools
may separately contain sentinel/page padding. Report both the configured
logical capacity and physical shapes. `slots * max_context_len` is a
preallocated-history denominator, not the observed number of live tokens.
Zero live tokens yields `null` per-live-token normalization. Use
`num_effective_layers` for model-layer normalization and a separate
`indexer_layers` count for real indexer layers; zero-row skipped GLM layers
remain visible placeholders and are not real indexer layers.

## Packed K and scale regions

`logical_regions` describes the **contents** of an existing physical tensor.
Each entry binds `tensor_name` to its local `allocation_group`, device and
physical storage bytes; `adds_physical_allocation` is always false.

| Layout | Key bytes per indexer token | Scale bytes per indexer token |
|---|---:|---:|
| GLM/DeepSeek FP8, block 128 | `index_head_dim` | `index_head_dim / 128 * 4` |
| DeepSeek packed FP4, block 32 | `index_head_dim / 2` | `index_head_dim / 32` |
| DeepSeek AITER separate FP4 | Actual payload tensor bytes | Actual scale tensor bytes |

Packed pages contain all K bytes followed by all scale bytes. GLM physical
rows have `page_size / index_kpool` indexer tokens; DeepSeek rows use the pool's
page size and record the main-sequence compression ratio separately. Preserve
logical shape, quantization block, layer ID, page size, KPool/compression ratio,
numeric format and known/potential preshuffle layout. A zero-row placeholder
has measured zero content bytes. Missing pools remain unmeasured.

The collector checks packed row shape and region extents/overlap without
calling `GetKAndS`, `get_index_k_scale_buffer`, dequantization or a gather kernel.
The `indexer_cache_accounting` helper deduplicates repeated alias-region
geometry, then reports content bytes separately from physical components.

## Offline validation and comparison

```bash
python benchmark/mpu_stages/memory_contract.py validate snapshot.json
python benchmark/mpu_stages/memory_contract.py indexer snapshot.json --output indexer-ledger.json
python benchmark/mpu_stages/memory_contract.py aggregate rank0.json rank1.json --output ranks.json
```

`aggregate_rank_snapshots` requires distinct ranks, known device UUIDs and an
explicit common `capture_group_id`. It reports whether all ranks were captured.
It sums process-owned named bytes, retains each local-device inventory, and
does not merge allocation IDs across ranks. Cross-process IPC/shared-storage
deduplication is unmeasured. NVML stays per physical device and per series;
peaks from several processes or different time windows are not summed.

`compare_to_dense(dense, candidate, controls=...)` requires the same model
family, phase, rank/world size and physical compute device. Source revisions
must be known; a deliberate source difference requires
`allow_source_difference=True` and is retained in the output.
`controls` is a dictionary of `{dense: value, candidate: value}` for:

- `model_revision`, `weight_format`, `parallelism`, `context_len`, `request_slots`
- `actual_batch`, `request_set_fingerprint`, `output_len`, `cuda_graph`

Each pair must be known, equal and match its actual snapshot context; an equal
caller-supplied pair cannot override real batch/workload/graph differences.
Missing actual controls fail closed. Only initialization/Graph warmup can use
`not_applicable` for actual_batch/request_set_fingerprint/output_len when both
actual context values are missing. Known values must still match. Missing category bytes yield
`null` deltas, not a Dense zero invented from absence. The output separately
reports named storage, inclusive categories, allocator, Graph pool and sampled
NVML deltas. Allocator peak deltas remain `null` unless both captures define
the same measured reset interval. Inclusive category deltas are not additive.
Subtracting prefill does not eliminate admission, queueing or batch differences.

## Future indexer K-cache experiment

`indexer_cache_accounting` exposes independent categories:

- K vectors and quantization scales
- Mapping/LRU and workspace
- Tail K/score and indexer compression state

An implementation should name actual mapping/LRU/workspace tensors using
`indexer_mapping_lru` and `indexer_workspace`. Do not classify all attention
workspaces as indexer workspaces. Optional counters are `lookups`, `hits`,
`misses`, `evictions`, `bytes_read_from_backing`, and `bytes_written_to_backing`.
They require a matching phase and explicit reset interval; missing counters
remain `null`, and hits plus misses must equal lookups when all three exist.
Zero lookups yields an undefined (`null`) hit rate.

Mechanism comparison starts after each model's own `4+0` output baseline runs:
keep checkpoint/quantization, Attention-DP4+EP4, request set, actual batch,
context/output lengths, Graph and backend fixed. Compare that model's HBM
indexer baseline to the proposed K cache, then to CPU/MPU backing with the same
sparse-selection rule. Report K/scale/metadata/workspace bytes and per-indexer-
layer/request-slot/live-token normalization independently of main KV and linear
state. Qwen Quest bounds are a useful metadata accounting reference, but they
are not a trained-indexer baseline; cross-model total byte comparisons do not
establish mechanism superiority. `4+4` needs real peer-access validation.

```bash
python -m pytest -q test/manual/mpu_stages/test_experiment_memory.py
```
