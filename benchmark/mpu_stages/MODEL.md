# DeepSeek-V4-Flash HiSparse stages

Source baseline: official main `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`.
Stage 03 uses four H200 compute GPUs and CPU pinned DDR C4 backing (`4+0`).
The upstream NoPD allocator, prefill staging, decode backup and miss swap-in
are present. This branch fixes the CUDA scan window for large HiSparse buffers.
A CPU configuration check is not model/GPU validation.

## H200 start point

Use Linux, the pinned branch's CUDA build dependencies and the official
`deepseek-ai/DeepSeek-V4-Flash` FP4 checkpoint. TP4/Marlin follows
`test/manual/dsv4/test_h200_fp4_flash.py`. FP8 repacks are a different baseline.
Start without speculative decoding, DP, PP, CP or PD. Do not copy EAGLE flags
from the general model cookbook: upstream rejects DSV4 HiSparse speculation.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --tp-size 4 --moe-runner-backend marlin \
  --attention-backend dsv4 --dsv4-attn-backend flashmla \
  --enable-hisparse --disable-radix-cache \
  --hisparse-config '{"top_k":512,"device_buffer_size":1024,"host_to_device_ratio":2,"swap_in_block_size":960}'
```

The checkpoint's `index_topk` overrides the JSON top-k; record its actual value.
C4 CPU backing is layer-first `uint8[pages,37440]`, 64 compressed slots/page:
576 value bytes per slot followed by a separate 8-byte scale region.
SWA, C128, C4 indexer and compression state remain on compute GPUs. `+0` does
not mean full-HBM KV. The optional full-HBM reference needs a separate name.

## Required acceptance (hardware pending)

Compare CPU-backed and MPU-backed runs with identical checkpoint, precision,
request token IDs/seeds, top-k/hot buffer, GPU pool sizing and parallel settings.
First verify deterministic token/logprob equality for short and long prompts,
C4/page boundaries, decode past capacity changes, repeated requests, aborts and
retractions; then enable graph replay. Check value and scale bytes separately.
Test prefill lengths not divisible by 4 or 256 and at least 257 decode tokens.
Long contexts must force actual C4 misses before any performance interpretation.

Report TTFT and TPOT as observed. For a NoPD decode-only estimate, subtract the
sum of matched prefill batch durations from measured run wall time, taking
max(TP-rank local CUDA duration) for each serial NoPD prefill batch. This is
an approximation: rank-local CUDA clocks do not provide an exact cross-device
interval union. Do not sum per-request TTFTs or TP-rank durations. Preserve both original and
estimated throughput, generated-token denominator and graph/timing overhead.

`host_to_device_ratio` participates in C4 GPU sizing. Freeze it and the actual
compute C4/SWA/C128/index pools when comparing backing capacity. Expose MPU
capacity independently; a logical address/indexer cap can limit the useful
ratio even when the backing allocation has space. The H200 simulator tests
CUDA peer load/store, not an unimplemented physical MPU driver or its price.


## Stage 04: four compute GPUs plus four storage GPUs

Use the same recipe and expose `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
Keep every model/GPU sizing option unchanged and replace only the JSON config:

```json
{"top_k":512,"device_buffer_size":1024,"host_to_device_ratio":2,"swap_in_block_size":960,"backing_store":"mpu","mpu_device_ids":[4,5,6,7],"mpu_capacity_ratio":2.0,"mpu_memory_reserve_bytes":1073741824}
```

TP rank i runs on CUDA-visible GPU i and allocates C4 on GPU 4+i. The four
storage GPUs run no model, indexer, collective or KV copy/initialization kernels.
Compute-side coherent `.cg` loads/stores copy value and scale regions; peer
writes include a system fence. Existing HiSparse streams/events retain producer
ordering. Storage lifetime ends only after compute work drains. Native CUDA
allocations and actual non-VMM allocator segments are checked; unsupported peer
access fails at startup. There is no CPU KV fallback or CPU-to-MPU KV staging.
`torch.empty` allocates storage; layer views only change host-side metadata.

`mpu_capacity_ratio` applies to usable C4 compressed slots per rank relative to
the physical compute C4 allocator, rounded up to 64-slot pages. It does not
change `host_to_device_ratio` or GPU index pools. Capacity beyond the logical
full-token/index space is reported as allocated backing and cannot create more
addressable requests. For a meaningful capacity sweep, freeze common logical
and GPU budgets high enough for the entire sweep; compare observed capacity
and throughput, including points limited by other pools. Record all four ranks.
The code uses conservative input + maximum output lifetime page leases for
both CPU and MPU runs; waiting requests cannot consume more backing than the
pool. This policy must stay identical between variants.

Use the official model ID above, or a local snapshot of exactly that checkpoint;
record its resolved Hugging Face revision and tokenizer revision before model
acceptance. No checkpoint was downloaded and no revision was guessed here.

```bash
python -m pytest test/manual/mpu/test_dsv4_cpu.py -q
# After CUDA dependencies and two peer-accessible GPUs are available:
PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False \
  python -m pytest test/manual/mpu/test_dsv4_gpu.py -q
```

Current validation is CPU-only. GPU tests cover split value/scale page-boundary
roundtrips, repeated graph replay and 2048-selected/4096-hot-buffer miss plans
with 256/960/1024-thread planners. They are unrun here. Complete eight-GPU TP4
model tests, abort/retract and capacity sweeps with the shared stage runner;
profile the storage GPUs to verify the no-kernel contract before accepting a
performance result. No hardware throughput, TTFT, TPOT or accuracy result is
claimed by this branch.

Local check result: 15 focused CPU tests passed, 4 CUDA tests skipped. The
upstream allocator unit test could not collect in Windows because the Unix
`resource` module is unavailable; it must still run in the Linux CUDA environment.
