# DeepSeek-V4-Flash HiSparse stages

Source baseline: official main `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`.
Stage 03 uses four H200 compute GPUs and CPU pinned DDR C4 backing (`4+0`).
The upstream NoPD allocator, prefill staging, decode backup and miss swap-in
are present. This branch fixes the CUDA scan window for large HiSparse buffers.
A CPU configuration check is not model/GPU validation.

## H200 start point

Use Linux and `sgl-project/DeepSeek-V4-Flash-FP8`, revision
`ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17`, with `SGLANG_DSV4_FP4_EXPERTS=0`.
The H200 FP8 recipe is grounded in `test/manual/dsv4/test_h200_fp8_flash.py`.
Both CPU and MPU stages use Attention-DP4 + EP4, effective attention/dense/MoE
TP=1, DeepGEMM and DeepEP. No speculative decoding, PP, CP or PD.
FP4/Marlin is not a DeepEP-compatible replacement in this source revision.

```bash
SGLANG_DSV4_FP4_EXPERTS=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 --trust-remote-code \
  --revision ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17 \
  --tp-size 4 --dp-size 1 --attn-dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --enable-dense-mlp-attn-tp --enable-dp-lm-head \
  --moe-runner-backend deep_gemm --moe-a2a-backend deepep --deepep-mode auto \
  --disable-overlap-schedule --load-balance-method round_robin \
  --cuda-graph-config '{"prefill":{"backend":"disabled"}}' \
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

Report TTFT and TPOT as observed. Decode-only estimation subtracts only pure
prefill EP steps using the maximum CUDA duration across four ranks per step.
Mixed prefill/decode steps remain in the denominator. Requests belong to one
Attention owner each; idle ranks still participate in EP. See
[ATTENTION_DP_EP.md](ATTENTION_DP_EP.md) for alignment and limitations.

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

Attention rank i runs on CUDA-visible GPU i and allocates C4 on GPU 4+i. The four
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

The pinned FP8 reference was fetched on 2026-10-27; metadata was hashed and
46 weight-shard checksums were recorded without downloading weight payloads.
Old FP4/Marlin results cannot be used as this recipe's CPU/MPU baseline.

The default `--max-total-tokens=131072`, input lengths 8192/32768, and concurrency
4/8/16 are unmeasured starting settings. Correctness generates 257 tokens and
uses original-token page boundaries of 256 (64 compressed C4 rows). Prefill
still needs compute KV capacity before staging; an allocation/startup failure
is a failed setup, not evidence of low throughput. Calibrate fit on H200.
For a 1/2/4/8 MPU sweep, set `--cpu-capacity-ratio 8 --capacity-ratios 1 2 4 8`
before running the complete CPU/MPU comparison. This freezes the shared host
index/address budget at 8; enlarge the common total token budget if needed,
then require identical measured compute pool sizes across every point.

```bash
python -m pytest test/manual/mpu/test_dsv4_cpu.py -q
# After CUDA dependencies and two peer-accessible GPUs are available:
PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False \
  python -m pytest test/manual/mpu/test_dsv4_gpu.py -q
```

Current validation is CPU-only. GPU tests cover split value/scale page-boundary
roundtrips, repeated graph replay and 2048-selected/4096-hot-buffer miss plans
with 256/960/1024-thread planners. They are unrun here. Complete eight-GPU Attention-DP4+EP4
model tests, abort/retract and capacity sweeps with the shared stage runner;
profile the storage GPUs to verify the no-kernel contract before accepting a
performance result. No hardware throughput, TTFT, TPOT or accuracy result is
claimed by this branch.

Local check result: 15 focused CPU tests passed, 4 CUDA tests skipped. The
upstream allocator unit test could not collect in Windows because the Unix
`resource` module is unavailable; it must still run in the Linux CUDA environment.
