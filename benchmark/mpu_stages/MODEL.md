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
union of actual prefill forward intervals from the measured run wall time;
do not sum per-request TTFTs or TP-rank durations. Preserve both original and
estimated throughput, generated-token denominator and graph/timing overhead.

`host_to_device_ratio` participates in C4 GPU sizing. Freeze it and the actual
compute C4/SWA/C128/index pools when comparing backing capacity. Expose MPU
capacity independently; a logical address/indexer cap can limit the useful
ratio even when the backing allocation has space. The H200 simulator tests
CUDA peer load/store, not an unimplemented physical MPU driver or its price.
