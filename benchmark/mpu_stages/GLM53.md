# GLM-5.3-Flash: NoPD TP4 HiSparse stages

These branches implement CPU DDR (4+0) and MPU (4+4) backing for the complete
DSA main KV history. `4+0` does **not** mean full GPU KV. Model execution,
DSA index K, and KDA recurrent/convolution state stay on the four compute GPUs.
No model or CUDA run has been performed in the Windows development environment.

## Source and model pins

- SGLang base: `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`.
- Checkpoint: `zai-org/GLM-5.3-Flash`, revision
  `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` (configuration inspected).
- Architecture: `Glm5NextForConditionalGeneration`, 45 text layers: 11 DSA
  layers at `[3,7,11,15,19,23,27,31,35,39,43]` and 34 KDA layers.
- The checkpoint has FP8 weights with excluded modules. BF16 below is the
  **KV cache** format; it does not request conversion of every weight to BF16.
- DSA latent width 512, RoPE cache width 0; each BF16 main-KV row is 1024 bytes
  per DSA layer. Index K uses a separate compressed pool and remains on GPU.
- `index_topk=2048`, `index_kpool=4`, `index_kpool_always_select_tail=true`:
  the physical selection width is 2051, including three independently valid
  tail columns. The hot buffer is 4096 plus its reserved append page.

## 4+0 CPU DDR baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path zai-org/GLM-5.3-Flash \
  --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
  --tp-size 4 --pp-size 1 --dp-size 1 --attn-dp-size 1 \
  --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_sparse \
  --kv-cache-dtype bfloat16 --page-size 64 \
  --disable-radix-cache --enable-hisparse \
  --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}' \
  --max-running-requests 4 --context-length 32768 \
  --reasoning-parser auto --tool-call-parser auto
```

No speculative/MTP, PD, attention DP/CP, or additional HiCache/LMCache tier is
enabled. On H200 the existing KPool resolver changes `flashmla_sparse` to FA3
when tail columns are present; record this effective backend as well as the CLI.
Keep this same attention path for the CPU and MPU comparison. The upstream
H200 regression recipe uses TP8, so TP4 startup/accuracy/capacity remains a
hardware acceptance item rather than a measured result here.

The host pool preserves `[11, backing_slots, 1, 512]` layer-first BF16 layout,
with 1024-byte token stride. Its existing allocator rounds CPU capacity up by
an extra page. `get_stage_capacity_info()` reports allocated payload bytes,
actual slots, actual used slots, requested/effective ratio, and excluded state.
Physical compute KV remains bounded independently of full backing history.
It also reports the compute padding page, backing padding beyond logical
capacity, and the logical admission limit. Logical history slots remain reserved
when their hot-cache mapping is released; the paged logical allocator rejects
decode growth at capacity and releases those slots only at request cleanup.

## Validation

Portable checks (run from the repository root):

```bash
python -m unittest discover -s test/manual/mpu_stages -p 'test_glm53_cpu.py'
```

The tests execute production AST definitions with CPU tensors: compact DSA
construction, wrapper/KDA ownership, mapping, KPool tails, kernel-call ABI,
scope rejection, capacity accounting, and real paged-allocator history lifetime,
decode exhaustion and release. They do not compile CUDA kernels.

Hardware acceptance must check startup, logits/selected locations versus the
same checkpoint and dense-history oracle, long-context hot-buffer eviction,
chunked prefill, decode append, abort/reuse, overlap and CUDA graphs. Then run
the common staged benchmark with fixed input/output lengths and concurrency;
retain per-rank capacity/timing traces and confirm all backing slots are freed
after the sequence. Performance numbers remain pending.

## Implementation provenance

Hybrid child-pool wiring, compact layer mapping, KPool tail handling and bounded
prefix-scan windows adapt the relevant pieces of PR #43533, inspected at
`8cfd1b6a2196a2c1ddc847724a57e4c4dd793298`. The PR was open and unmerged at
inspection. Its NIXL/shared-memory PD transport, author benchmarks and MTP
claims are not part of this NoPD baseline.
