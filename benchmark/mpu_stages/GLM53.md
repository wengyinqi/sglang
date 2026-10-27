# GLM-5.3-Flash: NoPD Attention-DP4 + EP4 HiSparse stages

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
  --tp-size 4 --pp-size 1 --dp-size 1 --attn-dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --enable-dense-mlp-attn-tp --enable-dp-lm-head \
  --moe-runner-backend deep_gemm --moe-a2a-backend deepep --deepep-mode low_latency \
  --disable-overlap-schedule --load-balance-method round_robin \
  --cuda-graph-config '{"prefill":{"backend":"disabled"}}' \
  --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_sparse \
  --kv-cache-dtype bfloat16 --page-size 64 \
  --disable-radix-cache --enable-hisparse \
  --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}' \
  --max-running-requests 4 --context-length 32768 \
  --reasoning-parser auto --tool-call-parser auto
```

No speculative/MTP, PD, CP, or additional HiCache/LMCache tier is
enabled. On H200 the existing KPool resolver changes `flashmla_sparse` to FA3
when tail columns are present; record this effective backend as well as the CLI.
Keep this same attention path for the CPU and MPU comparison. The upstream
H200 regression recipe uses TP8, so Attention-DP4 + EP4 startup/accuracy/capacity remains a
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

## 4+4 MPU backing

Use the same command above, expose `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
and replace the HiSparse JSON with:

```json
{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2,"backing_store":"mpu","mpu_device_ids":[4,5,6,7],"mpu_capacity_ratio":2.0,"mpu_memory_reserve_bytes":1073741824}
```

Each rank `i` executes on compute ordinal `i` and stores its complete DSA main
KV history on ordinal `4+i`. The standalone MPU pool uses one native allocation
of `[11, backing_slots, 1, 512]` BF16 per rank. Compute kernels initialize and
read/write the peer allocation through CUDA UVA/P2P; storage GPUs launch no KV
kernels. CPU holds page-allocation metadata only. P2P/allocator/layout failures
abort startup; they never construct a CPU KV fallback.

The existing HiSparse kernel computes the same selection/LRU miss plan with
payload I/O disabled, then a compute-GPU peer kernel replays that plan. Prefill
and decode append use the same peer record-copy interface. Shared-index
prefetch replays on the compute prefetch stream. Request free conservatively
synchronizes compute streams before returning backing pages, and pool destroy
drains them before dropping the allocation. These synchronization choices may
affect throughput and require hardware measurement.

Linux/NVIDIA CUDA, the native non-expandable allocator, an accessible directional
P2P pair and enough storage memory are required. Each storage allocation leaves
the configured reserve (default 1 GiB) unallocated. The tested local environment
has no CUDA device, so the CUDA modules have not been compiled or executed here.

### Capacity sweep contract

Set one fixed `--max-total-tokens` compute budget for every sweep point; do not
let free-memory profiling choose a new compute KV allocation between runs.
`host_to_device_ratio` fixes the logical-address/DSA-index budget on compute
GPUs, while `mpu_capacity_ratio` independently sizes the exposed MPU payload.
Select a fixed address ratio at least as large as the largest MPU ratio in the
sweep and use it for **all** CPU/MPU variants. For example, a 1x/2x/4x MPU sweep
uses `host_to_device_ratio=4` throughout and varies only `mpu_capacity_ratio`.
This increases fixed GPU index storage, so first validate that that fixed budget
fits; it does not increase the physical main-KV hot pool.
The provider also supplies the common metadata-only retained-pool signature:
hot main KV, DSA index K, KPool tail K/score, every KDA `mamba_cache` tensor and ReplaySSM
cursor buffers. The common timer records storage-deduplicated bytes and each
tensor's shape, dtype, strides, offset and alias group so the benchmark can
reject compute-state drift between CPU/MPU runs and capacity points.
Those retained bytes overlap `compute_kv_bytes`; do not add the two counters.

The MPU allocation rounds down to a whole page. Actual logical admission is
`min(fixed logical/index slots, MPU slots)`. Increasing MPU allocation beyond
the fixed index limit cannot improve usable history capacity; the provider
reports those bytes as `unaddressable_backing_slots`, not padding. Plot usable
capacity using `logical_admission_slots`/`usable_capacity_ratio`, and retain
`backing_slots`/`effective_capacity_ratio` as allocated-capacity measurements.
Serving request-length limits are capped by this logical admission limit and
retain the original physical prefill limit. NoPD stages KV only after the final
prefill chunk, so one input must still fit compute KV; additional backing does
not implement streaming prefill beyond that limit. The CPU
baseline has its existing extra backing page and reports it as padding.

For a full-HBM correctness oracle, omit `--enable-hisparse` and
`--hisparse-config`, keep BF16 KV, model revision, KPool attention backend and
parallelism identical, and choose an input size that fits its full GPU history.
That oracle is a separate validation variant; the requested `4+0` stage is CPU
DDR-backed HiSparse.

## Validation

Portable checks (run from the repository root):

```bash
python -m unittest discover -s test/manual/mpu_stages -p 'test_glm53*.py'
```

The tests execute production AST definitions with CPU tensors: compact DSA
construction, wrapper/KDA ownership, mapping, KPool tails, kernel-call ABI,
scope rejection, capacity accounting, and real paged-allocator history lifetime,
decode exhaustion and release. They do not compile CUDA kernels.
MPU tests additionally cover exact ordinal pairing, invalid configuration,
fixed-address capacity clamping, the standalone peer-allocation geometry,
chunk/append metadata allocation, exhaustion/reuse, compute pointer tables,
peer transfer/replay dispatch, shutdown ordering and capture dummy locations.

The opt-in GPU transfer test is:

```bash
SGLANG_TEST_GLM53_MPU=1 python -m unittest discover -s test/manual/mpu_stages -p 'test_glm53_gpu.py'
```

It exercises one compute/storage pair from the eight-device topology, BF16
all-layer writeback, per-layer load, fixed-size miss replay with invalid padded
rows, and CUDA graph replay. It does not load the model or establish Attention-DP4 + EP4 model
equivalence. It is skipped in the local CPU test run.

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
