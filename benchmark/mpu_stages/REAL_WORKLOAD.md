# Real-document CPU DDR / HBM Quest experiments

This extension keeps stage 01 as Qwen + Quest + HiSparse with CPU DDR backing.
The additional HBM recipe retains Quest and stores full KV on the same compute
GPU. It is a storage-path reference, not dense attention or simulated MPU.

Prepare a public LongBench-v2 JSON snapshot using its immutable dataset revision
and publisher SHA256. Record the dataset license and local tokenizer fingerprints:

```sh
python benchmark/mpu_stages/workload.py \
  --source-json /data/longbench-v2/data.json \
  --source-sha256 DATASET_PUBLISHER_SHA256 \
  --source-repo zai-org/LongBench-v2 --source-revision DATASET_REVISION \
  --source-license apache-2.0 --tokenizer /models/qwen \
  --input-lens 8192 8193 8256 32768 --num-prompts 64 --seed 1729 \
  --output-dir /data/frozen-longbench-qwen
```

Each request uses a real document prefix, its original question/options and the
tokenizer's chat envelope. Prefix, document and suffix are encoded separately to
control exact length. Documents are neither repeated nor padded. The same
documents and row order are retained across length buckets, variants, concurrency
points and repeats. Truncation can remove evidence needed to answer a question;
these inputs measure serving/storage performance, not official LongBench accuracy.

`workload.json` contains source provenance and hashes for each ID file. The native
benchmark's `frozen-ids` dataset verifies the file, per-row and tokenizer hashes,
lengths, vocabulary and distinct source IDs, then sends input_ids directly. It
never decodes/re-tokenizes, re-samples, fills missing rows or silently drops errors.
The existing random-ids default remains available.

Use `--workload /data/frozen-longbench-qwen/workload.json` with run.py or
correctness.py. Correctness uses frozen real inputs for long rounds and retains
the existing small/lifecycle fixtures. A full target-model manifest is still
required by the acceptance gate.

The bounded single-H800 campaign connects the three requested scopes:

```sh
python benchmark/mpu_stages/real_workload_experiment.py \
  --model /models/qwen --model-manifest /results/model-manifest.json \
  --workload /data/frozen-longbench-qwen/workload.json \
  --max-total-tokens 65536 --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 --top-k 2048 --device-buffer-size 4096 \
  --seed 1729 --warmup-requests 8 --profiles \
  --output-dir /results/real-campaign-001 --execute
```

| Block | Controlled workload | Measurement scope |
| --- | --- | --- |
| Correctness | Long/short rounds, CPU ratios 1/2/4/8, HBM, eager/graph | Target-model outputs and lifecycle acceptance |
| Paired | 8K/32K, 256 output, concurrency 1/4/32, CPU ratio 2 and HBM | 32 requests per point, five repeats |
| Additional capacity | 8K/32K, 256 output, concurrency 32, CPU ratios 1/4/8 | Reuse paired ratio 2 to complete 1/2/4/8 |
| Output-length diagnostic | Same eight 32K inputs, 256/1024 output, concurrency 1, CPU ratio 2 and HBM | Three repeats; separate from the formal capacity curve |
| Nsight | 32K, one request, CPU ratios 2/8 and HBM | Separate bounded decode captures; not throughput samples |

All formal points preserve physical compute KV, top-k, hot buffer, maximum running
requests, prefill chunk, seeds, graph mode and event timing. Capacity signatures
are checked across both independently executed blocks before combining their
results. Capacity-dependent GPU metadata is recorded separately. HBM pool sizes
are retained and can differ from CPU backing capacity; high-concurrency CPU/HBM
differences must not be attributed entirely to link latency.

The campaign plans 90 formal cases / 2,880 requests and 12 output-length diagnostic
cases / 96 requests. The smaller request count is explicit; P99 and bootstrap
statistics do not imply a configured SLO or a general business workload result.
Model startup/warmup are excluded from performance, and profiling samples are not
merged into performance statistics. No MPU cost or optimal ratio is fabricated.

`profile_decode.py --profile-variant hbm` uses the accepted HBM recipe. With a
frozen workload it captures the same input IDs as CPU profiling. After inspecting
the SQLite GPU identity evidence, provide a device map and run:

```sh
python benchmark/mpu_stages/bottleneck_analyze.py \
  --sqlite /results/profile/decode.sqlite --profile /results/profile/profile.json \
  --device-map /results/profile/device-map.json \
  --output /results/profile/bottleneck.json
```

The analyzer verifies device-role/trace hashes, retains named kernel categories
and unknown kernels, and reports both summed time and interval unions. Kernel-name
classification is heuristic; summed category fractions are not wall-time fractions.
Mapped-host load/store KV traffic is not inferred from CUDA memcpy bytes. Cache-hit
rates and total KV transactions require additional instrumentation.

These commands are implementation/verification entry points. Hardware claims are
established only by the fresh campaign's result files; the earlier sealed synthetic
baseline and the six-stage build receipts remain unchanged.
