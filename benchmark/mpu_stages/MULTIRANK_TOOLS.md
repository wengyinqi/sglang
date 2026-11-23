# Shared tools for stages 03–06

These branches use Attention-DP4 + EP4. The outer compute group has width four;
the instantiated attention/linear/vocabulary/MoE tensor partitions remain TP=1.
The single H800 cannot validate these four-rank model or peer-storage paths.
CPU fixtures validate accounting and failure rejection only.

## Frozen input and native HTTP frames

`workload.py` freezes real-document input IDs with source, tokenizer and row
hashes. `run.py --workload ... --workload-sha256 ...` passes the same ordered
IDs to every storage variant and repeat. Native `frozen-ids` lives in the new
`sglang.benchmark.datasets` package; `sglang.bench_serving` stays the upstream
compatibility wrapper. Cropped throughput inputs are explicitly separate from
the complete-input task-quality subset.

With `--output-details`, native `/generate` records monotonic request bounds,
actual cumulative completion counts and every HTTP frame. A multi-token frame
has no measured internal token timestamps; its amortized gap is not P99
single-token arrival latency.

## Steady decode and profile trigger

`steady_decode.py` first runs the existing complete native four-rank analyzer.
It then aligns global EP steps and requires unchanged request owner assignments,
one stream per rank, pure decode with optional idle EP participants, and
ordered launch timestamps. Reports retain actual global/local batch sizes,
rank launch skew and per-rank timelines. Client concurrency is still a separate
input control. No four-rank CUDA duration sum is called distributed wall time.

`profile_decode.py --profile-concurrency 4 --profile-stable-steps 16
--profile-batch-size 4` requires four actual owners and fresh aligned live
progress. Missing, duplicated, stale or misaligned rank evidence cannot arm the
stable trigger. Sessions expose actual model tensor partition audits and model
indexer configuration; forward rows expose CPU `seq_lens` and the model worker's
returned `actual_cuda_graph`, independent of configured Graph policy.

`decode_scope.py` retains the strict Qwen kernel verifier for a single-rank
capture. Four-rank DeepSeek/GLM captures can verify aligned trigger provenance;
their result remains `trigger_verified_pending_model_kernel_semantics` until
deployed model-source kernel semantics, all rank/device correlations and
complete captured calls have been checked. This status is not a verified
decode-kernel timing breakdown.

## Stability and task quality

`mixed_stability.py` freezes all four scheduler PIDs and lifetimes, validates
all four final idle flushes and zero used/reserved slots, merges request facts
by owner, and checks global DP mode/token vectors. A queued cancellation with
no forward remains ownerless. Compute UUIDs are separate from storage UUIDs:
four compute plus zero storage for CPU/HBM, or four compute plus four storage
for MPU. Resource trajectories stay per device; asynchronous peaks are not
summed into a simultaneous peak.

`task_quality.py` selects its model ID/revision from this branch's `stage.json`
(or an explicit `--stage`). DeepSeek/GLM comparisons require same-model native
sparse HBM and CPU results, plus MPU under `--require-mpu`. They do not add a
fabricated Dense attention variant. Effective trained indexer top-k and KPool
controls come from all instantiated model configurations; requested transfer
settings are separate. Inputs are complete, EOS is enabled, and every selected
sample remains in the denominator, including unscorable outputs and failures.

`real_workload_experiment.py --model-manifest ... --workload ... --profiles`
plans model-specific correctness boundaries, matching workload runs and all
four-owner diagnostics. Model startup, GPU capacity fit, DeepEP, actual model
correctness, device roles, peer load/store and performance remain hardware gates.
The existing frozen host/index budget guard still limits MPU capacity ratios.

## Acceptance

`acceptance.py --through portable --execute ...` includes shared workload,
HTTP-frame, steady, task-quality, mixed-load and four-rank contract fixtures.
Linux/Gloo, CUDA compilation/JIT, model import, device/P2P, correctness and
performance are separate ordered gates. A portable pass never unlocks a GPU
or model correctness claim.
