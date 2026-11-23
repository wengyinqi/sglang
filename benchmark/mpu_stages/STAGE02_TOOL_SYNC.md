# Qwen stage 02: tools shared with the audited stage 01 baseline

Stage 02 remains `02-qwen-quest-1gpu-1mpu`, with one compute GPU and MPU
backing storage. Default CUDA-visible ordinals are compute 0 and storage 1;
`--mpu-device-ids 1 2 3 4` selects four storage shards. The MPU stores the full
KV payload without CPU KV backing or computation on the storage devices.
TP, attention DP and EP remain 1 for this Qwen stage.

## Integrated experiment tools

- `run.py` saves a separate steady-decode analysis alongside end-to-end and
  prefill-subtracted NoPD estimates. Client concurrency is not decode batch.
- `profile_decode.py` uses observed consecutive decode forwards and request IDs
  to trigger a bounded capture; CPU reference capacity remains independent of
  the MPU capacity sweep. See [PROFILE_STABLE_TRIGGER.md](PROFILE_STABLE_TRIGGER.md).
- `task_quality.py` evaluates complete frozen inputs with normal stopping and
  compares Dense/HBM/CPU, optionally including MPU. See [TASK_QUALITY.md](TASK_QUALITY.md).
- `mixed_stability.py` tracks normal completion, cancellation, recovery and
  resource usage for one compute GPU plus configured storage GPUs. See
  [MIXED_STABILITY.md](MIXED_STABILITY.md).
- HTTP frame receipts, request latency, forward progress and cache diagnostic
  fields retain their explicit timing and payload scopes.

Existing MPU campaign controls, fixed CPU references, dynamic planned case
counts, strict complete-result checks, peer kernel classification and strict CLI
parsing are retained. `stage.json` is not replaced by the stage 01 CPU default.

`SGLANG_HISPARSE_CACHE_METRICS=1` currently instruments CPU-backed Qwen resolve
only. It is a separate diagnostic run, not a formal throughput configuration or
an MPU traffic measurement. The default MPU path leaves these counters disabled.

## Acceptance on the current single H800

Run portable CPU acceptance on the development host:

```sh
python benchmark/mpu_stages/acceptance.py --through portable \
  --output-dir /results/stage02-portable --execute
```

In the pinned Linux CUDA runtime, build and load this committed source:

```sh
python benchmark/mpu_stages/acceptance.py --through imports \
  --jit-matrix boundaries --output-dir /results/stage02-imports --execute
```

This executes Linux CPU contracts, production MPU CUDA object compilation,
CPU/MPU JIT factories and production imports. Preserve the actual `.so` files
with their reports. Native dependencies and the service image may be reused
when unchanged; distinguish that from rebuilding a complete image or wheelset.

Independent single-GPU regression targets are:

```sh
python -m pytest -q \
  python/sglang/jit_kernel/tests/test_hisparse_cache_metrics.py \
  test/manual/hisparse/test_quest_selector.py::TestQuestSelectorCUDA \
  test/manual/mpu_stages/test_mixed_stability_cuda.py \
  test/manual/hisparse/test_mpu_timing.py::TestForwardTrace::test_real_cuda_stream_eager_and_graph_replay \
  --junitxml=/results/stage02-single-gpu.xml
```

These run CPU-pinned resolve/counters, Quest eager/Graph and real-stream timing.
They do not execute stage 02's device/P2P, storage-GPU kernel, target-model or
performance gates. Those gates require a second peer-accessible GPU for 1+1,
or five total GPUs for 1+4. A requested subset passing does not pass later gates.

## Frozen reference and future paired runs

The audited stage 01 reference is commit
`b61d8c7f1529ff1971026ae68dcc42af80e7a592`, run
`2026-11-20-priority-002`. Its immutable source, model, workload, protocol,
raw evidence, reports and CSV hashes are bound by the separate D-drive
`2026-11-23-stage02-tool-sync-001/baseline01-freeze/baseline01-reference.json`.
The old CPU DDR result is not relabelled as a new MPU result.

Reuse its frozen inputs and model revision for new paired runs. All backends
in a new task-quality comparison must share the new source identity; do not
combine old b61 outputs and new stage 02 outputs under one source hash.
Match top-k, hot buffer, compute KV slots and padding, prefill partitioning,
Graph policy and sampling controls. Record actual TTFT/request-mean TPOT,
E2E throughput, NoPD estimate and steady windows separately. SLOs and complete
hardware hourly costs remain unspecified; no economic optimum is inferred.
