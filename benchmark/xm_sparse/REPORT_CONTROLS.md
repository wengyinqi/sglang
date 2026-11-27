# Source017 report controls and CPU replay

These report tools are separate from the model source. They never launch a
model and never alter a source017 receipt, native manifest, raw response or
seal. A passing CPU replay confirms the binding of evidence, not a new GPU
execution. Run the frozen delivery first on Linux with the original canonical
mounts. The external controller still requires its original source, model
configuration and workload paths; copying a JSON file to Windows does not
remap those paths or establish an independently restored model control.

## Explicit full32K/B4 precursor

`report_control_binding.py` replays the independently reviewed original
`large_cpu_baseline_control017.py` precursor gate. It requires a successful
true `[32768, 32768, 32768, 32768]` proof and its original acceptance and seal.
Choose the actual precursor ratio, either 4 or 8. Ratio 4 is preserved as
ratio 4; the later large CPU control uses ratio 8 separately.

The following CPU command is run only after the stated proof really passes.
Its paths must name the existing original files in the reviewed mount layout:

```bash
python /opt/report-tools/benchmark/xm_sparse/report_control_binding.py \
  --source-root /opt/mpu/ci-stage01/source \
  --source-archive /results/source-xm-017.tar.gz \
  --evidence-root /results \
  --correctness-campaign /results/results/correctness-mma-017-b4-full32k-ratio4/campaign.json \
  --correctness-seal /results/correctness-mma-017-b4-full32k-ratio4-seal.json \
  --full32k-acceptance /results/source017-full32k-b4-ratio4-cpu-acceptance.json \
  --model /models/qwen \
  --workload /data/frozen-xm-qwen-20261127/workload.json \
  --precursor-capacity-ratio 4 \
  --controller /results/tools/large_cpu_baseline_control017.py \
  --output /results/source017-full32k-b4-ratio4-report-binding.json
```

These names illustrate the explicit ratio4 phase contract, not completed
results. Never create a substitute acceptance to make the command pass.
Use the original ratio8 phase paths with `--precursor-capacity-ratio 8` when
that is the actual independently passed proof.

## Large CPU baseline and capacity display

The external actual-execution runner is `run_external_baseline017.py`.
The controller is `large_cpu_baseline_control017.py`; the dedicated byte
custody producer is `seal_external_baseline017.py`. Those delivered helpers
live under the evidence root and are not substituted by the campaign runner.
The current contract template has `gpu_execution_allowed=false` until a
reviewed ready plan and the required precursor/large-summary raw gates exist.

Supply the original baseline final receipt, ready plan, runner contract, job
file, original prerequisite seals and dedicated actual-execution custody to
`report.py --evidence`. The report rereads hashes and replays raw full inputs,
greedy IDs, finite logprobs, lifecycle and both same-B optimized comparisons.
A campaign seal alone cannot stand in for the external runner's custody.

Each capacity point retains client concurrency C, requested and effective
admission limits, actual Quest request-map rows R and context pages P, and
observed stable batch B. The admission limit is not the padded Quest tensor's
row count. P comes from the observed context-column shape and page size,
not CPU backing ratio. A point outside the independent small CPU/HBM proof
requires an exact B, admission limit and Quest-row match from the original
CPU baseline control. Only baseline, fused and combined-protected have the
large-control comparison contract.

Uncovered points remain in the original analysis and the report's diagnostic
table. A C12/admission12/B11 point cannot be covered by a B11/admission11
control, cannot become a claimed maxB8, and does not establish a global
maximum batch. Correctness output128 and timing output512 can have different
context/P allocations; their raw shapes stay separate.

## Complete HTTP paired effects

`client_paired_effects.py` independently recomputes complete HTTP singleton
token-frame arrival gaps, request TPOT, TTFT and complete output throughput.
It checks all six arms and three repeats against the source, workload,
commands, saved raw replay, texts and every HTTP frame. The timing requests
still require a separate same-source model proof.

```bash
python /opt/report-tools/benchmark/xm_sparse/client_paired_effects.py \
  --tool-root /opt/mpu/ci-stage01/source \
  --campaign /results/results/performance-mma-017-b1-16k32k/campaign.json \
  --saved-analysis /results/source017-performance-b1-analysis.json \
  --evidence-root /results \
  --expected-native-source 3ca21f0089b30e13b254491f2b086f073ad11f10 \
  --expected-core-sha256 d763476a8bc735c8197a9c0208a49493e0c4a66eb9c34974b46ccd5615fcade0 \
  --input-lens 16384 32768 --output-len 512 --client-concurrency 1 \
  --output /results/source017-http-paired-effects.json
```

The performance paths are pending until the model gates pass and that
campaign actually completes. Supply the derived effects JSON as another
`report.py --evidence` file. Table1A and its plots use complete HTTP mean TBT
and TPOT as the Table3-style main metrics. Table1B retains stable scheduler
intervals separately. Independent Nsight plots and layer spans remain
diagnostic measurements; none replaces the complete HTTP metric. Three
paired repeat percentages give a Student-t interval with df2; individual
token gaps are not independent repeats.

## Portable CPU checks

From the report-tool checkout, run:

```bash
python -m pytest -q \
  test/manual/xm_sparse/test_report_evidence.py \
  test/manual/xm_sparse/test_report_control_binding.py \
  test/manual/xm_sparse/test_client_paired_effects.py \
  test/manual/xm_sparse/test_analysis.py
```

These fixture guards exercise the evidence contract without model weights,
GPU, Nsight or server traffic. They do not grant any pending numerical,
capacity, P2P or performance result.
