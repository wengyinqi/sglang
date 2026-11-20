# 单卡 Qwen 稳态 decode 与流式尾延迟

`steady_decode.py` 对单计算 scheduler、非 overlap、NoPD 的完整 forward trace
做后处理。它先执行原有请求、prefill chunk、输出 token、idle flush 及单流校验，
再按**实际 request-ID 集合**切分连续 decode 区间。客户端并发相同不代表实际 batch 相同。

默认每段两侧各去掉 4 步，剩余至少 16 条连续 decode 记录才生成稳定窗口。
16 条记录形成 15 个 launch 间隔；最后一次 launch 只作右边界。窗口吞吐使用
实际 batch × launch 间隔数 / 首尾 scheduler launch 时间差。
prefill 插入和 request-ID 集合变化都会切分窗口。没有窗口时保存
`stable_window_observed=false`，不推断稳态收益。

```sh
python benchmark/mpu_stages/steady_decode.py \
  --result /results/case.jsonl --trace-dir /results/forward-trace \
  --output /results/case-steady.json
```

新的 `run.py` 单卡执行会自动保存 `<case-id>-steady.json` 和 manifest 中的
`steady_decode`。它要求本轮 `bench_serving.py --output-details` 生成的完整流式帧
证据；旧结果仍可用原 `analyze.py` 分析，缺失的帧证据不能补造。

计时字段分三种范围：

- `scheduler_launch_span_seconds`：服务端 host monotonic clock 的首尾 launch 间隔。
- `cuda_forward_seconds_over_same_intervals`：对应 forward 的 CUDA event 时长之和。
  与 host span 分别列出，不能相减当作 CPU 时间。
- `client`：客户端 E2E、TTFT、请求平均 TPOT 和原始 HTTP 帧到达间隔。
  帧可能同时包含多个 token，`amortized_chunk_gap_per_token_ms` 只是均摊值。
  只有所有正增量帧都各含一个 token 时，才提供 `single_token_arrival_gap_ms`。

请求平均 TPOT 沿用 `(request_latency - TTFT) / (output_tokens - 1)`；它不等于
逐 token ITL 的 P99。稳定窗口吞吐、整个 benchmark 的 E2E 吞吐及扣除 prefill 的
decode 估算也应分别列出。这里没有单独测量 queue/admission 时间或 KV hit/miss。

## 从实际稳定窗口触发独立 Nsight

在同一 clean source、模型快照和配置的 correctness 验收通过后，
`profile_decode.py` 支持两项新参数：

- `--profile-stable-steps 16`：已观察到至少 16 个同 request-ID 集合的连续 decode
  forward 才调用 `/start_profile`；默认 0 保留原触发方式。
- `--profile-batch-size N`：进一步要求实际 batch 正好为 N。此值应由先前容量实验
  的实际 batch 确定，不能直接使用客户端 concurrency。

例如，已经确定 CPU R2、C4 的实际稳定 batch 为 3 时，可在原有匹配 correctness 的
命令中增加 `--profile-concurrency 4 --profile-stable-steps 16 --profile-batch-size 3`。
本轮长输出建议 `--output-len 1024 --profile-steps 64`；短输出可能在触发或保守
收尾区间完成，工具会拒绝。触发前后快照分别保存在
`trigger_progress_before` / `trigger_progress_after_ack`。
快照仅读取 CPU metadata，不等待 CUDA event，不访问 KV payload。

## 验证多请求 decode 采集范围

`decode_scope.py` 当前专用于固定的 48 层、BF16 Qwen 单计算卡。它绑定 SQLite hash、
完整请求 token 覆盖、source commit 及已经从**实际部署源码**验证的 attention kernel
语义。`--semantics` 必须是 source verifier 的 `verified_from_deployed_sources` 输出，
不能直接使用仅表示已抓取源码的 `captured_from_deployed_runtime` receipt。

```sh
python benchmark/mpu_stages/decode_scope.py \
  --sqlite /results/profile/decode.sqlite --profile /results/profile/profile.json \
  --trace-dir /results/profile/forward-trace --semantics /results/decode-semantics.json \
  --output /results/profile/stable-decode-scope.json
```

验收对触发前 sequence 至触发 ack 后 sequence + 配置步数 + 2 的保守区间要求
同一实际 request-ID 集合、同一 stream、全部为 decode；实际 CUDA graph launch 数
只允许配置步数或多一个边界 step。每个 graph 必须完整含 48 个 Quest score 和
48 个已验证 specialization 的 attention 调用。未知或真正的 prefill kernel 会失败。

此验收没有记录精确 host↔Nsight 时间戳 join，因此明确使用保守边界；它也不替代
`profile_analyze.py` 的设备 UUID/角色验收。独立 Nsight 采集仅用于诊断，不能混入
未开 profiler 的性能数据。
