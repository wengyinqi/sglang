# MPU 实验计时口径

本目录的 `analyze.py` 配合 `SGLANG_HISPARSE_TIMING_DIR`，为单张计算 GPU 的
NoPD 实验提供实际 CUDA forward 时间与 decode-only 估算。当前只有 CPU
协议/计数回归结果；H200 上的事件计时、模型输出和性能验证仍待执行。

## 采集与关联

给每个服务进程的实验设置独立目录：

```bash
export SGLANG_HISPARSE_TIMING_DIR=/path/to/run/trace
# 在该环境中执行对应的 sglang.launch_server 命令。
```

用原生 `sglang.bench_serving --backend sglang --output-details` 保存每个请求的
真实 `meta_info.id`。计时记录也保存 scheduler 的 request ID，因此可以剔除
warmup、服务启动和 health-check 请求，不依靠不准确的 CPU 提交时间划分实验。
一批 prefill 无论含几个请求，都只计一次 CUDA 时间；chunked prefill 的所有分块
分别计时并检查 token 总数。

CUDA events 位于执行模型的 stream 上，覆盖 eager forward 或 CUDA Graph replay。
热路径不执行 device synchronize 或写文件；scheduler 完全空闲时排空尾部 events、
写出 JSONL 与 `idle_flush` 标记。事件记录和空闲时写盘本身仍有开销，性能比较双方
必须采用相同采集方式。分析必须在关闭服务之前完成，以免强制终止丢失尾部记录。

```bash
python benchmark/hisparse_mpu/analyze.py \
  --result /path/to/run/case.jsonl \
  --trace-dir /path/to/run/trace \
  --output /path/to/run/timing.json
```

脚本默认最多等待 10 秒让尾部落盘。结果文件必须只含一个 benchmark case。
供 runner 调用的函数为 `analyze_when_ready(result_dict, trace_dir, timeout=10.0)`。

## 指标

原始 `duration`、TTFT、TPOT、output throughput 保留为端到端测量值。
设 `T` 为该 case 的 benchmark wall time，`P` 为该 case 全部 prefill 批次的
CUDA event 时长之和，`N` 为总输出 token 数，`R` 为已完成请求数：

- `residual_seconds = T - P`。
- `decode_throughput_tokens_per_second = (N - R) / (T - P)`，排除 prefill 产生的首 token。
- `all_output_throughput_tokens_per_second = N / (T - P)`，另列全输出计数口径，便于复核。

这两种值都是去除 prefill forward 时间后的 **NoPD 反事实估算**。它们不代表实际
PD decode worker 的吞吐量，也不重新定义 TTFT 或 TPOT。残余时间仍包含请求排队、
CPU/网络间隔，以及定时 forward 以外的 KV staging 和其他工作；没有把这些时间
擅自归零。绝不能减去各请求 TTFT 之和：共享 prefill 批次会被重复扣除，而且 TTFT
本身还包含排队和传输。

## 校验范围

当前分析要求 TP=PP=DP=1、NoPD、关闭 radix cache、mixed chunk、two-batch
overlap 和 speculative decoding。可以使用普通 CPU/GPU overlap scheduler；其
每请求可能多执行一次 decode forward，结果中会显式记录额外数量。该功能与
two-batch overlap 不同。

分析器拒绝缺失或重复 CUDA events、未完成的 trace、多个计算 scheduler/stream、
混入其他请求的批次、失败 forward、丢失分块、prefix-cache 命中、retraction 导致
的重复 prefill，以及不完整/不一致的输出 token 计数。违反这些条件时保留原始证据，
修复实验配置或实现后重跑，不能把失败 case 当作容量或吞吐量收益。
