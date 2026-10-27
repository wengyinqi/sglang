# Decode diagnostics and capacity analysis

本轮提供代码和 CPU 验证；尚未在 GPU 上执行 Nsight 采集、模型推理或收益测量。
这些工具供六个阶段共用，`+0` 保持论文 HiSparse 的 CPU DDR 后备存储含义。

## 独立采集 decode trace

先完成同一源码、模型和存储配置的 `acceptance.py --through correctness --execute`。
再运行独立诊断，示例中所有参数应与已通过的 correctness 配置一致：

```sh
python benchmark/mpu_stages/profile_decode.py \
  --model /models/local-snapshot \
  --acceptance /results/correctness-acceptance/acceptance.json \
  --input-lens 32768 --capacity-ratios 2 --profile-steps 32 \
  --output-dir /results/profile-ratio2 --execute
```

不带 `--execute` 只生成计划。执行要求 Linux、GPU、Nsight Systems，以及匹配的
correctness 原始产物；CPU 测试通过不能解锁采集。归档源码保留 `.mpu-source.json`。

脚本复用 SGLang 两代代码均有的 `SGLANG_PROFILE_V2=1` 与 `/start_profile` 接口，
只启用 `CUDA_PROFILER` 的 `decode` 阶段。它先启动模型、warmup、flush cache，再启用
阶段触发器并发送诊断请求。采样使用重复 token 42；Qwen 为单请求，多卡阶段为
四个请求，覆盖四个 Attention owner。这些都是非性能样本。
生成长度至少比请求的采样步数多 4，给上游触发器留下结束边界；实际可能多包含一个
边界 step，必须检查原始时间线。NoPD 服务仍执行 prefill，但 trace 由 decode 触发。

Nsight 启动参数包括 CUDA Graph node trace，不要求 CPU perf 采样权限。保存 `.nsys-rep`、
SQLite、原始日志、forward trace、设备 UUID、工具版本、源码与 correctness 产物哈希。
结束时先停止自建 server，留给 Nsight 完成导出。产物状态为
`captured_pending_role_analysis`；文件存在本身不代表通路验收完成。

在 Nsight 中核对设备 UUID/PCI 信息后，提供显式映射。以下是**格式示例**，不是本机设备：

```json
{
  "schema_version": 1,
  "mapping_source": "Describe the Nsight device UUID/PCI evidence used",
  "devices": [
    {"trace_device_id": 0, "uuid": "GPU-compute-from-profile-json", "role": "compute"},
    {"trace_device_id": 1, "uuid": "GPU-storage-from-profile-json", "role": "storage"}
  ]
}
```

1+4 / 4+4 须列出全部设备；`trace_device_id` 不是 `CUDA_VISIBLE_DEVICES` 序号。
映射必须与 `profile.json` 中采集到的 UUID 和角色一一对应。

```sh
python benchmark/mpu_stages/profile_analyze.py \
  --sqlite /results/profile-ratio2/decode.sqlite \
  --profile /results/profile-ratio2/profile.json \
  --device-map /results/profile-ratio2/device-map.json \
  --output /results/profile-ratio2/role-analysis.json
```

分析器验证 SQLite 哈希，检查所有计算卡有 kernel 活动，报告存储卡上的 kernel、
主要 kernel 名称与 CUDA memcpy 字节。无 CUDA 活动、缺少计算卡或未知设备均阻塞；
存储卡出现 kernel 记为失败。设备映射由操作者提供，角色结论以此映射正确为条件。

**这不能证明 MPU 完全没有 CPU KV 数据访问。** GPU 的直接 P2P load/store 以及
mapped-host load/store 不一定出现在 memcpy 表中；CPU metadata 拷贝也不能当作 KV
payload。总 KV 字节及地址归属还需要硬件 transaction counter 或地址可追踪的证据。
kernel/copy 时长可能重叠，求和不等于墙钟时间。真实多 rank 覆盖与采样边界须看原始
trace；有界诊断中不会插入用于正确性对比的 `.cpu()` 操作。

## 容量、SLO 和成本

`run.py` 的原始 TTFT、TPOT、端到端吞吐和扣除 prefill 的 decode 估算继续分别保存。
DP4 只扣除全局纯 prefill 的 EP 步骤，混合 prefill/decode 步骤完整保留；每次重复
记录混合步骤数量，见 [ATTENTION_DP_EP.md](ATTENTION_DP_EP.md)。不能混合旧 TP
和新 DP 扣减方法计算重复区间。
新分析保留 P99 TTFT/TPOT，给出重复实验中位数的 95% bootstrap 区间和相邻比例增益
区间。默认至少 5 次重复才产生平台候选结论；原有默认 3 次仍可运行，但标为证据不足。
实际分配容量未增加、样本缺失、结果波动跨越阈值，都不会给出平台结论；明确下降记为
regression。区间不覆盖运行顺序、温度、频率、工作负载变化或扣除 prefill 的系统偏差。

先复制 `analysis-policy.example.json`，填入团队选定的 SLO 和每个**完整配置**的小时成本。
成本条目格式为 `{"variant":"mpu","ratio":2,"hourly_cost":YOUR_NUMBER}`。
`currency`、`source` 和 `cost_basis` 必填后才能使用成本；`cost_basis` 分别用
`simulator_hardware` 或 `hypothetical_mpu` 区分 H200 模拟器与假设 MPU 的经济模型。
同样 5 张 H200 上只改变 KV 分配比例，不代表购置成本随比例增加。

```sh
python benchmark/mpu_stages/run.py --model /models/local-snapshot \
  --capacity-ratios 1 2 4 8 --cpu-capacity-ratio 8 --repeats 5 \
  --analysis-policy /results/analysis-policy.json \
  --output-dir /results/performance --execute
```

只修改成本/SLO 时，可对已保存的 benchmark manifest 重新分析，无需重跑 GPU：

```sh
python benchmark/mpu_stages/capacity_analysis.py \
  --manifest /results/performance/manifest.json \
  --policy /results/analysis-policy.json --output /results/cost-review.json
```

每个重复都满足指定 SLO、重复数足够且已提供成本的配置，才参与“已测点中性价比最高”
比较。缺 P99 不会自动用 mean 代替。端到端 tokens/货币与估算 decode tokens/货币分列；
不声称全局经济最优或统计上已区分胜者。示例策略没有默认价格和 SLO，因此不会产生
经济最优结论。CPU 测试中的数值与 SQLite 均是明确标注的合成样本。
