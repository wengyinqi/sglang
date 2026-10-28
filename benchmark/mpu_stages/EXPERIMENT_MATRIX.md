# NoPD MPU PoC 实验矩阵 v1

`experiment-matrix.json` 固定本项目的负载和选择器预算，`experiment-policy.json`
保存统计门槛、实测延迟 SLO 和完整配置的成本。入口 `experiment_matrix.py` 读取
当前分支的 `stage.json`，输出配置摘要、全部子命令、门禁和请求计划。默认仅规划。
这是待 H200 验证的工程实验设计；这些规模尚未获得 GPU 容量或性能验证。

## 六阶段和可选拓扑

| 分支后缀（均以 `mpu-exp/` 开头） | 计算策略 | 全量后备 KV | 默认 profiles |
|---|---|---|---|
| 01-qwen-quest-1gpu-0mpu | 单计算卡，TP/DP/EP=1 | CPU DDR | baseline |
| 02-qwen-quest-1gpu-1mpu | 单计算卡，TP/DP/EP=1 | 1 张 H200 模拟 MPU | baseline、capacity |
| 03-deepseek-v4-flash-4gpu-0mpu | Attention-DP4 + EP4 | CPU DDR | baseline |
| 04-deepseek-v4-flash-4gpu-4mpu | Attention-DP4 + EP4 | 4 张 H200 模拟 MPU | baseline、capacity |
| 05-glm-5.3-flash-4gpu-0mpu | Attention-DP4 + EP4 | CPU DDR | baseline |
| 06-glm-5.3-flash-4gpu-4mpu | Attention-DP4 + EP4 | 4 张 H200 模拟 MPU | baseline、capacity |

阶段 02 增加 `--topology qwen-1plus4` 即生成独立 1+4 实验，不新建第七个分支。
四计算卡阶段的有效 Attention/MoE/dense TP=1。外层 `tp_size=4` 承载四个 DP/EP
计算进程，命令使用 `dp_size=1, attn_dp_size=4, ep_size=4`。
模拟 MPU 仅存 KV，保持各模型 HiSparse CPU 布局，无 CPU KV fallback。

## 固定负载

| 项目 | Qwen + Quest | DeepSeek V4 Flash | GLM 5.3 Flash |
|---|---:|---:|---:|
| 输入长度（token） | 8192、32768 | 8192、32768 | 8192、32768 |
| 输出长度（token） | 256 | 256 | 256 |
| 全局请求并发上限 | 1、4、8、16、32 | 4、8、16、32、64 | 4、8、16、32、64 |
| 每个点每轮请求数 | 256 | 512 | 512 |
| 每轮原生 warmup 请求数 | 32 | 64 | 64 |
| 独立计时重复数 | 5 | 5 | 5 |
| `max_total_tokens` 请求值 | 65536 | 131072 | 131072 |
| prefill chunk | 8192 | 4096 | 4096 |
| selector top-k | 2048 token | 512 C4 行 | 2048 token |
| hot buffer | 4096 token | 1024 C4 行 | 4096 token |

同一并发点每轮使用 `1729 + repeat_id` 种子，CPU/MPU 配对一致。
性能负载沿用原生 `random-ids`、greedy、固定输出长度、无限到达速率、禁用 radix cache
和 overlap schedule；不等同自然语言任务质量评测。warmup 复用原生实现并在计时前 flush，
不进入测量样本。每点至少 256 请求及 8 个并发波次；这仍不是稳定 P99 的统计保证。
性能阶段要求实际 decode graph 开启；自动回退 eager 会失败并保留 server-info。

默认每个 +0 阶段 50 个性能 case，每个 MPU 阶段 350 个（baseline 100、capacity 250）。
六阶段共 1200 个，可选 Qwen 1+4 再增加 350 个。首次到机先执行 baseline，验证容量、
时长和稳定性后再运行 capacity。OOM、超时或不完整结果不能记作 0 吞吐或平台。
调整矩阵时复制文件、修改 `id`，整组 CPU/MPU 使用同一份配置重新测量，不拼接异构结果。

## 两种容量对照

- `baseline`：CPU 与 MPU 的请求比例都是 2。+0 使用 CPU DDR；MPU 分支内部也重跑 CPU
  参考，避免跨分支、跨环境的旧结果直接混比。实际分配可能有 padding，报告区分“请求
  比例相同”与“物理 KV bytes/slots 相同”。
- `capacity`：CPU 比例固定 8，MPU 扫描 1、2、4、8。V4/GLM 同时固定 host/index 地址
  预算。这个 CPU 参考是固定的大容量基线，不能称为每个 MPU 点的等容量对照。

每个输入长度内，所有点必须保持实际计算 KV slots、bytes 和模型留存 GPU 池的
形状/stride/dtype/storage 别名一致。不同上下文长度单独形成曲线，不自动合并。
容量比使用 **实际后备 KV 分配字节 / 实际计算 KV 分配字节**，逐 rank 和汇总均保存；
汇总为 `sum(backing_kv_bytes) / sum(compute_kv_bytes)`。同时保留 slot 比例、单位、
padding 和 GPU 留存池。不得使用显卡标称显存或仅用卡数代替实际 KV 容量比。
此分母是当前模型 offload pool 的计算 KV，不包含权重及其他留存池；跨模型布局不可混比。

同一物理拓扑内扫描容量；Qwen 1+1 与 1+4 分开输出、分开分析。改变卡数也可能改变
聚合链路带宽，因此二者差异不能单独归因于容量。MPU 设备序号均为进程可见的 CUDA
ordinal，设备 UUID、P2P 与地址所有权以硬件验收和 profile 证据为准。

## 执行与验收

在对应实验分支、对应已构建环境内执行。规划无需 CUDA 或权重：

```sh
python benchmark/mpu_stages/experiment_matrix.py --output-dir /results/plan-stage
```

机器到位后，准备固定 revision 的本地权重和带可信参考、完整 checksum 的模型 manifest。
输出目录必须不存在，以下 `/models/...` 和 `/results/...` 为待替换路径：

```sh
python benchmark/mpu_stages/experiment_matrix.py \
  --profiles baseline --model /models/local-snapshot \
  --model-manifest /results/model-manifest.json \
  --output-dir /results/baseline --execute

# 在 MPU 分支执行；每种 topology 使用自己的 SLO/整机成本策略。
python benchmark/mpu_stages/experiment_matrix.py \
  --profiles capacity --model /models/local-snapshot \
  --model-manifest /results/model-manifest.json \
  --policy /results/selected-topology-policy.json \
  --output-dir /results/capacity --execute

# 仅阶段 02；保持计算 ordinal=0，存储 ordinal=1,2,3,4。
python benchmark/mpu_stages/experiment_matrix.py \
  --topology qwen-1plus4 --profiles capacity \
  --model /models/local-snapshot --model-manifest /results/model-manifest.json \
  --output-dir /results/qwen-1plus4 --execute
```

每个 profile 先调用现有 `acceptance.py --through correctness`，包括 CPU、compile、
JIT、imports、device、kernel、模型身份、正确性与生命周期门禁，再运行性能。
正确性覆盖每个比例、eager/graph、同 selector HBM 参考、三轮长短请求和 slot 复用；
greedy token ID 完全相同，logprob 绝对误差不超过 0.02。四卡阶段逐 Attention owner
覆盖 abort-prefill、abort-decode、retract-decode 与恢复。任何必需层失败/skip/缺失，
均停止后续 profile。它不能证明模型任务质量，也不能代替 profile 验证存储卡无计算。

`matrix.json` 保存精确参数和阶段/矩阵/策略哈希，`stage.json`、`policy.json` 是执行
快照。每个 profile 保留原生结果、trace、server-info、门禁和分析 JSON。完整测量记为
`measured`，不自动标记“性能收益通过”。运行中不要切分支或改生产源码。

## 收益、平台与成本判据

1. TTFT、TPOT、端到端吞吐保存实测值。另算 `(输出 token - 请求数)/(wall time - P)`；
   `P` 只累加全局纯 prefill 步骤的逐步 rank 最大 CUDA 时长，mixed 步骤不扣。
   这是 NoPD 的 decode 估算；不能当作实测 PD decode-only 吞吐。
2. 复用 2000 次配对 bootstrap 的 95% 区间，同时报告 CPU→MPU 的端到端与 decode 估算
   吞吐增幅。区间含 0 时不声称明确正收益。固定 CPU-first 顺序的热/频率漂移不在区间内，
   首轮结果需要结合独立重跑与硬件状态解释，不能宣称已排除系统偏差。
3. 每个 rank 的实际 slot 比例和 KV 字节比例都必须增长；至少 5 次完整重复。
   在扫描尾部至少连续两次容量增加的 decode 增益区间上界低于 5%，且不判为回退，才
   生成平台候选。后续显著收益、回退、缺样本或不确定结果打断候选链；不同拓扑不连接。
4. 项目尚未确定绝对 SLO。策略可保持空；填写时必须同时提供 `p99_ttft_ms` 和
   `p99_tpot_ms`（毫秒），每个重复都必须满足。未知阈值不会被虚构或以平均延迟替代。
5. 成本目前为空。填写整套配置的 `hourly_cost`、`currency`、来源和 `cost_basis`，
   区分 H200 模拟硬件与假设 MPU。固定 H200 拓扑内各分配比例的整机成本必须相同。
   每种拓扑使用独立策略；不能让 1+4 沿用 1+1 的价格。
6. 满足重复数和 SLO、有成本的点才参加“已测点中 tokens/货币最高”比较。端到端与
   decode 估算分列；不声称统计上胜者已分离、全局经济最优或真实 MPU 已获收益。

本轮只改实验工具、配置和测试，未改生产 CUDA/模型实现。旧离线包固定的源码提交和
bundle 哈希保持原样；使用新入口需获取本轮实验分支代码，不能将旧包声称为已包含新矩阵。
