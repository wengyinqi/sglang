# MPU parallelism contract

四个 V4/GLM 阶段使用 **Attention-DP4 + EP4**。Qwen 的 1+0、1+1、1+4
均只有一张计算卡，Attention-DP/EP/TP 为 1。`+0` 是 CPU DDR 后备存储。

SGLang 的 `tp_size=4` 是这里用于划分 Attention-DP 与 EP 的四进程组宽度；
实际 `attn_tp_size=1`、`moe_tp_size=1`、dense TP=1。完整模型仅有一个副本组，
所以新接口的 `--dp-size` 是 1，`--attn-dp-size` 和 `--ep-size` 是 4。
每个请求只属于一个 Attention rank，每个 rank 对应一张 MPU（CUDA ordinal 4+i）。
EP dispatch/combine 与调度同步仍然存在，不声称四张卡独立运行或无通信。

运行器固定 DeepEP low_latency、DeepGEMM、DP LM head、dense MLP 使用 attention
group；关闭 CP、prefill graph、overlap 和 speculative decoding。decode graph
仍作为单独正确性案例。计时启动时检查 resolved 参数，并遍历已实例化的
Column/RowParallelLinear 与 VocabParallelEmbedding（包含派生类），拒绝任何
tp_size>1 或缺失的分片信息。MoE 的有效 TP 另由 resolved EP group 检查。
这项模型实例检查需要加载模型，CPU 测试不替代它。

## 模型与配对

V4 改用 `sgl-project/DeepSeek-V4-Flash-FP8`，revision
`ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17`，设置
`SGLANG_DSV4_FP4_EXPERTS=0`。原 FP4/Marlin runner 仅注册无 A2A 路径，不能直接
与 DeepEP 拼接；这里采用仓库 H200 FP8 配方使用的后端。4+0/4+4 使用相同模型、
权重校验和、EP 和稀疏选择器。旧 FP4 结果不能作为本配方的性能对照。
GLM 保持原固定 checkpoint，使用 DeepGEMM 的 DeepEP dispatch/combine 接口。

## 请求与计时

验收请求轮至少有四个请求，使用 round-robin，要求四个 Attention owner 都有
实际请求证据。每个请求的完整 prefill/decode 只出现在一个 owner 上；迁移、重复、
缺失、残留容量都会失败。没有本地请求的 rank 仍参与 EP，必须提交完整 idle
forward 和末尾空闲容量；不能用启动时的零占用快照代替。

利用已有 scheduler gather 的 CPU 元数据记录全局 forward mode/token 向量，
不新增 GPU KV 回读或集合通信。四个 rank 的连续执行序号、全局向量和本地 token
计数必须一致。每个同步 EP 步骤使用四个 rank 的 CUDA 时长最大值：

- 全部活跃 rank 都是 prefill：可从 wall time 中减去该步骤估计。
- 部分 rank prefill、部分 rank decode：整步保留，避免误减有效 decode 工作。
- decode 与 idle：整步保留。

因此 decode-only 指标是扣除 **纯 prefill 步骤** 的估算，混合步骤中的 prefill
仍在分母中。报告单列 mixed_steps/mixed_forward_seconds 和每 rank 请求数；
CUDA events 不是跨卡同步时钟，step max 不代表精确墙钟区间。原始 TTFT/TPOT
和端到端吞吐不扣减。比较不同容量时应同时检查混合步骤比例，避免把调度构成变化
误认为纯粹的 KV I/O 收益。

## 验收边界

旧 TP4 镜像/报告保留为历史构建证据，不代表 Attention-DP4+EP4 验收。
新提交必须重新执行 portable、native build、production JIT 和容器 smoke；
生产模型实例审计、DeepEP 通信、KV 正确性与性能仍需要 H200/H800。
