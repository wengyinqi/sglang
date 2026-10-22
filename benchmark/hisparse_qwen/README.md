# Qwen + Quest / HiSparse 实验支持与验收

本目录对应 `hisparse_quest` 分支的 Qwen 实验修复。审查起点为
`e568f8a362b4ae3a597b28b75716309814aa33fb`。**原分支不能直接视为论文复现就绪：
已有主要实现，但存在可复现的正确性问题，以及真实 Qwen 路径的集成缺口。**
本次补齐这些缺口并提供回归测试、GPU 验收入口和实验脚本。

截至 2026-10-22，只有 Windows CPU 验证；没有运行 Qwen 模型、CUDA/Triton/
FlashInfer 内核、H800 或 GH200 吞吐实验。后续 H800 到位后，仍须完成下面的 GPU
验收才能确认实际可跑。代码已具备待验收的实现，不等于论文数值已经复现。

## 1. 论文到底要求什么

来源：[HiSparse, arXiv:2608.07009v1](https://arxiv.org/abs/2608.07009)，
第 9 页 §4.1、第 10 页 workloads/metrics、第 11 页 Figure 5。
本次读取的 PDF SHA256：
`7f6c545e6e54de09d6190574e643255c98d68870ed1261babbb09b0c7d3e7b91`。

| 项目 | 论文明确内容 | 本目录处理 |
|---|---|---|
| 模型 | `Qwen3-30B-A3B-Thinking-2507` | 默认使用该模型 |
| 选择器 | 所有层使用 Quest，`k=2048` | 两条路径共用同一 Quest 实现 |
| KV 精度 | BF16 | 两侧强制 BF16 KV；示例权重也用 BF16 |
| 硬件 | GH200 节点 | 脚本记录实际硬件；H800 属于独立平台验证 |
| 对照 | Quest 全 KV 驻留 HBM vs Quest + HiSparse | 不把 dense Qwen 当作 Figure 5 的性能基线 |
| 输入长度 | 图中标为 4K / 32K / 200K | 默认明确取 4096 / 32768 / 200000，可覆盖 |
| 工作负载 | 固定输出长度、闭环并发扫描，`bench_serving` | 逐个并发、重复运行，记录完整参数和失败 |
| 指标 | generation throughput、TTFT、TPOT | 保留原始 JSONL 与日志 |

论文 Figure 5 的 Qwen 结果为作者报告：

| 输入长度 | Quest | Quest + HiSparse | 单位 |
|---|---:|---:|---|
| 4K | 2430 | 2668 | generated tokens/s |
| 32K | 511 | 1824 | generated tokens/s |
| 200K | 111 | 520 | generated tokens/s |

这些不是本次测量值，也不是 H800 的性能承诺。论文没有完整披露 Figure 5 的逐点
并发、输出长度、Quest page size、GQA 多头聚合、具体启动命令及所有调优值。
§4.1 声称基线为 SGLang v0.5.11；本目录比较本分支中的两种 KV 驻留路径，
不能称为未经改动的官方 v0.5.11 基线。图 1/4/8 的 8K 输出不能直接代替图 5
未披露的输出长度。因此脚本要求显式指定 `--output-len`，不伪造论文参数。

官方 [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/blob/144afc2f379b542fdd4e85a1fcd5e1f79112d95d/config.json)
已按 revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` 核对：
48 层、32 query heads、4 KV heads、head dimension 128、上下文上限 262144。

## 2. 原来支持到哪一步，本次改了什么

| 环节 | 原分支状态 | 本次处理与证据边界 |
|---|---|---|
| Qwen MHA/GQA 模型调用链 | 已有 Quest-only 和 Quest+HiSparse backends | 保留该架构，补真实 Qwen 路径的 bounds hook |
| 全量 host KV + 有界 GPU hot cache | 已有 MHA host/device pools、LRU resolve、写回路径 | 核实物理 KV 池原本按 `size` 分配，纠正误称按 host ratio 膨胀的旧注释 |
| 选位集合 | recent window 与整页选择可能重复；4097 长度是反例 | 历史选页严格位于 recent window 之前，CPU 覆盖 2048-token 预算和跨页边界 |
| GQA 页评分 | 先平均 query heads，正负 query 可能抵消 | 先保留每个 query 的符号贡献，再聚合页上界；CPU 对照独立逐头参考计算 |
| 无效页与 padding | 无效 bounds 依赖 sentinel 算术；padding 存在无用/冲突写 | 显式屏蔽无效页；scatter 仅写有效项；GPU kernel parity 待执行 |
| Qwen fused RoPE/KV-write | `save_kv_cache=False` 时可能跳过 Quest prefill bounds | KV 已由融合路径写入时也构建 bounds，CPU 回归覆盖该分支 |
| Prefill backend 配置 | 允许使用不会构建 bounds 的独立 dense prefill backend | Quest-only 要求 prefill/decode 均走 `flashinfer_quest` |
| Chunked prefill | Quest-only 强行关闭已有逐 chunk bounds 重建路径 | 允许 chunked prefill；验证 prefix bounds 与完整 prefill 一致 |
| 异步 prefill bounds | D2H 完成可能早于 bounds reader，物理槽可提前复用 | staging 完成事件 join 两个 reader，再允许释放；abort 同步并清理 bounds |
| 长上下文辅助显存 | request 数量 × context 的 bounds 不计入 KV 预算 | 要求显式 `max_running_requests`，在 KV 分配前预留持久 metadata |
| CUDA graph | 有接口与 capture 路径，但页完成使用动态 shape 操作 | finalize 改固定 shape 更新，隔离 dummy request；真实 graph replay 待 GPU 验收 |
| 端到端测试 | 原 smoke 使用小模型和短 prompt，不能证明 sparse eviction 正确 | 新增超过 top-k/hot cache、跨页、混合长度、槽复用的 tokens/logprobs 对照 |
| 吞吐统计 | native benchmark 在空文本/缺失最终统计时可能保留请求的目标 token 数 | 改用服务端实际统计，截断流、abort 和缺失完成标志计为失败 |
| 论文级性能 | 没有本次独立结果 | 提供参数固定、记录硬件与失败的实验脚本，保留未验证状态 |

新的 GQA 规则仍是**每层一个共享 token 集**：每个 KV group 内平均各 query head
的 bounding-box upper bound，再跨 KV group 求和；不是给每个 query head 单独
维护不同的 KV 集。最近 64 个 token 始终保留，另外选择不与它重叠的历史整页。
论文没有披露这些细节，故这里明确称为本分支的共享 Quest 规则。
HiSparse 的正确性目标是保持这个规则下的选位和 attention 结果；
它不自动证明 Quest 相对 dense Qwen 的质量无损。

## 3. 支持范围与内存边界

当前实验路径为 Linux + NVIDIA CUDA、colocated prefill/decode、BF16 KV、关闭 radix
cache。Quest-only 使用 `--attention-backend flashinfer_quest`；HiSparse 使用
`--prefill-attention-backend flashinfer --decode-attention-backend flashinfer_hisparse`。
本轮不支持 speculative decoding、mixed chunk 或 Quest 的 PD disaggregation，
配置校验会拒绝这些组合。Prefill bounds bookkeeping 需要 host 操作，因此禁用
prefill piecewise CUDA graph；保留独立的 decode CUDA graph。
当前要求 pipeline parallel size 为 1；可以独立验证 tensor parallel。
PDMux 和 two-batch-overlap 会创建多个 Quest backend 实例，也不支持。
这里的 two-batch-overlap 与普通 overlap scheduling 是不同开关。

`quest_page_size=64` 是选择器页大小；`--page-size=64` 是 KV allocator 页大小，
两者概念不同。实验脚本显式设置两者，避免双方存储分页配置不同。
`device_buffer_size=4096` 表示每请求每层可缓存 4096 个 KV-record slots；
host ratio 影响 host capacity 和逻辑地址表，不会把物理 K/V payload 同比例放大。

按上述 Qwen config 计算，TP1 BF16 全 KV 为：

```text
48 layers * 2 (K,V) * 4 KV heads * 128 dimensions * 2 bytes
= 98,304 bytes/token = 96 KiB/token
```

200000-token prefill 的 KV 约 18.31 GiB，204800-token 则为 18.75 GiB；
`B=4096` 的 decode hot KV 约 384 MiB/请求。另有完整 host KV、页 bounds、
请求/地址/LRU 表和临时 workspace。TP1、page=64 时，200000 context 的
min/max bounds 约 293 MiB/请求；若错误使用默认 2048 个请求槽，单 bounds
便约 586 GiB。本次预算修复只保证持久分配被计入，不代替实际峰值测量。

**Chunked prefill 降低临时计算 workspace，但当前路径直到 prefill staging 完成
仍需容纳该请求完整 KV。** 30B 级 BF16 权重再加 200K KV、bounds、计算和 graph
workspace，单张 80GB H800 可能不足。建议先 TP1 小上下文验收，再根据实际
H800 数量、显存与互联测试 TP2/TP4 长上下文；不要提前承诺单卡 200K。
权重量化可另作实验，但必须记录它改变了对照配置。

## 4. 验证顺序

### CPU 回归（不安装完整 SGLang GPU 依赖）

```bash
python -m pytest -q test/manual/hisparse test/manual/test_hisparse_qwen_benchmark.py
```

CPU selector 测试调用真实 PyTorch 实现。backend/lifecycle 测试隔离真实方法并使用
CPU tensors 或事件模拟器：可以验证分支逻辑与依赖顺序，不能验证 CUDA stream
实际执行、FlashInfer attention 数值或服务进程启动。CUDA 专属测试在无 GPU 时
必须显示 skipped。

2026-10-22 本地最终结果：**46 passed、68 subtests passed、5 skipped**。
环境为 Windows、Python 3.12.14、CPU PyTorch 2.14.1、pytest 9.1.1；限制
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 后完成整套测试。
5 项跳过分别是 fused score、selection graph、finalize graph、scatter CUDA
测试和模型端到端验收。Black 26.1.0、isort 7.0.0、Ruff F401/F821、所有修改的
Python 文件语法编译及 `git diff --check` 通过。实验脚本 plan 模式已实际运行，
输出明确为 `planned`，没有模型运行或吞吐值。

### H800 到位后的 GPU 验收

使用本分支兼容的 Linux CUDA/Torch/Triton/FlashInfer/SGLang 安装环境，先运行
selector 文件中的 GPU parity、scatter 和 graph replay 测试，再做小模型端到端
对照。模型须已存在本地，避免把测试偷偷下载模型当成已具备环境。

```bash
export SGLANG_QUEST_E2E_MODEL=/models/Qwen2.5-0.5B-Instruct
python -m unittest test.manual.hisparse.test_quest_offload_equivalence -v
```

随后用论文 checkpoint 将 `SGLANG_QUEST_E2E_TOP_K=2048`，执行同一测试。
例如双卡验证可设置以下参数；实际静态内存比例仍须按机器余量调整：

```bash
export SGLANG_QUEST_E2E_MODEL=/models/Qwen3-30B-A3B-Thinking-2507
export SGLANG_QUEST_E2E_TOP_K=2048
export SGLANG_QUEST_E2E_TP_SIZE=2
export SGLANG_QUEST_E2E_MEM_FRACTION=0.85
export SGLANG_QUEST_E2E_TIMEOUT=3600
python -m unittest test.manual.hisparse.test_quest_offload_equivalence -v
```

它分别启动 Quest-HBM/HiSparse 的 eager/graph 进程，对比 greedy token IDs 和
output logprobs，包含超过 hot cache 的 prompt、80 步跨页 decode、混合 batch
和请求槽复用。此验收仍是较短的 sparse-regime 测试，最大 prompt 约 12K，
不覆盖论文 32K/200K。先通过 eager，再看 graph；之后还须验证 overlap、取消/retract、
长期负载与更高并发。不要用“生成了非空文本”替代这些验收。
该测试默认关闭普通 overlap；将 `SGLANG_QUEST_E2E_OVERLAP=1` 另跑才能检查它。

### 参数化吞吐实验

以下 `output-len=1024` 是明确选定的测试值，**不是声称论文图 5 使用 1024**。
默认仅写出命令与 `status=planned` 的 manifest，不加载模型。

```bash
python benchmark/hisparse_qwen/run.py \
  --output-len 1024 --input-lens 4096 32768 200000 \
  --concurrencies 1 2 4 8 --num-prompts 32 --repeats 3 \
  --tp-size 2 --output-dir /results/qwen-quest-plan
```

实际运行使用新的输出目录并添加 `--execute`。首次只选 4096 和低并发，开启
`--disable-cuda-graph --disable-overlap-schedule`；正确性通过后分别启用 graph 与
overlap，最后扩大至 32K/200K 和并发扫描。双方使用同样的 workload seed、
模型 revision、精度、context、scheduler cap 和并行度；只有 HiSparse 驻留路径不同。

脚本保存 git revision/dirty state、依赖版本、GPU/驱动/显存、主机内存、完整命令、
server/benchmark 日志及每点原始结果。子进程显式从本 checkout 的 `python/` 加载
SGLang，避免误跑环境中另一个安装版本。`random-ids` 配合 `--tokenize-prompt`
直接发送 input IDs，`--random-range-ratio 1` 固定输入/输出长度，用于合成吞吐
负载，不用于评估答案质量。图 5 的原始数据集/seed 未披露，须将此差异保留在报告中。

只有全部请求成功、输入/输出长度符合要求、指标有限且没有 request error，
该点才标 `success`；OOM、超时、部分完成都保留为失败，不进入有效吞吐比较。
一个点失败后停止该 server、跳过其后续点，避免残留请求污染下一次测量。
先比较相同并发的三次运行，再计算每侧并发扫描中的峰值；不得将一次最幸运的
样本或失败请求减少后的吞吐当作可复现提升。吞吐为 `bench_serving` 的端到端
generated tokens/s，包含 prefill 阶段，不冒充独立 decode-only 测量。

## 5. 尚待完成的验收

- Linux/CUDA 上编译并执行所有新 kernel parity 和 graph replay 测试。
- 小模型和真实 Qwen checkpoint 的 eager/graph output equivalence。
- TP2/TP4、overlap、并发取消/回收、重复 admission 和长时间 request-slot 复用。
- H800 实测 prefill 峰值、decode 显存、host pinned memory、传输带宽和吞吐曲线。
- 用真实文本集比较 dense Qwen 与本分支 Quest 的质量，以及检查不同 GQA 聚合策略。
- 获取论文完整 Qwen 配方后，才进一步判断是否能做到 Figure 5 的严格参数复现。
