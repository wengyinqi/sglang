# MPU 分阶段实验公共入口

`+0` 表示计算 GPU 配合 **HiSparse CPU DDR/pinned KV backing store**；它不是全量
KV 都驻留计算卡 HBM 的配置。当前分支的 `stage.json` 确定模型、TP 数量和 CPU/MPU
配置。六个阶段使用同一套脚本，模型专用设置保留在各分支的 recipe 中。

机器到位前先按 [PREHARDWARE.md](PREHARDWARE.md) 固定分支构建配方和模型清单。
新增 `acceptance.py` 复用下述运行器，将编译、导入、设备、内核、正确性和性能分层
验收；必需测试被跳过时停止后续层，不把计划或缺硬件误判为通过。

```bash
# 默认读取旁边的 stage.json，生成计划，不加载模型或启动 CUDA。
python benchmark/mpu_stages/correctness.py
python benchmark/mpu_stages/run.py

# 硬件与本地模型准备好后执行；每次输出目录必须不存在。
python benchmark/mpu_stages/correctness.py --model /models/model --output-dir results/check --execute
python benchmark/mpu_stages/run.py --model /models/model \
  --input-lens 8192 32768 --output-len 256 \
  --concurrencies 1 2 4 8 --num-prompts 32 --repeats 3 \
  --capacity-ratios 1 2 4 --output-dir results/perf --execute
```

输出 `manifest.json` 中的 `status=planned` 仅表示命令生成成功。没有 GPU 的开发机
能够运行 CPU 协议测试并检查计划；它不能证明 CUDA 编译、模型正确性或性能收益。

## 对照与正确性

CPU 阶段默认只使用 CPU HiSparse。MPU 阶段自动先运行 CPU HiSparse，再运行 MPU，
两者保留相同模型、稀疏 selector、计算 KV 预算、并发上限和生成参数。`--include-hbm`
仅在 recipe 明确提供保持相同 selector 的 `hbm` 配置时允许；它是额外的存储对照。
不要把 dense attention 与 sparse attention 的输出差异归为存储通路错误。

正确性脚本默认测试 eager 与 decode CUDA graph。`--disable-cuda-graph` 可只测试
eager，但不能据此宣称 graph 已通过。三轮请求覆盖超过 hot buffer 的长 prompt、
prefill chunk、长短请求混合和 KV slot 复用；每轮保存完整 token 输入与响应。
逐 token 要求 greedy token IDs 相同，选中 token 的 logprob 默认允许 0.02 绝对误差。
Near-tie 导致的分歧也会明确失败，需检查原始证据，不能静默忽略。

每轮完成后，脚本等待所有 TP rank 的计时 trace 出现该轮真实 request IDs，并取得
这些 forward 之后的空闲容量快照。实际 KV 占用与生命周期预留都必须归零。启动时的
零占用快照不能掩盖后续泄漏。MPU 的内部 padding、CPU pool 额外行和按最大输出长度
提前预留造成的排队都记录为实现事实，不能当作测量噪声丢弃。

## 容量与计时口径

Qwen 的 recipe 可要求 `--max-total-tokens` 与实际计算 KV slots 完全一致。对于
DSV4 C4 等压缩池，该参数还参与全量/SWA/压缩池的联合 profiling，不能简单除以 4
推算真实容量。此类 recipe 使用 `physical_budget_exact=false`，记录每个 rank 的
实际 pool slots、字节数和单位/范围；脚本仍强制同一 prompt 长度的 CPU/MPU、各容量
比例使用相同实际计算 KV slots 和字节数。任何 drift 都失败并保留证据。
可选 HBM 存储对照可能使用不同 pool 范围，保留其实际容量但不要求与 CPU/MPU 的
offload pool 相同；因此需单独解释该对照的显存差异，不能混入固定预算的平台判定。

CPU `host_to_device_ratio` 必须是正整数；MPU 比例可以是正浮点数。CPU baseline 的
容量比例由 `--cpu-capacity-ratio` 指定，默认 2；目标 MPU 的 `--capacity-ratios`
单独扫描。TP1 可用 `--mpu-device-ids 1 2 3 4` 测试多张存储卡；TP4 当前使用四个
存储 ordinal，各计算 rank 对应一张卡，例如 `4 5 6 7`。保持拓扑固定时扫描容量，
变更存储卡数量还会同时改变聚合链路带宽。

V4/GLM 的 `frozen_host_budget=true` 表示 CPU host ratio 同时固定 GPU index/逻辑地址
预算。扫描 `[1,2,4,8]` 前必须给所有 CPU/MPU 变体设置 `--cpu-capacity-ratio 8`，
然后通过 `--capacity-ratios 1 2 4 8` 只改变 MPU 分配。脚本拒绝超过这个固定地址
预算的扫点，也拒绝实际 `addressable_backing_slots < backing_slots` 的配置，防止
不可寻址的额外显存形成虚假的吞吐平台。Qwen recipe 可保持此选项为 false。
开启此选项还必须记录 GPU 留存池的实际字节数、形状、dtype、stride 与 storage
别名关系；V4 包含 SWA/C128/indexer/压缩状态，GLM 包含 indexer/KPool tail/线性
状态。脚本比较每个 rank 在 CPU/MPU 各比例及每轮请求结束时的这些信息，任何变化
都拒绝作为固定 GPU 预算的容量曲线。

原生 benchmark 保留端到端 TTFT、TPOT、输出吞吐量与完整请求计数。分析器另算：

`P = Σ 每个对齐 prefill batch 的 max(TP rank CUDA event 时长)`

`decode 吞吐估算 = (总输出 token − 请求数) / (benchmark wall time − P)`

TP1 时退化为单卡批次时间；TP4 **不把四张卡时间相加**，也不使用整轮最慢 rank 的
总和替代逐批次最大值。所有 rank 必须报告相同 batch 顺序、request IDs、模式与
prefill chunk 长度，trace 必须完整并在空闲时落盘。CUDA events 跨卡不具有同步的
绝对开始时间，因而 `P` 是明确标记的关键路径近似，不是精确分布式 prefill 时长。
NoPD 的 `wall time − P` 同样是反事实 decode-only 估算；TTFT/TPOT 不作此扣减。

排队、CPU/网络间隔、forward 以外的 staging、以及计时自身的开销仍保留。禁止减去
各请求 TTFT 之和，也不能把此估算描述成已运行 PD decode worker。缺失/重复 rank、
batch 不对齐、混杂请求、重算 prefill、未完成输出和非正残余时间均拒绝分析。分析和
容量验证在关闭服务前完成，避免强制终止丢失尾部证据。

`capacity-curve.json` 保存重复实验的中位数、范围和相邻比例吞吐增幅。默认低于 5%
只是待检查的平台标记；它不是统计显著性结论。需要结合 TTFT/TPOT SLO、真实硬件
成本、能耗和重复测量噪声确定性价比。H200 模拟 MPU 的测量不能直接等同真实 MPU。

## Recipe 与运行时接入约定

每个分支提供 `benchmark/mpu_stages/stage.json`，字段如下：

```json
{
  "schema_version": 1,
  "id": "02-qwen-quest-1gpu-1mpu",
  "family": "qwen",
  "tp_size": 1,
  "default_model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
  "default_revision": null,
  "variant": "mpu",
  "selector": "quest",
  "physical_budget_exact": true,
  "frozen_host_budget": false,
  "sparse_context_multiplier": 1,
  "page_size": 64,
  "server_args": ["--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--page-size", "64"],
  "variants": {
    "cpu": {
      "server_args": ["--enable-hisparse", "--prefill-attention-backend", "flashinfer", "--decode-attention-backend", "flashinfer_hisparse"],
      "hisparse_config": {"algorithm": "quest", "top_k": 2048, "device_buffer_size": 4096, "quest_page_size": 64},
      "ratio_key": "host_to_device_ratio"
    },
    "mpu": {
      "server_args": ["--enable-hisparse", "--prefill-attention-backend", "flashinfer", "--decode-attention-backend", "flashinfer_hisparse"],
      "hisparse_config": {"algorithm": "quest", "top_k": 2048, "device_buffer_size": 4096, "quest_page_size": 64, "backing_store": "mpu", "mpu_device_ids": [1]},
      "ratio_key": "mpu_capacity_ratio"
    }
  },
  "defaults": {"max_total_tokens": 65536, "chunked_prefill_size": 8192, "capacity_ratios": [2]}
}
```

公共脚本管理 TP/DP/PP、模型、上下文、请求上限、总 token 预算、chunk、graph 等参数，
recipe 中同名参数必须一致。可选 defaults 包括 `input_lens`、`output_len`、
`correctness_input_lens`、`correctness_output_len`、`concurrencies`、`num_prompts`、
`repeats`、`mem_fraction_static`、`cpu_capacity_ratio`。DSV4 C4 可设置
`sparse_context_multiplier=4`，将 hot buffer 的压缩行数换算为正确性 prompt 的原始
token 数。模型的 index_topk 等原生设置仍是权威，recipe 必须与实际 checkpoint 一致。
新 main 的 graph 验收读取 resolved `cuda_graph_config.decode.backend`；仅旧分支
没有该结构时读取 `disable_cuda_graph`，避免 legacy 标志与自动关闭的实际 backend 不一致。
新 main recipe 还必须设置 `"graph_max_bs_flag":"--cuda-graph-max-bs-decode"`，
因为它分别注册 decode/prefill 的最大 graph batch size，旧的 `--cuda-graph-max-bs`
会成为歧义参数。旧 Qwen recipe 不设置此属性，继续使用旧分支注册的原始参数名。

运行时使用 `SGLANG_MPU_STAGE_TIMING_DIR` 与
`sglang.srt.observability.mpu_stage_timing.MPUStageForwardTimer`：

1. `create(directory, server_args, tp_rank, pp_rank, dp_rank, capacity_provider=...)`。
   也可在池初始化后调用 `set_capacity_provider(callable)`。
   公共 helper `collect_stage_capacity(allocator, coordinator, tp_rank, tp_size)`
   在有 coordinator 时读取其 `get_stage_capacity_info()`，否则读取 HBM pool 的
   `size/get_kv_size_bytes()` 与 allocator 占用；返回值可以直接用作 provider。
2. 在实际执行 forward/graph replay 的 stream 中调用 `with timer.wrap(batch)`。
3. scheduler 完全空闲时调用 `timer.flush_idle()`；热路径不同步或写盘。

每个 rank 的 provider 返回 CPU 可序列化控制元数据：
`tp_rank`、`tp_size`、`backing_store`（`cpu/mpu/hbm`）、`physical_compute_slots`、
`backing_slots`、`compute_kv_bytes`、`backing_kv_bytes`、`storage_device_ids`、
`requested_capacity_ratio`、`effective_capacity_ratio`、`reserved_slots`、`used_slots`。
CPU 无预留机制时 `reserved_slots=0`；HBM 对照无 backing store 时 backing 字段可为空，
但其 compute 信息及空闲占用/预留仍必须有效。可以额外提供容量单位、pool 范围、
padding 和其他驻留池的说明。timer 自动为容量记录标注 `after_sequence`。

`describe_retained_gpu_pools({"pool_name": tensor_or_tree, ...})` 返回可直接合入
provider 的 `retained_gpu_pool_bytes` 与 `retained_gpu_pool_shapes`。传入值只接受
CUDA tensor、list/tuple、字符串 key 的 dict 或 None；模型 adapter 显式选择全部
需要冻结的留存池。helper 只读取 tensor metadata，跨全部输入去重 storage 字节数，
不读取 KV 值、不传输 payload，也不执行 GPU kernel。结果记录稳定的 allocation
group 以核对别名关系，不记录地址。`frozen_host_budget=true` 的 CPU/MPU 必须提供
非空签名，缺失时执行失败，不能仅靠配置名称宣称实际留存池已冻结。

```bash
python -m pytest test/manual/mpu_stages -q
```

这些 CPU 测试验证 TP 聚合、配置约束、实际容量一致性、空闲快照时序与失败路径；
最终仍需每个阶段的硬件正确性、kernel/profile 检查和性能验收。
# 构建与 JIT 验收更新

可复现 native 构建入口、依赖锁、实验 runtime 镜像打包和独立 `jit` 验收阶段，见
[NATIVE_BUILD.md](NATIVE_BUILD.md)。构建是否完成以对应源码提交的执行产物为准。

独立 decode profiler、设备角色检查、容量曲线重复区间与 SLO/成本策略，见
[PROFILING_AND_COST.md](PROFILING_AND_COST.md)。profiler 样本不参与性能统计。
