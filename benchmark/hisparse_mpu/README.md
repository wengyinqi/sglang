# HiSparse MPU 数据通路与 1+1 / 1+4 实验

本目录记录 Qwen + Quest 的 MPU backing store 实现及手动实验命令；当前六阶段的
默认配置以 `benchmark/mpu_stages/stage.json` 和仓库根目录 `MPU_EXPERIMENTS.md` 为准。
计算 H200 运行模型与 KV 读写
kernel，其余 H200 只分配 KV 显存，由计算卡发出 peer load/store。存储卡不启动
KV 计算 kernel；CPU 只管理请求、分配器和指针元数据。MPU 模式不分配 CPU pinned
KV backing store，不通过 CPU 中转 KV，也不在 P2P 不可用时回退到 CPU。

当前交付是实现和可执行验证入口。开发机没有 CUDA/H200；未执行 MPU CUDA
编译、双卡/多卡读写、模型正确性或吞吐量测试。下面的数值是实验参数，不是测量结果。
本次本地检查记录见 [VALIDATION.md](VALIDATION.md)。

## 范围与布局

- 一个计算 worker，TP=DP=PP=1，NoPD，BF16 Qwen MHA/GQA + Quest，关闭 radix cache。
- 计算卡为 CUDA-visible ordinal 0，存储卡使用独立 ordinal，可选 1–4 张。
- 每个存储 shard 是 `[2, layers, shard_rows, kv_heads, head_dim]`，K/V、layer、token
  行的地址语义沿用 HiSparse `layer_first` CPU KV 布局。全局 token slot 均匀分片到
  多张卡；额外填充行不计入可用容量。
- 计算卡保留 prefill 暂存与 decode hot KV；全量已接纳请求的 KV 保存在 MPU。
  `--max-total-tokens` 在 MPU 模式指定计算卡物理 KV slots，容量比例的分母也是这些
  slots。MPU 总可用 slots 为 `ceil(physical_slots × mpu_capacity_ratio)`，比例指所有
  存储卡之和，不是每张卡的比例。
- 初始实现的长请求 prefill 仍须满足计算卡暂存空间要求。多卡容量提高不等于计算卡
  可以 prefill 任意长单请求；大容量收益应通过允许的上下文长度与并发共同扫描。
- 接纳请求时按输入长度加最大输出长度保留 MPU 容量，直至完成/abort/retraction
  正确释放。该策略为后续 token 留出空间；即使当前实际 KV 还未占满，新的请求也
  可能等待。`reserved_mpu_slots` 与 `used_mpu_slots` 分别揭示预留和实际占用。
- 这里验证的是普通 KV cache 的存储替换；不承诺 MLA/DSA、量化 KV、speculative、
  PD、分布式计算、多租户或真实 MPU 驱动支持。

仅能使用 Linux CUDA、可进行计算卡到存储卡方向 P2P 访问的拓扑和 native CUDA
allocator。VMM/expandable segments、`cudaMallocAsync` 等不兼容分配方式应直接报错。
在正式实验前保存 `nvidia-smi topo -m`、GPU 型号、驱动与 CUDA 版本。NVLink 和 PCIe
的结果分开分析；H200 模拟器的带宽、延迟、能耗与成本不能直接当作真实 MPU 参数。

## 准备与启动

在该 checkout 安装 Linux CUDA SGLang 开发环境，参考
[`benchmark/hisparse_qwen/README.md`](../hisparse_qwen/README.md) 的 Qwen/Quest 依赖与
模型说明。安装依赖后运行命令时保留所有计算与存储 GPU 可见：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
# 使用 native allocator；不要开启 expandable_segments。
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
python -m sglang.launch_server \
  --model-path /models/Qwen3-0.6B --tp-size 1 --dp-size 1 --pp-size 1 \
  --base-gpu-id 0 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --enable-hisparse --prefill-attention-backend flashinfer \
  --decode-attention-backend flashinfer_hisparse \
  --disable-radix-cache --disable-overlap-schedule --page-size 64 \
  --max-total-tokens 4096 --max-running-requests 3 \
  --context-length 2048 --chunked-prefill-size 512 \
  --hisparse-config '{"algorithm":"quest","top_k":128,"quest_page_size":64,"device_buffer_size":256,"backing_store":"mpu","mpu_device_ids":[1],"mpu_capacity_ratio":2,"mpu_memory_reserve_bytes":1073741824}'
```

`/get_server_info` 的 `internal_states[0].hisparse_mpu_capacity` 提供实际计算卡容量、
MPU 可用 slots、分片与字节数。runner 强制检查实际值符合固定预算；不能仅凭请求
参数声称扫描使用了相同计算卡容量。MPU 大分配前保留的 1 GiB 是示例，可显式调整。

## 先验证正确性

脚本默认只输出计划，不启动服务。添加 `--execute` 才会运行。输出目录必须不存在，
避免覆盖证据。先使用本地小 Qwen 模型，通过后换成目标 Qwen 模型；不以小模型通过
替代 Qwen3-30B-A3B-Thinking-2507 验收。

```bash
python benchmark/hisparse_mpu/correctness.py \
  --model /models/Qwen3-0.6B --output-len 80 \
  --top-k 128 --device-buffer-size 256 --max-total-tokens 4096 \
  --chunked-prefill-size 512 --capacity-ratios 1 \
  --mpu-device-ids 1 --output-dir results/mpu-smoke-1p1 --execute
```

它先运行 Quest full-HBM eager 参考，再比较 MPU eager、Quest graph、MPU graph。
相同输入逐 token 比较 greedy token IDs 和选中 token 的 logprob（默认绝对误差
上限 0.02），记录第一处 token 分歧。三轮请求覆盖大于 hot buffer 的稀疏 decode、
prefill 分块、64-token 页边界、长短请求混合和释放后的 slot 复用。保存原始 token
输入、响应、服务配置和日志；任何丢失请求、短输出、NaN、token 分歧或误差超限都失败。
若浮点近似导致 near-tie 分歧，应检查原始证据，不能静默放宽验收或宣布等价。

用同一入口测试 1+4：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python benchmark/hisparse_mpu/correctness.py \
  --model /models/Qwen3-0.6B --output-len 80 \
  --top-k 128 --device-buffer-size 256 --max-total-tokens 4096 \
  --chunked-prefill-size 512 --capacity-ratios 1 \
  --mpu-device-ids 1 2 3 4 --output-dir results/mpu-smoke-1p4 --execute
```

这里刻意用总容量比例 1，使三条并发长请求的 KV 超过三个 shard 的总行数，有机会
覆盖所有存储卡。底层跨 shard 读写/resolve 测试还需单独运行；端到端 token 一致不
足以证明访问了每条硬件链路。测试入口：

```bash
SGLANG_RUN_MPU_GPU_TESTS=1 SGLANG_MPU_TEST_STORAGE_DEVICES=1,2,3,4 \
  python -m pytest test/manual/mpu/test_mpu_pool_gpu.py test/manual/mpu/test_mpu_resolve_gpu.py -q
SGLANG_MPU_E2E_MODEL=/models/Qwen3-0.6B \
  SGLANG_MPU_E2E_DEVICES=1,2,3,4 \
  python -m pytest test/manual/mpu/test_mpu_e2e_gpu.py -q
```

正确性 runner 每轮完成后轮询 MPU 占用与预留是否归零，超时仍未释放即失败。
未设置 opt-in 时，GPU 测试明确 skip；skip 不等于通过硬件验证。

## 固定计算 KV 预算扫描容量

```bash
python benchmark/hisparse_mpu/run.py \
  --model /models/Qwen3-30B-A3B-Thinking-2507 \
  --input-lens 8192 32768 --output-len 256 \
  --concurrencies 1 2 4 8 --num-prompts 32 --repeats 3 \
  --top-k 2048 --device-buffer-size 4096 --max-total-tokens 65536 \
  --chunked-prefill-size 8192 --capacity-ratios 1 2 4 8 \
  --mpu-device-ids 1 --output-dir results/qwen-mpu-1p1 --execute
```

这是待校准的扫描示例；模型、Quest 元数据、KV 暂存和 CUDA graph 都占计算卡显存，
配置超过设备容量时应失败并保留证据。以小扫描起步，依据实际显存降低物理 KV slots。
1+4 时改可见设备和 `--mpu-device-ids 1 2 3 4`，容量比例仍是四张卡总和。若想固定
拓扑比较容量效应，在同一个设备集合上只改比例；1+1 与 1+4 的差异还包含总链路带宽
和分片访问效应，不能归因于容量一个变量。

各比例采用同样的计算 KV slots、hot buffer、Quest top-k、模型版本、最大 running
requests、输入/输出长度、随机种子、prefill chunk、graph/overlap 配置与 CUDA 计时。
默认还运行 Quest full-HBM 对照；`--modes mpu` 可单独扫描 MPU。单个 case 超时后
停止该服务剩余 case，再启动新的组，避免未结束的请求污染下一次测量。

`manifest.json` 保留每个 case 的成功/失败状态；原生 `.jsonl` 保留 TTFT/TPOT 分布、
端到端 output throughput 和全部请求计数。`*-timing.json` 另列 CUDA prefill/decode
时长与 `wall time − prefill CUDA time` 的 decode-only **估算**。runner 在关闭服务
前等待空闲 flush，并拒绝缺失分块、retraction 重算、未完成输出或混杂请求。完整口径
见 [TIMING.md](TIMING.md)。不会减去各请求 TTFT，也不把去掉 prefill 的数值当作真实 TTFT。

`capacity-curve.json` 按输入长度和并发输出重复实验的中位数、最小/最大值和相邻比例
的吞吐量增幅，默认将低于 5% 标记为待检查的平台区。失败或缺失重复不参与平台判定。
它不是统计显著性检验：需要增加重复、检查噪声和 SLO。未提供 MPU 价格/功耗时不能
计算“最有性价比”的经济最优点；先找满足容量和延迟约束的最小比例，再结合真实
成本以及完整性能曲线确定。更高容量通常通过提高可驻留并发带来收益，因此也应看
各比例在相同 TTFT/TPOT 上限下的最大可持续吞吐量。

## 硬件到位后的验收顺序

1. 双卡 P2P/allocator 检查；底层远程 KV store/load、边界、resolve 命中/未命中、
   eviction、新 token、跨 shard 与 graph replay 测试。
2. 小 Qwen 的 1+1 eager/graph token 与 logprob 验收，再运行目标 Qwen 配置。
3. 在 profiler 中确认 KV kernels 只运行于计算卡，存储卡没有模型/读写 kernel，
   CPU 不出现 KV payload staging；观察 NVLink/PCIe 传输和实际分配。
4. 重复 1+4 正确性验收，再扫描容量/并发并采集完整 TTFT、TPOT、吞吐量与 decode
   估算。报告拓扑、实际容量、失败点与未验证项目，不从源码或 CPU 测试推导收益。
