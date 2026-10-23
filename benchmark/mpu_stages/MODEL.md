# Qwen + Quest：CPU DDR 与 MPU

阶段 01 (`1+0`) 使用 CPU pinned DDR 保存全量历史 K/V；阶段 02 (`1+1`)
以存储 H200 替换该层。二者同源于官方 `hisparse_quest`
`e568f8a362b4ae3a597b28b75716309814aa33fb`，包含前期 Quest、请求生命周期和
MPU 修复。每个分支的 `stage.json` 决定默认后备存储。

默认 checkpoint 为 `Qwen/Qwen3-30B-A3B-Thinking-2507`，固定 revision
`144afc2f379b542fdd4e85a1fcd5e1f79112d95d`，模型/KV 均 BF16，Quest top-k=2048、
页大小 64、每请求 hot buffer=4096。输出长度、并发和容量比例是明确的实验起点，
并非论文未公开参数的还原。论文对应关系和分支依赖见
[原 Qwen 指南](../hisparse_qwen/README.md)。

MPU 使用与 CPU `layer_first` 一致的 `[2, layers, rows, kv_heads, head_dim]`
布局。计算 GPU 发出 peer load/store；存储 GPU 只持有分配，不启动 KV kernel。
跨 1–4 张 MPU 均匀切分 token rows，`mpu_capacity_ratio` 指所有存储卡可用容量之和
除以计算卡物理 KV slots。CPU allocator 的额外行与 MPU shard padding 单独记录。
CPU 与 MPU 均按输入加最大输出长度保留完整生命周期容量，完成、abort、retraction
之后释放；逻辑 prefill 槽提前释放不会使 CPU DDR 基线绕过容量限制。

Quest bounds 与索引表保留在计算卡。`--max-total-tokens` 指定计算卡 KV 预算，
运行器要求实际值一致；容量字段 `compute_kv_bytes` 仅覆盖 K/V 主存储，不包含
selector/地址元数据、prefill workspace 和 graph 开销。逻辑地址表会随后备容量变化，
不能据此声称总 GPU 显存完全相同。当前 prefill 结束后才整体 staging，单请求输入
仍必须满足计算卡暂存空间上限；增加 MPU 容量主要用于扩展历史 KV 的并发驻留。

在 Linux CUDA 环境中先安装本分支依赖；此旧开发线使用 `sgl-kernel` 布局，不能
复用新 main 分支的安装目录。MPU 需 native CUDA allocator、可验证的非 VMM 分配
以及计算卡到存储卡方向的 P2P；无 CPU KV 回退。运行示例：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
python benchmark/mpu_stages/correctness.py --execute
python benchmark/mpu_stages/run.py --capacity-ratios 1 2 4 8 --execute
```

阶段 01 只需计算 GPU；阶段 02 默认还运行相同配置的 CPU DDR 对照。
`--include-hbm` 添加相同 Quest 的全驻留辅助参考。先用小 Qwen 与较小 top-k/buffer
验证数据通路，再换回默认目标模型；小模型通过不能替代目标 checkpoint 验收。

1+4 采用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4`，并给两种入口增加
`--mpu-device-ids 1 2 3 4`。必须另跑
[底层 MPU GPU 测试](../hisparse_mpu/README.md)，确认跨 shard 读写与每条 P2P 链路，
不能只根据最终输出推断所有存储卡都已被访问。

目前没有 CUDA 服务器。CPU 测试和计划生成只验证主机控制逻辑、计时分析与配置；
CUDA 编译、P2P、eager/graph 模型输出、长上下文稳定性和全部性能结果均待硬件验收。
