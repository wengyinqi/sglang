# 六阶段 MPU 实验分支

阶段中的第一个数字是计算 GPU 数，第二个数字是 MPU 数（当前用额外 H200 只作
存储模拟）。**`+0` 是 HiSparse 的 CPU DDR 后备存储，不是全 KV 驻留 GPU**。
六个阶段均按 NoPD 运行。CPU DDR 路径沿用 pinned memory；MPU 路径不通过 CPU
中转 KV，不使用 CPU KV fallback，存储卡不运行模型或 KV 计算。

| 阶段 | 分支 | 默认后备存储 | 代码来源 |
|---|---|---|---|
| 01 Qwen + Quest 1+0 | `mpu-exp/01-qwen-quest-1gpu-0mpu` | CPU DDR | 官方 `hisparse_quest` + 已完成的 Qwen/生命周期修复 |
| 02 Qwen + Quest 1+1 | `mpu-exp/02-qwen-quest-1gpu-1mpu` | 1 MPU，可配置 1+4 | 既有 `hisparse_mpu` 实现继续延展 |
| 03 DeepSeek V4 Flash 4+0 | `mpu-exp/03-deepseek-v4-flash-4gpu-0mpu` | CPU DDR C4 | 官方 main 的 V4 NoPD HiSparse + 实验修复 |
| 04 DeepSeek V4 Flash 4+4 | `mpu-exp/04-deepseek-v4-flash-4gpu-4mpu` | 每计算 rank 一个 MPU | 与 03 共用 V4 逻辑，替换 C4 后备存储 |
| 05 GLM-5.3 Flash 4+0 | `mpu-exp/05-glm-5.3-flash-4gpu-0mpu` | CPU DDR DSA KV | 官方 main + 混合池/层映射/KPool 支持 |
| 06 GLM-5.3 Flash 4+4 | `mpu-exp/06-glm-5.3-flash-4gpu-4mpu` | 每计算 rank 一个 MPU | 与 05 共用混合模型逻辑，替换 DSA KV 后备存储 |

每对分支共享相同的正确性修复，`benchmark/mpu_stages/stage.json` 指定当前阶段的
默认模型、TP、后备存储与启动参数。分支中存在共享 MPU 代码不意味着 `+0` 会分配
MPU 内存；实际选择由阶段配置控制。旧 `hisparse_mpu` 保留作历史检查点，后续工作
只进入 `mpu-exp/*`。

## 切换与验证

安装所选分支的 Linux CUDA 依赖，准备模型后，在仓库根目录运行：

```bash
git switch mpu-exp/01-qwen-quest-1gpu-0mpu
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python benchmark/mpu_stages/run.py --output-dir results/stage01-plan
# --execute 才启动 GPU 服务；输出目录必须不存在。
python benchmark/mpu_stages/correctness.py --model /models/Qwen3-30B-A3B-Thinking-2507 \
  --output-dir results/stage01-correctness --execute
python benchmark/mpu_stages/run.py --model /models/Qwen3-30B-A3B-Thinking-2507 \
  --output-dir results/stage01-performance --execute
```

其他阶段切换相应分支，入口保持一致；查看 `--help` 和本分支阶段配置再设置输入
长度、输出长度、并发、固定计算卡 KV 预算和容量比。Qwen 1+1 改为 1+4 时用
`CUDA_VISIBLE_DEVICES=0,1,2,3,4` 与 `--mpu-device-ids 1 2 3 4`。
4+4 用可见 ordinal `0..3` 计算、`4..7` 存储，默认 rank i 只访问 `4+i`。
可见 ordinal 受 `CUDA_VISIBLE_DEVICES` 重映射，不能直接混用物理卡号。

Qwen 实验分支来自较早的 Quest 开发线，V4/GLM 分支来自新的 main，依赖和内核位置
不同。每个模型系列使用与该分支匹配的环境；始终将当前 checkout 的 `python` 加入
`PYTHONPATH`，不要意外运行另一份已安装的 SGLang。

## 验证口径与存储范围

先比较相同 checkpoint、精度、稀疏选择器和输入下的 token IDs/logprob，再测性能。
`+0` 可用 GPU 全驻留版本作额外正确性参考；MPU 阶段主要与对应 CPU DDR 路径比较。
随机种子、Quest/DSA top-k、hot buffer、并发上限、prefill chunk、graph/overlap
配置及计时开关必须一致。改变后备容量时记录实际计算卡 KV slots/字节数；它们发生
变化的点不能当作仅改变 MPU 容量的实验。

- Qwen：搬运普通 MHA/GQA K/V，Quest bounds 仍在计算卡。
- DeepSeek V4：搬运 HiSparse 原本在 CPU 的 C4 KV；保留 64-token 页内 value/scale
  布局。SWA、C128、C4 indexer 和压缩状态仍在计算卡。
- GLM-5.3 Flash：搬运 DSA 主 KV；KDA recurrent/conv state 与选择器 index KV
  仍在计算卡。它们不是可直接按普通 token KV 搬运的同一种状态。

这些是对相应 HiSparse CPU 后备存储的替换，不表示把模型的全部状态都搬到 MPU。
H200 peer load/store 模拟验证也不等于真实 MPU 驱动、带宽、延迟或价格验证。

保留实测 TTFT、TPOT、端到端吞吐量；另报告 `wall time - prefill CUDA time` 的
decode 估计。TP4 的同一批次使用各 rank 时间的最大值估计关键路径，不能相加四卡
耗时，也不能减去各请求 TTFT 之和。估计仍含排队、CPU、传输和 staging 开销。
容量平台区需按输入长度、并发和 SLO 分析；没有 MPU 成本数据时不声称经济最优点。

V4/GLM 的 `host_to_device_ratio` 还控制计算侧索引/逻辑地址预算。默认先用比例 2
打通数据通路；扫描 1/2/4/8 时，使用 `--cpu-capacity-ratio 8 --capacity-ratios 1 2 4 8`
重新运行完整 CPU/MPU 组，预先固定较大的共同索引预算，只改变 MPU 容量。
运行器拒绝不可寻址的新增空间，并比较每个 rank 的主 KV、索引、压缩/循环状态
分配签名；签名字节包含主 KV，不能与 `compute_kv_bytes` 再相加。Qwen 的固定预算
仅覆盖主 K/V，逻辑地址元数据的大小仍可随后备容量变化，不能当作总 GPU 显存固定。

## 上游与实验代码隔离

`upstream` 指向 `https://github.com/sgl-project/sglang.git`；`origin` 指向个人 fork。
`main`、`hisparse-fix`、`hisparse_quest`、`hisparse-prefetch-indexshare`、
`hisparse-prefetch-indexshare-io-only` 始终只作官方镜像，不提交实验修改。六阶段代码
基线固定，更新官方镜像不会自动将新的 main 混入已准备的实验。

初次整理基线为 main `79eb10e182c62af6564e5d8aa2b62e19810b6cc2` 和 Quest 官方
`e568f8a362b4ae3a597b28b75716309814aa33fb`，保留前期修复
`105db04979914d9ca6ad723de40bd563b003b9f0`、MPU 实现
`01108050f1667ceeab3544d235ce7d5744dd66b0`。四条 HiSparse 官方分支各有不同开发
目的，没有累计合并两个 prefetch 原型。GLM 混合池参考 PR #43533 的实现思路，
该 PR 在整理时仍未合并；本实验的 NoPD 接入由本分支独立维护。

`python tools/sync_official_branches.py` 读取上游并显示计划；加 `--apply` 更新五个
本地镜像，加 `--push` 同步 fork。脚本拒绝覆盖任何 worktree 正在使用的官方分支，
对分叉或回退要求先保存备份 ref；它不会移动 `mpu-exp/*`。

目前没有 GPU 服务器。CPU 测试、静态检查和计划生成不代表 CUDA 编译、模型正确性、
长上下文稳定性或性能通过；各分支验证记录明确区分这些状态。
