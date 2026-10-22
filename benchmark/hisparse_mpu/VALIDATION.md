# MPU 数据通路交付检查（2026-10-22）

实现分支：`hisparse_mpu`；基于 Qwen/Quest 修复提交 `105db04979914d9ca6ad723de40bd563b003b9f0`。
此记录描述本地执行的检查，不代表 H200、H800 或真实 MPU 已通过验收。

| 项目 | 本次结果 | 待硬件验证 |
|---|---|---|
| MPU backing pool | 专用分配器、CPU 兼容布局、总容量分片、无 CPU KV fallback 已接入 | native CUDA 分配、定向 P2P 与可见性 |
| peer 数据通路 | 计算卡初始化、全层 store、分层 load、fused resolve 已接入 | CUDA 编译，1+1 / 1+4 数据逐项一致 |
| 生命周期 | 整请求容量预留、staging、decode 备份、finish/retract/abort 已接入 | 服务级并发、长 decode、graph 与 stream 压力 |
| Qwen 输出 | 小模型和目标模型共用 token / logprob 对照入口 | Quest HBM 对照下 eager / graph 等价 |
| 指标 | 原始 TTFT、TPOT、端到端吞吐，独立 prefill 扣除估算，固定实际容量扫描 | 实际收益、SLO 和容量平台区 |

本次执行环境为 Windows、Python 3.12.14、PyTorch 2.14.1 CPU，无 NVIDIA CUDA。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& D:\skyler\.venv-hisparse\Scripts\python.exe -m pytest -q `
  test/manual/hisparse test/manual/mpu test/manual/test_hisparse_qwen_benchmark.py
```

结果：**113 passed、18 skipped、244 subtests passed**。18 项 skip 是显式选择运行的
GPU/模型测试，不作为通过记录。CPU 测试覆盖容量计算、分配回收、配置拒绝、实际生产
协调方法的模拟调用、graph 捕获元数据、请求槽位预算、计时与实验计划；不模拟或
证明 CUDA kernel 数值正确性。另通过 Black、isort、Ruff F401/F821、clang-format
20.1.7 格式检查和 `git diff --check`。

已执行命令计划生成：1+1 正确性 4 组、1+4 正确性 4 组，以及 10 组 / 120 cases
的容量扫描。均为 `planned`、`execute=false`；没有生成任何实测吞吐量。

集成审查同时修复了原有 HiSparse 路径的三个问题：多轮 warp 前缀和覆盖未来计数、
graph 捕获未初始化实际 resolve 的位置表且留下 backing slot 0 尾部占位、staging
请求未计入新 prefill 的请求槽位预算。相关 CPU 回归通过，CUDA 回归已提供、未执行。

运行说明及限制见 [README.md](README.md)，指标定义见 [TIMING.md](TIMING.md)。
