# 本地验证记录：DeepSeek V4 Flash

日期：2026-10-23。环境：Windows、CPU PyTorch，无 CUDA/GPU。
阶段 03/04 使用相同实现，分别默认 CPU DDR 与 MPU C4 backing。

```text
python -m pytest -q test/manual/mpu test/manual/mpu_stages
46 passed, 4 skipped, 16 subtests passed
```

通过项覆盖 C4 64-row/37440-byte 布局、value/scale 地址、全生命周期页预留、容量
不足与释放、MPU rank 所有权、Graph 重捕获保护、保留 GPU pool 的分配签名，以及
真实 native HTTP stream 的请求 ID/完成 token 计数和 TP4 计时分析。
若生产方法使用隔离导入或 spy，测试只证明对应的 CPU 控制分支。

Python 编译/格式及 diff 检查通过。两阶段均生成正确性与性能计划；4+4 另验证
固定 host/index 预算 8 的比例 1/2/4/8 计划。启动参数使用本 main 的
`--cuda-graph-max-bs-decode`。这些步骤不启动服务、不测性能。

4 个 GPU 项目因无 CUDA 被跳过；上游 allocator 单元测试在 Windows 无法导入
Unix `resource`，仍需 Linux 环境执行。CUDA JIT 编译、value/scale P2P roundtrip、
TP4/八卡模型、abort/retract 压力、graph 输出与全部 TTFT/TPOT/吞吐量均未验证。

checkpoint config 已核对 revision `60d8d70770c6776ff598c94bb586a859a38244f1`，
未下载权重。默认预算只作为待校准起点，不能据此声称 H200 可容纳或已有性能收益。
