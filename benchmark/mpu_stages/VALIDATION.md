# 本地验证记录：GLM-5.3-Flash

日期：2026-10-23。环境：Windows、CPU PyTorch，无 CUDA/GPU。
阶段 05/06 使用相同混合模型实现，分别默认 CPU DDR 与 MPU DSA KV backing。

```text
python -m pytest -q test/manual/mpu_stages
55 passed, 1 skipped, 34 subtests passed
```

通过项覆盖混合池/global-to-dense 层映射、KPool tail 选择宽度、真实 paged
allocator 生命周期、逻辑/物理容量界限、MPU 配置和控制路径、Graph dummy maps、
留存 GPU 分配签名、native HTTP stream 的请求 ID/token 完成计数，以及 TP4
计时聚合。隔离了 GPU-only imports 的测试只验证 CPU 控制逻辑。

Python 编译/格式及 diff 检查通过。两阶段均生成正确性与性能计划；4+4 另生成
固定 index/address 预算 8 的比例 1/2/4/8 计划。使用新 main 的 decode graph
参数。计划均未启动服务，不能视作启动、内核或模型验证。

GPU 测试因没有 CUDA 跳过。待验收内容包括 CUDA JIT、DSA KV P2P、KPool 页尾与
KDA 状态协作、TP4/八卡目标 checkpoint 输出、eager/graph 等价、容量压力、
abort/retract、TTFT/TPOT/吞吐量及 decode 估算。不得将 CPU pass 或 GPU skip
标为这些验收通过。

checkpoint 固定为 `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`，权重与目标服务器
均未在此运行；所有默认参数是待校准起点。PR #43533 仍未合并，本分支是独立
维护的 NoPD 接入，不宣称该 PR 的 PD/MTP 功能已经完成。
