# 本地验证记录：Qwen + Quest

日期：2026-10-23。环境：Windows、Python 3.11、CPU PyTorch；没有 CUDA/GPU。
阶段 01/02 使用相同代码，区别是 `stage.json` 默认 CPU DDR 或 MPU。

```text
python -m pytest -q test/manual/hisparse test/manual/mpu
  test/manual/test_hisparse_qwen_benchmark.py test/manual/mpu_stages
132 passed, 18 skipped, 249 subtests passed

python -m pytest -q test/manual/test_sync_official_branches.py
3 passed
```

CPU 验证覆盖 Quest 选择/缓存控制、CPU/MPU 生命周期容量预留、释放与复用、地址布局、
stream 协议、请求 ID/输出计数、TP 计时聚合与容量漂移拒绝。部分生产方法隔离了
CUDA-only imports，用 spy 执行控制逻辑；这不证明 CUDA kernel 或模型推理正确。
跳过的 GPU/Triton 项目保留原测试入口，不能记作通过。

两阶段均生成 performance/correctness 计划；阶段 02 另生成 1+4 计划。计划未启动
服务、未下载权重、没有 TTFT/TPOT/吞吐量数据。默认 eager/graph 和同 selector HBM
对照命令均可生成。镜像同步脚本使用临时本地 Git remotes 验证事务更新、远程保护、
worktree 占用检查和实验分支保留。

硬件到位后必须补做：CUDA/sgl-kernel 编译、定向 P2P、所有 shard 的读写、graph
replay、目标 Qwen checkpoint token/logprob 一致性、长上下文和容量压力，以及
完整性能采集。保存拓扑、精度、版本、实际容量、失败记录和原始计时。
