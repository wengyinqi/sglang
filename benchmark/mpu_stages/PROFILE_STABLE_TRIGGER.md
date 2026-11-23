# 单计算卡的稳定 decode profiling

`profile_decode.py` 保留 02 分支 CPU DDR/MPU/HBM 的配置与正确性门禁，
并接入 `--profile-stable-steps` 和 `--profile-batch-size`。这两个参数约束
实际调度的 decode，不把客户端并发当成 batch。

```bash
python benchmark/mpu_stages/profile_decode.py \
  --acceptance /results/accepted/acceptance.json \
  --model /models/qwen --model-manifest /results/model-manifest.json \
  --workload /data/frozen/workload.json \
  --input-lens 32768 --output-len 1024 --num-prompts 32 \
  --capacity-ratios 2 --cpu-capacity-ratio 4 \
  --profile-variant mpu --profile-concurrency 8 \
  --profile-steps 64 --profile-stable-steps 16 --profile-batch-size 8 \
  --output-dir /results/profile-mpu-r2-c8 --execute
```

入口先启动已通过正确性验收的服务并预热，然后异步提交冻结输入。
从 `get_server_info` 的唯一计算 scheduler 读取 `mpu_forward_progress`，
要求连续 decode 的实际 request-ID 集合和 stream 不变、达到所需步数，
并满足可选实际 batch 约束。观察到合法状态后再发 `start_profile`，保存
触发前与接收后两份进度元数据。请求提前完成、失败、请求 ID 重复或
一直达不到该 batch 都会拒绝采集，不能用固定等待时间或过期状态代替。

请求生成预算至少保留稳定等待步数加采集步数和四步收尾余量。
服务 profile 停止边界仍须从实际 trace 判断，计划 64 步不能代替实际
Graph launch 数。进度是已提交 forward 的 CPU 元数据，不是 CUDA 完成时间；
kernel sum、活跃 union、首尾 span 和请求延迟分别报告。

02 分支的 `--profile-variant cpu` 始终使用独立的 `--cpu-capacity-ratio`；
MPU 使用本次 `--capacity-ratios` 的第一个点，HBM 不使用后备容量比。
CPU 参考的正确性身份也单独核验，不能把 MPU 扫点容量误当成 CPU 参考容量。
`--profile-batch-size` 需同时设置正的 `--profile-stable-steps`。不指定稳定
步数时保留原先的 decode profiler 触发方式。

单 H800 只能执行 CPU/HBM 的采集与通用回归。MPU 采集仍要求真实第二张
P2P 可用卡及相同配置的模型正确性验收；工具计划和 CPU 合同不构成 P2P 证据。
