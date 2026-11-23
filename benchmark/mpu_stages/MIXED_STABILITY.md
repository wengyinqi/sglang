# 单计算卡的连续混合负载

`mixed_stability.py` 是只管理客户端的入口。服务由实验总入口启动一次，客户端不重启服务、不改模型或缓存配置。默认持续 10,800 秒，使用已有公开冻结 8K / 32K 输入、80 / 256 / 1024 输出；定长输出便于核验取消和 forward token 覆盖，不用于任务准确率。

服务应为 NoPD、TP=DP=EP=1，关闭 radix cache、HiCache 和 overlap。启动服务时设置 `SGLANG_MPU_STAGE_TIMING_DIR`，同一容器内向客户端传入实际服务 PID 和 trace 目录。入口校验 `/proc/PID/stat` 的进程启动时间及唯一 scheduler trace 身份，避免用服务重启掩盖状态问题。

02 分支可对一张计算卡加 1-4 张 MPU 模拟存储卡运行同一客户端。计算和
存储 ordinal 来自服务实际 `base_gpu_id` 与解析后的 HiSparse `mpu_device_ids`，
客户端必须与服务使用相同的 `CUDA_VISIBLE_DEVICES`。开始执行时只查询 CUDA
设备属性，把这些 ordinal 绑定到 UUID 和 compute/storage 角色，不依据
`nvidia-smi` 的输出行顺序推断角色，也不启动存储卡计算。

```bash
python benchmark/mpu_stages/mixed_stability.py \
  --workload /data/frozen-longbench-qwen/workload.json \
  --trace-dir /results/stability-server-trace --server-pid "$SERVER_PID" \
  --port 30000 --duration-seconds 10800 --epoch-seconds 300 \
  --concurrency 8 --cancel-rate 0.1 --seed 1729 \
  --output-dir /results/stability-3hours --execute
```

没有 `--execute` 时只生成配置与输入身份计划。输出目录必须是新目录。所有请求直接发送冻结 token IDs；完整原文不重新编码、不产生合成 token。跨 epoch 重复公开输入是有意的槽位复用测试，源 ID、行号与 token hash 均保存。

到达方式为固定 seed 的 1 / 4 / 8 请求突发，受客户端最多 8 个在途请求约束；满载时截断本次突发，不建立无限客户端队列。`request-plan.jsonl` 记录实际提交顺序和样本，可用于重放。每 5 分钟停止新增请求并排空，这是保持同一服务进程的测量边界，不是服务重启。现有 CUDA 计时只在 scheduler 全 idle 落盘，因而此边界用于读取完整 forward、KV 回收和持久分配快照。

每次排空后发相同的长输入、64-token、greedy probe，逐 token / logprob 与初始 probe 比较；原 logprob 门限 0.02 不放宽。结束时调用 flush_cache，然后再发相同 probe，检查当前 idle KV `reserved_slots=used_slots=0`。持久 KV、固定 selector / 请求映射、容量相关元数据和后端描述符的分配签名必须保持一致；allocator free worklists 是动态项，记录而不误判为持久布局变化。

取消通过 timed 或观测到第 4 个输出 token 后发送 `/abort_request`。HTTP 200 只说明调用被接收。实际取消必须返回 terminal `finish_reason.type=abort`，并分别标注 trace 中的 decode、partial-prefill、prefill-complete-no-decode 或 no-forward-observed 阶段。完整 prefill 已产生首 token 而未运行 decode 的边界单列；没有 forward 的取消不能称为 prefill 中取消。已正常结束的晚取消保留 window-missed 状态。整个数小时实验至少要真实覆盖一次 decode 取消。

`requests/` 保留每请求终态、累计 token 数、流式 chunk 时间戳和取消证据。吞吐分别列正常请求输出以及包含取消的所有已生成 token；延迟统计只用正常完成请求。TPOT 是每请求平均值，P99 chunk gap 是正 token chunk 的观测间隔，不称为逐 token ITL。多 token 一个 chunk 时不均分伪造 token 时间。采样 `/server_info`、`/get_load`、选定服务/模型 scheduler 进程 RSS 和各选定 GPU 总显存；这些资源指标有明确作用域，不当成精确 KV 事务流量。

`gpu_roles` 冻结每个选定 CUDA ordinal、UUID 与角色；资源查询按这些 UUID 筛选。
计算或存储 UUID 缺失、替换、重复，或者持久存储分配与 `storage_device_ids`
发生变化都会使验收失败。其他 NVML 可见但没有参与该服务的 GPU 不计入这项采样。
`initial_final_gpu_used_delta_bytes_by_uuid` 按 UUID 分别记录计算和存储卡端点差值。
兼容字段 `initial_final_gpu_used_delta_bytes` 始终表示计算卡差值，不是所有卡的总和；
原单卡输出的这个标量含义不变。

`mixed-stability.json` 随每 epoch 更新，保存成功、失败和回收证据。功能通过表示同进程、请求和取消范围、零 idle KV 与 probe 一致；SLO、成本或通用“无泄漏”门限仍未定义，资源与吞吐/尾延迟漂移只给观测轨迹。真正 GPU / 模型结果必须由实际执行后的文件证明。

当前单 H800 能运行 CPU/HBM 路径的客户端回归。MPU 1+1/1+4 必须有实际
可互访的存储卡和此前正确性验收；CPU 合同、编译或单卡回归不代表 P2P 已通过。
