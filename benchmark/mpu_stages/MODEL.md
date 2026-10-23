# GLM-5.3-Flash 的 4+0 与 4+4

阶段 05 使用 CPU pinned DDR，阶段 06 使用每个 TP rank 独占的一张 MPU；默认值
由本分支 `stage.json` 选择。两分支共享相同的混合池、层映射、KPool tail 和生命周期
修复。实现来源为官方 main `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`，参考
未合并 PR #43533 的混合池适配思路，仅接入本实验需要的 NoPD 路径。

模型 checkpoint、精度、GPU/CPU/MPU 存储范围、具体参数和底层测试见
[GLM53.md](GLM53.md)。统一运行器为 `run.py`、`correctness.py`；默认只生成计划，
加 `--execute` 后才启动服务。所有变体均 TP4、无 speculative、无 PD，先比较
CPU DDR 与 MPU 的 token IDs/logprob；`--include-hbm` 添加相同稀疏算法的全驻留参考。

默认 `max_total_tokens=131072`、输入 8192/32768、输出 256、并发 1/2/4 是未测的
初始设置，不能保证给定 H200 环境一定装得下。上下文自动设置为输入加输出，避免
32768 输入被一个同样大小的上下文上限截断。正确性使用超过 4096 hot buffer 的
三条长输入、80 个输出 token、eager/graph 和槽位复用。

CPU 与 MPU 变体必须具有相同的 DSA 主 KV、index、KPool tail 和 KDA 状态分配。
运行器比较各 rank 的实际 tensor 布局和去重 storage 字节；这个签名包含主 KV，
不能与 `compute_kv_bytes` 再相加。默认索引预算支持容量比 2；扫描 1/2/4/8 时使用
`--cpu-capacity-ratio 8 --capacity-ratios 1 2 4 8` 统一冻结较大的索引预算，
并重新验证 CPU/MPU 的共同计算卡容量。不可寻址的 MPU 空间不会成为有效平台点。

CUDA 编译、4/8 卡推理、输出一致性与性能均未在本地执行；待硬件到位后按底层
读写、模型正确性、容量扫描的顺序验收，保留失败点和原始证据。
