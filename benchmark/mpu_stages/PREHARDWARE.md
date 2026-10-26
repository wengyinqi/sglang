# 机器到位前的准备与到货验收

六个实验分支复用本目录脚本，`stage.json` 决定模型和默认 CPU/MPU 后备存储。
`+0` 仍是 HiSparse CPU DDR；所有阶段是 NoPD。此入口不改变模型或缓存算法。

## 1. 固定所选分支的构建来源

切到要测的 `mpu-exp/*` 分支后，先提交需要包含的修改，再生成构建上下文：

```bash
python benchmark/mpu_stages/build_environment.py \
  --source-ref HEAD --output-dir /results/environment-plan
# 在装有 Docker 的 Linux 构建机执行另一份新计划：
python benchmark/mpu_stages/build_environment.py \
  --source-ref HEAD --output-dir /results/environment-build --execute
```

这会选择 Qwen 或 V4/GLM 对应的配方，并验证该提交 `pyproject.toml` 的核心版本。
源码来自固定 commit 的 `git archive`，不包含未提交文件；不能用输出目录里的
`recipe_generated` 证明 Docker 已构建。详情与镜像运行方法见
[environments/README.md](environments/README.md)。基础镜像标签和尚未锁定的间接
依赖会明确标记；保存实际构建的镜像 ID、RepoDigests、完整 pip freeze 和工具链记录。
切换 Qwen 与新 main 系列时必须切换环境。每个镜像都要对应待验收分支的源码提交。

## 2. 开发机先执行便携测试

```bash
python benchmark/mpu_stages/acceptance.py \
  --through portable --output-dir /results/portable --execute
```

它复用已有 CPU 控制逻辑测试，并检查新工具。没有 CUDA 时不会混入 GPU 测试再把
skip 当通过。输出的 `requested_scope_passed` 只覆盖请求的层；后续层为
`not_requested`。不加 `--execute` 时只生成 `planned` 计划。

在匹配的 Linux CUDA 开发镜像中，没有 GPU 也可尝试：

```bash
python benchmark/mpu_stages/acceptance.py \
  --through compile --output-dir /results/compile --execute
```

此处明确只把生产 MPU CUDA 头文件及其真实 FFI 导出编译成 Hopper `sm_90a` object，
记录 nvcc 命令、来源 SHA、产物 SHA 和依赖文件。它不证明链接、原生模块加载或 GPU
执行。缺 toolkit/FFI 头文件时会阻塞；编译错误会失败。需要驱动的导入和 JIT 加载
必须单独执行，不能用 Python 语法编译或 `find_spec` 代替：

```bash
python benchmark/mpu_stages/hardware_probe.py --mode imports --output /results/imports.json
python benchmark/mpu_stages/hardware_probe.py --mode compile --output /results/native-load.json
```

## 3. 准备本地模型身份

仓库已包含本轮从固定 revision 取得的三份
[校验和参考](environments/model-references/README.md)，可随代码带到服务器。
如需重新取得参考，以下命令只下载元数据及 LFS 校验和，不下载权重：

```bash
python benchmark/mpu_stages/model_snapshot.py fetch-reference --output /results/model-reference.json
```

权重准备好后，创建完整内容清单。该操作会读取所有权重字节，需要相应磁盘带宽和
时间；`verify` 也会按清单强度重新读取。只记录文件大小的默认清单不能通过模型验收层。

```bash
python benchmark/mpu_stages/model_snapshot.py create \
  --model /models/model \
  --reference benchmark/mpu_stages/environments/model-references/glm-5.3-flash.json \
  --full-checksum --output /results/model-manifest.json
```

以上以 GLM 为例；Qwen 使用 `qwen.json`，V4 使用 `deepseek-v4-flash.json`。

本地目录名或声明 revision 不证明权重来源。没有可信参考时，完整 SHA 清单证明的
是可重复的本地内容身份，报告的 `upstream_identity_verified` 仍为 false，并阻塞
总入口的目标模型验收层。每个模型
系列的两个分支可共享同一模型清单；不同模型不能共用。

## 4. 硬件到位后的统一入口

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python benchmark/mpu_stages/acceptance.py \
  --through correctness --model /models/model \
  --model-manifest /results/model-manifest.json \
  --output-dir /results/acceptance --execute
```

Qwen 1+1 使用计算卡加一张存储卡；1+4 加 `--mpu-device-ids 1 2 3 4`。
4+0 只需要四张计算卡；4+4 默认计算 ordinal `0..3` 配对存储 ordinal `4..7`。
传入的编号都是 `CUDA_VISIBLE_DEVICES` 重映射后的 CUDA ordinal，可见 UUID 也会保留。

顺序为 `portable → compile → imports → device → kernels → model_identity → correctness`。
加 `--through performance` 才继续性能测试，不能跳过前面的正确性层。通用输入长度、
并发、容量和超时参数沿用 `run.py`；例如更改固定索引预算与扫点时，两层都接收
`--cpu-capacity-ratio 8 --capacity-ratios 1 2 4 8`。这些参数仍需按实际装载容量校准。

- `device` 在每个计算 rank 和每个指定 MPU 上做定向 P2P 小数据读写、allocator 与
  显存预留检查。存储卡只分配空间，读写由计算卡发起；这不等于 profiler 证明。
- `kernels` 复用已有 CUDA 算子测试，Qwen 包含所选全部 shard。V4/GLM 将已有单对
  测试分别映射到全部四对设备，逐对执行；不是同时四 rank 的并发通信测试。
- `correctness` 复用已有运行器，强制加入相同 selector 的 HBM 存储参考，再比较
  CPU DDR/MPU、eager/graph、长短长请求和正常完成后的槽位复用。指定
  `--disable-cuda-graph` 后只验证 eager，报告不能被解释为 graph 已通过。
- 必需的 JUnit 测试出现 skip、空集合或缺失报告会阻塞后续步骤；测试失败会停止。
  总报告 `acceptance.json` 引用每层原始日志、JUnit、探针与模型 manifest，且保留
  `passed / failed / blocked / not_requested / not_run_after_failure`。

该入口不认证尚未补齐的 abort/retract 压力、四对设备同时执行、GLM 的 2051 列
真实 resolve→MPU 联合路径，也不把性能数据自动判为经济最优点。模型性能入口保留
实测 TTFT/TPOT/吞吐量与独立的 prefill 扣减估算；更多口径见 [README.md](README.md)。

## 本轮证据边界

2026-10-26 的开发主机是 Windows CPU 环境，无可用 Docker/CUDA 构建环境。可执行的
验证是便携测试、计划生成、源码归档和探针的阻塞行为；Linux object 编译、原生模块
加载、P2P、完整模型和性能均未在该主机执行。先保存这些明确状态，再在目标环境补验。
