# GPU 到位前的控制通路与迁移交付

六阶段定义不变：Qwen 1+0/1+1 保持单计算卡，DP=EP=TP=1；额外四进程测试只是旧版
控制代码的压力检查，不改变 Qwen 部署。DeepSeek/GLM 为 Attention-DP4+EP4，
有效 Attention/MoE/dense TP=1。+0 使用 CPU DDR 全量后备 KV；MPU 仅存储 KV。

## CPU 控制通路

Linux `acceptance.py --through portable --execute` 自动包含四项进程检查，缺少
Gloo、进程超时、缺报告或测试 skip 都不会算通过。Windows portable 不运行这些检查，
其报告的 `linux_cpu_control_executed=false`。也可单独执行：

```sh
python benchmark/mpu_stages/cpu_control.py --output /path/to/new-control-evidence
```

使用实际子进程与 Gloo，调用当前分支的 `DataParallelController` 路由方法、
`MLPSyncBatchInfo.all_gather`、`MPUSlotAllocator` 和 `HiSparseCoordinator`
回收方法。CPU fixture 用 AST 装载这些生产定义，以避开 GPU 模型依赖；报告保存
源文件哈希、进程 PID、退出码、各步 owner/token/mode、回收顺序及剩余容量。

覆盖单 rank、四 rank 的 round-robin/指定 owner 路由、idle、混合 prefill/decode、
graph eligibility 投票、abort/retract 后全量重分配，以及缺 rank 的真实 collective
超时和 rank 异常退出。进程等待有上限，失败后清理本次子进程。

请求内容/批次由测试构造，IPC 为管道；CUDA 流和完成事件是桩。它不启动真实调度服务，
不执行 HTTP abort API、GPU KV 数据读写、模型 forward 或 DeepEP/NVSHMEM。
CUDA 流竞态、P2P 数据正确性、模型输出和性能必须在机器到位后另行验收。

六个实验分支使用已登记的 `.github/workflows/run-cpu-ci.yml` 入口，调用可复用的
`mpu-cpu-control.yml`，在 push/PR 或手动 dispatch 时执行 CPU 验收。保留日志、JUnit
和进程证据，并核对实际测试 SHA；不修改 main 或官方 HiSparse 分支。

## 生成交付包

`delivery_bundle.py` 要求 Linux、Python 3.12+、Git 和 Docker。指定旧镜像的验收收据、
原生收据、JIT 收据及包含新工具的实验分支。输入数组示例：

同时接受 `runtime_image.py` 生成的嵌套 `plan` 收据和已有的平铺收据。
嵌套格式会核对原生收据 SHA256、源码提交、安装前缀、扩展清单及容器加载结果；
stage 和 wheel 身份从该哈希绑定的原生收据读取，原始镜像收据保持不变。

```json
[
  {
    "source_ref": "mpu-exp/05-glm-5.3-flash-4gpu-0mpu",
    "image_receipt": "/evidence/image-05/image-build.json",
    "native_receipt": "/evidence/native-05/native-build.json",
    "jit_receipt": "/opt/mpu/attention-dp-05-v2/acceptance/jit.json"
  }
]
```

```sh
python benchmark/mpu_stages/delivery_bundle.py create \
  --repo /path/to/sglang --spec /path/to/stages.json --output /path/to/new-bundle
```

每阶段带有 Docker image archive、源码 Git archive、原生 wheel、实际 JIT `.so`、
依赖收据、stage.json、launch.json 和固定 checkpoint 的文件/校验和参考清单。
完整模型权重不在包中。launch.json 的 `/models/selected` 是待替换路径；启动环境
包括 V4 的 `SGLANG_DSV4_FP4_EXPERTS=0`。默认 recipe 不是收益测量结果。

运行时与验收工具分别记录 `runtime_commit`、`tools_commit`，不会把旧镜像改标签
冒充新构建。只有测试/工具的明确白名单变更可复用运行时；生产代码、依赖锁或
stage.json 改变就会拒绝。恢复时再次逐文件核对生产源码、wheel 安装文件，并重新
加载原生扩展与搬迁后的 JIT 库。后者不运行 GPU 内核，也不等于重新调用 JIT 工厂。

每阶段单独压缩镜像，允许容量较小的目标机逐阶段恢复。检查 C/D 分区及 WSL
VHD 所在分区的真实空余空间；无需修改 WSL 内核。成功后将整目录复制到服务器，
并单独保留返回的 `manifest_sha256`。文件内哈希是完整性检查，不是数字签名。

## 恢复与验收

交付包根目录的 `restore.py` 可独立运行，不需要原仓库或 Git，也不依赖原主机
`/opt/mpu`、用户 CUDA/JIT 缓存或模型目录。镜像内部保留绝对 venv 前缀以保持 ABI。

```sh
python3 restore.py verify --bundle /path/to/bundle --expected-sha256 MANIFEST_SHA256
python3 restore.py restore --bundle /path/to/bundle \
  --expected-sha256 MANIFEST_SHA256 \
  --stage 05-glm-5.3-flash-4gpu-0mpu --output /path/to/new-restore-evidence
```

省略 `--stage` 会依次加载所有镜像；可重复该参数选择阶段。默认 Docker daemon
由 `DOCKER_HOST` 指定。需要隔离演练时使用新的 daemon/data-root；工具不会删除
任何旧镜像。恢复容器禁用网络，仅挂载交付包和新的证据目录，CPU 验收包括真实
Gloo 进程测试。输出 `restore.json`、镜像 ID/layer 清单、`.so` 加载记录及逐步日志。

通过 CPU 恢复后，仍需在 H200 上按 imports → device/P2P → kernels → 模型身份
与正确性 → performance/Nsight 验收。将已校验的 source.tar.gz 解压到服务器目录，
挂载模型与源码，以该源码的 `PYTHONPATH=.../python` 执行相应分支的验收工具。
真实 TTFT/TPOT 保留原值；扣除纯 prefill 的结果仅称 decode 估算。
