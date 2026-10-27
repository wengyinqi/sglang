# Native build and experimental runtime image

四卡 recipe 采用 Attention-DP4+EP4，Qwen 仍单计算卡；见
[ATTENTION_DP_EP.md](ATTENTION_DP_EP.md)。需要对修正后的源码提交重新构建。

六个实验分支共享此入口。`+0` 始终是 HiSparse CPU DDR 全量 KV 后备存储。
`main` 和四个官方 HiSparse 分支不需要这些工具。

## 验证范围

- `native_environment.py prepare`：Git 提交归档、LF、锁文件哈希，不复制未提交文件。
- `execute`：在独立 Linux 前缀安装完整依赖，构建所有默认 Rust 扩展，检查 wheel、
  安装依赖、加载原生扩展，再执行 `acceptance --through jit --jit-matrix boundaries`。
- `runtime_image.py`：将上述**成功结果**打包成实验服务镜像，验证实际容器的源码身份、
  原生扩展加载。它不等同于上游 Dockerfile 的 `BUILD_TYPE=all`。
- 完整 server imports 单独记录；缺少 `libcuda.so.1` 可标为 blocked，代码/ABI 错误仍失败。
  编译或加载成功均不代表 GPU 数据通路、模型正确性或性能通过。

以每次执行生成的 `native-build.json` 与 `image-build.json` 为准，计划文件不证明成功。
此前固定提交上的 26 个 JIT 加载结果不能替代新入口、新提交的执行结果。2026-10-27
曾遇到 WSL 启动“系统资源不足，无法完成请求的服务”；恢复时没有更新内核或配置。

## 固定输入

`environments/locks/` 包含 Qwen 与 DSA 两套 Python runtime/toolchain 锁，均记录版本与
分发包 SHA256。测试工具使用 pytest 9.1.1（已包含 subtests）。Qwen GRPC 的已验证
Cargo.lock 单独保存并复制到归档源码；DSA 使用源码内的 Cargo.lock。

执行环境要求 Ubuntu 24.04 / Python 3.12 / GCC 13.3.0 / Rust 1.99.1 / uv 0.13.5。
OS 包清单与内核版本会记录；apt、工具安装介质未全部锁定，不能称为完全可复现或
逐字节一致构建。不要修改 WSL 内核来运行这个入口。

依赖安装使用哈希检查、禁止 sdist、单任务编译。源码、venv、CUDA JIT cache 和 Cargo
缓存须放在 Linux 原生文件系统，报告/源码 tar 可放在 Windows 挂载目录。每次使用新的
输出目录与前缀；失败日志保留，不能覆盖成成功。缓存目录可复用。

## 使用

在 Windows 或 Linux 上从本分支的**已提交**源码准备计划：

```sh
python benchmark/mpu_stages/native_environment.py prepare \
  --source-ref mpu-exp/05-glm-5.3-flash-4gpu-0mpu \
  --output-dir /path/to/evidence/stage05 \
  --prefix /opt/mpu/stage05-build1
```

在匹配的 Ubuntu 环境执行（Windows 生成的目录使用对应 `/mnt/d/...` 路径）：

```sh
python3 benchmark/mpu_stages/native_environment.py execute \
  --output-dir /path/to/evidence/stage05 --cache /opt/mpu/cache
```

已有下载缓存时可加 `--offline --wheelhouse /path/to/verified-wheels`。离线模式需要完整
uv 与 Cargo 缓存；缺包会失败，不自动改成在线安装。Qwen `sglang-kernel` 使用固定
GitHub release URL；离线 wheelhouse 中应包含 URL 对应的原始文件名，工具核验 SHA256
后才改为本地 file URL。普通 registry pin 也会按包名、版本、Linux/Python 3.12 兼容
标签和锁内 SHA256 绑定本地 wheel，避免 `--find-links` 被同版本 registry 候选抢先。
有效安装输入另存为前缀下的 `*-effective.txt`。

成功后 `native-build.json` 的 `built=true`、`status=built_jit_verified` 才能打包镜像：

```sh
python3 benchmark/mpu_stages/runtime_image.py \
  --native-result /path/to/evidence/stage05/native-build.json \
  --base-image-id sha256:YOUR_VERIFIED_LOCAL_UBUNTU2404_IMAGE_ID \
  --image-tag sglang-mpu:stage05-YOUR_COMMIT \
  --output-dir /path/to/evidence/image05 --execute
```

基镜像必须有 Python 3.12、匹配的 OS 运行库和 JIT 所需 GCC。传入本地不可变 image ID，
不接受可漂移标签；Docker 使用 `--network none`，不会下载新依赖。镜像保留 venv 的
绝对前缀，默认入口为 `python -m sglang.launch_server`。GPU 机器仍需 NVIDIA 驱动与
NVIDIA Container Toolkit；无 GPU 的容器 smoke 不会尝试运行模型。

使用 `docker save` 导出已构建镜像、保存其 SHA256 与 `image-build.json` 后再迁移。
当前没有 registry 推送动作；Git 推送仅包含代码、锁和说明。

## 正式 JIT 阶段

```sh
python benchmark/mpu_stages/acceptance.py --through jit \
  --jit-matrix boundaries --output-dir /path/to/evidence/jit --execute
```

顺序为 `portable → compile → jit → imports → device → kernels → correctness → performance`。
`compile` 保留 object-only 含义。默认 JIT 矩阵六分支合计 26 项，`boundaries` 合计 40 项：
每分支额外检查 CPU/MPU resolve 的 2 倍 hot buffer；GLM 另加 group_topk=128 组件特化。
组件特化不代表模型选择器预算变更已受支持。报告保存实际工厂、参数、导出函数、源码
与 `.so` 哈希；验收拒绝缺项、参数替换和被改写的库。报告与 `.so` 必须一起保存，后续
验证会重新核验文件，不能只拷贝 JSON 后声称通过。
