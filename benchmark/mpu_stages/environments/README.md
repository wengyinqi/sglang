# 六阶段 Linux 环境与模型身份准备

这里准备构建配方、固定实验源码和模型身份证据。生成配方不等于完成镜像构建；镜像构建成功也不等于通过 CUDA 编译、GPU import、模型启动或性能验收。Windows 上只需 Python 标准库与 Git 即可生成和检查计划，不要求安装 Docker、WSL 或 CUDA。

## 两套依赖栈

| 配方 | stage.json family | 阶段 | pyproject 精确依赖 | CUDA 构建参数 |
|---|---|---|---|---|
| qwen-cu130 | qwen | 01、02 | torch 2.9.1；flashinfer-python 0.6.8.post1；sglang-kernel 0.4.1.post1 | 13.0.1 |
| dsa-cu130 | deepseek-v4-flash、glm-5.3-flash | 03–06 | torch 2.14.1；flashinfer-python 0.7.2；tilelang 0.1.12；sglang-kernel 0.4.10 | 13.0.3 |

生成器读取**指定 commit 中**的 `stage.json`、`python/pyproject.toml` 和 `docker/Dockerfile`。任一必备精确版本不匹配就失败，不会换成另一栈。Qwen 选用其 Dockerfile 已支持的 cu130 路径，与 pyproject 的 cu130 源一致；新栈沿用其 Dockerfile 的 CUDA 13.0.3。镜像目标为 Linux amd64；CUDA 驱动、H200 拓扑和模型支持仍需硬件预检。

## 固定源码并生成配方

在已经提交实验改动的仓库中运行；`--source-ref` 必须指向本地存在的已提交版本，解析后的完整 commit 会写入证据。可以从一个 worktree 为六个分支分别生成，源码不取当前 worktree 的未提交内容。

```powershell
python benchmark/mpu_stages/build_environment.py `
  --source-ref mpu-exp/01-qwen-quest-1gpu-0mpu `
  --output-dir artifacts/env-stage01 `
  --image-tag local/sglang-mpu:stage01

python benchmark/mpu_stages/build_environment.py `
  --source-ref mpu-exp/06-glm-5.3-flash-4gpu-4mpu `
  --output-dir artifacts/env-stage06 `
  --image-tag local/sglang-mpu:stage06
```

输出目录必须不存在或为空，避免覆盖已有证据。可选 `--stage PATH` 必须与该 commit 中的 stage.json 完全一致，默认直接读取该 commit 的配置。

- `source.tar`：由 `git -c core.autocrlf=false -c core.eol=lf archive <commit>` 生成，只含提交内文件，记录 SHA256。命令级覆盖避免 Windows 全局换行配置污染 Linux shell 文件，不修改用户 Git 设置。不会打包工作目录中的未跟踪文件、用户凭据或 `.git`。工作目录是否 dirty 另行记录，dirty 改动明确不包含在源码内。
- `build-context.tar`：保留原归档的权限位和符号链接，并增加几个明确生成的 `_mpu_build` 文件。无须在 Windows 解包或创建符号链接。
- `Dockerfile`：复用对应源码版本的仓库 Dockerfile，只将 SGLang 源码选择固定为 local archive、按 pyproject 修正 kernel/FlashInfer 构建参数，并增加来源与安装元数据采集层。不会从官方 release/main 克隆 SGLang 来替代实验分支。
- `environment-plan.json`：预期包版本、源码 commit、dirty 状态、归档与上下文 SHA、构建命令、镜像 tag、外部基础镜像引用和证据边界。`built=false`、image ID 为空。

运行和源码编译均使用 `/sgl-workspace/sglang` 内的归档源码。`PYTHONPATH` 固定到该目录的 `python`；源码目录内的 `.mpu-source.json` 保存 `source_commit`、`source_archive_sha256` 和 `provenance=git_archive`，供没有 `.git` 的镜像读取来源。完整源码及其构建输入是可审查的；若有尚未提交的实验修复，须先提交后生成。

## 显式构建与采集

在 Linux Docker 主机上，使用相同命令加 `--execute`，并选择新的输出目录：

```bash
python benchmark/mpu_stages/build_environment.py \
  --source-ref HEAD --output-dir artifacts/env-built \
  --image-tag local/sglang-mpu:validated-source --execute
```

也可以把已生成的 tar 传至 Docker 主机，按 environment-plan.json 的 `build.command` 运行，将 `build.stdin_file` 接到标准输入；要获得完整构建回执，应在主机调用同一模块的 `execute_build(plan, output_dir)`，且更新移动后的 stdin_file 路径。上下文 SHA256 必须仍匹配。

实际执行会记录 `docker-build.log`、`environment-build.json`、`image-inspect.json`、`environment-runtime.json` 和完整 `pip-freeze.txt`。读取镜像记录时按不可变的本地 image ID 运行，使用 `--network none`，不申请 GPU。任一必备包缺失/版本错误、SGLang 来源不匹配或 Docker 失败都会失败，不会标记已建成。成功状态是 `built_metadata_verified`，仅表示完成构建与安装元数据检查。

镜像本地 ID 与 registry digest 是不同概念。未推送的本地镜像可以没有 `RepoDigests`，此时忠实记录空列表，不能编造 digest。`--base-image` 可显式替换 NVIDIA base/runtime 镜像引用；新栈还支持 `--uv-image`。操作者可传入自己已经核实的 `name@sha256:...`；生成器只记录其格式，不声称已在线确认 registry。替换 base 时必须保持配方 CUDA 工具链兼容。

当前配方仍继承上游 Dockerfile 的 apt、传递依赖、uv latest 和辅助源码下载规则；完整 pip freeze 是构建后的证据，并不把这些输入自动变成预先锁定的依赖。`all_dependencies_locked=false` 始终显式保留。若需要逐字节重建同一镜像，还需另行固定所有外部镜像、apt/whl 与辅助源码产物。

## 本地模型与 tokenizer 身份

三个模型的预期仓库和固定 revision 取自各阶段 stage.json：Qwen 阶段共用一个模型身份，DeepSeek-V4 阶段共用一个身份，GLM-5.3 阶段共用一个身份。工具不把本地路径、HF 缓存目录名或声明的 revision 当作权重身份已经验证。

```bash
# 输出放在模型目录之外；默认只核对元数据与分片库存。
python benchmark/mpu_stages/model_snapshot.py create \
  --stage benchmark/mpu_stages/stage.json --model /models/target \
  --output artifacts/model-metadata.json

# 硬件验收所用的本地强身份：首次创建与随后 verify 都会读取全部权重。
python benchmark/mpu_stages/model_snapshot.py create \
  --stage benchmark/mpu_stages/stage.json --model /models/target \
  --output artifacts/model-full.json --full-checksum

python benchmark/mpu_stages/model_snapshot.py verify \
  --manifest artifacts/model-full.json --model /models/target \
  --stage benchmark/mpu_stages/stage.json --require-full-checksum \
  --output artifacts/model-verification.json
```

清单会记录 config、tokenizer、chat template、模型 Python 代码和权重索引的 SHA256，以及索引引用的全部 shard。缺分片、空文件、Git LFS 指针、路径越界、元数据变更、阶段 family/model/revision 不匹配都会被检测。`--full-checksum` 使用流式读取，不把权重整体装入内存；成本约为读取一遍全部权重的磁盘 I/O。进行创建与验证时应保持模型目录静止。

- `metadata_only`：元数据已哈希，权重只记录是否存在、长度及时间戳。不能证明权重内容相同，不满足 `--require-full-checksum`。
- `full-content-fingerprint`：所有本地权重内容均被 SHA256 覆盖，可在复制/换路径后复核相同内容；没有可信上游参考时，仍不能证明这些字节来自声明的 HF revision。
- `upstream_identity_verified=true`：只有完整权重及元数据的 SHA256 与匹配固定 revision 的可信参考全部一致才成立。此状态仍不代表模型启动/准确性验收。

可选在线获取小型元数据与 HF LFS SHA256，**不会下载权重 payload**：

```bash
python benchmark/mpu_stages/model_snapshot.py fetch-reference \
  --stage benchmark/mpu_stages/stage.json --output artifacts/model-reference.json
python benchmark/mpu_stages/model_snapshot.py create \
  --stage benchmark/mpu_stages/stage.json --model /models/target \
  --output artifacts/model-full.json --full-checksum \
  --reference artifacts/model-reference.json
```

`fetch-reference` 只针对公开 HF 仓库，以 HTTPS 的固定 revision API 和 metadata 内容为信任来源；输出不是签名文件，需保存和审查。离线/私有仓库可提供独立可信的同格式参考。网络失败、没有可用的 LFS SHA256、metadata 超出 64 MiB 上限会报错，不会补填未知身份。
