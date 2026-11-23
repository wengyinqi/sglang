# Quest 任务质量与 CPU DDR / HBM / MPU 存储对照

`task_quality.py` 独立于性能入口。它以完整 LongBench-v2 文档、原问题和选项组成
固定的聊天 prompt，并要求最终答案格式为 `Answer: X`。这是有硬件长度上限、使用
自定义答案格式 prompt 的小样本诊断，不是官方全量榜单评测。

在推理之前固定样本：seed=1729，各 domain 内打乱，然后按 domain 排序轮询选 24 条。
完整输入最多 32768 token，最大生成 8192，物理 token slots 65536。过长记录整条排除，
不裁剪或填充；全部 503 条记录的实际输入长度、是否入选及排除原因保存在
`selection.json`。某 domain 若没有足够的可容纳记录，实际分配及空缺均在 manifest 中披露。

使用已有原始数据、权重和全量 checksum 模型 manifest，不下载新的权重。
在服务器容器内、源码根目录执行（路径按 launcher 的实际 mount 保持一致）：

```bash
python benchmark/mpu_stages/task_quality.py freeze \
  --source-json /data/source/data.json \
  --source-sha256 15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2 \
  --tokenizer /models/qwen --model-manifest /results/model-manifest.json \
  --num-samples 24 --seed 1729 --max-input-tokens 32768 \
  --max-new-tokens 8192 --max-total-tokens 65536 \
  --output-dir /results/quality-inputs
```

父实验进程依次启动 CPU DDR R4、同卡 HBM Quest 和 dense FlashInfer 服务。
均保持 Qwen TP=PP=DP=EP=1、NoPD、BF16 模型/KV、page-size=64、
max-total-tokens=65536、prefill chunk=8192、prefill-max-requests=1，关闭 radix cache、
HiCache、overlap、自动截断、speculation 和 mixed chunk。共同使用 dense FlashInfer prefill。
Quest top-k=2048、quest-page-size=64；CPU 服务还设置 device-buffer-size=4096、host-to-device-ratio=4。
每个后端均运行相同 24 条冻结 IDs，temperature=0、ignore_eos=false，正常 EOS 停止。
`evaluate` 不负责启动或关闭服务，默认顺序请求，避免并发数改变质量对照。

```bash
python benchmark/mpu_stages/task_quality.py evaluate \
  --selection /results/quality-inputs/selection.json \
  --model-manifest /results/model-manifest.json \
  --source-identity /opt/mpu/ci-stage01/source/.mpu-source.json \
  --base-url http://127.0.0.1:30000 --variant cpu --capacity-ratio 4 --top-k 2048 \
  --output-dir /results/quality-cpu
```

HBM Quest 服务运行后用 `--variant hbm --output-dir /results/quality-hbm`；dense 服务运行后
用 `--variant dense --output-dir /results/quality-dense`，其余输入和预算不变。
入口检查服务实际配置，而非只相信 `--variant` 标签。
比较阶段重新核验 `server-info.json` hash、共同 prefill/精度参数和两条 Quest 路径的
selector 参数；不能把 prefill 分块或 page/hot-buffer 改变产生的数值差异归入存储搬运。

HTTP 超时或响应证据异常时，先对当前请求 ID 显式发送 `abort_request`，再调用
原生 `flush_cache?timeout=30`。abort 的 HTTP 200 仅表示接收；后者只有 scheduler
完整 idle（包括 HiSparse staging）时才成功，成功后才提交下一条。`--cleanup-timeout`
可调整这段排空等待。若排空未确认，停止后续提交，当前样本保留错误及清理证据，
其余入选样本列为 `not_run_after_cleanup_failure`；全部仍留在准确率分母中。
这类运行状态为 `failed_cleanup`，比较状态为 `compared_with_cleanup_failure`，不能称为
完整模型质量验收通过。入口不重启服务来掩盖恢复失败。

```bash
python benchmark/mpu_stages/task_quality.py compare \
  --selection /results/quality-inputs/selection.json \
  --runs /results/quality-dense /results/quality-hbm /results/quality-cpu \
  --output-dir /results/quality-comparison
```

结果包括原始完整响应、finish_reason、原生输出 token IDs、冻结输入 hash、JSON 与 JSONL、
每条答案、总体及 domain 准确率、dense/Quest 成败交换和 HBM/CPU 输出一致性。
比较阶段从完整原始响应重新解析评分，并拒绝漏样、换序、改输入、改模型或改采样配置。

Qwen Thinking 的聊天模板以 `<think>` 打开。只从输出 `</think>` 后解析最终答案；
思考区里的字母不能当成答案。length 截断、未闭合 thinking、歧义、异常输出均如实列为
不可评分，保留在全部事前入选样本的分母中，计为不正确；不能丢弃后只报告成功请求的准确率。
`completed_with_unscorable` 是已完成且含不可评分项，不是全部答案有效的验收状态。

CPU 合成测试只验证完整输入、停止策略、评分与证据合同，不证明实际模型质量：

```bash
python -m unittest discover -s test/manual/mpu_stages -p test_task_quality.py -v
```

## 02 分支的 MPU 路径

原有 dense / HBM Quest / CPU DDR Quest 三组接口与摘要字段继续兼容。02 分支新增
`evaluate --variant mpu`，用于未来一张计算 GPU 加一至四张 MPU 模拟存储卡的质量对照。
当前只有单 H800 时，CPU 合同通过不表示该 MPU/P2P 路径已经完成实卡验证。

服务必须实际开启 HiSparse、使用 `flashinfer_hisparse` decode 和 Quest，并设置
`backing_store=mpu`、`device_buffer_size=4096`、`quest_page_size=64`。MPU 容量使用
`mpu_capacity_ratio`，不把 `host_to_device_ratio` 当作 MPU 的容量。输入、计算 KV 预算、
selector、prefill、采样和停止策略保持相同；CPU DDR 与 MPU 后备容量比可以各自设置，
四组摘要会分别记录，不能把独立 MPU 容量扫描改写成 CPU 容量同时变化。

`--capacity-ratio` 指定本次 MPU 路径期望的容量比，`--mpu-device-ids` 指定期望的存储设备，
`--mpu-memory-reserve-bytes` 指定每张模拟存储卡的预留字节数。默认分别为 4、`[1]` 和
1073741824；`--compute-device-id` 默认 0，对应实际服务的 `base_gpu_id`。
存储设备必须有一至四个非负整数编号，不能重复，也不能与计算设备重叠。编号采用服务的
CUDA 可见设备命名空间；物理 UUID、P2P 拓扑和访问能力由独立硬件验收绑定。

使用实际部署的 02 源码身份路径，示例路径须按 launcher 的 mount 调整：

```bash
python benchmark/mpu_stages/task_quality.py evaluate \
  --selection /results/quality-inputs/selection.json \
  --model-manifest /results/model-manifest.json \
  --source-identity /opt/mpu/stage02/source/.mpu-source.json \
  --base-url http://127.0.0.1:30000 --variant mpu --capacity-ratio 4 \
  --mpu-device-ids 1 --mpu-memory-reserve-bytes 1073741824 \
  --compute-device-id 0 --top-k 2048 --output-dir /results/quality-mpu
```

1+4 时可改为 `--mpu-device-ids 1 2 3 4`，并独立指定 MPU 容量比，例如
`--capacity-ratio 8`。入口保存期望参数及服务实际解析后的 MPU 容量、设备顺序、reserve、
计算设备和 hot-buffer 控制；`compare` 重新检查这些参数、服务证据原 SHA 及公共预算。
服务端设备、reserve、容量或 selector 与保存的期望不匹配时，比较会失败。

四组比较显式使用 `--require-mpu`，漏掉任何一组都失败；未设置该参数且只有原三组时，
保留旧三组行为。出现 MPU 结果时同样自动要求完整四组。

```bash
python benchmark/mpu_stages/task_quality.py compare \
  --selection /results/quality-inputs/selection.json \
  --runs /results/quality-dense /results/quality-hbm /results/quality-cpu /results/quality-mpu \
  --require-mpu --output-dir /results/quality-comparison
```

四组保留 `hbm_cpu_final_answer_equal` 和 `hbm_cpu_generated_output_token_match`，新增
`hbm_mpu_final_answer_equal` / `cpu_mpu_final_answer_equal` 及对应
`*_generated_output_token_match`。token 匹配统计包括 length 停止输出；最终答案一致只统计
可解析答案。它们属于存储路径一致性，Dense 的各后端准确率及成败交换仍在
`variant_metrics` / `paired_effects` 中单列，不据此证明 Quest 普遍质量无损。

四组必须使用相同 `source_identity` 及其原文件 SHA。已冻结的 b61 版 1+0 结果继续作为
历史基线保存；不能把旧 b61 的 dense/HBM/CPU 结果与新 02 源码的 MPU 结果拼成四组
一致性证明。正式四组应在同一 02 源码部署下运行，保持完整冻结输入及模型身份不变。
