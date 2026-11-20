# 单 H800 的 Quest 任务质量诊断

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
