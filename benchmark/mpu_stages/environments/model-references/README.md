# 固定模型版本的校验和参考

2026-10-26 使用 `model_snapshot.fetch_reference` 从 Hugging Face 固定 revision API
取得如下参考。API 返回的 SHA 与阶段配置逐字一致；元数据文件实际下载后计算 SHA256，
权重只读取 API 提供的 LFS SHA256 与大小，没有下载权重 payload。

| 文件 | 模型固定 revision | 元数据文件 | 权重文件 |
|---|---|---:|---:|
| `qwen.json` | `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` | 8 | 16 |
| `deepseek-v4-flash.json` | `60d8d70770c6776ff598c94bb586a859a38244f1` | 11 | 46 |
| `glm-5.3-flash.json` | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | 7 | 62 |

这些是有来源、可检查的校验和清单，不是签名，也不是本地权重已经校验的证据。
将所选模型的 JSON 传给 `model_snapshot.py create --reference ... --full-checksum`，
工具才会比对实际本地内容。相同模型的 CPU/MPU 两个分支共享此参考。
新 revision 或自行转换的权重需要独立记录来源，不能冒用这些参考。
