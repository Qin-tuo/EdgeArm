# EdgeArm 本机模型与启动配置

模型默认位于 `/userdata/EdgeArm/.local/data/`，不提交到 Git。`EDGEARM_DATA_ROOT` 可修改数据根目录；更换目录时需一并迁移模型。

| 用途 | 主机目录 | 容器目录 | 完整性依据 |
|---|---|---|---|
| Qwen3-ASR | `.local/data/asr/` | `/opt/asr/models/` | `deploy/artifacts/rk_manifest.json` |
| Matcha/Vocos | `.local/data/tts/` | `/opt/tts/models/` | RK 清单及 `server/core/matcha_artifacts.json` |
| 本机 Qwen3-4B | `.local/data/llm/` | `/opt/llm/models/` | 本机 `artifacts.local.json` 和原发布清单 |
| agent Silero VAD | `server/core/assets/silero_vad.onnx` | `/opt/voice/assets/silero_vad.onnx` | `deploy/baseline.lock.json` |

## 本机已有的 Qwen3-4B

复用来源为 `/userdata/RK_LLM/artifacts/deploy/releases/42db74035211bf5b/model/`，对应 RK1828、Qwen3-4B 和 V1.1.0 工具链。四个文件作为完整配套导出复制到 EdgeArm，原发布目录不修改。

```text
.local/data/llm/
  Qwen3-4B.rknn
  Qwen3-4B.weight
  Qwen3-4B.embed.bin
  Qwen3-4B.tokenizer.gguf
  artifacts.local.json
  source-release-manifest.json
```

这套导出的模型图与仓库默认的 V1.0.4 清单不同。本机使用独立清单核对全部四个文件的大小和 SHA256；默认上游模型锁继续保留。模型完整性通过不代表已通过 EdgeArm 原生推理验收。

本机覆盖配置位于 `.local/deploy/models.local.yaml`，将 `RK1828_ARTIFACT_AUTO_DOWNLOAD` 设为 `0`，并将 `RK1828_ARTIFACT_MANIFEST` 指向 `/opt/llm/models/artifacts.local.json`。使用这套模型时，构建和启动均合并该配置：

```bash
cd /userdata/EdgeArm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml build
sudo scripts/compose.sh -f .local/deploy/models.local.yaml up -d
sudo scripts/compose.sh -f .local/deploy/models.local.yaml ps
```

`.local/` 文件仅存在于本机，克隆仓库不会获得这些模型和覆盖配置。不能将本机模型直接交给默认 V1.0.4 清单，也不能混用不同发布包的模型图和权重。原生运行仍需核对本机 RKNN3 Runtime、proxy、固件及服务 worker 的兼容性。

## ASR 和 TTS

下载覆盖所选 RK 清单的全部文件，包括 ASR tokenizer、embedding、三个 encoder、两个 decoder、服务端 VAD，以及 Matcha/Vocos/ORT 文件。Matcha 还需要发行包中的词典、音素表和语言资源；只有 `.rknn` 文件不足以运行。

下载先写入 `.part` 文件，核对大小和 SHA256 后安装到最终文件名。Matcha 压缩包及清单要求的解压文件分别校验。模型准备记录位于 `.local/reports/model-downloads.json`，下载日志位于 `.local/reports/model-downloads.log`。

当前准备范围不包含视觉 YOLOv8n-seg：该模型尚未确定目标数据、量化产物与 Runtime 配套清单。板端启动和硬件验收步骤见 [runbook](runbook.md)。

## 验证模型

先校验所选 ASR/TTS 清单、Matcha 前端资源及配套的 Qwen3-4B 四文件导出。脚本优先使用数据目录中的 `llm/artifacts.local.json`，也可用 `--llm-manifest` 指定其他配套清单：

```bash
cd /userdata/EdgeArm
source scripts/activate-env.sh
python scripts/test-models.py --verify-only
```

Docker daemon 启动后，仅启动待测的模型服务：

```bash
sudo scripts/compose.sh -f .local/deploy/models.local.yaml build speech llm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml up -d speech llm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml ps
```

两个服务健康后执行真实推理检查：

```bash
python scripts/test-models.py --services-only
```

测试依次检查 ASR/TTS 就绪标志和 LLM worker 状态，请求一条中文 LLM 回复，再合成“你好，今天天气真不错。”并交给 ASR 回读。TTS 必须输出可解码、非静音音频，归一化后的回读字符错误率默认不超过 0.3；这是单句 smoke 门槛，完整质量验收还需要真实录音和语料测试。

输出保存在 `.local/reports/model-smoke/<时间>/`，包括 `report.json`、`llm-response.json`、`asr-response.json` 和 `tts.wav`。退出码 0 表示所选检查通过，1 表示失败；仍需试听 WAV 并人工检查 LLM 回复。`--verify-only` 通过只证明文件完整性，服务未启动或加载失败时不会判为推理成功。
