# Lamp：OpenVoiceStream 本机改造基线

2026-10-09 用户确认：直接采用 OpenVoiceStream，硬件采用当前开发板。先复现已有工程，再接入 Lamp 外设；暂不新建业务框架或冻结 ROS 接口。

## 已选定的平台与源码

| 项目 | 选定内容 |
|---|---|
| 开发板 | Rockchip RK3588 CM3588 V2 Board，主机名 `cm3588-iseries` |
| 系统 | Ubuntu 22.04.5 ARM64，Linux 6.1.118，Python 3.10 |
| 内存 | 16GB 等级，系统可见约 15GiB |
| 加速卡 | RK1828 PCIe，设备节点 `/dev/pcie-rkep-0000:01:00.0` |
| 本机运行库 | RKNN3 1.1.0；设备固件版本和实际推理仍需验证 |
| 上游工程 | https://github.com/Seeed-Solution/openvoicestream |
| 源码基线 | `8fdc52ea65ce9a5dfd5dfe7e8b5d25943c269a11` |
| Rockchip 子模块 | `third_party/rkvoice-stream`，提交 `400e704a535e0ec00fa4f02d4ed2c4bc7b5dc308` |
| 对话依赖 | 已发布的 `voxedge==0.0.15a1`，通过 PyPI 安装 |

完整锁定信息见 [baseline.lock.json](../deploy/lamp/baseline.lock.json)。保留当前仓库的 `origin` 与历史，增加 `upstream` 远程及 `baseline/openvoicestream` 本地参考分支，采用源码快照导入；未改写原仓库历史。后续升级应对照锁定提交审查差异，不直接在当前分支执行 `git pull upstream main`。

本次准备已完成：ARM64 Python 开发依赖安装与 `pip check`、Docker 29.9.0 / Compose 5.6.0 本地工具安装、Compose 与 daemon 配置解析，以及 **117 项上游测试（全部通过）**。73 个已解析的第三方 Python 包版本记录于 [python.lock.txt](../deploy/lamp/python.lock.txt)。原始检查结果见 `.local/reports/check.log` 和 `.local/reports/baseline-tests.xml`；只读主机清单见 `.local/reports/host.json`。未启动模型服务、录音或舵机。

源码直接位于 `agent/`、`server/`、`services/` 等上游目录。Jetson 子模块保留上游引用，当前平台不初始化它。原架构书作为需求参考，其旧内存规格、首选模型及进程划分已被本次基线选择覆盖。

上游 GitHub Actions 文件保留在 [docs/upstream/ci.yml](upstream/ci.yml)，当前未启用云端 CI：首次推送时 GitHub 明确拒绝当前 OAuth 登录上传 `.github/workflows/ci.yml`（缺少 `workflow` 权限）。本机检查入口仍为 `scripts/lamp/check.sh`。后续获得工作流写入权限并适配 CI 后，再放回 `.github/workflows/`。

## 首轮运行链路

沿用 `deploy/docker-compose.conversation-rk3588-rk1828.yml`，通过 `deploy/lamp/compose.override.yaml` 添加本机配置。

| 功能 | 模型 / 推理位置 |
|---|---|
| VAD | Silero ONNX，RK3588 CPU |
| 可选唤醒 | Zipformer 中英 3M，sherpa-onnx / CPU；默认持续监听 |
| ASR | Qwen3-ASR，RKNN 编码器 + RKLLM 解码器 / RK3588 NPU |
| TTS | Matcha / ONNX Runtime CPU + Vocos / RKNN NPU |
| LLM | Qwen3-4B / RK1828，当前配套导出上下文 2048 |

Agent 配置默认连接本机 LLM，调试页面绑定回环地址，语音打断暂时关闭，待实际麦克风/AEC 验证后开启。板载 ES8389 声卡已枚举，但麦克风、扬声器物理连接尚未验证；不能据此认定音频可用。D435i 与五轴灯臂的整机接入尚未实施。

三个发布镜像已通过镜像仓库的匿名认证流程核验 Linux ARM64 清单，并锁定 digest。**发布镜像早于当前源码快照**：它们用于首轮硬件复现，不会自动包含工作区里修改的 Python 代码。修改业务代码后需要重建对应镜像，不能把旧镜像当作当前源码构建结果。

RK3588 源码构建所需原生库需按 `deploy/docker/Dockerfile.rk` 准备，RK1828 worker 与库需按 [上游构建说明](../services/rk1828-llm/BUILD.md) 准备。相关构建材料未齐备前，优先使用已锁定的发布镜像。模型大文件由上游下载器在首次启动时获取，尚未预下载，也尚未完成本机推理验收。

## 开发与检查

在仓库根目录执行：

```bash
git submodule update --init --depth=1 third_party/rkvoice-stream
scripts/lamp/setup-tools.sh
scripts/lamp/setup-dev.sh
scripts/lamp/check.sh
python3 scripts/lamp/doctor.py
```

所有工具、虚拟环境、下载和临时文件均位于仓库内的 `.local/` 或 `.venv/`。工具脚本只安装工具，不启动 Docker 或设备。`setup-dev.sh` 安装当前发布的 voxedge 包，不依赖上游开发脚本里未提供地址的相邻私有源码仓库。依赖锁、导入检查与上游测试只证明开发环境可用，不替代模型和实机测试。

`check.sh` 检查 Compose 合并结果、版本锁和 Agent 配置，并运行与该链路相关的上游测试；不会连接舵机、运行 NPU 推理或开始录音。`doctor.py` 只读检查本机条件，有运行阻塞时返回 1；它不会尝试自动重启加速卡。

## 本机容器与存储

根分区当前已满，Docker 镜像与容器数据必须放到 `/userdata`。本机配置为：

- Docker 数据：`.local/docker-data/`，配置见 `deploy/lamp/docker-daemon.json`。
- 模型与运行卷：`.local/data/{asr,tts,llm,resolved-profile,voice-data,agent-state}`。
- 检查与安装记录：`.local/reports/`。

本机没有运行中的 Docker daemon，当前账户也无法免密 sudo。已准备前台启动入口，由具备管理员权限的终端执行：

```bash
sudo scripts/lamp/start-docker.sh
```

该命令保持前台运行，数据目录按上述配置落在 `/userdata`。它不安装系统服务、不清理磁盘，也不替换已有 Docker daemon。若已有系统 Docker，先核对其 `DockerRootDir` 位于 `/userdata`，再使用它。

另一个终端可先下载锁定镜像：

```bash
sudo scripts/lamp/compose.sh pull
```

本机已有 `rk182x.service`，当前状态是 `failed`，退出码 252，未发现 transfer proxy 进程。设备节点、运行库和厂商固件文件存在，但当前账户看不到系统服务日志，故障原因尚未查明。需要管理员查看：

```bash
sudo journalctl -u rk182x.service -n 100 --no-pager
```

恢复现有厂商服务并确认 proxy 正常后，才进入模型运行验证；不混装其他版本固件或 RKNN3 库。另需恢复根分区可写空间，厂商启动脚本仍使用 `/tmp/rk-mdns.ini`；Docker 数据迁移不能替代这一系统条件。

## 后续实机启动入口

先完成以上运行条件和实际音频设备选择，再执行：

```bash
sudo scripts/lamp/compose.sh up -d
sudo scripts/lamp/compose.sh ps
sudo scripts/lamp/compose.sh logs --tail=100
```

可复制 `deploy/lamp/cm3588.env` 为 `deploy/lamp/cm3588.local.env`，设置 `LAMP_AUDIO_INPUT` / `LAMP_AUDIO_OUTPUT` 后，通过 `LAMP_ENV_FILE` 传入 `compose.sh`；本地覆盖文件不进入版本库。设备名应来自实际音频枚举，不预设板载声卡已接好麦克风。

随后按“本地语音闭环 → 五轴灯臂 → D435i/RKNN 视觉 → 并发资源测量”的顺序改造。保留上游对话和工具机制，ROS 2 是否接入控制层在外设阶段决定。
