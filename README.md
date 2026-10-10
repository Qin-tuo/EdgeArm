# EdgeArm

EdgeArm 是基于 OpenVoiceStream 的端侧语音机械臂项目，目标硬件为 **RK3588 16GB/128GB + PCIe RK1828 + SO-101 五关节及夹爪 + D435i + 接触传感器**。唯一应用入口是 `voice_so101`。

复用上游 agent/server、音频与会话、HTTP/WebSocket、工具注册、插件、执行器和 Rockchip 推理依赖。中文默认链路为 **Silero VAD → Qwen3-ASR（RK3588）→ Qwen3-4B（RK1828）→ Matcha/Vocos TTS**。启动采用按键说话，运动默认关闭。性能、设备驱动、标定和停止行为尚未完成本机实测。

- [架构与资源所有权](ARCHITECTURE.md)
- [裁剪范围与待实现清单](docs/scope.md)、[逐文件记录](docs/file-inventory.json)
- [部署、标定与实机验收步骤](docs/runbook.md)
- [软件验证与交付状态](docs/validation.md)
- [源码、模型与镜像版本锁](deploy/baseline.lock.json)、[上游来源与许可](NOTICE.md)

仓库：[Qin-tuo/EdgeArm](https://github.com/Qin-tuo/EdgeArm)。应用入口 `voice_so101` 与 `ovs_agent` Python 命名空间沿用 OVS 契约。

## 开发和检查

```bash
scripts/setup-tools.sh
scripts/setup-dev.sh
scripts/check.sh
python3 scripts/doctor.py
```

`setup-dev.sh` 安装语音开发依赖，不启动模型或外设。舵机 SDK 使用额外依赖 `agent[arm]`；相机和接触传感器的实际适配依赖在完成选型/板端验证后加入。现有开发环境可直接运行 `check.sh`。`doctor.py` 返回主机条件是否满足，不能证明推理或音频已验收。

## 板端启动

先按运行手册恢复主机 RKNN3 proxy，确认声卡与存储可用，再构建包含本次源码的镜像：

```bash
sudo scripts/start-docker.sh
# 在另一终端执行；已有合适的 Docker daemon 时无需再次启动。
sudo scripts/compose.sh build
sudo scripts/compose.sh up -d
sudo scripts/compose.sh ps
```

配置位于 `deploy/agent-config.yaml`；只在真实标定、限位和人工监护条件满足后设置 `metadata.robot.enabled: true`，配置固定串口与标定哈希并授予设备访问，再通过本地控制入口人工使能。`enabled` 只允许连接，不会自动启动力矩。旧发布镜像仅作为 digest 锁定的构建基础，不能代替本项目镜像。

本机调试页默认 `http://127.0.0.1:18000`，语音服务 `127.0.0.1:8621`，LLM 服务 `127.0.0.1:1828`。需要远程操作时使用 SSH 转发。禁止把未鉴权的调试控制页面直接暴露到局域网。

## 目录

| 目录 | 职责 |
|---|---|
| `agent/ovs_agent/` | OVS 共享语音 runtime 与唯一 `apps/voice_so101` 应用 |
| `server/` | OVS 语音服务、RK 后端契约、会话与资源管理 |
| `services/rk1828-llm/` | 本地 OpenAI-compatible LLM 服务 |
| `configs/profiles/` | 唯一 `rk3588-default` 语音 profile |
| `deploy/` | 唯一 Compose、源码镜像、配置与版本锁 |
| `scripts/` | 安装、检查、主机诊断及本地操作入口 |
| `third_party/rkvoice-stream/` | 固定版本的原始 Rockchip 依赖 |
| `agent/tests/`、`server/tests/`、`tests/` | 保留的语音回归与 SO-101 契约测试 |

RGB-D、接触 MCU、抓放规划/结果验证目前提供有界数据及任务扩展接口，尚无本机硬件适配。缺少后端或新鲜观察时抓放返回拒绝；仅有动作已发送不能播报任务完成。详见运行手册中的剩余工作。
