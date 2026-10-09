# voice-rebot-arm 端到端调通计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 voice-rebot-arm 在 Jetson (seeed@192.168.1.176) 上端到端跑通——唤醒→ASR→LLM→工具调用→机械臂动作→TTS 回话。

**Architecture:** 三个容器协作：seeed-voice (ASR+TTS+LLM 工具循环) + edge-llm (Qwen3 本地推理) + voice-rebot-arm (agent：麦克风/音箱/唤醒词/工具执行)。agent 通过 WebSocket 连 seeed-voice 的 /v2v/stream；seeed-voice 在 server-loop 模式下调 edge-llm 做 LLM 推理、发 SERVER_TOOL_CALL 给 agent 驱动机械臂。

**Tech Stack:** Docker Compose overlay, FastAPI, WebSocket, PulseAudio/ALSA, B601-DM 串口 SDK

---

## 发现的问题（按严重程度排序）

### 🔴 P0 — 阻塞所有功能

| # | 问题 | 根因 | 影响 |
|---|---|---|---|
| **1** | `actions.yaml` 路径不匹配 | Dockerfile 把代码拷到 `/opt/slv/agent/...`，但 config.yaml 默认找 `/opt/seeed/voice_rebot_arm/config/actions.yaml`，Dockerfile 未设 `CONFIG_DIR` | `ArmPlugin.setup()` 返回 False → 零个工具注册 → LLM 无工具可调 → 机械臂永远不动 |
| **2** | seeed-voice 缺 server-loop 环境变量 | `docker-compose.yml` 未设 `OVS_V2V_ENGINE=voxedge` + `OVS_V2V_SERVER_LOOP=1` | server 不跑 LLM 循环 → 不发 `SERVER_TOOL_CALL` → 即使工具注册了也不会被调 |
| **3** | agent 镜像需手动构建 | compose 无 `build:` stanza，默认 `voice-rebot-arm:dev` 不存在 | `docker compose up` 直接报 image not found |

### 🟠 P1 — 音频不通

| # | 问题 | 根因 | 影响 |
|---|---|---|---|
| **4** | `PULSE_SERVER` 无条件硬编码 | compose 写死 `unix:/run/user/1000/pulse/native`，若 Jetson 未跑 PulseAudio，socket 不存在 | PortAudio 打开失败 → 无麦克风无音箱 |
| **5** | `SPEAKER_DEVICE` 未设 | compose 没传 `SPEAKER_DEVICE`，config 默认空串 → `resolve_output_index` 回退系统默认 | TTS 音频送到 HDMI/APE 而不是 reSpeaker 扬声器 |
| **6** | Pulse config 主机路径可能不存在 | `${PULSE_CONFIG_DIR:-/home/seeed/.config/pulse}` 绑定挂载，若目录不存在 compose 报错 | 容器启动失败 |

### 🟡 P2 — 可能影响

| # | 问题 | 根因 | 影响 |
|---|---|---|---|
| **7** | edge-llm 健康检查端点未确认 | compose 用 `/health`，但 edge-llm 是外部镜像 | 若 endpoint 不存在 → `depends_on: service_healthy` 永远不满足 → agent 不启动 |
| **8** | `OVS_V2V_SYSTEM_PROMPT` 未设 | server 端的系统提示为 None，依赖 voxedge 引擎是否使用 agent 通过 `CLIENT_TOOL_ADVERTISE` 发送的提示 | LLM 可能不知道自己是机械臂控制器，不发工具调用 |

---

## 修复计划

### Task 1: 修复 actions.yaml 路径（P0 #1）

**Files:**
- Modify: `agent/Dockerfile.rebot-arm:116-121`

- [ ] **Step 1: 在 Dockerfile ENV 块中添加 CONFIG_DIR**

```dockerfile
ENV REBOT_REPO_ROOT=/opt/rebot \
    REBOT_CHANNEL=auto \
    OBSERVATION_PORT=8775 \
    MIC_INDEX=reSpeaker \
    WAKEWORD_MODEL="hey jarvis" \
    OVS_AGENT_SERVER_LOOP=1 \
    CONFIG_DIR=/opt/slv/agent/ovs_agent/apps/voice_rebot_arm
```

这样 `config.yaml` 里的 `${CONFIG_DIR:-...}/actions.yaml` 解析到镜像中实际存在的文件。

- [ ] **Step 2: 验证路径链条**

确认 `COPY agent /opt/slv/agent` (line 112) + `pip install` (line 113) 后，`/opt/slv/agent/ovs_agent/apps/voice_rebot_arm/actions.yaml` 确实存在。

- [ ] **Step 3: Commit**

```bash
git add agent/Dockerfile.rebot-arm
git commit -m "fix(voice-rebot-arm): set CONFIG_DIR so ArmPlugin finds actions.yaml"
```

---

### Task 2: 完善 compose overlay（P0 #2 #3 + P1 #4 #5 #6）

**Files:**
- Modify: `deploy/docker-compose.jetson-rebot.yml`

- [ ] **Step 1: 重写 overlay，修复所有已知问题**

```yaml
# OpenVoiceStream — Jetson rebot-arm overlay.
#
# Layers the B601-DM robot-arm agent + edge-llm on TOP of the base Jetson
# speech service. Use with docker compose multi-file:
#
#   docker compose \
#     -f deploy/docker-compose.yml \
#     -f deploy/docker-compose.jetson-rebot.yml \
#     up -d
#
# Build the agent image first (from repo root on the Jetson):
#   docker build -f agent/Dockerfile.rebot-arm -t voice-rebot-arm:dev .

services:
  # ── Patch speech: enable server-loop for tool calling ──────────────────
  speech:
    environment:
      - OVS_V2V_ENGINE=voxedge
      - OVS_V2V_SERVER_LOOP=1
      - EDGE_LLM_BASE_URL=http://edge-llm:8000/v1
      - EDGE_LLM_MODEL=${EDGE_LLM_MODEL:-qwen3}

  # ── Local LLM (OpenAI-compatible, Qwen3 on TensorRT) ──────────────────
  # Not built in this repo — pulled from registry.
  edge-llm:
    image: ${EDGE_LLM_IMAGE:-sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:qwen3-awq-orin-v2}
    container_name: edge-llm
    runtime: nvidia
    ipc: host
    ports:
      - "${EDGE_LLM_HOST_PORT:-8000}:8000"
    volumes:
      - /usr/local/cuda/lib64:/host-cuda:ro
      - /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro
      - /lib/aarch64-linux-gnu:/host-libs:ro
      - /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro
      - /usr/src/tensorrt:/usr/src/tensorrt:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 120s

  # ── Voice agent + robot arm (B601-DM) ──────────────────────────────────
  # Image built from agent/Dockerfile.rebot-arm (see header comment).
  voice-rebot-arm:
    image: ${AGENT_IMAGE:-voice-rebot-arm:dev}
    container_name: voice-rebot-arm
    user: "0"
    depends_on:
      speech:
        condition: service_healthy
      edge-llm:
        condition: service_started
    devices:
      - /dev/snd
      - ${REBOT_DEVICE:-/dev/ttyACM1}
    volumes:
      - /run/user:/run/user:ro
      - /dev/serial:/dev/serial:ro
    environment:
      VOICE_SERVICE_HOST: speech
      VOICE_SERVICE_PORT: "8000"
      LLM_SERVICE_HOST: edge-llm
      LLM_SERVICE_PORT: "8000"
      MIC_INDEX: ${MIC_INDEX:-reSpeaker}
      MIC_CHANNEL_SELECT: ${MIC_CHANNEL_SELECT:-0}
      WAKEWORD_MODEL: ${WAKEWORD_MODEL:-hey jarvis}
      REBOT_CHANNEL: ${REBOT_CHANNEL:-auto}
      OBSERVATION_PORT: ${OBSERVATION_PORT:-8775}
      SPEAKER_DEVICE: ${SPEAKER_DEVICE:-reSpeaker}
    ports:
      - "${OBSERVATION_PORT:-8775}:8775"
    restart: unless-stopped
```

关键改动说明：
- speech: 补上 `OVS_V2V_ENGINE=voxedge` + `OVS_V2V_SERVER_LOOP=1`（P0 #2）
- edge-llm: `depends_on` 改为 `condition: service_started`（避免 P2 #7 健康检查端点不确定的问题）
- voice-rebot-arm: 删除 `PULSE_SERVER`/`PULSE_CONFIG_DIR` 硬编码（P1 #4 #6），让 PortAudio 自行检测
- voice-rebot-arm: 添加 `SPEAKER_DEVICE: reSpeaker`（P1 #5）
- voice-rebot-arm: 删除不存在的 config 目录挂载

- [ ] **Step 2: Commit**

```bash
git add deploy/docker-compose.jetson-rebot.yml
git commit -m "fix(deploy): jetson-rebot overlay — server-loop vars, audio, edge-llm"
```

---

### Task 3: 在设备上验证 — 前置检查（不启动容器）

需要在 Jetson (seeed@192.168.1.176) 上执行。

- [ ] **Step 1: 确认 seeed-voice 是否已在跑**

```bash
ssh seeed@192.168.1.176
docker ps | grep -E "speech|seeed-voice"
```

期望看到 seeed-voice 容器 `Up (healthy)`。记下镜像 tag 和容器名。

- [ ] **Step 2: 确认 edge-llm 是否已在跑**

```bash
docker ps | grep edge-llm
```

如果在跑：记下镜像 tag。
如果不在跑：需要后续用 compose 拉起。

- [ ] **Step 3: 确认 B601-DM 串口设备**

```bash
ls -la /dev/serial/by-id/ | grep -i -E "damiao|b601|hdsc"
ls /dev/ttyACM*
```

记录实际设备节点（如 `/dev/ttyACM0` 或 `/dev/ttyACM1`）。

- [ ] **Step 4: 确认音频设备**

```bash
aplay -l | grep -i respeaker
arecord -l | grep -i respeaker
# PulseAudio 是否在跑？
pulseaudio --check && echo "PulseAudio running" || echo "PulseAudio NOT running"
ls /run/user/1000/pulse/native 2>/dev/null && echo "Pulse socket exists" || echo "No pulse socket"
```

如果 PulseAudio 在跑且 socket 存在：在 compose 里加回 `PULSE_SERVER`。
如果不在跑：保持当前配置（PortAudio 直接用 ALSA）。

- [ ] **Step 5: 确认 voxedge 是否在 seeed-voice 镜像里**

```bash
docker exec <seeed-voice容器名> python3 -c "import voxedge; print(voxedge.__file__)"
```

如果报 `ModuleNotFoundError`：当前镜像不支持 server-loop，需要更新镜像。

---

### Task 4: 在设备上构建 + 部署

- [ ] **Step 1: 拷代码到设备**（如果设备上没有最新代码）

```bash
# 从开发机
rsync -avz --exclude='.git' --exclude='agent/uv.lock' --exclude='__pycache__' \
  ./ seeed@192.168.1.176:/home/seeed/openvoicestream/
```

- [ ] **Step 2: 构建 agent 镜像**

```bash
ssh seeed@192.168.1.176
cd /home/seeed/openvoicestream
docker build -f agent/Dockerfile.rebot-arm -t voice-rebot-arm:dev .
```

首次构建约 10-15 分钟（克隆 SDK + pip install + 下载唤醒词模型）。

- [ ] **Step 3: 用 compose overlay 启动**

```bash
cd /home/seeed/openvoicestream
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.jetson-rebot.yml up -d
```

- [ ] **Step 4: 观察启动日志**

```bash
# 三个服务的日志
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.jetson-rebot.yml logs -f
```

期望看到：
- seeed-voice: `readyz` 返回 200，日志无报错
- edge-llm: 启动成功，加载模型
- voice-rebot-arm: 
  - `"ArmPlugin setup"` + `"actions.yaml loaded"` (NOT `"ArmPlugin disabled"`)
  - `"advertise_tools"` + 非空工具列表
  - `"SLV connected"` ws://speech:8000/v2v/stream

---

### Task 5: 端到端验证

- [ ] **Step 1: 验证唤醒词**

对着麦克风说 "Hey Jarvis"。日志应出现：
```
openwakeword: wake detected, score=0.XX
ConvState: SLEEPING → IDLE
```

- [ ] **Step 2: 验证 ASR**

唤醒后说一句话。日志应出现：
```
ASRPartial: ...
ASRFinal: "你说的话"
```

- [ ] **Step 3: 验证 LLM 工具调用**

唤醒后说 "挥个手" 或 "wave"。日志应出现：
```
ServerToolCall: name=wave, arguments={}
_spawn_tool_task: executing wave
RebotArmActuator: execute_sequence ...
```

如果看到 `"empty tools list"` 或没有 `ServerToolCall`：
- 检查 speech 容器的日志里 `OVS_V2V_SERVER_LOOP` 是否为 true
- 检查 agent 日志里 `ArmPlugin` 是否 disabled

- [ ] **Step 4: 验证 TTS 回话**

工具执行后日志应出现：
```
TTSStarted
TTSAudio: bytes=...
TTSDone
```

如果有 `TTSAudio` 但听不到声音：检查 `SPEAKER_DEVICE` 和音频输出设备。

---

## 风险和降级方案

| 风险 | 降级方案 |
|---|---|
| seeed-voice 镜像不含 voxedge → server-loop 启动崩溃 | 需要更新到 >= v1.12 的镜像（含 voxedge） |
| edge-llm 镜像不在 registry → 拉不下来 | 检查 `docker images` 是否本地已有 |
| B601-DM 不在 ttyACM1 → 串口连接失败 | `REBOT_CHANNEL=auto` 自动扫描；或手动指定 `REBOT_DEVICE` |
| 网络环境拉不到镜像（防火墙） | 用 `HF_ENDPOINT=https://hf-mirror.com` |
| PulseAudio 和 ALSA 设备冲突 | 统一：要么全走 Pulse（设 `PULSE_SERVER`），要么全走 ALSA（不设） |
