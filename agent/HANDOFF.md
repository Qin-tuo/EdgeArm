# OpenVoiceStream Agent — 交接文档

最近一次实机调试：2026-05-19，Mac 本机 + Orin NX 远端 SLV + edge-llm。
main 分支已 push origin（HEAD `d936d2d`）。

## 2026-05-19 SLV 侧重大变更（per-utterance ASR）

SLV `/v2v/stream` 适配层重写为 per-utterance ASR session 模型，修了三个连锁 race：

| Commit | 修复 |
|---|---|
| `1e123a3` 及之前 | per-utterance ASRSessionManager 框架 + 多轮 begin/end 协议 |
| `e2a3166` | BrokenPipeError → WorkerExitError 分类，让 SIGKILL 触发 restart_worker |
| `0c9972b` | endpoint_pending gen-stamp，防 stale VAD endpoint 触发错 utterance 的 finalize |
| `d936d2d` | manager.finalize_with_status 返回 accepted flag，丢掉的 finalize 不再 emit 空 asr_final |

**Orin NX 上线镜像**：`seeed-local-voice:jetson-v1.12-highperf-perutt-20260519c`
**回滚镜像**：`perutt-20260519b` / `perutt-20260519` / `bargein-asrfix-20260518` 均保留
**回滚命令**：`fleet exec orin-nx -- "cd /tmp/seeed-local-voice-release/deploy && sed -i 's|perutt-20260519c|perutt-20260519b|g' docker-compose.yml && docker compose -p seeed-local-voice-latest up -d speech"`

**对 agent 侧的含义**：之前 `asr_final watchdog`（commit `6cf7b77`）是为应对 SLV always_on 偶发丢空 final 设的兜底；per-utterance 落地后 SLV 不再发空 final（除非真的没有 ASR 文本），watchdog 还在但触发频率应明显下降。如果再观察到"agent 收不到 final 卡住"应优先怀疑 SLV 镜像版本而不是 agent 状态机。

---

## 1. 代码在哪里

| 仓库 | 路径 | 说明 |
|---|---|---|
| **agent**（本项目） | `/Users/harvest/project/seeed-local-voice/agent/` | Mac 本机跑，对话/状态机/dashboard |
| **edge-llm wrapper** | `/Users/harvest/project/edge-llm-chat-service/` | 包 TensorRT-Edge-LLM，HTTP 服务，docker 部署 Orin |
| **SLV (上游)** | `/Users/harvest/project/seeed-local-voice/` 父目录 | ASR/TTS WebSocket 服务，**不要碰它的 app/ 或 deploy/** |
| **TRT-Edge-LLM (上游 fork)** | `/Users/harvest/project/tensorrt-edge-llm/` | NVIDIA 引擎 + pybind，分支 `highperf/runtime-service` |

### Agent 关键文件

```
agent/ovs_agent/
├── app_base.py           # BaseApp orchestrator（状态机 / VAD / barge-in / watchdog）
├── app_mode.py           # AppMode strategy（chat/interpreter/monologue/transcribe）
├── session.py            # Session（token-aware trim、prefix_cache_disabled）
├── config.py             # Config dataclass + load_config(yaml)
├── slv_client.py         # SLV WebSocket 客户端（asr_eos / send_text / reconnect）
├── state.py              # ConvState 枚举
├── llm/
│   ├── openai_compat.py  # OpenAI 兼容 backend（含 A3 retry + SSE error detect）
│   └── edge_llm.py       # EdgeLLM 专用（prefix_cache fallback / A4 latch）
└── plugins/
    ├── debug_dashboard.py     # Dashboard backend (aiohttp WS)
    ├── llm_availability.py    # A1+A5 探活 + 5态机（含 UNKNOWN）
    └── static/{html,css,js}   # Dashboard 前端

agent/apps/multi_mode/
├── app.py                # MultiModeApp 入口
└── config.yaml           # 默认配置（SLV URL / LLM / VAD / system prompt）

agent/tests/               # 197 个 unit + non-Orin e2e 全通
agent/tests/e2e/           # 含 mock LLM resilience 套 + 真 Orin 套
```

### edge-llm 关键文件

```
edge-llm-chat-service/
├── edge_llm_chat_service/
│   ├── server.py         # serve()：上游 app + guard + structured errors
│   ├── guard.py          # input length guard + ASGI middleware + request_id
│   └── config.py
├── deploy/edge-llm/
│   ├── Dockerfile        # multi-stage，~127 MB（host mount CUDA/TRT）
│   ├── docker-compose.yml
│   ├── entrypoint.sh     # preflight + 自动下 engine + warmup + exec
│   ├── build.sh          # 唯一合法 build 入口
│   ├── healthcheck.sh
│   └── build-qwen3-engine-host.sh  # Orin host 上重 build engine
├── tests/                # 41 测试全通
└── DOCKERFILE_PLAN.md    # 设计 + 验证笔记
```

---

## 2. 怎么启动 Dashboard（Mac 本机）

### 前置
- Orin (`orin-nx`, Tailscale IP **100.82.225.102**) 上 SLV + edge-llm 都 healthy
  - SLV: `ws://100.82.225.102:8621/v2v/stream`（容器 `seeed-local-voice-latest-speech-1`）
  - edge-llm: `http://100.82.225.102:8000`（容器 `edge-llm-chat-service`，镜像 `qwen3-awq-orin-v2`）
- 检查命令：`fleet exec orin-nx -- 'docker ps | grep -E "edge-llm|speech"'`
  - 期望两个都 `Up (healthy)`

### 启动命令

```bash
cd /Users/harvest/project/seeed-local-voice/agent

NO_PROXY='100.82.225.102,localhost,127.0.0.1' \
no_proxy='100.82.225.102,localhost,127.0.0.1' \
OVS_SLV_URL='ws://100.82.225.102:8621/v2v/stream' \
OVS_LLM_URL='http://100.82.225.102:8000/v1' \
OVS_LLM_MODEL='Qwen/Qwen3-4B-AWQ' \
uv run ovs-agent run multi_mode
```

后台跑：在前面加 `nohup ... > /tmp/ovs-agent.log 2>&1 &`，看日志 `tail -f /tmp/ovs-agent.log`。

### Dashboard 地址
- **http://localhost:18000**
- 左半屏：对话气泡 / 事件流 / LLM 上下文
- 右半屏：SLV WS 状态 / latency / mic RMS / TTS / errors（带 type 分色）/ **LLM 健康** / AGENT 设置
- 顶部：思考状态 pill / mode 切换 / 设置 / 重连SLV / 中止TTS / **跳过麦克风直接打字** / 发送

### 关键 env / config 速查

| 变量 / 配置 | 默认 | 说明 |
|---|---|---|
| `NO_PROXY` | — | **必设**，否则 websockets 走 SOCKS proxy 崩 |
| `OVS_SLV_URL` | localhost:8621 | SLV WebSocket |
| `OVS_LLM_URL` | localhost:8000/v1 | edge-llm OpenAI API |
| `OVS_LLM_MODEL` | qwen2.5-3b-instruct | 模型名（要跟 edge-llm 一致） |
| `client_vad_threshold` (yaml) | **0.03（临时调高）** | Mac 内置麦默认 0.005 会被环境噪声触发死循环；TEMP 标注在 yaml 里 |
| `asr_final_timeout_s` (Config) | 3.0 | watchdog 超时 |

---

## 3. 已知问题 / 遗留待办

### A. LLM 回答太短 — 系统提示限制
**症状：** 让它讲故事只说"好啊给你讲个 X 的故事"，不真讲。

**根因：** `apps/multi_mode/config.yaml` system_prompt 写死了"控制在两三句话以内"。

**修法：** 改 yaml 的 system_prompt 加一段"用户要求详细/讲故事时不限句数"，或 Dashboard 右下角 AGENT 设置面板里改。

### B. Mac 内置麦 VAD 阈值临时调高了
**位置：** `apps/multi_mode/config.yaml` 第 22 行 `client_vad_threshold: 0.03`（原值 0.005，注释里有 TEMP 标记）

**TODO：**
1. **不要 commit 0.03**（已超 demo 期了再 revert）
2. 长期方案：给 yaml 加 `${OVS_VAD_THRESHOLD:-0.005}` env 覆盖，但需要在 `Config.__post_init__` 加 float 类型 coerce（yaml `_expand_env` 出来是 str）

### C. ASR 偶发卡死的链路 — 已基本解决但仍有可能
路径上有 3 重保护，按触发顺序：
1. `_eos_sent_this_turn` flag 在 ASRFinal handler **所有 path 入口** reset（commit `8afcf9d`）—— 防 duplicate_of_streamed / 空 final 漏 reset
2. `asr_final_timeout_s` watchdog（commit `6cf7b77`）—— SLV 不发空 final 时 3s 后强制回 IDLE
3. `mic_pump on_mic_rms` 限频 200ms（commit `b048381`）—— 防 dashboard 慢导致 mic 队列爆满丢 chunk

如果仍卡死，看 `tail /tmp/ovs-agent.log`，注意：
- `mic queue full -- dropping chunk` 大量 → mic 消费者又慢了，检查 dashboard WS 客户端
- `asr_final not received within 3.0s` → 正常，watchdog 在工作
- `send_json: WS closed mid-send` → SLV 重连竞态，watchdog 兜底

### D. Barge-in TTS 余音浪费
barge-in 路径不再调 `slv.abort()`（commit `fa13846`），SLV 会继续把已缓冲的 TTS 音频流过来，客户端 stop_playback 后丢弃。代价：浪费几百 ms 带宽，换 ASR 文字完整。

### E. transformers 没装 → token 计数走 ceil(chars*1.5) fallback
agent venv 和 edge-llm runtime image 都没装 `transformers`（避免拉 ~1GB PyTorch）。

edge-llm 这边 H1 修复后用 `tokenizers` rust lib（~10MB）+ engine_dir/tokenizer.json 直接精确算（commit `12364da`，已部署 `qwen3-awq-orin-v2`）。

agent 这边 `session.py` 仍走 char fallback —— 中文准确，英文偏保守。如果要精确，agent venv `uv add tokenizers` 然后改 `session.py` 走 tokenizers 路径（参考 edge-llm guard.py）。

### F. 4 个 Codex 评审标的 LOW（hypothetical / future-ops）
1. dashboard plugin double-start（已加 `_started` flag 修了，可关）
2. probe 拿 guard 400 input_too_long 误判（已修，分类为 UNKNOWN）
3. H3 upstream shim 全失败没监控告警（启动会 sys.exit(1)，docker 重启 loop，足够）
4. 测试覆盖盲区（双失败 / 并发 probe / dashboard 并发）

### G. 待 push GitHub
- `seeed-local-voice/agent`：main ahead origin/main ~50 commits（A1-A6+A7+H1-H3+MED+F1+watchdog+barge-in 全在里面）
- `edge-llm-chat-service`：本地 repo，**还没远端**，需要建 GitHub repo
- 你说要先改 README 再 push

### H. Agent 端那个 batch commit `c0299bf` 没拆
A1+A2+A3+A4+A5+A6 + H2-session 全一锅。要拆需要 `git rebase -i c0299bf~1`（系统规则不让我自动跑 `-i`），你自己来。

---

## 4. 远端 Orin 部署速查

### 现状
- Orin: `orin-nx`（`fleet list`），IP 100.82.225.102，用户 harvest
- edge-llm 镜像：`sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:qwen3-awq-orin-v2`（已 push，digest sha256:963ba8b...）
- engine：`/home/harvest/edgellm-workspace/Qwen3-4B-AWQ/engines-3072/`
- compose：`/home/harvest/edge-llm-chat-service/deploy/edge-llm/docker-compose.yml`

### 同步本机最新代码并重 build

```bash
# Mac 本机
fleet push orin-nx /Users/harvest/project/edge-llm-chat-service /home/harvest/edge-llm-chat-service
fleet exec orin-nx -- 'cd /home/harvest/edge-llm-chat-service && bash ./deploy/edge-llm/build.sh > /tmp/build.log 2>&1; tail -5 /tmp/build.log'
fleet exec orin-nx -- 'cd /home/harvest/edge-llm-chat-service/deploy/edge-llm && docker compose up -d'
```

### 看 edge-llm 日志
```bash
fleet exec orin-nx -- 'docker logs --tail 50 edge-llm-chat-service 2>&1 | grep -vE "Initializing plugin|MS\]"'
```

### 健康检查
```bash
curl -s http://100.82.225.102:8000/v1/models | python3 -m json.tool
curl -s -X POST http://100.82.225.102:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-AWQ","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
```

### 同 host 上其他容器（不要碰）
- `reachy-daemon`（reachy 机器人控制）
- `seeed-local-voice-latest-speech-1`（SLV 容器）

---

## 5. 防御性 hardening 一图速览

```
用户 utterance
    ↓
[client VAD] ─ noise rate-limit on_mic_rms broadcast ─ unblocks barge-in
    ↓
[SLV ASR] ─ asr_eos
    ↓
   ┌── ASRFinal 到 → reset _eos_sent flag (所有 path) → dispatch
   │
   └── 3s 不到 → watchdog 强制回 IDLE
    ↓
[LLMAvailability 5态机]
   HEALTHY / DEGRADED / DOWN / RECOVERING / UNKNOWN
   ConnectError → UNKNOWN（不算 fail）
   /v1/chat/completions probe（不是 /v1/models）
   DOWN 或 UNKNOWN → fail-fast 1ms 内抛 LLMUnavailable
    ↓
[Session]
   token-aware trim（max=3000，ceil×1.5 fallback）
   trim 时清 cache_warmed
    ↓
[A3 OpenAICompatBackend.stream]
   retry 1× / 0.5s backoff，只在第一帧前 retry
   SSE finish_reason=error → LLMStreamError（不 retry）
    ↓
[A4 EdgeLLMBackend.stream]
   prefix_cache 失败 → set prefix_cache_disabled latch → retry once（_retry_disabled=True 防双 retry）
    ↓
[edge-llm wrapper（Orin）]
   guard middleware：tokenizers rust lib + engine template → 400 input_too_long
   global structured error handler：所有错误 → {"error":{"code","message","context":{"request_id"}}}
   per-request X-Request-Id header + [req=xxx] log
   entrypoint preflight：build_provenance.txt 校验 tensor_rt + git_commit
   B2 warmup：真小推理通过才开 traffic
    ↓
[Dashboard 渲染]
   LLM 健康卡（5 状态 + 重新探测按钮）
   typed on_error 分色（红/橘/黄/灰）
   late client snapshot 含当前 LLM 健康状态
```

---

## 6. 测试速查

```bash
# Agent 全套（非 Orin 依赖）
cd /Users/harvest/project/seeed-local-voice/agent
uv run pytest tests/ \
  --ignore=tests/e2e/test_single_turn.py --ignore=tests/e2e/test_multi_turn.py \
  --ignore=tests/e2e/test_barge_in.py --ignore=tests/e2e/test_wake_word.py \
  --ignore=tests/e2e/test_stop_intent.py --ignore=tests/e2e/test_empty_final.py \
  --ignore=tests/e2e/test_reconnect.py --ignore=tests/e2e/test_idle_stability.py \
  --ignore=tests/test_e2e_orin.py
# 期望：197 passed, 1 skipped

# Agent 真 Orin e2e（需 SLV + edge-llm 都 healthy）
uv run pytest tests/e2e/test_{single_turn,multi_turn,barge_in,wake_word,stop_intent,empty_final,reconnect,idle_stability}.py -v
# 期望：9 passed, 1 skipped (test_stop_en_matches 是 ASR fixture 限制)

# edge-llm 全套
cd /Users/harvest/project/edge-llm-chat-service
uv run --extra tokenizers pytest tests/ -v
# 期望：41 passed
```

---

## 7. 关键 commit 索引

| 范围 | Commit | 说明 |
|---|---|---|
| Watchdog | `6cf7b77` | asr_final 超时 3s 强制 IDLE |
| EOS flag fix | `8afcf9d` | reset _eos_sent_this_turn on **所有** ASRFinal path |
| Mic queue | `b048381` | rate-limit on_mic_rms 防 barge-in 漏 |
| Barge-in fix | `fa13846` + `faf234d` | 不调 slv.abort，保 ASR 完整 |
| Dashboard | `8ee5e85` | typed on_error + 分色 + double-start guard |
| Availability | `47eb060` + `67cf0aa` | probe-400 分类 / UNKNOWN 状态 / F1 ConnectError |
| URL flap | `b49abcc` | probe 双 /v1 → 404 fix + char/4 log 改 char×1.5 |
| edge-llm guard | `12364da` | tokenizers rust lib 优先（精确 token 计数） |
| edge-llm B5+B6 | `6bc4b16` | 全局 structured error + request_id |
| edge-llm deploy | `b73c9a7` / `6a5beb8` / `b81b1a6` | engine guard / warmup / shm_size+regex |

完整：`git log --oneline origin/main..HEAD`（agent repo）+ `git log --oneline`（edge-llm repo）。
