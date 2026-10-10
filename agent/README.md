# SO-101 语音 Agent

基于 OpenVoiceStream，共享运行时位于 `ovs_agent/app_base.py` 和 `runtime.py`。唯一入口 `edgearm run edgearm`；默认配置在 `ovs_agent/apps/edgearm/config.yaml`，板端配置在 `deploy/agent-config.yaml`。

复用一个持续 WebSocket 会话，直接流式转发 LLM 文本到服务端 TTS，保留会话预算、音频热插拔和工具执行机制。应用只注册 ChatMode。机器人仅开放查询、抓放和取消工具，本地人工使能使用独立 Unix socket。

运行/标定见 [手册](../docs/runbook.md)，接口与未验证项见 [架构](../ARCHITECTURE.md)。
