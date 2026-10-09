# Spec: 流式翻译双 app（实时字幕 + 同传）

状态：READY FOR IMPLEMENTATION。由 codex 设计、主线程裁决未决点后定稿（2026-06-02）。
执行体照此实施，按"实施顺序"分模块 commit，每步带验证。

## 背景与设计决策（已与用户敲定，不要推翻）

基于流式 ASR 的 partial/finalize 机制做实时翻译。新增**两个 app** + **一个共享层**。

- 翻译**不基于原始 partial 逐帧翻**，也**不纯等 final**；走中间态：partial 流上做"稳定提交"，对已稳定片段翻译。
- **实时字幕**：re-translation 策略（提交线左侧锁定、右侧 tail 可刷新），无 TTS，输出到 dashboard side-by-side。
- **同传**：monotonic 策略（译文绝不回收），输出 TTS。带 `overlap_mode` 开关（clause-lag 默认 / 全双工重叠 opt-in）。
- 前端走 agent（复用 debug_dashboard WS + `on_translation` 广播）。
- 翻译后端（NLLB CT2）全复用，不动。

## 锚点核查（codex 已只读核查，行号以此为准）

- barge-in 触发条件实际在 `agent/ovs_agent/app_base.py:1840`（`1844` 是 logger 闭合行）
- `_build_translator` `app_base.py:143`；ASRPartial 分支 `app_base.py:1792`；`_broadcast("on_user_partial")` `app_base.py:1857`；`on_user_utterance` hook `app_base.py:588`；final 广播 `app_base.py:2021`
- `AppMode.barge_in_enabled` `app_mode.py:484`；`produces_tts` `app_mode.py:490`；mode override 存储 `config.py:129-133`
- plugin hooks `plugin.py:57`(on_user_partial)/`:60`(on_user_utterance)
- translator config `config.py:210-218`；validate 区段 `config.py:283-306`
- app 加载 `cli.py:27-44`（`ovs_agent.apps.<name>.app:App`）；自注册范本 `apps/voice_arm/__init__.py`
- SLV client：`send_text` `slv_client.py:491` / `flush_tts` `:496` / `asr_eos` `:503` / `events` `:575`
- 翻译后端 ABC `translator/base.py`（`async translate(text, src, tgt)`）；NLLB 服务 `services/translator/`
- `on_translation` payload 范本 `modes/interpreter.py:102-108`（original/translated/src_lang/tgt_lang/detected_language）
- 转录 mode 范本 `modes/transcribe.py`（barge_in_enabled=False, produces_tts=False）
- baseline app 范本 `apps/translator/app.py:7-22` + 其 `config.yaml`
- dashboard：plugin `plugins/debug_dashboard.py`，静态资源挂载 `:90-124`，WS 路由 `:97-100`，前端 `plugins/static/dashboard.{html,js}`，on_translation 渲染入口 `dashboard.js:1002-1004`，WS 连接 `dashboard.js:1438-1448`
- 前端范本 `docs/asr-realtime-demo.html`（mic→SLV /asr/stream，partial 虚线气泡 `:150-176`，双态样式 `:341-370`）

## 1. 共享层 `agent/ovs_agent/streaming_translate/`

### `committer.py` — SegmentCommitter（纯逻辑、无 IO、易单测）

```
SegmentCommitter(
    agreement_n: int = 2,
    strategy: Literal["retranslation","monotonic"] = "retranslation",
    clause_punct: str = "。！？；,.?;",
    min_commit_chars: int = 1,
)
```

数据结构：
- `ASRChunk(text, is_final=False, detected_language=None, ts=None)`
- `SegmentEvent(seq, source_text, committed_source, tail_source, is_final, strategy, revision)`

方法：
- `push_partial(text, detected_language=None, ts=None) -> list[SegmentEvent]`
- `finalize(text, detected_language=None, ts=None) -> list[SegmentEvent]`
- `reset() -> None`

状态：`_partials`、`_committed_source`、`_tail_source`、`_seq`、`_revision`

逻辑：
- Local Agreement：连续 N 次 partial 的公共前缀一致 → 提交该前缀（扣除已提交部分）
- 子句边界：公共前缀命中 `clause_punct` → 只提交到最后一个边界
- final：强制提交剩余尾巴
- `retranslation`：同一 `revision` 可覆盖右侧 tail
- `monotonic`：事件一旦发出即终态，绝不产生回收事件

### `echo_filter.py` — EchoFilter（同传重叠模式的软件自回声兜底）

维护最近播出译文环形缓存 `(text, ts)`。相似度用 `difflib.SequenceMatcher(None, norm_partial, norm_tts).ratio()`（normalize：去空白/标点、小写）。命中条件：partial 是最近译文子串，或 ratio >= 阈值 → 丢弃该 partial，不进 committer。
默认：阈值 `0.82`、时间窗 `4.0s`、最小长度 `4` 字符（短于此直接放行）。
理由：XVF3800 硬件 AEC 漏前 200-500ms（见 `app_base.py:1816` 注释），AEC 扛 90%、软件兜底扛漏网。

## 2. barge-in gate 接通（主线程裁决：方案确定）

改 `agent/ovs_agent/app_base.py:1840` 的 barge-in 条件。在打断路径前解析有效开关：

读取顺序：当前 mode override（`config.py:129-133`）→ mode class default（`app_mode.py:484`）→ app config default（新增 `barge_in_enabled`，见 §6）。

**向后兼容硬约束：未配置时视为 `True`（维持当前行为），现有 app 行为零变化。**

解析为 `False` 时：仍执行 `app_base.py:1857` 的 partial 投递（广播 + 新虚方法，见 §3），但**跳过** `:1840-:1856` 的打断路径（不 set BARGED_IN、不 cancel LLM turn、不 interrupt）。

## 3. BaseApp 新增 partial 虚方法（主线程裁决：方案确定）

`BaseApp` 当前没有原生 partial 子类 hook，只有插件广播。**给 BaseApp 加一个虚方法**，与 `on_user_utterance`（`app_base.py:588`）对称：

```python
async def on_user_partial(self, text: str, detected_language: str | None = None) -> None:
    """ASR partial. Default: no-op. Apps override to consume streaming partials."""
    return None
```

在 `app_base.py:1857` 现有 `_broadcast("on_user_partial", evt.text)` **紧挨处**调用 `await self.on_user_partial(evt.text, ...)`（broadcast 保留，给插件；虚方法给 app 子类）。两个新 app override 它。

## 4. App 1 `agent/ovs_agent/apps/live_caption/`

`app.py`：继承 `BaseApp`（参考 `apps/translator/app.py:7-22`）。`SegmentCommitter(strategy="retranslation")`。
- override `on_user_partial`：text 进 committer，每个 `SegmentEvent` 翻译后 `_broadcast("on_translation", payload)`（payload 沿用 `modes/interpreter.py:102-108`）
- override `on_user_utterance`：调 `committer.finalize()`，完整上下文重译覆盖 tail
- 无 TTS，`llm_backend: noop`
- **debounce（主线程裁决：纳入）**：已提交 clause 立即翻译；仅对易变 tail 做 `translate_debounce_ms`（默认 250ms）节流，避免狂打 NLLB

`config.yaml`：参考 `apps/translator/config.yaml`，含 `llm_backend: noop`、translator 字段、`committer_agreement_n`、`committer_min_commit_chars`、`translate_debounce_ms`。
`__init__.py`：导出 `App`。

## 5. App 2 `agent/ovs_agent/apps/simul_interpret/`

`app.py`：继承 `BaseApp`，`SegmentCommitter(strategy="monotonic")`。
- override `on_user_partial`：先过 `EchoFilter`（若 `echo_filter_enabled`）→ committer；每个事件翻译后 `await slv.send_text()`（`slv_client.py:491`）+ `await slv.flush_tts()`（`:496`）
- `overlap_mode: "off"`（默认 clause-lag）：仅 is_final/is_clause 事件播
- `overlap_mode: "on"`（全双工重叠）：partial 无条件投递；标注需 AEC 设备/耳机
- `barge_in_enabled: false`
- 译文播出时把文本喂给 EchoFilter 缓存
- 无 LLM

`config.yaml`：含 `overlap_mode`、`echo_filter_enabled`、`echo_similarity_threshold`、`echo_window_s`、`barge_in_enabled: false`、committer 字段、translator 字段、`translate_debounce_ms`。
`__init__.py`：导出 `App`。

## 6. config.py 新增字段 + 校验

在 `config.py:210-218` translator 字段之后新增：
- `committer_agreement_n: int = 2`
- `committer_min_commit_chars: int = 1`
- `translate_debounce_ms: int = 250`
- `overlap_mode: str = "off"`
- `echo_filter_enabled: bool = True`
- `echo_similarity_threshold: float = 0.82`
- `echo_window_s: float = 4.0`
- `barge_in_enabled: bool | None = None`（None = 维持现状 True；app 显式 false 才关）

校验加在 `config.py:283-306`：`committer_agreement_n >= 1`、`0 <= echo_similarity_threshold <= 1`、`echo_window_s > 0`、`overlap_mode in {"off","on"}`、`translate_debounce_ms >= 0`。

## 7. 前端

新页面 `agent/ovs_agent/plugins/static/live_caption.html` + `live_caption.js`：复制改造 `docs/asr-realtime-demo.html:150-176`（保留 Seeed 绿配色 + 气泡双态 `:341-370`）。dashboard plugin 增静态路由（核实 `debug_dashboard.py:90-124` 的挂载方式）。
WS 连接参考 `dashboard.js:1438-1448`。事件复用：`on_user_partial`（partial 气泡刷新）、`on_translation`（原文+译文 side-by-side，渲染入口 `dashboard.js:1002-1004`）。

## 8. 单测 `agent/ovs_agent/tests/test_streaming_translate.py`

- `test_committer_agreement`：N=2 相同前缀触发提交
- `test_committer_clause`：单次 partial 命中标点立即提交
- `test_committer_final_flush`：final 强制提交尾巴
- `test_committer_retranslation_tail_refresh`：同 revision 覆盖 tail
- `test_committer_monotonic_no_revoke`：monotonic 不发回收事件
- `test_barge_in_gate_disabled`：`barge_in_enabled=False` 时 partial 投递正常但打断不触发
- `test_echo_filter_high_similarity`：窗口内高相似丢弃
- `test_echo_filter_window_expiry`：超时放行
- `test_echo_filter_short_text`：短于 min_len 放行

## 9. 实施顺序（按可独立验证模块拆 commit）

| 步 | 内容 | 验证 |
|---|---|---|
| 1 | `streaming_translate/committer.py` + `echo_filter.py` + 单测 | `uv run pytest agent/ovs_agent/tests/test_streaming_translate.py` |
| 2 | `app_base.py` barge-in gate 接通 + `on_user_partial` 虚方法 + gate 单测 | gate 测试 PASS；现有 app 回归不变（跑现有 test_server_loop_integration 等） |
| 3 | config.py 新字段 + 校验 + `apps/live_caption/`（app.py/config.yaml/__init__.py） | 连 dashboard WS，partial → 收到 on_translation 事件 |
| 4 | `apps/simul_interpret/`（app.py/config.yaml/__init__.py） | send_text/flush_tts 调用日志可见；overlap on/off 行为区分 |
| 5 | `live_caption.html/js` + dashboard 静态路由 | 浏览器访问字幕页，side-by-side 渲染 partial + 译文 |

## 10. 风险/未决点

- 字幕页静态路由方式需执行体核实 `debug_dashboard.py:90-124` 资源挂载实现
- NLLB partial 频繁调用抖动 → 已用 `translate_debounce_ms` 缓解（§4）
- `echo_filter` difflib 在高 partial 频率下有轻微 CPU 开销，足够，无需外部依赖
- 真实 e2e（真机 ASR partial 时序、overlap 模式回声实测）需在边缘设备上验证，单测只覆盖逻辑层
