> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

# Concurrency Capability Framework

## 1. **ConcurrencyCapability data structure**

`ConcurrencyCapability` describes runtime concurrency, separate from feature capabilities such as `STREAMING`. 注意：`supports_parallel` 与 `supports_hot_reload` (见 `server/core/asr_backend.py:96` / `tts_backend.py:39`，`b85aeeb` 引入) 是**正交维度**——可并发 ≠ 可热 reload，反之亦然。Fields:

| field | valid values | default |
| --- | --- | --- |
| `supports_parallel` | `bool` | `False` |
| `max_concurrent` | integer `>=1`，或 `None` 表示**无硬上限**（实际由 VRAM/profile clamp 决定）。`None` 参与 `min()` 聚合时按 `+∞` 处理 | `1` |
| `is_stateful` | `bool` | `True` |
| `requires_exclusive_device` | `bool` — 语义：**跨 backend 互斥**（同一设备不能并存别的 backend 实例），不是"禁止同一 backend 内多 slot"。后者由 `supports_parallel`+`max_concurrent` 表达 | `True` for GPU/NPU, `False` for CPU |
| `scaling_mode` | `single_runtime_multiplex`, `multi_runtime_per_slot`, `per_call_isolated`, `external_managed` | `single_runtime_multiplex` |
| `vram_mb_per_slot` | `int` or `None` | `None` |

Sample declarations for current backends:

| backend | capability | evidence |
| --- | --- | --- |
| `trt_edge_llm_tts` | `supports_parallel=True`, `max_concurrent=OVS_TTS_WORKER_CONCURRENCY`, `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=single_runtime_multiplex` | `_WorkerIO` multiplexes N in-flight requests with per-request queues and a semaphore at `app/backends/jetson/trt_edge_llm_tts.py:486`; env concurrency is read at `app/backends/jetson/trt_edge_llm_tts.py:662` and refreshed before worker creation at `app/backends/jetson/trt_edge_llm_tts.py:935`; streaming no longer holds `_worker_lock` across chunks at `app/backends/jetson/trt_edge_llm_tts.py:1200`. |
| `trt_edge_llm_asr` | `supports_parallel=False`, `max_concurrent=1`, `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=single_runtime_multiplex` | worker mode is hot-reloadable but all worker requests are serialized by `_worker_lock` at `app/backends/jetson/trt_edge_llm_asr.py:147`, `app/backends/jetson/trt_edge_llm_asr.py:460`, and `app/backends/jetson/trt_edge_llm_asr.py:593`. |
| `kokoro_trt` | `supports_parallel=True`, `max_concurrent=K` (pre-allocated pool size, default 2), `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=single_runtime_multiplex` | **Updated 2026-05-25**: commits `a49a478` + `5eb67e9` 把 shared `self._ctx`/`self._split_ctxs`/`self._pool` 改成 shared engine weights + K 个 pre-allocated `_KokoroCtxSlot` (synthesize() 从队列借/还)。orin-nano N=2 实测 TTFA p50 3656→576ms, CUDA error 0, MD5 byte-identical (commit `5eb67e9` body)。Pool 大小是 `max_concurrent` 来源。 |
| `matcha_trt` | `supports_parallel=True`, `max_concurrent=K` (pre-allocated pool size, default 2), `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=single_runtime_multiplex` | **Updated 2026-05-25**: commits `a49a478` + `493776e` 同样改造。Drops shared `self._vocos_ctx`/`self._split_estimator_ctxs`/`self._cuda_pool`，改 per-call (later pool) 拿 context。Engines (weights) 仍共享。 |
| `qwen3_trt` | `supports_parallel=False`, `max_concurrent=1`, `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=single_runtime_multiplex` | registry entry exists at `server/core/tts_backend.py:134`; resident pybind pipeline at `app/backends/jetson/qwen3_trt.py:242`，unload 注释 C++ pipeline 无 close API at `app/backends/jetson/qwen3_trt.py:269`。**未做 N>=2 safety 改造**，保持 N=1。 |
| `paraformer_trt` | `supports_parallel=True`, `max_concurrent=None` (无硬上限，per-stream bundle 按 stream 数扩，受 VRAM 约束；聚合时按 +∞), `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=multi_runtime_per_slot` | **Updated 2026-05-25**: commit `e1e5424` 引入 `_ParaformerCtxBundle`，每个 stream 独占 enc/dec contexts + buffer cache（shape-keyed reuse 保留在 stream 内）。Backend 只剩 shared engines。配套 `ASRStream.close()`/`__del__` 在 `server/core/asr_backend.py:9` 释放 bundle (`e564328`)。语义上是 per-slot independent context，不是 in-flight multiplex。 |
| RK ASR/TTS | `supports_parallel=False`, `max_concurrent=1`, `is_stateful=True`, `requires_exclusive_device=True`, `scaling_mode=external_managed` | adapters delegate to `rkvoice_stream` at `app/backends/rk/asr.py:321` and `app/backends/rk/tts.py:35`; RK TTS says NPU teardown belongs to that repo at `app/backends/rk/tts.py:79`, and RK ASR mirrors that hot-reload limitation at `app/backends/rk/asr.py:380`. |
| desktop/CPU ASR/TTS | `supports_parallel=True`, `max_concurrent=4`, `is_stateful=True`, `requires_exclusive_device=False`, `scaling_mode=external_managed` | current target defaults give desktop `4` at `server/core/session_limiter.py:30`; CPU TTS is hot-reloadable and ORT-backed at `app/backends/cpu/sherpa.py:72`; CPU ASR is hot-reloadable and stores recognizer objects at `app/backends/cpu/sherpa_asr.py:190`. |

## 2. **ABC extension**

Add the descriptor as a classmethod, not an instance property, because scheduler ceilings should be available before expensive `preload()` and should not depend on mutable readiness. Existing feature `capabilities` remain instance properties because they can reflect loaded recognizers, as Sherpa ASR does at `app/backends/cpu/sherpa_asr.py:204`.

```python
@classmethod
def concurrency_capability(cls, profile: dict | None = None) -> ConcurrencyCapability: ...
```

Add that signature to `ASRBackend` and `TTSBackend`. `server/core/llm_backend.py` does not exist; `rg --files server/core | rg 'llm|backend'` only found `backend_manager.py`, `asr_backend.py`, and `tts_backend.py`. If an LLM ABC is added in P1, give it the same classmethod signature.

## 3. **session_limiter refactor**

Replace target-derived defaults at `server/core/session_limiter.py:30` with backend-derived ceilings. Use one global session gate and compute `limit = min(asr.max_concurrent, tts.max_concurrent)`, because a voice session consumes both ASR and TTS in the same end-user workflow, and the current limiter is global/reject-not-queue at `server/core/session_limiter.py:1`. Per-modality queues would change admission behavior and should remain out of scope for this refactor.

Compatibility shim: if a backend lacks `concurrency_capability`, synthesize `ConcurrencyCapability(False, 1, True, True, single_runtime_multiplex, None)`. This preserves safety for older backends and matches the unknown fallback `1` at `server/core/session_limiter.py:36`. Profile overrides may only downgrade: `effective = min(backend_ceiling, profile.max_concurrent_sessions, env_override)`. Upgrades are forbidden; if profile or env exceeds a backend ceiling, log and clamp, because current profile/env precedence allows direct override at `server/core/session_limiter.py:88` and `server/core/session_limiter.py:99`.

## 4. **execution_policy ↔ capability coordination**

Backend capability is the ceiling; profile `execution_policy` lowers the coordinator mode. `concurrent` may be used only when both 活动 backend 的 `supports_parallel=True` 且 `max_concurrent > 1`。降级规则：**只要任一 backend `supports_parallel=False`，resolve 为 `serialized`**（不再用 `requires_exclusive_device` 作为降级触发——后者只描述跨 backend 互斥，不否决同 backend 多 slot）。如 profile 显式 `exclusive`，保持 `exclusive`。The current coordinator takes `policy.get("mode", "concurrent")` directly at `server/core/coordinator.py:20`, creates a lock for `serialized`/`exclusive` at `server/core/coordinator.py:22`, bypasses the lock in `concurrent` at `server/core/coordinator.py:38`, and unloads the previous slot only in `exclusive` at `server/core/coordinator.py:42`. Change only this resolution logic before `BackendCoordinator` is constructed.

## 5. **Generic worker_io abstraction**

Extract the existing TRT TTS `_WorkerIO` to `server/core/worker_io.py`; the current implementation starts at `app/backends/jetson/trt_edge_llm_tts.py:486`. Public API:

```python
def __init__(self, proc, max_concurrent: int): ...
async def send_request(self, request_id: str, payload: dict) -> AsyncIterator[dict]: ...
def cancel(self, request_id: str) -> None: ...
def close(self) -> None: ...
```

The abstraction keeps one stdin lock, one stdout reader, an in-flight map keyed by `request_id`, and a semaphore. Existing TTS behavior already inserts the queue before stdin write at `app/backends/jetson/trt_edge_llm_tts.py:537`, writes under `_stdin_lock` at `app/backends/jetson/trt_edge_llm_tts.py:545`, yields until `done`/`cancelled` at `app/backends/jetson/trt_edge_llm_tts.py:552`, and cancels by writing `{"type":"cancel","id":...}` at `app/backends/jetson/trt_edge_llm_tts.py:567`.

ASR streaming:

```python
rid = new_id()
async for event in io.send_request(rid, {"id": rid, "event": "decode", "audio": b64}):
    if event["event"] == "partial": yield event["text"]
    if event["event"] == "final":
        # ASRStream.finalize() 协议（commit 4d3450a）返回 (text, detected_language)
        return event["text"], event.get("language")
```

TTS streaming:

```python
rid = new_id()
async for event in io.send_request(rid, {"id": rid, "stream": True, "text": text}):
    if event["event"] == "audio": yield event["pcm"]
    if event["event"] == "done": return
```

**Executor cap 协调（`c704589` 风险点）**: `server/main.py` 中 `tts_stream_executor` 的 `max_workers` 与 backend `max_concurrent` 必须对齐——历史曾因 lazy startup 解析顺序错误导致永久使用 global default (`c704589`)。worker_io 的 semaphore 是 backend 层 ceiling，executor cap 是 HTTP 层 ceiling，两者**必须取自同一 capability 来源**。建议：limiter resolve 时一并产出 `executor_max_workers` 供 main 装配，避免两处独立计算漂移。

## 6. **Rollout phases**

**P0 (纯框架，零行为变化)**: add `ConcurrencyCapability` and ABC classmethod defaults in `server/core/asr_backend.py` and `server/core/tts_backend.py`; optionally add `server/core/concurrency_capability.py`。**各 backend 填能力声明但 `session_limiter` 仍走旧 target default 路径**，limiter 不读 capability。Verify with existing backend factory imports at `server/core/asr_backend.py:126` and `server/core/tts_backend.py:130`; rollback by removing the new descriptor and classmethods.

**P1 (切换聚合 + worker_io 抽象)**: extract `server/core/worker_io.py`, migrate TRT TTS to import it, and refactor `server/core/session_limiter.py` to read from capability + policy-resolution wiring near `server/main.py:511`、`tts_stream_executor` cap 改读 capability (`c704589` 风险点)。
**预期行为变化**: matcha/kokoro/paraformer 声明 N≥2 后，orin-nano profile (target default 当前=1) 的有效会话上限从 1 升到 2（`min(asr=∞, tts=2)=2`）——这是 commits `a49a478`/`5eb67e9`/`493776e`/`e1e5424` 已经做出来的 N≥2 safety 收益的落地，真机验过（CUDA error 0, MD5 byte-identical, `5eb67e9` body）。orin-nx (=2) / RK (=1, external_managed) 不变。想保守 profile 可写 `max_concurrent_sessions: 1` 走 clamp 路径降回。Verify TTS worker concurrency tests + orin-nano N=2 真机冒烟 (`bench/perf/load_2client_tts.py`); rollback by restoring local `_WorkerIO` and old `resolve_limit()`.

**P2 (VRAM budget 占位)**: add VRAM-aware scheduling placeholders using `vram_mb_per_slot` only as metadata; no hard enforcement until measured budgets exist. Touch `server/core/session_limiter.py` and health reporting near `server/main.py:882`. Verify health payloads and conservative default behavior. Rollback by ignoring the optional VRAM field.

Capability defaults are conservative: undeclared backends resolve to `max_concurrent=1`。P0 要求 ABC 加签名 + 已知 N≥2-safe backend (matcha/kokoro/paraformer + trt_edge_llm_tts) 填能力声明；其余 backend (qwen3_trt / moss_tts_nano / RK / CPU) 未填时走默认值，行为不变。

## 7. **决策 (2026-05-25)**

- **Profile/env 超 ceiling**: warn + 静默 clamp。`session_limiter` 在 resolve 时打 `logger.warning` 标明哪一层（profile/env）被 clamp 到 backend ceiling，运行时按 ceiling 跑。理由：profile 跨设备复用是常态，hard fail 会让同一 profile 在 orin-nx vs orin-nano 必须分叉。
- **LLM 调度独立**: 未来 LLM ABC 不并入 voice-session limiter，单独 token/slot 调度器。理由：LLM 可能跨 session 复用（共享 KV cache / 批处理），与 ASR/TTS 的"一会话一槽位"语义不同。P0/P1 不实现，仅在 ABC 设计时预留 hook，不强行约束。
- **CI 验证分层**: 单元 mock 测 limiter 聚合数学和 worker_io demux 协议正确性（pytest，快、确定）；真机冒烟在 orin-nx 跑 TTS N=2（`bench/perf/load_2client_tts.py` 已存在），ASR N>1 等实施后补类似 bench。理由：memory `tts_n2_phase_b_stability_landed` 显示纯 mock 抓不到 C++ race，必须真机；但 limiter 聚合逻辑可纯单元完全覆盖。

## 8. **Open questions /留白**

- How should P2 track a process-wide VRAM budget across ASR, TTS, and future LLM runtimes?
- 真机冒烟的 pass 阈值（TTFA 倍率、错误率、CUDA error 计数）具体怎么定？参考 TTS N=2 已用的 ≤1.5× TTFA gate
- How should subprocess workers report runtime-discovered limits back to Python (claimed N=2 实际炸的检测)?
