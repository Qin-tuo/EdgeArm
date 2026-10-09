# MOSS-TTS-Nano Jetson Orin C++ TRT 适配 Postmortem

**作者：** suharvest
**周期：** 2026-05-23 ~ 2026-05-24（连续约 36 小时密集 debug 跨日）
**最终结果：** Production cutover 完成 — TTFA **157 ms** on Orin NX，19× 快于 ORT CPU EP fallback (3000 ms)，3 段中文 prompt ASR CER=0，5 次重复 byte-identical。
**关联 commits：**
- 主项目 `seeed-local-voice` @ `4f15d34` — docs/profiles cutover
- Fork `TensorRT-Edge-LLM` @ `3c6c263` (branch `qwen3-tts-highperf-runtime-w8a16`) — runtime + worker + KV dtype fix

---

## TL;DR

- 目标：OpenMOSS/MOSS-TTS-Nano 0.1B（12L LM + 16-codebook RVQ codec, 12.5Hz frame, 48kHz stereo）port 到 Jetson Orin NX 的 TensorRT 路径，无 PyTorch 依赖，跟现有 qwen3_tts 同一 production 架构。
- 路径：ONNX surgery（FP16 + delta KV + If-rank fix） → trtexec 6 engines → CUDA kernel + C++ runtime → JSON-line worker → Python backend → 集成。
- 真正吃掉 80% 时间的不是 LM/codec 适配，而是 **1 行 `sizeof(half)` 硬编码**。这一行让 KV scratch buffer 是 engine IO 实际字节数的一半，frame 1 起 layer 间 KV 互相覆盖。
- 因为 frame 0 的 audio sample 走 prefill `globalHidden`（独立 FP32 binding）算出来，**frame 0 byte-identical ORT 的 trick 反复欺骗了 6+ 轮诊断**，问题被误诊成「模型 EOS 截断」整整一周。
- 真根因不是任何「TTS 移植教科书」上常见的 attention/RoPE/sampling/paged-KV 问题。在 6 个嫌疑全部 refute 之后，**只有重新审视"事实集合"，对 KV scratch 做 byte-level dump，才发现 misalignment 比例是精准的 2:1**。
- 一旦定位，修复极小（dynamic `getTensorDataType()` probe，~30 行）。

---

## 1. 项目背景

### 模型
OpenMOSS/MOSS-TTS-Nano 0.1B：12-layer transformer LM（hidden=768, 12 heads, head_dim=64）+ RVQ audio codec（16 codebook, 12.5Hz frame rate, 48kHz stereo 输出）。`input_ids` row width = 17（1 text + 16 RVQ audio codebook），每个 audio frame 需要 1 个 global LM step + 16 步 local RVQ 采样 + 1 次 codec_decode_step。

### 目标场景
Jetson 边缘部署，跟 qwen3_tts 同一 production stack（C++ stdin/stdout JSON-line worker + Python `TTSBackend` + HTTP `/tts` / `/tts/stream` / `/tts/clone`）。**不能依赖 PyTorch**（aarch64 wheel + glibc + 显存预算都不允许）。

### Reference 资产
- 官方 ONNX export 脚本（`onnx/export_hf_to_tts_onnx.py`）
- 官方 Python ORT runtime（`onnx_tts_runtime.py` / `ort_cpu_runtime.py`）
- 独立 codec 仓库 `OpenMOSS/MOSS-Audio-Tokenizer-Nano-ONNX`（不需要自己 trace）

### 性能 baseline
qwen3_tts 在 AGX Orin TTFA 370 ms。trtexec 阶段实测 MOSS engine 单步延迟，T7 预测 TTFA 30-150 ms，**2.5-10× 收益**预期 → 决策值得花 10-15 PD。

---

## 2. 关键选择 / 决策点

| # | 决策 | 选项 | 选择 | 理由 | 当时风险评估 |
|---|---|---|---|---|---|
| 2.1 | Worker 协议 | JSON-line stdin/stdout vs gRPC vs Unix socket | JSON-line | 跟 qwen3_tts 同架构，Python backend code 可整段复用 | 低 |
| 2.2 | KV cache 布局 | paged (qwen3_tts 风格) vs linear | linear | MOSS 12 层全 global attn 无 PD 分离；fork 实际**没有** paged allocator（codex 读真实代码确认）；N≤4 边缘场景 linear 144 MB/slot 完全够 | 中（与 spec 早期 paged 设计冲突，砍掉 4-5 PD） |
| 2.3 | ONNX export 路径 | WSL2 GPU 训练机 vs Mac 直接导出 | WSL2 | PyTorch + CUDA + 显存，Mac 跑不动 0.1B model.half() | 低 |
| 2.4 | KV present mode | 全量 sequence vs delta last-token | prefill 全量 + decode delta | 节省 14-16% per-step（实测 prefill 3.23→2.77ms, decode 3.34→2.79ms） | 低（但导出 patch 后期成为误诊轴） |
| 2.5 | Local 16-step | 自己写 16-step 串行循环 vs 调用官方 `local_fixed_sampled_frame` | 后者 | codex 阶段 5 Risk-5 揭示：官方有 one-shot frame sampler 内置 16 步循环 + sampling，**省 80% C++ 代码**。是整个项目最大的早期节省 | 低 |
| 2.6 | ORT fallback 是否保留 | 删 vs 保留 | 保留 | ORT CPU EP 已 verify CER=0；做 deterministic fallback 价值高；profile 切换零成本 | 低 |
| 2.7 | Codec ONNX If-rank | 自己 patch ONNX vs 重导出 codec | patch（onnx.helper Unsqueeze） | 不需要原仓库的 PyTorch source；通用 surgery script 可复用 | 中（patch 失败要 fallback ORT codec） |

---

## 3. 适配 timeline + 坑 trail

### 阶段 1 — Recon + 难度评估（~1 PD）

**做了什么：** codex 读官方 `ort_cpu_runtime.py` / `onnx_tts_runtime.py` / `export_hf_to_tts_onnx.py`，画 IO schema 流程图，dump 每个 tensor 的 dtype/shape/dynamic-axes，与 trt_edge_llm 接口对比。

**关键发现：**
- `input_ids` 是 3D `[1, S, 17]`（1 text col + 16 RVQ codebook col），不是 1D 序列。
- KV 是 flat per-layer `[B, T, H, D]` FP32（与我们 fork 的 FP16 假设不一样）。
- 独立 codec 仓库提供了完整 codec ONNX，不需要自己 trace。
- 本地 `local_transformer` 16 步串行是**模型架构硬约束**，TRT 单步压不掉。

**坑：** 一开始假设 fork 有 paged KV allocator（spec `an internal engineering spec (not in the public mirror)` 是按 Option B paged 写的）。codex 读真实代码后确认 fork 的 `KVCacheManager` 只是 flat `[maxBatchSize, 2, numKVHeads, maxSeqLen, headDim]`，没有 block manager。**砍掉 4-5 PD paged allocator 工作量**，改走线性 KV（spec `an internal engineering spec (not in the public mirror)` 命名仍叫 "paged" 实际是 linear，是历史遗留命名问题）。

**产出：** `memory/moss_tts_nano_port_recon.md` + `an internal engineering spec (not in the public mirror)` + `an internal engineering spec (not in the public mirror)`。

### 阶段 2 — ONNX patch + 重导出（~1.5 PD）

**做了什么：** WSL2 上对 `export_hf_to_tts_onnx.py` 写 unified diff `scripts/patches/moss-tts-nano-paged-kv-fp16.patch`（137 行），核心：
- Wire-level KV dtype FP32 → FP16
- Decode_step `present_*` 输出从 full sequence 改成 `[B, 1, H, D]` delta
- Prefill 保留 full sequence（首次填 KV pool）
- `dynamic_axes` 命名调整为 `kv_delta_seq`
- meta JSON 加 `kv_cache_dtype`/`present_kv_output_mode_prefill/decode` 等字段

**坑 1：** Codec ONNX (`OpenMOSS/MOSS-Audio-Tokenizer-Nano-ONNX`) 的 `If` 节点 then/else 分支输出 rank 不一致（rank-2 vs rank-3）。ORT 容忍，TRT 10.3 ONNX parser 严格拒绝：
```
[E] /If_OutputLayer: IIfConditionalOutputLayer inputs must have the same shape.
    Shapes are [1,-1] and [1,1,-1]
```

**解决：** 写 `scripts/fix_moss_codec_onnx_if_rank.py` — 递归遍历 If 节点，在 rank-2 输出分支前插 `Unsqueeze(axis=1)` 拉齐到 rank-3。通用模板，未来其他模型也可套。

**ORT smoke：** 改完图后在 WSL2 上 ORT CPU EP 跑通 dummy 输入产 WAV，证明语义没破。**没在目标设备上跑** — T1 阶段不该上目标设备。

### 阶段 3 — Engine build（~1 PD）

**做了什么：** Orin NX 上 trtexec FP16 build 全部 6 engines（5 TTS + 1 codec），显式 `--minShapes/--optShapes/--maxShapes`：

| Engine | mean step | engine plan |
|---|---|---|
| prefill | 2.77 ms | 215 MB |
| decode_step | 2.79 ms | 214 MB |
| local_decoder | 0.93 ms | 62 MB |
| local_cached_step | 0.93 ms | 110 MB |
| local_fixed_sampled_frame | 8.21 ms | 66 MB |
| codec_decode_step | 6.86 ms | 27.6 MB |

**坑 2：** Codec engine `--maxShapes=audio_codes:1x8x16` 默认 8 frame 太小，runtime chunk_frames 必须 cap 在 8 否则 `decodeFrames exceeds maxFrames`。

**解决：** 在 worker 里加 `kCodecMaxFramesPerBatch = 8` 常量，自动 clamp 请求的 `chunk_frames` / `first_chunk_frames`。文档化在 worker_p1 memory 里。

**坑 3：** trtexec 单 build 跑 10-30 分钟 tactic search，不用 `2>&1 | tee` 留 log，perf 数字会丢。所有 6 个 build 都 log 落盘。

**TTFA 预测（含 codec 6.86 ms）：** 4-frame chunk wall TTFA ~120-160 ms — go signal。

### 阶段 4 — CUDA kernel + C++ runtime（~3.5 PD）

**做了什么：**
- `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}` — `appendMossLinearKV` host launcher + 2 path（headDim=64 fast path + generic path）+ 5/5 gtest PASSED in standalone link。
- `cpp/runtime/mossTtsNanoRuntime.{cpp,h}` (~1500 行) — Slot pool + RequestGuard RAII + `prefill()` / `decodeStep()` / `sampleFrame()` / `decodeFrames()` / `generate()` + codec ping-pong streaming state（4 transformer_offset + 12 attention cache）。

**坑 4：** Fork 的 `cpp/kernels/cuteDSLArtifact/aarch64/sm_87/include/gdn_decode.h:6` 用 `static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)` 做 shim，CUDA 12.6 原生已有该符号 → `declared 'extern' and later 'static' [-fpermissive]`。影响 `cuteDslGDNRunner.cpp.o` / `cuteDslSSDRunner.cpp.o`，full `unitTest` build 链接失败。

**解决：** 绕过 — 用 nvcc 直接 standalone link 一个 gtest binary 走 kernel 单测，不走 full `unitTest` target。修复 cuteDSL header 留给后续 upstream PR。

**坑 5（小，但被记下来）：** Build script 一开始不 rebuild runtime `.o`，改完 `.cpp` 后 link 还在用旧 obj，**导致后面 KV dtype debug 反复怀疑改动没生效**。修法：`build_moss_worker.sh` 加 mtime 检查自动 rebuild runtime `.o`。

### 阶段 5 — Worker P1 + E2E smoke（~1 PD）

**做了什么：** `cpp/workers/moss_tts_nano_worker.cpp` (25 KB) — main + JSON-line 协议（`worker_ready` / `ready` / `chunk` / `done` / `error`）+ SentencePiece tokenize + 可选 codec_encode ORT session（voice clone 用）+ 流式 chunk emit。

**Dummy 输入端到端 smoke (orin-nx)：** TTFA 97 ms / RTF 0.17 / 8s 48kHz stereo WAV 产出。SOX stat 验证非全 0 信号。

### 阶段 6 — Voice agent 集成（~1 PD）

**做了什么：** Python `MossTtsNanoBackend` 类（`app/backends/jetson/moss_tts_nano.py`, 508 行）：subprocess.Popen spawn worker + queue.Queue chunk 分发 + `synthesize` / `generate_streaming` / `clone_voice` / `extract_speaker_embedding`。

**坑 6：** 默认 voice "Lingyu" prompt_audio_codes 是 218 rows（vs "Junhao" 98 rows），prefill `input_ids` shape 超过 default `--maxShapes=input_ids:1x128x17`。

**解决：** 调大 profile maxShapes + rebuild prefill engine（`--maxShapes=input_ids:1x256x17,attention_mask:1x256`），同步 lift `kKvProfileMaxPast = 256`。

**P5 端到端：** TTFA 120 ms via Python backend，13.76s 真合成（dummy zero token 噪声音频）。下一步真实 text 验证 → 走到下一个坑。

### 阶段 7 — 第一次质量危机：「长文本被截断」（约 0.5 PD 浪费）

**症状：** 短 utterance "你好" CER=0 ✓，但中长文本被截断：
- 中文本 "你好，今天天气真不错" → ASR 只识别出 "你好"
- 长文本 "这是一个语音合成测试，希望听起来自然" → 同样截断在前几个字

**第一次诊断（错）：** Worker probe `sampleFrame` 输出 `should_continue` flag — frame 17 时模型自然返回 0 → "**模型 EOS hallucination**"。
- 短 prompt 输出 9 frame 自然停 → 看起来合理
- 长 prompt 也停在 frame 17 → 解释为「prompt 长度不够触发模型继续」

**记到 memory：** `moss_tts_nano_model_quality_findings.md` 写「voice prompt 长度影响 EOS，**模型局限，非 worker bug**」。这一份 memory 是**误诊**，事后用一条 `INVALIDATED` 注释 + 链接到真相 doc 留档（不删，保留学习价值）。

**临时退路：** 切 Python ORT subprocess worker 路径（`deploy/jetson-workers/moss_tts_nano_ort_worker.py`），CPU EP 实测：
- 中文本 "你好，今天天气真不错" CER=0 ✓
- 长文本完整识别 ✓
- 39-token 长文本完整识别 ✓
- TTFA 3000 ms（慢但 deterministic 且对内容正确）

**当时结论：** 「TRT engine 跟 ORT 在同 ONNX 上数值发散，先发 ORT 路径，TRT 留作未来高性能候选」。

**这个结论是错的。** 不是数值发散，是 buffer-sizing bug。但 ORT 路径作为 fallback 留下来后来很有用。

### 阶段 8 — v16 FP32 rebuild + 第二次质量危机：「audio 全 garbage」（多轮 debug 跨 1 周）

**重启 TRT 路径的契机：** 怀疑「TRT vs ORT 数值发散」需要先排除 FP16 量化漂移。试验：把 v16 所有 5 个 decode 路径的 engine 重 build 成 FP32 KV IO（attention 仍 FP32 compute）。

**预期：** 数值精度上对齐 ORT → EOS 行为应一致。

**实测：**
- ✓ `should_continue` flag 行为修正了 — 短文本停在 frame 9，中文本 frame 26，长文本 frame 46（自然 EOS，跟 Python ORT 一致）。
- ✗ Audio 却完全 garbage — peak 信号 -0.999 clipping，DC offset，4kHz 高频啸叫。ASR 返回空字符串或乱码。
- 部分场景 peak < 0.001 完全静音。

**就这样进入 6+ 轮诊断噩梦。**

### 阶段 9 — 6+ 轮 diagnostic trail（每轮一个嫌疑，全部 refute）

> **这段是整份 postmortem 最重要的部分。** 写详细，让后人不要重新走一遍。

每一轮都派 codex 第二意见 + 主线程 fact-check。每个嫌疑 refute 后没意识到「事实集合本身」可能有问题，连续 round 2/3 接着错前提推下去。

#### 轮 1 — codebook count 8 vs 16 mismatch
**假说：** Codec 编码用 8 个 codebook，LM 输出 16 → 16 个被截到 8 → corruption。
**Refute：** ONNX meta JSON `n_vq = 16`，Python ORT 也是 16，runtime 也读到 16。双方一致，无 mismatch。

#### 轮 2 — K-RoPE TRT fusion 错位
**假说：** TRT 在 FP32 rebuild 时把 K-RoPE 的 sin/cos rotation 跟 K projection fusion 顺序变了 → 数值偏移。
**Refute：** Polygraphy `decode_step` engine FP32 单步 inline test PASS（atol 1e-2，跟 ORT 输出 byte-byte match）。**这里的 PASS 后来发现是误导的** — Polygraphy 测的是 inline build 的 engine、喂 numpy 输入，不走 runtime 的 host buffer，所以 buffer-sizing bug 看不到。当时没意识到这点，把 Polygraphy PASS 当成「engine 没问题」一锤定音。

#### 轮 3 — KV dim order BTHD vs BHTD
**假说：** MOSS 用 `[B, T, H, D]`，qwen3_tts 用 `[1, H, T, D]`，appendMossLinearKV kernel 可能搞反了维度顺序。
**Refute：** 重新看 kernel 代码 + ONNX meta `kv_layout_order = ["B","T","H","D"]`。kernel 严格按 BTHD 写。读 onnx graph：present_key dim names 也是 `[batch, seq, num_heads, head_dim]`。无 mismatch。

#### 轮 4 — audio_codes layout transpose
**假说：** Codec 输入 `audio_codes` 实际形状是 `[B, 16, T]` 而非 worker 假设的 `[B, T, 16]` → 16 codebook 维度跟 frame 维度 transpose。
**Refute：** ONNX `audio_codes` input shape annotation `[batch, frame, n_vq]`，Python ORT 调用也明确传 `[1, T, 16]`，runtime 一致。

#### 轮 5 — global_hidden dtype 误传
**假说：** v16 rebuild 把 `global_hidden` 也换成了 FP32，runtime 仍按 FP16 setTensorAddress → reinterpret cast garbage。
**Refute：** Runtime 启动时打印 `mGlobalHiddenDtype = kFLOAT (FP32)`，所有 binding 用 sizeof(float)。**注意：global_hidden 的 dtype 已经被正确动态 probe，但 KV 的没有。** 这是一个关键 hint，当时没抓住。

#### 轮 6 — audio_codebook_size 默认 2048 vs ONNX 1024
**假说：** Runtime hardcode `audio_codebook_size = 2048`，但 ONNX 实际 vocab 是 1024 → sampling 时溢出 index → 取到 garbage embedding。
**Refute：** Runtime `readMetadata()` 从 `meta.model_config.audio_codebook_size` 正确读到 1024。Log 打印验证。

#### 轮 7 — codec engine 数值 / If-rank patch 未生效
**假说：** Codec ONNX If-rank surgery 出错，patch 后的 codec engine 数值不对。
**Refute：** Cross-decode test — 把 Python ORT 跑出的 audio token sequence（正确的） dump 出来喂 C++ runtime 的 codec engine → 输出**干净中文语音**。所以 codec engine 是对的，问题在 audio token 本身。

#### 轮 8 — codec batch / streaming state ping-pong
**假说：** Codec 4-buffer ping-pong 有 race，跨 frame state 串。
**Refute：** Single-frame call 同样 garbage；调用模式与 Python ORT match。

#### 轮 9 — sampler RNG 序列
**假说：** `local_fixed_sampled_frame` 内置 sampler 用 PCG64 RNG，C++ 端没正确 seed → 跟 ORT 偏离。
**部分 disprove：** Fixed RNG 0.5 强制采样 → silence；inject PCG64 frame 0 byte-identical to ORT → frame 1 又 diverge。**关键观察：frame 0 byte-equal。** 当时记入「frame-0 trick」但没意识到这意味着上游 prefill 是对的，所以 KV 写回路径才是嫌疑。

#### 轮 10 — prefill engine 数值漂移
**假说：** Prefill engine FP32 rebuild 后输出 `global_hidden` 数值有累积漂移。
**Refute：** Polygraphy PASS atol 1e-2；dump prefill 输出 byte-compare ORT，前 12 KB byte-identical。

#### 轮 11 — prefill input_ids 拼接错
**假说：** `input_ids` 在 Python 端拼接顺序错（prompt template + ref audio + text）。
**Refute：** Dump runtime 跟 Python 跑的 `input_ids.bin`，md5 byte-identical。

#### 轮 12 — sample_frame engine 数值 + hidden 传递
**假说：** `local_fixed_sampled_frame` 拿到的 global_hidden 在 frame 边界被污染。
**Refute：** Fixed RNG seed 模式下 **frame 0 byte-identical** → sample_frame engine 输入输出 frame 0 是对的。

### 阶段 10 — 真根因抓到（约 4 小时）

**改变诊断策略：** 6+ 轮全 refute 之后，主线程意识到「所有嫌疑都基于一个隐含假设：runtime KV scratch 跟 engine 期望的 IO 是同 dtype」。这条假设没人验过。

**决定性动作：** 不再听 codex 提新嫌疑，自己做 byte-level dump bisect：
1. 在 `decodeStep()` frame 1 入口前，把 `slot.globalKvDevice[layer=0]` 的 `past_key_0` 内容 cudaMemcpy 到 host，dump 成 hex。
2. Python ORT 在同 prompt + 同 frame 跑一遍，dump 它的 `past_key_0` input 内容到 hex。
3. Diff。

**结果：** 前 ~12 KB byte-identical → byte 12288 开始，C++ 版本的内容跟 ORT 的内容**整体右移了 2 字节**。Misalignment 比例精准 2:1。

**5 分钟内的 grep：**
```
$ grep -n 'sizeof(half)' cpp/runtime/mossTtsNanoRuntime.cpp
181:    mGlobalLayerKvBytes = 2 * mMaxSeqLen * H * D * sizeof(half);
183:    mLocalLayerKvBytes  = 2 * 16 * H * D * sizeof(half);
185:    mMaxPresentLayerBytes = 2 * maxStep * H * D * sizeof(half);
460:    auto layerOffset = layer * (2 * mMaxSeqLen * H * D * sizeof(half));
511:    cudaMemcpyAsync(..., S * H * D * sizeof(half), ...);
558:    cudaMemcpyAsync(..., 1 * H * D * sizeof(half), ...);
580:    cudaMemcpyAsync(..., 1 * H * D * sizeof(half), ...);
659:    sizeof(half) * pastLen * H * D
```

**8 处 hardcode。** 全部都假设 KV element = 2 bytes。但 v16 engine 现在是 FP32 IO，element = 4 bytes。

**交叉验证：** 在 runtime 加一行
```cpp
auto kvDtype = mDecodeEngine->getTensorDataType("past_key_0");
std::cerr << "[moss] KV element dtype=" << int(kvDtype) << "\n";
```
启动打印 `dtype=0`（kFLOAT）。**bug 确认。**

**为什么 byte 12288 才开始 diverge：** 12288 = 12 (heads) × 64 (head_dim) × 2 (bytes) × 8 (tokens) — 大约写完第一个 layer 的前 8 个 token 位置后，越过半 buffer 界限开始压第二个 layer 的 region。Layer 0 自身写入是连续的，所以 prefix byte-identical 一段；从越界点开始 layer 0 K 覆盖 layer 0 V 的 region，layer 0 V 写覆盖 layer 1 K，layer 1 K 写覆盖 layer 1 V，等等。每一层都有部分内容被下一层 K/V 写覆盖。Frame 0 token 输出走的是 prefill 的 `globalHiddenDevice`（一个**独立的 FP32 binding**，跟 KV scratch 不共享内存），所以 frame 0 byte-identical ORT — 这就是「frame-0 trick」的真正成因。Frame 1 起 `sampleFrame` 调 `decode_step` engine 读了被污染的 `past_key/value` → output token 飘向模型先验的高频字符 → ASR 看着像 EOS hallucination。

### 阶段 11 — 修复

**Diff：** `cpp/runtime/mossTtsNanoRuntime.{cpp,h}`:

```cpp
// 新增 helper
size_t MossTtsNanoRuntime::tensorElementSize(nvinfer1::DataType dtype) const {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 0;
    }
}

// loadEngines() 里
auto kvDtype = mDecodeEngine->getTensorDataType("past_key_0");
mKvElementSize = tensorElementSize(kvDtype);
std::cerr << "[moss] KV element dtype=" << int(kvDtype)
          << " size=" << mKvElementSize << " bytes (FP32=0 FP16=1)\n";

// 全部 8 处 sizeof(half) → mKvElementSize
```

**Header：** 加 `size_t mKvElementSize{2};` 成员（默认 FP16 back-compat）。

**Build script：** `build_moss_worker.sh` 加 mtime 检查自动 rebuild runtime `.o`（避免再被 stale obj 骗）。

### 阶段 12 — 生产化落地

**Burn-in 5 测试（2026-05-24 on orin-nx）：**

| 场景 | TTFA | Audio | md5 |
|---|---|---|---|
| short "你好" | 152 ms | 0.56 s | `3989341fccd21950508dccc27bb22300` |
| medium "你好，今天天气真不错" | 154 ms | 4.24 s | `fbe5a9a2e07c63fa88a3bbe1866bfdc1` |
| longer (41-char zh) | 156 ms | 9.60 s | `c2f9217bb1c0af0c90f5a94bb513b69e` |
| voice clone (zh_1_ref) | 1063 ms | 1.36 s | `b89bb62595e8a92716ec7c0281d446c2` |
| 5× repeat medium | 152-153 ms | 4.24 s | all 5 == `fbe5a9a2...` ✓ byte-identical |

ASR CER=0 across 3 Chinese prompts via radxa qwen3_asr_rk。**5 sequential runs byte-identical** → 证明 KV scratch 没残留状态污染 cross-invocation。

**Profile 架构：**
- `jetson-moss-tts-nano-trt` (NEW, DEFAULT) → C++ binary @ `/opt/jv-workers/moss_tts_nano_worker` md5 `3017b3f34bb9c4cbc8391f65ecd84541`
- `jetson-moss-tts-nano` (FALLBACK) → Python ORT @ `/opt/jv-workers/moss_tts_nano_ort_worker.py` TTFA 3000 ms

Pre-fix binary 备份在 `/opt/jv-workers/moss_tts_nano_worker.before_kvdtype_fix` md5 `7be68fe0c2d83042227d14d314e7d4e2`（debug 对照用）。

**Commit：**
- Fork @ `3c6c263` — runtime + worker + KV fix
- Main @ `4f15d34` — profile + runbook + spec + playbook §10.5

**Memory + gotcha：**
- `[[moss_tts_nano_trt_production_ready]]` 写新真相
- `[[moss_tts_nano_model_quality_findings]]` 顶部加 `INVALIDATED` 注释链到真相 doc
- `[[moss_tts_nano_ort_path_production_ready]]` 顶部加更新说明（ORT 现在是 fallback）
- Gotcha #15 入 `~/.claude/skills/device-gotchas/references/gotchas-jetson.md`

---

## 4. Root Cause Deep Dive

### 4.1 为什么 frame 0 byte-identical（关键迷局）

Frame 0 的 audio sample 通过下面这条链路产生：

```
prefill engine
  → 输出 globalHiddenDevice [B, S, H*D] (independent FP32 binding)
  → local_decoder engine (frame-0 only, one-shot)
     输入 globalHiddenDevice → 输出 frame-0 audio_codes [1, 1, 16]
  → codec_decode_step → PCM
```

**这条链路完全不读 KV scratch buffer。** `globalHiddenDevice` 是 prefill 输出的一个独立 device tensor，单独 cudaMalloc，跟 `slot.globalKvDevice` 不共享内存。Prefill 写 `present_key/value` 到 KV scratch 用的是 hardcoded `sizeof(half)` 步进，**写入数据本身的 byte 内容是对的**，只是在 buffer 里的「布局」是错的（layer-0 K 写到了 layer-0 K + layer-0 V 的合并区，但因为是第一次写、layer-0 V 还没写，所以 frame 0 时没人去读到那些越界的字节）。

Frame 1 起调 `decode_step` engine，开始读 `past_key/past_value`。Engine 期望按 FP32 step (4 bytes/element) 读取，而 scratch 里的内容是「**正确的 FP16 数据但按 sizeof(half) 步进排布**」。读出来 dtype 没错（engine 直接当 FP32 读出来），但 element 间布局错位 — 读到的不是「layer-0 K, layer-0 V, layer-1 K, ...」连续，而是「layer-0 K 前半 + layer-0 K 后半 + layer-0 V 前半 + ...」交错。

**关键观察：corruption 是 clean 的** — 没 NaN、没 crash、没明显异常值。Token 分布飘向模型先验的高频字符（中文里就是常见尾词），整体看着像「模型生成了一段无意义但 grammatically 合理的中文」→ 完美伪装成 EOS hallucination。

### 4.2 为什么 Polygraphy 永远 PASS

Polygraphy 跑的是「inline build engine」+ 「喂 numpy 输入」+ 「读 numpy 输出」，**完全不走 runtime 的 host buffer / KV scratch 路径**。Engine 本身没问题，是 runtime 给 engine 喂的 KV scratch 内容布局错。

**教训：** Polygraphy / 任何 engine-level numerical diff tool 都只能验「engine ↔ ONNX 语义一致」，不能验「runtime 整链路正确」。后者需要 dump 真实 runtime device buffer 跟 reference 对比。

### 4.3 Fix 修了什么

8 处 `sizeof(half)` 全部换成 `mKvElementSize`（启动期 dynamic probe）。具体行（fork @ 3c6c263 之前的位置）：

| 行 | 用途 |
|---|---|
| 181 | `mGlobalLayerKvBytes` 单层全 KV scratch size |
| 183 | `mLocalLayerKvBytes` 单层 local KV scratch size |
| 185 | `mMaxPresentLayerBytes` 单层 present staging |
| 460 | Slot 内单 layer offset |
| 511 | Prefill copy-back present → KV scratch 字节数 |
| 558 | Decode step present write 字节数 |
| 580 | Decode step copy-back |
| 659 | Decode step past_key shape 总 element count → bytes |

Header 加 `size_t mKvElementSize{2};` member。`loadEngines()` 启动期一次 probe。Future FP16 rebuild Just Works（probe 返回 kHALF→2）。

代码详见 fork @ `3c6c263`。

---

## 5. 经验教训

### 5.1 通用规则（直接抄进各 device-gotchas）

1. **跨组件 ABI 用 dynamic introspection，永远不要 hardcode dtype element size**。`engine->getTensorDataType(binding)` 是基本动作，runtime 启动期必须 probe 所有 IO binding dtype 并打 log。
2. **Engine rebuild 必须 verify worker rebuild**。`.o` mtime 检查必装。任何 build script 不验证 stale obj，就有可能在 KV dtype 这类 ABI 问题上骗自己整周。
3. **多帧累积模型至少验 frame 2-3**，不要只看 frame 0。Frame 0 通常走 prefill 的 standalone tensor，跟 decode loop 的 stateful KV 不共享内存。
4. **Frame 0 byte-identical 不证整链路正确**。这是反直觉的，但 MOSS 案例证明 prefill 链路完全独立于 decode KV。
5. **Polygraphy / engine-level 数值 diff PASS 不证 runtime 正确**。Engine 单步 PASS 跟 runtime stateful loop PASS 是两件事。
6. **任何 runtime 启动期一定打 dtype probe log**。MOSS 案例事后看，一行 `[moss] KV element dtype=X size=Y bytes` 能省 1 周。

### 5.2 诊断方法论

1. **N 个嫌疑全 refute 后，必有共同盲区**。第 6 轮失败时主线程没及时回退一步审视「事实集合本身是否真」，而是接着派 codex 提第 7、第 8 个嫌疑。这是浪费时间最大的轮次。**1 次修复失败应立即派 codex；3 次修复失败应停 debug，重新审视前提**。
2. **Codex 第二意见容易基于错误前提推断**。Codex 看到 frame 0 byte-equal 就承接「prefill / sampler / engine 都没问题」这个假设继续推，导致 round 2-12 都在围绕「decode 路径上的算法问题」转。Codex 在「事实集合不全」时无法自检 → **派 codex 前先 fact-check 现有事实，确认它们真的是事实而不是没验过的假设**。
3. **Inline test PASS 不等于 production PASS**：Polygraphy decode_step PASS 跟生产 runtime decode loop PASS 是两件不同的事。**Production-side test 必须跑 real runtime 的 device buffer dump + byte-compare reference**。
4. **Fixed RNG + fixed input 是 bisect divergence 起点的最强工具**。MOSS frame-0 byte-equal 这个观察本身是金矿（说明 prefill 链路完全 OK），但需要正确解读 — 它的意思不是「runtime 都对」，而是「prefill output 都对」，反推「decode loop 输入有问题」→ 那 decode loop 的输入是什么？`past_key/past_value` 来自 KV scratch → 该 dump 的就是 KV scratch。可惜这个推理链当时没走通。
5. **Decisive bisect 比 single-point probe 强**。Round 12 之前都在单变量 probe（看 should_continue / 看 RNG / 看 prefill output），没人把「整个 KV scratch」当一个变量做 byte-byte compare。**State-corruption bug 的 ground truth 是 state dump byte-compare，不是任何单点观测**。

### 5.3 Anti-patterns（不要重复犯）

- ❌ **看到「frame 17 截断」立刻怪模型 EOS**，而不是怀疑 runtime 累积状态 bug。Stateful runtime 在 frame-1 起就该是首要嫌疑，而不是模型本身。
- ❌ **没有 sync engine rebuild 跟 worker rebuild**。V16 engine 切 FP32 时 worker `.o` 没 rebuild → 半天 debug 看到的还是旧逻辑。`.o` mtime 检查是基本卫生。
- ❌ **Codex round 1 refute 后没充分质疑「事实集合」**，连续 round 2-12 接着错前提推。3 次连续假设失败应触发 review。
- ❌ **单点诊断**（看一个变量看完看下一个）替代 **decisive bisect**（dump 完整 state byte-compare）。前者每轮 30 分钟 × 12 轮 = 6 小时全无进展，后者 1 小时直达根因。
- ❌ **不打 dtype/shape 启动期 log**。任何 runtime 启动期不打全 binding dtype + size 的 log 是裸奔。
- ❌ **Memory 写早写错**。`moss_tts_nano_model_quality_findings.md` 在 round 7 时就被写成「模型 EOS 局限」存档了 — 误诊一旦进 memory 会污染后续决策。改正：误诊 memory 顶部加 `INVALIDATED + 链接真相 doc`，不删（留学习价值）。

---

## 6. 复现 checklist

按这个 checklist 走可以重新部署一遍。每一步有 stop-criteria，不通过别跨阶段。

```bash
# === T1: 环境准备 (dev GPU 机器，WSL2) ===
[ ] PyTorch + CUDA + ONNX 1.16+ + onnxruntime 1.20+
[ ] git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git
[ ] huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M --local-dir ~/models/...
[ ] huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir ...

# === T2: ONNX patch + 重导出 (dev GPU 机器) ===
[ ] git apply scripts/patches/moss-tts-nano-paged-kv-fp16.patch
[ ] python onnx/export_hf_to_tts_onnx.py --checkpoint-path ... --output-dir ... --opset 17
[ ] python scripts/fix_moss_codec_onnx_if_rank.py --in-dir ... --out-dir ...
[ ] onnx.checker.check_model() 全部 6 个 ONNX 通过
[ ] ORT CPU EP dummy smoke 跑通（不是目标设备）

# === T3: Engine build (Orin NX 目标机) ===
[ ] 拷贝 ONNX bundle + codec bundle 到 Orin NX
[ ] bash scripts/build_moss_tts_engines.sh （内含 --minShapes/--optShapes/--maxShapes）
[ ] 全部 6 个 .plan 文件生成
[ ] Python tensorrt API dump 每个 engine 的 IO，确认 KV dtype 跟你预期一致
[ ] 单 engine trtexec --loadEngine 拿稳态延迟，跟 trtexec memory 的预期对照

# === T4: C++ worker build (Orin NX host, NOT container) ===
[ ] cd ~/TensorRT-Edge-LLM && git checkout qwen3-tts-highperf-runtime-w8a16
[ ] git log -1 --oneline  # 确认包含 commit 3c6c263 或更新
[ ] EDGELLM_SRC=$PWD ORT_ROOT=/usr/local/onnxruntime SP_ROOT=/usr OUT=/tmp/moss_tts_nano_worker \
        bash cpp/workers/build_moss_worker.sh
[ ] md5sum /tmp/moss_tts_nano_worker 跟 production binary 一致

# === T5: Deploy + dtype probe verify ===
[ ] sudo cp /opt/jv-workers/moss_tts_nano_worker /opt/jv-workers/moss_tts_nano_worker.bak.$(date +%Y%m%d-%H%M)
[ ] sudo install -m 0755 /tmp/moss_tts_nano_worker /opt/jv-workers/moss_tts_nano_worker
[ ] Spawn worker，在 stderr 看到 [moss] KV element dtype=... size=... bytes  ← 没这行就是旧 binary
[ ] 把这条 log 字段记进 deployment notes

# === T6: Profile select ===
[ ] export OVS_PROFILE=jetson-moss-tts-nano-trt  （TRT 默认）
[ ] 或 OVS_PROFILE=jetson-moss-tts-nano （ORT fallback，CPU EP）

# === T7: Smoke test ===
[ ] python3 bench/perf/smoke_moss_tts_backend.py --text "你好，今天天气真不错"
[ ] TTFA < 200 ms
[ ] WAV duration ~= text 长度 / 4 字/秒
[ ] sox stat: peak < 0.95, RMS > 0.05, rough freq 500-1500 Hz
[ ] ASR via radxa qwen3_asr_rk 或类似 → CER < 5%
[ ] 同一 prompt 跑 5 次 → 5 个 WAV md5 byte-identical
```

**验收门槛（任何一项 fail 都打回重 debug，不要 ship）：**
1. TTFA < 200 ms on Orin NX FP32 engine path
2. ASR CER < 5% on 短/中/长 三种中文 prompt
3. 5 sequential same-prompt runs byte-identical
4. Worker stderr 有 `[moss] KV element dtype=...` log
5. SOX peak 不 clipping，rough freq 在人声基频范围

---

## 7. 给 RK NPU port 的建议（下一站）

把 MOSS-TTS-Nano 或同类 LM-based TTS port 到 RK3576/RK3588 NPU 时：

### 直接复用
- T0 侦察方法论（读官方 Python runtime 源码画流程图）
- T1 ONNX patch（FP16 + delta KV + If-rank surgery）
- T5 真实输入测试 + ASR 验证流程
- T6 服务化框架（Python backend / HTTP layer 不变）
- **本 postmortem 的 6+ 轮 diagnostic trail 就是 R&D 阶段的避坑清单**

### 需要重写
- **T3 编译**：换 `rknn-toolkit2` 替代 trtexec。注意 RKNN 动态 shape 支持比 TRT 弱，多数 RKNN 模型用 fixed shape + 多个 engine 切换（不同 seq_len 各 build 一个 plan）。
- **T4 C++ runtime**：换 RKNN C API（`rknn_init` / `rknn_inputs_set` / `rknn_run` / `rknn_outputs_get`）。没有 `setTensorAddress` zero-copy，每次必须 memcpy（**RK 性能上限的硬约束**）。没有 CUDA stream / 异步，所有调用 sync。

### 强制要求（不要省）
1. **不要假设 toolchain 跟 trtexec 同语义**。RKNN 算子覆盖 < TRT，常见 fallback 算子（Where, ScatterND, 复杂 If）必须 ONNX 替换或 CPU fallback。
2. **一定 dump 第一个 RK runtime 输出 byte-compare ORT**。**这是 MOSS 那 6+ 轮 debug 没做的事，下次必做**。Frame 0 / Frame 1 / Frame 2 三个时间点的完整 state dump 都做 byte-compare。
3. **KV dtype / shape ABI 严格 verify**。`rknn.list_inputs()` + `rknn.query_perf_detail()` 启动期跑一遍 dump 全部 IO dtype/shape，与 runtime 假设的字节布局对一遍。**MOSS 案例如果在 RK 上重发，这条规则能省整周**。
4. **写 frame 0 + frame 1 + frame 2 byte-identical regression test**，每次改 runtime 跑一次。把 fixed-RNG + fixed-input 下的 3-frame WAV md5 当回归 gate。
5. **runtime 启动期打全 binding dtype + size + element-size**，跟 MOSS 现在的 `[moss] KV element dtype=...` 一样的 log 模板。

### RK 特有的注意点
（参考 `~/.claude/skills/device-gotchas/references/gotchas-rk3576.md` / `gotchas-rk3588.md`）

- RKNN 模型加载耗内存比 TRT 高（含完整算子图 JSON）。0.1B + 6 engine 在 RK3576 4GB 可能紧。
- `rknn_run` 阻塞无法 cancel — 上层 worker 要 spawn 子线程包一层 cancel 机制（参考 [[tts_worker_phase1_landed]] 的 cancel 协议）。
- ZeroCopy mode 需要 `--core-mask=0x01` 单核绑定，多核会 race。
- KV cache 必须用 zero-copy buffer，否则每步 memcpy 几十 MB 性能崩。

### 工作量重估
跟 Jetson 比 RK 多 ~3 PD，主要在 toolchain compile 阶段（多 fixed-shape engine + 算子兼容性调）和 C++ runtime（缺 zero-copy 多 memcpy 优化）。

---

## 8. 引用 / 关联文档

### Commits
- 主项目 `seeed-local-voice` @ `4f15d34` — "docs(moss-tts-nano): C++ TRT production cutover + KV dtype fix spec"
- Fork `TensorRT-Edge-LLM` (branch `qwen3-tts-highperf-runtime-w8a16`) @ `3c6c263` — "feat(moss-tts-nano): C++ TRT runtime + worker + KV dtype ABI fix"

### Specs / Runbooks / Playbook
- `an internal engineering spec (not in the public mirror)` — 完整 root cause + diagnostic trail（这份 postmortem 的精简版本）
- `an internal engineering spec (not in the public mirror)` — C++ runtime 线性 KV 设计 spec
- `an internal engineering spec (not in the public mirror)` — 原 Option B paged KV 设计（后改 linear，命名遗留）
- `docs/runbooks/moss-tts-nano-deployment.md` — 双路径部署 runbook
- `docs/playbooks/tts-model-edge-port-playbook.md` — 通用 TTS 边缘移植 playbook（§10.5 KV dtype ABI 教训）

### Scripts / Patches
- `scripts/patches/moss-tts-nano-paged-kv-fp16.patch` — ONNX wire FP16 + delta KV patch (137 行)
- `scripts/fix_moss_codec_onnx_if_rank.py` — Codec ONNX If-rank surgery（通用模板）
- `scripts/build_moss_tts_engines.sh` — trtexec recipe + 显式 shape profile

### Memory（项目长记忆，按时间序）
- `moss_tts_nano_port_recon.md` — T0 侦察
- `moss_tts_nano_trtexec_orin_nx.md` — T2 实测延迟 + codec If-rank fix 落地
- `moss_tts_nano_kernel_chunk1_done.md` — Chunk 1+2 落地详情
- `moss_tts_nano_worker_p1_done.md` — P1 worker 协议 + 实测中文合成
- `moss_tts_nano_smoke_e2e_done.md` — runtime 端到端（dummy 输入）
- `moss_tts_nano_model_quality_findings.md` — **INVALIDATED** 误诊存档（顶部链到真相）
- `moss_tts_nano_ort_path_production_ready.md` — ORT path（现在是 fallback）
- `moss_tts_nano_trt_production_ready.md` — TRT path 最终生产真相

### Gotcha
- `~/.claude/skills/device-gotchas/references/gotchas-jetson.md` 末尾 #15 — KV buffer dtype ABI mismatch

### Related context
- `[[trt_edge_llm_fork_path]]` — fork 位置 + 生产分支
- `[[tts_n2_phase_b_stability_landed]]` — slot lifecycle 模板（runtime 复用）
- `[[trt_edge_llm_tts_env_staleness]]` — Python backend module-level env 陷阱

---

## 9. 最后一句

**这次 port 实际编码 + engine + 整合大约 7 PD，本来该 14 PD 完成。但 KV dtype bug 因为「frame-0 byte-equal trick」误导 + 「N 个嫌疑全 refute 后没回审事实集合」白白吃掉 5-7 PD。**

下次再 port 任何 stateful LM-based 模型到 edge NPU，第一行 runtime startup log 必须打全部 IO binding 的 dtype + size。这一行 log 是用整周 debug 时间换来的。
