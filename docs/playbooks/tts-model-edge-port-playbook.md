# TTS 模型边缘端口移植 Playbook

**写于：** 2026-05-23，刚做完 MOSS-TTS-Nano 0.1B 在 Jetson Orin NX 上的全链路适配（TTFA 97ms，3.8× 快于 qwen3_tts baseline）。
**目标读者：** 下一次把同类生成式 TTS（LM + audio codec 架构）移植到 Jetson / RK NPU / Hailo / 其他 NPU 时的工程师。
**适用模型形态：** Encoder-Decoder 自回归 TTS，LM 主体 + RVQ codec decoder（MOSS / Qwen3-TTS / SoundStream 系 / EnCodec 系 / Tortoise 系），不太适用于纯 vocoder（HiFi-GAN / Matcha）。

---

## 0. 一句话总结

**90% 的时间花在「ONNX 和 runtime 形状对不上」上，不是花在模型本身。** 把这一类问题（ONNX 静/动态形状、dtype 边界、runtime profile、stateful KV 接口）的检查清单内化到流程里，下次移植从 2 周缩到 3-5 天。

---

## 1. 阶段拆分（按依赖关系）

```
T0 侦察（codex / 读官方 Python runtime）
  ↓
T1 ONNX 重导出 + patch（在 GPU 机器上跑，不在目标设备上跑）
  ↓
T2 编译目标设备 engine plan（trtexec / rknn_toolkit / hef_compiler）
  ↓
T3 C++ runtime 实现（按目标设备 API：TRT / RKNN-C / Hailo SDK）
  ↓
T4 端到端 smoke（dummy 输入证管线通）
  ↓
T5 真实输入验证（tokenizer + ASR 验内容）
  ↓
T6 服务化（worker process + Python backend + HTTP）
```

每阶段的 stop-criteria 是下一阶段的前提，**不要跨阶段试错**。

---

## 2. T0 侦察阶段：先读 Python 运行时，不要看 ONNX 文件

### 必做

1. **先 clone 模型仓库**：找到官方 inference runtime 的 Python 源码（不是模型权重，是推理代码）。例如 MOSS 的 `onnx_tts_runtime.py` / `ort_cpu_runtime.py`。
2. **从主推理函数往下读**：找 `generate()` / `synthesize()` 这种入口，画一遍流程图：text → tokenize → prefill → decode loop → codec stream → audio。
3. **理清 stateful 点**：哪些张量是 cross-step persistent（KV cache / codec streaming state / RNG state）。
4. **看输入张量的 prompt 结构**：MOSS 的 `input_ids` 是 `[seq, 17]`，col 0 = text，col 1-16 = RVQ audio，特殊 token 是 `im_start` / `audio_user_slot` / `audio_assistant_slot` 等。**这种"伪多通道 token"结构必须从源码而不是文档反推**。
5. **找推理路径的可选分支**：有些模型有 `local_decoder`（custom sampling）+ `local_fixed_sampled_frame`（one-shot），别盲目实现复杂分支（MOSS 案例：原 spec 让我实现 16 步串行循环，**实际官方默认用 one-shot frame sampler**，C++ 省 80% 代码）。

### 必避

- ❌ 不要从 ONNX 文件反推协议——dtype / dim name 都可能误导
- ❌ 不要相信 LLM 给的 anchor 行号——派 codex 拿真实文件
- ❌ 不要在没看官方 Python 代码前就开始写 C++ runtime

### 工具

- `gh api repos/.../contents/file -H "Accept: application/vnd.github.raw"` 拉单个文件
- `curl -sL https://raw.githubusercontent.com/.../file.py` 备用
- 派 codex agent 读完整文件后产 IO schema 报告（不要让它"读 spec"，让它读源码）

### 产出

一份 markdown 侦察报告（参考 `memory/moss_tts_nano_port_recon.md`），含：
- 完整 IO tensor 名 + dtype + shape + dynamic axes
- Stateful tensor 清单 + state schema 来源（meta JSON or 硬编码）
- prompt 结构示意 + 特殊 token id
- 与现有 backend（如 qwen3_tts）的接口差异点
- "复用 vs 重写 vs 新写" 工作量矩阵

---

## 3. T1 ONNX 重导出 + patch

### 关键决策点

#### Dtype 边界（FP32/FP16/BF16/INT8）

- **看模型原版 ONNX 里 KV / attention 是 FP32 还是 FP16**。MOSS 原版 KV 是 FP32，我们 patch 成 wire FP16 + 内部 cast 回 FP32 compute（避免 FP16 attention 精度回归）。
- **永远不要 import `onnxconverter_common.float16`** 来后处理整图——做 export-time `model.half()` 或 patch dummy tensor dtype。理由：后处理改图会破坏 dynamic_axes 命名 + external_data 引用。
- **RK NPU 注意**：RK3588 支持 FP16 + INT8，FP32 自动 fallback 到 CPU 很慢。导出时直接 FP16，输入仍可 INT32（token ID）。

#### KV cache 接口（最容易踩坑）

- 原版多半是 flat `[B, T, H, D]` per-layer，每步返回**全量** present。这种接口在 N>1 并发场景下 IO 带宽爆炸。
- **改造方向 A**（推荐多数边缘场景）：让 ONNX present 输出只返回 last-token delta `[B, 1, H, D]`。runtime 在 C++ 侧 append 到 per-slot 线性 buffer。
  - MOSS 案例：14-16% 速度提升（prefill 3.23→2.77ms, decode 3.34→2.79ms）
  - 实现 5 行：`flatten_present` 加 `delta_only=True` 分支，`key[:, -1:, :, :]`
  - **prefill 必须保留 full sequence**（首次填充 KV pool），decode 才用 delta
- **改造方向 B**（paged KV）：除非你目标设备的 KV manager 真的支持 paged（NVIDIA TRT-LLM 上游支持，但**很多 fork 实际是线性 KV**——我们自己 fork 就是），否则别做。MOSS 案例验证：fork 没 paged allocator，线性 5-6 PD vs paged 10-15 PD，性能差异在 N≤4 边缘场景可忽略。

#### Dynamic axes 必须显式重命名

- 改完 ONNX，把 `dynamic_axes` 的 dim name 也改（如 `total_seq` → `kv_delta_seq`），后续 dump meta JSON 时这是 runtime IO binding 的关键线索。

### Patch 文件管理

- **永远生成 unified diff**（`diff -u`），不要手编 patch
- 用 `patch --dry-run -p1` 先验证再 apply
- patch 留在 `scripts/patches/<model>-<purpose>.patch`，commit 进主仓库

### 必踩坑

1. **第三方 ONNX `If` 节点 rank 不一致**：MOSS codec ONNX 的 `If` 节点 then/else 分支输出 rank-2 vs rank-3。ORT 容忍，TRT/RKNN 严格。**修法：onnx.helper graph surgery，在 low-rank 分支插 Unsqueeze**。脚本 `scripts/fix_moss_codec_onnx_if_rank.py` 是通用模板，可直接套别的模型。
2. **External data 引用**：ONNX 主图 .onnx 文件可能只有 KB 级（结构），权重在 .data 旁文件。改完图重新 `onnx.save` 后要 `merge_shared_external_data` 重新 dedup。
3. **`onnx.checker.check_model` 必须过**：否则 TRT/RKNN parse 会给出难以诊断的报错。

### 产出

- 一份可 `git apply` 的 unified diff patch
- 重导出后产物清单（onnx 文件 md5 + size + dtype 变化）
- ORT smoke 通过证明语义正确（**这一步在 dev GPU 机器上做，不要去目标设备**）

---

## 4. T2 编译目标设备 engine plan

### 通用陷阱（所有 NPU 后端通用）

#### **静态 vs 动态 shape profile**

**这是 MOSS 适配遇到次数最多的问题。** 默认 `trtexec --onnx=foo.onnx --fp16 --saveEngine=foo.plan` 会用 ONNX 里的**示例形状**作为静态 profile baked 进 engine。Runtime 用任何别的 shape 都会 `Static dimension mismatch`。

**正确做法**：所有动态轴显式指定 `--minShapes` / `--optShapes` / `--maxShapes`：

```bash
trtexec --onnx=prefill.onnx --fp16 \
  --minShapes=input_ids:1x1x17,attention_mask:1x1,past_valid_lengths:1 \
  --optShapes=input_ids:1x32x17,attention_mask:1x32,past_valid_lengths:1 \
  --maxShapes=input_ids:1x128x17,attention_mask:1x128,past_valid_lengths:1 \
  --saveEngine=prefill.plan
```

每个动态张量、每条轴都要列。MOSS decode_step 有 12 层 past_key/past_value × 4 个 shape 字段 = 48 个 shape 声明，写一次脚本生成。

**RK 对位**：`rknn_toolkit2.config(dynamic_input=[...])` 配置类似，每个张量给 min/opt/max 三个 shape，注意 RKNN 的 dynamic shape 支持比 TRT 弱，**多数 RKNN 模型用 fixed shape + 多个 engine 切换**（不同 seq_len 各 build 一个 plan）。

#### **必做：build 完用 Python API 验 engine I/O**

```python
import tensorrt as trt
runtime = trt.Runtime(trt.Logger())
engine = runtime.deserialize_cuda_engine(open("foo.plan", "rb").read())
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    print(name, engine.get_tensor_dtype(name), engine.get_tensor_shape(name))
```

这比从 trtexec stdout 反推可靠太多。MOSS 案例：我们认为 `past_key_*` 是 dynamic 实际 baked 成 static `[1, 1, 12, 64]`，python dump 一眼看清。

**RK 对位**：`rknn.list_inputs()` + `rknn.query_perf_detail()`。

#### Shared external data

如果 ONNX 用 external `.data` 文件存权重，**编译时要确保 .data 在同目录**或者用 `--onnx=` 路径包含 data。否则 engine 体积异常小（只含结构），运行时 weight 都是 0 / garbage。

#### **必避**

- ❌ 一次 build 5+ engine 不留 stdout log。trtexec tactic search 会跑 10-30 分钟，stdout 关掉就丢了 perf 数字。每个 build 用 `2>&1 | tee build_<name>.log` 留底。
- ❌ build 在临时 `/tmp/` 上跑。设备重启就没了，必须移到持久目录。

### 产出

- 6 个 .plan 文件（5 TTS + 1 codec），各自 md5 + size + 单步延迟（trtexec --loadEngine 跑一次拿 mean GPU compute）
- 一份 `engine_profiles.txt` 记录每个 plan 的 min/opt/max profile
- VRAM 总占用 estimate（活跃 ExecutionContext × N slot 后的峰值）

---

## 5. T3 C++ Runtime 实现

### 架构原则

#### 复用现有同类 runtime 模板

如果项目里已有同类 backend（MOSS 案例：qwen3_tts），直接 mirror 结构：
- 同样的 slot pool + acquirePoolSlot / releasePoolSlot / RequestGuard RAII
- 同样的 per-slot ExecutionContext + per-slot CUDA stream（[[tts_n2_phase_b_stability_landed]] 教训）
- 同样的 per-slot pre-allocated scratch tensor（避免运行时 cudaMalloc）

不要自己发明新模式。

#### 每个 engine 一个 IExecutionContext per slot

不要让多个 slot 共享 context（N=2 时会撞 race）。预 alloc 全部 5+ engine × N slot 个 context，构造期 N=1 即可，先证 single-client。

### MOSS 总结的可移植坑

#### 坑 1：tensor name 不要硬编码假设

我曾假设 prefill 也有 `past_key_*` 输入（因为 decode 有），实际 prefill 只有 `present_*` 输出。**写 setInputShape / setTensorAddress 时永远用 `hasTensor(engine, name)` 守一下**。模式：

```cpp
if (hasTensor(engine, pk.c_str())) {
    ctx.setInputShape(pk.c_str(), pastShape);
    ctx.setTensorAddress(pk.c_str(), layerKvPtr(slot, layer, 0));
}
```

省一堆 `Static dimension mismatch` debug 时间。

#### 坑 2：meta JSON 字段位置不同模型不一样

MOSS 把 `audio_codebook_sizes`/`n_vq` 放在 `model_config.*` 下，不是 `tts_config.*`（我猜的）。**写 readMetadata 时 try 多个路径**：

```cpp
if (meta.contains("model_config") && mc.contains("n_vq")) ...
else if (meta.contains("tts_config") && tc.contains("n_vq")) ...
else fallback_default;
```

或者在 T0 侦察阶段就 dump 一份 meta 的 key 树，直接照着写。

#### 坑 3：输入 token row_width

MOSS `input_ids` 是 `[seq, 17]` 3D，不是 `[seq]` 1D。我第一版 setInputShape 用 `Dims2{1, seqLen}` 撞 mismatch。**确认每个张量真实 rank** 是 T0 阶段必做项。

#### 坑 4：stateful streaming state 必须 ping-pong

Codec streaming 输出是 next-step input。**不要 cudaMemcpyDeviceToDevice 同步 buffer**，用 `std::swap` 双 buffer 指针。MOSS codec 12 layer × 4 个 cache tensor × ping-pong = 96 个指针，写一个 swap loop 完事：

```cpp
for (auto& cache : state.attn) {
    std::swap(cache.offsetIn, cache.offsetOut);
    std::swap(cache.keysIn, cache.keysOut);
    // ...
}
```

#### 坑 5：cudaMemset(positions, 0xff) trick

`int32` cache positions 初始化为 -1 用 `cudaMemsetAsync(ptr, 0xff, n*4, stream)`，因为 `0xffffffff` == -1 二进制补码。一行解决，不需要单独 kernel。

#### 坑 6：FP16 boundary cast 不是 FP16 compute

在 ONNX patch 阶段我们让 wire = FP16，但内部 attention 仍 cast 回 FP32 compute。在 runtime 别脑补成 FP16 全链路——`setTensorAddress` 给 ONNX 喂 FP16 buffer，ONNX 内部该 cast 还是 cast。

### Build / test 策略

1. **每写完一个文件先 .o 编译验证**，不要写一大堆再 link
2. **优先 standalone gtest**，绕开 full project link（MOSS 案例：fork 的 cuteDSLArtifact 在 CUDA 12.6 下编译错，我们用 standalone link 绕过去）
3. **smoke test 用 dummy zero 输入**，先证不崩、shape 通、不 NaN，再去 hook 真实 tokenizer

### **必避**

- ❌ 不要在 C++ 里实现 tokenizer / text normalizer（SentencePiece + WeText FST），它们是 CPU-bound + 库依赖重，放 Python 上层
- ❌ 不要在 runtime 里塞采样逻辑（temperature / top-k / nucleus），让 ONNX engine 内置 sampler（MOSS 的 local_fixed_sampled_frame 就这么干，省 90% C++ 代码）
- ❌ 不要新写 paged KV allocator——如果 fork 没有就用线性 KV

---

## 6. T4 端到端 smoke

### 最小 smoke 协议

```cpp
int main(int argc, char** argv) {
    Runtime rt(argv[1], maxSlots=1, maxSeqLen=512);
    auto guard = rt.beginRequest();
    
    std::vector<int32_t> dummyInputIds(1 * 17, 0);  // 1 token × row_width 17, all zero
    std::vector<int32_t> dummyAttnMask(1, 1);
    std::vector<float> pcmOut;
    
    rt.generate(guard.slot(), dummyInputIds, dummyAttnMask, /*cfg*/{}, pcmOut);
    
    writeWavStereoF32(argv[2], pcmOut);
    printf("SMOKE_OK samples=%zu\n", pcmOut.size());
}
```

成功标准：
- 不崩
- pcm_samples > 0
- WAV 文件能用 `soundfile`/`sox` 读
- TTFA 在 T2 trtexec 单步延迟预测的合理范围内（MOSS 预测 30-150ms，实测 97ms）

**注意**：dummy 输入下音频是噪声，sox 测 RMS > 0 + 非全静默就算通。**不要用 ASR 验证 dummy 输入**——那是 T5 任务。

### 调试链路

每次 smoke 失败按这个顺序排查：

1. Engine load 成功？（log "Loaded engine size: N MiB"）
2. ExecutionContext create 成功？
3. setTensorAddress 全 OK？（`Static dimension mismatch` → 回去看 T2 profile）
4. enqueueV3 返回 true？（false → 一般 shape 没 setInputShape 完）
5. cudaStreamSynchronize 不报错？
6. cudaMemcpy DeviceToHost 拿到非零数据？

MOSS 适配每个 bug 都在这 6 步里某一步。

---

## 7. T5 真实输入 + ASR 验证

这一步是 **T4 之后**，**T6 之前**。目的：用真实 input_ids 跑出可懂语音，喂同语言 ASR 验内容字错率。

### Python 端拼 input_ids

照 T0 侦察出来的 prompt 结构，写 Python helper：

```python
def build_input_ids(text: str, ref_audio_codes: np.ndarray | None) -> np.ndarray:
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load("tokenizer.model")
    text_ids = sp.EncodeAsIds(normalize_text(text))  # WeText FST
    
    rows = []
    rows.append([im_start_id] + [audio_pad_id]*16)
    if ref_audio_codes is not None:
        rows.append([audio_start_id] + [audio_pad_id]*16)
        for frame in ref_audio_codes:
            rows.append([audio_pad_id] + frame.tolist())
        rows.append([audio_end_id] + [audio_pad_id]*16)
    for t in text_ids:
        rows.append([t] + [audio_pad_id]*16)
    rows.append([im_end_id] + [audio_pad_id]*16)
    rows.append([audio_assistant_slot_id] + [audio_pad_id]*16)
    
    return np.asarray(rows, dtype=np.int32)
```

dump 成 `.bin`，C++ smoke 读 stdin / `fread`。

### ASR 验证流程

```bash
# 生成 wav
./moss_tts_smoke ./bundle ./input_ids.bin ./out.wav

# 喂 ASR
curl -X POST http://orin-nx:8080/asr \
  -F "audio=@./out.wav" -F "language=zh"
# 期待响应 text 与输入 text 字符差异 < 10%（容忍一些音素错）
```

**注意**：dummy zero 输入跑出的 wav，ASR 会回 "嗯..." 之类无意义字符——这是符合预期的，**别误以为是 bug**。

### Voice clone 验证

把 zh_1.wav（5s reference）codec_encode 出 audio_codes，拼在 input_ids 前缀。跑出来的 wav 应该音色匹配（人耳听 + 简单 speaker embedding cosine 验证）。

---

## 8. T6 服务化

完整 mirror 现有 production backend：

| Qwen3_TTS 组件 | 新 backend 对应物 | 工作量 |
|---|---|---|
| `qwen3_tts_worker` C++ stdin/stdout | `<model>_tts_worker.cpp` | 2 PD |
| `Qwen3TRTBackend` Python class | `<Model>TTSBackend` | 1.5 PD |
| Registry + profile JSON | 注册 `"jetson.<model>"` | 0.3 PD |
| Artifact bundle + manifest | `/opt/models/<model>/` 布局 | 0.5 PD |
| HTTP smoke + ASR | bench/perf 脚本 | 0.7 PD |
| Docker / compose | env vars + Dockerfile 路径 | 0.5 PD |

具体接口在 `docs/runbooks/tts-phase-b-deployment.md` 和 `server/core/tts_backend.py`。

---

## 9. RK NPU 移植对位

我们做 Jetson 适配的经验对 RK 大部分能直接用，关键差异：

### 直接复用

- ✅ T0 侦察（Python 源码读法）
- ✅ T1 ONNX patch（FP16 + delta KV + If-rank surgery）
- ✅ T5 真实输入测试 + ASR 验证流程
- ✅ T6 服务化框架（Python backend / HTTP layer 不变）
- ✅ smoke 协议 / 故障排查 6 步法

### 需要改写

- **T2 编译**：用 `rknn-toolkit2` 替代 trtexec
  ```python
  from rknn.api import RKNN
  rknn = RKNN(verbose=True)
  rknn.config(target_platform='rk3588', dynamic_input=[[[1,1,17],[1,32,17],[1,128,17]]])
  rknn.load_onnx(model='prefill.onnx')
  rknn.build(do_quantization=False)  # FP16
  rknn.export_rknn('prefill.rknn')
  ```
- **T3 C++ runtime**：换 RKNN C API
  - `rknn_init` / `rknn_inputs_set` / `rknn_run` / `rknn_outputs_get`
  - 没有 `setTensorAddress` 这种 zero-copy 直接绑指针的 API，每次必须 memcpy（**这是 RK 性能上限的硬约束**）
  - 没有 CUDA stream / 异步——所有调用 sync
  - 没有 Execution Context per slot 概念，每个 model handle 是单线程
- **T2.5 算子兼容性**：RKNN 算子覆盖 < TRT 算子覆盖。建议提前跑一遍 `rknn.list_inputs/outputs()` 在 toolkit2 PC 端模拟，不通过的算子要 ONNX 替换或 fallback CPU。常见 fallback 算子：`Where`, `ScatterND`, 复杂 `If` 控制流。
- **T2.6 量化**：RK 主打 INT8。如果做 INT8 quant，需要 calibration dataset（100-500 条 utterance）。FP16 也可以，但 NPU 算力打折。

### RK 特有的坑（来自 device-gotchas）

参考 `~/.claude/skills/device-gotchas/gotchas-rk3576.md` 和 `gotchas-rk3588.md`：

- RKNN 模型加载耗内存比 TRT 高（含完整算子图 JSON）
- `rknn_run` 阻塞，无法 cancel——上层 worker 要 spawn 子线程包一层
- ZeroCopy mode 需要 `--core-mask=0x01` 单核绑定，多核会 race
- KV cache 必须用 zero-copy buffer，否则每步 memcpy 几十 MB 性能崩

### 工作量重估（参考）

| 阶段 | Jetson 实际 | RK 估算 |
|---|---|---|
| T0 侦察 | 1 d | 1 d（同步） |
| T1 ONNX patch + 重导出 | 1.5 d | 1.5 d（同步） |
| T2 编译 engine | 1 d | 2 d（多个 fixed-shape engine 切换 + 算子兼容性调） |
| T3 C++ runtime | 5 d | 6 d（RKNN API 没 TRT 灵活，多 memcpy） |
| T4 smoke | 0.5 d | 1 d（缺 cuda-gdb 这种工具，debug 慢） |
| T5 真实输入 + ASR | 0.5 d | 0.5 d（同步） |
| T6 服务化 | 5 d | 5 d（同步） |
| **合计** | **14 d** | **17 d** |

RK 比 Jetson 多 ~3 PD，主要在 T2 + T3。

---

## 10. 通用经验（无关后端）

### 委托模式

- **不要让 codex 给你"凭空"代码**——它会捏造 anchor 行号、tensor 名、API 签名。让它先**读真实文件**（用本地 vendor/ 缓存），再给 diff。
- **每次派 codex 都给完整上下文 + 真实文件路径**，不要说"基于先前的 spec"——上下文丢失。
- **派执行体（general-purpose）做 build/deploy 时必带红线 + EVIDENCE 模板**（CLAUDE.md 里有完整版）。

### 增量验证

- **每个 chunk 写完先编译 .o**，再 link。MOSS 4 个 chunk 全 .o 编译干净后再做端到端 smoke，每次错都局部化在最近一个 chunk。
- **不要写 1000 行后第一次跑**。

### Memory 沉淀

- 每次踩坑后立刻写 memory，含**根因 + 修法 + file:line**。下次再碰类似 bug 直接 grep memory。
- MOSS 适配产了 8 条 memory（recon、trtexec、chunk1、e2e_done...），下次回来 5 分钟能续上。

### 工具链

- `fleet transfer` 设备间直传，不走 Mac 中转（关键路径上能省 10 min）
- `fleet ssh device -- "while pgrep -f trtexec > /dev/null; do sleep 30; done"` 让远端阻塞 SSH 替代 polling
- Python `onnx.load(path, load_external_data=True)` + `onnx.checker.check_model()` 是改 ONNX 后的必跑检查
- `nvinfer1` Python binding (`tensorrt` 包) dump engine I/O 是 T2-T3 调试最快路径

---

## 10.5 KV buffer dtype ABI mismatch（2026-05-24 MOSS-TTS-Nano case study）

**症状**：frame-0 token 与参考 (ORT) byte-identical，frame ≥1 drift。ASR 验到「trailing-token hallucination / EOS-like 截断」之类 model-quality 现象。

**根因**：runtime 里 KV 缓冲区（`past_key_*`、`past_value_*`、`present_key_*`、`present_value_*`）字节数硬编码 `sizeof(half)`。当 ONNX/engine rebuild 把 KV IO 切到 FP32（例如为了追另一个 divergence 假说）时，buffer 实际 half-size → 各层 KV write 互相覆盖 → frame 1 之后污染。

**为什么很难发现**：
1. frame-0 的 audio sample 是从 prefill 的 `globalHidden` 算出来的，这是一个**独立**的 device binding，不走 KV scratch。所以 RNG-fixed 复现验出来 frame-0 是 byte-equal，给人「runtime 是对的」的错觉。
2. 污染不会 NaN / 不会 crash，只是 token 分布飘向模型先验的「常见尾段」（中文里就是大量重复字），看着像 EOS hallucination。
3. 几轮试错都在「肯定是 attention mask / RoPE / sampling / paged-KV 布局」转——因为这些是 TTS 移植教科书里 frame-drift 的常见 root cause。

**怎么 5 分钟内排查这一类问题**：
- 在 `mDecodeEngine->getTensorDataType("past_key_0")` 查一次，跟 runtime 里假设的元素大小对一遍。
- 在 runtime 启动日志里 *永远* 打一行 `[runtime] KV element dtype=<int> size=<n> bytes`。这一行能避免 1 周的迷路。
- grep runtime 源码里 `sizeof(half)` / `sizeof(float)` 等 dtype 常量——如果它们用在和 engine IO 关联的 buffer 上，应该被 `engine->getTensorDataType()` 探测替换。

**修复模板**：
- 加 helper：`size_t tensorElementSize(nvinfer1::DataType dtype)`（kFLOAT→4, kHALF→2, kINT32→4, kINT8→1, kBOOL→1）。
- 在 `loadEngines()` 里，所有需要算 KV/hidden buffer 大小的 dtype 都从 engine 探测一次，存成 member（`mKvElementSize`、`mGlobalHiddenDtype` 等）。
- 全文 grep 把所有 `sizeof(half)` 等硬编码替换成 member。
- 打 log，回归测：frame-0 byte-equal、frame ≥1 ASR CER 应同时通过。

**详细 case study + 6 轮 diagnostic trail**：`an internal engineering spec (not in the public mirror)`。

---

## 11. 反模式（不要做）

- ❌ **猜测式修复**：错了不查根因，把可能的修改全试一遍——次次都更糟
- ❌ **多目标合一次派**：让 codex 同时改 ONNX + 写 C++ + 设计协议——它什么都做不完
- ❌ **绕过 stop-criteria**：T2 没验通就开始 T3——会浪费 T3 的所有工作量
- ❌ **过度配置**：profile 一上来就支持 N=8 + voice clone + 多语言——MVP 单 client 跑通再加
- ❌ **不读 git history**：现有 backend 已经踩过的坑，不要再踩第二次（参考 [[tts_n2_phase_b_stability_landed]] 等 memory）
- ❌ **runtime 里硬编码 dtype 元素大小**：永远从 engine IO 探测，否则 engine rebuild 一换精度就会重蹈 MOSS KV ABI mismatch（[[moss_tts_nano_trt_production_ready]]）

---

## 12. 快速 checklist（移植任何新 TTS 时打印贴墙）

```
[ ] T0  读官方 Python 推理代码，画流程图
[ ] T0  侦察报告 commit 进 memory
[ ] T1  ONNX patch 用 unified diff，dry-run 验证
[ ] T1  ORT smoke 在 dev GPU 机器跑通（不在目标设备）
[ ] T2  trtexec / rknn build 显式 --minShapes/--optShapes/--maxShapes
[ ] T2  build log 落盘 + Python API dump engine I/O 验证
[ ] T2  单 engine trtexec --loadEngine 拿稳态延迟
[ ] T3  Mirror 现有 backend 的 slot lifecycle，不自己发明
[ ] T3  每写一个文件 .o 编译验证，不要堆完再 link
[ ] T3  hasTensor() 守每个 setInputShape / setTensorAddress
[ ] T4  Dummy zero smoke 跑出 WAV（sox 非全 0）
[ ] T5  Python tokenize + 真实 input_ids dump bin
[ ] T5  ASR 字错率 < 15% 算通过
[ ] T6  HTTP /tts + /tts/stream + /tts/clone 都通
[ ] T6  bench script + ASR 验内容自动化
[ ] T6  Docker compose + Dockerfile 提供 env vars
[ ] *   每个里程碑 commit + memory 沉淀
```

---

## 13. 相关项目内 memory 索引

- [[moss_tts_nano_port_recon]] — T0 侦察示例
- [[moss_tts_nano_trtexec_orin_nx]] — T2 实测延迟
- [[moss_tts_nano_kernel_chunk1_done]] — T3 增量 build 验证
- [[moss_tts_nano_smoke_e2e_done]] — T4 端到端 milestone
- [[trt_edge_llm_fork_path]] — fork 位置（RK 不适用）
- [[tts_n2_phase_b_stability_landed]] — slot lifecycle / per-slot scratch 模式
- [[trt_edge_llm_tts_env_staleness]] — module-level env 陷阱（Python backend 写法）

---

**写完这份 playbook，是希望下一个移植不用再花 14 PD。把 checklist 当 stop-criteria 用，不要跨阶段试错。**
