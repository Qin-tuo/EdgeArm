# 框架检查与不更换内核的运行方案

检查日期：2026-10-10 至 2026-10-12。主机为 CM3588 V2 / RK3588 / ARM64 / Ubuntu 22.04.5，内核 6.1.118；项目使用 venv + pip。原生部署可绕开容器限制；完整机器人链路尚未验收。

## 检查结果

| 层级 | 发现与处理 | 当前结果 |
|---|---|---|
| Python | 补齐 ASR 的 tokenizers 和默认 RK 后端的 Lite2 安装声明，锁定其依赖 | tokenizers 0.23.3、Lite2 2.3.2，pip check 通过 |
| Agent token 预算 | 可选 transformers 未安装，计数回退为 char×1.5；固定前缀 5 token 下历史预算为 995 | 基本对话通过，精确上下文边界尚未验收；不能盲目按启动日志建议将输入预算提高到 2005，worker 总上下文仅 2048 |
| 原生库 | 检查 PortAudio、libsndfile、libgomp、libstdc++、espeak-ng；语音 Runtime 严格校验 | 本机可加载，ASR/TTS 实际推理通过 |
| Docker 工具 | 原安装脚本缺少 Buildx | 已安装并固定 Buildx v0.38.0 ARM64 下载哈希 |
| NPU 监控 | debugfs 权限错误令正常语音服务的 readyz 返回 503 | 已处理权限异常并仅从 NPU devfreq 回退；实测 readyz 200 |
| LLM 故障处理 | 断链仍可能显示就绪，空回复可能返回成功 | 已撤销故障就绪状态；空回复/worker 错误返回 502，流式返回 error |
| Compose 健康 | 原检查只看 HTTP 200 | 已检查 ASR/TTS 布尔状态和 LLM status |
| RK1828 普通推理 | EFFICIENT/EFFICIENT 的此前传输样本通过；其他三组模式复现断链，两次无 SMI 对照首条 31-token 请求失败 | SMI 操作不是必要触发条件；根因、冷上电、长期稳定性及答案质量仍未验收 |
| LLM token 统计 | 项目非流式 API 的 usage 固定为 0，流式 completion 数按输出片段计数 | 不能据此评估真实 token 用量；本次测试从 worker 日志关联实际 prefill/decode 计数，精确 API 统计仍待实现 |
| RK1828 工具调用 | 配套 V1.1.0 worker 上，set_function_tools 返回 0 后仍 ERROR_PIPE，设备 proxy 消失 | 原生入口默认拒绝 tools，实际 HTTP 400 后 health 保持 200；根因未解决 |
| RK1828 服务恢复 | restart 的 DDR 请求曾失败；重启主机可恢复 proxy 与模型初始化 | 初始化恢复不能证明真实推理可用，仍会在首轮或多轮断链 |
| 页面文字输入 | 原 send_text 只转发给 TTS，未调用 agent 的 LLM | 当前 client-loop 已改为正常对话入口，处理休眠唤醒和重叠请求 |
| 模型版本 | 本机 Qwen3 为 V1.1.0，默认清单为 V1.0.4 | 本机使用独立配套清单；不能混用四个导出文件 |
| 机械臂依赖 | 已安装 feetech-servo-sdk 1.0.0 / pyserial 3.5 并验证导入 | 真实标定、串口和运动未验收 |
| 机器人闭环 | D435i/MCU 未接入，GraspBackend.plan/verify 无默认实现 | 仍需开发和实物验收，安装依赖不能补齐 |

10 月 12 日推送前软件回归：**1257 passed、2 skipped**，另有 5 条 FastAPI 生命周期弃用警告；依赖、Docker/Compose 配置、待提交文件的产品审计与文档链接检查通过。工作区直接运行 `scripts/check.sh` 时，链接审计被一份未纳入提交的本地规划文档中的外部电脑路径阻断；因此单独运行完整测试套件，并将产品审计的文档范围限制为 Git 索引文件。软件检查不启动容器或验证外设。新增回归覆盖 debugfs 权限、误用 GPU/DDR 状态、LLM 断链/空回复、合法工具回复、Compose 健康判断、服务启动等待、页面文字对话以及诊断失败后停止请求。

最近基本推理报告：`.local/reports/model-smoke/20261010T064816541144Z/report.json`。LLM 正常中文回复；TTS 为 16 kHz、2.72 秒、RMS 0.141 的 WAV；ASR 回读“你好，今天天气真不错”，CER 0。该报告生成于后续工具调用故障之前。

工具调用复现日志位于 `.local/reports/inference-check/native-llm-tool-failure.log`；请求结果为 `llm-protocol.json`，故障后状态为 `health-after-tool-failure.json`。依赖和内核检查分别为 `dependency-audit.json`、`doctor-framework.json`。这些证据仅存在于本机。

2026-10-10 15:24 页面文字对话复测：同一 `/api/control/send_text` 接口进入 LLM 并收到 token、TTS 音频和对话完成事件，音频 58368 字节，播放设备自动选择 HDMI。LLM 请求与设备 proxy 保持正常。记录为 `.local/reports/inference-check/page-dialogue.json`、`native-agent-dialogue.log`。

该次算术问题“17乘23等于多少？只回答数字。”得到 `401`，正确答案为 `391`。报告分别记录 `pipeline_passed=true` 与 `answer_check.passed=false`，整项 `passed=false`。这验证了调用链，也揭示回复准确性问题；不代表模型质量通过。文本输入跳过 ASR，因此仅观察 RK3588 的 ASR 负载不能判断 LLM 是否工作。

新版 worker 与工具拦截启用后再次恢复语音服务与 agent。最新基本模型报告为 `.local/reports/model-smoke/20261010T080755620837Z/report.json`：LLM 非空中文回复；TTS 16 kHz、2.72 秒非静音 WAV；ASR 回读 CER 0。16:12 页面 `/api/control/send_text` 实测得到“你好！”，收到 LLM token、37888 字节 TTS PCM 和 `on_assistant_done`，LLM health 保持 ok。证据为 `.local/reports/inference-check/page-dialogue-restored.json` 和 `native-agent-restored.log`。页面仍监听 `127.0.0.1:18000`；麦克风自动选 ES8389、扬声器选 HDMI，USB 麦克风与实际听音未验收。此次后台 speech/agent 进程不会自动随主机重启恢复。

## 原生启动

启动脚本沿用所选模型、端口和 Python 环境，无需启动 Docker。必须已有校验通过的原生 Runtime 和模型，Git 克隆不会提供这些大型文件：

- speech：`.local/runtime/speech/MANIFEST.json`、`librknnrt.so`、`librkllmrt.so`；本机由锁定 ARM64 speech 镜像提取，保留哈希校验。
- LLM：默认 `.local/runtime/rk1828-v1.1.0/rknn_qwen3_demo`、完整 `lib/` 和哈希清单；`.local/data/llm/artifacts.local.json` 对应本机 V1.1.0 四文件导出。旧 `.local/runtime/rk1828` 仅保留诊断证据，不再作为默认入口；它按 1.0.4 SDK 编译，替换双客户端库不能保证 ABI 配套。
- 模型目录：`.local/data/asr`、`.local/data/tts`、`.local/data/llm`，来源见 [模型说明](model-storage.md)。

首次准备 Python 环境，无需运行 setup-tools：

```bash
cd /userdata/EdgeArm
scripts/setup-dev.sh
source scripts/activate-env.sh
```

恢复 RK1828 并确认 Online、Health OK、父/设备 proxy 均存在后，在不同终端启动；已占用相同端口时先停止旧启动终端：

```bash
# 终端一；当前 RK1828 设备权限要求 root。
sudo env EDGEARM_RK1828_PREFILL_MODE=efficient scripts/start-native-llm.sh
```

```bash
# 终端二；从默认 profile 生成本机路径，严格校验 Runtime。
scripts/start-native-speech.sh
```

```bash
# 终端三；基本模型检查通过后再启动语音 agent。
source scripts/activate-env.sh
python scripts/test-models.py --services-only
scripts/start-native-agent.sh
```

原生 agent 默认关闭模型工具调用及机械臂连接，仅用于普通语音对话。原生 LLM 服务同时默认设置 `RK1828_ALLOW_TOOLS=0`，在访问 worker 前对带 tools 的请求返回 HTTP 400；`RK1828_KV_REUSE=0` 对应本轮普通推理基线。调试页监听本机 18000 端口，可使用 SSH 转发访问。启动时最多等待 180 秒，只有 ASR/TTS 与 LLM 均就绪才启动 agent；超时会列出未就绪的服务和端口。可用 `EDGEARM_SERVICE_WAIT_S=300 scripts/start-native-agent.sh` 调整等待时间。录放音尚需实测；脚本语法及配置加载通过不等于整机验收。

`scripts/start-native-agent.sh --prepare-only` 只生成 `.local/deploy/agent-native.yaml`。只有真实 tools 请求验收后才能同时启用 LLM 的 `RK1828_ALLOW_TOOLS=1` 与 agent 的 `EDGEARM_NATIVE_TOOLS=1`；当前两者均关闭。机械臂连接仍关闭，真实动作按 runbook 单独使能和验收。

speech 支持 `EDGEARM_NATIVE_RUNTIME` / `EDGEARM_NATIVE_DATA`；LLM 支持 `EDGEARM_NATIVE_LLM_RUNTIME` / `EDGEARM_NATIVE_LLM_MODELS` / `EDGEARM_NATIVE_LLM_MANIFEST` 覆盖路径。sudo 下使用 `sudo env NAME=value ...` 显式传入。

## RK1828 恢复与修复

先停止 LLM 启动终端，确认无其他 RK1828 任务，然后在管理员终端恢复设备：

```bash
sudo rknn-smi reset -t hw -d 0
sleep 5
sudo systemctl restart rk182x.service
sleep 10
sudo rknn-smi info -t board -d 0
ps -eo pid,args | rg 'rknn3_transfer_proxy'
```

现有服务会重新加载已安装的板卡固件并启动 proxy，没有更换系统内核。仅设备 Online、健康正常且设备 proxy 存在时才重新加载模型。复位无效时可能仍需整机重启或断电恢复；这与重刷内核不同。本轮未执行上述恢复命令。

版本配套须用 **V1.1.0 headers、Model Zoo、库** 在匹配 Ubuntu 22.04 ABI 的环境重新编译 server-mode worker，方法见 [构建说明](../services/rk1828-llm/BUILD.md)。不能只替换 `.so` 就宣称兼容。

本机已在独立目录编译 V1.1.0 worker，SDK revision 为 `174e44c77230735b1458946debb62b3982c1ee58`。产物和哈希在 `.local/runtime/rk1828-v1.1.0/MANIFEST.json`，构建日志在 `.local/reports/inference-check/worker-v1.1.0-build.log`；动态依赖可加载，两份 RKNN3 库与主机哈希一致。原 worker 未覆盖，新版已成为原生默认入口。

新版关闭 KV reuse 后两轮各 8 个普通请求均正常完成，实际 decode_tokens 为 2–159，速度约 55–58 token/s，未再出现 8192 的异常统计。这只能验收本轮样本，不能证明长期稳定。两轮均出现特定算术问法答 `401`（应为 `391`）、历史中的“小林”答“小智”。当前 Python shim 将历史与 system 文本展平为单个用户输入，角色语义与模型质量均需进一步排查，不能将错误仅归因于量化。

**16:30 更新：普通多轮仍失败。** 用户在页面依次输入“你好”“你是谁”“什么模型”，第三句对应 worker req#13：prompt 320 字节、prefill_tokens=83、首 token 时间为 0、decode_tokens=0，随后 ERROR_PIPE、设备 proxy 消失，health 503。上一轮为 prefill_tokens=51。实际进程环境确认 `RK1828_KV_REUSE=0`、`RK1828_ALLOW_TOOLS=0`，maps 确认加载独立目录中的 V1.1.0 两份库，不能再将普通对话失败归因于配置回退或仅限 tools。

已停止诊断 agent 并保存完整历史、原始 320 字节展平输入、请求 JSON、应用/内核日志、进程环境与实际库映射：`.local/reports/inference-check/plain-failure-20261010-163011/`。主机当前约 10 GiB 可用内存，采集到的故障时段内核日志未出现 OOM、PCIe/AER 或 Oops 记录，故障后 root rknn-smi 无法初始化。这些记录不能排除设备端故障或供电问题；用户说明通过底板 12V 供电，电流及板卡电源状态未验证。51→83 token 的变化仅为 prefill 输入长度线索，尚未证明某个阈值故障。

**17:17 厂商原版对照也失败。** 17:12:44 主机重启后，仅启动同 SDK 未修改的 `rkllm3-server` 与配套双库，未运行 speech、agent 或项目 Python worker。文件哈希验证通过，仅监听本机 1829，保持 EFFICIENT / PERFORMANCE、8 核、2048 上下文和 512 输出上限。第一个请求重放原始 320 字节展平输入，厂商日志同样显示 `n_prompt_tokens = 83`，随后出现 `ERROR_PIPE (-9)`；设备 proxy PID 360 消失，父 proxy PID 338 和 HTTP listener 仍存在，请求在 60 秒后超时。只发送这一次请求，结构化历史与连续增长会话测试均未执行。

证据为 `.local/reports/inference-check/vendor-llm-replay.json`、`vendor-llm-v1.1.0.log` 和 `.local/runtime/rkllm3-server-v1.1.0/MANIFEST.json`；诊断入口为 `.local/diagnostics/start-vendor-llm.sh`。这说明相同故障可在厂商路径复现，无需页面、多轮累积、项目 IPC 或 tools；尚不能确定模型编译、Runtime、EP 固件或硬件中的哪项是根因，也不能据此排除项目存在其他问题。83-token 输入长度仍只是线索，未验证长度阈值或不同 prefill 模式。HTTP 端口继续监听不能证明设备健康。

本次 root 日志与实际库映射采集脚本为 `.local/diagnostics/collect-vendor-llm-failure.sh`，写入独立 `vendor-failure-20261010-171726/` 目录，避免覆盖 16:30 证据。该脚本只读，不复位、不重启、不调用 SMI 或发推理请求；须在恢复设备之前运行。

用户已完成采集：`process-3031-runtime-maps.log` 确认实际加载暂存目录内的两份 V1.1.0 库；启动日志有 `Downloading firmware OK`，故障后只剩父 proxy 和 HTTP 服务。主机可用内存 14,421,752 kB，17:16 后采集的内核日志 2264 行无 OOM、PCIe/AER、Oops 或温度告警匹配。`/var/crash` 未发现可读取的崩溃报告，未取得设备端 crash dump，因此退出原因仍未知。厂商记录的 `1__3_token_in.txt` 保存完整 83 个输入 token。复现材料压缩包为 `.local/reports/inference-check/rk1828-vendor-repro-20261010.zip`，包含输入、命令、版本/哈希和本次日志，不包含模型权重。

**EFFICIENT prefill 临时对照通过本轮样本。** 用户允许临时修改后，17:27:43 重启主机，并用 `.local/diagnostics/start-vendor-efficient-prefill.sh` 启动同一厂商服务。work 和 prefill 均回读为 EFFICIENT，其他模型与启动参数保持一致。原始 83-token 故障输入成功返回，完整角色历史为 114 token 也通过；随后 8 轮真实增长对话、22 轮增长历史和 1 个完整 SSE 响应全部完成，共 33 次请求。最大输入为 1523 token；SSE 有非空内容、`finish_reason=stop` 与 `[DONE]`，结束后 health 200 且设备 proxy PID 370 保留。实际姓名回忆正确，但 17×23 仍答 401，应为 391；传输正常不能代表答案质量合格。

设置/回读记录为 `vendor-mode-efficient.log`，请求报告为 `vendor-efficient-prefill-replay.json`、`vendor-efficient-history.json`，汇总为 `prefill-mode-comparison.json`，均在 `.local/reports/inference-check/`。PERFORMANCE 原故障包与 EFFICIENT 本轮证据已合并为 `rk1828-prefill-comparison-20261010.zip`。

本轮结果支持故障与 PERFORMANCE prefill 路径有关，但还不能区分固件、编译图、运行库或模式相关硬件负载，未证明长度阈值。`scripts/setup-rk1828-mode.sh` 支持显式 `EDGEARM_RK1828_PREFILL_MODE=efficient`，并阻止在厂商服务或其他 worker 运行时修改模式；PERFORMANCE 仍是未显式设置时的默认值，未更改用户的正式默认偏好。

**项目 worker 的 EFFICIENT prefill 对照也通过本轮传输样本。** 停止厂商服务后，直接用 `.local/diagnostics/start-matched-efficient-prefill.sh` 启动配套项目 worker，没有再次重启板子或板卡服务。日志确认两种模式均为 EFFICIENT、KV reuse 关闭，并通过 Runtime 和模型哈希校验。41 次请求包括原始 83-token 输入、8 轮对话、30 轮增长历史和 1 个完整 SSE，设备 proxy 始终保留；汇总为 `matched-efficient-summary.json`，明细为 `matched-efficient-replay.json` 与 `matched-efficient-history.json`。

测试也发现项目非流式 API 的 usage 是固定 0 占位值，使原定的 1500-token 停止条件未触发，测试实际完成了固定上限 30 轮。已将每条请求按顺序关联 worker 日志计数，最大 prefill 为 2052；该结果不能验收配置为 2048 的实际上下文边界。后续测试已增加独立的 22 轮上限，不再单独依赖 API 的用量统计。项目的角色展平仍存在：第 5 轮姓名问题答“通义千问”，算术问题复述了题目，第 8 轮姓名才答对；不能将传输样本通过当作模型质量通过。

随后恢复原生 speech 与 agent。最新基本模型报告为 `.local/reports/model-smoke/20261010T094424489435Z/report.json`：LLM 非空中文回答，TTS 为 16 kHz、2.72 秒非静音 WAV，ASR 回读 CER 0。页面测试 `page-efficient-prefill.json` 共 5 轮，其中完整重放原故障之前的预热招呼及“你好”“你是谁”“什么模型”；第 4 轮 worker req#46 再次为 320 字节、83 token，这次成功返回并完成 TTS 和 `on_assistant_done`。五轮都有回答与 PCM 音频，结束后 LLM health 为 ok、speech ASR/TTS 为 true、页面 HTTP 200，设备 proxy PID 370 存在。麦克风为 ES8389、输出为 HDMI，真实听音与 USB 麦克风仍未验收。随后为厂商日志复现停止了这些应用；它们不随主机重启自动恢复，日常启动采用 runbook 中显式 EFFICIENT prefill 的临时命令。上述对照和基本模型结果已整理为 `.local/reports/inference-check/rk1828-complete-checks-20261010.zip`。

**18:06 厂商要求的双端日志复现成功。** 控制台执行 `uart_irq_set_owner subsoc`、`rknn_set_log_level 3`、`rknn_server_set_log_level 2`；主机应用导出 `RKNN_LOG_LEVEL=3`。同一配套 V1.1.0 worker、Qwen3-4B 模型、8 核、2048 上下文及关闭 KV reuse/tools，回读 work=EFFICIENT、prefill=PERFORMANCE。前三次预热/对话输入 31、40、51 token 均返回 HTTP 200；第 4 次“什么模型”输入 320 字节、83 token，18:06:28 报 ERROR_PIPE (-9)，零输出，HTTP 502。复现程序立即停止后续请求。

设备端完整日志最后记录 `rknn3_run` 输入 `Tc=83`，随后控制台持续出现 `PCIe receive length (4294967295) exceeds buffer size (256)` 及异常字节；未捕获明确 EP panic/assert。请求失败当时设备 proxy PID 370 尚短暂存在，结束终端后复查只剩父 proxy PID 348，因此进程存在也不能证明通信正常。主机日志未发现明确 OOM/AER/Oops，根因仍未知。日志级别显著影响推理速度，本次耗时不能用于性能验收。

完整原文、请求/响应、失败输入、系统日志、模型/Runtime 哈希及采集源码在 `.local/reports/inference-check/rk1828-manufacturer-logs-efficient-performance-20261010.zip`；原始目录为 `vendor-logs/20261010-175953-efficient-performance/`。模式准备与采集入口已纳入 [runbook](runbook.md)。其余模式的“无法推理”先作为用户报告记录，再按下面实测更新；不要将未测试组合列为实测失败。

**10 月 12 日 09:50，NORMAL / EFFICIENT 也取得双端故障日志。** 主机重启恢复后，设置并回读 work=NORMAL（1）、prefill=EFFICIENT（0），其余模型、Runtime、8 核、上下文和请求参数不变。控制台确认 RKNN 日志 3、server 日志 2，主机应用 `RKNN_LOG_LEVEL=3`。前四次 31/40/51/83-token 请求均成功，原先 EFFICIENT/PERFORMANCE 的 83-token 输入此次返回 29 个 token；第五次“请用一句话说明语音助手如何工作。”实际输入 129 token，09:50:28 报 ERROR_PIPE (-9)、零输出、HTTP 502，程序立即停止。

设备端最后记录本次 prefill 的 `Tc=128`，随后出现同类 PCIe 接收长度 4294967295 告警和异常字节；未捕获明确 panic，不能据此证明 129-token 阈值。停止两个终端后，仅父 proxy PID 350 留存，设备 proxy PID 373 消失。因此 NORMAL/EFFICIENT 本轮能推理短请求，实测为多轮增长后断连，并非首轮全部无法推理。完整日志为 `.local/reports/inference-check/rk1828-manufacturer-logs-normal-efficient-20261012.zip`，目录虽然 10 月 10 日准备，实际采集时间以原始日志中的 10 月 12 日为准。

**10 月 12 日 10:05，NORMAL / PERFORMANCE 首轮即断链。** 再次重启恢复，模式回读 NORMAL（1）/PERFORMANCE（1），同一模型、配套 Runtime、8 核、2048 上下文与双端详细日志。初始化在 09:58:24 完成，10:05 发出第一条“你好，请用一句简短中文打招呼。”；worker req#0 实际输入 31 token，10:05:05 报 ERROR_PIPE (-9)，零输出、HTTP 502。仅发送这一条，未进入多轮，程序立即停止。两端采集正常结束并保存系统日志，文件哈希及压缩包完整性通过。

本组日志为 `.local/reports/inference-check/rk1828-manufacturer-logs-normal-performance-20261012.zip`；三组汇总为 `rk1828-manufacturer-mode-logs-20261012-3-cases.zip`。PERFORMANCE/EFFICIENT、PERFORMANCE/PERFORMANCE 的双端采集仍待完成，EFFICIENT/EFFICIENT 也尚未补采同级别日志。模式矩阵为 `manufacturer-mode-matrix.json`；每一组分别记录实测结果，相同通信错误不代表根因已确认。

**10 月 12 日，无 SMI 对照两次均在首轮断连。** 按厂商提出的对照条件，分别在 15:24:07 和 15:43:32 主机重启后，使用独立诊断入口跳过 `setup-rk1828-mode.sh`，不执行 SMI 查询/设置，也不打开 rknn-console。第一次用户明确确认重启后未执行 SMI；第二次沿用相同无 SMI 操作要求。所检查的 `/bin/rknn3_startup` 开机脚本未包含 SMI 命令。未回读 work/prefill，实际模式保持未知，不据此推定出厂默认值。

两次都使用同一 Qwen3-4B 模型、配套 V1.1.0 worker/双库、8 核、2048 上下文、512 输出上限，关闭 KV reuse/tools，主机 `RKNN_LOG_LEVEL=3`。两次初始化均完成并输出 `READY`，健康接口返回 ok；随后第一条“你好，请用一句简短中文打招呼。”均为 31-token 输入、零输出、ERROR_PIPE (-9)、HTTP 502，清理 KV cache 又报 ERROR_NO_DEVICE (-4)，设备 proxy 退出。失败请求分别发生在 15:41:42、16:03:14–15，耗时 0.354 秒和 0.364 秒。第一次就绪后约 45.7 秒发请求，第二次约 2.0 秒；后续请求立即停止。

本轮证明：**在主机重启后，不执行 SMI 也能复现故障，SMI 操作不是此次故障的必要触发条件。模型可以初始化，但生成没有跑通。** 两次均为主机重启，尚未验证完全断电后的首次冷上电；也不能据此排除运行模式、固件、Runtime、模型或硬件因素。第一次日志为 `.local/reports/inference-check/rk1828-reboot-no-smi-attempt1-20261012.zip`，原始目录分别为 `vendor-logs/20261012-152724-reboot-no-smi/` 和 `vendor-logs/20261012-154844-reboot-no-smi-attempt2/`。本次没有设备控制台日志，避免在该对照中加入额外控制操作。后续模式测试暂停。

新版首次工具请求仍在 `set_function_tools ret=0` 后出现 `ERROR_PIPE`、设备 proxy 消失；HTTP 502、health 503。证据为 `.local/reports/inference-check/llm-v1.1.0-tools.json` 和 `native-llm-v1.1.0-tool-failure-20261010.log`。因此配套编译修复了统计与本轮普通请求运行条件，未解决 tools 断链，后续须由供应商排查 Runtime/固件。重启恢复并启用服务端工具拦截后，实际工具请求返回 400，health 保持 200，设备 proxy 保留，证据为 `llm-v1.1.0-tool-guard.json`。普通请求报告为 `llm-v1.1.0-first-run.json` 和 `llm-v1.1.0-repeated.json`；相关软件回归 **41 passed**。

15:43 的服务恢复失败日志为 `.local/reports/inference-check/rk1828-recovery-journal.log`：开机曾有 `Downloading firmware OK`，随后手动 restart 在 `Re-Init pcie boot` 后报 `check ddr load addr request failed`、`Device request state: 0xffa70000`，约五秒后退出 234。这是固件加载阶段的失败，尚不能据此判断硬件永久故障。不要反复 restart 或加载模型；先保存日志并重启主机，开机后直接检查设备状态。已成功加载时无需再 restart。重启仍不能恢复再按板卡恢复流程考虑冷断电，不能把重启或复位视为刷写内核。

服务启动失败时先读取日志：

```bash
sudo journalctl -u rk182x.service -b -n 100 --no-pager
```

主机恢复后，先确认设备 Online、Health OK 和设备 proxy 存在，再运行新 worker；服务 active 与设备节点存在都不能单独证明设备可推理。

15:50 主机重新启动后，`rk182x.service` 的 ExecStart 返回 0，父 proxy 与 `0000:01:00.0` 设备 proxy 均存在。用户通过 root `rknn-smi info -t board -d 0` 确认设备 Online、固件 1.1.0。此次重启恢复了服务，尚未证明后续连续推理稳定。

新 worker 暂存独立目录并记录源码/二进制/库哈希；恢复后依次验收普通请求、SSE、tools、多轮及重复请求，确认每轮结束后设备仍 Online。若一致 SDK 仍复现，交由供应商排查固件/Runtime。不要无限重试设备掉线，也不要未经验证改动用户指定的 EFFICIENT / PERFORMANCE 模式。

## Docker 边界

当前内核禁用了 OverlayFS、POSIX_MQUEUE、BPF_SYSCALL/CGROUP_BPF、MEMCG、CGROUP_PIDS 及 bridge/veth/netfilter 所需功能。实际顺序：overlay2 报 `no such device`；VFS 能启动 daemon，但容器报 mqueue 缺失；覆盖挂载后仍报 `bpf_prog_query(BPF_CGROUP_DEVICE): function not implemented`。privileged、host IPC 未解决；Compose 内存限制也不受支持。

`/proc/config.gz` 显示这些功能未编译，本机也无对应模块；pip、普通 modprobe 或更换 Docker 版本不能完整补齐。保留 VFS daemon 可获取镜像/提取文件，应用采用原生方式。

已准备的 [内核配置片段](../deploy/kernel/docker.config) 仅供未来配套 BSP 构建参考，未编译、安装或刷写。本次按用户要求使用原生方案。Docker 功能依据 [Moby 官方检查脚本](https://github.com/moby/moby/blob/master/contrib/check-config.sh)，Buildx 安装依据 [官方说明](https://github.com/docker/buildx#installing)。
