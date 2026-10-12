# EdgeArm 板端启动、模型测试与硬件验收

截至 2026-10-10，本机所选 ASR/TTS 资源及 Qwen3-4B 配套导出的 19 个文件已通过大小和 SHA256 校验。原生运行下的 ASR/LLM/TTS 基本推理检查已通过：LLM 返回正常中文并正确计算 17 × 23 = 391；TTS 生成 16 kHz、2.72 秒的非静音 WAV；ASR 回读“你好，今天天气真不错”，归一化字符错误率为 0。真实录放音、机械臂、D435i 和 MCU 接触链路仍待验收。

本次完整推理报告为 `.local/reports/model-smoke/20261010T063101348698Z/report.json`，相关响应和 WAV 位于同一目录；模式回读和单独 LLM 测试记录位于 `.local/reports/inference-check/`。这些是本机测试记录，不随 Git 克隆分发，单句检查也不代表完整质量或长时间稳定性验收。

**当前主机的启动限制：** BSP 内核 6.1.118 没有启用 OverlayFS、bridge/veth、POSIX 消息队列、内存 cgroup，实际容器启动还出现 `bpf_prog_query(BPF_CGROUP_DEVICE)` 功能缺失。下文标准容器方案尚未在本机通过；先补齐 BSP 的容器支持，再验收容器部署。VFS 测试配置仅解决 Docker daemon 启动，不解决后续容器运行错误。

**不更换内核的方案：** 使用仓库内的 `scripts/start-native-llm.sh`、`scripts/start-native-speech.sh` 和 `scripts/start-native-agent.sh`。原生 LLM 使用配套 V1.1.0 worker，关闭 KV reuse 并拒绝 tools；agent 关闭工具与机械臂。**PERFORMANCE prefill 下项目 worker 与厂商服务均在同一 83-token 输入断链；用户授权临时改为 EFFICIENT prefill 后，厂商服务 33 次、项目 worker 41 次请求及页面原始触发顺序均通过传输检查。** 当前采用显式 EFFICIENT prefill 作为临时绕过方式；长期稳定性、精确 token 统计及答案质量仍未验收。未指定模式的脚本入口仍默认 PERFORMANCE。检查结果、恢复步骤见 [框架检查报告](framework-audit.md)。

## 本机完整启动流程（原生）

本机使用 venv + pip，模型、配套 Runtime 和 Python 环境已准备好。日常开机无需重复安装依赖。首次准备环境时执行 `cd /userdata/EdgeArm && scripts/setup-dev.sh`；手动运行 Python 命令前可执行 `source scripts/activate-env.sh`。三个启动脚本会自行使用项目虚拟环境。以下服务保持前台运行，关闭相应终端会停止服务。

开机后先检查板卡服务和设备：

```bash
cd /userdata/EdgeArm
systemctl status rk182x.service --no-pager
sudo rknn-smi info -t board -d 0
```

确认服务 active、Device Status 为 Online、Firmware Version 为 1.1.0 后继续。板卡服务已启用开机启动；正常时不要再运行 `systemctl restart rk182x.service`。若启动失败，先保存 `sudo journalctl -u rk182x.service -b -n 100 --no-pager`，按 [检查报告](framework-audit.md) 恢复设备，再加载模型。

终端一：启动 LLM，保持运行：

```bash
cd /userdata/EdgeArm
sudo env EDGEARM_RK1828_PREFILL_MODE=efficient scripts/start-native-llm.sh
```

脚本验证 Runtime 和模型哈希，上述命令显式设置并回读 work=EFFICIENT、prefill=EFFICIENT，使用配套 V1.1.0 worker，关闭 KV reuse 和工具请求。看到 `READY 1` 与 `Uvicorn running on http://127.0.0.1:1828` 后继续。当前设备权限要求以 root 启动此服务。

这条命令是本机已复测通过的临时配置；移除环境变量会回到 PERFORMANCE prefill，当前模型在该模式仍会断链。先停止其他 LLM worker 或厂商服务，避免同时占用设备和更改运行模式。本机带日志的等价入口为 `.local/diagnostics/start-matched-efficient-prefill.sh`；短时样本不能替代长期稳定性验收。

终端二：启动 ASR/TTS，保持运行：

```bash
cd /userdata/EdgeArm
scripts/start-native-speech.sh
```

看到 `Speech service ready` 与 `Uvicorn running on http://127.0.0.1:8621` 后继续。语音 Runtime 来自锁定 speech 镜像，保留严格校验；无需额外启动 Docker 或用于远程 Toolkit 调试的 rknn_server。

终端三：可先执行一次基本模型检查，再启动 agent：

```bash
cd /userdata/EdgeArm
# 可选检查：必须在 agent 占用唯一语音会话之前运行。
.venv/bin/python scripts/test-models.py --services-only
scripts/start-native-agent.sh
```

模型检查应得到 `PASS model inference smoke`。agent 会自动等待 ASR/TTS 和 LLM 就绪，看到 `debug_dashboard listening on http://127.0.0.1:18000` 后即可访问。默认使用普通对话，关闭工具和机械臂。测试已启动的 agent 时使用页面流程，不要额外占用语音会话进行同一轮 HTTP ASR 检查。

在电脑上建立 SSH 转发并保持该终端打开：

```bash
ssh -N -L 18000:127.0.0.1:18000 ubuntu@172.16.51.112
```

浏览器打开 <http://127.0.0.1:18000>，可通过文字输入测试 LLM/TTS。语音输入使用板端连接的麦克风；当前自动选择 ES8389 输入和 HDMI 输出，USB 麦克风接入后可使用页面的“重启麦克风”重新打开输入流。真实听音与新麦克风仍需实测。

关闭时依次在 agent、speech、LLM 终端按 Ctrl+C。重启主机后重新启动这三个应用服务；目前仅板卡底层服务开机自启。已有服务在运行时不要重复启动，可用 `ss -ltn '( sport = :1828 or sport = :8621 or sport = :18000 )'` 查看端口。

本机 `.local/runtime/`、`.local/data/`、诊断脚本和生成的 profile 不随 Git 克隆分发；上述流程以现有配套资源为前提，构建和来源见 [LLM 构建说明](../services/rk1828-llm/BUILD.md) 与 [模型说明](model-storage.md)。

若只需启动本机 Docker daemon，使用已准备的独立 VFS 配置并保持前台运行：

```bash
cd /userdata/EdgeArm
sudo scripts/start-docker.sh .local/deploy/docker-test.json
```

文件完整性、服务就绪和真实推理分别判断。测试中曾出现 LLM `/health` 返回 ok，但请求返回空内容、日志出现 `ERROR_PIPE`，设备 proxy 随后消失。重启曾恢复短请求，EFFICIENT / PERFORMANCE 的真实多轮仍反复断链；当前只有显式 EFFICIENT prefill 的传输样本通过。不要把恢复后能启动视为故障已经解决，也不要在断链后反复发送请求。

## 厂商日志与模式复现

厂商要求同时保存 RK182x 控制台和 RK3588 的 `RKNN_LOG_LEVEL=3` 应用日志。先停止 agent、speech、所有 LLM 应用与旧控制台；每组使用独立目录，避免覆盖故障现场。以下入口以本机已校验的 V1.1.0 Runtime 和模型为前提，不安装或更换固件。

准备一组测试，例如 EFFICIENT / PERFORMANCE；该命令只创建目录和终端入口，不修改设备：

```bash
cd /userdata/EdgeArm
.venv/bin/python scripts/prepare-rk1828-log-case.py efficient performance
```

终端一执行下列命令并保持控制台，随后在控制台逐条输入厂商的三个命令：

```bash
sudo /userdata/EdgeArm/.local/diagnostics/start-manufacturer-console.sh
```

```text
uart_irq_set_owner subsoc
rknn_set_log_level 3
rknn_server_set_log_level 2
```

终端二启动应用，脚本设置并回读本组模式，自动导出 `RKNN_LOG_LEVEL=3`，将完整 stdout/stderr 写入 `rk3588-host.log`：

```bash
sudo /userdata/EdgeArm/.local/diagnostics/start-manufacturer-llm.sh
```

看到 `READY 1` 后，终端三执行一次复现：

```bash
cd /userdata/EdgeArm
read -r RK1828_CASE < .local/diagnostics/manufacturer-active-case.txt
.venv/bin/python scripts/reproduce-rk1828-modes.py "$RK1828_CASE"
```

程序先检查就绪状态，发送预热及真实增长历史，最多 16 次请求；使用 worker 实际 token 数限制历史，不依赖 API 的占位 usage。HTTP 错误、空回复、proxy 消失或初始化失败时停止，不重试、不复位。`requests.json` 保存实际请求、响应和失败位置；同一目录不能重复测试。详细日志会降低推理速度，不能用来衡量正常运行性能。

完成或发生故障后，在控制台按 **Ctrl+X** 结束原始日志记录，在应用终端按 **Ctrl+C**。应用采集脚本随后保存系统日志。确认目录内有 `rk182x-console.log`、`rk3588-host.log`、`requests.json`、`kernel-journal.log`、`proxy-journal.log` 和 `dmesg.log`，再恢复设备。断链后若只剩父 proxy，不继续下一组；此前热重启板卡服务曾在固件加载阶段失败，先保存日志，再重启主机恢复，开机正常时无需额外 restart 板卡服务。重启后重新打开控制台并输入三个日志命令。

两端采集都结束后，以普通用户打包并检查原始文件哈希；目录未完成、缺日志级别成功回显或包已存在时会拒绝覆盖：

```bash
.venv/bin/python scripts/package-rk1828-logs.py "$RK1828_CASE"
```

模式映射来自本机 SDK：work 为 `efficient=0`、`normal=1`、`performance=2`；prefill 只有 `efficient=0`、`performance=1`，共六种组合。下一组用不同参数重新执行准备命令，并确认没有旧模型/控制台进程。

| Work / Prefill | 截至 2026-10-12 10:07 的证据 |
| --- | --- |
| EFFICIENT / PERFORMANCE | 已按厂商日志要求复现：预热后第三句“什么模型”，83-token 输入，零输出、ERROR_PIPE、HTTP 502 |
| EFFICIENT / EFFICIENT | 此前厂商服务 33 次、项目 worker 41 次及页面 5 轮传输样本通过；尚未采集同级别双端对照日志 |
| NORMAL / EFFICIENT | 已按厂商日志要求复现：前四次请求成功（含 83-token 原故障输入），第五次 129-token 输入零输出、ERROR_PIPE、HTTP 502 |
| NORMAL / PERFORMANCE | 已按厂商日志要求复现：首条预热请求，31-token 输入，零输出、ERROR_PIPE、HTTP 502 |
| PERFORMANCE / EFFICIENT | 用户报告无法推理；本轮待测 |
| PERFORMANCE / PERFORMANCE | 用户报告无法推理；本轮待测 |

本机首组完整日志包：`.local/reports/inference-check/rk1828-manufacturer-logs-efficient-performance-20261010.zip`，包含两端原文、失败请求、版本哈希和采集源码。RK182x 最后记录输入 `Tc=83`，随后出现 `PCIe receive length (4294967295) exceeds buffer size (256)`；未取得明确 EP panic/assert，不能据此确定固件、编译图、运行库或供电根因。

NORMAL / EFFICIENT 日志包为 `.local/reports/inference-check/rk1828-manufacturer-logs-normal-efficient-20261012.zip`。该目录于 10 月 10 日准备，实际复现时间为 10 月 12 日 09:49–09:50。设备端最后记录 prefill 输入 `Tc=128`，主机总输入为 129 token，随后出现同类 PCIe 接收异常；不能据此认定固定长度阈值。该组合能够完成短请求，实测现象为增长历史后断连。六种组合的本机汇总为 `.local/reports/inference-check/manufacturer-mode-matrix.json`。

NORMAL / PERFORMANCE 在 10 月 12 日 10:05 的第一条预热请求中断链，主机实际输入 31 token、零输出、HTTP 502，尚未进入多轮。完整日志包为 `.local/reports/inference-check/rk1828-manufacturer-logs-normal-performance-20261012.zip`，前三组汇总包为 `.local/reports/inference-check/rk1828-manufacturer-mode-logs-20261012-3-cases.zip`。相同 ERROR_PIPE/PCIe 异常表示通信中断，仍不能直接确定这些模式是否有同一根因。

最终交付给厂商的三组原始日志及 Markdown 说明已合并为 `.local/reports/inference-check/rk1828-three-mode-logs-20261012.zip`。该包解压后按模式分目录，包含完整原文、请求、系统日志、源脚本、版本哈希及 `README.md`，不包含模型权重。

### 不执行 SMI 的对照结果

2026-10-12 又完成两次独立主机重启后的无 SMI 对照：启动前不查询/设置 SMI、不打开控制台，仅跳过原生启动流程中的模式设置，其余模型和 Runtime 参数保持一致。两次都能初始化到 `READY`，但首条 31-token 请求均零输出、ERROR_PIPE、HTTP 502，设备 proxy 退出。**无需 SMI 操作也能复现；初始化成功不等于推理成功。** 实际 work/prefill 未查询，不能认定其值；完全断电首次上电仍未验证，详情见 [检查报告](framework-audit.md)。

本机专用 `.local/diagnostics/start-no-smi-llm.sh` 绑定单次 boot ID 和独立日志目录，已经使用的入口会拒绝重试或覆盖。准备新的无 SMI 对照应复制并审核普通启动入口，跳过 `scripts/setup-rk1828-mode.sh`，保留模型/Runtime 哈希校验和单次初始化限制；不要直接运行普通 `start-native-llm.sh`，它会调用 SMI。设备断链后先保存日志再恢复，避免继续在故障状态下请求。当前测试已暂停。

## 1. 开发环境

原生 agent 启动时会等待两个模型服务就绪，默认上限 180 秒，超时列出服务与端口。可用 `EDGEARM_SERVICE_WAIT_S=300 scripts/start-native-agent.sh` 延长等待。启动脚本会拒绝重复启动；18000 端口已占用时先确认已有 agent，不要增加实例，否则会触发语音服务单会话限制。

当前 client-loop 下，页面文字输入已接入 agent 对话流程，而非直接朗读输入；可以从聊天历史核对 user/assistant 消息。文字输入跳过 ASR，语音输入还需按住说话并检查板端麦克风。已有 agent 占用唯一会话时，speech `/readyz` 的 `sessions_full` 表示没有额外会话容量；模型健康状态仍按 `/health` 判断。

```bash
cd /userdata/EdgeArm
scripts/setup-tools.sh
scripts/setup-dev.sh
source scripts/activate-env.sh
scripts/check.sh
```

以后每个终端只需 `cd /userdata/EdgeArm` 并执行 `source scripts/activate-env.sh`。该脚本启用 `.venv`、配置本地控制 socket 和 Silero VAD 路径，并清除 ROS 注入的 `PYTHONPATH`。完整应用依赖下文启动的 speech 和 LLM 服务；环境激活本身不启动这些服务。

Python 语音依赖使用 `deploy/python.lock.txt`，voxedge=0.0.15a1。舵机依赖额外安装：

```bash
.venv/bin/python -m pip install -e './agent[arm]' -c deploy/python.lock.txt
```

feetech-servo-sdk=1.0.0、pyserial=3.5 已在本机安装并验证导入，真实串口与舵机动作仍未验收。RGB-D 需要板端 librealsense、RKNN2 模型/Runtime，MCU 协议仍待选型；这些不进入无外设语音开发环境。

## 2. 恢复主机前置条件

确认 ARM64 RK3588、16GB/128GB 规格、Ubuntu 用户态与 BSP。确认 /dev/pcie-rkep-*、RKNN3 库、设备固件和 transfer proxy 的兼容版本；仅文件存在不能证明推理可用。检查板端服务及日志：

```bash
systemctl is-active rk182x.service rknn2-server.service
sudo journalctl -u rk182x.service -n 100 --no-pager
sudo journalctl -u rknn2-server.service -n 100 --no-pager
```

`rknn2-server.service` 用于 RKNN2 相关板端服务，`rk182x.service` 提供 RK1828 transfer proxy；两者均不能代替模型推理检查。本机原生 speech 使用 `.local/runtime/speech` 中已校验的配套运行库，按上面的原生流程启动；无需额外运行 rknn2-server。

在没有 LLM worker 运行时，设置 RK1828 的 Work Mode 为 EFFICIENT（0）、Prefill Mode 为 PERFORMANCE（1），并回读确认：

```bash
sudo scripts/setup-rk1828-mode.sh
```

该脚本检查服务状态、拒绝在 LLM worker 运行时切换模式，不执行硬件复位。重启或设备复位后重新确认模式，再加载模型。

模型和 Docker 数据放在 /userdata，保留系统与下载峰值空间。确认实际麦克风/扬声器连接和全双工 USB 声卡；禁止用声卡枚举代替音频闭环验证。容器默认只给 agent 声卡设备，机械臂未接入。

## 3. 容器部署参考（当前 BSP 尚不能运行）

本机 Qwen3-4B 来自 RK_LLM 的 V1.1.0 导出，已复制到 `.local/data/llm/` 并按原发布清单校验。仓库默认清单对应上游 V1.0.4 导出；两套模型图文件的哈希不同，直接采用默认清单会被完整性检查拒绝。模型位置及来源见 [模型说明](model-storage.md)。

完整启动命令如下。首次验收先按 3.1–3.4 分别测试模型服务，通过后再启动 agent：

```bash
cd /userdata/EdgeArm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml up -d --build
```

本机覆盖配置只调整 LLM 的两项设置：

```yaml
services:
  llm:
    environment:
      RK1828_ARTIFACT_AUTO_DOWNLOAD: "0"
      RK1828_ARTIFACT_MANIFEST: /opt/llm/models/artifacts.local.json
```

第一项关闭 LLM 自动下载，使用已准备的四个模型文件；第二项选择与本机导出配套的大小和 SHA256 清单。容器路径 `/opt/llm/models/` 对应主机 `.local/data/llm/`。覆盖配置及 `artifacts.local.json` 均为本机文件，位于 Git 忽略的 `.local/` 目录；新克隆需先准备模型和本机配置。

`--build` 构建软件容器镜像；服务启动后会实际加载模型并开始推理服务初始化。已有的文件校验记录与板端推理验收分别记录。

所有产品镜像均需构建源码 overlay；固定发布镜像仅提供原生运行库基础。本机已拉取 LLM 基础镜像，但源码镜像构建的 RUN 步骤因内核缺少 mqueue 支持而失败；原生推理通过不代表容器构建和启动通过。

### 3.1 校验已准备的模型

```bash
cd /userdata/EdgeArm
source scripts/activate-env.sh
python scripts/test-models.py --verify-only
```

脚本核对 ASR/TTS 清单、Matcha 前端资源及 Qwen3-4B 四文件导出，优先使用 `.local/data/llm/artifacts.local.json`。本机预期校验 19 个文件，输出 `PASS model file verification.`，报告中 `artifacts.inference_tested` 为 `false`。若出现文件缺失、大小或 SHA256 不符，先恢复对应发布包的完整文件，再进行下一步。

### 3.2 启动 Docker 和模型服务

主机尚无可用 Docker daemon 时，先执行：

```bash
sudo scripts/start-docker.sh
```

该进程保持前台，在另一个终端执行下面的启动步骤；已有可用 Docker daemon 时直接执行：

```bash
cd /userdata/EdgeArm
source scripts/activate-env.sh
mkdir -p .local/data/{asr,tts,llm,voice-data,agent-state} .local/run/edgearm
chmod 700 .local/run/edgearm
python scripts/doctor.py --json
sudo scripts/compose.sh -f .local/deploy/models.local.yaml build speech llm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml up -d speech llm
sudo scripts/compose.sh -f .local/deploy/models.local.yaml ps
sudo scripts/compose.sh -f .local/deploy/models.local.yaml logs --tail=100 speech llm
```

doctor 是环境诊断，不执行模型推理。若当前用户无法访问 Docker socket，需结合管理员权限下的 Compose 输出判断 daemon 状态。首次构建会下载镜像及依赖，首次启动还需加载模型；等待初始化完成，检查日志及健康接口：

```bash
curl -fsS http://127.0.0.1:8621/health | python -m json.tool
curl -fsS http://127.0.0.1:1828/health | python -m json.tool
```

speech 必须返回 `asr: true`、`tts: true`，LLM 必须返回 `status: "ok"`。HTTP 200 或容器显示 running/healthy 仍需结合响应内容判断；下一步脚本会检查这些字段。

后续管理这套服务时也合并同一个覆盖配置，例如查看日志、重启或停止时保留 `-f .local/deploy/models.local.yaml`。

构建需要网络拉取 digest 锁定基础镜像与 Python 包。Compose 配置的 speech、LLM、agent 内存上限分别为 8/3/1GiB，禁止 swap；当前主机缺少内存 cgroup，Docker 明确报告不支持内存限制，因此这些限额尚未生效。本次原生检查不验证容器资源限制；须补齐 BSP 支持并测 RSS/PSS、MemAvailable 和 RK1828 内存后再验收。LLM 失败启动不自动重试，先排查 proxy/驱动/固件。

### 3.3 测试 LLM、TTS 和 ASR 真实推理

在已激活环境的终端执行：

```bash
python scripts/test-models.py --services-only
```

脚本通过本机 HTTP 服务依次执行以下检查，不需要麦克风或机械臂：

| 检查 | 通过条件 |
|---|---|
| 服务就绪 | ASR/TTS 就绪标志为 true，LLM worker 状态为 ok |
| Qwen3-4B | 对中文自我介绍请求返回非空文本；人工检查回复是否合理 |
| Matcha/Vocos | 合成“你好，今天天气真不错。”，WAV 可解码、采样值有限、时长至少 0.1 秒、RMS 至少 0.0001 |
| Qwen3-ASR | 回读上一步合成音频；去除标点、空白并统一大小写后的字符错误率 CER 不超过 0.3 |

成功输出 `PASS model inference smoke; listen to tts.wav and review the LLM reply.`。这是单句基本功能检查；仍需试听合成音频、核对回复，并补充真实录音和语料测试来判断质量。脚本记录的是整次 HTTP 请求耗时，不能据此宣称首包延迟、实时率或整条语音链路延迟达标。

每次运行的报告位于 `.local/reports/model-smoke/<UTC 时间>/`：

| 文件 | 用途 |
|---|---|
| `report.json` | 所选测试模式、通过/失败状态、错误原因及已完成步骤的结果 |
| `llm-response.json` | LLM 原始响应 |
| `tts.wav` | 合成音频，供试听和 ASR 回读 |
| `asr-response.json` | ASR 原始响应 |

脚本遇到失败会停止后续步骤，因此对应响应文件可能尚未生成。正常检查的退出码为 0，运行失败为 1；`--verify-only` 返回 0 只代表文件校验通过。服务启动后，也可以运行 `python scripts/test-models.py` 一次完成文件校验和推理检查。

### 3.4 失败排查

先查看终端输出和本次 `report.json` 的 `error`，再结合模型服务日志定位：

```bash
sudo scripts/compose.sh -f .local/deploy/models.local.yaml ps
sudo scripts/compose.sh -f .local/deploy/models.local.yaml logs --tail=200 speech llm
sudo journalctl -u rk182x.service -n 100 --no-pager
free -h
```

| 现象 | 处理方向 |
|---|---|
| 文件不存在、大小或 SHA256 不符 | 检查 `.local/data/`、配套清单及完整导出；本机 V1.1.0 LLM 必须合并 `models.local.yaml` |
| Docker daemon 无法连接 | 在单独终端启动 `sudo scripts/start-docker.sh`，保留进程；检查管理员权限下的 Compose 输出 |
| HTTP connection refused | 检查 speech/llm 是否启动、端口和初始化日志；环境激活不会启动服务 |
| ASR/TTS 为 false，或 LLM 状态不是 ok | 模型尚未就绪或 worker 初始化失败；从日志检查挂载路径、原生运行库、proxy/固件版本及内存 |
| LLM 回复为空或请求超时 | 检查 worker 日志和 RK1828 服务；默认单请求超时 180 秒，可用 `--timeout` 调整并重新记录 |
| TTS 无法解码、过短或静音 | 检查 Matcha 前端资源和 Vocos 配套模型、运行库，核对服务日志 |
| ASR 回读 CER 超限 | 试听本次 `tts.wav` 并对照识别文本，区分合成问题和识别问题；补充真实录音测试 |

### 3.5 启动 agent 并验证录放音

模型服务推理检查通过后，再启动 agent：

```bash
sudo scripts/compose.sh -f .local/deploy/models.local.yaml build agent
sudo scripts/compose.sh -f .local/deploy/models.local.yaml up -d agent
sudo scripts/compose.sh -f .local/deploy/models.local.yaml ps
sudo scripts/compose.sh -f .local/deploy/models.local.yaml logs --tail=100 agent
```

修改声卡配置时，复制 `deploy/cm3588.env` 为 `deploy/cm3588.local.env`，设置 `EDGEARM_AUDIO_INPUT` / `EDGEARM_AUDIO_OUTPUT`。使用 sudo 时显式传递环境文件，后续管理命令也使用同一路径：

```bash
sudo env EDGEARM_ENV_FILE=/userdata/EdgeArm/deploy/cm3588.local.env scripts/compose.sh -f .local/deploy/models.local.yaml up -d agent
```

默认配置固定中文、单会话、2048 token LLM 上下文和按键说话；回声/打断验证后才能调整。

本地调试页 `127.0.0.1:18000`；SSH 转发后用按键说话按钮。语音服务 `127.0.0.1:8621`、LLM `127.0.0.1:1828`。使用 health、实际 ASR/LLM/TTS 和录放音样本验证，不能只看容器 running。

## 4. SO-101 连接、人工使能和停止

先核对实物供电、电平、共地、限流、电机ID1..6和物理断电/被动支撑。只使用稳定 `/dev/serial/by-id/` 名称。真实标定包含全部六电机；每电机记录 id、drive_mode、homing_offset、range_min、range_max。当前适配器要求 drive_mode=0，范围有限且匹配实际机械限位；对电机端 homing offset、方向、角度/夹爪映射必须另行实机核对。

将真实标定文件放到 `.local/data/agent-state/calibration/so101.json`，人工核对后计算 SHA256，填写 `metadata.robot.actuator.calibration_sha256`。哈希只能绑定文件，不能自动认可标定。没有模板全量程标定，也不生成假的标定文件。

配置 `metadata.robot.enabled: true` 后，通过 Compose 覆盖文件显式授予该串口设备访问，并把本地配置挂载到同一配置路径；使用 compose.sh 的标准 `-f <overlay>` 参数合并。确认设备用户/组权限并重建/重启 agent。enabled 只连接并读反馈，启动不启动力矩；不要把语音/LLM 容器的 /dev 或 privileged 权限直接复制给 agent。

读取状态和人工使能：

```bash
.venv/bin/python scripts/control.py status
.venv/bin/python scripts/control.py enable
.venv/bin/python scripts/control.py stop
```

Unix socket 位于 `.local/run/edgearm/control.sock`，目录700、文件600；由容器运行用户创建，客户端需要相同用户或主机管理员权限。该控制客户端还未在本轮真实 socket 环境运行。异常退出残留 socket 时，先确认旧进程已终止再删除；启动不会自动夺取现有 socket。

停止先撤销运动权限，再请求所有关节保持当前反馈，包括夹爪；软件返回 hold_feedback_stable 不等于已验证物理急停或持物安全。故障不能由 LLM重新使能。人工使能、低速动作、反馈年龄、停止和取消必须在监护条件下逐项验收。关闭或物理断电可能失去支撑，必须按实际机械结构设计。

## 5. 感知/接触/抓放接入与剩余工作

- D435i：确认USB3带宽、真实profile、RGB/深度配对、CameraInfo、有效深度与固定相机到基座外参。原始帧在相机所有者处理；只向 SensorStore 发布带版本与时间的结构化 Scene。
- RKNN2：适配 YOLOv8n-seg 与目标数据，测深度组合和可达候选；不要把类别/掩码宣称为完整抓取位姿。CPU参考与量化精度单独记录。
- 接触：完成ADC/MCU选型、采样协议、标定、序号和时钟域转换；发布 Contact。健康/新鲜度检查失败时禁止任务。
- 抓放：实现 GraspBackend.plan/verify，配置已测放置区ID和受限工作区；五自由度姿态约束、轨迹速度/限位、多源结果判定与有限恢复必须共用现有执行器。当前应用没有默认后端，不会返回伪成功。
- 并发：测语音与视觉共享NPU的核心分工、排队、CPU抖动、整机内存、长时间温度与故障恢复；记录上游能力和本项目实测结果。

ROS 2 / ros2_control / MoveIt 属于后续选项；采用时须关闭直接串口路径、移交唯一设备所有权。CPU低资源LLM方案待同机测量后另立配置，本轮没有保留多套默认模型。

## 6. 剩余工作验收门槛

| 工作 | 通过条件与记录 |
|---|---|
| 固定 BSP / RK1828 / 镜像 | 实际构建并启动；记录 SDK、proxy/固件、库与模型哈希，ASR/LLM/TTS 样本及主机/EP 内存 |
| 模型基本功能 | 文件校验和真实推理分别通过；保存 model-smoke 报告、LLM/ASR 响应及 WAV，人工核对回复和试听，并补充真实录音测试 |
| 麦克风 / 扬声器 | 保留已有真实语音测试样本并补充板端采集样本；记录录放音、PTT、重连；AEC/打断另行验收 |
| SO-101 标定 / SDK | 验证六电机 ID/方向/offset/量程、SDK 分组读反馈完整性及有限串口超时；人工使能不跳变 |
| 停止 / 取消 | 监护下测各阶段停止和反馈年龄；包含拔线、读写故障、持物、不自动松爪、断电支撑 |
| D435i / MCU | 实测相机 profile/外参、源时钟转换与接触标定；过期、乱序、丢设备必须禁止任务 |
| 抓放闭环 | 实现 plan/verify 与运动中接触/健康检查、受限恢复；复用唯一总线，禁止仅依赖静态轨迹与延迟宣称闭环 |
| 整机并发 / 热运行 | 测 NPU 共存、内存峰值、排队/控制抖动、温度和故障恢复；保存真实结果和失败样本 |

当前 plan/verify 是扩展契约，静态序列并未实现运动中接触反馈控制。没有默认后端、未经实机校验时，不开放自主抓放。ROS / MoveIt 和 CPU 降级不属于本轮默认启动范围。
