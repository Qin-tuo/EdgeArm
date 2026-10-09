# 桌面台灯陪伴机器人单板系统架构与选型

> 项目：Edge Lamp Companion / `edge-lamp`  
> 当前选定硬件：本机 CM3588 V2 / RK3588，16GB RAM，Ubuntu 22.04 ARM64，Linux 6.1.118；RK1828 通过 PCIe 连接；数据位于 `/userdata`  
> 日期：2026-10-09  
> 状态：已选定 OpenVoiceStream 改造基线，尚未完成本机硬件验收  
> 完整产品任务：[项目二计划书](/Users/barry/Desktop/AI-Infra-Course/project-plans/02-desktop-lamp-rockchip.md)

**选型更新（2026-10-09）：** 用户已确认采用 OpenVoiceStream，硬件就是当前开发板。当前执行依据为 [Lamp 基线说明](docs/lamp-baseline.md) 与 `deploy/lamp/`。下文保留原架构方案作为需求参考；其中 8GB/64GB、固定 C++/ROS 进程划分及 SenseVoice/Melo/Qwen2.5 首选模型均属于旧方案，不作为当前实施约束。首轮沿用上游 Qwen3-ASR、Matcha/Vocos、Qwen3-4B 链路，接口和外设细节在实际接入时收敛。

原方案正文：

本阶段先定义单板上实际运行的软件系统：模块分工、模型与 Runtime、外设连接、数据格式、通信接口和部署结构。主线是机器人 C++/Linux 系统与异构端侧推理，语音和视觉作为真实产品负载。

首版选择：**ROS 2 + CycloneDDS 连接机器人模块；RKNN2 承载 RK3588 视觉；sherpa-onnx 承载 CPU 语音；RKNN3 承载待验收的 RK1828 对话推理；llama.cpp 提供对话 CPU 基线。** 原始媒体在所属进程内处理，跨进程主要传观察、文本、状态和受限任务。

## 1. 平台条件与系统版本

| 条件 | 已知内容 | 影响 |
|---|---|---|
| RK3588 | 用户确认 8GB RAM / 64GB 存储 | 小对话模型；限制缓存、上下文与媒体记录 |
| Linux 内核 | 用户确认 6.1.118 | 保留板卡 BSP，不为安装 ROS 随意换通用内核 |
| RK1828 | 用户确认 PCIe 连接 | 使用厂商驱动与 transfer proxy，不自研 DMA |
| Linux 发行版、板卡型号 | 尚未提供 | 决定 ROS 版本、用户态 ABI、USB 拓扑 |
| RK1828 自身内存、固件和 SDK | 尚未提供 | 决定模型和上下文；不能拿主机 8GB 替代此信息 |
| 舵机适配板、音频设备 | 具体型号待核验 | 电压、总线和音频同步能力需逐项确认 |

Ubuntu 22.04 使用 ROS 2 Humble，Ubuntu 24.04 使用 Jazzy。若为 Debian 或其他厂商镜像，先核验源码构建与依赖，不套用 Ubuntu apt 命令，也不因内核是 6.1 就认定发行版。优先保留已经支持 RK1828 的厂商系统。

RealSense ROS 驱动列出了 Humble/Ubuntu 22.04 与 Jazzy/Ubuntu 24.04 的组合；这不证明用户的 ARM64 BSP、USB 控制器和 6.1.118 内核已兼容 D435i。驱动与 librealsense 版本配套固定，先验收再考虑改内核。[RealSense ROS 驱动](https://github.com/realsenseai/realsense-ros)

## 2. 整机拓扑

```text
D435i -- USB 3 --------> RK3588: perception_container
                            RGB + depth -> YOLOv8n/RKNN2 -> objects/target

全双工 USB 声卡 -------> RK3588: lamp_voice
  麦克风 / 扬声器           ALSA -> AEC/VAD/KWS -> SenseVoice -> text
                            speak text -> MeloTTS -> PCM -> ALSA

观察与文本 -- ROS 2 / CycloneDDS --> RK3588: lamp_companion
                                       状态机 / 命令路由 / 对话 / 网页
                                           |
                        本地 HTTP/JSON       | ROS Action/Service
                              |            v
                         LLM server   RK3588: lamp_actuator
                              |       安全 / 仲裁 / ros2_control
                     RKNN3 transfer proxy  |
                              |            +-- USB 串口适配器
                             PCIe          |       -> 半双工 TTL 舵机总线
                              |            |       -> SO-101 五关节
                           RK1828          |
                        对话模型推理        +-- USB CDC -> Pico -> LED/按钮

CPU 基线：LLM server 改为 llama-server，不同时加载协处理器后端
硬件急停：独立于 ROS、Linux、模型和网页链路
```

ROS 2 是模块间协议，HTTP 是应用调用模型服务的协议，PCIe 是主机与协处理器的物理连接。它们互不替代；交互模块不应自己封装 PCIe DMA 描述符或舵机数据包。

## 3. 进程与资源所有权

四个业务进程，不按每个小功能建立一个服务。

| 进程 | 内部组件 | 独占资源 | 输出 |
|---|---|---|---|
| `perception_container` | RealSense 组件、RGB-D 对齐、YOLO、深度过滤、目标保持 | D435i、RKNN2 推理上下文 | 物体列表、人物目标、感知健康状态 |
| `lamp_voice` | ALSA、重采样、AEC、VAD、KWS、ASR、TTS | 声卡、语音模型、PCM 缓冲 | 唤醒/打断事件、识别文本、播放状态 |
| `lamp_companion` | 状态机、命令路由、上下文、LLM 客户端、工具校验、网页桥接 | 对话请求、本地偏好数据库 | 受限动作、照明请求、合成文本、产品状态 |
| `lamp_actuator` | 动作仲裁、安全检查、controller manager、舵机插件、LED/按键 | 舵机/Pico 串口、运动使能状态 | 关节状态、动作结果、按钮、安全状态 |

另外运行标准 `robot_state_publisher`，由 URDF 与关节状态发布 TF。对话后端为 `llama-server`，或厂商 `rkllm3-server` 加 transfer proxy；最终服务数量以 SDK 启动方式为准，不将厂商辅助服务藏在“四个进程”的说法里。

相机与检测在同一 ROS 组件容器，优先使用 intra-process 通信减少拷贝；音频内部用有界队列连接。**同进程不自动等于零拷贝**，需要核对图像转换、消息所有权、RKNN 输入与实际拷贝次数。

控制进程不加载 ASR/LLM/TTS，不同步等待 HTTP。推理使用工作线程，不占用 ROS 回调线程；控制更新线程不做 SQLite、日志落盘或长时间串口重试。推理进程异常时控制仍能拒绝新任务并执行经过验证的停止策略。

## 4. 固定模型与推理后端

下表是首轮明确选型，不是“每个模型都已经兼容 RK1828”的清单。

| 功能 | 首轮模型 | 格式与 Runtime | 位置 |
|---|---|---|---|
| 人物/常见物体 | Rockchip 适配 YOLOv8n | 优化 ONNX 转 INT8 `.rknn`；RKNN2 C API | RK3588 NPU |
| VAD | Silero VAD | ONNX；sherpa-onnx C++ / ONNX Runtime | RK3588 CPU |
| 唤醒 | Zipformer 中英 KWS 3M，2025-12-20 | chunk-8；INT8 encoder/joiner + FP32 decoder；sherpa | RK3588 CPU |
| ASR | SenseVoiceSmall，sherpa 2025-09-09 INT8 导出包 | `model.int8.onnx` + tokens；sherpa | RK3588 CPU |
| TTS | MeloTTS-Chinese 的 `vits-melo-tts-zh_en` 导出包 | ONNX + 词典/tokens/规则；sherpa | RK3588 CPU |
| 对话 | `Qwen/Qwen2.5-1.5B-Instruct` | GGUF Q4_K_M / llama.cpp 基线；RKNN3 为协处理器候选 | CPU 基线；目标 RK1828 |

### 4.1 视觉

使用 Rockchip Model Zoo 的 YOLOv8n 导出与后处理路径。官方适配图与原版 Ultralytics 输出结构不同，不能把任意原版 ONNX 与这套 C++ 后处理混用。类别、尺寸、量化、坐标还原和 NMS 成套固定。深度来自 D435i，不再跑深度估计模型。[Rockchip YOLOv8 示例](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/README.md)

首版 `person` 驱动在场与跟随，其他类别作为精简对话上下文。它只能描述检测类别与粗略位置，不代表能读文字、识别任意物品或理解整个桌面。VLM 不占首版常驻内存；以后按单帧任务单独评测。

### 4.2 语音

统一使用 sherpa-onnx C++ 接口，不把 PyTorch/FunASR/MeloTTS 原始 Python 训练环境搬进常驻业务。SenseVoice 路径是 **VAD 分段后识别完整语句**，不是逐 token 增量 ASR；真正流式 ASR 需要另评测 Zipformer 等模型。[SenseVoice 部署文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)

KWS 初始短语“小灯小灯”，按中文词表生成 keywords 并测误唤醒。chunk-8 是模型分块，不是 8ms；准确率与唤醒整链路延迟实测。[KWS 模型与配置](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html)

Melo 的 sherpa 导出模型原生输出 44.1kHz，词典与规范化资源必须一起部署。先按标点分句合成和播放，不宣称模型原生逐 token 音频流。CPU 合成 RTF 若不达标，需要优化或替换部署模型，不能套用其他机器的性能。[Melo 部署包](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)、[原始中文模型](https://huggingface.co/myshell-ai/MeloTTS-Chinese)

### 4.3 对话与 RK1828 门槛

固定 Qwen2.5-1.5B checkpoint、Tokenizer 与 chat template；CPU 以 llama.cpp GGUF Q4_K_M 建立基线。上下文初值 2048 token，一次最多 128 个新 token，一个生成任务；这些是初值而非性能保证。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)、[llama.cpp](https://github.com/ggml-org/llama.cpp)

RK1828 路径为 ONNX -> RKNN3 -> 配套模型/权重/Tokenizer/embedding -> Runtime 或 `rkllm3-server`。GGUF 权重、RKNN2 模型不能直接交给 RKNN3。SDK 说明了 RK1828、PCIe 协处理器模式与 OpenAI-compatible 服务，但上下文、采样参数、流式与取消行为要按用户版本核验。[RKNN3 SDK](https://github.com/airockchip/rknn3-toolkit)

**当前公开 Qwen2.5 示例用的是 3B，转换参数仍写 `rk1820`。** 它不能证明 1.5B 已兼容用户 RK1828。先跑 BSP/SDK 已支持模型验证 PCIe，再核验目标 checkpoint 的转换、输出和内存；若只能跑另一个模型，登记差异，不把不同模型的速度直接解释为硬件加速比。[官方 Qwen2.5 示例](https://github.com/airockchip/rknn3-model-zoo/blob/main/examples/Qwen2_5/README.md)

对话 CPU/协处理器后端只开一个。ASR/TTS 首版不争抢 RK1828，测出瓶颈后再迁移；SDK 列出的模型家族不是目标权重的兼容证明。每个模型冻结来源、许可、哈希、精度与转换版本；训练适配后的模型重新走端侧验证。

## 5. 外设连接与驱动

| 外设 | 物理连接 | 软件入口 | 内容 |
|---|---|---|---|
| D435i | USB 3，优先直连主机 | librealsense + realsense2_camera | RGB、深度、内参、时间戳 |
| 麦克风/扬声器 | 共用一块全双工 USB Audio 声卡；按需接功放 | ALSA `libasound` | 捕获与播放 PCM |
| SO-101 五关节 | USB 总线适配器 -> Feetech 半双工 TTL -> 串联舵机 | 核验现有适配板；ros2_control 硬件插件 | ID、位置目标、位置/故障反馈 |
| LED/按钮 | 无现成接口时，Pico/RP2040 USB CDC；PWM/GPIO 接驱动/按钮 | actuator 串口驱动 + Pico SDK 固件 | 亮度、灯色、按键、心跳、确认 |
| RK1828 | 已有 PCIe | 厂商驱动 + RKNN3 Runtime/proxy | SDK 管理的模型与推理数据 |
| 调试电脑 | 有线网/局域网 | SSH；显式鉴权的网页入口 | 配置/诊断，不是在线推理依赖 |

Pico 是没有现成灯控接口时的默认分支；现有控制板已经支持调光/按键则直接替代，ROS 接口保持一致。Pico 只输出控制信号，不直接驱动大功率 LED；电平、PWM、隔离/共地和电源按实际驱动核验。[Pico 官方能力](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html)

SO-101 舵机不是普通 GPIO/PWM 舵机，不能用 Pico PWM 替代其 TTL 总线。舵机供电与 USB 通信分别规划，确认手中电压版本及适配板电流；USB 数据连接不意味着 USB 给整臂供电。[SO-101 官方指南](https://huggingface.co/docs/lerobot/so101)

板卡口不足时才加供电可靠的 USB hub，测相机、声卡、串口并发时掉帧/xrun。用 udev 稳定名称 `/dev/lamp_servo`、`/dev/lamp_io` 和声卡 ID，不依赖变化的 `ttyUSB0` / card 0。

## 6. 媒体数据路径

### 6.1 RGB-D

候选 RGB/depth 为 640x480 / 30fps，检测取最新帧，目标处理 10-15fps。先查询相机实际支持 profile 并记录真实格式，不只看 launch 参数。保留采集时间、主机到达时间、frame ID 和深度单位。

```text
RGB8 + Z16 depth + CameraInfo
 -> 深度对齐到 RGB / 时间配对
 -> RGB letterbox -> RKNN2 -> 检测框还原至原图
 -> 框内有效深度过滤 -> camera_optical_frame 中的点
 -> TF 外参变换 -> base_link 中的点
 -> TargetObservation / ObjectObservation
```

此规格未压缩像素量为 `640*480*(3+2)*30 = 46.08 MB/s`，不含额外传输开销。帧队列初值 2，积压丢旧帧；媒体处理同容器，跨 DDS 发送结构化观察。全量图像/rosbag/预览按需开启，不让每个进程都订阅原始 RGB-D。

目标记录类别、置信度、像素框、相机/基座坐标、深度有效性、跟踪 ID、采集时间。深度无效仍可报告检测到人，但禁止伪造距离或无限期沿用旧坐标。

### 6.2 音频

首版同一声卡采集/播放。若设备以 48kHz 工作，采集下采样到 16kHz mono；语音内部用连续浮点 PCM 缓冲，测试文件用 PCM16。16kHz mono PCM16 为 32KB/s；20ms 为 320 samples / 640B。

```text
ALSA capture -> 重采样/帧缓冲 -> AEC -> VAD + KWS
 -> 端点分段 -> SenseVoice -> 最终文本 -> ROS

ROS Speak -> 规范化/分句 -> MeloTTS 44.1kHz PCM
 -> 重采样/音量 -> ALSA playback
 -> 实际播放参考缓冲 -> AEC
```

AEC/重采样首选 SpeexDSP C 接口。参考信号对应真正送往声卡的波形，包含音量、缓冲延迟与采样率处理；不能拿未播放的 TTS 波形作同步参考。同声卡也需测时钟与延迟；独立 USB 麦克风/另一声卡扬声器不能默认获得有效 AEC。[Speex AEC 与重采样](https://www.speex.org/docs/manual/speex-manual/node7.html)

先交付半双工，回声/双讲测试通过再启用语音打断。ALSA/AEC 块和模型输入窗口分开管理，模型窗口不必是 20ms。原始 PCM 留在 voice 内部，不经 ROS 绕行多个进程；对外发送文本、端点/唤醒事件与播放进度。

## 7. 通信框架与接口契约

### 7.1 框架

ROS 使用 `rclcpp` 与 `rmw_cyclonedds_cpp`；首版仅单板本地 DDS 通信，按 ROS 版本核对 localhost 配置，不默认把动作接口广播到局域网。[CycloneDDS ROS 接入](https://github.com/ros2/rmw_cyclonedds)

Topic 用于观察/状态，Service 用于短配置，Action 用于长动作/播放。取消与执行完成由执行侧确认；“发送取消”不是“实际停下”。不再叠加 MQTT、ZeroMQ、Redis 或自研总线。

### 7.2 ROS 接口

以下是项目定义，不是厂商已实现的接口。

| 接口 | 类型 | 方向 | 数据/含义 |
|---|---|---|---|
| `/lamp/target` | Topic / `TargetObservation` | perception -> companion/actuator | 单目标坐标、深度有效性、时间 |
| `/lamp/objects` | Topic / 包装物体列表消息 | perception -> companion | 有界类别/位置列表，不传全图 |
| `/lamp/voice/event` | Topic / `VoiceEvent` | voice -> companion | 唤醒、话音、端点、打断与事件 ID |
| `/lamp/voice/transcript` | Topic / `Transcript` | voice -> companion | 最终文本、utterance ID、请求关联 |
| `/lamp/state` | Topic / `LampState` | companion -> voice/actuator/web | 交互状态、请求 ID、模式、故障 |
| `/lamp/safety_state` | Topic / `SafetyState` | actuator -> companion/voice/web | 使能、急停锁存、限位/反馈故障；执行侧权威状态 |
| `/lamp/heartbeat` | Topic / `Heartbeat` | companion -> actuator | boot ID、序号、单调时间；丢失后拒绝新动作 |
| `/lamp/button` | Topic / `ButtonEvent` | actuator -> companion | 去抖后的按键、序号与时间 |
| `/joint_states`、`/tf`、`/tf_static` | 标准 Topic | actuator/rsp -> 控制与感知 | 关节反馈/坐标；静态 TF 用标准配置 |
| `/lamp/health` | Topic / `DiagnosticArray` | 各进程 -> companion/web | 后端、驱动、资源和掉线状态 |
| `/lamp/speak` | Action / `Speak` | companion -> voice | 文本片段/请求 ID；进度、结果、取消 |
| `/lamp/gesture` | Action / `PlayGesture` | companion -> actuator | 白名单动作名/幅度/时长；结果、取消 |
| `/lamp/look_at` | Action / `LookAt` | companion -> actuator | 基座坐标系目标/有效期；限制工作区、结果和取消 |
| `/lamp/set_light` | Service / `SetLight` | companion -> actuator | 亮度/灯色；拒绝或设备确认，不伪称光学反馈 |
| `/lamp/set_mode` | Service / `SetMode` | web -> companion | 静音、休眠、免打扰、跟随开关 |
| `/lamp/set_timer` | Service / `SetTimer` | web/命令路由 -> companion | 有界时长、取消与免打扰规则 |
| `/lamp/stop` | Service / `Stop` | voice/companion/web -> actuator | 高优先级受控停止，不是硬件急停 |
| `/lamp/enable_motion` | Service / `SetBool` | 经确认的本地控制 -> actuator | 显式重新使能，LLM 无权限 |

连续观察采用 best-effort/keep-last 1-2；语音事件、任务和安全状态采用 reliable、有界历史。运动命令 volatile，不用 transient-local 重放旧动作；状态快照可单独保留最新值。发布订阅 QoS 必须匹配。[ROS 2 QoS](https://raw.githubusercontent.com/ros2/ros2_documentation/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)

Reliable 不等于恰好执行一次或硬实时。请求/事件携带 ID 并去重；运动任务含发送单调时间、TTL 和 boot ID，拒绝过期与上次开机命令。各进程使用同一单板单调时钟判断年龄；Header 时间另留给工具，不用可跳变墙钟计算有效期。心跳只关心最新值，使用有界历史和接收侧超时，不能靠 DDS 重传延长运动许可。

### 7.3 模型与网页

内部对话契约固定 messages 输入、文本片段、最终状态与 request ID。CPU 后端调用 llama-server OpenAI-compatible HTTP；RK1828 后端接 `rkllm3-server`，按 SDK 核对端点、流式与采样参数，差异集中在一个客户端适配文件。

迟到的取消结果直接丢弃；若服务不能中止计算，等待旧计算结束或受控重启模型服务再放行新任务。关闭 HTTP 连接不等于释放了协处理器计算。

网页和模型服务默认绑定 `127.0.0.1`；网页静态 HTML/CSS/JavaScript，C++ `cpp-httplib` 提供 REST/SSE 和本地 HTTP 客户端。局域网访问显式开启鉴权，不暴露底层轨迹；网页和语音一样经过受限 ROS 入口。

## 8. 行为与运动框架

交互状态为 `idle/listening/thinking/speaking/resting/fault`。运动使能/急停锁存单独维护，跟随是受仲裁的模式，不因 speaking 自动解除故障。

停止、静音、亮一点/暗一点等固定命令先走规则，不等待 LLM。开放对话用小模型，观察转换为精简上下文。模型工具输出完整 JSON 校验后受理，不从未完成流式文本中抽出动作执行；禁止 shell、任意轨迹与重新使能。

有效用户打断首先由 voice 在本进程停止旧音频，再通知 companion 使旧 request ID 失效，取消生成/待播句子及关联动作。若模型还在计算，迟到结果也不能重新播放或运动；网络取消能力与本地实际停播分别计时。

运动复用 **ros2_control + joint_trajectory_controller** 的位置接口、轨迹插值与结果监控，只适配一个 Feetech 硬件插件并复用核验过的协议实现。动作库为五关节关键帧与时间；不重写通用轨迹控制器，不引入首版不需要的 MoveIt 全场景规划。[官方轨迹控制器](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)

actuator 仲裁节点调用底层 FollowJointTrajectory；底层接口不向网页/LLM开放。硬件插件写串口前仍独立核验使能、反馈年龄、软限位和速度范围。ROS 单位为 rad/m，插件转舵机单位；取消后保持/减速按 ROS 版本与实机确认，不假定各发行版具有同一取消功能。

跟随只在固定安全工作区内调整朝向：TF 标定后计算方向，映射到受限 yaw/pitch 动作；不追求五轴任意笛卡尔位姿。URDF 对应实际灯头/轴向，最差负载测试通过后才能运动；复杂姿态以后再加 KDL/MoveIt。

控制目标初值 30Hz；目标年龄超过 300ms、关节反馈超时或上层心跳丢失时，禁止新任务并进入验收过的停止策略。门限按总线/负载测试冻结。Linux 失效仍需物理急停与防坠；DDS、ros2_control 和 MCU 心跳不是安全认证。

## 9. 仓库结构

下面是未来 `edge-lamp` 仓库分配，本次未创建代码或安装依赖。

```text
edge-lamp/
  src/
    lamp_interfaces/     # msg / srv / action
    lamp_perception/     # RGB-D、YOLO/RKNN2、目标与深度
    lamp_voice/          # ALSA、SpeexDSP、sherpa、播放与取消
    lamp_companion/      # 状态机、命令、LLM、网页桥接
    lamp_actuator/       # ros2_control 插件、安全、LED/按键
    lamp_bringup/        # launch、参数、URDF、标定、后端选择
  firmware/lamp_io/      # 仅 Pico 灯控分支需要
  web/                  # 本地控制台静态文件
  tools/                # 导出/转换、板卡自检与评测
  tests/                # 契约、回放、故障与集成
  models/manifest.json  # 来源、精度、版本、哈希；不存大权重
  docs/                 # 架构、接线、模型卡、评测与复现
```

业务和设备 Runtime 使用 C++17；构建用 ament/CMake、colcon，参数用 ROS YAML，HTTP/日志用 nlohmann/json，偏好用 SQLite。Python 用于训练/数据/导出/转换/评测；厂商服务若依赖 Python，独立固定环境，不进入控制循环。

工作站隔离 RKNN2/RKNN3 转换环境，板上只放推理产物与 Runtime，不常驻训练栈。固定依赖版本/commit，记录 ARM64、glibc、SDK/Runtime/驱动/固件关系。

## 10. 资源与部署约束

以下为设计目标，不是实测。

- 8GB 主机常态使用目标约 6GB 内，保留约 2GB 系统余量；RK1828 内存独立登记，主机缓冲/KV 同样测量。
- 一个对话任务，最多两个待播句子；过期任务清理，不累积多轮生成。
- ASR/TTS 线程初值各 2，KWS/VAD 各 1；压力冲突时串行/限流，不让各个库默认开满 CPU。
- RK3588 NPU 首版只跑 YOLO；CPU 对话回退降低视觉/线程压力，不同时加载另一份对话后端。
- 默认不存原始音视频；日志轮转上限 1GB，授权调试记录总额初值 5GB，剩余低于 15GB 停止非必要记录并告警。
- 不靠 swap 弥补长期超额模型，不宣称 8GB 能并发常驻大型 VLM/LLM/高质量音色克隆。

程序/网页放 `/opt/edge-lamp`，配置 `/etc/edge-lamp`，模型/SQLite/授权记录 `/var/lib/edge-lamp`；journald 或有界 JSONL 管理日志。

systemd 管理进程，ROS launch 管理组件/参数；厂商 Python 服务用独立 venv，不加入 Kubernetes。顺序为设备/SDK 自检 -> 后端健康 -> 感知/语音 -> 交互就绪。控制默认禁止运动，标定/安全检查与用户明确使能后才放行；重启不自动清除急停或使能运动。

Pico 灯控用有界行式 JSON/USB CDC，包含版本、序号、操作与参数，最大长度/超时固定；设备确认、按钮事件带序号。心跳初值 100ms，500ms 丢失后灯进入约定亮度，不负责舵机断电、不替代急停，也不添加 micro-ROS。

## 11. 上层框架搭建顺序

| 顺序 | 任务 | 结果 |
|---|---|---|
| 1 | 板卡/BSP 盘点 | 发行版、内存、USB/PCIe、SDK/驱动/固件清单 |
| 2 | 后端最小验收 | D435i RGB-D；RK3588 YOLO/RKNN2；RK1828 官方模型/PCIe |
| 3 | 冻结包与契约 | 六个 ROS 包、接口、所有权、设备稳定名 |
| 4 | ROS 运行环境 | 本地 DDS、容器、TF、参数、自检/启动 |
| 5 | 视觉真实链路 | 相机 -> RKNN2 -> 有时间戳/有效距离的观察 |
| 6 | 语音真实链路 | 声卡 -> VAD/ASR -> 文本；TTS -> 实际播放 |
| 7 | 对话/行为 | CPU 或已验收 RK1828 二选一；状态、取消、工具校验 |
| 8 | 执行/照明 | 禁止运动检查接口 -> 低速单轴/五轴；灯/按钮 |
| 9 | 并发/故障 | 迟到结果、断线、重启、压力、内存与存储限制 |

回放/替身用于单元与故障测试。本阶段交付**可在板上启动的部署结构、真实设备/模型链路、接口契约和版本清单**。机械电气未验收时禁止自主运动；软件就绪不替代硬件门槛。

完整产品、数据适配与稳定性仍按主计划，实现排期在发行版与 SDK 最小验收后确定。
