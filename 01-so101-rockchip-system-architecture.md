# RK 机械臂单板系统架构与选型

> 项目范围：RK 端侧语音交互与多传感器机械臂；原外部项目书未随仓库提供。当前交付见 [软件架构](ARCHITECTURE.md) 和 [验证状态](docs/validation.md)。  
> 平台：RK3588 16GB/128GB，内核 6.1.118；PCIe RK1828  
> 日期：2026-10-09；本文件定义目标架构，未完成用户板卡验收

## 1. 架构决定

以 OpenVoiceStream 固定源码快照为基线开发，已在当前仓库裁剪并扩展现有 app、plugin、actuator 和 device profile，保留许可、来源与版本锁；未创建远程 fork。整套在线系统仍运行在 RK 上，保留 SO-101 夹爪，Jetson 不进入拓扑。

先复现上游再增量适配：server/agent、HTTP/WebSocket、会话与模型后端保持原有设计。首版机器人模块通过 agent 的插件/执行器机制连接；ROS 2、CycloneDDS 与 ros2_control 保留为后续标准机器人接口方案，不作为启动门槛。

本地 LLM 沿用 OpenAI-compatible HTTP，RK1828 传输沿用厂商 Runtime/proxy。关键感知、控制或测量需要原生实现时接入 C++17 模块/worker，而不是把现有 Python 会话框架全部改写。

## 2. 首版板端拓扑

```text
麦克风/扬声器 --> OVS agent: edgearm（已迁移；硬件未验收）
                        |
                        +-- 原有音频/会话/工具注册
                        +-- RGB-D感知插件 <-- USB 3 / D435i
                        |      RKNN2 / 深度 / 基座候选
                        +-- 抓放任务插件：结果判定 / 有限恢复
                        |      接触/压力 <-- ADC/MCU / USB CDC候选
                        +-- SO-101执行器：唯一总线所有者
                        |      受控worker / 限位 / 反馈 / 停止
                        |          --> USB适配器 / TTL总线 / 五关节+夹爪
                        |
                        +-- /v2v/stream --> OVS server
                                                ASR / 工具编排 / TTS
                                                |
                                     本地LLM接口 / RKNN3 proxy
                                                |
                                             PCIe RK1828

本地控制台 --> 同一工具/任务入口；物理急停和被动支撑独立
CPU降级：RK主机小模型替换LLM服务，不并行常驻两份
```

感知、任务和执行是职责，不预先规定每个职责都开新进程。阻塞采集、推理与串口操作不能放在音频事件循环里；执行用受控 worker，必要的控制隔离按实际调用链与测试确定。D435i 固定在底座/支架，接触 ADC/MCU 尚待选型。

## 3. 上游扩展点与资源所有权

下表保留设计职责。edgearm、共享 runtime、SO101Plugin、SO 执行器及数据契约已实现；D435i/MCU/抓放后端待接入。旧应用仅作为来源记录，已从产品入口移除。

| 层 | 复用内容 | 本项目改动 |
|---|---|---|
| server / voxedge | 语音会话、ASR/TTS/VAD、后端与工具编排 | 首轮不改核心引擎；仅在实际缺少目标后端时扩展 |
| agent app | voice_arm / MultiModeApp 的生命周期与音频链 | 新 app 组合机械臂、感知、任务插件，不复制一套 BaseApp |
| actuator | SOArmActuator、现有注册机制、动作与观察接口 | 优先复用 SO 驱动，适配真实 SO-101 标定、安全和反馈 |
| perception / grasp | reBot 的插件组织、相机抓取与 dashboard 思路 | D435i 固定外参、RKNN2、多传感器结果与恢复 |
| configs / deploy | 原有 profile、compose 和部署校验 | 新增 16GB 板卡配置，本地 RK1828 服务、设备与资源限制 |
| tests / benchmark | 上游会话/agent 测试与测量工具 | 自己的契约、故障、抓放、并发与优化对照 |

voice_arm 已通过 ArmPlugin 挂接执行器，驱动由 app 注册而非框架硬编码；新增 SO-101 扩展应沿用该机制。[应用入口](https://github.com/Seeed-Solution/openvoicestream/blob/main/agent/ovs_agent/apps/voice_arm/app.py)、[执行器注册](https://github.com/Seeed-Solution/openvoicestream/blob/main/agent/ovs_agent/actuators/factory.py)

源码中的 SOArmActuator 优先使用仓库内 MinimalSOFollower，并有 SO101FollowerConfig 的备用路径；因此先验证现有实现，不默认再写一份 Feetech 驱动。但演示代码的占位标定不能用于自主运动，启动力矩/运动状态必须核对并改为真实标定通过后人工使能。[SO 执行器源码](https://github.com/Seeed-Solution/openvoicestream/blob/main/agent/ovs_agent/apps/voice_arm/so_arm.py)

音频设备只由 agent 持有；相机、接触接口各有一个所有者；舵机总线仅由执行器及其串行 worker 持有。查询/网页/抓放共享观察和同一动作仲裁，不能另开串口或让多个动作序列交错执行。

上游 SO 执行器的动作序列会占用锁，缓存观察可能在运动期间变旧；本项目加入接触/关节闭环前必须测反馈年龄与采样周期，并调整执行/采样路径。取消异步任务不等于底层串口线程或机械臂停止；停止需明确传到执行侧并确认真实结果。不会仅凭上游重连、busy 标志或提示词宣称安全。[动作插件](https://github.com/Seeed-Solution/openvoicestream/blob/main/agent/ovs_agent/plugins/actuator_actions.py)

若后续采用 ROS 控制，执行器插件改为 ROS 客户端，将唯一串口所有权移交控制进程，关闭原直接串口路径；不能双驱动。模型服务仍由 OVS 管理，不重建平行语音服务。

## 4. 模型与推理框架

| 任务 | 首轮候选 | 部署路径与验收 |
|---|---|---|
| 目标检测/分割 | Rockchip 适配 YOLOv8n-seg | RK3588 NPU / RKNN2 C API；ONNX CPU 参考 |
| 深度 | D435i 原生深度 | librealsense，对齐、无效深度过滤，不常驻深度估计模型 |
| VAD | OVS 已支持的 Silero | CPU；按 profile 核验 |
| 中文 ASR | 上游 Qwen3-ASR RK 参考；SenseVoice CPU 为低资源候选 | 先单会话，核对当前 OVS 后端与产物；不把分段 ASR 称为增量流式 |
| 中文 TTS | 上游 Matcha-TTS / Vocos 参考 | 核验 OVS 当前 RKNN/CPU 路径、RTF、采样率与内存 |
| 对话/工具 | 上游 RK1828 Qwen3-4B 参考 | RKNN3 + 本地服务；具体权重、内存、工具格式待验收 |
| LLM CPU 基线 | Qwen2.5-1.5B-Instruct 低位宽 GGUF 候选 | RK3588 llama.cpp；工具成功率与响应速度另测 |

YOLOv8n-seg 只提供类别/掩码，不直接给完整抓取位姿。结合深度、物体几何和桌面约束生成可达候选；已知类别之外要独立收集/标注和适配数据。[RK3588 分割示例](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8_seg/README.md)

上游当前 RK3588 compose 已写有 Qwen3-ASR、Matcha 和本地 RK1828 Qwen3-4B 服务连接；因此不必自研新的 LLM 网络协议。用户已更正为 16GB/128GB，主机内存容量与参考配置一致，但其语音容器 12GB 上限不代表整机资源预算或实际占用。固定源 commit、镜像、profile、模型哈希，核验本机 BSP/驱动、服务部署与视觉/控制并发。[上游部署配置](https://github.com/Seeed-Solution/openvoicestream/blob/main/deploy/docker-compose.radxa.yml)

RK3588 的 RKNN2 与 RK1828 的 RKNN3 分离；模型与 Runtime 不能混用。厂商支持 RK1828 不等于具体 checkpoint 已转换，也不证明主机 BSP 已能同时稳定运行两种驱动。[RKNN3 平台与 Runtime](https://github.com/airockchip/rknn3-toolkit)

### 一套产品默认配置与后续实验

- **后续低资源实验（未保留部署入口）：**视觉 RK3588 NPU，语音 CPU 候选，LLM 首选已验收 RK1828；协处理器未通时在同一 RK 主机降低模型/任务频率。
- **当前产品默认路径（尚未实机验收）：**OVS ASR/TTS 和视觉共享 RK3588 NPU，LLM 保留 RK1828；测核心分配、排队、内存和控制抖动，必要时串行调度或降低视觉占空比。

上游语音已包含 NPU 核心配置，不能再让视觉默认占满全部核心后假设互不影响。先测再确定单一产品默认配置；不把理论 TOPS 相加当作实际加速，也不强制全部模型同时常驻。

首轮 LLM 单任务、短上下文/短回答；上下文 2048 token、输出 96-128 token 是可调整初值，不是 Qwen3-4B 适配证明。CPU 与协处理器若使用不同模型，只能比较产品方案，不能宣称同模型加速比。

## 5. 外设连接与数据传输

| 外设 | 连接 | 软件与数据 |
|---|---|---|
| D435i | USB 3，优先直连 | 待接入 D435i 适配器 + librealsense；ROS 驱动为可选扩展 |
| 音频 | 优先同一全双工 USB 声卡，型号待确认 | OVS agent 音频插件；采集/播放 PCM |
| SO-101 | USB 总线适配器 -> Feetech TTL | 核验现有协议/驱动；目标、实际位置与错误状态 |
| 接触/压力 | 传感器 -> 匹配 ADC/MCU -> USB CDC 候选 | 标定后的测量、序号、时间、健康状态 |
| RK1828 | 已确认 PCIe | 厂商驱动、RKNN3 与 transfer proxy |
| 调试电脑 | 有线网/局域网 | SSH、配置/诊断；不参与在线推理 |

USB 通信不表示整臂可从 USB 供电。不要用普通 PWM 代替 SO-101 TTL 舵机总线；电平、电压、共地、限流与供电必须按实物核验。用 udev/固定设备 ID，不依赖 ttyUSB0 或声卡 0 永远不变。

RGB-D 初始候选 640x480/30fps；RGB8 + Z16 未压缩约 46.08MB/s，不含传输开销。查询真实 profile，队列初值 2，积压丢旧帧；先在所属感知模块处理，模块间主要传结构化结果。采用 ROS 后才评测 intra-process/DDS，不默认同进程零拷贝。

```text
RGB + 深度 + CameraInfo
 -> 配对/对齐 -> RKNN2分割 -> 掩码内有效深度
 -> 相机坐标 -> 固定外参 -> 基座抓取候选
 -> 可达/限位/过期检查 -> 任务执行
 -> 接触 + 关节 + 再观察 -> 结果/恢复
```

相机、MCU 和主机可能使用不同时间域；保存采集/到达时间和偏差估计，不把异步采样叫作严格硬件同步。主机任务有效期使用单板单调时钟，不使用可能跳变的墙钟。

音频首轮统一到 16kHz mono PCM16 的 ASR 输入候选，即 32KB/s；实际采样率由声卡与后端确认。OVS agent/server 沿用上游 WebSocket/HTTP 契约传 PCM、文本和工具事件，不增加第二条原始音频总线。TTS 输出采样率由后端返回，播放前显式重采样。

先半双工/按键说话，再验收回声与用户打断。上游 voice_arm 的默认唤醒与 reSpeaker/Orin 校准不证明任意麦克风可用；中文唤醒需要已训练支持的模型，不能只改名称。AEC 使用真实播放参考或经验证的硬件，不靠 VAD 宣称全双工。[voice_arm 适配边界](https://github.com/Seeed-Solution/openvoicestream/blob/main/agent/ovs_agent/apps/voice_arm/README.md)

## 6. 通信契约与控制

首版通过上游 ToolRegistry 与机器人插件/执行器调用完成任务，复用现有事件、观察与结果接口；感知/接触队列保持有界。运动请求带 ID、boot ID、有效期和取消状态；执行侧去重、拒绝过期命令，不能靠重复调用重试未知结果的动作。

以下 ROS 接口仅为后续接入 ROS 时的候选，不是首版必建清单，也不是已实现的厂商接口。ROS 发行版按实际用户态选择，不由内核 6.1.118 推断。[RealSense ROS 支持说明](https://github.com/realsenseai/realsense-ros)

| 接口 | 类型 | 内容 |
|---|---|---|
| /robot/objects | Topic | 类别、掩码摘要/候选、坐标、置信度、深度有效性与时间 |
| /robot/contact | Topic | 接触测量、标定版本、序号与时间 |
| /joint_states、/tf、/tf_static | 标准 Topic | 本体反馈、运动与固定坐标系 |
| /robot/state、/robot/health | Topic | 任务阶段、关联 ID、故障与后端状态 |
| /robot/safety_state | Topic | actuator 发布的权威使能/急停/反馈状态 |
| /robot/execute_task | Action | 物体 ID、目标区、请求 ID、有效期；进度、结果与取消 |
| /robot/stop | Service | 高优先级受控停止；不是物理急停 |
| /robot/enable_motion | Service | 本地人工重新使能；LLM 无权限 |

观察使用有界 best-effort，任务/故障使用匹配的 reliable QoS；运动命令不持久重放。可靠传输不等于恰好执行一次或硬实时。运动请求带 ID、boot ID 和有效期；执行侧去重、拒绝过期与上次开机命令。[ROS 2 QoS](https://raw.githubusercontent.com/ros2/ros2_documentation/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)

LLM 只请求查询、抓放和取消等语义工具；对象/放置区引用已验证 ID，不直接指定任意坐标、关节轨迹或 shell。完整工具参数校验后才交给抓放任务插件；模糊目标先澄清。OVS 工具返回任务接受与实际完成应分清，未完成不能播报“已经放好”。

停止/取消先走本地高优先级入口，不依赖 LLM 重新理解。播放取消、生成取消与机械臂实际停止分别确认；迟到工具/音频直接丢弃。持物状态下取消不默认松爪，保持/安全放置按实机验收定义。

首版复用并验证上游 SO 执行器/动作表示，明确归一化、角度、编码器与夹爪映射；不要把 actions.yaml 的度直接当作 rad。任务结果不能只依赖 delay/sleep，需要反馈与接触判定。若采用 ROS，再使用 ros2_control + joint_trajectory_controller，关节单位与 URDF 匹配，并替换直接串口所有者。[轨迹控制器](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)

任务状态初始为 idle/observing/approaching/grasping/verifying/transferring/placing/recovering/fault。受限运动学使用已有库（C++ 首选 Orocos KDL 候选），根据五自由度能力限制抓取姿态；不要求任意六维位姿，也不手写通用 IK。初期固定低速轨迹与已测工作区，复杂 MoveIt 避障非前置要求。

## 7. 仓库与部署

当前仓库基于 OpenVoiceStream 固定快照完成专用化裁剪；目录遵循 OVS 原有契约。无关应用、平台模型和部署入口已移除，逐文件记录见 [裁剪清单](docs/file-inventory.json)。

```text
EdgeArm/
  server/                         # 复用语音服务与会话
  agent/ovs_agent/
    runtime.py                    # 迁移 OVS 共享生命周期，只注册 ChatMode
    apps/edgearm/              # 唯一 app、SO 插件、执行器及数据契约
    actuators/                    # 复用注册/契约
  configs/profiles/rk3588-default.json
  deploy/                   # 唯一默认部署、配置和锁
  services/rk1828-llm/             # 原有本地 LLM 服务
  third_party/rkvoice-stream/      # 固定版本原始依赖
  agent/tests/ server/tests/ tests/
  scripts/                  # 安装、检查、诊断和本地控制
  docs/                     # 范围、启动、验证与剩余工作
```

上游 commit、镜像、模型 manifest 和 Python 约束见 [版本锁](deploy/baseline.lock.json)。本轮未创建 GitHub fork、构建镜像或启动实机模型。保留 LICENSE/NOTICE/来源，不改名后宣称全部原创。旧的未跟踪实验记录移到忽略目录 `.local/archive/pre-so101`，不作为本项目验收证据。

voxedge 是独立依赖，RK Runtime 也涉及子模块；clone 产品仓库不等于依赖齐全。优先使用上游已固定的发布版本/配套镜像，只在确需修改引擎时维护对应源码/构建版本。不把其他平台的所有模型/依赖装入 RK 主机。[上游开发说明](https://github.com/Seeed-Solution/openvoicestream/blob/main/DEVELOP.md)

新增 C++ 感知/执行 Runtime 优先包装为插件使用的原生库/worker；确需 ROS 时才增加 ament/CMake/colcon 包，避免一开始新建五个独立机器人包。每条引入的新线程/进程都说明资源所有权和取消语义。

16GB 主机以全部业务进程常态合计约 12GB 内、保留约 4GB 系统与峰值余量为初步规划，不是实测结论，也不是给每个容器各分配 12GB。按 /proc/MemAvailable、进程 RSS/PSS、容器与协处理器测量调整。RK1828 自身内存与主机内存分别登记。单会话、有限图像/播放/任务队列，不同时加载多个对话后端，不靠 swap 掩盖长期超额。

模型与授权记录独立目录，日志轮转，128GB 存储保留余量；原始媒体默认不落盘。当前使用唯一 Compose 管理语音/LLM/agent，后续可由 systemd 管理该入口；核验 Runtime/设备权限、网络端口与资源后固定配置，不引入 Kubernetes。

服务默认本机访问；上游示例的 host network、0.0.0.0、privileged 和全 /dev 挂载须逐项核对，不能当作最终安全配置。网页局域网访问必须显式开启与鉴权。启动默认禁止运动；先自检/后端健康，再服务就绪，人工确认后使能。

## 8. 后续板端验收顺序

1. 源码已裁剪并固定依赖和模型清单；在真实板端确认发行版/板型/RK1828 内存与 SDK，重新执行主机诊断。
2. 先复现上游语音单会话及本地 LLM，在 16GB 上记录内存/延迟，不先换模型或改核心引擎。
3. 无自主运动地核验现有 SO 执行器、电机 ID、真实标定与读反馈；默认禁止运动，不使用占位标定抓放。
4. 低速监护下验证停止、限位、单写者、反馈年龄与动作取消，通过 edgearm 跑一个受限动作；当前软件检查不能替代该步骤。
5. 接入 D435i/RKNN2、固定外参和接触反馈，通过插件完成已知位置抓放、多源判定与有限恢复。
6. 测单模型与混合负载，再确定 NPU 分工、需要下沉的 C++ 模块和是否引入 ROS；新增功能保留原基线回归。
7. 数据/轻量适配、量化、故障/热运行和优化对照；分别记录上游已有能力与自己的修改结果。

硬件或 SDK 核验未通过时继续非运动链路/回放测试，不转为必须先在 Jetson 完成。全部性能数字在实测前只作为设计初值。
