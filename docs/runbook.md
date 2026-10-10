# RK SO-101 启动与后续验收

本手册区分软件交付和硬件验收。真实音频、RK1828 推理、机械臂、D435i 和 MCU 接触链路均未在本轮运行。原主机历史清单不证明当前设备可用；先重新运行 doctor。

## 1. 开发环境

```bash
scripts/setup-tools.sh
scripts/setup-dev.sh
scripts/check.sh
python3 scripts/doctor.py --json
```

Python 语音依赖使用 `deploy/python.lock.txt`，voxedge=0.0.15a1。舵机依赖额外安装：

```bash
.venv/bin/python -m pip install -e './agent[arm]' -c deploy/python.lock.txt
```

feetech-servo-sdk=1.0.0、pyserial=3.5 为板端候选锁定版本，未在本轮安装/验证。RGB-D 需要板端 librealsense、RKNN2 模型/Runtime，MCU 协议仍待选型；这些不进入无外设语音开发环境。

## 2. 恢复主机前置条件

确认 ARM64 RK3588、16GB/128GB 规格、Ubuntu 用户态与 BSP。确认 /dev/pcie-rkep-*、RKNN3 库、设备固件和 transfer proxy 的兼容版本；仅文件存在不能证明推理可用。历史环境曾有 rk182x.service 失败，需在真实主机复查并查看日志：

```bash
sudo journalctl -u rk182x.service -n 100 --no-pager
```

模型和 Docker 数据放在 /userdata，保留系统与下载峰值空间。确认实际麦克风/扬声器连接和全双工 USB 声卡；禁止用声卡枚举代替音频闭环验证。容器默认只给 agent 声卡设备，机械臂未接入。

## 3. 构建和启动唯一默认方案

所有产品镜像均需构建源码 overlay；固定发布镜像仅提供原生运行库基础。当前没有构建、拉取或运行这些镜像，不应直接宣称启动成功。

```bash
sudo scripts/start-docker.sh
```

该进程保持前台；另一个终端：

```bash
mkdir -p .local/data/{asr,tts,llm,voice-data,agent-state} .local/run/edgearm
chmod 700 .local/run/edgearm
sudo scripts/compose.sh build
sudo scripts/compose.sh up -d
sudo scripts/compose.sh ps
sudo scripts/compose.sh logs --tail=100
```

构建不依赖手工复制其他平台模型，但需要网络拉取 digest 锁定基础镜像与 Python 包；本轮受限环境未验证网络构建。speech、LLM、agent 内存上限分别为 8/3/1GiB，禁止 swap；加载峰值未知，若不足须先测 RSS/PSS、MemAvailable 和 RK1828 内存，重新分配/降低负载后再验收。LLM 失败启动不自动重试，先排查 proxy/驱动/固件。

修改声卡配置可复制 `deploy/cm3588.env` 为 `cm3588.local.env`，设置 EDGEARM_AUDIO_INPUT/OUTPUT，通过 EDGEARM_ENV_FILE 传给 compose.sh。默认配置固定中文、单会话、2048 token LLM 上下文和按键说话；回声/打断验证后才能调整。

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
| 麦克风 / 扬声器 | 保留已有真实语音测试样本并补充板端采集样本；记录录放音、PTT、重连；AEC/打断另行验收 |
| SO-101 标定 / SDK | 验证六电机 ID/方向/offset/量程、SDK 分组读反馈完整性及有限串口超时；人工使能不跳变 |
| 停止 / 取消 | 监护下测各阶段停止和反馈年龄；包含拔线、读写故障、持物、不自动松爪、断电支撑 |
| D435i / MCU | 实测相机 profile/外参、源时钟转换与接触标定；过期、乱序、丢设备必须禁止任务 |
| 抓放闭环 | 实现 plan/verify 与运动中接触/健康检查、受限恢复；复用唯一总线，禁止仅依赖静态轨迹与延迟宣称闭环 |
| 整机并发 / 热运行 | 测 NPU 共存、内存峰值、排队/控制抖动、温度和故障恢复；保存真实结果和失败样本 |

当前 plan/verify 是扩展契约，静态序列并未实现运动中接触反馈控制。没有默认后端、未经实机校验时，不开放自主抓放。ROS / MoveIt 和 CPU 降级不属于本轮默认启动范围。
