# RK SO-101 当前软件架构

目标要求来自 [系统架构书](01-so101-rockchip-system-architecture.md)，平台为 RK3588 16GB/128GB + PCIe RK1828。当前源码在 OpenVoiceStream 固定快照上裁剪；没有另建语音框架。

```text
USB 音频 → edgearm / OVS BaseApp + VoiceRuntime
                ├─ ChatMode / Session / ToolRegistry / 音频与停止意图
                ├─ SO101Plugin → 单一串行 worker → SOArmActuator → MinimalSOFollower → TTL
                ├─ SensorStore ← RGB-D 感知所有者（待接入 D435i / RKNN2）
                │              ← 接触接口所有者（待选型 ADC/MCU）
                └─ OVS /v2v/stream → server → RK3588 ASR/VAD/TTS
                         └─ OpenAI-compatible HTTP → RK1828 本地 LLM / 厂商 proxy
本地操作客户端 → 0600 Unix socket → 查询 / 停止 / 人工使能
```

## 复用和裁剪

`runtime.py` 迁自上游 MultiModeApp，保留事件、会话和音频处理，仅注册 ChatMode。`edgearm/app.py` 复用上游 conversation 的生命周期和可选 KWS 接法。SO 驱动迁自上游 voice_arm，继续使用 Actuator ABC 与工厂注册；修正占位标定、启动力矩、动作限位和取消行为。原应用目录、跨平台部署、其他模型配置与演示已移除。

voxedge 固定 `0.0.15a1`，rkvoice-stream 固定提交，保留第三方内部实现与原始许可。server 的通用协议/能力/健康接口仍沿用上游；产品 ASR/TTS 注册表仅接受 `rk.asr` 和 `rk.tts`，默认 profile 与下载路径仅保留 Qwen3-ASR + Matcha/Vocos。

## 所有权和安全状态

音频由 agent 唯一持有。SO101Plugin 的单线程 executor 是唯一总线调用入口，执行器锁序列化全部串口操作；查询只读取缓存。RGB-D 和 MCU 的未来适配器各持有一个设备，采集/推理在线程或受控 worker 中执行，通过应用事件循环发布结构化样本。SensorStore 每种样本仅保留最新一份，物体最多 32 个，保存序号、采集/到达时间、来源时钟和标定版本；任务使用主机单调时钟。

启动不启动力矩，不生成标定。舵机连接需要完整六电机真实标定；使能另需人工核对后的文件 SHA256。该哈希证明文件身份，不能证明标定测量正确或电机寄存器已匹配。现有适配器只接受 drive_mode=0；非零反向配置需单独适配/验收。驱动沿用上游角度换算：五关节为度，夹爪为百分比，原始范围由标定决定。

动作必须包含全部六关节；整段发送前检查有限数值、标定范围和每帧变化/延迟的速度界限。停止先设置线程事件、撤销运动权限，再发当前反馈位置作为保持目标；保留夹爪，不自动释放。稳定反馈只是一项软件观察，尚无物理停止或持物安全验收。断连/故障后需要人工恢复；重启与重新连接不会自动恢复运动权限。

## 语义任务接口

LLM 仅看到 `robot_query`、`robot_grasp_place`、`robot_cancel`。工具拒绝额外/缺失/错误类型参数；任务仅引用当前物体与配置的放置区 ID，不能写任意坐标、轨迹或 shell。运动使能只通过文件权限为 0600 的本地 Unix socket，不作为 LLM 工具。

请求包含 request_id、boot/cancel token 和最多五秒的有效期。最多保存 256 条请求记录，满时拒绝新请求，避免遗忘执行结果后重放动作。取消更新令牌，拒绝迟到工具；任何时刻只接受一个运动任务。accepted、completed、cancelled、rejected、fault 分开；完成必须有动作后新鲜的场景/接触观察以及独立结果验证。当前无默认抓放后端，缺少任一前置条件会拒绝任务。

ROS 2、CycloneDDS、ros2_control、MoveIt 保留为后续选项。接入 ROS 时须将串口所有权完全移交，停用直接串口驱动；不得建立第二套语音服务或双驱动。

## 部署与资源

唯一配置为 `deploy/compose.yaml`，三个服务分别构建源码 overlay，基础镜像 digest 固定。语音/LLM 依赖主机 BSP 与厂商 Runtime；初始兼容配置仍使用 privileged 和 /dev 挂载，设备权限缩减必须经过板端验证。agent 默认仅授予声卡访问，不授予机械臂/相机/MCU 设备。

语音 8GiB、LLM 3GiB、agent 1GiB 是初步容器限制，合计 12GiB，禁止 swap 扩容；它们不是实测占用，可能需要根据模型加载峰值调整。RK1828 自身内存单独登记。模型缓存和 Docker 数据位于 /userdata，日志轮转/热运行/NPU 并发仍需实机验收。
