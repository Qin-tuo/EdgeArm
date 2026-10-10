# SO-101 复用的 OVS 并发契约

保留 `server/core/concurrency_capability.py`、`coordinator.py`、`session_limiter.py`、`worker_io.py` 和原有会话测试。产品默认只启用 `rk.asr` / `rk.tts`、`rk3588-default` 与一个语音会话。

| 层 | 所有权与边界 |
|---|---|
| ASR/TTS 后端 | 沿用 voxedge/rkvoice-stream 生命周期与能力声明；RK profile 采用共享 NPU 串行策略 |
| 会话与调度 | OVS limiter/coordinator 管理会话槽、队列、取消及资源租约；能力声明不是硬件并发验证 |
| WorkerIO | 按 request ID 分发事件；EOF、取消及异常须释放 in-flight 和 semaphore，不接纳未注册请求的事件 |
| agent 音频 | OVS capture/playback 所有者；阻塞采集和原生设备初始化在实际音频启动时发生 |
| SO-101 | 插件的单线程 executor 是串口唯一入口；查询缓存、动作读取反馈、取消先发送线程事件 |
| RGB-D / 接触 | 待接入的设备所有者处理原始数据，通过应用事件循环发布有界结构化样本 |

`ConcurrencyCapability` 保留 `supports_parallel`、`max_concurrent`、`is_stateful`、`requires_exclusive_device`、`scaling_mode`、`vram_mb_per_slot`。可并发与可热重载是不同能力；产品不会因一般能力字段存在而启用另一平台或模型。

RK3588 ASR 核心配置沿用固定 profile，视觉接入前必须实测分工、排队、CPU 抖动、模型加载峰值与温度。16GB 板卡上的 8/3/1GiB 容器限制属于初步预算。配置验证、fake backend 及线程测试不证明 NPU 并发、音频延迟或物理停止达到要求。

取消任务不意味着底层 native worker 已退出；SO 串口 worker 不被弃置，停止确认独立记录。ROS 接入时必须移交总线所有权。具体安全契约见 [当前架构](../../ARCHITECTURE.md)，实机步骤见 [运行手册](../runbook.md)。
