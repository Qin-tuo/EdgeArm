# SO-101 语义工具

`voice_so101` 复用 OVS `ToolRegistry` 和工具执行循环，使用专属注册表。默认配置只公开下列三个工具；人工使能独立于 LLM。

| 工具 | 用途 |
|---|---|
| `robot_query()` | 查询当前状态、物体/放置区 ID、boot/cancel token 和主机单调时间 |
| `robot_grasp_place(object_id, zone_id, request_id, boot_id, expires_at)` | 请求抓放已验证物体到已配置放置区 |
| `robot_cancel()` | 撤销运动权限并请求保持当前关节/夹爪 |

参数使用类型生成的 JSON Schema，并设置 `additionalProperties: false`。注册表拒绝未知、缺失或类型错误参数；任务入口继续核验 ID、有效期、去重、忙碌状态、新鲜感知、接触健康和人工使能。期限使用同一主机单调时钟，最多五秒。只传任务意图，不能传关节、任意坐标或 shell。

```yaml
tools_enabled: true
tools_default_allowlist: [robot_query, robot_grasp_place, robot_cancel]
tools_max_iterations: 3
```

`accepted` 表示任务被接受。通过 `robot_query` 查询实际结果；只有 `completed` 才表示结果已通过适配器独立验证。动作发送结束后必须取得新的场景、接触与关节反馈；当前没有默认 D435i、MCU 或抓放后端，前置条件不满足时返回拒绝。

OVS 每轮工具仍顺序执行，发出 `tool_call_started`、`tool_call_completed`、`tool_call_error` 等事件。工具调用结束和物理任务结束是两个事件。长任务由机器人插件管理唯一活动任务，工具立即返回接受状态；取消通过线程事件传给执行器，不靠取消 Python 协程代替机械臂停止。

工具作者应让 `asyncio.CancelledError` 传播；不得在 planner/verifier 中直接访问总线。新增任务或感知适配器必须共用现有总线所有者、SensorStore 和结果判定契约。接口及实机验收见 [架构](../../ARCHITECTURE.md) 和 [运行手册](../runbook.md)。
