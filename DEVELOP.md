# 开发

先读 [主文档](README.md)、[架构](ARCHITECTURE.md) 和 [运行手册](docs/runbook.md)。环境安装使用 `scripts/setup-dev.sh`，只初始化 Rockchip 子模块，Python 约束来自 `deploy/python.lock.txt`。

检查入口是 `scripts/check.sh`；软件测试不会启动模型、录音或真实电机。运行器会报告受限 IPC 环境并仅在测试中限制 selector 等待，软件测试结果不支持实机时延/安全结论。

新功能通过现有 OVS Plugin、ToolRegistry、Actuator 与 SensorStore/GraspBackend 接入。禁止复制 BaseApp/Session，禁止绕过唯一总线所有者。更新公共语音接口须运行会话和工具回归；驱动改动须覆盖默认禁动、限位、取消和启动/停止竞态。
