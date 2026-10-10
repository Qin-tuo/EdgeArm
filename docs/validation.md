# EdgeArm 软件交付与验证

2026-10-10：项目更名为 EdgeArm，部署、脚本、项目文档分别集中到 `deploy/`、`scripts/`、`docs/`。清理 89 个空目录和两个无用文件，恢复保留的真实语音测试样本。项目 CLI 为 `edgearm`，Python 命名空间和唯一应用继续使用 `ovs_agent` / `voice_so101`。

`scripts/check.sh` 正常退出：**1214 项通过、2 项跳过、0 失败**，保留 5 条上游弃用告警；详见 [检查记录](software-checks.json)。同时通过 CLI、依赖、Compose、锁、导入、文件清单、文档链接、Shell 语法及 `git diff --check`。产品目录中无空文件夹；Git、虚拟环境、本地状态和原始依赖内部不属于清理范围。

本次环境允许 TCP 和本地 IPC，原先受环境限制的 45 项网络测试已执行并通过。测试运行器显式加载项目的 pytest-asyncio 插件，避免主机 ROS/colcon 插件依赖影响测试；没有更改生产事件循环。仍跳过 1 项需要真实 ASR 服务的测试和 1 项需要 voxedge>=0.0.15a2 的中文 URL 读法测试；本项目固定 voxedge=0.0.15a1。

软件验证覆盖默认禁止运动、严格工具参数、标定/限位/速度、反馈超时、取消与使能竞争、迟到计划、去重及任务结果判定。机械臂和传感器用明确 fake 验证软件契约。Silero ONNX 的 CPU 测试使用仓库已有资产与真实语音样本，不等于真实声卡、麦克风或回声验收。

本轮未构建或运行产品容器、下载外部模型、执行 RK 板端推理或真实机械臂/D435i/MCU。抓放 plan/verify 仍待硬件适配，运动中接触闭环尚未实现。BSP/SDK/固件、真实标定、停止与持物、峰值内存及 NPU 并发仍需实测。

逐文件记录见 [inventory](file-inventory.json)，启动及剩余工作见 [runbook](runbook.md)。日志、JUnit、Compose 展开结果保存在忽略目录 `.local/reports/`，旧历史报告保存在 `.local/archive/pre-so101`。
