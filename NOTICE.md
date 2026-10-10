# 来源与许可

本项目基于 [OpenVoiceStream](https://github.com/Seeed-Solution/openvoicestream) 源码快照 `8fdc52ea65ce9a5dfd5dfe7e8b5d25943c269a11`，原始 [MIT LICENSE](LICENSE) 保留，未宣称上游语音框架和驱动为本项目原创。裁剪前本仓库提交及逐文件操作见 [清单](docs/file-inventory.json)。

`agent/ovs_agent/runtime.py` 来自 MultiModeApp；`apps/voice_so101/app.py` 来自 conversation；`minimal_so_follower.py` 和 `so_arm.py` 来自 voice_arm。改动包括产品入口、默认运动关闭、严格标定/动作/取消契约。

voxedge 使用已发布的 `0.0.15a1` 依赖；rkvoice-stream 子模块固定 `400e704a535e0ec00fa4f02d4ed2c4bc7b5dc308`，不修改其上游内容及许可。模型、厂商 Runtime 与容器基础镜像是独立第三方产物，具体来源、SHA256/digest 记录在模型 manifest 和 [版本锁](deploy/baseline.lock.json)。SDK、模型及运行库的使用条件分别按其原始许可核验。
