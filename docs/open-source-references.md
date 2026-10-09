# GitHub 开源项目参考与采用范围

参考核对日期为 2026 年 10 月 9 日。采用的是模块组织和公共接口边界，当前实现为独立代码；没有复制上游源码、安装 SDK、下载权重或将上游项目加入默认构建。可核对的默认分支提交与 GitHub 元数据保存于 [references.json](../third_party/references.json)。这些提交是调研记录，不是已经集成的依赖锁文件。

| 项目 | 公开能力与设计线索 | 本框架采用的方式 |
| --- | --- | --- |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 提供本地 ASR、TTS、VAD 及 C/C++ 接口 | 语音能力拆成业务处理器，前后处理与库接口封装留在模块内部 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 区分推理库、CLI 和服务端应用 | 核心运行库与应用装配分开，模型后端不控制整体业务流程 |
| [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | 展示模型导出与转换，以及 Python/C API 的设备推理 | 资产、转换工具与端侧运行解耦，保留独立的后端和精度记录 |
| [rknn-llm](https://github.com/airockchip/rknn-llm) | 区分 PC 上转换量化与板端 C/C++ runtime | 工作站产物通过清单登记后交给设备适配器，核心不包含工具链类型 |
| [wyoming-satellite](https://github.com/rhasspy/wyoming-satellite) | 区分采音、转写、合成开始结束和播放完成事件 | 阶段、首输出与任务完成采用不同事件，避免把生成音频当作实际播放 |

wyoming-satellite 在本次 GitHub 元数据检查中已归档，仅作为事件边界的历史参考，不作为持续维护的依赖。其远端 satellite 部署方式也没有被直接引入本项目。[仓库元数据](https://api.github.com/repos/rhasspy/wyoming-satellite)

RKNN 与 RKLLM 的公开说明不能用来推断 RK1828 的 SDK、固件或模型支持。RK1828 以后使用自己的适配器，依据实际厂商接口验证；它与 RK3588 RKNN/RKLLM 路径分别记录。[RKNN 官方说明](https://github.com/airockchip/rknn_model_zoo/blob/main/README.md)、[RKLLM 官方说明](https://github.com/airockchip/rknn-llm/blob/main/README.md)

GitHub 元数据标识 sherpa-onnx 和 RKNN Model Zoo 为 Apache-2.0，llama.cpp 与 wyoming-satellite 为 MIT；rknn-llm 的标识为 NOASSERTION。具体 SDK、二进制和模型文件需要分别按对应许可处理，仓库元数据不是它们的统一授权结论。当前没有引入第三方代码，因此这些记录不改变本仓库的代码分发许可。

