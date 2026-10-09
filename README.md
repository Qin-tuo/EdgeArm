# Robot-voice-stack

当前目标是搭建可更换业务流程、模型和硬件的通用 C++ 工程框架。离线语音与 RK3588/RK1828 保留为候选应用方向，语音链路作为示例模块，不绑定核心运行层。

- [原始任务书](02-companion-voice-rockchip%281%29.md)
- [任务拆解与公开数据方案](docs/task-breakdown.md)
- [实施计划书与工时排期](docs/project-plan.md)
- [通用框架架构](docs/architecture.md)
- [GitHub 参考项目与采用范围](docs/open-source-references.md)
- [业务模块与后端扩展方法](docs/extending.md)

已实现配置、类型化数据包、处理器注册、顺序流水线、有界异步任务队列、合作式取消、包含排队时间的超时、错误恢复和 JSONL 指标记录。默认构建只需要 C++17、CMake 和线程库；启用测试还需要 Python 3，不要求模型、音频设备、NPU SDK 或训练工作站。

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

只构建核心库和测试，可增加 `-DRVS_BUILD_EXAMPLES=OFF`；不运行测试可增加 `-DBUILD_TESTING=OFF`。

## 运行示例

以下命令在仓库根目录执行：

```bash
# 通用文本处理
./build/rvs-run --config configs/echo.ini --input "你好"

# 模拟语音处理器装配，生成固定回答和静音 WAV
./build/rvs-run --config configs/voice-mock.ini --input "介绍一下这个项目" --repeat 3

# 只依赖核心的数值处理示例
./build/rvs-custom-demo

# 查看注册的处理器
./build/rvs-run --list-processors
```

运行结果为 JSON，最后一行给出独立的产物目录。目录中包含配置快照、运行说明、`metrics.jsonl` 和示例产物；默认位于忽略 Git 的 `runs/`，可用 `--output-root` 修改。

模拟语音链路使用转写夹具、固定回复和静音采样，所有相关事件标记 `simulated=true`。它验证框架接口与调度行为；真实 ASR、LLM、TTS、麦克风、扬声器和硬件加速尚未接入。`voice-cpu.example.ini` 是未来接入模板，当前运行会明确返回 `unsupported`。

## 指标与资产工具

```bash
python3 tools/benchmarks/summarize.py runs/<run-id>/metrics.jsonl
```

汇总输出各阶段的样本数、状态计数、成功样本 P50/P95，并区分后端、模型与模拟标记。模型与数据清单登记方式见 [assets/README.md](assets/README.md)。

原任务书和 320 小时计划保留为历史候选方案，后续目标确定后再更新实施范围。当前不执行硬件联调、SSH 工作站检查或训练。
