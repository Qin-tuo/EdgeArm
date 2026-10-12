# RK1828 本地 LLM 服务

容器部署通过 [Compose](../../deploy/compose.yaml) 和 `deploy/Dockerfile.llm` 构建。基础镜像/模型来源固定在 [版本锁](../../deploy/baseline.lock.json)。当前 BSP 阻断容器运行；配套 V1.1.0 worker 的短样本通过，但真实普通多轮和 tools 均出现设备断链。同 SDK 未修改的厂商 HTTP 服务也在首个请求复现相同 83-token 输入故障；更换项目 worker 未解决稳定性问题。原生服务默认拒绝工具请求，答案质量尚未通过，见 [检查报告](../../docs/framework-audit.md)。

用户授权临时将 prefill 改为 EFFICIENT、work 保持 EFFICIENT 后，同一厂商服务通过原始故障输入、连续历史和 SSE，共 33 次请求；项目 worker 随后也通过 41 次请求和页面原故障顺序。模式入口支持显式 `EDGEARM_RK1828_PREFILL_MODE=efficient`，默认 PERFORMANCE 未改变。这仅是 EFFICIENT/EFFICIENT 的短期传输结果；NORMAL/EFFICIENT 随后仍复现断链，两次完全跳过 SMI 的主机重启对照也在首条 31-token 请求失败。根因、冷上电、长期稳定性、角色历史与答案质量仍未验收。API 的 usage 尚为占位值，实测 token 数应参考 worker 日志。

`rk1828_llm_server.py` 和 `entrypoint.sh` 沿用 OVS 的 OpenAI-compatible shim；默认本机 1828 端口、2048 上下文、单加载尝试。主机 RKNN3 库与 proxy/EP 固件必须配套，speech 的 RKNN2/RKLLM 语音路径不能替代 RKNN3。排查步骤见 [运行手册](../../docs/runbook.md)。

## 厂商 worker 维护

容器默认使用锁定基础镜像中的原生 worker。本机原生入口改用独立构建的 V1.1.0 worker，不改变发布镜像锁。维护 `patches/Qwen3-cpp-main.cc` 时必须取得匹配厂商 SDK，将源码复制到隔离目录，再构建；不要覆盖 SDK 原始交付文件、安装系统驱动或固件。

本次在 ARM64 Ubuntu 22.04 / GCC 11.4 原生构建，使用 model-zoo revision `174e44c77230735b1458946debb62b3982c1ee58`。最小源码树只需要 `3rdparty`、`utils` 和 `examples/Qwen3/cpp`，无需复制模型：

```bash
cd /userdata/EdgeArm
MODEL_ZOO=/userdata/RK_LLM/artifacts/upgrade/v1.1.0/rknn3-model-zoo
WORKER_SOURCE="$PWD/.local/build/rknn3-worker-v1.1.0"
WORKER_STAGE="$PWD/.local/runtime/rk1828-v1.1.0"
mkdir -p "$WORKER_SOURCE/examples/Qwen3"
cp -a "$MODEL_ZOO/3rdparty" "$MODEL_ZOO/utils" "$WORKER_SOURCE/"
cp -a "$MODEL_ZOO/examples/Qwen3/cpp" "$WORKER_SOURCE/examples/Qwen3/"
cp services/rk1828-llm/patches/Qwen3-cpp-main.cc "$WORKER_SOURCE/examples/Qwen3/cpp/main.cc"
cmake -S "$WORKER_SOURCE/examples/Qwen3/cpp" -B "$WORKER_SOURCE/build-native" \
  -DTARGET_SOC=rk3588 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DDISABLE_RGA=ON -DDISABLE_LIBJPEG=ON -DCMAKE_INSTALL_PREFIX="$WORKER_STAGE"
cmake --build "$WORKER_SOURCE/build-native" -j4
cmake --install "$WORKER_SOURCE/build-native"
```

构建命令应只用于新的暂存目录，不能覆盖正在运行的 worker。编译产物记录二进制、完整 lib 目录、SHA256、SDK revision、源码哈希及硬件验收结果。`scripts/start-native-llm.sh` 校验 Runtime `MANIFEST.json` 中 `files` 的 `size_bytes` 和 `sha256`，再验证模型并设置执行模式。本机清单为 `.local/runtime/rk1828-v1.1.0/MANIFEST.json`；这些忽略目录内的产物不会随 Git 克隆提供。重新制作并验收基础镜像后才更新产品锁。

worker 保留 stderr `READY 1` 与 stdin/stdout 帧协议；stdout 仅传长度前缀 token/EOS，诊断不能混入。服务线程串行持有 worker 并排空请求，HTTP 客户端断开也不能遗弃未读帧。工具格式、KV 行为、长会话、取消恢复和超时须在匹配 SDK 上重新测量。

模型 manifest 固定一组 Qwen3-4B 产物；入口验证大小和 SHA256 后才加载。主机内存与 RK1828 EP 内存分别记录。失败后先诊断驱动、proxy/固件、库与产物完整性，不反复尝试模型或复位卡。
