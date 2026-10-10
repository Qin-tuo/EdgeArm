# RK1828 本地 LLM 服务

产品只通过 [唯一部署](../../deploy/compose.yaml) 和 `deploy/Dockerfile.llm` 构建。基础镜像/模型来源固定在 [版本锁](../../deploy/baseline.lock.json)。当前未构建或运行；旧板测量不代表本机验收。

`rk1828_llm_server.py` 和 `entrypoint.sh` 沿用 OVS 的 OpenAI-compatible shim；默认本机 1828 端口、2048 上下文、单加载尝试。主机 RKNN3 库与 proxy/EP 固件必须配套，speech 的 RKNN2/RKLLM 语音路径不能替代 RKNN3。排查步骤见 [运行手册](../../docs/runbook.md)。

## 厂商 worker 维护

默认使用锁定基础镜像中的原生 worker。只有明确需要修改原生代码、并已取得匹配厂商 SDK 时，才维护 `patches/Qwen3-cpp-main.cc` 并在真实 RK1828 主机通过 model-zoo 的配套脚本构建：

```bash
cd <匹配版本的rknn3-model-zoo>
cp <本仓库>/services/rk1828-llm/patches/Qwen3-cpp-main.cc examples/Qwen3/cpp/main.cc
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3
```

编译产物可在忽略目录 `deploy/rk1828-runtime/` 暂存，记录二进制、完整 lib 目录、SHA256、SDK/proxy/固件版本并更新 MANIFEST；这不是第二个产品部署入口。重新制作并验收基础镜像后才更新产品锁，不能假设本地编译二进制与已发布 ABI 相同。

worker 保留 stderr `READY 1` 与 stdin/stdout 帧协议；stdout 仅传长度前缀 token/EOS，诊断不能混入。服务线程串行持有 worker 并排空请求，HTTP 客户端断开也不能遗弃未读帧。工具格式、KV 行为、长会话、取消恢复和超时须在匹配 SDK 上重新测量。

模型 manifest 固定一组 Qwen3-4B 产物；入口验证大小和 SHA256 后才加载。主机内存与 RK1828 EP 内存分别记录。失败后先诊断驱动、proxy/固件、库与产物完整性，不反复尝试模型或复位卡。
