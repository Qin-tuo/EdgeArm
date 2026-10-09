# 模型与数据资产登记

此目录保存小型清单与模板。权重、数据集和大量运行产物放在仓库外，使用配置引用。`manifests/model.example.json` 是未填写的结构示例，不代表已经准备或验证了模型。

已有文件可以登记为模型、数据集或普通产物：

```bash
python3 tools/assets/register.py \
  --id experiment-model-v1 --kind model \
  --source https://example.org/model-origin --license reviewed-license \
  --output assets/manifests/experiment-model-v1.json /path/to/model-file
```

来源与许可参数必须替换为实际信息。工具计算文件 SHA256、记录大小并使用独占创建写入清单；不会覆盖已有清单，也不会复制或下载文件。`permission_verified=false` 表示登记不等于授权核验。

