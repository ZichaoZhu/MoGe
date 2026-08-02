# exp4 检查点记录

检查点未提交到 Git，只保存在 ZJU3DV-S115 的用户安全目录：

`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp4_hypersim_multiscene_generalization/artifacts/training_final`

| 文件 | 含义 | 步数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| `checkpoint.pt` | 验证点图 Rel 最低 | 0 | 203,095,101 | `3fb3cfe22ed187dba7e1e0e45e3e474adc1677ea2b2d7512815aebd7c86dd93c` |
| `latest_checkpoint.pt` | 完整训练结束 | 8000 | 609,120,043 | `013a36487d924e2338370ea71e17e030dfc1d8152d5e7234842e7cee600150a9` |

最佳检查点为 step 0 不是保存错误：SSR 输出层以零初始化开始，后续验证点图 Rel 均未
优于初始值，因此预注册的模型选择规则保留了严格恒等映射。分析训练后 SSR 行为时应
使用 `latest_checkpoint.pt`。
