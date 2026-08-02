# 检查点记录

正式检查点未复制到本地 Git 仓库，保存在训练服务器：

```text
/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp2_hypersim_smallset_overfit/artifacts/checkpoint.pt
```

- 文件大小：609,109,557 字节（约 581 MiB）
- SHA-256：`cbb63bccac283492c9fe2fb6451dfb9258157a37803a86fde56f616c320d719f`
- 内容：SSR 参数、AdamW 优化器状态、完成步数、基础模型标识和运行参数

已实际加载该文件，并在相同数据和配置下从第 2000 步继续执行第 2001 步。恢复后的
训练损失为 0.02919，前向、反向、优化器更新、评测和新检查点保存均成功。
