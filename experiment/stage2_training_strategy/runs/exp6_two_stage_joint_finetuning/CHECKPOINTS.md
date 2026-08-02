# exp6 检查点记录

正式检查点未提交到 Git，只保存在 ZJU3DV-S115 的用户安全目录：

`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp6_two_stage_joint_finetuning/artifacts/training`

| 文件 | 含义 | 步数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| `checkpoint.pt` | 完整验证集 K=3 点图 Rel 最低 | 250 | 1,508,130,399 | `ee4b6522ea264b06ae216d7bcfad8f964481a48f01fd3eb924028caae28f7368` |
| `latest_checkpoint.pt` | 完整训练结束，含优化器状态 | 2500 | 4,469,874,577 | `e393aa2855522a42b7f024d2009dee346f5fdffb6f524177453e33898097a444` |
| `resume_checkpoint.pt` | 最近周期边界，可断点恢复 | 2500 | 4,469,874,577 | `0f6efc2af77e3a6f79b09b1891ea5fe48824894f8db292b446bd757ca8b37217` |

最佳检查点位于预热阶段，不是保存错误。它只按验证集选择，测试集未参与选模。最终
检查点保留用于诊断完整联合训练后期的 SSR 正贡献与基础模型退化，不应作为当前默认
推理权重。
