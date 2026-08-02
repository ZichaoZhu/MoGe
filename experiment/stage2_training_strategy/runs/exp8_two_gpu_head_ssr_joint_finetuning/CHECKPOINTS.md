# exp8 检查点

检查点仅保存在服务器安全目录：

`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp8_two_gpu_head_ssr_joint_finetuning/artifacts/training`

| 文件 | 含义 | 大小（字节） | SHA-256 |
|---|---|---:|---|
| `checkpoint.pt` | 验证 K=3 最佳；step 0 | 1,508,130,463 | `7f55539817192228a8dcaa7b1e8f83b855bce3eccf6cdbb31ebcd07b6898cd90` |
| `latest_checkpoint.pt` | step 600 完整模型与优化器 | 2,034,630,921 | `2a05d61b2ae8a2617a299587c2494df1a892ac678fe4bf6fbae66cb4356cf871` |
| `resume_checkpoint.pt` | step 600 断点恢复状态 | 2,034,630,921 | `f9d3b4c2bed3858f28667bafa42d6bef6a28b5fd86511656d6a73c732a5d6c2a` |

`checkpoint.pt` 为 step 0 不是保存错误，而是因为所有训练后检查点的验证 K=3 点图 Rel
均未优于初始值。step 600 的训练效果与诊断可视化使用 `latest_checkpoint.pt`。
