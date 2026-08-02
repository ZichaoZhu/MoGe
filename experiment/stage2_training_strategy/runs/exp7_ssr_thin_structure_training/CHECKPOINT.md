# exp7 检查点记录

所有检查点仅保存在 ZJU3DV-S115，不提交到 Git。

服务器实验目录：

```text
/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp7_ssr_thin_structure_training
```

| 用途 | 步数 | 相对路径 | 大小（字节） | SHA-256 |
|---|---:|---|---:|---|
| 验证集最优 | 0 | `artifacts/training/checkpoint.pt` | 203,095,101 | `a19e0c3709900fd88baf189717d6cb7f5f03f39502f9da5f18204337e20fb94d` |
| 训练集最优 | 3800 | `artifacts/train_best_tracking/train_best_checkpoint.pt` | 609,120,043 | `ac1472b5088a1c0b6995e15b9989d82d535b98e543236c559915ceaa8769cb3a` |
| 最终状态 | 4000 | `artifacts/training/latest_checkpoint.pt` | 609,120,043 | `39baf8bc0117438d3fb85b546fb40ac96d712d8642c73e45e8113899e464b4e5` |

验证集最优是零初始化恒等映射，说明训练后的 SSR 没有跨场景泛化。训练集最优用于
本实验的训练拟合指标与 GIF，不应作为可泛化模型使用。

三者均为 SSR-only 检查点；基础模型固定从
`Ruicheng/moge-2-vitl-normal` 加载。训练集最优和最终状态额外包含 AdamW 状态与显式
随机数生成器状态，因此能够恢复训练。
