# 阶段二：论文式训练策略与模块贡献

## 阶段目标

比较固定 Base、只训练 SSR、更新二维几何头、分离训练、联合训练以及不同数据规模，
并通过同一检查点的 K=0 与 K≥1 区分 Base/2D Head 和 SSR 的贡献。

## 原实验映射

| 原实验 | 内容 | 主要结果 |
|---|---|---|
| [Exp6](runs/exp6_two_stage_joint_finetuning/README.md) | 两阶段 Base 与 SSR 联合微调 | 验证/测试改善 2.26%/6.08%，主要来自 K=0 |
| [Exp7](runs/exp7_ssr_thin_structure_training/README.md) | 冻结 Base 的细结构 SSR 训练 | 训练改善 41.71%/49.47%，验证明显恶化 |
| [Exp8](runs/exp8_two_gpu_head_ssr_joint_finetuning/README.md) | 双卡二维模块与 SSR 联合训练 | K=0 验证/测试改善，K=3 相对 K=0 退化 |
| [Exp9](runs/exp9_thin_structure_single_image_staged_overfit/README.md) | 五张单图论文式两阶段极限过拟合 | 3/5 张最终 K=3 优于 K=0 |
| [Exp10](runs/exp10_hypersim_48_staged_joint_overfit/README.md) | 48 图两阶段训练 | step 3300 后 SSR 漂移，安全中止 |
| [Exp11](runs/exp11_hypersim_100_train_staged_joint_overfit/README.md) | 100 图两阶段训练 | 低学习率抑制坍塌，但 detached 长训仍漂移 |
| [Exp12](runs/exp12_hypersim_100_immediate_joint_finetuning/README.md) | 从 Exp11 最佳点立即联合训练 | 训练小幅改善，验证 K=3 恶化 1.72% |
| [Exp13](runs/exp13_fixed_base_ssr_diagnostic/README.md) | 固定原始 Base 的 SSR 学习率诊断 | 高学习率拟合训练结构，低学习率更稳定 |
| [Exp14](runs/exp14_head_ssr_short_joint/README.md) | 强 SSR 检查点短程解冻二维头 | K=0 三划分均改善，K=3 未超过起点 |
| [Exp15](runs/exp15_detached_head_ssr_coadaptation/README.md) | Detached 二维头与 SSR 共适应 | 训练 K=3 改善约 25%，K=5 与留出结构仍退化 |

## 阶段结论

- 固定 Base、只训练 SSR 不是当前小数据复现的充分方案。
- 更新 Base/2D Head 能稳定改善 K=0，是多次实验中的主要收益来源。
- 分离预热有助于避免早期 SSR 梯度破坏 Base；联合训练允许 refined loss 进一步塑造
  二维表示，两者都具有合理性。
- 当前结果尚未证明 SSR 在留出数据上的独立贡献：K=3 经常不优于同检查点 K=0。
- 48/100 图仍远小于论文的大规模混合训练，单一 Hypersim 域会产生明显过拟合。

本阶段验证了论文训练路线的重要性和当前实现的学习能力，但不能据此宣称已经复现
论文总体性能或泛化能力。

## 交互式查看器

Exp9 的 `viewer/` 保留在原归档中，可比较训练前后、不同 K、不同样本和实验入口。
当前线上版本仍为 `https://moge3-exp9-viewer.vercel.app/`。
