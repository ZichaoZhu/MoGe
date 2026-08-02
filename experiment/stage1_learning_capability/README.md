# 阶段一：SSR 学习能力与小数据基线

## 阶段目标

先回答最基础的问题：当前 MoGe-3/SSR 实现是否能够学习几何残差，以及训练集改善
能否自然迁移到独立样本。

## 原实验映射

| 原实验 | 内容 | 主要结果 |
|---|---|---|
| [Exp1](runs/exp1_hypersim_single_batch_overfit/README.md) | 单批次 SSR 过拟合 | 深度 Rel 由 1.844% 降至 0.278% |
| [Exp2](runs/exp2_hypersim_smallset_overfit/README.md) | 多场景小数据 SSR 训练 | 训练深度 Rel 由 3.163% 降至 1.271%，独立场景未改善 |
| [Exp3](runs/exp3_fine_structure_visualization/README.md) | 细结构迭代可视化 | 训练细结构 K=3 点图 Rel 由 6.374% 降至 2.765%，验证仍退化 |
| [Exp4](runs/exp4_hypersim_multiscene_generalization/README.md) | 多场景 SSR 泛化评测 | 训练点图改善 13.03%，验证恶化 3.33% |
| [Exp5](runs/exp5_ssr_refinement_stability_ablation/README.md) | K 与边缘损失稳定性消融 | 四组最佳检查点均为 step 0；K=1 较稳定但验证仍恶化 |

## 阶段结论

- 单图和小训练集可以明显过拟合，证明体素化、像素回映射、SSR 反向传播和残差更新
  链路具备学习能力。
- K 增大能够增强修正能力，也会放大不稳定性；K=1 通常比 K=3 稳定。
- 固定 Base、只训练 SSR 时，训练集收益不能可靠迁移到独立样本。
- 点云 GIF 适合观察离轴细杆形态，但不能替代定量指标。

本阶段证明的是“实现能学”，不是“方法已泛化”，也不是完整论文效果复现。
