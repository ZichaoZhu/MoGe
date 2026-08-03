# 阶段三：SSR 稳定性与归一化诊断

## 阶段目标

解释阶段二后期出现的残差漂移：它究竟是普通 OOM、梯度爆炸、Base 状态变化，
还是 SSR 内部统计和递归精修造成的几何坍塌。

## 原实验映射

| 原实验 | 内容 | 主要结果 |
|---|---|---|
| [Exp16](runs/exp16_post_detach_short_joint/README.md) | 联合阶段残差越界诊断 | 反向前检测到绝对对数残差 0.5331 并安全终止，并非先发生 OOM |
| [Exp17](runs/exp17_ssr_residual_outlier_scan/README.md) | 全数据残差离群扫描 | train 模式无越界，eval 模式 6 张越界，最大 1.0458 |
| [Exp18](runs/exp18_base_ssr_mode_factorial/README.md) | Base/SSR 模式四组合析因 | Base 模式无影响，离群由 SSR eval 模式触发 |
| [Exp19](runs/exp19_ssr_batchnorm_recalibration/README.md) | BatchNorm 运行统计重校准 | 离群 6→2、最大残差 1.0458→0.8124，未根治 |
| [Exp20](runs/exp20_stateless_ssr_batch_statistics/README.md) | 单图即时统计推理 | 离群全部消失，但验证/测试仍退化 |
| [Exp21](runs/exp21_iteration_specific_bn_statistics/README.md) | 逐轮独立 BatchNorm 统计 | 有小幅改善，仍有 3 张离群 |
| [Exp22](runs/exp22_batch_independent_normalization_screen/README.md) | LayerNorm/GroupNorm 筛选 | 极端残差消失，但 SSR 修正接近恒等 |
| [Exp23](runs/exp23_true_microbatch2_batchnorm/README.md) | 真实双图 microbatch 短程对照 | 短程无离群，尚不足以检验强修正阶段 |
| [Exp24](runs/exp24_microbatch2_long_detached/README.md) | 真实双图 microbatch 长程对照 | 训练改善约 21%，离群和最大残差下降，留出集仍退化 |
| [Exp28](runs/exp28_smooth_bounded_residual_joint/README.md) | 平滑有界残差联合训练 | 稳定完成 200 次联合更新并改善训练集，验证集全图仍退化 |
| [Exp29](runs/exp29_smooth_bounded_residual_long_joint/README.md) | 平滑有界残差长程联合训练 | 续训至 step 3600；最佳 step 2800 训练集改善，原始残差最终触发安全停止，留出集仍退化 |

## 因果链

```text
小 microbatch / 单域训练
        ↓
SSR BatchNorm 运行统计偏移
        ↓
eval 模式产生异常对数深度残差
        ↓
共享 SSR 在 K 轮中递归放大
        ↓
深度跨度和活动体素范围扩大
        ↓
显存增长，严重时触发 OOM
```

## 阶段结论

- 更准确的故障名称是“SSR 残差/几何状态坍塌”，不是单一的梯度爆炸。
- SSR BatchNorm 的 train/eval 统计失配是已定位的直接因素，Base 模式不是原因。
- 更大的真实 microbatch 可以改善统计质量和稳定性，但不能解决留出集退化。
- LayerNorm/GroupNorm 能消除极端残差，却同时使有效精修接近消失，所以“替换 Norm”
  不是已经成立的最终方案。
- `0.1*tanh(raw/0.1)` 能阻止单轮几何更新越界并让联合训练继续，但它没有
  自动带来留出域收益。
- 数据集多样性不足主要解释过拟合；它与 batch 统计问题相关，但尚未被证明是残差
  爆炸的直接原因。
