# Exp11：100 张 Hypersim 两阶段联合训练

## 当前状态

用户已审核并通过 52 张新增图片。实验复用 Exp10 已审核的 48 张训练图，
合并为 100 张 Hypersim 训练图；验证集和测试集继续使用 Exp10 中场景组隔离的
16/16 张图片。

首次正式分支在 step 4478 因连续大梯度被稳定性保护终止；从 step 3000 最佳
模型启动的低学习率恢复分支在 step 4000 由人工监控保护性停止。恢复分支没有
OOM、NaN、梯度爆炸或 SSR 数值坍塌，但训练集 K=3 Point Rel 相对最佳点的回退
从 `0.48%`、`0.91%`、`2.05%` 加速至 `5.74%`，因此没有继续消耗算力进入联合
阶段。当前 Exp11 的最佳结果仍为 step 3000，训练集 K=3 Point Rel 为
`3.3448%`。

候选筛选不读取任何模型预测或验证、测试指标，只使用训练源中的 RGB 和
Hypersim 像素精确深度。现有 Exp10 的全部场景以及验证、测试场景组均被排除，
以避免相邻帧重复和数据泄漏。

## 候选审核

最终候选共 52 张，来自 52 个不同场景，覆盖 42 个场景组；每个场景组最多
2 张。它们与 Exp10 的 48 张训练图不存在样本或场景重叠，也不包含 Exp10
验证集、测试集所在的 `ai_043`、`ai_047`、`ai_046`、`ai_050` 场景组。

联系表中的红框是 GT 细结构检测器给出的参考区域，只用于帮助审核；后续训练
仍使用完整 RGB 和完整深度图，并不会只训练红框区域。

- [候选 N01–N13](data_review/candidate_52/candidate_52_contact_sheet_01.jpg)
- [候选 N14–N26](data_review/candidate_52/candidate_52_contact_sheet_02.jpg)
- [候选 N27–N39](data_review/candidate_52/candidate_52_contact_sheet_03.jpg)
- [候选 N40–N52](data_review/candidate_52/candidate_52_contact_sheet_04.jpg)
- [候选清单与源文件记录](data_review/candidate_52/candidate_52.json)
- [人工替换记录](data_review/candidate_curation.json)

候选已于 2026-07-31 获得用户确认，审批记录见
[candidate_approval.json](data_review/candidate_approval.json)。

## 训练设置

- 输入分辨率：`384×512`，训练 `K=3`。
- 总计 10000 个优化步；前 5000 步将 Base 特征从 SSR 输入处断开，后 5000
  步联合训练 Base、二维几何头和 SSR。
- DINO 前 1000 步冻结学习率，至第 2000 步完成预热。
- SSR、二维头、DINO 学习率分别为 `1e-6`、`1e-6`、`1e-7`。
- 多卡通信使用 Exp10 已验证的 Gloo 后端；首次四卡 NCCL 冒烟在模型加载前
  由第 4 个进程触发 `SIGSEGV`，失败记录予以保留，不作为训练结果。
- 每张卡 microbatch 为 1；四卡或两卡时全局 batch 为 8，三卡时为 6。
- 正式训练前先执行 12 步多卡冒烟测试；训练期间每 10 分钟记录一次进程、
  GPU、最新训练步和周期评测状态。
- 启动器不会终止其他用户的进程，只选择启动时剩余显存不少于 18000 MiB、
  利用率不高于 30% 的 GPU，并在符合条件的卡中使用尽可能多的 GPU。

## 首次训练中断与恢复

首次四卡正式分支在 step 4478 由梯度稳定性保护主动终止。训练集 K=3 Point
Rel 在 step 3000 达到最低的 `3.3448%`，之后连续退化；step
3823/4013/4179/4427/4461 共跳过五次梯度范数超过 100 的更新，最终一次保护
事件的梯度范数为 `172.44`。这不是 OOM，且异常更新没有写入模型。

恢复分支固定从 SHA-256 为
`1a665e760516fde4e00082b9743a0fd8c7eed98d298cb76bb012b813227c98ce`
的 step 3000 最佳模型开始，重置优化器，并将 SSR、二维头、DINO 学习率分别
降至 `5e-7`、`5e-7`、`5e-8`。恢复时最多使用两张满足 18000 MiB 空闲显存
门槛的 GPU，并继续每 600 秒记录一次健康状态。

## 恢复分支结果

恢复分支使用物理 GPU 1、0 和 Gloo 后端，从 step 3000 运行至 step 4000，
始终处于 `detached_warmup` 阶段。监控结果如下：

| Step | 训练 K=0 Point Rel | 训练 K=3 Point Rel | 相对最佳点回退 | 验证 K=3 Point Rel |
|---:|---:|---:|---:|---:|
| 3000 | 3.3537% | 3.3448% | 0.00% | 10.4008% |
| 3250 | 3.3700% | 3.3608% | 0.48% | 10.7731% |
| 3500 | 3.3869% | 3.3751% | 0.91% | 11.1758% |
| 3750 | 3.4325% | 3.4134% | 2.05% | 11.5337% |
| 4000 | 3.5585% | 3.5368% | 5.74% | 12.1429% |

step 4000 的 K=3 仍比同一步 K=0 好 `0.61%`，说明 SSR 仍提供正向残差修正；
恢复分支的 K=3 也比首次分支同一步的 `6.1990%` 好 `42.95%`。因此降低学习率
成功抑制了原始分支的快速坍塌，但没有阻止 Base 在 detached 阶段持续漂移。

恢复的 1000 个优化步中，最大裁剪前梯度范数为 `19.5580`，没有梯度达到
100，没有跳过更新；SSR 最大绝对对数深度残差为 `0.02359`。监控于
2026-08-01 04:19:46（Asia/Shanghai）向经过命令行和输出目录双重核验的 Exp11
`torchrun` 发送 SIGTERM。两个 worker 正常退出，GPU 显存释放，其他用户进程
未受影响。包装脚本因此记录 `exit_code=1`；这表示“由监控主动停止”，不表示
OOM 或程序自身崩溃。

结果文件：

- [首次分支周期评测](results/original_evaluation_history.csv)
- [首次分支 Point Rel 主曲线](results/visualizations/original_point_rel_curve.png)
- [首次分支完整指标面板](results/visualizations/original_metrics_dashboard.png)
- [恢复分支周期评测](results/recovery_evaluation_history.csv)
- [恢复分支逐步训练记录](results/recovery_training_history.csv)
- [10 分钟监督记录](results/recovery_supervisor_checks.jsonl)
- [监督终态](results/recovery_supervisor_status.json)
- [监控结论摘要](results/recovery_monitoring_summary.json)

首次分支图由 `tools/moge3/plot_exp11_evaluation_history.py` 从周期评测 CSV
直接生成，使用原始评测点，不进行平滑。绿色虚线标记训练集 K=3 Point Rel
最低的 step 3000；淡红背景表示最低点之后的持续漂移区间；红色虚线标记
step 4478 的稳定性保护停止位置。PDF 矢量版本与 PNG 位于同一目录。

服务器保留的主要检查点：

- 首次分支 step 3000 最佳模型：
  `1a665e760516fde4e00082b9743a0fd8c7eed98d298cb76bb012b813227c98ce`
  （约 1.5 GB）；
- 恢复分支最佳模型：
  `c088e875e0086055f4be36fc587230ab827e52918666940416af8f10bcba141a`
  （约 1.5 GB）；
- 恢复分支 step 4000 完整恢复点：
  `5c0e64cacc17f61040fd1e03f0b8fbb36c72994e67417e3f4915657b0d24151f`
  （约 4.2 GB）。

本次结果表明：当前训练目标能够保持 SSR 的局部正向修正，并且低学习率可以解决
梯度爆炸；但 detached 阶段继续训练 Base 会使主要点图指标单调恶化。后续若要
验证联合阶段，应从 step 3000 最佳模型启动独立分支，调整阶段切换或 Base
更新策略，而不应从 step 4000 漂移点继续恢复。
