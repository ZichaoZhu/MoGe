# Exp12：100 张 Hypersim 从最佳点立即联合训练

## 目标

Exp11 原始分支在 step 3000 达到训练集 K=3 Point Rel `3.3448%`，随后
Base、SSR 和总训练损失同步漂移；固定到 step 5000 才解除 detach 不适合当前
100 张小数据配置。

Exp12 固定从 Exp11 step 3000 最佳检查点重新开始，使用新优化器，并从
step 3001 起直接进入联合训练，使 K=1/2/3 精修损失能够同时更新 SSR、二维
几何头和 DINO。实验只验证“小数据上提前联合能否继续改善训练拟合并保持稳定”，
不宣称论文规模的泛化结果。

## 配置

- 数据：复用 Exp11 的 100/16/16 张训练、验证、测试图片。
- 分辨率：`384×512`；训练 `K=3`。
- 两卡 Gloo；每卡 micro-batch 1；全局 batch 16。
- 联合训练范围：step 3001–5000，最多 2000 个新优化步。
- 峰值学习率：
  - SSR：`5e-7`
  - Neck/2D Head：`2e-7`
  - DINO：`1e-8`
- 从 step 3000 到5000进行余弦衰减，最终学习率为峰值的10%。
- 每100步评测完整100张训练图和16张验证图；以训练集 K=3 Point Rel
  保存最佳检查点。
- 如果连续5次评测没有至少0.2%的相对改善，监督器安全停止该分支。
- 梯度范数超过100时跳过更新；最多允许5次、最多连续2次。
- SSR 最大绝对对数深度残差超过5，或 K=3 同时超过10%且为 K=0 的3倍时
  立即终止。

## 运行

`run_single.sh` 支持 `smoke` 和 `formal` 两种模式；`run_pipeline.sh` 先执行
冒烟测试，再从同一个 Exp11 最佳检查点启动正式分支。`supervise_training.py`
只选择当前用户允许范围内且满足显存/利用率门槛的 GPU，不终止其他用户任务，
并每60秒记录进程、GPU、最新训练步、评测和早停状态。

服务器大型检查点仅保存在 `artifacts/`，由 Git 忽略。正式结果完成或触发早停
后再补充。

## 运行结果

正式训练于 2026-08-01 14:27 启动，使用物理 GPU 1、0，从 Exp11 step 3000
最佳模型恢复并重置优化器。监督器在 17:02 检测到平台期，于完整保存 step 3600
评测后主动停止训练。训练没有发生 OOM、非有限数值、SSR 残差越界或梯度更新
跳过；最后一次梯度范数为 `1.3824`，最大绝对对数深度残差为 `0.01672`。

| 时间点 | 训练 K=0 Point Rel | 训练 K=3 Point Rel | 验证 K=0 Point Rel | 验证 K=3 Point Rel |
|---|---:|---:|---:|---:|
| 恢复起点 step 3000 | 3.3537% | 3.3448% | 10.4076% | 10.4008% |
| 最佳训练点 step 3300 | 3.3346% | **3.3203%** | 10.5805% | 10.5793% |
| 停止点 step 3600 | 3.3440% | 3.3251% | 10.7095% | 10.7044% |

相对恢复起点，step 3300 的训练 K=3 Point Rel 改善 `0.73%`；其中 Base/2D
Head 的 K=0 Point Rel 改善 `0.57%`，同一检查点 SSR 再带来 `0.43%` 的相对
Point Rel 降低。这说明立即联合训练能够稳定地继续优化 Base 与 SSR，且 SSR
保持正向修正。

但验证 K=3 Point Rel 在 step 3300 已比起点恶化 `1.72%`，step 3600 恶化
`2.92%`。验证集只在 step 3100 短暂达到 `10.3928%`，随后持续上升。因此本次
实验没有解决 100 张小数据下的泛化问题；它证明的是联合反向传播可以稳定运行并
小幅提高训练拟合，而不是取得了更好的验证效果。

自动停止依据为训练 K=3 Point Rel：相对于 step 3100 的显著改善参考值，step
3200–3600 连续五次评测均未达到 `0.2%` 的最小相对改善。训练集的绝对最低值
虽出现在 step 3300，但改善幅度不足以重置“显著改善”窗口，符合预设早停规则。
包装脚本因收到监督器发出的 SIGINT 记录 `exit_code=1`；日志中的
`KeyboardInterrupt` 是主动早停的结果，不是程序自身崩溃。

服务器保留：

- step 3300 最佳模型，约 1.5 GB，SHA-256：
  `80b62541db30e4a2c2cef34a8608d01eb53f9b81ae3ad44cd5309381e5d35f1c`；
- step 3600 完整恢复点，约 4.47 GB，SHA-256：
  `2caf3faed76dfa443240c4a11cdd3474bd1c4b4f76e79d3ab9deae60e68e1781`。

小型结果已归档：

- [周期评测原始数据](results/evaluation_history.csv)
- [逐步训练原始数据](results/training_history.csv)
- [结果摘要](results/result_summary.json)
- [监督器终态](results/supervisor_status.json)
- [平台期决定](results/plateau_decision.json)
- [正式训练日志](results/formal_training.log.txt)
- [原始指标可视化](results/visualizations/evaluation_dashboard.png)

图表由 `tools/moge3/plot_exp12_evaluation_history.py` 从周期评测 CSV 直接
生成，不做平滑；绿色虚线为训练集最佳 step 3300，淡红区域表示此后没有产生
新低点，红色虚线为平台期停止 step 3600。

## 最佳检查点离线 K 扫描

训练结束后，使用同一个离线评测器分别加载联合前 step 3000 和最佳 step 3300，
在完整 100/16/16 张 train/val/test 上重新计算 K=0/1/3/5。下表为 Point Rel；
这套一致的离线评测结果与训练期间周期评测存在轻微实现差异，因此阶段比较只使用
下表内部数值。

| 划分 | 阶段 | K=0 | K=1 | K=3 | K=5 |
|---|---|---:|---:|---:|---:|
| Train | 联合前 step 3000 | 3.3541% | 3.3483% | 3.3450% | 3.3501% |
| Train | 联合后 step 3300 | 3.3338% | 3.3231% | **3.3191%** | 3.3325% |
| Val | 联合前 step 3000 | 10.3401% | 10.3368% | 10.3317% | 10.3291% |
| Val | 联合后 step 3300 | 10.5417% | **10.5401%** | 10.5401% | 10.5426% |
| Test | 联合前 step 3000 | **5.7980%** | 5.7985% | 5.8008% | 5.8063% |
| Test | 联合后 step 3300 | 5.8476% | 5.8452% | **5.8446%** | 5.8495% |

按 K=3 比较联合前后，训练集改善 `0.77%`，验证集恶化 `2.02%`，测试集恶化
`0.76%`。在最终模型内部比较 K=3 与 K=0，Point Rel 相对降低分别为
train `0.44%`、val `0.015%`、test `0.050%`；逐帧改善比例为
`85% / 56.25% / 37.5%`。K=5 在训练和测试上已经回退，因此当前默认仍采用
K=3。

机器可读结果：

- [联合前完整评测](results/initial_step3000_full_evaluation/report.json)
- [联合后完整评测](results/best_step3300_full_evaluation/report.json)
- [离线对比摘要](results/posthoc_evaluation_summary.json)

## 多实验交互式查看器

查看器已从 Exp9 专用页面扩展为多实验入口，默认打开 Exp12，同时保留原有 Exp9。
Exp12 开放三张在查看预测前锁定的代表图片：

| 划分 | 样本 | 锁定裁剪 |
|---|---|---|
| Train | `ai_054_008_cam_00_frame.0000` | `[160,160,352,352]` |
| Val | `ai_047_001_cam_00_frame.0042` | `[160,96,352,288]` |
| Test | `ai_046_001_cam_00_frame.0000` | `[224,128,416,320]` |

每张均保存联合前/后 K=0/1/3/5 的原始 XYZ 与逐像素 RGB，共 24 份 PLY、
70,786,680 字节。网页默认左侧为“联合前 K=0”，右侧为“联合后 K=3”，可切换
实验、样本、阶段、K、完整场景/裁剪、原始/GT 对齐坐标，以及彩色点云/SSR
体素壳。切换阶段或 K 时相机保持不变。

所选 train/val/test 裁剪的 K=3 Point Rel 联合前→后分别为
`5.9655%→6.0328%`、`8.3368%→8.4327%`、`9.1858%→9.1407%`。它们没有按预测
结果挑选，因此如实包含两个局部退化案例；网页用途是展示实际实验效果，不是只展示
成功样本。

- [点云导出报告](results/viewer_export/pointcloud_export_report.json)
- [三张样本逐阶段指标](results/viewer_export/selected_sample_metrics.json)
- [锁定选择记录](results/viewer_selection/fine_sample_selection.json)
- [训练样本验收截图](results/viewer_acceptance/exp12_sample_1_dual_view.png)
- [验证样本验收截图](results/viewer_acceptance/exp12_sample_2_dual_view.png)
- [测试样本验收截图](results/viewer_acceptance/exp12_sample_3_dual_view.png)

公开地址保持为 <https://moge3-exp9-viewer.vercel.app>，页面内新增 Exp9/Exp12
切换，不另建重复站点。
