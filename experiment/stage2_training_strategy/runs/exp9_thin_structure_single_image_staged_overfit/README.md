# Exp9：五张细结构单图的论文式两阶段极限过拟合

## 结论

实验已完成。当前实现能够在 3/5 张单图上把基础二维几何和 SSR 都训练到较高精度，
但未达到预定的 5/5 稳定过拟合目标：

- 3/5 张最终 K=3 在全图和细结构区域均优于同检查点 K=0；
- 2/5 张达到全图深度 Rel <1% 且 δ1.01 >95%；
- 3/5 张达到细结构点图 Rel <2%；
- 最成功的样本 05 最终 K=3 全图点图 Rel 为 0.279%，细结构点图 Rel 为
  0.337%，全图深度 Rel 为 0.255%；
- 样本 01 和 07 属于同一个 `ai_019_004` 场景，训练中基础二维分支发生退化，
  最终 K=0 与 K=3 均失效。它们作为失败样本完整保留。

因此，本实验验证了论文式“阶段一梯度隔离、阶段二联合训练”的代码路径确实能够学习，
也证明 SSR 能在成功样本中产生独立增益；但当前固定超参数对不同单图并不稳健，不能据此
宣称训练流程已经达到完整论文的稳定性。

## 实验设计

五张 Hypersim 图片是五个完全独立的任务。每个任务都从同一份 MoGe-2 ViT-L 权重和
零初始化 SSR 开始：

1. 阶段一中，K=0 几何损失训练 DINO、neck 和 points head；K=1–3 损失只训练
   SSR，进入 SSR 的点图和视觉特征均 `detach`；
2. 阶段二解除 `detach`，K=1–3 损失通过 SSR 反传至二维分支和 DINO，同时保留
   K=0 的独立基础几何监督；
3. 每 100 步评测 K∈{0,1,3,5}，按全图 K=3 点图 Rel 与锁定细结构 K=3 点图
   Rel 之和选择最佳联合检查点；
4. 每阶段至少训练 2000 步，达到平台期、成功阈值或 5000 步上限后停止。

统一配置为 384×512、`num_tokens=2500`、batch size 1、训练 K=3、不使用增强。
AdamW 的 SSR、二维头和 DINO 峰值学习率分别为 `2e-5`、`1e-5` 和 `5e-7`；
DINO 前 1000 步冻结，1001–2000 步线性预热。损失由等权的全局、径向局部
`{4,16,64}` 和边缘损失组成。不预测法线，mask、normal 和 metric-scale 等无关
辅助头均冻结。

训练前的零初始化恒等检查要求 K=0/1/3/5 逐元素完全一致；五个正式任务均通过，
冒烟测试的最大绝对误差为 0。

## 主要结果

表中 Rel 与 δ 均按百分数表示。“阶段一 K0/K3”用于拆分二维模块和 SSR 的独立贡献；
“最终”采用联合阶段选择分数最低的检查点，而非平台期的最后一步。

| 样本 | 初始 K0 点图 Rel | 阶段一 K0 / K3 | 最终 K0 / K3 | 最终 K3 深度 Rel | δ1.01 | 细结构点图 Rel | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 01 `ai_019_004` f0000 | 9.129% | 108.207% / 108.207% | 108.207% / 108.207% | 100.000% | 0.00% | 104.232% | 训练退化 |
| 04 `ai_054_008` f0000 | 8.209% | 1.736% / 1.259% | 1.710% / 1.149% | 1.044% | 75.25% | 0.738% | SSR 有效，深度阈值略未达到 |
| 05 `ai_053_018` f0000 | 14.554% | 0.562% / 0.257% | 0.637% / 0.279% | 0.255% | 99.35% | 0.337% | 达到全部目标 |
| 06 `ai_002_003` f0000 | 3.742% | 1.262% / 0.468% | 1.158% / 0.492% | 0.439% | 96.55% | 0.553% | 达到全部目标 |
| 07 `ai_019_004` f0099 | 8.769% | 108.207% / 108.207% | 108.207% / 108.207% | 100.000% | 0.00% | 107.660% | 训练退化 |

完整的三个时间点、四个 K、三个评测范围和五项指标共 180 行，见
[metrics_summary.csv](metrics_summary.csv) 和
[metrics_summary.json](metrics_summary.json)。

## 各阶段贡献

- 初始 K=0→阶段一 K=0：样本 04、05、06 的全图点图 Rel 分别改善约
  78.9%、96.1% 和 66.3%，说明二维几何分支确实参与了训练，并非只训练 SSR。
- 阶段一同检查点 K=0→K=3：上述三张分别改善约 27.5%、54.2% 和 62.9%，
  证明隔离阶段中的 SSR 学到了独立残差修正。
- 阶段一 K=3→最终 K=3：样本 04 继续改善；样本 05 的全图点图略回退、细结构继续
  改善；样本 06 略回退。联合训练不是所有样本的单调增益来源。
- 初始 K=0→最终 K=3：成功的三张全图点图 Rel 总改善约 86.0%、98.1% 和
  86.9%。

样本 01 和 07 在二维分支训练时输出塌缩到接近零深度，随后对齐指标饱和、梯度变为
零，SSR 无法恢复。两者没有 NaN 或 OOM，并按最短阶段长度运行至平台期。样本 01
首次运行还暴露了评测器过滤全部非正预测后崩溃的问题；修复后从第 700 步恢复，能够
正确把退化报告为 100% 深度 Rel。该评测修复没有改变训练退化本身。

## 可视化

每张图片均提供完整裁剪和训练前锁定细结构掩码两种 GIF。动画固定为单向
yaw −90°→+90°、46 帧、110 ms/帧，并依次比较 GT 与最终 K=0/1/3/5。最左侧始终
显示原图和锁定裁剪框。

| 样本 | 完整裁剪 | 锁定细结构 | 三阶段误差 |
|---|---|---|---|
| 01 | [GIF](01_ai_019_004_cam_00_frame.0000/artifacts/final_crop_orbit.gif) | [GIF](01_ai_019_004_cam_00_frame.0000/artifacts/final_structure_orbit.gif) | [点图](01_ai_019_004_cam_00_frame.0000/artifacts/stage_point_error_comparison.png) |
| 04 | [GIF](04_ai_054_008_cam_00_frame.0000/artifacts/final_crop_orbit.gif) | [GIF](04_ai_054_008_cam_00_frame.0000/artifacts/final_structure_orbit.gif) | [点图](04_ai_054_008_cam_00_frame.0000/artifacts/stage_point_error_comparison.png) |
| 05 | [GIF](05_ai_053_018_cam_00_frame.0000/artifacts/final_crop_orbit.gif) | [GIF](05_ai_053_018_cam_00_frame.0000/artifacts/final_structure_orbit.gif) | [点图](05_ai_053_018_cam_00_frame.0000/artifacts/stage_point_error_comparison.png) |
| 06 | [GIF](06_ai_002_003_cam_00_frame.0000/artifacts/final_crop_orbit.gif) | [GIF](06_ai_002_003_cam_00_frame.0000/artifacts/final_structure_orbit.gif) | [点图](06_ai_002_003_cam_00_frame.0000/artifacts/stage_point_error_comparison.png) |
| 07 | [GIF](07_ai_019_004_cam_00_frame.0099/artifacts/final_crop_orbit.gif) | [GIF](07_ai_019_004_cam_00_frame.0099/artifacts/final_structure_orbit.gif) | [点图](07_ai_019_004_cam_00_frame.0099/artifacts/stage_point_error_comparison.png) |

另有三阶段深度图、训练曲线、PDF 和 GIF 静态预览保存在各样本的 `artifacts/` 中。
10 个 GIF 均已逐帧解码验证，共 46 帧且帧时长均为 110 ms。

## 检查点和复现接口

30 个正式检查点共 83.10 GiB，只保存在服务器：

```text
/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/
experiment/stage2_training_strategy/runs/exp9_thin_structure_single_image_staged_overfit/<样本>/checkpoints/
```

每个样本包含 `initial.pt`、`stage1.pt`、`best_joint.pt`、`final.pt`、`last.pt`
和 `resume.pt`。相对路径、字节数和 SHA-256 见
[checkpoint_manifest.json](checkpoint_manifest.json)。本地 Git 不保存这些大型文件。

服务器启动入口为 [run_all.sh](run_all.sh)，可自动跳过完整任务并从 `resume.pt`
恢复不完整任务；[render_all.sh](render_all.sh) 负责最终渲染和统一汇总。调度器在
四卡中只选择满足显存余量和低利用率条件的 GPU，每个单图任务独占一张卡，不使用
DDP。

## 验证

- 样本 04 的 384×512 四步冒烟测试通过，观测到的额外显存峰值约 13.46 GiB；
- 阶段一 refined loss 不向二维模块和 DINO 反传，阶段二可完整反传；
- DINO 冻结/预热、平台期、断点恢复、三阶段重载、全图对齐裁剪指标和 GIF 解码均有
  自动测试；
- 服务器最终全仓测试为 `67 passed, 16 warnings in 13.77s`；
- 调度复核结果为 `complete` 且无未归档失败进程。

机器可读验收记录见 [verification_report.json](verification_report.json)。

本实验只证明小数据上的学习能力与实现路径，不衡量跨图泛化，也不等同于论文使用完整
数据和算力的数值复现。

## 训练前点云归档与交互式查看器

Exp9 现已补充全部五张样本在任何训练开始前的原始预测点云。导出器逐一加载每张任务
自己的 `initial.pt`，而不是使用训练后的 Base 输出倒推。五张图的 K=0/1/3/5
最大逐元素误差均为 0，确认零初始化 SSR 在训练前是严格恒等映射。

样本 05、04、06 还导出了 `final.pt` 的 K=0/1/3/5，因而可以明确拆分：

- 训练前 K=0：未经 Exp9 微调的 MoGe-2 Base；
- 训练后 K=0：经过联合训练微调的 Base/2D Head；
- 训练后 K=1/3/5：同一训练后 Base 经相应次数 SSR 精修。

PLY 始终保存网络原始 XYZ 和逐像素 RGB，共 196,608 点。清单单独保存全图求得的
尺度和光轴平移，网页中的“GT 对齐”仅在渲染时应用，避免把对齐后的坐标误称为原始
预测。五张初始点云和三张最终点云共 17 份唯一 PLY，总计 50,140,565 字节。
机器可读结果见 [pointcloud_export_report.json](pointcloud_export_report.json) 和
[点云校验清单](viewer/public/data/pointcloud_sha256.json)。

[双窗口查看器](viewer/README.md) 已扩展为统一的多实验入口。Exp9 仍提供样本
05、04、06 的训练前/训练后对比；新增 Exp12 的 train/val/test 三张代表图，
用于查看 100 张训练图联合微调前后的变化。页面可切换 K=0/1/3/5、完整场景/
细结构裁剪、原始/GT 对齐坐标和彩色点云/SSR 体素壳。
两个窗口支持独立旋转、平移、缩放与视野复位，也可同步相机进行固定视角比较。
切换训练阶段或 K 时会保留当前相机；每个窗口可把左键显式切换为旋转或平移，
右键不再参与相机操作。

供组内分享的公开 Vercel 地址为
<https://moge3-exp9-viewer.vercel.app>；原有 Sites 部署继续保持仅本人可访问。

实现采用 Three.js、React Three Fiber 和 OrbitControls。未采用 Potree，因为本实验
每幅图只有约 19.7 万点，不需要面向十亿级点云的八叉树转换；未采用需要常驻
Python/WebSocket 服务的 Viser，使查看器可以作为静态实验资产独立部署。

三张真实数据验收截图：

| 样本 | 双窗口截图 |
|---|---|
| 05 | [查看](artifacts/viewer_acceptance/sample_5_dual_view.png) |
| 04 | [查看](artifacts/viewer_acceptance/sample_4_dual_view.png) |
| 06 | [查看](artifacts/viewer_acceptance/sample_6_dual_view.png) |

初版查看器完成时，服务器全仓回归测试为 `73 passed, 16 warnings in 11.78s`。
多实验版本另外通过 8 项 Python 导出测试、12 项前端单元测试、3 项真实 WebGL
浏览器测试、Next.js/Vinext 双生产构建及依赖漏洞审计。完整记录见
[viewer_verification_report.json](viewer_verification_report.json)。
