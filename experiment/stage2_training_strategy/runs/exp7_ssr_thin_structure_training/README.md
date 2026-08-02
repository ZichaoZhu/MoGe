# exp7_ssr_thin_structure_training

## 当前状态

实验已完成。首轮只采用 Hypersim；TartanAir 保留为后续备选。

本实验替代已经回撤的错误 exp7。旧实验联合训练了基础模型和 SSR，与本实验目标不符；
旧目录、服务器数据、检查点和产物均已删除，而且旧实验从未形成 Git 提交。

## 实验目标

采用冻结基础模型、只训练精修器的路线：

- 从 `Ruicheng/moge-2-vitl-normal` 初始化；
- 完全冻结 DINO、二维解码器和全部基础模型预测头；
- 仅训练 SSR；
- 增加包含栏杆、椅腿、窗格、楼梯扶手、吊灯线、栅格和细支撑结构的训练样本；
- 在固定基础模型输出上比较 K=0 与 K=3，判断 SSR 是否真正改善细结构。

不预测法线，也不使用法线监督。

## 强制训练约束

正式训练开始前必须用自动测试验证：

1. 除 `model.ssr` 外，所有模型参数均为 `requires_grad=False`；
2. 优化器参数集合与 SSR 参数集合完全相等；
3. 基础点图和视觉特征在训练前缓存，训练循环不重复执行基础模型；
4. SSR 输入与缓存特征全部断开梯度；
5. 一个优化步骤后，基础模型权重哈希保持不变且基础模型梯度均为空；
6. 只允许 SSR 权重、优化器状态和实验统计进入训练检查点；
7. 模型选择比较同一检查点的 K=0 与 K=3，不能把基础模型微调收益记作 SSR 收益。

## 训练前数据审核

### Hypersim

[24 张候选联系表](data_review/curated_candidates/hypersim_curated_24_contact_sheet.jpg)
中的每个条目左侧为完整 RGB 和红色候选框，右侧为放大候选区域。候选具备 RGB、
光心欧氏深度和相机参数，数据语义已在 exp1–exp6 中验证。

当前优先建议使用 Hypersim。它与 exp2 的数据和训练代码一致，可把实验变量限制为
“细结构样本数量和多样性”，避免同时引入数据集域变化。

### TartanAir V2

[24 张候选联系表](data_review/curated_candidates/tartanair_curated_24_contact_sheet.jpg)
来自已下载的 ConstructionSite、CarWelding 和 AmusementPark。ConstructionSite
包含钢筋、脚手架和起重机，CarWelding 包含安全围栏、轨道和厂房桁架，
AmusementPark 包含过山车轨道与细支撑。

这些画面有较多细结构。[RGB/Depth 对照表](data_review/curated_candidates/tartanair_curated_24_rgb_depth_contact_sheet.jpg)
显示候选细结构在深度图中具有清晰且像素对齐的几何边界。

数据审计已经确认：

- 八个已下载环境共有 105,610 对 RGB/Depth，按轨迹和帧号完全配对；
- V2 原始图像为 640×640、90° FoV，内参为
  `fx=fy=320, cx=cy=320`；
- 深度 PNG 是无损保存的 little-endian float32 光轴 Z 深度，单位为米；
- 送入现有 MoGe 点图加载器前需要转换为光心欧氏距离：
  `distance = z * sqrt(1 + x_normalized^2 + y_normalized^2)`。

机器可读证据保存在
[tartanair_modality_audit.json](data_review/curated_candidates/tartanair_modality_audit.json)。
审计对照官方 `castacks/tartanairpy` 提交
`158a6844d782942110967325ca3082f50ab2bfc7`。

TartanAir 在数据语义上可以接入，但它会同时引入明显的视觉域变化。因此仍不建议在
第一轮实验中与 Hypersim 混合。

原始的八个环境均匀抽帧表保存在
[`data_review/tartanair_uniform_samples`](data_review/tartanair_uniform_samples)，用于说明
为何优先保留上述三个环境。

## 用户确认结果

用户于 2026-07-28 确认保留全部候选图。为控制变量，首轮只使用 24 张 Hypersim
候选图准备训练数据；TartanAir 暂不参与训练。

## 首轮实验设计

- 训练：24 张已确认候选，来自 23 个场景、15 个互不相同的 `ai_XXX` 场景组；
- 验证：`ai_043` 与 `ai_047` 两个未参与训练的场景组，共 16 张；
- 测试：`ai_046` 与 `ai_050` 两个未参与训练和选模的场景组，共 16 张；
- 分辨率：384×512，明显高于 exp2 的 192×256；
- 训练：K=3、4000 步、batch size 2、AdamW、SSR 学习率 `2e-5`；
- 检查点仍按验证集点图 Rel 选择，同时保留最终检查点以诊断训练集拟合。

完整划分见 [data_spec.json](data_spec.json)，配置见 [config.json](config.json)。

### 稳定性试跑

正式配置前曾以 exp2 的 `2e-4` 学习率在 384×512 上试跑。step 380 出现残差与梯度
尖峰，下一步点图对齐因显存膨胀而 OOM；step 200 时训练集虽已开始改善，验证集已经
退化。该失败运行完整保存在服务器 `artifacts/failed_lr2e4_step380/`，不作为实验
结果。正式运行从零初始化重启，并采用 exp4/exp5 在相同分辨率验证稳定的 `2e-5`。

## SSR-only 验证

正式报告确认训练链路满足冻结约束：

- 模型共 607 组参数张量，其中基础模型 481 组全部冻结；
- SSR 的 126 组参数张量全部可训练，优化器也严格只包含这 126 组；
- 缓存的基础点图与视觉特征均已断开梯度；
- 缓存前后基础模型状态 SHA-256 均为
  `502c221fa55c99cdeec938a3d9dd3eaae6740ca5f8a3bea6130ee6cda7c34271`；
- 检查点只保存 SSR、优化器和实验状态，不保存或更新基础模型；
- 零初始化 K=3 与 K=0 的最大绝对误差为 0；
- 不预测法线，也不使用法线监督。

因此下面的 K=0→K=3 变化只能来自 SSR，不能归因于基础模型微调。

## 实验结果

### 训练集拟合

按训练集点图 Rel 保存的最优检查点为 step 3800。24 张训练图的点图 Rel 和深度 Rel
均得到改善，18/24 张的边界 F1 得到改善。

| 指标 | K=0 | K=3 | 变化 |
|---|---:|---:|---:|
| 点图 Rel | 7.350% | **4.284%** | 相对下降 41.71% |
| 深度 Rel | 6.303% | **3.185%** | 相对下降 49.47% |
| 深度 δ1.01 | 25.05% | **47.29%** | 增加 22.25 个百分点 |
| 深度 δ1.25 | 93.84% | **97.42%** | 增加 3.58 个百分点 |
| 边界 F1 | 0.7962 | **0.8187** | 增加 0.0225 |

精修次数消融显示 K=3 的点图和深度误差最低；K=1 的边界 F1 最高，K=5 则出现
过度精修：

| K | 点图 Rel | 深度 Rel | 边界 F1 |
|---:|---:|---:|---:|
| 0 | 7.350% | 6.303% | 0.7962 |
| 1 | 4.456% | 3.482% | **0.8322** |
| 3 | **4.284%** | **3.185%** | 0.8187 |
| 5 | 4.731% | 3.486% | 0.7998 |

### 独立场景

同一个 step 3800 检查点没有泛化到独立场景：

| 划分 | K=0 点图 Rel | K=3 点图 Rel | 点图变化 | 深度变化 | 边界 F1 变化 |
|---|---:|---:|---:|---:|---:|
| 验证 | 8.904% | 9.887% | 恶化 11.04% | 恶化 6.91% | −0.0555 |
| 测试 | 7.314% | 7.805% | 恶化 6.72% | 恶化 6.20% | −0.0292 |

验证集只有 3/16 张点图与深度 Rel 改善，测试集只有 1/16 张改善。按验证集点图 Rel
选择出的最佳检查点仍是 step 0，即严格恒等映射。因此本实验通过了“优质细结构小数据
拟合”目标，但不能作为 SSR 泛化能力的证据。

### 细结构 GIF

训练前从 24 张已确认候选中固定 8 个代表区域，选择过程只读取 RGB 与真值。最终动画
使用 step 3800、K=3，并以真值深度最近 50% 的点构造显示掩码；该掩码只用于可视化。
动画均为单向 yaw −90°→+90°，46 帧、110 ms/帧。

8/8 个局部区域的点图和深度 Rel 均改善：

| 局部均值 | K=0 | K=3 | 变化 |
|---|---:|---:|---:|
| 点图 Rel | 10.36% | **6.27%** | 相对下降 39.44% |
| 深度 Rel | 9.69% | **5.64%** | 相对下降 41.77% |

- [8 张 GIF 索引](artifacts/gif_gallery/INDEX.md)
- [静态总览](artifacts/gif_gallery/gallery_contact_sheet.jpg)
- [机器可读 GIF 报告](artifacts/gif_gallery/gallery_report.json)
- [逐区域指标](artifacts/gif_gallery/gallery_metrics.csv)

## 资源与稳定性

- 稳定训练及周期评测：4863.39 秒；
- 含最终全量评测：4927.10 秒，约 82.12 分钟；
- 平均每步（含周期评测）：1.216 秒；
- PyTorch 峰值已分配显存：9.65 GiB；
- 全程无 NaN、Inf 或第二次 OOM；
- 服务器专项测试 8 项、GIF 工具测试 4 项均通过。

## 产物

- [验证集选模报告](artifacts/training/report.json)：最佳为 step 0；
- [训练最优完整评测](artifacts/train_best_evaluation/report.json)：step 3800；
- [训练曲线](artifacts/training/training_curves.png)；
- [周期评测记录](artifacts/training/evaluation_history.csv)；
- [逐帧完整指标](artifacts/train_best_evaluation/per_frame_metrics.csv)；
- [定性对比](artifacts/training/qualitative_comparison.png)；
- [稳定正式运行日志](artifacts/exp7_run.log)；
- [失败高学习率日志](artifacts/failed_lr2e4_step380/exp7_run.log)；
- [检查点记录](CHECKPOINT.md)；
- [产物校验和](SHA256SUMS)。

## 结论

第一性原理上，这次实验回答了两个不同问题：

1. 固定基础模型后，SSR 能否从细结构真值中学到有效残差？能。训练集与 8 个固定
   细结构区域均有大幅改善，且基础模型哈希严格不变。
2. 仅用 24 张精挑图片，SSR 能否把这种修正推广到新场景？不能。验证和测试均退化，
   验证选模甚至只能选择零初始化。

所以当前检查点适合作为“SSR 可学习性”和后续训练策略的调试基线，不应接入
InfiniDepth 作为可泛化模块。下一步若继续，应增加细结构场景数量与多样性，并继续
使用独立验证集约束过拟合；不能只依赖训练集 GIF。
