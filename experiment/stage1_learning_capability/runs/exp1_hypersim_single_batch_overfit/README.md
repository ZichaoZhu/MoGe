# exp1_hypersim_single_batch_overfit

## 实验目标

在扩大数据集之前，验证复现的 MoGe-3 自引导稀疏精修器能否从真实 RGB 图像中学习
几何残差。本实验用于检查完整流水线和优化过程是否正确，不用于评估泛化能力。

## 相较基线的变更

冻结预训练 MoGe-2 ViT-L 编码器和所有二维预测头，只优化包含 5070 万参数的稀疏三维
SSR。不进行法线预测，也不使用法线监督。

## 实验配置

- 日期：2026-07-24
- 实现代码提交：`d71fc96`
- 服务器：ZJU3DV-S115，单张 RTX 4090
- 数据来源：Hypersim `ai_001_001/cam_00/frame.0000`
- 输入：色调映射 RGB 和 `depth_meters`，不使用法线数据
- 图像尺寸：192×256
- 批量大小：固定 1 个样本
- 基础检查点：`Ruicheng/moge-2-vitl-normal`
- 精修次数：K=3，多轮共享 SSR 权重
- 优化设置：200 步，AdamW，SSR 学习率 2e-4，权重衰减 1e-2
- 损失函数：全局仿射不变点图损失、α={4,16,64} 的径向局部损失，以及每轮精修
  预测的边缘角度损失

Hypersim 保存的是像素对应三维点到相机光心的欧氏距离。真值点图通过场景对应的
`M_cam_from_uv`、归一化相机射线和
`[x,y,z]_Hypersim -> [x,-y,-z]_MoGe` 坐标变换重建。与 Hypersim 世界坐标真值交叉
验证后，点误差均值为 0.67 mm，p99 为 1.58 mm。

## 实验结果

| 指标 | 冻结的 K=0 基线 | 训练后的 K=3 |
|---|---:|---:|
| 对齐点图 Rel | 0.020082 | **0.003811** |
| 对齐深度 Rel | 0.018444 | **0.002783** |
| 深度 δ1.01 | 0.511800 | **0.965698** |
| 深度 δ1.25 | 0.999003 | **0.999858** |

- 总损失：0.048328 → 0.014564
- 运行时间：84.87 秒，平均每步 0.424 秒
- GPU 峰值分配显存：4.63 GB
- SSR 初始残差：严格为零
- 数值状态：未出现 NaN 或 Inf
- 检查点重载：严格加载，无缺失或额外键；在新进程中复现了最终指标

## 实验结论

真实数据几何转换、预训练基础模型、稀疏体素化、迭代 SSR、论文几何损失、优化器、
检查点和评测路径均可以端到端正常运行。指标大幅提升说明 SSR 具备拟合局部几何修正
的能力。由于训练过程中只重复使用一帧图像，本实验不能衡量模型的跨场景泛化能力。

## 三维旋转动画

补充的 [三维点图旋转 GIF](artifacts/single_batch_point_map_orbit.gif) 在同一视角下并列
展示 GT、冻结基础模型 K=0 和训练后 K=3。预测点图先使用实验相同的光轴仿射对齐，
再以 RGB 作为点颜色，从 -35° 到 +35° 往返旋转，共 28 帧。该动画用于从离轴视角
检查表面弯曲、层间错位和飞点；它比正视深度图更容易展示 SSR 单帧过拟合前后的三维
几何差异，但不能提供泛化证据。补充后服务器完整测试为 37 passed，产物与原检查点
的 SHA-256 校验全部通过。

## 曲线数据说明

原始运行只保存了首尾指标，没有保存逐步损失。为了忠实重绘带坐标轴的曲线，使用完全
相同的配置重新运行了 200 步，并将逐步损失保存在 `artifacts/loss_curve.csv`。由于
SpConv 存在轻微非确定性，重跑的最终损失为 0.014684、对齐深度 Rel 为 0.003180；
上表仍记录原始运行及其对应检查点的结果。重跑完整报告单独保存在
`metrics/loss_curve_rerun_report.json`，没有覆盖原始报告。

## 文件说明

- `config.json`：结构化实验配置
- `run.sh`：服务器完整复现入口
- `data/manifest.json`：样本来源、几何约定和校验和
- `metrics/report.json`：原始最终报告
- `metrics/loss_curve_rerun_report.json`：用于补充逐步曲线数据的同配置重跑报告
- `artifacts/comparison.png`：RGB、真值、精修前和精修后深度对比
- `artifacts/loss_curve.png` 和 `artifacts/loss_curve.pdf`：带坐标轴、图例和 10 步
  滑动平均的优化曲线
- `artifacts/loss_curve.csv`：绘图使用的逐步损失及滑动平均原始数据
- `artifacts/single_batch_point_map_orbit.gif`：GT/K=0/K=3 对齐点图的离轴旋转动画
- `artifacts/single_batch_point_map_orbit_report.json`：动画对应的检查点、指标与渲染参数
- `artifacts/*.npy`：用于分析的对齐深度结果
- `artifacts/SHA256SUMS`：包括未提交检查点在内的产物校验和

609 MB 检查点和受许可证约束的 RGB/深度输入保存在服务器的本实验目录中，不提交到
Git。

## 补充：用户指定区域的细结构 GIF

在唯一训练帧上按 RGB/GT 锁定三个区域，并使用同一个 step 200、K=3 检查点重绘。
每张 GIF 的最左列显示完整输入图和红色裁剪框，第二列显示裁剪及 GT 深度掩码，随后
固定同一相机依次显示 GT、K=0 和 K=3：

| 区域 | K=0 点图 Rel | K=3 点图 Rel | 动画 |
|---|---:|---:|---|
| 完整圆形毛巾架与悬挂毛巾 | 1.666% | **0.299%** | [GIF](artifacts/rod_visualization/train_rod_orbit.gif) |
| 用户蓝框：右侧大窗 | 1.234% | **0.436%** | [GIF](artifacts/rod_visualization_window/train_rod_orbit.gif) |
| 用户蓝框：左下浴缸水龙头 | 5.886% | **1.762%** | [GIF](artifacts/rod_visualization_faucet/train_rod_orbit.gif) |

对应静态预览和机器可读报告分别保存在三个 `rod_visualization*` 目录。裁剪与显示掩码
只根据 RGB/GT 决定，不参与训练或检查点选择；Rel 也只是该显示掩码上的诊断值，不是
论文 Local 指标。exp1 只有这一张训练图，因此这些动画只说明单图拟合能力，不提供
验证或测试泛化证据。动画采用单向 −90°→+90°、每步 4°、每帧 110 ms，共 46 帧，
一轮 5.06 秒；复现入口为 [render_rod_gifs.sh](render_rod_gifs.sh)。
