# MoGe-3 多实验三窗口点云查看器

本目录提供 Exp9、Exp12、Exp20 与 Exp29 点云的统一交互式浏览器。页面依次选择
实验、数据划分和图片。窗口 A 显示真实点云，窗口 B/C 可独立选择训练前/后与
K。默认入口为 `Exp29 → Train → 图片 1`，B 显示最佳检查点 K=0，C 显示
同一检查点 K=3。

- Exp9：开放样本 05、04、06，展示三项彼此独立的单图极限过拟合；
- Exp12：开放一张训练、一张验证和一张测试图片，展示 100 张训练图联合微调前后；
- Exp20：每个 train/val/test 划分开放五张，展示 Exp15 step 800 权重使用
  单图即时 BatchNorm 统计时的 K=0/1/3/5，并与 Exp15 训练开始前的
  MoGe-2 初始化比较；
- Exp29：每个 train/val/test 划分开放五张，按训练前 GT 中的长细杆显著性
  人工锁定，同时保留三张改善和两张退化案例；
- Exp12 的训练样本与 Exp9 样本 04 是同一帧，可直接比较单图过拟合与百图训练。

实验目录位于 `public/data/experiments.json`。Exp9 清单保持
`public/data/manifest.json`，Exp12、Exp20 与 Exp29 的清单和资产分别位于
`public/data/exp12/`、`public/data/exp20/`、`public/data/exp29/`。

## 数据语义

- PLY 保存模型输出的原始 MoGe 相机坐标 XYZ，没有把 GT 对齐结果写回文件；
- Exp29 另存 Hypersim 深度与相机内参反投影得到的真实相机坐标点云；
  两张验证图共 9 个无效 GT 深度以零哨兵保留像素顺序，前端明确跳过，不插值；
- “GT 对齐”在浏览器中应用
  \(P_{\mathrm{aligned}}=sP_{\mathrm{raw}}+(0,0,t)\)；
- Three.js 显示坐标使用 `[x,-y,-z]`；
- 训练前 SSR 为零初始化，K=0/1/3/5 逐元素相同，因此只保存一份
  `initial_k0.ply`，其他 K 在清单中使用别名；此规则适用于 Exp9 与 Exp20；
- Exp12 的“联合前”是已经训练到 step 3000 的 Exp11 模型，其 SSR 不为零，
  因此联合前/后均实际保存 K=0/1/3/5，不使用别名；
- Exp20 不是一份新权重。其“训练前”严格重建 Exp15 的起点：官方
  `Ruicheng/moge-2-vitl-normal`、seed 151 与零初始化 SSR；“训练后”恢复
  Exp15 step 800，并在不读取或更新 running buffers 的情况下让 SSR 使用当前
  单图稀疏特征统计；
- Exp29 只展示最终 step 2800 检查点，不重复存储续训起点。Base、2D Head
  和 SSR 均来自同一检查点，SSR 使用保存的 BatchNorm running statistics，
  每轮应用 `0.1*tanh(raw_residual/0.1)`；左右 K=0/K=3 对比表示是否启用
  SSR 精修，不表示两个不同训练阶段；窗口 A 是真实点云，B/C 是预测点云；
- SSR 体素模式使用 `[depth,row,column]`，其中
  `depth=round(200*log(Z_raw))`。为了显示居中只减去当前裁剪的中位 depth bin，
  不改变体素间相对关系。

五张 Exp9 样本的训练前点云均已归档。网站开放的样本 05、04、06 还包含
真实点云和最终检查点的 K=0/1/3/5，共 20 份唯一 PLY。文件、字节数与 SHA-256 位于
`public/data/pointcloud_sha256.json`，前端接口位于 `public/data/manifest.json`。

Exp12 为三个锁定样本保存真实点云及联合前/后 K=0/1/3/5，共 27 份 PLY、
79,635,015
字节。文件校验位于 `public/data/exp12/pointcloud_sha256.json`。样本仅依据
RGB 与 GT 细结构候选联系表选定，在查看任何 Exp12 预测前已锁定。

Exp20 保存 15 张图片的训练前 K=0，以及训练后 K=0/1/3/5；训练前
K=1/3/5 使用严格恒等别名，并为每张图增加真实点云，因此共 90 份唯一 PLY、
135 个逻辑点云、265,450,050 字节。
Train 选择 K=3 相对 K=0 改善最大的五张；Validation/Test 固定选择改善排序
第 1、5、9、12、16 名，既展示成功案例也保留退化案例。选择依据、裁剪框和
SHA-256 分别归档在 Exp20 的 `viewer_selection.json`、导出报告与
`public/data/exp20/pointcloud_sha256.json`。

全图指标使用 Exp20 正式评测归档值。SpConv CUDA 重跑在阈值附近存在轻微
非确定性，本次点云重算相对正式评测的 Point Rel 最大绝对偏差为
`2.68e-4`；该偏差与有效容差均保存在导出报告，未被当作新的实验结果。

Exp29 保存 15 张图片的真实点云和最终 K=0/1/3/5，共 75 份 PLY、
221,208,375 字节。Train/Validation/Test 均优先选择栏杆、管线、扶手、椅腿、
窗格和隔断竖杆等长细结构，并保留三张改善和两张退化案例，避免网站只展示成功
结果。完整评测中 K=3 优于 K=0 的比例分别为 87%、18.75% 与 37.5%；这说明
训练域修正能力明显，但不能据此宣称留出域泛化成功。

Exp9、Exp12、Exp20 的 21 份增补 GT 统一由
`tools/moge3/export_viewer_ground_truths.py` 生成，导出清单位于
`public/data/ground_truth_export_report.json`。因此网站四个实验的全部开放样本
现在都具有真实点云窗口，不再使用缺失占位。

## 本地运行

```bash
npm install
npm run dev
```

访问 `http://localhost:3000`，也可以从同一局域网访问
`http://10.162.196.95:3000`。如果开发服务器已在修改配置前启动，需要先按
`Ctrl+C` 终止旧进程，再重新执行 `npm run dev`，否则 Next.js 的 HMR 来源白名单
不会生效。交互方式为：

- 每个窗口可把左键切换为“旋转”或“平移”；
- 中键拖动或滚轮：缩放；
- 右键不再控制相机，避免与浏览器菜单或手势冲突；
- “适配视野”：回到当前点云的默认视角；
- “相机同步”：使所有已归档窗口使用相同相机；
- 鼠标悬浮或键盘聚焦图片卡片：放大预览原始 RGB；
- “原始输出”：查看未对齐 XYZ，此时因相对尺度不同自动关闭相机同步；
- “SSR 体素壳”：查看锁定裁剪对应的离散稀疏体素。

页面会把当前选择写入 URL，例如：

```text
?experiment=exp20_stateless_ssr_batch_statistics&split=train&sample=1&leftStage=initial&rightStage=final&leftK=0&rightK=3
```

刷新或分享链接后会恢复实验、划分、图片、左右阶段和 K。切换阶段或 K
只替换点云几何，不会重置旋转、平移或缩放；切换实验、划分或图片会自动适配
一次视野。

## 验证

```bash
npm run typecheck
npm test
npm run build
npm run test:e2e
npm audit --audit-level=high
```

Exp9 浏览器验收截图位于上级实验目录的 `artifacts/viewer_acceptance/`；
Exp12 截图位于
`../../exp12_hypersim_100_immediate_joint_finetuning/results/viewer_acceptance/`。
Exp20 的 train/val/test 验收截图位于阶段三对应实验的
`results/viewer_acceptance/`。Exp29 同样为三个划分各保存一张改善案例和
一张退化案例的三窗口截图，共六张。
需要主动更新截图时使用
`UPDATE_ACCEPTANCE_SCREENSHOTS=1 npm run test:e2e`；普通测试不会改写归档图片。

用于 Sites 的 vinext 构建命令为：

```bash
npm run build:sites
```

构建产物位于 `dist/`，不提交到 Git。`npm run build:next` 仍可验证原生
Next.js 构建；vinext 仅作为 Sites 所需的兼容部署后端，不改变页面源码和交互逻辑。

Vercel 使用 `vercel.json` 中的 `npm run build:next`，直接部署原生 Next.js
产物；它与 Sites 的 vinext 构建互不影响。

当部署包不需要重复携带大型点云时，可以在构建 Sites 前设置静态资产源：

```bash
NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN=https://moge3-exp9-viewer.vercel.app \
  npm run build:sites
```

此时实验目录、清单、RGB 与 PLY 均从 Vercel 的 `/data/*` 读取；Vercel 对该路径
返回跨域读取头。普通本地和 Vercel 构建不设置此变量，仍使用同源相对路径。

公开生产地址：<https://moge3-exp9-viewer.vercel.app>
