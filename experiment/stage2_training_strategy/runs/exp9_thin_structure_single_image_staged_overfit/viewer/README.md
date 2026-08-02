# MoGe-3 多实验双窗口点云查看器

本目录提供 Exp9 与 Exp12 点云的统一交互式浏览器。页面顶部先选择实验，再选择
样本；两个窗口可独立选择阶段和 K。默认实验为 Exp12，左侧显示联合前 K=0，
右侧显示联合后 K=3。

- Exp9：开放样本 05、04、06，展示三项彼此独立的单图极限过拟合；
- Exp12：开放一张训练、一张验证和一张测试图片，展示 100 张训练图联合微调前后；
- Exp12 的训练样本与 Exp9 样本 04 是同一帧，可直接比较单图过拟合与百图训练。

实验目录位于 `public/data/experiments.json`。Exp9 清单保持
`public/data/manifest.json`，Exp12 清单与资产位于 `public/data/exp12/`。

## 数据语义

- PLY 保存模型输出的原始 MoGe 相机坐标 XYZ，没有把 GT 对齐结果写回文件；
- “GT 对齐”在浏览器中应用
  \(P_{\mathrm{aligned}}=sP_{\mathrm{raw}}+(0,0,t)\)；
- Three.js 显示坐标使用 `[x,-y,-z]`；
- 训练前 SSR 为零初始化，K=0/1/3/5 逐元素相同，因此只保存一份
  `initial_k0.ply`，其他 K 在清单中使用别名；此规则仅适用于 Exp9；
- Exp12 的“联合前”是已经训练到 step 3000 的 Exp11 模型，其 SSR 不为零，
  因此联合前/后均实际保存 K=0/1/3/5，不使用别名；
- SSR 体素模式使用 `[depth,row,column]`，其中
  `depth=round(200*log(Z_raw))`。为了显示居中只减去当前裁剪的中位 depth bin，
  不改变体素间相对关系。

五张 Exp9 样本的训练前点云均已归档。样本 05、04、06 还包含最终检查点的
K=0/1/3/5，共 17 份唯一 PLY。文件、字节数与 SHA-256 位于
`public/data/pointcloud_sha256.json`，前端接口位于 `public/data/manifest.json`。

Exp12 为三个锁定样本保存联合前/后 K=0/1/3/5，共 24 份 PLY、70,786,680
字节。文件校验位于 `public/data/exp12/pointcloud_sha256.json`。样本仅依据
RGB 与 GT 细结构候选联系表选定，在查看任何 Exp12 预测前已锁定。

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
- “相机同步”：使两个窗口使用相同相机；
- “原始输出”：查看未对齐 XYZ，此时因相对尺度不同自动关闭相机同步；
- “SSR 体素壳”：查看锁定裁剪对应的离散稀疏体素。

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

公开生产地址：<https://moge3-exp9-viewer.vercel.app>
