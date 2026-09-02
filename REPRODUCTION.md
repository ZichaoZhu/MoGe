# MoGe-3 复现记录

## 固定基线

| 项目 | 固定版本 |
| --- | --- |
| 上游代码 | `https://github.com/microsoft/MoGe.git` |
| 上游 commit | `74fbce054ebed49800de42d0ad0e83495065719a` (`V3 (#158)`) |
| 本仓库分支 | `repro/moge3-74fbce0` |
| 官方模型 | `Ruicheng/moge-3-vitl` |
| 模型 revision | `a96e58bad16a94c9a3c193a5d4cd75b4b6906c94` |
| 模型许可 | MIT |
| 官方评测集 | `Ruicheng/monocular-geometry-evaluation` |
| 评测集 revision | `574610d77c0cc14d911a1ff46ec529ba934c8960` |

评测集仓库未声明统一许可证。使用和引用其中各数据集时，应分别遵守原数据集许可证；不将数据文件提交或重新分发到本仓库。

## 复现范围

按以下顺序执行，上一阶段通过后再进入下一阶段：

1. 使用官方 `moge-3-vitl` checkpoint 完成固定输入的推理 smoke test。
2. 使用官方 `configs/eval/moge3.json` 评测 checkpoint，并区分官方报告值与本地复现值。
3. 是否复现训练由独立实验决定；训练结果不得与官方 checkpoint 评测混写。

当前状态：代码、模型和评测数据 revision 已固定；推理与评测尚未执行，暂无复现指标。

## 记录要求

每次运行至少记录：

- 代码 commit、模型 revision、数据 revision 和配置文件哈希；
- 命令、随机种子、Python 依赖锁文件、GPU、驱动和 CUDA 版本；
- 输入清单、原始日志、逐数据集指标和聚合方式；
- 相对上游代码、配置或评测口径的全部改动。

数据、checkpoint、虚拟环境和运行输出只保存在实验存储中，不提交到 Git。Git 只跟踪代码、配置、命令、轻量级 provenance 和结果报告。

## 结果边界

- “官方结果”仅指论文或上游仓库公开的数值。
- “checkpoint 复现”指固定官方权重后的本地评测，不等同于训练复现。
- “训练复现”必须从论文规定的初始化和数据开始，并单独报告随机种子、训练预算和方差。
- 任何修改版实验都使用新分支和新输出目录，不覆盖本基线结果。

引用方式沿用上游 `README.md` 中的 MoGe-3 BibTeX，并同时引用实际使用的评测数据集。
