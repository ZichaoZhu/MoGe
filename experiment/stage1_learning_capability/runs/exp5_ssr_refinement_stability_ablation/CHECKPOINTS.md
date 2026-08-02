# exp5 检查点记录

检查点未提交到 Git，只保存在 ZJU3DV-S115 的用户安全目录：

`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/exp5_ssr_refinement_stability_ablation/artifacts/runs`

| 组别与文件 | 含义 | 步数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| A `checkpoint.pt` | 最佳验证点图 Rel | 0 | 203,095,165 | `0344f313fe5541aae3476eed580ad3d6b5c3d2368567d9fec4452c3b7cae912f` |
| A `latest_checkpoint.pt` | 完整训练结束 | 2000 | 609,120,107 | `132e0e0cd3a84ec1a5bb54e6744a425704285e32c5ffa71a7ffb1f62b3328eec` |
| B `checkpoint.pt` | 最佳验证点图 Rel | 0 | 203,095,165 | `996789798e075a085ed43807b6fcb6cda926dabf7523ffafa1c7d543f3c12838` |
| B `latest_checkpoint.pt` | 完整训练结束 | 2000 | 609,120,107 | `b603e415081117f8df680c0bd0245730b0444ff09c70cee97f8211e96c32a3bb` |
| C `checkpoint.pt` | 最佳验证点图 Rel | 0 | 203,095,165 | `cd303c5f0ed4602d6fb24a0e5242f7575ad3a957db2c8becd9a3bbf516fc5da9` |
| C `latest_checkpoint.pt` | 完整训练结束 | 2000 | 609,120,107 | `203a3a064c91035233f2e8abb69f25ca68f6f36598f9393cca606da75e18d189` |
| D `checkpoint.pt` | 最佳验证点图 Rel | 0 | 203,095,165 | `ad6ef730b508df90929dbd21e46c9bff0dc6a4fe6db662555124e9c423d111ea` |
| D `latest_checkpoint.pt` | 完整训练结束 | 2000 | 609,120,107 | `07153f64ce6cfab232b9cfd7bbd29180935efb8721dc0f62507e3640c62ab431` |

A/B/C/D 分别对应 README 中的四组消融。所有最佳检查点都是 step 0，不是保存错误：
后续检查点的完整验证集点图 Rel 均未优于零初始化。分析训练后 SSR 行为时应使用各组
的 `latest_checkpoint.pt`。
