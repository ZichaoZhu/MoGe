# exp8_two_gpu_head_ssr_joint_finetuning 训练图细结构 GIF

选区只依据 RGB 与真值深度，并在渲染预测前锁定。所有动画均为单向 yaw −90°→+90°，共 46 帧。

| # | 样本 | 结构 | K=0 点图 Rel | K=3 点图 Rel | GIF |
|---:|---|---|---:|---:|---|
| 01 | `ai_019_004_cam_00_frame.0000` | 楼梯扶手与平行栏杆 | 9.883% | 7.408% | [查看](items/01_ai_019_004_cam_00_frame.0000/train_rod_orbit.gif) |
| 02 | `ai_052_008_cam_01_frame.0099` | 桌腿与椅子细支撑 | 2.197% | 1.766% | [查看](items/02_ai_052_008_cam_01_frame.0099/train_rod_orbit.gif) |
| 03 | `ai_026_013_cam_00_frame.0000` | 放射状屋顶细梁 | 5.621% | 5.175% | [查看](items/03_ai_026_013_cam_00_frame.0000/train_rod_orbit.gif) |
| 04 | `ai_054_008_cam_00_frame.0000` | 悬空楼梯踏板与支架 | 5.941% | 5.695% | [查看](items/04_ai_054_008_cam_00_frame.0000/train_rod_orbit.gif) |
| 05 | `ai_053_018_cam_00_frame.0000` | 密集竖直隔断杆 | 11.173% | 11.620% | [查看](items/05_ai_053_018_cam_00_frame.0000/train_rod_orbit.gif) |
| 06 | `ai_002_003_cam_00_frame.0000` | 楼梯栏杆与斜向扶手 | 7.348% | 5.912% | [查看](items/06_ai_002_003_cam_00_frame.0000/train_rod_orbit.gif) |
| 07 | `ai_019_004_cam_00_frame.0099` | 玻璃楼梯细拉杆 | 18.453% | 18.679% | [查看](items/07_ai_019_004_cam_00_frame.0099/train_rod_orbit.gif) |
| 08 | `ai_009_001_cam_00_frame.0099` | 装饰性细柱与弯曲格栅 | 3.374% | 3.352% | [查看](items/08_ai_009_001_cam_00_frame.0099/train_rod_orbit.gif) |
