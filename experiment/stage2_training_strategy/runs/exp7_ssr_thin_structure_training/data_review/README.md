# 数据审核说明

- `curated_candidates/hypersim_curated_24_contact_sheet.jpg`：人工二次筛选的 24 张
  Hypersim 代表帧，含候选区域放大图。
- `curated_candidates/hypersim_curated_24.json`：Hypersim 候选的场景、相机、帧、
  RGB、深度和裁剪坐标。
- `curated_candidates/tartanair_curated_24_contact_sheet.jpg`：从三个较合适的
  TartanAir V2 环境中筛出的 24 张代表帧。
- `curated_candidates/tartanair_curated_24_rgb_depth_contact_sheet.jpg`：相同
  24 帧的 RGB 与反深度对照，用于人工检查细结构深度边界和像素对齐。
- `curated_candidates/tartanair_curated_24.json`：TartanAir 候选的环境、轨迹、
  帧和 ZIP 成员路径。
- `curated_candidates/tartanair_modality_audit.json`：八个环境的 RGB/Depth
  配对数量、相机内参、深度编码与 Z 深度到光心欧氏距离的转换规则。
- `tartanair_uniform_samples/`：八个已下载环境的均匀抽帧审计表，不代表最终选择。

所有联系表仅用于训练前人工审核，没有读取模型预测或测试集表现。
