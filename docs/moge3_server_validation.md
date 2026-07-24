# ZJU3DV-S115 validation report

Validation date: 2026-07-24. Repository base commit:
`925b8ed835a7a9cdb7578ba15c658a0afc969030`.

## Environment

| Component | Result |
|---|---|
| GPUs | 4 × NVIDIA GeForce RTX 4090, 50,864,390,144 bytes each, SM89 |
| Python | 3.12.13 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| SpConv | spconv-cu126 2.3.8 |
| accelerate | 1.14.0 |
| isolated environment | `/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/envs/moge3` |

Every physical GPU passed an independent `SparseConv3d` forward and backward.
All configured caches and temporary paths resolve below
`/mnt/data/home/zhuzichao`.

## Correctness

- 19 unit tests pass: geometry factorization, voxel counts, discontinuity
  separation, pixel remapping, DINO feature sampling, iterative re-voxelization,
  affine invariance, radial local loss corner cases, path guarding, optimizer
  schedules, and synthetic/real gradient routing.
- The MoGe-2 ViT-L checkpoint (1.32 GB) loads into v3 with `strict=False` only
  for the new SSR parameters and removed normal head.
- Normal prediction is absent.
- The zero-initialized SSR has exactly zero residual and bitwise-identical point
  maps for K=1, K=3, and K=5 relative to K=0 within the same forward.
- Full model parameter count: 376,948,294; SSR parameter count: 50,739,073.
- A 100-step synthetic overfit reduces log-depth MSE from
  0.003508728 to 0.000039277.
- Accelerate checkpoints save, restore, and continue optimization.

## Performance probes

These are SSR-only forward/backward probes, not full-model benchmark numbers.

| Input / configuration | Peak allocated GPU memory | Time per optimization step |
|---|---:|---:|
| 64×64, K=1, small U-Net | 51,651,584 B | 212 ms |
| 384×384, K=3, production U-Net | 3,872,129,536 B | 2.35 s |
| 700×700, K=3, production U-Net | 11,843,052,032 B | 6.46 s |
| 4×64×64, K=3, four-rank Gloo | 80,463,872 B/rank | 141 ms |

Active voxel counts at 700×700 were
`[490000, 136060, 36150, 9445, 2437]`.

## Real Hypersim single-batch overfit

The authorized `/nas1/datasets/hypersim/raw` mount was used strictly as a
read-only source. Only frame 0000 from `ai_001_001/cam_00` was checksum-copied
into the project experiment directory. The batch contains a tone-mapped RGB
image and `depth_meters`; no normal map was copied or used.

Hypersim depth stores Euclidean camera distance rather than planar Z depth.
Ground-truth points were reconstructed with the scene-specific
`M_cam_from_uv`, normalized camera rays, and the coordinate conversion
`[x,y,z]_Hypersim -> [x,-y,-z]_MoGe`. A read-only cross-check against the
Hypersim position map gave 0.67 mm mean point error and 1.58 mm p99 error.

The MoGe-2 ViT-L base and all 2D heads were frozen. The production 50.7M
parameter SSR alone was optimized for 200 steps at 192×256, K=3, with the
global, radial local (α=4,16,64), and edge geometry losses.

| Metric | Frozen K=0 base | K=3 after overfit |
|---|---:|---:|
| aligned point Rel | 0.020082 | 0.003811 |
| aligned depth Rel | 0.018444 | 0.002783 |
| depth δ1.01 | 0.511800 | 0.965698 |
| depth δ1.25 | 0.999003 | 0.999858 |

The summed training loss decreased from 0.048328 to 0.014564. Training took
84.9 seconds (0.424 seconds/step after initialization) and peaked at 4.63 GB
allocated GPU memory. The zero-initialized residual was exactly zero before
optimization. A fresh process strictly reloaded the SSR checkpoint with no
missing or unexpected keys and reproduced depth Rel 0.002783 exactly.

Artifacts are stored under:
`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/experiments/moge3/hypersim-single-batch/run-200`.
The checkpoint SHA-256 is
`6ed6998ed7cf8270890a8579c9c4e7f23245b431aaa439170ed6d65df863e495`.

## Distributed-runtime finding

NCCL DDP currently segfaults in
`torch.distributed._verify_param_shape_across_processes` on this server. The
same failure reproduces with `tools/moge3/ddp_probe.py`, which contains only one
dense `torch.nn.Linear` and no MoGe or SpConv code. Disabling NCCL P2P, IB, cuMem,
and SpConv JIT does not change the failure.

The four-GPU MoGe-3 synthetic run passes with the Gloo process-group fallback.
This establishes model/DDP correctness but is slower than NCCL. Formal
large-scale training should wait for the server PyTorch/NCCL runtime to be
repaired or replaced with a verified environment.

## Data cleanup

With explicit authorization for the no-normal reproduction, the following
irrelevant data were permanently removed:

- Taskonomy/Omnidata normal: about 238 GB;
- Taskonomy/Omnidata reshading: about 137 GB;
- unusable Hypersim RGB+normal derivative: about 51 GB;
- empty held-out and Taskonomy shell directories.

About 426 GB was recovered. The 469 GB RGB/depth/mask Taskonomy candidate was
preserved. No path outside `/mnt/data/home/zhuzichao` was modified.
