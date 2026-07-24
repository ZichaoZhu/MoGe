# MoGe-3 reproduction status

This branch reconstructs the architecture described by the MoGe-3 paper on top
of the official MoGe-2 repository. MoGe-3 source code and checkpoints were not
publicly available when this work started, so unpublished U-Net details are
centralized in `configs/train/v3.json` and must not be treated as official.

## Implemented scope

- factorized geometry \(q=(X/Z,Y/Z,\log Z)\);
- self-guided voxelization \(c=(i,j,\operatorname{round}(D\log Z))\), with
  \(D=200\) by default and re-voxelization after every refinement;
- a shared-weight four-level SpConv sparse 3D U-Net;
- DINO feature injection at the sparse bottleneck;
- zero-initialized log-depth residual output;
- iterative \(K=0\ldots7\) inference, default \(K=3\);
- global affine-invariant, radial-partition local, and edge losses for every
  geometry estimate;
- synthetic-only SSR gradient routing and the 5,000-step base/refiner detach;
- paper learning rates, warm-up, decay, AdamW, and gradient clipping;
- CPU reference backend, remote path guard, and synthetic smoke tests.

Normal prediction and normal supervision are deliberately disabled for this
reproduction. Existing MoGe v1/v2 behavior remains available.

## Server layout

All writable server paths are below `/mnt/data/home/zhuzichao`:

```text
2026_TPAMI_InfiniGeometry/
├── third_party/MoGe-3
├── envs/moge3
├── cache/moge3
├── checkpoints/moge3
├── experiments/moge3
└── tmp/moge3
```

Set cache variables before every run:

```bash
export MOGE3_ROOT=/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry
export TMPDIR="$MOGE3_ROOT/tmp/moge3"
export HF_HOME="$MOGE3_ROOT/cache/moge3/huggingface"
export TORCH_HOME="$MOGE3_ROOT/cache/moge3/torch"
export PIP_CACHE_DIR="$MOGE3_ROOT/cache/moge3/pip"
export XDG_CACHE_HOME="$MOGE3_ROOT/cache/moge3/xdg"
export CONDA_PKGS_DIRS="$MOGE3_ROOT/cache/moge3/conda"
```

The laboratory proxy can be enabled for one command with
`HTTPS_PROXY=http://10.130.136.133:7890` and
`HTTP_PROXY=http://10.130.136.133:7890`. It is intentionally not installed as a
global shell, Git, or Conda setting.

## Smoke tests

From the repository root on the server:

```bash
ENV=/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/envs/moge3
OUT=/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/experiments/moge3/smoke-64
"$ENV/bin/python" -m pytest -q tests
"$ENV/bin/python" -m moge.scripts.smoke_ssr --output "$OUT" --height 64 --width 64
```

Four-process DDP:

```bash
"$ENV/bin/accelerate" launch --num_processes 4 \
  -m moge.scripts.smoke_ssr \
  --output /mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/experiments/moge3/smoke-ddp \
  --height 64 --width 64 --steps 20
```

`--ddp-backend gloo` is available as a diagnostic fallback when the server's
NCCL runtime itself fails the dense-PyTorch DDP probe. It is not the recommended
backend for formal training.

The current milestone does not download datasets or start a real-data training
run. Dataset integration remains blocked on the explicit data strategy.
