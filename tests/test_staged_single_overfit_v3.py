import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from moge.scripts.overfit_hypersim_staged_v3 import (
    atomic_torch_save,
    backbone_learning_rate,
    build_structure_mask,
    configure_depth_trainable_modules,
    loss_window_improvement,
    meaningful_best_update,
    plateau_reached,
    _scope_metrics,
    zero_identity_error,
)
from tools.moge3.visualize_staged_single_overfit import save_gif


class _TinyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssr = nn.Linear(1, 1)
        self.neck = nn.Linear(1, 1)
        self.points_head = nn.Linear(1, 1)
        self.mask_head = nn.Linear(1, 1)
        self.scale_head = nn.Linear(1, 1)
        self.normal_head = nn.Linear(1, 1)
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(1, 1)


class _TinyRoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(0.4))
        self.visual = nn.Parameter(torch.tensor(0.2))
        self.ssr = nn.Parameter(torch.tensor(0.1))

    def forward(self, detached: bool):
        base = self.base + self.visual
        refiner_input = base.detach() if detached else base
        refined = refiner_input + self.ssr
        return base, refined


def test_plateau_requires_both_metric_and_loss_stagnation():
    assert not plateau_reached(
        {"k0": 5, "k3": 4},
        patience=5,
        loss_improvement=0.001,
    )
    assert not plateau_reached(
        {"k0": 5, "k3": 5},
        patience=5,
        loss_improvement=0.006,
    )
    assert plateau_reached(
        {"k0": 5, "k3": 5},
        patience=5,
        loss_improvement=0.004,
    )


def test_loss_window_and_meaningful_best_use_relative_half_percent():
    losses = [1.0] * 100 + [0.996] * 100
    assert loss_window_improvement(losses, window=200, chunk=100) == pytest.approx(
        0.004
    )
    best, changed = meaningful_best_update(None, 0.10, 0.005)
    assert changed and best == 0.10
    best, changed = meaningful_best_update(best, 0.0996, 0.005)
    assert not changed and best == 0.10
    best, changed = meaningful_best_update(best, 0.0994, 0.005)
    assert changed and best == 0.0994


def test_backbone_freeze_and_linear_warmup_match_exp9():
    assert backbone_learning_rate(
        1000,
        freeze_steps=1000,
        warmup_end=2000,
        peak_lr=5e-7,
    ) == 0.0
    assert backbone_learning_rate(
        1500,
        freeze_steps=1000,
        warmup_end=2000,
        peak_lr=5e-7,
    ) == pytest.approx(2.5e-7)
    assert backbone_learning_rate(
        2000,
        freeze_steps=1000,
        warmup_end=2000,
        peak_lr=5e-7,
    ) == pytest.approx(5e-7)


def test_only_depth_pipeline_modules_are_trainable():
    model = _TinyDepthModel()
    counts = configure_depth_trainable_modules(model)
    assert set(counts) == {"ssr", "neck", "points_head", "backbone"}
    for name, parameter in model.named_parameters():
        expected = name.startswith(
            ("ssr.", "neck.", "points_head.", "encoder.backbone.")
        )
        assert parameter.requires_grad is expected


def test_detached_stage_refined_loss_cannot_update_base_or_visual_modules():
    model = _TinyRoutingModel()
    base, refined = model(detached=True)
    refined.square().backward(retain_graph=True)
    assert model.ssr.grad is not None
    assert model.base.grad is None
    assert model.visual.grad is None

    model.zero_grad(set_to_none=True)
    base, refined = model(detached=True)
    (base.square() + refined.square()).backward()
    assert model.base.grad is not None
    assert model.visual.grad is not None
    detached_base_gradient = model.base.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    base, refined = model(detached=False)
    (base.square() + refined.square()).backward()
    assert model.ssr.grad is not None
    assert model.base.grad.abs() > detached_base_gradient.abs()
    assert model.visual.grad.abs() > detached_base_gradient.abs()


def test_locked_structure_mask_is_derived_only_from_ground_truth_depth():
    depth = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)
    rows, cols = torch.meshgrid(
        torch.linspace(-0.3, 0.3, 8),
        torch.linspace(-0.4, 0.4, 8),
        indexing="ij",
    )
    points = torch.stack((cols * depth, rows * depth, depth), dim=-1)
    mask = build_structure_mask(
        points,
        (0, 0, 8, 8),
        {"type": "near_quantile", "quantile": 0.5},
    )
    assert mask.shape == (8, 8)
    assert mask.sum().item() == 32


def test_negative_aligned_depth_is_penalized_instead_of_removed():
    gt = torch.ones(8, 8, 3)
    gt[..., :2] = 0.0
    prediction = gt.clone()
    prediction[..., 2] = -2.0
    metrics = _scope_metrics(
        prediction,
        gt,
        torch.ones(8, 8, dtype=torch.bool),
        boundary_threshold=0.03,
    )
    assert metrics["pixels"] == 64
    assert metrics["depth_rel"] == pytest.approx(3.0)
    assert metrics["depth_delta_1.01"] == 0.0


def test_zero_identity_and_atomic_checkpoint_reload(tmp_path):
    base = torch.randn(3, 4, 3)
    assert zero_identity_error({0: base, 1: base, 3: base, 5: base}) == 0.0
    target = tmp_path / "resume.pt"
    payload = {
        "model": {"weight": torch.arange(4)},
        "optimizer": {"state": {"step": 17}},
        "step": 17,
        "phase": "joint",
        "trackers": {"joint_stale": 2},
    }
    atomic_torch_save(payload, target)
    loaded = torch.load(target, weights_only=False)
    assert loaded["step"] == 17
    assert loaded["phase"] == "joint"
    torch.testing.assert_close(loaded["model"]["weight"], torch.arange(4))
    assert not target.with_name(f"{target.name}.incomplete").exists()


def test_exp9_gif_can_be_decoded_frame_by_frame(tmp_path, monkeypatch):
    height = width = 8
    rows, cols = np.meshgrid(
        np.linspace(-0.2, 0.2, height),
        np.linspace(-0.3, 0.3, width),
        indexing="ij",
    )
    depth = np.ones((height, width), dtype=np.float32) * 2.0
    points = np.stack((cols * depth, rows * depth, depth), axis=-1)
    image = np.full((height, width, 3), 0.6, dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)
    output = tmp_path / "test.gif"
    monkeypatch.setattr(
        "tools.moge3.visualize_staged_single_overfit.render_orbit_frame",
        lambda **kwargs: Image.new(
            "RGB",
            (24, 16),
            (int(kwargs["yaw"] + 90), 80, 140),
        ),
    )
    save_gif(
        output,
        label="complete crop",
        full_image=image,
        crop=(0, 0, width, height),
        crop_image=image,
        gt_points=points,
        predictions={f"Final K={k}": points.copy() for k in (0, 1, 3, 5)},
        metrics={
            f"Final K={k}": {"point_rel": 0.0}
            for k in (0, 1, 3, 5)
        },
        mask=mask,
        structure_mask=mask,
        yaw_values=(-90.0, 0.0, 90.0),
        pitch=-8.0,
        duration_ms=110,
    )
    with Image.open(output) as gif:
        assert gif.n_frames == 3
        for frame in range(gif.n_frames):
            gif.seek(frame)
            gif.convert("RGB").load()
