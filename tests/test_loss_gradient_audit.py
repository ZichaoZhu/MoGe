import torch

from tools.moge3.audit_loss_gradients import (
    aggregate_rows,
    gradient_statistics,
    select_locked_samples,
)
from experiment.stage4_loss_gradient_audit.runs.exp25_loss_gradient_structure_audit.plot_results import mean_metric
from experiment.stage4_loss_gradient_audit.runs.exp25_loss_gradient_structure_audit.summarize_results import (
    relative_reduction,
    summarize,
)
from moge.scripts.train_hypersim_joint_v3 import RawSample


def _sample(sample_id: str, split: str, has_crop: bool = True) -> RawSample:
    return RawSample(
        sample_id=sample_id,
        split=split,
        scene="scene",
        frame=0,
        image=torch.zeros(3, 2, 3),
        gt_points=torch.ones(2, 3, 3),
        crop_xyxy=(0, 0, 2, 2) if has_crop else None,
        structure_mask_config={"type": "near_quantile", "quantile": 0.5},
    )


def test_select_locked_samples_preserves_manifest_order_and_leaveout():
    samples = [
        _sample("t0", "train"),
        _sample("t1", "train"),
        _sample("t2", "train"),
        _sample("v0", "val"),
        _sample("x0", "test"),
        _sample("v-no-roi", "val", has_crop=False),
    ]
    selected = select_locked_samples(samples, train_limit=2)
    assert [sample.sample_id for sample in selected] == ["t0", "t1", "v0", "x0"]


def test_gradient_statistics_reports_spatial_enrichment():
    gradient = torch.tensor([[1.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    valid = torch.ones_like(gradient, dtype=torch.bool)
    masks = {
        "valid": valid,
        "crop": torch.tensor(
            [[False, False, False], [False, True, False]]
        ),
        "structure": torch.tensor(
            [[True, False, False], [False, True, False]]
        ),
        "gt_boundary": torch.tensor(
            [[True, True, False], [False, False, False]]
        ),
    }
    stats = gradient_statistics(gradient, masks)
    assert stats["gradient_l1_mass"] == 4.0
    assert stats["crop_gradient_mass_share"] == 0.5
    assert stats["crop_pixel_share"] == 1 / 6
    assert stats["crop_gradient_enrichment"] == 3.0


def test_aggregate_rows_groups_split_k_and_objective():
    rows = [
        {
            "sample_id": "a",
            "scene": "s",
            "frame": 0,
            "split": "train",
            "k": 0,
            "objective": "global",
            "loss": 1.0,
        },
        {
            "sample_id": "b",
            "scene": "s",
            "frame": 1,
            "split": "train",
            "k": 0,
            "objective": "global",
            "loss": 3.0,
        },
    ]
    aggregates = aggregate_rows(rows)
    assert len(aggregates) == 1
    assert aggregates[0]["sample_count"] == 2
    assert aggregates[0]["loss_mean"] == 2.0
    assert aggregates[0]["loss_median"] == 2.0


def test_plot_mean_metric_filters_scope():
    rows = [
        {
            "split": "train",
            "k": "0",
            "objective": "global",
            "loss": "1.0",
        },
        {
            "split": "train",
            "k": "0",
            "objective": "global",
            "loss": "3.0",
        },
        {
            "split": "val",
            "k": "0",
            "objective": "global",
            "loss": "100.0",
        },
    ]
    assert (
        mean_metric(
            rows,
            split="train",
            step=0,
            objective="global",
            metric="loss",
        )
        == 2.0
    )


def test_exp25_summary_separates_train_and_leaveout_direction():
    rows = []
    for split, k0, k3 in (
        ("train", 10.0, 8.0),
        ("val", 10.0, 11.0),
        ("test", 10.0, 12.0),
    ):
        for step, combined in ((0, k0), (3, k3)):
            for objective, loss in (
                ("global", 1.0),
                ("local", 2.0),
                ("edge_paper_min", 0.1),
                ("combined_paper", combined),
            ):
                rows.append(
                    {
                        "split": split,
                        "k": str(step),
                        "objective": objective,
                        "loss": str(loss),
                        "gradient_l1_mass": "1",
                        "structure_gradient_enrichment": "1",
                        "gt_boundary_gradient_enrichment": "1",
                        "cosine_global_local": "0.2",
                        "cosine_global_edge": "0.1",
                        "cosine_local_edge": "0.3",
                    }
                )
    report = {
        "checkpoint_step": 800,
        "checkpoint_sha256": "abc",
        "shape": [384, 512],
        "selection": {},
        "edge_formula": {},
        "peak_cuda_memory_bytes": {},
    }
    payload = summarize(rows, report)
    assert payload["objective_change_k0_to_k3"]["train"][
        "k3_relative_reduction"
    ] == 0.2
    assert payload["objective_change_k0_to_k3"]["val"][
        "k3_relative_reduction"
    ] == -0.1
    assert relative_reduction(10.0, 12.0) == -0.2
