import math

import torch

from moge.scripts.train_hypersim_joint_v3 import RawSample
from tools.moge3.audit_ssr_parameter_gradients import (
    aggregate_rows,
    combine_gradients,
    gradient_cosine,
    gradient_statistics,
    parameter_group,
    restrict_gradients,
    select_microbatches,
)
from experiment.stage4_loss_gradient_audit.runs.exp26_ssr_parameter_gradient_conflict_audit.summarize_results import (
    group_energy_shares,
    ratio_by_batch,
)


def _sample(sample_id: str, split: str) -> RawSample:
    return RawSample(
        sample_id=sample_id,
        split=split,
        scene="scene",
        frame=0,
        image=torch.zeros(3, 2, 2),
        gt_points=torch.ones(2, 2, 3),
    )


def test_select_microbatches_is_manifest_ordered_and_split_local():
    samples = [
        _sample("t0", "train"),
        _sample("v0", "val"),
        _sample("t1", "train"),
        _sample("t2", "train"),
        _sample("v1", "val"),
        _sample("t3", "train"),
    ]
    batches = select_microbatches(
        samples,
        splits=("train", "val"),
        pairs_per_split=1,
    )
    assert [
        (split, index, [sample.sample_id for sample in pair])
        for split, index, pair in batches
    ] == [
        ("train", 0, ["t0", "t1"]),
        ("val", 0, ["v0", "v1"]),
    ]


def test_combine_gradients_and_cosine_are_exact():
    global_gradient = (torch.tensor([1.0, 0.0]), None)
    edge_gradient = (torch.tensor([0.0, 2.0]), torch.tensor([3.0]))
    combined = combine_gradients(
        [global_gradient, edge_gradient],
        [1.0, 0.5],
    )
    torch.testing.assert_close(combined[0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(combined[1], torch.tensor([1.5]))
    assert gradient_cosine(global_gradient, edge_gradient) == 0.0
    assert math.isclose(
        gradient_cosine(combined, combined),
        1.0,
        rel_tol=1e-12,
    )


def test_gradient_statistics_counts_inactive_parameters():
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    )
    named = tuple(module.named_parameters())
    gradients = (torch.ones_like(named[0][1]), None)
    statistics = gradient_statistics(named, gradients)
    assert statistics["parameter_numel"] == 6
    assert statistics["active_parameter_numel"] == 4
    assert statistics["active_parameter_tensors"] == 1
    assert math.isclose(statistics["gradient_l2"], 2.0)
    assert math.isclose(statistics["gradient_rms"], 2 / math.sqrt(6))
    assert math.isclose(statistics["gradient_nonzero_fraction"], 4 / 6)


def test_parameter_group_covers_sparse_unet_families():
    assert parameter_group("input_projection.weight") == "input_fusion"
    assert parameter_group("unet.visual_projection.weight") == "input_fusion"
    assert parameter_group("encoder_blocks.2.0.conv1.weight") == "encoder"
    assert parameter_group("encoder_blocks.4.0.conv1.weight") == "bottleneck"
    assert parameter_group("bottleneck_fusion.weight") == "bottleneck"
    assert parameter_group("decoder_blocks.0.0.conv1.weight") == "decoder"
    assert parameter_group("output_layer.weight") == "output"


def test_restrict_gradients_separates_output_layer():
    named = (
        ("unet.encoder_blocks.0.weight", torch.nn.Parameter(torch.ones(1))),
        ("unet.output_layer.weight", torch.nn.Parameter(torch.ones(1))),
    )
    gradients = (torch.tensor([2.0]), torch.tensor([3.0]))
    non_output = restrict_gradients(named, gradients, scope="non_output")
    output = restrict_gradients(named, gradients, scope="output")
    assert non_output[0] is gradients[0] and non_output[1] is None
    assert output[0] is None and output[1] is gradients[1]


def test_aggregate_rows_keeps_objective_scope():
    rows = [
        {"split": "train", "objective": "global", "gradient_l2": 1.0},
        {"split": "train", "objective": "global", "gradient_l2": 3.0},
        {"split": "val", "objective": "global", "gradient_l2": 9.0},
    ]
    aggregated = aggregate_rows(rows, keys=("split", "objective"))
    train = next(row for row in aggregated if row["split"] == "train")
    assert train["batch_count"] == 2
    assert train["gradient_l2_mean"] == 2.0


def test_group_energy_share_uses_squared_l2_norm():
    objective_rows = [
        {
            "batch_id": "train_00",
            "split": "train",
            "objective": "edge_paper",
            "gradient_l2": "5",
        }
    ]
    group_rows = [
        {
            "batch_id": "train_00",
            "split": "train",
            "objective": "edge_paper",
            "parameter_group": "output",
            "gradient_l2": "4",
        },
        {
            "batch_id": "train_00",
            "split": "train",
            "objective": "edge_paper",
            "parameter_group": "encoder",
            "gradient_l2": "3",
        },
    ]
    shares = group_energy_shares(objective_rows, group_rows)
    output = next(
        row for row in shares if row["parameter_group"] == "output"
    )
    assert output["energy_share"] == 16 / 25


def test_ratio_by_batch_averages_ratios_not_ratio_of_means():
    rows = []
    for batch_id, edge, global_ in (
        ("train_00", 1.0, 2.0),
        ("train_01", 6.0, 3.0),
    ):
        for objective, value in (
            ("edge_paper", edge),
            ("global", global_),
        ):
            rows.append(
                {
                    "batch_id": batch_id,
                    "split": "train",
                    "objective": objective,
                    "gradient_l2": str(value),
                }
            )
    assert ratio_by_batch(
        rows,
        "edge_paper",
        "global",
        "gradient_l2",
    ) == {"train": 1.25}
