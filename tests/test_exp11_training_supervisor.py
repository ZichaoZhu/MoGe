from __future__ import annotations

from experiment.stage2_training_strategy.runs.exp11_hypersim_100_train_staged_joint_overfit.supervise_training import (
    select_gpus,
)


def test_selects_largest_safe_gpu_set_in_free_memory_order() -> None:
    records = [
        {"index": 0, "memory_free_mib": 23177, "utilization_percent": 21},
        {"index": 1, "memory_free_mib": 33317, "utilization_percent": 0},
        {"index": 2, "memory_free_mib": 19995, "utilization_percent": 0},
        {"index": 3, "memory_free_mib": 32403, "utilization_percent": 0},
    ]
    assert select_gpus(
        records,
        minimum_free_mib=18000,
        maximum_utilization=30,
        maximum_count=4,
    ) == [1, 3, 0, 2]


def test_excludes_gpu_below_memory_or_above_utilization_threshold() -> None:
    records = [
        {"index": 0, "memory_free_mib": 17999, "utilization_percent": 0},
        {"index": 1, "memory_free_mib": 30000, "utilization_percent": 31},
        {"index": 2, "memory_free_mib": 20000, "utilization_percent": 30},
    ]
    assert select_gpus(
        records,
        minimum_free_mib=18000,
        maximum_utilization=30,
        maximum_count=4,
    ) == [2]
