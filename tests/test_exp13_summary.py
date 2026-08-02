from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiment"
    / "stage2_training_strategy"
    / "runs"
    / "exp13_fixed_base_ssr_diagnostic"
    / "summarize_results.py"
)
SPEC = importlib.util.spec_from_file_location("exp13_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_summary_selects_lowest_locked_structure_rel():
    rows = [
        {
            "step": "0",
            "train/k0_point_rel": "0.10",
            "train/k3_point_rel": "0.10",
            "train/k0_structure_point_rel": "0.20",
            "train/k3_structure_point_rel": "0.20",
            "val/k0_point_rel": "0.10",
            "val/k3_point_rel": "0.10",
            "val/k0_structure_point_rel": "0.20",
            "val/k3_structure_point_rel": "0.20",
        },
        {
            "step": "400",
            "train/k0_point_rel": "0.10",
            "train/k3_point_rel": "0.095",
            "train/k0_structure_point_rel": "0.20",
            "train/k3_structure_point_rel": "0.18",
            "val/k0_point_rel": "0.10",
            "val/k3_point_rel": "0.105",
            "val/k0_structure_point_rel": "0.20",
            "val/k3_structure_point_rel": "0.16",
        },
        {
            "step": "800",
            "train/k0_point_rel": "0.10",
            "train/k3_point_rel": "0.09",
            "train/k0_structure_point_rel": "0.20",
            "train/k3_structure_point_rel": "0.15",
            "val/k0_point_rel": "0.10",
            "val/k3_point_rel": "0.11",
            "val/k0_structure_point_rel": "0.20",
            "val/k3_structure_point_rel": "0.18",
        },
    ]
    result = MODULE.training_summary("lr", rows)
    assert result["status"] == "complete"
    assert result["best_train_structure"]["step"] == 800
    assert result["best_validation_structure"]["step"] == 400
    assert (
        result["best_train_structure"]["train_structure_relative_reduction"]
        == pytest.approx(0.25)
    )
