from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiment"
    / "stage2_training_strategy"
    / "runs"
    / "exp13_fixed_base_ssr_diagnostic"
    / "supervise_training.py"
)
SPEC = importlib.util.spec_from_file_location("exp13_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gpu_selection_prefers_more_free_memory_and_honors_exclusions():
    records = [
        {
            "index": 0,
            "memory_free_mib": 20000,
            "memory_used_mib": 28000,
            "utilization_percent": 0,
        },
        {
            "index": 1,
            "memory_free_mib": 30000,
            "memory_used_mib": 18000,
            "utilization_percent": 5,
        },
        {
            "index": 2,
            "memory_free_mib": 40000,
            "memory_used_mib": 8000,
            "utilization_percent": 80,
        },
    ]
    assert MODULE.select_gpus(
        records,
        minimum_free_mib=18000,
        maximum_utilization=10,
        excluded={0},
        maximum_count=2,
    ) == [1]
