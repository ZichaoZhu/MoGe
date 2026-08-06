import json
from pathlib import Path

from tools.moge3.export_exp30_viewer import (
    DEFAULT_RELATIVE_PATHS,
    EXPERIMENT,
    SPLITS,
    _stage_paths,
)
from tools.moge3.export_viewer_experiment import validate_selection


SELECTION = (
    Path(__file__).parents[1]
    / "experiment/stage5_scaling_validation/runs/"
    "exp30_hypersim_100_long_two_stage_overfit/viewer_selection.json"
)


def test_exp30_viewer_selection_has_five_unique_samples_per_split():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    validate_selection(
        selection,
        experiment=EXPERIMENT,
        splits=SPLITS,
    )
    ids = [
        row["id"]
        for split in SPLITS
        for row in selection["splits"][split]
    ]
    assert len(ids) == len(set(ids)) == 15


def test_exp30_selection_records_train_success_and_holdout_failures():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert all(
        row["outcome"] == "improved"
        for row in selection["splits"]["train"]
    )
    for split in ("val", "test"):
        outcomes = {
            row["outcome"] for row in selection["splits"][split]
        }
        assert outcomes == {"improved", "degraded"}


def test_exp30_stage_paths_lock_the_three_archived_steps(tmp_path):
    paths = {
        key: tmp_path / relative
        for key, relative in DEFAULT_RELATIVE_PATHS.items()
    }
    stages = _stage_paths(paths)
    assert tuple(stages) == ("initial", "stage1", "final")
    assert [stages[name][2] for name in stages] == [0, 20_000, 30_000]
