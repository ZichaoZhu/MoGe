import json

import pytest

from tools.moge3.run_exp30 import (
    GpuState,
    best_checkpoint,
    eligible_gpus,
    find_data_root,
    full_scores,
    window_improvement,
)


def test_gpu_selection_requires_28_gib_free():
    states = [
        GpuState(0, 30_000, 10),
        GpuState(1, 28_000, 0),
        GpuState(2, 40_000, 90),
    ]
    assert eligible_gpus(states, 28_672) == [0, 2]


def test_full_composite_score_and_plateau_window(tmp_path):
    history = tmp_path / "full_evaluation_history.csv"
    history.write_text(
        "step,train/k3_point_rel,train/k3_structure_point_rel\n"
        "15000,0.03,0.04\n"
        "20000,0.025,0.035\n",
        encoding="utf-8",
    )
    scores = full_scores([history])
    assert [step for step, _ in scores] == [15000, 20000]
    assert [score for _, score in scores] == pytest.approx([0.07, 0.06])
    assert window_improvement(
        scores,
        end_step=20000,
        window_steps=5000,
    ) == pytest.approx(1 / 7)


def test_best_checkpoint_is_selected_across_recovery_attempts(tmp_path):
    attempts = []
    for index, score in enumerate((0.08, 0.06)):
        attempt = tmp_path / f"attempt_{index:02d}"
        attempt.mkdir()
        (attempt / "checkpoint.pt").write_bytes(b"checkpoint")
        (attempt / "checkpoint_metadata.json").write_text(
            json.dumps({"step": 1000 + index, "score": score}),
            encoding="utf-8",
        )
        attempts.append(attempt)
    checkpoint, metadata = best_checkpoint(attempts)
    assert checkpoint.parent.name == "attempt_01"
    assert metadata["score"] == 0.06


def test_data_root_supports_the_server_legacy_layout(tmp_path):
    repository = tmp_path / "repo"
    legacy = (
        repository
        / "experiment/exp11_hypersim_100_train_staged_joint_overfit/data"
    )
    legacy.mkdir(parents=True)
    assert find_data_root(repository, safe_root=tmp_path) == legacy
