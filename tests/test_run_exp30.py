import json

import pytest

from tools.moge3.run_exp30 import (
    GpuState,
    best_checkpoint,
    eligible_gpus,
    failed_stage1_continuation,
    find_data_root,
    full_scores,
    latest_verified_checkpoint,
    training_launcher,
    update_status,
    window_improvement,
)


def test_gpu_selection_requires_28_gib_free():
    states = [
        GpuState(0, 30_000, 10),
        GpuState(1, 28_000, 0),
        GpuState(2, 40_000, 90),
    ]
    assert eligible_gpus(states, 28_672) == [2, 0]


def test_single_and_multi_gpu_use_the_same_gloo_launcher(tmp_path):
    single = training_launcher(tmp_path / "python", process_count=1)
    multi = training_launcher(tmp_path / "python", process_count=4)
    assert "torch.distributed.run" in single
    assert "--nproc_per_node=1" in single
    assert "--nproc_per_node=4" in multi


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


def test_update_status_clears_stale_failure_and_wait_fields(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "status.json").write_text(
        json.dumps(
            {
                "started_at": "earlier",
                "state": "waiting_for_gpu",
                "error": "RuntimeError",
                "message": "old failure",
                "failed_at": "earlier",
                "requested_gpu_count": 4,
                "eligible_gpu_indices": [3],
                "gpu_states": [],
                "next_check_seconds": 600,
            }
        ),
        encoding="utf-8",
    )

    update_status(tmp_path, state="running", phase="stage1")

    payload = json.loads((artifacts / "status.json").read_text())
    assert payload["state"] == "running"
    assert payload["phase"] == "stage1"
    assert payload["started_at"] == "earlier"
    for key in (
        "error",
        "message",
        "failed_at",
        "requested_gpu_count",
        "eligible_gpu_indices",
        "gpu_states",
        "next_check_seconds",
    ):
        assert key not in payload


def test_stability_recovery_uses_cross_attempt_verified_milestone(tmp_path):
    attempts = []
    for index in range(3):
        attempt = tmp_path / f"attempt_{index:02d}"
        attempt.mkdir()
        (attempt / "resume_checkpoint.pt").write_bytes(b"unsafe-resume")
        attempts.append(attempt)
    milestones = attempts[0] / "milestones"
    milestones.mkdir()
    safe_2500 = milestones / "step_002500.pt"
    safe_2500.write_bytes(b"safe")
    later_milestones = attempts[2] / "milestones"
    later_milestones.mkdir()
    safe_5000 = later_milestones / "step_005000.pt"
    safe_5000.write_bytes(b"safer")

    assert latest_verified_checkpoint(attempts) == safe_5000


def test_stability_recovery_never_falls_back_to_resume_checkpoint(tmp_path):
    attempt = tmp_path / "attempt_00"
    attempt.mkdir()
    (attempt / "resume_checkpoint.pt").write_bytes(b"unsafe-resume")
    initial = attempt / "initial_checkpoint.pt"
    initial.write_bytes(b"known-safe-initialization")

    assert latest_verified_checkpoint([attempt]) == initial


def test_failed_stage1_continuation_creates_new_attempt_from_milestone(
    tmp_path,
):
    experiment = tmp_path / "exp30"
    stage = experiment / "artifacts" / "training" / "stage1"
    for index in range(3):
        attempt = stage / f"attempt_{index:02d}"
        attempt.mkdir(parents=True)
        (attempt / "resume_checkpoint.pt").write_bytes(b"unsafe-resume")
    milestones = stage / "attempt_00" / "milestones"
    milestones.mkdir()
    safe = milestones / "step_002500.pt"
    safe.write_bytes(b"safe")
    (experiment / "artifacts" / "status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "phase": "stage1",
                "recovery_count": 2,
                "learning_rate_scale": 0.25,
            }
        ),
        encoding="utf-8",
    )

    continuation = failed_stage1_continuation(experiment)

    assert continuation.next_attempt_index == 3
    assert continuation.recovery_count == 2
    assert continuation.learning_rate_scale == 0.25
    assert continuation.checkpoint == safe
