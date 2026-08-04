from tools.moge3.monitor_exp30 import (
    decide_action,
    latest_logged_step,
)


def test_alive_pipeline_is_only_observed():
    decision = decide_action(
        state="running",
        phase="stage1",
        pipeline_alive=True,
        failure_kind="unknown",
        recovery_count=3,
        maximum_recoveries=5,
    )
    assert decision.action == "observe"


def test_external_oom_is_restarted_without_consuming_stability_budget():
    decision = decide_action(
        state="failed",
        phase="stage1",
        pipeline_alive=False,
        failure_kind="oom",
        recovery_count=3,
        maximum_recoveries=5,
    )
    assert decision.action == "restart"
    assert decision.effective_recovery_count == 3


def test_stability_failure_accounts_for_the_next_recovery():
    decision = decide_action(
        state="failed",
        phase="stage1",
        pipeline_alive=False,
        failure_kind="stability",
        recovery_count=3,
        maximum_recoveries=5,
    )
    assert decision.action == "restart"
    assert decision.effective_recovery_count == 4


def test_recovery_budget_is_not_bypassed():
    decision = decide_action(
        state="failed",
        phase="stage1",
        pipeline_alive=False,
        failure_kind="stability",
        recovery_count=5,
        maximum_recoveries=5,
    )
    assert decision.action == "stop_terminal"
    assert decision.effective_recovery_count == 6


def test_unknown_failure_is_not_restarted_blindly():
    decision = decide_action(
        state="failed",
        phase="stage1",
        pipeline_alive=False,
        failure_kind="unknown",
        recovery_count=3,
        maximum_recoveries=5,
    )
    assert decision.action == "stop_terminal"


def test_latest_logged_step_reads_append_only_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        '{"event": "launch"}\n'
        '{"step": 2525, "loss": 0.1}\n'
        '{"step": 2550, "loss": 0.09}\n',
        encoding="utf-8",
    )
    assert latest_logged_step(log) == 2550
