from __future__ import annotations

from pathlib import Path

from r1_uav_nav.training.supervisor import (
    SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
    SupervisorConfig,
    SupervisorDecision,
    WorkerHeartbeat,
    WorkerPhase,
    decide_supervisor_action,
    read_heartbeat,
    run_fake_worker_smoke,
    write_heartbeat,
)


def _heartbeat(
    phase: WorkerPhase = WorkerPhase.TRAINING,
    *,
    safe_bundle: str | None = "bundle",
) -> WorkerHeartbeat:
    return WorkerHeartbeat(
        schema_version=SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
        run_id="run",
        worker_id="worker",
        pid=123,
        sequence=4,
        recorded_at="2026-07-26T00:00:00+00:00",
        phase=phase,
        global_timesteps=100,
        completed_episodes=2,
        latest_course={"profile_id": "easy", "base_seed": 1100},
        latest_safe_bundle=safe_bundle,
        cleanup_status=None,
        last_error=None,
    )


def test_heartbeat_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, _heartbeat())
    assert read_heartbeat(path) == _heartbeat()
    assert not tuple(tmp_path.glob("*.tmp-*"))


def test_stale_training_requires_termination_but_checkpoint_gets_grace() -> None:
    config = SupervisorConfig(stale_timeout_s=10.0, checkpoint_grace_s=30.0)
    assert (
        decide_supervisor_action(
            config=config,
            heartbeat=_heartbeat(),
            seconds_since_progress=11.0,
            worker_running=True,
        )
        is SupervisorDecision.STALE_TERMINATION_REQUIRED
    )
    assert (
        decide_supervisor_action(
            config=config,
            heartbeat=_heartbeat(WorkerPhase.CHECKPOINTING),
            seconds_since_progress=11.0,
            worker_running=True,
        )
        is SupervisorDecision.WAIT_CHECKPOINT_GRACE
    )


def test_restart_requires_bundle_termination_and_safe_survey() -> None:
    config = SupervisorConfig()
    assert (
        decide_supervisor_action(
            config=config,
            heartbeat=_heartbeat(safe_bundle=None),
            seconds_since_progress=70.0,
            worker_running=False,
        )
        is SupervisorDecision.BLOCKED_NO_SAFE_BUNDLE
    )
    assert (
        decide_supervisor_action(
            config=config,
            heartbeat=_heartbeat(),
            seconds_since_progress=70.0,
            worker_running=False,
            worker_termination_confirmed=True,
            simulator_safe_state_confirmed=False,
        )
        is SupervisorDecision.BLOCKED_UNSAFE_SIMULATOR
    )
    assert (
        decide_supervisor_action(
            config=config,
            heartbeat=_heartbeat(),
            seconds_since_progress=70.0,
            worker_running=False,
            worker_termination_confirmed=True,
            simulator_safe_state_confirmed=True,
        )
        is SupervisorDecision.RESTART_FROM_BUNDLE
    )


def test_spawned_fake_worker_emits_progress_and_exits_without_termination(
    tmp_path: Path,
) -> None:
    final = run_fake_worker_smoke(
        tmp_path / "heartbeat.json",
        heartbeat_count=2,
        interval_s=0.001,
    )
    assert final.phase is WorkerPhase.STOPPED
    assert final.sequence == 2
