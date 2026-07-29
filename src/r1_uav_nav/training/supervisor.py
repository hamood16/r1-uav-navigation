"""Offline M13.7 heartbeat protocol and recovery decision engine."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from r1_uav_nav.training.long_run_state import utc_now

SUPERVISOR_HEARTBEAT_SCHEMA_VERSION = 1
SUPERVISOR_EVENT_SCHEMA_VERSION = 1


class WorkerPhase(str, Enum):
    """Training worker phases reported through heartbeat progress."""

    STARTING = "starting"
    RESETTING = "resetting"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    VALIDATION = "validation"
    CLEANUP = "cleanup"
    STOPPED = "stopped"
    ERROR = "error"


class SupervisorDecision(str, Enum):
    """Pure supervisor outcomes; Phase A never terminates a process."""

    CONTINUE = "continue"
    WAIT_CHECKPOINT_GRACE = "wait_checkpoint_grace"
    STALE_TERMINATION_REQUIRED = "stale_termination_required"
    BLOCKED_NO_SAFE_BUNDLE = "blocked_no_safe_bundle"
    BLOCKED_UNSAFE_SIMULATOR = "blocked_unsafe_simulator"
    RESTART_FROM_BUNDLE = "restart_from_bundle"
    COMPLETE = "complete"


@dataclass(frozen=True)
class SupervisorConfig:
    """Heartbeat age and future restart-budget policy."""

    heartbeat_interval_s: float = 5.0
    stale_timeout_s: float = 60.0
    checkpoint_grace_s: float = 180.0
    maximum_restart_attempts: int = 2
    restart_backoff_s: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "heartbeat_interval_s",
            "stale_timeout_s",
            "checkpoint_grace_s",
            "restart_backoff_s",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.checkpoint_grace_s < self.stale_timeout_s:
            raise ValueError("checkpoint grace must not be shorter than stale timeout")
        if (
            not isinstance(self.maximum_restart_attempts, int)
            or isinstance(self.maximum_restart_attempts, bool)
            or self.maximum_restart_attempts < 0
        ):
            raise ValueError("maximum_restart_attempts must be non-negative")


@dataclass(frozen=True)
class WorkerHeartbeat:
    """Atomic worker progress evidence."""

    schema_version: int
    run_id: str
    worker_id: str
    pid: int
    sequence: int
    recorded_at: str
    phase: WorkerPhase
    global_timesteps: int
    completed_episodes: int
    latest_course: dict[str, Any] | None
    latest_safe_bundle: str | None
    cleanup_status: str | None
    last_error: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SUPERVISOR_HEARTBEAT_SCHEMA_VERSION:
            raise ValueError("unsupported heartbeat schema")
        if not self.run_id or not self.worker_id:
            raise ValueError("heartbeat run and worker IDs must not be empty")
        for name in ("pid", "sequence", "global_timesteps", "completed_episodes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> WorkerHeartbeat:
        values = dict(raw)
        values["phase"] = WorkerPhase(values["phase"])
        return cls(**values)


@dataclass(frozen=True)
class SupervisorEvent:
    """One recorded supervisor observation or decision."""

    schema_version: int
    recorded_at: str
    decision: SupervisorDecision
    worker_id: str | None
    restart_attempt: int
    reason: str
    latest_safe_bundle: str | None


def write_heartbeat(path: Path, heartbeat: WorkerHeartbeat) -> None:
    """Atomically replace the current heartbeat."""
    _atomic_json(path, asdict(heartbeat))


def read_heartbeat(path: Path) -> WorkerHeartbeat:
    """Load and validate heartbeat evidence."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("heartbeat must contain a JSON object")
    return WorkerHeartbeat.from_mapping(raw)


def decide_supervisor_action(
    *,
    config: SupervisorConfig,
    heartbeat: WorkerHeartbeat | None,
    seconds_since_progress: float,
    worker_running: bool,
    restart_attempt: int = 0,
    worker_termination_confirmed: bool = False,
    simulator_safe_state_confirmed: bool = False,
) -> SupervisorDecision:
    """Choose a recovery action without mutating worker or simulator state."""
    if not math.isfinite(seconds_since_progress) or seconds_since_progress < 0:
        raise ValueError("seconds_since_progress must be finite and non-negative")
    if restart_attempt < 0:
        raise ValueError("restart_attempt must be non-negative")
    if heartbeat is not None and heartbeat.phase is WorkerPhase.STOPPED:
        return SupervisorDecision.COMPLETE
    if worker_running:
        timeout = (
            config.checkpoint_grace_s
            if heartbeat is not None and heartbeat.phase is WorkerPhase.CHECKPOINTING
            else config.stale_timeout_s
        )
        if seconds_since_progress <= timeout:
            if (
                heartbeat is not None
                and heartbeat.phase is WorkerPhase.CHECKPOINTING
                and seconds_since_progress > config.stale_timeout_s
            ):
                return SupervisorDecision.WAIT_CHECKPOINT_GRACE
            return SupervisorDecision.CONTINUE
        return SupervisorDecision.STALE_TERMINATION_REQUIRED
    if heartbeat is None or not heartbeat.latest_safe_bundle:
        return SupervisorDecision.BLOCKED_NO_SAFE_BUNDLE
    if restart_attempt >= config.maximum_restart_attempts:
        return SupervisorDecision.BLOCKED_UNSAFE_SIMULATOR
    if not worker_termination_confirmed or not simulator_safe_state_confirmed:
        return SupervisorDecision.BLOCKED_UNSAFE_SIMULATOR
    return SupervisorDecision.RESTART_FROM_BUNDLE


def supervisor_event(
    *,
    decision: SupervisorDecision,
    heartbeat: WorkerHeartbeat | None,
    restart_attempt: int,
    reason: str,
) -> SupervisorEvent:
    """Build stable supervisor report evidence."""
    return SupervisorEvent(
        schema_version=SUPERVISOR_EVENT_SCHEMA_VERSION,
        recorded_at=utc_now(),
        decision=decision,
        worker_id=heartbeat.worker_id if heartbeat is not None else None,
        restart_attempt=restart_attempt,
        reason=reason,
        latest_safe_bundle=(
            heartbeat.latest_safe_bundle if heartbeat is not None else None
        ),
    )


def run_fake_worker_smoke(
    heartbeat_path: Path,
    *,
    heartbeat_count: int = 3,
    interval_s: float = 0.02,
    join_timeout_s: float = 10.0,
) -> WorkerHeartbeat:
    """Spawn one non-simulator worker and observe its normal completion."""
    if heartbeat_count <= 0:
        raise ValueError("heartbeat_count must be positive")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_fake_worker,
        args=(heartbeat_path, heartbeat_count, interval_s),
    )
    process.start()
    process.join(join_timeout_s)
    if process.is_alive():
        raise TimeoutError(
            "fake worker did not exit; Phase A supervisor does not terminate processes"
        )
    if process.exitcode != 0:
        raise RuntimeError(f"fake worker exited with code {process.exitcode}")
    final = read_heartbeat(heartbeat_path)
    if final.phase is not WorkerPhase.STOPPED:
        raise RuntimeError("fake worker did not emit a stopped heartbeat")
    return final


def _fake_worker(path: Path, count: int, interval_s: float) -> None:
    run_id = f"fake-{uuid.uuid4().hex[:8]}"
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    for index in range(count):
        phase = WorkerPhase.STARTING if index == 0 else WorkerPhase.TRAINING
        write_heartbeat(
            path,
            WorkerHeartbeat(
                schema_version=SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
                run_id=run_id,
                worker_id=worker_id,
                pid=os.getpid(),
                sequence=index,
                recorded_at=utc_now(),
                phase=phase,
                global_timesteps=max(0, index - 1),
                completed_episodes=0,
                latest_course=None,
                latest_safe_bundle=None,
                cleanup_status=None,
                last_error=None,
            ),
        )
        time.sleep(interval_s)
    write_heartbeat(
        path,
        WorkerHeartbeat(
            schema_version=SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
            run_id=run_id,
            worker_id=worker_id,
            pid=os.getpid(),
            sequence=count,
            recorded_at=utc_now(),
            phase=WorkerPhase.STOPPED,
            global_timesteps=max(0, count - 1),
            completed_episodes=0,
            latest_course=None,
            latest_safe_bundle=None,
            cleanup_status="not_applicable",
            last_error=None,
        ),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "SupervisorConfig",
    "SupervisorDecision",
    "SupervisorEvent",
    "WorkerHeartbeat",
    "WorkerPhase",
    "decide_supervisor_action",
    "read_heartbeat",
    "run_fake_worker_smoke",
    "supervisor_event",
    "write_heartbeat",
]
