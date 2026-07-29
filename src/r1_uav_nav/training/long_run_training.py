"""Offline-first M13.7 TD3 checkpoint and resume orchestration."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from r1_uav_nav.sim.static_course import load_course_suite_config
from r1_uav_nav.training.long_run_state import (
    LONG_RUN_STATE_SCHEMA_VERSION,
    LongRunRunState,
    ResumeMode,
    ResumePlan,
    SafeCheckpoint,
    canonical_digest,
    load_rng_evidence,
    restore_rng_state,
    save_checkpoint_bundle,
    utc_now,
)
from r1_uav_nav.training.supervisor import WorkerHeartbeat, write_heartbeat
from r1_uav_nav.training.validation_callback import (
    DEFAULT_MONITORING_CASES,
    DEFAULT_PROMOTION_CASES,
    ValidationCase,
)

LONG_RUN_CONFIG_SCHEMA_VERSION = 1
DEFAULT_LONG_RUN_CONFIG_PATH = Path("configs/training/m13_7_long_run_td3.yaml")


@dataclass(frozen=True)
class LongRunOutputConfig:
    """Ignored output roots for one long-run experiment."""

    model_root: str = "results/trained_models/m13_7_long_run_td3"
    log_root: str = "results/logs/m13_7_long_run_td3"
    report_root: str = "results/reports/m13/training"

    def __post_init__(self) -> None:
        for name in ("model_root", "log_root", "report_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be repository-relative")
            if not path.parts or path.parts[0] != "results":
                raise ValueError(f"{name} must remain under results/")


@dataclass(frozen=True)
class LongRunValidationConfig:
    """Deterministic monitoring and promotion schedules."""

    monitoring_interval_steps: int = 5_000
    promotion_interval_steps: int = 20_000
    episodes_per_case: int = 1
    monitoring_cases: tuple[ValidationCase, ...] = DEFAULT_MONITORING_CASES
    promotion_cases: tuple[ValidationCase, ...] = DEFAULT_PROMOTION_CASES
    cleanup_hard_gate: bool = True

    def __post_init__(self) -> None:
        for name in (
            "monitoring_interval_steps",
            "promotion_interval_steps",
            "episodes_per_case",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.cleanup_hard_gate:
            raise ValueError("M13.7 Phase A requires the cleanup hard gate")
        if not self.monitoring_cases or not self.promotion_cases:
            raise ValueError("validation case sets must not be empty")
        for name in ("monitoring_cases", "promotion_cases"):
            identities = [item.identity for item in getattr(self, name)]
            if len(identities) != len(set(identities)):
                raise ValueError(f"{name} must contain unique identities")


@dataclass(frozen=True)
class LongRunTD3Config:
    """Strict Phase A TD3 infrastructure configuration."""

    schema_version: int = LONG_RUN_CONFIG_SCHEMA_VERSION
    experiment_name: str = "m13_7_long_run_td3"
    algorithm: str = "td3"
    seed: int = 1370
    device: str = "cpu"
    policy: str = "MlpPolicy"
    additional_timesteps: int = 100
    learning_rate: float = 0.0003
    buffer_size: int = 10_000
    learning_starts: int = 20
    batch_size: int = 16
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_std: tuple[float, float, float] = (0.15, 0.15, 0.05)
    policy_hidden_layers: tuple[int, ...] = (256, 256)
    checkpoint_interval_steps: int = 1_000
    course_suite_path: str = "configs/planning/m13_3_voxel_astar.yaml"
    obstacle_environment_config_path: str = "configs/env/m13_5_obstacle_uav_env.yaml"
    training_profile_ids: tuple[str, ...] = ("easy", "medium", "hard")
    output: LongRunOutputConfig = field(default_factory=LongRunOutputConfig)
    validation: LongRunValidationConfig = field(default_factory=LongRunValidationConfig)

    def __post_init__(self) -> None:
        if self.schema_version != LONG_RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported M13.7 long-run configuration schema")
        if self.algorithm != "td3":
            raise ValueError("M13.7 Phase A supports only TD3")
        if (
            not self.experiment_name
            or self.experiment_name != self.experiment_name.strip()
        ):
            raise ValueError("experiment_name must not be empty or padded")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.policy != "MlpPolicy":
            raise ValueError("Phase A supports only MlpPolicy")
        for name in (
            "seed",
            "additional_timesteps",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "train_freq",
            "gradient_steps",
            "policy_delay",
            "checkpoint_interval_steps",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "learning_starts"} else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size must not exceed buffer_size")
        for name in (
            "learning_rate",
            "gamma",
            "tau",
            "target_policy_noise",
            "target_noise_clip",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.action_noise_std or any(
            not math.isfinite(value) or value < 0 for value in self.action_noise_std
        ):
            raise ValueError("action_noise_std must contain non-negative values")
        if len(self.action_noise_std) != 3:
            raise ValueError("action_noise_std must match the three-value action")
        if not self.policy_hidden_layers or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.policy_hidden_layers
        ):
            raise ValueError("policy_hidden_layers must contain positive integers")
        if not self.training_profile_ids or len(set(self.training_profile_ids)) != len(
            self.training_profile_ids
        ):
            raise ValueError("training_profile_ids must be non-empty and unique")
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in self.training_profile_ids
        ):
            raise ValueError("training_profile_ids must contain unpadded names")
        for name in (
            "course_suite_path",
            "obstacle_environment_config_path",
        ):
            value = getattr(self, name)
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a repository-relative path")

    @property
    def full_config_digest(self) -> str:
        return canonical_digest(asdict(self))

    @property
    def compatibility_digest(self) -> str:
        compatible = asdict(self)
        compatible.pop("additional_timesteps")
        compatible.pop("checkpoint_interval_steps")
        compatible.pop("output")
        validation = compatible["validation"]
        validation.pop("monitoring_interval_steps")
        validation.pop("promotion_interval_steps")
        return canonical_digest(compatible)

    def resolved_snapshot(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class LongRunTrainingResult:
    """Bounded Phase A training/resume result."""

    run_id: str
    resume_mode: ResumeMode
    initial_timesteps: int
    final_timesteps: int
    checkpoint: SafeCheckpoint
    restored_rng_sources: dict[str, bool]


class SafeCheckpointCallback(BaseCallback):
    """Save a complete bundle after SB3 stores a completed environment step."""

    def __init__(
        self,
        *,
        config: LongRunTD3Config,
        run_id: str,
        run_root: Path,
        resolved_config: Mapping[str, Any],
        initial_sequence: int = 0,
        initial_completed_episodes: int = 0,
        next_checkpoint_step: int | None = None,
        created_at: str | None = None,
        parent_run_id: str | None = None,
        warm_start_source_model_digest: str | None = None,
        course_pool: Sequence[Mapping[str, Any]] | None = None,
        heartbeat_path: Path | None = None,
        worker_id: str = "phase-a-worker",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.run_id = run_id
        self.run_root = run_root
        self.resolved_config = dict(resolved_config)
        self.sequence = initial_sequence
        self.completed_episodes = initial_completed_episodes
        self.next_checkpoint_step = (
            next_checkpoint_step
            if next_checkpoint_step is not None
            else config.checkpoint_interval_steps
        )
        self.heartbeat_path = heartbeat_path
        self.worker_id = worker_id
        self.created_at = created_at or utc_now()
        self.parent_run_id = parent_run_id
        self.warm_start_source_model_digest = warm_start_source_model_digest
        self.course_pool = tuple(dict(item) for item in (course_pool or ()))
        self._checkpoint_due = False
        self.latest_checkpoint: SafeCheckpoint | None = None
        self.last_info: dict[str, Any] = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if infos:
            self.last_info = dict(infos[0])
        if dones is not None and bool(dones[0]):
            self.completed_episodes += 1
        if self.num_timesteps >= self.next_checkpoint_step:
            self._checkpoint_due = True
        return True

    def _on_rollout_end(self) -> None:
        if not self._checkpoint_due:
            return
        self.save_now()
        while self.next_checkpoint_step <= self.num_timesteps:
            self.next_checkpoint_step += self.config.checkpoint_interval_steps
        self._checkpoint_due = False

    def save_now(self) -> SafeCheckpoint:
        """Save one immutable bundle at the current completed-step boundary."""
        self.sequence += 1
        replay = getattr(self.model, "replay_buffer", None)
        replay_size = int(replay.size()) if replay is not None else 0
        persisted_next_checkpoint_step = self.next_checkpoint_step
        while persisted_next_checkpoint_step <= self.model.num_timesteps:
            persisted_next_checkpoint_step += self.config.checkpoint_interval_steps
        state = LongRunRunState(
            schema_version=LONG_RUN_STATE_SCHEMA_VERSION,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            warm_start_source_model_digest=self.warm_start_source_model_digest,
            created_at=self.created_at,
            updated_at=utc_now(),
            global_timesteps=int(self.model.num_timesteps),
            update_count=int(getattr(self.model, "_n_updates", 0)),
            completed_episodes=self.completed_episodes,
            replay_buffer_size=replay_size,
            checkpoint_sequence=self.sequence,
            next_checkpoint_step=persisted_next_checkpoint_step,
            next_monitoring_step=self.config.validation.monitoring_interval_steps,
            next_promotion_step=self.config.validation.promotion_interval_steps,
            full_config_digest=self.config.full_config_digest,
            compatibility_digest=self.config.compatibility_digest,
            environment_config_digest=canonical_digest(
                self.config.obstacle_environment_config_path
            ),
            observation_signature=_space_signature(self.model.observation_space),
            action_signature=_space_signature(self.model.action_space),
            active_course=_sanitized_course(self.last_info),
            last_completed_course=(
                _sanitized_course(self.last_info)
                if self.last_info.get("termination_reason") is not None
                else None
            ),
            course_pool=(
                self.course_pool
                if self.course_pool
                else tuple(
                    {"profile_id": item}
                    for item in sorted(self.config.training_profile_ids)
                )
            ),
            course_pool_digest=canonical_digest(
                self.course_pool
                if self.course_pool
                else sorted(self.config.training_profile_ids)
            ),
            episode_in_progress=self.last_info.get("termination_reason") is None,
            deterministic_resume_guaranteed=False,
        )
        if self.heartbeat_path is not None:
            self._heartbeat("checkpointing", state)
        self.latest_checkpoint = save_checkpoint_bundle(
            model=self.model,
            run_state=state,
            resolved_config=self.resolved_config,
            run_root=self.run_root,
            env=self.training_env,
        )
        if self.heartbeat_path is not None:
            self._heartbeat("training", state)
        return self.latest_checkpoint

    def _heartbeat(self, phase: str, state: LongRunRunState) -> None:
        from r1_uav_nav.training.supervisor import (
            SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
            WorkerPhase,
        )

        write_heartbeat(
            self.heartbeat_path,
            WorkerHeartbeat(
                schema_version=SUPERVISOR_HEARTBEAT_SCHEMA_VERSION,
                run_id=self.run_id,
                worker_id=self.worker_id,
                pid=os.getpid(),
                sequence=self.sequence,
                recorded_at=utc_now(),
                phase=WorkerPhase(phase),
                global_timesteps=state.global_timesteps,
                completed_episodes=state.completed_episodes,
                latest_course=state.active_course,
                latest_safe_bundle=(
                    str(self.latest_checkpoint.directory)
                    if self.latest_checkpoint is not None
                    else None
                ),
                cleanup_status=None,
                last_error=None,
            ),
        )


def load_long_run_td3_config(path: Path) -> LongRunTD3Config:
    """Load and strictly validate the M13.7 YAML configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("M13.7 configuration must contain a mapping")
    allowed = {
        "schema_version",
        "experiment_name",
        "algorithm",
        "seed",
        "device",
        "policy",
        "additional_timesteps",
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "tau",
        "train_freq",
        "gradient_steps",
        "policy_delay",
        "target_policy_noise",
        "target_noise_clip",
        "action_noise_std",
        "policy_hidden_layers",
        "checkpoint_interval_steps",
        "course_suite_path",
        "obstacle_environment_config_path",
        "training_profile_ids",
        "output",
        "validation",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown M13.7 configuration fields: {sorted(unknown)}")
    output = _output_config(raw.get("output", {}))
    validation = _validation_config(raw.get("validation", {}))
    values = {
        key: value for key, value in raw.items() if key not in {"output", "validation"}
    }
    for name in ("action_noise_std", "policy_hidden_layers", "training_profile_ids"):
        if name in values:
            values[name] = tuple(values[name])
    return LongRunTD3Config(**values, output=output, validation=validation)


def validate_long_run_configuration(
    config: LongRunTD3Config,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate course pools and output roots without importing a simulator."""
    suite_path = repository_root / config.course_suite_path
    env_path = repository_root / config.obstacle_environment_config_path
    if not env_path.is_file():
        raise FileNotFoundError(f"M13.5 environment config not found: {env_path}")
    suite = load_course_suite_config(suite_path)
    training_pairs: list[tuple[str, int]] = []
    for profile_id in sorted(config.training_profile_ids):
        profile = suite.profile(profile_id)
        if profile.split.value != "training":
            raise ValueError(f"profile {profile_id!r} is not a training profile")
        training_pairs.extend((profile_id, seed) for seed in profile.base_seeds)
    declared = {
        (profile.profile_id, seed)
        for profile in suite.profiles
        for seed in profile.base_seeds
    }
    for case in (
        *config.validation.monitoring_cases,
        *config.validation.promotion_cases,
    ):
        if (case.profile_id, case.base_seed) not in declared:
            raise ValueError(
                f"undeclared validation course {case.profile_id}/{case.base_seed}"
            )
    for root in asdict(config.output).values():
        _require_results_path(repository_root, Path(root))
    return {
        "schema_version": config.schema_version,
        "configuration_digest": config.full_config_digest,
        "compatibility_digest": config.compatibility_digest,
        "training_pairs": tuple(training_pairs),
        "monitoring_cases": tuple(
            case.identity for case in config.validation.monitoring_cases
        ),
        "promotion_cases": tuple(
            case.identity for case in config.validation.promotion_cases
        ),
        "live_training_enabled": False,
    }


def run_fake_td3_smoke(
    config: LongRunTD3Config,
    *,
    repository_root: Path,
    resume_plan: ResumePlan | None = None,
    run_id: str | None = None,
    additional_timesteps: int | None = None,
    env_factory: Callable[[], gym.Env] | None = None,
) -> LongRunTrainingResult:
    """Run a tiny CPU-only TD3 checkpoint/resume cycle without AirSim."""
    timesteps = (
        config.additional_timesteps
        if additional_timesteps is None
        else additional_timesteps
    )
    if timesteps <= 0:
        raise ValueError("additional_timesteps must be positive")
    effective_config = replace(config, additional_timesteps=timesteps)
    validation_evidence = validate_long_run_configuration(
        effective_config, repository_root=repository_root
    )
    plan = resume_plan or ResumePlan(mode=ResumeMode.NEW)
    env = (env_factory or _TinyLongRunEnv)()
    resolved_config = effective_config.resolved_snapshot()
    restored: dict[str, bool] = {}
    initial_timesteps = 0
    source_state: LongRunRunState | None = None
    if plan.mode is ResumeMode.NEW:
        effective_run_id = run_id or f"m13-7-{uuid.uuid4().hex[:12]}"
        model = _new_td3(effective_config, env)
    elif plan.mode is ResumeMode.MODEL_ONLY_WARM_START:
        effective_run_id = run_id or f"m13-7-warm-{uuid.uuid4().hex[:10]}"
        model = TD3.load(
            plan.model_path,
            env=env,
            device="cpu",
            force_reset=True,
        )
    else:
        if plan.run_state_path is None or plan.replay_buffer_path is None:
            raise ValueError("full resume plan is incomplete")
        source_state = LongRunRunState.from_mapping(
            yaml.safe_load(plan.run_state_path.read_text(encoding="utf-8"))
        )
        effective_run_id = source_state.run_id
        model = TD3.load(
            plan.model_path,
            env=env,
            device="cpu",
            force_reset=True,
        )
        model.load_replay_buffer(
            plan.replay_buffer_path,
            truncate_last_traj=True,
        )
        restored = restore_rng_state(
            load_rng_evidence(plan.run_state_path.parent / "rng_state.json"),
            env=env,
        )
        initial_timesteps = int(model.num_timesteps)
    run_root = repository_root / effective_config.output.model_root / effective_run_id
    initial_sequence = source_state.checkpoint_sequence if source_state else 0
    callback = SafeCheckpointCallback(
        config=effective_config,
        run_id=effective_run_id,
        run_root=run_root,
        resolved_config=resolved_config,
        initial_sequence=initial_sequence,
        initial_completed_episodes=(
            source_state.completed_episodes if source_state else 0
        ),
        parent_run_id=(
            plan.source_run_id
            if plan.mode is ResumeMode.MODEL_ONLY_WARM_START
            else source_state.parent_run_id if source_state else None
        ),
        warm_start_source_model_digest=(
            plan.source_model_digest
            if plan.mode is ResumeMode.MODEL_ONLY_WARM_START
            else source_state.warm_start_source_model_digest if source_state else None
        ),
        course_pool=tuple(
            {"profile_id": profile_id, "base_seed": base_seed}
            for profile_id, base_seed in validation_evidence["training_pairs"]
        ),
        next_checkpoint_step=(
            initial_timesteps + effective_config.checkpoint_interval_steps
        ),
    )
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=(plan.mode is not ResumeMode.FULL),
            tb_log_name=effective_config.experiment_name,
        )
        if callback.latest_checkpoint is None:
            callback.save_now()
        assert callback.latest_checkpoint is not None
        return LongRunTrainingResult(
            run_id=effective_run_id,
            resume_mode=plan.mode,
            initial_timesteps=initial_timesteps,
            final_timesteps=int(model.num_timesteps),
            checkpoint=callback.latest_checkpoint,
            restored_rng_sources=restored,
        )
    finally:
        env.close()


def _new_td3(config: LongRunTD3Config, env: gym.Env) -> TD3:
    action_noise = NormalActionNoise(
        mean=np.zeros(3, dtype=np.float32),
        sigma=np.asarray(config.action_noise_std, dtype=np.float32),
    )
    return TD3(
        config.policy,
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        policy_delay=config.policy_delay,
        target_policy_noise=config.target_policy_noise,
        target_noise_clip=config.target_noise_clip,
        action_noise=action_noise,
        policy_kwargs={"net_arch": list(config.policy_hidden_layers)},
        replay_buffer_kwargs={"handle_timeout_termination": True},
        seed=config.seed,
        device="cpu",
        verbose=0,
    )


class _TinyLongRunEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.full(10, -1.0, dtype=np.float32),
                    np.zeros(73, dtype=np.float32),
                )
            ),
            high=np.ones(83, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step = 0
        observation = np.zeros(83, dtype=np.float32)
        observation[10:83] = 1.0
        return observation, {
            "profile_id": "fake",
            "base_seed": 0,
            "accepted_candidate_seed": 0,
            "scene_digest": "fake",
            "termination_reason": None,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("fake smoke action must be finite with shape (3,)")
        self._step += 1
        terminated = self._step >= 4
        observation = np.zeros(83, dtype=np.float32)
        observation[:3] = np.clip(action, -1.0, 1.0)
        observation[10:83] = 1.0
        return (
            observation,
            float(-np.linalg.norm(action)),
            terminated,
            False,
            {
                "profile_id": "fake",
                "base_seed": 0,
                "accepted_candidate_seed": 0,
                "scene_digest": "fake",
                "termination_reason": "goal_reached" if terminated else None,
            },
        )


def _output_config(raw: Any) -> LongRunOutputConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("output configuration must be a mapping")
    allowed = {"model_root", "log_root", "report_root"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown output fields: {sorted(unknown)}")
    return LongRunOutputConfig(**raw)


def _validation_config(raw: Any) -> LongRunValidationConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("validation configuration must be a mapping")
    allowed = {
        "monitoring_interval_steps",
        "promotion_interval_steps",
        "episodes_per_case",
        "monitoring_cases",
        "promotion_cases",
        "cleanup_hard_gate",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown validation fields: {sorted(unknown)}")
    values = dict(raw)
    for name in ("monitoring_cases", "promotion_cases"):
        if name in values:
            entries = values[name]
            if not isinstance(entries, list):
                raise ValueError(f"{name} must be a list")
            values[name] = tuple(ValidationCase(**entry) for entry in entries)
    return LongRunValidationConfig(**values)


def _require_results_path(repository_root: Path, relative: Path) -> None:
    root = repository_root.resolve()
    target = (root / relative).resolve()
    results = (root / "results").resolve()
    try:
        target.relative_to(results)
    except ValueError as exc:
        raise ValueError("M13.7 output path must remain under results/") from exc


def _space_signature(space: spaces.Space[Any]) -> str:
    if not isinstance(space, spaces.Box):
        raise ValueError("M13.7 supports Box observation and action spaces")
    return canonical_digest(
        {
            "type": "Box",
            "shape": space.shape,
            "dtype": str(space.dtype),
            "low": np.asarray(space.low).tolist(),
            "high": np.asarray(space.high).tolist(),
        }
    )


def _sanitized_course(info: Mapping[str, Any]) -> dict[str, Any] | None:
    keys = (
        "profile_id",
        "base_seed",
        "accepted_candidate_seed",
        "attempt_index",
        "scene_digest",
    )
    evidence = {key: info[key] for key in keys if key in info}
    return evidence or None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "DEFAULT_LONG_RUN_CONFIG_PATH",
    "LongRunOutputConfig",
    "LongRunTD3Config",
    "LongRunTrainingResult",
    "LongRunValidationConfig",
    "SafeCheckpointCallback",
    "load_long_run_td3_config",
    "run_fake_td3_smoke",
    "validate_long_run_configuration",
]
