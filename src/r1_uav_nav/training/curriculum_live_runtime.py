"""Lazily imported runtime for bounded M13.8 supervised live pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from r1_uav_nav.training.curriculum import CurriculumState
from r1_uav_nav.training.curriculum_evidence import TrajectorySummaryCollector
from r1_uav_nav.training.curriculum_live_pilot import (
    M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION,
    LivePilotReport,
    PreparedLivePilot,
    default_live_pilot_report_path,
    save_live_pilot_report,
)
from r1_uav_nav.training.long_run_state import (
    LongRunRunState,
    ResumeMode,
    SafeCheckpoint,
    load_rng_evidence,
    restore_rng_state,
    validate_checkpoint_bundle,
)
from r1_uav_nav.training.long_run_training import (
    SafeCheckpointCallback,
    create_td3_model,
)


@dataclass(frozen=True)
class PilotRuntimeEnvironment:
    """Injected environment plus cleanup and broad-reset evidence."""

    env: gym.Env
    cleanup: Callable[[], Any]
    broad_reset_evidence: Callable[[], Mapping[str, Any]]
    position_scales_m: tuple[float, float, float] = (20.0, 20.0, 8.0)


@dataclass(frozen=True)
class PilotRuntimeResult:
    """Runtime output returned after report persistence."""

    report: LivePilotReport
    report_path: Path


class _PilotProgressCallback(BaseCallback):
    def __init__(
        self,
        *,
        initial_global_timesteps: int,
        target_global_timesteps: int,
        cumulative_cap: int,
        state: CurriculumState,
    ) -> None:
        super().__init__()
        self.initial_global_timesteps = initial_global_timesteps
        self.target_global_timesteps = target_global_timesteps
        self.cumulative_cap = cumulative_cap
        self.initial_stage_steps = state.stage_completed_timesteps
        self.state = state
        self.overrun = False

    def _on_step(self) -> bool:
        segment_steps = max(0, int(self.num_timesteps) - self.initial_global_timesteps)
        cumulative = self.initial_stage_steps + segment_steps
        self.state = replace(self.state, stage_completed_timesteps=cumulative)
        if cumulative > self.cumulative_cap:
            self.overrun = True
            return False
        return int(self.num_timesteps) < self.target_global_timesteps

    def current_state(self) -> Mapping[str, Any]:
        return self.state.to_dict()


class _PilotMetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        stage_id: str,
        start_relative: tuple[float, float, float],
        goal_relative: tuple[float, float, float],
        position_scales_m: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.start_relative = start_relative
        self.goal_relative = goal_relative
        self.position_scales_m = position_scales_m
        self.episodes: list[dict[str, Any]] = []
        self.route_shapes: list[dict[str, Any]] = []
        self.safety_incident = False
        self._collector = self._new_collector()

    def _new_collector(self) -> TrajectorySummaryCollector | None:
        if self.stage_id != "stage-1":
            return None
        return TrajectorySummaryCollector(
            start_relative=self.start_relative,
            goal_relative=self.goal_relative,
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        observations = self.locals.get("new_obs")
        dones = self.locals.get("dones")
        if not infos or observations is None:
            return True
        info = dict(infos[0])
        observation = np.asarray(observations[0], dtype=np.float32)
        if self._collector is not None:
            self._collector.add(
                observation=observation,
                position_scales_m=self.position_scales_m,
                info=info,
            )
        collision = bool(info.get("collision", False))
        safety = bool(
            info.get("ground_clearance_violation", False)
            or info.get("workspace_violation", False)
            or info.get("sensor_failure", False)
        )
        self.safety_incident = self.safety_incident or collision or safety
        if dones is not None and bool(dones[0]):
            self.episodes.append(
                _episode_summary(info, collision=collision, safety=safety)
            )
            if self._collector is not None:
                self.route_shapes.append(
                    asdict(
                        self._collector.result(
                            successful=bool(info.get("success", False)),
                            collision=collision,
                            safety_violation=safety,
                        )
                    )
                )
            self._collector = self._new_collector()
        return True


def execute_live_pilot(
    prepared: PreparedLivePilot,
    *,
    repository_root: Path,
    environment_factory: (
        Callable[[PreparedLivePilot], PilotRuntimeEnvironment] | None
    ) = None,
    report_path: Path | None = None,
) -> PilotRuntimeResult:
    """Execute one bounded segment after pure preflight has already succeeded."""
    started = datetime.now(timezone.utc)
    runtime_environment: PilotRuntimeEnvironment | None = None
    cleanup: Any | None = None
    errors: list[str] = []
    interrupted = False
    initial_checkpoint: SafeCheckpoint | None = None
    final_checkpoint: SafeCheckpoint | None = None
    latest_checkpoint: SafeCheckpoint | None = None
    replay_size: int | None = None
    actual_segment = 0
    final_state = prepared.curriculum_state
    metrics: _PilotMetricsCallback | None = None
    progress: _PilotProgressCallback | None = None

    try:
        factory = environment_factory or _build_live_environment
        runtime_environment = factory(prepared)
        env = runtime_environment.env
        model, source_state, restored = _load_or_create_model(prepared, env)
        del restored  # Persisted by M13.7 checkpoint RNG evidence.
        initial_global = int(model.num_timesteps)
        target_global = initial_global + prepared.requested_timesteps
        progress = _PilotProgressCallback(
            initial_global_timesteps=initial_global,
            target_global_timesteps=target_global,
            cumulative_cap=prepared.metadata.cumulative_cap_steps,
            state=prepared.curriculum_state,
        )
        metrics = _PilotMetricsCallback(
            stage_id=prepared.stage.stage_id,
            start_relative=(0.0, 0.0, 0.0),
            goal_relative=_goal_relative(prepared),
            position_scales_m=runtime_environment.position_scales_m,
        )
        checkpoint = _checkpoint_callback(
            prepared,
            source_state=source_state,
            curriculum_state_provider=progress.current_state,
        )
        checkpoint.init_callback(model)
        initial_checkpoint = checkpoint.save_now()
        latest_checkpoint = initial_checkpoint

        callbacks = CallbackList([metrics, progress, checkpoint])
        model.learn(
            total_timesteps=prepared.requested_timesteps,
            callback=callbacks,
            reset_num_timesteps=(prepared.resume_plan.mode is not ResumeMode.FULL),
            tb_log_name=prepared.long_run_config.experiment_name,
        )
        actual_segment = max(0, int(model.num_timesteps) - initial_global)
        final_state = progress.state
        if (
            progress.overrun
            or final_state.stage_completed_timesteps
            > prepared.metadata.cumulative_cap_steps
        ):
            errors.append("TD3 timestep accounting exceeded the cumulative pilot cap")
        else:
            final_checkpoint = checkpoint.save_now()
            latest_checkpoint = final_checkpoint
            replay = getattr(model, "replay_buffer", None)
            replay_size = int(replay.size()) if replay is not None else 0
    except KeyboardInterrupt:
        interrupted = True
        errors.append("KeyboardInterrupt: operator interrupted the live pilot")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if progress is not None:
            final_state = progress.state
        if runtime_environment is not None:
            try:
                cleanup = runtime_environment.cleanup()
            except BaseException as exc:
                errors.append(f"cleanup {type(exc).__name__}: {exc}")

    broad_reset = (
        dict(runtime_environment.broad_reset_evidence())
        if runtime_environment is not None
        else {"guard_installed": False, "reset_attempted": False}
    )
    cleanup_mapping = _cleanup_mapping(cleanup)
    cleanup_success = bool(cleanup is not None and getattr(cleanup, "succeeded", False))
    checkpoint_success = bool(
        initial_checkpoint is not None
        and final_checkpoint is not None
        and _checkpoint_is_valid(initial_checkpoint)
        and _checkpoint_is_valid(final_checkpoint)
    )
    safety_incident_free = bool(
        metrics is not None
        and not metrics.safety_incident
        and not broad_reset.get("reset_attempted", False)
    )
    infrastructure_success = bool(
        initial_checkpoint is not None
        and actual_segment > 0
        and not interrupted
        and not errors
        and not broad_reset.get("reset_attempted", False)
    )
    checks = {
        "preflight_succeeded": True,
        "infrastructure_succeeded": infrastructure_success,
        "safety_incident_free": safety_incident_free,
        "checkpoints_succeeded": checkpoint_success,
        "named_cleanup_succeeded": cleanup_success,
        "broad_simulator_reset_not_used": not broad_reset.get("reset_attempted", False),
        "not_interrupted": not interrupted,
        "no_operational_errors": not errors,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    report = LivePilotReport(
        schema_version=M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION,
        run_id=prepared.run_id,
        started_at=started.isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        command_mode=(
            "resume-pilot"
            if prepared.resume_plan.mode is ResumeMode.FULL
            else "pilot-stage"
        ),
        stage_id=prepared.stage.stage_id,
        pilot_kind=prepared.pilot_kind.value,
        profile_id=prepared.course.result.profile_id,
        base_seed=prepared.course.result.base_seed,
        accepted_candidate_seed=prepared.course.result.accepted_candidate_seed,
        requested_timesteps=prepared.requested_timesteps,
        actual_segment_timesteps=actual_segment,
        cumulative_timesteps=final_state.stage_completed_timesteps,
        cumulative_cap_timesteps=prepared.metadata.cumulative_cap_steps,
        curriculum_config_digest=prepared.curriculum.config_digest,
        pilot_config_digest=prepared.config.config_digest,
        pilot_metadata_digest=prepared.metadata.metadata_digest,
        m13_6_suite_digest=prepared.m13_6_evidence.suite_digest,
        scene_digest=prepared.course.result.scene_digest,
        occupancy_digest=prepared.course.result.occupancy_digest,
        solvability_digest=prepared.course.result.solvability_digest,
        initial_checkpoint=_checkpoint_mapping(initial_checkpoint, repository_root),
        final_checkpoint=_checkpoint_mapping(final_checkpoint, repository_root),
        latest_safe_checkpoint=_checkpoint_mapping(latest_checkpoint, repository_root),
        replay_size=replay_size,
        curriculum_state=final_state.to_dict(),
        episode_summaries=tuple(metrics.episodes if metrics is not None else ()),
        route_shape_summaries=tuple(
            metrics.route_shapes if metrics is not None else ()
        ),
        broad_reset_evidence=broad_reset,
        cleanup_evidence=cleanup_mapping,
        preflight_success=True,
        infrastructure_success=infrastructure_success,
        safety_incident_free=safety_incident_free,
        checkpoint_success=checkpoint_success,
        cleanup_success=cleanup_success,
        report_success=all(checks.values()),
        interrupted=interrupted,
        pilot_only=True,
        promotion_claimed=False,
        learned_avoidance_claimed=False,
        final_policy_claimed=False,
        final_generalization_claimed=False,
        real_world_claimed=False,
        acceptance_checks=checks,
        acceptance_failures=failures,
        errors=tuple(errors),
        limitations=_limitations(),
    )
    destination = report_path or default_live_pilot_report_path(prepared)
    saved = save_live_pilot_report(
        report,
        destination,
        repository_root=repository_root,
        approved_root=repository_root / prepared.config.outputs.report_root,
    )
    return PilotRuntimeResult(report, saved)


def _load_or_create_model(
    prepared: PreparedLivePilot,
    env: gym.Env,
) -> tuple[TD3, LongRunRunState | None, dict[str, bool]]:
    plan = prepared.resume_plan
    if plan.mode is ResumeMode.NEW:
        return create_td3_model(prepared.long_run_config, env), None, {}
    if plan.mode is ResumeMode.MODEL_ONLY_WARM_START:
        model = TD3.load(
            plan.model_path,
            env=env,
            device=prepared.long_run_config.device,
            force_reset=True,
        )
        model.num_timesteps = 0
        return model, None, {}
    if plan.run_state_path is None or plan.replay_buffer_path is None:
        raise ValueError("full live-pilot resume plan is incomplete")
    safe = validate_checkpoint_bundle(
        plan.run_state_path.parent,
        expected_compatibility_digest=prepared.long_run_config.compatibility_digest,
    )
    model = TD3.load(
        plan.model_path,
        env=env,
        device=prepared.long_run_config.device,
        force_reset=True,
    )
    model.load_replay_buffer(plan.replay_buffer_path, truncate_last_traj=True)
    restored = restore_rng_state(
        load_rng_evidence(plan.run_state_path.parent / "rng_state.json"),
        env=env,
    )
    return model, safe.run_state, restored


def _checkpoint_callback(
    prepared: PreparedLivePilot,
    *,
    source_state: LongRunRunState | None,
    curriculum_state_provider: Callable[[], Mapping[str, Any]],
) -> SafeCheckpointCallback:
    return SafeCheckpointCallback(
        config=prepared.long_run_config,
        run_id=prepared.run_id,
        run_root=prepared.run_root,
        resolved_config=prepared.long_run_config.resolved_snapshot(),
        initial_sequence=source_state.checkpoint_sequence if source_state else 0,
        initial_completed_episodes=(
            source_state.completed_episodes if source_state else 0
        ),
        next_checkpoint_step=(
            int(source_state.global_timesteps)
            + prepared.long_run_config.checkpoint_interval_steps
            if source_state is not None
            else prepared.long_run_config.checkpoint_interval_steps
        ),
        parent_run_id=(
            prepared.resume_plan.source_run_id
            if prepared.resume_plan.mode is ResumeMode.MODEL_ONLY_WARM_START
            else source_state.parent_run_id if source_state else None
        ),
        warm_start_source_model_digest=(
            prepared.resume_plan.source_model_digest
            if prepared.resume_plan.mode is ResumeMode.MODEL_ONLY_WARM_START
            else (
                source_state.warm_start_source_model_digest
                if source_state is not None
                else None
            )
        ),
        course_pool=(
            {
                "profile_id": prepared.stage.profile_id,
                "base_seed": prepared.stage.base_seed,
            },
        ),
        curriculum_state_provider=curriculum_state_provider,
        worker_id="m13-8-supervised-pilot",
    )


def _build_live_environment(prepared: PreparedLivePilot) -> PilotRuntimeEnvironment:
    # This function is reached only after prepare_live_pilot has completed.
    from r1_uav_nav.envs.colosseum_obstacle_uav_env import (
        ColosseumObstacleUAVEnv,
        CourseSelectionMode,
        ObstacleCourseSelectionConfig,
        ObstacleRuntimeAuthorization,
        load_colosseum_obstacle_uav_env_config,
    )
    from r1_uav_nav.sim.colosseum_client import (
        create_multirotor_client,
        import_colosseum_client_module,
    )
    from r1_uav_nav.sim.lidar_live_validation import BroadResetGuard

    root = prepared.repository_root
    source = load_colosseum_obstacle_uav_env_config(
        root / prepared.config.obstacle_environment_config_path
    )
    module = import_colosseum_client_module(prepared.config.client_module)
    raw_client = create_multirotor_client(module)
    guarded = BroadResetGuard(raw_client)
    course = ObstacleCourseSelectionConfig(
        course_suite_path=prepared.curriculum.course_suites["m13_8"],
        asset_catalog_path=source.course.asset_catalog_path,
        lidar_config_path=source.course.lidar_config_path,
        mode=CourseSelectionMode.FIXED,
        fixed_profile_id=prepared.stage.profile_id,
        fixed_base_seed=prepared.stage.base_seed,
        seeded_profile_ids=(prepared.stage.profile_id,),
        allow_external_test_endpoints=False,
    )
    authorization = ObstacleRuntimeAuthorization(
        allow_live_rpc=True,
        allow_scene_mutation=True,
        confirm_scene_area_clear=True,
        confirm_no_visible_collision=True,
        allow_debug_markers=True,
        allow_marker_flush=True,
        allow_flight=True,
        allow_start_positioning=True,
        confirm_clear_airspace=True,
    )
    env = ColosseumObstacleUAVEnv(
        replace(
            source,
            course=course,
            authorization=authorization,
        ),
        client_factory=lambda: guarded,
        client_module=module,
        repository_root=root,
    )
    pinned = _AcceptedCourseWrapper(env, prepared.course)
    return PilotRuntimeEnvironment(
        env=pinned,
        cleanup=env.close_with_result,
        broad_reset_evidence=lambda: {
            "guard_installed": True,
            "reset_attempted": guarded.reset_attempted,
        },
        position_scales_m=(
            source.navigation.workspace_xy_limit,
            source.navigation.workspace_xy_limit,
            max(
                source.navigation.workspace_up_limit,
                source.navigation.workspace_down_limit,
            ),
        ),
    )


class _AcceptedCourseWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, course: Any) -> None:
        super().__init__(env)
        self.course = course

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise ValueError("live pilot owns the accepted reset options")
        return self.env.reset(
            seed=seed,
            options={"validated_course": self.course},
        )


def _goal_relative(prepared: PreparedLivePilot) -> tuple[float, float, float]:
    start = prepared.course.scene.start_anchor
    goal = prepared.course.scene.goal_approach
    return (goal.x - start.x, goal.y - start.y, goal.z - start.z)


def _episode_summary(
    info: Mapping[str, Any],
    *,
    collision: bool,
    safety: bool,
) -> dict[str, Any]:
    return {
        "profile_id": info.get("profile_id"),
        "base_seed": info.get("base_seed"),
        "accepted_candidate_seed": info.get("accepted_candidate_seed"),
        "step_count": info.get("step_count"),
        "success": bool(info.get("success", False)),
        "collision": collision,
        "safety_violation": safety,
        "sensor_failure": bool(info.get("sensor_failure", False)),
        "termination_reason": info.get("termination_reason"),
        "path_length_m": info.get("path_length_m"),
        "distance_to_goal_m": info.get("distance_to_goal"),
        "minimum_clearance_m": info.get("reward_clearance_m"),
    }


def _checkpoint_mapping(
    checkpoint: SafeCheckpoint | None,
    repository_root: Path,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    try:
        relative = checkpoint.directory.resolve().relative_to(repository_root.resolve())
    except ValueError:
        relative = Path("results") / "external-checkpoint"
    return {
        "relative_directory": relative.as_posix(),
        "global_timesteps": checkpoint.manifest.global_timesteps,
        "checkpoint_sequence": checkpoint.manifest.checkpoint_sequence,
        "compatibility_digest": checkpoint.manifest.compatibility_digest,
        "replay_buffer_size": checkpoint.manifest.replay_buffer_size,
    }


def _checkpoint_is_valid(checkpoint: SafeCheckpoint) -> bool:
    try:
        validate_checkpoint_bundle(checkpoint.directory)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _cleanup_mapping(cleanup: Any | None) -> dict[str, Any] | None:
    if cleanup is None:
        return None
    try:
        return _jsonable(asdict(cleanup))
    except TypeError:
        return {
            "succeeded": bool(getattr(cleanup, "succeeded", False)),
            "scene_cleanup_deferred": bool(
                getattr(cleanup, "scene_cleanup_deferred", False)
            ),
            "scene_cleanup_deferred_reason": getattr(
                cleanup, "scene_cleanup_deferred_reason", None
            ),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _limitations() -> tuple[str, ...]:
    return (
        "This is bounded supervised pilot infrastructure, not curriculum promotion.",
        "No learned obstacle-avoidance or final-generalization claim is made.",
        "M13.3 does not represent undocumented built-in Blocks geometry.",
        "LiDAR evidence depends on configured beam density and sector aggregation.",
        "No dynamic-obstacle, camera, SLAM, mapping, or real-world claim is made.",
    )


__all__ = [
    "PilotRuntimeEnvironment",
    "PilotRuntimeResult",
    "execute_live_pilot",
]
