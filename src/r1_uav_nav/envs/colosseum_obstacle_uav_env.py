"""Composition-based M13.5 obstacle-aware Colosseum environment."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from r1_uav_nav.envs.colosseum_lidar_uav_env import (
    ColosseumLidarUAVEnv,
    ColosseumLidarUAVEnvConfig,
)
from r1_uav_nav.envs.colosseum_uav_env import ColosseumUAVEnvConfig
from r1_uav_nav.envs.obstacle_reward import (
    ClearanceEvidence,
    ObstacleRewardBreakdown,
    ObstacleRewardConfig,
    calculate_clearance_evidence,
    calculate_obstacle_reward,
)
from r1_uav_nav.sim.colosseum_client import (
    CleanupResult,
    confirm_connection,
    create_multirotor_client,
    import_colosseum_client_module,
)
from r1_uav_nav.sim.colosseum_scene import (
    ColosseumSceneManager,
    MaterializationConfig,
    MaterializedScene,
    SceneCleanupResult,
    cleanup_scene_resources,
)
from r1_uav_nav.sim.lidar_features import (
    LIDAR_OBSERVATION_SIZE,
    LidarFeatureConfig,
    load_lidar_feature_config,
)
from r1_uav_nav.sim.scene_specification import (
    AssetCatalog,
    Bounds3D,
    Vector3,
    load_asset_catalog,
)
from r1_uav_nav.sim.static_course import (
    CourseSplit,
    CourseSuiteConfig,
    ValidatedCourse,
    generate_solvable_course,
    load_course_suite_config,
    require_solvable_course,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D

M13_5_OBSTACLE_ENV_CONFIG_SCHEMA_VERSION = 1
M13_5_OBSTACLE_INFO_SCHEMA_VERSION = 1

_REASON_COLLISION = "collision"
_REASON_GROUND = "ground_clearance_violation"
_REASON_WORKSPACE = "workspace_violation"
_REASON_GOAL = "goal_reached"
_REASON_MAX_STEPS = "max_steps"
_REASON_WATCHDOG = "watchdog_timeout"
_REASON_SENSOR = "sensor_failure"
_REASON_RPC = "rpc_recovery"


class CourseSelectionMode(str, Enum):
    """Supported deterministic course-selection policies."""

    FIXED = "fixed"
    SEEDED = "seeded"


class ObstacleEnvironmentError(RuntimeError):
    """Base error for M13.5 environment lifecycle failures."""


class ObstacleResetError(ObstacleEnvironmentError):
    """Raised after an episode reset fails and cleanup is attempted."""

    def __init__(
        self,
        message: str,
        *,
        operational_error: BaseException,
        cleanup_result: ObstacleEnvironmentCleanupResult | None,
    ) -> None:
        self.operational_error = operational_error
        self.cleanup_result = cleanup_result
        super().__init__(message)


@dataclass(frozen=True)
class ObstacleCourseSelectionConfig:
    """Course registry, deterministic selection, and resource paths."""

    course_suite_path: str = "configs/planning/m13_3_voxel_astar.yaml"
    asset_catalog_path: str = "configs/scenes/m13_2_assets.yaml"
    lidar_config_path: str = "configs/sensing/m13_4_lidar_features.yaml"
    mode: CourseSelectionMode = CourseSelectionMode.FIXED
    fixed_profile_id: str = "easy"
    fixed_base_seed: int = 1100
    seeded_profile_ids: tuple[str, ...] = ("easy", "medium", "hard")
    allow_external_test_endpoints: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CourseSelectionMode):
            raise ValueError("mode must be a CourseSelectionMode")
        for name in (
            "course_suite_path",
            "asset_catalog_path",
            "lidar_config_path",
            "fixed_profile_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            not isinstance(self.fixed_base_seed, int)
            or isinstance(self.fixed_base_seed, bool)
            or self.fixed_base_seed < 0
        ):
            raise ValueError("fixed_base_seed must be a non-negative integer")
        if not self.seeded_profile_ids or any(
            not isinstance(value, str) or not value.strip()
            for value in self.seeded_profile_ids
        ):
            raise ValueError("seeded_profile_ids must contain non-empty names")
        if len(set(self.seeded_profile_ids)) != len(self.seeded_profile_ids):
            raise ValueError("seeded_profile_ids must be unique")
        if not isinstance(self.allow_external_test_endpoints, bool):
            raise ValueError("allow_external_test_endpoints must be boolean")


@dataclass(frozen=True)
class ObstacleRuntimeAuthorization:
    """Explicit live authorizations checked before simulator import."""

    allow_live_rpc: bool = False
    allow_scene_mutation: bool = False
    confirm_scene_area_clear: bool = False
    confirm_no_visible_collision: bool = False
    allow_debug_markers: bool = False
    allow_marker_flush: bool = False
    allow_flight: bool = False
    allow_start_positioning: bool = False
    confirm_clear_airspace: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in asdict(self).values()):
            raise ValueError("runtime authorization values must be boolean")

    @property
    def complete(self) -> bool:
        return all(asdict(self).values())


@dataclass(frozen=True)
class ColosseumObstacleUAVEnvConfig:
    """Versioned M13.5 environment configuration."""

    schema_version: int = M13_5_OBSTACLE_ENV_CONFIG_SCHEMA_VERSION
    navigation: ColosseumUAVEnvConfig = field(
        default_factory=lambda: ColosseumUAVEnvConfig(max_episode_steps=200)
    )
    reward: ObstacleRewardConfig = field(default_factory=ObstacleRewardConfig)
    course: ObstacleCourseSelectionConfig = field(
        default_factory=ObstacleCourseSelectionConfig
    )
    authorization: ObstacleRuntimeAuthorization = field(
        default_factory=ObstacleRuntimeAuthorization
    )
    max_episode_steps: int = 200
    watchdog_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.schema_version != M13_5_OBSTACLE_ENV_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported M13.5 obstacle environment schema")
        if not isinstance(self.navigation, ColosseumUAVEnvConfig):
            raise ValueError("navigation must be a ColosseumUAVEnvConfig")
        if not isinstance(self.reward, ObstacleRewardConfig):
            raise ValueError("reward must be an ObstacleRewardConfig")
        if not isinstance(self.course, ObstacleCourseSelectionConfig):
            raise ValueError("course must be an ObstacleCourseSelectionConfig")
        if not isinstance(self.authorization, ObstacleRuntimeAuthorization):
            raise ValueError("authorization must be an ObstacleRuntimeAuthorization")
        _validate_navigation_config(self.navigation)
        if (
            not isinstance(self.max_episode_steps, int)
            or isinstance(self.max_episode_steps, bool)
            or self.max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if (
            not isinstance(self.watchdog_timeout_s, (int, float))
            or isinstance(self.watchdog_timeout_s, bool)
            or not math.isfinite(float(self.watchdog_timeout_s))
            or self.watchdog_timeout_s <= 0
        ):
            raise ValueError("watchdog_timeout_s must be finite and positive")


@dataclass(frozen=True)
class ObstacleEpisodeResetOptions:
    """Optional accepted course or explicitly gated fake endpoint episode."""

    validated_course: ValidatedCourse | None = None
    start_anchor: Position3D | None = None
    goal_approach: Position3D | None = None
    workspace_bounds: Bounds3D | None = None
    scene_id: str = "external-test"
    scene_digest: str = "external-test"
    profile_id: str = "external-test"
    base_seed: int = 0
    accepted_candidate_seed: int = 0
    attempt_index: int = 0
    start_id: str = "start"
    goal_id: str = "goal"

    def __post_init__(self) -> None:
        external = (self.start_anchor, self.goal_approach, self.workspace_bounds)
        supplied = sum(item is not None for item in external)
        if supplied not in {0, 3}:
            raise ValueError(
                "external start, goal, and workspace must be provided together"
            )
        if self.validated_course is not None and supplied:
            raise ValueError(
                "validated_course and external endpoint evidence are mutually exclusive"
            )

    @property
    def uses_external_endpoints(self) -> bool:
        return self.start_anchor is not None


@dataclass(frozen=True)
class ObstacleEnvironmentCleanupResult:
    """Independent UAV and scene cleanup evidence."""

    uav_cleanup: CleanupResult | None
    uav_lifecycle_evidence: dict[str, Any] | None
    scene_cleanup: tuple[SceneCleanupResult, ...]
    scene_cleanup_deferred: bool
    scene_cleanup_deferred_reason: str | None
    errors: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        uav_ok = (
            self.uav_cleanup is None or not self.uav_cleanup.safety_critical_failure
        )
        return (
            uav_ok
            and not self.scene_cleanup_deferred
            and all(result.succeeded for result in self.scene_cleanup)
            and not self.errors
        )

    @property
    def landing_success(self) -> bool | None:
        evidence = self.uav_lifecycle_evidence
        if not evidence:
            return None
        attempts = tuple(evidence.get("cleanup_attempts", ()))
        if not attempts:
            return None
        return bool(attempts[-1].get("landing_confirmed", False))


@dataclass(frozen=True)
class _EpisodeContext:
    scene_id: str
    scene_digest: str
    profile_id: str
    base_seed: int
    accepted_candidate_seed: int
    attempt_index: int
    start_id: str
    goal_id: str
    start_anchor: Position3D
    goal_approach: Position3D
    workspace: Bounds3D


class ColosseumObstacleUAVEnv(gym.Env[np.ndarray, np.ndarray]):
    """Obstacle-aware wrapper around the validated M13.4 LiDAR environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ColosseumObstacleUAVEnvConfig | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
        client_module: ModuleType | None = None,
        client_module_loader: Callable[[str], ModuleType] | None = None,
        scene_manager_factory: (
            Callable[[Any, ModuleType, AssetCatalog], ColosseumSceneManager] | None
        ) = None,
        inner_environment_factory: (
            Callable[[ColosseumLidarUAVEnvConfig, Any], ColosseumLidarUAVEnv] | None
        ) = None,
        repository_root: str | Path | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.obstacle_config = config or ColosseumObstacleUAVEnvConfig()
        self.client_factory = client_factory
        self.client_module = client_module
        self.client_module_loader = (
            client_module_loader or import_colosseum_client_module
        )
        self.scene_manager_factory = scene_manager_factory
        self.inner_environment_factory = inner_environment_factory
        self.repository_root = Path(repository_root or Path.cwd()).resolve()
        self.monotonic_fn = monotonic_fn
        self.sleep_fn = sleep_fn

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.full(10, -1.0, dtype=np.float32),
                    np.zeros(73, dtype=np.float32),
                )
            ),
            high=np.ones(LIDAR_OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )

        self.client: Any | None = None
        self.scene_manager: ColosseumSceneManager | None = None
        self.inner_env: ColosseumLidarUAVEnv | None = None
        self.lidar_config: LidarFeatureConfig | None = None
        self.episode_context: _EpisodeContext | None = None
        self.last_observation: np.ndarray | None = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_distance_to_goal: float | None = None
        self.previous_position: Position3D | None = None
        self.last_valid_clearance_m: float | None = None
        self.path_length_m = 0.0
        self.step_count = 0
        self.safety_override_count = 0
        self.episode_started_at: float | None = None
        self.has_reset = False
        self.episode_complete = False
        self.closed = False
        self.last_cleanup_result: ObstacleEnvironmentCleanupResult | None = None
        self.prior_episode_cleanup: ObstacleEnvironmentCleanupResult | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select a course and start one named obstacle-aware episode."""
        if self.closed:
            raise RuntimeError("Cannot reset a closed ColosseumObstacleUAVEnv.")
        gym.Env.reset(self, seed=seed)
        reset_options = _coerce_reset_options(options)
        prepared_course = self._resolve_course(reset_options)
        self._validate_pre_client_mode(reset_options)

        self.prior_episode_cleanup = self._close_inner_environment()
        if (
            self.prior_episode_cleanup is not None
            and not self.prior_episode_cleanup.succeeded
        ):
            raise ObstacleResetError(
                "Previous obstacle episode cleanup failed.",
                operational_error=ObstacleEnvironmentError(
                    "previous named cleanup was not safe"
                ),
                cleanup_result=self.prior_episode_cleanup,
            )

        try:
            context = self._prepare_episode_context(
                reset_options,
                prepared_course,
            )
            inner = self._create_inner_environment(context)
            self.inner_env = inner
            observation, inner_info = inner.reset(
                seed=seed,
                options={
                    "start_anchor": _position_tuple(context.start_anchor),
                    "goal_approach": _position_tuple(context.goal_approach),
                },
            )
            observation = self._validate_observation(observation)
            lidar_config = self._require_lidar_config()
            clearance = calculate_clearance_evidence(
                observation[10:82],
                lidar_valid=bool(observation[82] == 1.0),
                lidar_config=lidar_config,
                reward_config=self.obstacle_config.reward,
                last_valid_clearance_m=None,
            )
            if clearance.measured_clearance_m is None:
                raise ObstacleEnvironmentError(
                    "reset did not establish valid LiDAR clearance"
                )
            self.last_valid_clearance_m = clearance.measured_clearance_m
            current_position = _position_from_info(inner_info)
            distance = _finite_info_float(inner_info, "distance_to_goal")
            self.episode_context = context
            self.last_observation = observation.copy()
            self.previous_action = np.zeros(3, dtype=np.float32)
            self.previous_distance_to_goal = distance
            self.previous_position = current_position
            self.path_length_m = 0.0
            self.step_count = 0
            self.safety_override_count = 0
            self.episode_started_at = self.monotonic_fn()
            self.has_reset = True
            self.episode_complete = False
            info = self._build_info(
                inner_info,
                previous_distance=distance,
                distance=distance,
                progress=0.0,
                clearance=clearance,
                action=np.zeros(3, dtype=np.float32),
                previous_action=np.zeros(3, dtype=np.float32),
                action_change=0.0,
                reward=None,
                terminated=False,
                truncated=False,
                termination_reason=None,
            )
            return observation, info
        except BaseException as exc:
            cleanup = self._cleanup_after_reset_failure()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ObstacleResetError(
                f"Obstacle environment reset failed: {type(exc).__name__}: {exc}",
                operational_error=exc,
                cleanup_result=cleanup,
            ) from exc

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Forward a world-NED velocity action and calculate the M13.5 reward."""
        self._validate_step_allowed()
        action_array = _validated_action(action)
        inner = self._require_inner()
        previous_distance = self._require_previous_distance()
        previous_action = self.previous_action.copy()

        try:
            observation, _, inner_terminated, inner_truncated, inner_info = inner.step(
                action_array
            )
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError, AssertionError)):
                raise
            return self._recover_rpc_step(exc, action_array, previous_action)

        observation = self._validate_observation(observation)
        distance = _finite_info_float(inner_info, "distance_to_goal")
        current_position = _position_from_info(inner_info)
        self.path_length_m += _distance(
            current_position,
            self._require_previous_position(),
        )
        self.step_count += 1

        clearance = calculate_clearance_evidence(
            observation[10:82],
            lidar_valid=bool(observation[82] == 1.0),
            lidar_config=self._require_lidar_config(),
            reward_config=self.obstacle_config.reward,
            last_valid_clearance_m=self.last_valid_clearance_m,
        )
        if clearance.measured_clearance_m is not None:
            self.last_valid_clearance_m = clearance.measured_clearance_m

        collision = bool(inner_info.get("collision", False))
        ground_violation = (
            _finite_info_float(inner_info, "ground_clearance")
            < self.obstacle_config.navigation.min_ground_clearance
        )
        workspace_violation = not _inside_bounds(
            current_position,
            self._require_context().workspace,
        )
        goal_success = distance <= self.obstacle_config.navigation.goal_tolerance
        termination_reason = _primary_terminal_reason(
            collision,
            ground_violation,
            workspace_violation,
            goal_success,
        )
        terminated = termination_reason is not None
        truncated = bool(inner_truncated)
        if not terminated and bool(inner_terminated):
            termination_reason = str(
                inner_info.get("termination_reason") or _REASON_COLLISION
            )
            terminated = True

        if (
            not terminated
            and not truncated
            and self.step_count >= int(self.obstacle_config.max_episode_steps)
        ):
            truncated = True
            termination_reason = _REASON_MAX_STEPS
        if not terminated and not truncated and self._watchdog_expired():
            truncated = True
            termination_reason = _REASON_WATCHDOG
        if truncated and inner_info.get("sensor_failure"):
            termination_reason = _REASON_SENSOR

        new_outer_outcome = (terminated or truncated) and not (
            inner_terminated or inner_truncated
        )
        hover_errors: tuple[str, ...] = ()
        if new_outer_outcome:
            hover_errors = inner.request_named_terminal_hover()
            self.safety_override_count += 1
        elif terminated or truncated:
            self.safety_override_count += 1

        progress = previous_distance - distance
        action_change = float(np.linalg.norm(action_array - previous_action))
        breakdown = calculate_obstacle_reward(
            previous_distance_to_goal=previous_distance,
            distance_to_goal=distance,
            action=action_array,
            previous_action=previous_action,
            clearance=clearance,
            terminal_reason=termination_reason,
            config=self.obstacle_config.reward,
        )
        if terminated or truncated:
            self.episode_complete = True
        else:
            self.previous_distance_to_goal = distance
            self.previous_position = current_position
            self.previous_action = action_array.copy()
        self.last_observation = observation.copy()

        info = self._build_info(
            inner_info,
            previous_distance=previous_distance,
            distance=distance,
            progress=progress,
            clearance=clearance,
            action=action_array,
            previous_action=previous_action,
            action_change=action_change,
            reward=breakdown,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
        )
        info["terminal_hover_errors"] = hover_errors
        return observation, breakdown.total, terminated, truncated, info

    def close(self) -> None:
        """Close the environment and perform ordered cleanup."""
        self.close_with_result()

    def close_with_result(self) -> ObstacleEnvironmentCleanupResult | None:
        """Close named UAV control before exact scene and marker cleanup."""
        if self.closed:
            return self.last_cleanup_result
        self.last_cleanup_result = self._close_inner_environment(
            include_scene_cleanup=True
        )
        self.closed = True
        return self.last_cleanup_result

    def _resolve_course(
        self,
        options: ObstacleEpisodeResetOptions,
    ) -> ValidatedCourse | None:
        if options.uses_external_endpoints:
            if not self.obstacle_config.course.allow_external_test_endpoints:
                raise ValueError(
                    "endpoint-only episodes require the explicit test-only gate"
                )
            return None
        if options.validated_course is not None:
            return require_solvable_course(options.validated_course)
        suite = self._load_course_suite()
        profile_id, base_seed = self._select_profile_and_seed(suite)
        return generate_solvable_course(
            suite,
            profile_id,
            base_seed,
            repository_root=self.repository_root,
        )

    def _select_profile_and_seed(
        self,
        suite: CourseSuiteConfig,
    ) -> tuple[str, int]:
        selection = self.obstacle_config.course
        if selection.mode is CourseSelectionMode.FIXED:
            profile = suite.profile(selection.fixed_profile_id)
            if selection.fixed_base_seed not in profile.base_seeds:
                raise ValueError("fixed base seed is not declared by the profile")
            return profile.profile_id, selection.fixed_base_seed
        allowed = set(selection.seeded_profile_ids)
        pairs = sorted(
            (profile.profile_id, seed)
            for profile in suite.profiles
            if profile.profile_id in allowed and profile.split is CourseSplit.TRAINING
            for seed in profile.base_seeds
        )
        if not pairs:
            raise ValueError("seeded course pool is empty")
        index = int(self.np_random.integers(0, len(pairs)))
        return pairs[index]

    def _validate_pre_client_mode(
        self,
        options: ObstacleEpisodeResetOptions,
    ) -> None:
        if options.uses_external_endpoints:
            if self.client_factory is None:
                raise ValueError(
                    "endpoint-only mode requires an injected fake client factory"
                )
            return
        if not self.obstacle_config.authorization.complete:
            raise ValueError(
                "live scene episodes require every explicit M13.5 authorization"
            )

    def _prepare_episode_context(
        self,
        options: ObstacleEpisodeResetOptions,
        course: ValidatedCourse | None,
    ) -> _EpisodeContext:
        self.lidar_config = load_lidar_feature_config(
            self._resource_path(self.obstacle_config.course.lidar_config_path)
        )
        if options.uses_external_endpoints:
            assert options.start_anchor is not None
            assert options.goal_approach is not None
            assert options.workspace_bounds is not None
            self._ensure_client()
            return _EpisodeContext(
                options.scene_id,
                options.scene_digest,
                options.profile_id,
                options.base_seed,
                options.accepted_candidate_seed,
                options.attempt_index,
                options.start_id,
                options.goal_id,
                options.start_anchor,
                options.goal_approach,
                options.workspace_bounds,
            )

        assert course is not None
        client = self._ensure_client()
        module = self._ensure_client_module()
        catalog = load_asset_catalog(
            self._resource_path(self.obstacle_config.course.asset_catalog_path)
        )
        manager = self._ensure_scene_manager(client, module, catalog)
        auth = self.obstacle_config.authorization
        materialized, _ = manager.reset_scene(
            course.scene,
            MaterializationConfig(
                vehicle_name=self._require_lidar_config().vehicle_name,
                allow_scene_mutation=auth.allow_scene_mutation,
                confirm_scene_area_clear=auth.confirm_scene_area_clear,
                confirm_no_visible_collision=auth.confirm_no_visible_collision,
                allow_debug_markers=auth.allow_debug_markers,
                allow_marker_flush=auth.allow_marker_flush,
            ),
        )
        return _context_from_materialized(course, materialized)

    def _create_inner_environment(
        self,
        context: _EpisodeContext,
    ) -> ColosseumLidarUAVEnv:
        navigation = _navigation_for_context(
            self.obstacle_config.navigation,
            context,
            self.obstacle_config.max_episode_steps,
        )
        config = ColosseumLidarUAVEnvConfig(
            navigation=navigation,
            lidar=self._require_lidar_config(),
            confirm_no_visible_collision=(
                self.obstacle_config.authorization.confirm_no_visible_collision
                or self.obstacle_config.course.allow_external_test_endpoints
            ),
        )
        client = self._require_client()
        if self.inner_environment_factory is not None:
            return self.inner_environment_factory(config, client)
        return ColosseumLidarUAVEnv(
            config,
            client_factory=lambda: client,
            sleep_fn=self.sleep_fn,
        )

    def _build_info(
        self,
        inner_info: Mapping[str, Any],
        *,
        previous_distance: float,
        distance: float,
        progress: float,
        clearance: ClearanceEvidence,
        action: np.ndarray,
        previous_action: np.ndarray,
        action_change: float,
        reward: ObstacleRewardBreakdown | None,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        context = self._require_context()
        lidar = inner_info.get("lidar")
        consecutive_invalid = (
            lidar.get("consecutive_invalid_scans", 0)
            if isinstance(lidar, Mapping)
            else 0
        )
        return {
            "schema_version": M13_5_OBSTACLE_INFO_SCHEMA_VERSION,
            "scene_digest": context.scene_digest,
            "profile_id": context.profile_id,
            "base_seed": context.base_seed,
            "accepted_candidate_seed": context.accepted_candidate_seed,
            "attempt_index": context.attempt_index,
            "start_id": context.start_id,
            "goal_id": context.goal_id,
            "distance_to_goal": distance,
            "previous_distance_to_goal": previous_distance,
            "goal_progress": progress,
            "minimum_lidar_clearance_m": clearance.measured_clearance_m,
            "reward_clearance_m": clearance.reward_clearance_m,
            "clearance_source": clearance.source,
            "collision": bool(inner_info.get("collision", False)),
            "workspace_violation": termination_reason == _REASON_WORKSPACE,
            "ground_clearance_violation": termination_reason == _REASON_GROUND,
            "unsafe_clearance": clearance.unsafe,
            "sensor_failure": bool(inner_info.get("sensor_failure", False)),
            "consecutive_invalid_scans": int(consecutive_invalid),
            "path_length_m": self.path_length_m,
            "previous_action": [float(value) for value in previous_action],
            "action": [float(value) for value in action],
            "action_magnitude": float(np.linalg.norm(action)),
            "action_change_magnitude": action_change,
            "reward_breakdown": asdict(reward) if reward is not None else None,
            "success": termination_reason == _REASON_GOAL,
            "safety_override_count": self.safety_override_count,
            "step_count": self.step_count,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "prior_episode_cleanup": _cleanup_mapping(self.prior_episode_cleanup),
        }

    def _recover_rpc_step(
        self,
        error: Exception,
        action: np.ndarray,
        previous_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation = self._require_last_observation().copy()
        previous_distance = self._require_previous_distance()
        clearance = calculate_clearance_evidence(
            observation[10:82],
            lidar_valid=False,
            lidar_config=self._require_lidar_config(),
            reward_config=self.obstacle_config.reward,
            last_valid_clearance_m=self.last_valid_clearance_m,
        )
        breakdown = calculate_obstacle_reward(
            previous_distance_to_goal=previous_distance,
            distance_to_goal=previous_distance,
            action=action,
            previous_action=previous_action,
            clearance=clearance,
            terminal_reason=_REASON_RPC,
            config=self.obstacle_config.reward,
        )
        cleanup = self._close_inner_environment()
        self.last_cleanup_result = cleanup
        self.episode_complete = True
        self.step_count += 1
        self.safety_override_count += 1
        info = self._build_info(
            {},
            previous_distance=previous_distance,
            distance=previous_distance,
            progress=0.0,
            clearance=clearance,
            action=action,
            previous_action=previous_action,
            action_change=float(np.linalg.norm(action - previous_action)),
            reward=breakdown,
            terminated=False,
            truncated=True,
            termination_reason=_REASON_RPC,
        )
        info["rpc_error"] = f"{type(error).__name__}: {error}"
        info["cleanup_result"] = _cleanup_mapping(cleanup)
        return observation, breakdown.total, False, True, info

    def _cleanup_after_reset_failure(self) -> ObstacleEnvironmentCleanupResult | None:
        return self._close_inner_environment(include_scene_cleanup=True)

    def _close_inner_environment(
        self,
        *,
        include_scene_cleanup: bool = False,
    ) -> ObstacleEnvironmentCleanupResult | None:
        inner = self.inner_env
        uav_cleanup: CleanupResult | None = None
        lifecycle: dict[str, Any] | None = None
        errors: list[str] = []
        if inner is not None:
            try:
                uav_cleanup = inner.close_with_result()
            except BaseException as exc:
                errors.append(f"UAV cleanup raised {type(exc).__name__}: {exc}")
            lifecycle = dict(inner.lifecycle_evidence)
            self.inner_env = None

        unsafe = bool(
            errors or (uav_cleanup is not None and uav_cleanup.safety_critical_failure)
        )
        scene_results: tuple[SceneCleanupResult, ...] = ()
        deferred = False
        deferred_reason: str | None = None
        if include_scene_cleanup:
            if unsafe:
                deferred = True
                deferred_reason = "named UAV cleanup was not conclusively safe"
            else:
                runtime = self._active_scene_runtime()
                if runtime is not None:
                    try:
                        scene_results = cleanup_scene_resources(
                            self._require_client(), runtime
                        )
                    except BaseException as exc:
                        errors.append(
                            f"scene cleanup raised {type(exc).__name__}: {exc}"
                        )
                    if self.scene_manager is not None and all(
                        item.succeeded for item in scene_results
                    ):
                        self.scene_manager.active_runtime = None
        if inner is None and not include_scene_cleanup and not errors:
            return None
        return ObstacleEnvironmentCleanupResult(
            uav_cleanup,
            lifecycle,
            scene_results,
            deferred,
            deferred_reason,
            tuple(errors),
        )

    def _active_scene_runtime(self) -> Any | None:
        if self.scene_manager is None:
            return None
        return self.scene_manager.active_runtime or self.scene_manager.last_runtime

    def _ensure_client(self) -> Any:
        if self.client is None:
            if self.client_factory is not None:
                self.client = self.client_factory()
            else:
                module = self._ensure_client_module()
                self.client = create_multirotor_client(module)
            confirm_connection(self.client)
        return self.client

    def _ensure_client_module(self) -> ModuleType:
        if self.client_module is None:
            self.client_module = self.client_module_loader(
                self.obstacle_config.navigation.client_module
            )
        return self.client_module

    def _ensure_scene_manager(
        self,
        client: Any,
        module: ModuleType,
        catalog: AssetCatalog,
    ) -> ColosseumSceneManager:
        if self.scene_manager is None:
            if self.scene_manager_factory is not None:
                self.scene_manager = self.scene_manager_factory(client, module, catalog)
            else:
                self.scene_manager = ColosseumSceneManager(
                    client,
                    module,
                    catalog,
                    sleep_fn=self.sleep_fn,
                )
        return self.scene_manager

    def _load_course_suite(self) -> CourseSuiteConfig:
        return load_course_suite_config(
            self._resource_path(self.obstacle_config.course.course_suite_path)
        )

    def _resource_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repository_root / path

    def _watchdog_expired(self) -> bool:
        if self.episode_started_at is None:
            raise RuntimeError("episode watchdog was not initialized")
        return (
            self.monotonic_fn() - self.episode_started_at
            >= self.obstacle_config.watchdog_timeout_s
        )

    def _validate_observation(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation)
        if value.shape != (LIDAR_OBSERVATION_SIZE,):
            raise ObstacleEnvironmentError("observation must have shape (83,)")
        if value.dtype != np.float32:
            raise ObstacleEnvironmentError("observation must use float32")
        if not np.all(np.isfinite(value)):
            raise ObstacleEnvironmentError("observation must be finite")
        if not self.observation_space.contains(value):
            raise ObstacleEnvironmentError("observation is outside declared bounds")
        return value

    def _validate_step_allowed(self) -> None:
        if self.closed:
            raise RuntimeError("Cannot step a closed ColosseumObstacleUAVEnv.")
        if not self.has_reset:
            raise RuntimeError("Call reset before step.")
        if self.episode_complete:
            raise RuntimeError("Episode is complete; call reset before stepping.")

    def _require_inner(self) -> ColosseumLidarUAVEnv:
        if self.inner_env is None:
            raise RuntimeError("inner LiDAR environment is unavailable")
        return self.inner_env

    def _require_context(self) -> _EpisodeContext:
        if self.episode_context is None:
            raise RuntimeError("episode course context is unavailable")
        return self.episode_context

    def _require_lidar_config(self) -> LidarFeatureConfig:
        if self.lidar_config is None:
            raise RuntimeError("LiDAR configuration is unavailable")
        return self.lidar_config

    def _require_previous_distance(self) -> float:
        if self.previous_distance_to_goal is None:
            raise RuntimeError("previous goal distance is unavailable")
        return self.previous_distance_to_goal

    def _require_previous_position(self) -> Position3D:
        if self.previous_position is None:
            raise RuntimeError("previous measured position is unavailable")
        return self.previous_position

    def _require_last_observation(self) -> np.ndarray:
        if self.last_observation is None:
            raise RuntimeError("last observation is unavailable")
        return self.last_observation

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError("simulator client is unavailable")
        return self.client


def load_colosseum_obstacle_uav_env_config(
    path: str | Path,
) -> ColosseumObstacleUAVEnvConfig:
    """Load the strict versioned M13.5 environment YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("M13.5 environment config must contain a mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "max_episode_steps",
            "watchdog_timeout_s",
            "navigation",
            "reward",
            "course",
            "authorization",
        },
        "root",
    )
    navigation_raw = _mapping(raw.get("navigation", {}), "navigation")
    reward_raw = _mapping(raw.get("reward", {}), "reward")
    course_raw = _mapping(raw.get("course", {}), "course")
    authorization_raw = _mapping(raw.get("authorization", {}), "authorization")
    _reject_unknown(
        navigation_raw,
        set(ColosseumUAVEnvConfig.__dataclass_fields__),
        "navigation",
    )
    _reject_unknown(
        reward_raw,
        set(ObstacleRewardConfig.__dataclass_fields__),
        "reward",
    )
    _reject_unknown(
        course_raw,
        set(ObstacleCourseSelectionConfig.__dataclass_fields__),
        "course",
    )
    _reject_unknown(
        authorization_raw,
        set(ObstacleRuntimeAuthorization.__dataclass_fields__),
        "authorization",
    )
    navigation_values = dict(navigation_raw)
    if "default_goal_offset" in navigation_values:
        navigation_values["default_goal_offset"] = tuple(
            navigation_values["default_goal_offset"]
        )
    course_values = dict(course_raw)
    if "mode" in course_values:
        course_values["mode"] = CourseSelectionMode(str(course_values["mode"]))
    if "seeded_profile_ids" in course_values:
        course_values["seeded_profile_ids"] = tuple(course_values["seeded_profile_ids"])
    return ColosseumObstacleUAVEnvConfig(
        schema_version=_required_int(raw, "schema_version"),
        navigation=ColosseumUAVEnvConfig(**navigation_values),
        reward=ObstacleRewardConfig(**reward_raw),
        course=ObstacleCourseSelectionConfig(**course_values),
        authorization=ObstacleRuntimeAuthorization(**authorization_raw),
        max_episode_steps=_required_int(raw, "max_episode_steps"),
        watchdog_timeout_s=_required_float(raw, "watchdog_timeout_s"),
    )


def _coerce_reset_options(
    options: dict[str, Any] | None,
) -> ObstacleEpisodeResetOptions:
    if not options:
        return ObstacleEpisodeResetOptions()
    allowed = set(ObstacleEpisodeResetOptions.__dataclass_fields__)
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown reset option keys: {sorted(unknown)}")
    values = dict(options)
    for name in ("start_anchor", "goal_approach"):
        if name in values and values[name] is not None:
            values[name] = _coerce_position(values[name], name)
    if "workspace_bounds" in values and isinstance(values["workspace_bounds"], Mapping):
        values["workspace_bounds"] = Bounds3D(**values["workspace_bounds"])
    return ObstacleEpisodeResetOptions(**values)


def _context_from_materialized(
    course: ValidatedCourse,
    materialized: MaterializedScene,
) -> _EpisodeContext:
    result = course.result
    return _EpisodeContext(
        materialized.scene_id,
        result.scene_digest,
        result.profile_id,
        result.base_seed,
        result.accepted_candidate_seed,
        result.attempt_index,
        course.scene.config.start_pad.name,
        course.scene.config.goal_pad.name,
        _position(materialized.start_anchor_world),
        _position(materialized.goal_approach_world),
        materialized.workspace_world,
    )


def _navigation_for_context(
    base: ColosseumUAVEnvConfig,
    context: _EpisodeContext,
    max_episode_steps: int,
) -> ColosseumUAVEnvConfig:
    anchor = context.start_anchor
    bounds = context.workspace
    xy_limit = max(
        abs(bounds.min_x - anchor.x),
        abs(bounds.max_x - anchor.x),
        abs(bounds.min_y - anchor.y),
        abs(bounds.max_y - anchor.y),
        base.workspace_xy_limit,
    )
    up_limit = max(anchor.z - bounds.min_z, base.workspace_up_limit)
    down_limit = max(bounds.max_z - anchor.z, base.workspace_down_limit)
    return replace(
        base,
        workspace_xy_limit=xy_limit,
        workspace_up_limit=up_limit,
        workspace_down_limit=down_limit,
        max_episode_steps=max_episode_steps,
    )


def _primary_terminal_reason(
    collision: bool,
    ground_violation: bool,
    workspace_violation: bool,
    goal_success: bool,
) -> str | None:
    if collision:
        return _REASON_COLLISION
    if ground_violation:
        return _REASON_GROUND
    if workspace_violation:
        return _REASON_WORKSPACE
    if goal_success:
        return _REASON_GOAL
    return None


def _inside_bounds(position: Position3D, bounds: Bounds3D) -> bool:
    return (
        bounds.min_x <= position.x <= bounds.max_x
        and bounds.min_y <= position.y <= bounds.max_y
        and bounds.min_z <= position.z <= bounds.max_z
    )


def _validated_action(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (3,):
        raise ValueError("action must have shape (3,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("action must contain finite values")
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def _position_from_info(info: Mapping[str, Any]) -> Position3D:
    value = info.get("measured_position")
    return _coerce_position(value, "measured_position")


def _coerce_position(value: Any, name: str) -> Position3D:
    if isinstance(value, Position3D):
        result = value
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        result = Position3D(float(value[0]), float(value[1]), float(value[2]))
    else:
        raise ValueError(f"{name} must contain three coordinates")
    if not all(math.isfinite(item) for item in _position_tuple(result)):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def _position(value: Vector3) -> Position3D:
    return Position3D(value.x, value.y, value.z)


def _position_tuple(value: Position3D) -> tuple[float, float, float]:
    return (value.x, value.y, value.z)


def _distance(first: Position3D, second: Position3D) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def _finite_info_float(info: Mapping[str, Any], key: str) -> float:
    try:
        value = float(info[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ObstacleEnvironmentError(f"inner info lacks finite {key}") from exc
    if not math.isfinite(value):
        raise ObstacleEnvironmentError(f"inner info lacks finite {key}")
    return value


def _cleanup_mapping(
    result: ObstacleEnvironmentCleanupResult | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "uav_cleanup": (
            asdict(result.uav_cleanup) if result.uav_cleanup is not None else None
        ),
        "scene_cleanup": [asdict(item) for item in result.scene_cleanup],
        "scene_cleanup_deferred": result.scene_cleanup_deferred,
        "scene_cleanup_deferred_reason": result.scene_cleanup_deferred_reason,
        "landing_success": result.landing_success,
        "errors": list(result.errors),
        "succeeded": result.succeeded,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")


def _required_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{key} must be an integer")
    return result


def _required_float(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ValueError(f"{key} must be numeric")
    converted = float(result)
    if not math.isfinite(converted):
        raise ValueError(f"{key} must be finite")
    return converted


def _validate_navigation_config(config: ColosseumUAVEnvConfig) -> None:
    positive_names = (
        "anchor_altitude",
        "min_ground_clearance",
        "workspace_xy_limit",
        "workspace_up_limit",
        "workspace_down_limit",
        "max_horizontal_velocity",
        "max_vertical_velocity",
        "control_duration",
        "anchor_move_velocity",
        "anchor_move_timeout",
        "goal_tolerance",
        "min_goal_distance",
    )
    finite_names = (
        "progress_reward_scale",
        "step_penalty",
        "action_penalty_scale",
        "success_reward",
        "collision_penalty",
        "out_of_bounds_penalty",
    )
    for name in positive_names:
        value = getattr(config, name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"navigation.{name} must be finite and positive")
    for name in finite_names:
        value = getattr(config, name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"navigation.{name} must be finite")
    if not isinstance(config.client_module, str) or not config.client_module.strip():
        raise ValueError("navigation.client_module must not be empty")
    if not isinstance(config.random_goal, bool):
        raise ValueError("navigation.random_goal must be boolean")
    if (
        not isinstance(config.max_episode_steps, int)
        or isinstance(config.max_episode_steps, bool)
        or config.max_episode_steps <= 0
    ):
        raise ValueError("navigation.max_episode_steps must be a positive integer")
    if config.anchor_altitude < config.min_ground_clearance:
        raise ValueError("navigation.anchor_altitude must satisfy min_ground_clearance")
    if config.min_goal_distance <= config.goal_tolerance:
        raise ValueError("navigation.min_goal_distance must exceed goal_tolerance")
    offset = config.default_goal_offset
    if not isinstance(offset, (tuple, list)) or len(offset) != 3:
        raise ValueError("navigation.default_goal_offset must have three values")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in offset
    ):
        raise ValueError("navigation.default_goal_offset must contain finite values")
