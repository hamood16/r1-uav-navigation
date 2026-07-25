"""Opt-in Colosseum environment with fixed-size SensorLocalFrame LiDAR features."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from r1_uav_nav.envs.colosseum_uav_env import (
    OBSERVATION_SIZE,
    ColosseumUAVEnv,
    ColosseumUAVEnvConfig,
    ColosseumUAVState,
)
from r1_uav_nav.sim.colosseum_capabilities import (
    CollisionClassification,
    classify_collision_samples,
    sample_collision_information,
)
from r1_uav_nav.sim.colosseum_client import CleanupResult, CleanupState
from r1_uav_nav.sim.colosseum_lidar import read_lidar_features
from r1_uav_nav.sim.lidar_features import (
    LIDAR_OBSERVATION_SIZE,
    LidarFaultTracker,
    LidarFeatureConfig,
    LidarFeatureResult,
    LidarTimestampTracker,
    extraction_evidence,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D, calculate_position_error

_GROUNDED_SPEED_TOLERANCE_M_S = 0.05
_REASON_SENSOR_FAILURE = "sensor_failure"


class LidarResetError(RuntimeError):
    """Raised after reset-time LiDAR retries and named cleanup are exhausted."""


@dataclass(frozen=True)
class ColosseumLidarUAVEnvConfig:
    """Navigation and sensing configuration for the opt-in environment."""

    navigation: ColosseumUAVEnvConfig = field(default_factory=ColosseumUAVEnvConfig)
    lidar: LidarFeatureConfig = field(default_factory=LidarFeatureConfig)
    confirm_no_visible_collision: bool = False
    start_anchor_position_tolerance_m: float = 0.75
    start_anchor_speed_tolerance_m_s: float = 0.1
    start_anchor_confirmation_timeout_s: float = 5.0
    start_anchor_poll_interval_s: float = 0.2
    start_anchor_consecutive_samples: int = 3
    return_transit_altitude_m: float = 2.5
    landing_position_tolerance_m: float = 0.75
    landing_confirmation_timeout_s: float = 5.0
    landing_poll_interval_s: float = 0.2
    touchdown_consecutive_samples: int = 3
    final_state_confirmation_timeout_s: float = 5.0
    final_state_poll_interval_s: float = 0.2

    def __post_init__(self) -> None:
        if not self.confirm_no_visible_collision:
            raise ValueError(
                "LiDAR environment requires explicit no-visible-collision confirmation"
            )
        for name in (
            "start_anchor_position_tolerance_m",
            "start_anchor_speed_tolerance_m_s",
            "start_anchor_confirmation_timeout_s",
            "start_anchor_poll_interval_s",
            "return_transit_altitude_m",
            "landing_position_tolerance_m",
            "landing_confirmation_timeout_s",
            "landing_poll_interval_s",
            "final_state_confirmation_timeout_s",
            "final_state_poll_interval_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "start_anchor_consecutive_samples",
            "touchdown_consecutive_samples",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class LidarEpisodeResetOptions:
    """Optional externally materialized scene endpoints."""

    start_anchor: Position3D | None = None
    goal_approach: Position3D | None = None

    def __post_init__(self) -> None:
        if (self.start_anchor is None) != (self.goal_approach is None):
            raise ValueError("start_anchor and goal_approach must be provided together")


class ColosseumLidarUAVEnv(ColosseumUAVEnv):
    """Named lifecycle environment that never performs a broad simulator reset."""

    def __init__(
        self,
        config: ColosseumLidarUAVEnvConfig,
        client_factory: Callable[[], Any] | None = None,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config.navigation, client_factory)
        self.lidar_env_config = config
        self.lidar_config = config.lidar
        self.sleep_fn = sleep_fn
        self.timestamp_tracker = LidarTimestampTracker(
            self.lidar_config.maximum_repeated_timestamp_transitions
        )
        self.fault_tracker = LidarFaultTracker(
            self.lidar_config.maximum_consecutive_invalid_scans
        )
        self.requires_cleanup = False
        self.last_lidar_result: LidarFeatureResult | None = None
        self.last_navigation_observation: np.ndarray | None = None
        self.last_sensor_safety_errors: tuple[str, ...] = ()
        self.original_ground_position: Position3D | None = None
        self.collision_baseline_timestamp: int | None = None
        self.lifecycle_evidence: dict[str, Any] = _new_lifecycle_evidence()
        self.lifecycle_evidence["landing_position_tolerance_m"] = (
            self.lidar_env_config.landing_position_tolerance_m
        )
        self.cleanup_attempt_count = 0
        self.primary_cleanup_result: CleanupResult | None = None
        self.recovery_cleanup_result: CleanupResult | None = None
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.full(OBSERVATION_SIZE, -1.0, dtype=np.float32),
                    np.zeros(self.lidar_config.feature_count + 1, dtype=np.float32),
                )
            ),
            high=np.ones(LIDAR_OBSERVATION_SIZE, dtype=np.float32),
            shape=(LIDAR_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    @property
    def vehicle_name(self) -> str:
        return self.lidar_config.vehicle_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a named episode without calling the global simulator reset."""
        if self.closed:
            raise RuntimeError("Cannot reset a closed ColosseumLidarUAVEnv.")
        if (
            self.cleanup_safety_critical_failure_seen
            and not self._owns_control_state()
            and not self.requires_cleanup
        ):
            raise RuntimeError("Previous safety-critical cleanup failed.")

        gym.Env.reset(self, seed=seed)
        client = self._get_or_create_client()
        if self._owns_control_state() or self.requires_cleanup:
            cleanup = self._cleanup_named_control()
            self._record_named_cleanup_result(cleanup)
            if cleanup.safety_critical_failure:
                raise RuntimeError(
                    "Named cleanup failed before LiDAR environment reset."
                )

        self._reset_episode_bookkeeping()
        self.requires_cleanup = False
        self.timestamp_tracker.reset()
        self.fault_tracker.reset()
        self.last_lidar_result = None
        self.last_navigation_observation = None
        self.last_sensor_safety_errors = ()
        self.original_ground_position = None
        self.collision_baseline_timestamp = None
        self.lifecycle_evidence = _new_lifecycle_evidence()
        self.lifecycle_evidence["landing_position_tolerance_m"] = (
            self.lidar_env_config.landing_position_tolerance_m
        )
        self.cleanup_attempt_count = 0
        self.primary_cleanup_result = None
        self.recovery_cleanup_result = None
        self.last_cleanup_result = None
        self.cleanup_safety_critical_failure_seen = False

        try:
            initial_state = self._read_named_state()
            initial_position, initial_velocity = self._extract_position_and_velocity(
                initial_state
            )
            if not _is_landed(initial_state):
                raise RuntimeError("SimpleFlight must be landed before reset.")
            if _speed(initial_velocity) > _GROUNDED_SPEED_TOLERANCE_M_S:
                raise RuntimeError("SimpleFlight must be stationary before reset.")
            api_enabled = client.isApiControlEnabled(vehicle_name=self.vehicle_name)
            if not isinstance(api_enabled, bool) or api_enabled:
                raise RuntimeError("API control must be disabled before reset.")
            _, collision_samples = sample_collision_information(
                client,
                vehicle_name=self.vehicle_name,
                sleep_fn=self.sleep_fn,
            )
            collision_assessment = classify_collision_samples(
                collision_samples,
                is_landed=True,
                measured_speed=_speed(initial_velocity),
                api_control_enabled=False,
                operator_confirmed_stable=(
                    self.lidar_env_config.confirm_no_visible_collision
                ),
            )
            if collision_assessment.classification not in {
                CollisionClassification.NO_COLLISION,
                CollisionClassification.EXPECTED_GROUND_CONTACT,
            }:
                raise RuntimeError("Grounded collision evidence is unsafe.")

            self.original_ground_position = initial_position
            self.collision_baseline_timestamp = collision_assessment.baseline_timestamp
            self.lifecycle_evidence["original_ground_position"] = _position_evidence(
                initial_position
            )
            self.lifecycle_evidence["collision_baseline_timestamp"] = (
                self.collision_baseline_timestamp
            )
            self.ground_reference_z = initial_position.z
            reset_options = _coerce_reset_options(options)
            target_anchor, goal_position = self._resolve_episode_points(
                initial_position, reset_options
            )
            self._validate_position_safety(target_anchor)
            self.anchor_position = target_anchor
            self._validate_goal_position(target_anchor, goal_position)
            self.goal_position = goal_position

            self._enable_named_control_and_takeoff(client)
            self._move_named_to_anchor(client, target_anchor)
            anchor_state, anchor_position = self._confirm_start_anchor(target_anchor)
            self.anchor_position = anchor_position
            measured_state = self._build_named_state(anchor_state)
            self.previous_distance_to_goal = calculate_position_error(
                measured_state.position, goal_position
            )
            lidar_result = self._acquire_reset_lidar()
            self.last_lidar_result = lidar_result
            self.has_reset = True
            navigation = super()._build_observation(measured_state)
            self.last_navigation_observation = navigation.copy()
            observation = _augment_observation(navigation, lidar_result)
            info = super()._build_info(
                measured_state,
                success=False,
                out_of_bounds=False,
                termination_reason=None,
            )
            info["lidar"] = extraction_evidence(lidar_result)
            info["sensor_failure"] = False
            info["start_anchor_confirmation"] = dict(
                self.lifecycle_evidence["start_anchor_confirmation"]
            )
            return observation, info
        except BaseException as exc:
            if self._owns_control_state() or self.requires_cleanup:
                cleanup = self._cleanup_named_control()
                self._record_named_cleanup_result(cleanup)
            if isinstance(exc, LidarResetError):
                raise
            raise

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one navigation action and append current LiDAR features."""
        if self.requires_cleanup:
            raise RuntimeError("Sensor failure requires reset or close before step.")
        navigation, reward, terminated, truncated, info = super().step(action)
        self.last_navigation_observation = navigation.copy()
        lidar_result = read_lidar_features(
            self._require_client(), self.lidar_config, self.timestamp_tracker
        )
        self.last_lidar_result = lidar_result
        self.fault_tracker.record(lidar_result.lidar_valid == 1.0)
        sensor_failure = (
            not terminated and not truncated and self.fault_tracker.threshold_reached
        )
        if sensor_failure:
            self.last_sensor_safety_errors = self._apply_sensor_failure_safety()
            terminated = False
            truncated = True
            self.episode_complete = True
            self.requires_cleanup = True
            info["termination_reason"] = _REASON_SENSOR_FAILURE
        info["lidar"] = extraction_evidence(lidar_result)
        info["lidar"][
            "consecutive_invalid_scans"
        ] = self.fault_tracker.consecutive_invalid_scans
        info["sensor_failure"] = sensor_failure
        info["sensor_safety_errors"] = self.last_sensor_safety_errors
        return (
            _augment_observation(navigation, lidar_result),
            reward,
            terminated,
            truncated,
            info,
        )

    def close_with_result(self) -> CleanupResult | None:
        """Close using exact named cleanup and no scene mutation."""
        if self.closed and self.last_cleanup_result is not None:
            return self.last_cleanup_result
        if self.client is not None and (
            self._owns_control_state() or self.requires_cleanup
        ):
            self._record_named_cleanup_result(self._cleanup_named_control())
            if (
                self.last_cleanup_result is not None
                and self.last_cleanup_result.safety_critical_failure
                and self.requires_cleanup
                and self.cleanup_attempt_count < 2
            ):
                self._record_named_cleanup_result(self._cleanup_named_control())
        self.closed = True
        return self.last_cleanup_result

    def _resolve_episode_points(
        self,
        initial_position: Position3D,
        options: LidarEpisodeResetOptions,
    ) -> tuple[Position3D, Position3D]:
        if options.start_anchor is not None and options.goal_approach is not None:
            return options.start_anchor, options.goal_approach
        anchor = Position3D(
            initial_position.x,
            initial_position.y,
            initial_position.z - self.config.anchor_altitude,
        )
        offset = self._select_goal_offset({})
        goal = Position3D(
            anchor.x + offset[0],
            anchor.y + offset[1],
            anchor.z + offset[2],
        )
        return anchor, goal

    def _enable_named_control_and_takeoff(self, client: Any) -> None:
        client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.cleanup_state = replace(self.cleanup_state, api_control_enabled=True)
        client.armDisarm(True, vehicle_name=self.vehicle_name)
        self.cleanup_state = replace(self.cleanup_state, armed=True)
        self.cleanup_state = replace(self.cleanup_state, takeoff_attempted=True)
        client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.cleanup_state = replace(self.cleanup_state, airborne=True)

    def _move_named_to_anchor(self, client: Any, anchor: Position3D) -> None:
        client.moveToPositionAsync(
            anchor.x,
            anchor.y,
            anchor.z,
            self.config.anchor_move_velocity,
            timeout_sec=self.config.anchor_move_timeout,
            vehicle_name=self.vehicle_name,
        ).join()

    def _confirm_start_anchor(self, target: Position3D) -> tuple[Any, Position3D]:
        client = self._require_client()
        client.hoverAsync(vehicle_name=self.vehicle_name).join()
        evidence = self.lifecycle_evidence["start_anchor_confirmation"]
        evidence["requested_start_anchor"] = _position_evidence(target)
        maximum_attempts = _confirmation_attempt_budget(
            self.lidar_env_config.start_anchor_confirmation_timeout_s,
            self.lidar_env_config.start_anchor_poll_interval_s,
        )
        consecutive = 0
        last_state: Any | None = None
        last_position: Position3D | None = None
        last_error: float | None = None
        last_speed: float | None = None
        for attempt in range(1, maximum_attempts + 1):
            evidence["confirmation_attempts"] = attempt
            last_state = self._read_named_state()
            last_position, velocity = self._extract_position_and_velocity(last_state)
            last_error = calculate_position_error(last_position, target)
            last_speed = _speed(velocity)
            rejection_reasons: list[str] = []
            if last_error > self.lidar_env_config.start_anchor_position_tolerance_m:
                rejection_reasons.append("position outside start-anchor tolerance")
            if last_speed > self.lidar_env_config.start_anchor_speed_tolerance_m_s:
                rejection_reasons.append("speed above stable-hover tolerance")
            collision_rejection = self._collision_rejection_reason()
            if collision_rejection:
                rejection_reasons.append(collision_rejection)
            consecutive = 0 if rejection_reasons else consecutive + 1
            evidence.update(
                {
                    "measured_positions": (
                        *evidence["measured_positions"],
                        _position_evidence(last_position),
                    ),
                    "measured_start_anchor": _position_evidence(last_position),
                    "position_error_m": last_error,
                    "position_tolerance_m": (
                        self.lidar_env_config.start_anchor_position_tolerance_m
                    ),
                    "measured_speed_m_s": last_speed,
                    "speed_tolerance_m_s": (
                        self.lidar_env_config.start_anchor_speed_tolerance_m_s
                    ),
                    "consecutive_accepted_samples": consecutive,
                    "required_consecutive_samples": (
                        self.lidar_env_config.start_anchor_consecutive_samples
                    ),
                    "rejection_reason": (
                        "; ".join(rejection_reasons) if rejection_reasons else None
                    ),
                }
            )
            if consecutive >= self.lidar_env_config.start_anchor_consecutive_samples:
                evidence["confirmation_success"] = True
                return last_state, last_position
            if attempt < maximum_attempts:
                self.sleep_fn(self.lidar_env_config.start_anchor_poll_interval_s)

        raise RuntimeError(
            "Start-anchor confirmation timed out after "
            f"{maximum_attempts} attempts; "
            f"position_error_m={last_error!r}; "
            "position_tolerance_m="
            f"{self.lidar_env_config.start_anchor_position_tolerance_m:.6f}; "
            f"speed_m_s={last_speed!r}; "
            "last_position="
            f"{_position_evidence(last_position) if last_position else None}"
        )

    def _read_named_state(self) -> Any:
        return self._require_client().getMultirotorState(vehicle_name=self.vehicle_name)

    def _read_state(self) -> Any:
        return self._read_named_state()

    def _move_by_velocity(self, client: Any, velocity: Position3D) -> None:
        client.moveByVelocityAsync(
            velocity.x,
            velocity.y,
            velocity.z,
            self.config.control_duration,
            vehicle_name=self.vehicle_name,
        ).join()

    def _build_named_state(self, raw_state: Any) -> ColosseumUAVState:
        position, velocity = self._extract_position_and_velocity(raw_state)
        collision = self._require_client().simGetCollisionInfo(
            vehicle_name=self.vehicle_name
        )
        return ColosseumUAVState(
            position,
            velocity,
            bool(getattr(collision, "has_collided", False)),
            self._require_ground_reference_z(),
            self._require_anchor_position(),
            self._require_goal_position(),
        )

    def _build_state(self, raw_state: Any) -> ColosseumUAVState:
        return self._build_named_state(raw_state)

    def _apply_terminal_hover(self) -> None:
        try:
            self._require_client().hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception as exc:
            self.last_terminal_safety_error = str(exc)

    def _acquire_reset_lidar(self) -> LidarFeatureResult:
        last: LidarFeatureResult | None = None
        for attempt in range(self.lidar_config.reset_scan_attempts):
            last = read_lidar_features(
                self._require_client(), self.lidar_config, self.timestamp_tracker
            )
            if last.lidar_valid == 1.0:
                self.fault_tracker.reset()
                return last
            if attempt + 1 < self.lidar_config.reset_scan_attempts:
                self.sleep_fn(self.lidar_config.reset_scan_interval_s)
        self._apply_sensor_failure_safety()
        self.requires_cleanup = True
        raise LidarResetError(
            "No valid fresh LiDAR scan appeared during bounded reset retries; "
            f"last_status={last.status.value if last else 'unavailable'}"
        )

    def _apply_sensor_failure_safety(self) -> tuple[str, ...]:
        errors: list[str] = []
        client = self._require_client()
        try:
            client.moveByVelocityAsync(
                0.0,
                0.0,
                0.0,
                self.lidar_config.safety_zero_velocity_duration_s,
                vehicle_name=self.vehicle_name,
            ).join()
        except BaseException as exc:
            errors.append(f"zero-velocity safety command raised {type(exc).__name__}")
        try:
            client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except BaseException as exc:
            errors.append(f"hover safety command raised {type(exc).__name__}")
        return tuple(errors)

    def _cleanup_named_control(self) -> CleanupResult:
        self.cleanup_attempt_count += 1
        attempt_number = self.cleanup_attempt_count
        attempted: list[str] = []
        succeeded: list[str] = []
        errors: list[str] = []
        evidence = _new_cleanup_attempt_evidence(attempt_number)
        self.lifecycle_evidence["cleanup_attempts"] = (
            *self.lifecycle_evidence["cleanup_attempts"],
            evidence,
        )
        self.lifecycle_evidence["recovery_retry_required"] = attempt_number > 1
        client = self._require_client()

        owns_flight = (
            self.cleanup_state.takeoff_attempted or self.cleanup_state.airborne
        )
        if owns_flight:
            _attempt_cleanup_action(
                "hoverAsync",
                lambda: client.hoverAsync(vehicle_name=self.vehicle_name).join(),
                attempted,
                succeeded,
                errors,
            )
            self._attempt_return_to_original_ground(
                attempted, succeeded, errors, evidence
            )
            _attempt_cleanup_action(
                "landAsync",
                lambda: client.landAsync(vehicle_name=self.vehicle_name).join(),
                attempted,
                succeeded,
                errors,
            )
            try:
                touchdown_state = self._confirm_physical_touchdown(evidence)
                self.cleanup_state = replace(
                    self.cleanup_state,
                    airborne=False,
                    takeoff_attempted=False,
                )
                evidence["landed_state_before_disarm"] = getattr(
                    touchdown_state, "landed_state", None
                )
            except BaseException as exc:
                errors.append(
                    "physical touchdown confirmation raised "
                    f"{type(exc).__name__}: {exc}"
                )

        if self.cleanup_state.armed:
            disarm_succeeded = _attempt_cleanup_action(
                "armDisarm",
                lambda: client.armDisarm(False, vehicle_name=self.vehicle_name),
                attempted,
                succeeded,
                errors,
            )
            if disarm_succeeded:
                self.cleanup_state = replace(self.cleanup_state, armed=False)
        if self.cleanup_state.api_control_enabled:
            release_succeeded = _attempt_cleanup_action(
                "enableApiControl",
                lambda: client.enableApiControl(False, vehicle_name=self.vehicle_name),
                attempted,
                succeeded,
                errors,
            )
            if release_succeeded:
                self.cleanup_state = replace(
                    self.cleanup_state, api_control_enabled=False
                )

        try:
            self._confirm_final_landed_state(evidence)
        except BaseException as exc:
            errors.append(
                "final landed-state confirmation raised " f"{type(exc).__name__}: {exc}"
            )

        safe = not errors
        if safe:
            self.cleanup_state = CleanupState()
            self.requires_cleanup = False
        else:
            self.requires_cleanup = True
        evidence.update(
            {
                "actions_attempted": tuple(attempted),
                "actions_succeeded": tuple(succeeded),
                "errors": tuple(errors),
                "succeeded": safe,
            }
        )
        return CleanupResult(
            tuple(attempted),
            tuple(succeeded),
            tuple(errors),
            not safe,
        )

    def _attempt_return_to_original_ground(
        self,
        attempted: list[str],
        succeeded: list[str],
        errors: list[str],
        evidence: dict[str, Any],
    ) -> None:
        original = self.original_ground_position
        if original is None:
            errors.append("original grounded position is unavailable")
            evidence["return_rejection_reason"] = errors[-1]
            return
        return_target = Position3D(
            original.x,
            original.y,
            original.z - self.lidar_env_config.return_transit_altitude_m,
        )
        evidence["return_target"] = _position_evidence(return_target)
        client = self._require_client()
        moved = _attempt_cleanup_action(
            "moveToPositionAsync",
            lambda: client.moveToPositionAsync(
                return_target.x,
                return_target.y,
                return_target.z,
                self.config.anchor_move_velocity,
                timeout_sec=self.config.anchor_move_timeout,
                vehicle_name=self.vehicle_name,
            ).join(),
            attempted,
            succeeded,
            errors,
        )
        if not moved:
            evidence["return_rejection_reason"] = (
                "return-to-original-ground movement failed"
            )
            return
        _attempt_cleanup_action(
            "hoverAsync",
            lambda: client.hoverAsync(vehicle_name=self.vehicle_name).join(),
            attempted,
            succeeded,
            errors,
        )
        try:
            _state, position = self._confirm_stable_return_target(
                return_target, evidence
            )
            evidence["confirmed_return_airborne_position"] = _position_evidence(
                position
            )
            evidence["return_target_confirmed"] = True
        except BaseException as exc:
            evidence["return_rejection_reason"] = f"{type(exc).__name__}: {exc}"
            errors.append(
                "return-target confirmation raised " f"{type(exc).__name__}: {exc}"
            )

    def _confirm_stable_return_target(
        self, target: Position3D, evidence: dict[str, Any]
    ) -> tuple[Any, Position3D]:
        maximum_attempts = _confirmation_attempt_budget(
            self.lidar_env_config.start_anchor_confirmation_timeout_s,
            self.lidar_env_config.start_anchor_poll_interval_s,
        )
        consecutive = 0
        last_state: Any | None = None
        last_position: Position3D | None = None
        for attempt in range(1, maximum_attempts + 1):
            last_state = self._read_named_state()
            last_position, velocity = self._extract_position_and_velocity(last_state)
            error = calculate_position_error(last_position, target)
            speed = _speed(velocity)
            rejection = (
                error > self.lidar_env_config.start_anchor_position_tolerance_m
                or speed > self.lidar_env_config.start_anchor_speed_tolerance_m_s
                or self._collision_rejection_reason() is not None
            )
            consecutive = 0 if rejection else consecutive + 1
            evidence.update(
                {
                    "return_confirmation_attempts": attempt,
                    "return_airborne_position": _position_evidence(last_position),
                    "return_airborne_position_error_m": error,
                    "return_airborne_speed_m_s": speed,
                    "return_consecutive_samples": consecutive,
                }
            )
            if consecutive >= self.lidar_env_config.start_anchor_consecutive_samples:
                return last_state, last_position
            if attempt < maximum_attempts:
                self.sleep_fn(self.lidar_env_config.start_anchor_poll_interval_s)
        raise RuntimeError(
            "return-target confirmation timed out; "
            f"attempts={maximum_attempts}; "
            "last_position="
            f"{_position_evidence(last_position) if last_position else None}"
        )

    def _confirm_physical_touchdown(self, evidence: dict[str, Any]) -> Any:
        original = self.original_ground_position
        if original is None:
            raise RuntimeError("original grounded position is unavailable")
        maximum_attempts = _confirmation_attempt_budget(
            self.lidar_env_config.landing_confirmation_timeout_s,
            self.lidar_env_config.landing_poll_interval_s,
        )
        consecutive = 0
        last_state: Any | None = None
        for attempt in range(1, maximum_attempts + 1):
            last_state = self._read_named_state()
            position, velocity = self._extract_position_and_velocity(last_state)
            speed = _speed(velocity)
            position_error = calculate_position_error(position, original)
            rejection_reasons: list[str] = []
            if speed > _GROUNDED_SPEED_TOLERANCE_M_S:
                rejection_reasons.append("speed above grounded tolerance")
            if position_error > self.lidar_env_config.landing_position_tolerance_m:
                rejection_reasons.append("position outside landing tolerance")
            if (
                abs(position.z - original.z)
                > self.lidar_env_config.landing_position_tolerance_m
            ):
                rejection_reasons.append("vertical position outside landing tolerance")
            collision_rejection = self._collision_rejection_reason()
            if collision_rejection:
                rejection_reasons.append(collision_rejection)
            consecutive = 0 if rejection_reasons else consecutive + 1
            evidence.update(
                {
                    "touchdown_confirmation_attempts": attempt,
                    "touchdown_consecutive_samples": consecutive,
                    "touchdown_position": _position_evidence(position),
                    "touchdown_position_error_m": position_error,
                    "touchdown_speed_m_s": speed,
                    "touchdown_rejection_reason": (
                        "; ".join(rejection_reasons) if rejection_reasons else None
                    ),
                }
            )
            if consecutive >= self.lidar_env_config.touchdown_consecutive_samples:
                evidence["physical_touchdown_confirmed"] = True
                return last_state
            if attempt < maximum_attempts:
                self.sleep_fn(self.lidar_env_config.landing_poll_interval_s)
        raise RuntimeError(
            "physical touchdown confirmation timed out; "
            f"attempts={maximum_attempts}; "
            f"last_position_error_m={evidence['touchdown_position_error_m']!r}; "
            f"last_speed_m_s={evidence['touchdown_speed_m_s']!r}"
        )

    def _confirm_final_landed_state(self, evidence: dict[str, Any]) -> None:
        original = self.original_ground_position
        if original is None:
            raise RuntimeError("original grounded position is unavailable")
        maximum_attempts = _confirmation_attempt_budget(
            self.lidar_env_config.final_state_confirmation_timeout_s,
            self.lidar_env_config.final_state_poll_interval_s,
        )
        for attempt in range(1, maximum_attempts + 1):
            state = self._read_named_state()
            position, velocity = self._extract_position_and_velocity(state)
            speed = _speed(velocity)
            api_enabled = self._require_client().isApiControlEnabled(
                vehicle_name=self.vehicle_name
            )
            position_error = calculate_position_error(position, original)
            rejection_reasons: list[str] = []
            if not _is_landed(state):
                rejection_reasons.append("landed state has not converged")
            if speed > _GROUNDED_SPEED_TOLERANCE_M_S:
                rejection_reasons.append("speed above grounded tolerance")
            if position_error > self.lidar_env_config.landing_position_tolerance_m:
                rejection_reasons.append("position outside landing tolerance")
            if api_enabled is not False:
                rejection_reasons.append("API control remains enabled")
            collision_rejection = self._collision_rejection_reason()
            if collision_rejection:
                rejection_reasons.append(collision_rejection)
            evidence.update(
                {
                    "final_landed_state_attempts": attempt,
                    "final_landed_state": getattr(state, "landed_state", None),
                    "final_position": _position_evidence(position),
                    "final_position_error_m": position_error,
                    "final_speed_m_s": speed,
                    "final_api_control_enabled": api_enabled,
                    "api_control_released": api_enabled is False,
                    "returned_to_original_ground": (
                        position_error
                        <= self.lidar_env_config.landing_position_tolerance_m
                    ),
                    "landing_confirmed": not rejection_reasons,
                    "final_rejection_reason": (
                        "; ".join(rejection_reasons) if rejection_reasons else None
                    ),
                }
            )
            if not rejection_reasons:
                return
            if attempt < maximum_attempts:
                self.sleep_fn(self.lidar_env_config.final_state_poll_interval_s)
        raise RuntimeError(
            "final landed-state confirmation timed out; "
            f"attempts={maximum_attempts}; "
            f"last_landed_state={evidence['final_landed_state']!r}; "
            f"last_speed_m_s={evidence['final_speed_m_s']!r}; "
            f"last_position_error_m={evidence['final_position_error_m']!r}"
        )

    def _collision_rejection_reason(self) -> str | None:
        collision = self._require_client().simGetCollisionInfo(
            vehicle_name=self.vehicle_name
        )
        has_collided = getattr(collision, "has_collided", None)
        if not isinstance(has_collided, bool):
            return "collision evidence is unavailable or malformed"
        if not has_collided:
            return None
        timestamp = getattr(collision, "time_stamp", None)
        if (
            self.collision_baseline_timestamp is None
            or timestamp is None
            or timestamp != self.collision_baseline_timestamp
        ):
            return "new or ambiguous collision evidence"
        return None

    def _record_named_cleanup_result(self, result: CleanupResult) -> None:
        if self.primary_cleanup_result is None:
            self.primary_cleanup_result = result
        else:
            self.recovery_cleanup_result = result
        self.last_cleanup_result = result
        self.cleanup_safety_critical_failure_seen = result.safety_critical_failure
        if not result.safety_critical_failure:
            self.cleanup_state = CleanupState()
            self.requires_cleanup = False

    def _owns_control_state(self) -> bool:
        return any(
            (
                self.cleanup_state.api_control_enabled,
                self.cleanup_state.armed,
                self.cleanup_state.takeoff_attempted,
                self.cleanup_state.airborne,
            )
        )


def _augment_observation(
    navigation: np.ndarray, lidar: LidarFeatureResult
) -> np.ndarray:
    observation = np.concatenate(
        (
            navigation.astype(np.float32, copy=False),
            lidar.features,
            np.asarray([lidar.lidar_valid], dtype=np.float32),
        )
    ).astype(np.float32)
    if observation.shape != (LIDAR_OBSERVATION_SIZE,):
        raise RuntimeError("LiDAR observation must have shape (83,)")
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("LiDAR observation contains non-finite values")
    return observation


def _coerce_reset_options(
    options: Mapping[str, Any] | None,
) -> LidarEpisodeResetOptions:
    if options is None:
        return LidarEpisodeResetOptions()
    allowed = {"start_anchor", "goal_approach"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown LiDAR reset options: {sorted(unknown)}")
    return LidarEpisodeResetOptions(
        _coerce_position(options.get("start_anchor"), "start_anchor"),
        _coerce_position(options.get("goal_approach"), "goal_approach"),
    )


def _coerce_position(value: Any, name: str) -> Position3D | None:
    if value is None:
        return None
    if isinstance(value, Position3D):
        result = value
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        result = Position3D(float(value[0]), float(value[1]), float(value[2]))
    else:
        raise ValueError(f"{name} must contain three coordinates")
    if not all(math.isfinite(item) for item in (result.x, result.y, result.z)):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def _speed(velocity: Position3D) -> float:
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _position_evidence(position: Position3D) -> dict[str, float]:
    return {"x": position.x, "y": position.y, "z": position.z}


def _confirmation_attempt_budget(timeout_s: float, interval_s: float) -> int:
    return int(math.floor((timeout_s + 1e-12) / interval_s)) + 1


def _new_lifecycle_evidence() -> dict[str, Any]:
    return {
        "original_ground_position": None,
        "landing_position_tolerance_m": None,
        "collision_baseline_timestamp": None,
        "start_anchor_confirmation": {
            "requested_start_anchor": None,
            "measured_positions": (),
            "measured_start_anchor": None,
            "position_error_m": None,
            "position_tolerance_m": None,
            "measured_speed_m_s": None,
            "speed_tolerance_m_s": None,
            "confirmation_attempts": 0,
            "consecutive_accepted_samples": 0,
            "required_consecutive_samples": 0,
            "rejection_reason": None,
            "confirmation_success": False,
        },
        "cleanup_attempts": (),
        "recovery_retry_required": False,
    }


def _new_cleanup_attempt_evidence(attempt_number: int) -> dict[str, Any]:
    return {
        "attempt_number": attempt_number,
        "return_target": None,
        "return_target_confirmed": False,
        "return_confirmation_attempts": 0,
        "return_airborne_position": None,
        "return_airborne_position_error_m": None,
        "return_airborne_speed_m_s": None,
        "return_consecutive_samples": 0,
        "confirmed_return_airborne_position": None,
        "return_rejection_reason": None,
        "touchdown_confirmation_attempts": 0,
        "touchdown_consecutive_samples": 0,
        "touchdown_position": None,
        "touchdown_position_error_m": None,
        "touchdown_speed_m_s": None,
        "touchdown_rejection_reason": None,
        "physical_touchdown_confirmed": False,
        "landed_state_before_disarm": None,
        "final_landed_state_attempts": 0,
        "final_landed_state": None,
        "final_position": None,
        "final_position_error_m": None,
        "final_speed_m_s": None,
        "final_api_control_enabled": None,
        "api_control_released": False,
        "returned_to_original_ground": False,
        "landing_confirmed": False,
        "final_rejection_reason": None,
        "actions_attempted": (),
        "actions_succeeded": (),
        "errors": (),
        "succeeded": False,
    }


def _attempt_cleanup_action(
    name: str,
    action: Callable[[], Any],
    attempted: list[str],
    succeeded: list[str],
    errors: list[str],
) -> bool:
    attempted.append(name)
    try:
        action()
    except BaseException as exc:
        errors.append(f"{name} raised {type(exc).__name__}: {exc}")
        return False
    succeeded.append(name)
    return True


def _is_landed(state: Any) -> bool:
    value = getattr(state, "landed_state", None)
    return value == 0 or str(value).lower().endswith("landed")
