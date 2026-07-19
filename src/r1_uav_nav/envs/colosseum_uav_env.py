"""Gymnasium wrapper for Colosseum multirotor goal navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from r1_uav_nav.sim import (
    CleanupResult,
    CleanupState,
    ColosseumClientError,
    cleanup_after_control,
    confirm_connection,
    create_multirotor_client,
    import_colosseum_client_module,
    read_collision_status,
    read_multirotor_state,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D, calculate_position_error

OBSERVATION_SIZE = 10
_TERMINATION_NONE = None
_REASON_GOAL_REACHED = "goal_reached"
_REASON_COLLISION = "collision"
_REASON_OUT_OF_BOUNDS = "out_of_bounds"
_REASON_GROUND_CLEARANCE = "ground_clearance_violation"
_REASON_MAX_STEPS = "max_steps"


@dataclass(frozen=True)
class ColosseumUAVEnvConfig:
    """Configuration for the Colosseum Gymnasium UAV environment."""

    client_module: str = "airsim"
    anchor_altitude: float = 2.0
    min_ground_clearance: float = 1.0
    workspace_xy_limit: float = 5.0
    workspace_up_limit: float = 3.0
    workspace_down_limit: float = 0.25
    max_horizontal_velocity: float = 1.0
    max_vertical_velocity: float = 0.5
    control_duration: float = 0.5
    anchor_move_velocity: float = 0.5
    anchor_move_timeout: float = 20.0
    goal_tolerance: float = 0.5
    min_goal_distance: float = 1.0
    max_episode_steps: int = 100
    default_goal_offset: tuple[float, float, float] = (3.0, 0.0, 0.0)
    random_goal: bool = False
    progress_reward_scale: float = 1.0
    step_penalty: float = -0.02
    action_penalty_scale: float = 0.01
    success_reward: float = 10.0
    collision_penalty: float = -10.0
    out_of_bounds_penalty: float = -5.0


@dataclass(frozen=True)
class ColosseumUAVState:
    """Measured Colosseum UAV state used by the Gymnasium wrapper."""

    position: Position3D
    linear_velocity: Position3D
    collision: bool
    ground_reference_z: float
    anchor_position: Position3D
    goal_position: Position3D


class ColosseumUAVEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for Colosseum 3D goal-reaching navigation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ColosseumUAVEnvConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or ColosseumUAVEnvConfig()
        _validate_config(self.config)
        self.client_factory = client_factory
        self.client: Any | None = None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.cleanup_state = CleanupState()
        self.last_cleanup_result: CleanupResult | None = None
        self.cleanup_safety_critical_failure_seen = False
        self.last_terminal_safety_error: str | None = None
        self.ground_reference_z: float | None = None
        self.anchor_position: Position3D | None = None
        self.goal_position: Position3D | None = None
        self.previous_distance_to_goal: float | None = None
        self.step_count = 0
        self.has_reset = False
        self.episode_complete = False
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulator and return the initial observation."""
        if self.closed:
            raise RuntimeError("Cannot reset a closed ColosseumUAVEnv.")

        super().reset(seed=seed)
        options = options or {}
        self._cleanup_previous_episode()
        self._reset_episode_bookkeeping()

        try:
            client = self._get_or_create_client()
            initial_state = self._reset_client_and_read_grounded_state(client)
            client = self._require_client()
            initial_position, _ = self._extract_position_and_velocity(initial_state)
            self.ground_reference_z = initial_position.z
            target_anchor = Position3D(
                initial_position.x,
                initial_position.y,
                initial_position.z - self.config.anchor_altitude,
            )
            self._validate_position_safety(target_anchor)
            goal_offset = self._select_goal_offset(options)
            self._validate_goal_offset_for_planned_anchor(goal_offset, target_anchor)

            self._enable_control_and_takeoff(client)
            self._move_to_anchor(client, target_anchor)

            anchor_state = self._read_state()
            anchor_position, _ = self._extract_position_and_velocity(anchor_state)
            self.anchor_position = anchor_position
            goal_position = self._resolve_goal(anchor_position, goal_offset)
            self._validate_goal_position(anchor_position, goal_position)
            self.goal_position = goal_position

            measured_state = self._build_state(anchor_state)
            distance = calculate_position_error(
                measured_state.position,
                goal_position,
            )
            self.previous_distance_to_goal = distance
            self.has_reset = True
            observation = self._build_observation(measured_state)
            return observation, self._build_info(
                measured_state,
                success=False,
                out_of_bounds=False,
                termination_reason=_TERMINATION_NONE,
            )
        except Exception:
            if self.client is not None:
                self._record_cleanup_result(
                    cleanup_after_control(
                        self.client,
                        self.cleanup_state,
                    )
                )
            self.cleanup_state = CleanupState()
            raise

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a normalized velocity action."""
        self._validate_step_allowed()
        clipped_action = self._validate_and_clip_action(action)
        velocity = self._scale_action_to_velocity(clipped_action)

        client = self._require_client()
        self._move_by_velocity(client, velocity)
        raw_state = self._read_state()
        measured_state = self._build_state(raw_state)
        distance_to_goal = calculate_position_error(
            measured_state.position,
            measured_state.goal_position,
        )
        ground_clearance = self._calculate_ground_clearance(measured_state.position)
        out_of_bounds = self._is_out_of_bounds(measured_state.position)
        ground_violation = ground_clearance < self.config.min_ground_clearance
        success = distance_to_goal <= self.config.goal_tolerance
        terminated, truncated, termination_reason = self._determine_done(
            collision=measured_state.collision,
            ground_violation=ground_violation,
            out_of_bounds=out_of_bounds,
            success=success,
        )
        reward = self._calculate_reward(
            distance_to_goal=distance_to_goal,
            action=clipped_action,
            success=success and termination_reason == _REASON_GOAL_REACHED,
            collision=measured_state.collision,
            out_of_bounds=out_of_bounds or ground_violation,
        )

        self.step_count += 1
        if not terminated and not truncated:
            self.previous_distance_to_goal = distance_to_goal
        else:
            self.episode_complete = True
            self._apply_terminal_hover()

        observation = self._build_observation(measured_state)
        info = self._build_info(
            measured_state,
            success=success and termination_reason == _REASON_GOAL_REACHED,
            out_of_bounds=out_of_bounds or ground_violation,
            termination_reason=termination_reason,
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the environment and clean up simulator control state."""
        self.close_with_result()

    def close_with_result(self) -> CleanupResult | None:
        """Close the environment and return the cleanup result, if any."""
        if self.closed and self.last_cleanup_result is not None:
            return self.last_cleanup_result
        if self.client is not None:
            cleanup_result = cleanup_after_control(
                self.client,
                self.cleanup_state,
            )
            self._record_cleanup_result(cleanup_result)
        self.cleanup_state = CleanupState()
        self.closed = True
        return self.last_cleanup_result

    def _get_or_create_client(self) -> Any:
        if self.client is None:
            self.client = (
                self.client_factory()
                if self.client_factory is not None
                else self._create_default_client()
            )
            confirm_connection(self.client)
        return self.client

    def _create_default_client(self) -> Any:
        client_module = import_colosseum_client_module(self.config.client_module)
        return create_multirotor_client(client_module)

    def _reset_client_and_read_grounded_state(self, client: Any) -> Any:
        try:
            return self._reset_client_once(client)
        except ColosseumClientError:
            self.client = self._recreate_client_once()
            return self._reset_client_once(self.client)

    def _reset_client_once(self, client: Any) -> Any:
        reset_method = getattr(client, "reset", None)
        if reset_method is None:
            raise ColosseumClientError("Client does not provide reset().")
        try:
            reset_method()
            confirm_connection(client)
            return read_multirotor_state(client)
        except Exception as exc:
            raise ColosseumClientError("Simulator reset failed.") from exc

    def _recreate_client_once(self) -> Any:
        if self.client_factory is None:
            client = self._create_default_client()
        else:
            client = self.client_factory()
        confirm_connection(client)
        return client

    def _cleanup_previous_episode(self) -> None:
        if self.client is not None:
            self._record_cleanup_result(
                cleanup_after_control(
                    self.client,
                    self.cleanup_state,
                )
            )
            self.cleanup_state = CleanupState()

    def _record_cleanup_result(self, cleanup_result: CleanupResult) -> None:
        self.last_cleanup_result = _merge_cleanup_results(
            self.last_cleanup_result,
            cleanup_result,
        )
        if cleanup_result.safety_critical_failure:
            self.cleanup_safety_critical_failure_seen = True

    def _reset_episode_bookkeeping(self) -> None:
        self.cleanup_state = CleanupState()
        self.last_terminal_safety_error = None
        self.ground_reference_z = None
        self.anchor_position = None
        self.goal_position = None
        self.previous_distance_to_goal = None
        self.step_count = 0
        self.has_reset = False
        self.episode_complete = False

    def _enable_control_and_takeoff(self, client: Any) -> None:
        self._call_client_method(client, "enableApiControl", True)
        self.cleanup_state = replace(self.cleanup_state, api_control_enabled=True)
        self._call_client_method(client, "armDisarm", True)
        self.cleanup_state = replace(self.cleanup_state, armed=True)
        self.cleanup_state = replace(self.cleanup_state, takeoff_attempted=True)
        self._join_async(self._call_client_method(client, "takeoffAsync"))
        self.cleanup_state = replace(self.cleanup_state, airborne=True)

    def _move_to_anchor(self, client: Any, anchor: Position3D) -> None:
        self._join_async(
            self._call_client_method(
                client,
                "moveToPositionAsync",
                anchor.x,
                anchor.y,
                anchor.z,
                self.config.anchor_move_velocity,
                timeout_sec=self.config.anchor_move_timeout,
            )
        )

    def _move_by_velocity(self, client: Any, velocity: Position3D) -> None:
        self._join_async(
            self._call_client_method(
                client,
                "moveByVelocityAsync",
                velocity.x,
                velocity.y,
                velocity.z,
                self.config.control_duration,
            )
        )

    def _read_state(self) -> Any:
        return read_multirotor_state(self._require_client())

    def _build_state(self, raw_state: Any) -> ColosseumUAVState:
        position, linear_velocity = self._extract_position_and_velocity(raw_state)
        return ColosseumUAVState(
            position=position,
            linear_velocity=linear_velocity,
            collision=read_collision_status(self._require_client(), raw_state),
            ground_reference_z=self._require_ground_reference_z(),
            anchor_position=self._require_anchor_position(),
            goal_position=self._require_goal_position(),
        )

    def _extract_position_and_velocity(
        self,
        raw_state: Any,
    ) -> tuple[Position3D, Position3D]:
        kinematics = getattr(raw_state, "kinematics_estimated", None)
        raw_position = getattr(kinematics, "position", None)
        raw_velocity = getattr(kinematics, "linear_velocity", None)
        position = _extract_vector(raw_position, "position")
        velocity = _extract_vector(raw_velocity, "linear velocity")
        return position, velocity

    def _select_goal_offset(
        self,
        options: dict[str, Any],
    ) -> tuple[float, float, float]:
        if "goal_offset" in options:
            return _coerce_goal_offset(options["goal_offset"])
        if self.config.random_goal:
            return self._sample_random_goal_offset()
        return _coerce_goal_offset(self.config.default_goal_offset)

    def _sample_random_goal_offset(self) -> tuple[float, float, float]:
        low_z = -self.config.workspace_up_limit
        high_z = min(
            self.config.workspace_down_limit,
            self.config.anchor_altitude - self.config.min_ground_clearance,
        )
        for _ in range(100):
            goal_offset = (
                float(
                    self.np_random.uniform(
                        -self.config.workspace_xy_limit,
                        self.config.workspace_xy_limit,
                    )
                ),
                float(
                    self.np_random.uniform(
                        -self.config.workspace_xy_limit,
                        self.config.workspace_xy_limit,
                    )
                ),
                float(self.np_random.uniform(low_z, high_z)),
            )
            if _distance_from_origin(goal_offset) >= self.config.min_goal_distance:
                return goal_offset
        raise ValueError("Could not sample a safe random goal offset.")

    def _resolve_goal(
        self,
        anchor: Position3D,
        offset: tuple[float, float, float],
    ) -> Position3D:
        if _distance_from_origin(offset) < self.config.min_goal_distance:
            raise ValueError("goal_offset must be at least min_goal_distance away.")
        return Position3D(
            anchor.x + offset[0],
            anchor.y + offset[1],
            anchor.z + offset[2],
        )

    def _validate_goal_position(
        self,
        anchor: Position3D,
        goal_position: Position3D,
    ) -> None:
        self._validate_position_safety(goal_position)
        distance_from_anchor = calculate_position_error(anchor, goal_position)
        if distance_from_anchor < self.config.min_goal_distance:
            raise ValueError("goal is too close to the anchor.")

    def _validate_goal_offset_for_planned_anchor(
        self,
        offset: tuple[float, float, float],
        planned_anchor: Position3D,
    ) -> None:
        _validate_goal_offset_for_config(self.config, offset, "goal_offset")
        predicted_goal = Position3D(
            planned_anchor.x + offset[0],
            planned_anchor.y + offset[1],
            planned_anchor.z + offset[2],
        )
        self._validate_position_safety(predicted_goal)

    def _validate_position_safety(self, position: Position3D) -> None:
        ground_reference_z = self._require_ground_reference_z()
        clearance = ground_reference_z - position.z
        if clearance < self.config.min_ground_clearance:
            raise ValueError("position violates minimum ground clearance.")
        if self.anchor_position is not None and self._is_out_of_bounds(position):
            raise ValueError("position is outside configured workspace.")

    def _build_observation(self, state: ColosseumUAVState) -> np.ndarray:
        anchor = state.anchor_position
        goal = state.goal_position
        position = state.position
        config = self.config
        vertical_scale = max(config.workspace_up_limit, config.workspace_down_limit)
        full_vertical_span = config.workspace_up_limit + config.workspace_down_limit
        distance = calculate_position_error(position, goal)
        diagonal = self._workspace_diagonal()

        observation = np.array(
            [
                (position.x - anchor.x) / config.workspace_xy_limit,
                (position.y - anchor.y) / config.workspace_xy_limit,
                (position.z - anchor.z) / vertical_scale,
                (goal.x - position.x) / (2.0 * config.workspace_xy_limit),
                (goal.y - position.y) / (2.0 * config.workspace_xy_limit),
                (goal.z - position.z) / full_vertical_span,
                state.linear_velocity.x / config.max_horizontal_velocity,
                state.linear_velocity.y / config.max_horizontal_velocity,
                state.linear_velocity.z / config.max_vertical_velocity,
                distance / diagonal,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(observation)):
            raise ColosseumClientError("Observation contains non-finite values.")
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _build_info(
        self,
        state: ColosseumUAVState,
        *,
        success: bool,
        out_of_bounds: bool,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "measured_position": _position_tuple(state.position),
            "anchor_position": _position_tuple(state.anchor_position),
            "goal_position": _position_tuple(state.goal_position),
            "distance_to_goal": calculate_position_error(
                state.position,
                state.goal_position,
            ),
            "collision": state.collision,
            "success": success,
            "out_of_bounds": out_of_bounds,
            "ground_reference_z": state.ground_reference_z,
            "ground_clearance": self._calculate_ground_clearance(state.position),
            "step_count": self.step_count,
            "termination_reason": termination_reason,
        }

    def _calculate_reward(
        self,
        *,
        distance_to_goal: float,
        action: np.ndarray,
        success: bool,
        collision: bool,
        out_of_bounds: bool,
    ) -> float:
        previous_distance = self._require_previous_distance()
        progress = previous_distance - distance_to_goal
        reward = (
            self.config.progress_reward_scale * progress
            + self.config.step_penalty
            - self.config.action_penalty_scale * float(np.linalg.norm(action))
        )
        if success:
            reward += self.config.success_reward
        if collision:
            reward += self.config.collision_penalty
        if out_of_bounds:
            reward += self.config.out_of_bounds_penalty
        return float(reward)

    def _determine_done(
        self,
        *,
        collision: bool,
        ground_violation: bool,
        out_of_bounds: bool,
        success: bool,
    ) -> tuple[bool, bool, str | None]:
        if collision:
            return True, False, _REASON_COLLISION
        if ground_violation:
            return True, False, _REASON_GROUND_CLEARANCE
        if out_of_bounds:
            return True, False, _REASON_OUT_OF_BOUNDS
        if success:
            return True, False, _REASON_GOAL_REACHED
        if self.step_count + 1 >= self.config.max_episode_steps:
            return False, True, _REASON_MAX_STEPS
        return False, False, _TERMINATION_NONE

    def _apply_terminal_hover(self) -> None:
        hover_method = getattr(self._require_client(), "hoverAsync", None)
        if hover_method is None:
            return
        try:
            self._join_async(hover_method())
        except Exception as exc:
            self.last_terminal_safety_error = str(exc)

    def _is_out_of_bounds(self, position: Position3D) -> bool:
        anchor = self._require_anchor_position()
        rel_x = position.x - anchor.x
        rel_y = position.y - anchor.y
        rel_z = position.z - anchor.z
        return (
            abs(rel_x) > self.config.workspace_xy_limit
            or abs(rel_y) > self.config.workspace_xy_limit
            or rel_z < -self.config.workspace_up_limit
            or rel_z > self.config.workspace_down_limit
        )

    def _calculate_ground_clearance(self, position: Position3D) -> float:
        return self._require_ground_reference_z() - position.z

    def _scale_action_to_velocity(self, action: np.ndarray) -> Position3D:
        return Position3D(
            float(action[0]) * self.config.max_horizontal_velocity,
            float(action[1]) * self.config.max_horizontal_velocity,
            float(action[2]) * self.config.max_vertical_velocity,
        )

    def _validate_and_clip_action(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (3,):
            raise ValueError("action must have shape (3,)")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain finite values")
        return np.clip(action_array, -1.0, 1.0).astype(np.float32)

    def _validate_step_allowed(self) -> None:
        if self.closed:
            raise RuntimeError("Cannot step a closed ColosseumUAVEnv.")
        if not self.has_reset:
            raise RuntimeError("Call reset before step.")
        if self.episode_complete:
            raise RuntimeError("Episode is complete; call reset before stepping again.")

    def _call_client_method(
        self,
        client: Any,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        method = getattr(client, method_name, None)
        if method is None:
            raise ColosseumClientError(f"Client does not provide {method_name}.")
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            raise ColosseumClientError(f"{method_name} failed.") from exc

    def _join_async(self, async_result: Any) -> None:
        join_method = getattr(async_result, "join", None)
        if join_method is None:
            return
        try:
            join_method()
        except Exception as exc:
            raise ColosseumClientError(
                "Asynchronous simulator command failed."
            ) from exc

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError("Simulator client is not available.")
        return self.client

    def _require_ground_reference_z(self) -> float:
        if self.ground_reference_z is None:
            raise RuntimeError("Ground reference is not available.")
        return self.ground_reference_z

    def _require_anchor_position(self) -> Position3D:
        if self.anchor_position is None:
            raise RuntimeError("Anchor position is not available.")
        return self.anchor_position

    def _require_goal_position(self) -> Position3D:
        if self.goal_position is None:
            raise RuntimeError("Goal position is not available.")
        return self.goal_position

    def _require_previous_distance(self) -> float:
        if self.previous_distance_to_goal is None:
            raise RuntimeError("Previous goal distance is not available.")
        return self.previous_distance_to_goal

    def _workspace_diagonal(self) -> float:
        return math.sqrt(
            (2.0 * self.config.workspace_xy_limit) ** 2
            + (2.0 * self.config.workspace_xy_limit) ** 2
            + (self.config.workspace_up_limit + self.config.workspace_down_limit) ** 2
        )


def _validate_config(config: ColosseumUAVEnvConfig) -> None:
    _require_positive("anchor_altitude", config.anchor_altitude)
    _require_positive("min_ground_clearance", config.min_ground_clearance)
    _require_positive("workspace_xy_limit", config.workspace_xy_limit)
    _require_positive("workspace_up_limit", config.workspace_up_limit)
    _require_positive("workspace_down_limit", config.workspace_down_limit)
    _require_positive("max_horizontal_velocity", config.max_horizontal_velocity)
    _require_positive("max_vertical_velocity", config.max_vertical_velocity)
    _require_positive("control_duration", config.control_duration)
    _require_positive("anchor_move_velocity", config.anchor_move_velocity)
    _require_positive("anchor_move_timeout", config.anchor_move_timeout)
    _require_positive("goal_tolerance", config.goal_tolerance)
    _require_positive("min_goal_distance", config.min_goal_distance)
    _require_finite("progress_reward_scale", config.progress_reward_scale)
    _require_finite("step_penalty", config.step_penalty)
    _require_finite("action_penalty_scale", config.action_penalty_scale)
    _require_finite("success_reward", config.success_reward)
    _require_finite("collision_penalty", config.collision_penalty)
    _require_finite("out_of_bounds_penalty", config.out_of_bounds_penalty)
    if config.anchor_altitude < config.min_ground_clearance:
        raise ValueError("anchor_altitude must satisfy min_ground_clearance")
    if config.min_goal_distance <= config.goal_tolerance:
        raise ValueError("min_goal_distance must be greater than goal_tolerance")
    if not isinstance(config.max_episode_steps, int) or isinstance(
        config.max_episode_steps,
        bool,
    ):
        raise ValueError("max_episode_steps must be a positive integer")
    if config.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    max_reachable_distance = _calculate_max_reachable_goal_distance(config)
    if config.min_goal_distance > max_reachable_distance:
        raise ValueError(
            "min_goal_distance is greater than the maximum reachable workspace "
            "distance"
        )
    default_goal_offset = _coerce_goal_offset(config.default_goal_offset)
    _validate_goal_offset_for_config(config, default_goal_offset, "default_goal_offset")


def _extract_vector(vector: Any, label: str) -> Position3D:
    try:
        position = Position3D(
            float(vector.x_val),
            float(vector.y_val),
            float(vector.z_val),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ColosseumClientError(f"Could not extract {label} from state.") from exc
    if not all(math.isfinite(value) for value in (position.x, position.y, position.z)):
        raise ColosseumClientError(f"{label} contains non-finite values.")
    return position


def _coerce_goal_offset(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("goal_offset must contain three values")
    goal_offset = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in goal_offset):
        raise ValueError("goal_offset must contain finite values")
    return goal_offset


def _distance_from_origin(offset: tuple[float, float, float]) -> float:
    return math.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2)


def _validate_goal_offset_for_config(
    config: ColosseumUAVEnvConfig,
    offset: tuple[float, float, float],
    label: str,
) -> None:
    if _distance_from_origin(offset) < config.min_goal_distance:
        raise ValueError(f"{label} must be at least min_goal_distance away")
    if abs(offset[0]) > config.workspace_xy_limit:
        raise ValueError(f"{label} x offset is outside workspace_xy_limit")
    if abs(offset[1]) > config.workspace_xy_limit:
        raise ValueError(f"{label} y offset is outside workspace_xy_limit")
    if offset[2] < -config.workspace_up_limit:
        raise ValueError(f"{label} z offset exceeds workspace_up_limit")
    if offset[2] > config.workspace_down_limit:
        raise ValueError(f"{label} z offset exceeds workspace_down_limit")
    predicted_clearance = config.anchor_altitude - offset[2]
    if predicted_clearance < config.min_ground_clearance:
        raise ValueError(f"{label} violates minimum ground clearance")


def _calculate_max_reachable_goal_distance(config: ColosseumUAVEnvConfig) -> float:
    safe_down_limit = min(
        config.workspace_down_limit,
        config.anchor_altitude - config.min_ground_clearance,
    )
    max_z_magnitude = max(config.workspace_up_limit, safe_down_limit)
    return math.sqrt(
        config.workspace_xy_limit**2 + config.workspace_xy_limit**2 + max_z_magnitude**2
    )


def _merge_cleanup_results(
    existing: CleanupResult | None,
    new_result: CleanupResult,
) -> CleanupResult:
    if existing is None:
        return new_result
    if not _cleanup_result_has_meaningful_content(new_result):
        return existing
    return CleanupResult(
        actions_attempted=existing.actions_attempted + new_result.actions_attempted,
        actions_succeeded=existing.actions_succeeded + new_result.actions_succeeded,
        errors=existing.errors + new_result.errors,
        safety_critical_failure=(
            existing.safety_critical_failure or new_result.safety_critical_failure
        ),
    )


def _cleanup_result_has_meaningful_content(cleanup_result: CleanupResult) -> bool:
    return bool(
        cleanup_result.actions_attempted
        or cleanup_result.actions_succeeded
        or cleanup_result.errors
    )


def _position_tuple(position: Position3D) -> tuple[float, float, float]:
    return (position.x, position.y, position.z)


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
