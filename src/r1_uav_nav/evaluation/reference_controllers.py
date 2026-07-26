"""Simulator-independent scripted reference controllers for M13.6."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from r1_uav_nav.planners.voxel_astar import Point3D, VoxelGrid, validate_segment

REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION = 1
ACTION_SHAPE = (3,)
OBSERVATION_SHAPE = (83,)


class ControllerError(ValueError):
    """Raised when a controller input or route violates its contract."""


class ControllerPrivilege(str, Enum):
    """Stable privilege classification for reference-controller evidence."""

    NONE = "non_privileged"
    REFERENCE_PATH = "privileged_reference_path"


@dataclass(frozen=True)
class ControllerStateSpec:
    """Read-only scales needed to decode the protected M13.5 observation."""

    position_scales_m: tuple[float, float, float]
    goal_displacement_scales_m: tuple[float, float, float]
    velocity_scales_m_s: tuple[float, float, float]
    action_low: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    action_high: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "position_scales_m",
            "goal_displacement_scales_m",
            "velocity_scales_m_s",
        ):
            _positive_triplet(getattr(self, name), name)
        _finite_triplet(self.action_low, "action_low")
        _finite_triplet(self.action_high, "action_high")
        if any(
            low >= high
            for low, high in zip(self.action_low, self.action_high, strict=True)
        ):
            raise ControllerError(
                "every action lower bound must be below its upper bound"
            )


@dataclass(frozen=True)
class ControllerStepInput:
    """One immutable controller input with the protected policy observation."""

    observation: np.ndarray
    step_index: int
    info: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation)
        if observation.shape != OBSERVATION_SHAPE:
            raise ControllerError("controller observation must have shape (83,)")
        if observation.dtype != np.float32:
            raise ControllerError("controller observation must use float32")
        if not np.all(np.isfinite(observation)):
            raise ControllerError("controller observation must be finite")
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise ControllerError("step_index must be a non-negative integer")


@dataclass(frozen=True)
class ControllerDecision:
    """A validated bounded action and sanitized controller diagnostics."""

    action: np.ndarray
    target_label: str | None = None
    target_error_m: float | None = None
    waypoint_index: int | None = None

    def __post_init__(self) -> None:
        action = validate_controller_action(self.action)
        object.__setattr__(self, "action", action)
        if self.target_error_m is not None:
            _finite_nonnegative(self.target_error_m, "target_error_m")
        if self.waypoint_index is not None and self.waypoint_index < 0:
            raise ControllerError("waypoint_index must be non-negative")


@runtime_checkable
class ReferenceController(Protocol):
    """Common interface used by offline tests and the live orchestrator."""

    @property
    def controller_id(self) -> str: ...

    @property
    def privilege(self) -> ControllerPrivilege: ...

    @property
    def configuration_digest(self) -> str: ...

    def reset(self) -> None: ...

    def act(self, step_input: ControllerStepInput) -> ControllerDecision: ...


@dataclass(frozen=True)
class RandomControllerConfig:
    """Seeded untrained baseline configuration."""

    schema_version: int = REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION
    seed: int = 0

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ControllerError("random controller seed must be an integer")


class RandomController:
    """Observation-independent baseline using only a local NumPy generator."""

    def __init__(self, config: RandomControllerConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    @property
    def controller_id(self) -> str:
        return "random"

    @property
    def privilege(self) -> ControllerPrivilege:
        return ControllerPrivilege.NONE

    @property
    def configuration_digest(self) -> str:
        return _digest(asdict(self.config))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def act(self, step_input: ControllerStepInput) -> ControllerDecision:
        _ = step_input
        action = self._rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        return ControllerDecision(action)


@dataclass(frozen=True)
class DirectGoalControllerConfig:
    """Navigation-prefix-only proportional controller configuration."""

    schema_version: int = REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION
    proportional_gain: float = 1.0
    velocity_damping: float = 0.15
    cruise_speed_m_s: float = 0.50
    slowdown_radius_m: float = 1.0

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _finite_positive(self.proportional_gain, "proportional_gain")
        _finite_nonnegative(self.velocity_damping, "velocity_damping")
        _finite_positive(self.cruise_speed_m_s, "cruise_speed_m_s")
        _finite_positive(self.slowdown_radius_m, "slowdown_radius_m")


class DirectGoalController:
    """Non-privileged controller using only goal displacement and velocity."""

    def __init__(
        self,
        config: DirectGoalControllerConfig,
        state_spec: ControllerStateSpec,
    ) -> None:
        self.config = config
        self.state_spec = state_spec

    @property
    def controller_id(self) -> str:
        return "direct"

    @property
    def privilege(self) -> ControllerPrivilege:
        return ControllerPrivilege.NONE

    @property
    def configuration_digest(self) -> str:
        return _digest(
            {"config": asdict(self.config), "state_spec": asdict(self.state_spec)}
        )

    def reset(self) -> None:
        return None

    def act(self, step_input: ControllerStepInput) -> ControllerDecision:
        observation = step_input.observation
        displacement = np.asarray(observation[3:6], dtype=np.float64) * np.asarray(
            self.state_spec.goal_displacement_scales_m
        )
        velocity = np.asarray(observation[6:9], dtype=np.float64) * np.asarray(
            self.state_spec.velocity_scales_m_s
        )
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-9:
            return ControllerDecision(
                np.zeros(3, dtype=np.float32),
                target_label="goal",
                target_error_m=distance,
            )
        speed = self.config.cruise_speed_m_s * min(
            1.0, distance / self.config.slowdown_radius_m
        )
        desired_velocity = displacement / distance * speed
        command_velocity = (
            self.config.proportional_gain * desired_velocity
            - self.config.velocity_damping * velocity
        )
        action = _velocity_to_action(command_velocity, self.state_spec)
        return ControllerDecision(
            action,
            target_label="goal",
            target_error_m=distance,
        )


@dataclass(frozen=True)
class OracleWaypointControllerConfig:
    """Privileged deterministic reference-path follower configuration."""

    schema_version: int = REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION
    maximum_segment_length_m: float = 1.0
    waypoint_tolerance_m: float = 0.30
    proportional_gain: float = 1.0
    velocity_damping: float = 0.15
    cruise_speed_m_s: float = 0.40
    collinearity_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in (
            "maximum_segment_length_m",
            "waypoint_tolerance_m",
            "proportional_gain",
            "cruise_speed_m_s",
            "collinearity_tolerance",
        ):
            _finite_positive(getattr(self, name), name)
        _finite_nonnegative(self.velocity_damping, "velocity_damping")


class OracleWaypointController:
    """Privileged A* reference-path controller for course executability evidence."""

    def __init__(
        self,
        config: OracleWaypointControllerConfig,
        state_spec: ControllerStateSpec,
        *,
        reference_path: Sequence[Point3D],
        grid: VoxelGrid,
    ) -> None:
        self.config = config
        self.state_spec = state_spec
        self.route = compress_reference_path(
            reference_path,
            maximum_segment_length_m=config.maximum_segment_length_m,
            collinearity_tolerance=config.collinearity_tolerance,
            grid=grid,
        )
        self.route_offsets = tuple(
            tuple(
                value - origin
                for value, origin in zip(point, self.route[0], strict=True)
            )
            for point in self.route
        )
        self.waypoint_index = 1 if len(self.route) > 1 else 0
        self.completed_waypoints = 0

    @property
    def controller_id(self) -> str:
        return "oracle"

    @property
    def privilege(self) -> ControllerPrivilege:
        return ControllerPrivilege.REFERENCE_PATH

    @property
    def configuration_digest(self) -> str:
        return _digest(
            {"config": asdict(self.config), "state_spec": asdict(self.state_spec)}
        )

    @property
    def route_digest(self) -> str:
        return _digest(
            {
                "schema_version": REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION,
                "route": self.route,
            }
        )

    @property
    def waypoint_count(self) -> int:
        return len(self.route)

    def reset(self) -> None:
        self.waypoint_index = 1 if len(self.route) > 1 else 0
        self.completed_waypoints = 0

    def act(self, step_input: ControllerStepInput) -> ControllerDecision:
        observation = step_input.observation
        current_offset = np.asarray(observation[0:3], dtype=np.float64) * np.asarray(
            self.state_spec.position_scales_m
        )
        velocity = np.asarray(observation[6:9], dtype=np.float64) * np.asarray(
            self.state_spec.velocity_scales_m_s
        )

        while self.waypoint_index < len(self.route_offsets) - 1:
            error = float(
                np.linalg.norm(
                    np.asarray(self.route_offsets[self.waypoint_index]) - current_offset
                )
            )
            if error > self.config.waypoint_tolerance_m:
                break
            self.completed_waypoints += 1
            self.waypoint_index += 1

        target = np.asarray(self.route_offsets[self.waypoint_index])
        displacement = target - current_offset
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-9:
            action = np.zeros(3, dtype=np.float32)
        else:
            speed = self.config.cruise_speed_m_s * min(
                1.0, distance / self.config.waypoint_tolerance_m
            )
            desired_velocity = displacement / distance * speed
            action = _velocity_to_action(
                self.config.proportional_gain * desired_velocity
                - self.config.velocity_damping * velocity,
                self.state_spec,
            )
        return ControllerDecision(
            action,
            target_label=f"oracle-waypoint-{self.waypoint_index:03d}",
            target_error_m=distance,
            waypoint_index=self.waypoint_index,
        )


def compress_reference_path(
    reference_path: Sequence[Point3D],
    *,
    maximum_segment_length_m: float,
    collinearity_tolerance: float = 1e-8,
    grid: VoxelGrid | None = None,
) -> tuple[Point3D, ...]:
    """Preserve endpoints and turns, split long segments, and revalidate them."""
    _finite_positive(maximum_segment_length_m, "maximum_segment_length_m")
    _finite_positive(collinearity_tolerance, "collinearity_tolerance")
    points = tuple(_point(point, "reference path point") for point in reference_path)
    if len(points) < 2:
        raise ControllerError("oracle reference path must contain at least two points")
    if _distance(points[0], points[-1]) <= collinearity_tolerance:
        raise ControllerError("oracle reference path endpoints must be distinct")

    turns: list[Point3D] = [points[0]]
    for index in range(1, len(points) - 1):
        first = np.asarray(points[index]) - np.asarray(points[index - 1])
        second = np.asarray(points[index + 1]) - np.asarray(points[index])
        if np.linalg.norm(first) <= collinearity_tolerance:
            continue
        if np.linalg.norm(second) <= collinearity_tolerance:
            continue
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        if np.linalg.norm(first - second) > collinearity_tolerance:
            turns.append(points[index])
    turns.append(points[-1])

    route: list[Point3D] = [turns[0]]
    for start, end in zip(turns, turns[1:], strict=False):
        length = _distance(start, end)
        if length <= collinearity_tolerance:
            continue
        subdivisions = max(1, math.ceil(length / maximum_segment_length_m))
        for index in range(1, subdivisions + 1):
            fraction = index / subdivisions
            route.append(
                tuple(
                    start_value + (end_value - start_value) * fraction
                    for start_value, end_value in zip(start, end, strict=True)
                )
            )

    if len(route) < 2:
        raise ControllerError("oracle route compression produced no usable route")
    if grid is not None:
        for start, end in zip(route, route[1:], strict=False):
            result = validate_segment(grid, start, end)
            if not result.clear:
                raise ControllerError(
                    "compressed oracle route failed segment validation: "
                    f"{result.reason}"
                )
    return tuple(route)


def validate_controller_action(action: Any) -> np.ndarray:
    """Return one copied finite float32 action within the fixed action contract."""
    value = np.asarray(action)
    if value.shape != ACTION_SHAPE:
        raise ControllerError("controller action must have shape (3,)")
    if not np.issubdtype(value.dtype, np.number):
        raise ControllerError("controller action must be numeric")
    value = value.astype(np.float32, copy=True)
    if not np.all(np.isfinite(value)):
        raise ControllerError("controller action must be finite")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ControllerError("controller action must remain within [-1, 1]")
    return value


def _velocity_to_action(
    velocity: np.ndarray, state_spec: ControllerStateSpec
) -> np.ndarray:
    action = np.asarray(velocity, dtype=np.float64) / np.asarray(
        state_spec.velocity_scales_m_s
    )
    maximum = float(np.max(np.abs(action)))
    if maximum > 1.0:
        action /= maximum
    return validate_controller_action(action.astype(np.float32))


def _point(value: Sequence[float], name: str) -> Point3D:
    if len(value) != 3:
        raise ControllerError(f"{name} must contain three coordinates")
    point = tuple(float(item) for item in value)
    _finite_triplet(point, name)
    return point  # type: ignore[return-value]


def _distance(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        sum(
            (second_value - first_value) ** 2
            for first_value, second_value in zip(first, second, strict=True)
        )
    )


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema(value: int) -> None:
    if value != REFERENCE_CONTROLLER_CONFIG_SCHEMA_VERSION:
        raise ControllerError("unsupported reference-controller schema version")


def _finite_triplet(value: Sequence[float], name: str) -> None:
    if len(value) != 3 or any(not math.isfinite(float(item)) for item in value):
        raise ControllerError(f"{name} must contain three finite values")


def _positive_triplet(value: Sequence[float], name: str) -> None:
    _finite_triplet(value, name)
    if any(float(item) <= 0 for item in value):
        raise ControllerError(f"{name} values must be positive")


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ControllerError(f"{name} must be finite and positive")


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ControllerError(f"{name} must be finite and non-negative")
