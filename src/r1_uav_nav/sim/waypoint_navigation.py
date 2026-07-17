"""Scripted Colosseum waypoint-navigation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from r1_uav_nav.sim.colosseum_client import ColosseumClientError, read_multirotor_state

HORIZONTAL_SQUARE_ROUTE = "horizontal-square"
FIGURE_EIGHT_ROUTE = "figure-eight"
VERTICAL_SQUARE_ROUTE = "vertical-square"
ALL_ROUTES = "all"
ROUTE_SEQUENCE = (HORIZONTAL_SQUARE_ROUTE, FIGURE_EIGHT_ROUTE, VERTICAL_SQUARE_ROUTE)
ROUTE_CHOICES = (*ROUTE_SEQUENCE, ALL_ROUTES)
MIN_FIGURE_EIGHT_SAMPLES = 8
MAX_FIGURE_EIGHT_SAMPLES = 64
DEFAULT_FIGURE_EIGHT_SAMPLES = 13
DEFAULT_VELOCITY = 0.5
DEFAULT_WAYPOINT_TOLERANCE = 0.5
DEFAULT_FIGURE_EIGHT_X_SCALE = 3.0
DEFAULT_FIGURE_EIGHT_Y_SCALE = 2.0
DEFAULT_WAYPOINT_TIMEOUT = 20.0
DEFAULT_CORRECTION_SPEED = 0.25
DEFAULT_CORRECTION_MAX_DURATION = 2.0
DEFAULT_CORRECTION_ATTEMPTS = 3
DEFAULT_CORRECTION_SETTLE_DELAY = 0.1


@dataclass(frozen=True)
class Position3D:
    """Simulator-independent 3D position."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WaypointOffset:
    """Relative waypoint offset from the measured route anchor."""

    dx: float
    dy: float
    dz: float


@dataclass(frozen=True)
class RouteParameters:
    """Configurable route-shape parameters."""

    horizontal_square_side_length: float = 2.0
    figure_eight_x_scale: float = DEFAULT_FIGURE_EIGHT_X_SCALE
    figure_eight_y_scale: float = DEFAULT_FIGURE_EIGHT_Y_SCALE
    figure_eight_samples: int = DEFAULT_FIGURE_EIGHT_SAMPLES
    vertical_square_width: float = 2.0
    vertical_square_height: float = 1.0


@dataclass(frozen=True)
class NavigationConfig:
    """Validated waypoint-navigation settings."""

    route: str = HORIZONTAL_SQUARE_ROUTE
    anchor_altitude: float = 2.0
    velocity: float = DEFAULT_VELOCITY
    waypoint_tolerance: float = DEFAULT_WAYPOINT_TOLERANCE
    waypoint_timeout: float = DEFAULT_WAYPOINT_TIMEOUT
    correction_speed: float = DEFAULT_CORRECTION_SPEED
    correction_max_duration: float = DEFAULT_CORRECTION_MAX_DURATION
    correction_attempts: int = DEFAULT_CORRECTION_ATTEMPTS
    correction_settle_delay: float = DEFAULT_CORRECTION_SETTLE_DELAY
    min_ground_clearance: float = 1.0
    hover_between_routes: float = 1.0
    route_parameters: RouteParameters = RouteParameters()


@dataclass(frozen=True)
class WaypointProgress:
    """Progress record for one requested waypoint."""

    route_name: str
    waypoint_index: int
    num_waypoints: int
    requested_position: Position3D
    measured_position: Position3D
    position_error: float
    collision_detected: bool
    correction_attempt: int = 0


@dataclass(frozen=True)
class RouteExecutionResult:
    """Summary for one scripted route execution."""

    route_name: str
    num_waypoints_requested: int
    num_waypoints_completed: int
    max_position_error: float
    final_position_error: float
    collision_occurred: bool
    returned_to_anchor: bool


ProgressCallback = Callable[[WaypointProgress], None]


def get_route_names() -> tuple[str, ...]:
    """Return executable route names, excluding the suite alias."""
    return ROUTE_SEQUENCE


def validate_navigation_config(config: NavigationConfig) -> None:
    """Validate safety-critical navigation settings before arming."""
    if config.route not in ROUTE_CHOICES:
        raise ValueError(f"unknown route: {config.route}")
    _require_positive("anchor_altitude", config.anchor_altitude)
    _require_positive("velocity", config.velocity)
    _require_positive("waypoint_tolerance", config.waypoint_tolerance)
    _require_positive("waypoint_timeout", config.waypoint_timeout)
    _require_positive("correction_speed", config.correction_speed)
    _require_positive("correction_max_duration", config.correction_max_duration)
    if config.correction_attempts < 0:
        raise ValueError("correction_attempts must be non-negative")
    if config.correction_settle_delay < 0.0:
        raise ValueError("correction_settle_delay must be non-negative")
    _require_positive("min_ground_clearance", config.min_ground_clearance)
    if config.hover_between_routes < 0.0:
        raise ValueError("hover_between_routes must be non-negative")
    _validate_route_parameters(config.route_parameters)


def _validate_route_parameters(route_parameters: RouteParameters) -> None:
    _require_positive(
        "horizontal_square_side_length",
        route_parameters.horizontal_square_side_length,
    )
    _require_positive("figure_eight_x_scale", route_parameters.figure_eight_x_scale)
    _require_positive("figure_eight_y_scale", route_parameters.figure_eight_y_scale)
    if not (
        MIN_FIGURE_EIGHT_SAMPLES
        <= route_parameters.figure_eight_samples
        <= MAX_FIGURE_EIGHT_SAMPLES
    ):
        raise ValueError(
            "figure_eight_samples must be between "
            f"{MIN_FIGURE_EIGHT_SAMPLES} and {MAX_FIGURE_EIGHT_SAMPLES}"
        )
    _require_positive("vertical_square_width", route_parameters.vertical_square_width)
    _require_positive("vertical_square_height", route_parameters.vertical_square_height)


def generate_horizontal_square(side_length: float) -> tuple[WaypointOffset, ...]:
    """Generate a horizontal square route that starts and ends at the anchor."""
    _require_positive("side_length", side_length)
    return (
        WaypointOffset(0.0, 0.0, 0.0),
        WaypointOffset(side_length, 0.0, 0.0),
        WaypointOffset(side_length, side_length, 0.0),
        WaypointOffset(0.0, side_length, 0.0),
        WaypointOffset(0.0, 0.0, 0.0),
    )


def generate_figure_eight(
    x_scale: float,
    y_scale: float,
    samples: int,
) -> tuple[WaypointOffset, ...]:
    """Generate an inclusive Gerono-style horizontal figure-eight route."""
    _require_positive("x_scale", x_scale)
    _require_positive("y_scale", y_scale)
    if not MIN_FIGURE_EIGHT_SAMPLES <= samples <= MAX_FIGURE_EIGHT_SAMPLES:
        raise ValueError(
            "samples must be between "
            f"{MIN_FIGURE_EIGHT_SAMPLES} and {MAX_FIGURE_EIGHT_SAMPLES}"
        )

    offsets: list[WaypointOffset] = []
    for index in range(samples):
        if index == 0 or index == samples - 1:
            offsets.append(WaypointOffset(0.0, 0.0, 0.0))
            continue

        t_value = 2.0 * math.pi * index / (samples - 1)
        offsets.append(
            WaypointOffset(
                x_scale * math.sin(t_value),
                y_scale * math.sin(t_value) * math.cos(t_value),
                0.0,
            )
        )

    return tuple(offsets)


def generate_vertical_square(width: float, height: float) -> tuple[WaypointOffset, ...]:
    """Generate a vertical x-z square route with y fixed."""
    _require_positive("width", width)
    _require_positive("height", height)
    return (
        WaypointOffset(0.0, 0.0, 0.0),
        WaypointOffset(width, 0.0, 0.0),
        WaypointOffset(width, 0.0, -height),
        WaypointOffset(0.0, 0.0, -height),
        WaypointOffset(0.0, 0.0, 0.0),
    )


def get_route_offsets(
    route_name: str,
    route_parameters: RouteParameters,
) -> tuple[WaypointOffset, ...]:
    """Return relative waypoint offsets for a named route."""
    if route_name == HORIZONTAL_SQUARE_ROUTE:
        return generate_horizontal_square(
            route_parameters.horizontal_square_side_length
        )
    if route_name == FIGURE_EIGHT_ROUTE:
        return generate_figure_eight(
            route_parameters.figure_eight_x_scale,
            route_parameters.figure_eight_y_scale,
            route_parameters.figure_eight_samples,
        )
    if route_name == VERTICAL_SQUARE_ROUTE:
        return generate_vertical_square(
            route_parameters.vertical_square_width,
            route_parameters.vertical_square_height,
        )
    raise ValueError(f"unknown route: {route_name}")


def expand_route_selection(route_name: str) -> tuple[str, ...]:
    """Expand a route selection into executable route names."""
    if route_name == ALL_ROUTES:
        return ROUTE_SEQUENCE
    if route_name in ROUTE_SEQUENCE:
        return (route_name,)
    raise ValueError(f"unknown route: {route_name}")


def resolve_waypoints(
    anchor: Position3D,
    offsets: Sequence[WaypointOffset],
) -> tuple[Position3D, ...]:
    """Resolve relative waypoint offsets against a measured anchor."""
    return tuple(
        Position3D(
            anchor.x + offset.dx,
            anchor.y + offset.dy,
            anchor.z + offset.dz,
        )
        for offset in offsets
    )


def validate_route_clearance(
    absolute_waypoints: Sequence[Position3D],
    ground_reference_z: float,
    min_ground_clearance: float,
) -> None:
    """Validate absolute waypoints against measured ground-reference z."""
    _require_positive("min_ground_clearance", min_ground_clearance)
    for waypoint in absolute_waypoints:
        clearance = ground_reference_z - waypoint.z
        if clearance < min_ground_clearance:
            raise ValueError(
                "waypoint violates minimum ground clearance: "
                f"clearance={clearance:.3f}, required={min_ground_clearance:.3f}"
            )


def calculate_position_error(first: Position3D, second: Position3D) -> float:
    """Calculate Euclidean 3D distance between two positions."""
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def extract_position_from_state(state: Any) -> Position3D:
    """Extract an estimated position from an AirSim-style multirotor state."""
    kinematics = getattr(state, "kinematics_estimated", None)
    position = getattr(kinematics, "position", None)
    try:
        return Position3D(
            float(position.x_val),
            float(position.y_val),
            float(position.z_val),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ColosseumClientError("Could not extract position from state.") from exc


def read_collision_status(client: Any, state: Any) -> bool:
    """Read collision status from state or an optional client fallback."""
    state_collision = getattr(state, "collision", None)
    if state_collision is not None and hasattr(state_collision, "has_collided"):
        return bool(state_collision.has_collided)

    collision_method = getattr(client, "simGetCollisionInfo", None)
    if collision_method is None:
        return False

    try:
        collision_info = collision_method()
    except Exception as exc:
        raise ColosseumClientError("Could not read collision info.") from exc

    return bool(getattr(collision_info, "has_collided", False))


def validate_selected_routes_for_anchor(
    config: NavigationConfig,
    anchor: Position3D,
    ground_reference_z: float,
) -> None:
    """Validate every selected route against a proposed or measured anchor."""
    for route_name in expand_route_selection(config.route):
        offsets = get_route_offsets(route_name, config.route_parameters)
        waypoints = resolve_waypoints(anchor, offsets)
        validate_route_clearance(
            waypoints,
            ground_reference_z,
            config.min_ground_clearance,
        )


def execute_route(
    client: Any,
    route_name: str,
    anchor: Position3D,
    ground_reference_z: float,
    config: NavigationConfig,
    progress_callback: ProgressCallback | None = None,
    settle_sleep_fn: Callable[[float], None] | None = None,
) -> RouteExecutionResult:
    """Execute one route and verify each absolute waypoint."""
    offsets = get_route_offsets(route_name, config.route_parameters)
    absolute_waypoints = resolve_waypoints(anchor, offsets)
    validate_route_clearance(
        absolute_waypoints,
        ground_reference_z,
        config.min_ground_clearance,
    )

    max_position_error = 0.0
    final_position_error = math.inf
    completed = 0

    for waypoint_index, requested_position in enumerate(absolute_waypoints, start=1):
        async_result = _call_move_to_position(
            client,
            requested_position,
            config.velocity,
            config.waypoint_timeout,
        )
        _wait_for_async_result(async_result)

        verification = _verify_waypoint(
            client=client,
            route_name=route_name,
            waypoint_index=waypoint_index,
            num_waypoints=len(absolute_waypoints),
            requested_position=requested_position,
            config=config,
            progress_callback=progress_callback,
            settle_sleep_fn=settle_sleep_fn,
        )
        max_position_error = max(max_position_error, verification.max_error)
        final_position_error = verification.final_error
        completed += 1

    returned_to_anchor = final_position_error <= config.waypoint_tolerance
    return RouteExecutionResult(
        route_name=route_name,
        num_waypoints_requested=len(absolute_waypoints),
        num_waypoints_completed=completed,
        max_position_error=max_position_error,
        final_position_error=final_position_error,
        collision_occurred=False,
        returned_to_anchor=returned_to_anchor,
    )


def execute_route_suite(
    client: Any,
    route_names: Sequence[str],
    anchor: Position3D,
    ground_reference_z: float,
    config: NavigationConfig,
    progress_callback: ProgressCallback | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> tuple[RouteExecutionResult, ...]:
    """Execute routes in order, stopping immediately on the first failure."""
    results: list[RouteExecutionResult] = []
    for route_index, route_name in enumerate(route_names):
        result = execute_route(
            client,
            route_name,
            anchor,
            ground_reference_z,
            config,
            progress_callback,
            sleep_fn,
        )
        results.append(result)
        if route_index < len(route_names) - 1 and config.hover_between_routes > 0.0:
            _hover_between_routes(client)
            if sleep_fn is not None:
                sleep_fn(config.hover_between_routes)

    return tuple(results)


def _call_move_to_position(
    client: Any,
    position: Position3D,
    velocity: float,
    waypoint_timeout: float,
) -> Any:
    move_method = getattr(client, "moveToPositionAsync", None)
    if move_method is None:
        raise ColosseumClientError("Client does not provide moveToPositionAsync.")

    try:
        return move_method(
            position.x,
            position.y,
            position.z,
            velocity,
            timeout_sec=waypoint_timeout,
        )
    except Exception as exc:
        raise ColosseumClientError("moveToPositionAsync failed.") from exc


@dataclass(frozen=True)
class _WaypointVerification:
    final_error: float
    max_error: float


def _verify_waypoint(
    *,
    client: Any,
    route_name: str,
    waypoint_index: int,
    num_waypoints: int,
    requested_position: Position3D,
    config: NavigationConfig,
    progress_callback: ProgressCallback | None,
    settle_sleep_fn: Callable[[float], None] | None,
) -> _WaypointVerification:
    measured_position, position_error, collision_detected = _read_waypoint_status(
        client,
        requested_position,
    )
    _emit_progress(
        progress_callback,
        route_name,
        waypoint_index,
        num_waypoints,
        requested_position,
        measured_position,
        position_error,
        collision_detected,
        correction_attempt=0,
    )
    _raise_if_collision(route_name, waypoint_index, collision_detected)
    if position_error <= config.waypoint_tolerance:
        return _WaypointVerification(
            final_error=position_error,
            max_error=position_error,
        )

    max_error = position_error

    for correction_attempt in range(1, config.correction_attempts + 1):
        _run_correction_attempt(
            client,
            requested_position,
            measured_position,
            position_error,
            config,
        )
        if config.correction_settle_delay > 0.0 and settle_sleep_fn is not None:
            settle_sleep_fn(config.correction_settle_delay)

        measured_position, position_error, collision_detected = _read_waypoint_status(
            client,
            requested_position,
        )
        _emit_progress(
            progress_callback,
            route_name,
            waypoint_index,
            num_waypoints,
            requested_position,
            measured_position,
            position_error,
            collision_detected,
            correction_attempt=correction_attempt,
        )
        max_error = max(max_error, position_error)
        _raise_if_collision(route_name, waypoint_index, collision_detected)
        if position_error <= config.waypoint_tolerance:
            return _WaypointVerification(
                final_error=position_error,
                max_error=max_error,
            )

    raise ColosseumClientError(
        f"Waypoint tolerance exceeded during {route_name} at waypoint "
        f"{waypoint_index}: final_error={position_error:.3f}, "
        f"tolerance={config.waypoint_tolerance:.3f}, "
        f"correction_attempts={config.correction_attempts}"
    )


def _read_waypoint_status(
    client: Any,
    requested_position: Position3D,
) -> tuple[Position3D, float, bool]:
    state = read_multirotor_state(client)
    measured_position = extract_position_from_state(state)
    position_error = calculate_position_error(requested_position, measured_position)
    collision_detected = read_collision_status(client, state)
    return measured_position, position_error, collision_detected


def _emit_progress(
    progress_callback: ProgressCallback | None,
    route_name: str,
    waypoint_index: int,
    num_waypoints: int,
    requested_position: Position3D,
    measured_position: Position3D,
    position_error: float,
    collision_detected: bool,
    correction_attempt: int,
) -> None:
    if progress_callback is None:
        return

    progress_callback(
        WaypointProgress(
            route_name=route_name,
            waypoint_index=waypoint_index,
            num_waypoints=num_waypoints,
            requested_position=requested_position,
            measured_position=measured_position,
            position_error=position_error,
            collision_detected=collision_detected,
            correction_attempt=correction_attempt,
        )
    )


def _raise_if_collision(
    route_name: str,
    waypoint_index: int,
    collision_detected: bool,
) -> None:
    if collision_detected:
        raise ColosseumClientError(
            f"Collision detected during {route_name} at waypoint {waypoint_index}."
        )


def _run_correction_attempt(
    client: Any,
    requested_position: Position3D,
    measured_position: Position3D,
    position_error: float,
    config: NavigationConfig,
) -> None:
    move_by_velocity = getattr(client, "moveByVelocityAsync", None)
    if move_by_velocity is None:
        raise ColosseumClientError(
            "Waypoint correction requires moveByVelocityAsync, but the client "
            "does not provide it."
        )

    correction_speed = min(config.velocity, config.correction_speed)
    duration = min(position_error / correction_speed, config.correction_max_duration)
    velocity_scale = correction_speed / position_error
    vx = (requested_position.x - measured_position.x) * velocity_scale
    vy = (requested_position.y - measured_position.y) * velocity_scale
    vz = (requested_position.z - measured_position.z) * velocity_scale

    try:
        _wait_for_async_result(move_by_velocity(vx, vy, vz, duration))
    except Exception as exc:
        raise ColosseumClientError("moveByVelocityAsync correction failed.") from exc

    hover_method = getattr(client, "hoverAsync", None)
    if hover_method is not None:
        _wait_for_async_result(hover_method())


def _hover_between_routes(client: Any) -> None:
    hover_method = getattr(client, "hoverAsync", None)
    if hover_method is None:
        return
    try:
        _wait_for_async_result(hover_method())
    except Exception as exc:
        raise ColosseumClientError("hoverAsync failed between routes.") from exc


def _wait_for_async_result(async_result: Any) -> None:
    join_method = getattr(async_result, "join", None)
    if join_method is None:
        return
    try:
        join_method()
    except Exception as exc:
        raise ColosseumClientError("Asynchronous movement command failed.") from exc


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
