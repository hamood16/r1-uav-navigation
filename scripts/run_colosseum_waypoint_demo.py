"""Run scripted Colosseum waypoint-navigation routes."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any, Sequence

from r1_uav_nav.sim import (
    CleanupResult,
    CleanupState,
    ColosseumClientError,
    ColosseumClientImportError,
    cleanup_after_control,
    confirm_connection,
    create_multirotor_client,
    import_colosseum_client_module,
    read_multirotor_state,
)
from r1_uav_nav.sim.waypoint_navigation import (
    DEFAULT_FIGURE_EIGHT_SAMPLES,
    DEFAULT_FIGURE_EIGHT_X_SCALE,
    DEFAULT_FIGURE_EIGHT_Y_SCALE,
    DEFAULT_VELOCITY,
    DEFAULT_WAYPOINT_TIMEOUT,
    DEFAULT_WAYPOINT_TOLERANCE,
    HORIZONTAL_SQUARE_ROUTE,
    ROUTE_CHOICES,
    NavigationConfig,
    Position3D,
    RouteExecutionResult,
    RouteParameters,
    WaypointProgress,
    execute_route_suite,
    expand_route_selection,
    extract_position_from_state,
    validate_navigation_config,
    validate_selected_routes_for_anchor,
)

DEFAULT_CLIENT_MODULE = "airsim"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse waypoint-demo CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run scripted Colosseum waypoint-navigation routes.",
    )
    parser.add_argument("--client-module", default=DEFAULT_CLIENT_MODULE)
    parser.add_argument(
        "--route",
        choices=ROUTE_CHOICES,
        default=HORIZONTAL_SQUARE_ROUTE,
    )
    parser.add_argument("--anchor-altitude", type=float, default=2.0)
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY)
    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=DEFAULT_WAYPOINT_TOLERANCE,
    )
    parser.add_argument(
        "--waypoint-timeout",
        type=float,
        default=DEFAULT_WAYPOINT_TIMEOUT,
    )
    parser.add_argument("--horizontal-square-side-length", type=float, default=2.0)
    parser.add_argument(
        "--figure-eight-x-scale",
        type=float,
        default=DEFAULT_FIGURE_EIGHT_X_SCALE,
    )
    parser.add_argument(
        "--figure-eight-y-scale",
        type=float,
        default=DEFAULT_FIGURE_EIGHT_Y_SCALE,
    )
    parser.add_argument(
        "--figure-eight-samples",
        type=int,
        default=DEFAULT_FIGURE_EIGHT_SAMPLES,
    )
    parser.add_argument("--vertical-square-width", type=float, default=2.0)
    parser.add_argument("--vertical-square-height", type=float, default=1.0)
    parser.add_argument("--min-ground-clearance", type=float, default=1.0)
    parser.add_argument("--hover-between-routes", type=float, default=1.0)
    return parser.parse_args(argv)


def build_navigation_config(args: argparse.Namespace) -> NavigationConfig:
    """Convert parsed CLI args into a typed navigation config."""
    route_parameters = RouteParameters(
        horizontal_square_side_length=args.horizontal_square_side_length,
        figure_eight_x_scale=args.figure_eight_x_scale,
        figure_eight_y_scale=args.figure_eight_y_scale,
        figure_eight_samples=args.figure_eight_samples,
        vertical_square_width=args.vertical_square_width,
        vertical_square_height=args.vertical_square_height,
    )
    config = NavigationConfig(
        route=args.route,
        anchor_altitude=args.anchor_altitude,
        velocity=args.velocity,
        waypoint_tolerance=args.waypoint_tolerance,
        waypoint_timeout=args.waypoint_timeout,
        min_ground_clearance=args.min_ground_clearance,
        hover_between_routes=args.hover_between_routes,
        route_parameters=route_parameters,
    )
    validate_navigation_config(config)
    return config


def main() -> int:
    """Run the waypoint-navigation demo."""
    args = parse_args()
    return run_waypoint_demo(args, sleep_fn=time.sleep)


def run_waypoint_demo(
    args: argparse.Namespace,
    sleep_fn: Any = time.sleep,
) -> int:
    """Run the waypoint-navigation demo from parsed arguments."""
    cleanup_state = CleanupState()
    cleanup_result: CleanupResult | None = None
    operation_failed = False
    client: Any | None = None

    try:
        config = build_navigation_config(args)
        client_module = import_colosseum_client_module(args.client_module)
        client = create_multirotor_client(client_module)
        confirm_connection(client)

        initial_state = read_multirotor_state(client)
        initial_position = extract_position_from_state(initial_state)
        ground_reference_z = initial_position.z
        target_anchor = Position3D(
            initial_position.x,
            initial_position.y,
            initial_position.z - config.anchor_altitude,
        )
        validate_selected_routes_for_anchor(config, target_anchor, ground_reference_z)

        _call_client_method(client, "enableApiControl", True)
        cleanup_state = replace(cleanup_state, api_control_enabled=True)
        _call_client_method(client, "armDisarm", True)
        cleanup_state = replace(cleanup_state, armed=True)

        cleanup_state = replace(cleanup_state, takeoff_attempted=True)
        _join_async_result(_call_client_method(client, "takeoffAsync"))
        cleanup_state = replace(cleanup_state, airborne=True)

        _join_async_result(
            _call_client_method(
                client,
                "moveToPositionAsync",
                target_anchor.x,
                target_anchor.y,
                target_anchor.z,
                config.velocity,
                timeout_sec=config.waypoint_timeout,
            )
        )
        anchor_state = read_multirotor_state(client)
        anchor = extract_position_from_state(anchor_state)
        validate_selected_routes_for_anchor(config, anchor, ground_reference_z)

        print("Initial position:", _format_position(initial_position))
        print("Measured route anchor:", _format_position(anchor))
        route_names = expand_route_selection(config.route)
        results = execute_route_suite(
            client,
            route_names,
            anchor,
            ground_reference_z,
            config,
            progress_callback=_print_progress,
            sleep_fn=sleep_fn,
        )
        final_state = read_multirotor_state(client)
        final_position = extract_position_from_state(final_state)
        print("Final airborne position:", _format_position(final_position))
        _print_results(results)
    except (
        ColosseumClientError,
        ColosseumClientImportError,
        ValueError,
        AttributeError,
    ) as exc:
        operation_failed = True
        print(f"Colosseum waypoint demo failed: {exc}")
    except KeyboardInterrupt:
        operation_failed = True
        print("Colosseum waypoint demo interrupted by user.")
    finally:
        cleanup_result = (
            cleanup_after_control(client, cleanup_state) if client is not None else None
        )
        if cleanup_result is not None:
            _print_cleanup_result(cleanup_result)

    if operation_failed:
        return 1
    if cleanup_result is not None and cleanup_result.safety_critical_failure:
        print("Colosseum waypoint demo failed during safety-critical cleanup.")
        return 1

    print("Colosseum waypoint demo complete.")
    return 0


def _call_client_method(
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


def _join_async_result(async_result: object) -> None:
    join_method = getattr(async_result, "join", None)
    if join_method is None:
        return
    try:
        join_method()
    except Exception as exc:
        raise ColosseumClientError("Asynchronous simulator command failed.") from exc


def _print_progress(progress: WaypointProgress) -> None:
    print(
        f"[{progress.route_name}] waypoint "
        f"{progress.waypoint_index}/{progress.num_waypoints}: "
        f"target={_format_position(progress.requested_position)} "
        f"measured={_format_position(progress.measured_position)} "
        f"error={progress.position_error:.3f} "
        f"collision={progress.collision_detected}"
    )


def _print_results(results: Sequence[RouteExecutionResult]) -> None:
    print("Route results:")
    for result in results:
        print(
            f"- {result.route_name}: completed "
            f"{result.num_waypoints_completed}/{result.num_waypoints_requested}, "
            f"max_error={result.max_position_error:.3f}, "
            f"final_error={result.final_position_error:.3f}, "
            f"collision={result.collision_occurred}, "
            f"returned_to_anchor={result.returned_to_anchor}"
        )


def _print_cleanup_result(cleanup_result: CleanupResult) -> None:
    if cleanup_result.actions_succeeded:
        print("Cleanup actions:")
        for action in cleanup_result.actions_succeeded:
            print(f"- {action}")
    if cleanup_result.errors:
        print("Cleanup errors:")
        for error in cleanup_result.errors:
            print(f"- {error}")


def _format_position(position: Position3D) -> str:
    return f"(x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f})"


if __name__ == "__main__":
    raise SystemExit(main())
