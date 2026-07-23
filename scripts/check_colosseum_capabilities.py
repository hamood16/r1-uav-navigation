"""Run staged Colosseum capability probes with explicit safety authorization."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from r1_uav_nav.sim.colosseum_capabilities import (
    DEFAULT_REPORTS_DIR,
    AirborneProbeConfig,
    CapabilityObservation,
    CapabilityProbeError,
    CapabilityProbeReport,
    CleanupDomainResult,
    ConnectionProbeConfig,
    GroundedLidarProbeConfig,
    LidarProbeConfig,
    PerformanceProbeConfig,
    ProbeRuntimeState,
    SceneMutationConfig,
    SceneSurveyConfig,
    cleanup_probe_domains,
    create_probe_client,
    generate_report_path,
    inspect_active_lidar_settings,
    inspect_client_capabilities,
    load_client_module,
    prepare_airborne_probe,
    probe_debug_markers,
    probe_grounded_lidar,
    probe_lidar,
    probe_performance,
    probe_scene_mutation,
    save_capability_report,
    survey_scene,
    validate_grounded_preflight,
    validate_marker_hold_seconds,
    validate_material_name,
    validate_mutation_hold_seconds,
    validate_report_output_path,
)
from r1_uav_nav.sim.colosseum_client import ColosseumClientImportError

DEFAULT_COMMAND = "inspect-client"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse capability-probe arguments without importing the external client."""
    parser = argparse.ArgumentParser(
        description="Inspect and safely probe Colosseum/AirSim capabilities.",
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")

    inspect_client = subparsers.add_parser(
        "inspect-client",
        help="Inspect client methods without constructing a simulator client.",
    )
    _add_common_arguments(inspect_client, suppress_defaults=True)

    survey = subparsers.add_parser(
        "survey",
        help="Run a read-only scene and settings survey.",
    )
    _add_common_arguments(survey, suppress_defaults=True)
    survey.add_argument("--object-regex", default=".*")
    survey.add_argument("--max-objects", type=int, default=100)
    survey.add_argument("--confirm-no-visible-collision", action="store_true")

    markers = subparsers.add_parser(
        "markers",
        help="Create and clean up temporary persistent debug markers.",
    )
    _add_common_arguments(markers, suppress_defaults=True)
    markers.add_argument("--allow-debug-markers", action="store_true")
    markers.add_argument("--allow-marker-flush", action="store_true")
    markers.add_argument("--marker-hold-seconds", type=float, default=0.0)

    mutation = subparsers.add_parser(
        "mutation",
        help="Exercise one uniquely named temporary scene object.",
    )
    _add_common_arguments(mutation, suppress_defaults=True)
    mutation.add_argument("--asset-name", required=True)
    mutation.add_argument("--allow-scene-mutation", action="store_true")
    mutation.add_argument("--confirm-spawn-area-clear", action="store_true")
    mutation.add_argument("--confirm-vehicle-disarmed", action="store_true")
    mutation.add_argument(
        "--spawn-offset", type=_parse_vector, default=(3.0, 3.0, -1.0)
    )
    mutation.add_argument("--material-name", default=None)
    mutation.add_argument("--mutation-hold-seconds", type=float, default=0.0)

    lidar = subparsers.add_parser(
        "lidar",
        help="Validate bounded raw LiDAR scans after controlled takeoff.",
    )
    _add_common_arguments(lidar, suppress_defaults=True)
    _add_airborne_arguments(lidar)
    lidar.add_argument("--lidar-name", default="")
    lidar.add_argument("--scan-count", type=int, default=20)
    lidar.add_argument("--scan-interval", type=float, default=0.2)
    lidar.add_argument("--stale-threshold", type=int, default=3)
    lidar.add_argument("--warm-up-attempts", type=int, default=10)
    lidar.add_argument("--warm-up-interval", type=float, default=0.2)
    lidar.add_argument("--settle-interval", type=float, default=0.5)
    lidar.add_argument("--coordinate-frame-experiment", action="store_true")
    lidar.add_argument("--allow-coordinate-motion", action="store_true")
    lidar.add_argument("--coordinate-scan-attempts", type=int, default=5)
    lidar.add_argument("--yaw-return-tolerance", type=float, default=5.0)
    lidar.add_argument("--visualize-lidar", action="store_true")
    lidar.add_argument("--allow-marker-flush", action="store_true")
    lidar.add_argument(
        "--lidar-visualization-hold-seconds",
        type=float,
        default=8.0,
    )
    lidar.add_argument(
        "--lidar-visualization-max-points",
        type=int,
        default=2000,
    )
    lidar.add_argument(
        "--lidar-visualization-max-rays",
        type=int,
        default=64,
    )

    grounded_lidar = subparsers.add_parser(
        "grounded-lidar",
        help="Validate bounded raw LiDAR scans without acquiring UAV control.",
    )
    _add_common_arguments(grounded_lidar, suppress_defaults=True)
    grounded_lidar.add_argument("--lidar-name", required=True)
    grounded_lidar.add_argument("--scan-count", type=int, default=20)
    grounded_lidar.add_argument("--scan-interval", type=float, default=0.2)
    grounded_lidar.add_argument("--stale-threshold", type=int, default=3)
    grounded_lidar.add_argument("--warm-up-attempts", type=int, default=10)
    grounded_lidar.add_argument("--warm-up-interval", type=float, default=0.2)
    grounded_lidar.add_argument("--confirm-no-visible-collision", action="store_true")

    performance = subparsers.add_parser(
        "performance",
        help="Measure bounded simulator RPC rates.",
    )
    _add_common_arguments(performance, suppress_defaults=True)
    _add_airborne_arguments(performance)
    performance.add_argument("--iterations", type=int, default=20)
    performance.add_argument("--include-lidar", action="store_true")
    performance.add_argument("--lidar-name", default="")
    performance.add_argument("--include-control", action="store_true")
    performance.add_argument("--control-duration", type=float, default=0.1)
    performance.add_argument("--probe-pause", action="store_true")
    performance.add_argument("--allow-pause", action="store_true")

    parser.set_defaults(command=DEFAULT_COMMAND)
    return parser.parse_args(argv)


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Add shared options without letting subparsers overwrite root values."""

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument("--client-module", default=default("airsim"))
    parser.add_argument("--host", default=default("127.0.0.1"))
    parser.add_argument("--port", type=int, default=default(41451))
    parser.add_argument("--rpc-timeout", type=float, default=default(30.0))
    parser.add_argument("--vehicle-name", default=default(""))
    parser.add_argument("--output-dir", type=Path, default=default(DEFAULT_REPORTS_DIR))
    parser.add_argument("--output-path", type=Path, default=default(None))


def main() -> int:
    """Run one probe mode and return a process exit status."""
    return run_probe(parse_args())


def run_probe(args: argparse.Namespace, *, repository_root: Path | None = None) -> int:
    """Run a parsed probe command with cleanup and report preservation."""
    root = (repository_root or Path.cwd()).resolve()
    started = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex
    observations: list[CapabilityObservation] = []
    data: dict[str, Any] = {}
    errors: list[str] = []
    cleanup_results: tuple[CleanupDomainResult, ...] = ()
    runtime = ProbeRuntimeState()
    client: Any | None = None
    interrupted = False

    try:
        connection = _build_connection_config(args)
        _validate_command_authorization(args)
        grounded_lidar_config = (
            _build_grounded_lidar_config(args)
            if args.command == "grounded-lidar"
            else None
        )
        airborne_config = (
            _build_airborne_config(args)
            if args.command in {"lidar", "performance"}
            else None
        )
        lidar_config = (
            _build_lidar_config(args, airborne_config)
            if args.command == "lidar" and airborne_config is not None
            else None
        )
        performance_config = (
            _build_performance_config(args, airborne_config)
            if args.command == "performance" and airborne_config is not None
            else None
        )
        client_module = load_client_module(connection.client_module)
        observations.extend(inspect_client_capabilities(client_module))
        data["client_module_version"] = str(
            getattr(client_module, "__version__", "unknown")
        )

        if args.command != "inspect-client":
            client = create_probe_client(client_module, connection)
            _confirm_connection(client)
            data["connection"] = _read_version_evidence(client)
            data["selected_vehicle_name"] = connection.vehicle_name

        if args.command == "survey":
            survey_config = SceneSurveyConfig(
                connection=connection,
                object_regex=args.object_regex,
                max_objects=args.max_objects,
                confirm_no_visible_collision=args.confirm_no_visible_collision,
            )
            survey_observations, survey_data = survey_scene(
                client,
                survey_config,
                client_module=client_module,
            )
            observations.extend(survey_observations)
            data["survey"] = survey_data
            if not survey_data["measured_state"]["safe_for_later_stages"]:
                raise CapabilityProbeError(
                    "Read-only survey found an unsafe vehicle state; later stages "
                    "must not proceed."
                )
        elif args.command == "markers":
            state = _require_client(client).getMultirotorState()
            position = validate_grounded_preflight(
                client,
                client_module,
                state,
                operator_confirmed_stable=True,
            )
            observations.extend(
                probe_debug_markers(
                    client,
                    client_module,
                    position,
                    runtime,
                    allow_debug_markers=args.allow_debug_markers,
                    allow_marker_flush=args.allow_marker_flush,
                    marker_hold_seconds=args.marker_hold_seconds,
                )
            )
            data["markers"] = {
                "derived_from_measured_position": True,
                "hold_seconds": args.marker_hold_seconds,
                "operator_visibility_confirmation": "pending",
            }
        elif args.command == "mutation":
            mutation_config = SceneMutationConfig(
                asset_name=args.asset_name,
                allow_scene_mutation=args.allow_scene_mutation,
                confirm_spawn_area_clear=args.confirm_spawn_area_clear,
                confirm_vehicle_disarmed=args.confirm_vehicle_disarmed,
                spawn_offset=args.spawn_offset,
                material_name=args.material_name,
                mutation_hold_seconds=args.mutation_hold_seconds,
            )
            mutation_observations, mutation_data = probe_scene_mutation(
                client, client_module, mutation_config, runtime
            )
            observations.extend(mutation_observations)
            data["mutation"] = mutation_data
            data["operator_object_confirmation"] = "pending"
        elif args.command == "lidar":
            if airborne_config is None or lidar_config is None:
                raise CapabilityProbeError("Airborne LiDAR configuration is missing.")
            settings_verification = inspect_active_lidar_settings(
                client,
                airborne_config.vehicle_name,
                lidar_config.lidar_name,
            )
            data["airborne_settings"] = settings_verification
            if not settings_verification["profile_matches"]:
                raise CapabilityProbeError(
                    "Active settings do not match the exact provisional M13.1 profile."
                )
            context = prepare_airborne_probe(
                client,
                client_module,
                airborne_config,
                runtime,
                lidar_config.lidar_name,
                settings_verification=settings_verification,
            )
            lidar_observations, lidar_data = probe_lidar(
                client,
                lidar_config,
                context,
                client_module=client_module,
                runtime=runtime,
            )
            observations.extend(lidar_observations)
            data["airborne_context"] = asdict(context)
            data["lidar"] = lidar_data
            if not lidar_data["airborne_scan_gate_passed"]:
                raise CapabilityProbeError(
                    "Airborne LiDAR evidence did not pass the strict scan gate."
                )
        elif args.command == "grounded-lidar":
            if grounded_lidar_config is None:
                raise CapabilityProbeError("Grounded LiDAR configuration is missing.")
            grounded_observations, grounded_data = probe_grounded_lidar(
                client,
                client_module,
                grounded_lidar_config,
            )
            observations.extend(grounded_observations)
            data["grounded_lidar"] = grounded_data
            if not grounded_data["ready_for_airborne_validation"]:
                raise CapabilityProbeError(
                    "Grounded LiDAR evidence did not pass the airborne gate."
                )
        elif args.command == "performance":
            if performance_config is None:
                raise CapabilityProbeError("Performance configuration is missing.")
            context = None
            if args.include_control:
                if airborne_config is None:
                    raise CapabilityProbeError(
                        "Airborne performance configuration is missing."
                    )
                settings_verification = inspect_active_lidar_settings(
                    client,
                    airborne_config.vehicle_name,
                    args.lidar_name,
                )
                data["airborne_settings"] = settings_verification
                if not settings_verification["profile_matches"]:
                    raise CapabilityProbeError(
                        "Active settings do not match the exact provisional M13.1 "
                        "profile."
                    )
                context = prepare_airborne_probe(
                    client,
                    client_module,
                    airborne_config,
                    runtime,
                    args.lidar_name,
                    settings_verification=settings_verification,
                )
                data["airborne_context"] = asdict(context)
            if args.probe_pause:
                _run_pause_probe(
                    client,
                    client_module,
                    performance_config.vehicle_name,
                    args.confirm_no_visible_collision,
                )
            data["performance"] = probe_performance(
                client, performance_config, airborne_context=context
            )
        elif args.command != "inspect-client":
            raise ValueError(f"Unknown capability command {args.command!r}.")
    except KeyboardInterrupt:
        interrupted = True
        runtime.client_compromised = True
        errors.append("Probe interrupted; restart Blocks before another live stage.")
    except (CapabilityProbeError, ColosseumClientImportError, ValueError) as exc:
        if _indicates_client_corruption(exc):
            runtime.client_compromised = True
        errors.append(str(exc))
    except Exception as exc:
        if _indicates_client_corruption(exc):
            runtime.client_compromised = True
        errors.append(f"Unexpected probe failure: {type(exc).__name__}")
    finally:
        cleanup_results = cleanup_probe_domains(client, runtime)

    cleanup_failed = any(
        result.attempted and not result.succeeded for result in cleanup_results
    )
    if cleanup_failed:
        errors.append("One or more independent cleanup domains failed.")
    if runtime.client_compromised:
        errors.append(
            "Client may be unusable; restart Blocks and use a fresh Python process."
        )

    completed = datetime.now(timezone.utc)
    report = CapabilityProbeReport(
        schema_version="1.0",
        run_id=run_id,
        mode=args.command,
        started_at_utc=started.isoformat(),
        completed_at_utc=completed.isoformat(),
        success=not errors and not cleanup_failed,
        interrupted=interrupted,
        observations=tuple(observations),
        data=data,
        cleanup_results=cleanup_results,
        errors=tuple(errors),
    )

    output_path = args.output_path or generate_report_path(
        args.command,
        args.output_dir,
        timestamp=completed.isoformat(),
        run_id=run_id,
    )
    try:
        validate_report_output_path(output_path, root)
        save_capability_report(report, output_path)
    except (OSError, ValueError) as exc:
        print(f"Capability report was not saved: {exc}")
        return 1

    _print_report_summary(report, output_path)
    if interrupted:
        return 130
    return 0 if report.success else 1


def _add_airborne_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-flight", action="store_true")
    parser.add_argument("--confirm-clear-airspace", action="store_true")
    parser.add_argument("--confirm-no-visible-collision", action="store_true")
    parser.add_argument("--confirm-grounded-lidar-passed", action="store_true")
    parser.add_argument("--anchor-altitude", type=float, default=2.0)
    parser.add_argument("--min-ground-clearance", type=float, default=1.0)
    parser.add_argument("--anchor-velocity", type=float, default=0.5)
    parser.add_argument("--movement-timeout", type=float, default=20.0)
    parser.add_argument("--movement-tolerance", type=float, default=0.75)


def _build_connection_config(args: argparse.Namespace) -> ConnectionProbeConfig:
    return ConnectionProbeConfig(
        client_module=args.client_module,
        host=args.host,
        port=args.port,
        rpc_timeout=args.rpc_timeout,
        vehicle_name=args.vehicle_name,
    )


def _build_airborne_config(args: argparse.Namespace) -> AirborneProbeConfig:
    return AirborneProbeConfig(
        vehicle_name=args.vehicle_name,
        allow_flight=args.allow_flight,
        confirm_clear_airspace=args.confirm_clear_airspace,
        confirm_no_visible_collision=args.confirm_no_visible_collision,
        confirm_grounded_lidar_passed=args.confirm_grounded_lidar_passed,
        anchor_altitude=args.anchor_altitude,
        min_ground_clearance=args.min_ground_clearance,
        anchor_velocity=args.anchor_velocity,
        movement_timeout=args.movement_timeout,
        movement_tolerance=args.movement_tolerance,
    )


def _build_lidar_config(
    args: argparse.Namespace,
    airborne: AirborneProbeConfig,
) -> LidarProbeConfig:
    return LidarProbeConfig(
        airborne=airborne,
        lidar_name=args.lidar_name,
        scan_count=args.scan_count,
        scan_interval=args.scan_interval,
        stale_threshold=args.stale_threshold,
        warm_up_attempts=args.warm_up_attempts,
        warm_up_interval=args.warm_up_interval,
        settle_interval=args.settle_interval,
        coordinate_frame_experiment=args.coordinate_frame_experiment,
        allow_coordinate_motion=args.allow_coordinate_motion,
        coordinate_scan_attempts=args.coordinate_scan_attempts,
        yaw_return_tolerance_degrees=args.yaw_return_tolerance,
        visualize_lidar=args.visualize_lidar,
        allow_marker_flush=args.allow_marker_flush,
        visualization_hold_seconds=args.lidar_visualization_hold_seconds,
        visualization_max_points=args.lidar_visualization_max_points,
        visualization_max_rays=args.lidar_visualization_max_rays,
    )


def _build_performance_config(
    args: argparse.Namespace,
    airborne: AirborneProbeConfig,
) -> PerformanceProbeConfig:
    return PerformanceProbeConfig(
        vehicle_name=args.vehicle_name,
        iterations=args.iterations,
        include_lidar=args.include_lidar,
        lidar_name=args.lidar_name,
        include_control=args.include_control,
        control_duration=args.control_duration,
        probe_pause=args.probe_pause,
        allow_pause=args.allow_pause,
        airborne=airborne,
    )


def _build_grounded_lidar_config(
    args: argparse.Namespace,
) -> GroundedLidarProbeConfig:
    return GroundedLidarProbeConfig(
        vehicle_name=args.vehicle_name,
        lidar_name=args.lidar_name,
        scan_count=args.scan_count,
        scan_interval=args.scan_interval,
        stale_threshold=args.stale_threshold,
        warm_up_attempts=args.warm_up_attempts,
        warm_up_interval=args.warm_up_interval,
        confirm_no_visible_collision=args.confirm_no_visible_collision,
    )


def _validate_command_authorization(args: argparse.Namespace) -> None:
    if args.command == "markers" and not (
        args.allow_debug_markers and args.allow_marker_flush
    ):
        raise ValueError("markers requires both explicit marker authorization flags")
    if args.command == "markers":
        validate_marker_hold_seconds(args.marker_hold_seconds)
    if args.command == "mutation" and not (
        args.allow_scene_mutation
        and args.confirm_spawn_area_clear
        and args.confirm_vehicle_disarmed
    ):
        raise ValueError(
            "mutation requires authorization, clear-area confirmation, and disarm "
            "confirmation"
        )
    if args.command == "mutation":
        validate_mutation_hold_seconds(args.mutation_hold_seconds)
        validate_material_name(args.material_name)
    if args.command == "lidar":
        if not args.vehicle_name.strip():
            raise ValueError("lidar requires a non-empty vehicle name")
        if not args.lidar_name.strip():
            raise ValueError("lidar requires a non-empty LiDAR name")
        if not (
            args.allow_flight
            and args.confirm_clear_airspace
            and args.confirm_no_visible_collision
            and args.confirm_grounded_lidar_passed
        ):
            raise ValueError(
                "lidar requires flight, clear-airspace, no-visible-collision, "
                "and grounded-LiDAR confirmations"
            )
        if args.visualize_lidar and not args.allow_marker_flush:
            raise ValueError(
                "LiDAR visualization requires explicit marker-flush authorization"
            )
    if args.command == "performance":
        if not args.vehicle_name.strip():
            raise ValueError("performance requires a non-empty vehicle name")
        if (args.include_lidar or args.include_control) and not args.lidar_name.strip():
            raise ValueError(
                "LiDAR or control performance requires a non-empty LiDAR name"
            )
        if args.include_control and not (
            args.allow_flight
            and args.confirm_clear_airspace
            and args.confirm_no_visible_collision
            and args.confirm_grounded_lidar_passed
        ):
            raise ValueError(
                "control performance requires flight, clear-airspace, "
                "no-visible-collision, and grounded-LiDAR confirmations"
            )
        if args.probe_pause and not args.allow_pause:
            raise ValueError("pause performance requires allow_pause")


def _confirm_connection(client: Any) -> None:
    try:
        client.confirmConnection()
    except Exception as exc:
        raise CapabilityProbeError("Could not confirm simulator connection.") from exc


def _read_version_evidence(client: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, method_name in (
        ("client_version", "getClientVersion"),
        ("server_version", "getServerVersion"),
        ("minimum_server_version", "getMinRequiredServerVersion"),
        ("minimum_client_version", "getMinRequiredClientVersion"),
    ):
        method = getattr(client, method_name, None)
        result[label] = method() if method is not None else None
    return result


def _run_pause_probe(
    client: Any,
    client_module: Any,
    vehicle_name: str,
    confirm_no_visible_collision: bool,
) -> None:
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    validate_grounded_preflight(
        client,
        client_module,
        state,
        operator_confirmed_stable=confirm_no_visible_collision,
        vehicle_name=vehicle_name,
    )
    pause_state = getattr(client, "simIsPause", None)
    if pause_state is None:
        raise CapabilityProbeError("Client does not provide simIsPause.")
    initial = bool(pause_state())
    try:
        client.simPause(True)
        if not bool(pause_state()):
            raise CapabilityProbeError("Simulator did not enter paused state.")
        client.simContinueForTime(0.1)
    finally:
        client.simPause(initial)


def _require_client(client: Any | None) -> Any:
    if client is None:
        raise CapabilityProbeError("Live probe has no simulator client.")
    return client


def _parse_vector(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected x,y,z floating-point values"
        ) from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected exactly three comma-separated values"
        )
    return parts  # type: ignore[return-value]


def _indicates_client_corruption(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "timeout",
                "timed out",
                "ioloop",
                "tornado",
                "loop is already",
            )
        ):
            return True
        current = current.__cause__
    return False


def _print_report_summary(report: CapabilityProbeReport, output_path: Path) -> None:
    print(f"Capability probe mode: {report.mode}")
    print(f"Success: {report.success}")
    print(f"Report: {output_path}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
    print(
        "Live RPC success is not practical-support evidence; record required "
        "operator confirmations in the M13.1 documentation."
    )


if __name__ == "__main__":
    raise SystemExit(main())
