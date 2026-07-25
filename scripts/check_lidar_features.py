"""Offline-first M13.4 LiDAR feature validation CLI."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from r1_uav_nav.sim.colosseum_capabilities import validate_report_output_path
from r1_uav_nav.sim.colosseum_client import (
    confirm_connection,
    create_multirotor_client,
    import_colosseum_client_module,
)
from r1_uav_nav.sim.colosseum_lidar import (
    GroundedLidarFeatureProbeConfig,
    build_easy_known_geometry_evidence,
    build_live_report,
    probe_grounded_lidar_features,
    save_lidar_live_report,
)
from r1_uav_nav.sim.lidar_features import (
    LidarScanInput,
    extract_lidar_features,
    extraction_evidence,
    load_lidar_feature_config,
)
from r1_uav_nav.sim.lidar_live_validation import (
    AIRBORNE_SMOKE_PROFILE,
    AIRBORNE_SMOKE_SEED,
    DEFAULT_ASSET_CATALOG,
    DEFAULT_COURSE_CONFIG,
    KNOWN_GEOMETRY_PROFILE,
    KNOWN_GEOMETRY_SEED,
    LiveAuthorizationEvidence,
    execute_airborne_smoke,
    execute_known_geometry_live,
    prepare_lidar_live_run,
)
from r1_uav_nav.sim.static_course import (
    generate_solvable_course,
    load_course_suite_config,
)

DEFAULT_CONFIG = Path("configs/sensing/m13_4_lidar_features.yaml")
DEFAULT_REPORT_DIR = Path("results/reports/m13/lidar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fixed-size M13.4 LiDAR features."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("validate", help="validate config and one offline fixture")
    geometry = subparsers.add_parser(
        "known-geometry", help="recompute accepted easy-seed comparison evidence"
    )
    geometry.add_argument("--course-config", type=Path, default=DEFAULT_COURSE_CONFIG)
    geometry.add_argument("--course-profile", default="easy")
    geometry.add_argument("--course-seed", type=int, default=1100)
    grounded = subparsers.add_parser(
        "grounded", help="supervised read-only grounded feature probe"
    )
    grounded.add_argument("--vehicle-name", required=True)
    grounded.add_argument("--lidar-name", required=True)
    grounded.add_argument("--scan-count", type=int, default=20)
    grounded.add_argument("--scan-interval", type=float, default=0.2)
    grounded.add_argument("--allow-live-rpc", action="store_true")
    grounded.add_argument("--confirm-no-visible-collision", action="store_true")
    grounded.add_argument("--client-module", default="airsim")
    known_live = subparsers.add_parser(
        "known-geometry-live",
        help="supervised easy-course known-distance validation",
    )
    _add_live_course_arguments(
        known_live,
        default_profile=KNOWN_GEOMETRY_PROFILE,
        default_seed=KNOWN_GEOMETRY_SEED,
    )
    known_live.add_argument("--scan-count", type=int, default=10)
    known_live.add_argument("--scan-interval", type=float, default=0.2)
    airborne = subparsers.add_parser(
        "airborne-smoke",
        help="supervised medium-course 83-value observation smoke test",
    )
    _add_live_course_arguments(
        airborne,
        default_profile=AIRBORNE_SMOKE_PROFILE,
        default_seed=AIRBORNE_SMOKE_SEED,
    )
    airborne.add_argument("--step-count", type=int, default=5)
    parser.set_defaults(command="validate")
    return parser


def _add_live_course_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_profile: str,
    default_seed: int,
) -> None:
    parser.add_argument("--course-config", type=Path, default=DEFAULT_COURSE_CONFIG)
    parser.add_argument("--asset-catalog", type=Path, default=DEFAULT_ASSET_CATALOG)
    parser.add_argument("--course-profile", default=default_profile)
    parser.add_argument("--course-seed", type=int, default=default_seed)
    parser.add_argument("--vehicle-name", required=True)
    parser.add_argument("--lidar-name", required=True)
    parser.add_argument("--client-module", default="airsim")
    parser.add_argument("--allow-live-rpc", action="store_true")
    parser.add_argument("--allow-scene-mutation", action="store_true")
    parser.add_argument("--confirm-scene-area-clear", action="store_true")
    parser.add_argument("--confirm-no-visible-collision", action="store_true")
    parser.add_argument("--allow-debug-markers", action="store_true")
    parser.add_argument("--allow-marker-flush", action="store_true")
    parser.add_argument("--allow-flight", action="store_true")
    parser.add_argument("--allow-start-positioning", action="store_true")
    parser.add_argument("--confirm-clear-airspace", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    return run(parse_args())


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
    client_module_loader: Callable[[str], Any] = import_colosseum_client_module,
    client_factory: Callable[[Any], Any] = create_multirotor_client,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    config = load_lidar_feature_config(args.config)
    started = datetime.now(timezone.utc)
    if args.command == "validate":
        result = extract_lidar_features(
            LidarScanInput((2.0, 0.0, 0.0), 1),
            config,
        )
        print(
            json.dumps(
                {
                    "configuration_digest": config.configuration_digest,
                    "feature_shape": list(result.features.shape),
                    "fixture": extraction_evidence(result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "known-geometry":
        if args.course_profile != "easy" or args.course_seed != 1100:
            raise ValueError("known-geometry requires easy seed 1100")
        suite = load_course_suite_config(args.course_config)
        course = generate_solvable_course(
            suite, args.course_profile, args.course_seed, repository_root=root
        )
        evidence = build_easy_known_geometry_evidence(course, config)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    if args.command in {"known-geometry-live", "airborne-smoke"}:
        sample_count = (
            args.scan_count
            if args.command == "known-geometry-live"
            else args.step_count
        )
        prepared = prepare_lidar_live_run(
            mode=args.command,
            repository_root=root,
            feature_config=config,
            course_config_path=args.course_config,
            asset_catalog_path=args.asset_catalog,
            output_dir=args.output_dir,
            profile_id=args.course_profile,
            base_seed=args.course_seed,
            vehicle_name=args.vehicle_name,
            lidar_name=args.lidar_name,
            authorizations=_live_authorizations(args),
            sample_count=sample_count,
            sample_interval_s=(
                args.scan_interval if args.command == "known-geometry-live" else 0.0
            ),
        )
        client_module = client_module_loader(args.client_module)
        client = client_factory(client_module)
        confirm_connection(client)
        execution = (
            execute_known_geometry_live(
                prepared,
                client,
                client_module,
                sleep_fn=sleep_fn,
            )
            if args.command == "known-geometry-live"
            else execute_airborne_smoke(
                prepared,
                client,
                client_module,
                sleep_fn=sleep_fn,
            )
        )
        shape = (72,) if args.command == "known-geometry-live" else (83,)
        report = build_live_report(
            mode=args.command,
            config=config,
            started_at=started,
            data=execution.data,
            errors=execution.errors,
            interrupted=execution.interrupted,
            cleanup=execution.cleanup,
            feature_shape=shape,
        )
        prefix = (
            "m13_4_known_geometry_live_"
            if args.command == "known-geometry-live"
            else "m13_4_airborne_smoke_"
        )
        output = prepared.output_dir / (
            f"{prefix}{started.strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{report.run_id[:8]}.json"
        )
        validate_report_output_path(output, root)
        save_lidar_live_report(report, output)
        print(f"Report: {output}")
        return 0 if report.success else 1
    if args.command != "grounded":
        raise ValueError(f"unsupported command {args.command!r}")

    _validate_grounded_arguments(args, config)
    client_module = client_module_loader(args.client_module)
    client = client_factory(client_module)
    confirm_connection(client)
    errors: tuple[str, ...] = ()
    interrupted = False
    data: dict[str, Any] = {}
    try:
        data = probe_grounded_lidar_features(
            client,
            client_module,
            GroundedLidarFeatureProbeConfig(
                config,
                args.scan_count,
                args.scan_interval,
                args.confirm_no_visible_collision,
            ),
            sleep_fn=sleep_fn,
        )
    except KeyboardInterrupt:
        interrupted = True
        errors = ("Operation interrupted by the operator.",)
    except BaseException as exc:
        errors = (f"{type(exc).__name__}: {exc}",)
    report = build_live_report(
        mode="grounded",
        config=config,
        started_at=started,
        data=data,
        errors=errors,
        interrupted=interrupted,
    )
    output = args.output_dir / (
        f"m13_4_grounded_{started.strftime('%Y%m%dT%H%M%S%fZ')}_{report.run_id[:8]}.json"
    )
    validate_report_output_path(output, root)
    save_lidar_live_report(report, output)
    print(f"Report: {output}")
    return 0 if report.success else 1


def _live_authorizations(args: argparse.Namespace) -> LiveAuthorizationEvidence:
    return LiveAuthorizationEvidence(
        allow_live_rpc=args.allow_live_rpc,
        allow_scene_mutation=args.allow_scene_mutation,
        confirm_scene_area_clear=args.confirm_scene_area_clear,
        confirm_no_visible_collision=args.confirm_no_visible_collision,
        allow_debug_markers=args.allow_debug_markers,
        allow_marker_flush=args.allow_marker_flush,
        allow_flight=args.allow_flight,
        allow_start_positioning=args.allow_start_positioning,
        confirm_clear_airspace=args.confirm_clear_airspace,
    )


def _validate_grounded_arguments(args: argparse.Namespace, config: Any) -> None:
    if not args.allow_live_rpc:
        raise ValueError("grounded mode requires --allow-live-rpc")
    if not args.confirm_no_visible_collision:
        raise ValueError("grounded mode requires --confirm-no-visible-collision")
    if args.vehicle_name != config.vehicle_name:
        raise ValueError("vehicle name does not match the feature configuration")
    if args.lidar_name != config.lidar_name:
        raise ValueError("LiDAR name does not match the feature configuration")
    if not 1 <= args.scan_count <= 100:
        raise ValueError("scan-count must be between 1 and 100")
    if not 0 <= args.scan_interval <= 5:
        raise ValueError("scan-interval must be between 0 and 5 seconds")


if __name__ == "__main__":
    raise SystemExit(main())
