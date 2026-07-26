"""Offline-first CLI for M13.6 scripted reference-controller validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from r1_uav_nav.evaluation.m13_6_course_validation import (
    DEFAULT_CONFIG_PATH,
    M13_6LiveAuthorizations,
    default_report_path,
    execute_reference_episode,
    load_m13_6_config,
    prepare_reference_run,
    save_reference_episode_report,
    summarize_episode_reports,
    validate_offline_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M13.6 scripted reference controllers."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate",
        help="validate configuration, matrix, courses, and oracle plans offline",
    )
    preview = subparsers.add_parser(
        "preview",
        help="show sanitized controller/course plans offline",
    )
    _add_identity_arguments(preview, required=False)

    live = subparsers.add_parser(
        "run",
        help="run one explicitly authorized supervised live episode",
    )
    _add_identity_arguments(live, required=True)
    live.add_argument("--vehicle-name", default="SimpleFlight")
    live.add_argument("--lidar-name", default="LidarSensor1")
    live.add_argument("--allow-live-rpc", action="store_true")
    live.add_argument("--allow-scene-mutation", action="store_true")
    live.add_argument("--confirm-scene-area-clear", action="store_true")
    live.add_argument("--confirm-no-visible-collision", action="store_true")
    live.add_argument("--allow-debug-markers", action="store_true")
    live.add_argument("--allow-marker-flush", action="store_true")
    live.add_argument("--allow-flight", action="store_true")
    live.add_argument("--allow-start-positioning", action="store_true")
    live.add_argument("--confirm-clear-airspace", action="store_true")
    live.add_argument("--confirm-preflight-survey-passed", action="store_true")
    live.add_argument("--confirm-grounded-lidar-passed", action="store_true")
    live.add_argument("--allow-optional-hard", action="store_true")
    live.add_argument("--confirm-required-stages-passed", action="store_true")

    summary = subparsers.add_parser(
        "summarize",
        help="validate explicit episode reports against the required matrix",
    )
    summary.add_argument("reports", type=Path, nargs="+")
    return parser


def _add_identity_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
) -> None:
    parser.add_argument(
        "--controller",
        choices=("random", "direct", "oracle"),
        required=required,
    )
    parser.add_argument("--course-profile", required=required)
    parser.add_argument("--course-seed", type=int, required=required)
    parser.add_argument("--controller-seed", type=int)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    return run(parse_args())


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
    client_module_loader: Callable[[str], Any] | None = None,
    client_factory: Callable[[Any], Any] | None = None,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_m13_6_config(config_path)

    if args.command == "validate":
        plans = validate_offline_configuration(config, repository_root=root)
        print(
            json.dumps(
                {
                    "configuration_digest": config.configuration_digest,
                    "validated_plan_count": len(plans),
                    "plans": plans,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preview":
        plans = validate_offline_configuration(config, repository_root=root)
        requested = (args.controller, args.course_profile, args.course_seed)
        supplied = tuple(value is not None for value in requested)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "preview controller, profile, and course seed must be supplied together"
            )
        if all(supplied):
            plans = tuple(
                item
                for item in plans
                if item["controller_id"] == args.controller
                and item["course_profile"] == args.course_profile
                and item["base_seed"] == args.course_seed
                and item["controller_seed"] == args.controller_seed
            )
            if len(plans) != 1:
                raise ValueError("preview identity is not declared by the matrix")
        print(json.dumps(plans, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize_episode_reports(config, args.reports)
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return 0 if summary.report_success else 1
    if args.command != "run":
        raise ValueError(f"unsupported command {args.command!r}")
    if args.vehicle_name != config.vehicle_name:
        raise ValueError("run vehicle name does not match the M13.6 configuration")
    if args.lidar_name != config.lidar_name:
        raise ValueError("run LiDAR name does not match the M13.6 configuration")

    prepared = prepare_reference_run(
        config,
        repository_root=root,
        controller_id=args.controller,
        course_profile=args.course_profile,
        base_seed=args.course_seed,
        controller_seed=args.controller_seed,
        authorizations=_authorizations(args),
        report_directory=args.output_dir,
        allow_optional_hard=args.allow_optional_hard,
        confirm_required_stages_passed=args.confirm_required_stages_passed,
    )

    if client_module_loader is None or client_factory is None:
        from r1_uav_nav.sim.colosseum_client import (
            create_multirotor_client,
            import_colosseum_client_module,
        )

        client_module_loader = client_module_loader or import_colosseum_client_module
        client_factory = client_factory or create_multirotor_client
    module = client_module_loader(config.client_module)
    client = client_factory(module)
    report = execute_reference_episode(
        prepared,
        client=client,
        client_module=module,
        repository_root=root,
    )
    output = default_report_path(prepared, run_id=report.run_id)
    save_reference_episode_report(report, output, repository_root=root)
    print(f"Report: {output}")
    return 0 if report.report_success else 1


def _authorizations(args: argparse.Namespace) -> M13_6LiveAuthorizations:
    return M13_6LiveAuthorizations(
        allow_live_rpc=args.allow_live_rpc,
        allow_scene_mutation=args.allow_scene_mutation,
        confirm_scene_area_clear=args.confirm_scene_area_clear,
        confirm_no_visible_collision=args.confirm_no_visible_collision,
        allow_debug_markers=args.allow_debug_markers,
        allow_marker_flush=args.allow_marker_flush,
        allow_flight=args.allow_flight,
        allow_start_positioning=args.allow_start_positioning,
        confirm_clear_airspace=args.confirm_clear_airspace,
        confirm_preflight_survey_passed=args.confirm_preflight_survey_passed,
        confirm_grounded_lidar_passed=args.confirm_grounded_lidar_passed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
