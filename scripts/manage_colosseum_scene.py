"""Validate and supervise deterministic M13.2 Colosseum scenes."""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from r1_uav_nav.sim.colosseum_capabilities import validate_report_output_path
from r1_uav_nav.sim.colosseum_client import (
    confirm_connection,
    create_multirotor_client,
    import_colosseum_client_module,
)
from r1_uav_nav.sim.colosseum_scene import (
    DEFAULT_OWNERSHIP_DIR,
    DEFAULT_SCENE_REPORTS_DIR,
    AssetCalibrationProbe,
    ColosseumSceneManager,
    MaterializationConfig,
    OwnershipManifestError,
    PrebuiltVerifySceneBackend,
    RuntimeSpawnSceneBackend,
    SceneLifecycleReport,
    SceneRuntimeState,
    VehiclePositioningConfig,
    cleanup_scene_resources,
    recover_owned_scene,
    save_scene_report,
)
from r1_uav_nav.sim.scene_specification import (
    canonical_scene_dict,
    load_asset_catalog,
    load_scene_config,
    resolve_scene,
)
from r1_uav_nav.sim.static_course import (
    ValidatedCourse,
    course_report_dict,
    generate_solvable_course,
    load_course_suite_config,
    resolve_profile_scene_path,
)

DEFAULT_SCENE_CONFIG = Path("configs/scenes/m13_2_minimal.yaml")
DEFAULT_ASSET_CATALOG = Path("configs/scenes/m13_2_assets.yaml")
DEFAULT_COURSE_CONFIG = Path("configs/planning/m13_3_voxel_astar.yaml")
DEFAULT_COMMAND = "validate"
MAX_HOLD_SECONDS = 15.0


class _ExplicitSceneConfigAction(argparse.Action):
    """Remember an explicit scene path without changing its parsed value."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        namespace.scene_config_explicit = True


def build_parser() -> argparse.ArgumentParser:
    """Build a parser whose default path never imports the external client."""
    parser = argparse.ArgumentParser(
        description="Validate or supervise one deterministic M13.2 scene."
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=DEFAULT_SCENE_CONFIG,
        action=_ExplicitSceneConfigAction,
    )
    parser.add_argument("--asset-catalog", type=Path, default=DEFAULT_ASSET_CATALOG)
    parser.add_argument("--course-config", type=Path, default=DEFAULT_COURSE_CONFIG)
    parser.add_argument("--course-profile")
    parser.add_argument("--course-seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCENE_REPORTS_DIR)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("validate", help="validate one scene offline")
    generate = subparsers.add_parser(
        "generate", help="resolve and preview one scene offline"
    )
    generate.add_argument("--output-path", type=Path)

    materialize = subparsers.add_parser(
        "materialize", help="supervised runtime scene materialization"
    )
    _add_live_arguments(materialize)
    materialize.add_argument(
        "--backend",
        choices=("runtime_spawn", "prebuilt_verify"),
        default="runtime_spawn",
    )
    materialize.add_argument("--allow-scene-mutation", action="store_true")
    materialize.add_argument("--confirm-scene-area-clear", action="store_true")
    materialize.add_argument("--allow-debug-markers", action="store_true")
    materialize.add_argument("--allow-marker-flush", action="store_true")
    materialize.add_argument("--hold-seconds", type=float, default=0.0)
    materialize.add_argument("--repeat", type=int, default=1)
    materialize.add_argument("--position-start", action="store_true")
    materialize.add_argument("--allow-flight", action="store_true")
    materialize.add_argument("--allow-start-positioning", action="store_true")
    materialize.add_argument("--confirm-clear-airspace", action="store_true")

    recover = subparsers.add_parser(
        "cleanup", help="recover exact owned names from a manifest"
    )
    _add_live_arguments(recover)
    recover.add_argument("--ownership-source", type=Path, required=True)
    recover.add_argument("--allow-scene-mutation", action="store_true")
    recover.add_argument("--allow-recovery", action="store_true")
    recover.add_argument(
        "--expected-backend",
        choices=("runtime_spawn", "asset_calibration"),
        default="runtime_spawn",
    )

    calibrate = subparsers.add_parser(
        "calibrate-asset",
        help="prepare a supervised nominal Cube calibration report",
    )
    _add_live_arguments(calibrate)
    calibrate.add_argument("--asset-name", required=True)
    calibrate.add_argument("--allow-scene-mutation", action="store_true")
    calibrate.add_argument("--confirm-scene-area-clear", action="store_true")
    calibrate.add_argument("--allow-debug-markers", action="store_true")
    calibrate.add_argument("--allow-marker-flush", action="store_true")
    calibrate.add_argument("--hold-seconds", type=float, default=8.0)

    parser.set_defaults(command=DEFAULT_COMMAND, scene_config_explicit=False)
    return parser


def _add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--client-module", default="airsim")
    parser.add_argument("--vehicle-name", default="SimpleFlight")
    parser.add_argument("--confirm-no-visible-collision", action="store_true")


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
    """Run one mode, preserving cleanup and report evidence."""
    root = (repository_root or Path.cwd()).resolve()
    scene, course = _resolve_scene_or_course(args, root)
    catalog = (
        load_asset_catalog(args.asset_catalog)
        if args.command == "materialize"
        else None
    )
    if args.command == "validate":
        assert scene is not None
        print(
            f"Scene {scene.config.scene_id!r} is valid: " f"digest={scene.scene_digest}"
        )
        if course is not None:
            print(
                f"Course {course.result.profile_id!r} is solvable: "
                f"digest={course.result.solvability_digest}"
            )
        return 0
    if args.command == "generate":
        assert scene is not None
        payload = json.dumps(
            canonical_scene_dict(scene.config), indent=2, sort_keys=True
        )
        if args.output_path is None:
            print(payload)
        else:
            validate_report_output_path(args.output_path, root)
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(payload + "\n", encoding="utf-8", newline="\n")
        return 0

    _validate_live_arguments(args, scene)
    client_module = client_module_loader(args.client_module)
    client = client_factory(client_module)
    confirm_connection(client)
    run_id = uuid.uuid4().hex
    runtime: SceneRuntimeState | None = None
    materialized = None
    report_data: dict[str, Any] = {}
    if course is not None:
        report_data["static_course_solvability"] = course_report_dict(course)
    cleanup_results = ()
    errors: list[str] = []
    interrupted = False
    report_scene_id: str | None = None
    report_scene_digest: str | None = None
    report_materialization_digest: str | None = None
    repetitions: list[dict[str, Any]] = []
    active_repetition_index: int | None = None

    try:
        if args.command == "cleanup":
            recovered, result = recover_owned_scene(
                client,
                args.ownership_source,
                allow_scene_mutation=args.allow_scene_mutation,
                allow_recovery=args.allow_recovery,
                expected_backend=args.expected_backend,
            )
            report_scene_id = recovered.scene_id
            report_scene_digest = recovered.scene_digest
            report_materialization_digest = recovered.materialization_digest
            cleanup_results = (result,)
            if not result.succeeded:
                errors.extend(result.errors)
        elif args.command == "calibrate-asset":
            probe = AssetCalibrationProbe(
                client,
                client_module,
                ownership_dir=DEFAULT_OWNERSHIP_DIR,
                sleep_fn=sleep_fn,
            )
            try:
                result, runtime = probe.run(
                    asset_name=args.asset_name,
                    vehicle_name=args.vehicle_name,
                    allow_scene_mutation=args.allow_scene_mutation,
                    confirm_scene_area_clear=args.confirm_scene_area_clear,
                    confirm_no_visible_collision=args.confirm_no_visible_collision,
                    allow_debug_markers=args.allow_debug_markers,
                    allow_marker_flush=args.allow_marker_flush,
                    hold_seconds=args.hold_seconds,
                    run_id=run_id,
                )
                report_data["asset_calibration"] = result
                report_data["catalog_updated"] = False
                report_scene_id = "asset-calibration"
                report_scene_digest = runtime.manifest.scene_digest
                report_materialization_digest = runtime.manifest.materialization_digest
            except BaseException:
                runtime = probe.last_runtime
                raise
        else:
            assert scene is not None
            assert catalog is not None
            config = MaterializationConfig(
                vehicle_name=args.vehicle_name,
                allow_scene_mutation=args.allow_scene_mutation,
                confirm_scene_area_clear=args.confirm_scene_area_clear,
                confirm_no_visible_collision=args.confirm_no_visible_collision,
                allow_debug_markers=args.allow_debug_markers,
                allow_marker_flush=args.allow_marker_flush,
            )
            backend = (
                RuntimeSpawnSceneBackend(client, client_module)
                if args.backend == "runtime_spawn"
                else PrebuiltVerifySceneBackend(client)
            )
            manager = ColosseumSceneManager(
                client,
                client_module,
                catalog,
                ownership_dir=DEFAULT_OWNERSHIP_DIR,
                sleep_fn=sleep_fn,
            )
            for index in range(args.repeat):
                try:
                    materialized, runtime = manager.reset_scene(
                        scene,
                        config,
                        backend=backend,
                        run_id=f"{run_id}-{index + 1}",
                    )
                except BaseException:
                    if (
                        active_repetition_index is not None
                        and manager.last_reset_cleanup_results
                    ):
                        repetitions[active_repetition_index][
                            "cleanup_results"
                        ] = manager.last_reset_cleanup_results
                    active_repetition_index = None
                    runtime = manager.last_runtime
                    raise
                if (
                    active_repetition_index is not None
                    and manager.last_reset_cleanup_results
                ):
                    repetitions[active_repetition_index][
                        "cleanup_results"
                    ] = manager.last_reset_cleanup_results
                repetitions.append(
                    _build_repetition_evidence(materialized, iteration=index + 1)
                )
                active_repetition_index = len(repetitions) - 1
                if args.hold_seconds:
                    print(f"Scene is materialized for {args.hold_seconds:.1f} seconds.")
                    sleep_fn(args.hold_seconds)
            report_data["repetitions"] = repetitions
            report_scene_id = materialized.scene_id
            report_scene_digest = materialized.scene_digest
            report_materialization_digest = materialized.materialization_digest
            if args.position_start:
                from r1_uav_nav.sim.colosseum_scene import (
                    position_vehicle_at_start_and_return,
                )

                report_data["vehicle_positioning"] = (
                    position_vehicle_at_start_and_return(
                        client,
                        client_module,
                        materialized,
                        runtime,
                        VehiclePositioningConfig(
                            vehicle_name=args.vehicle_name,
                            allow_flight=args.allow_flight,
                            allow_start_positioning=args.allow_start_positioning,
                            confirm_clear_airspace=args.confirm_clear_airspace,
                            confirm_no_visible_collision=args.confirm_no_visible_collision,
                        ),
                        sleep_fn=sleep_fn,
                    )
                )
    except KeyboardInterrupt:
        interrupted = True
        errors.append("Operation interrupted by the operator.")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if runtime is not None:
            if runtime.vehicle_positioning_evidence:
                report_data.setdefault(
                    "vehicle_positioning",
                    runtime.vehicle_positioning_evidence,
                )
            cleanup_results = cleanup_scene_resources(client, runtime)
            if active_repetition_index is not None:
                repetitions[active_repetition_index][
                    "cleanup_results"
                ] = cleanup_results
            errors.extend(
                error
                for result in cleanup_results
                if not result.succeeded
                for error in result.errors
            )
            report_data["ownership_manifest"] = runtime.manifest
            report_data["ownership_evidence_complete"] = True

    completed = datetime.now(timezone.utc)
    report = SceneLifecycleReport(
        schema_version="1.0",
        run_id=run_id,
        mode=args.command,
        success=not errors and not interrupted,
        interrupted=interrupted,
        scene_id=report_scene_id,
        scene_digest=report_scene_digest,
        materialization_digest=report_materialization_digest,
        selected_vehicle_name=getattr(args, "vehicle_name", None),
        materialized_scene=materialized,
        data=report_data,
        cleanup_results=tuple(cleanup_results),
        errors=tuple(errors),
    )
    filename = (
        f"m13_2_{args.command}_{completed.strftime('%Y%m%dT%H%M%S')}_"
        f"{run_id[:8]}.json"
    )
    output = args.output_dir / filename
    validate_report_output_path(output, root)
    save_scene_report(report, output)
    print(f"Report: {output}")
    if interrupted:
        return 130
    return 0 if report.success else 1


def _resolve_scene_or_course(
    args: argparse.Namespace,
    root: Path,
) -> tuple[Any | None, ValidatedCourse | None]:
    scene_commands = {"validate", "generate", "materialize"}
    if args.command not in scene_commands:
        if args.course_profile is not None or args.course_seed is not None:
            raise ValueError(
                "course arguments apply only to validate, generate, or materialize"
            )
        return None, None

    if args.course_profile is None:
        if args.course_seed is not None:
            raise ValueError("--course-seed requires --course-profile")
        return resolve_scene(load_scene_config(args.scene_config)), None

    if args.course_seed is None:
        raise ValueError("--course-profile requires --course-seed")
    suite = load_course_suite_config(args.course_config)
    profile = suite.profile(args.course_profile)
    profile_scene_path = resolve_profile_scene_path(profile, root)
    if args.scene_config_explicit:
        supplied_scene_path = (
            args.scene_config
            if args.scene_config.is_absolute()
            else root / args.scene_config
        ).resolve()
        if supplied_scene_path != profile_scene_path:
            raise ValueError(
                "--scene-config conflicts with the selected course profile"
            )
    course = generate_solvable_course(
        suite,
        profile.profile_id,
        args.course_seed,
        repository_root=root,
    )
    return course.scene, course


def _validate_live_arguments(
    args: argparse.Namespace, scene: Any | None = None
) -> None:
    if not args.vehicle_name.strip():
        raise ValueError("vehicle_name must not be empty")
    hold = float(getattr(args, "hold_seconds", 0.0))
    if not math.isfinite(hold) or not 0.0 <= hold <= MAX_HOLD_SECONDS:
        raise ValueError("hold_seconds must be finite and between 0 and 15")
    if args.command == "materialize":
        if scene is None:
            raise ValueError("materialize requires a resolved scene")
        if not 1 <= args.repeat <= 3:
            raise ValueError("repeat must be between 1 and 3")
        if args.backend == "runtime_spawn" and not args.allow_scene_mutation:
            raise ValueError("runtime_spawn requires --allow-scene-mutation")
        if args.backend == "runtime_spawn" and not args.confirm_scene_area_clear:
            raise ValueError("runtime_spawn requires --confirm-scene-area-clear")
        if not args.confirm_no_visible_collision:
            raise ValueError("materialization requires collision confirmation")
        requires_marker_overlay = (
            scene.config.goal_pad.appearance.marker_color_rgba is not None
        )
        if requires_marker_overlay and not args.allow_debug_markers:
            raise ValueError(
                "scene requires --allow-debug-markers for its goal overlay"
            )
        if (
            requires_marker_overlay or args.allow_debug_markers
        ) and not args.allow_marker_flush:
            raise ValueError("debug markers require --allow-marker-flush")
        if args.position_start and not all(
            (
                args.allow_flight,
                args.allow_start_positioning,
                args.confirm_clear_airspace,
                args.confirm_no_visible_collision,
            )
        ):
            raise ValueError("position-start lacks required flight authorization")
    if args.command == "cleanup" and not (
        args.allow_scene_mutation and args.allow_recovery
    ):
        raise OwnershipManifestError(
            "cleanup requires mutation and recovery authorization"
        )
    if args.command == "calibrate-asset":
        if not all(
            (
                args.allow_scene_mutation,
                args.confirm_scene_area_clear,
                args.allow_debug_markers,
                args.allow_marker_flush,
                args.confirm_no_visible_collision,
            )
        ):
            raise ValueError("calibration lacks required authorization")


def _build_repetition_evidence(
    materialized: Any,
    *,
    iteration: int,
    cleanup_results: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build directly comparable evidence for one complete materialization."""
    return {
        "iteration": iteration,
        "scene_digest": materialized.scene_digest,
        "materialization_digest": materialized.materialization_digest,
        "exact_names": {
            "requested": tuple(item.requested_name for item in materialized.objects),
            "returned": tuple(item.returned_name for item in materialized.objects),
        },
        "objects": tuple(
            {
                "specification_name": item.specification_name,
                "requested_name": item.requested_name,
                "returned_name": item.returned_name,
                "requested_transform": item.requested_transform,
                "measured_center_position": item.measured_center_position,
                "measured_scale": item.measured_scale,
                "measured_yaw_degrees": item.measured_yaw_degrees,
            }
            for item in materialized.objects
        ),
        "cleanup_results": tuple(cleanup_results),
    }


if __name__ == "__main__":
    raise SystemExit(main())
