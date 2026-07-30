"""Offline-only M13.8 static-obstacle curriculum tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from r1_uav_nav.training.curriculum import (
    DEFAULT_CURRICULUM_CONFIG_PATH,
    CurriculumState,
    enable_curriculum_mode,
    load_curriculum_config,
    validate_curriculum_configuration,
)
from r1_uav_nav.training.curriculum_live_pilot import (
    DEFAULT_LIVE_PILOT_CONFIG_PATH,
    GitEvidence,
    LivePilotAuthorizations,
    PreparedLivePilot,
    load_live_pilot_config,
    prepare_live_pilot,
    summarize_live_pilot_reports,
)
from r1_uav_nav.training.long_run_state import (
    LongRunRunState,
    validate_checkpoint_bundle,
    validate_resume_request,
)
from r1_uav_nav.training.long_run_training import (
    LongRunOutputConfig,
    load_long_run_td3_config,
    run_fake_td3_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M13.8 curriculum infrastructure offline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CURRICULUM_CONFIG_PATH,
    )
    parser.add_argument(
        "--live-pilot-config",
        type=Path,
        default=DEFAULT_LIVE_PILOT_CONFIG_PATH,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate config, splits, and courses")
    subparsers.add_parser("preview-stages", help="print sanitized stage definitions")
    fake = subparsers.add_parser(
        "fake-smoke", help="run tiny TD3 checkpoint and full-resume smoke"
    )
    fake.add_argument("--timesteps", type=int, default=100)
    fake.add_argument("--run-id")
    inspect = subparsers.add_parser(
        "inspect-curriculum-state", help="inspect state JSON or checkpoint bundle"
    )
    inspect.add_argument("source", type=Path)
    summarize = subparsers.add_parser(
        "summarize", help="summarize ignored curriculum JSON reports"
    )
    summarize.add_argument("reports", nargs="+", type=Path)
    preflight = subparsers.add_parser(
        "preflight-live-pilot",
        help="validate one supervised live pilot without importing a client",
    )
    _add_live_pilot_arguments(preflight)
    pilot = subparsers.add_parser(
        "pilot-stage",
        help="run one explicitly authorized bounded Stage 0 or Stage 1 segment",
    )
    _add_live_pilot_arguments(pilot)
    resume = subparsers.add_parser(
        "resume-pilot",
        help="strictly resume one authorized bounded pilot segment",
    )
    _add_live_pilot_arguments(resume, require_resume_latest=True)
    live_summary = subparsers.add_parser(
        "summarize-live-pilot",
        help="summarize explicit ignored Phase B reports offline",
    )
    live_summary.add_argument("reports", nargs="+", type=Path)
    return parser


def _add_live_pilot_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_resume_latest: bool = False,
) -> None:
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--course-profile", required=True)
    parser.add_argument("--course-seed", type=int, required=True)
    parser.add_argument("--pilot-kind", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--max-timesteps", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--vehicle-name", default="SimpleFlight")
    parser.add_argument("--lidar-name", default="LidarSensor1")
    parser.add_argument("--m13-6-report", type=Path, action="append", default=[])
    parser.add_argument("--expected-m13-6-suite-digest", required=True)
    parser.add_argument("--preflight-survey-report", type=Path, required=True)
    parser.add_argument("--preflight-survey-digest", required=True)
    parser.add_argument("--grounded-lidar-report", type=Path, required=True)
    parser.add_argument("--grounded-lidar-digest", required=True)
    parser.add_argument("--stage-0-pilot-report", type=Path)
    parser.add_argument("--stage-0-pilot-digest")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument("--resume-run-state", type=Path)
    parser.add_argument("--resume-latest", type=Path, required=require_resume_latest)
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--allow-partial-resume", action="store_true")
    for flag in (
        "allow-live-rpc",
        "allow-scene-mutation",
        "allow-debug-markers",
        "allow-marker-flush",
        "allow-flight",
        "allow-start-positioning",
        "allow-training",
        "confirm-results-root-ignored",
        "confirm-m13-6-supervised-evidence-accepted",
        "confirm-preflight-survey-passed",
        "confirm-grounded-lidar-passed",
        "confirm-clear-airspace",
        "confirm-scene-area-clear",
        "confirm-no-visible-collision",
        "confirm-manual-operator-present",
        "confirm-named-cleanup-required",
    ):
        parser.add_argument(f"--{flag}", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
    runtime_loader: Callable[[], Any] | None = None,
    git_inspector: Callable[[Path], GitEvidence] | None = None,
    ignore_checker: Callable[[Path, Path], bool] | None = None,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_curriculum_config(config_path)

    if args.command == "validate":
        evidence = validate_curriculum_configuration(
            config,
            repository_root=root,
            verify_courses=True,
        )
        print(json.dumps(evidence, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "preview-stages":
        validate_curriculum_configuration(
            config,
            repository_root=root,
            verify_courses=False,
        )
        print(
            json.dumps(
                {
                    "schema_version": config.schema_version,
                    "curriculum_id": config.curriculum_id,
                    "configuration_digest": config.config_digest,
                    "stages": [
                        {
                            "stage_id": stage.stage_id,
                            "description": stage.description,
                            "minimum_stage_steps": stage.minimum_stage_steps,
                            "maximum_stage_steps": stage.maximum_stage_steps,
                            "validation_interval_steps": (
                                stage.validation_interval_steps
                            ),
                            "training_course_count": len(stage.training_courses),
                            "validation_course_count": len(stage.validation_courses),
                            "gate": asdict(stage.gate),
                        }
                        for stage in config.stages
                    ],
                    "live_training_enabled": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fake-smoke":
        return _fake_smoke(args, config=config, root=root)
    if args.command == "inspect-curriculum-state":
        source = args.source if args.source.is_absolute() else root / args.source
        print(
            json.dumps(
                _inspect_state(source),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    if args.command == "summarize":
        print(
            json.dumps(
                _summarize_reports(args.reports, root=root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "summarize-live-pilot":
        live_config = _load_live_config(args, root)
        summary = summarize_live_pilot_reports(
            args.reports,
            repository_root=root,
            approved_root=root / live_config.outputs.report_root,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["successful_report_count"] == summary["report_count"] else 1
    if args.command in {
        "preflight-live-pilot",
        "pilot-stage",
        "resume-pilot",
    }:
        prepared = _prepare_live(
            args,
            root=root,
            git_inspector=git_inspector,
            ignore_checker=ignore_checker,
        )
        if args.command == "preflight-live-pilot":
            print(
                json.dumps(
                    _prepared_evidence(prepared),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if runtime_loader is None:
            from r1_uav_nav.training.curriculum_live_runtime import (
                execute_live_pilot,
            )

            executor = execute_live_pilot
        else:
            executor = runtime_loader()
        result = executor(prepared, repository_root=root)
        print(f"Report: {result.report_path}")
        return 0 if result.report.report_success else 1
    raise ValueError(f"unsupported command {args.command!r}")


def _load_live_config(args: argparse.Namespace, root: Path) -> Any:
    path = args.live_pilot_config
    source = path if path.is_absolute() else root / path
    return load_live_pilot_config(source)


def _prepare_live(
    args: argparse.Namespace,
    *,
    root: Path,
    git_inspector: Callable[[Path], GitEvidence] | None,
    ignore_checker: Callable[[Path, Path], bool] | None,
) -> PreparedLivePilot:
    return prepare_live_pilot(
        _load_live_config(args, root),
        repository_root=root,
        stage_id=args.stage_id,
        profile_id=args.course_profile,
        base_seed=args.course_seed,
        pilot_kind=args.pilot_kind,
        requested_timesteps=args.max_timesteps,
        authorizations=_live_authorizations(args),
        m13_6_report_paths=args.m13_6_report,
        expected_m13_6_suite_digest=args.expected_m13_6_suite_digest,
        preflight_survey_path=args.preflight_survey_report,
        expected_preflight_survey_digest=args.preflight_survey_digest,
        grounded_lidar_path=args.grounded_lidar_report,
        expected_grounded_lidar_digest=args.grounded_lidar_digest,
        stage_0_report_path=args.stage_0_pilot_report,
        expected_stage_0_report_digest=args.stage_0_pilot_digest,
        vehicle_name=args.vehicle_name,
        lidar_name=args.lidar_name,
        run_id=args.run_id,
        resume_checkpoint=args.resume_checkpoint,
        resume_replay_buffer=args.resume_replay_buffer,
        resume_run_state=args.resume_run_state,
        resume_latest=args.resume_latest,
        reset_num_timesteps=args.reset_num_timesteps,
        allow_partial_resume=args.allow_partial_resume,
        report_directory=args.output_dir,
        git_inspector=git_inspector,
        ignore_checker=ignore_checker,
    )


def _live_authorizations(args: argparse.Namespace) -> LivePilotAuthorizations:
    return LivePilotAuthorizations(
        **{
            name: bool(getattr(args, name))
            for name in LivePilotAuthorizations.__dataclass_fields__
        }
    )


def _prepared_evidence(prepared: PreparedLivePilot) -> dict[str, Any]:
    result = prepared.course.result
    return {
        "preflight_success": True,
        "runtime_imported": False,
        "run_id": prepared.run_id,
        "stage_id": prepared.stage.stage_id,
        "pilot_kind": prepared.pilot_kind.value,
        "requested_timesteps": prepared.requested_timesteps,
        "completed_pilot_timesteps": prepared.completed_pilot_timesteps,
        "remaining_timesteps": prepared.remaining_timesteps,
        "profile_id": result.profile_id,
        "base_seed": result.base_seed,
        "accepted_candidate_seed": result.accepted_candidate_seed,
        "scene_digest": result.scene_digest,
        "occupancy_digest": result.occupancy_digest,
        "solvability_digest": result.solvability_digest,
        "m13_6_suite_digest": prepared.m13_6_evidence.suite_digest,
        "curriculum_config_digest": prepared.curriculum.config_digest,
        "pilot_config_digest": prepared.config.config_digest,
        "pilot_metadata_digest": prepared.metadata.metadata_digest,
        "resume_mode": prepared.resume_plan.mode.value,
        "git": asdict(prepared.git_evidence),
        "claim_flags": {
            "promotion_claimed": False,
            "learned_avoidance_claimed": False,
            "final_generalization_claimed": False,
            "real_world_claimed": False,
        },
    }


def _fake_smoke(
    args: argparse.Namespace,
    *,
    config: Any,
    root: Path,
) -> int:
    if not 100 <= args.timesteps <= 500:
        raise ValueError("--timesteps must be within the Phase A range 100-500")
    validate_curriculum_configuration(
        config,
        repository_root=root,
        verify_courses=False,
    )
    long_run_path = root / config.long_run_config_path
    long_run = load_long_run_td3_config(long_run_path)
    effective = replace(
        enable_curriculum_mode(long_run, config),
        additional_timesteps=args.timesteps,
        learning_starts=1,
        batch_size=2,
        buffer_size=max(512, args.timesteps * 3),
        policy_hidden_layers=(8, 8),
        checkpoint_interval_steps=max(1_000, args.timesteps * 3),
        output=LongRunOutputConfig(
            model_root=config.output.model_root,
            log_root=config.output.log_root,
            report_root=config.output.report_root,
        ),
    )
    state = replace(
        CurriculumState.initial(config),
        stage_completed_timesteps=args.timesteps,
    )
    first = run_fake_td3_smoke(
        effective,
        repository_root=root,
        run_id=args.run_id,
        curriculum_state_provider=state.to_dict,
    )
    plan = validate_resume_request(
        resume_latest=first.checkpoint.directory.parent.parent,
        expected_compatibility_digest=effective.compatibility_digest,
    )
    second = run_fake_td3_smoke(
        effective,
        repository_root=root,
        resume_plan=plan,
    )
    validated = validate_checkpoint_bundle(second.checkpoint.directory)
    resumed_state = CurriculumState.from_mapping(validated.run_state.curriculum_state)
    print(
        json.dumps(
            {
                "run_id": second.run_id,
                "initial_timesteps": first.initial_timesteps,
                "first_final_timesteps": first.final_timesteps,
                "resumed_initial_timesteps": second.initial_timesteps,
                "resumed_final_timesteps": second.final_timesteps,
                "checkpoint": str(second.checkpoint.directory),
                "curriculum_stage": resumed_state.stage_id,
                "curriculum_config_digest": (resumed_state.curriculum_config_digest),
                "replay_size": second.checkpoint.manifest.replay_buffer_size,
                "full_resume_verified": True,
                "live_training_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _inspect_state(source: Path) -> dict[str, Any]:
    if source.is_dir():
        state = validate_checkpoint_bundle(source).run_state.curriculum_state
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if "curriculum_state" in raw:
            state = LongRunRunState.from_mapping(raw).curriculum_state
        else:
            state = raw
    return CurriculumState.from_mapping(state).to_dict()


def _summarize_reports(paths: Sequence[Path], *, root: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        source = path if path.is_absolute() else root / path
        _require_results_path(root, source)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("curriculum report must contain a mapping")
        reports.append(raw)
    return {
        "schema_version": 1,
        "report_count": len(reports),
        "stage_ids": sorted(
            {
                str(item["stage_id"])
                for item in reports
                if item.get("stage_id") is not None
            }
        ),
        "promotion_count": sum(
            item.get("status") in {"promoted", "curriculum_complete"}
            for item in reports
        ),
        "live_training_executed": False,
    }


def _require_results_path(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to((root / "results").resolve())
    except ValueError as exc:
        raise ValueError("curriculum input/output must remain under results/") from exc


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"M13.8 command failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
