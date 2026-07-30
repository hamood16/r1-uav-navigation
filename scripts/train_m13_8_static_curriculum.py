"""Offline-only M13.8 static-obstacle curriculum tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from r1_uav_nav.training.curriculum import (
    DEFAULT_CURRICULUM_CONFIG_PATH,
    CurriculumState,
    enable_curriculum_mode,
    load_curriculum_config,
    validate_curriculum_configuration,
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
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
    raise ValueError(f"unsupported command {args.command!r}")


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
        print(f"M13.8 offline command failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
