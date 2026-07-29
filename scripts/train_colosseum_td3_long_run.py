"""Offline Phase A tooling for M13.7 long-run TD3 infrastructure."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from r1_uav_nav.training.long_run_state import (
    ResumeMode,
    validate_resume_request,
)
from r1_uav_nav.training.long_run_training import (
    DEFAULT_LONG_RUN_CONFIG_PATH,
    load_long_run_td3_config,
    run_fake_td3_smoke,
    validate_long_run_configuration,
)
from r1_uav_nav.training.supervisor import run_fake_worker_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M13.7 resumable TD3 infrastructure offline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_LONG_RUN_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate configuration and optional resume evidence"
    )
    _add_resume_arguments(validate)

    inspect_resume = subparsers.add_parser(
        "inspect-resume", help="inspect one complete safe resume bundle"
    )
    _add_resume_arguments(inspect_resume)

    fake = subparsers.add_parser(
        "fake-smoke", help="run a tiny simulator-independent TD3 save/resume smoke"
    )
    _add_resume_arguments(fake)
    fake.add_argument("--additional-timesteps", type=int)
    fake.add_argument("--run-id")

    supervisor = subparsers.add_parser(
        "supervisor-smoke", help="spawn a short non-simulator heartbeat worker"
    )
    supervisor.add_argument(
        "--heartbeat-output",
        type=Path,
        default=Path("results/reports/m13/training/fake_worker_heartbeat.json"),
    )
    return parser


def _add_resume_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument("--resume-run-state", type=Path)
    parser.add_argument("--resume-latest", type=Path)
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--allow-partial-resume", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_long_run_td3_config(config_path)
    validation = validate_long_run_configuration(config, repository_root=root)

    if args.command == "supervisor-smoke":
        output = (
            args.heartbeat_output
            if args.heartbeat_output.is_absolute()
            else root / args.heartbeat_output
        )
        _require_results_output(root, output)
        heartbeat = run_fake_worker_smoke(output)
        print(json.dumps(asdict(heartbeat), indent=2, sort_keys=True, default=str))
        return 0

    plan = _resume_plan(args, config.compatibility_digest)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    **validation,
                    "resume_mode": plan.mode.value,
                    "live_training_enabled": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect-resume":
        if plan.mode is ResumeMode.NEW:
            raise ValueError("inspect-resume requires full or warm-start evidence")
        print(
            json.dumps(
                {
                    "mode": plan.mode.value,
                    "source_run_id": plan.source_run_id,
                    "source_model_digest": plan.source_model_digest,
                    "model": str(plan.model_path),
                    "replay_buffer": str(plan.replay_buffer_path),
                    "run_state": str(plan.run_state_path),
                    "reset_num_timesteps": plan.reset_num_timesteps,
                    "creates_new_lineage": plan.creates_new_lineage,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command != "fake-smoke":
        raise ValueError(f"unsupported command {args.command!r}")
    if args.additional_timesteps is not None:
        if args.additional_timesteps <= 0:
            raise ValueError("--additional-timesteps must be positive")
        config = replace(config, additional_timesteps=args.additional_timesteps)
    result = run_fake_td3_smoke(
        config,
        repository_root=root,
        resume_plan=plan,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "resume_mode": result.resume_mode.value,
                "initial_timesteps": result.initial_timesteps,
                "final_timesteps": result.final_timesteps,
                "checkpoint": str(result.checkpoint.directory),
                "live_training_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resume_plan(args: argparse.Namespace, compatibility_digest: str):
    return validate_resume_request(
        resume_checkpoint=getattr(args, "resume_checkpoint", None),
        resume_replay_buffer=getattr(args, "resume_replay_buffer", None),
        resume_run_state=getattr(args, "resume_run_state", None),
        resume_latest=getattr(args, "resume_latest", None),
        reset_num_timesteps=getattr(args, "reset_num_timesteps", False),
        allow_partial_resume=getattr(args, "allow_partial_resume", False),
        expected_compatibility_digest=compatibility_digest,
    )


def _require_results_output(root: Path, output: Path) -> None:
    try:
        output.resolve().relative_to((root / "results").resolve())
    except ValueError as exc:
        raise ValueError(
            "output must remain under the ignored results directory"
        ) from exc


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"M13.7 offline command failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
