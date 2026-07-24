"""Validate deterministic M13.3 static courses without a simulator client."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from r1_uav_nav.sim.colosseum_capabilities import validate_report_output_path
from r1_uav_nav.sim.static_course import (
    ValidatedCourse,
    generate_solvable_course,
    load_course_suite_config,
    save_course_report,
)

DEFAULT_COURSE_CONFIG = Path("configs/planning/m13_3_voxel_astar.yaml")
DEFAULT_REPORT_DIR = Path("results/reports/m13/courses")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M13.3 static-course solvability offline."
    )
    parser.add_argument(
        "--course-config",
        type=Path,
        default=DEFAULT_COURSE_CONFIG,
        help="authoritative course-suite configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="ignored directory for course evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate one declared profile and base seed",
    )
    validate.add_argument(
        "--course-profile",
        "--profile",
        dest="course_profile",
        required=True,
    )
    validate.add_argument("--seed", type=int)
    validate.add_argument("--output-path", type=Path)

    subparsers.add_parser(
        "validate-all",
        help="recompute every tracked profile baseline",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    return run(parse_args())


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    suite = load_course_suite_config(args.course_config)
    courses: list[ValidatedCourse] = []

    if args.command == "validate":
        profile = suite.profile(args.course_profile)
        seed = _select_seed(profile.base_seeds, args.seed)
        courses.append(
            generate_solvable_course(
                suite,
                profile.profile_id,
                seed,
                repository_root=root,
            )
        )
    elif args.command == "validate-all":
        for profile in suite.profiles:
            for seed in profile.base_seeds:
                courses.append(
                    generate_solvable_course(
                        suite,
                        profile.profile_id,
                        seed,
                        repository_root=root,
                    )
                )
    else:
        raise ValueError(f"unsupported command {args.command!r}")

    for index, course in enumerate(courses):
        output = _report_path(args, course, index=index, count=len(courses))
        validate_report_output_path(output, root)
        save_course_report(course, output)
        result = course.result
        path = result.path_result
        print(
            f"{result.profile_id}:{result.base_seed} accepted "
            f"candidate={result.accepted_candidate_seed} "
            f"length={path.reference_path_length_m:.3f} m "
            f"digest={result.solvability_digest}"
        )
        print(f"Report: {output}")
    return 0


def _select_seed(declared_seeds: tuple[int, ...], requested: int | None) -> int:
    if requested is None:
        if len(declared_seeds) != 1:
            raise ValueError(
                "--seed is required when a profile declares multiple base seeds"
            )
        return declared_seeds[0]
    if requested not in declared_seeds:
        raise ValueError(
            f"seed {requested} is not declared for the selected course profile"
        )
    return requested


def _report_path(
    args: argparse.Namespace,
    course: ValidatedCourse,
    *,
    index: int,
    count: int,
) -> Path:
    explicit = getattr(args, "output_path", None)
    if explicit is not None:
        if count != 1 or index != 0:
            raise ValueError("--output-path is valid only for one course")
        return explicit
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = uuid.uuid4().hex[:8]
    return args.output_dir / (
        f"m13_3_{course.result.profile_id}_{course.result.base_seed}_"
        f"{timestamp}_{run_id}.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
