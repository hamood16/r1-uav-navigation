"""Offline Phase A throughput evidence tooling for M13.7."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from r1_uav_nav.training.long_run_state import canonical_digest
from r1_uav_nav.training.throughput_benchmark import (
    ThroughputBenchmarkConfig,
    ThroughputEpisodeSample,
    load_throughput_report,
    save_throughput_report,
    summarize_throughput,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M13.7 throughput evidence offline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the Phase A benchmark contract")
    fake = subparsers.add_parser(
        "fake-smoke", help="write deterministic fake timing evidence"
    )
    fake.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/reports/m13/training/benchmarks/" "m13_7_fake_throughput.json"
        ),
    )
    summary = subparsers.add_parser(
        "summarize", help="summarize explicit throughput reports"
    )
    summary.add_argument("reports", type=Path, nargs="+")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_root: Path | None = None,
) -> int:
    root = (repository_root or Path.cwd()).resolve()
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "modes": ["validate", "fake-smoke", "summarize"],
                    "live_benchmark_enabled": False,
                    "simulator_clock_scale": 1.0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "summarize":
        reports = [load_throughput_report(path) for path in args.reports]
        accepted = all(bool(report["accepted"]) for report in reports)
        print(
            json.dumps(
                {
                    "report_count": len(reports),
                    "all_accepted": accepted,
                    "variants": [report["variant_id"] for report in reports],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if accepted else 1
    if args.command != "fake-smoke":
        raise ValueError(f"unsupported command {args.command!r}")
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        output.resolve().relative_to((root / "results").resolve())
    except ValueError as exc:
        raise ValueError("benchmark output must remain under results/") from exc
    config = ThroughputBenchmarkConfig(
        variant_id="offline-fake",
        environment_config_digest=canonical_digest("fake-m13-5-environment"),
        control_duration_s=0.2,
        checkpoint_interval_steps=1_000,
    )
    report = summarize_throughput(
        config,
        (
            ThroughputEpisodeSample(
                reset_latency_s=0.01,
                action_step_latencies_s=(0.02, 0.02, 0.03),
                lidar_latencies_s=(0.005, 0.006, 0.005),
                episode_wall_time_s=0.09,
                active_step_time_s=0.07,
                checkpoint_overhead_s=0.01,
                logging_overhead_s=0.001,
                cleanup_success=True,
                cleanup_status="fake_success",
            ),
        ),
    )
    save_throughput_report(report, output)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"M13.7 throughput command failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
