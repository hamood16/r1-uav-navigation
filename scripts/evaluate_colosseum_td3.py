"""Evaluate TD3, random, or scripted-forward policies in ColosseumUAVEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from r1_uav_nav.training import (
    DEFAULT_CONFIG_PATH,
    EVALUATION_POLICY_KINDS,
    apply_evaluation_overrides,
    evaluate_colosseum_td3,
    load_colosseum_td3_config,
    resolve_device,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Colosseum TD3 evaluation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Colosseum TD3 and baseline policies.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--policy",
        choices=EVALUATION_POLICY_KINDS,
        default=None,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> int:
    """Run Colosseum TD3 or baseline evaluation."""
    args = parse_args()
    try:
        config = load_colosseum_td3_config(args.config)
        config = apply_evaluation_overrides(
            config,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            policy=args.policy,
            device=args.device,
            seed=args.seed,
            reports_dir=args.reports_dir,
        )
        print(f"Resolved device: {resolve_device(config.device)}")
        result = evaluate_colosseum_td3(config)
    except Exception as exc:
        print(f"Colosseum TD3 evaluation failed before startup: {exc}")
        return 1

    if result.error_message:
        print(f"Colosseum TD3 evaluation failed: {result.error_message}")
    if result.cleanup_error_message:
        print(f"Cleanup error: {result.cleanup_error_message}")
    if result.metrics_error_message:
        print(f"Metrics error: {result.metrics_error_message}")
    if result.metrics_path is not None:
        print(f"Evaluation metrics: {result.metrics_path}")
    if result.cleanup_safety_critical_failure:
        print("Safety-critical cleanup failure occurred.")
    if result.exit_code == 0:
        print("Colosseum TD3 evaluation complete.")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
