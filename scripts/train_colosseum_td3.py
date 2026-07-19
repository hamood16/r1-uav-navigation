"""Train a TD3 baseline on the live Colosseum Gymnasium UAV environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from r1_uav_nav.training import (
    DEFAULT_CONFIG_PATH,
    apply_training_overrides,
    load_colosseum_td3_config,
    resolve_device,
    train_colosseum_td3,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Colosseum TD3 training CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a bounded TD3 baseline on ColosseumUAVEnv.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-output-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> int:
    """Run Colosseum TD3 training."""
    args = parse_args()
    try:
        config = load_colosseum_td3_config(args.config)
        config = apply_training_overrides(
            config,
            total_timesteps=args.total_timesteps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
            seed=args.seed,
            model_output_dir=args.model_output_dir,
            tensorboard_log_dir=args.tensorboard_log_dir,
            reports_dir=args.reports_dir,
        )
        print(f"Resolved device: {resolve_device(config.device)}")
        result = train_colosseum_td3(config)
    except Exception as exc:
        print(f"Colosseum TD3 training failed before startup: {exc}")
        return 1

    if result.error_message:
        print(f"Colosseum TD3 training failed: {result.error_message}")
    if result.checkpoint_error_message:
        print(f"Checkpoint error: {result.checkpoint_error_message}")
    if result.cleanup_error_message:
        print(f"Cleanup error: {result.cleanup_error_message}")
    if result.metrics_error_message:
        print(f"Metrics error: {result.metrics_error_message}")
    if result.final_checkpoint_path is not None:
        print(f"Final checkpoint: {result.final_checkpoint_path}")
    if result.metrics_path is not None:
        print(f"Training metrics: {result.metrics_path}")
    if result.cleanup_safety_critical_failure:
        print("Safety-critical cleanup failure occurred.")
    if result.exit_code == 0:
        print("Colosseum TD3 training complete.")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
