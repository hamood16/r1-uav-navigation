"""Train the TD3 baseline on ContinuousDynamicUAVEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from stable_baselines3.common.env_checker import check_env

from r1_uav_nav.agents.td3_agent import create_td3_model
from r1_uav_nav.utils import create_continuous_dynamic_uav_env_from_config, load_config

ENV_CONFIG_PATH = Path("configs/env/continuous_dynamic_2d.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/td3_continuous_dynamic_full.yaml")
TENSORBOARD_RUN_NAME = "td3_continuous_dynamic_full"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse TD3 continuous dynamic training arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TD3 model on ContinuousDynamicUAVEnv.",
    )
    parser.add_argument("--env-config", type=Path, default=ENV_CONFIG_PATH)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG_PATH)
    parser.add_argument("--tensorboard-run-name", default=None)
    return parser.parse_args(argv)


def main() -> None:
    """Run full TD3 baseline training."""
    args = parse_args()
    env = create_continuous_dynamic_uav_env_from_config(args.env_config)
    training_config = load_config(args.training_config)

    check_env(env)

    model_output_path = Path(training_config["model_output_path"])
    tensorboard_log_dir = Path(training_config["tensorboard_log_dir"])
    tensorboard_run_name = args.tensorboard_run_name or tensorboard_log_dir.name
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    model = create_td3_model(
        env=env,
        training_config=training_config,
        tensorboard_log=str(tensorboard_log_dir),
    )
    total_timesteps = training_config["total_timesteps"]
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tensorboard_run_name,
    )
    model.save(model_output_path)

    print("TD3 continuous dynamic training complete.")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Model saved to: {model_output_path}")
    print(f"TensorBoard logs: {tensorboard_log_dir}")


if __name__ == "__main__":
    main()
