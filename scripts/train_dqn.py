"""Train a debug DQN agent on the GridUAVEnv environment."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from r1_uav_nav.agents.dqn_agent import create_dqn_model
from r1_uav_nav.utils import create_grid_uav_env_from_config, load_config

ENV_CONFIG_PATH = Path("configs/env/grid_2d.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/dqn_debug.yaml")


def main() -> None:
    """Run a short DQN training job using YAML configuration files."""
    env = create_grid_uav_env_from_config(ENV_CONFIG_PATH)
    training_config = load_config(TRAINING_CONFIG_PATH)

    check_env(env)

    model_output_path = Path(training_config["model_output_path"])
    tensorboard_log_dir = Path(training_config["tensorboard_log_dir"])
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    model = create_dqn_model(
        env=env,
        training_config=training_config,
        tensorboard_log=str(tensorboard_log_dir),
    )
    total_timesteps = training_config["total_timesteps"]
    model.learn(total_timesteps=total_timesteps)
    model.save(model_output_path)

    print("DQN training complete.")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Model saved to: {model_output_path}")
    print(f"TensorBoard logs: {tensorboard_log_dir}")


if __name__ == "__main__":
    main()
