"""Stable-Baselines3 DQN agent helpers."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import DQN


def create_dqn_model(
    env: Any,
    training_config: dict[str, Any],
    tensorboard_log: str | None = None,
) -> DQN:
    """Create an untrained Stable-Baselines3 DQN model."""
    algorithm = training_config.get("algorithm")
    if algorithm != "DQN":
        raise ValueError(f"Expected algorithm to be 'DQN', got {algorithm!r}")

    return DQN(
        policy=training_config["policy"],
        env=env,
        learning_rate=training_config["learning_rate"],
        buffer_size=training_config["buffer_size"],
        learning_starts=training_config["learning_starts"],
        batch_size=training_config["batch_size"],
        gamma=training_config["gamma"],
        train_freq=training_config["train_freq"],
        target_update_interval=training_config["target_update_interval"],
        exploration_initial_eps=training_config["exploration_initial_eps"],
        exploration_final_eps=training_config["exploration_final_eps"],
        exploration_fraction=training_config["exploration_fraction"],
        seed=training_config["seed"],
        verbose=training_config["verbose"],
        device=training_config["device"],
        tensorboard_log=tensorboard_log,
    )
