"""Stable-Baselines3 TD3 agent helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise


def create_td3_model(
    env: Any,
    training_config: dict[str, Any],
    tensorboard_log: str | None = None,
) -> TD3:
    """Create an untrained Stable-Baselines3 TD3 model."""
    algorithm = training_config.get("algorithm")
    if algorithm != "TD3":
        raise ValueError(f"Expected algorithm to be 'TD3', got {algorithm!r}")

    n_actions = env.action_space.shape[-1]
    action_noise_std = training_config["action_noise_std"]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_std * np.ones(n_actions),
    )

    return TD3(
        policy=training_config["policy"],
        env=env,
        learning_rate=training_config["learning_rate"],
        buffer_size=training_config["buffer_size"],
        learning_starts=training_config["learning_starts"],
        batch_size=training_config["batch_size"],
        gamma=training_config["gamma"],
        tau=training_config["tau"],
        train_freq=training_config["train_freq"],
        gradient_steps=training_config["gradient_steps"],
        policy_delay=training_config["policy_delay"],
        target_policy_noise=training_config["target_policy_noise"],
        target_noise_clip=training_config["target_noise_clip"],
        action_noise=action_noise,
        seed=training_config["seed"],
        verbose=training_config["verbose"],
        device=training_config["device"],
        tensorboard_log=tensorboard_log,
    )
