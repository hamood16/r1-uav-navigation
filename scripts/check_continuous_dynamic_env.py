"""Smoke-check the continuous dynamic UAV environment."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from r1_uav_nav.utils import create_continuous_dynamic_uav_env_from_config

ENV_CONFIG_PATH = Path("configs/env/continuous_dynamic_2d.yaml")


def main() -> None:
    """Run a quick SB3 compatibility and random-action smoke check."""
    env = create_continuous_dynamic_uav_env_from_config(ENV_CONFIG_PATH)
    check_env(env)

    env.reset(seed=42)
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    print("Continuous dynamic UAV environment smoke check passed.")


if __name__ == "__main__":
    main()
