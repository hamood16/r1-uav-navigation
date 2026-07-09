"""Utilities for loading project configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from r1_uav_nav.envs import DynamicGridUAVEnv, GridUAVEnv

_GRID_UAV_ENV_NAME = "GridUAVEnv"
_DYNAMIC_GRID_UAV_ENV_NAME = "DynamicGridUAVEnv"
_GRID_UAV_ENV_REQUIRED_KEYS = (
    "grid_size",
    "max_steps",
    "num_obstacles",
    "random_start",
    "random_goal",
)
_GRID_UAV_ENV_OPTIONAL_KEYS = (
    "use_lidar",
    "step_penalty",
    "hover_penalty",
    "boundary_penalty",
    "collision_penalty",
    "goal_reward",
    "timeout_penalty",
    "progress_reward_scale",
)
_DYNAMIC_GRID_UAV_ENV_REQUIRED_KEYS = (
    "grid_size",
    "max_steps",
    "num_dynamic_obstacles",
    "random_start",
    "random_goal",
)
_DYNAMIC_GRID_UAV_ENV_OPTIONAL_KEYS = (
    "step_penalty",
    "hover_penalty",
    "boundary_penalty",
    "collision_penalty",
    "goal_reward",
    "timeout_penalty",
    "progress_reward_scale",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary: {config_path}")

    return config


def create_grid_uav_env_from_config(path: str | Path) -> GridUAVEnv:
    """Create a GridUAVEnv from a YAML configuration file."""
    config = load_config(path)

    env_name = config.get("env_name")
    if env_name != _GRID_UAV_ENV_NAME:
        raise ValueError(
            f"Expected env_name to be {_GRID_UAV_ENV_NAME!r}, got {env_name!r}"
        )

    missing_keys = [key for key in _GRID_UAV_ENV_REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            "Missing required GridUAVEnv config keys: " + ", ".join(missing_keys)
        )

    env_kwargs = {
        "grid_size": config["grid_size"],
        "max_steps": config["max_steps"],
        "num_obstacles": config["num_obstacles"],
        "random_start": config["random_start"],
        "random_goal": config["random_goal"],
    }
    env_kwargs.update(
        {key: config[key] for key in _GRID_UAV_ENV_OPTIONAL_KEYS if key in config}
    )

    return GridUAVEnv(**env_kwargs)


def create_dynamic_grid_uav_env_from_config(path: str | Path) -> DynamicGridUAVEnv:
    """Create a DynamicGridUAVEnv from a YAML configuration file."""
    config = load_config(path)

    env_name = config.get("env_name")
    if env_name != _DYNAMIC_GRID_UAV_ENV_NAME:
        raise ValueError(
            f"Expected env_name to be {_DYNAMIC_GRID_UAV_ENV_NAME!r}, got "
            f"{env_name!r}"
        )

    missing_keys = [
        key for key in _DYNAMIC_GRID_UAV_ENV_REQUIRED_KEYS if key not in config
    ]
    if missing_keys:
        raise ValueError(
            "Missing required DynamicGridUAVEnv config keys: " + ", ".join(missing_keys)
        )

    env_kwargs = {
        "grid_size": config["grid_size"],
        "max_steps": config["max_steps"],
        "num_dynamic_obstacles": config["num_dynamic_obstacles"],
        "random_start": config["random_start"],
        "random_goal": config["random_goal"],
    }
    env_kwargs.update(
        {
            key: config[key]
            for key in _DYNAMIC_GRID_UAV_ENV_OPTIONAL_KEYS
            if key in config
        }
    )

    return DynamicGridUAVEnv(**env_kwargs)
