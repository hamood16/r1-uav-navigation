from pathlib import Path

import pytest

from r1_uav_nav.envs import ContinuousDynamicUAVEnv, DynamicGridUAVEnv
from r1_uav_nav.utils import (
    create_continuous_dynamic_uav_env_from_config,
    create_dynamic_grid_uav_env_from_config,
    load_config,
)

DYNAMIC_CONFIG_PATHS = {
    "easy": Path("configs/env/dynamic_grid_2d_easy.yaml"),
    "medium": Path("configs/env/dynamic_grid_2d_medium.yaml"),
    "hard": Path("configs/env/dynamic_grid_2d_hard.yaml"),
}
CONTINUOUS_CONFIG_PATHS = {
    "easy": Path("configs/env/continuous_dynamic_2d_easy.yaml"),
    "medium": Path("configs/env/continuous_dynamic_2d_medium.yaml"),
    "hard": Path("configs/env/continuous_dynamic_2d_hard.yaml"),
}
BASELINE_CONFIG_PATHS = [
    Path("configs/env/dynamic_grid_2d.yaml"),
    Path("configs/env/continuous_dynamic_2d.yaml"),
]


@pytest.mark.parametrize(
    "config_path",
    [*DYNAMIC_CONFIG_PATHS.values(), *CONTINUOUS_CONFIG_PATHS.values()],
)
def test_m10_difficulty_configs_load(config_path: Path) -> None:
    config = load_config(config_path)

    assert config


@pytest.mark.parametrize("difficulty, config_path", DYNAMIC_CONFIG_PATHS.items())
def test_dynamic_difficulty_configs_create_env(
    difficulty: str,
    config_path: Path,
) -> None:
    config = load_config(config_path)

    env = create_dynamic_grid_uav_env_from_config(config_path)

    assert difficulty in DYNAMIC_CONFIG_PATHS
    assert isinstance(env, DynamicGridUAVEnv)
    assert env.grid_size == config["grid_size"]
    assert env.max_steps == config["max_steps"]
    assert env.num_dynamic_obstacles == config["num_dynamic_obstacles"]
    assert env.random_start is config["random_start"]
    assert env.random_goal is config["random_goal"]
    assert env.step_penalty == pytest.approx(config["step_penalty"])
    assert env.hover_penalty == pytest.approx(config["hover_penalty"])
    assert env.boundary_penalty == pytest.approx(config["boundary_penalty"])
    assert env.collision_penalty == pytest.approx(config["collision_penalty"])
    assert env.goal_reward == pytest.approx(config["goal_reward"])
    assert env.timeout_penalty == pytest.approx(config["timeout_penalty"])
    assert env.progress_reward_scale == pytest.approx(config["progress_reward_scale"])


@pytest.mark.parametrize("difficulty, config_path", CONTINUOUS_CONFIG_PATHS.items())
def test_continuous_difficulty_configs_create_env(
    difficulty: str,
    config_path: Path,
) -> None:
    config = load_config(config_path)

    env = create_continuous_dynamic_uav_env_from_config(config_path)

    assert difficulty in CONTINUOUS_CONFIG_PATHS
    assert isinstance(env, ContinuousDynamicUAVEnv)
    assert env.world_size == pytest.approx(config["world_size"])
    assert env.max_steps == config["max_steps"]
    assert env.num_dynamic_obstacles == config["num_dynamic_obstacles"]
    assert env.max_uav_speed == pytest.approx(config["max_uav_speed"])
    assert env.obstacle_speed == pytest.approx(config["obstacle_speed"])
    assert env.dt == pytest.approx(config["dt"])
    assert env.collision_radius == pytest.approx(config["collision_radius"])
    assert env.goal_radius == pytest.approx(config["goal_radius"])
    assert env.random_start is config["random_start"]
    assert env.random_goal is config["random_goal"]
    assert env.step_penalty == pytest.approx(config["step_penalty"])
    assert env.boundary_penalty == pytest.approx(config["boundary_penalty"])
    assert env.collision_penalty == pytest.approx(config["collision_penalty"])
    assert env.goal_reward == pytest.approx(config["goal_reward"])
    assert env.timeout_penalty == pytest.approx(config["timeout_penalty"])
    assert env.progress_reward_scale == pytest.approx(config["progress_reward_scale"])


def test_dynamic_difficulty_values_are_monotonic() -> None:
    easy, medium, hard = _load_dynamic_difficulty_configs()

    assert (
        easy["num_dynamic_obstacles"]
        < medium["num_dynamic_obstacles"]
        < hard["num_dynamic_obstacles"]
    )
    assert easy["max_steps"] > medium["max_steps"] > hard["max_steps"]


def test_continuous_difficulty_values_are_monotonic() -> None:
    easy, medium, hard = _load_continuous_difficulty_configs()

    assert (
        easy["num_dynamic_obstacles"]
        < medium["num_dynamic_obstacles"]
        < hard["num_dynamic_obstacles"]
    )
    assert easy["max_steps"] > medium["max_steps"] > hard["max_steps"]
    assert (
        easy["collision_radius"] < medium["collision_radius"] < hard["collision_radius"]
    )
    assert easy["goal_radius"] > medium["goal_radius"] > hard["goal_radius"]


def test_medium_dynamic_config_matches_baseline_values() -> None:
    baseline_config = load_config("configs/env/dynamic_grid_2d.yaml")
    medium_config = load_config(DYNAMIC_CONFIG_PATHS["medium"])

    assert medium_config == baseline_config


def test_medium_continuous_config_matches_baseline_values() -> None:
    baseline_config = load_config("configs/env/continuous_dynamic_2d.yaml")
    medium_config = load_config(CONTINUOUS_CONFIG_PATHS["medium"])

    assert medium_config == baseline_config


@pytest.mark.parametrize("config_path", BASELINE_CONFIG_PATHS)
def test_baseline_config_files_still_exist(config_path: Path) -> None:
    assert config_path.exists()


def _load_dynamic_difficulty_configs() -> tuple[dict, dict, dict]:
    return (
        load_config(DYNAMIC_CONFIG_PATHS["easy"]),
        load_config(DYNAMIC_CONFIG_PATHS["medium"]),
        load_config(DYNAMIC_CONFIG_PATHS["hard"]),
    )


def _load_continuous_difficulty_configs() -> tuple[dict, dict, dict]:
    return (
        load_config(CONTINUOUS_CONFIG_PATHS["easy"]),
        load_config(CONTINUOUS_CONFIG_PATHS["medium"]),
        load_config(CONTINUOUS_CONFIG_PATHS["hard"]),
    )
