from pathlib import Path

import pytest

from r1_uav_nav.envs import (
    ContinuousDynamicUAVEnv,
    DynamicGridUAVEnv,
    GridUAVEnv,
)
from r1_uav_nav.utils import (
    create_continuous_dynamic_uav_env_from_config,
    create_dynamic_grid_uav_env_from_config,
    create_grid_uav_env_from_config,
    load_config,
)


def test_load_config_loads_valid_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env_name: GridUAVEnv",
                "grid_size: 5",
                "max_steps: 25",
                "num_obstacles: 3",
                "random_start: true",
                "random_goal: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == {
        "env_name": "GridUAVEnv",
        "grid_size": 5,
        "max_steps": 25,
        "num_obstacles": 3,
        "random_start": True,
        "random_goal": False,
    }


def test_create_grid_uav_env_from_project_config_returns_environment() -> None:
    env = create_grid_uav_env_from_config("configs/env/grid_2d.yaml")

    assert isinstance(env, GridUAVEnv)


def test_create_grid_uav_env_from_project_config_matches_yaml_values() -> None:
    config = load_config("configs/env/grid_2d.yaml")

    env = create_grid_uav_env_from_config("configs/env/grid_2d.yaml")

    assert env.grid_size == config["grid_size"]
    assert env.max_steps == config["max_steps"]
    assert env.num_obstacles == config["num_obstacles"]
    assert env.random_start is config["random_start"]
    assert env.random_goal is config["random_goal"]
    assert env.use_lidar is False


def test_create_grid_uav_env_from_shaped_config_sets_reward_attributes() -> None:
    config = load_config("configs/env/grid_2d_static_full.yaml")

    env = create_grid_uav_env_from_config("configs/env/grid_2d_static_full.yaml")

    assert isinstance(env, GridUAVEnv)
    assert env.use_lidar is True
    assert env.step_penalty == pytest.approx(config["step_penalty"])
    assert env.hover_penalty == pytest.approx(config["hover_penalty"])
    assert env.boundary_penalty == pytest.approx(config["boundary_penalty"])
    assert env.collision_penalty == pytest.approx(config["collision_penalty"])
    assert env.goal_reward == pytest.approx(config["goal_reward"])
    assert env.timeout_penalty == pytest.approx(config["timeout_penalty"])
    assert env.progress_reward_scale == pytest.approx(config["progress_reward_scale"])


def test_create_dynamic_grid_uav_env_from_config_returns_environment() -> None:
    env = create_dynamic_grid_uav_env_from_config("configs/env/dynamic_grid_2d.yaml")

    assert isinstance(env, DynamicGridUAVEnv)


def test_create_dynamic_grid_uav_env_from_config_matches_yaml_values() -> None:
    config = load_config("configs/env/dynamic_grid_2d.yaml")

    env = create_dynamic_grid_uav_env_from_config("configs/env/dynamic_grid_2d.yaml")

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


def test_create_continuous_dynamic_uav_env_from_config_returns_environment() -> None:
    env = create_continuous_dynamic_uav_env_from_config(
        "configs/env/continuous_dynamic_2d.yaml"
    )

    assert isinstance(env, ContinuousDynamicUAVEnv)


def test_create_continuous_dynamic_uav_env_from_config_matches_yaml_values() -> None:
    config = load_config("configs/env/continuous_dynamic_2d.yaml")

    env = create_continuous_dynamic_uav_env_from_config(
        "configs/env/continuous_dynamic_2d.yaml"
    )

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


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing_path)


def test_load_config_raises_for_empty_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Config file is empty"):
        load_config(config_path)


def test_load_config_raises_for_non_dictionary_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a dictionary"):
        load_config(config_path)


def test_create_grid_uav_env_from_config_raises_for_invalid_env_name(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "wrong_env.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env_name: OtherEnv",
                "grid_size: 5",
                "max_steps: 25",
                "num_obstacles: 3",
                "random_start: true",
                "random_goal: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected env_name"):
        create_grid_uav_env_from_config(config_path)


def test_create_grid_uav_env_from_config_raises_for_missing_required_key(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing_key.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env_name: GridUAVEnv",
                "grid_size: 5",
                "num_obstacles: 3",
                "random_start: true",
                "random_goal: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required GridUAVEnv config keys"):
        create_grid_uav_env_from_config(config_path)
