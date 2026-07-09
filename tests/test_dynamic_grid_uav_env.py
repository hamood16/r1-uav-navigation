import numpy as np
import pytest

from r1_uav_nav.envs import DynamicGridUAVEnv, DynamicObstacle


def test_dynamic_environment_can_be_instantiated() -> None:
    env = DynamicGridUAVEnv()

    assert env.action_space.n == 5


def test_dynamic_reset_returns_valid_observation_and_info() -> None:
    env = DynamicGridUAVEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (9,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_dynamic_reset_places_obstacles_inside_grid_without_start_goal_overlap() -> (
    None
):
    env = DynamicGridUAVEnv(grid_size=6, num_dynamic_obstacles=4)

    env.reset(seed=42)

    obstacle_positions = [obstacle.position for obstacle in env.dynamic_obstacles]
    assert all(
        0 <= x < env.grid_size and 0 <= y < env.grid_size for x, y in obstacle_positions
    )
    assert env.uav_position not in obstacle_positions
    assert env.goal_position not in obstacle_positions


def test_dynamic_obstacle_moves_after_step() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0, 0)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = [DynamicObstacle(position=(1, 1), velocity=(1, 0))]

    env.step(4)

    assert env.dynamic_obstacles[0].position == (2, 1)


def test_dynamic_obstacle_bounces_off_wall() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0, 0)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = [DynamicObstacle(position=(4, 2), velocity=(1, 0))]

    env.step(4)

    assert env.dynamic_obstacles[0].position == (3, 2)
    assert env.dynamic_obstacles[0].velocity == (-1, 0)


def test_uav_moving_into_dynamic_obstacle_terminates_with_info() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = [DynamicObstacle(position=(2, 3), velocity=(1, 0))]

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(env.collision_penalty)
    assert terminated is True
    assert truncated is False
    assert info == {
        "is_success": False,
        "is_collision": True,
        "collision_type": "uav_into_obstacle",
    }


def test_dynamic_obstacle_moving_into_uav_terminates_with_info() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = [DynamicObstacle(position=(2, 1), velocity=(0, 1))]

    _, reward, terminated, truncated, info = env.step(4)

    assert reward == pytest.approx(env.collision_penalty)
    assert terminated is True
    assert truncated is False
    assert info == {
        "is_success": False,
        "is_collision": True,
        "collision_type": "obstacle_into_uav",
    }


def test_goal_reaching_sets_success_info_when_no_collision_occurs() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (2, 3)
    env.dynamic_obstacles = []

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(env.goal_reward)
    assert terminated is True
    assert truncated is False
    assert info == {
        "is_success": True,
        "is_collision": False,
        "collision_type": None,
    }


def test_nearest_dynamic_obstacle_fields_appear_in_observation() -> None:
    env = DynamicGridUAVEnv(grid_size=5, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0, 0)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = [
        DynamicObstacle(position=(3, 0), velocity=(-1, 0)),
        DynamicObstacle(position=(1, 0), velocity=(1, 0)),
    ]

    observation = env._get_observation()

    np.testing.assert_array_equal(
        observation,
        np.array([0, 0, 4, 4, 1, 0, 1, 0, 1], dtype=np.float32),
    )


@pytest.mark.parametrize("action", [-1, 5, 1.5, "up", None])
def test_dynamic_invalid_actions_raise_value_error(action: object) -> None:
    env = DynamicGridUAVEnv()
    env.reset(seed=42)

    with pytest.raises(ValueError, match="invalid action"):
        env.step(action)  # type: ignore[arg-type]


def test_dynamic_max_steps_truncates_episode() -> None:
    env = DynamicGridUAVEnv(grid_size=5, max_steps=1, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0, 0)
    env.goal_position = (4, 4)
    env.dynamic_obstacles = []

    _, _, terminated, truncated, info = env.step(4)

    assert terminated is False
    assert truncated is True
    assert info["is_success"] is False
    assert info["is_collision"] is False


def test_dynamic_step_returns_gymnasium_five_value_tuple() -> None:
    env = DynamicGridUAVEnv()
    env.reset(seed=42)

    result = env.step(4)

    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
