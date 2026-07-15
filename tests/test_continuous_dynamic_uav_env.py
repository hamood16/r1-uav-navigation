import math

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from r1_uav_nav.envs import ContinuousDynamicObstacle, ContinuousDynamicUAVEnv


def test_continuous_dynamic_environment_can_be_instantiated() -> None:
    env = ContinuousDynamicUAVEnv()

    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert isinstance(env.observation_space, spaces.Box)


def test_continuous_dynamic_reset_returns_valid_observation_and_info() -> None:
    env = ContinuousDynamicUAVEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (9,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_continuous_dynamic_reset_avoids_trivial_or_invalid_states() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=4)

    env.reset(seed=42)

    assert _distance(env.uav_position, env.goal_position) >= 2.0 * env.goal_radius
    for obstacle in env.dynamic_obstacles:
        assert _distance(env.uav_position, obstacle.position) > env.collision_radius
        assert _distance(env.goal_position, obstacle.position) > env.goal_radius


def test_continuous_dynamic_uav_moves_with_continuous_action() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (1.0, 1.0)
    env.goal_position = (5.0, 1.0)

    env.step(np.array([0.5, 0.0], dtype=np.float32))

    assert env.uav_position == pytest.approx((1.5, 1.0))


def test_continuous_dynamic_action_is_clipped() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (1.0, 1.0)
    env.goal_position = (5.0, 1.0)

    env.step(np.array([3.0, 0.0], dtype=np.float32))

    assert env.uav_position == pytest.approx((2.0, 1.0))


@pytest.mark.parametrize(
    "action",
    [
        np.array([1.0], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_continuous_dynamic_wrong_shaped_actions_raise_value_error(
    action: np.ndarray,
) -> None:
    env = ContinuousDynamicUAVEnv()
    env.reset(seed=42)

    with pytest.raises(ValueError, match="shape"):
        env.step(action)


def test_continuous_dynamic_boundary_clipping_applies_boundary_penalty() -> None:
    env = ContinuousDynamicUAVEnv(
        num_dynamic_obstacles=0,
        step_penalty=-0.02,
        boundary_penalty=-0.50,
        progress_reward_scale=0.0,
    )
    env.reset(seed=42)
    env.uav_position = (0.1, 1.0)
    env.goal_position = (5.0, 1.0)

    _, reward, terminated, truncated, _ = env.step(
        np.array([-1.0, 0.0], dtype=np.float32)
    )

    assert env.uav_position == pytest.approx((0.0, 1.0))
    assert reward == pytest.approx(-0.52)
    assert terminated is False
    assert truncated is False


def test_continuous_dynamic_obstacle_moves_after_step() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0.0, 0.0)
    env.goal_position = (10.0, 10.0)
    env.dynamic_obstacles = [
        ContinuousDynamicObstacle(position=(2.0, 2.0), velocity=(0.5, 0.0))
    ]

    env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert env.dynamic_obstacles[0].position == pytest.approx((2.5, 2.0))


def test_continuous_dynamic_obstacle_bounces_off_world_boundary() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (0.0, 0.0)
    env.goal_position = (10.0, 10.0)
    env.dynamic_obstacles = [
        ContinuousDynamicObstacle(position=(9.8, 2.0), velocity=(0.5, 0.0))
    ]

    env.step(np.array([0.0, 0.0], dtype=np.float32))

    obstacle = env.dynamic_obstacles[0]
    assert obstacle.position == pytest.approx((9.3, 2.0))
    assert obstacle.velocity == pytest.approx((-0.5, 0.0))


def test_continuous_dynamic_obstacle_state_stays_within_bounds() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=3)

    env.reset(seed=42)
    env.step(np.array([0.0, 0.0], dtype=np.float32))

    for obstacle in env.dynamic_obstacles:
        assert 0.0 <= obstacle.position[0] <= env.world_size
        assert 0.0 <= obstacle.position[1] <= env.world_size
        assert abs(obstacle.velocity[0]) <= env.obstacle_speed
        assert abs(obstacle.velocity[1]) <= env.obstacle_speed


def test_continuous_dynamic_collision_terminates_with_info() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (1.0, 1.0)
    env.goal_position = (10.0, 10.0)
    env.dynamic_obstacles = [
        ContinuousDynamicObstacle(position=(1.2, 1.0), velocity=(0.0, 0.0))
    ]

    _, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert reward == pytest.approx(env.collision_penalty)
    assert terminated is True
    assert truncated is False
    assert info == {
        "is_success": False,
        "is_collision": True,
        "collision_type": "uav_obstacle_collision",
    }


def test_continuous_dynamic_goal_reaching_terminates_with_info() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (1.0, 1.0)
    env.goal_position = (2.0, 1.0)
    env.dynamic_obstacles = []

    _, reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert reward == pytest.approx(env.goal_reward)
    assert terminated is True
    assert truncated is False
    assert info == {
        "is_success": True,
        "is_collision": False,
        "collision_type": None,
    }


def test_continuous_dynamic_max_steps_truncates() -> None:
    env = ContinuousDynamicUAVEnv(max_steps=1, num_dynamic_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (1.0, 1.0)
    env.goal_position = (10.0, 10.0)
    env.dynamic_obstacles = []

    _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is True
    assert info["is_success"] is False
    assert info["is_collision"] is False


def test_continuous_dynamic_step_returns_gymnasium_five_value_tuple() -> None:
    env = ContinuousDynamicUAVEnv()
    env.reset(seed=42)

    result = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (9,)
    assert observation.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_continuous_dynamic_step_observation_belongs_to_observation_space() -> None:
    env = ContinuousDynamicUAVEnv(num_dynamic_obstacles=0)
    env.reset(seed=42)

    observation, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert env.observation_space.contains(observation)


def test_continuous_dynamic_environment_passes_sb3_check_env() -> None:
    env = ContinuousDynamicUAVEnv(
        world_size=5.0,
        max_steps=5,
        num_dynamic_obstacles=1,
    )

    check_env(env)


def _distance(
    first_position: tuple[float, float],
    second_position: tuple[float, float],
) -> float:
    return math.hypot(
        first_position[0] - second_position[0],
        first_position[1] - second_position[1],
    )
