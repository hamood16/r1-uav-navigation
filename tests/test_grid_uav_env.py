import math

import numpy as np
import pytest

from r1_uav_nav.envs import GridUAVEnv


def test_environment_can_be_instantiated() -> None:
    env = GridUAVEnv()

    assert env.action_space.n == 5


@pytest.mark.parametrize(
    ("kwargs", "error_message"),
    [
        ({"grid_size": 1}, "grid_size must be at least 2"),
        ({"max_steps": 0}, "max_steps must be at least 1"),
        ({"num_obstacles": -1}, "num_obstacles cannot be negative"),
        (
            {"grid_size": 3, "num_obstacles": 8},
            "num_obstacles cannot exceed 7",
        ),
    ],
)
def test_constructor_rejects_invalid_parameters(
    kwargs: dict[str, int], error_message: str
) -> None:
    with pytest.raises(ValueError, match=error_message):
        GridUAVEnv(**kwargs)


def test_reset_returns_valid_observation_and_info() -> None:
    env = GridUAVEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_default_step_observation_shape_remains_five_values() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0)
    env.reset(seed=42)

    observation, _, _, _, _ = env.step(4)

    assert observation.shape == (5,)


def test_lidar_reset_observation_has_nine_values_and_belongs_to_space() -> None:
    env = GridUAVEnv(use_lidar=True)

    observation, _ = env.reset(seed=42)

    assert observation.shape == (9,)
    assert env.observation_space.contains(observation)


def test_lidar_values_are_appended_in_up_down_left_right_order() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0, use_lidar=True)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = {(2, 3), (0, 2)}

    observation = env._get_observation()

    assert observation[-4:].tolist() == pytest.approx([0.0, 2.0, 1.0, 2.0])


def test_lidar_detects_wall_clearance_correctly() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0, use_lidar=True)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = set()

    observation = env._get_observation()

    assert observation[-4:].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_lidar_updates_after_movement() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0, use_lidar=True)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = {(2, 4)}

    observation, _, _, _, _ = env.step(3)

    assert env.uav_position == (3, 2)
    assert observation[-4:].tolist() == pytest.approx([2.0, 2.0, 3.0, 1.0])


def test_reset_places_all_entities_in_distinct_valid_cells() -> None:
    env = GridUAVEnv(grid_size=6, num_obstacles=8)

    env.reset(seed=42)

    positions = [env.uav_position, env.goal_position, *env.obstacles]
    assert all(0 <= x < env.grid_size and 0 <= y < env.grid_size for x, y in positions)
    assert env.uav_position != env.goal_position
    assert env.uav_position not in env.obstacles
    assert env.goal_position not in env.obstacles
    assert len(env.obstacles) == env.num_obstacles


def test_reset_uses_fixed_start_and_goal_when_randomisation_is_disabled() -> None:
    env = GridUAVEnv(
        grid_size=6,
        num_obstacles=4,
        random_start=False,
        random_goal=False,
    )

    env.reset(seed=42)

    assert env.uav_position == (0, 0)
    assert env.goal_position == (5, 5)


def test_reset_is_reproducible_with_the_same_seed() -> None:
    first_env = GridUAVEnv(grid_size=6, num_obstacles=8)
    second_env = GridUAVEnv(grid_size=6, num_obstacles=8)

    first_observation, _ = first_env.reset(seed=123)
    second_observation, _ = second_env.reset(seed=123)

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_env.uav_position == second_env.uav_position
    assert first_env.goal_position == second_env.goal_position
    assert first_env.obstacles == second_env.obstacles


@pytest.mark.parametrize("action", [1, 2])
def test_boundary_attempt_keeps_position_and_applies_penalty(action: int) -> None:
    env = GridUAVEnv(
        grid_size=4,
        max_steps=2,
        num_obstacles=0,
        random_start=False,
        random_goal=False,
    )
    env.reset(seed=42)

    _, reward, terminated, truncated, _ = env.step(action)

    assert env.uav_position == (0, 0)
    assert reward == pytest.approx(-0.10)
    assert terminated is False
    assert truncated is False
    assert env.current_step == 1


@pytest.mark.parametrize(
    ("action", "expected_position"),
    [
        (0, (2, 3)),
        (1, (2, 1)),
        (2, (1, 2)),
        (3, (3, 2)),
    ],
)
def test_movement_actions_update_position(
    action: int, expected_position: tuple[int, int]
) -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(action)

    assert env.uav_position == expected_position
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_hover_keeps_position_and_applies_ordinary_reward() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)

    _, reward, terminated, truncated, _ = env.step(4)

    assert env.uav_position == (2, 2)
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_obstacle_collision_keeps_position_and_terminates() -> None:
    env = GridUAVEnv(grid_size=5, max_steps=1, num_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = {(2, 3)}

    _, reward, terminated, truncated, _ = env.step(0)

    assert env.uav_position == (2, 2)
    assert reward == pytest.approx(-1.0)
    assert terminated is True
    assert truncated is False


def test_reaching_goal_updates_position_and_terminates() -> None:
    env = GridUAVEnv(grid_size=5, max_steps=1, num_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (2, 3)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(0)

    assert env.uav_position == env.goal_position
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False


def test_custom_goal_reward_is_used() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0, goal_reward=10.0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (2, 3)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == pytest.approx(10.0)
    assert terminated is True
    assert truncated is False


def test_custom_collision_penalty_is_used() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0, collision_penalty=-2.0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = {(2, 3)}

    _, reward, terminated, truncated, _ = env.step(0)

    assert env.uav_position == (2, 2)
    assert reward == pytest.approx(-2.0)
    assert terminated is True
    assert truncated is False


def test_progress_reward_is_positive_when_moving_closer_to_goal() -> None:
    env = GridUAVEnv(
        grid_size=5,
        num_obstacles=0,
        step_penalty=-0.02,
        progress_reward_scale=0.5,
    )
    env.reset(seed=42)
    env.uav_position = (0, 0)
    env.goal_position = (2, 0)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(3)

    assert env.uav_position == (1, 0)
    assert reward == pytest.approx(0.48)
    assert terminated is False
    assert truncated is False


def test_progress_reward_is_negative_when_moving_farther_from_goal() -> None:
    env = GridUAVEnv(
        grid_size=5,
        num_obstacles=0,
        step_penalty=-0.02,
        progress_reward_scale=0.5,
    )
    env.reset(seed=42)
    env.uav_position = (1, 0)
    env.goal_position = (0, 0)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(3)

    assert env.uav_position == (2, 0)
    assert reward == pytest.approx(-0.52)
    assert terminated is False
    assert truncated is False


def test_hover_has_zero_progress_shaping() -> None:
    env = GridUAVEnv(
        grid_size=5,
        num_obstacles=0,
        hover_penalty=-0.05,
        progress_reward_scale=1.0,
    )
    env.reset(seed=42)
    env.uav_position = (1, 0)
    env.goal_position = (4, 0)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(4)

    assert env.uav_position == (1, 0)
    assert reward == pytest.approx(-0.05)
    assert terminated is False
    assert truncated is False


def test_max_steps_truncates_episode_without_termination() -> None:
    env = GridUAVEnv(grid_size=5, max_steps=2, num_obstacles=0)
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)

    _, _, first_terminated, first_truncated, _ = env.step(4)
    _, _, second_terminated, second_truncated, _ = env.step(4)

    assert first_terminated is False
    assert first_truncated is False
    assert second_terminated is False
    assert second_truncated is True
    assert env.current_step == 2


def test_timeout_penalty_is_added_on_truncation() -> None:
    env = GridUAVEnv(
        grid_size=5,
        max_steps=1,
        num_obstacles=0,
        step_penalty=-0.02,
        timeout_penalty=-2.0,
    )
    env.reset(seed=42)
    env.uav_position = (2, 2)
    env.goal_position = (4, 4)
    env.obstacles = set()

    _, reward, terminated, truncated, _ = env.step(4)

    assert reward == pytest.approx(-2.02)
    assert terminated is False
    assert truncated is True


@pytest.mark.parametrize("action", [-1, 5, 1.5, "up", None])
def test_invalid_actions_raise_value_error(action: object) -> None:
    env = GridUAVEnv()
    env.reset(seed=42)

    with pytest.raises(ValueError, match="invalid action"):
        env.step(action)  # type: ignore[arg-type]


def test_no_obstacles_uses_grid_diagonal_for_distance() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0)

    observation, _ = env.reset(seed=42)

    assert observation[4] == pytest.approx(math.hypot(4, 4))


def test_reset_and_step_observations_belong_to_observation_space() -> None:
    env = GridUAVEnv(grid_size=5, num_obstacles=0)

    reset_observation, _ = env.reset(seed=42)
    step_observation, _, _, _, _ = env.step(4)

    assert env.observation_space.contains(reset_observation)
    assert env.observation_space.contains(step_observation)


def test_step_returns_gymnasium_five_value_tuple() -> None:
    env = GridUAVEnv()
    env.reset(seed=42)

    result = env.step(4)

    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_environment_runs_for_random_steps() -> None:
    env = GridUAVEnv(max_steps=8)
    env.reset(seed=42)

    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
