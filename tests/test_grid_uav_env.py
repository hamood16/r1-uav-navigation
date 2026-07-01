import numpy as np

from r1_uav_nav.envs import GridUAVEnv


def test_environment_can_be_instantiated() -> None:
    env = GridUAVEnv()

    assert env.action_space.n == 5


def test_reset_returns_valid_observation_and_info() -> None:
    env = GridUAVEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


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
