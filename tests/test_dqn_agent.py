import pytest
from stable_baselines3 import DQN

from r1_uav_nav.agents.dqn_agent import create_dqn_model
from r1_uav_nav.envs import GridUAVEnv


def _valid_dqn_config() -> dict[str, object]:
    return {
        "algorithm": "DQN",
        "policy": "MlpPolicy",
        "learning_rate": 0.0001,
        "buffer_size": 100,
        "learning_starts": 10,
        "batch_size": 16,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 50,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.3,
        "seed": 42,
        "verbose": 0,
        "device": "cpu",
    }


def test_create_dqn_model_returns_dqn_instance() -> None:
    env = GridUAVEnv(grid_size=5, max_steps=20, num_obstacles=0)

    model = create_dqn_model(env=env, training_config=_valid_dqn_config())

    assert isinstance(model, DQN)


def test_create_dqn_model_rejects_non_dqn_algorithm() -> None:
    env = GridUAVEnv(grid_size=5, max_steps=20, num_obstacles=0)
    config = _valid_dqn_config()
    config["algorithm"] = "TD3"

    with pytest.raises(ValueError, match="Expected algorithm"):
        create_dqn_model(env=env, training_config=config)
