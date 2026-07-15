import pytest
from stable_baselines3 import TD3

from r1_uav_nav.agents.td3_agent import create_td3_model
from r1_uav_nav.envs import ContinuousDynamicUAVEnv


def _valid_td3_config() -> dict[str, object]:
    return {
        "algorithm": "TD3",
        "policy": "MlpPolicy",
        "learning_rate": 0.0003,
        "buffer_size": 100,
        "learning_starts": 10,
        "batch_size": 16,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "action_noise_std": 0.1,
        "seed": 42,
        "verbose": 0,
        "device": "cpu",
    }


def test_create_td3_model_returns_td3_instance() -> None:
    env = ContinuousDynamicUAVEnv(
        world_size=5.0,
        max_steps=20,
        num_dynamic_obstacles=0,
    )

    model = create_td3_model(env=env, training_config=_valid_td3_config())

    assert isinstance(model, TD3)


def test_create_td3_model_rejects_non_td3_algorithm() -> None:
    env = ContinuousDynamicUAVEnv(
        world_size=5.0,
        max_steps=20,
        num_dynamic_obstacles=0,
    )
    config = _valid_td3_config()
    config["algorithm"] = "DQN"

    with pytest.raises(ValueError, match="Expected algorithm"):
        create_td3_model(env=env, training_config=config)
