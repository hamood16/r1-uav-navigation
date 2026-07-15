from r1_uav_nav.utils import load_config


def test_td3_continuous_dynamic_config_loads() -> None:
    config = load_config("configs/training/td3_continuous_dynamic_full.yaml")

    assert config["algorithm"] == "TD3"


def test_td3_continuous_dynamic_config_values() -> None:
    config = load_config("configs/training/td3_continuous_dynamic_full.yaml")

    assert config["total_timesteps"] >= 300000
    assert config["action_noise_std"] > 0
    assert "td3_continuous_dynamic_full.zip" in config["model_output_path"]
    assert "td3_continuous_dynamic_full" in config["tensorboard_log_dir"]


def test_td3_paths_are_separate_from_dqn_paths() -> None:
    td3_config = load_config("configs/training/td3_continuous_dynamic_full.yaml")
    dqn_config = load_config("configs/training/dqn_dynamic_full.yaml")

    assert td3_config["model_output_path"] != dqn_config["model_output_path"]
    assert td3_config["tensorboard_log_dir"] != dqn_config["tensorboard_log_dir"]
