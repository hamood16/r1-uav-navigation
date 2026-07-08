from r1_uav_nav.utils import load_config


def test_static_dqn_config_uses_dqn_algorithm() -> None:
    config = load_config("configs/training/dqn_static_full.yaml")

    assert config["algorithm"] == "DQN"


def test_static_dqn_config_trains_longer_than_debug_config() -> None:
    static_config = load_config("configs/training/dqn_static_full.yaml")
    debug_config = load_config("configs/training/dqn_debug.yaml")

    assert static_config["total_timesteps"] > debug_config["total_timesteps"]


def test_static_dqn_config_uses_static_output_paths() -> None:
    config = load_config("configs/training/dqn_static_full.yaml")

    assert config["model_output_path"] == "results/trained_models/dqn_static_full.zip"
    assert config["tensorboard_log_dir"] == "results/logs/dqn_static_full"
