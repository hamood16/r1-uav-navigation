from r1_uav_nav.evaluation.rollout_selection import (
    select_fastest_successful_rollout,
    select_longest_rollout,
    select_mean_length_rollout,
)
from r1_uav_nav.utils import load_config


def test_dynamic_dqn_config_uses_dqn_algorithm() -> None:
    config = load_config("configs/training/dqn_dynamic_full.yaml")

    assert config["algorithm"] == "DQN"


def test_dynamic_dqn_config_uses_full_training_timesteps() -> None:
    config = load_config("configs/training/dqn_dynamic_full.yaml")

    assert config["total_timesteps"] >= 300000


def test_dynamic_dqn_config_uses_dynamic_output_paths() -> None:
    config = load_config("configs/training/dqn_dynamic_full.yaml")

    assert "dqn_dynamic_full.zip" in config["model_output_path"]
    assert "dqn_dynamic_full" in config["tensorboard_log_dir"]


def test_dynamic_dqn_config_paths_differ_from_static_paths() -> None:
    dynamic_config = load_config("configs/training/dqn_dynamic_full.yaml")
    static_config = load_config("configs/training/dqn_static_full.yaml")

    assert dynamic_config["model_output_path"] != static_config["model_output_path"]
    assert dynamic_config["tensorboard_log_dir"] != static_config["tensorboard_log_dir"]


def test_select_fastest_successful_rollout_prefers_lowest_step_success() -> None:
    rollouts = _sample_rollouts()

    selected_rollout = select_fastest_successful_rollout(rollouts)

    assert selected_rollout["name"] == "fast_success"


def test_select_mean_length_rollout_prefers_closest_successful_mean() -> None:
    rollouts = _sample_rollouts()

    selected_rollout = select_mean_length_rollout(rollouts)

    assert selected_rollout["name"] == "medium_success"


def test_select_longest_rollout_prefers_longest_success() -> None:
    rollouts = _sample_rollouts()

    selected_rollout = select_longest_rollout(rollouts)

    assert selected_rollout["name"] == "long_success"


def test_rollout_selection_falls_back_when_no_rollout_succeeds() -> None:
    rollouts = [
        {"name": "first_failure", "success": False, "steps": 9, "path_length": 3.0},
        {"name": "long_failure", "success": False, "steps": 12, "path_length": 9.0},
        {"name": "mid_failure", "success": False, "steps": 10, "path_length": 6.0},
    ]

    assert select_fastest_successful_rollout(rollouts)["name"] == "first_failure"
    assert select_mean_length_rollout(rollouts)["name"] == "mid_failure"
    assert select_longest_rollout(rollouts)["name"] == "long_failure"


def _sample_rollouts() -> list[dict]:
    return [
        {"name": "failure", "success": False, "steps": 5, "path_length": 20.0},
        {"name": "fast_success", "success": True, "steps": 4, "path_length": 2.0},
        {
            "name": "medium_success",
            "success": True,
            "steps": 8,
            "path_length": 5.0,
        },
        {"name": "long_success", "success": True, "steps": 12, "path_length": 8.0},
    ]
