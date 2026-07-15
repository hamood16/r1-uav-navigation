from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from r1_uav_nav.utils import load_config

DQN_BASELINE_CONFIG_PATH = Path("configs/training/dqn_dynamic_full.yaml")
DQN_TUNED_CONFIG_PATH = Path("configs/training/dqn_dynamic_tuned.yaml")
TD3_BASELINE_CONFIG_PATH = Path("configs/training/td3_continuous_dynamic_full.yaml")
TD3_TUNED_CONFIG_PATH = Path("configs/training/td3_continuous_dynamic_tuned.yaml")

REQUIRED_DQN_KEYS = {
    "algorithm",
    "policy",
    "total_timesteps",
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "train_freq",
    "target_update_interval",
    "exploration_initial_eps",
    "exploration_final_eps",
    "exploration_fraction",
    "seed",
    "verbose",
    "device",
    "model_output_path",
    "tensorboard_log_dir",
}
REQUIRED_TD3_KEYS = {
    "algorithm",
    "policy",
    "total_timesteps",
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "tau",
    "train_freq",
    "gradient_steps",
    "policy_delay",
    "target_policy_noise",
    "target_noise_clip",
    "action_noise_std",
    "seed",
    "verbose",
    "device",
    "model_output_path",
    "tensorboard_log_dir",
}


def test_tuned_dqn_config_loads_with_expected_values() -> None:
    baseline_config = load_config(DQN_BASELINE_CONFIG_PATH)
    tuned_config = load_config(DQN_TUNED_CONFIG_PATH)

    assert tuned_config["algorithm"] == "DQN"
    assert REQUIRED_DQN_KEYS.issubset(tuned_config)
    assert tuned_config["total_timesteps"] >= baseline_config["total_timesteps"]
    assert tuned_config["model_output_path"] != baseline_config["model_output_path"]
    assert tuned_config["tensorboard_log_dir"] != baseline_config["tensorboard_log_dir"]
    assert tuned_config["model_output_path"] == (
        "results/trained_models/dqn_dynamic_tuned.zip"
    )
    assert tuned_config["tensorboard_log_dir"] == "results/logs/dqn_dynamic_tuned"


def test_tuned_td3_config_loads_with_expected_values() -> None:
    baseline_config = load_config(TD3_BASELINE_CONFIG_PATH)
    tuned_config = load_config(TD3_TUNED_CONFIG_PATH)

    assert tuned_config["algorithm"] == "TD3"
    assert REQUIRED_TD3_KEYS.issubset(tuned_config)
    assert tuned_config["total_timesteps"] >= baseline_config["total_timesteps"]
    assert tuned_config["model_output_path"] != baseline_config["model_output_path"]
    assert tuned_config["tensorboard_log_dir"] != baseline_config["tensorboard_log_dir"]
    assert tuned_config["model_output_path"] == (
        "results/trained_models/td3_continuous_dynamic_tuned.zip"
    )
    assert tuned_config["tensorboard_log_dir"] == (
        "results/logs/td3_continuous_dynamic_tuned"
    )


def test_dynamic_dqn_train_parse_args_defaults() -> None:
    module = _load_script_module("train_dynamic_dqn.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/dynamic_grid_2d.yaml")
    assert args.training_config == DQN_BASELINE_CONFIG_PATH
    assert args.tensorboard_run_name is None


def test_dynamic_dqn_train_parse_args_custom_values() -> None:
    module = _load_script_module("train_dynamic_dqn.py")

    args = module.parse_args(
        [
            "--env-config",
            "configs/env/dynamic_grid_2d_medium.yaml",
            "--training-config",
            str(DQN_TUNED_CONFIG_PATH),
            "--tensorboard-run-name",
            "dqn_dynamic_tuned_custom",
        ]
    )

    assert args.env_config == Path("configs/env/dynamic_grid_2d_medium.yaml")
    assert args.training_config == DQN_TUNED_CONFIG_PATH
    assert args.tensorboard_run_name == "dqn_dynamic_tuned_custom"


def test_td3_train_parse_args_defaults() -> None:
    module = _load_script_module("train_td3_continuous_dynamic.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/continuous_dynamic_2d.yaml")
    assert args.training_config == TD3_BASELINE_CONFIG_PATH
    assert args.tensorboard_run_name is None


def test_td3_train_parse_args_custom_values() -> None:
    module = _load_script_module("train_td3_continuous_dynamic.py")

    args = module.parse_args(
        [
            "--env-config",
            "configs/env/continuous_dynamic_2d_medium.yaml",
            "--training-config",
            str(TD3_TUNED_CONFIG_PATH),
            "--tensorboard-run-name",
            "td3_continuous_dynamic_tuned_custom",
        ]
    )

    assert args.env_config == Path("configs/env/continuous_dynamic_2d_medium.yaml")
    assert args.training_config == TD3_TUNED_CONFIG_PATH
    assert args.tensorboard_run_name == "td3_continuous_dynamic_tuned_custom"


def test_dynamic_dqn_evaluate_parse_args_defaults() -> None:
    module = _load_script_module("evaluate_dynamic_dqn.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/dynamic_grid_2d.yaml")
    assert args.training_config == DQN_BASELINE_CONFIG_PATH
    assert args.plots_dir == Path("results/plots/dynamic")
    assert args.episodes == 100
    assert args.seed == 42


def test_dynamic_dqn_evaluate_parse_args_custom_values() -> None:
    module = _load_script_module("evaluate_dynamic_dqn.py")

    args = module.parse_args(
        [
            "--env-config",
            "configs/env/dynamic_grid_2d_medium.yaml",
            "--training-config",
            str(DQN_TUNED_CONFIG_PATH),
            "--plots-dir",
            "results/plots/dynamic_tuned",
            "--episodes",
            "25",
            "--seed",
            "123",
        ]
    )

    assert args.env_config == Path("configs/env/dynamic_grid_2d_medium.yaml")
    assert args.training_config == DQN_TUNED_CONFIG_PATH
    assert args.plots_dir == Path("results/plots/dynamic_tuned")
    assert args.episodes == 25
    assert args.seed == 123


def test_td3_evaluate_parse_args_defaults() -> None:
    module = _load_script_module("evaluate_td3_continuous_dynamic.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/continuous_dynamic_2d.yaml")
    assert args.training_config == TD3_BASELINE_CONFIG_PATH
    assert args.plots_dir == Path("results/plots/td3_continuous_dynamic")
    assert args.episodes == 100
    assert args.seed == 42


def test_td3_evaluate_parse_args_custom_values() -> None:
    module = _load_script_module("evaluate_td3_continuous_dynamic.py")

    args = module.parse_args(
        [
            "--env-config",
            "configs/env/continuous_dynamic_2d_medium.yaml",
            "--training-config",
            str(TD3_TUNED_CONFIG_PATH),
            "--plots-dir",
            "results/plots/td3_continuous_dynamic_tuned",
            "--episodes",
            "25",
            "--seed",
            "123",
        ]
    )

    assert args.env_config == Path("configs/env/continuous_dynamic_2d_medium.yaml")
    assert args.training_config == TD3_TUNED_CONFIG_PATH
    assert args.plots_dir == Path("results/plots/td3_continuous_dynamic_tuned")
    assert args.episodes == 25
    assert args.seed == 123


def _load_script_module(script_name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = spec_from_file_location(script_name.removesuffix(".py"), script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
