from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import stable_baselines3 as sb3
from gymnasium import Env, spaces
from stable_baselines3 import TD3

import r1_uav_nav.training.colosseum_td3 as colosseum_td3_module
from r1_uav_nav.agents.td3_agent import create_td3_model
from r1_uav_nav.envs import ColosseumUAVEnvConfig
from r1_uav_nav.training.colosseum_td3 import (
    ColosseumTD3Config,
    apply_evaluation_overrides,
    apply_training_overrides,
    build_training_config_dict,
    colosseum_td3_config_from_dict,
    evaluate_colosseum_td3,
    load_colosseum_td3_config,
    resolve_device,
    train_colosseum_td3,
    verify_output_paths_ignored,
)

DOC_PATH = Path("docs/m12_colosseum_td3_baseline.md")
CONFIG_PATH = Path("configs/training/td3_colosseum_baseline.yaml")


class FakeCleanupResult:
    def __init__(self, safety_critical_failure: bool) -> None:
        self.safety_critical_failure = safety_critical_failure


class FakeReplayBuffer:
    def __init__(self, size: int = 0) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class FakeModel:
    def __init__(
        self,
        *,
        env: object,
        training_config: dict[str, object],
        fail_save: bool = False,
    ) -> None:
        self.env = env
        self.training_config = training_config
        self.fail_save = fail_save
        self.replay_buffer = FakeReplayBuffer()
        self.logger = SimpleNamespace(name_to_value={})
        self._n_updates = 0
        self.num_timesteps = 0
        self.saved_paths: list[Path] = []

    def learn(self, total_timesteps: int, callback: object, tb_log_name: str) -> None:
        callback.init_callback(self)
        for step_index in range(1, total_timesteps + 1):
            self.num_timesteps = step_index
            self.replay_buffer._size = step_index
            if step_index > self.training_config["learning_starts"]:
                self._n_updates += 1
                self.logger.name_to_value = {
                    "train/actor_loss": -0.1,
                    "train/critic_loss": 0.2,
                }
            callback.locals = {
                "rewards": np.asarray([1.0], dtype=np.float32),
                "dones": np.asarray([step_index % 25 == 0]),
                "infos": [
                    {
                        "distance_to_goal": max(0.0, 3.0 - step_index * 0.01),
                        "success": step_index % 50 == 0,
                        "collision": False,
                        "out_of_bounds": False,
                        "termination_reason": (
                            "goal_reached" if step_index % 50 == 0 else "max_steps"
                        ),
                        "TimeLimit.truncated": step_index % 50 != 0,
                    }
                ],
            }
            callback.on_step()

    def save(self, output_path: Path) -> None:
        if self.fail_save:
            raise RuntimeError("checkpoint save failed")
        self.saved_paths.append(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("fake model", encoding="utf-8")


class FakeEvalModel:
    def predict(
        self, observation: np.ndarray, deterministic: bool
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), None


class FakeColosseumTrainingEnv:
    def __init__(
        self,
        *,
        cleanup_failure: bool = False,
        fail_reset: bool = False,
        intermediate_cleanup_failure: bool = False,
    ) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.cleanup_failure = cleanup_failure
        self.fail_reset = fail_reset
        self.intermediate_cleanup_failure = intermediate_cleanup_failure
        self.cleanup_safety_critical_failure_seen = False
        self.close_count = 0
        self.reset_count = 0
        self.step_count = 0
        self.actions: list[np.ndarray] = []
        self.last_cleanup_result: FakeCleanupResult | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        self.reset_count += 1
        self.step_count = 0
        if self.fail_reset:
            self.last_cleanup_result = FakeCleanupResult(True)
            self.cleanup_safety_critical_failure_seen = True
            raise RuntimeError("reset failed")
        if self.intermediate_cleanup_failure and self.reset_count == 2:
            self.cleanup_safety_critical_failure_seen = True
            self.last_cleanup_result = FakeCleanupResult(True)
        return np.zeros(10, dtype=np.float32), {
            "distance_to_goal": 3.0,
            "termination_reason": None,
        }

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.step_count += 1
        terminated = self.step_count >= 2
        info = {
            "distance_to_goal": max(0.0, 3.0 - self.step_count),
            "success": terminated,
            "collision": False,
            "out_of_bounds": False,
            "termination_reason": "goal_reached" if terminated else None,
        }
        return np.zeros(10, dtype=np.float32), 1.0, terminated, False, info

    def close(self) -> None:
        self.close_count += 1

    def close_with_result(self) -> FakeCleanupResult:
        self.close_count += 1
        self.last_cleanup_result = FakeCleanupResult(self.cleanup_failure)
        return self.last_cleanup_result


class FakeResettingModel(FakeModel):
    def learn(self, total_timesteps: int, callback: object, tb_log_name: str) -> None:
        callback.init_callback(self)
        observation, _ = self.env.reset()
        for step_index in range(1, total_timesteps + 1):
            self.num_timesteps = step_index
            self.replay_buffer._size = step_index
            if step_index > self.training_config["learning_starts"]:
                self._n_updates += 1
            observation, reward, terminated, truncated, info = self.env.step(
                np.zeros(3, dtype=np.float32)
            )
            callback.locals = {
                "rewards": np.asarray([reward], dtype=np.float32),
                "dones": np.asarray([terminated or truncated]),
                "infos": [info],
            }
            callback.on_step()
            if terminated or truncated:
                observation, _ = self.env.reset()


class TinyContinuousEnv(Env):
    def __init__(self, *, terminal_mode: str = "success") -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.terminal_mode = terminal_mode
        self.step_count = 0
        self.episode_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(10, dtype=np.float32), {"distance_to_goal": 1.0}

    def step(self, action: np.ndarray):
        self.step_count += 1
        observation = np.full(10, 0.1, dtype=np.float32)
        reward = float(1.0 - np.linalg.norm(action))
        terminated = False
        truncated = False
        reason = None
        if self.step_count >= 1:
            if self.terminal_mode == "alternating":
                terminated = self.episode_count % 2 == 0
                truncated = not terminated
                reason = "goal_reached" if terminated else "max_steps"
                self.episode_count += 1
            elif self.terminal_mode == "truncated":
                truncated = True
                reason = "max_steps"
            else:
                terminated = True
                reason = "goal_reached"
        info = {
            "distance_to_goal": 0.5,
            "success": terminated,
            "collision": False,
            "out_of_bounds": False,
            "termination_reason": reason,
        }
        return observation, reward, terminated, truncated, info


def test_docs_and_config_exist_and_capture_m12_5_intent() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    config = load_colosseum_td3_config(CONFIG_PATH)

    assert "TD3" in doc_text
    assert "fixed-goal" in doc_text
    assert "max_vertical_velocity = 0.2" in doc_text
    assert "Stable-Baselines3" in doc_text
    assert config.total_timesteps == 100
    assert config.learning_starts == 20
    assert config.batch_size == 16
    assert config.checkpoint_interval == 50
    assert config.env_config.default_goal_offset == pytest.approx((3.0, 0.0, 0.0))
    assert config.env_config.max_vertical_velocity == pytest.approx(0.2)
    assert config.action_noise_std == pytest.approx((0.15, 0.15, 0.05))


def test_output_paths_are_ignored_by_git() -> None:
    verify_output_paths_ignored(
        [
            Path("results/trained_models/colosseum_td3_baseline/"),
            Path("results/logs/colosseum_td3_baseline/"),
            Path("results/reports/m12/"),
        ]
    )


@pytest.mark.parametrize(
    "raw_config",
    [
        {"total_timesteps": 20, "learning_starts": 20},
        {"total_timesteps": 100, "batch_size": 101},
        {"action_noise_std": [0.1, 0.2]},
        {"learning_rate": float("nan")},
        {"device": "quantum"},
    ],
)
def test_invalid_colosseum_td3_config_values_raise(
    raw_config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        colosseum_td3_config_from_dict(raw_config)


def test_training_overrides_and_device_resolution(tmp_path: Path) -> None:
    config = apply_training_overrides(
        ColosseumTD3Config(),
        total_timesteps=120,
        learning_starts=30,
        batch_size=16,
        checkpoint_interval=60,
        device="cpu",
        seed=7,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
    )

    assert config.total_timesteps == 120
    assert config.learning_starts == 30
    assert config.checkpoint_interval == 60
    assert config.device == "cpu"
    assert config.seed == 7
    assert resolve_device("cpu") == "cpu"


def test_create_td3_model_accepts_per_axis_noise_cpu() -> None:
    env = TinyContinuousEnv()
    training_config = build_training_config_dict(
        ColosseumTD3Config(
            device="cpu",
            verbose=0,
            buffer_size=64,
            batch_size=16,
            policy_kwargs={"net_arch": [8, 8]},
        )
    )

    model = create_td3_model(env=env, training_config=training_config)

    assert isinstance(model, TD3)
    assert model.action_noise is not None
    assert model.action_noise._sigma == pytest.approx(np.asarray([0.15, 0.15, 0.05]))


def test_real_sb3_cpu_integration_updates_saves_loads_and_predicts(
    tmp_path: Path,
) -> None:
    env = TinyContinuousEnv()
    training_config = build_training_config_dict(
        ColosseumTD3Config(
            device="cpu",
            verbose=0,
            total_timesteps=8,
            learning_starts=2,
            buffer_size=64,
            batch_size=2,
            policy_delay=2,
            policy_kwargs={"net_arch": [8, 8]},
        )
    )
    model = create_td3_model(env=env, training_config=training_config)

    model.learn(total_timesteps=8)
    checkpoint = tmp_path / "td3_tiny.zip"
    model.save(checkpoint)
    loaded_model = TD3.load(checkpoint, env=env, device="cpu")
    action, _ = loaded_model.predict(np.zeros(10, dtype=np.float32), deterministic=True)

    assert model.observation_space.shape == (10,)
    assert model.action_space.shape == (3,)
    assert model.replay_buffer.size() > 0
    assert model._n_updates > 0
    assert checkpoint.exists()
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


def test_installed_sb3_timeout_truncation_bootstrapping_behavior() -> None:
    assert sb3.__version__ == "2.9.0"
    env = TinyContinuousEnv(terminal_mode="alternating")
    training_config = build_training_config_dict(
        ColosseumTD3Config(
            device="cpu",
            verbose=0,
            total_timesteps=8,
            learning_starts=2,
            buffer_size=64,
            batch_size=2,
            policy_delay=2,
            policy_kwargs={"net_arch": [8, 8]},
        )
    )
    model = create_td3_model(env=env, training_config=training_config)

    model.learn(total_timesteps=8)

    replay_buffer = model.replay_buffer
    assert replay_buffer.handle_timeout_termination
    stored_size = replay_buffer.size()
    timeout_indices = np.where(replay_buffer.timeouts[:stored_size, 0] == 1.0)[0]
    terminal_indices = np.where(
        (replay_buffer.dones[:stored_size, 0] == 1.0)
        & (replay_buffer.timeouts[:stored_size, 0] == 0.0)
    )[0]
    timeout_sample = replay_buffer._get_samples(np.asarray([timeout_indices[0]]))
    terminal_sample = replay_buffer._get_samples(np.asarray([terminal_indices[0]]))

    assert len(timeout_indices) > 0
    assert len(terminal_indices) > 0
    assert float(timeout_sample.dones.cpu().numpy()[0, 0]) == pytest.approx(0.0)
    assert float(terminal_sample.dones.cpu().numpy()[0, 0]) == pytest.approx(1.0)


def test_train_colosseum_td3_saves_checkpoints_and_metrics(tmp_path: Path) -> None:
    env = FakeColosseumTrainingEnv()
    config = ColosseumTD3Config(
        total_timesteps=100,
        learning_starts=20,
        batch_size=16,
        checkpoint_interval=50,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        verbose=0,
    )

    result = train_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_factory=lambda env, training_config, tensorboard_log: FakeModel(
            env=env,
            training_config=training_config,
        ),
        require_ignored_outputs=False,
    )

    assert result.exit_code == 0
    assert result.metrics_path is not None
    assert result.metrics_path.exists()
    assert (config.checkpoint_dir / "step_50.zip").exists()
    assert (config.checkpoint_dir / "step_100.zip").exists()
    assert config.final_checkpoint_path.exists()
    assert config.best_checkpoint_path.exists()
    assert config.training_episode_metrics_path.exists()


def test_training_cleanup_failure_is_reported_and_preserved(tmp_path: Path) -> None:
    env = FakeColosseumTrainingEnv(cleanup_failure=True)
    config = ColosseumTD3Config(
        total_timesteps=100,
        learning_starts=20,
        batch_size=16,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        verbose=0,
    )

    result = train_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_factory=lambda env, training_config, tensorboard_log: FakeModel(
            env=env,
            training_config=training_config,
        ),
        require_ignored_outputs=False,
    )

    assert result.exit_code == 1
    assert result.cleanup_safety_critical_failure
    assert env.close_count == 1


def test_training_closes_env_when_checkpoint_save_fails(tmp_path: Path) -> None:
    env = FakeColosseumTrainingEnv(cleanup_failure=True)
    config = ColosseumTD3Config(
        total_timesteps=100,
        learning_starts=20,
        batch_size=16,
        checkpoint_interval=50,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        verbose=0,
    )

    result = train_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_factory=lambda env, training_config, tensorboard_log: FakeModel(
            env=env,
            training_config=training_config,
            fail_save=True,
        ),
        require_ignored_outputs=False,
    )

    assert env.close_count == 1
    assert result.exit_code == 1
    assert result.checkpoint_error_message is not None
    assert "checkpoint save failed" in result.checkpoint_error_message
    assert result.cleanup_safety_critical_failure
    assert result.metrics_path is not None
    assert result.metrics_path.exists()


def test_intermediate_cleanup_failure_survives_successful_final_close(
    tmp_path: Path,
) -> None:
    env = FakeColosseumTrainingEnv(intermediate_cleanup_failure=True)
    config = ColosseumTD3Config(
        total_timesteps=30,
        learning_starts=5,
        batch_size=8,
        checkpoint_interval=15,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        verbose=0,
    )

    result = train_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_factory=lambda env, training_config, tensorboard_log: FakeResettingModel(
            env=env,
            training_config=training_config,
        ),
        require_ignored_outputs=False,
    )

    assert env.reset_count > 1
    assert env.close_count == 1
    assert result.exit_code == 1
    assert result.cleanup_safety_critical_failure


def test_training_metrics_save_failure_returns_nonzero_and_closes_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FakeColosseumTrainingEnv()
    config = ColosseumTD3Config(
        total_timesteps=30,
        learning_starts=5,
        batch_size=8,
        checkpoint_interval=15,
        model_output_dir=tmp_path / "models",
        tensorboard_log_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        verbose=0,
    )

    def fail_json_write(output_path: Path, data: dict[str, object]) -> None:
        raise RuntimeError("metrics write failed")

    monkeypatch.setattr(colosseum_td3_module, "_write_json", fail_json_write)

    result = colosseum_td3_module.train_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_factory=lambda env, training_config, tensorboard_log: FakeModel(
            env=env,
            training_config=training_config,
        ),
        require_ignored_outputs=False,
    )

    assert env.close_count == 1
    assert result.exit_code == 1
    assert result.metrics_error_message is not None
    assert "metrics write failed" in result.metrics_error_message


def test_evaluate_colosseum_td3_random_and_scripted_do_not_load_model(
    tmp_path: Path,
) -> None:
    for policy in ("random", "scripted-forward"):
        env = FakeColosseumTrainingEnv()
        config = apply_evaluation_overrides(
            ColosseumTD3Config(
                reports_dir=tmp_path / policy,
                model_output_dir=tmp_path / "models",
                tensorboard_log_dir=tmp_path / "logs",
                verbose=0,
            ),
            policy=policy,
            episodes=2,
        )

        result = evaluate_colosseum_td3(
            config,
            env_factory=lambda _env_config, env=env: env,
            model_loader=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("model_loader should not be called")
            ),
            require_ignored_outputs=False,
        )

        assert result.exit_code == 0
        assert result.metrics_path is not None
        assert result.metrics_path.exists()
        assert result.metrics_path.name == (
            f"colosseum_td3_baseline_{policy}_evaluation_summary.json"
        )
        assert len(env.actions) == 4


def test_evaluate_colosseum_td3_loads_deterministic_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_text("fake model", encoding="utf-8")
    env = FakeColosseumTrainingEnv()
    config = apply_evaluation_overrides(
        ColosseumTD3Config(
            reports_dir=tmp_path / "reports",
            model_output_dir=tmp_path / "models",
            tensorboard_log_dir=tmp_path / "logs",
            verbose=0,
        ),
        checkpoint=checkpoint,
        policy="td3",
        episodes=1,
    )

    result = evaluate_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        model_loader=lambda *args, **kwargs: FakeEvalModel(),
        require_ignored_outputs=False,
    )

    assert result.exit_code == 0
    assert env.actions[0] == pytest.approx(np.asarray([1.0, 0.0, 0.0]))
    assert result.metrics_path is not None
    assert (
        result.metrics_path.name == "colosseum_td3_baseline_td3_evaluation_summary.json"
    )


def test_evaluation_cleanup_failure_returns_nonzero(tmp_path: Path) -> None:
    env = FakeColosseumTrainingEnv(cleanup_failure=True)
    config = apply_evaluation_overrides(
        ColosseumTD3Config(
            reports_dir=tmp_path / "reports",
            model_output_dir=tmp_path / "models",
            tensorboard_log_dir=tmp_path / "logs",
            verbose=0,
        ),
        policy="random",
        episodes=1,
    )

    result = evaluate_colosseum_td3(
        config,
        env_factory=lambda _env_config: env,
        require_ignored_outputs=False,
    )

    assert result.exit_code == 1
    assert result.cleanup_safety_critical_failure


def test_train_and_evaluate_scripts_parse_defaults_and_overrides() -> None:
    train_script = _load_script("train_colosseum_td3")
    eval_script = _load_script("evaluate_colosseum_td3")

    train_args = train_script.parse_args(
        [
            "--total-timesteps",
            "100",
            "--learning-starts",
            "20",
            "--batch-size",
            "16",
            "--checkpoint-interval",
            "50",
            "--device",
            "cpu",
        ]
    )
    eval_args = eval_script.parse_args(
        [
            "--policy",
            "scripted-forward",
            "--episodes",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert train_args.total_timesteps == 100
    assert train_args.learning_starts == 20
    assert train_args.checkpoint_interval == 50
    assert train_args.device == "cpu"
    assert eval_args.policy == "scripted-forward"
    assert eval_args.episodes == 2
    assert eval_args.device == "cpu"


def test_config_builds_expected_colosseum_env_config() -> None:
    config = colosseum_td3_config_from_dict(
        {
            "env": {
                "default_goal_offset": [3.0, 0.0, 0.0],
                "max_episode_steps": 40,
                "max_vertical_velocity": 0.2,
            }
        }
    )

    assert isinstance(config.env_config, ColosseumUAVEnvConfig)
    assert config.env_config.default_goal_offset == pytest.approx((3.0, 0.0, 0.0))
    assert config.env_config.max_episode_steps == 40
    assert config.env_config.max_vertical_velocity == pytest.approx(0.2)


def _load_script(module_name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{module_name}.py"
    spec = spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
