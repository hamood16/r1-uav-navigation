"""Colosseum TD3 baseline training and evaluation helpers."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback

from r1_uav_nav.agents.td3_agent import create_td3_model
from r1_uav_nav.envs import ColosseumUAVEnv, ColosseumUAVEnvConfig
from r1_uav_nav.utils import load_config

DEFAULT_CONFIG_PATH = Path("configs/training/td3_colosseum_baseline.yaml")
DEFAULT_EXPERIMENT_NAME = "colosseum_td3_baseline"
DEFAULT_MODEL_OUTPUT_DIR = Path("results/trained_models/colosseum_td3_baseline")
DEFAULT_TENSORBOARD_LOG_DIR = Path("results/logs/colosseum_td3_baseline")
DEFAULT_REPORTS_DIR = Path("results/reports/m12")
DEFAULT_FINAL_CHECKPOINT_NAME = "final.zip"
DEFAULT_BEST_CHECKPOINT_NAME = "best_training_episode.zip"
DEFAULT_POLICY_KIND = "td3"
EVALUATION_POLICY_KINDS = ("td3", "random", "scripted-forward")


@dataclass(frozen=True)
class ColosseumTD3Config:
    """Typed configuration for the first Colosseum TD3 baseline."""

    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    seed: int = 42
    device: str = "auto"
    policy: str = "MlpPolicy"
    total_timesteps: int = 100
    learning_rate: float = 0.0003
    buffer_size: int = 10_000
    learning_starts: int = 20
    batch_size: int = 16
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_std: tuple[float, ...] = (0.15, 0.15, 0.05)
    verbose: int = 1
    checkpoint_interval: int = 50
    save_best_checkpoint: bool = True
    model_output_dir: Path = DEFAULT_MODEL_OUTPUT_DIR
    tensorboard_log_dir: Path = DEFAULT_TENSORBOARD_LOG_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR
    final_checkpoint_name: str = DEFAULT_FINAL_CHECKPOINT_NAME
    best_checkpoint_name: str = DEFAULT_BEST_CHECKPOINT_NAME
    evaluation_episodes: int = 3
    evaluation_policy: str = DEFAULT_POLICY_KIND
    eval_checkpoint_path: Path | None = None
    env_config: ColosseumUAVEnvConfig = ColosseumUAVEnvConfig(
        max_vertical_velocity=0.2,
    )
    policy_kwargs: dict[str, Any] | None = None

    @property
    def checkpoint_dir(self) -> Path:
        """Return the directory used for periodic checkpoints."""
        return self.model_output_dir / "checkpoints"

    @property
    def final_checkpoint_path(self) -> Path:
        """Return the final checkpoint path."""
        return self.model_output_dir / self.final_checkpoint_name

    @property
    def best_checkpoint_path(self) -> Path:
        """Return the best observed training-episode checkpoint path."""
        return self.model_output_dir / self.best_checkpoint_name

    @property
    def training_metrics_path(self) -> Path:
        """Return the JSON summary path for training metrics."""
        return self.reports_dir / f"{self.experiment_name}_training_summary.json"

    @property
    def training_episode_metrics_path(self) -> Path:
        """Return the CSV path for per-episode training metrics."""
        return self.reports_dir / f"{self.experiment_name}_training_episodes.csv"

    @property
    def evaluation_metrics_path(self) -> Path:
        """Return the JSON summary path for evaluation metrics."""
        return (
            self.reports_dir
            / f"{self.experiment_name}_{self.evaluation_policy}_evaluation_summary.json"
        )


@dataclass(frozen=True)
class TrainingEpisodeMetrics:
    """Metrics for one training episode observed by the callback."""

    episode: int
    global_step: int
    episode_return: float
    episode_length: int
    final_distance_to_goal: float
    min_distance_to_goal: float
    success: bool
    collision: bool
    out_of_bounds: bool
    ground_clearance_violation: bool
    truncated: bool
    termination_reason: str | None
    actor_loss: float | None
    critic_loss: float | None
    replay_buffer_size: int
    exploration_noise_scale: tuple[float, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationEpisodeMetrics:
    """Metrics for one Colosseum policy evaluation episode."""

    episode: int
    episode_return: float
    episode_length: int
    final_distance_to_goal: float
    min_distance_to_goal: float
    success: bool
    collision: bool
    out_of_bounds: bool
    ground_clearance_violation: bool
    truncated: bool
    termination_reason: str | None


@dataclass(frozen=True)
class ColosseumTD3RunResult:
    """Process-style result for train/evaluation scripts."""

    exit_code: int
    metrics_path: Path | None
    final_checkpoint_path: Path | None = None
    error_message: str | None = None
    checkpoint_error_message: str | None = None
    cleanup_error_message: str | None = None
    metrics_error_message: str | None = None
    cleanup_safety_critical_failure: bool = False


class ColosseumTD3MetricsCallback(BaseCallback):
    """Collect training metrics and save periodic/best checkpoints."""

    def __init__(
        self,
        config: ColosseumTD3Config,
        *,
        start_time: float | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.start_time = start_time if start_time is not None else time.monotonic()
        self.episode_metrics: list[TrainingEpisodeMetrics] = []
        self.current_return = 0.0
        self.current_length = 0
        self.current_min_distance = math.inf
        self.best_score: tuple[int, float, float] | None = None

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]

        distance = _safe_float(info.get("distance_to_goal"), default=math.inf)
        self.current_return += reward
        self.current_length += 1
        self.current_min_distance = min(self.current_min_distance, distance)

        if self.num_timesteps % self.config.checkpoint_interval == 0:
            _save_model(
                self.model,
                self.config.checkpoint_dir / f"step_{self.num_timesteps}.zip",
            )

        if done:
            metrics = self._build_episode_metrics(info)
            self.episode_metrics.append(metrics)
            if self.config.save_best_checkpoint and self._is_new_best(metrics):
                _save_model(self.model, self.config.best_checkpoint_path)
            self.current_return = 0.0
            self.current_length = 0
            self.current_min_distance = math.inf

        return True

    def _build_episode_metrics(self, info: dict[str, Any]) -> TrainingEpisodeMetrics:
        termination_reason = info.get("termination_reason")
        truncated = bool(info.get("TimeLimit.truncated", False))
        return TrainingEpisodeMetrics(
            episode=len(self.episode_metrics) + 1,
            global_step=self.num_timesteps,
            episode_return=self.current_return,
            episode_length=self.current_length,
            final_distance_to_goal=_safe_float(
                info.get("distance_to_goal"),
                default=math.inf,
            ),
            min_distance_to_goal=self.current_min_distance,
            success=bool(info.get("success", False)),
            collision=bool(info.get("collision", False)),
            out_of_bounds=bool(info.get("out_of_bounds", False)),
            ground_clearance_violation=termination_reason
            == "ground_clearance_violation",
            truncated=truncated,
            termination_reason=(
                str(termination_reason) if termination_reason is not None else None
            ),
            actor_loss=_logger_value(self.model, "train/actor_loss"),
            critic_loss=_logger_value(self.model, "train/critic_loss"),
            replay_buffer_size=_replay_buffer_size(self.model),
            exploration_noise_scale=tuple(
                float(value) for value in self.config.action_noise_std
            ),
            elapsed_seconds=time.monotonic() - self.start_time,
        )

    def _is_new_best(self, metrics: TrainingEpisodeMetrics) -> bool:
        score = (
            int(metrics.success),
            -metrics.final_distance_to_goal,
            metrics.episode_return,
        )
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            return True
        return False


def load_colosseum_td3_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ColosseumTD3Config:
    """Load the Colosseum TD3 YAML config."""
    raw_config = load_config(path)
    return colosseum_td3_config_from_dict(raw_config)


def colosseum_td3_config_from_dict(raw_config: dict[str, Any]) -> ColosseumTD3Config:
    """Create a typed TD3 config from a dictionary."""
    env_config = _env_config_from_dict(raw_config.get("env", {}))
    config = ColosseumTD3Config(
        experiment_name=str(raw_config.get("experiment_name", DEFAULT_EXPERIMENT_NAME)),
        seed=int(raw_config.get("seed", 42)),
        device=str(raw_config.get("device", "auto")),
        policy=str(raw_config.get("policy", "MlpPolicy")),
        total_timesteps=int(raw_config.get("total_timesteps", 100)),
        learning_rate=float(raw_config.get("learning_rate", 0.0003)),
        buffer_size=int(raw_config.get("buffer_size", 10_000)),
        learning_starts=int(raw_config.get("learning_starts", 20)),
        batch_size=int(raw_config.get("batch_size", 16)),
        gamma=float(raw_config.get("gamma", 0.99)),
        tau=float(raw_config.get("tau", 0.005)),
        train_freq=int(raw_config.get("train_freq", 1)),
        gradient_steps=int(raw_config.get("gradient_steps", 1)),
        policy_delay=int(raw_config.get("policy_delay", 2)),
        target_policy_noise=float(raw_config.get("target_policy_noise", 0.2)),
        target_noise_clip=float(raw_config.get("target_noise_clip", 0.5)),
        action_noise_std=_coerce_noise_std(
            raw_config.get("action_noise_std", (0.15, 0.15, 0.05))
        ),
        verbose=int(raw_config.get("verbose", 1)),
        checkpoint_interval=int(raw_config.get("checkpoint_interval", 50)),
        save_best_checkpoint=bool(raw_config.get("save_best_checkpoint", True)),
        model_output_dir=Path(
            raw_config.get("model_output_dir", DEFAULT_MODEL_OUTPUT_DIR)
        ),
        tensorboard_log_dir=Path(
            raw_config.get("tensorboard_log_dir", DEFAULT_TENSORBOARD_LOG_DIR)
        ),
        reports_dir=Path(raw_config.get("reports_dir", DEFAULT_REPORTS_DIR)),
        final_checkpoint_name=str(
            raw_config.get("final_checkpoint_name", DEFAULT_FINAL_CHECKPOINT_NAME)
        ),
        best_checkpoint_name=str(
            raw_config.get("best_checkpoint_name", DEFAULT_BEST_CHECKPOINT_NAME)
        ),
        evaluation_episodes=int(raw_config.get("evaluation_episodes", 3)),
        evaluation_policy=str(raw_config.get("evaluation_policy", DEFAULT_POLICY_KIND)),
        eval_checkpoint_path=_optional_path(raw_config.get("eval_checkpoint_path")),
        env_config=env_config,
        policy_kwargs=raw_config.get("policy_kwargs"),
    )
    validate_colosseum_td3_config(config)
    return config


def apply_training_overrides(
    config: ColosseumTD3Config,
    *,
    total_timesteps: int | None = None,
    learning_starts: int | None = None,
    batch_size: int | None = None,
    checkpoint_interval: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    model_output_dir: Path | None = None,
    tensorboard_log_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> ColosseumTD3Config:
    """Apply CLI overrides to a typed TD3 config."""
    updated = replace(
        config,
        total_timesteps=(
            total_timesteps if total_timesteps is not None else config.total_timesteps
        ),
        learning_starts=(
            learning_starts if learning_starts is not None else config.learning_starts
        ),
        batch_size=batch_size if batch_size is not None else config.batch_size,
        checkpoint_interval=(
            checkpoint_interval
            if checkpoint_interval is not None
            else config.checkpoint_interval
        ),
        device=device if device is not None else config.device,
        seed=seed if seed is not None else config.seed,
        model_output_dir=(
            model_output_dir
            if model_output_dir is not None
            else config.model_output_dir
        ),
        tensorboard_log_dir=(
            tensorboard_log_dir
            if tensorboard_log_dir is not None
            else config.tensorboard_log_dir
        ),
        reports_dir=reports_dir if reports_dir is not None else config.reports_dir,
    )
    validate_colosseum_td3_config(updated)
    return updated


def apply_evaluation_overrides(
    config: ColosseumTD3Config,
    *,
    checkpoint: Path | None = None,
    episodes: int | None = None,
    policy: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    reports_dir: Path | None = None,
) -> ColosseumTD3Config:
    """Apply CLI overrides for deterministic or baseline evaluation."""
    updated = replace(
        config,
        eval_checkpoint_path=(
            checkpoint if checkpoint is not None else config.eval_checkpoint_path
        ),
        evaluation_episodes=(
            episodes if episodes is not None else config.evaluation_episodes
        ),
        evaluation_policy=policy if policy is not None else config.evaluation_policy,
        device=device if device is not None else config.device,
        seed=seed if seed is not None else config.seed,
        reports_dir=reports_dir if reports_dir is not None else config.reports_dir,
    )
    validate_colosseum_td3_config(updated)
    return updated


def validate_colosseum_td3_config(config: ColosseumTD3Config) -> None:
    """Validate Colosseum TD3 training/evaluation config."""
    if not config.experiment_name:
        raise ValueError("experiment_name must not be empty")
    _require_positive_int("seed", config.seed, allow_zero=True)
    _require_device(config.device)
    for name in (
        "total_timesteps",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "policy_delay",
        "checkpoint_interval",
        "evaluation_episodes",
    ):
        _require_positive_int(
            name,
            getattr(config, name),
            allow_zero=name == "learning_starts",
        )
    if config.learning_starts >= config.total_timesteps:
        raise ValueError("learning_starts must be less than total_timesteps")
    if config.batch_size > config.buffer_size:
        raise ValueError("batch_size must not exceed buffer_size")
    if config.batch_size > config.total_timesteps:
        raise ValueError("batch_size must not exceed total_timesteps")
    for name in (
        "learning_rate",
        "gamma",
        "tau",
        "target_policy_noise",
        "target_noise_clip",
    ):
        _require_positive_float(name, getattr(config, name))
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    _validate_noise_std(config.action_noise_std)
    if config.evaluation_policy not in EVALUATION_POLICY_KINDS:
        raise ValueError(
            "evaluation_policy must be one of " + ", ".join(EVALUATION_POLICY_KINDS)
        )


def resolve_device(device: str) -> str:
    """Resolve auto/cpu/cuda into a concrete PyTorch device string."""
    _require_device(device)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def build_training_config_dict(config: ColosseumTD3Config) -> dict[str, Any]:
    """Build the dictionary expected by create_td3_model."""
    training_config: dict[str, Any] = {
        "algorithm": "TD3",
        "policy": config.policy,
        "learning_rate": config.learning_rate,
        "buffer_size": config.buffer_size,
        "learning_starts": config.learning_starts,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "tau": config.tau,
        "train_freq": config.train_freq,
        "gradient_steps": config.gradient_steps,
        "policy_delay": config.policy_delay,
        "target_policy_noise": config.target_policy_noise,
        "target_noise_clip": config.target_noise_clip,
        "action_noise_std": config.action_noise_std,
        "seed": config.seed,
        "verbose": config.verbose,
        "device": resolve_device(config.device),
        "replay_buffer_kwargs": {"handle_timeout_termination": True},
    }
    if config.policy_kwargs is not None:
        training_config["policy_kwargs"] = config.policy_kwargs
    return training_config


def train_colosseum_td3(
    config: ColosseumTD3Config,
    *,
    env_factory: Callable[[ColosseumUAVEnvConfig], Any] | None = None,
    model_factory: Callable[..., Any] = create_td3_model,
    require_ignored_outputs: bool = True,
) -> ColosseumTD3RunResult:
    """Train TD3 on ColosseumUAVEnv and save checkpoints/metrics."""
    env = None
    model = None
    callback: ColosseumTD3MetricsCallback | None = None
    error_message: str | None = None
    checkpoint_error_message: str | None = None
    cleanup_error_message: str | None = None
    metrics_error_message: str | None = None
    metrics_path: Path | None = None
    cleanup_failure = False
    interrupted = False
    start_time = time.monotonic()
    _prepare_output_dirs(config, require_ignored_outputs=require_ignored_outputs)
    _seed_everything(config.seed)

    try:
        env = _make_env(config, env_factory)
        _seed_space(env.action_space, config.seed)
        model = model_factory(
            env=env,
            training_config=build_training_config_dict(config),
            tensorboard_log=str(config.tensorboard_log_dir),
        )
        callback = ColosseumTD3MetricsCallback(config, start_time=start_time)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            tb_log_name=config.experiment_name,
        )
    except KeyboardInterrupt:
        interrupted = True
        error_message = "Training interrupted by user."
    except Exception as exc:
        error_message = str(exc)
    finally:
        try:
            if model is not None:
                _save_model(model, config.final_checkpoint_path)
        except Exception as exc:
            checkpoint_error_message = f"Final checkpoint save failed: {exc}"
        finally:
            cleanup_failure, cleanup_error_message = (
                _close_env_and_detect_cleanup_failure(env)
            )
        try:
            metrics_path = _save_training_outputs(
                config=config,
                callback=callback,
                model=model,
                cleanup_failure=cleanup_failure,
                interrupted=interrupted,
                error_message=error_message,
                checkpoint_error_message=checkpoint_error_message,
                cleanup_error_message=cleanup_error_message,
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as exc:
            metrics_error_message = f"Metrics save failed: {exc}"

    exit_code = int(
        error_message is not None
        or checkpoint_error_message is not None
        or cleanup_error_message is not None
        or metrics_error_message is not None
        or cleanup_failure
    )
    return ColosseumTD3RunResult(
        exit_code=exit_code,
        metrics_path=metrics_path,
        final_checkpoint_path=(
            config.final_checkpoint_path if model is not None else None
        ),
        error_message=error_message,
        checkpoint_error_message=checkpoint_error_message,
        cleanup_error_message=cleanup_error_message,
        metrics_error_message=metrics_error_message,
        cleanup_safety_critical_failure=cleanup_failure,
    )


def evaluate_colosseum_td3(
    config: ColosseumTD3Config,
    *,
    env_factory: Callable[[ColosseumUAVEnvConfig], Any] | None = None,
    model_loader: Callable[..., Any] = TD3.load,
    require_ignored_outputs: bool = True,
) -> ColosseumTD3RunResult:
    """Evaluate a TD3 checkpoint or baseline policy in ColosseumUAVEnv."""
    env = None
    model = None
    error_message: str | None = None
    cleanup_error_message: str | None = None
    metrics_error_message: str | None = None
    metrics_path: Path | None = None
    cleanup_failure = False
    episode_metrics: list[EvaluationEpisodeMetrics] = []
    start_time = time.monotonic()
    _prepare_output_dirs(config, require_ignored_outputs=require_ignored_outputs)
    _seed_everything(config.seed)

    try:
        env = _make_env(config, env_factory)
        _seed_space(env.action_space, config.seed)
        policy_kind = config.evaluation_policy
        if policy_kind == "td3":
            checkpoint = config.eval_checkpoint_path or config.final_checkpoint_path
            if not checkpoint.exists():
                raise FileNotFoundError(f"TD3 checkpoint not found: {checkpoint}")
            model = model_loader(
                checkpoint, env=env, device=resolve_device(config.device)
            )

        rng = np.random.default_rng(config.seed)
        for episode_index in range(config.evaluation_episodes):
            observation, info = env.reset(seed=config.seed + episode_index)
            episode_metrics.append(
                _run_evaluation_episode(
                    env=env,
                    model=model,
                    policy_kind=policy_kind,
                    observation=observation,
                    reset_info=info,
                    episode_index=episode_index + 1,
                    rng=rng,
                )
            )
    except KeyboardInterrupt:
        error_message = "Evaluation interrupted by user."
    except Exception as exc:
        error_message = str(exc)
    finally:
        cleanup_failure, cleanup_error_message = _close_env_and_detect_cleanup_failure(
            env
        )
        try:
            metrics_path = _save_evaluation_outputs(
                config=config,
                episode_metrics=episode_metrics,
                cleanup_failure=cleanup_failure,
                error_message=error_message,
                cleanup_error_message=cleanup_error_message,
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as exc:
            metrics_error_message = f"Metrics save failed: {exc}"

    exit_code = int(
        error_message is not None
        or cleanup_error_message is not None
        or metrics_error_message is not None
        or cleanup_failure
    )
    return ColosseumTD3RunResult(
        exit_code=exit_code,
        metrics_path=metrics_path,
        final_checkpoint_path=None,
        error_message=error_message,
        cleanup_error_message=cleanup_error_message,
        metrics_error_message=metrics_error_message,
        cleanup_safety_critical_failure=cleanup_failure,
    )


def verify_output_paths_ignored(paths: Sequence[Path]) -> None:
    """Verify generated-output paths are ignored by Git."""
    for path in paths:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(f"Generated output path is not ignored by Git: {path}")


def _run_evaluation_episode(
    *,
    env: Any,
    model: Any,
    policy_kind: str,
    observation: np.ndarray,
    reset_info: dict[str, Any],
    episode_index: int,
    rng: np.random.Generator,
) -> EvaluationEpisodeMetrics:
    total_reward = 0.0
    episode_length = 0
    min_distance = _safe_float(reset_info.get("distance_to_goal"), default=math.inf)
    terminated = False
    truncated = False
    final_info = reset_info

    while not terminated and not truncated:
        action = _select_evaluation_action(policy_kind, model, observation, env, rng)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        episode_length += 1
        min_distance = min(
            min_distance,
            _safe_float(final_info.get("distance_to_goal"), default=math.inf),
        )

    termination_reason = final_info.get("termination_reason")
    return EvaluationEpisodeMetrics(
        episode=episode_index,
        episode_return=total_reward,
        episode_length=episode_length,
        final_distance_to_goal=_safe_float(
            final_info.get("distance_to_goal"),
            default=math.inf,
        ),
        min_distance_to_goal=min_distance,
        success=bool(final_info.get("success", False)),
        collision=bool(final_info.get("collision", False)),
        out_of_bounds=bool(final_info.get("out_of_bounds", False)),
        ground_clearance_violation=termination_reason == "ground_clearance_violation",
        truncated=bool(truncated),
        termination_reason=(
            str(termination_reason) if termination_reason is not None else None
        ),
    )


def _select_evaluation_action(
    policy_kind: str,
    model: Any,
    observation: np.ndarray,
    env: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if policy_kind == "td3":
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    if policy_kind == "random":
        return np.asarray(
            rng.uniform(low=env.action_space.low, high=env.action_space.high),
            dtype=np.float32,
        )
    if policy_kind == "scripted-forward":
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    raise ValueError(f"Unknown evaluation policy: {policy_kind}")


def _env_config_from_dict(raw_env_config: dict[str, Any]) -> ColosseumUAVEnvConfig:
    config_kwargs = dict(raw_env_config)
    if "default_goal_offset" in config_kwargs:
        config_kwargs["default_goal_offset"] = tuple(
            config_kwargs["default_goal_offset"]
        )
    return ColosseumUAVEnvConfig(**config_kwargs)


def _make_env(
    config: ColosseumTD3Config,
    env_factory: Callable[[ColosseumUAVEnvConfig], Any] | None,
) -> Any:
    if env_factory is not None:
        return env_factory(config.env_config)
    return ColosseumUAVEnv(config.env_config)


def _prepare_output_dirs(
    config: ColosseumTD3Config,
    *,
    require_ignored_outputs: bool,
) -> None:
    if require_ignored_outputs:
        verify_output_paths_ignored(
            [
                config.model_output_dir,
                config.tensorboard_log_dir,
                config.reports_dir,
            ]
        )
    for path in (
        config.model_output_dir,
        config.checkpoint_dir,
        config.tensorboard_log_dir,
        config.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _save_training_outputs(
    *,
    config: ColosseumTD3Config,
    callback: ColosseumTD3MetricsCallback | None,
    model: Any,
    cleanup_failure: bool,
    interrupted: bool,
    error_message: str | None,
    checkpoint_error_message: str | None,
    cleanup_error_message: str | None,
    elapsed_seconds: float,
) -> Path:
    episode_metrics = callback.episode_metrics if callback is not None else []
    _write_training_episode_csv(config.training_episode_metrics_path, episode_metrics)
    summary = {
        "experiment_name": config.experiment_name,
        "total_timesteps": config.total_timesteps,
        "learning_starts": config.learning_starts,
        "batch_size": config.batch_size,
        "num_episodes": len(episode_metrics),
        "replay_buffer_size": _replay_buffer_size(model),
        "num_updates": int(getattr(model, "_n_updates", 0)) if model is not None else 0,
        "final_checkpoint_path": str(config.final_checkpoint_path),
        "best_checkpoint_path": str(config.best_checkpoint_path),
        "cleanup_safety_critical_failure": cleanup_failure,
        "interrupted": interrupted,
        "error_message": error_message,
        "checkpoint_error_message": checkpoint_error_message,
        "cleanup_error_message": cleanup_error_message,
        "elapsed_seconds": elapsed_seconds,
        "config": _config_to_jsonable(config),
    }
    _write_json(config.training_metrics_path, summary)
    return config.training_metrics_path


def _save_evaluation_outputs(
    *,
    config: ColosseumTD3Config,
    episode_metrics: Sequence[EvaluationEpisodeMetrics],
    cleanup_failure: bool,
    error_message: str | None,
    cleanup_error_message: str | None,
    elapsed_seconds: float,
) -> Path:
    summary = {
        "experiment_name": config.experiment_name,
        "policy": config.evaluation_policy,
        "episodes": [asdict(metric) for metric in episode_metrics],
        "num_episodes": len(episode_metrics),
        "success_rate": _mean_bool(metric.success for metric in episode_metrics),
        "average_return": _mean_float(
            metric.episode_return for metric in episode_metrics
        ),
        "average_final_distance": _mean_float(
            metric.final_distance_to_goal for metric in episode_metrics
        ),
        "cleanup_safety_critical_failure": cleanup_failure,
        "error_message": error_message,
        "cleanup_error_message": cleanup_error_message,
        "elapsed_seconds": elapsed_seconds,
        "config": _config_to_jsonable(config),
    }
    _write_json(config.evaluation_metrics_path, summary)
    return config.evaluation_metrics_path


def _write_training_episode_csv(
    output_path: Path,
    metrics: Sequence[TrainingEpisodeMetrics],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TrainingEpisodeMetrics.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))


def _write_json(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _close_env_and_detect_cleanup_failure(env: Any | None) -> tuple[bool, str | None]:
    if env is None:
        return False, None
    cleanup_error_message = None
    close_with_result = getattr(env, "close_with_result", None)
    try:
        if close_with_result is not None:
            cleanup_result = close_with_result()
        else:
            env.close()
            cleanup_result = getattr(env, "last_cleanup_result", None)
    except Exception as exc:
        cleanup_result = getattr(env, "last_cleanup_result", None)
        cleanup_error_message = f"Environment cleanup failed: {exc}"
    cleanup_failure = bool(
        getattr(cleanup_result, "safety_critical_failure", False)
        or getattr(env, "cleanup_safety_critical_failure_seen", False)
    )
    if cleanup_error_message is not None:
        cleanup_failure = True
    return cleanup_failure, cleanup_error_message


def _save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_space(space: Any, seed: int) -> None:
    seed_method = getattr(space, "seed", None)
    if seed_method is not None:
        seed_method(seed)


def _config_to_jsonable(config: ColosseumTD3Config) -> dict[str, Any]:
    data = asdict(config)
    for key in (
        "model_output_dir",
        "tensorboard_log_dir",
        "reports_dir",
        "eval_checkpoint_path",
    ):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def _coerce_noise_std(value: Any) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(float(item) for item in value)
    raise ValueError("action_noise_std must be a number or sequence of numbers")


def _validate_noise_std(noise_std: tuple[float, ...]) -> None:
    if len(noise_std) not in (1, 3):
        raise ValueError("action_noise_std must be scalar or length 3")
    for value in noise_std:
        _require_positive_float("action_noise_std", value)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _require_device(device: str) -> None:
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of auto, cpu, cuda")


def _require_positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    elif value < 1:
        raise ValueError(f"{name} must be positive")


def _require_positive_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _logger_value(model: Any, key: str) -> float | None:
    logger = getattr(model, "logger", None)
    values = getattr(logger, "name_to_value", {})
    value = values.get(key)
    return float(value) if value is not None else None


def _replay_buffer_size(model: Any) -> int:
    replay_buffer = getattr(model, "replay_buffer", None)
    size_method = getattr(replay_buffer, "size", None)
    if size_method is None:
        return 0
    return int(size_method())


def _safe_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean_bool(values: Sequence[bool] | Any) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def _mean_float(values: Sequence[float] | Any) -> float | None:
    values = [float(value) for value in values]
    if not values:
        return None
    return sum(values) / len(values)
