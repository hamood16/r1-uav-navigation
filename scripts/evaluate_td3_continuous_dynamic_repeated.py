"""Repeated evaluation for the trained TD3 continuous dynamic baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from stable_baselines3 import TD3

from r1_uav_nav.evaluation import (
    EpisodeResult,
    calculate_path_length,
    summarise_episode_results,
)
from r1_uav_nav.evaluation.repeated_evaluation import (
    RepeatedEvaluationRun,
    save_repeated_evaluation_summary,
    summarise_repeated_evaluations,
)
from r1_uav_nav.utils import create_continuous_dynamic_uav_env_from_config, load_config

DEFAULT_ENV_CONFIG_PATH = Path("configs/env/continuous_dynamic_2d.yaml")
DEFAULT_TRAINING_CONFIG_PATH = Path("configs/training/td3_continuous_dynamic_full.yaml")
DEFAULT_OUTPUT_PATH = Path(
    "results/reports/m10/td3_continuous_dynamic_repeated_eval.json"
)
DEFAULT_REPEATS = 5
DEFAULT_EPISODES_PER_REPEAT = 100
DEFAULT_BASE_SEED = 42
TRAINING_COMMAND = "python scripts/train_td3_continuous_dynamic.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse repeated TD3 continuous dynamic evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Repeated evaluation for the trained TD3 continuous baseline.",
    )
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--episodes-per-repeat",
        type=int,
        default=DEFAULT_EPISODES_PER_REPEAT,
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args(argv)


def main() -> None:
    """Run repeated deterministic evaluation for TD3 continuous dynamic."""
    args = parse_args()
    _validate_args(args)

    env = create_continuous_dynamic_uav_env_from_config(args.env_config)
    training_config = load_config(args.training_config)
    model_output_path = Path(training_config["model_output_path"])
    if not model_output_path.exists():
        raise FileNotFoundError(
            f"TD3 model not found at {model_output_path}. "
            f"Run `{TRAINING_COMMAND}` first."
        )

    model = TD3.load(model_output_path, env=env)
    runs = [
        _evaluate_repeat(
            env=env,
            model=model,
            repeat_index=repeat_index,
            repeat_seed=args.base_seed + repeat_index * 1000,
            episodes_per_repeat=args.episodes_per_repeat,
        )
        for repeat_index in range(args.repeats)
    ]
    summary = summarise_repeated_evaluations(
        runs,
        metadata={
            "algorithm": "TD3",
            "environment": "ContinuousDynamicUAVEnv",
            "env_config_path": str(args.env_config),
            "training_config_path": str(args.training_config),
            "model_path": str(model_output_path),
            "base_seed": args.base_seed,
        },
    )
    saved_path = save_repeated_evaluation_summary(summary, args.output_path)

    print("Repeated TD3 continuous dynamic evaluation complete.")
    print(f"Repeats: {summary['num_repeats']}")
    print(f"Episodes per repeat: {summary['episodes_per_repeat']}")
    _print_metric_summary(summary)
    print(f"Summary path: {saved_path}")


def _evaluate_repeat(
    env: Any,
    model: TD3,
    repeat_index: int,
    repeat_seed: int,
    episodes_per_repeat: int,
) -> RepeatedEvaluationRun:
    episode_results: list[EpisodeResult] = []
    timeout_count = 0

    for episode_index in range(episodes_per_repeat):
        episode_seed = repeat_seed + episode_index
        observation, _ = env.reset(seed=episode_seed)
        uav_positions = [env.uav_position]
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {
            "is_success": False,
            "is_collision": False,
            "collision_type": None,
        }

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            steps += 1
            uav_positions.append(env.uav_position)

        timeout_count += int(truncated)
        episode_results.append(
            EpisodeResult(
                total_reward=total_reward,
                steps=steps,
                success=bool(final_info.get("is_success", False)),
                collision=bool(final_info.get("is_collision", False)),
                path_length=calculate_path_length(uav_positions),
            )
        )

    summary = summarise_episode_results(episode_results)
    return RepeatedEvaluationRun(
        repeat_index=repeat_index,
        seed=repeat_seed,
        num_episodes=summary.num_episodes,
        success_rate=summary.success_rate,
        collision_rate=summary.collision_rate,
        timeout_rate=timeout_count / summary.num_episodes,
        average_reward=summary.average_reward,
        average_steps=summary.average_steps,
        average_path_length=summary.average_path_length,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if args.episodes_per_repeat < 1:
        raise ValueError("episodes-per-repeat must be at least 1")


def _print_metric_summary(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    for metric_name in (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "average_reward",
        "average_steps",
        "average_path_length",
    ):
        metric = metrics[metric_name]
        print(f"{metric_name}: mean={metric['mean']:.4f}, std={metric['std']:.4f}")


if __name__ == "__main__":
    main()
