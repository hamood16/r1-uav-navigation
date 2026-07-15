"""Evaluate the trained TD3 baseline on ContinuousDynamicUAVEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import TD3

from r1_uav_nav.evaluation import (
    EpisodeResult,
    calculate_path_length,
    plot_collision_rate_bar,
    plot_continuous_dynamic_trajectory_gif,
    plot_continuous_dynamic_trajectory_png,
    plot_reward_curve,
    plot_success_rate_bar,
    summarise_episode_results,
)
from r1_uav_nav.evaluation.rollout_selection import (
    Rollout,
    select_fastest_successful_rollout,
    select_longest_rollout,
    select_mean_length_rollout,
)
from r1_uav_nav.utils import create_continuous_dynamic_uav_env_from_config, load_config

ENV_CONFIG_PATH = Path("configs/env/continuous_dynamic_2d.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/td3_continuous_dynamic_full.yaml")
PLOTS_DIR = Path("results/plots/td3_continuous_dynamic")
NUM_EVAL_EPISODES = 100
EVAL_SEED = 42
TRAINING_COMMAND = "python scripts/train_td3_continuous_dynamic.py"


def main() -> None:
    """Run deterministic evaluation for the trained TD3 baseline."""
    env = create_continuous_dynamic_uav_env_from_config(ENV_CONFIG_PATH)
    training_config = load_config(TRAINING_CONFIG_PATH)
    model_output_path = Path(training_config["model_output_path"])
    if not model_output_path.exists():
        raise FileNotFoundError(
            f"TD3 model not found at {model_output_path}. "
            f"Run `{TRAINING_COMMAND}` first."
        )

    model = TD3.load(model_output_path, env=env)

    episode_results: list[EpisodeResult] = []
    episode_rewards: list[float] = []
    rollouts: list[Rollout] = []
    timeout_count = 0

    for episode_index in range(NUM_EVAL_EPISODES):
        observation, _ = env.reset(seed=EVAL_SEED + episode_index)
        uav_positions = [env.uav_position]
        dynamic_obstacle_positions = [_dynamic_obstacle_positions(env)]
        start_position = env.uav_position
        goal_position = env.goal_position
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
            dynamic_obstacle_positions.append(_dynamic_obstacle_positions(env))

        path_length = calculate_path_length(uav_positions)
        success = bool(final_info.get("is_success", False))
        collision = bool(final_info.get("is_collision", False))
        timeout = bool(truncated)
        collision_step = steps if collision else None
        timeout_count += int(timeout)

        episode_rewards.append(total_reward)
        rollouts.append(
            {
                "uav_positions": uav_positions,
                "dynamic_obstacle_positions": dynamic_obstacle_positions,
                "start_position": start_position,
                "goal_position": goal_position,
                "world_size": env.world_size,
                "success": success,
                "collision": collision,
                "truncated": timeout,
                "collision_type": final_info.get("collision_type"),
                "collision_step": collision_step,
                "steps": steps,
                "total_reward": total_reward,
                "path_length": path_length,
            }
        )
        episode_results.append(
            EpisodeResult(
                total_reward=total_reward,
                steps=steps,
                success=success,
                collision=collision,
                path_length=path_length,
            )
        )

    summary = summarise_episode_results(episode_results)
    selected_fastest_rollout = select_fastest_successful_rollout(rollouts)
    selected_mean_length_rollout = select_mean_length_rollout(rollouts)
    selected_longest_rollout = select_longest_rollout(rollouts)
    plot_paths = [
        _plot_continuous_rollout(
            selected_fastest_rollout,
            PLOTS_DIR / "trajectory.png",
        ),
        _plot_continuous_rollout(
            selected_mean_length_rollout,
            PLOTS_DIR / "trajectory_mean_length.png",
        ),
        _plot_continuous_rollout(
            selected_longest_rollout,
            PLOTS_DIR / "trajectory_longest.png",
        ),
        _animate_continuous_rollout(
            selected_fastest_rollout,
            PLOTS_DIR / "trajectory.gif",
        ),
        _animate_continuous_rollout(
            selected_mean_length_rollout,
            PLOTS_DIR / "trajectory_mean_length.gif",
        ),
        _animate_continuous_rollout(
            selected_longest_rollout,
            PLOTS_DIR / "trajectory_longest.gif",
        ),
        plot_reward_curve(
            episode_rewards=episode_rewards,
            output_path=PLOTS_DIR / "reward_curve.png",
        ),
        plot_success_rate_bar(
            summary=summary,
            output_path=PLOTS_DIR / "success_rate.png",
        ),
        plot_collision_rate_bar(
            summary=summary,
            output_path=PLOTS_DIR / "collision_rate.png",
        ),
    ]

    timeout_rate = timeout_count / summary.num_episodes
    print("TD3 continuous dynamic evaluation complete.")
    print(f"Episodes: {summary.num_episodes}")
    print(f"Success rate: {summary.success_rate:.2f}")
    print(f"Collision rate: {summary.collision_rate:.2f}")
    print(f"Average reward: {summary.average_reward:.2f}")
    print(f"Average steps: {summary.average_steps:.2f}")
    print(f"Average path length: {summary.average_path_length:.2f}")
    print(f"Timeout rate: {timeout_rate:.2f}")
    print(f"Model path: {model_output_path}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


def _plot_continuous_rollout(rollout: Rollout, output_path: Path) -> Path:
    return plot_continuous_dynamic_trajectory_png(
        uav_positions=rollout["uav_positions"],
        dynamic_obstacle_positions=rollout["dynamic_obstacle_positions"],
        start_position=rollout["start_position"],
        goal_position=rollout["goal_position"],
        world_size=rollout["world_size"],
        output_path=output_path,
        collision_step=rollout["collision_step"],
    )


def _animate_continuous_rollout(rollout: Rollout, output_path: Path) -> Path:
    return plot_continuous_dynamic_trajectory_gif(
        uav_positions=rollout["uav_positions"],
        dynamic_obstacle_positions=rollout["dynamic_obstacle_positions"],
        start_position=rollout["start_position"],
        goal_position=rollout["goal_position"],
        world_size=rollout["world_size"],
        output_path=output_path,
        collision_step=rollout["collision_step"],
    )


def _dynamic_obstacle_positions(env: Any) -> list[tuple[float, float]]:
    return [obstacle.position for obstacle in env.dynamic_obstacles]


if __name__ == "__main__":
    main()
