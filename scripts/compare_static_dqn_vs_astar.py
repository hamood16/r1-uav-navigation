"""Compare the trained static DQN baseline against A* on shared layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

from r1_uav_nav.evaluation import (
    calculate_path_length,
    plot_metric_comparison,
    plot_trajectory,
    plot_trajectory_overlay,
)
from r1_uav_nav.evaluation.static_comparison import (
    StaticLayout,
    assert_layouts_match,
    select_representative_shared_success_episode,
    summarise_static_comparison,
)
from r1_uav_nav.planners import find_astar_path
from r1_uav_nav.utils import create_grid_uav_env_from_config, load_config

ENV_CONFIG_PATH = Path("configs/env/grid_2d_static_full.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/dqn_static_full.yaml")
PLOTS_DIR = Path("results/plots/comparison/static_dqn_vs_astar")
SUMMARY_PATH = PLOTS_DIR / "comparison_summary.json"
NUM_EVAL_EPISODES = 100
EVAL_SEED = 42
TRAINING_COMMAND = "python scripts/train_static_dqn.py"
ComparisonRecord = dict[str, Any]


def main() -> None:
    """Run the static DQN-vs-A* comparison."""
    env = create_grid_uav_env_from_config(ENV_CONFIG_PATH)
    training_config = load_config(TRAINING_CONFIG_PATH)
    model_output_path = Path(training_config["model_output_path"])
    if not model_output_path.exists():
        raise FileNotFoundError(
            f"Static DQN model not found at {model_output_path}. "
            f"Run `{TRAINING_COMMAND}` first."
        )

    model = DQN.load(model_output_path, env=env)
    records: list[ComparisonRecord] = []

    for episode_index in range(NUM_EVAL_EPISODES):
        episode_seed = EVAL_SEED + episode_index
        env.reset(seed=episode_seed)
        astar_layout = _layout_from_env(env)
        astar_record = _evaluate_astar_layout(astar_layout)

        observation, _ = env.reset(seed=episode_seed)
        dqn_layout = _layout_from_env(env)
        assert_layouts_match(astar_layout, dqn_layout)
        dqn_record = _evaluate_dqn_episode(env, model, observation)

        records.append(
            {
                "episode_index": episode_index,
                "seed": episode_seed,
                "layout": {
                    "start_position": astar_layout.start_position,
                    "goal_position": astar_layout.goal_position,
                    "obstacles": sorted(astar_layout.obstacles),
                    "grid_size": astar_layout.grid_size,
                },
                "astar": astar_record,
                "dqn": dqn_record,
            }
        )

    summary = summarise_static_comparison(
        records,
        eval_seed=EVAL_SEED,
        model_path=str(model_output_path),
    )
    selected_record = select_representative_shared_success_episode(records)
    summary["selected_episode"] = {
        "episode_index": selected_record["episode_index"],
        "seed": selected_record["seed"],
    }

    plot_paths = _save_comparison_plots(summary, selected_record)
    summary_path = _save_summary(summary)

    print("Static DQN vs A* comparison complete.")
    print(f"Episodes: {summary['num_episodes']}")
    print(f"A* success rate: {summary['astar']['success_rate']:.2f}")
    print(f"A* failure rate: {summary['astar']['failure_rate']:.2f}")
    print(
        "A* average successful path length: "
        f"{summary['astar']['average_successful_path_length']:.2f}"
    )
    print(
        "A* average successful steps: "
        f"{summary['astar']['average_successful_steps']:.2f}"
    )
    print(f"DQN success rate: {summary['dqn']['success_rate']:.2f}")
    print(f"DQN collision rate: {summary['dqn']['collision_rate']:.2f}")
    print(f"DQN timeout rate: {summary['dqn']['timeout_rate']:.2f}")
    print(
        f"DQN average reward over all episodes: {summary['dqn']['average_reward']:.2f}"
    )
    print(
        "DQN average path length over all episodes: "
        f"{summary['dqn']['average_path_length']:.2f}"
    )
    print(f"DQN average steps over all episodes: {summary['dqn']['average_steps']:.2f}")
    print(
        "Shared-success count: "
        f"{summary['shared_success_comparison']['shared_success_count']}"
    )
    print(f"Summary path: {summary_path}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


def _evaluate_astar_layout(layout: StaticLayout) -> dict[str, Any]:
    planned_path = find_astar_path(
        start=layout.start_position,
        goal=layout.goal_position,
        obstacles=set(layout.obstacles),
        grid_size=layout.grid_size,
    )
    success = planned_path is not None
    path = planned_path if planned_path is not None else [layout.start_position]
    return {
        "success": success,
        "failure": not success,
        "collision": False,
        "path": path,
        "steps": len(path) - 1 if success else 0,
        "path_length": calculate_path_length(path) if success else 0.0,
    }


def _evaluate_dqn_episode(
    env: Any,
    model: DQN,
    observation: Any,
) -> dict[str, Any]:
    positions = [env.uav_position]
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        positions.append(env.uav_position)

    success = bool(
        info.get("is_success", terminated and env.uav_position == env.goal_position)
    )
    collision = bool(info.get("is_collision", terminated and not success))
    timeout = bool(truncated)

    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "total_reward": total_reward,
        "positions": positions,
        "steps": steps,
        "path_length": calculate_path_length(positions),
    }


def _layout_from_env(env: Any) -> StaticLayout:
    return StaticLayout(
        start_position=env.uav_position,
        goal_position=env.goal_position,
        obstacles=frozenset(env.obstacles),
        grid_size=env.grid_size,
    )


def _save_comparison_plots(
    summary: dict[str, Any],
    selected_record: ComparisonRecord,
) -> list[Path]:
    layout = selected_record["layout"]
    return [
        plot_metric_comparison(
            labels=["A*", "DQN"],
            values=[
                summary["astar"]["success_rate"],
                summary["dqn"]["success_rate"],
            ],
            title="Static success rate comparison",
            ylabel="Success rate",
            output_path=PLOTS_DIR / "success_rate_comparison.png",
        ),
        plot_metric_comparison(
            labels=["A* successful avg", "DQN all episodes avg"],
            values=[
                summary["astar"]["average_successful_path_length"],
                summary["dqn"]["average_path_length"],
            ],
            title="Static path length comparison",
            ylabel="Path length",
            output_path=PLOTS_DIR / "path_length_comparison.png",
        ),
        plot_metric_comparison(
            labels=["A* successful avg", "DQN all episodes avg"],
            values=[
                summary["astar"]["average_successful_steps"],
                summary["dqn"]["average_steps"],
            ],
            title="Static steps comparison",
            ylabel="Steps",
            output_path=PLOTS_DIR / "steps_comparison.png",
        ),
        plot_metric_comparison(
            labels=["A*", "DQN"],
            values=[
                summary["astar"]["collision_rate"],
                summary["dqn"]["collision_rate"],
            ],
            title="Static collision rate comparison",
            ylabel="Collision rate",
            output_path=PLOTS_DIR / "collision_rate_comparison.png",
        ),
        plot_trajectory(
            trajectory_positions=selected_record["astar"]["path"],
            obstacles=layout["obstacles"],
            start_position=layout["start_position"],
            goal_position=layout["goal_position"],
            grid_size=layout["grid_size"],
            output_path=PLOTS_DIR / "trajectory_astar.png",
        ),
        plot_trajectory(
            trajectory_positions=selected_record["dqn"]["positions"],
            obstacles=layout["obstacles"],
            start_position=layout["start_position"],
            goal_position=layout["goal_position"],
            grid_size=layout["grid_size"],
            output_path=PLOTS_DIR / "trajectory_dqn.png",
        ),
        plot_trajectory_overlay(
            astar_positions=selected_record["astar"]["path"],
            dqn_positions=selected_record["dqn"]["positions"],
            obstacles=layout["obstacles"],
            start_position=layout["start_position"],
            goal_position=layout["goal_position"],
            grid_size=layout["grid_size"],
            output_path=PLOTS_DIR / "trajectory_overlay.png",
        ),
    ]


def _save_summary(summary: dict[str, Any]) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return SUMMARY_PATH


if __name__ == "__main__":
    main()
