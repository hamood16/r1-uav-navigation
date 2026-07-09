"""Generate dynamic GridUAVEnv rollout visualisations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from r1_uav_nav.envs import DynamicGridUAVEnv, DynamicObstacle
from r1_uav_nav.evaluation import (
    calculate_path_length,
    plot_dynamic_trajectory_gif,
    plot_dynamic_trajectory_png,
)
from r1_uav_nav.utils import create_dynamic_grid_uav_env_from_config

ENV_CONFIG_PATH = Path("configs/env/dynamic_grid_2d.yaml")
PLOTS_DIR = Path("results/plots/dynamic")
NUM_ROLLOUTS = 5
ROLLOUT_SEED = 42

_ACTION_DELTAS = {
    0: (0, 1),
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0),
    4: (0, 0),
}


def main() -> None:
    """Run safety-aware dynamic rollouts and save visualisations."""
    env = create_dynamic_grid_uav_env_from_config(ENV_CONFIG_PATH)

    rollouts = [
        _run_rollout(env, seed=ROLLOUT_SEED + rollout_index)
        for rollout_index in range(NUM_ROLLOUTS)
    ]
    successful_rollouts = [rollout for rollout in rollouts if rollout["success"]]

    selected_fastest_rollout = (
        min(successful_rollouts, key=lambda rollout: rollout["steps"])
        if successful_rollouts
        else rollouts[0]
    )
    mean_length_candidates = successful_rollouts if successful_rollouts else rollouts
    mean_path_length = sum(
        rollout["path_length"] for rollout in mean_length_candidates
    ) / len(mean_length_candidates)
    selected_mean_length_rollout = min(
        mean_length_candidates,
        key=lambda rollout: abs(rollout["path_length"] - mean_path_length),
    )
    selected_longest_rollout = max(
        mean_length_candidates,
        key=lambda rollout: rollout["path_length"],
    )

    plot_paths = [
        _plot_dynamic_rollout(
            selected_fastest_rollout,
            PLOTS_DIR / "dynamic_trajectory.png",
        ),
        _plot_dynamic_rollout(
            selected_mean_length_rollout,
            PLOTS_DIR / "dynamic_trajectory_mean_length.png",
        ),
        _plot_dynamic_rollout(
            selected_longest_rollout,
            PLOTS_DIR / "dynamic_trajectory_longest.png",
        ),
        plot_dynamic_trajectory_gif(
            uav_positions=selected_fastest_rollout["uav_positions"],
            dynamic_obstacle_positions=selected_fastest_rollout[
                "dynamic_obstacle_positions"
            ],
            start_position=selected_fastest_rollout["start_position"],
            goal_position=selected_fastest_rollout["goal_position"],
            grid_size=selected_fastest_rollout["grid_size"],
            output_path=PLOTS_DIR / "dynamic_trajectory.gif",
            collision_step=selected_fastest_rollout["collision_step"],
        ),
    ]

    print("Dynamic environment visualisation complete.")
    print(f"Rollouts: {NUM_ROLLOUTS}")
    print(f"Selected rollout success: {selected_fastest_rollout['success']}")
    print(f"Selected rollout collision: {selected_fastest_rollout['collision']}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


def _run_rollout(env: DynamicGridUAVEnv, seed: int) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    del observation

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
        action = _choose_safety_aware_greedy_action(env)
        _, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        steps += 1
        uav_positions.append(env.uav_position)
        dynamic_obstacle_positions.append(_dynamic_obstacle_positions(env))

    collision_step = steps if final_info["is_collision"] else None
    return {
        "uav_positions": uav_positions,
        "dynamic_obstacle_positions": dynamic_obstacle_positions,
        "start_position": start_position,
        "goal_position": goal_position,
        "grid_size": env.grid_size,
        "success": final_info["is_success"],
        "collision": final_info["is_collision"],
        "collision_type": final_info["collision_type"],
        "collision_step": collision_step,
        "steps": steps,
        "total_reward": total_reward,
        "path_length": calculate_path_length(uav_positions),
    }


def _choose_safety_aware_greedy_action(env: DynamicGridUAVEnv) -> int:
    current_obstacle_positions = {
        obstacle.position for obstacle in env.dynamic_obstacles
    }
    predicted_obstacle_positions = {
        _predict_next_obstacle_position(env, obstacle)
        for obstacle in env.dynamic_obstacles
    }
    candidate_actions = []

    for action, (dx, dy) in _ACTION_DELTAS.items():
        candidate_position = (env.uav_position[0] + dx, env.uav_position[1] + dy)
        if action != 4 and not _is_within_grid(candidate_position, env.grid_size):
            continue
        if candidate_position in current_obstacle_positions:
            continue
        if candidate_position in predicted_obstacle_positions:
            continue

        distance_to_goal = math.hypot(
            candidate_position[0] - env.goal_position[0],
            candidate_position[1] - env.goal_position[1],
        )
        candidate_actions.append((distance_to_goal, action))

    if not candidate_actions:
        return 4

    return min(candidate_actions)[1]


def _predict_next_obstacle_position(
    env: DynamicGridUAVEnv,
    obstacle: DynamicObstacle,
) -> tuple[int, int]:
    x, y = obstacle.position
    vx, vy = obstacle.velocity
    next_position = (x + vx, y + vy)

    if _is_within_grid(next_position, env.grid_size):
        return next_position

    if not 0 <= next_position[0] < env.grid_size:
        vx *= -1
    if not 0 <= next_position[1] < env.grid_size:
        vy *= -1

    return x + vx, y + vy


def _plot_dynamic_rollout(rollout: dict[str, Any], output_path: Path) -> Path:
    return plot_dynamic_trajectory_png(
        uav_positions=rollout["uav_positions"],
        dynamic_obstacle_positions=rollout["dynamic_obstacle_positions"],
        start_position=rollout["start_position"],
        goal_position=rollout["goal_position"],
        grid_size=rollout["grid_size"],
        output_path=output_path,
        collision_step=rollout["collision_step"],
    )


def _dynamic_obstacle_positions(env: DynamicGridUAVEnv) -> list[tuple[int, int]]:
    return [obstacle.position for obstacle in env.dynamic_obstacles]


def _is_within_grid(position: tuple[int, int], grid_size: int) -> bool:
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size


if __name__ == "__main__":
    main()
