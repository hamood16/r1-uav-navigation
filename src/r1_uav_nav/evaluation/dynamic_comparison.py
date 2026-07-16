"""Fixed comparison data for tuned dynamic DQN and TD3 results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicAlgorithmResult:
    """Repeated-evaluation summary for one dynamic navigation algorithm."""

    label: str
    control_type: str
    environment: str
    success_rate: float
    success_rate_std: float
    collision_rate: float
    collision_rate_std: float
    timeout_rate: float
    timeout_rate_std: float
    average_reward: float
    average_reward_std: float
    average_steps: float
    average_steps_std: float
    average_path_length: float
    average_path_length_std: float


TUNED_DQN_RESULT = DynamicAlgorithmResult(
    label="Tuned DQN",
    control_type="Discrete actions",
    environment="DynamicGridUAVEnv",
    success_rate=0.8920,
    success_rate_std=0.0232,
    collision_rate=0.1080,
    collision_rate_std=0.0232,
    timeout_rate=0.0000,
    timeout_rate_std=0.0000,
    average_reward=9.0604,
    average_reward_std=0.4605,
    average_steps=7.1080,
    average_steps_std=0.1888,
    average_path_length=6.9740,
    average_path_length_std=0.1648,
)
TUNED_TD3_RESULT = DynamicAlgorithmResult(
    label="Tuned TD3",
    control_type="Continuous velocity",
    environment="ContinuousDynamicUAVEnv",
    success_rate=0.9760,
    success_rate_std=0.0162,
    collision_rate=0.0240,
    collision_rate_std=0.0162,
    timeout_rate=0.0000,
    timeout_rate_std=0.0000,
    average_reward=10.7858,
    average_reward_std=0.3273,
    average_steps=5.9600,
    average_steps_std=0.1071,
    average_path_length=6.1078,
    average_path_length_std=0.2168,
)


def get_dynamic_comparison_results() -> tuple[DynamicAlgorithmResult, ...]:
    """Return tuned dynamic RL comparison results in display order."""
    return (TUNED_DQN_RESULT, TUNED_TD3_RESULT)


def get_comparison_metric_names() -> tuple[str, ...]:
    """Return metric names used for the DQN-vs-TD3 comparison plots."""
    return (
        "success_rate",
        "collision_rate",
        "average_reward",
        "average_steps",
        "average_path_length",
        "timeout_rate",
    )
