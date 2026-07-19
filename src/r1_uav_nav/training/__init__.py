"""Training utilities for UAV navigation experiments."""

from r1_uav_nav.training.colosseum_td3 import (
    DEFAULT_CONFIG_PATH,
    EVALUATION_POLICY_KINDS,
    ColosseumTD3Config,
    ColosseumTD3RunResult,
    EvaluationEpisodeMetrics,
    TrainingEpisodeMetrics,
    apply_evaluation_overrides,
    apply_training_overrides,
    evaluate_colosseum_td3,
    load_colosseum_td3_config,
    resolve_device,
    train_colosseum_td3,
)

__all__ = [
    "ColosseumTD3Config",
    "ColosseumTD3RunResult",
    "DEFAULT_CONFIG_PATH",
    "EVALUATION_POLICY_KINDS",
    "EvaluationEpisodeMetrics",
    "TrainingEpisodeMetrics",
    "apply_evaluation_overrides",
    "apply_training_overrides",
    "evaluate_colosseum_td3",
    "load_colosseum_td3_config",
    "resolve_device",
    "train_colosseum_td3",
]
