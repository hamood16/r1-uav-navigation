import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

from r1_uav_nav.evaluation.repeated_evaluation import (
    RepeatedEvaluationRun,
    calculate_mean_std,
    save_repeated_evaluation_summary,
    summarise_repeated_evaluations,
)


def test_calculate_mean_std_uses_population_standard_deviation() -> None:
    result = calculate_mean_std([1.0, 2.0, 3.0])

    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx((2.0 / 3.0) ** 0.5)


def test_calculate_mean_std_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty values"):
        calculate_mean_std([])


def test_summarise_repeated_evaluations_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="empty repeated evaluation runs"):
        summarise_repeated_evaluations([])


def test_summarise_repeated_evaluations_returns_json_serialisable_summary() -> None:
    summary = summarise_repeated_evaluations(
        _sample_runs(),
        metadata={"algorithm": "DQN", "base_seed": 42},
    )

    encoded_summary = json.dumps(summary)

    assert encoded_summary
    assert summary["metadata"] == {"algorithm": "DQN", "base_seed": 42}
    assert summary["num_repeats"] == 2
    assert summary["episodes_per_repeat"] == 100
    assert summary["seeds"] == [42, 1042]
    assert summary["metrics"]["success_rate"]["mean"] == pytest.approx(0.75)
    assert summary["metrics"]["success_rate"]["std"] == pytest.approx(0.05)
    assert summary["metrics"]["timeout_rate"]["mean"] == pytest.approx(0.075)
    assert len(summary["runs"]) == 2


def test_save_repeated_evaluation_summary_writes_json(tmp_path: Path) -> None:
    summary = summarise_repeated_evaluations(_sample_runs())
    output_path = tmp_path / "reports" / "summary.json"

    saved_path = save_repeated_evaluation_summary(summary, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_dynamic_dqn_repeated_parse_args_defaults() -> None:
    module = _load_script_module("evaluate_dynamic_dqn_repeated.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/dynamic_grid_2d.yaml")
    assert args.training_config == Path("configs/training/dqn_dynamic_full.yaml")
    assert args.output_path == Path(
        "results/reports/m10/dynamic_dqn_repeated_eval.json"
    )
    assert args.repeats == 5
    assert args.episodes_per_repeat == 100
    assert args.base_seed == 42


def test_td3_repeated_parse_args_defaults() -> None:
    module = _load_script_module("evaluate_td3_continuous_dynamic_repeated.py")

    args = module.parse_args([])

    assert args.env_config == Path("configs/env/continuous_dynamic_2d.yaml")
    assert args.training_config == Path(
        "configs/training/td3_continuous_dynamic_full.yaml"
    )
    assert args.output_path == Path(
        "results/reports/m10/td3_continuous_dynamic_repeated_eval.json"
    )
    assert args.repeats == 5
    assert args.episodes_per_repeat == 100
    assert args.base_seed == 42


def _sample_runs() -> list[RepeatedEvaluationRun]:
    return [
        RepeatedEvaluationRun(
            repeat_index=0,
            seed=42,
            num_episodes=100,
            success_rate=0.70,
            collision_rate=0.20,
            timeout_rate=0.10,
            average_reward=6.0,
            average_steps=12.0,
            average_path_length=10.0,
        ),
        RepeatedEvaluationRun(
            repeat_index=1,
            seed=1042,
            num_episodes=100,
            success_rate=0.80,
            collision_rate=0.10,
            timeout_rate=0.05,
            average_reward=8.0,
            average_steps=10.0,
            average_path_length=8.0,
        ),
    ]


def _load_script_module(script_name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = spec_from_file_location(script_name.removesuffix(".py"), script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
