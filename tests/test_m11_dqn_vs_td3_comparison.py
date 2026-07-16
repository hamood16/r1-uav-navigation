from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

from r1_uav_nav.evaluation.dynamic_comparison import (
    get_comparison_metric_names,
    get_dynamic_comparison_results,
)

DOC_PATH = Path("docs/results/dqn_vs_td3_dynamic.md")
EXPECTED_PLOT_FILENAMES = {
    "success_rate_comparison.png",
    "collision_rate_comparison.png",
    "average_reward_comparison.png",
    "average_steps_comparison.png",
    "path_length_comparison.png",
    "timeout_rate_comparison.png",
}


def test_dqn_vs_td3_comparison_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_dqn_vs_td3_comparison_doc_mentions_core_topics() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "DQN" in doc_text
    assert "TD3" in doc_text
    assert "discrete" in doc_text
    assert "continuous" in doc_text
    assert "repeated evaluation" in doc_text


def test_dqn_vs_td3_comparison_doc_contains_key_values_and_caveat() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "0.8920" in doc_text
    assert "0.9760" in doc_text
    assert "not directly apples-to-apples" in doc_text


def test_dynamic_comparison_helper_returns_expected_metrics_and_values() -> None:
    metric_names = get_comparison_metric_names()
    results = get_dynamic_comparison_results()
    dqn_result, td3_result = results

    assert metric_names == (
        "success_rate",
        "collision_rate",
        "average_reward",
        "average_steps",
        "average_path_length",
        "timeout_rate",
    )
    assert dqn_result.success_rate == pytest.approx(0.8920)
    assert dqn_result.collision_rate == pytest.approx(0.1080)
    assert td3_result.success_rate == pytest.approx(0.9760)
    assert td3_result.collision_rate == pytest.approx(0.0240)


def test_plot_script_parse_args_defaults() -> None:
    module = _load_plot_script_module()

    args = module.parse_args([])

    assert args.output_dir == Path("results/plots/comparison/dqn_vs_td3")


def test_generate_comparison_plots_writes_expected_pngs(tmp_path: Path) -> None:
    module = _load_plot_script_module()

    plot_paths = module.generate_comparison_plots(tmp_path)

    assert {plot_path.name for plot_path in plot_paths} == EXPECTED_PLOT_FILENAMES
    assert all(plot_path.parent == tmp_path for plot_path in plot_paths)
    assert all(plot_path.exists() for plot_path in plot_paths)
    assert all(plot_path.stat().st_size > 0 for plot_path in plot_paths)


def _load_plot_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_dqn_vs_td3_comparison_plots.py"
    )
    spec = spec_from_file_location("generate_dqn_vs_td3_comparison_plots", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
