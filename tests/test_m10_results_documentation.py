from pathlib import Path

REPORT_PATH = Path("docs/m10_dynamic_rl_results.md")


def test_m10_dynamic_rl_results_report_exists() -> None:
    assert REPORT_PATH.exists()


def test_m10_dynamic_rl_results_report_mentions_core_topics() -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    assert "DQN" in report_text
    assert "TD3" in report_text
    assert "repeated evaluation" in report_text


def test_m10_dynamic_rl_results_report_contains_key_repeated_success_rates() -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    assert "0.8920" in report_text
    assert "0.9760" in report_text


def test_m10_dynamic_rl_results_report_contains_required_caveat() -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    assert "not directly apples-to-apples" in report_text


def test_m10_dynamic_rl_results_report_contains_repeated_report_paths() -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    assert "results/reports/m10/dqn_dynamic_tuned_repeated_eval.json" in report_text
    assert (
        "results/reports/m10/td3_continuous_dynamic_tuned_repeated_eval.json"
        in report_text
    )
