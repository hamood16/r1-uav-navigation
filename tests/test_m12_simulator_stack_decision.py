from pathlib import Path

DOC_PATH = Path("docs/m12_simulator_stack_decision.md")


def test_m12_simulator_stack_decision_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_m12_simulator_stack_decision_doc_mentions_core_stack_choices() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "Colosseum" in doc_text
    assert "AirSim" in doc_text
    assert "PX4" in doc_text


def test_m12_simulator_stack_decision_doc_mentions_staged_roadmap() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "M12.2" in doc_text
    assert "M12.3" in doc_text
    assert "M12.4" in doc_text
    assert "M12.5" in doc_text


def test_m12_simulator_stack_decision_doc_mentions_initial_scope() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "no camera-based RL at first" in doc_text
    assert "continuous velocity commands" in doc_text
