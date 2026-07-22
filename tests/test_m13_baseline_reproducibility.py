from pathlib import Path

README_PATH = Path("README.md")
BASELINE_DOC_PATH = Path("docs/m13_0_baseline_reproducibility.md")
M12_TD3_DOC_PATH = Path("docs/m12_colosseum_td3_baseline.md")
RUNTIME_REQUIREMENTS_PATH = Path("requirements.txt")
DEV_REQUIREMENTS_PATH = Path("requirements-dev.txt")
COLOSSEUM_REQUIREMENTS_PATH = Path("requirements-colosseum.txt")
CI_WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def test_m13_baseline_doc_records_freeze_facts() -> None:
    doc_text = BASELINE_DOC_PATH.read_text(encoding="utf-8")

    assert BASELINE_DOC_PATH.exists()
    assert "m12.5-complete" in doc_text
    assert "ab83487d3b125dd4072f0cbe3823900be9a10d64" in doc_text
    assert "Python >=3.11,<3.12" in doc_text
    assert "v2.0.0-beta" in doc_text
    assert "7b9658a1" in doc_text
    assert "msgpack==0.6.2" in doc_text
    assert "2,000" in doc_text
    assert "100%" in doc_text
    assert "0.459 m" in doc_text
    assert "does not add LiDAR" in doc_text


def test_readme_describes_current_project_state_without_stale_title() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")

    assert readme_text.startswith("# r1-UAV-navigation")
    assert "# Cleaner UAV" not in readme_text
    assert "implemented through Stable-Baselines3" in readme_text
    assert "A*: implemented" in readme_text
    assert "future options only" in readme_text
    assert "2,000-step TD3 Stage B run" in readme_text
    assert "LiDAR observations" in readme_text
    assert "docs/m13_0_baseline_reproducibility.md" in readme_text
    assert "Unreal Engine later" not in readme_text
    assert "AirSim / Colosseum / Cosys-AirSim later" not in readme_text


def test_m12_5_markdown_fences_are_balanced_and_results_preserved() -> None:
    doc_text = M12_TD3_DOC_PATH.read_text(encoding="utf-8")

    assert doc_text.count("```") % 2 == 0
    assert "| Random policy | 0 | 3 | 0% | -1.260 | 3.081 m |" in doc_text
    assert (
        "| Scripted-forward reference | 0 | 3 | 100% | 12.422 | 0.368 m |" in doc_text
    )
    assert "| TD3 smoke checkpoint | 100 | 3 | 0% | -6.254 | 3.347 m |" in doc_text
    assert "| TD3 Stage A checkpoint | 500 | 5 | 0% | -8.366 | 5.917 m |" in doc_text
    assert (
        "| TD3 Stage B checkpoint | 2,000 | 5 | 100% | 12.305 | 0.459 m |" in doc_text
    )
    assert "Velocity: x=0.0, y=0.0, z=0.0\n```" in doc_text


def test_requirements_are_structured_pinned_and_path_free() -> None:
    runtime_lines = _active_requirement_lines(RUNTIME_REQUIREMENTS_PATH)
    dev_lines = _active_requirement_lines(DEV_REQUIREMENTS_PATH)
    colosseum_lines = _active_requirement_lines(COLOSSEUM_REQUIREMENTS_PATH)
    combined_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            RUNTIME_REQUIREMENTS_PATH,
            DEV_REQUIREMENTS_PATH,
            COLOSSEUM_REQUIREMENTS_PATH,
        )
    )

    assert "pandas" not in combined_text
    assert "opencv-python" not in combined_text
    assert "torch==2.12.1+cu126" not in combined_text
    assert "torch==2.12.1" in runtime_lines
    assert "stable-baselines3==2.9.0" in runtime_lines
    assert "-r requirements.txt" in dev_lines
    assert "-r requirements.txt" in colosseum_lines
    assert "pytest==9.1.1" in dev_lines
    assert "ruff==0.15.19" in dev_lines
    assert "black==26.5.1" in dev_lines
    assert "msgpack==0.6.2" in colosseum_lines
    assert "msgpack-rpc-python==0.4.1" in colosseum_lines

    for line in runtime_lines + dev_lines + colosseum_lines:
        if line.startswith("-r "):
            continue
        assert "==" in line

    assert "-e " not in combined_text
    assert ".venv" not in combined_text
    assert "C:\\" not in combined_text
    assert "/home/" not in combined_text


def test_ci_workflow_is_simulator_independent_python_311() -> None:
    workflow_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert CI_WORKFLOW_PATH.exists()
    assert 'python-version: "3.11"' in workflow_text
    assert "permissions:" in workflow_text
    assert "contents: read" in workflow_text
    assert "python -m pip check" in workflow_text
    assert "python -m pytest" in workflow_text
    assert "python -m ruff check ." in workflow_text
    assert "python -m black --check --no-cache ." in workflow_text
    assert "requirements-dev.txt" in workflow_text

    forbidden_terms = (
        "Colosseum",
        "Blocks",
        "Unreal",
        "CUDA",
        "train_colosseum_td3",
        "evaluate_colosseum_td3",
        "check_colosseum_connection",
        "run_colosseum_waypoint_demo",
    )
    for term in forbidden_terms:
        assert term not in workflow_text


def _active_requirement_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines
