from __future__ import annotations

import builtins
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_colosseum_td3_long_run.py"
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_colosseum_throughput.py"
DOC_PATH = ROOT / "docs" / "m13_7_long_run_training_infrastructure.md"


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_and_cli_modules_import_with_airsim_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"airsim", "cosysairsim"}:
            raise AssertionError("offline M13.7 imported a simulator client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import r1_uav_nav.training  # noqa: F401

    _load_script(TRAIN_SCRIPT, "m13_7_train_cli_import_test")
    _load_script(BENCHMARK_SCRIPT, "m13_7_benchmark_cli_import_test")


def test_validate_command_is_offline_and_accepts_no_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(TRAIN_SCRIPT, "m13_7_train_cli_validate_test")
    args = module.parse_args(["validate"])
    assert module.run(args, repository_root=ROOT) == 0
    output = capsys.readouterr().out
    assert '"resume_mode": "new"' in output
    assert '"live_training_enabled": false' in output


def test_resume_argument_error_occurs_during_offline_validation() -> None:
    module = _load_script(TRAIN_SCRIPT, "m13_7_train_cli_resume_test")
    args = module.parse_args(["validate", "--resume-checkpoint", "missing-model.zip"])
    with pytest.raises(ValueError, match="requires --resume-run-state"):
        module.run(args, repository_root=ROOT)


def test_benchmark_validate_mode_is_offline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(BENCHMARK_SCRIPT, "m13_7_benchmark_cli_test")
    assert module.run(module.parse_args(["validate"]), repository_root=ROOT) == 0
    assert '"live_benchmark_enabled": false' in capsys.readouterr().out


def test_fake_smoke_then_inspect_latest_succeeds_without_simulator_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_cli_repository(tmp_path)
    monkeypatch.setitem(sys.modules, "airsim", None)
    monkeypatch.setitem(sys.modules, "cosysairsim", None)
    module = _load_script(TRAIN_SCRIPT, "m13_7_train_cli_digest_regression")
    run_id = "cli-digest-regression"
    fake_args = module.parse_args(
        [
            "fake-smoke",
            "--additional-timesteps",
            "20",
            "--run-id",
            run_id,
        ]
    )
    assert module.run(fake_args, repository_root=tmp_path) == 0
    fake_output = capsys.readouterr().out
    assert '"final_timesteps": 20' in fake_output

    run_root = tmp_path / "results" / "trained_models" / "m13_7_long_run_td3" / run_id
    inspect_args = module.parse_args(
        ["inspect-resume", "--resume-latest", str(run_root)]
    )
    assert module.run(inspect_args, repository_root=tmp_path) == 0
    inspect_output = capsys.readouterr().out
    assert '"mode": "full"' in inspect_output
    assert '"reset_num_timesteps": false' in inspect_output


def test_documentation_retains_phase_a_and_no_policy_claims() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "Phase A" in text
    assert "does not run Blocks" in text
    assert "does not demonstrate an obstacle-aware policy" in text
    assert "model-only warm start" in text
    assert "Process exit alone is not safe-state evidence" in text


def _prepare_cli_repository(root: Path) -> None:
    for relative in (
        Path("configs/training/m13_7_long_run_td3.yaml"),
        Path("configs/planning/m13_3_voxel_astar.yaml"),
        Path("configs/env/m13_5_obstacle_uav_env.yaml"),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, destination)
