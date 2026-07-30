from __future__ import annotations

import builtins
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_m13_8_static_curriculum.py"


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_and_import_are_simulator_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"airsim", "cosysairsim"}:
            raise AssertionError("offline M13.8 imported a simulator client")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load_script("m13_8_cli_import")
    assert "fake-smoke" in module.build_parser().format_help()
    with pytest.raises(SystemExit) as exit_info:
        module.parse_args(["--help"])
    assert exit_info.value.code == 0


def test_preview_exposes_gated_phase_b_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("m13_8_cli_preview")
    choices = module.build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {
        "validate",
        "preview-stages",
        "fake-smoke",
        "summarize",
        "inspect-curriculum-state",
        "preflight-live-pilot",
        "pilot-stage",
        "resume-pilot",
        "summarize-live-pilot",
    }
    assert module.run(module.parse_args(["preview-stages"]), repository_root=ROOT) == 0
    output = capsys.readouterr().out
    assert '"live_training_enabled": false' in output
    assert '"stage_id": "stage-5"' in output


def test_fake_smoke_persists_curriculum_through_full_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_repository(tmp_path)
    module = _load_script("m13_8_cli_fake_smoke")
    args = module.parse_args(
        ["fake-smoke", "--timesteps", "100", "--run-id", "curriculum-smoke"]
    )
    assert module.run(args, repository_root=tmp_path) == 0
    output = capsys.readouterr().out
    assert '"full_resume_verified": true' in output
    assert '"curriculum_stage": "stage-0"' in output
    assert '"live_training_executed": false' in output


def test_fake_smoke_rejects_long_budget() -> None:
    module = _load_script("m13_8_cli_budget")
    with pytest.raises(ValueError, match="100-500"):
        module.run(
            module.parse_args(["fake-smoke", "--timesteps", "501"]),
            repository_root=ROOT,
        )


def _prepare_repository(root: Path) -> None:
    for relative in (
        "configs/training/m13_8_static_obstacle_curriculum.yaml",
        "configs/training/m13_7_long_run_td3.yaml",
        "configs/planning/m13_8_curriculum_courses.yaml",
        "configs/planning/m13_3_voxel_astar.yaml",
        "configs/env/m13_5_obstacle_uav_env.yaml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, destination)
    (root / "results").mkdir()
