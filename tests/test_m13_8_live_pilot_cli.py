from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_m13_8_static_curriculum.py"
RUNTIME_MODULE = "r1_uav_nav.training.curriculum_live_runtime"


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimum_live_args(command: str) -> list[str]:
    return [
        command,
        "--stage-id",
        "stage-0",
        "--course-profile",
        "curriculum-empty-train",
        "--course-seed",
        "20000",
        "--pilot-kind",
        "smoke",
        "--max-timesteps",
        "100",
        "--expected-m13-6-suite-digest",
        "0" * 64,
        "--preflight-survey-report",
        "missing-preflight.json",
        "--preflight-survey-digest",
        "0" * 64,
        "--grounded-lidar-report",
        "missing-grounded.json",
        "--grounded-lidar-digest",
        "0" * 64,
    ]


def test_help_lists_phase_b_commands_without_importing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop(RUNTIME_MODULE, None)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"airsim", "cosysairsim"}:
            raise AssertionError("CLI help imported a simulator module")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load_script("m13_8_live_help")
    choices = module.build_parser()._subparsers._group_actions[0].choices
    assert {
        "preflight-live-pilot",
        "pilot-stage",
        "resume-pilot",
        "summarize-live-pilot",
    }.issubset(choices)
    assert RUNTIME_MODULE not in sys.modules


def test_missing_authorization_fails_before_runtime_loader() -> None:
    module = _load_script("m13_8_live_preflight_gate")
    called = False

    def runtime_loader():
        nonlocal called
        called = True
        raise AssertionError("runtime loader must not be reached")

    with pytest.raises(ValueError, match="required authorization"):
        module.run(
            module.parse_args(_minimum_live_args("pilot-stage")),
            repository_root=ROOT,
            runtime_loader=runtime_loader,
        )
    assert not called


def test_stage_two_fails_before_runtime_loader() -> None:
    module = _load_script("m13_8_live_stage_gate")
    args = _minimum_live_args("pilot-stage")
    args[args.index("stage-0")] = "stage-2"
    authorization_flags = [
        f"--{name.replace('_', '-')}"
        for name in module.LivePilotAuthorizations.__dataclass_fields__
    ]
    with pytest.raises(ValueError, match="stage-0 and stage-1"):
        module.run(
            module.parse_args([*args, *authorization_flags]),
            repository_root=ROOT,
            runtime_loader=lambda: (_ for _ in ()).throw(
                AssertionError("runtime loader must not be reached")
            ),
        )
