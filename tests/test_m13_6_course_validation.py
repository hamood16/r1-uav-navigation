from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from r1_uav_nav.evaluation.m13_6_course_validation import (
    M13_6LiveAuthorizations,
    execute_reference_episode,
    load_m13_6_config,
    prepare_reference_run,
    save_reference_episode_report,
    summarize_episode_reports,
    validate_offline_configuration,
)
from r1_uav_nav.evaluation.reference_controllers import ControllerStateSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "evaluation" / "m13_6_reference_controllers.yaml"
SCRIPT_PATH = ROOT / "scripts" / "check_m13_6_reference_controllers.py"


def _authorizations(**changes: bool) -> M13_6LiveAuthorizations:
    values = {name: True for name in M13_6LiveAuthorizations.__dataclass_fields__}
    values.update(changes)
    return M13_6LiveAuthorizations(**values)


@dataclass(frozen=True)
class FakeCleanup:
    succeeded: bool = True
    scene_cleanup_deferred: bool = False
    scene_cleanup_deferred_reason: str | None = None
    actions: tuple[str, ...] = ("uav-cleanup", "scene-cleanup", "marker-cleanup")


class FakeClient:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        raise AssertionError("broad reset must not be called")


class FakeReferenceEnvironment:
    def __init__(
        self,
        *_: Any,
        outcome: str = "goal",
        events: list[str] | None = None,
        **__: Any,
    ) -> None:
        self.outcome = outcome
        self.events = events if events is not None else []
        self.step_calls = 0
        self.closed = False
        self.action_space = None
        self.observation_space = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert seed is not None
        assert options is not None and options["validated_course"].result.accepted
        self.events.append("reset")
        return self._observation(True), self._info(distance=9.0, clearance=2.0)

    def controller_state_spec(self) -> ControllerStateSpec:
        return ControllerStateSpec(
            (20.0, 20.0, 6.0),
            (40.0, 40.0, 12.0),
            (0.5, 0.5, 0.4),
        )

    def latest_lidar_evidence(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "diagnostics": {
                "timestamp": 100 + self.step_calls,
                "timestamp_status": "fresh",
                "nearest_overall_m": 2.0,
            },
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_calls += 1
        self.events.append("step")
        if self.outcome == "exception":
            raise RuntimeError("fake controller-run failure")
        if self.outcome == "interrupt":
            raise KeyboardInterrupt
        if self.outcome == "sensor":
            return (
                self._observation(False),
                -1.0,
                False,
                True,
                self._info(
                    distance=8.9,
                    clearance=2.0,
                    sensor_failure=True,
                    reason="sensor_failure",
                ),
            )
        return (
            self._observation(True),
            5.0,
            True,
            False,
            self._info(
                distance=0.1,
                clearance=2.0,
                success=True,
                reason="goal_reached",
            ),
        )

    def close_with_result(self) -> FakeCleanup:
        self.events.append("uav-cleanup")
        self.events.append("scene-cleanup")
        self.closed = True
        if self.outcome == "unsafe_cleanup":
            return FakeCleanup(
                succeeded=False,
                scene_cleanup_deferred=True,
                scene_cleanup_deferred_reason="UAV state was not conclusively safe",
                actions=("uav-cleanup",),
            )
        return FakeCleanup()

    @staticmethod
    def _observation(valid: bool) -> np.ndarray:
        observation = np.zeros(83, dtype=np.float32)
        observation[3] = 0.2
        observation[10:82] = 0.5 if valid else 1.0
        observation[82] = 1.0 if valid else 0.0
        return observation

    @staticmethod
    def _info(
        *,
        distance: float,
        clearance: float,
        success: bool = False,
        sensor_failure: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "distance_to_goal": distance,
            "reward_clearance_m": clearance,
            "clearance_source": "current",
            "collision": False,
            "success": success,
            "sensor_failure": sensor_failure,
            "path_length_m": 0.25,
            "termination_reason": reason,
        }


class FakeEnvironmentFactory:
    def __init__(self, outcome: str, events: list[str] | None = None) -> None:
        self.outcome = outcome
        self.events = events if events is not None else []
        self.instance: FakeReferenceEnvironment | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> FakeReferenceEnvironment:
        self.instance = FakeReferenceEnvironment(
            *args,
            outcome=self.outcome,
            events=self.events,
            **kwargs,
        )
        return self.instance


@pytest.fixture(scope="module")
def config():
    return load_m13_6_config(CONFIG_PATH)


def _prepared(
    config: Any,
    *,
    controller: str = "direct",
    profile: str = "empty",
    seed: int = 0,
    controller_seed: int | None = None,
):
    return prepare_reference_run(
        config,
        repository_root=ROOT,
        controller_id=controller,
        course_profile=profile,
        base_seed=seed,
        controller_seed=controller_seed,
        authorizations=_authorizations(),
    )


def test_offline_configuration_validates_complete_matrix(config: Any) -> None:
    plans = validate_offline_configuration(config, repository_root=ROOT)
    assert len(plans) == 10
    assert sum(item["controller_id"] == "oracle" for item in plans) == 4
    assert all("reference_path" not in item for item in plans)


@pytest.mark.parametrize(
    "missing",
    tuple(M13_6LiveAuthorizations.__dataclass_fields__),
)
def test_each_authorization_fails_pre_client(config: Any, missing: str) -> None:
    with pytest.raises(ValueError, match="authorization"):
        prepare_reference_run(
            config,
            repository_root=ROOT,
            controller_id="direct",
            course_profile="empty",
            base_seed=0,
            controller_seed=None,
            authorizations=_authorizations(**{missing: False}),
        )


def test_wrong_pairing_and_optional_hard_are_rejected(config: Any) -> None:
    with pytest.raises(ValueError, match="not declared"):
        prepare_reference_run(
            config,
            repository_root=ROOT,
            controller_id="direct",
            course_profile="medium",
            base_seed=2200,
            controller_seed=None,
            authorizations=_authorizations(),
        )
    with pytest.raises(ValueError, match="optional hard"):
        _prepared(config, controller="oracle", profile="hard", seed=3100)


def test_unsafe_report_directory_fails_before_live_import(
    config: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="reports must remain"):
        prepare_reference_run(
            config,
            repository_root=ROOT,
            controller_id="direct",
            course_profile="empty",
            base_seed=0,
            controller_seed=None,
            authorizations=_authorizations(),
            report_directory=tmp_path,
        )


def test_direct_goal_success_report_and_cleanup(config: Any) -> None:
    prepared = _prepared(config)
    events: list[str] = []
    factory = FakeEnvironmentFactory("goal", events)
    client = FakeClient()
    report = execute_reference_episode(
        prepared,
        client=client,
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=factory,
    )
    assert report.episode_success
    assert report.report_success
    assert not report.expected_baseline_failure
    assert report.validation_config_digest == config.configuration_digest
    assert all(report.authorization_evidence.values())
    assert report.step_trace[0]["controller_action_generated"]
    assert events[-2:] == ["uav-cleanup", "scene-cleanup"]
    assert client.reset_calls == 0


def test_direct_medium_clearance_abort_happens_before_action(config: Any) -> None:
    prepared = _prepared(config, controller="direct", profile="medium", seed=2100)

    class AbortEnvironment(FakeReferenceEnvironment):
        def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            observation, info = super().reset(**kwargs)
            info["reward_clearance_m"] = 0.9
            return observation, info

    def factory(*args: Any, **kwargs: Any) -> AbortEnvironment:
        return AbortEnvironment(*args, **kwargs)

    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=factory,
    )
    assert report.clearance_abort
    assert report.step_count == 0
    assert report.expected_baseline_failure
    assert report.report_success


def test_persistent_sensor_failure_remains_truncation(config: Any) -> None:
    prepared = _prepared(config)
    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=FakeEnvironmentFactory("sensor"),
    )
    assert report.truncated
    assert report.termination_reason == "sensor_failure"
    assert not report.report_success


def test_controller_run_exception_still_cleans_and_writes_failed_report(
    config: Any, tmp_path: Path
) -> None:
    prepared = _prepared(config)
    events: list[str] = []
    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=FakeEnvironmentFactory("exception", events),
    )
    assert not report.report_success
    assert report.errors
    assert events[-2:] == ["uav-cleanup", "scene-cleanup"]
    output = tmp_path / "failed.json"
    save_reference_episode_report(report, output, repository_root=ROOT)
    saved = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(saved).lower()
    assert "point_cloud" not in serialized
    assert "reference_path" not in serialized
    assert saved["report_success"] is False


def test_interruption_is_recorded_after_cleanup(config: Any) -> None:
    prepared = _prepared(config)
    events: list[str] = []
    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=FakeEnvironmentFactory("interrupt", events),
    )
    assert report.interrupted
    assert not report.report_success
    assert events[-2:] == ["uav-cleanup", "scene-cleanup"]


def test_uncertain_uav_cleanup_defers_scene_acceptance(config: Any) -> None:
    prepared = _prepared(config)
    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=FakeEnvironmentFactory("unsafe_cleanup"),
    )
    assert report.episode_success
    assert not report.report_success
    assert report.cleanup_evidence is not None
    assert report.cleanup_evidence["scene_cleanup_deferred"] is True


def test_suite_summary_requires_exact_successful_matrix(
    config: Any, tmp_path: Path
) -> None:
    paths: list[Path] = []
    for index, entry in enumerate(config.required_matrix):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "controller_id": entry.controller_id.value,
                    "course_profile": entry.course_profile,
                    "base_seed": entry.base_seed,
                    "controller_seed": entry.controller_seed,
                    "report_success": True,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    summary = summarize_episode_reports(config, paths)
    assert summary.report_success
    paths.pop()
    failed = summarize_episode_reports(config, paths)
    assert not failed.report_success
    assert failed.missing_identities


def test_cli_module_import_and_help_do_not_import_airsim(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "airsim", None)
    spec = importlib.util.spec_from_file_location("m13_6_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["run", "--help"])
    assert exc.value.code == 0


def test_transient_invalid_scan_uses_zero_action_hold(config: Any) -> None:
    prepared = _prepared(config)

    class TransientEnvironment(FakeReferenceEnvironment):
        def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            observation, info = super().reset(**kwargs)
            observation[82] = 0.0
            return observation, info

    holder: list[TransientEnvironment] = []

    def factory(*args: Any, **kwargs: Any) -> TransientEnvironment:
        instance = TransientEnvironment(*args, **kwargs)
        holder.append(instance)
        return instance

    report = execute_reference_episode(
        prepared,
        client=FakeClient(),
        client_module=ModuleType("fake_airsim"),
        repository_root=ROOT,
        environment_factory=factory,
    )
    assert report.step_trace[0]["controller_action_generated"] is False
    assert report.step_trace[0]["action"] == [0.0, 0.0, 0.0]
    assert holder[0].step_calls == 1
