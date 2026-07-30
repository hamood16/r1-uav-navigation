from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from r1_uav_nav.evaluation.m13_6_course_validation import (
    load_m13_6_config,
    summarize_episode_reports,
)
from r1_uav_nav.training.curriculum import CurriculumState
from r1_uav_nav.training.curriculum_live_pilot import (
    DEFAULT_LIVE_PILOT_CONFIG_PATH,
    GitEvidence,
    LivePilotAuthorizations,
    PilotKind,
    load_live_pilot_config,
    prepare_live_pilot,
)
from r1_uav_nav.training.curriculum_live_runtime import (
    PilotRuntimeEnvironment,
    execute_live_pilot,
)
from r1_uav_nav.training.long_run_state import (
    canonical_digest,
    validate_checkpoint_bundle,
    validate_resume_request,
)

ROOT = Path(__file__).resolve().parents[1]


def _authorizations(**changes: bool) -> LivePilotAuthorizations:
    values = {name: True for name in LivePilotAuthorizations.__dataclass_fields__}
    values.update(changes)
    return LivePilotAuthorizations(**values)


def _prepare_repo(root: Path) -> None:
    shutil.copytree(ROOT / "configs", root / "configs")
    for relative in (
        "results/reports/m13/reference_controllers",
        "results/reports/m13/lidar",
        "results/reports/m13/curriculum",
        "results/trained_models/m13_8_static_curriculum",
        "results/logs/m13_8_static_curriculum",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _m13_6_reports(root: Path) -> tuple[list[Path], str]:
    config = load_m13_6_config(
        root / "configs/evaluation/m13_6_reference_controllers.yaml"
    )
    reports: list[Path] = []
    digests: list[str] = []
    for index, entry in enumerate(config.required_matrix):
        path = (
            root
            / "results/reports/m13/reference_controllers"
            / f"accepted-{index}.json"
        )
        digest = _write_json(
            path,
            {
                "schema_version": 1,
                "controller_id": entry.controller_id.value,
                "course_profile": entry.course_profile,
                "base_seed": entry.base_seed,
                "controller_seed": entry.controller_seed,
                "report_success": True,
                "run_id": f"accepted-{index}",
                "authorization_evidence": {"allow_live_rpc": True},
                "cleanup_evidence": {"succeeded": True},
                "broad_reset_evidence": {
                    "guard_installed": True,
                    "reset_attempted": False,
                },
            },
        )
        reports.append(path)
        digests.append(digest)
    summary = summarize_episode_reports(config, reports)
    suite_digest = canonical_digest(
        {
            "summary": asdict(summary),
            "report_digests": sorted(digests),
        }
    )
    return reports, suite_digest


def _prerequisites(root: Path) -> tuple[Path, str, Path, str]:
    survey = root / "results/reports/m13/preflight.json"
    grounded = root / "results/reports/m13/lidar/grounded.json"
    survey_digest = _write_json(survey, {"schema_version": 1, "success": True})
    grounded_digest = _write_json(
        grounded,
        {"schema_version": 1, "success": True},
    )
    return survey, survey_digest, grounded, grounded_digest


def _prepared(root: Path, *, stage_id: str = "stage-0"):
    config = load_live_pilot_config(root / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    reports, suite_digest = _m13_6_reports(root)
    survey, survey_digest, grounded, grounded_digest = _prerequisites(root)
    stage = config.stage(stage_id)
    return prepare_live_pilot(
        config,
        repository_root=root,
        stage_id=stage_id,
        profile_id=stage.profile_id,
        base_seed=stage.base_seed,
        pilot_kind=PilotKind.SMOKE,
        requested_timesteps=100,
        authorizations=_authorizations(),
        m13_6_report_paths=reports,
        expected_m13_6_suite_digest=suite_digest,
        preflight_survey_path=survey,
        expected_preflight_survey_digest=survey_digest,
        grounded_lidar_path=grounded,
        expected_grounded_lidar_digest=grounded_digest,
        git_inspector=lambda _: GitEvidence("test", "a" * 40, True),
        ignore_checker=lambda _root, _path: True,
        run_id=f"{stage_id}-smoke",
    )


def test_config_locks_stage_identity_and_budgets() -> None:
    config = load_live_pilot_config(ROOT / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    assert [stage.stage_id for stage in config.stages] == ["stage-0", "stage-1"]
    assert config.stage("stage-0").profile_id == "curriculum-empty-train"
    assert config.stage("stage-1").base_seed == 21000
    assert config.stage("stage-0").smoke.cumulative_cap_steps == 500
    assert config.stage("stage-1").pilot.minimum_requested_steps == 2000


def test_missing_authorizations_fail_before_evidence_loading(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    config = load_live_pilot_config(tmp_path / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    with pytest.raises(ValueError, match="allow_live_rpc"):
        prepare_live_pilot(
            config,
            repository_root=tmp_path,
            stage_id="stage-0",
            profile_id="curriculum-empty-train",
            base_seed=20000,
            pilot_kind="smoke",
            requested_timesteps=100,
            authorizations=_authorizations(allow_live_rpc=False),
            m13_6_report_paths=[],
            expected_m13_6_suite_digest="0" * 64,
            preflight_survey_path=Path("missing"),
            expected_preflight_survey_digest="0" * 64,
            grounded_lidar_path=Path("missing"),
            expected_grounded_lidar_digest="0" * 64,
        )


def test_stage_two_and_excessive_budget_fail_before_runtime(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    config = load_live_pilot_config(tmp_path / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    common = dict(
        config=config,
        repository_root=tmp_path,
        profile_id="curriculum-empty-train",
        base_seed=20000,
        pilot_kind="smoke",
        requested_timesteps=100,
        authorizations=_authorizations(),
        m13_6_report_paths=[],
        expected_m13_6_suite_digest="0" * 64,
        preflight_survey_path=Path("missing"),
        expected_preflight_survey_digest="0" * 64,
        grounded_lidar_path=Path("missing"),
        expected_grounded_lidar_digest="0" * 64,
    )
    with pytest.raises(ValueError, match="stage-0 and stage-1"):
        prepare_live_pilot(stage_id="stage-2", **common)
    with pytest.raises(ValueError, match="exceeds remaining cumulative cap"):
        prepare_live_pilot(
            stage_id="stage-0",
            requested_timesteps=501,
            **{
                key: value
                for key, value in common.items()
                if key != "requested_timesteps"
            },
        )


def test_names_reserved_seed_and_dirty_worktree_fail_early(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    config = load_live_pilot_config(tmp_path / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    common = dict(
        config=config,
        repository_root=tmp_path,
        stage_id="stage-0",
        profile_id="curriculum-empty-train",
        base_seed=20000,
        pilot_kind="smoke",
        requested_timesteps=100,
        authorizations=_authorizations(),
        m13_6_report_paths=[],
        expected_m13_6_suite_digest="0" * 64,
        preflight_survey_path=Path("missing"),
        expected_preflight_survey_digest="0" * 64,
        grounded_lidar_path=Path("missing"),
        expected_grounded_lidar_digest="0" * 64,
        git_inspector=lambda _: GitEvidence("test", "a" * 40, True),
    )
    with pytest.raises(ValueError, match="vehicle name"):
        prepare_live_pilot(vehicle_name="WrongVehicle", **common)
    with pytest.raises(ValueError, match="final-held-out"):
        prepare_live_pilot(
            base_seed=9100,
            **{key: value for key, value in common.items() if key != "base_seed"},
        )
    with pytest.raises(ValueError, match="worktree must be clean"):
        prepare_live_pilot(
            **{
                **common,
                "git_inspector": lambda _: GitEvidence("test", "a" * 40, False),
            }
        )


def test_complete_offline_preflight_builds_locked_stage_zero(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    prepared = _prepared(tmp_path)
    assert prepared.stage.stage_id == "stage-0"
    assert prepared.course.result.profile_id == "curriculum-empty-train"
    assert prepared.course.result.obstacle_count == 0
    assert prepared.remaining_timesteps == 500
    assert prepared.metadata.promotion_claimed is False
    assert prepared.long_run_config.execution_metadata is not None


def test_stage_one_requires_accepted_stage_zero_report(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    config = load_live_pilot_config(tmp_path / DEFAULT_LIVE_PILOT_CONFIG_PATH)
    reports, suite_digest = _m13_6_reports(tmp_path)
    survey, survey_digest, grounded, grounded_digest = _prerequisites(tmp_path)
    with pytest.raises(ValueError, match="requires an accepted Stage 0"):
        prepare_live_pilot(
            config,
            repository_root=tmp_path,
            stage_id="stage-1",
            profile_id="curriculum-blocker-train",
            base_seed=21000,
            pilot_kind="smoke",
            requested_timesteps=100,
            authorizations=_authorizations(),
            m13_6_report_paths=reports,
            expected_m13_6_suite_digest=suite_digest,
            preflight_survey_path=survey,
            expected_preflight_survey_digest=survey_digest,
            grounded_lidar_path=grounded,
            expected_grounded_lidar_digest=grounded_digest,
        )
    stage_0 = tmp_path / "results/reports/m13/curriculum/stage-0.json"
    stage_0_digest = _write_json(
        stage_0,
        {
            "schema_version": 1,
            "stage_id": "stage-0",
            "profile_id": "curriculum-empty-train",
            "base_seed": 20000,
            "report_success": True,
            "checkpoint_success": True,
            "cleanup_success": True,
            "pilot_only": True,
            "promotion_claimed": False,
            "learned_avoidance_claimed": False,
            "final_policy_claimed": False,
            "final_generalization_claimed": False,
            "real_world_claimed": False,
        },
    )
    prepared = prepare_live_pilot(
        config,
        repository_root=tmp_path,
        stage_id="stage-1",
        profile_id="curriculum-blocker-train",
        base_seed=21000,
        pilot_kind="smoke",
        requested_timesteps=100,
        authorizations=_authorizations(),
        m13_6_report_paths=reports,
        expected_m13_6_suite_digest=suite_digest,
        preflight_survey_path=survey,
        expected_preflight_survey_digest=survey_digest,
        grounded_lidar_path=grounded,
        expected_grounded_lidar_digest=grounded_digest,
        stage_0_report_path=stage_0,
        expected_stage_0_report_digest=stage_0_digest,
        git_inspector=lambda _: GitEvidence("test", "a" * 40, True),
        ignore_checker=lambda _root, _path: True,
    )
    assert prepared.course.result.obstacle_count == 1
    assert prepared.course.result.path_result.direct_line_clear is False
    assert prepared.curriculum_state.stage_id == "stage-1"
    assert prepared.curriculum_state.completed_stage_ids == ()
    assert prepared.metadata.learned_avoidance_claimed is False


@dataclass(frozen=True)
class _Cleanup:
    succeeded: bool = True
    scene_cleanup_deferred: bool = False
    scene_cleanup_deferred_reason: str | None = None


class _FakePilotEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (np.full(10, -1.0, dtype=np.float32), np.zeros(73, np.float32))
            ),
            high=np.ones(83, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.step_index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        observation = np.zeros(83, dtype=np.float32)
        observation[10:] = 1.0
        return observation, _info(None, 0)

    def step(self, action):
        self.step_index += 1
        observation = np.zeros(83, dtype=np.float32)
        observation[0] = min(self.step_index / 20.0, 1.0)
        observation[10:] = 1.0
        done = self.step_index >= 5
        return (
            observation,
            0.1,
            done,
            False,
            _info(
                "goal_reached" if done else None,
                self.step_index,
            ),
        )


class _InterruptedPilotEnv(_FakePilotEnv):
    def step(self, action):
        raise KeyboardInterrupt


def _info(reason: str | None, step: int) -> dict[str, Any]:
    return {
        "profile_id": "curriculum-empty-train",
        "base_seed": 20000,
        "accepted_candidate_seed": 20000,
        "scene_digest": "a" * 64,
        "step_count": step,
        "distance_to_goal": max(0.0, 5.0 - step),
        "path_length_m": float(step),
        "reward_clearance_m": 5.0,
        "collision": False,
        "ground_clearance_violation": False,
        "workspace_violation": False,
        "sensor_failure": False,
        "success": reason == "goal_reached",
        "termination_reason": reason,
    }


def test_fake_runtime_saves_initial_and_final_safe_bundles(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    prepared = _prepared(tmp_path)
    prepared = replace(
        prepared,
        long_run_config=replace(
            prepared.long_run_config,
            policy_hidden_layers=(8, 8),
            learning_starts=1000,
            batch_size=2,
            buffer_size=512,
            checkpoint_interval_steps=1000,
        ),
    )
    result = execute_live_pilot(
        prepared,
        repository_root=tmp_path,
        environment_factory=lambda _: PilotRuntimeEnvironment(
            _FakePilotEnv(),
            lambda: _Cleanup(),
            lambda: {"guard_installed": True, "reset_attempted": False},
        ),
    )
    assert result.report.report_success
    assert result.report.actual_segment_timesteps == 100
    assert result.report.initial_checkpoint is not None
    assert result.report.final_checkpoint is not None
    final_directory = tmp_path / result.report.final_checkpoint["relative_directory"]
    safe = validate_checkpoint_bundle(final_directory)
    assert safe.run_state.curriculum_state["stage_id"] == "stage-0"
    assert safe.run_state.curriculum_state["stage_completed_timesteps"] == 100
    resume_plan = validate_resume_request(
        resume_latest=prepared.run_root,
        expected_compatibility_digest=prepared.long_run_config.compatibility_digest,
    )
    resumed = replace(
        prepared,
        resume_plan=resume_plan,
        curriculum_state=CurriculumState.from_mapping(safe.run_state.curriculum_state),
        completed_pilot_timesteps=100,
        remaining_timesteps=400,
    )
    resumed_result = execute_live_pilot(
        resumed,
        repository_root=tmp_path,
        environment_factory=lambda _: PilotRuntimeEnvironment(
            _FakePilotEnv(),
            lambda: _Cleanup(),
            lambda: {"guard_installed": True, "reset_attempted": False},
        ),
    )
    assert resumed_result.report.actual_segment_timesteps == 100
    assert resumed_result.report.cumulative_timesteps == 200
    assert resumed_result.report.replay_size is not None
    assert resumed_result.report.replay_size >= 100
    serialized = result.report_path.read_text(encoding="utf-8").lower()
    assert "raw_point_cloud" not in serialized
    assert '"promotion_claimed": false' in serialized

    interrupted_prepared = replace(
        prepared,
        run_id="interrupted-smoke",
        run_root=(
            tmp_path
            / "results/trained_models/m13_8_static_curriculum/interrupted-smoke"
        ),
        report_directory=(
            tmp_path / "results/reports/m13/curriculum/interrupted-smoke"
        ),
    )
    interrupted = execute_live_pilot(
        interrupted_prepared,
        repository_root=tmp_path,
        environment_factory=lambda _: PilotRuntimeEnvironment(
            _InterruptedPilotEnv(),
            lambda: _Cleanup(succeeded=False, scene_cleanup_deferred=True),
            lambda: {"guard_installed": True, "reset_attempted": False},
        ),
    )
    assert interrupted.report.interrupted
    assert not interrupted.report.report_success
    assert interrupted.report.initial_checkpoint is not None
    assert interrupted.report.final_checkpoint is None
    assert interrupted.report.latest_safe_checkpoint is not None
    assert interrupted.report.cleanup_success is False
