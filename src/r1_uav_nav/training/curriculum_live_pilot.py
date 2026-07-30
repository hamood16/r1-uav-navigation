"""Pure preparation and evidence contracts for M13.8 supervised live pilots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from r1_uav_nav.evaluation.m13_6_course_validation import (
    M13_6_EPISODE_REPORT_SCHEMA_VERSION,
    M13_6ReferenceControllerConfig,
    load_m13_6_config,
    summarize_episode_reports,
)
from r1_uav_nav.sim.static_course import (
    ValidatedCourse,
    generate_solvable_course,
    load_course_suite_config,
    require_solvable_course,
)
from r1_uav_nav.training.curriculum import (
    M13_9_RESERVED_FINAL_TEST_SEEDS,
    CurriculumConfig,
    CurriculumState,
    enable_curriculum_mode,
    load_curriculum_config,
    validate_curriculum_configuration,
)
from r1_uav_nav.training.long_run_state import (
    ResumeMode,
    ResumePlan,
    canonical_digest,
    load_resolved_config,
    validate_checkpoint_bundle,
    validate_resume_request,
)
from r1_uav_nav.training.long_run_training import (
    LongRunOutputConfig,
    LongRunTD3Config,
    load_long_run_td3_config,
)

LIVE_PILOT_CONFIG_SCHEMA_VERSION = 1
LIVE_PILOT_METADATA_SCHEMA_VERSION = 1
M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION = 1
M13_8_LIVE_PILOT_SUMMARY_SCHEMA_VERSION = 1
DEFAULT_LIVE_PILOT_CONFIG_PATH = Path(
    "configs/training/m13_8_supervised_live_pilot.yaml"
)
EXPECTED_VEHICLE_NAME = "SimpleFlight"
EXPECTED_LIDAR_NAME = "LidarSensor1"
LOCKED_M13_8_FEASIBILITY_DIGEST = (
    "7ab64205ccb7fd055d2ed0b5f42528d1e7be0772ed46224197427b96c6baa1b6"
)


class PilotKind(str, Enum):
    SMOKE = "smoke"
    PILOT = "pilot"


@dataclass(frozen=True)
class PilotBudget:
    minimum_requested_steps: int
    cumulative_cap_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_requested_steps, int)
            or isinstance(self.minimum_requested_steps, bool)
            or self.minimum_requested_steps <= 0
        ):
            raise ValueError("minimum_requested_steps must be a positive integer")
        if (
            not isinstance(self.cumulative_cap_steps, int)
            or isinstance(self.cumulative_cap_steps, bool)
            or self.cumulative_cap_steps < self.minimum_requested_steps
        ):
            raise ValueError("cumulative cap must cover the minimum request")


@dataclass(frozen=True)
class LivePilotStage:
    stage_id: str
    profile_id: str
    base_seed: int
    obstacle_count: int
    smoke: PilotBudget
    pilot: PilotBudget
    requires_stage_0_report: bool

    def __post_init__(self) -> None:
        if self.stage_id not in {"stage-0", "stage-1"}:
            raise ValueError("live pilot stage must be stage-0 or stage-1")
        if not self.profile_id.strip():
            raise ValueError("live pilot profile must not be empty")
        if self.base_seed in M13_9_RESERVED_FINAL_TEST_SEEDS:
            raise ValueError("M13.9 final-held-out seed is not allowed")
        if self.obstacle_count not in {0, 1}:
            raise ValueError("Phase B supports only zero or one obstacle")

    def budget(self, kind: PilotKind) -> PilotBudget:
        return self.smoke if kind is PilotKind.SMOKE else self.pilot


@dataclass(frozen=True)
class LivePilotOutputConfig:
    model_root: str
    log_root: str
    report_root: str
    m13_6_report_root: str
    prerequisite_report_root: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            path = Path(value)
            if (
                not value
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "results"
            ):
                raise ValueError(f"{name} must be repository-relative under results/")


@dataclass(frozen=True)
class LivePilotConfig:
    schema_version: int
    curriculum_config_path: str
    m13_6_config_path: str
    obstacle_environment_config_path: str
    vehicle_name: str
    lidar_name: str
    client_module: str
    locked_feasibility_digest: str
    outputs: LivePilotOutputConfig
    stages: tuple[LivePilotStage, ...]

    def __post_init__(self) -> None:
        if self.schema_version != LIVE_PILOT_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported M13.8 live-pilot configuration schema")
        for name in (
            "curriculum_config_path",
            "m13_6_config_path",
            "obstacle_environment_config_path",
        ):
            _repository_relative(getattr(self, name), name)
        if self.vehicle_name != EXPECTED_VEHICLE_NAME:
            raise ValueError("live pilot requires vehicle SimpleFlight")
        if self.lidar_name != EXPECTED_LIDAR_NAME:
            raise ValueError("live pilot requires LiDAR LidarSensor1")
        if not self.client_module.strip():
            raise ValueError("client_module must not be empty")
        if self.locked_feasibility_digest != LOCKED_M13_8_FEASIBILITY_DIGEST:
            raise ValueError("M13.8 locked feasibility digest does not match")
        if tuple(item.stage_id for item in self.stages) != ("stage-0", "stage-1"):
            raise ValueError("live pilot must define exactly stage-0 and stage-1")
        expected = (
            ("stage-0", "curriculum-empty-train", 20000, 0, False),
            ("stage-1", "curriculum-blocker-train", 21000, 1, True),
        )
        actual = tuple(
            (
                item.stage_id,
                item.profile_id,
                item.base_seed,
                item.obstacle_count,
                item.requires_stage_0_report,
            )
            for item in self.stages
        )
        if actual != expected:
            raise ValueError("live pilot stage/course identities are locked")
        for stage in self.stages:
            if stage.smoke != PilotBudget(100, 500):
                raise ValueError("smoke budget must be 100-500 cumulative steps")
            if stage.pilot != PilotBudget(2000, 5000):
                raise ValueError("pilot budget must be 2000-5000 cumulative steps")

    @property
    def config_digest(self) -> str:
        return canonical_digest(asdict(self))

    def stage(self, stage_id: str) -> LivePilotStage:
        matches = [stage for stage in self.stages if stage.stage_id == stage_id]
        if len(matches) != 1:
            raise ValueError(
                "live pilots permit only stage-0 and stage-1; " f"received {stage_id!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class LivePilotAuthorizations:
    allow_live_rpc: bool = False
    allow_scene_mutation: bool = False
    allow_debug_markers: bool = False
    allow_marker_flush: bool = False
    allow_flight: bool = False
    allow_start_positioning: bool = False
    allow_training: bool = False
    confirm_results_root_ignored: bool = False
    confirm_m13_6_supervised_evidence_accepted: bool = False
    confirm_preflight_survey_passed: bool = False
    confirm_grounded_lidar_passed: bool = False
    confirm_clear_airspace: bool = False
    confirm_scene_area_clear: bool = False
    confirm_no_visible_collision: bool = False
    confirm_manual_operator_present: bool = False
    confirm_named_cleanup_required: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in asdict(self).values()):
            raise ValueError("live-pilot authorizations must be boolean")

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value is not True)


@dataclass(frozen=True)
class GitEvidence:
    branch: str
    commit: str
    tracked_worktree_clean: bool

    def __post_init__(self) -> None:
        if not self.branch.strip():
            raise ValueError("Git branch evidence must not be empty")
        if len(self.commit) < 7:
            raise ValueError("Git commit evidence is malformed")


@dataclass(frozen=True)
class AcceptedReportEvidence:
    relative_path: str
    content_digest: str
    schema_version: int
    mode: str


@dataclass(frozen=True)
class M13_6SuiteEvidence:
    suite_digest: str
    report_digests: tuple[str, ...]
    report_paths: tuple[str, ...]
    required_identities: tuple[tuple[str, str, int, int | None], ...]


@dataclass(frozen=True)
class LivePilotMetadata:
    schema_version: int
    stage_id: str
    profile_id: str
    base_seed: int
    accepted_candidate_seed: int
    pilot_kind: PilotKind
    cumulative_cap_steps: int
    scene_digest: str
    occupancy_digest: str
    solvability_digest: str
    m13_6_suite_digest: str
    observation_contract: dict[str, Any]
    action_contract: dict[str, Any]
    pilot_only: bool = True
    promotion_claimed: bool = False
    learned_avoidance_claimed: bool = False
    final_policy_claimed: bool = False
    final_generalization_claimed: bool = False
    real_world_claimed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LIVE_PILOT_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported live-pilot metadata schema")
        if not self.pilot_only or any(
            (
                self.promotion_claimed,
                self.learned_avoidance_claimed,
                self.final_policy_claimed,
                self.final_generalization_claimed,
                self.real_world_claimed,
            )
        ):
            raise ValueError("live-pilot claim fields must remain false")
        if self.base_seed in M13_9_RESERVED_FINAL_TEST_SEEDS:
            raise ValueError("live-pilot metadata contains a final-held-out seed")
        for name in ("scene_digest", "occupancy_digest", "solvability_digest"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be SHA-256")

    @property
    def metadata_digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class PreparedLivePilot:
    repository_root: Path
    config: LivePilotConfig
    curriculum: CurriculumConfig
    course: ValidatedCourse
    stage: LivePilotStage
    pilot_kind: PilotKind
    requested_timesteps: int
    completed_pilot_timesteps: int
    remaining_timesteps: int
    run_id: str
    resume_plan: ResumePlan
    curriculum_state: CurriculumState
    long_run_config: LongRunTD3Config
    metadata: LivePilotMetadata
    m13_6_evidence: M13_6SuiteEvidence
    preflight_survey: AcceptedReportEvidence
    grounded_lidar: AcceptedReportEvidence
    stage_0_report: AcceptedReportEvidence | None
    authorizations: LivePilotAuthorizations
    git_evidence: GitEvidence
    report_directory: Path
    run_root: Path


@dataclass(frozen=True)
class LivePilotReport:
    schema_version: int
    run_id: str
    started_at: str
    completed_at: str
    command_mode: str
    stage_id: str
    pilot_kind: str
    profile_id: str
    base_seed: int
    accepted_candidate_seed: int
    requested_timesteps: int
    actual_segment_timesteps: int
    cumulative_timesteps: int
    cumulative_cap_timesteps: int
    curriculum_config_digest: str
    pilot_config_digest: str
    pilot_metadata_digest: str
    m13_6_suite_digest: str
    scene_digest: str
    occupancy_digest: str
    solvability_digest: str
    initial_checkpoint: dict[str, Any] | None
    final_checkpoint: dict[str, Any] | None
    latest_safe_checkpoint: dict[str, Any] | None
    replay_size: int | None
    curriculum_state: dict[str, Any]
    episode_summaries: tuple[dict[str, Any], ...]
    route_shape_summaries: tuple[dict[str, Any], ...]
    broad_reset_evidence: dict[str, Any]
    cleanup_evidence: dict[str, Any] | None
    preflight_success: bool
    infrastructure_success: bool
    safety_incident_free: bool
    checkpoint_success: bool
    cleanup_success: bool
    report_success: bool
    interrupted: bool
    pilot_only: bool
    promotion_claimed: bool
    learned_avoidance_claimed: bool
    final_policy_claimed: bool
    final_generalization_claimed: bool
    real_world_claimed: bool
    acceptance_checks: dict[str, bool]
    acceptance_failures: tuple[str, ...]
    errors: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported M13.8 live-pilot report schema")
        if not self.pilot_only or any(
            (
                self.promotion_claimed,
                self.learned_avoidance_claimed,
                self.final_policy_claimed,
                self.final_generalization_claimed,
                self.real_world_claimed,
            )
        ):
            raise ValueError("live-pilot reports cannot make performance claims")
        expected_success = all(self.acceptance_checks.values())
        if self.report_success != expected_success:
            raise ValueError("report_success must match all acceptance checks")
        expected_failures = tuple(
            name for name, passed in self.acceptance_checks.items() if not passed
        )
        if self.acceptance_failures != expected_failures:
            raise ValueError("acceptance_failures do not match checks")
        assert_sanitized_report(asdict(self))


def load_live_pilot_config(path: str | Path) -> LivePilotConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "live-pilot configuration")
    _keys(
        root,
        {
            "schema_version",
            "curriculum_config_path",
            "m13_6_config_path",
            "obstacle_environment_config_path",
            "vehicle_name",
            "lidar_name",
            "client_module",
            "locked_feasibility_digest",
            "outputs",
            "stages",
        },
        "live-pilot configuration",
    )
    output = _mapping(root["outputs"], "outputs")
    _keys(
        output,
        {
            "model_root",
            "log_root",
            "report_root",
            "m13_6_report_root",
            "prerequisite_report_root",
        },
        "outputs",
    )
    stages = _list(root["stages"], "stages")
    return LivePilotConfig(
        schema_version=_integer(root["schema_version"], "schema_version"),
        curriculum_config_path=_string(
            root["curriculum_config_path"], "curriculum_config_path"
        ),
        m13_6_config_path=_string(root["m13_6_config_path"], "m13_6_config_path"),
        obstacle_environment_config_path=_string(
            root["obstacle_environment_config_path"],
            "obstacle_environment_config_path",
        ),
        vehicle_name=_string(root["vehicle_name"], "vehicle_name"),
        lidar_name=_string(root["lidar_name"], "lidar_name"),
        client_module=_string(root["client_module"], "client_module"),
        locked_feasibility_digest=_string(
            root["locked_feasibility_digest"], "locked_feasibility_digest"
        ),
        outputs=LivePilotOutputConfig(
            **{key: _string(value, key) for key, value in output.items()}
        ),
        stages=tuple(_stage(item) for item in stages),
    )


def prepare_live_pilot(
    config: LivePilotConfig,
    *,
    repository_root: Path,
    stage_id: str,
    profile_id: str,
    base_seed: int,
    pilot_kind: str | PilotKind,
    requested_timesteps: int,
    authorizations: LivePilotAuthorizations,
    m13_6_report_paths: Sequence[Path],
    expected_m13_6_suite_digest: str,
    preflight_survey_path: Path,
    expected_preflight_survey_digest: str,
    grounded_lidar_path: Path,
    expected_grounded_lidar_digest: str,
    stage_0_report_path: Path | None = None,
    expected_stage_0_report_digest: str | None = None,
    vehicle_name: str = EXPECTED_VEHICLE_NAME,
    lidar_name: str = EXPECTED_LIDAR_NAME,
    run_id: str | None = None,
    resume_checkpoint: Path | None = None,
    resume_replay_buffer: Path | None = None,
    resume_run_state: Path | None = None,
    resume_latest: Path | None = None,
    reset_num_timesteps: bool = False,
    allow_partial_resume: bool = False,
    report_directory: Path | None = None,
    git_inspector: Callable[[Path], GitEvidence] | None = None,
    ignore_checker: Callable[[Path, Path], bool] | None = None,
) -> PreparedLivePilot:
    """Validate a complete live plan without importing or mutating a simulator."""
    root = repository_root.resolve()
    missing = authorizations.missing()
    if missing:
        raise ValueError(
            "live pilot lacks required authorization: " + ", ".join(missing)
        )
    if vehicle_name != config.vehicle_name:
        raise ValueError("vehicle name does not match live-pilot configuration")
    if lidar_name != config.lidar_name:
        raise ValueError("LiDAR name does not match live-pilot configuration")

    stage = config.stage(stage_id)
    if base_seed in M13_9_RESERVED_FINAL_TEST_SEEDS:
        raise ValueError("M13.9 final-held-out seed is forbidden")
    if (profile_id, base_seed) != (stage.profile_id, stage.base_seed):
        raise ValueError(
            f"{stage.stage_id} requires {stage.profile_id} seed {stage.base_seed}"
        )
    kind = PilotKind(pilot_kind)
    budget = stage.budget(kind)
    if (
        not isinstance(requested_timesteps, int)
        or isinstance(requested_timesteps, bool)
        or requested_timesteps < budget.minimum_requested_steps
    ):
        raise ValueError(
            f"{kind.value} requests must be at least "
            f"{budget.minimum_requested_steps} steps"
        )
    if requested_timesteps > budget.cumulative_cap_steps:
        raise ValueError(
            "requested segment exceeds remaining cumulative cap "
            f"({budget.cumulative_cap_steps} steps)"
        )
    if stage.requires_stage_0_report and (
        stage_0_report_path is None or not expected_stage_0_report_digest
    ):
        raise ValueError("stage-1 requires an accepted Stage 0 pilot report and digest")
    git_evidence = (git_inspector or inspect_git)(root)
    if not git_evidence.tracked_worktree_clean:
        raise ValueError("tracked Git worktree must be clean for live execution")

    curriculum_path = _resolve(root, config.curriculum_config_path)
    curriculum = load_curriculum_config(curriculum_path)
    validation = validate_curriculum_configuration(
        curriculum,
        repository_root=root,
        verify_courses=True,
    )
    if validation["feasibility_digest"] != config.locked_feasibility_digest:
        raise ValueError("locked M13.8 feasibility digest changed")

    course_ref = next(
        (
            item
            for item in curriculum.stage(stage_id).training_courses
            if (item.profile_id, item.base_seed) == (profile_id, base_seed)
        ),
        None,
    )
    if course_ref is None or course_ref.suite_id != "m13_8":
        raise ValueError("live-pilot course is not the locked M13.8 training course")
    suite = load_course_suite_config(
        _resolve(root, curriculum.course_suites[course_ref.suite_id])
    )
    course = require_solvable_course(
        generate_solvable_course(
            suite,
            profile_id,
            base_seed,
            repository_root=root,
        )
    )
    result = course.result
    if result.obstacle_count != stage.obstacle_count:
        raise ValueError("live-pilot course obstacle count changed")

    m13_6_config = load_m13_6_config(_resolve(root, config.m13_6_config_path))
    m13_6_evidence = validate_m13_6_evidence(
        m13_6_config,
        m13_6_report_paths,
        repository_root=root,
        approved_root=_resolve(root, config.outputs.m13_6_report_root),
        expected_suite_digest=expected_m13_6_suite_digest,
    )
    preflight = _accepted_prerequisite(
        preflight_survey_path,
        expected_digest=expected_preflight_survey_digest,
        repository_root=root,
        approved_root=_resolve(root, config.outputs.prerequisite_report_root),
        mode="preflight-survey",
    )
    grounded = _accepted_prerequisite(
        grounded_lidar_path,
        expected_digest=expected_grounded_lidar_digest,
        repository_root=root,
        approved_root=_resolve(root, config.outputs.prerequisite_report_root),
        mode="grounded-lidar",
    )
    stage_0_evidence: AcceptedReportEvidence | None = None
    if stage.requires_stage_0_report:
        assert stage_0_report_path is not None
        assert expected_stage_0_report_digest is not None
        stage_0_evidence = validate_stage_0_pilot_report(
            stage_0_report_path,
            expected_digest=expected_stage_0_report_digest,
            repository_root=root,
            approved_root=_resolve(root, config.outputs.report_root),
        )

    metadata = LivePilotMetadata(
        schema_version=LIVE_PILOT_METADATA_SCHEMA_VERSION,
        stage_id=stage.stage_id,
        profile_id=result.profile_id,
        base_seed=result.base_seed,
        accepted_candidate_seed=result.accepted_candidate_seed,
        pilot_kind=kind,
        cumulative_cap_steps=budget.cumulative_cap_steps,
        scene_digest=result.scene_digest,
        occupancy_digest=result.occupancy_digest,
        solvability_digest=result.solvability_digest,
        m13_6_suite_digest=m13_6_evidence.suite_digest,
        observation_contract={"shape": [83], "dtype": "float32"},
        action_contract={
            "shape": [3],
            "dtype": "float32",
            "bounds": [-1.0, 1.0],
            "frame": "world_ned",
        },
    )
    long_run_source = load_long_run_td3_config(
        _resolve(root, curriculum.long_run_config_path)
    )
    output = LongRunOutputConfig(
        model_root=config.outputs.model_root,
        log_root=config.outputs.log_root,
        report_root=config.outputs.report_root,
    )
    effective_long_run = replace(
        enable_curriculum_mode(long_run_source, curriculum, stage_id=stage_id),
        experiment_name="m13_8_supervised_live_pilot",
        additional_timesteps=requested_timesteps,
        course_suite_path=curriculum.course_suites["m13_8"],
        obstacle_environment_config_path=config.obstacle_environment_config_path,
        training_profile_ids=(profile_id,),
        execution_metadata=_jsonable(asdict(metadata)),
        output=output,
    )
    resume_plan = validate_resume_request(
        resume_checkpoint=_optional_resolve(root, resume_checkpoint),
        resume_replay_buffer=_optional_resolve(root, resume_replay_buffer),
        resume_run_state=_optional_resolve(root, resume_run_state),
        resume_latest=_optional_resolve(root, resume_latest),
        reset_num_timesteps=reset_num_timesteps,
        allow_partial_resume=allow_partial_resume,
        expected_compatibility_digest=(
            None if allow_partial_resume else effective_long_run.compatibility_digest
        ),
    )
    if stage_id == "stage-1" and resume_plan.mode is ResumeMode.MODEL_ONLY_WARM_START:
        raise ValueError("Stage 1 model-only warm start is forbidden")

    completed = 0
    if resume_plan.mode is ResumeMode.FULL:
        assert resume_plan.run_state_path is not None
        safe = validate_checkpoint_bundle(
            resume_plan.run_state_path.parent,
            expected_compatibility_digest=effective_long_run.compatibility_digest,
        )
        source_config = load_resolved_config(
            resume_plan.run_state_path.parent / "resolved_config.json"
        )
        if source_config.get("execution_metadata") != _jsonable(asdict(metadata)):
            raise ValueError("resume pilot metadata does not match the requested run")
        source_state = CurriculumState.from_mapping(safe.run_state.curriculum_state)
        if source_state.stage_id != stage_id:
            raise ValueError("resume curriculum stage does not match pilot stage")
        completed = source_state.stage_completed_timesteps
        effective_run_id = safe.run_state.run_id
        state = source_state
    else:
        effective_run_id = run_id or (
            f"m13-8-{stage_id}-{kind.value}-{uuid.uuid4().hex[:10]}"
        )
        state = CurriculumState.initial(curriculum)
        if stage_id == "stage-1":
            state = replace(state, stage_id="stage-1", stage_index=1)

    remaining = budget.cumulative_cap_steps - completed
    if remaining <= 0:
        raise ValueError("pilot cumulative timestep cap has already been reached")
    if requested_timesteps > remaining:
        raise ValueError(
            f"requested segment exceeds remaining cumulative cap ({remaining} steps)"
        )
    if state.validation_history or state.completed_stage_ids:
        raise ValueError("live pilots must not contain promotion validation history")

    report_root = _resolve(root, config.outputs.report_root)
    model_root = _resolve(root, config.outputs.model_root)
    log_root = _resolve(root, config.outputs.log_root)
    checker = ignore_checker or is_git_ignored
    for generated_root in (report_root, model_root, log_root):
        _require_under(generated_root, root / "results", "generated output")
        if not checker(root, generated_root):
            raise ValueError(
                f"generated output root is not Git-ignored: {generated_root}"
            )

    requested_report_dir = (
        _resolve(root, report_directory)
        if report_directory is not None
        else report_root / effective_run_id
    )
    _require_under(requested_report_dir, report_root, "pilot report")
    run_root = model_root / effective_run_id
    return PreparedLivePilot(
        repository_root=root,
        config=config,
        curriculum=curriculum,
        course=course,
        stage=stage,
        pilot_kind=kind,
        requested_timesteps=requested_timesteps,
        completed_pilot_timesteps=completed,
        remaining_timesteps=remaining,
        run_id=effective_run_id,
        resume_plan=resume_plan,
        curriculum_state=state,
        long_run_config=effective_long_run,
        metadata=metadata,
        m13_6_evidence=m13_6_evidence,
        preflight_survey=preflight,
        grounded_lidar=grounded,
        stage_0_report=stage_0_evidence,
        authorizations=authorizations,
        git_evidence=git_evidence,
        report_directory=requested_report_dir,
        run_root=run_root,
    )


def validate_m13_6_evidence(
    config: M13_6ReferenceControllerConfig,
    report_paths: Sequence[Path],
    *,
    repository_root: Path,
    approved_root: Path,
    expected_suite_digest: str,
) -> M13_6SuiteEvidence:
    if not report_paths:
        identities = ", ".join(str(item.identity) for item in config.required_matrix)
        raise ValueError(f"M13.6 evidence is missing required reports: {identities}")
    resolved: list[Path] = []
    digests: list[str] = []
    for source in report_paths:
        path = _resolve(repository_root, source)
        _require_under(path, approved_root, "M13.6 report")
        payload = _read_mapping(path)
        if payload.get("schema_version") != M13_6_EPISODE_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported M13.6 report schema: {path.name}")
        _require_live_m13_6_report(payload, path.name)
        assert_sanitized_report(payload)
        resolved.append(path)
        digests.append(_sha256_file(path))
    summary = summarize_episode_reports(config, resolved)
    if not summary.report_success:
        details = []
        if summary.missing:
            details.append(f"missing={list(summary.missing)!r}")
        if summary.duplicates:
            details.append(f"duplicates={list(summary.duplicates)!r}")
        if summary.failed:
            details.append(f"failed={list(summary.failed)!r}")
        if summary.unexpected:
            details.append(f"unexpected={list(summary.unexpected)!r}")
        raise ValueError("M13.6 supervised matrix is incomplete: " + "; ".join(details))
    suite_payload = {
        "summary": _jsonable(asdict(summary)),
        "report_digests": sorted(digests),
    }
    suite_digest = canonical_digest(suite_payload)
    if suite_digest != expected_suite_digest:
        raise ValueError(
            "M13.6 suite digest mismatch: "
            f"expected {expected_suite_digest}, measured {suite_digest}"
        )
    return M13_6SuiteEvidence(
        suite_digest=suite_digest,
        report_digests=tuple(sorted(digests)),
        report_paths=tuple(_relative(path, repository_root) for path in resolved),
        required_identities=tuple(item.identity for item in config.required_matrix),
    )


def validate_stage_0_pilot_report(
    path: Path,
    *,
    expected_digest: str,
    repository_root: Path,
    approved_root: Path,
) -> AcceptedReportEvidence:
    source = _resolve(repository_root, path)
    _require_under(source, approved_root, "Stage 0 pilot report")
    payload = _read_mapping(source)
    digest = _sha256_file(source)
    if digest != expected_digest:
        raise ValueError("Stage 0 pilot report digest mismatch")
    checks = {
        "schema": payload.get("schema_version")
        == M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION,
        "stage": payload.get("stage_id") == "stage-0",
        "profile": payload.get("profile_id") == "curriculum-empty-train",
        "seed": payload.get("base_seed") == 20000,
        "report_success": payload.get("report_success") is True,
        "checkpoint_success": payload.get("checkpoint_success") is True,
        "cleanup_success": payload.get("cleanup_success") is True,
        "pilot_only": payload.get("pilot_only") is True,
        "promotion_not_claimed": payload.get("promotion_claimed") is False,
        "learning_not_claimed": payload.get("learned_avoidance_claimed") is False,
        "final_not_claimed": payload.get("final_policy_claimed") is False,
        "generalization_not_claimed": (
            payload.get("final_generalization_claimed") is False
        ),
        "real_world_not_claimed": payload.get("real_world_claimed") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("Stage 0 pilot report is not accepted: " + ", ".join(failures))
    assert_sanitized_report(payload)
    return AcceptedReportEvidence(
        _relative(source, repository_root),
        digest,
        int(payload["schema_version"]),
        "stage-0-live-pilot",
    )


def save_live_pilot_report(
    report: LivePilotReport,
    output_path: Path,
    *,
    repository_root: Path,
    approved_root: Path,
) -> Path:
    destination = _resolve(repository_root, output_path)
    _require_under(destination, approved_root, "live-pilot report")
    payload = _jsonable(asdict(report))
    assert_sanitized_report(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def default_live_pilot_report_path(
    prepared: PreparedLivePilot,
    *,
    run_token: str | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    token = run_token or uuid.uuid4().hex[:8]
    return prepared.report_directory / (
        f"m13_8_{prepared.stage.stage_id}_{prepared.pilot_kind.value}_"
        f"{timestamp}_{token}.json"
    )


def summarize_live_pilot_reports(
    paths: Sequence[Path],
    *,
    repository_root: Path,
    approved_root: Path,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("summarize-live-pilot requires report paths")
    reports: list[Mapping[str, Any]] = []
    for path in paths:
        source = _resolve(repository_root, path)
        _require_under(source, approved_root, "live-pilot report")
        payload = _read_mapping(source)
        if payload.get("schema_version") != M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported live-pilot report schema: {source.name}")
        assert_sanitized_report(payload)
        reports.append(payload)
    return {
        "schema_version": M13_8_LIVE_PILOT_SUMMARY_SCHEMA_VERSION,
        "report_count": len(reports),
        "successful_report_count": sum(
            item.get("report_success") is True for item in reports
        ),
        "stage_ids": sorted({str(item.get("stage_id")) for item in reports}),
        "live_execution_observed": any(
            item.get("infrastructure_success") is not None for item in reports
        ),
        "promotion_claimed": False,
        "learned_avoidance_claimed": False,
        "final_generalization_claimed": False,
        "real_world_claimed": False,
    }


def inspect_git(repository_root: Path) -> GitEvidence:
    branch = _git(repository_root, "branch", "--show-current")
    commit = _git(repository_root, "rev-parse", "HEAD")
    tracked = _git(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    return GitEvidence(branch, commit, tracked == "")


def is_git_ignored(repository_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative.as_posix()],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def assert_sanitized_report(payload: Mapping[str, Any]) -> None:
    keys = _all_keys(payload)
    forbidden = {
        "raw_point_cloud",
        "point_cloud",
        "raw_trajectory",
        "trajectory_points",
        "astar_path",
        "reference_path",
        "obstacle_coordinates",
        "obstacles",
        "settings_path",
        "environment_variables",
    }
    if forbidden.intersection(keys):
        raise ValueError("live-pilot evidence contains a prohibited payload")


def _accepted_prerequisite(
    path: Path,
    *,
    expected_digest: str,
    repository_root: Path,
    approved_root: Path,
    mode: str,
) -> AcceptedReportEvidence:
    source = _resolve(repository_root, path)
    _require_under(source, approved_root, f"{mode} report")
    payload = _read_mapping(source)
    digest = _sha256_file(source)
    if digest != expected_digest:
        raise ValueError(f"{mode} report digest mismatch")
    success = payload.get("success")
    if success is None:
        success = payload.get("report_success")
    if success is not True:
        raise ValueError(f"{mode} report is not accepted")
    assert_sanitized_report(payload)
    schema = payload.get("schema_version")
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise ValueError(f"{mode} report schema is missing")
    return AcceptedReportEvidence(
        _relative(source, repository_root),
        digest,
        schema,
        mode,
    )


def _require_live_m13_6_report(payload: Mapping[str, Any], label: str) -> None:
    checks = {
        "report_success": payload.get("report_success") is True,
        "authorization_evidence": isinstance(
            payload.get("authorization_evidence"), Mapping
        )
        and all(payload["authorization_evidence"].values()),
        "cleanup_evidence": isinstance(payload.get("cleanup_evidence"), Mapping),
        "broad_reset_guard": isinstance(payload.get("broad_reset_evidence"), Mapping)
        and payload["broad_reset_evidence"].get("guard_installed") is True
        and payload["broad_reset_evidence"].get("reset_attempted") is False,
        "run_id": isinstance(payload.get("run_id"), str)
        and bool(payload.get("run_id")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            f"M13.6 report {label} lacks accepted live evidence: " + ", ".join(failures)
        )


def _stage(value: Any) -> LivePilotStage:
    raw = _mapping(value, "stage")
    _keys(
        raw,
        {
            "stage_id",
            "profile_id",
            "base_seed",
            "obstacle_count",
            "requires_stage_0_report",
            "smoke",
            "pilot",
        },
        "stage",
    )
    return LivePilotStage(
        stage_id=_string(raw["stage_id"], "stage_id"),
        profile_id=_string(raw["profile_id"], "profile_id"),
        base_seed=_integer(raw["base_seed"], "base_seed"),
        obstacle_count=_integer(raw["obstacle_count"], "obstacle_count"),
        requires_stage_0_report=_boolean(
            raw["requires_stage_0_report"], "requires_stage_0_report"
        ),
        smoke=_budget(raw["smoke"], "smoke"),
        pilot=_budget(raw["pilot"], "pilot"),
    )


def _budget(value: Any, label: str) -> PilotBudget:
    raw = _mapping(value, label)
    _keys(raw, {"minimum_requested_steps", "cumulative_cap_steps"}, label)
    return PilotBudget(
        _integer(raw["minimum_requested_steps"], "minimum_requested_steps"),
        _integer(raw["cumulative_cap_steps"], "cumulative_cap_steps"),
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Git inspection failed: " + (completed.stderr.strip() or "unknown error")
        )
    return completed.stdout.strip()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("evidence path must remain repository-relative") from exc


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )


def _optional_resolve(root: Path, path: Path | None) -> Path | None:
    return None if path is None else _resolve(root, path)


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {root}") from exc


def _repository_relative(value: str, label: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must contain a list")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {label} fields: {sorted(missing)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty unpadded string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        output = {str(key) for key in value}
        for item in value.values():
            output.update(_all_keys(item))
        return output
    if isinstance(value, (tuple, list)):
        output: set[str] = set()
        for item in value:
            output.update(_all_keys(item))
        return output
    return set()


__all__ = [
    "DEFAULT_LIVE_PILOT_CONFIG_PATH",
    "GitEvidence",
    "LIVE_PILOT_CONFIG_SCHEMA_VERSION",
    "LIVE_PILOT_METADATA_SCHEMA_VERSION",
    "LivePilotAuthorizations",
    "LivePilotConfig",
    "LivePilotMetadata",
    "LivePilotReport",
    "M13_8_LIVE_PILOT_REPORT_SCHEMA_VERSION",
    "PilotKind",
    "PreparedLivePilot",
    "assert_sanitized_report",
    "default_live_pilot_report_path",
    "inspect_git",
    "is_git_ignored",
    "load_live_pilot_config",
    "prepare_live_pilot",
    "save_live_pilot_report",
    "summarize_live_pilot_reports",
    "validate_m13_6_evidence",
    "validate_stage_0_pilot_report",
]
