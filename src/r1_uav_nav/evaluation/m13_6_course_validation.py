"""Offline-first M13.6 scripted-controller course validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from r1_uav_nav.envs.colosseum_obstacle_uav_env import (
    ColosseumObstacleUAVEnv,
    ObstacleCourseSelectionConfig,
    ObstacleRuntimeAuthorization,
    load_colosseum_obstacle_uav_env_config,
)
from r1_uav_nav.evaluation.reference_controllers import (
    ControllerDecision,
    ControllerPrivilege,
    ControllerStepInput,
    DirectGoalController,
    DirectGoalControllerConfig,
    OracleWaypointController,
    OracleWaypointControllerConfig,
    RandomController,
    RandomControllerConfig,
    ReferenceController,
)
from r1_uav_nav.sim.colosseum_capabilities import validate_report_output_path
from r1_uav_nav.sim.lidar_features import (
    LidarFeatureConfig,
    load_lidar_feature_config,
)
from r1_uav_nav.sim.lidar_live_validation import BroadResetGuard
from r1_uav_nav.sim.static_course import (
    ValidatedCourse,
    generate_solvable_course,
    load_course_suite_config,
    require_solvable_course,
)

M13_6_CONFIG_SCHEMA_VERSION = 1
M13_6_EPISODE_REPORT_SCHEMA_VERSION = 1
M13_6_SUITE_SUMMARY_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("configs/evaluation/m13_6_reference_controllers.yaml")
DEFAULT_REPORT_DIR = Path("results/reports/m13/reference_controllers")
EXPECTED_VEHICLE_NAME = "SimpleFlight"
EXPECTED_LIDAR_NAME = "LidarSensor1"


class ControllerKind(str, Enum):
    """Implemented M13.6 controller identifiers."""

    RANDOM = "random"
    DIRECT = "direct"
    ORACLE = "oracle"


class ExpectedOutcome(str, Enum):
    """Expected task-level behavior for one matrix entry."""

    GOAL_SUCCESS = "goal_success"
    BASELINE_FAILURE = "baseline_failure"


@dataclass(frozen=True)
class ReferenceControllerMatrixEntry:
    """One approved controller/course/seed live-validation pairing."""

    controller_id: ControllerKind
    course_profile: str
    base_seed: int
    expected_outcome: ExpectedOutcome
    controller_seed: int | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.course_profile.strip():
            raise ValueError("course_profile must not be empty")
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise ValueError("base_seed must be an integer")
        if self.controller_id is ControllerKind.RANDOM:
            if not isinstance(self.controller_seed, int) or isinstance(
                self.controller_seed, bool
            ):
                raise ValueError("random entries require an integer controller seed")
        elif self.controller_seed is not None:
            raise ValueError("only random entries may declare controller_seed")

    @property
    def identity(self) -> tuple[str, str, int, int | None]:
        return (
            self.controller_id.value,
            self.course_profile,
            self.base_seed,
            self.controller_seed,
        )


@dataclass(frozen=True)
class M13_6LiveAuthorizations:
    """All acknowledgements required before importing a simulator client."""

    allow_live_rpc: bool = False
    allow_scene_mutation: bool = False
    confirm_scene_area_clear: bool = False
    confirm_no_visible_collision: bool = False
    allow_debug_markers: bool = False
    allow_marker_flush: bool = False
    allow_flight: bool = False
    allow_start_positioning: bool = False
    confirm_clear_airspace: bool = False
    confirm_preflight_survey_passed: bool = False
    confirm_grounded_lidar_passed: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in asdict(self).values()):
            raise ValueError("live authorization values must be boolean")

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value is not True)


@dataclass(frozen=True)
class M13_6ReferenceControllerConfig:
    """Strict versioned software and live-run configuration."""

    schema_version: int
    vehicle_name: str
    lidar_name: str
    client_module: str
    course_suite_path: str
    obstacle_env_config_path: str
    lidar_config_path: str
    report_directory: str
    control_duration_s: float
    max_horizontal_velocity_m_s: float
    max_vertical_velocity_m_s: float
    episode_step_limit: int
    watchdog_timeout_s: float
    clearance_abort_m: float
    direct: DirectGoalControllerConfig
    oracle: OracleWaypointControllerConfig
    required_matrix: tuple[ReferenceControllerMatrixEntry, ...]
    optional_matrix: tuple[ReferenceControllerMatrixEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != M13_6_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported M13.6 configuration schema")
        for name in (
            "vehicle_name",
            "lidar_name",
            "client_module",
            "course_suite_path",
            "obstacle_env_config_path",
            "lidar_config_path",
            "report_directory",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "control_duration_s",
            "max_horizontal_velocity_m_s",
            "max_vertical_velocity_m_s",
            "watchdog_timeout_s",
            "clearance_abort_m",
        ):
            _positive_float(getattr(self, name), name)
        if (
            not isinstance(self.episode_step_limit, int)
            or isinstance(self.episode_step_limit, bool)
            or self.episode_step_limit <= 0
        ):
            raise ValueError("episode_step_limit must be a positive integer")
        validate_required_matrix(self.required_matrix, self.optional_matrix)

    @property
    def configuration_digest(self) -> str:
        return _digest(_jsonable(asdict(self)))


@dataclass(frozen=True)
class PreparedReferenceRun:
    """Inputs validated entirely offline before simulator import."""

    config: M13_6ReferenceControllerConfig
    matrix_entry: ReferenceControllerMatrixEntry
    course: ValidatedCourse
    lidar_config: LidarFeatureConfig
    authorizations: M13_6LiveAuthorizations
    report_directory: Path


@dataclass(frozen=True)
class ReferenceEpisodeReport:
    """Sanitized one-episode reference-controller evidence."""

    schema_version: int
    run_id: str
    started_at: str
    completed_at: str
    controller_id: str
    controller_privilege: str
    controller_seed: int | None
    controller_config_digest: str
    validation_config_digest: str
    expected_outcome: str
    course_profile: str
    base_seed: int
    accepted_candidate_seed: int
    attempt_index: int
    scene_digest: str
    occupancy_digest: str
    solvability_digest: str
    direct_line_clear: bool
    observation_contract: dict[str, Any]
    action_contract: dict[str, Any]
    step_count: int
    total_reward: float
    path_length_m: float
    final_distance_to_goal_m: float | None
    total_progress_m: float
    episode_success: bool
    report_success: bool
    expected_baseline_failure: bool
    clearance_abort: bool
    collision: bool
    termination_reason: str | None
    truncated: bool
    interrupted: bool
    oracle_evidence: dict[str, Any] | None
    step_trace: tuple[dict[str, Any], ...]
    lidar_summary: dict[str, Any]
    broad_reset_evidence: dict[str, Any]
    authorization_evidence: dict[str, bool]
    cleanup_evidence: dict[str, Any] | None
    acceptance_checks: dict[str, bool]
    acceptance_failures: tuple[str, ...]
    errors: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceSuiteSummary:
    """Required-matrix validation over explicit episode reports."""

    schema_version: int
    report_success: bool
    required_identities: tuple[tuple[str, str, int, int | None], ...]
    observed_identities: tuple[tuple[str, str, int, int | None], ...]
    missing_identities: tuple[tuple[str, str, int, int | None], ...]
    duplicate_identities: tuple[tuple[str, str, int, int | None], ...]
    unexpected_identities: tuple[tuple[str, str, int, int | None], ...]
    failed_identities: tuple[tuple[str, str, int, int | None], ...]


def load_m13_6_config(path: str | Path) -> M13_6ReferenceControllerConfig:
    """Load and strictly validate the M13.6 YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("M13.6 config must contain a mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "vehicle_name",
            "lidar_name",
            "client_module",
            "course_suite_path",
            "obstacle_env_config_path",
            "lidar_config_path",
            "report_directory",
            "runtime",
            "direct",
            "oracle",
            "required_matrix",
            "optional_matrix",
        },
        "root",
    )
    runtime = _mapping(raw.get("runtime"), "runtime")
    _reject_unknown(
        runtime,
        {
            "control_duration_s",
            "max_horizontal_velocity_m_s",
            "max_vertical_velocity_m_s",
            "episode_step_limit",
            "watchdog_timeout_s",
            "clearance_abort_m",
        },
        "runtime",
    )
    direct_raw = _mapping(raw.get("direct"), "direct")
    oracle_raw = _mapping(raw.get("oracle"), "oracle")
    _reject_unknown(
        direct_raw,
        set(DirectGoalControllerConfig.__dataclass_fields__) - {"schema_version"},
        "direct",
    )
    _reject_unknown(
        oracle_raw,
        set(OracleWaypointControllerConfig.__dataclass_fields__) - {"schema_version"},
        "oracle",
    )
    required = tuple(
        _matrix_entry(item, optional=False)
        for item in _sequence(raw.get("required_matrix"), "required_matrix")
    )
    optional = tuple(
        _matrix_entry(item, optional=True)
        for item in _sequence(raw.get("optional_matrix", ()), "optional_matrix")
    )
    return M13_6ReferenceControllerConfig(
        schema_version=_integer(raw, "schema_version"),
        vehicle_name=_string(raw, "vehicle_name"),
        lidar_name=_string(raw, "lidar_name"),
        client_module=_string(raw, "client_module"),
        course_suite_path=_string(raw, "course_suite_path"),
        obstacle_env_config_path=_string(raw, "obstacle_env_config_path"),
        lidar_config_path=_string(raw, "lidar_config_path"),
        report_directory=_string(raw, "report_directory"),
        control_duration_s=_number(runtime, "control_duration_s"),
        max_horizontal_velocity_m_s=_number(runtime, "max_horizontal_velocity_m_s"),
        max_vertical_velocity_m_s=_number(runtime, "max_vertical_velocity_m_s"),
        episode_step_limit=_integer(runtime, "episode_step_limit"),
        watchdog_timeout_s=_number(runtime, "watchdog_timeout_s"),
        clearance_abort_m=_number(runtime, "clearance_abort_m"),
        direct=DirectGoalControllerConfig(**dict(direct_raw)),
        oracle=OracleWaypointControllerConfig(**dict(oracle_raw)),
        required_matrix=required,
        optional_matrix=optional,
    )


def validate_required_matrix(
    required: Sequence[ReferenceControllerMatrixEntry],
    optional: Sequence[ReferenceControllerMatrixEntry] = (),
) -> None:
    """Require the accepted M13.6 baseline matrix exactly."""
    expected = {
        ("direct", "empty", 0, None, "goal_success"),
        ("oracle", "empty", 0, None, "goal_success"),
        ("direct", "easy", 1100, None, "goal_success"),
        ("oracle", "easy", 1100, None, "goal_success"),
        ("direct", "medium", 2100, None, "baseline_failure"),
        ("oracle", "medium", 2100, None, "goal_success"),
        ("random", "medium", 2100, 1360, "baseline_failure"),
        ("random", "medium", 2100, 1361, "baseline_failure"),
        ("random", "medium", 2100, 1362, "baseline_failure"),
    }
    actual = {
        (*item.identity, item.expected_outcome.value)
        for item in required
        if not item.optional
    }
    if actual != expected or len(required) != len(expected):
        raise ValueError("required_matrix does not match the accepted M13.6 matrix")
    optional_expected = {
        ("oracle", "hard", 3100, None, "goal_success"),
    }
    actual_optional = {
        (*item.identity, item.expected_outcome.value)
        for item in optional
        if item.optional
    }
    if actual_optional != optional_expected or len(optional) != 1:
        raise ValueError("optional_matrix must contain only oracle hard seed 3100")


def prepare_reference_run(
    config: M13_6ReferenceControllerConfig,
    *,
    repository_root: Path,
    controller_id: str,
    course_profile: str,
    base_seed: int,
    controller_seed: int | None,
    authorizations: M13_6LiveAuthorizations,
    report_directory: Path | None = None,
    allow_optional_hard: bool = False,
    confirm_required_stages_passed: bool = False,
) -> PreparedReferenceRun:
    """Validate names, matrix, authorizations, output, and course before import."""
    if config.vehicle_name != EXPECTED_VEHICLE_NAME:
        raise ValueError("M13.6 requires vehicle name SimpleFlight")
    if config.lidar_name != EXPECTED_LIDAR_NAME:
        raise ValueError("M13.6 requires LiDAR name LidarSensor1")
    try:
        identity = (
            ControllerKind(controller_id).value,
            course_profile,
            int(base_seed),
            controller_seed,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid controller/course identity") from exc
    matches = [
        item
        for item in (*config.required_matrix, *config.optional_matrix)
        if item.identity == identity
    ]
    if len(matches) != 1:
        raise ValueError("controller/course/seed pairing is not declared")
    entry = matches[0]
    if entry.optional and not (allow_optional_hard and confirm_required_stages_passed):
        raise ValueError(
            "optional hard run requires --allow-optional-hard and "
            "--confirm-required-stages-passed"
        )
    missing = authorizations.missing()
    if missing:
        raise ValueError("live run lacks required authorization: " + ", ".join(missing))

    root = repository_root.resolve()
    approved = (root / config.report_directory).resolve()
    requested = report_directory or Path(config.report_directory)
    resolved_report_dir = (
        requested if requested.is_absolute() else root / requested
    ).resolve()
    if not resolved_report_dir.is_relative_to(approved):
        raise ValueError(
            "M13.6 reports must remain under "
            "results/reports/m13/reference_controllers"
        )
    validate_report_output_path(resolved_report_dir / "preflight.json", root)

    lidar_config = load_lidar_feature_config(_resolve(root, config.lidar_config_path))
    if lidar_config.vehicle_name != config.vehicle_name:
        raise ValueError("configured vehicle name does not match M13.4 sensing")
    if lidar_config.lidar_name != config.lidar_name:
        raise ValueError("configured LiDAR name does not match M13.4 sensing")
    suite = load_course_suite_config(_resolve(root, config.course_suite_path))
    profile = suite.profile(course_profile)
    if base_seed not in profile.base_seeds:
        raise ValueError("base seed is not declared by the selected course profile")
    course = require_solvable_course(
        generate_solvable_course(
            suite,
            course_profile,
            base_seed,
            repository_root=root,
        )
    )
    if controller_id == ControllerKind.ORACLE.value:
        _validate_oracle_plan(course, config)
    return PreparedReferenceRun(
        config,
        entry,
        course,
        lidar_config,
        authorizations,
        resolved_report_dir,
    )


def validate_offline_configuration(
    config: M13_6ReferenceControllerConfig,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], ...]:
    """Validate every required and optional course/controller plan offline."""
    root = repository_root.resolve()
    suite = load_course_suite_config(_resolve(root, config.course_suite_path))
    lidar = load_lidar_feature_config(_resolve(root, config.lidar_config_path))
    if (lidar.vehicle_name, lidar.lidar_name) != (
        config.vehicle_name,
        config.lidar_name,
    ):
        raise ValueError("M13.6 names do not match the M13.4 sensing config")
    output: list[dict[str, Any]] = []
    course_cache: dict[tuple[str, int], ValidatedCourse] = {}
    for entry in (*config.required_matrix, *config.optional_matrix):
        key = (entry.course_profile, entry.base_seed)
        if key not in course_cache:
            profile = suite.profile(entry.course_profile)
            if entry.base_seed not in profile.base_seeds:
                raise ValueError(f"undeclared course pairing {key!r}")
            course_cache[key] = require_solvable_course(
                generate_solvable_course(
                    suite,
                    entry.course_profile,
                    entry.base_seed,
                    repository_root=root,
                )
            )
        course = course_cache[key]
        oracle = (
            _validate_oracle_plan(course, config)
            if entry.controller_id is ControllerKind.ORACLE
            else None
        )
        output.append(_preview_entry(entry, course, oracle))
    return tuple(output)


def create_reference_controller(
    prepared: PreparedReferenceRun,
    env: ColosseumObstacleUAVEnv,
) -> ReferenceController:
    """Construct the declared controller after environment reset."""
    entry = prepared.matrix_entry
    state_spec = env.controller_state_spec()
    if entry.controller_id is ControllerKind.RANDOM:
        assert entry.controller_seed is not None
        return RandomController(RandomControllerConfig(seed=entry.controller_seed))
    if entry.controller_id is ControllerKind.DIRECT:
        return DirectGoalController(prepared.config.direct, state_spec)
    path = prepared.course.result.path_result.reference_path
    return OracleWaypointController(
        prepared.config.oracle,
        state_spec,
        reference_path=path,
        grid=prepared.course.grid,
    )


def execute_reference_episode(
    prepared: PreparedReferenceRun,
    *,
    client: Any,
    client_module: ModuleType,
    repository_root: Path,
    environment_factory: Callable[..., ColosseumObstacleUAVEnv] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ReferenceEpisodeReport:
    """Execute one bounded episode and always attempt ordered environment cleanup."""
    started = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex
    guard = BroadResetGuard(client)
    env: ColosseumObstacleUAVEnv | None = None
    controller: ReferenceController | None = None
    trace: list[dict[str, Any]] = []
    errors: list[str] = []
    interrupted = False
    clearance_abort = False
    collision = False
    episode_success = False
    truncated = False
    termination_reason: str | None = None
    total_reward = 0.0
    initial_distance: float | None = None
    final_distance: float | None = None
    final_info: Mapping[str, Any] = {}
    cleanup: Any | None = None

    try:
        env_config = _live_environment_config(prepared, repository_root)
        factory = environment_factory or ColosseumObstacleUAVEnv
        env = factory(
            env_config,
            client_factory=lambda: guard,
            client_module=client_module,
            repository_root=repository_root,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
        )
        observation, info = env.reset(
            seed=(
                prepared.matrix_entry.controller_seed
                if prepared.matrix_entry.controller_seed is not None
                else prepared.matrix_entry.base_seed
            ),
            options={"validated_course": prepared.course},
        )
        controller = create_reference_controller(prepared, env)
        controller.reset()
        initial_distance = _optional_finite(info.get("distance_to_goal"))
        final_distance = initial_distance
        final_info = info

        for step_index in range(prepared.config.episode_step_limit):
            lidar_valid = bool(float(observation[82]) == 1.0)
            if not lidar_valid:
                decision = ControllerDecision(
                    np.zeros(3, dtype=np.float32),
                    target_label="sensor-hold",
                )
                controller_action_generated = False
            else:
                clearance = _optional_finite(info.get("reward_clearance_m"))
                if (
                    clearance is not None
                    and clearance <= prepared.config.clearance_abort_m
                ):
                    clearance_abort = True
                    termination_reason = "clearance_abort"
                    break
                decision = controller.act(
                    ControllerStepInput(observation, step_index, info)
                )
                controller_action_generated = True

            observation, reward, terminated, step_truncated, info = env.step(
                decision.action
            )
            total_reward += float(reward)
            final_info = info
            final_distance = _optional_finite(info.get("distance_to_goal"))
            collision = collision or bool(info.get("collision", False))
            episode_success = bool(info.get("success", False))
            truncated = bool(step_truncated)
            termination_reason = (
                str(info["termination_reason"])
                if info.get("termination_reason") is not None
                else None
            )
            trace.append(
                _trace_item(
                    step_index,
                    decision,
                    controller_action_generated=controller_action_generated,
                    observation=observation,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(step_truncated),
                    info=info,
                    lidar_evidence=env.latest_lidar_evidence(),
                )
            )
            if terminated or step_truncated:
                break
    except KeyboardInterrupt as exc:
        interrupted = True
        errors.append(f"{type(exc).__name__}: operator interrupted the run")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                cleanup = env.close_with_result()
            except BaseException as exc:
                errors.append(f"cleanup {type(exc).__name__}: {exc}")

    cleanup_evidence = _jsonable(asdict(cleanup)) if cleanup is not None else None
    cleanup_success = bool(cleanup is not None and cleanup.succeeded)
    expected_failure = (
        prepared.matrix_entry.expected_outcome is ExpectedOutcome.BASELINE_FAILURE
    )
    outcome_matches = _outcome_matches(
        prepared.matrix_entry,
        episode_success=episode_success,
        clearance_abort=clearance_abort,
        collision=collision,
        termination_reason=termination_reason,
        final_info=final_info,
    )
    checks = {
        "expected_outcome_observed": outcome_matches,
        "named_uav_and_scene_cleanup_succeeded": cleanup_success,
        "broad_simulator_reset_not_used": not guard.reset_attempted,
        "no_operational_errors": not errors,
        "not_interrupted": not interrupted,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    report_success = all(checks.values())
    oracle_evidence = _oracle_report_evidence(controller)
    lidar_summary = _summarize_lidar_trace(trace)
    path_length = _optional_finite(final_info.get("path_length_m")) or 0.0
    progress = (
        initial_distance - final_distance
        if initial_distance is not None and final_distance is not None
        else 0.0
    )
    result = prepared.course.result
    report = ReferenceEpisodeReport(
        schema_version=M13_6_EPISODE_REPORT_SCHEMA_VERSION,
        run_id=run_id,
        started_at=started.isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        controller_id=prepared.matrix_entry.controller_id.value,
        controller_privilege=(
            controller.privilege.value
            if controller is not None
            else _declared_privilege(prepared.matrix_entry.controller_id).value
        ),
        controller_seed=prepared.matrix_entry.controller_seed,
        controller_config_digest=(
            controller.configuration_digest
            if controller is not None
            else _declared_controller_digest(prepared)
        ),
        validation_config_digest=prepared.config.configuration_digest,
        expected_outcome=prepared.matrix_entry.expected_outcome.value,
        course_profile=result.profile_id,
        base_seed=result.base_seed,
        accepted_candidate_seed=result.accepted_candidate_seed,
        attempt_index=result.attempt_index,
        scene_digest=result.scene_digest,
        occupancy_digest=result.occupancy_digest,
        solvability_digest=result.solvability_digest,
        direct_line_clear=result.path_result.direct_line_clear,
        observation_contract={
            "shape": [83],
            "dtype": "float32",
            "navigation_indices": [0, 9],
            "lidar_indices": [10, 81],
            "lidar_valid_index": 82,
        },
        action_contract={
            "shape": [3],
            "dtype": "float32",
            "bounds": [-1.0, 1.0],
            "frame": "world_ned",
        },
        step_count=len(trace),
        total_reward=float(total_reward),
        path_length_m=float(path_length),
        final_distance_to_goal_m=final_distance,
        total_progress_m=float(progress),
        episode_success=episode_success,
        report_success=report_success,
        expected_baseline_failure=expected_failure,
        clearance_abort=clearance_abort,
        collision=collision,
        termination_reason=termination_reason,
        truncated=truncated,
        interrupted=interrupted,
        oracle_evidence=oracle_evidence,
        step_trace=tuple(trace),
        lidar_summary=lidar_summary,
        broad_reset_evidence={
            "guard_installed": True,
            "reset_attempted": guard.reset_attempted,
        },
        authorization_evidence=asdict(prepared.authorizations),
        cleanup_evidence=cleanup_evidence,
        acceptance_checks=checks,
        acceptance_failures=failures,
        errors=tuple(errors),
        limitations=_limitations(),
    )
    _assert_report_sanitized(_jsonable(asdict(report)))
    return report


def save_reference_episode_report(
    report: ReferenceEpisodeReport,
    output_path: str | Path,
    *,
    repository_root: Path,
) -> Path:
    """Atomically write one ignored, sanitized report."""
    destination = Path(output_path)
    validate_report_output_path(destination, repository_root)
    payload = _jsonable(asdict(report))
    _assert_report_sanitized(payload)
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


def default_report_path(
    prepared: PreparedReferenceRun,
    *,
    started_at: datetime | None = None,
    run_id: str | None = None,
) -> Path:
    """Build one unique ignored report path without writing it."""
    timestamp = (started_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    identity = run_id or uuid.uuid4().hex
    entry = prepared.matrix_entry
    seed_suffix = (
        f"_controller-{entry.controller_seed}"
        if entry.controller_seed is not None
        else ""
    )
    return prepared.report_directory / (
        f"m13_6_{entry.controller_id.value}_{entry.course_profile}_"
        f"{entry.base_seed}{seed_suffix}_{timestamp}_{identity[:8]}.json"
    )


def summarize_episode_reports(
    config: M13_6ReferenceControllerConfig,
    report_paths: Sequence[str | Path],
) -> ReferenceSuiteSummary:
    """Validate explicit reports against the complete required matrix."""
    if not report_paths:
        raise ValueError("summarize requires explicit episode report paths")
    required = tuple(item.identity for item in config.required_matrix)
    allowed_optional = {item.identity for item in config.optional_matrix}
    observed: list[tuple[str, str, int, int | None]] = []
    succeeded: dict[tuple[str, str, int, int | None], bool] = {}
    counts: dict[tuple[str, str, int, int | None], int] = {}
    for path in report_paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("episode report must contain a mapping")
        if raw.get("schema_version") != M13_6_EPISODE_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported M13.6 episode report schema")
        identity = (
            str(raw.get("controller_id")),
            str(raw.get("course_profile")),
            _strict_report_int(raw.get("base_seed"), "base_seed"),
            _optional_report_int(raw.get("controller_seed"), "controller_seed"),
        )
        observed.append(identity)
        counts[identity] = counts.get(identity, 0) + 1
        succeeded[identity] = bool(raw.get("report_success", False))
    required_set = set(required)
    observed_set = set(observed)
    missing = tuple(sorted(required_set - observed_set))
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    unexpected = tuple(sorted(observed_set - required_set - allowed_optional, key=str))
    failed = tuple(sorted(key for key in required if not succeeded.get(key, False)))
    success = not (missing or duplicates or unexpected or failed)
    return ReferenceSuiteSummary(
        M13_6_SUITE_SUMMARY_SCHEMA_VERSION,
        success,
        required,
        tuple(observed),
        missing,
        duplicates,
        unexpected,
        failed,
    )


def _live_environment_config(
    prepared: PreparedReferenceRun,
    repository_root: Path,
) -> Any:
    source = load_colosseum_obstacle_uav_env_config(
        _resolve(repository_root, prepared.config.obstacle_env_config_path)
    )
    navigation = replace(
        source.navigation,
        client_module=prepared.config.client_module,
        max_horizontal_velocity=prepared.config.max_horizontal_velocity_m_s,
        max_vertical_velocity=prepared.config.max_vertical_velocity_m_s,
        control_duration=prepared.config.control_duration_s,
        max_episode_steps=prepared.config.episode_step_limit,
    )
    course = ObstacleCourseSelectionConfig(
        course_suite_path=prepared.config.course_suite_path,
        asset_catalog_path=source.course.asset_catalog_path,
        lidar_config_path=prepared.config.lidar_config_path,
        mode=source.course.mode,
        fixed_profile_id=prepared.matrix_entry.course_profile,
        fixed_base_seed=prepared.matrix_entry.base_seed,
        seeded_profile_ids=source.course.seeded_profile_ids,
        allow_external_test_endpoints=False,
    )
    authorization = ObstacleRuntimeAuthorization(
        allow_live_rpc=True,
        allow_scene_mutation=True,
        confirm_scene_area_clear=True,
        confirm_no_visible_collision=True,
        allow_debug_markers=True,
        allow_marker_flush=True,
        allow_flight=True,
        allow_start_positioning=True,
        confirm_clear_airspace=True,
    )
    return replace(
        source,
        navigation=navigation,
        course=course,
        authorization=authorization,
        max_episode_steps=prepared.config.episode_step_limit,
        watchdog_timeout_s=prepared.config.watchdog_timeout_s,
    )


def _validate_oracle_plan(
    course: ValidatedCourse,
    config: M13_6ReferenceControllerConfig,
) -> dict[str, Any]:
    from r1_uav_nav.evaluation.reference_controllers import (
        ControllerStateSpec,
    )

    state_spec = ControllerStateSpec(
        position_scales_m=(1.0, 1.0, 1.0),
        goal_displacement_scales_m=(1.0, 1.0, 1.0),
        velocity_scales_m_s=(
            config.max_horizontal_velocity_m_s,
            config.max_horizontal_velocity_m_s,
            config.max_vertical_velocity_m_s,
        ),
    )
    controller = OracleWaypointController(
        config.oracle,
        state_spec,
        reference_path=course.result.path_result.reference_path,
        grid=course.grid,
    )
    return {
        "privileged": True,
        "waypoint_count": controller.waypoint_count,
        "route_digest": controller.route_digest,
        "maximum_segment_length_m": config.oracle.maximum_segment_length_m,
    }


def _preview_entry(
    entry: ReferenceControllerMatrixEntry,
    course: ValidatedCourse,
    oracle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = course.result
    return {
        "controller_id": entry.controller_id.value,
        "controller_privilege": _declared_privilege(entry.controller_id).value,
        "controller_seed": entry.controller_seed,
        "course_profile": result.profile_id,
        "base_seed": result.base_seed,
        "accepted_candidate_seed": result.accepted_candidate_seed,
        "attempt_index": result.attempt_index,
        "scene_digest": result.scene_digest,
        "occupancy_digest": result.occupancy_digest,
        "solvability_digest": result.solvability_digest,
        "direct_line_clear": result.path_result.direct_line_clear,
        "expected_outcome": entry.expected_outcome.value,
        "optional": entry.optional,
        "oracle_route": dict(oracle) if oracle is not None else None,
    }


def _trace_item(
    step_index: int,
    decision: ControllerDecision,
    *,
    controller_action_generated: bool,
    observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    lidar_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = (
        lidar_evidence.get("diagnostics", {})
        if isinstance(lidar_evidence, Mapping)
        else {}
    )
    return {
        "step_index": step_index,
        "controller_action_generated": controller_action_generated,
        "action": [float(value) for value in decision.action],
        "target_label": decision.target_label,
        "target_error_m": decision.target_error_m,
        "waypoint_index": decision.waypoint_index,
        "reward": reward,
        "distance_to_goal_m": _optional_finite(info.get("distance_to_goal")),
        "path_length_m": _optional_finite(info.get("path_length_m")),
        "clearance_m": _optional_finite(info.get("reward_clearance_m")),
        "clearance_source": info.get("clearance_source"),
        "lidar_valid": float(observation[82]),
        "lidar_status": (
            lidar_evidence.get("status")
            if isinstance(lidar_evidence, Mapping)
            else None
        ),
        "lidar_timestamp": diagnostics.get("timestamp"),
        "lidar_timestamp_status": diagnostics.get("timestamp_status"),
        "nearest_overall_m": diagnostics.get("nearest_overall_m"),
        "collision": bool(info.get("collision", False)),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": info.get("termination_reason"),
    }


def _outcome_matches(
    entry: ReferenceControllerMatrixEntry,
    *,
    episode_success: bool,
    clearance_abort: bool,
    collision: bool,
    termination_reason: str | None,
    final_info: Mapping[str, Any],
) -> bool:
    if entry.expected_outcome is ExpectedOutcome.GOAL_SUCCESS:
        return (
            episode_success
            and termination_reason == "goal_reached"
            and not clearance_abort
            and not collision
        )
    if entry.controller_id is ControllerKind.DIRECT:
        return not episode_success and (clearance_abort or collision)
    return not episode_success and not bool(final_info.get("success", False))


def _oracle_report_evidence(
    controller: ReferenceController | None,
) -> dict[str, Any] | None:
    if not isinstance(controller, OracleWaypointController):
        return None
    return {
        "privileged_reference_controller": True,
        "waypoint_count": controller.waypoint_count,
        "completed_waypoint_count": controller.completed_waypoints,
        "route_digest": controller.route_digest,
        "full_route_coordinates_included": False,
    }


def _summarize_lidar_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_count = sum(float(item.get("lidar_valid", 0.0)) == 1.0 for item in trace)
    statuses = [item.get("lidar_status") for item in trace]
    timestamps = [item.get("lidar_timestamp") for item in trace]
    clearances = [
        float(item["clearance_m"])
        for item in trace
        if item.get("clearance_m") is not None
    ]
    return {
        "sample_count": len(trace),
        "valid_sample_count": valid_count,
        "invalid_sample_count": len(trace) - valid_count,
        "statuses": statuses,
        "timestamps": timestamps,
        "minimum_clearance_m": min(clearances) if clearances else None,
        "raw_scan_payload_included": False,
    }


def _declared_privilege(controller: ControllerKind) -> ControllerPrivilege:
    return (
        ControllerPrivilege.REFERENCE_PATH
        if controller is ControllerKind.ORACLE
        else ControllerPrivilege.NONE
    )


def _declared_controller_digest(prepared: PreparedReferenceRun) -> str:
    entry = prepared.matrix_entry
    if entry.controller_id is ControllerKind.RANDOM:
        return _digest(asdict(RandomControllerConfig(seed=entry.controller_seed or 0)))
    if entry.controller_id is ControllerKind.DIRECT:
        return _digest(asdict(prepared.config.direct))
    return _digest(asdict(prepared.config.oracle))


def _limitations() -> tuple[str, ...]:
    return (
        (
            "The oracle controller is privileged reference evidence, "
            "not deployable intelligence."
        ),
        "Built-in Blocks geometry is not represented by M13.3 course occupancy.",
        "Physical simulator collision response is not proven by static solvability.",
        "LiDAR clearance depends on beam density and angular sector aggregation.",
        "No obstacle-aware policy was trained.",
        "No camera, SLAM, dynamic-obstacle, mapping, or real-world claim is made.",
    )


def _assert_report_sanitized(value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, sort_keys=True).lower()
    forbidden = (
        "raw_point_cloud",
        "reference_path_length",
        "oracle_path_length",
        "obstacle_coordinates",
        "settings_path",
    )
    if any(token in serialized for token in forbidden):
        raise ValueError("report contains a prohibited evidence field")


def _matrix_entry(value: Any, *, optional: bool) -> ReferenceControllerMatrixEntry:
    raw = _mapping(value, "matrix entry")
    _reject_unknown(
        raw,
        {
            "controller_id",
            "course_profile",
            "base_seed",
            "controller_seed",
            "expected_outcome",
        },
        "matrix entry",
    )
    return ReferenceControllerMatrixEntry(
        ControllerKind(_string(raw, "controller_id")),
        _string(raw, "course_profile"),
        _integer(raw, "base_seed"),
        ExpectedOutcome(_string(raw, "expected_outcome")),
        (
            _integer(raw, "controller_seed")
            if raw.get("controller_seed") is not None
            else None
        ),
        optional,
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must contain a list")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")


def _string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{key} must be an integer")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if (
        not isinstance(result, (int, float))
        or isinstance(result, bool)
        or not math.isfinite(float(result))
    ):
        raise ValueError(f"{key} must be finite and numeric")
    return float(result)


def _positive_float(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _strict_report_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"report {label} must be an integer")
    return value


def _optional_report_int(value: Any, label: str) -> int | None:
    return None if value is None else _strict_report_int(value, label)


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
