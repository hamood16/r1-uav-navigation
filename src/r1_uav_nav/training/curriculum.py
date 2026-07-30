"""Strict offline M13.8 curriculum configuration and progression state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from r1_uav_nav.sim.static_course import (
    CourseSuiteConfig,
    generate_solvable_course,
    load_course_suite_config,
)
from r1_uav_nav.training.curriculum_evidence import (
    CurriculumValidationSummary,
    write_curriculum_report,
)
from r1_uav_nav.training.curriculum_sampling import (
    CurriculumCourseRef,
    RobustnessPerturbationConfig,
)
from r1_uav_nav.training.long_run_state import canonical_digest

CURRICULUM_CONFIG_SCHEMA_VERSION = 1
CURRICULUM_STATE_SCHEMA_VERSION = 1
PROMOTION_DECISION_SCHEMA_VERSION = 1
M13_9_RESERVED_FINAL_TEST_SEEDS = (
    9100,
    9200,
    9300,
    10100,
    10200,
    10300,
)
DEFAULT_CURRICULUM_CONFIG_PATH = Path(
    "configs/training/m13_8_static_obstacle_curriculum.yaml"
)


class CurriculumStatus(str, Enum):
    ACTIVE = "active"
    PROMOTED = "promoted"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class PromotionStatus(str, Enum):
    WAITING_MINIMUM_STEPS = "waiting_minimum_steps"
    VALIDATION_FAILED = "validation_failed"
    PASS_RECORDED = "pass_recorded"
    PROMOTED = "promoted"
    CURRICULUM_COMPLETE = "curriculum_complete"
    STAGE_BUDGET_EXHAUSTED = "stage_budget_exhausted"


@dataclass(frozen=True)
class CurriculumOutputConfig:
    model_root: str
    log_root: str
    report_root: str

    def __post_init__(self) -> None:
        for name in ("model_root", "log_root", "report_root"):
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
class PromotionGate:
    minimum_episode_count: int
    minimum_success_rate: float
    maximum_collision_rate: float
    maximum_timeout_rate: float
    minimum_cleanup_success_rate: float
    maximum_safety_violation_rate: float
    maximum_sensor_failure_rate: float = 1.0
    subgroup_minimum_success_rates: dict[str, float] = field(default_factory=dict)
    minimum_non_straight_success_rate: float | None = None
    minimum_direct_success_advantage: float | None = None
    maximum_earlier_success_regression: float = 1.0
    maximum_earlier_collision_regression: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_episode_count, int)
            or isinstance(self.minimum_episode_count, bool)
            or self.minimum_episode_count <= 0
        ):
            raise ValueError("minimum_episode_count must be positive")
        for name in (
            "minimum_success_rate",
            "maximum_collision_rate",
            "maximum_timeout_rate",
            "minimum_cleanup_success_rate",
            "maximum_safety_violation_rate",
            "maximum_sensor_failure_rate",
            "maximum_earlier_success_regression",
            "maximum_earlier_collision_regression",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in (
            "minimum_non_straight_success_rate",
            "minimum_direct_success_advantage",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be within [0, 1]")
        if any(
            not key or not 0 <= float(value) <= 1
            for key, value in self.subgroup_minimum_success_rates.items()
        ):
            raise ValueError("subgroup success thresholds must be named and bounded")


@dataclass(frozen=True)
class SamplingLevel:
    level_id: str
    difficulty_weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.level_id or self.level_id != self.level_id.strip():
            raise ValueError("sampling level_id must not be empty or padded")
        if not self.difficulty_weights:
            raise ValueError("sampling level requires difficulty weights")
        if any(
            not name or not math.isfinite(float(value)) or float(value) <= 0
            for name, value in self.difficulty_weights.items()
        ):
            raise ValueError("difficulty weights must be named, finite, and positive")


@dataclass(frozen=True)
class CurriculumStageConfig:
    stage_id: str
    index: int
    description: str
    minimum_stage_steps: int
    maximum_stage_steps: int
    validation_interval_steps: int
    training_courses: tuple[CurriculumCourseRef, ...]
    validation_courses: tuple[CurriculumCourseRef, ...]
    gate: PromotionGate
    sampling_levels: tuple[SamplingLevel, ...] = ()
    robustness: RobustnessPerturbationConfig | None = None

    def __post_init__(self) -> None:
        if not self.stage_id or self.stage_id != self.stage_id.strip():
            raise ValueError("stage_id must not be empty or padded")
        if self.index < 0:
            raise ValueError("stage index must be non-negative")
        if not self.description.strip():
            raise ValueError("stage description must not be empty")
        if self.minimum_stage_steps <= 0:
            raise ValueError("minimum_stage_steps must be positive")
        if self.maximum_stage_steps < self.minimum_stage_steps:
            raise ValueError("maximum stage steps must cover the minimum")
        if self.validation_interval_steps <= 0:
            raise ValueError("validation interval must be positive")
        if not self.training_courses or not self.validation_courses:
            raise ValueError("every stage requires training and validation courses")
        if any(item.role != "training" for item in self.training_courses):
            raise ValueError("stage training courses must use the training role")
        if any(
            item.role != "curriculum_validation" for item in self.validation_courses
        ):
            raise ValueError(
                "stage validation courses must use curriculum_validation role"
            )


@dataclass(frozen=True)
class CurriculumConfig:
    schema_version: int
    curriculum_id: str
    long_run_config_path: str
    course_suites: dict[str, str]
    reserved_final_test_seeds: tuple[int, ...]
    required_consecutive_passes: int
    replay_on_promotion: str
    output: CurriculumOutputConfig
    stages: tuple[CurriculumStageConfig, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported M13.8 curriculum configuration schema")
        if not self.curriculum_id or self.curriculum_id != self.curriculum_id.strip():
            raise ValueError("curriculum_id must not be empty or padded")
        if self.replay_on_promotion != "preserve":
            raise ValueError("M13.8 Phase A preserves replay on promotion")
        if self.required_consecutive_passes != 3:
            raise ValueError("M13.8 requires exactly three consecutive passes")
        _repository_relative(self.long_run_config_path, "long_run_config_path")
        if set(self.course_suites) != {"m13_3", "m13_8"}:
            raise ValueError("course_suites must define exactly m13_3 and m13_8")
        for suite_id, path in self.course_suites.items():
            if not suite_id.strip():
                raise ValueError("course suite IDs must not be empty")
            _repository_relative(path, f"course suite {suite_id}")
        if len(self.stages) != 6:
            raise ValueError("M13.8 must define exactly six stages")
        if tuple(item.index for item in self.stages) != tuple(range(6)):
            raise ValueError("M13.8 stages must be ordered from index 0 through 5")
        if tuple(item.stage_id for item in self.stages) != tuple(
            f"stage-{index}" for index in range(6)
        ):
            raise ValueError("M13.8 stage IDs must be stage-0 through stage-5")
        if len(self.reserved_final_test_seeds) != len(
            set(self.reserved_final_test_seeds)
        ):
            raise ValueError("reserved final-test seeds must be unique")
        if tuple(self.reserved_final_test_seeds) != M13_9_RESERVED_FINAL_TEST_SEEDS:
            raise ValueError("M13.9 reserved final-test seeds must remain locked")
        _validate_partitions(self)

    @property
    def config_digest(self) -> str:
        return canonical_digest(asdict(self))

    def stage(self, stage_id: str) -> CurriculumStageConfig:
        matches = [item for item in self.stages if item.stage_id == stage_id]
        if len(matches) != 1:
            raise ValueError(f"unknown curriculum stage {stage_id!r}")
        return matches[0]


@dataclass(frozen=True)
class CurriculumState:
    schema_version: int
    curriculum_id: str
    curriculum_config_digest: str
    stage_id: str
    stage_index: int
    status: CurriculumStatus
    stage_start_global_timestep: int
    stage_completed_timesteps: int
    sampling_level_index: int
    consecutive_pass_count: int
    completed_stage_ids: tuple[str, ...]
    validation_history: tuple[dict[str, Any], ...]
    best_validation_by_stage: dict[str, dict[str, Any]]
    course_sampler_state: dict[str, Any] | None = None
    robustness_sampler_state: dict[str, Any] | None = None
    last_promotion_decision_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported curriculum-state schema")
        if not self.curriculum_id or not self.stage_id:
            raise ValueError("curriculum and stage IDs must not be empty")
        if len(self.curriculum_config_digest) != 64:
            raise ValueError("curriculum_config_digest must be SHA-256")
        for name in (
            "stage_index",
            "stage_start_global_timestep",
            "stage_completed_timesteps",
            "sampling_level_index",
            "consecutive_pass_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if len(self.validation_history) > 3:
            raise ValueError(
                "curriculum state retains at most three validation windows"
            )
        if (
            self.last_promotion_decision_digest is not None
            and len(self.last_promotion_decision_digest) != 64
        ):
            raise ValueError("promotion decision digest must be SHA-256")

    @classmethod
    def initial(cls, config: CurriculumConfig) -> CurriculumState:
        return cls(
            schema_version=CURRICULUM_STATE_SCHEMA_VERSION,
            curriculum_id=config.curriculum_id,
            curriculum_config_digest=config.config_digest,
            stage_id=config.stages[0].stage_id,
            stage_index=0,
            status=CurriculumStatus.ACTIVE,
            stage_start_global_timestep=0,
            stage_completed_timesteps=0,
            sampling_level_index=0,
            consecutive_pass_count=0,
            completed_stage_ids=(),
            validation_history=(),
            best_validation_by_stage={},
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["status"] = self.status.value
        return values

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CurriculumState:
        values = dict(raw)
        values["status"] = CurriculumStatus(values["status"])
        values["completed_stage_ids"] = tuple(values.get("completed_stage_ids", ()))
        values["validation_history"] = tuple(values.get("validation_history", ()))
        return cls(**values)


@dataclass(frozen=True)
class PromotionDecision:
    schema_version: int
    stage_id: str
    status: PromotionStatus
    passed: bool
    failures: tuple[str, ...]
    consecutive_pass_count: int
    promoted_to_stage_id: str | None
    decision_digest: str


def evaluate_promotion(
    config: CurriculumConfig,
    state: CurriculumState,
    summary: CurriculumValidationSummary,
) -> tuple[PromotionDecision, CurriculumState]:
    """Evaluate one complete offline validation window deterministically."""
    stage = config.stage(state.stage_id)
    if summary.stage_id != stage.stage_id:
        raise ValueError("validation summary does not match active stage")
    if state.curriculum_config_digest != config.config_digest:
        raise ValueError("curriculum state config digest mismatch")
    failures = _gate_failures(stage.gate, summary)
    if state.stage_completed_timesteps < stage.minimum_stage_steps:
        status = PromotionStatus.WAITING_MINIMUM_STEPS
        failures.append("minimum stage steps not reached")
    elif failures:
        status = PromotionStatus.VALIDATION_FAILED
    else:
        status = PromotionStatus.PASS_RECORDED
    pass_count = state.consecutive_pass_count + 1 if not failures else 0
    promoted_to: str | None = None
    new_status = CurriculumStatus.ACTIVE
    new_stage_id = state.stage_id
    new_stage_index = state.stage_index
    completed = state.completed_stage_ids
    stage_start = state.stage_start_global_timestep
    stage_steps = state.stage_completed_timesteps
    sampling_level = state.sampling_level_index

    if (
        not failures
        and stage.sampling_levels
        and sampling_level < len(stage.sampling_levels) - 1
    ):
        sampling_level += 1
        pass_count = 0
    elif not failures and pass_count >= config.required_consecutive_passes:
        completed = completed + (stage.stage_id,)
        if stage.index == len(config.stages) - 1:
            status = PromotionStatus.CURRICULUM_COMPLETE
            new_status = CurriculumStatus.COMPLETE
        else:
            promoted_to = config.stages[stage.index + 1].stage_id
            status = PromotionStatus.PROMOTED
            new_status = CurriculumStatus.PROMOTED
            new_stage_id = promoted_to
            new_stage_index = stage.index + 1
            stage_start += stage_steps
            stage_steps = 0
            sampling_level = 0
            pass_count = 0
    elif state.stage_completed_timesteps >= stage.maximum_stage_steps:
        status = PromotionStatus.STAGE_BUDGET_EXHAUSTED
        new_status = CurriculumStatus.BLOCKED
        failures.append("maximum stage budget exhausted")

    history_item = {
        "stage_id": summary.stage_id,
        "validation_digest": summary.with_digest().validation_digest,
        "passed": not failures,
    }
    best_by_stage = dict(state.best_validation_by_stage)
    candidate_rank = (
        summary.success_rate,
        -summary.collision_rate,
        summary.cleanup_success_rate,
        -summary.timeout_rate,
    )
    prior = best_by_stage.get(stage.stage_id)
    prior_rank = tuple(prior.get("rank", ())) if prior is not None else None
    if prior_rank is None or candidate_rank > prior_rank:
        best_by_stage[stage.stage_id] = {
            "rank": candidate_rank,
            "validation_digest": history_item["validation_digest"],
        }
    identity = {
        "stage_id": stage.stage_id,
        "status": status.value,
        "failures": failures,
        "pass_count": pass_count,
        "promoted_to": promoted_to,
        "validation_digest": history_item["validation_digest"],
    }
    decision_digest = canonical_digest(identity)
    decision = PromotionDecision(
        schema_version=PROMOTION_DECISION_SCHEMA_VERSION,
        stage_id=stage.stage_id,
        status=status,
        passed=not failures,
        failures=tuple(failures),
        consecutive_pass_count=pass_count,
        promoted_to_stage_id=promoted_to,
        decision_digest=decision_digest,
    )
    updated = replace(
        state,
        stage_id=new_stage_id,
        stage_index=new_stage_index,
        status=new_status,
        stage_start_global_timestep=stage_start,
        stage_completed_timesteps=stage_steps,
        sampling_level_index=sampling_level,
        consecutive_pass_count=pass_count,
        completed_stage_ids=completed,
        validation_history=(state.validation_history + (history_item,))[-3:],
        best_validation_by_stage=best_by_stage,
        last_promotion_decision_digest=decision_digest,
    )
    return decision, updated


def enable_curriculum_mode(
    long_run_config: Any,
    curriculum: CurriculumConfig,
    *,
    stage_id: str = "stage-0",
) -> Any:
    """Enable curriculum metadata without mutating M13.7 validation defaults.

    M13.8's progression engine owns its separate multi-suite validation pool.
    """
    curriculum.stage(stage_id)
    return replace(
        long_run_config,
        curriculum_id=curriculum.curriculum_id,
        curriculum_config_digest=curriculum.config_digest,
    )


def curriculum_state_for_resume(
    config: CurriculumConfig,
    *,
    resume_mode: Any,
    source_state: Mapping[str, Any] | None = None,
) -> CurriculumState:
    """Resolve the strict full-resume or Stage 0 warm-start curriculum state."""
    mode = getattr(resume_mode, "value", str(resume_mode))
    if mode == "full":
        if source_state is None:
            raise ValueError("full curriculum resume requires source state")
        restored = CurriculumState.from_mapping(source_state)
        if restored.curriculum_config_digest != config.config_digest:
            raise ValueError("curriculum config digest mismatch")
        return restored
    if mode in {"new", "model_only_warm_start"}:
        return CurriculumState.initial(config)
    raise ValueError(f"unsupported curriculum resume mode {mode!r}")


def load_curriculum_config(path: str | Path) -> CurriculumConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "curriculum configuration")
    _keys(
        root,
        {
            "schema_version",
            "curriculum_id",
            "long_run_config_path",
            "course_suites",
            "reserved_final_test_seeds",
            "required_consecutive_passes",
            "replay_on_promotion",
            "output",
            "stages",
        },
        "curriculum configuration",
    )
    suites = _mapping(root["course_suites"], "course_suites")
    output_raw = _mapping(root["output"], "output")
    _keys(output_raw, {"model_root", "log_root", "report_root"}, "output")
    stages_raw = root["stages"]
    if not isinstance(stages_raw, list):
        raise ValueError("stages must be a list")
    return CurriculumConfig(
        schema_version=_integer(root["schema_version"], "schema_version"),
        curriculum_id=_string(root["curriculum_id"], "curriculum_id"),
        long_run_config_path=_string(
            root["long_run_config_path"], "long_run_config_path"
        ),
        course_suites={
            _string(key, "suite_id"): _string(value, f"course_suites.{key}")
            for key, value in suites.items()
        },
        reserved_final_test_seeds=tuple(
            _integer(item, "reserved seed")
            for item in _list(
                root["reserved_final_test_seeds"],
                "reserved_final_test_seeds",
            )
        ),
        required_consecutive_passes=_integer(
            root["required_consecutive_passes"],
            "required_consecutive_passes",
        ),
        replay_on_promotion=_string(root["replay_on_promotion"], "replay_on_promotion"),
        output=CurriculumOutputConfig(
            model_root=_string(output_raw["model_root"], "model_root"),
            log_root=_string(output_raw["log_root"], "log_root"),
            report_root=_string(output_raw["report_root"], "report_root"),
        ),
        stages=tuple(_stage(item) for item in stages_raw),
    )


def save_curriculum_state(
    path: Path,
    state: CurriculumState,
    *,
    repository_root: Path,
) -> Path:
    return write_curriculum_report(
        path,
        state.to_dict(),
        repository_root=repository_root,
    )


def load_curriculum_state(path: Path) -> CurriculumState:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("curriculum state must contain a mapping")
    return CurriculumState.from_mapping(raw)


def validate_curriculum_configuration(
    config: CurriculumConfig,
    *,
    repository_root: Path,
    verify_courses: bool = True,
) -> dict[str, Any]:
    """Validate declared course identities and optional offline feasibility."""
    suites: dict[str, CourseSuiteConfig] = {}
    for suite_id, relative in config.course_suites.items():
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"course suite not found: {path}")
        suites[suite_id] = load_course_suite_config(path)
    unique: dict[tuple[str, str, int], CurriculumCourseRef] = {}
    for stage in config.stages:
        for course in (*stage.training_courses, *stage.validation_courses):
            suite = suites[course.suite_id]
            profile = suite.profile(course.profile_id)
            if course.base_seed not in profile.base_seeds:
                raise ValueError(f"undeclared curriculum course {course.identity!r}")
            if course.suite_id == "m13_8":
                expected_split = (
                    "training" if course.role == "training" else "validation"
                )
                if profile.split.value != expected_split:
                    raise ValueError(
                        f"M13.8 course {course.identity!r} has the wrong split"
                    )
            unique[course.identity] = course
    held_out = {
        seed
        for profile in suites["m13_3"].profiles
        if profile.split.value == "held_out"
        for seed in profile.base_seeds
    }
    if set(config.reserved_final_test_seeds) != held_out:
        raise ValueError("reserved final-test seeds do not match M13.3 held-out")
    feasibility: list[dict[str, Any]] = []
    if verify_courses:
        for identity in sorted(unique):
            course_ref = unique[identity]
            course = generate_solvable_course(
                suites[course_ref.suite_id],
                course_ref.profile_id,
                course_ref.base_seed,
                repository_root=repository_root,
            )
            result = course.result
            feasibility.append(
                {
                    "suite_id": course_ref.suite_id,
                    "profile_id": result.profile_id,
                    "base_seed": result.base_seed,
                    "accepted_candidate_seed": result.accepted_candidate_seed,
                    "attempt_index": result.attempt_index,
                    "scene_digest": result.scene_digest,
                    "occupancy_digest": result.occupancy_digest,
                    "solvability_digest": result.solvability_digest,
                    "direct_line_clear": result.path_result.direct_line_clear,
                    "reference_path_length_m": (
                        result.path_result.reference_path_length_m
                    ),
                }
            )
    blocker = next(
        (
            item
            for item in feasibility
            if item["profile_id"] == "curriculum-blocker-train"
        ),
        None,
    )
    if verify_courses and (
        blocker is None or blocker["direct_line_clear"] is not False
    ):
        raise ValueError("Stage 1 blocker must be solvable with a blocked direct line")
    for path in asdict(config.output).values():
        resolved = (repository_root / path).resolve()
        try:
            resolved.relative_to((repository_root / "results").resolve())
        except ValueError as exc:
            raise ValueError("curriculum output must remain under results/") from exc
    return {
        "schema_version": config.schema_version,
        "curriculum_id": config.curriculum_id,
        "curriculum_config_digest": config.config_digest,
        "stage_ids": tuple(item.stage_id for item in config.stages),
        "course_count": len(unique),
        "feasibility": tuple(feasibility),
        "feasibility_digest": canonical_digest(feasibility),
        "reserved_final_test_seeds": config.reserved_final_test_seeds,
        "live_training_enabled": False,
    }


def _stage(raw: Any) -> CurriculumStageConfig:
    values = _mapping(raw, "stage")
    _keys(
        values,
        {
            "stage_id",
            "index",
            "description",
            "minimum_stage_steps",
            "maximum_stage_steps",
            "validation_interval_steps",
            "training_courses",
            "validation_courses",
            "gate",
            "sampling_levels",
            "robustness",
        },
        "stage",
        optional={"sampling_levels", "robustness"},
    )
    gate_raw = _mapping(values["gate"], "gate")
    gate_fields = {
        "minimum_episode_count",
        "minimum_success_rate",
        "maximum_collision_rate",
        "maximum_timeout_rate",
        "minimum_cleanup_success_rate",
        "maximum_safety_violation_rate",
        "maximum_sensor_failure_rate",
        "subgroup_minimum_success_rates",
        "minimum_non_straight_success_rate",
        "minimum_direct_success_advantage",
        "maximum_earlier_success_regression",
        "maximum_earlier_collision_regression",
    }
    _keys(
        gate_raw,
        gate_fields,
        "gate",
        optional=gate_fields
        - {
            "minimum_episode_count",
            "minimum_success_rate",
            "maximum_collision_rate",
            "maximum_timeout_rate",
            "minimum_cleanup_success_rate",
            "maximum_safety_violation_rate",
        },
    )
    subgroup = _mapping(
        gate_raw.get("subgroup_minimum_success_rates", {}),
        "subgroup_minimum_success_rates",
    )
    levels = tuple(
        _sampling_level(item)
        for item in _list(values.get("sampling_levels", []), "sampling_levels")
    )
    robustness_raw = values.get("robustness")
    robustness = (
        _robustness(_mapping(robustness_raw, "robustness"))
        if robustness_raw is not None
        else None
    )
    return CurriculumStageConfig(
        stage_id=_string(values["stage_id"], "stage_id"),
        index=_integer(values["index"], "index"),
        description=_string(values["description"], "description"),
        minimum_stage_steps=_integer(
            values["minimum_stage_steps"], "minimum_stage_steps"
        ),
        maximum_stage_steps=_integer(
            values["maximum_stage_steps"], "maximum_stage_steps"
        ),
        validation_interval_steps=_integer(
            values["validation_interval_steps"], "validation_interval_steps"
        ),
        training_courses=tuple(
            _course(item, expected_role="training")
            for item in _flatten_course_list(
                _list(values["training_courses"], "training_courses")
            )
        ),
        validation_courses=tuple(
            _course(item, expected_role="curriculum_validation")
            for item in _flatten_course_list(
                _list(values["validation_courses"], "validation_courses")
            )
        ),
        gate=PromotionGate(
            minimum_episode_count=_integer(
                gate_raw["minimum_episode_count"], "minimum_episode_count"
            ),
            minimum_success_rate=_number(
                gate_raw["minimum_success_rate"], "minimum_success_rate"
            ),
            maximum_collision_rate=_number(
                gate_raw["maximum_collision_rate"], "maximum_collision_rate"
            ),
            maximum_timeout_rate=_number(
                gate_raw["maximum_timeout_rate"], "maximum_timeout_rate"
            ),
            minimum_cleanup_success_rate=_number(
                gate_raw["minimum_cleanup_success_rate"],
                "minimum_cleanup_success_rate",
            ),
            maximum_safety_violation_rate=_number(
                gate_raw["maximum_safety_violation_rate"],
                "maximum_safety_violation_rate",
            ),
            maximum_sensor_failure_rate=_number(
                gate_raw.get("maximum_sensor_failure_rate", 1.0),
                "maximum_sensor_failure_rate",
            ),
            subgroup_minimum_success_rates={
                _string(key, "subgroup name"): _number(value, f"subgroup {key}")
                for key, value in subgroup.items()
            },
            minimum_non_straight_success_rate=_optional_number(
                gate_raw.get("minimum_non_straight_success_rate"),
                "minimum_non_straight_success_rate",
            ),
            minimum_direct_success_advantage=_optional_number(
                gate_raw.get("minimum_direct_success_advantage"),
                "minimum_direct_success_advantage",
            ),
            maximum_earlier_success_regression=_number(
                gate_raw.get("maximum_earlier_success_regression", 1.0),
                "maximum_earlier_success_regression",
            ),
            maximum_earlier_collision_regression=_number(
                gate_raw.get("maximum_earlier_collision_regression", 1.0),
                "maximum_earlier_collision_regression",
            ),
        ),
        sampling_levels=levels,
        robustness=robustness,
    )


def _course(raw: Any, *, expected_role: str) -> CurriculumCourseRef:
    values = _mapping(raw, "course reference")
    _keys(
        values,
        {
            "suite_id",
            "profile_id",
            "base_seed",
            "role",
            "difficulty",
            "direction",
            "weight",
        },
        "course reference",
        optional={"direction", "weight"},
    )
    course = CurriculumCourseRef(
        suite_id=_string(values["suite_id"], "suite_id"),
        profile_id=_string(values["profile_id"], "profile_id"),
        base_seed=_integer(values["base_seed"], "base_seed"),
        role=_string(values["role"], "role"),
        difficulty=_string(values["difficulty"], "difficulty"),
        direction=_string(values.get("direction", "forward"), "direction"),
        weight=_number(values.get("weight", 1.0), "weight"),
    )
    if course.role != expected_role:
        raise ValueError(f"course role must be {expected_role}")
    return course


def _flatten_course_list(values: Sequence[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in values:
        if isinstance(item, list):
            flattened.extend(_flatten_course_list(item))
        else:
            flattened.append(item)
    return flattened


def _sampling_level(raw: Any) -> SamplingLevel:
    values = _mapping(raw, "sampling level")
    _keys(values, {"level_id", "difficulty_weights"}, "sampling level")
    weights = _mapping(values["difficulty_weights"], "difficulty_weights")
    return SamplingLevel(
        level_id=_string(values["level_id"], "level_id"),
        difficulty_weights={
            _string(key, "difficulty"): _number(value, f"weight {key}")
            for key, value in weights.items()
        },
    )


def _robustness(raw: Mapping[str, Any]) -> RobustnessPerturbationConfig:
    fields = {
        "lidar_noise_std_min",
        "lidar_noise_std_max",
        "sector_dropout_probability_min",
        "sector_dropout_probability_max",
        "velocity_response_scale_min",
        "velocity_response_scale_max",
        "control_duration_scale_min",
        "control_duration_scale_max",
    }
    _keys(raw, fields, "robustness")
    return RobustnessPerturbationConfig(
        **{name: _number(raw[name], name) for name in fields}
    )


def _gate_failures(
    gate: PromotionGate,
    summary: CurriculumValidationSummary,
) -> list[str]:
    failures: list[str] = []
    checks = (
        (summary.complete, "validation set is incomplete"),
        (
            summary.episode_count >= gate.minimum_episode_count,
            "minimum validation episode count not reached",
        ),
        (
            summary.success_rate >= gate.minimum_success_rate,
            "success rate is below threshold",
        ),
        (
            summary.collision_rate <= gate.maximum_collision_rate,
            "collision rate exceeds threshold",
        ),
        (
            summary.timeout_rate <= gate.maximum_timeout_rate,
            "timeout rate exceeds threshold",
        ),
        (
            summary.cleanup_success_rate >= gate.minimum_cleanup_success_rate,
            "cleanup hard gate failed",
        ),
        (
            summary.safety_violation_rate <= gate.maximum_safety_violation_rate,
            "safety violation rate exceeds threshold",
        ),
        (
            summary.sensor_failure_rate <= gate.maximum_sensor_failure_rate,
            "sensor failure rate exceeds threshold",
        ),
        (
            summary.earlier_stage_success_regression
            <= gate.maximum_earlier_success_regression,
            "earlier-stage success regressed",
        ),
        (
            summary.earlier_stage_collision_regression
            <= gate.maximum_earlier_collision_regression,
            "earlier-stage collision rate regressed",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    for subgroup, minimum in gate.subgroup_minimum_success_rates.items():
        if summary.subgroup_success_rates.get(subgroup, -1.0) < minimum:
            failures.append(f"subgroup {subgroup!r} success is below threshold")
    if gate.minimum_non_straight_success_rate is not None and (
        summary.non_straight_success_rate is None
        or summary.non_straight_success_rate < gate.minimum_non_straight_success_rate
    ):
        failures.append("non-straight route evidence is below threshold")
    if gate.minimum_direct_success_advantage is not None and (
        summary.direct_success_advantage is None
        or summary.direct_success_advantage < gate.minimum_direct_success_advantage
    ):
        failures.append("direct-controller advantage gate failed")
    return failures


def _validate_partitions(config: CurriculumConfig) -> None:
    role_by_identity: dict[tuple[str, str, int], str] = {}
    training_seeds: set[int] = set()
    validation_seeds: set[int] = set()
    for stage in config.stages:
        for course in (*stage.training_courses, *stage.validation_courses):
            existing = role_by_identity.setdefault(course.identity, course.role)
            if existing != course.role:
                raise ValueError("course identity appears in multiple split roles")
            target = training_seeds if course.role == "training" else validation_seeds
            target.add(course.base_seed)
            if course.base_seed in config.reserved_final_test_seeds:
                raise ValueError("reserved final-test seed leaked into curriculum")
    if training_seeds.intersection(validation_seeds):
        raise ValueError("training and curriculum-validation seeds must be disjoint")
    bases = sorted(training_seeds | validation_seeds)
    for left, right in zip(bases, bases[1:], strict=False):
        if left >= 20_000 and right - left < 32:
            raise ValueError("M13.8 candidate seed blocks overlap")


def _repository_relative(value: str, name: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _keys(
    raw: Mapping[str, Any],
    allowed: set[str],
    name: str,
    *,
    optional: set[str] = frozenset(),
) -> None:
    unknown = set(raw) - allowed
    missing = allowed - optional - set(raw)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty unpadded string")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)
