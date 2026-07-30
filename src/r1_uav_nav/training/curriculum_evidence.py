"""Sanitized M13.8 route, baseline, validation, and report evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from r1_uav_nav.training.long_run_state import canonical_digest

ROUTE_SHAPE_EVIDENCE_SCHEMA_VERSION = 1
CURRICULUM_BASELINE_SCHEMA_VERSION = 1
CURRICULUM_VALIDATION_SCHEMA_VERSION = 1
PROMOTION_DECISION_SCHEMA_VERSION = 1
CURRICULUM_SUMMARY_SCHEMA_VERSION = 1


class BaselineEvidenceScope(str, Enum):
    OFFLINE_FAKE = "offline_fake"
    SUPERVISED_LIVE = "supervised_live"


@dataclass(frozen=True)
class CurriculumBaselineEvidence:
    schema_version: int
    evidence_scope: BaselineEvidenceScope
    controller_id: str
    privileged: bool
    controller_config_digest: str
    suite_id: str
    profile_id: str
    base_seed: int
    accepted_candidate_seed: int
    scene_digest: str
    solvability_digest: str
    success_rate: float
    collision_rate: float
    cleanup_success_rate: float
    report_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_BASELINE_SCHEMA_VERSION:
            raise ValueError("unsupported curriculum baseline schema")
        for name in (
            "controller_config_digest",
            "scene_digest",
            "solvability_digest",
            "report_digest",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name in ("success_rate", "collision_rate", "cleanup_success_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.controller_id == "oracle" and not self.privileged:
            raise ValueError("oracle baseline evidence must be marked privileged")
        if self.controller_id in {"random", "direct"} and self.privileged:
            raise ValueError("random and direct baselines must be non-privileged")

    def require_current(
        self,
        *,
        controller_config_digest: str,
        scene_digest: str,
        solvability_digest: str,
        required_scope: BaselineEvidenceScope,
    ) -> None:
        if self.evidence_scope is not required_scope:
            raise ValueError("baseline evidence scope does not satisfy this gate")
        expected = (
            controller_config_digest,
            scene_digest,
            solvability_digest,
        )
        actual = (
            self.controller_config_digest,
            self.scene_digest,
            self.solvability_digest,
        )
        if actual != expected:
            raise ValueError("baseline evidence is stale or belongs to another course")


@dataclass(frozen=True)
class RouteShapeEvidence:
    schema_version: int
    point_count: int
    straight_line_distance_m: float
    travelled_path_length_m: float
    travelled_to_direct_ratio: float
    maximum_perpendicular_deviation_m: float
    maximum_lateral_deviation_m: float
    maximum_vertical_deviation_m: float
    minimum_clearance_m: float | None
    initial_relative_position: tuple[float, float, float]
    final_relative_position: tuple[float, float, float]
    trajectory_digest: str
    non_straight_qualified: bool


class TrajectorySummaryCollector:
    """Accumulate bounded route-shape metrics without storing a trajectory."""

    def __init__(
        self,
        *,
        start_relative: Sequence[float],
        goal_relative: Sequence[float],
    ) -> None:
        self._start = _point(start_relative, "start_relative")
        self._goal = _point(goal_relative, "goal_relative")
        self._direct = self._goal - self._start
        self._direct_distance = float(np.linalg.norm(self._direct))
        if self._direct_distance <= 1e-9:
            raise ValueError("route-shape endpoints must be distinct")
        self._point_count = 0
        self._travelled = 0.0
        self._max_perpendicular = 0.0
        self._max_lateral = 0.0
        self._max_vertical = 0.0
        self._minimum_clearance: float | None = None
        self._reported_path_length: float | None = None
        self._previous: np.ndarray | None = None
        self._initial: np.ndarray | None = None
        self._final: np.ndarray | None = None
        self._digest = hashlib.sha256()

    def add(
        self,
        *,
        observation: np.ndarray,
        position_scales_m: Sequence[float],
        info: Mapping[str, Any],
    ) -> None:
        values = np.asarray(observation, dtype=np.float64)
        scales = _point(position_scales_m, "position_scales_m")
        if values.shape != (83,) or not np.all(np.isfinite(values)):
            raise ValueError("trajectory observation must be finite with shape (83,)")
        if np.any(scales <= 0):
            raise ValueError("position scales must be positive")
        position = values[:3] * scales
        if self._previous is not None:
            self._travelled += float(np.linalg.norm(position - self._previous))
        if self._initial is None:
            self._initial = position.copy()
        self._previous = position.copy()
        self._final = position.copy()
        self._point_count += 1

        along = float(np.dot(position - self._start, self._direct)) / (
            self._direct_distance**2
        )
        projection = self._start + np.clip(along, 0.0, 1.0) * self._direct
        delta = position - projection
        self._max_perpendicular = max(
            self._max_perpendicular, float(np.linalg.norm(delta))
        )
        self._max_lateral = max(self._max_lateral, float(np.linalg.norm(delta[:2])))
        self._max_vertical = max(self._max_vertical, abs(float(delta[2])))
        clearance = info.get("reward_clearance_m")
        if clearance is None:
            clearance = info.get("minimum_lidar_clearance_m")
        if clearance is not None:
            measured = float(clearance)
            if not math.isfinite(measured) or measured < 0:
                raise ValueError("trajectory clearance must be finite and non-negative")
            self._minimum_clearance = (
                measured
                if self._minimum_clearance is None
                else min(self._minimum_clearance, measured)
            )
        reported_path = info.get("path_length_m")
        if reported_path is not None:
            measured_path = float(reported_path)
            if not math.isfinite(measured_path) or measured_path < 0:
                raise ValueError("reported path length must be finite and non-negative")
            if (
                self._reported_path_length is not None
                and measured_path + 1e-9 < self._reported_path_length
            ):
                raise ValueError("reported path length must not decrease")
            self._reported_path_length = measured_path
        quantized = np.round(position, decimals=4).astype("<f8", copy=False)
        self._digest.update(quantized.tobytes())

    def result(
        self,
        *,
        successful: bool,
        collision: bool,
        safety_violation: bool,
    ) -> RouteShapeEvidence:
        if self._initial is None or self._final is None:
            raise ValueError("trajectory summary requires at least one point")
        reported_path = (
            self._reported_path_length
            if self._reported_path_length is not None
            else self._travelled
        )
        ratio = reported_path / self._direct_distance
        qualified = (
            successful
            and not collision
            and not safety_violation
            and ratio >= 1.05
            and self._max_perpendicular >= 1.0
        )
        return RouteShapeEvidence(
            schema_version=ROUTE_SHAPE_EVIDENCE_SCHEMA_VERSION,
            point_count=self._point_count,
            straight_line_distance_m=self._direct_distance,
            travelled_path_length_m=reported_path,
            travelled_to_direct_ratio=ratio,
            maximum_perpendicular_deviation_m=self._max_perpendicular,
            maximum_lateral_deviation_m=self._max_lateral,
            maximum_vertical_deviation_m=self._max_vertical,
            minimum_clearance_m=self._minimum_clearance,
            initial_relative_position=tuple(float(v) for v in self._initial),
            final_relative_position=tuple(float(v) for v in self._final),
            trajectory_digest=self._digest.hexdigest(),
            non_straight_qualified=qualified,
        )


@dataclass(frozen=True)
class CurriculumValidationSummary:
    schema_version: int
    stage_id: str
    global_timesteps: int
    episode_count: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    cleanup_success_rate: float
    safety_violation_rate: float
    sensor_failure_rate: float
    subgroup_success_rates: dict[str, float]
    non_straight_success_rate: float | None = None
    direct_success_advantage: float | None = None
    earlier_stage_success_regression: float = 0.0
    earlier_stage_collision_regression: float = 0.0
    complete: bool = True
    validation_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported curriculum validation schema")
        if self.episode_count <= 0:
            raise ValueError("validation summary requires episodes")
        for name in (
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "cleanup_success_rate",
            "safety_violation_rate",
            "sensor_failure_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.validation_digest and len(self.validation_digest) != 64:
            raise ValueError("validation_digest must be SHA-256")

    def with_digest(self) -> CurriculumValidationSummary:
        if self.validation_digest:
            return self
        values = asdict(self)
        values["validation_digest"] = ""
        return CurriculumValidationSummary(
            **{**values, "validation_digest": canonical_digest(values)}
        )


def write_curriculum_report(
    path: Path,
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    """Atomically write sanitized evidence below results/."""
    resolved = path.resolve()
    try:
        resolved.relative_to((repository_root / "results").resolve())
    except ValueError as exc:
        raise ValueError("curriculum report must remain under results/") from exc
    forbidden = {
        "raw_point_cloud",
        "point_cloud",
        "astar_path",
        "reference_path",
        "obstacles",
        "settings_path",
    }
    if forbidden.intersection(_all_keys(payload)):
        raise ValueError("curriculum report contains a prohibited payload")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)
    return resolved


def baseline_report_digest(payload: Mapping[str, Any]) -> str:
    return canonical_digest(payload)


def require_baseline_set(
    evidence: Sequence[CurriculumBaselineEvidence],
    *,
    required_controller_ids: Sequence[str],
    required_scope: BaselineEvidenceScope,
    expected_controller_config_digests: Mapping[str, str],
    scene_digest: str,
    solvability_digest: str,
) -> tuple[CurriculumBaselineEvidence, ...]:
    """Require one current baseline per controller without accepting stale data."""
    by_controller: dict[str, CurriculumBaselineEvidence] = {}
    for item in evidence:
        if item.controller_id in by_controller:
            raise ValueError(f"duplicate baseline for {item.controller_id!r}")
        by_controller[item.controller_id] = item
    missing = set(required_controller_ids) - set(by_controller)
    if missing:
        raise ValueError(f"missing required baselines: {sorted(missing)}")
    selected: list[CurriculumBaselineEvidence] = []
    for controller_id in required_controller_ids:
        if controller_id not in expected_controller_config_digests:
            raise ValueError(f"missing controller digest for {controller_id!r}")
        item = by_controller[controller_id]
        item.require_current(
            controller_config_digest=expected_controller_config_digests[controller_id],
            scene_digest=scene_digest,
            solvability_digest=solvability_digest,
            required_scope=required_scope,
        )
        selected.append(item)
    return tuple(selected)


def _point(values: Sequence[float], name: str) -> np.ndarray:
    point = np.asarray(values, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain three finite values")
    return point


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(item) for item in value}
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()
