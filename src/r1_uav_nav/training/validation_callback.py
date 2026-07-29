"""Deterministic M13.7 validation ranking and best-model promotion."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from r1_uav_nav.training.long_run_state import (
    canonical_digest,
    capture_rng_state,
    restore_rng_state,
    utc_now,
)

VALIDATION_REPORT_SCHEMA_VERSION = 1
VALIDATION_STATE_SCHEMA_VERSION = 1


class ValidationTier(str, Enum):
    """Validation tiers with distinct checkpoint-promotion authority."""

    MONITORING = "monitoring"
    PROMOTION = "promotion"


@dataclass(frozen=True)
class ValidationCase:
    """One fixed course and evaluation seed."""

    profile_id: str
    base_seed: int
    episode_seed: int

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id != self.profile_id.strip():
            raise ValueError("validation profile_id must not be empty or padded")
        for name in ("base_seed", "episode_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.profile_id, self.base_seed, self.episode_seed


DEFAULT_MONITORING_CASES = (
    ValidationCase("empty", 0, 100_000),
    ValidationCase("easy", 1100, 101_100),
    ValidationCase("medium", 2100, 102_100),
)
DEFAULT_PROMOTION_CASES = (
    ValidationCase("empty", 0, 200_000),
    ValidationCase("held-out-reverse", 9100, 209_100),
    ValidationCase("held-out-elevated", 10100, 210_100),
)


@dataclass(frozen=True)
class ValidationEpisodeResult:
    """Sanitized deterministic evaluation result for one case."""

    profile_id: str
    base_seed: int
    episode_seed: int
    episode_return: float
    success: bool
    collision: bool
    landing_success: bool
    cleanup_success: bool
    final_distance_m: float
    travelled_path_length_m: float
    reference_path_length_m: float | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "episode_return",
            "final_distance_m",
            "travelled_path_length_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.final_distance_m < 0 or self.travelled_path_length_m < 0:
            raise ValueError("validation distances must not be negative")
        if self.reference_path_length_m is not None and (
            not math.isfinite(float(self.reference_path_length_m))
            or self.reference_path_length_m < 0
        ):
            raise ValueError("reference_path_length_m must be finite and non-negative")

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.profile_id, self.base_seed, self.episode_seed

    @property
    def safe_cleanup(self) -> bool:
        return self.landing_success and self.cleanup_success

    @property
    def path_efficiency(self) -> float | None:
        if (
            not self.success
            or self.reference_path_length_m is None
            or self.travelled_path_length_m <= 0
        ):
            return None
        return min(
            1.0,
            max(0.0, self.reference_path_length_m / self.travelled_path_length_m),
        )


@dataclass(frozen=True, order=True)
class ValidationRank:
    """Lexicographic best-model rank; larger is always better."""

    success_rate: float
    negative_collision_rate: float
    safe_cleanup_rate: float
    mean_path_efficiency: float
    negative_mean_final_distance_m: float
    mean_return: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in self.as_tuple()):
            raise ValueError("validation rank values must be finite")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.success_rate,
            self.negative_collision_rate,
            self.safe_cleanup_rate,
            self.mean_path_efficiency,
            self.negative_mean_final_distance_m,
            self.mean_return,
        )


@dataclass(frozen=True)
class ValidationSummary:
    """Complete tier result and promotion decision evidence."""

    schema_version: int
    validation_id: str
    created_at: str
    tier: ValidationTier
    global_timesteps: int
    expected_case_count: int
    results: tuple[ValidationEpisodeResult, ...]
    complete: bool
    cleanup_gate_passed: bool
    eligible_for_promotion: bool
    promoted: bool
    rank: ValidationRank
    validation_digest: str
    failures: tuple[str, ...]


class ValidationEvaluator(Protocol):
    """Injected evaluator boundary; Phase A implementations use fake envs."""

    def __call__(
        self,
        *,
        model: Any,
        tier: ValidationTier,
        cases: tuple[ValidationCase, ...],
        global_timesteps: int,
    ) -> Sequence[ValidationEpisodeResult]: ...


def summarize_validation(
    *,
    tier: ValidationTier,
    cases: Sequence[ValidationCase],
    results: Sequence[ValidationEpisodeResult],
    global_timesteps: int,
    best_rank: ValidationRank | None = None,
) -> ValidationSummary:
    """Validate a complete result set and calculate deterministic promotion rank."""
    expected = tuple(item.identity for item in cases)
    actual = tuple(item.identity for item in results)
    failures: list[str] = []
    if len(actual) != len(set(actual)):
        failures.append("duplicate validation result identity")
    if set(actual) != set(expected) or len(actual) != len(expected):
        failures.append("validation result set is incomplete or unexpected")
    complete = not failures
    cleanup_gate = complete and all(item.safe_cleanup for item in results)
    if complete and not cleanup_gate:
        failures.append("landing or cleanup evidence failed or was inconclusive")
    rank = calculate_validation_rank(results)
    eligible = tier is ValidationTier.PROMOTION and complete and cleanup_gate
    promoted = eligible and (best_rank is None or rank > best_rank)
    identity = {
        "tier": tier.value,
        "global_timesteps": global_timesteps,
        "expected": expected,
        "results": [asdict(item) for item in results],
    }
    return ValidationSummary(
        schema_version=VALIDATION_REPORT_SCHEMA_VERSION,
        validation_id=canonical_digest(identity)[:16],
        created_at=utc_now(),
        tier=tier,
        global_timesteps=global_timesteps,
        expected_case_count=len(expected),
        results=tuple(results),
        complete=complete,
        cleanup_gate_passed=cleanup_gate,
        eligible_for_promotion=eligible,
        promoted=promoted,
        rank=rank,
        validation_digest=canonical_digest(identity),
        failures=tuple(failures),
    )


def calculate_validation_rank(
    results: Sequence[ValidationEpisodeResult],
) -> ValidationRank:
    """Calculate the approved lexicographic validation rank."""
    if not results:
        return ValidationRank(0.0, -1.0, 0.0, 0.0, -1e300, -1e300)
    count = len(results)
    successes = sum(item.success for item in results)
    collisions = sum(item.collision for item in results)
    cleanups = sum(item.safe_cleanup for item in results)
    efficiencies = [
        value for item in results if (value := item.path_efficiency) is not None
    ]
    return ValidationRank(
        success_rate=successes / count,
        negative_collision_rate=-(collisions / count),
        safe_cleanup_rate=cleanups / count,
        mean_path_efficiency=(
            sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        ),
        negative_mean_final_distance_m=-(
            sum(item.final_distance_m for item in results) / count
        ),
        mean_return=sum(item.episode_return for item in results) / count,
    )


class DeterministicValidationCallback(BaseCallback):
    """Schedule injected deterministic validation and promotion at rollout bounds."""

    def __init__(
        self,
        *,
        evaluator: ValidationEvaluator,
        report_directory: Path,
        model_output_directory: Path,
        monitoring_cases: Sequence[ValidationCase] = DEFAULT_MONITORING_CASES,
        promotion_cases: Sequence[ValidationCase] = DEFAULT_PROMOTION_CASES,
        monitoring_interval: int = 5_000,
        promotion_interval: int = 20_000,
        initial_best_rank: ValidationRank | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if monitoring_interval <= 0 or promotion_interval <= 0:
            raise ValueError("validation intervals must be positive")
        self.evaluator = evaluator
        self.report_directory = Path(report_directory)
        self.model_output_directory = Path(model_output_directory)
        self.monitoring_cases = tuple(monitoring_cases)
        self.promotion_cases = tuple(promotion_cases)
        self.monitoring_interval = monitoring_interval
        self.promotion_interval = promotion_interval
        self.best_rank = initial_best_rank
        self.next_monitoring_step = monitoring_interval
        self.next_promotion_step = promotion_interval
        self.summaries: list[ValidationSummary] = []
        self._monitoring_due = False
        self._promotion_due = False

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_monitoring_step:
            self._monitoring_due = True
        if self.num_timesteps >= self.next_promotion_step:
            self._promotion_due = True
        return True

    def _on_rollout_end(self) -> None:
        if self._monitoring_due:
            self._run_tier(ValidationTier.MONITORING, self.monitoring_cases)
            while self.next_monitoring_step <= self.num_timesteps:
                self.next_monitoring_step += self.monitoring_interval
            self._monitoring_due = False
        if self._promotion_due:
            self._run_tier(ValidationTier.PROMOTION, self.promotion_cases)
            while self.next_promotion_step <= self.num_timesteps:
                self.next_promotion_step += self.promotion_interval
            self._promotion_due = False

    def run_now(self, tier: ValidationTier) -> ValidationSummary:
        """Run one tier directly for offline tests and bounded tooling."""
        cases = (
            self.monitoring_cases
            if tier is ValidationTier.MONITORING
            else self.promotion_cases
        )
        return self._run_tier(tier, cases)

    def _run_tier(
        self,
        tier: ValidationTier,
        cases: tuple[ValidationCase, ...],
    ) -> ValidationSummary:
        rng_evidence = capture_rng_state()
        try:
            results = tuple(
                self.evaluator(
                    model=self.model,
                    tier=tier,
                    cases=cases,
                    global_timesteps=self.num_timesteps,
                )
            )
        finally:
            restore_rng_state(rng_evidence)
        summary = summarize_validation(
            tier=tier,
            cases=cases,
            results=results,
            global_timesteps=self.num_timesteps,
            best_rank=self.best_rank,
        )
        if summary.promoted:
            self.model_output_directory.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_output_directory / "best_validation_model.zip")
            self.best_rank = summary.rank
            _atomic_json(
                self.model_output_directory / "best_validation_state.json",
                {
                    "schema_version": VALIDATION_STATE_SCHEMA_VERSION,
                    "validation_id": summary.validation_id,
                    "global_timesteps": summary.global_timesteps,
                    "rank": summary.rank.as_tuple(),
                    "validation_digest": summary.validation_digest,
                },
            )
        self.summaries.append(summary)
        save_validation_summary(summary, self.report_directory)
        return summary


def save_validation_summary(
    summary: ValidationSummary,
    report_directory: Path,
) -> Path:
    """Persist immutable JSON and atomically rebuild a compact CSV index."""
    report_directory.mkdir(parents=True, exist_ok=True)
    path = report_directory / (
        f"validation_{summary.global_timesteps:012d}_"
        f"{summary.tier.value}_{summary.validation_id}.json"
    )
    repeat = 0
    while path.exists():
        repeat += 1
        path = report_directory / (
            f"validation_{summary.global_timesteps:012d}_"
            f"{summary.tier.value}_{summary.validation_id}"
            f"_repeat-{repeat:03d}.json"
        )
    _atomic_json(path, asdict(summary))
    records: list[dict[str, Any]] = []
    for report in sorted(report_directory.glob("validation_*.json")):
        raw = json.loads(report.read_text(encoding="utf-8"))
        rank = raw["rank"]
        records.append(
            {
                "validation_id": raw["validation_id"],
                "global_timesteps": raw["global_timesteps"],
                "tier": raw["tier"],
                "complete": raw["complete"],
                "cleanup_gate_passed": raw["cleanup_gate_passed"],
                "promoted": raw["promoted"],
                "success_rate": rank["success_rate"],
                "collision_rate": -float(rank["negative_collision_rate"]),
                "safe_cleanup_rate": rank["safe_cleanup_rate"],
                "mean_path_efficiency": rank["mean_path_efficiency"],
                "mean_final_distance_m": -float(rank["negative_mean_final_distance_m"]),
                "mean_return": rank["mean_return"],
            }
        )
    _atomic_csv(report_directory / "validation_summary.csv", records)
    return path


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fieldnames = list(records[0]) if records else ["validation_id"]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "DEFAULT_MONITORING_CASES",
    "DEFAULT_PROMOTION_CASES",
    "DeterministicValidationCallback",
    "ValidationCase",
    "ValidationEpisodeResult",
    "ValidationRank",
    "ValidationSummary",
    "ValidationTier",
    "calculate_validation_rank",
    "save_validation_summary",
    "summarize_validation",
]
