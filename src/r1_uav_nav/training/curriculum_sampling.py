"""Deterministic M13.8 curriculum course and robustness sampling."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from r1_uav_nav.training.long_run_state import canonical_digest


@dataclass(frozen=True)
class CurriculumCourseRef:
    """One course identity with an explicit curriculum split role."""

    suite_id: str
    profile_id: str
    base_seed: int
    role: str
    difficulty: str
    direction: str = "forward"
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("suite_id", "profile_id", "role", "difficulty", "direction"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty unpadded string")
        if self.role not in {
            "training",
            "curriculum_validation",
            "final_test_reserved",
        }:
            raise ValueError("unsupported curriculum course role")
        if (
            not isinstance(self.base_seed, int)
            or isinstance(self.base_seed, bool)
            or self.base_seed < 0
        ):
            raise ValueError("base_seed must be a non-negative integer")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("course weight must be finite and positive")

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.suite_id, self.profile_id, self.base_seed


@dataclass(frozen=True)
class CurriculumSamplerState:
    """JSON-compatible local NumPy generator state."""

    schema_version: int
    bit_generator: str
    state: dict[str, Any]
    sample_count: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported curriculum sampler-state schema")
        if self.bit_generator != "PCG64":
            raise ValueError("M13.8 supports only the PCG64 bit generator")
        if not isinstance(self.state, dict) or not self.state:
            raise ValueError("sampler state must be a non-empty mapping")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")


class DeterministicCurriculumSampler:
    """Sample stable course references without touching global RNG state."""

    def __init__(
        self,
        courses: Sequence[CurriculumCourseRef],
        *,
        seed: int,
        state: CurriculumSamplerState | None = None,
        difficulty_weights: Mapping[str, float] | None = None,
    ) -> None:
        if not courses:
            raise ValueError("curriculum sampler requires at least one course")
        identities = [item.identity for item in courses]
        if len(identities) != len(set(identities)):
            raise ValueError("curriculum sampler course identities must be unique")
        self._courses = tuple(sorted(courses, key=lambda item: item.identity))
        self._difficulty_weights = (
            {str(name): float(weight) for name, weight in difficulty_weights.items()}
            if difficulty_weights is not None
            else None
        )
        if self._difficulty_weights is not None:
            missing = {item.difficulty for item in self._courses} - set(
                self._difficulty_weights
            )
            if missing:
                raise ValueError(
                    f"difficulty weights are missing groups: {sorted(missing)}"
                )
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self._difficulty_weights.values()
            ):
                raise ValueError("difficulty weights must be finite and positive")
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._sample_count = 0
        if state is not None:
            self._rng.bit_generator.state = copy.deepcopy(state.state)
            self._sample_count = state.sample_count

    def sample(self) -> CurriculumCourseRef:
        candidates = self._courses
        if self._difficulty_weights is not None:
            groups = tuple(sorted({item.difficulty for item in self._courses}))
            group_weights = np.asarray(
                [self._difficulty_weights[item] for item in groups],
                dtype=np.float64,
            )
            group = groups[
                int(
                    self._rng.choice(
                        len(groups),
                        p=group_weights / float(np.sum(group_weights)),
                    )
                )
            ]
            candidates = tuple(
                item for item in self._courses if item.difficulty == group
            )
        weights = np.asarray([item.weight for item in candidates], dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
        index = int(self._rng.choice(len(candidates), p=probabilities))
        self._sample_count += 1
        return candidates[index]

    def snapshot(self) -> CurriculumSamplerState:
        return CurriculumSamplerState(
            schema_version=1,
            bit_generator=type(self._rng.bit_generator).__name__,
            state=copy.deepcopy(self._rng.bit_generator.state),
            sample_count=self._sample_count,
        )


@dataclass(frozen=True)
class RobustnessPerturbationConfig:
    """Offline-only Stage 5 perturbation bounds."""

    lidar_noise_std_min: float = 0.0
    lidar_noise_std_max: float = 0.02
    sector_dropout_probability_min: float = 0.0
    sector_dropout_probability_max: float = 0.05
    velocity_response_scale_min: float = 0.9
    velocity_response_scale_max: float = 1.1
    control_duration_scale_min: float = 0.9
    control_duration_scale_max: float = 1.1

    def __post_init__(self) -> None:
        for lower_name, upper_name in (
            ("lidar_noise_std_min", "lidar_noise_std_max"),
            (
                "sector_dropout_probability_min",
                "sector_dropout_probability_max",
            ),
            ("velocity_response_scale_min", "velocity_response_scale_max"),
            ("control_duration_scale_min", "control_duration_scale_max"),
        ):
            lower = float(getattr(self, lower_name))
            upper = float(getattr(self, upper_name))
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("perturbation bounds must be finite")
            if lower < 0 or upper < lower:
                raise ValueError("perturbation bounds must be ordered and non-negative")
        if self.sector_dropout_probability_max > 1:
            raise ValueError("sector dropout probability must not exceed one")
        if self.velocity_response_scale_min <= 0:
            raise ValueError("velocity response scale must be positive")
        if self.control_duration_scale_min <= 0:
            raise ValueError("control duration scale must be positive")


@dataclass(frozen=True)
class RobustnessPerturbation:
    """One deterministic offline policy-view perturbation."""

    lidar_noise_std: float
    sector_dropout_probability: float
    velocity_response_scale: float
    control_duration_scale: float
    seed: int

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


class RobustnessPerturbationSampler:
    """Generate Stage 5 perturbations with a private RNG."""

    def __init__(
        self,
        config: RobustnessPerturbationConfig,
        *,
        seed: int,
    ) -> None:
        self.config = config
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def sample(self) -> RobustnessPerturbation:
        child_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        return RobustnessPerturbation(
            lidar_noise_std=float(
                self._rng.uniform(
                    self.config.lidar_noise_std_min,
                    self.config.lidar_noise_std_max,
                )
            ),
            sector_dropout_probability=float(
                self._rng.uniform(
                    self.config.sector_dropout_probability_min,
                    self.config.sector_dropout_probability_max,
                )
            ),
            velocity_response_scale=float(
                self._rng.uniform(
                    self.config.velocity_response_scale_min,
                    self.config.velocity_response_scale_max,
                )
            ),
            control_duration_scale=float(
                self._rng.uniform(
                    self.config.control_duration_scale_min,
                    self.config.control_duration_scale_max,
                )
            ),
            seed=child_seed,
        )


def apply_policy_view_perturbation(
    observation: np.ndarray,
    perturbation: RobustnessPerturbation,
) -> np.ndarray:
    """Return a perturbed copy of the 83-value policy observation.

    This helper never changes the source observation or any safety evidence.
    """
    source = np.asarray(observation, dtype=np.float32)
    if source.shape != (83,) or not np.all(np.isfinite(source)):
        raise ValueError("policy observation must be finite with shape (83,)")
    result = source.copy()
    rng = np.random.Generator(np.random.PCG64(perturbation.seed))
    lidar = result[10:82].astype(np.float64)
    if perturbation.lidar_noise_std:
        lidar += rng.normal(0.0, perturbation.lidar_noise_std, size=lidar.shape)
    if perturbation.sector_dropout_probability:
        dropout = rng.random(lidar.shape) < perturbation.sector_dropout_probability
        lidar[dropout] = 1.0
    result[10:82] = np.clip(lidar, 0.0, 1.0).astype(np.float32)
    return result


def sampler_state_from_mapping(raw: Mapping[str, Any]) -> CurriculumSamplerState:
    return CurriculumSamplerState(
        schema_version=int(raw["schema_version"]),
        bit_generator=str(raw["bit_generator"]),
        state=dict(raw["state"]),
        sample_count=int(raw["sample_count"]),
    )
