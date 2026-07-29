"""Simulator-independent M13.7 throughput evidence and comparison helpers."""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from r1_uav_nav.training.long_run_state import canonical_digest, utc_now

THROUGHPUT_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ThroughputBenchmarkConfig:
    """One benchmark variant and its operator-controlled knobs."""

    variant_id: str
    environment_config_digest: str
    control_duration_s: float
    checkpoint_interval_steps: int
    console_logging: str = "minimal"
    tracing_enabled: bool = False
    simulator_clock_scale: float = 1.0
    graphics_preset: str = "unchanged"

    def __post_init__(self) -> None:
        if not self.variant_id or self.variant_id != self.variant_id.strip():
            raise ValueError("variant_id must not be empty or padded")
        if len(self.environment_config_digest) != 64:
            raise ValueError("environment_config_digest must be SHA-256")
        if not math.isfinite(self.control_duration_s) or self.control_duration_s <= 0:
            raise ValueError("control_duration_s must be finite and positive")
        if self.checkpoint_interval_steps <= 0:
            raise ValueError("checkpoint_interval_steps must be positive")
        if self.simulator_clock_scale != 1.0:
            raise ValueError("Phase A records only simulator_clock_scale=1.0 evidence")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    @property
    def knobs(self) -> dict[str, Any]:
        return {
            "control_duration_s": self.control_duration_s,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "console_logging": self.console_logging,
            "tracing_enabled": self.tracing_enabled,
            "simulator_clock_scale": self.simulator_clock_scale,
            "graphics_preset": self.graphics_preset,
        }


@dataclass(frozen=True)
class ThroughputEpisodeSample:
    """Raw aggregate latencies for one bounded benchmark episode."""

    reset_latency_s: float
    action_step_latencies_s: tuple[float, ...]
    lidar_latencies_s: tuple[float, ...]
    episode_wall_time_s: float
    active_step_time_s: float
    checkpoint_overhead_s: float
    logging_overhead_s: float
    cleanup_success: bool
    cleanup_status: str

    def __post_init__(self) -> None:
        values = (
            self.reset_latency_s,
            self.episode_wall_time_s,
            self.active_step_time_s,
            self.checkpoint_overhead_s,
            self.logging_overhead_s,
            *self.action_step_latencies_s,
            *self.lidar_latencies_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("throughput timings must be finite and non-negative")
        if not self.action_step_latencies_s:
            raise ValueError("at least one action-step latency is required")
        if not self.cleanup_status.strip():
            raise ValueError("cleanup_status must not be empty")


@dataclass(frozen=True)
class LatencySummary:
    """Stable percentile summary without wall-clock assertions."""

    count: int
    minimum_s: float
    mean_s: float
    median_s: float
    p95_s: float
    maximum_s: float


@dataclass(frozen=True)
class ThroughputBenchmarkReport:
    """Schema-versioned throughput and cleanup evidence."""

    schema_version: int
    report_id: str
    created_at: str
    variant_id: str
    configuration_digest: str
    environment_config_digest: str
    knobs: dict[str, Any]
    episode_count: int
    step_count: int
    reset_latency: LatencySummary
    action_step_latency: LatencySummary
    lidar_latency: LatencySummary | None
    lidar_latency_unavailable_reason: str | None
    episode_wall_time: LatencySummary
    checkpoint_overhead: LatencySummary
    logging_overhead: LatencySummary
    active_steps_per_second: float
    end_to_end_steps_per_second: float
    cleanup_success: bool
    cleanup_statuses: tuple[str, ...]
    accepted: bool
    limitations: tuple[str, ...]


def summarize_throughput(
    config: ThroughputBenchmarkConfig,
    samples: Sequence[ThroughputEpisodeSample],
) -> ThroughputBenchmarkReport:
    """Build deterministic benchmark evidence from measured samples."""
    if not samples:
        raise ValueError("at least one throughput sample is required")
    action_latencies = tuple(
        value for sample in samples for value in sample.action_step_latencies_s
    )
    lidar_latencies = tuple(
        value for sample in samples for value in sample.lidar_latencies_s
    )
    step_count = len(action_latencies)
    active_seconds = sum(sample.active_step_time_s for sample in samples)
    wall_seconds = sum(sample.episode_wall_time_s for sample in samples)
    cleanup_success = all(sample.cleanup_success for sample in samples)
    identity = {
        "config": asdict(config),
        "samples": [asdict(sample) for sample in samples],
    }
    return ThroughputBenchmarkReport(
        schema_version=THROUGHPUT_REPORT_SCHEMA_VERSION,
        report_id=canonical_digest(identity)[:16],
        created_at=utc_now(),
        variant_id=config.variant_id,
        configuration_digest=config.digest,
        environment_config_digest=config.environment_config_digest,
        knobs=config.knobs,
        episode_count=len(samples),
        step_count=step_count,
        reset_latency=_summarize(tuple(sample.reset_latency_s for sample in samples)),
        action_step_latency=_summarize(action_latencies),
        lidar_latency=_summarize(lidar_latencies) if lidar_latencies else None,
        lidar_latency_unavailable_reason=(
            None if lidar_latencies else "LiDAR latency was not instrumented."
        ),
        episode_wall_time=_summarize(
            tuple(sample.episode_wall_time_s for sample in samples)
        ),
        checkpoint_overhead=_summarize(
            tuple(sample.checkpoint_overhead_s for sample in samples)
        ),
        logging_overhead=_summarize(
            tuple(sample.logging_overhead_s for sample in samples)
        ),
        active_steps_per_second=(
            step_count / active_seconds if active_seconds > 0 else 0.0
        ),
        end_to_end_steps_per_second=(
            step_count / wall_seconds if wall_seconds > 0 else 0.0
        ),
        cleanup_success=cleanup_success,
        cleanup_statuses=tuple(sample.cleanup_status for sample in samples),
        accepted=cleanup_success,
        limitations=(
            "Phase A uses fake/offline timing and makes no Colosseum throughput claim.",
            "Simulator clock scale is fixed at 1.0.",
        ),
    )


def changed_benchmark_knob(
    baseline: ThroughputBenchmarkConfig,
    candidate: ThroughputBenchmarkConfig,
) -> str:
    """Require a candidate comparison to change exactly one benchmark knob."""
    if baseline.environment_config_digest != candidate.environment_config_digest:
        raise ValueError("benchmark environment digests must match")
    changed = [
        name for name, value in baseline.knobs.items() if candidate.knobs[name] != value
    ]
    if len(changed) != 1:
        raise ValueError("benchmark comparison must change exactly one knob")
    return changed[0]


def save_throughput_report(
    report: ThroughputBenchmarkReport,
    output_path: Path,
) -> None:
    """Atomically persist a benchmark report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                asdict(report),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_throughput_report(path: Path) -> Mapping[str, Any]:
    """Load report evidence for offline summarization."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("throughput report must contain a JSON object")
    if raw.get("schema_version") != THROUGHPUT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported throughput report schema")
    return raw


def _summarize(values: Sequence[float]) -> LatencySummary:
    if not values:
        raise ValueError("latency summary requires at least one value")
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LatencySummary(
        count=len(ordered),
        minimum_s=ordered[0],
        mean_s=statistics.fmean(ordered),
        median_s=statistics.median(ordered),
        p95_s=ordered[p95_index],
        maximum_s=ordered[-1],
    )


__all__ = [
    "LatencySummary",
    "ThroughputBenchmarkConfig",
    "ThroughputBenchmarkReport",
    "ThroughputEpisodeSample",
    "changed_benchmark_knob",
    "load_throughput_report",
    "save_throughput_report",
    "summarize_throughput",
]
