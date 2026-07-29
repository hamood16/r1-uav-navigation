from __future__ import annotations

from dataclasses import replace

import pytest

from r1_uav_nav.training.long_run_state import canonical_digest
from r1_uav_nav.training.throughput_benchmark import (
    ThroughputBenchmarkConfig,
    ThroughputEpisodeSample,
    changed_benchmark_knob,
    summarize_throughput,
)


def _config() -> ThroughputBenchmarkConfig:
    return ThroughputBenchmarkConfig(
        variant_id="baseline",
        environment_config_digest=canonical_digest("environment"),
        control_duration_s=0.2,
        checkpoint_interval_steps=1000,
    )


def _sample(*, cleanup: bool = True) -> ThroughputEpisodeSample:
    return ThroughputEpisodeSample(
        reset_latency_s=1.0,
        action_step_latencies_s=(0.2, 0.3, 0.4),
        lidar_latencies_s=(0.05, 0.06, 0.07),
        episode_wall_time_s=2.0,
        active_step_time_s=0.9,
        checkpoint_overhead_s=0.1,
        logging_overhead_s=0.02,
        cleanup_success=cleanup,
        cleanup_status="success" if cleanup else "failed",
    )


def test_report_records_latency_throughput_and_cleanup() -> None:
    report = summarize_throughput(_config(), (_sample(),))
    assert report.step_count == 3
    assert report.action_step_latency.mean_s == pytest.approx(0.3)
    assert report.lidar_latency is not None
    assert report.checkpoint_overhead.maximum_s == pytest.approx(0.1)
    assert report.active_steps_per_second == pytest.approx(3 / 0.9)
    assert report.accepted


def test_cleanup_failure_blocks_benchmark_acceptance() -> None:
    report = summarize_throughput(_config(), (_sample(cleanup=False),))
    assert not report.cleanup_success
    assert not report.accepted


def test_comparison_requires_exactly_one_changed_knob() -> None:
    baseline = _config()
    candidate = replace(
        baseline,
        variant_id="less-logging",
        console_logging="none",
    )
    assert changed_benchmark_knob(baseline, candidate) == "console_logging"
    with pytest.raises(ValueError, match="exactly one"):
        changed_benchmark_knob(
            baseline,
            replace(
                candidate,
                control_duration_s=0.1,
            ),
        )
    with pytest.raises(ValueError, match="exactly one"):
        changed_benchmark_knob(
            baseline,
            replace(baseline, variant_id="renamed"),
        )


def test_phase_a_rejects_simulator_speedup() -> None:
    with pytest.raises(ValueError, match="clock_scale=1.0"):
        replace(_config(), simulator_clock_scale=2.0)
