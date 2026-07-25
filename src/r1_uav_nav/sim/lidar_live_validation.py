"""Supervised M13.4 live validation orchestration with offline-first gates."""

from __future__ import annotations

import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np

from r1_uav_nav.envs.colosseum_lidar_uav_env import (
    ColosseumLidarUAVEnv,
    ColosseumLidarUAVEnvConfig,
)
from r1_uav_nav.envs.colosseum_uav_env import ColosseumUAVEnvConfig
from r1_uav_nav.sim.colosseum_capabilities import validate_report_output_path
from r1_uav_nav.sim.colosseum_lidar import (
    build_easy_known_geometry_evidence,
    lidar_scan_input_from_data,
)
from r1_uav_nav.sim.colosseum_scene import (
    ColosseumSceneManager,
    MaterializationConfig,
    SceneCleanupResult,
    SceneRuntimeState,
    StartAnchorReadOnlyContext,
    VehiclePositioningConfig,
    cleanup_scene_resources,
    position_vehicle_at_start_and_return,
)
from r1_uav_nav.sim.lidar_features import (
    LIDAR_FEATURE_COUNT,
    LIDAR_OBSERVATION_SIZE,
    LidarFeatureConfig,
    LidarTimestampStatus,
    LidarTimestampTracker,
    extract_lidar_features,
    extraction_evidence,
)
from r1_uav_nav.sim.scene_specification import AssetCatalog, load_asset_catalog
from r1_uav_nav.sim.static_course import (
    ValidatedCourse,
    generate_solvable_course,
    load_course_suite_config,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D

KNOWN_GEOMETRY_PROFILE = "easy"
KNOWN_GEOMETRY_SEED = 1100
AIRBORNE_SMOKE_PROFILE = "medium"
AIRBORNE_SMOKE_SEED = 2100
DEFAULT_ASSET_CATALOG = Path("configs/scenes/m13_2_assets.yaml")
DEFAULT_COURSE_CONFIG = Path("configs/planning/m13_3_voxel_astar.yaml")
DEFAULT_LIDAR_REPORT_DIR = Path("results/reports/m13/lidar")
_GROUNDED_SPEED_TOLERANCE_M_S = 0.05


@dataclass(frozen=True)
class LiveAuthorizationEvidence:
    """Every operator gate required before live scene or flight work."""

    allow_live_rpc: bool
    allow_scene_mutation: bool
    confirm_scene_area_clear: bool
    confirm_no_visible_collision: bool
    allow_debug_markers: bool
    allow_marker_flush: bool
    allow_flight: bool
    allow_start_positioning: bool
    confirm_clear_airspace: bool

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value is not True)


@dataclass(frozen=True)
class PreparedLidarLiveRun:
    """Fully offline-validated inputs safe to use before client import."""

    mode: str
    feature_config: LidarFeatureConfig
    course: ValidatedCourse
    asset_catalog: AssetCatalog
    authorizations: LiveAuthorizationEvidence
    output_dir: Path
    sample_count: int
    sample_interval_s: float
    known_geometry: dict[str, Any] | None


@dataclass(frozen=True)
class LiveExecutionResult:
    """Sanitized execution evidence ready for the existing live report."""

    data: dict[str, Any]
    cleanup: dict[str, Any]
    errors: tuple[str, ...]
    interrupted: bool


class BroadResetGuard:
    """Proxy that makes any broad simulator reset an explicit run failure."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.reset_attempted = False

    def reset(self) -> None:
        self.reset_attempted = True
        raise RuntimeError(
            "broad simulator reset is forbidden in M13.4 live validation"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def prepare_lidar_live_run(
    *,
    mode: str,
    repository_root: Path,
    feature_config: LidarFeatureConfig,
    course_config_path: Path,
    asset_catalog_path: Path,
    output_dir: Path,
    profile_id: str,
    base_seed: int,
    vehicle_name: str,
    lidar_name: str,
    authorizations: LiveAuthorizationEvidence,
    sample_count: int,
    sample_interval_s: float = 0.0,
) -> PreparedLidarLiveRun:
    """Validate every offline input and authorization before client import."""
    expected = {
        "known-geometry-live": (KNOWN_GEOMETRY_PROFILE, KNOWN_GEOMETRY_SEED),
        "airborne-smoke": (AIRBORNE_SMOKE_PROFILE, AIRBORNE_SMOKE_SEED),
    }
    if mode not in expected:
        raise ValueError(f"unsupported M13.4 live mode {mode!r}")
    expected_profile, expected_seed = expected[mode]
    if profile_id != expected_profile:
        raise ValueError(f"{mode} requires course profile {expected_profile!r}")
    if base_seed != expected_seed:
        raise ValueError(f"{mode} requires declared base seed {expected_seed}")
    if vehicle_name != feature_config.vehicle_name:
        raise ValueError("vehicle name does not match the LiDAR feature configuration")
    if lidar_name != feature_config.lidar_name:
        raise ValueError("LiDAR name does not match the LiDAR feature configuration")
    missing = authorizations.missing()
    if missing:
        raise ValueError(
            "live mode lacks required authorization: " + ", ".join(missing)
        )
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise ValueError("sample count must be an integer")
    maximum_count = 20
    if not 1 <= sample_count <= maximum_count:
        raise ValueError(f"sample count must be between 1 and {maximum_count}")
    if not math.isfinite(sample_interval_s) or not 0 <= sample_interval_s <= 2.0:
        raise ValueError("scan interval must be finite and between 0 and 2 seconds")

    root = repository_root.resolve()
    approved = (root / DEFAULT_LIDAR_REPORT_DIR).resolve()
    resolved_output = (
        output_dir if output_dir.is_absolute() else root / output_dir
    ).resolve()
    if not resolved_output.is_relative_to(approved):
        raise ValueError(
            "M13.4 live reports must remain under results/reports/m13/lidar"
        )
    validate_report_output_path(resolved_output / "preflight.json", root)

    course_path = (
        course_config_path
        if course_config_path.is_absolute()
        else root / course_config_path
    )
    suite = load_course_suite_config(course_path)
    profile = suite.profile(profile_id)
    if base_seed not in profile.base_seeds:
        raise ValueError(
            f"seed {base_seed} is not declared by course profile {profile_id!r}"
        )
    course = generate_solvable_course(
        suite, profile_id, base_seed, repository_root=root
    )
    if not course.result.accepted:
        raise ValueError("course did not pass M13.3 solvability validation")
    catalog_path = (
        asset_catalog_path
        if asset_catalog_path.is_absolute()
        else root / asset_catalog_path
    )
    catalog = load_asset_catalog(catalog_path)
    known = (
        build_easy_known_geometry_evidence(course, feature_config)
        if mode == "known-geometry-live"
        else None
    )
    return PreparedLidarLiveRun(
        mode,
        feature_config,
        course,
        catalog,
        authorizations,
        resolved_output,
        sample_count,
        sample_interval_s,
        known,
    )


def collect_known_geometry_samples(
    context: StartAnchorReadOnlyContext,
    config: LidarFeatureConfig,
    known_geometry: Mapping[str, Any],
    *,
    scan_count: int,
    scan_interval_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect strict feature evidence without exposing geometry to extraction."""
    output = evidence if evidence is not None else {}
    expected_index = int(known_geometry["expected_flattened_feature_index"])
    clear_index = int(known_geometry["clear_comparison_flattened_feature_index"])
    expected_distance = float(known_geometry["expected_nearest_surface_distance_m"])
    tolerance = float(known_geometry["accepted_absolute_tolerance_m"])
    tracker = LidarTimestampTracker(config.maximum_repeated_timestamp_transitions)
    extractions: list[dict[str, Any]] = []
    measured_normalized: list[float] = []
    measured_metres: list[float] = []
    clear_normalized: list[float] = []
    clear_metres: list[float] = []
    output.update(
        {
            "requested_scan_count": scan_count,
            "collected_scan_count": 0,
            "expected_comparison_index": expected_index,
            "clear_comparison_index": clear_index,
            "expected_distance_m": expected_distance,
            "accepted_absolute_tolerance_m": tolerance,
            "extractions": extractions,
            "measured_normalized_values": measured_normalized,
            "measured_distances_m": measured_metres,
            "clear_sector_normalized_values": clear_normalized,
            "clear_sector_distances_m": clear_metres,
        }
    )
    for index in range(scan_count):
        raw = context.read_lidar()
        result = extract_lidar_features(
            lidar_scan_input_from_data(raw), config, tracker
        )
        extraction = extraction_evidence(result)
        extractions.append(extraction)
        output["collected_scan_count"] = len(extractions)
        normalized = float(result.features[expected_index])
        clear_value = float(result.features[clear_index])
        measured_normalized.append(normalized)
        measured_metres.append(_denormalize_distance(normalized, config))
        clear_normalized.append(clear_value)
        clear_metres.append(_denormalize_distance(clear_value, config))
        if index + 1 < scan_count and scan_interval_s:
            sleep_fn(scan_interval_s)

    median_distance = statistics.median(measured_metres)
    difference = abs(median_distance - expected_distance)
    output.update(
        {
            "minimum_distance_m": min(measured_metres),
            "maximum_distance_m": max(measured_metres),
            "mean_distance_m": statistics.fmean(measured_metres),
            "median_distance_m": median_distance,
            "absolute_difference_m": difference,
            "distance_within_tolerance": difference <= tolerance,
        }
    )
    checks = {
        "requested_scan_count_collected": len(extractions) == scan_count,
        "every_scan_policy_valid": all(
            item["lidar_valid"] == 1.0 for item in extractions
        ),
        "feature_shape_exactly_72": all(
            item["feature_shape"] == [LIDAR_FEATURE_COUNT] for item in extractions
        ),
        "features_finite_and_bounded": all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for item in extractions
            for value in item["sector_values"]
        ),
        "timestamps_first_valid_then_fresh": _timestamp_sequence_is_fresh(extractions),
        "distance_within_tolerance": difference <= tolerance,
    }
    output["acceptance_checks"] = checks
    output["acceptance_failures"] = tuple(
        name for name, passed in checks.items() if not passed
    )
    return output


def execute_known_geometry_live(
    prepared: PreparedLidarLiveRun,
    client: Any,
    client_module: ModuleType,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    manager_factory: Callable[..., Any] = ColosseumSceneManager,
    position_fn: Callable[..., dict[str, Any]] = position_vehicle_at_start_and_return,
    cleanup_fn: Callable[
        [Any, SceneRuntimeState], tuple[SceneCleanupResult, ...]
    ] = cleanup_scene_resources,
) -> LiveExecutionResult:
    """Materialize easy seed 1100, sample at its anchor, and clean safely."""
    guarded = BroadResetGuard(client)
    run_id = uuid.uuid4().hex
    runtime: SceneRuntimeState | None = None
    materialized: Any | None = None
    positioning: dict[str, Any] = {}
    measurement: dict[str, Any] = {}
    cleanup_results: tuple[SceneCleanupResult, ...] = ()
    errors: list[str] = []
    interrupted = False
    manager: Any | None = None
    try:
        manager = manager_factory(
            guarded,
            client_module,
            prepared.asset_catalog,
            sleep_fn=sleep_fn,
        )
        materialized, runtime = manager.materialize(
            prepared.course.scene,
            _materialization_config(prepared),
            run_id=run_id,
        )

        def callback(context: StartAnchorReadOnlyContext) -> dict[str, Any]:
            try:
                return collect_known_geometry_samples(
                    context,
                    prepared.feature_config,
                    prepared.known_geometry or {},
                    scan_count=prepared.sample_count,
                    scan_interval_s=prepared.sample_interval_s,
                    sleep_fn=sleep_fn,
                    evidence=measurement,
                )
            except BaseException as exc:
                measurement["collection_error"] = f"{type(exc).__name__}: {exc}"
                raise

        positioning = position_fn(
            guarded,
            client_module,
            materialized,
            runtime,
            _positioning_config(prepared),
            start_anchor_callback=callback,
            sleep_fn=sleep_fn,
        )
    except KeyboardInterrupt:
        interrupted = True
        errors.append("Operation interrupted by the operator.")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if runtime is None and manager is not None:
            runtime = getattr(manager, "last_runtime", None)
        if runtime is not None:
            if runtime.vehicle_positioning_evidence:
                positioning = runtime.vehicle_positioning_evidence
            try:
                cleanup_results = cleanup_fn(guarded, runtime)
            except BaseException as exc:
                errors.append(f"scene cleanup raised {type(exc).__name__}: {exc}")
            errors.extend(_cleanup_errors(cleanup_results))

    final_state = _final_uav_state_evidence(
        guarded, client_module, prepared.feature_config.vehicle_name
    )
    course = _course_identity(prepared.course)
    cleanup = _cleanup_evidence(cleanup_results)
    checks = _known_geometry_acceptance_checks(
        prepared,
        materialized,
        measurement,
        positioning,
        cleanup_results,
        final_state,
        guarded,
    )
    data = {
        "course": course,
        "authorization_evidence": asdict(prepared.authorizations),
        "known_geometry": prepared.known_geometry,
        "measurement": measurement,
        "vehicle_positioning": positioning,
        "scene_cleanup": cleanup,
        "final_uav_state": final_state,
        "broad_reset_attempted": guarded.reset_attempted,
        "acceptance_checks": checks,
        "acceptance_failures": tuple(
            name for name, passed in checks.items() if not passed
        ),
    }
    return LiveExecutionResult(data, cleanup, tuple(errors), interrupted)


def execute_airborne_smoke(
    prepared: PreparedLidarLiveRun,
    client: Any,
    client_module: ModuleType,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    manager_factory: Callable[..., Any] = ColosseumSceneManager,
    environment_factory: Callable[..., Any] | None = None,
    cleanup_fn: Callable[
        [Any, SceneRuntimeState], tuple[SceneCleanupResult, ...]
    ] = cleanup_scene_resources,
) -> LiveExecutionResult:
    """Run bounded zero-action observations in the opt-in LiDAR environment."""
    guarded = BroadResetGuard(client)
    run_id = uuid.uuid4().hex
    runtime: SceneRuntimeState | None = None
    materialized: Any | None = None
    manager: Any | None = None
    environment: Any | None = None
    environment_cleanup: Any | None = None
    environment_lifecycle: dict[str, Any] = {}
    observations: list[dict[str, Any]] = []
    cleanup_results: tuple[SceneCleanupResult, ...] = ()
    final_state: dict[str, Any] = {}
    scene_cleanup_performed = False
    scene_cleanup_deferred_reason: str | None = None
    errors: list[str] = []
    interrupted = False
    try:
        manager = manager_factory(
            guarded,
            client_module,
            prepared.asset_catalog,
            sleep_fn=sleep_fn,
        )
        materialized, runtime = manager.materialize(
            prepared.course.scene,
            _materialization_config(prepared),
            run_id=run_id,
        )
        factory = environment_factory or _default_environment_factory
        environment = factory(prepared, guarded, sleep_fn)
        options = {
            "start_anchor": _position(materialized.start_anchor_world),
            "goal_approach": _position(materialized.goal_approach_world),
        }
        observation, info = environment.reset(options=options)
        observations.append(
            _observation_evidence(environment, observation, info, "reset")
        )
        for index in range(prepared.sample_count):
            action = np.zeros(environment.action_space.shape, dtype=np.float32)
            observation, _reward, terminated, truncated, info = environment.step(action)
            observations.append(
                _observation_evidence(
                    environment,
                    observation,
                    info,
                    f"step-{index + 1}",
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        interrupted = True
        errors.append("Operation interrupted by the operator.")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if environment is not None:
            try:
                environment_cleanup = environment.close_with_result()
            except BaseException as exc:
                errors.append(f"environment cleanup raised {type(exc).__name__}: {exc}")
            environment_lifecycle = dict(getattr(environment, "lifecycle_evidence", {}))
        final_state = _final_uav_state_evidence(
            guarded, client_module, prepared.feature_config.vehicle_name
        )
        if runtime is None and manager is not None:
            runtime = getattr(manager, "last_runtime", None)
        scene_cleanup_safe, scene_cleanup_deferred_reason = (
            _scene_cleanup_safety_decision(
                final_state,
                environment_lifecycle,
                environment_was_created=environment is not None,
            )
        )
        if runtime is not None and scene_cleanup_safe:
            scene_cleanup_performed = True
            try:
                cleanup_results = cleanup_fn(guarded, runtime)
            except BaseException as exc:
                errors.append(f"scene cleanup raised {type(exc).__name__}: {exc}")
            errors.extend(_cleanup_errors(cleanup_results))
        elif runtime is None:
            scene_cleanup_deferred_reason = (
                "no materialized scene runtime required cleanup"
            )
    cleanup = {
        "environment": _serialize_cleanup_result(environment_cleanup),
        "scene": _cleanup_evidence(cleanup_results),
        "scene_cleanup_performed": scene_cleanup_performed,
        "scene_cleanup_deferred": (runtime is not None and not scene_cleanup_performed),
        "scene_cleanup_deferred_reason": scene_cleanup_deferred_reason,
    }
    checks = _airborne_acceptance_checks(
        prepared,
        materialized,
        observations,
        environment_cleanup,
        cleanup_results,
        final_state,
        guarded,
        environment_lifecycle,
        scene_cleanup_performed,
    )
    data = {
        "course": _course_identity(prepared.course),
        "authorization_evidence": asdict(prepared.authorizations),
        "requested_step_count": prepared.sample_count,
        "observation_count": len(observations),
        "observations": observations,
        "environment_cleanup": _serialize_cleanup_result(environment_cleanup),
        "environment_lifecycle": environment_lifecycle,
        "scene_cleanup": _cleanup_evidence(cleanup_results),
        "scene_cleanup_performed": scene_cleanup_performed,
        "scene_cleanup_deferred": (runtime is not None and not scene_cleanup_performed),
        "scene_cleanup_deferred_reason": scene_cleanup_deferred_reason,
        "final_uav_state": final_state,
        "start_pad_appearance": _start_pad_appearance_evidence(prepared, materialized),
        "broad_reset_attempted": guarded.reset_attempted,
        "acceptance_checks": checks,
        "acceptance_failures": tuple(
            name for name, passed in checks.items() if not passed
        ),
    }
    return LiveExecutionResult(data, cleanup, tuple(errors), interrupted)


def _default_environment_factory(
    prepared: PreparedLidarLiveRun,
    client: Any,
    sleep_fn: Callable[[float], None],
) -> ColosseumLidarUAVEnv:
    navigation = ColosseumUAVEnvConfig(
        workspace_xy_limit=20.0,
        workspace_up_limit=10.0,
        workspace_down_limit=2.0,
        max_episode_steps=max(20, prepared.sample_count + 5),
    )
    return ColosseumLidarUAVEnv(
        ColosseumLidarUAVEnvConfig(
            navigation=navigation,
            lidar=prepared.feature_config,
            confirm_no_visible_collision=(
                prepared.authorizations.confirm_no_visible_collision
            ),
        ),
        client_factory=lambda: client,
        sleep_fn=sleep_fn,
    )


def _materialization_config(
    prepared: PreparedLidarLiveRun,
) -> MaterializationConfig:
    auth = prepared.authorizations
    return MaterializationConfig(
        vehicle_name=prepared.feature_config.vehicle_name,
        allow_scene_mutation=auth.allow_scene_mutation,
        confirm_scene_area_clear=auth.confirm_scene_area_clear,
        confirm_no_visible_collision=auth.confirm_no_visible_collision,
        allow_debug_markers=auth.allow_debug_markers,
        allow_marker_flush=auth.allow_marker_flush,
    )


def _positioning_config(prepared: PreparedLidarLiveRun) -> VehiclePositioningConfig:
    auth = prepared.authorizations
    return VehiclePositioningConfig(
        vehicle_name=prepared.feature_config.vehicle_name,
        allow_flight=auth.allow_flight,
        allow_start_positioning=auth.allow_start_positioning,
        confirm_clear_airspace=auth.confirm_clear_airspace,
        confirm_no_visible_collision=auth.confirm_no_visible_collision,
        start_anchor_lidar_name=prepared.feature_config.lidar_name,
        start_anchor_callback_maximum_samples=max(20, prepared.sample_count),
    )


def _denormalize_distance(value: float, config: LidarFeatureConfig) -> float:
    return config.minimum_range_m + value * (
        config.maximum_range_m - config.minimum_range_m
    )


def _timestamp_sequence_is_fresh(extractions: list[dict[str, Any]]) -> bool:
    statuses = [item["diagnostics"]["timestamp_status"] for item in extractions]
    return (
        bool(statuses)
        and statuses[0] == LidarTimestampStatus.FIRST_VALID.value
        and all(status == LidarTimestampStatus.FRESH.value for status in statuses[1:])
    )


def _course_identity(course: ValidatedCourse) -> dict[str, Any]:
    result = course.result
    return {
        "profile_id": result.profile_id,
        "base_seed": result.base_seed,
        "accepted_candidate_seed": result.accepted_candidate_seed,
        "attempt_index": result.attempt_index,
        "scene_digest": result.scene_digest,
        "occupancy_digest": result.occupancy_digest,
        "solvability_digest": result.solvability_digest,
        "solvable": result.accepted,
    }


def _cleanup_errors(results: tuple[SceneCleanupResult, ...]) -> tuple[str, ...]:
    return tuple(
        error for result in results if not result.succeeded for error in result.errors
    )


def _cleanup_evidence(
    results: tuple[SceneCleanupResult, ...],
) -> dict[str, Any]:
    return {
        "results": tuple(asdict(result) for result in results),
        "uav_cleanup": _domain_cleanup(results, "uav"),
        "object_cleanup": _domain_cleanup(results, "objects"),
        "marker_cleanup": _domain_cleanup(results, "markers"),
    }


def _domain_cleanup(
    results: tuple[SceneCleanupResult, ...], domain: str
) -> dict[str, Any] | None:
    match = next((result for result in results if result.domain == domain), None)
    return asdict(match) if match is not None else None


def _final_uav_state_evidence(
    client: Any, client_module: ModuleType, vehicle_name: str
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "landed": False,
        "stationary": False,
        "speed_m_s": None,
        "position": None,
        "api_control_enabled": None,
        "armed_state": "unavailable",
        "errors": (),
    }
    errors: list[str] = []
    try:
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        kinematics = getattr(state, "kinematics_estimated", None)
        position = getattr(kinematics, "position", None)
        velocity = getattr(kinematics, "linear_velocity", None)
        position_values = (
            float(position.x_val),
            float(position.y_val),
            float(position.z_val),
        )
        velocity_values = (
            float(velocity.x_val),
            float(velocity.y_val),
            float(velocity.z_val),
        )
        state_values = (*position_values, *velocity_values)
        if not all(math.isfinite(value) for value in state_values):
            raise ValueError("final state contains non-finite values")
        speed = math.sqrt(sum(value * value for value in velocity_values))
        landed_type = getattr(client_module, "LandedState", None)
        landed_value = getattr(landed_type, "Landed", 0)
        evidence["position"] = position_values
        evidence["speed_m_s"] = speed
        evidence["stationary"] = speed <= _GROUNDED_SPEED_TOLERANCE_M_S
        evidence["landed"] = getattr(state, "landed_state", None) == landed_value
    except BaseException as exc:
        errors.append(f"state verification raised {type(exc).__name__}: {exc}")
    try:
        evidence["api_control_enabled"] = client.isApiControlEnabled(
            vehicle_name=vehicle_name
        )
    except BaseException as exc:
        errors.append(f"API-control verification raised {type(exc).__name__}: {exc}")
    evidence["errors"] = tuple(errors)
    return evidence


def _known_geometry_acceptance_checks(
    prepared: PreparedLidarLiveRun,
    materialized: Any | None,
    measurement: Mapping[str, Any],
    positioning: Mapping[str, Any],
    cleanup_results: tuple[SceneCleanupResult, ...],
    final_state: Mapping[str, Any],
    guarded: BroadResetGuard,
) -> dict[str, bool]:
    measurement_checks = measurement.get("acceptance_checks", {})
    return {
        "course_is_authoritative_and_solvable": (
            prepared.course.result.profile_id == KNOWN_GEOMETRY_PROFILE
            and prepared.course.result.base_seed == KNOWN_GEOMETRY_SEED
            and prepared.course.result.accepted
        ),
        "scene_materialized": materialized is not None,
        "measurement_acceptance_passed": bool(measurement_checks)
        and all(measurement_checks.values()),
        "safe_return_and_landing_confirmed": (
            positioning.get("returned_to_original_ground") is True
            and positioning.get("landing_confirmed") is True
            and positioning.get("api_control_released") is True
        ),
        "object_cleanup_succeeded": _cleanup_domain_succeeded(
            cleanup_results, "objects"
        ),
        "marker_cleanup_succeeded": _cleanup_domain_succeeded(
            cleanup_results, "markers"
        ),
        "final_vehicle_landed": final_state.get("landed") is True,
        "final_vehicle_stationary": final_state.get("stationary") is True,
        "final_api_control_disabled": (final_state.get("api_control_enabled") is False),
        "broad_reset_not_used": not guarded.reset_attempted,
    }


def _airborne_acceptance_checks(
    prepared: PreparedLidarLiveRun,
    materialized: Any | None,
    observations: list[dict[str, Any]],
    environment_cleanup: Any,
    cleanup_results: tuple[SceneCleanupResult, ...],
    final_state: Mapping[str, Any],
    guarded: BroadResetGuard,
    environment_lifecycle: Mapping[str, Any],
    scene_cleanup_performed: bool,
) -> dict[str, bool]:
    expected_count = prepared.sample_count + 1
    cleanup_attempts = environment_lifecycle.get("cleanup_attempts", ())
    final_attempt = cleanup_attempts[-1] if cleanup_attempts else {}
    returned_to_original = (
        final_attempt.get("returned_to_original_ground") is True
        or _position_within_tolerance(
            environment_lifecycle.get("original_ground_position"),
            final_state.get("position"),
            environment_lifecycle.get("landing_position_tolerance_m"),
        )
    ) and (
        final_state.get("landed") is True
        and final_state.get("stationary") is True
        and final_state.get("api_control_enabled") is False
    )
    return {
        "course_is_authoritative_and_solvable": (
            prepared.course.result.profile_id == AIRBORNE_SMOKE_PROFILE
            and prepared.course.result.base_seed == AIRBORNE_SMOKE_SEED
            and prepared.course.result.accepted
        ),
        "scene_materialized": materialized is not None,
        "requested_observations_collected": len(observations) == expected_count,
        "every_observation_valid": bool(observations)
        and all(item["accepted"] for item in observations),
        "timestamps_first_valid_then_fresh": _observation_timestamps_are_fresh(
            observations
        ),
        "no_sensor_failure": all(
            item["sensor_failure"] is False for item in observations
        ),
        "no_unexpected_termination": all(
            not item["terminated"] and not item["truncated"] for item in observations
        ),
        "environment_cleanup_succeeded": not _cleanup_result_failed(
            environment_cleanup
        ),
        "returned_to_original_ground": returned_to_original,
        "scene_cleanup_performed_after_safe_return": scene_cleanup_performed,
        "object_cleanup_succeeded": _cleanup_domain_succeeded(
            cleanup_results, "objects"
        ),
        "marker_cleanup_succeeded": _cleanup_domain_succeeded(
            cleanup_results, "markers"
        ),
        "final_vehicle_landed": final_state.get("landed") is True,
        "final_vehicle_stationary": final_state.get("stationary") is True,
        "final_api_control_disabled": (final_state.get("api_control_enabled") is False),
        "broad_reset_not_used": not guarded.reset_attempted,
    }


def _scene_cleanup_safety_decision(
    final_state: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    *,
    environment_was_created: bool,
) -> tuple[bool, str | None]:
    if final_state.get("landed") is not True:
        return False, "final named vehicle state is not confirmed landed"
    if final_state.get("stationary") is not True:
        return False, "final named vehicle state is not confirmed stationary"
    if final_state.get("api_control_enabled") is not False:
        return False, "final named vehicle API control is not confirmed disabled"
    if not environment_was_created:
        return True, None
    original = lifecycle.get("original_ground_position")
    final_position = final_state.get("position")
    tolerance = lifecycle.get("landing_position_tolerance_m")
    returned_to_original = _position_within_tolerance(
        original, final_position, tolerance
    )
    attempts = lifecycle.get("cleanup_attempts", ())
    if not attempts:
        if returned_to_original:
            return True, None
        return False, "environment supplied no original-ground cleanup evidence"
    final_attempt = attempts[-1]
    if (
        final_attempt.get("returned_to_original_ground") is not True
        and not returned_to_original
    ):
        return False, "return to the recorded original ground point is unverified"
    if final_attempt.get("landing_confirmed") is not True and not returned_to_original:
        return False, "final landing confirmation is unavailable"
    return True, None


def _position_within_tolerance(expected: Any, actual: Any, tolerance: Any) -> bool:
    if not isinstance(expected, Mapping):
        return False
    if not isinstance(actual, (tuple, list)) or len(actual) != 3:
        return False
    try:
        expected_values = tuple(float(expected[name]) for name in ("x", "y", "z"))
        actual_values = tuple(float(value) for value in actual)
        limit = float(tolerance)
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(limit) or limit <= 0:
        return False
    values = (*expected_values, *actual_values)
    if not all(math.isfinite(value) for value in values):
        return False
    return (
        math.sqrt(
            sum(
                (expected_value - actual_value) ** 2
                for expected_value, actual_value in zip(
                    expected_values, actual_values, strict=True
                )
            )
        )
        <= limit
    )


def _start_pad_appearance_evidence(
    prepared: PreparedLidarLiveRun, materialized: Any | None
) -> dict[str, Any]:
    appearance = prepared.course.scene.config.start_pad.appearance
    materialized_object = None
    for item in getattr(materialized, "objects", ()):
        if getattr(item, "specification_name", None) == "start-pad":
            materialized_object = item
            break
    return {
        "requested_material_name": appearance.material_name,
        "requested_marker_color_rgba": appearance.marker_color_rgba,
        "requests_expected_red_appearance": bool(
            appearance.material_name and "red" in appearance.material_name.casefold()
        ),
        "material_assignment_attempted": (
            getattr(materialized_object, "material_assignment_succeeded", None)
            is not None
        ),
        "material_assignment_succeeded": getattr(
            materialized_object, "material_assignment_succeeded", None
        ),
        "cosmetic_only": True,
        "affects_airborne_smoke_acceptance": False,
    }


def _cleanup_domain_succeeded(
    results: tuple[SceneCleanupResult, ...], domain: str
) -> bool:
    result = next((item for item in results if item.domain == domain), None)
    return result is not None and result.succeeded


def _observation_evidence(
    environment: Any,
    observation: np.ndarray,
    info: Mapping[str, Any],
    label: str,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    lidar = info.get("lidar", {})
    navigation = getattr(environment, "last_navigation_observation", None)
    shape_valid = observation.shape == (LIDAR_OBSERVATION_SIZE,)
    dtype_valid = observation.dtype == np.float32
    finite = bool(np.all(np.isfinite(observation)))
    contained = bool(environment.observation_space.contains(observation))
    navigation_unchanged = (
        isinstance(navigation, np.ndarray)
        and navigation.shape == (10,)
        and np.array_equal(observation[:10], navigation)
    )
    lidar_features_match = (
        isinstance(lidar, Mapping)
        and len(lidar.get("sector_values", ())) == LIDAR_FEATURE_COUNT
        and np.array_equal(
            observation[10:82],
            np.asarray(lidar["sector_values"], dtype=np.float32),
        )
    )
    valid_flag = float(observation[82]) if observation.size > 82 else None
    sensor_failure = bool(info.get("sensor_failure", False))
    accepted = all(
        (
            shape_valid,
            dtype_valid,
            finite,
            contained,
            navigation_unchanged,
            lidar_features_match,
            valid_flag == 1.0,
            not sensor_failure,
            not terminated,
            not truncated,
        )
    )
    return {
        "label": label,
        "shape": list(observation.shape),
        "dtype": str(observation.dtype),
        "finite": finite,
        "observation_space_contains": contained,
        "navigation_prefix_unchanged": navigation_unchanged,
        "lidar_features_match": lidar_features_match,
        "lidar_valid_flag": valid_flag,
        "extraction_status": lidar.get("status"),
        "timestamp_status": lidar.get("diagnostics", {}).get("timestamp_status"),
        "terminated": terminated,
        "truncated": truncated,
        "sensor_failure": sensor_failure,
        "termination_reason": info.get("termination_reason"),
        "accepted": accepted,
    }


def _observation_timestamps_are_fresh(
    observations: list[dict[str, Any]],
) -> bool:
    statuses = [item.get("timestamp_status") for item in observations]
    return (
        bool(statuses)
        and statuses[0] == LidarTimestampStatus.FIRST_VALID.value
        and all(status == LidarTimestampStatus.FRESH.value for status in statuses[1:])
    )


def _cleanup_result_failed(result: Any) -> bool:
    return result is not None and bool(getattr(result, "safety_critical_failure", True))


def _serialize_cleanup_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    try:
        return asdict(result)
    except TypeError:
        return {
            "safety_critical_failure": bool(
                getattr(result, "safety_critical_failure", True)
            ),
            "errors": tuple(getattr(result, "errors", ())),
        }


def _position(value: Any) -> Position3D:
    return Position3D(float(value.x), float(value.y), float(value.z))
