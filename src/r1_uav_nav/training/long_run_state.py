"""Schema-versioned M13.7 checkpoint, resume, and random-state utilities."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import random
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import stable_baselines3 as sb3
import torch

LONG_RUN_STATE_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
RNG_STATE_SCHEMA_VERSION = 1
LATEST_POINTER_SCHEMA_VERSION = 1

MODEL_FILENAME = "model.zip"
REPLAY_BUFFER_FILENAME = "replay_buffer.pkl"
RUN_STATE_FILENAME = "run_state.json"
RNG_STATE_FILENAME = "rng_state.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "latest.json"

_REQUIRED_ARTIFACTS = (
    MODEL_FILENAME,
    REPLAY_BUFFER_FILENAME,
    RUN_STATE_FILENAME,
    RNG_STATE_FILENAME,
    RESOLVED_CONFIG_FILENAME,
)


class ResumeMode(str, Enum):
    """Supported M13.7 startup modes."""

    NEW = "new"
    FULL = "full"
    MODEL_ONLY_WARM_START = "model_only_warm_start"


@dataclass(frozen=True)
class ArtifactEvidence:
    """Integrity evidence for one checkpoint artifact."""

    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.filename not in _REQUIRED_ARTIFACTS:
            raise ValueError(f"unsupported checkpoint artifact {self.filename!r}")
        if self.size_bytes < 0:
            raise ValueError("artifact size must not be negative")
        if len(self.sha256) != 64:
            raise ValueError("artifact SHA-256 digest must contain 64 characters")


@dataclass(frozen=True)
class LongRunRunState:
    """Persistent training progress that is independent of simulator state."""

    schema_version: int = LONG_RUN_STATE_SCHEMA_VERSION
    run_id: str = ""
    parent_run_id: str | None = None
    warm_start_source_model_digest: str | None = None
    created_at: str = ""
    updated_at: str = ""
    algorithm: str = "td3"
    stable_baselines3_version: str = sb3.__version__
    python_version: str = platform.python_version()
    global_timesteps: int = 0
    update_count: int = 0
    completed_episodes: int = 0
    replay_buffer_size: int = 0
    checkpoint_sequence: int = 0
    next_checkpoint_step: int = 0
    next_monitoring_step: int = 0
    next_promotion_step: int = 0
    best_validation_rank: tuple[float, ...] | None = None
    best_validation_source: str | None = None
    full_config_digest: str = ""
    compatibility_digest: str = ""
    environment_config_digest: str = ""
    observation_signature: str = ""
    action_signature: str = ""
    active_course: dict[str, Any] | None = None
    last_completed_course: dict[str, Any] | None = None
    course_pool: tuple[dict[str, Any], ...] = ()
    course_pool_digest: str = ""
    course_sampler_state: dict[str, Any] | None = None
    curriculum_state: dict[str, Any] = field(
        default_factory=lambda: {
            "schema_version": 1,
            "stage_id": "none",
            "completed_stages": [],
            "progress": 0.0,
        }
    )
    worker_attempt: int = 0
    supervisor_events: tuple[dict[str, Any], ...] = ()
    episode_in_progress: bool = False
    deterministic_resume_guaranteed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LONG_RUN_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported long-run state schema")
        if not self.run_id or self.run_id != self.run_id.strip():
            raise ValueError("run_id must not be empty or padded")
        if self.algorithm != "td3":
            raise ValueError("M13.7 Phase A supports only TD3")
        if not self.created_at or not self.updated_at:
            raise ValueError("run-state timestamps must not be empty")
        for name in (
            "full_config_digest",
            "compatibility_digest",
            "environment_config_digest",
            "observation_signature",
            "action_signature",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if (
            self.warm_start_source_model_digest is not None
            and len(self.warm_start_source_model_digest) != 64
        ):
            raise ValueError("warm-start source model digest must be SHA-256")
        for name in (
            "global_timesteps",
            "update_count",
            "completed_episodes",
            "replay_buffer_size",
            "checkpoint_sequence",
            "next_checkpoint_step",
            "next_monitoring_step",
            "next_promotion_step",
            "worker_attempt",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.best_validation_rank is not None and any(
            not math.isfinite(float(value)) for value in self.best_validation_rank
        ):
            raise ValueError("best_validation_rank values must be finite")
        _validate_curriculum_state(self.curriculum_state)

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible run-state evidence."""
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> LongRunRunState:
        """Build and validate a run state from persisted JSON."""
        values = dict(raw)
        rank = values.get("best_validation_rank")
        if rank is not None:
            values["best_validation_rank"] = tuple(float(item) for item in rank)
        values["course_pool"] = tuple(values.get("course_pool", ()))
        values["supervisor_events"] = tuple(values.get("supervisor_events", ()))
        return cls(**values)


def _validate_curriculum_state(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("curriculum_state must be a mapping")
    stage_id = value.get("stage_id")
    if stage_id == "none":
        if value.get("schema_version") != 1:
            raise ValueError("legacy curriculum state must use schema version 1")
        return
    required = {
        "schema_version",
        "curriculum_id",
        "curriculum_config_digest",
        "stage_id",
        "stage_index",
        "status",
        "stage_start_global_timestep",
        "stage_completed_timesteps",
        "sampling_level_index",
        "consecutive_pass_count",
        "completed_stage_ids",
        "validation_history",
        "best_validation_by_stage",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"curriculum_state is missing required fields: {sorted(missing)}"
        )
    if value.get("schema_version") != 1:
        raise ValueError("unsupported curriculum-state schema")
    if len(str(value.get("curriculum_config_digest", ""))) != 64:
        raise ValueError("curriculum config digest must be SHA-256")
    if value.get("status") not in {"active", "promoted", "blocked", "complete"}:
        raise ValueError("unsupported curriculum status")
    for name in (
        "stage_index",
        "stage_start_global_timestep",
        "stage_completed_timesteps",
        "sampling_level_index",
        "consecutive_pass_count",
    ):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"curriculum {name} must be a non-negative integer")


@dataclass(frozen=True)
class CheckpointManifest:
    """Manifest proving an immutable checkpoint bundle is complete."""

    schema_version: int
    status: str
    run_id: str
    global_timesteps: int
    checkpoint_sequence: int
    created_at: str
    algorithm: str
    stable_baselines3_version: str
    compatibility_digest: str
    observation_signature: str
    action_signature: str
    replay_buffer_class: str
    replay_buffer_capacity: int
    replay_buffer_size: int
    handle_timeout_termination: bool | None
    optimize_memory_usage: bool | None
    episode_in_progress: bool
    artifacts: tuple[ArtifactEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint manifest schema")
        if self.status != "complete":
            raise ValueError("checkpoint manifest is not complete")
        if self.algorithm != "td3":
            raise ValueError("checkpoint algorithm must be TD3")
        if len(self.compatibility_digest) != 64:
            raise ValueError("manifest compatibility digest must be SHA-256")
        if len(self.observation_signature) != 64 or len(self.action_signature) != 64:
            raise ValueError("manifest space signatures must be SHA-256")
        if self.replay_buffer_class != "ReplayBuffer":
            raise ValueError("Phase A checkpoints require the SB3 ReplayBuffer")
        if self.replay_buffer_capacity <= 0:
            raise ValueError("replay buffer capacity must be positive")
        if self.handle_timeout_termination is not True:
            raise ValueError("replay buffer must handle timeout termination")
        names = tuple(item.filename for item in self.artifacts)
        if len(names) != len(set(names)) or set(names) != set(_REQUIRED_ARTIFACTS):
            raise ValueError("checkpoint manifest artifact set is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CheckpointManifest:
        values = dict(raw)
        artifacts = values.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("checkpoint artifacts must be a list")
        values["artifacts"] = tuple(ArtifactEvidence(**item) for item in artifacts)
        return cls(**values)


@dataclass(frozen=True)
class ResumePlan:
    """Pre-start decision for a new, full-resume, or warm-start run."""

    mode: ResumeMode
    run_state_path: Path | None = None
    model_path: Path | None = None
    replay_buffer_path: Path | None = None
    manifest_path: Path | None = None
    source_run_id: str | None = None
    source_model_digest: str | None = None
    reset_num_timesteps: bool = True
    creates_new_lineage: bool = True


@dataclass(frozen=True)
class SafeCheckpoint:
    """A discovered and integrity-checked checkpoint bundle."""

    directory: Path
    manifest: CheckpointManifest
    run_state: LongRunRunState

    @property
    def model_path(self) -> Path:
        return self.directory / MODEL_FILENAME

    @property
    def replay_buffer_path(self) -> Path:
        return self.directory / REPLAY_BUFFER_FILENAME

    @property
    def run_state_path(self) -> Path:
        return self.directory / RUN_STATE_FILENAME


def utc_now() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_rng_state(
    *,
    env: Any | None = None,
    course_sampler: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Capture practical local and process RNG state without simulator claims."""
    sources: dict[str, Any] = {
        "python_random": _jsonable(random.getstate()),
        "numpy_global": _numpy_legacy_state_to_json(np.random.get_state()),
        "torch_cpu": _bytes_to_text(bytes(torch.get_rng_state().tolist())),
        "torch_cuda": [],
        "environment_np_random": _generator_state(env, "np_random"),
        "action_space_np_random": _generator_state(
            getattr(env, "action_space", None), "np_random"
        ),
        "course_sampler": (
            _jsonable(course_sampler.bit_generator.state)
            if course_sampler is not None
            else None
        ),
    }
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        sources["torch_cuda"] = [
            _bytes_to_text(bytes(state.tolist()))
            for state in torch.cuda.get_rng_state_all()
        ]
    available = {
        name: value is not None
        for name, value in sources.items()
        if name not in {"torch_cuda"}
    }
    available["torch_cuda"] = cuda_available
    return {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "sources": sources,
        "available": available,
        "deterministic_resume_guaranteed": False,
        "limitations": [
            "Simulator physics and an in-progress environment episode are not "
            "restored.",
            "CUDA kernels may remain nondeterministic.",
        ],
    }


def restore_rng_state(
    evidence: Mapping[str, Any],
    *,
    env: Any | None = None,
    course_sampler: np.random.Generator | None = None,
) -> dict[str, bool]:
    """Restore supported RNG sources and report which restorations succeeded."""
    if evidence.get("schema_version") != RNG_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported RNG-state schema")
    sources = evidence.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("RNG-state sources must be a mapping")
    restored: dict[str, bool] = {}
    random.setstate(_nested_tuple(sources["python_random"]))
    restored["python_random"] = True
    np.random.set_state(_numpy_legacy_state_from_json(sources["numpy_global"]))
    restored["numpy_global"] = True
    torch.set_rng_state(
        torch.tensor(list(_text_to_bytes(str(sources["torch_cpu"]))), dtype=torch.uint8)
    )
    restored["torch_cpu"] = True
    cuda_states = sources.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                torch.tensor(list(_text_to_bytes(item)), dtype=torch.uint8)
                for item in cuda_states
            ]
        )
        restored["torch_cuda"] = True
    else:
        restored["torch_cuda"] = False
    restored["environment_np_random"] = _restore_generator_state(
        env, "np_random", sources.get("environment_np_random")
    )
    restored["action_space_np_random"] = _restore_generator_state(
        getattr(env, "action_space", None),
        "np_random",
        sources.get("action_space_np_random"),
    )
    if course_sampler is not None and sources.get("course_sampler") is not None:
        course_sampler.bit_generator.state = sources["course_sampler"]
        restored["course_sampler"] = True
    else:
        restored["course_sampler"] = False
    return restored


def save_checkpoint_bundle(
    *,
    model: Any,
    run_state: LongRunRunState,
    resolved_config: Mapping[str, Any],
    run_root: Path,
    env: Any | None = None,
    course_sampler: np.random.Generator | None = None,
) -> SafeCheckpoint:
    """Atomically save an immutable model/replay/state checkpoint bundle."""
    checkpoints = run_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    bundle_name = (
        f"step-{run_state.global_timesteps:012d}"
        f"-seq-{run_state.checkpoint_sequence:06d}"
    )
    destination = checkpoints / bundle_name
    if destination.exists():
        raise FileExistsError(f"checkpoint bundle already exists: {destination}")
    temporary = checkpoints / f".{bundle_name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False)
    try:
        model.save(temporary / MODEL_FILENAME)
        model.save_replay_buffer(temporary / REPLAY_BUFFER_FILENAME)
        _write_json(temporary / RUN_STATE_FILENAME, run_state.to_dict())
        rng_state = capture_rng_state(env=env, course_sampler=course_sampler)
        _write_json(temporary / RNG_STATE_FILENAME, rng_state)
        _write_json(temporary / RESOLVED_CONFIG_FILENAME, resolved_config)
        artifacts = tuple(
            _artifact_evidence(temporary / filename) for filename in _REQUIRED_ARTIFACTS
        )
        replay = getattr(model, "replay_buffer", None)
        manifest = CheckpointManifest(
            schema_version=CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            status="complete",
            run_id=run_state.run_id,
            global_timesteps=run_state.global_timesteps,
            checkpoint_sequence=run_state.checkpoint_sequence,
            created_at=utc_now(),
            algorithm=run_state.algorithm,
            stable_baselines3_version=run_state.stable_baselines3_version,
            compatibility_digest=run_state.compatibility_digest,
            observation_signature=run_state.observation_signature,
            action_signature=run_state.action_signature,
            replay_buffer_class=(
                type(replay).__name__ if replay is not None else "unavailable"
            ),
            replay_buffer_capacity=int(getattr(replay, "buffer_size", 0)),
            replay_buffer_size=_replay_size(replay),
            handle_timeout_termination=getattr(
                replay, "handle_timeout_termination", None
            ),
            optimize_memory_usage=getattr(replay, "optimize_memory_usage", None),
            episode_in_progress=run_state.episode_in_progress,
            artifacts=artifacts,
        )
        _write_json(temporary / MANIFEST_FILENAME, manifest.to_dict())
        _flush_directory(temporary)
        os.replace(temporary, destination)
        _flush_directory(checkpoints)
        _atomic_json(
            run_root / LATEST_FILENAME,
            {
                "schema_version": LATEST_POINTER_SCHEMA_VERSION,
                "run_id": run_state.run_id,
                "bundle": destination.name,
                "global_timesteps": run_state.global_timesteps,
                "checkpoint_sequence": run_state.checkpoint_sequence,
            },
        )
        return validate_checkpoint_bundle(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_checkpoint_bundle(
    directory: Path,
    *,
    expected_compatibility_digest: str | None = None,
) -> SafeCheckpoint:
    """Validate manifest, artifact integrity, and run-state consistency."""
    source = directory.resolve()
    manifest = CheckpointManifest.from_mapping(
        _read_mapping(source / MANIFEST_FILENAME)
    )
    run_state = LongRunRunState.from_mapping(_read_mapping(source / RUN_STATE_FILENAME))
    if manifest.run_id != run_state.run_id:
        raise ValueError("checkpoint manifest and run state have different run IDs")
    if manifest.global_timesteps != run_state.global_timesteps:
        raise ValueError("checkpoint timestep evidence is inconsistent")
    if manifest.checkpoint_sequence != run_state.checkpoint_sequence:
        raise ValueError("checkpoint sequence evidence is inconsistent")
    if manifest.compatibility_digest != run_state.compatibility_digest:
        raise ValueError("checkpoint compatibility evidence is inconsistent")
    resolved_config = _read_mapping(source / RESOLVED_CONFIG_FILENAME)
    if canonical_digest(resolved_config) != run_state.full_config_digest:
        raise ValueError("resolved configuration digest does not match run state")
    if (
        expected_compatibility_digest is not None
        and manifest.compatibility_digest != expected_compatibility_digest
    ):
        raise ValueError("checkpoint is incompatible with the requested configuration")
    for artifact in manifest.artifacts:
        path = source / artifact.filename
        actual = _artifact_evidence(path)
        if actual != artifact:
            raise ValueError(
                f"checkpoint artifact integrity failed: {artifact.filename}"
            )
    if manifest.stable_baselines3_version != sb3.__version__:
        raise ValueError(
            "checkpoint Stable-Baselines3 version does not match the runtime"
        )
    return SafeCheckpoint(source, manifest, run_state)


def discover_latest_safe_checkpoint(
    run_root: Path,
    *,
    expected_compatibility_digest: str | None = None,
) -> SafeCheckpoint:
    """Find the highest-step complete checkpoint without trusting filenames."""
    checkpoints = run_root.resolve() / "checkpoints"
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoints}")
    valid: list[SafeCheckpoint] = []
    for child in checkpoints.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            valid.append(
                validate_checkpoint_bundle(
                    child,
                    expected_compatibility_digest=expected_compatibility_digest,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not valid:
        raise FileNotFoundError("no complete compatible checkpoint bundle was found")
    valid.sort(
        key=lambda item: (
            item.manifest.global_timesteps,
            item.manifest.checkpoint_sequence,
            item.directory.name,
        )
    )
    return valid[-1]


def validate_resume_request(
    *,
    resume_checkpoint: Path | None = None,
    resume_replay_buffer: Path | None = None,
    resume_run_state: Path | None = None,
    resume_latest: Path | None = None,
    reset_num_timesteps: bool = False,
    allow_partial_resume: bool = False,
    expected_compatibility_digest: str | None = None,
) -> ResumePlan:
    """Resolve strict full-resume or explicit model-only warm-start arguments."""
    supplied = any(
        item is not None
        for item in (
            resume_checkpoint,
            resume_replay_buffer,
            resume_run_state,
            resume_latest,
        )
    )
    if not supplied:
        if reset_num_timesteps or allow_partial_resume:
            raise ValueError("resume flags require a resume artifact")
        return ResumePlan(mode=ResumeMode.NEW)
    if resume_latest is not None:
        if any(
            item is not None
            for item in (resume_checkpoint, resume_replay_buffer, resume_run_state)
        ):
            raise ValueError("--resume-latest cannot be combined with explicit files")
        safe = discover_latest_safe_checkpoint(
            resume_latest,
            expected_compatibility_digest=expected_compatibility_digest,
        )
        if reset_num_timesteps or allow_partial_resume:
            raise ValueError(
                "full resume cannot reset timesteps or allow partial state"
            )
        return _full_resume_plan(safe)
    if allow_partial_resume:
        if not reset_num_timesteps:
            raise ValueError("model-only warm start requires --reset-num-timesteps")
        if resume_checkpoint is None:
            raise ValueError("model-only warm start requires --resume-checkpoint")
        if resume_replay_buffer is not None or resume_run_state is not None:
            raise ValueError("model-only warm start forbids replay and run state")
        model_path = _normalized_model_path(resume_checkpoint.resolve())
        return ResumePlan(
            mode=ResumeMode.MODEL_ONLY_WARM_START,
            model_path=model_path,
            source_model_digest=_sha256_file(model_path),
            reset_num_timesteps=True,
            creates_new_lineage=True,
        )
    if reset_num_timesteps:
        raise ValueError("full resume must preserve the timestep counter")
    if resume_run_state is None:
        raise ValueError("full resume requires --resume-run-state")
    safe = validate_checkpoint_bundle(
        resume_run_state.resolve().parent,
        expected_compatibility_digest=expected_compatibility_digest,
    )
    if (
        resume_checkpoint is not None
        and resume_checkpoint.resolve() != safe.model_path.resolve()
    ):
        raise ValueError("explicit model path does not match the run-state bundle")
    if (
        resume_replay_buffer is not None
        and resume_replay_buffer.resolve() != safe.replay_buffer_path.resolve()
    ):
        raise ValueError("explicit replay path does not match the run-state bundle")
    return _full_resume_plan(safe)


def load_rng_evidence(path: Path) -> dict[str, Any]:
    """Load persisted RNG evidence."""
    return dict(_read_mapping(path))


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Load a resolved configuration snapshot."""
    return dict(_read_mapping(path))


def _full_resume_plan(safe: SafeCheckpoint) -> ResumePlan:
    return ResumePlan(
        mode=ResumeMode.FULL,
        run_state_path=safe.run_state_path,
        model_path=safe.model_path,
        replay_buffer_path=safe.replay_buffer_path,
        manifest_path=safe.directory / MANIFEST_FILENAME,
        source_run_id=safe.run_state.run_id,
        reset_num_timesteps=False,
        creates_new_lineage=False,
    )


def _artifact_evidence(path: Path) -> ArtifactEvidence:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint artifact not found: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return ArtifactEvidence(path.name, size, digest.hexdigest())


def _normalized_model_path(path: Path) -> Path:
    if path.is_file():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(f"model checkpoint not found: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
        _flush_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return raw


def _flush_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replay_size(replay: Any | None) -> int:
    if replay is None:
        return 0
    size = replay.size()
    if not isinstance(size, (int, np.integer)) or int(size) < 0:
        raise ValueError("replay buffer returned an invalid size")
    return int(size)


def _generator_state(owner: Any | None, attribute: str) -> dict[str, Any] | None:
    generator = getattr(owner, attribute, None)
    if isinstance(generator, np.random.Generator):
        return _jsonable(generator.bit_generator.state)
    return None


def _restore_generator_state(
    owner: Any | None,
    attribute: str,
    state: Any,
) -> bool:
    if owner is None or state is None:
        return False
    generator = getattr(owner, attribute, None)
    if not isinstance(generator, np.random.Generator):
        return False
    generator.bit_generator.state = state
    return True


def _numpy_legacy_state_to_json(state: Sequence[Any]) -> dict[str, Any]:
    return {
        "bit_generator": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_legacy_state_from_json(raw: Any) -> tuple[Any, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError("NumPy global RNG state must be a mapping")
    return (
        str(raw["bit_generator"]),
        np.asarray(raw["keys"], dtype=np.uint32),
        int(raw["position"]),
        int(raw["has_gauss"]),
        float(raw["cached_gaussian"]),
    )


def _bytes_to_text(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _text_to_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CheckpointManifest",
    "LONG_RUN_STATE_SCHEMA_VERSION",
    "LongRunRunState",
    "ResumeMode",
    "ResumePlan",
    "SafeCheckpoint",
    "canonical_digest",
    "capture_rng_state",
    "discover_latest_safe_checkpoint",
    "load_resolved_config",
    "load_rng_evidence",
    "restore_rng_state",
    "save_checkpoint_bundle",
    "utc_now",
    "validate_checkpoint_bundle",
    "validate_resume_request",
]
