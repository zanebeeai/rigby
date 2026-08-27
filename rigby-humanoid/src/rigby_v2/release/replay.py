"""Self-verifying replay bundles and exact native MuJoCo replay audit."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rigby_v2.hashing import canonical_json_bytes, content_hash, sha256_bytes, validate_sha256
from rigby_v2.simulation import (
    NativeMujocoRuntime,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    model_source_hash,
)

from .errors import ReleaseIntegrityError


MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ReplayArtifactBinding:
    logical_name: str
    sha256: str
    size_bytes: int
    media_type: str
    archive_path: str

    def __post_init__(self) -> None:
        if not self.logical_name or not self.media_type or self.size_bytes < 0:
            raise ValueError("replay artifact fields are invalid")
        validate_sha256(self.sha256)
        expected = f"objects/sha256/{self.sha256[:2]}/{self.sha256[2:]}"
        if self.archive_path != expected:
            raise ValueError("replay artifact path is not content addressed")


@dataclass(frozen=True)
class ReplayRuntimeBinding:
    mujoco_version: str
    physics_hz: int
    timestep_s: float
    controller_id: str
    controller_config_sha256: str
    dependency_lock_sha256: str
    dependency_versions: dict[str, str]

    def __post_init__(self) -> None:
        if (
            not self.mujoco_version
            or self.physics_hz <= 0
            or self.timestep_s <= 0
            or not self.controller_id
            or not self.dependency_versions
        ):
            raise ValueError("runtime replay binding is incomplete")
        validate_sha256(self.controller_config_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if any(not str(name).strip() or not str(version).strip() for name, version in self.dependency_versions.items()):
            raise ValueError("dependency version names and values cannot be blank")


@dataclass(frozen=True)
class ReplayBundleManifest:
    schema_version: str
    bundle_id: str
    program: ReplayArtifactBinding
    rig: ReplayArtifactBinding
    scene: ReplayArtifactBinding
    mjz: ReplayArtifactBinding
    expected_traces: dict[str, ReplayArtifactBinding]
    runtime: ReplayRuntimeBinding
    seeds: tuple[int, ...]
    artifacts: dict[str, ReplayArtifactBinding]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or not self.bundle_id:
            raise ValueError("unsupported replay bundle manifest")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)) or any(seed < 0 for seed in self.seeds):
            raise ValueError("replay seeds must be nonnegative and unique")
        expected_keys = {str(seed) for seed in self.seeds}
        if set(self.expected_traces) != expected_keys:
            raise ValueError("every seed requires exactly one expected trace")
        core = (
            (self.program, "program"),
            (self.rig, "rig"),
            (self.scene, "scene"),
            (self.mjz, "mjz"),
        )
        if any(binding.logical_name != expected for binding, expected in core):
            raise ValueError("core replay artifact logical names are fixed")
        if any(
            binding.logical_name != f"expected_trace:{seed}"
            for seed, binding in self.expected_traces.items()
        ):
            raise ValueError("expected trace logical names must bind their seed")
        if any(name != binding.logical_name for name, binding in self.artifacts.items()):
            raise ValueError("extra replay artifact keys must match logical names")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayBundleManifest":
        def artifact(item: Mapping[str, Any]) -> ReplayArtifactBinding:
            return ReplayArtifactBinding(**dict(item))

        return cls(
            schema_version=str(value["schema_version"]),
            bundle_id=str(value["bundle_id"]),
            program=artifact(value["program"]),
            rig=artifact(value["rig"]),
            scene=artifact(value["scene"]),
            mjz=artifact(value["mjz"]),
            expected_traces={key: artifact(item) for key, item in value["expected_traces"].items()},
            runtime=ReplayRuntimeBinding(**dict(value["runtime"])),
            seeds=tuple(int(seed) for seed in value["seeds"]),
            artifacts={key: artifact(item) for key, item in value.get("artifacts", {}).items()},
        )


@dataclass(frozen=True)
class VerifiedReplayBundle:
    manifest: ReplayBundleManifest
    payloads: dict[str, bytes]
    manifest_sha256: str


@dataclass(frozen=True)
class ReplayAuditEnvironment:
    mujoco_version: str
    physics_hz: int
    timestep_s: float
    controller_id: str
    controller_config_sha256: str
    dependency_lock_sha256: str
    dependency_versions: dict[str, str]


@dataclass(frozen=True)
class ReplayAuditReport:
    exact: bool
    reason: str | None
    seeds: tuple[int, ...]
    repeat_count: int
    results: tuple[SimulationResult, ...]


def _artifact(logical_name: str, value: bytes, media_type: str) -> ReplayArtifactBinding:
    digest = sha256_bytes(value)
    return ReplayArtifactBinding(
        logical_name=logical_name,
        sha256=digest,
        size_bytes=len(value),
        media_type=media_type,
        archive_path=f"objects/sha256/{digest[:2]}/{digest[2:]}",
    )


def _publish_file_no_replace(temporary: str, destination: Path) -> None:
    """Atomically publish a same-filesystem file without a clobber race."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(destination) from None
    Path(temporary).unlink()


def encode_simulation_trace(result: SimulationResult) -> bytes:
    if result.status is not SimulationStatus.COMPLETED:
        raise ValueError("only completed authoritative simulation traces can be encoded")
    trace = result.trace
    contacts = [
        {
            "time_s": frame.time_s,
            "contacts": [asdict(contact) for contact in frame.contacts],
        }
        for frame in trace.contacts
    ]
    return canonical_json_bytes(
        {
            "model_hash": result.model_hash,
            "times_s": trace.times_s.tolist(),
            "qpos": trace.qpos.tolist(),
            "qvel": trace.qvel.tolist(),
            "ctrl": trace.ctrl.tolist(),
            "actuator_force": trace.actuator_force.tolist(),
            "generalized_effort": trace.generalized_effort.tolist(),
            "contacts": contacts,
        }
    )


def controller_config_hash(request: SimulationRequest) -> str:
    """Bind every controller/root-selection setting that can affect stepping."""

    return content_hash(asdict(request.config))


def create_replay_bundle(
    destination: Path,
    *,
    bundle_id: str,
    program: bytes,
    rig: bytes,
    scene: bytes,
    mjz: bytes,
    expected_traces: Mapping[int, bytes],
    runtime: ReplayRuntimeBinding,
    extra_artifacts: Mapping[str, tuple[bytes, str]] | None = None,
) -> ReplayBundleManifest:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    core_payloads = {
        "program": (program, "application/json"),
        "rig": (rig, "application/json"),
        "scene": (scene, "application/json"),
        "mjz": (mjz, "application/vnd.mujoco.mjz"),
    }
    trace_bindings = {
        str(seed): _artifact(f"expected_trace:{seed}", value, "application/json")
        for seed, value in sorted(expected_traces.items())
    }
    extras = {
        name: _artifact(name, value, media_type)
        for name, (value, media_type) in sorted((extra_artifacts or {}).items())
    }
    manifest = ReplayBundleManifest(
        schema_version="1.0",
        bundle_id=bundle_id,
        program=_artifact("program", program, "application/json"),
        rig=_artifact("rig", rig, "application/json"),
        scene=_artifact("scene", scene, "application/json"),
        mjz=_artifact("mjz", mjz, "application/vnd.mujoco.mjz"),
        expected_traces=trace_bindings,
        runtime=runtime,
        seeds=tuple(sorted(expected_traces)),
        artifacts=extras,
    )
    manifest_dict = manifest.to_dict()
    manifest_sha256 = sha256_bytes(canonical_json_bytes(manifest_dict))
    envelope = canonical_json_bytes(
        {"manifest": manifest_dict, "manifest_sha256": manifest_sha256}
    )
    payload_by_binding: dict[ReplayArtifactBinding, bytes] = {}
    for name, binding in (("program", manifest.program), ("rig", manifest.rig), ("scene", manifest.scene), ("mjz", manifest.mjz)):
        payload_by_binding[binding] = core_payloads[name][0]
    for seed, binding in manifest.expected_traces.items():
        payload_by_binding[binding] = expected_traces[int(seed)]
    for name, binding in manifest.artifacts.items():
        payload_by_binding[binding] = (extra_artifacts or {})[name][0]

    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.staging-", delete=False
        ) as stream:
            temporary = stream.name
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("bundle.json", envelope)
            written: set[str] = set()
            for binding, payload in payload_by_binding.items():
                if binding.archive_path not in written:
                    archive.writestr(binding.archive_path, payload)
                    written.add(binding.archive_path)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        _publish_file_no_replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return manifest


def _safe_archive_names(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ReleaseIntegrityError("archive contains too many entries")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ReleaseIntegrityError("archive contains duplicate paths")
    if sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ReleaseIntegrityError("archive expands beyond the safety limit")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ReleaseIntegrityError("archive contains an unsafe path")
    return set(names)


def verify_replay_bundle(path: Path) -> VerifiedReplayBundle:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = _safe_archive_names(archive)
            if "bundle.json" not in names:
                raise ReleaseIntegrityError("replay bundle manifest is missing")
            envelope = json.loads(archive.read("bundle.json"))
            manifest_dict = envelope["manifest"]
            expected_manifest_hash = validate_sha256(envelope["manifest_sha256"])
            actual_manifest_hash = sha256_bytes(canonical_json_bytes(manifest_dict))
            if actual_manifest_hash != expected_manifest_hash:
                raise ReleaseIntegrityError("replay manifest checksum mismatch")
            manifest = ReplayBundleManifest.from_dict(manifest_dict)
            bindings = [manifest.program, manifest.rig, manifest.scene, manifest.mjz]
            bindings += list(manifest.expected_traces.values())
            bindings += list(manifest.artifacts.values())
            expected_paths = {"bundle.json", *(binding.archive_path for binding in bindings)}
            if names != expected_paths:
                raise ReleaseIntegrityError("replay archive has missing or unbound files")
            payloads: dict[str, bytes] = {}
            for binding in bindings:
                payload = archive.read(binding.archive_path)
                if len(payload) != binding.size_bytes or sha256_bytes(payload) != binding.sha256:
                    raise ReleaseIntegrityError(
                        f"replay artifact {binding.logical_name!r} failed integrity verification"
                    )
                payloads[binding.logical_name] = payload
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ReleaseIntegrityError):
            raise
        raise ReleaseIntegrityError(f"invalid replay bundle: {exc}") from exc
    return VerifiedReplayBundle(manifest, payloads, actual_manifest_hash)


class ExactReplayAuditor:
    def __init__(self, runtime: NativeMujocoRuntime | None = None) -> None:
        self.runtime = runtime or NativeMujocoRuntime()

    def audit(
        self,
        bundle: VerifiedReplayBundle,
        environment: ReplayAuditEnvironment,
        request_builder: Callable[[VerifiedReplayBundle, int], SimulationRequest],
        *,
        repeat_count: int = 3,
    ) -> ReplayAuditReport:
        if repeat_count < 3:
            raise ValueError("exact replay audit requires at least three repeats")
        expected_environment = ReplayAuditEnvironment(**asdict(bundle.manifest.runtime))
        if environment != expected_environment:
            return ReplayAuditReport(False, "runtime or dependency binding mismatch", bundle.manifest.seeds, repeat_count, ())
        results: list[SimulationResult] = []
        for seed in bundle.manifest.seeds:
            request = request_builder(bundle, seed)
            if model_source_hash(request) != bundle.manifest.mjz.sha256:
                return ReplayAuditReport(False, "request MJZ does not match replay manifest", bundle.manifest.seeds, repeat_count, tuple(results))
            if controller_config_hash(request) != environment.controller_config_sha256:
                return ReplayAuditReport(False, "controller configuration changed", bundle.manifest.seeds, repeat_count, tuple(results))
            expected_trace = bundle.payloads[f"expected_trace:{seed}"]
            for _ in range(repeat_count):
                result = self.runtime.simulate(request)
                results.append(result)
                if result.status is not SimulationStatus.COMPLETED:
                    return ReplayAuditReport(False, "replayed simulation did not complete", bundle.manifest.seeds, repeat_count, tuple(results))
                if result.diagnostics.get("controller") != environment.controller_id:
                    return ReplayAuditReport(False, "controller identity changed", bundle.manifest.seeds, repeat_count, tuple(results))
                if result.diagnostics.get("physics_hz") != environment.physics_hz:
                    return ReplayAuditReport(False, "physics frequency changed", bundle.manifest.seeds, repeat_count, tuple(results))
                if result.diagnostics.get("timestep_s") != environment.timestep_s:
                    return ReplayAuditReport(False, "physics timestep changed", bundle.manifest.seeds, repeat_count, tuple(results))
                if result.diagnostics.get("native_mujoco_version") != environment.mujoco_version:
                    return ReplayAuditReport(False, "MuJoCo version changed", bundle.manifest.seeds, repeat_count, tuple(results))
                if encode_simulation_trace(result) != expected_trace:
                    return ReplayAuditReport(False, "authoritative trace differs byte-for-byte", bundle.manifest.seeds, repeat_count, tuple(results))
        return ReplayAuditReport(True, None, bundle.manifest.seeds, repeat_count, tuple(results))
