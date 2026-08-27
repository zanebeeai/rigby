from __future__ import annotations

import json
import zipfile
from io import BytesIO

import mujoco
import pytest

from rigby_core.hashing import canonical_json_bytes
from rigby_v2.release import (
    ExactReplayAuditor,
    ReleaseIntegrityError,
    ReplayAuditEnvironment,
    ReplayRuntimeBinding,
    controller_config_hash,
    create_replay_bundle,
    encode_simulation_trace,
    verify_replay_bundle,
)
from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
)

pytestmark = pytest.mark.medium


HASH_C = "c" * 64


def _mjz() -> tuple[bytes, mujoco.MjModel]:
    xml = """
    <mujoco model="release_replay">
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="pelvis" pos="0 0 1"><freejoint name="root"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="arm"><joint name="joint" damping="1"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="joint" ctrllimited="true" ctrlrange="-5 5"/></actuator>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    archive = BytesIO()
    spec.to_zip(archive)
    return archive.getvalue(), model


def _request(mjz: bytes, model: mujoco.MjModel) -> SimulationRequest:
    return SimulationRequest(
        model_xml=None,
        model_mjz=mjz,
        trajectory=ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv)),
        config=SimulationConfig(duration_s=0.05),
        request_id="release-replay",
    )


def _bundle(tmp_path):
    mjz, model = _mjz()
    request = _request(mjz, model)
    result = NativeMujocoRuntime().simulate(request)
    runtime = ReplayRuntimeBinding(
        mujoco_version=result.diagnostics["native_mujoco_version"],
        physics_hz=result.diagnostics["physics_hz"],
        timestep_s=result.diagnostics["timestep_s"],
        controller_id=result.diagnostics["controller"],
        controller_config_sha256=controller_config_hash(request),
        dependency_lock_sha256=HASH_C,
        dependency_versions={"mujoco": mujoco.__version__, "rigby_v2": "0.1.0"},
    )
    path = tmp_path / "exact.rigby-replay"
    manifest = create_replay_bundle(
        path,
        bundle_id="exact-release-replay",
        program=canonical_json_bytes({"program": "wave", "schema_version": "2.0"}),
        rig=canonical_json_bytes({"rig_id": "release-rig"}),
        scene=canonical_json_bytes({"scene_id": "release-scene"}),
        mjz=mjz,
        expected_traces={7: encode_simulation_trace(result)},
        runtime=runtime,
        extra_artifacts={"evidence/orbit": (b"video", "video/mp4")},
    )
    return path, manifest, request


def test_replay_bundle_binds_inputs_dependencies_artifacts_and_audits_exactly(tmp_path) -> None:
    path, manifest, request = _bundle(tmp_path)
    verified = verify_replay_bundle(path)
    environment = ReplayAuditEnvironment(**manifest.runtime.__dict__)

    def request_builder(bundle, seed):
        assert seed == 7
        spec = mujoco.MjSpec.from_zip(BytesIO(bundle.payloads["mjz"]))
        model = spec.compile()
        return _request(bundle.payloads["mjz"], model)

    report = ExactReplayAuditor().audit(
        verified, environment, request_builder, repeat_count=3
    )

    assert report.exact
    assert report.reason is None
    assert len(report.results) == 3
    assert verified.manifest.program.sha256 == manifest.program.sha256
    assert verified.manifest.rig.sha256 == manifest.rig.sha256
    assert verified.manifest.scene.sha256 == manifest.scene.sha256
    assert verified.manifest.mjz.sha256 == manifest.mjz.sha256
    assert verified.payloads["evidence/orbit"] == b"video"
    assert request.model_mjz == verified.payloads["mjz"]
    with pytest.raises(FileExistsError):
        create_replay_bundle(
            path,
            bundle_id="no-overwrite",
            program=b"{}",
            rig=b"{}",
            scene=b"{}",
            mjz=b"x",
            expected_traces={1: b"{}"},
            runtime=manifest.runtime,
        )


def _rewrite_archive(source, destination, transform):
    with zipfile.ZipFile(source, "r") as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            output_archive.writestr(info.filename, transform(info.filename, input_archive.read(info.filename)))


def test_tampered_replay_manifest_and_payload_are_refused(tmp_path) -> None:
    path, manifest, _ = _bundle(tmp_path)
    tampered_manifest = tmp_path / "tampered-manifest.zip"

    def alter_manifest(name, payload):
        if name != "bundle.json":
            return payload
        envelope = json.loads(payload)
        envelope["manifest"]["bundle_id"] = "tampered"
        return canonical_json_bytes(envelope)

    _rewrite_archive(path, tampered_manifest, alter_manifest)
    with pytest.raises(ReleaseIntegrityError, match="manifest checksum"):
        verify_replay_bundle(tampered_manifest)

    tampered_payload = tmp_path / "tampered-payload.zip"

    def alter_payload(name, payload):
        return b"tampered" if name == manifest.mjz.archive_path else payload

    _rewrite_archive(path, tampered_payload, alter_payload)
    with pytest.raises(ReleaseIntegrityError, match="failed integrity"):
        verify_replay_bundle(tampered_payload)


def test_replay_audit_fails_closed_on_environment_or_controller_drift(tmp_path) -> None:
    path, manifest, request = _bundle(tmp_path)
    verified = verify_replay_bundle(path)
    environment = ReplayAuditEnvironment(**manifest.runtime.__dict__)
    wrong_environment = ReplayAuditEnvironment(
        **{**environment.__dict__, "dependency_versions": {"mujoco": "changed"}}
    )
    rejected = ExactReplayAuditor().audit(
        verified, wrong_environment, lambda bundle, seed: request
    )
    assert not rejected.exact
    assert rejected.reason == "runtime or dependency binding mismatch"

    changed_request = SimulationRequest(
        model_xml=None,
        model_mjz=request.model_mjz,
        trajectory=request.trajectory,
        config=SimulationConfig(duration_s=0.10),
        request_id=request.request_id,
    )
    changed = ExactReplayAuditor().audit(
        verified, environment, lambda bundle, seed: changed_request
    )
    assert not changed.exact
    assert changed.reason == "controller configuration changed"

