from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile

import mujoco
import numpy as np
import pytest

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.evidence import (
    EvidenceError,
    EvidenceFailureReason,
    EvidenceRenderConfig,
    MujocoEvidenceRenderer,
)
from rigby_v2.flywheel.schemas import EvidenceCameraRole
from rigby_v2.simulation.metrics import ContactFrame
from rigby_v2.simulation.runtime import SimulationTrace

pytestmark = pytest.mark.slow


MODEL_XML = r"""
<mujoco model="evidence_test">
  <visual><global fovy="45"/><quality shadowsize="256"/></visual>
  <worldbody>
    <geom type="plane" size="2 2 .1" rgba=".2 .25 .3 1"/>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="capsule" size=".08 .2" rgba=".7 .5 .3 1" mass="2"/>
      <site name="gaze_origin" pos="0 -.08 .18"/>
      <site name="right_palm" pos=".2 -.1 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _trace(model: mujoco.MjModel) -> SimulationTrace:
    times = np.arange(49, dtype=np.float64) / 240.0
    qpos = np.repeat(model.qpos0[None, :], len(times), axis=0)
    qpos[:, 0] += np.linspace(0, 0.08, len(times))
    qvel = np.zeros((len(times), model.nv))
    qvel[:, 0] = 0.4
    return SimulationTrace(
        times_s=times,
        qpos=qpos,
        qvel=qvel,
        ctrl=np.empty((len(times), 0)),
        contacts=tuple(ContactFrame(float(time), ()) for time in times),
        actuator_force=np.empty((len(times), 0)),
        generalized_effort=np.zeros((len(times), model.nv)),
    )


def test_authoritative_trace_renders_three_raw_and_video_views_at_30_fps(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    renderer = MujocoEvidenceRenderer(
        store, EvidenceRenderConfig(width=160, height=120)
    )
    try:
        evidence = renderer.render(
            model,
            _trace(model),
            candidate_id="candidate-1",
            anonymous_id="anon-1",
            metrics={"penetration_m": 0.0},
            task_site="right_palm",
        )
    except EvidenceError as error:
        if error.reason is EvidenceFailureReason.RENDER_BACKEND_UNAVAILABLE:
            pytest.skip(f"local MuJoCo GL backend unavailable: {error}")
        raise
    assert {camera.role for camera in evidence.cameras} == set(EvidenceCameraRole)
    assert evidence.trace.media_type == "application/zip"
    assert store.exists(evidence.trace, verify=True)
    for camera in evidence.cameras:
        assert camera.fps == 30
        assert np.diff(camera.timestamps_s) == pytest.approx(1 / 30)
        assert len(camera.timestamps_s) == 7
        assert len(camera.world_from_camera) == 7
        assert camera.video.media_type == "video/mp4"
        assert store.exists(camera.video, verify=True)
        with zipfile.ZipFile(io.BytesIO(store.read_bytes(camera.raw_frames))) as archive:
            pngs = sorted(name for name in archive.namelist() if name.endswith(".png"))
            manifest = json.loads(archive.read("manifest.json"))
        assert len(pngs) == 7
        assert manifest["fps"] == 30
        assert manifest["timestamps_s"] == pytest.approx(camera.timestamps_s)
        assert len(manifest["world_from_camera"]) == 7

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        video_path = store.resolve(evidence.cameras[0].video)
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["r_frame_rate"] == "30/1"
        assert int(stream["nb_frames"]) == 7


def test_invalid_trace_and_missing_video_backend_are_typed(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    bad = _trace(model)
    bad.times_s[1] = bad.times_s[0]
    with pytest.raises(EvidenceError) as invalid:
        MujocoEvidenceRenderer(store).render(
            model, bad, candidate_id="bad", anonymous_id="bad"
        )
    assert invalid.value.reason is EvidenceFailureReason.INVALID_TRACE

    renderer = MujocoEvidenceRenderer(
        store,
        EvidenceRenderConfig(width=160, height=120, ffmpeg_path=str(tmp_path / "missing-ffmpeg")),
    )
    with pytest.raises(EvidenceError) as unavailable:
        renderer.render(model, _trace(model), candidate_id="bad-video", anonymous_id="bad-video")

    # Check the reason we caught; do not wrap `pytest.raises` in `try/except
    # EvidenceError`. That was the previous form and it could not fire:
    # `pytest.raises` *catches* the error, so the `except` never sees one, and a
    # headless host fell through to the assertion below with the wrong reason.
    # On a machine with no GL the render backend fails before the missing ffmpeg
    # is ever reached, so that is a fact about the host, not about the typing.
    if unavailable.value.reason is EvidenceFailureReason.RENDER_BACKEND_UNAVAILABLE:
        pytest.skip(f"MuJoCo GL backend unavailable on this host: {unavailable.value}")

    assert unavailable.value.reason is EvidenceFailureReason.VIDEO_BACKEND_UNAVAILABLE
    assert unavailable.value.code.value == "unsupported"
