from __future__ import annotations

import dataclasses
import io
import math
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from PIL import Image

from rigby_v2.artifacts import ArtifactStore
from rigby_v2.flywheel.schemas import (
    CameraEvidenceV1,
    CandidateEvidenceV1,
    EvidenceCameraRole,
)
from rigby_v2.hashing import canonical_json_bytes, content_hash
from rigby_v2.simulation.runtime import SimulationTrace

from .errors import EvidenceError, EvidenceFailureReason


EVIDENCE_FPS = 30


@dataclass(frozen=True)
class EvidenceRenderConfig:
    width: int = 320
    height: int = 240
    fov_y_degrees: float = 45.0
    ffmpeg_path: str | None = None
    video_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("evidence dimensions must be positive even integers")
        if not 1.0 < self.fov_y_degrees < 179.0:
            raise ValueError("evidence vertical field of view must lie in (1, 179)")
        if self.video_timeout_s <= 0.0:
            raise ValueError("video timeout must be positive")


def anonymous_id_for(semantic_plan_hash: str, candidate_id: str) -> str:
    return "anon-" + content_hash(
        {"semantic_plan_hash": semantic_plan_hash, "candidate_id": candidate_id}
    )[:16]


def _zip_bytes(files: Iterable[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return output.getvalue()


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(value), allow_pickle=False)
    return output.getvalue()


def trace_archive_bytes(trace: SimulationTrace) -> bytes:
    contacts = [dataclasses.asdict(frame) for frame in trace.contacts]
    return _zip_bytes(
        (
            ("times_s.npy", _npy_bytes(trace.times_s)),
            ("qpos.npy", _npy_bytes(trace.qpos)),
            ("qvel.npy", _npy_bytes(trace.qvel)),
            ("ctrl.npy", _npy_bytes(trace.ctrl)),
            ("actuator_force.npy", _npy_bytes(trace.actuator_force)),
            ("generalized_effort.npy", _npy_bytes(trace.generalized_effort)),
            ("contacts.json", canonical_json_bytes(contacts)),
        )
    )


def _evidence_times(trace: SimulationTrace) -> np.ndarray:
    if trace.times_s.ndim != 1 or len(trace.times_s) < 2:
        raise EvidenceError(
            EvidenceFailureReason.INVALID_TRACE,
            "Authoritative trace requires at least two timestamps",
        )
    if np.any(~np.isfinite(trace.times_s)) or np.any(np.diff(trace.times_s) <= 0.0):
        raise EvidenceError(
            EvidenceFailureReason.INVALID_TRACE,
            "Authoritative trace timestamps must be finite and strictly increasing",
        )
    start = float(trace.times_s[0])
    end = float(trace.times_s[-1])
    frame_count = int(math.floor((end - start) * EVIDENCE_FPS + 1e-9)) + 1
    if frame_count < 2:
        raise EvidenceError(
            EvidenceFailureReason.INVALID_TRACE,
            "Trace is shorter than one 30 FPS evidence interval",
        )
    return start + np.arange(frame_count, dtype=np.float64) / EVIDENCE_FPS


def _interpolated_state(
    model: mujoco.MjModel, trace: SimulationTrace, time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    right = int(np.searchsorted(trace.times_s, time_s, side="right"))
    right = min(max(right, 1), len(trace.times_s) - 1)
    left = right - 1
    span = float(trace.times_s[right] - trace.times_s[left])
    alpha = (time_s - float(trace.times_s[left])) / span
    qpos = trace.qpos[left].copy()
    tangent = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        tangent,
        span,
        trace.qpos[left],
        trace.qpos[right],
    )
    mujoco.mj_integratePos(model, qpos, tangent, alpha * span)
    qvel = (1.0 - alpha) * trace.qvel[left] + alpha * trace.qvel[right]
    return qpos, qvel


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("camera direction is degenerate")
    return vector / norm


def _render_camera_transform(renderer: mujoco.Renderer) -> np.ndarray:
    """Read the exact GL camera MuJoCo used after scene projection."""

    camera = renderer.scene.camera[0]
    position = np.asarray(camera.pos, dtype=np.float64).copy()
    forward = _normalize(np.asarray(camera.forward, dtype=np.float64))
    up = _normalize(np.asarray(camera.up, dtype=np.float64))
    right = _normalize(np.cross(forward, up))
    up = _normalize(np.cross(right, forward))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, up, -forward))
    transform[:3, 3] = position
    return transform


def _mjv_camera(position: np.ndarray, look_at: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    offset = position - look_at
    distance = max(0.01, float(np.linalg.norm(offset)))
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = look_at
    camera.distance = distance
    camera.azimuth = math.degrees(math.atan2(float(offset[1]), float(offset[0])))
    camera.elevation = math.degrees(math.asin(float(offset[2]) / distance))
    return camera


def _named_site(model: mujoco.MjModel, names: Iterable[str]) -> int | None:
    for name in names:
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if identifier >= 0:
            return identifier
    return None


def _target_point(model: mujoco.MjModel, data: mujoco.MjData, task_site: str | None) -> np.ndarray:
    names = [task_site] if task_site else []
    names.extend(
        ("right_palm", "left_palm", "fingertip", "right_index_tip", "left_index_tip")
    )
    site = _named_site(model, (name for name in names if name))
    if site is not None:
        return data.site_xpos[site].copy()
    return data.subtree_com[0].copy()


def _camera_pose(
    role: EvidenceCameraRole,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    progress: float,
    task_site: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    center = data.subtree_com[0].copy()
    extent = max(float(model.stat.extent), 0.5)
    if role is EvidenceCameraRole.ORBIT:
        angle = math.radians(135.0 + 35.0 * progress)
        position = center + np.asarray(
            [2.0 * extent * math.cos(angle), 2.0 * extent * math.sin(angle), 0.65 * extent]
        )
        return position, center
    if role is EvidenceCameraRole.EGOCENTRIC:
        gaze_site = _named_site(model, ("gaze_origin", "head_center", "sternum"))
        if gaze_site is not None:
            position = data.site_xpos[gaze_site].copy()
            orientation = data.site_xmat[gaze_site].reshape(3, 3)
            forward = orientation @ np.asarray([0.0, -1.0, 0.0])
        else:
            highest_body = int(np.argmax(data.xpos[:, 2]))
            position = data.xpos[highest_body].copy()
            orientation = data.xmat[highest_body].reshape(3, 3)
            forward = orientation @ np.asarray([0.0, -1.0, 0.0])
        forward = _normalize(forward)
        position = position + 0.04 * forward
        return position, position + extent * forward
    target = _target_point(model, data, task_site)
    position = target + np.asarray([0.55 * extent, 0.45 * extent, 0.28 * extent])
    return position, target


def _intrinsics(config: EvidenceRenderConfig) -> tuple[float, ...]:
    focal = 0.5 * config.height / math.tan(math.radians(config.fov_y_degrees) / 2.0)
    return (
        focal,
        0.0,
        (config.width - 1.0) / 2.0,
        0.0,
        focal,
        (config.height - 1.0) / 2.0,
        0.0,
        0.0,
        1.0,
    )


def _png_bytes(frame: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
        output, format="PNG", optimize=False, compress_level=6
    )
    return output.getvalue()


def _video_bytes(
    frames: list[bytes], config: EvidenceRenderConfig, role: EvidenceCameraRole
) -> bytes:
    executable = config.ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise EvidenceError(
            EvidenceFailureReason.VIDEO_BACKEND_UNAVAILABLE,
            "ffmpeg is not installed; authoritative video evidence cannot be encoded",
        )
    with tempfile.TemporaryDirectory(prefix="rigby-evidence-") as folder:
        root = Path(folder)
        for index, frame in enumerate(frames):
            (root / f"{index:06d}.png").write_bytes(frame)
        output = root / f"{role.value}.mp4"
        command = [
            str(executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(EVIDENCE_FPS),
            "-start_number",
            "0",
            "-i",
            str(root / "%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(EVIDENCE_FPS),
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(output),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=config.video_timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as error:
            stderr = getattr(error, "stderr", b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise EvidenceError(
                EvidenceFailureReason.VIDEO_BACKEND_UNAVAILABLE,
                "ffmpeg could not encode 30 FPS evidence",
                details={"error": str(error), "stderr": str(stderr)[-1000:]},
            ) from error
        return output.read_bytes()


class MujocoEvidenceRenderer:
    def __init__(
        self, store: ArtifactStore, config: EvidenceRenderConfig | None = None
    ) -> None:
        self.store = store
        self.config = config or EvidenceRenderConfig()

    def render(
        self,
        model: mujoco.MjModel,
        trace: SimulationTrace,
        *,
        candidate_id: str,
        anonymous_id: str,
        metrics: dict[str, float] | None = None,
        task_site: str | None = None,
    ) -> CandidateEvidenceV1:
        if trace.qpos.shape != (len(trace.times_s), model.nq) or trace.qvel.shape != (
            len(trace.times_s),
            model.nv,
        ):
            raise EvidenceError(
                EvidenceFailureReason.INVALID_TRACE,
                "Trace generalized state dimensions do not match the MuJoCo model",
            )
        if np.any(~np.isfinite(trace.qpos)) or np.any(~np.isfinite(trace.qvel)):
            raise EvidenceError(
                EvidenceFailureReason.INVALID_TRACE,
                "Trace generalized state must contain only finite values",
            )
        timestamps = _evidence_times(trace)
        trace_ref = self.store.put_bytes(
            trace_archive_bytes(trace),
            media_type="application/zip",
            filename=f"{anonymous_id}.authoritative-trace.zip",
        )
        original_fov = float(model.vis.global_.fovy)
        model.vis.global_.fovy = self.config.fov_y_degrees
        try:
            renderer = mujoco.Renderer(
                model, height=self.config.height, width=self.config.width
            )
        except Exception as error:
            model.vis.global_.fovy = original_fov
            raise EvidenceError(
                EvidenceFailureReason.RENDER_BACKEND_UNAVAILABLE,
                "MuJoCo offscreen rendering is unavailable",
                details={"error": str(error)},
            ) from error
        data = mujoco.MjData(model)
        rendered: list[CameraEvidenceV1] = []
        try:
            for role in EvidenceCameraRole:
                frames: list[bytes] = []
                transforms: list[tuple[float, ...]] = []
                duration = float(timestamps[-1] - timestamps[0])
                for time_s in timestamps:
                    data.qpos[:], data.qvel[:] = _interpolated_state(model, trace, float(time_s))
                    data.time = float(time_s)
                    mujoco.mj_forward(model, data)
                    progress = (
                        float((time_s - timestamps[0]) / duration) if duration > 0 else 0.0
                    )
                    position, look_at = _camera_pose(
                        role, model, data, progress, task_site
                    )
                    renderer.update_scene(data, camera=_mjv_camera(position, look_at))
                    transform = _render_camera_transform(renderer)
                    transforms.append(tuple(float(value) for value in transform.reshape(-1)))
                    frames.append(_png_bytes(renderer.render()))

                camera_id = f"{anonymous_id}-{role.value}"
                manifest = {
                    "schema_version": "raw_png_evidence.v1",
                    "camera_id": camera_id,
                    "role": role.value,
                    "fps": EVIDENCE_FPS,
                    "timestamps_s": timestamps.tolist(),
                    "world_from_camera": transforms,
                    "intrinsics": _intrinsics(self.config),
                    "frames": [f"frames/{index:06d}.png" for index in range(len(frames))],
                }
                raw_archive = _zip_bytes(
                    [("manifest.json", canonical_json_bytes(manifest))]
                    + [
                        (f"frames/{index:06d}.png", value)
                        for index, value in enumerate(frames)
                    ]
                )
                raw_ref = self.store.put_bytes(
                    raw_archive,
                    media_type="application/zip",
                    filename=f"{camera_id}.raw-png.zip",
                )
                video_ref = self.store.put_bytes(
                    _video_bytes(frames, self.config, role),
                    media_type="video/mp4",
                    filename=f"{camera_id}.mp4",
                )
                rendered.append(
                    CameraEvidenceV1(
                        camera_id=camera_id,
                        role=role,
                        raw_frames=raw_ref,
                        video=video_ref,
                        timestamps_s=tuple(float(value) for value in timestamps),
                        world_from_camera=tuple(transforms),
                        intrinsics=_intrinsics(self.config),
                    )
                )
        except EvidenceError:
            raise
        except Exception as error:
            raise EvidenceError(
                EvidenceFailureReason.RENDER_FAILED,
                "Authoritative trace evidence rendering failed",
                details={"error": str(error)},
            ) from error
        finally:
            renderer.close()
            model.vis.global_.fovy = original_fov
        return CandidateEvidenceV1(
            anonymous_id=anonymous_id,
            candidate_id=candidate_id,
            trace=trace_ref,
            cameras=tuple(rendered),
            metrics=metrics or {},
        )
