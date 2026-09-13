"""The declared sensors sampled as physics runs, for a monitor deciding as it goes.

The same configurations, rates, latencies and sample contents as the
recorded streams, produced from the live data after every physics step
instead of from a sealed record afterwards. A sample observed at time t
becomes available to the evaluator at t plus the sensor's latency, which
is the time it is stamped with. The oracle kind is never sampled live: a
running monitor has no privileged state to read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import mujoco
import numpy as np
from rigby_core.skills import ConditionalV1, ConditionalVerdictV1, EvidenceKind, EvidenceSampleV1, SampleQuality, SensorConfigurationV1, SensorSpecV1, evaluate

from ..contact.placement import PlacementGoal
from ..contracts import EffectorV1, RobotAssetManifestV1, SiteSemantic
from ..grounding.grounder import figure_site_for
from ..grounding.workspace import WorkspaceFrame, build_workspace_frame
from .sensors import CAMERAS, CameraSpec, group_forces_from_data, visible_fraction


KEEP_S = 12.0
"""How much history a live sensor keeps: longer than any window, shorter than an episode."""


@dataclass
class LiveStream:
    sensor: SensorSpecV1
    next_s: float
    times: list[float] = field(default_factory=list)
    quality: list[SampleQuality] = field(default_factory=list)
    values: list[dict[str, float]] = field(default_factory=list)
    provenance: str = ""

    def push(self, time_s: float, quality: SampleQuality, values: dict[str, float]) -> None:
        self.times.append(time_s)
        self.quality.append(quality)
        self.values.append(values)
        while self.times and self.times[0] < time_s - KEEP_S:
            self.times.pop(0)
            self.quality.pop(0)
            self.values.pop(0)

    def window(self, start_s: float, end_s: float) -> list[EvidenceSampleV1]:
        lo = int(np.searchsorted(self.times, start_s - 1e-9, side="left"))
        hi = int(np.searchsorted(self.times, end_s + 1e-9, side="right"))
        return [EvidenceSampleV1(sensor_id=self.sensor.sensor_id, kind=self.sensor.kind, time_s=self.times[i], quality=self.quality[i], values=self.values[i], provenance=self.provenance)
                for i in range(lo, hi)]

    def latest(self, now_s: float, *, valid_only: bool = True) -> tuple[float, dict[str, float]] | None:
        hi = int(np.searchsorted(self.times, now_s + 1e-9, side="right"))
        for i in range(hi - 1, -1, -1):
            if not valid_only or self.quality[i] is SampleQuality.VALID:
                return self.times[i], self.values[i]
        return None


class LiveSensing:
    """Every sensor of a configuration, sampling the live data."""

    def __init__(self, model: mujoco.MjModel, manifest: RobotAssetManifestV1, effectors: tuple[EffectorV1, ...], configuration: SensorConfigurationV1, goal: PlacementGoal, *,
                 object_geom: str = "scene_block_geom", seed: str = "") -> None:
        self.model = model
        self.configuration = configuration
        self.goal = goal
        self.object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
        self.object_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")])
        self.half = np.array(model.geom_size[self.object_geom], dtype=float)
        self.effectors = {e.chain_id: e for e in effectors}
        self.geomid = np.zeros(1, dtype=np.int32)
        self.streams: dict[str, LiveStream] = {}
        self.cameras: dict[str, CameraSpec] = {}
        self.rng: dict[str, np.random.Generator] = {}
        self.frames: dict[str, WorkspaceFrame] = {}
        self.members: dict[str, dict[str, frozenset[int]]] = {}
        self.sites: dict[str, int] = {}
        self.dofs = [(d.name, int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, d.joint)]), int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, d.joint)]))
                     for d in manifest.dofs]
        for sensor in configuration.sensors:
            if sensor.oracle:
                continue
            self.streams[sensor.sensor_id] = LiveStream(sensor, next_s=0.0, provenance=f"{sensor.sensor_id} live at {sensor.rate_hz:g} Hz, latency {sensor.latency_s * 1000:.0f} ms")
            if sensor.kind is EvidenceKind.OBJECT_POSE:
                spec = CAMERAS[sensor.sensor_id.split(":", 1)[1]]
                self.cameras[sensor.sensor_id] = spec
                digest = hashlib.sha256(f"{seed}|{sensor.sensor_id}".encode()).digest()[:8]
                self.rng[sensor.sensor_id] = np.random.default_rng(int.from_bytes(digest, "little"))
            if sensor.entity and sensor.entity not in self.frames:
                effector = self.effectors[sensor.entity]
                chain = next(c for c in manifest.morphology.chains if c.chain_id == effector.chain_id)
                self.frames[sensor.entity] = build_workspace_frame(model, manifest.morphology, chain, figure_site=figure_site_for(manifest, effector.chain_id))
                members = {}
                for body in effector.member_bodies:
                    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                    start, count = (int(model.body_geomadr[identifier]), int(model.body_geomnum[identifier])) if identifier >= 0 else (0, 0)
                    members[body] = frozenset(range(start, start + count))
                self.members[sensor.entity] = members
                site = -1
                for semantic in (SiteSemantic.GRASP_POINT, SiteSemantic.GRASP_CENTER):
                    for candidate in manifest.morphology.sites:
                        if candidate.semantic is semantic and candidate.name.startswith(effector.chain_id):
                            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, candidate.name)
                            if site >= 0:
                                break
                    if site >= 0:
                        break
                self.sites[sensor.entity] = site
        self._started = False

    # -- sampling ----------------------------------------------------------------------
    def observe(self, data: mujoco.MjData) -> None:
        """Sample every sensor whose next instant has come."""

        now = float(data.time)
        if not self._started:
            for stream in self.streams.values():
                stream.next_s = now
            self._started = True
        for sensor_id, stream in self.streams.items():
            sensor = stream.sensor
            if now + 1e-9 < stream.next_s:
                continue
            stream.next_s += 1.0 / sensor.rate_hz
            stamp = now + sensor.latency_s
            kind = sensor.kind
            if kind is EvidenceKind.JOINT_ENCODERS:
                row = {}
                for name, qadr, dadr in self.dofs:
                    row[f"q:{name}"] = float(data.qpos[qadr])
                    row[f"dq:{name}"] = float(data.qvel[dadr])
                stream.push(stamp, SampleQuality.VALID, row)
            elif kind is EvidenceKind.EFFECTOR_POSE:
                p = data.site_xpos[self.sites[sensor.entity]]
                stream.push(stamp, SampleQuality.VALID, {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
            elif kind is EvidenceKind.CONTACT_FORCE:
                effector = self.effectors[sensor.entity]
                groups = group_forces_from_data(self.model, data, self.members[sensor.entity], effector.opposition_groups, self.object_geom)
                row = {f"group_{k}_n": float(f) for k, f in enumerate(groups)}
                row["max_force_n"] = float(max(groups, default=0.0))
                stream.push(stamp, SampleQuality.VALID, row)
            elif kind is EvidenceKind.OBJECT_POSE:
                spec = self.cameras[sensor_id]
                origin = np.array(spec.position_m, dtype=float)
                fraction, centre = visible_fraction(self.model, data, self.object_geom, self.half, origin, self.geomid)
                if float(np.linalg.norm(centre - origin)) > spec.range_m:
                    stream.push(stamp, SampleQuality.MISSING, {"visible_fraction": 0.0})
                elif fraction >= spec.visible_fraction:
                    reported = centre + self.rng[sensor_id].normal(0.0, spec.noise_m, size=3)
                    stream.push(stamp, SampleQuality.VALID, {"x": float(reported[0]), "y": float(reported[1]), "z": float(reported[2]), "visible_fraction": fraction})
                else:
                    stream.push(stamp, SampleQuality.OCCLUDED, {"visible_fraction": fraction})
            elif kind is EvidenceKind.REGION:
                low, high = self.goal.region_minimum_m, self.goal.region_maximum_m
                stream.push(stamp, SampleQuality.VALID, {"min_x": low[0], "min_y": low[1], "min_z": low[2], "max_x": high[0], "max_y": high[1], "max_z": high[2]})
            elif kind is EvidenceKind.REACH_ENVELOPE:
                frame = self.frames[sensor.entity]
                point = self.last_object_position(now)
                azimuth, elevation = frame.bearing_of(point) if point is not None else (0.0, 0.0)
                stream.push(stamp, SampleQuality.VALID, {"origin_x": float(frame.origin[0]), "origin_y": float(frame.origin[1]), "origin_z": float(frame.origin[2]),
                                                          "inner_m": float(frame.inner_reach(azimuth, elevation)), "outer_m": float(frame.directional_reach(azimuth, elevation)),
                                                          "azimuth_deg": float(azimuth), "elevation_deg": float(elevation)})

    def grip_force_now(self, data: mujoco.MjData, entity: str) -> float:
        """The largest opposition-group force on the object in ``data`` as it
        stands, read the way the contact sensor reads it; for a restart that
        has no stream yet and has to learn whether the closure is engaged."""

        if entity not in self.members:
            return 0.0
        effector = self.effectors[entity]
        groups = group_forces_from_data(self.model, data, self.members[entity], effector.opposition_groups, self.object_geom)
        return float(max(groups, default=0.0))

    # -- reading -------------------------------------------------------------------------
    def samples(self, start_s: float, end_s: float) -> list[EvidenceSampleV1]:
        out: list[EvidenceSampleV1] = []
        for stream in self.streams.values():
            out.extend(stream.window(start_s, end_s))
        return out

    def last_object_position(self, now_s: float, *, max_age_s: float | None = None) -> np.ndarray | None:
        """The newest position any camera reported at or before ``now_s``, within ``max_age_s`` if given."""

        best = None
        for sensor_id in self.cameras:
            latest = self.streams[sensor_id].latest(now_s)
            if latest is None:
                continue
            if max_age_s is not None and now_s - latest[0] > max_age_s + 1e-9:
                continue
            if best is None or latest[0] > best[0]:
                best = latest
        return None if best is None else np.array([best[1]["x"], best[1]["y"], best[1]["z"]], dtype=float)

    def mean_object_position(self, now_s: float, window_s: float, *, min_samples: int = 4) -> np.ndarray | None:
        """The mean of the positions the cameras reported over the last
        ``window_s``: an estimate whose noise is a frame's over the root of
        the number of frames, for an object that has been seen still."""

        track = []
        for sensor_id in self.cameras:
            for sample in self.streams[sensor_id].window(now_s - window_s, now_s):
                if sample.quality is SampleQuality.VALID:
                    track.append([sample.values["x"], sample.values["y"], sample.values["z"]])
        if len(track) < min_samples:
            return None
        return np.mean(np.asarray(track, dtype=float), axis=0)

    def object_drift_mps(self, now_s: float, window_s: float, *, min_samples: int = 4) -> float | None:
        """How fast the sensed object moved over the last ``window_s``: the
        mean position of the window's second half against its first half,
        over half the window, so the camera's noise averages out and a
        resting object reads as resting; ``None`` with too few sightings
        in either half."""

        track = []
        for sensor_id in self.cameras:
            stream = self.streams[sensor_id]
            for sample in stream.window(now_s - window_s, now_s):
                if sample.quality is SampleQuality.VALID:
                    track.append((sample.time_s, np.array([sample.values["x"], sample.values["y"], sample.values["z"]])))
        middle = now_s - 0.5 * window_s
        first = [p for t, p in track if t < middle]
        second = [p for t, p in track if t >= middle]
        if len(first) < min_samples or len(second) < min_samples:
            return None
        return float(np.linalg.norm(np.mean(second, axis=0) - np.mean(first, axis=0)) / (0.5 * window_s))

    def camera_state(self, now_s: float) -> dict[str, dict]:
        """What each camera last reported: its age and whether it saw the object."""

        out = {}
        for sensor_id in self.cameras:
            latest = self.streams[sensor_id].latest(now_s, valid_only=False)
            out[sensor_id] = {"age_s": None if latest is None else now_s - latest[0], "visible_fraction": None if latest is None else latest[1].get("visible_fraction")}
        return out

    def grip_force_n(self, manipulator: str, now_s: float) -> float:
        latest = self.streams[f"contact:{manipulator}"].latest(now_s) if f"contact:{manipulator}" in self.streams else None
        return 0.0 if latest is None else float(latest[1].get("max_force_n", 0.0))


def decide_live(conditional: ConditionalV1, sensing: LiveSensing, now_s: float) -> ConditionalVerdictV1:
    start = now_s - conditional.window.duration_s
    return evaluate(conditional, sensing.configuration, sensing.samples(start, now_s), now_s)
