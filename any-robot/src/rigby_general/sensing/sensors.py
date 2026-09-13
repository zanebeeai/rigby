"""Declared sensor configurations, and the evidence they produce from a record.

Each sensor samples the recorded run at its own rate and latency. Encoders
read the recorded joints; the effector pose is the encoders through the
calibrated model; contact force is the recorded contact wrench on each of
the gripper's members against the object, reported per opposition group;
a camera reports the object's position when most of its rays reach the
object first, and reports occlusion when they hit something else -- the
robot's own hand over a top-down grasp, a screen in the world, whatever the
world holds; the region is the task's geometry; the reach envelope is the
manipulator's calibrated shell along the bearing of the last sensed object.
The oracle sensor reports the full state and is marked as such; the
evaluator never reads it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import mujoco
import numpy as np
from rigby_core.skills import EvidenceKind, EvidenceSampleV1, SampleQuality, SensorConfigurationV1, SensorSpecV1

from ..contracts import EffectorV1
from ..grounding.grounder import figure_site_for
from ..grounding.workspace import WorkspaceFrame, build_workspace_frame
from .episode import Episode


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position_m: tuple[float, float, float]
    rate_hz: float = 30.0
    latency_s: float = 0.033
    max_age_s: float = 0.2
    visible_fraction: float = 0.5
    """Of the rays cast at the object (its centre and five face centres): how
    many must reach it first for the sample to report a position."""
    noise_m: float = 0.002
    """Standard deviation of the reported position, per axis, seeded per trace and sensor."""
    range_m: float = float("inf")
    """Beyond this distance the camera reports nothing (a missing sample), not a position; unbounded by default, as the G09 campaign was scored."""


CAMERAS = {
    "overhead": CameraSpec("overhead", (-0.08, 0.65, 1.2)),
    "front": CameraSpec("front", (-0.08, 1.35, 0.55)),
    "side_low": CameraSpec("side_low", (0.45, 0.65, 0.35)),
}
"""Where the cameras stand in the G06 fixed world: above the two fixtures,
beyond them looking back, and low to the right where a hand closing on the
cube stands between the lens and the cube."""

ENCODER_HZ = 500.0
EFFECTOR_HZ = 100.0
CONTACT_HZ = 200.0
GEOMETRY_HZ = 10.0
ORACLE_HZ = 100.0


def sensor_specs(name: str, effectors: tuple[EffectorV1, ...]) -> tuple[SensorSpecV1, ...]:
    """The named configuration's sensors, one contact, pose and reach sensor per manipulator."""

    common = [SensorSpecV1(sensor_id="encoders", kind=EvidenceKind.JOINT_ENCODERS, rate_hz=ENCODER_HZ, max_age_s=0.02, description="joint positions and velocities of every declared dof")]
    for effector in effectors:
        common.append(SensorSpecV1(sensor_id=f"effector_pose:{effector.chain_id}", kind=EvidenceKind.EFFECTOR_POSE, rate_hz=EFFECTOR_HZ, max_age_s=0.05, entity=effector.chain_id,
                                   description="the grasp point through the calibrated model from the encoders"))
    geometry = [SensorSpecV1(sensor_id="region", kind=EvidenceKind.REGION, rate_hz=GEOMETRY_HZ, max_age_s=1.0, description="the task's destination region")]
    reach = [SensorSpecV1(sensor_id=f"reach:{e.chain_id}", kind=EvidenceKind.REACH_ENVELOPE, rate_hz=GEOMETRY_HZ, max_age_s=1.0, entity=e.chain_id,
                          description="the manipulator's calibrated reach shell along the bearing of the last sensed object position") for e in effectors]
    contact = [SensorSpecV1(sensor_id=f"contact:{e.chain_id}", kind=EvidenceKind.CONTACT_FORCE, rate_hz=CONTACT_HZ, max_age_s=0.05, entity=e.chain_id,
                            description=f"normal force on each opposition group of {e.chain_id} from the object: " + "; ".join(",".join(g) for g in e.opposition_groups)) for e in effectors]

    def camera(spec: CameraSpec) -> SensorSpecV1:
        return SensorSpecV1(sensor_id=f"camera:{spec.name}", kind=EvidenceKind.OBJECT_POSE, rate_hz=spec.rate_hz, latency_s=spec.latency_s, max_age_s=spec.max_age_s, occludable=True,
                            description=f"object position by ray visibility from {spec.position_m}; occluded below {spec.visible_fraction:.0%} of rays; noise {spec.noise_m * 1000:.0f} mm")

    oracle = SensorSpecV1(sensor_id="oracle", kind=EvidenceKind.ORACLE_STATE, rate_hz=ORACLE_HZ, max_age_s=0.02, oracle=True, description="privileged full state; labels only")
    if name == "overhead_contact":
        return tuple(common + contact + [camera(CAMERAS["overhead"])] + geometry + reach)
    if name == "front_contact":
        return tuple(common + contact + [camera(CAMERAS["front"])] + geometry + reach)
    if name == "side_contact":
        return tuple(common + contact + [camera(CAMERAS["side_low"])] + geometry + reach)
    if name == "contact_only":
        return tuple(common + contact + geometry + reach)
    if name == "front_vision_only":
        return tuple(common + [camera(CAMERAS["front"])] + geometry + reach)
    if name == "side_vision_only":
        return tuple(common + [camera(CAMERAS["side_low"])] + geometry + reach)
    if name == "front_contact_with_oracle":
        return tuple(common + contact + [camera(CAMERAS["front"])] + geometry + reach + [oracle])
    if name == "front_overhead_contact":
        return tuple(common + contact + [camera(CAMERAS["front"]), camera(CAMERAS["overhead"])] + geometry + reach)
    raise KeyError(f"no sensor configuration named {name!r}")


CONFIGURATIONS = ("overhead_contact", "front_contact", "side_contact", "contact_only", "front_vision_only", "side_vision_only", "front_contact_with_oracle")
"""The configurations the G09 campaign scored. ``front_overhead_contact``
(both cameras with contact) is what the G10 skill observes with."""


def configuration(name: str, effectors: tuple[EffectorV1, ...]) -> SensorConfigurationV1:
    descriptions = {
        "overhead_contact": "encoders, gripper contact, a camera above the fixtures",
        "front_contact": "encoders, gripper contact, a camera beyond the fixtures looking back",
        "side_contact": "encoders, gripper contact, a low camera to the side that a closing hand stands in front of",
        "contact_only": "encoders and gripper contact; no camera",
        "front_vision_only": "encoders and the front camera; no contact sensing",
        "side_vision_only": "encoders and the low side camera; no contact sensing: a hand closing on the cube hides it",
        "front_contact_with_oracle": "front_contact plus a privileged oracle sensor, present to show it is never read",
        "front_overhead_contact": "encoders, gripper contact, the front camera and the overhead camera: what one cannot see the other may",
    }
    return SensorConfigurationV1(configuration_id=name, sensors=sensor_specs(name, effectors), description=descriptions[name])


@dataclass
class Stream:
    sensor: SensorSpecV1
    times: np.ndarray
    quality: list[SampleQuality]
    values: list[dict[str, float]]
    provenance: str

    def window(self, start_s: float, end_s: float) -> list[EvidenceSampleV1]:
        lo = int(np.searchsorted(self.times, start_s - 1e-9, side="left"))
        hi = int(np.searchsorted(self.times, end_s + 1e-9, side="right"))
        return [EvidenceSampleV1(sensor_id=self.sensor.sensor_id, kind=self.sensor.kind, time_s=float(self.times[i]), quality=self.quality[i], values=self.values[i], provenance=self.provenance)
                for i in range(lo, hi)]

    def truncated(self, last_s: float) -> "Stream":
        keep = int(np.searchsorted(self.times, last_s + 1e-9, side="right"))
        return Stream(self.sensor, self.times[:keep], self.quality[:keep], self.values[:keep], self.provenance + f"; frozen at {last_s:.3f}s")

    def thinned(self, rate_hz: float) -> "Stream":
        if not len(self.times):
            return self
        keep, next_time = [], float(self.times[0])
        for i, t in enumerate(self.times):
            if t + 1e-9 >= next_time:
                keep.append(i)
                next_time = t + 1.0 / rate_hz
        return Stream(self.sensor.model_copy(update={"rate_hz": rate_hz}), self.times[keep], [self.quality[i] for i in keep], [self.values[i] for i in keep], self.provenance + f"; thinned to {rate_hz:g} Hz")


def sample_times(episode: Episode, rate_hz: float, latency_s: float) -> np.ndarray:
    if episode.refusal:
        return np.array([episode.start_s + latency_s])
    count = int(np.floor((episode.end_s - episode.start_s - latency_s) * rate_hz)) + 1
    return episode.start_s + latency_s + np.arange(max(count, 1)) / rate_hz


def _noise(episode: Episode, sensor_id: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{episode.trace_sha256}|{sensor_id}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


FACES = [np.zeros(3), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])]
"""Where a camera's rays are aimed: the object's centre and five of its faces (not the underside)."""


def visible_fraction(model: mujoco.MjModel, data: mujoco.MjData, object_geom: int, half: np.ndarray, origin: np.ndarray, geomid: np.ndarray) -> tuple[float, np.ndarray]:
    """The fraction of the rays from ``origin`` that reach the object first,
    against the world as ``data`` has it, and the object's centre."""

    centre = np.array(data.geom_xpos[object_geom], dtype=float)
    rotation = np.array(data.geom_xmat[object_geom], dtype=float).reshape(3, 3)
    seen = 0
    for face in FACES:
        target = centre + rotation @ (half * face)
        vector = target - origin
        distance = float(np.linalg.norm(vector))
        mujoco.mj_ray(model, data, origin, vector / distance, None, 1, -1, geomid)
        seen += int(geomid[0] == object_geom)
    return seen / len(FACES), centre


def group_forces_from_data(model: mujoco.MjModel, data: mujoco.MjData, member_geoms: dict[str, frozenset[int]], groups, object_geom: int) -> list[float]:
    """Normal force on each opposition group from the object, from the live contacts."""

    forces = {body: 0.0 for body in member_geoms}
    buffer = np.zeros(6, dtype=float)
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if object_geom not in (first, second):
            continue
        other = second if first == object_geom else first
        for body, geoms in member_geoms.items():
            if other in geoms:
                mujoco.mj_contactForce(model, data, index, buffer)
                forces[body] += abs(float(buffer[0]))
                break
    return [max((forces.get(m, 0.0) for m in group), default=0.0) for group in groups]


def camera_stream(episode: Episode, sensor: SensorSpecV1, spec: CameraSpec, *, model: mujoco.MjModel | None = None) -> Stream:
    """Rays from the camera to the object's centre and five face centres,
    against the world as recorded (or ``model``, a copy with more in it)."""

    model = episode.model if model is None else model
    data = mujoco.MjData(model)
    times = sample_times(episode, spec.rate_hz, spec.latency_s)
    rng = _noise(episode, sensor.sensor_id)
    origin = np.array(spec.position_m, dtype=float)
    geomid = np.zeros(1, dtype=np.int32)
    obj = episode.object_geom
    half = episode.object_half_extent_m
    quality, values = [], []
    for t in times:
        index = episode.index_at(t - spec.latency_s)
        data.qpos[:] = episode.record.arrays["qpos"][index]
        if model.nmocap:
            # A recorded occluder stands where the record's user input put it.
            user = episode.record.arrays["user_input"][index]
            mujoco.mj_setState(model, data, user, int(episode.record.arrays["input_spec"]))
        mujoco.mj_kinematics(model, data)
        fraction, centre = visible_fraction(model, data, obj, half, origin, geomid)
        if float(np.linalg.norm(centre - origin)) > spec.range_m:
            quality.append(SampleQuality.MISSING)
            values.append({"visible_fraction": 0.0})
            continue
        if fraction >= spec.visible_fraction:
            reported = centre + rng.normal(0.0, spec.noise_m, size=3)
            quality.append(SampleQuality.VALID)
            values.append({"x": float(reported[0]), "y": float(reported[1]), "z": float(reported[2]), "visible_fraction": fraction})
        else:
            quality.append(SampleQuality.OCCLUDED)
            values.append({"visible_fraction": fraction})
    return Stream(sensor, times, quality, values, f"camera {spec.name} at {spec.position_m}, {len(FACES)} rays, visible when >= {spec.visible_fraction:.0%} reach the object")


def frame_for(episode: Episode, effector: EffectorV1) -> WorkspaceFrame:
    chain = next(c for c in episode.manifest.morphology.chains if c.chain_id == effector.chain_id)
    return build_workspace_frame(episode.model, episode.manifest.morphology, chain, figure_site=figure_site_for(episode.manifest, effector.chain_id))


@dataclass
class EvidenceStreams:
    """Every sensor of a configuration, sampled over one episode."""

    episode: Episode
    configuration: SensorConfigurationV1
    streams: dict[str, Stream] = field(default_factory=dict)

    @classmethod
    def build(cls, episode: Episode, configuration: SensorConfigurationV1) -> "EvidenceStreams":
        streams: dict[str, Stream] = {}
        effectors = {e.chain_id: e for e in episode.effectors}
        arrays = episode.record.arrays
        dofs = [(d.name, int(episode.model.jnt_qposadr[mujoco.mj_name2id(episode.model, mujoco.mjtObj.mjOBJ_JOINT, d.joint)]),
                 int(episode.model.jnt_dofadr[mujoco.mj_name2id(episode.model, mujoco.mjtObj.mjOBJ_JOINT, d.joint)])) for d in episode.manifest.dofs]
        cameras: list[Stream] = []
        for sensor in configuration.sensors:
            kind = sensor.kind
            if kind is EvidenceKind.JOINT_ENCODERS:
                times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
                values = []
                for t in times:
                    i = episode.index_at(t - sensor.latency_s)
                    row = {}
                    for name, qadr, dadr in dofs:
                        row[f"q:{name}"] = float(arrays["qpos"][i][qadr])
                        row[f"dq:{name}"] = float(arrays["qvel"][i][dadr])
                    values.append(row)
                streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), values, f"joint encoders at {sensor.rate_hz:g} Hz")
            elif kind is EvidenceKind.EFFECTOR_POSE:
                effector = effectors[sensor.entity]
                site = episode.grasp_site(effector)
                times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
                values = []
                for t in times:
                    data = episode.kinematics(episode.index_at(t - sensor.latency_s))
                    p = data.site_xpos[site]
                    values.append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
                streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), values, f"grasp point of {effector.chain_id} through the calibrated model at {sensor.rate_hz:g} Hz")
            elif kind is EvidenceKind.CONTACT_FORCE:
                effector = effectors[sensor.entity]
                times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
                values = []
                for t in times:
                    groups = episode.group_forces(effector, episode.index_at(t - sensor.latency_s))
                    row = {f"group_{k}_n": float(f) for k, f in enumerate(groups)}
                    row["max_force_n"] = float(max(groups, default=0.0))
                    values.append(row)
                streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), values, f"contact wrench on the members of {effector.chain_id} at {sensor.rate_hz:g} Hz")
            elif kind is EvidenceKind.OBJECT_POSE:
                spec = CAMERAS[sensor.sensor_id.split(":", 1)[1]]
                stream = camera_stream(episode, sensor, spec)
                streams[sensor.sensor_id] = stream
                cameras.append(stream)
            elif kind is EvidenceKind.REGION:
                times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
                low, high = episode.goal.region_minimum_m, episode.goal.region_maximum_m
                row = {"min_x": low[0], "min_y": low[1], "min_z": low[2], "max_x": high[0], "max_y": high[1], "max_z": high[2]}
                streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), [dict(row) for _ in times], "the task's destination region, absolute metres")
            elif kind is EvidenceKind.ORACLE_STATE:
                times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
                values = []
                for t in times:
                    p = episode.object_position(episode.index_at(t - sensor.latency_s))
                    values.append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "privileged": 1.0})
                streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), values, "privileged full state; labels only")
        # The reach shell is read along the bearing of the last position a
        # camera reported; with no camera in the configuration or nothing seen
        # yet, along the manipulator's working direction.
        for sensor in configuration.sensors:
            if sensor.kind is not EvidenceKind.REACH_ENVELOPE:
                continue
            effector = effectors[sensor.entity]
            frame = frame_for(episode, effector)
            times = sample_times(episode, sensor.rate_hz, sensor.latency_s)
            values = []
            for t in times:
                point = None
                for camera in cameras:
                    lo = int(np.searchsorted(camera.times, t + 1e-9, side="right"))
                    for i in range(lo - 1, -1, -1):
                        if camera.quality[i] is SampleQuality.VALID:
                            point = np.array([camera.values[i]["x"], camera.values[i]["y"], camera.values[i]["z"]])
                            break
                    if point is not None:
                        break
                azimuth, elevation = frame.bearing_of(point) if point is not None else (0.0, 0.0)
                values.append({"origin_x": float(frame.origin[0]), "origin_y": float(frame.origin[1]), "origin_z": float(frame.origin[2]),
                               "inner_m": float(frame.inner_reach(azimuth, elevation)), "outer_m": float(frame.directional_reach(azimuth, elevation)),
                               "azimuth_deg": float(azimuth), "elevation_deg": float(elevation)})
            streams[sensor.sensor_id] = Stream(sensor, times, [SampleQuality.VALID] * len(times), values, f"calibrated reach shell of {effector.chain_id} along the bearing of the last sensed object position")
        return cls(episode, configuration, streams)

    def samples(self, start_s: float, end_s: float) -> list[EvidenceSampleV1]:
        out: list[EvidenceSampleV1] = []
        for stream in self.streams.values():
            out.extend(stream.window(start_s, end_s))
        return out

    def replaced(self, **streams: Stream) -> "EvidenceStreams":
        merged = dict(self.streams)
        merged.update(streams)
        return EvidenceStreams(self.episode, self.configuration, merged)

    def without(self, sensor_ids: set[str], configuration_id: str) -> "EvidenceStreams":
        sensors = tuple(s for s in self.configuration.sensors if s.sensor_id not in sensor_ids)
        configuration = SensorConfigurationV1(configuration_id=configuration_id, sensors=sensors, description=self.configuration.description + f"; without {sorted(sensor_ids)}")
        return EvidenceStreams(self.episode, configuration, {k: v for k, v in self.streams.items() if k not in sensor_ids})
