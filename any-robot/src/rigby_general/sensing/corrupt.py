"""Engineered degradations of the evidence: what a monitor must not be fooled by.

Each takes the streams a configuration produced and returns streams with
less in them. A screen is a physical thing: a box is added to a copy of
the recorded world and the camera's rays are cast again against it, so
what the camera then reports is what it would have seen with the screen
there. A frozen sensor stops reporting at an instant; the samples it
produced before stay, and stay dated. An absent sensor is simply not in
the configuration. A thinned sensor reports at a lower rate.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
from rigby_core.skills import EvidenceKind, SensorConfigurationV1

from .episode import Episode
from .sensors import CAMERAS, EvidenceStreams, camera_stream


def occluded_by_screen(streams: EvidenceStreams, camera_id: str, centre_m: tuple[float, float, float], half_m: tuple[float, float, float]) -> EvidenceStreams:
    """The same episode seen past a screen between the camera and the fixtures."""

    episode = streams.episode
    xml = (episode.root / "model.xml").read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "geom", name="g09_screen", type="box", size=" ".join(f"{v:.4f}" for v in half_m), pos=" ".join(f"{v:.4f}" for v in centre_m),
                  contype="0", conaffinity="0", rgba="0.2 0.2 0.2 0.8")
    compiler = root.find("compiler")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != episode.model.nq:
        raise ValueError("the screened world changed the configuration layout")
    sensor = next(s for s in streams.configuration.sensors if s.sensor_id == camera_id)
    spec = CAMERAS[camera_id.split(":", 1)[1]]
    stream = camera_stream(episode, sensor, spec, model=model)
    stream.provenance += f"; a {2 * half_m[0]:.2f} x {2 * half_m[1]:.2f} m screen at {centre_m}"
    return streams.replaced(**{camera_id: stream})


def stale(streams: EvidenceStreams, sensor_ids: set[str], frozen_at_s: float) -> EvidenceStreams:
    """The named sensors report nothing after ``frozen_at_s``."""

    return streams.replaced(**{k: v.truncated(frozen_at_s) for k, v in streams.streams.items() if k in sensor_ids})


def absent(streams: EvidenceStreams, kind: EvidenceKind) -> EvidenceStreams:
    """The configuration without any sensor of ``kind``."""

    missing = {s.sensor_id for s in streams.configuration.sensors if s.kind is kind}
    return streams.without(missing, f"{streams.configuration.configuration_id}-without-{kind.value}")


def sparse(streams: EvidenceStreams, sensor_ids: set[str], rate_hz: float) -> EvidenceStreams:
    """The named sensors thinned to ``rate_hz``; the configuration declares the new rate."""

    thinned = {k: v.thinned(rate_hz) for k, v in streams.streams.items() if k in sensor_ids}
    sensors = tuple(thinned[s.sensor_id].sensor if s.sensor_id in thinned else s for s in streams.configuration.sensors)
    configuration = SensorConfigurationV1(configuration_id=f"{streams.configuration.configuration_id}-sparse", sensors=sensors, description=streams.configuration.description + f"; {sorted(sensor_ids)} at {rate_hz:g} Hz")
    merged = dict(streams.streams)
    merged.update(thinned)
    return EvidenceStreams(streams.episode, configuration, merged)


def sensors_of(streams: EvidenceStreams, kind: EvidenceKind) -> set[str]:
    return {s.sensor_id for s in streams.configuration.sensors if s.kind is kind}


__all__ = ["absent", "occluded_by_screen", "sensors_of", "sparse", "stale", "Episode"]
