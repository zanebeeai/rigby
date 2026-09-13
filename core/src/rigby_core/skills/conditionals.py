"""Observable conditionals: a predicate about the world, decided from declared
evidence over a window, or not decided at all.

A conditional says which entities it is about, which kinds of evidence it
needs and from which roles, over what temporal window, by what rule with
what parameters, when it abstains, and what happens when it does. It is
decided by an evaluator that sees only samples from the sensors a
configuration declares: if a required kind has no sensor, the newest
sample is older than the window allows, too few samples are valid (an
occluded camera reports a sample of quality ``occluded``, not a pose), or
there are fewer samples than the rule needs, the verdict is ``unknown`` and
the conditional's fallback says what to do about it. A conditional never
lists privileged state among its evidence; an oracle sensor exists so that
labels can be made from the full state, separately and never as an input.

The six rules here are pure functions of flattened numbers, so the same
conditional binds to any sensor configuration that supplies the kinds it
names, and their thresholds are parameters of a named policy, set by
calibration on recorded episodes rather than read off a robot description.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Callable, Literal, Self

from pydantic import Field, model_validator

from ..contracts import Contract


class EvidenceKind(StrEnum):
    JOINT_ENCODERS = "joint_encoders"
    """Joint positions and velocities of the body."""
    CONTACT_FORCE = "contact_force"
    """Normal force on declared members from contact with the object, and
    whether members on opposing sides are both loaded."""
    OBJECT_POSE = "object_pose"
    """An object's position from a perception sensor that can be occluded."""
    EFFECTOR_POSE = "effector_pose"
    """The manipulator's grasp point, from the encoders through the model."""
    REACH_ENVELOPE = "reach_envelope"
    """The measured reach of the manipulator toward a queried point."""
    REGION = "region"
    """Task geometry: an axis-aligned region the task names."""
    ORACLE_STATE = "oracle_state"
    """Privileged full state. For labels only; a conditional may not require it."""


class SampleQuality(StrEnum):
    VALID = "valid"
    OCCLUDED = "occluded"
    MISSING = "missing"


class SensorSpecV1(Contract):
    sensor_id: str = Field(min_length=1)
    kind: EvidenceKind
    rate_hz: float = Field(gt=0.0)
    latency_s: float = Field(default=0.0, ge=0.0)
    max_age_s: float = Field(gt=0.0)
    """How old this sensor's newest sample may be before it counts as stale."""
    occludable: bool = False
    oracle: bool = False
    entity: str = ""
    """The entity this sensor observes, when it observes one in particular:
    a contact sensor reads one manipulator's members. Empty for a sensor
    that observes whatever is in front of it."""
    description: str = ""


class SensorConfigurationV1(Contract):
    configuration_id: str = Field(min_length=1)
    sensors: tuple[SensorSpecV1, ...] = ()
    description: str = ""

    @model_validator(mode="after")
    def distinct(self) -> Self:
        ids = [s.sensor_id for s in self.sensors]
        if len(set(ids)) != len(ids):
            raise ValueError("sensor identifiers must be distinct")
        return self

    def of_kind(self, kind: EvidenceKind) -> tuple[SensorSpecV1, ...]:
        return tuple(s for s in self.sensors if s.kind is kind and not s.oracle)

    @property
    def has_oracle(self) -> bool:
        return any(s.oracle for s in self.sensors)


class EvidenceSampleV1(Contract):
    sensor_id: str = Field(min_length=1)
    kind: EvidenceKind
    time_s: float = Field(ge=0.0)
    quality: SampleQuality = SampleQuality.VALID
    values: dict[str, float] = {}
    provenance: str = ""


class TemporalWindowV1(Contract):
    duration_s: float = Field(gt=0.0)
    max_age_s: float = Field(gt=0.0)
    """How old the newest sample of any required kind may be at decision time."""


class DecisionRuleV1(Contract):
    rule: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    parameters: dict[str, float] = {}
    policy: str = Field(min_length=1)
    """The named calibration the parameters come from."""


class AbstentionV1(Contract):
    min_valid_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    """Of the samples in the window, from each required sensor, that must be valid."""
    note: str = "missing source, stale samples, occlusion or too few samples decide nothing"


class FallbackV1(Contract):
    action: Literal["re_observe", "fail", "abstain"]
    budget: int = Field(default=1, ge=0, le=8)


class EvidenceRequirementV1(Contract):
    kind: EvidenceKind
    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    required: bool = True
    min_samples: int = Field(default=1, ge=1)
    """The fewest valid samples of this kind the window must hold to decide."""


class ConditionalV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    entities: dict[str, str] = {}
    """Role to entity identifier."""
    evidence: tuple[EvidenceRequirementV1, ...] = Field(min_length=1)
    window: TemporalWindowV1
    rule: DecisionRuleV1
    abstention: AbstentionV1 = AbstentionV1()
    fallback: FallbackV1
    description: str = ""

    @model_validator(mode="after")
    def never_privileged(self) -> Self:
        if any(e.kind is EvidenceKind.ORACLE_STATE for e in self.evidence):
            raise ValueError(f"{self.name}: a conditional may not require privileged state")
        for requirement in self.evidence:
            if requirement.role not in self.entities:
                raise ValueError(f"{self.name}: evidence names role {requirement.role!r} the conditional has no entity for")
        return self


class Decision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ConditionalVerdictV1(Contract):
    name: str
    decision: Decision
    reason: str = ""
    window_start_s: float
    window_end_s: float
    samples_used: int = 0
    sensors_used: tuple[str, ...] = ()
    kinds_used: tuple[EvidenceKind, ...] = ()
    fallback: Literal["re_observe", "fail", "abstain"] | None = None
    detail: dict[str, float] = {}


RuleFn = Callable[[dict[EvidenceKind, list[EvidenceSampleV1]], dict[str, float]], tuple[bool, dict[str, float]]]


def evaluate(conditional: ConditionalV1, configuration: SensorConfigurationV1, samples: list[EvidenceSampleV1], now_s: float,
             rules: dict[str, RuleFn] | None = None) -> ConditionalVerdictV1:
    """Decide ``conditional`` at ``now_s`` from ``samples``, which are all the
    samples the configuration's sensors produced. Anything the abstention
    rules refuse is ``unknown`` with the reason and the fallback."""

    rules = rules if rules is not None else RULES
    if conditional.rule.rule not in rules:
        raise KeyError(f"no rule named {conditional.rule.rule!r}")
    start = now_s - conditional.window.duration_s
    used: dict[EvidenceKind, list[EvidenceSampleV1]] = {}
    sensors_used: list[str] = []

    def unknown(reason: str, **detail: float) -> ConditionalVerdictV1:
        return ConditionalVerdictV1(name=conditional.name, decision=Decision.UNKNOWN, reason=reason, window_start_s=start, window_end_s=now_s,
                                    samples_used=sum(len(v) for v in used.values()), sensors_used=tuple(sensors_used),
                                    kinds_used=tuple(used), fallback=conditional.fallback.action, detail=detail)

    for requirement in conditional.evidence:
        entity = conditional.entities[requirement.role]
        sensors = tuple(s for s in configuration.of_kind(requirement.kind) if not s.entity or s.entity == entity)
        if not sensors:
            if requirement.required:
                return unknown(f"missing_source:{requirement.kind.value}")
            continue
        # Every sensor of the kind that is fresh and mostly valid contributes
        # its valid samples; a kind is undecidable only when no sensor of it
        # is usable, and then the reason is the first sensor's. A second
        # camera the hand stands under does not blind a first that sees.
        kind_samples: list[EvidenceSampleV1] = []
        refusals: list[tuple[str, dict[str, float]]] = []
        for sensor in sensors:
            own = [s for s in samples if s.sensor_id == sensor.sensor_id and start - 1e-9 <= s.time_s <= now_s + 1e-9]
            if not own:
                refusals.append((f"stale:{sensor.sensor_id}:no_sample_in_window", {}))
                continue
            newest = max(s.time_s for s in own)
            if now_s - newest > min(sensor.max_age_s, conditional.window.max_age_s) + 1e-9:
                refusals.append((f"stale:{sensor.sensor_id}", {"age_s": now_s - newest}))
                continue
            valid = [s for s in own if s.quality is SampleQuality.VALID]
            fraction = len(valid) / len(own)
            if fraction < conditional.abstention.min_valid_fraction:
                worst = max((s.quality for s in own if s.quality is not SampleQuality.VALID), key=lambda q: q.value, default=SampleQuality.MISSING)
                refusals.append((f"{worst.value}:{sensor.sensor_id}", {"valid_fraction": fraction}))
                continue
            kind_samples.extend(valid)
            sensors_used.append(sensor.sensor_id)
        if not kind_samples and refusals:
            if requirement.required:
                reason, detail = refusals[0]
                return unknown(reason, **detail)
            continue
        if len(kind_samples) < requirement.min_samples:
            if requirement.required:
                return unknown(f"insufficient_samples:{requirement.kind.value}", samples=float(len(kind_samples)))
            continue
        if kind_samples:
            used[requirement.kind] = sorted(kind_samples, key=lambda s: s.time_s)
    holds, detail = rules[conditional.rule.rule](used, dict(conditional.rule.parameters))
    return ConditionalVerdictV1(name=conditional.name, decision=Decision.PASS if holds else Decision.FAIL, reason="", window_start_s=start, window_end_s=now_s,
                                samples_used=sum(len(v) for v in used.values()), sensors_used=tuple(sensors_used), kinds_used=tuple(used), fallback=None, detail=detail)


# --------------------------------------------------------------------------
# the six rules, as pure functions of flattened numbers
# --------------------------------------------------------------------------


def _positions(samples: list[EvidenceSampleV1]) -> list[tuple[float, tuple[float, float, float]]]:
    return [(s.time_s, (s.values["x"], s.values["y"], s.values["z"])) for s in samples if all(k in s.values for k in ("x", "y", "z"))]


def _norm(v) -> float:
    return math.sqrt(sum(c * c for c in v))


def rule_reachable(used, p):
    """The object's newest sensed position lies within the manipulator's
    measured reach toward it: inner and outer reach along its bearing."""

    poses = _positions(used.get(EvidenceKind.OBJECT_POSE, []))
    envelope = used.get(EvidenceKind.REACH_ENVELOPE, [])
    if not poses or not envelope:
        return False, {}
    t, (x, y, z) = poses[-1]
    e = envelope[-1].values
    distance = _norm((x - e["origin_x"], y - e["origin_y"], z - e["origin_z"]))
    margin = p.get("margin_fraction", 0.0)
    inside = e["inner_m"] * (1.0 - margin) <= distance <= e["outer_m"] * (1.0 + margin)
    return inside, {"distance_m": distance, "inner_m": e["inner_m"], "outer_m": e["outer_m"]}


def _group_forces(sample: EvidenceSampleV1) -> list[float]:
    return [v for k, v in sorted(sample.values.items()) if k.startswith("group_") and k.endswith("_n")]


def _opposed(sample: EvidenceSampleV1, threshold_n: float) -> bool:
    """Every opposition group of the sensor loaded above the threshold."""

    groups = _group_forces(sample)
    return len(groups) >= 2 and all(f >= threshold_n for f in groups)


def rule_opposition_established(used, p):
    """Members on opposing sides are both loaded above the contact threshold,
    in at least the required fraction of the window's samples."""

    samples = used.get(EvidenceKind.CONTACT_FORCE, [])
    if not samples:
        return False, {}
    loaded = [s for s in samples if _opposed(s, p["contact_force_n"])]
    fraction = len(loaded) / len(samples)
    return fraction >= p.get("min_fraction", 0.9), {"opposed_fraction": fraction, "peak_force_n": max(max(_group_forces(s), default=0.0) for s in samples)}


def rule_held(used, p):
    """Opposition sustained across the whole window; and, when a pose sensor
    is present, the object's newest sensed position off its support by at
    least the lift threshold."""

    holds, detail = rule_opposition_established(used, {"contact_force_n": p["contact_force_n"], "min_fraction": p.get("min_fraction", 0.95)})
    if not holds:
        return False, detail
    poses = _positions(used.get(EvidenceKind.OBJECT_POSE, []))
    if poses:
        _, (_, _, z) = poses[-1]
        detail["newest_z_m"] = z
        if z < p["support_top_m"] + p["half_height_m"] + p["lift_threshold_m"]:
            return False, detail
    return True, detail


def _span(track: list[tuple[float, tuple[float, float, float]]]) -> tuple[float, tuple[float, float, float]]:
    """Duration and displacement from the first sample to the last."""

    (t0, p0), (t1, p1) = track[0], track[-1]
    return t1 - t0, tuple(b - a for a, b in zip(p0, p1))


def rule_moving_with_robot(used, p):
    """Over the window the grasp point travelled at least the minimum speed,
    and the object's sensed displacement matches the grasp point's within a
    fraction of that travel plus the position tolerance."""

    obj = _positions(used.get(EvidenceKind.OBJECT_POSE, []))
    eff = _positions(used.get(EvidenceKind.EFFECTOR_POSE, []))
    if len(obj) < 2 or len(eff) < 2:
        return False, {}
    # Pair the effector track to the object's sampling instants, so both
    # displacements span the same interval.
    first = min(eff, key=lambda item: abs(item[0] - obj[0][0]))
    last = min(eff, key=lambda item: abs(item[0] - obj[-1][0]))
    tolerance = p.get("pairing_tolerance_s", 0.05)
    if abs(first[0] - obj[0][0]) > tolerance or abs(last[0] - obj[-1][0]) > tolerance or last[0] <= first[0]:
        return False, {"paired": 0.0}
    duration, effector_displacement = _span([first, last])
    _, object_displacement = _span(obj)
    travel = _norm(effector_displacement)
    speed = travel / duration
    mismatch = _norm(tuple(a - b for a, b in zip(object_displacement, effector_displacement)))
    allowed = p.get("max_mismatch_fraction", 0.5) * travel + p.get("position_tolerance_m", 0.0)
    return speed >= p["min_speed_mps"] and mismatch <= allowed, {"effector_speed_mps": speed, "effector_travel_m": travel, "mismatch_m": mismatch, "allowed_mismatch_m": allowed, "paired": 1.0}


def rule_stably_placed(used, p):
    """The object inside the region on every sample of the window (to within
    the position tolerance), its end-to-end drift under the stillness limit,
    its samples spread no wider than the tolerance, and no member loaded."""

    poses = _positions(used.get(EvidenceKind.OBJECT_POSE, []))
    region = used.get(EvidenceKind.REGION, [])
    contacts = used.get(EvidenceKind.CONTACT_FORCE, [])
    if not poses or not region:
        return False, {}
    r = region[-1].values
    half = p.get("half_extent_m", 0.0)
    tol = p.get("position_tolerance_m", 0.0)
    inside = all(r["min_x"] + half - tol <= x <= r["max_x"] - half + tol and r["min_y"] + half - tol <= y <= r["max_y"] - half + tol and r["min_z"] + half - tol <= z <= r["max_z"] - half + tol for _, (x, y, z) in poses)
    duration, drift = _span(poses)
    speed = _norm(drift) / duration if duration > 0 else 0.0
    mean = tuple(sum(c[i] for _, c in poses) / len(poses) for i in range(3))
    spread = max(_norm(tuple(a - b for a, b in zip(c, mean))) for _, c in poses)
    still = speed <= p["max_speed_mps"] and spread <= p.get("max_spread_m", float("inf"))
    released = all(max(_group_forces(s), default=0.0) < p["contact_force_n"] for s in contacts) if contacts else True
    return inside and still and released, {"inside": float(inside), "drift_speed_mps": speed, "spread_m": spread, "released": float(released), "samples": float(len(poses))}


def rule_area_clear(used, p):
    """No sensed object position within the margin of the region on any sample of the window."""

    poses = _positions(used.get(EvidenceKind.OBJECT_POSE, []))
    region = used.get(EvidenceKind.REGION, [])
    if not poses or not region:
        return False, {}
    r = region[-1].values
    margin = p.get("margin_m", 0.0)
    inside = [x for _, (x, y, z) in poses if r["min_x"] - margin <= x <= r["max_x"] + margin and r["min_y"] - margin <= y <= r["max_y"] + margin and r["min_z"] - margin <= z <= r["max_z"] + margin]
    return not inside, {"samples_inside": float(len(inside)), "samples": float(len(poses))}


RULES: dict[str, RuleFn] = {
    "reachable": rule_reachable,
    "opposition_established": rule_opposition_established,
    "held": rule_held,
    "moving_with_robot": rule_moving_with_robot,
    "stably_placed": rule_stably_placed,
    "area_clear": rule_area_clear,
}
PREDICATES = tuple(RULES)
