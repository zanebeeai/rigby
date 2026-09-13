"""The six conditionals, instantiated for a task under a calibrated policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rigby_core.skills import (
    AbstentionV1,
    ConditionalV1,
    ConditionalVerdictV1,
    DecisionRuleV1,
    EvidenceKind,
    EvidenceRequirementV1,
    FallbackV1,
    TemporalWindowV1,
    evaluate,
)

from ..contracts import EffectorV1
from .episode import Episode
from .sensors import EvidenceStreams


def load_policy(path: Path) -> dict:
    return json.loads(Path(path).read_bytes())


def policy_digest(policy: dict) -> str:
    return hashlib.sha256((json.dumps(policy, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")).hexdigest()


def conditionals_for(episode: Episode, effector: EffectorV1, policy: dict) -> dict[str, ConditionalV1]:
    """The predicate contracts for one manipulator and the task's object and
    region, bound over a recorded episode."""

    return bind_conditionals(policy, effector.chain_id, support_top_m=float(episode.support_top_m), half_height_m=float(episode.object_half_extent_m[2]),
                             half_extent_m=float(max(episode.object_half_extent_m)))


def bind_conditionals(policy: dict, manipulator: str, *, support_top_m: float, half_height_m: float, half_extent_m: float) -> dict[str, ConditionalV1]:
    """The predicate contracts for one manipulator and the task's object and
    region. Thresholds come from the policy; the object's size, its
    support's height and the region come from the task, never from the
    robot description."""

    name = policy["policy_id"]
    thresholds = policy["rules"]
    windows = policy["windows"]
    height = 2.0 * half_height_m
    half_extent = half_extent_m
    entities = {"manipulator": manipulator, "object": "cube", "region": "destination"}

    def window(rule: str) -> TemporalWindowV1:
        w = windows[rule]
        return TemporalWindowV1(duration_s=w["duration_s"], max_age_s=w["max_age_s"])

    def requirement(kind: EvidenceKind, role: str, required: bool = True, rule: str | None = None) -> EvidenceRequirementV1:
        """The primary evidence of a rule (its first requirement) needs the
        policy's minimum sample count; static kinds need one sample."""

        return EvidenceRequirementV1(kind=kind, role=role, required=required, min_samples=windows[rule]["min_samples"] if rule else 1)

    abstention = AbstentionV1(min_valid_fraction=policy["abstention"]["min_valid_fraction"])
    contact_n = float(thresholds["contact_force_n"])
    tolerance = float(thresholds["position_tolerance_m"])
    out = {
        "reachable": ConditionalV1(
            name="reachable", entities=entities,
            evidence=(requirement(EvidenceKind.OBJECT_POSE, "object", rule="reachable"), requirement(EvidenceKind.REACH_ENVELOPE, "manipulator")),
            window=window("reachable"),
            rule=DecisionRuleV1(rule="reachable", parameters={"margin_fraction": float(thresholds["reachable"]["margin_fraction"])}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="re_observe", budget=2),
            description="the object's sensed position lies within the manipulator's calibrated reach shell along its bearing"),
        "opposition_established": ConditionalV1(
            name="opposition_established", entities=entities,
            evidence=(requirement(EvidenceKind.CONTACT_FORCE, "manipulator", rule="opposition_established"),),
            window=window("opposition_established"),
            rule=DecisionRuleV1(rule="opposition_established", parameters={"contact_force_n": contact_n, "min_fraction": float(thresholds["opposition_established"]["min_fraction"])}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="fail", budget=0),
            description="every opposition group of the manipulator loaded by the object above the contact threshold"),
        "held": ConditionalV1(
            name="held", entities=entities,
            evidence=(requirement(EvidenceKind.CONTACT_FORCE, "manipulator", rule="held"), requirement(EvidenceKind.OBJECT_POSE, "object", required=False)),
            window=window("held"),
            rule=DecisionRuleV1(rule="held", parameters={"contact_force_n": contact_n, "min_fraction": float(thresholds["held"]["min_fraction"]),
                                                          "support_top_m": support_top_m, "half_height_m": 0.5 * height,
                                                          "lift_threshold_m": float(thresholds["held"]["lift_fraction_of_height"]) * height}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="re_observe", budget=1),
            description="opposition sustained through the window and, when a camera sees the object, the object off its support"),
        "moving_with_robot": ConditionalV1(
            name="moving_with_robot", entities=entities,
            evidence=(requirement(EvidenceKind.OBJECT_POSE, "object", rule="moving_with_robot"), requirement(EvidenceKind.EFFECTOR_POSE, "manipulator")),
            window=window("moving_with_robot"),
            rule=DecisionRuleV1(rule="moving_with_robot", parameters={"min_speed_mps": float(thresholds["moving_with_robot"]["min_speed_mps"]),
                                                                       "max_mismatch_fraction": float(thresholds["moving_with_robot"]["max_mismatch_fraction"]),
                                                                       "position_tolerance_m": tolerance,
                                                                       "pairing_tolerance_s": float(thresholds["moving_with_robot"]["pairing_tolerance_s"])}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="re_observe", budget=2),
            description="the object's sensed velocity tracks the grasp point's while the grasp point moves"),
        "stably_placed": ConditionalV1(
            name="stably_placed", entities=entities,
            evidence=(requirement(EvidenceKind.OBJECT_POSE, "object", rule="stably_placed"), requirement(EvidenceKind.REGION, "region"), requirement(EvidenceKind.CONTACT_FORCE, "manipulator", required=False)),
            window=window("stably_placed"),
            rule=DecisionRuleV1(rule="stably_placed", parameters={"max_speed_mps": float(thresholds["stably_placed"]["max_speed_mps"]), "max_spread_m": float(thresholds["stably_placed"]["max_spread_m"]),
                                                                   "position_tolerance_m": tolerance, "contact_force_n": contact_n, "half_extent_m": half_extent}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="re_observe", budget=3),
            description="the object inside the region on every sample of the window, still, with no member loaded"),
        "area_clear": ConditionalV1(
            name="area_clear", entities=entities,
            evidence=(requirement(EvidenceKind.OBJECT_POSE, "object", rule="area_clear"), requirement(EvidenceKind.REGION, "region")),
            window=window("area_clear"),
            rule=DecisionRuleV1(rule="area_clear", parameters={"margin_m": half_extent + tolerance}, policy=name),
            abstention=abstention, fallback=FallbackV1(action="re_observe", budget=2),
            description="no sensed object position within the object's half extent of the region on any sample of the window"),
    }
    return out


def decide(conditional: ConditionalV1, streams: EvidenceStreams, now_s: float) -> ConditionalVerdictV1:
    """The conditional decided at ``now_s`` from the streams' samples in its window."""

    start = now_s - conditional.window.duration_s
    return evaluate(conditional, streams.configuration, streams.samples(start, now_s), now_s)
