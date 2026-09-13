"""Observable conditionals: representation, abstention, the six rules, and no privileged access."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rigby_core.skills import (
    PREDICATES,
    RULES,
    AbstentionV1,
    ConditionalV1,
    Decision,
    DecisionRuleV1,
    EvidenceKind,
    EvidenceRequirementV1,
    EvidenceSampleV1,
    FallbackV1,
    SampleQuality,
    SensorConfigurationV1,
    SensorSpecV1,
    TemporalWindowV1,
    evaluate,
)


def sensor(sensor_id: str, kind: EvidenceKind, *, max_age_s: float = 0.1, entity: str = "", oracle: bool = False, occludable: bool = False) -> SensorSpecV1:
    return SensorSpecV1(sensor_id=sensor_id, kind=kind, rate_hz=100.0, max_age_s=max_age_s, entity=entity, oracle=oracle, occludable=occludable)


def configuration(*sensors: SensorSpecV1) -> SensorConfigurationV1:
    return SensorConfigurationV1(configuration_id="test", sensors=sensors)


def conditional(name: str, evidence, *, rule: str, parameters: dict, duration_s: float = 0.5, max_age_s: float = 0.1, min_valid_fraction: float = 0.8) -> ConditionalV1:
    return ConditionalV1(name=name, entities={"manipulator": "palm", "object": "cube", "region": "destination"}, evidence=tuple(evidence),
                         window=TemporalWindowV1(duration_s=duration_s, max_age_s=max_age_s),
                         rule=DecisionRuleV1(rule=rule, parameters=parameters, policy="test-policy"),
                         abstention=AbstentionV1(min_valid_fraction=min_valid_fraction), fallback=FallbackV1(action="re_observe", budget=1))


def contact_samples(times, forces, sensor_id="contact:palm"):
    return [EvidenceSampleV1(sensor_id=sensor_id, kind=EvidenceKind.CONTACT_FORCE, time_s=t, values={"group_0_n": f[0], "group_1_n": f[1]}) for t, f in zip(times, forces)]


def pose_samples(times, points, sensor_id="camera:front", quality=SampleQuality.VALID):
    return [EvidenceSampleV1(sensor_id=sensor_id, kind=EvidenceKind.OBJECT_POSE, time_s=t, quality=quality, values={"x": p[0], "y": p[1], "z": p[2]} if quality is SampleQuality.VALID else {})
            for t, p in zip(times, points)]


OPPOSITION = conditional("opposition_established", [EvidenceRequirementV1(kind=EvidenceKind.CONTACT_FORCE, role="manipulator", min_samples=2)], rule="opposition_established",
                         parameters={"contact_force_n": 0.5, "min_fraction": 0.9})
CONTACT = sensor("contact:palm", EvidenceKind.CONTACT_FORCE, entity="palm")
TIMES = [0.51 + 0.01 * i for i in range(50)]


# -- representation -------------------------------------------------------------


def test_conditional_carries_every_declared_part():
    c = OPPOSITION
    assert c.entities["manipulator"] == "palm"
    assert c.evidence[0].kind is EvidenceKind.CONTACT_FORCE and c.evidence[0].role == "manipulator"
    assert c.window.duration_s == 0.5 and c.evidence[0].min_samples == 2 and c.rule.policy == "test-policy"
    assert c.abstention.min_valid_fraction == 0.8 and c.fallback.action == "re_observe"
    assert set(PREDICATES) == set(RULES) == {"reachable", "opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear"}


def test_conditional_refuses_privileged_state_and_unbound_roles():
    with pytest.raises(ValidationError, match="privileged"):
        conditional("held", [EvidenceRequirementV1(kind=EvidenceKind.ORACLE_STATE, role="object")], rule="held", parameters={})
    with pytest.raises(ValidationError, match="no entity"):
        ConditionalV1(name="x", entities={}, evidence=(EvidenceRequirementV1(kind=EvidenceKind.CONTACT_FORCE, role="manipulator"),),
                      window=TemporalWindowV1(duration_s=0.5, max_age_s=0.1), rule=DecisionRuleV1(rule="held", policy="p"), fallback=FallbackV1(action="fail"))


def test_configuration_refuses_duplicate_sensors():
    with pytest.raises(ValidationError, match="distinct"):
        configuration(CONTACT, CONTACT)


# -- deciding and abstaining --------------------------------------------------------


def test_opposition_passes_and_fails_on_evidence():
    held = contact_samples(TIMES, [(1.0, 1.2)] * 50)
    verdict = evaluate(OPPOSITION, configuration(CONTACT), held, now_s=1.0)
    assert verdict.decision is Decision.PASS and verdict.sensors_used == ("contact:palm",) and verdict.fallback is None
    one_sided = contact_samples(TIMES, [(1.0, 0.0)] * 50)
    assert evaluate(OPPOSITION, configuration(CONTACT), one_sided, now_s=1.0).decision is Decision.FAIL


def test_missing_source_is_unknown_with_fallback():
    verdict = evaluate(OPPOSITION, configuration(sensor("encoders", EvidenceKind.JOINT_ENCODERS)), [], now_s=1.0)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason == "missing_source:contact_force" and verdict.fallback == "re_observe"


def test_stale_sensor_is_unknown_not_a_pass():
    held = contact_samples(TIMES, [(1.0, 1.2)] * 50)
    verdict = evaluate(OPPOSITION, configuration(CONTACT), held, now_s=1.5)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason.startswith("stale:contact:palm") and verdict.detail["age_s"] == pytest.approx(0.5)


def test_occluded_samples_are_unknown_below_the_valid_fraction():
    c = conditional("area_clear", [EvidenceRequirementV1(kind=EvidenceKind.OBJECT_POSE, role="object"), EvidenceRequirementV1(kind=EvidenceKind.REGION, role="region")],
                    rule="area_clear", parameters={"margin_m": 0.0})
    camera = sensor("camera:front", EvidenceKind.OBJECT_POSE, occludable=True)
    region = sensor("region", EvidenceKind.REGION, max_age_s=10.0)
    region_sample = [EvidenceSampleV1(sensor_id="region", kind=EvidenceKind.REGION, time_s=1.0, values={"min_x": 0, "min_y": 0, "min_z": 0, "max_x": 1, "max_y": 1, "max_z": 1})]
    seen = pose_samples(TIMES, [(2.0, 2.0, 2.0)] * 50)
    hidden = pose_samples(TIMES[:40], [(2.0, 2.0, 2.0)] * 40, quality=SampleQuality.OCCLUDED) + pose_samples(TIMES[40:], [(2.0, 2.0, 2.0)] * 10)
    assert evaluate(c, configuration(camera, region), seen + region_sample, now_s=1.0).decision is Decision.PASS
    verdict = evaluate(c, configuration(camera, region), hidden + region_sample, now_s=1.0)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason == "occluded:camera:front" and verdict.detail["valid_fraction"] == pytest.approx(0.2)


def test_too_few_samples_is_unknown():
    c = OPPOSITION.model_copy(update={"evidence": (EvidenceRequirementV1(kind=EvidenceKind.CONTACT_FORCE, role="manipulator", min_samples=20),)})
    verdict = evaluate(c, configuration(CONTACT), contact_samples(TIMES[-5:], [(1.0, 1.2)] * 5), now_s=1.0)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason == "insufficient_samples:contact_force"


def test_optional_evidence_is_dropped_when_stale_rather_than_used():
    c = conditional("held", [EvidenceRequirementV1(kind=EvidenceKind.CONTACT_FORCE, role="manipulator"), EvidenceRequirementV1(kind=EvidenceKind.OBJECT_POSE, role="object", required=False)],
                    rule="held", parameters={"contact_force_n": 0.5, "min_fraction": 0.95, "support_top_m": 0.26, "half_height_m": 0.015, "lift_threshold_m": 0.015})
    camera = sensor("camera:front", EvidenceKind.OBJECT_POSE, occludable=True)
    held = contact_samples(TIMES, [(1.0, 1.2)] * 50)
    on_bench_fresh = pose_samples(TIMES, [(0.0, 0.0, 0.275)] * 50)
    assert evaluate(c, configuration(CONTACT, camera), held + on_bench_fresh, now_s=1.0).decision is Decision.FAIL
    on_bench_stale = pose_samples([0.2, 0.3], [(0.0, 0.0, 0.275)] * 2)
    verdict = evaluate(c, configuration(CONTACT, camera), held + on_bench_stale, now_s=1.0)
    assert verdict.decision is Decision.PASS and verdict.kinds_used == (EvidenceKind.CONTACT_FORCE,)


# -- no privileged access -------------------------------------------------------------


def test_oracle_sensor_is_never_read():
    oracle = sensor("oracle", EvidenceKind.ORACLE_STATE, oracle=True)
    lies = [EvidenceSampleV1(sensor_id="oracle", kind=EvidenceKind.ORACLE_STATE, time_s=t, values={"held": 1.0}) for t in TIMES]
    one_sided = contact_samples(TIMES, [(1.0, 0.0)] * 50)
    without = evaluate(OPPOSITION, configuration(CONTACT), one_sided, now_s=1.0)
    with_oracle = evaluate(OPPOSITION, configuration(CONTACT, oracle), one_sided + lies, now_s=1.0)
    assert without == with_oracle and with_oracle.decision is Decision.FAIL
    # An oracle-only configuration supplies no source for the kind.
    assert evaluate(OPPOSITION, configuration(oracle), lies, now_s=1.0).reason == "missing_source:contact_force"


def test_sensors_are_scoped_to_the_entity_named_by_the_role():
    left = sensor("contact:left", EvidenceKind.CONTACT_FORCE, entity="left")
    right = sensor("contact:right", EvidenceKind.CONTACT_FORCE, entity="right")
    c = OPPOSITION.model_copy(update={"entities": {"manipulator": "left", "object": "cube", "region": "destination"}})
    right_holds = contact_samples(TIMES, [(1.0, 1.2)] * 50, sensor_id="contact:right")
    left_open = contact_samples(TIMES, [(0.0, 0.0)] * 50, sensor_id="contact:left")
    verdict = evaluate(c, configuration(left, right), right_holds + left_open, now_s=1.0)
    assert verdict.decision is Decision.FAIL and verdict.sensors_used == ("contact:left",)


def test_same_contract_binds_to_different_configurations():
    """One conditional, two configurations: decided where the kind exists, unknown where it does not; the contract is the same object."""

    held = contact_samples(TIMES, [(1.0, 1.2)] * 50)
    contact_only = configuration(CONTACT)
    vision_only = configuration(sensor("camera:front", EvidenceKind.OBJECT_POSE))
    assert evaluate(OPPOSITION, contact_only, held, now_s=1.0).decision is Decision.PASS
    assert evaluate(OPPOSITION, vision_only, held, now_s=1.0).reason == "missing_source:contact_force"
    assert OPPOSITION.model_dump() == OPPOSITION.model_dump()


# -- the rules --------------------------------------------------------------------


def test_reachable_rule_uses_the_shell_along_the_bearing():
    used = {EvidenceKind.OBJECT_POSE: pose_samples([1.0], [(0.5, 0.0, 0.0)]),
            EvidenceKind.REACH_ENVELOPE: [EvidenceSampleV1(sensor_id="reach", kind=EvidenceKind.REACH_ENVELOPE, time_s=1.0, values={"origin_x": 0, "origin_y": 0, "origin_z": 0, "inner_m": 0.1, "outer_m": 0.6})]}
    assert RULES["reachable"](used, {"margin_fraction": 0.0})[0]
    used[EvidenceKind.OBJECT_POSE] = pose_samples([1.0], [(0.7, 0.0, 0.0)])
    assert not RULES["reachable"](used, {"margin_fraction": 0.0})[0]
    assert RULES["reachable"](used, {"margin_fraction": 0.2})[0]


def test_moving_with_robot_compares_window_displacements():
    times = [0.6, 0.7, 0.8, 0.9, 1.0]
    effector = [EvidenceSampleV1(sensor_id="effector_pose:palm", kind=EvidenceKind.EFFECTOR_POSE, time_s=t, values={"x": 0.1 * i, "y": 0.0, "z": 0.0}) for i, t in enumerate(times)]
    together = pose_samples(times, [(0.1 * i + 0.02, 0.0, 0.0) for i in range(5)])
    left_behind = pose_samples(times, [(0.0, 0.0, 0.0)] * 5)
    p = {"min_speed_mps": 0.05, "max_mismatch_fraction": 0.5, "position_tolerance_m": 0.006, "pairing_tolerance_s": 0.05}
    assert RULES["moving_with_robot"]({EvidenceKind.OBJECT_POSE: together, EvidenceKind.EFFECTOR_POSE: effector}, p)[0]
    assert not RULES["moving_with_robot"]({EvidenceKind.OBJECT_POSE: left_behind, EvidenceKind.EFFECTOR_POSE: effector}, p)[0]
    still = [s.model_copy(update={"values": {"x": 0.0, "y": 0.0, "z": 0.0}}) for s in effector]
    assert not RULES["moving_with_robot"]({EvidenceKind.OBJECT_POSE: left_behind, EvidenceKind.EFFECTOR_POSE: still}, p)[0]


def test_stably_placed_and_area_clear_are_complements_on_the_region():
    region = [EvidenceSampleV1(sensor_id="region", kind=EvidenceKind.REGION, time_s=1.0, values={"min_x": 0, "min_y": 0, "min_z": 0, "max_x": 1, "max_y": 1, "max_z": 1})]
    inside = pose_samples([0.6, 0.8, 1.0], [(0.5, 0.5, 0.5)] * 3)
    outside = pose_samples([0.6, 0.8, 1.0], [(2.0, 0.5, 0.5)] * 3)
    placed = {"max_speed_mps": 0.01, "max_spread_m": 0.01, "position_tolerance_m": 0.0, "contact_force_n": 0.5, "half_extent_m": 0.1}
    assert RULES["stably_placed"]({EvidenceKind.OBJECT_POSE: inside, EvidenceKind.REGION: region}, placed)[0]
    assert not RULES["stably_placed"]({EvidenceKind.OBJECT_POSE: outside, EvidenceKind.REGION: region}, placed)[0]
    loaded = contact_samples([1.0], [(1.0, 1.0)])
    assert not RULES["stably_placed"]({EvidenceKind.OBJECT_POSE: inside, EvidenceKind.REGION: region, EvidenceKind.CONTACT_FORCE: loaded}, placed)[0]
    drifting = pose_samples([0.6, 0.8, 1.0], [(0.5, 0.5, 0.5), (0.51, 0.5, 0.5), (0.52, 0.5, 0.5)])
    assert not RULES["stably_placed"]({EvidenceKind.OBJECT_POSE: drifting, EvidenceKind.REGION: region}, placed)[0]
    assert RULES["area_clear"]({EvidenceKind.OBJECT_POSE: outside, EvidenceKind.REGION: region}, {"margin_m": 0.1})[0]
    assert not RULES["area_clear"]({EvidenceKind.OBJECT_POSE: inside, EvidenceKind.REGION: region}, {"margin_m": 0.1})[0]


def test_verdict_records_its_window_and_evidence():
    held = contact_samples(TIMES, [(1.0, 1.2)] * 50)
    verdict = evaluate(OPPOSITION, configuration(CONTACT), held, now_s=1.0)
    assert (verdict.window_start_s, verdict.window_end_s) == (0.5, 1.0)
    assert verdict.samples_used == 50 and verdict.kinds_used == (EvidenceKind.CONTACT_FORCE,)
    assert verdict.detail["opposed_fraction"] == 1.0 and verdict.detail["peak_force_n"] == 1.2


def test_a_second_sensor_of_the_kind_that_is_blind_does_not_blind_the_first():
    c = conditional("area_clear", [EvidenceRequirementV1(kind=EvidenceKind.OBJECT_POSE, role="object"), EvidenceRequirementV1(kind=EvidenceKind.REGION, role="region")],
                    rule="area_clear", parameters={"margin_m": 0.0})
    front = sensor("camera:front", EvidenceKind.OBJECT_POSE, occludable=True)
    overhead = sensor("camera:overhead", EvidenceKind.OBJECT_POSE, occludable=True)
    region = sensor("region", EvidenceKind.REGION, max_age_s=10.0)
    region_sample = [EvidenceSampleV1(sensor_id="region", kind=EvidenceKind.REGION, time_s=1.0, values={"min_x": 0, "min_y": 0, "min_z": 0, "max_x": 1, "max_y": 1, "max_z": 1})]
    seen = pose_samples(TIMES, [(2.0, 2.0, 2.0)] * 50)
    hidden = pose_samples(TIMES, [(2.0, 2.0, 2.0)] * 50, sensor_id="camera:overhead", quality=SampleQuality.OCCLUDED)
    verdict = evaluate(c, configuration(front, overhead, region), seen + hidden + region_sample, now_s=1.0)
    assert verdict.decision is Decision.PASS
    assert "camera:front" in verdict.sensors_used and "camera:overhead" not in verdict.sensors_used
    both_hidden = pose_samples(TIMES, [(2.0, 2.0, 2.0)] * 50, quality=SampleQuality.OCCLUDED) + hidden
    verdict = evaluate(c, configuration(front, overhead, region), both_hidden + region_sample, now_s=1.0)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason == "occluded:camera:front"
