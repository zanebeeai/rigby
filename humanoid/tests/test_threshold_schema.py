"""Schema guard for config/thresholds.v1.json.

Plan 08 §5. The point of the file is that no gate can carry a number without
saying where the number came from, so an entry missing its provenance must fail
the suite rather than be caught in review.
"""

from __future__ import annotations

import json
import struct

import pytest

from rigby_poc.thresholds import (
    EMPIRICAL_KINDS,
    REQUIRED_SOURCE_FIELDS,
    THRESHOLDS_FILE,
    Threshold,
    ThresholdError,
    schema_version,
    threshold,
    thresholds,
    value_of,
)

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _raw() -> dict:
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))


def test_file_is_committed_and_versioned() -> None:
    assert THRESHOLDS_FILE.exists()
    assert schema_version() == "1.0"


def test_every_entry_declares_a_value() -> None:
    """Either a scalar ``value`` or a ``default`` plus a ``by_family`` map."""

    for key, entry in _raw()["thresholds"].items():
        scalar = "value" in entry
        per_family = "default" in entry and "by_family" in entry
        assert scalar != per_family, (
            f"{key} must declare exactly one of `value` or `default`+`by_family`"
        )


@pytest.mark.parametrize("field", ["unit", "applies_to", "source"])
def test_every_entry_declares_the_mandatory_fields(field: str) -> None:
    for key, entry in _raw()["thresholds"].items():
        assert field in entry, f"{key} is missing the mandatory `{field}`"


def test_applies_to_is_a_non_empty_list() -> None:
    for key, entry in _raw()["thresholds"].items():
        assert isinstance(entry["applies_to"], list) and entry["applies_to"], (
            f"{key}.applies_to must be a non-empty list"
        )


def test_source_kind_is_known() -> None:
    for key, entry in _raw()["thresholds"].items():
        kind = entry["source"].get("kind")
        assert kind in REQUIRED_SOURCE_FIELDS, f"{key} has unknown source kind {kind!r}"


def test_every_source_cites_something() -> None:
    for key, entry in _raw()["thresholds"].items():
        cites = entry["source"].get("cites")
        assert isinstance(cites, list) and cites, f"{key}.source.cites must be non-empty"
        for citation in cites:
            assert isinstance(citation, str) and citation.strip(), (
                f"{key} has an empty citation"
            )


def test_source_carries_the_fields_its_kind_requires() -> None:
    for key, entry in _raw()["thresholds"].items():
        source = entry["source"]
        for field in REQUIRED_SOURCE_FIELDS[source["kind"]]:
            assert field in source and source[field] not in (None, ""), (
                f"{key}.source is kind={source['kind']!r} so it must state `{field}`"
            )


def test_an_empirical_source_cannot_omit_its_n() -> None:
    """The rule that stops a ceiling from being stated as a bare float.

    A source that claims the number came from data must be structurally unable
    to hold that number without the n it was derived from. Plan 10 §9.5 makes
    the same rule a failing guard for published rates.
    """

    for key, entry in _raw()["thresholds"].items():
        source = entry["source"]
        if source["kind"] not in EMPIRICAL_KINDS:
            continue
        assert isinstance(source.get("n"), int) and source["n"] > 0, (
            f"{key} claims an empirical source but states no positive n"
        )


def test_a_measured_source_states_the_zero_it_was_measured_against() -> None:
    """A correct n over a correct computation against the wrong zero is still wrong.

    Lane `anatomy` published, then retracted, an elbow hyperextension figure that
    compared rig-rest-relative DOF deltas against clinical references assuming
    anatomical neutral. The n was right and the arithmetic was right. No
    n-and-baseline check catches that; requiring the reference next to the number
    does.
    """

    for key, entry in _raw()["thresholds"].items():
        source = entry["source"]
        if source["kind"] not in EMPIRICAL_KINDS:
            continue
        assert source.get("reference_frame"), (
            f"{key} claims an empirical source but does not state the reference "
            f"frame its measurement is relative to"
        )


def test_a_provisional_source_says_so_in_its_rationale() -> None:
    """`provisional` is honest only if the entry does not read as validated."""

    for key, entry in _raw()["thresholds"].items():
        if entry["source"]["kind"] != "provisional":
            continue
        rationale = entry["source"]["rationale"]
        assert len(rationale) > 40, f"{key} is provisional with a one-word rationale"


def test_per_family_entries_declare_their_derivation_contract() -> None:
    """An empty ``by_family`` is legal, but it must name who fills it.

    08 §6.2 ships `default` only on purpose. What makes that honest rather than
    an omission is that the entry states the owner and the fields any future
    value must carry.
    """

    for key, entry in _raw()["thresholds"].items():
        if "by_family" not in entry:
            continue
        contract = entry.get("by_family_contract")
        assert contract, f"{key} has by_family but no by_family_contract"
        assert contract.get("owner"), f"{key}.by_family_contract must name an owner"
        for field in ("n", "bone_set"):
            assert field in contract.get("requires", []), (
                f"{key}.by_family_contract must require `{field}` of any value written "
                f"into it"
            )


def test_an_unfilled_by_family_states_why_it_is_unfilled() -> None:
    """An empty map is legal only if it is a decision rather than an omission.

    08 §6.2 resolved 2026-08-24: not derivable from the corpus. The contract has
    to carry the reason, or a later reader cannot tell "nobody got to it" from
    "we established this cannot be done from this source".
    """

    for key, entry in _raw()["thresholds"].items():
        if entry.get("by_family") != {}:
            continue
        contract = entry["by_family_contract"]
        assert contract.get("resolution"), (
            f"{key}.by_family is empty; the contract must say why, not just who"
        )
        assert contract.get("resolved"), f"{key}.by_family_contract must be dated"


def test_by_family_values_are_a_subset_of_the_declared_families() -> None:
    for key, entry in _raw()["thresholds"].items():
        if "by_family" not in entry:
            continue
        declared = set(entry["by_family_contract"].get("families", []))
        unknown = set(entry["by_family"]) - declared
        assert not unknown, f"{key}.by_family has undeclared families {sorted(unknown)}"


def test_keys_are_dotted_and_namespaced() -> None:
    for key in _raw()["thresholds"]:
        assert "." in key, f"{key} must be namespaced, e.g. signal.<name>"
        assert key == key.lower(), f"{key} must be lower case"


def test_loader_returns_thresholds() -> None:
    loaded = thresholds()
    assert loaded
    assert all(isinstance(value, Threshold) for value in loaded.values())
    assert set(loaded) == set(_raw()["thresholds"])


def test_unknown_key_fails_loudly() -> None:
    with pytest.raises(ThresholdError):
        threshold("signal.does_not_exist")


def test_scalar_lookup_resolves() -> None:
    assert value_of("physics.root_drift_max_m") == pytest.approx(1e-6)
    assert value_of("signal.discontinuity_rad") == pytest.approx(0.35)


def test_empty_by_family_resolves_to_default_for_every_family() -> None:
    """The deliberate 08 §6.2 state: `default` only, per-family derived in 10."""

    entry = threshold("signal.angular_velocity_max_rad_s")
    assert entry.is_per_family
    assert entry.by_family == {}
    for family in ("gesture", "strike", "full_body", "locomotion"):
        assert entry.for_family(family) == pytest.approx(9.5)
    assert entry.for_family(None) == pytest.approx(9.5)


def test_applies_to_wildcard() -> None:
    assert threshold("physics.root_drift_max_m").applies("strike")
    assert threshold("physics.min_lift_m").applies("grasp")
    assert not threshold("physics.min_lift_m").applies("gesture")


def test_ported_values_match_their_cited_config_source() -> None:
    """08a is a port. If a ported number drifts from its origin, fail here.

    This is what makes 08a auditable: every value below is read back out of the
    file it was cited from, so a typo in the port cannot pass silently.
    """

    criteria = json.loads(
        (THRESHOLDS_FILE.parents[1] / "acceptance_criteria.yaml").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (THRESHOLDS_FILE.parent / "motion_quality_reference.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (
            THRESHOLDS_FILE.parent / "rig_profiles" / "mesh2motion-human-vrm1.json"
        ).read_text(encoding="utf-8")
    )

    limits = quality["hard_limits"]
    assert value_of("signal.angular_velocity_max_rad_s") == limits["angular_velocity_rad_s"]
    assert value_of("signal.angular_acceleration_max_rad_s2") == limits["angular_acceleration_rad_s2"]
    assert value_of("signal.angular_jerk_max_rad_s3") == limits["angular_jerk_rad_s3"]
    assert value_of("anatomy.wrist_swing_max_rad") == limits["wrist_swing_rad"]
    assert value_of("anatomy.wrist_twist_max_rad") == limits["wrist_twist_rad"]
    assert value_of("anatomy.forearm_twist_max_rad") == limits["forearm_twist_rad"]
    assert (
        value_of("signal.min_active_hand_visibility_fraction")
        == limits["minimum_active_hand_visibility_fraction"]
    )

    safety = criteria["safety"]
    assert value_of("safety.joint_limit_violations_max") == safety["max_joint_limit_violations"]
    assert value_of("safety.nan_count_max") == safety["max_nan_count"]
    assert value_of("signal.discontinuity_count_max") == safety["max_discontinuities"]
    assert value_of("safety.penetration_max_m") == safety["max_penetration_m"]
    assert (
        value_of("safety.unresolved_non_hand_collisions_max")
        == safety["max_unresolved_non_hand_collisions"]
    )
    assert value_of("physics.foot_drift_max_m") == safety["max_foot_drift_m"]

    proof = criteria["physical_proof"]
    assert value_of("physics.min_lift_m") == proof["min_lift_m"]
    assert value_of("physics.min_hold_s") == proof["min_hold_s"]
    assert value_of("physics.max_vertical_drift_m") == proof["max_vertical_drift_m"]
    assert value_of("physics.max_palm_relative_slip_m") == proof["max_palm_relative_slip_m"]

    forward = criteria["semantic_forward_space"]
    assert value_of("semantic.reach_tolerance_m") == forward["reach_tolerance_m"]
    assert value_of("semantic.min_forward_projection_m") == forward["min_forward_projection_m"]
    assert (
        value_of("camera.max_angle_from_rig_forward_deg")
        == criteria["egocentric_camera_orientation"]["max_angle_from_rig_forward_deg"]
    )
    assert (
        value_of("physics.max_coordinate_roundtrip_error_m")
        == criteria["structural_physics"]["max_coordinate_roundtrip_error_m"]
    )

    calibration = profile["coordinate_calibration"]
    assert value_of("anatomy.shoulder_origin_m") == calibration["rest_world_pivots_m"]["leftUpperArm"]
    # The arm-length entries were retired with the primitives constants they
    # duplicated: generation and measurement now derive per-side lengths from
    # the rig itself (kinematics.arm_calibration). The profile's rounded copy
    # deliberately remains -- it feeds evals/orientation.py's tolerance-gated
    # reach plausibility check, a consumer class where 4 decimals is honest.


def test_generator_ceilings_are_bit_identical_to_the_constants_they_replace() -> None:
    """§1.1 items 6 and 7. Approximate equality is not good enough here.

    These three values replace literals that feed a frame-count division and a
    clamp, so a sub-ulp difference is a different clip, not a rounding detail.
    The subdivision target is the one that proves the point: 0.35 * 0.4 is
    0.13999999999999999 in IEEE754, so storing the factor and multiplying at
    read time would have silently changed `ceil(target_delta / 0.14)` at exact
    boundaries. They are stored as values for that reason; `derived_as` records
    the derivation the plan asked for without letting float arithmetic into the
    hot path.
    """

    expected = {
        "anatomy.forearm_twist_generator_max_rad": 1.30,
        "anatomy.wrist_twist_generator_max_rad": 0.12,
        "signal.discontinuity_generator_target_rad": 0.14,
    }
    for key, literal in expected.items():
        actual = value_of(key)
        assert struct.pack("<d", actual) == struct.pack("<d", literal), (
            f"{key} is {actual!r}, not bit-identical to the literal {literal!r} it replaces"
        )


def test_every_derived_value_records_what_it_derives_from() -> None:
    for key, entry in _raw()["thresholds"].items():
        if "derived_as" not in entry:
            continue
        derivation = entry["derived_as"]
        assert derivation["of"] in _raw()["thresholds"], (
            f"{key}.derived_as.of names {derivation['of']!r}, which is not a threshold"
        )
        assert isinstance(derivation.get("factor"), (int, float))
        assert "exact" in derivation, (
            f"{key}.derived_as must say whether the product reproduces the value exactly"
        )


def test_a_derivation_marked_inexact_really_is_inexact() -> None:
    """Guards the reason these are values rather than products.

    If a future edit makes the product exact, `exact` must be updated -- and if
    it makes it *further* from the value, that is a real drift and must fail.
    """

    for key, entry in _raw()["thresholds"].items():
        derivation = entry.get("derived_as")
        if not derivation:
            continue
        product = value_of(derivation["of"]) * derivation["factor"]
        assert product == pytest.approx(entry["value"], rel=1e-9), (
            f"{key} claims to derive from {derivation['of']} x {derivation['factor']} "
            f"but that is {product!r}, not {entry['value']!r}"
        )
        exact = struct.pack("<d", product) == struct.pack("<d", entry["value"])
        assert exact == derivation["exact"], (
            f"{key}.derived_as.exact says {derivation['exact']} but the product is "
            f"{'' if exact else 'not '}bit-identical. Update the flag deliberately."
        )


def test_contradictions_record_what_they_supersede() -> None:
    """Every §1.1 resolution states the losing value and why it lost."""

    resolved = {
        key: entry
        for key, entry in _raw()["thresholds"].items()
        if "resolves" in entry or "resolves" in entry.get("related", {})
    }
    items = {
        text
        for entry in resolved.values()
        for text in [entry.get("resolves") or entry["related"]["resolves"]]
    }
    for item in (2, 3, 4, 6, 7):
        assert any(f"item {item}" in text for text in items), (
            f"§1.1 item {item} has no recorded resolution"
        )

    for key, entry in resolved.items():
        if "supersedes" not in entry:
            continue
        superseded = entry["supersedes"]
        for field in ("key", "value", "why"):
            assert superseded.get(field) not in (None, ""), (
                f"{key}.supersedes must state `{field}`"
            )
        assert superseded["value"] != entry.get("value"), (
            f"{key} claims to supersede a value identical to its own"
        )
