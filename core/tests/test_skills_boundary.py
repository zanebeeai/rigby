"""A boundary is checked on pose, velocity, contact mode, ownership, belief
freshness and resources, and only some of what fails admits a path repair."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rigby_core.skills.boundary import (
    BoundaryStateV1,
    ContactMode,
    InitiationSetV1,
    JointStateV1,
    Repair,
    check_boundary,
)


def joint(name="j0", position=0.0, velocity=0.0, minimum=-1.0, maximum=1.0, velocity_limit=2.0) -> JointStateV1:
    return JointStateV1(name=name, position=position, velocity=velocity, minimum=minimum, maximum=maximum, velocity_limit=velocity_limit)


def boundary(*joints, mode=ContactMode.FREE, held=None, belief_age_s=0.0, owned=()) -> BoundaryStateV1:
    return BoundaryStateV1(time_s=1.0, joints=tuple(joints) or (joint(),), contact_mode=mode, held=held or {}, belief_age_s=belief_age_s, owned=owned, manipulator="effector:a")


def initiation(mode=ContactMode.FREE, required_held=None, **overrides) -> InitiationSetV1:
    fields = dict(skill_id="next", contact_mode=mode, required_held=required_held or {}, max_belief_age_s=5.0, required_resources=("effector:a",))
    fields.update(overrides)
    return InitiationSetV1(**fields)


def test_a_compatible_boundary_needs_no_repair() -> None:
    verdict = check_boundary(boundary(joint(position=0.3, velocity=0.01)), initiation())
    assert verdict.compatible and verdict.repair is Repair.NONE and not verdict.violations


def test_a_joint_past_its_limit_is_repaired_by_a_move_to_the_nearest_admissible_position() -> None:
    verdict = check_boundary(boundary(joint(position=1.05)), initiation(limit_margin_fraction=0.02))
    assert not verdict.compatible and verdict.repair is Repair.JOINT_MOVE
    [violation] = verdict.violations
    assert violation.code == "joint_beyond_limit" and violation.measured == 1.05 and violation.limit == 1.0
    assert verdict.joint_targets == {"j0": pytest.approx(1.0 - 0.04)}


def test_a_joint_inside_the_margin_is_also_a_move() -> None:
    verdict = check_boundary(boundary(joint(position=-0.99)), initiation(limit_margin_fraction=0.02))
    [violation] = verdict.violations
    assert violation.code == "joint_inside_margin" and verdict.joint_targets == {"j0": pytest.approx(-0.96)}
    assert check_boundary(boundary(joint(position=-0.95)), initiation(limit_margin_fraction=0.02)).compatible


def test_residual_velocity_is_settled() -> None:
    verdict = check_boundary(boundary(joint(velocity=0.5, velocity_limit=2.0)), initiation(speed_fraction=0.05))
    [violation] = verdict.violations
    assert violation.code == "velocity_too_high" and violation.limit == pytest.approx(0.1) and verdict.repair is Repair.SETTLE


def test_a_stale_belief_is_observed() -> None:
    verdict = check_boundary(boundary(belief_age_s=9.0), initiation(max_belief_age_s=5.0))
    [violation] = verdict.violations
    assert violation.code == "belief_stale" and verdict.repair is Repair.OBSERVE


def test_a_contact_mode_mismatch_admits_no_path() -> None:
    verdict = check_boundary(boundary(mode=ContactMode.HOLDING, held={"cube": "effector:a"}), initiation(mode=ContactMode.FREE))
    codes = {v.code for v in verdict.violations}
    assert codes == {"contact_mode_mismatch", "ownership_mismatch"}
    assert verdict.repair is Repair.CONTACT_CHANGE and verdict.rejected
    assert not any(v.repairable_by_path for v in verdict.violations)


def test_a_skill_that_expects_to_hold_cannot_be_given_the_object_by_a_path() -> None:
    verdict = check_boundary(boundary(joint(position=0.99)), initiation(mode=ContactMode.HOLDING, required_held={"cube": "effector:a"}))
    codes = [v.code for v in verdict.violations]
    assert "joint_inside_margin" in codes and "contact_mode_mismatch" in codes and "ownership_mismatch" in codes
    assert verdict.repair is Repair.CONTACT_CHANGE, "the unrepairable violation decides, even beside a repairable one"
    assert verdict.joint_targets == {}


def test_holding_with_the_right_resource_satisfies_a_holding_initiation() -> None:
    verdict = check_boundary(boundary(mode=ContactMode.HOLDING, held={"cube": "effector:a"}), initiation(mode=ContactMode.HOLDING, required_held={"cube": "effector:a"}))
    assert verdict.compatible
    wrong = check_boundary(boundary(mode=ContactMode.HOLDING, held={"cube": "effector:b"}), initiation(mode=ContactMode.HOLDING, required_held={"cube": "effector:a"}))
    assert [v.code for v in wrong.violations] == ["ownership_mismatch"] and wrong.rejected
    assert "held by effector:b" in wrong.violations[0].detail


def test_a_resource_still_owned_must_be_released_first() -> None:
    verdict = check_boundary(boundary(owned=("effector:a",)), initiation(required_resources=("effector:a",)))
    [violation] = verdict.violations
    assert violation.code == "resource_conflict" and verdict.repair is Repair.RELEASE_RESOURCE and verdict.rejected


def test_the_path_repairs_are_ordered_move_then_settle_then_observe() -> None:
    both = check_boundary(boundary(joint(position=1.2, velocity=0.9), belief_age_s=9.0), initiation())
    assert {v.code for v in both.violations} == {"joint_beyond_limit", "velocity_too_high", "belief_stale"}
    assert both.repair is Repair.JOINT_MOVE
    assert check_boundary(boundary(joint(velocity=0.9), belief_age_s=9.0), initiation()).repair is Repair.SETTLE


def test_an_object_that_must_be_resting_cannot_be_put_down_by_a_path() -> None:
    verdict = check_boundary(boundary(), initiation(required_resting=("cube",)))
    [violation] = verdict.violations
    assert violation.code == "object_not_resting" and verdict.rejected and verdict.repair is Repair.CONTACT_CHANGE
    resting = BoundaryStateV1(time_s=1.0, joints=(joint(),), contact_mode=ContactMode.FREE, resting_on={"cube": "bench"}, belief_age_s=0.0, manipulator="effector:a")
    assert check_boundary(resting, initiation(required_resting=("cube",))).compatible


def test_joint_states_must_be_finite_with_increasing_ranges() -> None:
    with pytest.raises(ValidationError):
        JointStateV1(name="j", position=float("nan"), velocity=0.0, minimum=-1.0, maximum=1.0, velocity_limit=1.0)
    with pytest.raises(ValidationError):
        JointStateV1(name="j", position=0.0, velocity=0.0, minimum=1.0, maximum=-1.0, velocity_limit=1.0)


def test_the_verdict_round_trips() -> None:
    from rigby_core.skills.boundary import BoundaryVerdictV1

    verdict = check_boundary(boundary(joint(position=1.05, velocity=0.3)), initiation())
    again = BoundaryVerdictV1.model_validate_json(verdict.model_dump_json())
    assert again == verdict and again.content_hash() == verdict.content_hash()
