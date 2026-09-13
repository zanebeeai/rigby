"""Two certified skills composed through the boundary check, on physics.

The neutral checker is pinned in core; this pins what the body supplies and
what the composition does with it, on the jaw arm in the registered G06 fixed
world: the contact mode is measured from opposition, not read from a plan; a
compatible boundary lets the second skill run; a boundary reached still
moving is braked to rest along a velocity-matched quintic and verified
again before the second skill; a joint pushed past its limit is moved back
inside the margin and verified again; a skill that ends holding cannot be
followed by one that begins free, nor a free one by one that begins
holding, and no motion is proposed to bridge either; a stale belief is
observed; a resource still owned is a conflict; and the same composition
run without the check is what the check exists to prevent, caught by the
free-motion joint gate over the record. Every composition is one physics
record whose recorded controls replay across every boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics
from rigby_core.skills import ContactMode, Repair

from rigby_general.contact.placement import PlacementGoal
from rigby_general.contact.transfer import transfer_scene_from_environment
from rigby_general.grounding.grounder import _collision_guard, figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.transitions import arm_joint_names, compose, joint_move_skill, measure_boundary, transfer_skill


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
PROTOCOL = Path(__file__).resolve().parents[1] / "assets" / "general" / "research-protocols" / "g06-transfer-v1"
CORPUS = Path(__file__).resolve().parents[1] / "assets" / "general" / "research-protocols" / "g08-transitions-v1"
BODY = "zoo_jaw_arm"


@pytest.fixture(scope="module")
def jaw():
    env = EnvironmentV1.model_validate_json((PROTOCOL / "environment.json").read_bytes())
    raw = json.loads((PROTOCOL / "goal.json").read_bytes())
    goal = PlacementGoal(region_minimum_m=tuple(raw["region_minimum_m"]), region_maximum_m=tuple(raw["region_maximum_m"]), dwell_s=raw["dwell_s"],
                         maximum_linear_speed_mps=raw["maximum_linear_speed_mps"], maximum_angular_speed_radps=raw["maximum_angular_speed_radps"])
    source = ZOO_ROOT / BODY / "robot.urdf"
    robot = ingest_robot(source, robot_id=BODY)
    effector = robot.morphology.grasping_effectors[0]
    chain = next(c for c in robot.morphology.chains if c.chain_id == effector.chain_id)
    frame = build_workspace_frame(robot.finalized.model, robot.morphology, chain, figure_site=figure_site_for(robot.manifest, effector.chain_id))
    scene = transfer_scene_from_environment(robot.manifest, robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=goal, asset_root=source.parent)
    return {"robot": robot, "manifest": robot.manifest, "effector": effector, "frame": frame, "scene": scene, "model": scene.model,
            "guard": _collision_guard(robot.manifest, scene.model), "arm": arm_joint_names(scene.model, effector, frame)}


def mid_pose(jaw, seed: int = 1) -> dict[str, float]:
    """A feasible terminal pose from the registered corpus for this body."""

    corpus = json.loads((CORPUS / "corpus.json").read_bytes())
    case = [c for c in corpus["feasible_cases"] if c["zoo_id"] == BODY][seed]
    return dict(case["first"]["targets"])


def skills(jaw):
    m, man, eff, fr, sc, g = jaw["model"], jaw["manifest"], jaw["effector"], jaw["frame"], jaw["scene"], jaw["guard"]
    return {
        "motion": lambda **kw: joint_move_skill(m, man, eff, fr, g, skill_id=kw.pop("skill_id", "motion"), **kw),
        "transfer": lambda: transfer_skill(m, man, sc, eff, fr, skill_id="transfer"),
        "acquire": lambda: transfer_skill(m, man, sc, eff, fr, skill_id="acquire_carry", phase_range=("approach", "carry")),
        "place": lambda: transfer_skill(m, man, sc, eff, fr, skill_id="place", phase_range=("lower", "dwell")),
    }


def run(jaw, first, second, *, validate=True, belief_age_s=0.0):
    recorder = PhysicsRecorder(jaw["model"])
    record = compose(jaw["model"], jaw["manifest"], jaw["effector"], jaw["frame"], first, second, recorder, guard=jaw["guard"], validate=validate, belief_age_s=belief_age_s)
    return record, recorder


def test_the_contact_mode_is_measured_from_opposition(jaw) -> None:
    s = skills(jaw)
    acquire = s["acquire"]()
    outcome = acquire.run(None, None, None)
    assert outcome.certified and outcome.continuation is not None
    boundary, raw = measure_boundary(jaw["model"], jaw["manifest"], jaw["effector"], jaw["frame"], outcome.continuation, belief_age_s=0.0)
    assert boundary.contact_mode is ContactMode.HOLDING and boundary.held == {"cube": f"effector:{jaw['effector'].chain_id}"}
    assert raw["opposition"] and len(raw["contacted_members"]) >= 2 and raw["peak_force_n"] > 0.0
    motion = s["motion"](targets=mid_pose(jaw)).run(None, None, None)
    resting, raw = measure_boundary(jaw["model"], jaw["manifest"], jaw["effector"], jaw["frame"], motion.continuation, belief_age_s=0.0)
    assert resting.contact_mode is ContactMode.FREE and resting.held == {} and resting.resting_on == {"cube": "env_fixture_bench"}


def test_a_compatible_boundary_lets_the_second_skill_run_on_one_replayable_record(jaw) -> None:
    s = skills(jaw)
    record, recorder = run(jaw, s["motion"](targets=mid_pose(jaw)), s["transfer"]())
    assert record.first.certified and record.verdicts[0].compatible and not record.repairs
    assert record.second is not None and record.second.certified and record.composed_success
    assert record.gate_violations == []
    times = np.asarray(recorder.rows["time_s"])
    assert np.all(np.diff(times) > 0)
    assert replay_physics(jaw["model"], recorder.finish())["agrees"]


def test_a_boundary_reached_still_moving_is_braked_then_verified_again(jaw) -> None:
    s = skills(jaw)
    record, recorder = run(jaw, s["motion"](targets=mid_pose(jaw), stop_fraction=0.45), s["transfer"]())
    first = record.verdicts[0]
    assert not first.compatible and first.repair is Repair.SETTLE and all(v.code == "velocity_too_high" for v in first.violations)
    [repair] = record.repairs
    assert repair.kind is Repair.SETTLE and repair.executed and repair.verdict_after is not None and repair.verdict_after.compatible
    assert repair.peak_speed_fraction <= 1.0, "braking along a velocity-matched quintic stays under the velocity limits"
    assert record.re_verified and record.second is not None and record.second.certified and record.composed_success
    assert record.cost.physics_s == pytest.approx(repair.physics_s) and record.cost.repairs == (Repair.SETTLE,)


def test_a_joint_past_its_limit_is_moved_back_inside_the_margin_and_verified_again(jaw) -> None:
    s = skills(jaw)
    dofs = {d.joint: d for d in jaw["manifest"].dofs}
    targets = mid_pose(jaw)
    base = dofs[jaw["arm"][0]]
    targets[base.name] = base.maximum + 0.03 * (base.maximum - base.minimum)
    record, recorder = run(jaw, s["motion"](targets=targets), s["transfer"]())
    assert not record.first.certified and record.first.gate == "joint_position_limit", "the first skill is not certified by its own gate"
    first = record.verdicts[0]
    assert not first.compatible and first.repair is Repair.JOINT_MOVE and first.violations[0].code == "joint_beyond_limit" and first.violations[0].subject == base.name
    repair = record.repairs[0]
    assert repair.kind is Repair.JOINT_MOVE and repair.executed and repair.verdict_after.compatible
    inside = repair.boundary_after.joint(base.name)
    assert inside.position <= base.maximum - 0.02 * (base.maximum - base.minimum)
    assert record.re_verified and record.second is not None
    for extra in record.repairs[1:]:
        # If the transfer could not plan from the repaired pose, the inserted
        # transition is a guarded move to its reference configuration,
        # verified again before the one further attempt.
        assert extra.kind is Repair.JOINT_MOVE and extra.detail.startswith("reroute") and extra.verdict_after is not None
    assert not record.composed_success, "a composition whose first skill broke a limit is not a success, whatever the second did"


def test_without_the_check_the_second_skill_runs_from_a_broken_boundary_and_the_gate_catches_it(jaw) -> None:
    s = skills(jaw)
    dofs = {d.joint: d for d in jaw["manifest"].dofs}
    targets = mid_pose(jaw)
    base = dofs[jaw["arm"][0]]
    targets[base.name] = base.maximum + 0.03 * (base.maximum - base.minimum)
    record, recorder = run(jaw, s["motion"](targets=targets), s["transfer"](), validate=False)
    assert not record.validated and record.repairs == [] and record.second is not None
    assert any(g["code"] == "joint_position_limit" and g["subject"] == base.name and g["scope"] == "second" for g in record.gate_violations)
    assert not record.composed_success


def test_a_skill_that_ends_holding_cannot_be_followed_by_one_that_begins_free(jaw) -> None:
    s = skills(jaw)
    record, recorder = run(jaw, s["acquire"](), s["transfer"]())
    assert record.boundary.contact_mode is ContactMode.HOLDING
    assert record.rejected and record.rejection == "contact_mode_mismatch" and record.second is None
    assert record.repairs == [], "no path is proposed to bridge a contact-mode mismatch"
    codes = {v.code for v in record.verdicts[0].violations}
    assert {"contact_mode_mismatch", "ownership_mismatch", "object_not_resting"} <= codes


def test_a_free_skill_cannot_be_followed_by_one_that_begins_holding(jaw) -> None:
    s = skills(jaw)
    record, recorder = run(jaw, s["motion"](targets=mid_pose(jaw)), s["place"]())
    assert record.rejected and record.rejection == "contact_mode_mismatch" and record.second is None and record.repairs == []


def test_an_acquisition_then_a_placement_compose_through_a_holding_boundary(jaw) -> None:
    s = skills(jaw)
    record, recorder = run(jaw, s["acquire"](), s["place"]())
    assert record.boundary.contact_mode is ContactMode.HOLDING
    assert record.second is not None and record.second.certified and record.second.measurements["placement_success"]
    assert record.composed_success
    for repair in record.repairs:
        assert repair.kind is Repair.SETTLE and repair.boundary_after.contact_mode is ContactMode.HOLDING, "the object stays held through the transition"
    assert replay_physics(jaw["model"], recorder.finish())["agrees"]


def test_a_stale_belief_is_observed_and_a_kept_resource_is_a_conflict(jaw) -> None:
    s = skills(jaw)
    record, _ = run(jaw, s["motion"](targets=mid_pose(jaw)), s["transfer"](), belief_age_s=30.0)
    assert [v.code for v in record.verdicts[0].violations] == ["belief_stale"]
    assert record.repairs[0].kind is Repair.OBSERVE and record.re_verified and record.composed_success and record.belief_age_s == 0.0
    record, _ = run(jaw, s["motion"](targets=mid_pose(jaw), keeps_resources=(f"effector:{jaw['effector'].chain_id}",)), s["transfer"]())
    assert record.rejected and record.rejection == "resource_conflict" and record.second is None
