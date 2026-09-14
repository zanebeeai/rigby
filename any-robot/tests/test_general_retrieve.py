"""Loco-manipulation on the mobile bodies: the corrected jaws, the manipulator's reach, the retrieve session's schedule and invariant, a recorded trial that replays.

The second-version dog and biped hang their jaws below the last arm
link (the first version's wrist capsule ran between the fingers); the
reach solves on the declared grasp site in the limb's own joints; the
retrieve session names the support set and the holding limb per phase,
excludes the holding limb from the drive, never writes the root or the
object after placement, and records every step so the trial replays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_core.simulation.recording import replay_physics
from rigby_general.mobility import check_mobile_integrity, load_mobile_body, measure_mobile_body, measure_stance
from rigby_general.mobility.manipulation import manipulators_of
from rigby_general.mobility.retrieve import RetrieveDisturbance, RetrieveSession, grasp_poses, run_retrieve
from rigby_general.mobility.trials import course_world_xml
from rigby_general.mobility.validate import place
from rigby_general.mobility.world import course_v1

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "assets/general/mobile"
sys.path.insert(0, str(ROOT / "scripts"))
import build_mobile_zoo as zoo  # noqa: E402

V2 = ("mobile_dog_arm_v2", "mobile_wheeled_biped_v2", "mobile_octopus_v2")


@pytest.fixture(scope="module")
def bodies():
    return {body_id: load_mobile_body(MOBILE / body_id) for body_id in V2}


def test_the_v2_bodies_rebuild_byte_for_byte_and_the_v1_bodies_are_untouched(tmp_path):
    for builder in zoo.BUILDERS + zoo.BUILDERS_V2:
        b, declaration = builder()
        zoo.write_body(b, declaration, tmp_path)
        committed = (MOBILE / b.robot_id / "robot.xml").read_text(encoding="utf-8")
        assert committed == (tmp_path / b.robot_id / "robot.xml").read_text(encoding="utf-8"), b.robot_id


def test_the_v2_jaw_hangs_below_the_last_link_and_the_v1_jaw_did_not(bodies):
    for body_id, last_link, length in (("mobile_dog_arm_v2", "arm_link_2", 0.08), ("mobile_wheeled_biped_v2", "arm_link_1", 0.20)):
        model = bodies[body_id].model
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_palm")
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[palm])) == last_link
        assert model.body_pos[palm][2] == pytest.approx(-(length + 0.03))
        v1 = load_mobile_body(MOBILE / body_id.removesuffix("_v2")).model
        assert v1.body_pos[mujoco.mj_name2id(v1, mujoco.mjtObj.mjOBJ_BODY, "arm_palm")][2] == pytest.approx(-0.03), "the first version's palm sat inside the last link"


def test_the_v2_bodies_pass_integrity_and_hold_their_stances(bodies):
    for body_id in V2:
        body = bodies[body_id]
        assert check_mobile_integrity(body).passed
        manifest = measure_mobile_body(body)
        assert manifest.manipulators[0].reach_m > 0.5
        if body_id == "mobile_octopus_v2":
            v1 = measure_mobile_body(load_mobile_body(MOBILE / "mobile_octopus"))
            assert manifest.manipulators[0].aperture_m > 2.0 * v1.manipulators[0].aperture_m, "the v2 pincer parts far wider than the first version's"
        for stance, declared in body.declaration["stances"].items():
            out = measure_stance(body, stance)
            measurement = out[0] if isinstance(out, tuple) else out
            assert measurement.statically_stable == declared["statically_stable"], (body_id, stance)


def test_the_reach_solves_where_the_arm_can_go_and_says_where_it_cannot(bodies):
    body = bodies["mobile_dog_arm_v2"]
    model = body.floor_model
    data = mujoco.MjData(model)
    place(model, data, body, body.declaration["stances"]["parked"]["joints"])
    mujoco.mj_forward(model, data)
    arm = manipulators_of(body, model)[0]
    torso = np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")])
    near = arm.solve(data, torso + np.array([0.45, 0.0, -0.05]), down=0.3)
    assert near.reached and near.residual_m < arm.TOLERANCE_M
    assert set(near.joints) == set(arm.joints)
    far = arm.solve(data, torso + np.array([1.5, 0.0, 0.0]))
    assert not far.reached and far.residual_m > 0.5
    assert arm.site_position_of(data, near.joints) == pytest.approx(torso + np.array([0.45, 0.0, -0.05]), abs=0.01)


def test_the_session_names_support_and_holding_per_phase_and_the_tray_rule(bodies):
    body = bodies["mobile_octopus_v2"]
    session = RetrieveSession(body, course_v1(), seed=1, cap_s=30.0)
    assert grasp_poses(body).manipulation_stance is None and grasp_poses(bodies["mobile_dog_arm_v2"]).manipulation_stance == "parked"
    record = session.begin("approach")
    assert set(record.support) == set(body.declaration["support_members"]) and record.holding == ""
    session.holding_limb = "tentacle_0"
    session.locomotor.excluded_limbs = {"tentacle_0"}
    carry = session.begin("carry")
    assert carry.holding == "tentacle_0"
    moving = {mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in session.limb_bodies["tentacle_0"]}
    assert "t0_seg_1" in moving and "t0_root" not in moving, "the tentacle's segments move with it; its root mount is part of the mantle"
    assert not (set(carry.support) & moving), "the holding tentacle's segments are not support members while it holds"
    assert "t0_root" in carry.support and len(carry.support) == len(body.declaration["support_members"]) - len(set(body.declaration["support_members"]) & moving)
    assert session.schedule[-1]["drive_excludes"] == ["tentacle_0"]
    assert not session.cube_in_tray()[0]
    cube = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, "course_cube_free")
    adr = int(session.model.jnt_qposadr[cube])
    session.data.qpos[adr: adr + 3] = (3.42, 2.18, 0.086)
    session.data.qvel[:] = 0.0
    mujoco.mj_forward(session.model, session.data)
    assert session.cube_in_tray()[0]
    disturbance = RetrieveDisturbance(kind="object", detail="2 N sideways", phase="carry", offset_s=6.0, duration_s=0.5, magnitude=2.0, direction=(0.0, 1.0, 0.0))
    assert disturbance.as_json()["phase"] == "carry"


def test_a_short_retrieve_trial_records_every_step_and_replays_exactly(bodies):
    body = bodies["mobile_wheeled_biped_v2"]
    course = course_v1()
    result = run_retrieve(body, course, seed=7, cap_s=8.0)
    assert not result.success and result.reason
    assert result.samples == 4001 or result.samples < 4001
    assert result.phases and result.phases[0].phase == "approach"
    assert result.actuation["root_writes"] == "none after placement" and result.actuation["object_writes"] == "none"
    model = mujoco.MjSpec.from_string(course_world_xml(body, course, None)).compile()
    replay = replay_physics(model, result.run.record)
    assert replay["agrees"] and replay["max_state_error"] == 0.0
