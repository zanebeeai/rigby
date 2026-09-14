"""The three mobile bodies: built, ingested, measured, settled and judged on the common course.

The builder writes the bodies fresh; the mobility ingest holds them to
the floating-base rules; the manifest reads its numbers off the model;
the settling tests find the stances their authors called stable stable
and the one they called unstable unstable, on the members declared; a
recovery from a small roll succeeds where it is meant to; the course
freezes to a hash and the feasibility map disagrees with the bodies
exactly where their geometry says it should.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_general.mobility import check_mobile_integrity, load_mobile_body, measure_mobile_body, measure_stance, recovery_trial
from rigby_general.mobility.contracts import MobileBodyManifestV1
from rigby_general.mobility.world import course_v1, feasibility

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_mobile_zoo as zoo  # noqa: E402

BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("mobile")
    for builder in zoo.BUILDERS:
        zoo.write_body(*builder(), root)
    return {body_id: load_mobile_body(root / body_id) for body_id in BODIES}


def test_the_built_bodies_match_the_committed_assets(built):
    for body_id, body in built.items():
        committed = (ROOT / "assets/general/mobile" / body_id / "robot.xml").read_text(encoding="utf-8")
        assert committed == body.xml, f"{body_id}: the committed model is not what the builder writes"


def test_every_body_passes_the_floating_base_integrity_rules(built):
    for body_id, body in built.items():
        report = check_mobile_integrity(body)
        assert report.passed, (body_id, report.violations)
        assert report.free_joints == 1 and report.actuator_count == report.joint_count
        assert report.inertia_checked_bodies == report.body_count


def test_the_manifest_is_measured_not_copied(built):
    for body_id, body in built.items():
        manifest = measure_mobile_body(body)
        assert manifest.model_sha256 == body.model_sha256 and manifest.mass_kg == pytest.approx(float(body.model.body_mass.sum()))
        assert len(manifest.joints) == body.model.njnt - 1
        for joint in manifest.joints:
            j = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
            if joint.minimum is not None:
                assert (joint.minimum, joint.maximum) == pytest.approx(tuple(float(v) for v in body.model.jnt_range[j]))
        assert {limb.role for limb in manifest.limbs} == {"mobile_dog_arm": {"leg", "arm"}, "mobile_wheeled_biped": {"wheel_leg", "arm"}, "mobile_octopus": {"tentacle"}}[body_id]
        for manipulator in manifest.manipulators:
            assert 0.02 < manipulator.aperture_m < 0.12 and 0.2 < manipulator.reach_m < 0.8
        assert all(member.friction > 0 for member in manifest.support_members)
        MobileBodyManifestV1.model_validate_json(manifest.model_dump_json())


def test_stances_settle_as_declared(built):
    for body_id, body in built.items():
        for stance, declared in body.declaration["stances"].items():
            measurement, run = measure_stance(body, stance, duration_s=3.0)
            assert measurement.statically_stable == declared["statically_stable"], (body_id, stance, measurement)
            if declared["statically_stable"]:
                assert measurement.undeclared_contacts == () and measurement.tilt_deg < 15.0 and measurement.settled
            assert len(run.record.arrays["time_s"]) == int(round(3.0 / body.floor_model.opt.timestep)) + 1


def test_a_small_roll_is_recovered_where_the_body_can_hold_itself(built):
    for body_id, body in built.items():
        working = body.declaration["working_stance"]
        stance = working if body.declaration["stances"][working]["statically_stable"] else "parked"
        height = measure_stance(body, stance, duration_s=2.0)[0].base_height_m
        trial, _ = recovery_trial(body, stance, "roll_8", perturbation="rolled 8 degrees", roll_deg=8.0, pitch_deg=0.0, drop_m=0.03, supported=True, duration_s=5.0, expected_height_m=height)
        assert trial.recovered, (body_id, trial)
    biped = built["mobile_wheeled_biped"]
    trial, _ = recovery_trial(biped, "standing", "released_standing", perturbation="released standing", roll_deg=0.0, pitch_deg=0.0, drop_m=0.02, supported=False, duration_s=3.0)
    assert not trial.recovered and trial.final_tilt_deg > 30.0, "without a balance controller the biped does not stand"


def test_the_course_freezes_and_the_feasibility_map_follows_the_geometry(built):
    course = course_v1()
    assert course.sha256() == course_v1().sha256()
    assert {b["branch_id"] for b in course.branches} == {"corridor", "doorway", "ramp", "step", "station", "tray"}
    model = mujoco.MjSpec.from_string(built["mobile_dog_arm"].floor_xml.replace('<light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>', course.mjcf_features(), 1)).compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "course_cube") >= 0
    geometry = {"mobile_dog_arm": {"arm_base_height_above_base_m": 0.10, "wheel_radius_m": None, "leg_length_m": 0.41, "mantle_rise_m": None},
                "mobile_wheeled_biped": {"arm_base_height_above_base_m": 0.18, "wheel_radius_m": 0.08, "leg_length_m": None, "mantle_rise_m": None},
                "mobile_octopus": {"arm_base_height_above_base_m": -0.06, "wheel_radius_m": None, "leg_length_m": None, "mantle_rise_m": 0.017}}
    verdicts = {}
    for body_id, body in built.items():
        manifest = measure_mobile_body(body)
        stances = tuple(measure_stance(body, s, duration_s=2.0)[0] for s in body.declaration["stances"])
        verdicts[body_id] = {v["branch_id"]: v["feasible"] for v in feasibility(course, manifest, stances, **geometry[body_id])["verdicts"]}
    assert all(verdicts[b]["corridor"] for b in BODIES) and all(verdicts[b]["ramp"] for b in BODIES)
    assert not verdicts["mobile_octopus"]["doorway"], "1.2 m across, the octopus does not fit a 0.7 m doorway"
    assert not verdicts["mobile_wheeled_biped"]["step"], "an 8 cm wheel does not climb a 5 cm step"
    assert not verdicts["mobile_octopus"]["step"], "the mantle rides 1.7 cm off the floor"
    assert verdicts["mobile_dog_arm"]["step"] and verdicts["mobile_dog_arm"]["doorway"]
    assert all(verdicts[b]["station"] and verdicts[b]["tray"] for b in BODIES), "every body reaches the station and the tray in a stance it can hold"
