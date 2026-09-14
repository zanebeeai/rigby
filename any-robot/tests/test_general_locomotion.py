"""Locomotion for the three mobile bodies: the analytic leg IK, the body-neutral navigator, each drive making way on the flat floor
through its own actuators, and a recorded course trial that replays exactly with its declared external input.

The controllers are hand-authored per body and say so; the navigator
knows a base pose and a drive interface only; a trial writes the root
once at placement and never again, applies a push only as recorded user
input, and the recorded controls replay to the recorded states.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_core.simulation.recording import replay_physics
from rigby_general.mobility import load_mobile_body
from rigby_general.mobility.locomotion import BaseState, DogTrot, Navigator, OctopusCrawl, WheeledBalance, leg_fk, leg_ik, make_locomotor, wrap
from rigby_general.mobility.trials import Perturbation, course_world_xml, run_travel
from rigby_general.mobility.validate import place
from rigby_general.mobility.world import course_v1

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "assets/general/mobile"
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")


@pytest.fixture(scope="module")
def bodies():
    return {body_id: load_mobile_body(MOBILE / body_id) for body_id in BODIES}


def drive(body, seconds: float, command):
    """Run a drive on the flat floor from its working stance; return the base states sampled every 0.5 s."""

    model = body.floor_model
    data = mujoco.MjData(model)
    locomotor = make_locomotor(body, model)
    place(model, data, body, locomotor.stance)
    locomotor.reset(data)
    states = []
    for step in range(int(seconds / model.opt.timestep)):
        v, omega = command(float(data.time))
        control = locomotor.control(data, v, omega)
        assert control.shape == (model.nu,) and np.all(np.isfinite(control))
        assert np.all(control >= model.actuator_ctrlrange[:, 0] - 1e-9) and np.all(control <= model.actuator_ctrlrange[:, 1] + 1e-9)
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        if step % 250 == 0:
            states.append(locomotor.base_state(data))
    return locomotor, states


def test_leg_ik_inverts_leg_fk_in_the_dog_convention():
    for hip, knee in ((0.55, -1.15), (0.2, -0.6), (0.9, -1.8), (-0.3, -0.9)):
        x, z = leg_fk(hip, knee)
        assert leg_ik(x, z) == pytest.approx((hip, knee), abs=1e-6)
    assert leg_fk(0.0, 0.0) == pytest.approx((0.0, -0.41))


def test_every_drive_is_hand_authored_and_says_so(bodies):
    kinds = {}
    for body_id, body in bodies.items():
        locomotor = make_locomotor(body, body.floor_model)
        kinds[body_id] = type(locomotor)
        assert "analytic, hand-authored" in locomotor.provenance
        assert 0.0 < locomotor.min_speed_mps < locomotor.max_speed_mps
    assert kinds == {"mobile_dog_arm": DogTrot, "mobile_wheeled_biped": WheeledBalance, "mobile_octopus": OctopusCrawl}


@pytest.mark.parametrize("body_id,seconds,least_m", [("mobile_dog_arm", 8.0, 0.8), ("mobile_wheeled_biped", 8.0, 1.2), ("mobile_octopus", 14.0, 0.6)])
def test_each_drive_makes_way_forward_and_stays_upright(bodies, body_id, seconds, least_m):
    body = bodies[body_id]
    locomotor, states = drive(body, seconds, lambda t: (0.0, 0.0) if t < 1.5 else (locomotor_max(body), 0.0))
    final = states[-1]
    assert final.position[0] > least_m, (body_id, final.position)
    assert abs(final.position[1]) < 0.4 and abs(math.degrees(final.pitch)) < 25.0 and abs(math.degrees(final.roll)) < 25.0
    assert abs(math.degrees(final.yaw)) < 20.0


def locomotor_max(body) -> float:
    return make_locomotor(body, body.floor_model).max_speed_mps


@pytest.mark.parametrize("body_id,seconds,least_deg", [("mobile_dog_arm", 8.0, 60.0), ("mobile_wheeled_biped", 8.0, 60.0), ("mobile_octopus", 10.0, 60.0)])
def test_each_drive_turns_counter_clockwise_for_a_positive_turn_rate(bodies, body_id, seconds, least_deg):
    body = bodies[body_id]
    locomotor, states = drive(body, seconds, lambda t: (0.0, 0.0) if t < 1.5 else (0.0, 0.5))
    turned = sum(wrap(b.yaw - a.yaw) for a, b in zip(states, states[1:]))
    assert math.degrees(turned) > least_deg, (body_id, math.degrees(turned))
    assert np.linalg.norm(states[-1].position[:2]) < 1.0, "a turn in place may creep (the trot slips) but does not travel"


def test_the_navigator_steers_stops_holds_and_releases_without_knowing_the_body():
    class Drive:
        max_speed_mps, min_speed_mps, max_turn_radps = 0.3, 0.05, 0.6

    nav = Navigator(waypoints=[(1.2, 0.0), (0.0, 0.0)])
    at = lambda x, y, yaw: BaseState(position=np.array([x, y, 0.3]), yaw=yaw, pitch=0.0, roll=0.0, velocity=np.zeros(3), angular=np.zeros(3))
    v, omega = nav.command(at(0.0, 0.0, 0.0), 0.0, Drive())
    assert v == pytest.approx(0.3) and omega == 0.0
    v, omega = nav.command(at(0.0, 0.0, math.radians(90)), 0.1, Drive())
    assert v == 0.0 and omega < 0.0, "facing away: turn first"
    v, omega = nav.command(at(0.95, 0.0, 0.0), 0.2, Drive())
    assert v == 0.0 and nav.arrived_at == 0.2 and nav.log[-1]["event"] == "arrived"
    assert nav.command(at(0.8, 0.0, 0.0), 1.0, Drive()) == (0.0, 0.0), "inside the release radius it still counts as arrived"
    assert nav.command(at(0.8, 0.0, 0.0), 2.3, Drive()) == (0.0, 0.0) and nav.index == 1 and nav.log[-1]["event"] == "released"
    v, omega = nav.command(at(0.8, 0.0, 0.0), 2.4, Drive())
    assert v == 0.0 and abs(omega) == pytest.approx(0.6), "the next waypoint is behind: turn in place"
    sign = math.copysign(1.0, omega)
    v, omega = nav.command(at(0.8, 0.0, -sign * 0.02), 2.5, Drive())
    assert math.copysign(1.0, omega) == sign, "the turn direction is committed across the seam"
    v, omega = nav.command(at(0.8, 0.0, math.pi), 3.0, Drive())
    assert v > 0.0 and abs(omega) < 0.1
    v, omega = nav.command(at(0.31, 0.0, math.pi), 3.5, Drive())
    assert v == pytest.approx(0.05), "at the edge of the radius it asks for the drive's minimum, not less"
    nav.command(at(0.1, 0.0, math.pi), 4.0, Drive())
    assert not nav.done
    nav.command(at(0.1, 0.0, math.pi), 6.1, Drive())
    assert nav.done


def test_the_course_world_carries_the_patch_and_the_trial_replays_exactly(bodies):
    body = bodies["mobile_wheeled_biped"]
    course = course_v1()
    patch = Perturbation(kind="patch", detail="slick", patch_friction=0.2, patch_centre_m=(0.6, 0.0), patch_half_m=(0.3, 0.6))
    assert 'name="patch_slick"' in course_world_xml(body, course, patch) and 'name="patch_slick"' not in course_world_xml(body, course, None)
    push = Perturbation(kind="push", detail="20 N sideways", at_s=3.0, duration_s=0.3, magnitude=20.0, direction=(0.0, 1.0, 0.0))
    result = run_travel(body, course, seed=3, waypoints=[(0.8, 0.0)], cap_s=12.0, perturbation=push)
    assert result.samples == 6001 or result.samples < 6001
    assert result.perturbation_log[0]["event"] == "on" and result.perturbation_log[0]["time_s"] == pytest.approx(3.0, abs=0.003)
    assert result.actuation["provenance"].startswith("analytic, hand-authored") and result.actuation["root_writes"] == "none after placement"
    assert set(result.support["declared_contact_fraction"]) == {"left_wheel", "right_wheel", "torso"}
    model = mujoco.MjSpec.from_string(course_world_xml(body, course, push)).compile()
    record = result.run.record
    user = record.arrays["user_input"]
    assert np.abs(user).max() > 0.0, "the push is in the recorded user input"
    replay = replay_physics(model, record)
    assert replay["agrees"] and replay["max_state_error"] == 0.0
    if result.success:
        assert result.reached and result.stable_at_end and not result.fell
