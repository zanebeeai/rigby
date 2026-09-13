"""The skill store bound to a body: contexts from the session as it stands, and a restart from observation on physics.

A context's facets come from what the session runs with, not from what a
caller says: swapping the controller configuration changes the controller
facet and nothing else; a layout that moves the fixtures changes no facet
and only the values. The goal drawn around a moved platform reproduces the
registered G06 goal on the registered world. And on the jaw arm, the
transfer tree interrupted as the grasp completes is restarted in a fresh
session that learns from the contact and the cameras that the cube is
held, skips the acquisition, and finishes the transfer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rigby_core.skills import ContextDimension, Verdict, compare
from rigby_core.skills.examples import transfer_object_library

from rigby_general.gates.control import ControllerConfig
from rigby_general.sensing import load_policy
from rigby_general.skills import TransferObjectSession
from rigby_general.skills.skill_store import CLAIMED_RANGES, context_of, draw_validation_set, goal_for, layouts, perturbed, restart, run_tree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g10_corpus as g10  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return g10.environment()


@pytest.fixture(scope="module")
def policy():
    return load_policy(g10.G09 / "policy.json")


def open_jaw(world, policy, **kwargs):
    return TransferObjectSession.open("zoo_jaw_arm", ROOT / "assets/general/zoo/zoo_jaw_arm/robot.urdf", world, goal_for(world), policy, configuration_name="front_overhead_contact", **kwargs)


def test_goal_for_reproduces_the_registered_goal(world):
    registered = g10.registered_goal()
    drawn = goal_for(world)
    assert drawn.region_minimum_m == pytest.approx(registered.region_minimum_m)
    assert drawn.region_maximum_m == pytest.approx(registered.region_maximum_m)
    assert (drawn.dwell_s, drawn.maximum_linear_speed_mps, drawn.maximum_angular_speed_radps) == (registered.dwell_s, registered.maximum_linear_speed_mps, registered.maximum_angular_speed_radps)


def test_layouts_keep_every_value_inside_the_claimed_ranges(world, policy):
    ranges = {r.quantity: r for r in CLAIMED_RANGES}
    for name, layout in layouts(world).items():
        session = open_jaw(layout, policy, seed_label=name)
        context = context_of(session)
        for quantity, value in context.values.items():
            assert ranges[quantity].low <= value <= ranges[quantity].high, (name, quantity, value)
        base = context_of(open_jaw(world, policy, seed_label="base"))
        verdict = compare(base, context)
        assert verdict.valid, (name, verdict.differences)


def test_facets_come_from_what_the_session_runs_with(world, policy):
    base = context_of(open_jaw(world, policy, seed_label="a"))
    softer = context_of(open_jaw(world, policy, seed_label="b", controller_config=ControllerConfig(natural_frequency_hz=7.0)))
    verdict = compare(base, softer)
    assert not verdict.valid and [d.dimension for d in verdict.differences] == [ContextDimension.CONTROLLER]
    one_camera = context_of(TransferObjectSession.open("zoo_jaw_arm", ROOT / "assets/general/zoo/zoo_jaw_arm/robot.urdf", world, goal_for(world), policy, configuration_name="front_contact", seed_label="c"))
    assert [d.dimension for d in compare(base, one_camera).differences] == [ContextDimension.SENSORS]
    bigger = world.model_copy(update={"objects": (world.objects[0].model_copy(update={"size_m": (0.0175, 0.0175, 0.0175)}),)})
    assert [d.dimension for d in compare(base, context_of(open_jaw(bigger, policy, seed_label="d"))).differences] == [ContextDimension.GEOMETRY]
    assert [d.dimension for d in compare(base, context_of(open_jaw(world, policy, seed_label="e"), friction_assumption={"object_friction_range": [0.5, 0.9]})).differences] == [ContextDimension.FRICTION]
    assert [d.dimension for d in compare(base, context_of(open_jaw(world, policy, seed_label="f"), evidence_schema={"episode_protocol": "x/2"})).differences] == [ContextDimension.EVIDENCE_SCHEMA]
    slick = world.model_copy(update={"objects": (world.objects[0].model_copy(update={"friction": 0.6}),)})
    out_of_range = compare(base, context_of(open_jaw(slick, policy, seed_label="g")))
    assert not out_of_range.valid and out_of_range.differences[0].quantity == "object_friction"


def test_validation_set_is_reproducible_and_disjoint_from_development_seeds():
    first, draws = draw_validation_set("zoo_jaw_arm", set_id="s", seeds=(1000, 1001), rng_seed=1, threshold=2)
    again, draws_again = draw_validation_set("zoo_jaw_arm", set_id="s", seeds=(1000, 1001), rng_seed=1, threshold=2)
    assert first.content_hash() == again.content_hash() and draws == draws_again
    assert all(d["seed"] >= 1000 for d in draws)
    assert first.independent_of and first.threshold == 2


def test_restart_after_the_grasp_reconstructs_the_hold_and_finishes(world, policy):
    library = transfer_object_library()
    _, draws = draw_validation_set("zoo_jaw_arm", set_id="t", seeds=(1000,), rng_seed=20261111, threshold=1)
    scene = perturbed(world, draws[0])
    session = open_jaw(scene, policy, seed_label="first")
    arguments = {"object": "cube", "destination": "platform", "effector": session.effector.chain_id}
    first = run_tree(session, library, "transfer_object", arguments, checkpoint_after=2)
    assert first.record.verdict is Verdict.INTERRUPTED and first.checkpoint is not None
    assert [c["leaf"] for c in first.runtime.calls] == ["observe_object", "acquire"]
    assert json.loads(first.checkpoint.model_dump_json())["world_state"]["state"] is not None
    second = restart(first.checkpoint, library, "transfer_object", arguments, "zoo_jaw_arm", ROOT / "assets/general/zoo/zoo_jaw_arm/robot.urdf", scene, goal_for(scene), policy,
                     configuration_name="front_overhead_contact", seed_label="second")
    trail = second.reconstruction["trail"]
    assert trail["closure_engaged"] and trail["held"]["decision"] == "pass"
    assert second.reconstruction["facts"]["held:cube"] is True
    assert "acquire" not in [c.get("leaf") for c in second.runtime.calls]
    assert second.record.verdict is Verdict.SUCCESS
    assert second.session.time_s > first.session.time_s
