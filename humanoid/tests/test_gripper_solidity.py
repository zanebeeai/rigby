"""What the arm is solid against, and what it is deliberately not.

The arm's links were declared with no contype or conaffinity at all, which in
MuJoCo means both default to zero and the geom collides with NOTHING. Every link
was a ghost: the arm swung straight through the bin and through the bench, and
only the plate and the finger pads were ever stopped by anything. Nobody had
noticed because the hand -- the part being watched -- was solid.

Giving the links a mask then broke the arm in the opposite direction. The
pedestal the arm is bolted to carried the furniture mask, so the links began
colliding with their own mount: 28.6 mm of permanent penetration at base_hub,
22 mm at seg1, and 0.35 rad of joint drift in 1500 steps while the arm was
commanded to hold perfectly still.

Both failures are silent -- an arm passing through a bin and an arm fighting its
own post both look like an arm -- so both are pinned here.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_poc.gripper.physics.model import computed_torque, make

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def body():
    return make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=0.72)


def _masks(body, name):
    index = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert index >= 0, f"no geom called {name!r}"
    return int(body.model.geom_contype[index]), int(
        body.model.geom_conaffinity[index])


def _collide(body, one, two):
    contype_a, affinity_a = _masks(body, one)
    contype_b, affinity_b = _masks(body, two)
    return bool((contype_a & affinity_b) or (contype_b & affinity_a))


LINKS = ("base_hub", "seg1", "seg2", "seg3")


@pytest.mark.parametrize("link", LINKS)
def test_a_link_is_solid_against_the_bin_and_the_bench(body, link):
    """The arm cannot pass through the thing it is meant to reach into."""
    assert _collide(body, link, "bin_wall0")
    assert _collide(body, link, "bin_floor")
    assert _collide(body, link, "table")


@pytest.mark.parametrize("link", LINKS)
def test_a_link_does_not_fight_its_own_mount(body, link):
    """The pedestal is the arm's post, not an obstacle in the room."""
    assert not _collide(body, link, "pedestal")


@pytest.mark.parametrize("link", LINKS)
def test_a_link_does_not_collide_with_the_hand_or_the_block(body, link):
    """Deliberate exclusions, each for its own reason.

    The hand is part of the same body, and neighbouring capsules on a chain
    touch by construction. The block belongs to the fingers: a link brushing
    past should not knock it across the bench mid-reach.
    """
    assert not _collide(body, link, "plate_geom")
    assert not _collide(body, link, "left_geom")
    assert not _collide(body, link, "block_geom")


def test_the_links_do_not_collide_with_each_other(body):
    for near, far in zip(LINKS, LINKS[1:]):
        assert not _collide(body, near, far)


def test_the_hand_is_still_solid_against_everything_it_was(body):
    """The change to the links must not have quietly cost the hand anything."""
    for other in ("bin_wall0", "bin_floor", "table", "block_geom", "pedestal"):
        assert _collide(body, "plate_geom", other), other


def test_the_arm_rests_without_touching_anything(body):
    """At its start pose, nothing is in contact and nothing drifts.

    This is the pedestal failure stated as a measurement rather than as a mask:
    an arm told to hold still, which is being pushed by its own mount, moves.
    """
    fresh = make(np.asarray([0.025, 0.025, 0.03]),
                 np.asarray([0.0, 0.30, 0.76]), table_top=0.72)
    mujoco.mj_forward(fresh.model, fresh.data)
    touching = [
        (mujoco.mj_id2name(fresh.model, mujoco.mjtObj.mjOBJ_GEOM,
                           fresh.data.contact[i].geom1),
         mujoco.mj_id2name(fresh.model, mujoco.mjtObj.mjOBJ_GEOM,
                           fresh.data.contact[i].geom2))
        for i in range(fresh.data.ncon)]
    assert touching == [], touching

    held = np.asarray(fresh.q())
    for _step in range(1500):
        fresh.data.ctrl[:] = computed_torque(fresh, held, 0.0)
        mujoco.mj_step(fresh.model, fresh.data)
    drift = np.abs(np.asarray(fresh.q())[:4] - held[:4])
    assert float(drift.max()) < 0.01, drift
