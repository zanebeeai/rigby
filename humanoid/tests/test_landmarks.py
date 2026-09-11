"""Perceiving named things, and refusing to perceive them wrongly.

The model is stood in for throughout. Its job here is to point at pixels, and
what these tests are about is what happens to those pixels afterwards -- so the
picks come from the segmentation buffer, which is where a competent model would
point, with jitter added to stand for how imprecisely it points.

Segmentation truth is used ONLY to make the stand-in picks and to grade the
answers. Nothing in the code under test can see it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_poc.gripper.decision.pick_and_place import bin_of
from rigby_poc.gripper.physics.model import make
from rigby_poc.gripper.sensing import landmarks as L

pytestmark = pytest.mark.medium


@pytest.fixture(scope="module")
def scene():
    body = make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=0.72)
    model = body.model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
             for i in range(model.ngeom)]
    groups = {
        "block": [i for i, n in enumerate(names) if n == "block_geom"],
        "bin": [i for i, n in enumerate(names)
                if n.startswith("bin_") and n != "bin_riser"],
    }
    viewer = mujoco.Renderer(model, height=L.PICK_H, width=L.PICK_W)
    viewer.enable_segmentation_rendering()
    truth = {"block": np.asarray(body.block()),
             "bin": np.asarray(bin_of()["centre"], dtype=float)}
    return body, viewer, groups, truth


def _picks(scene, item, jitter=0.0, seed=3):
    body, viewer, groups, _truth = scene
    rng = np.random.default_rng(seed)
    out = {}
    for camera in L.CORNERS:
        viewer.update_scene(body.data, camera=camera)
        lit = np.isin(viewer.render()[:, :, 0], groups[item])
        if int(lit.sum()) < 40:
            continue
        rows, cols = np.nonzero(lit)
        out[camera] = [float(cols.mean()) + rng.normal(0, jitter),
                       float(rows.mean()) + rng.normal(0, jitter)]
    return out


def test_the_bin_is_perceived_at_all(scene):
    """The failure that started this: the bin was invisible to every instrument.

    It was painted the same blue-grey as the bench it stood on -- bin pixels
    ran 57-88 mean luminance against the bench's 72-87 -- so nothing could
    segment it, and the machine knew where it was only because a manifest said
    so. Two cameras is the minimum for a position; the bin should be visible to
    more than that.
    """
    body, _viewer, _groups, truth = scene
    world = L.World()
    got = world.anchor(body, "bin", _picks(scene, "bin"), now=0.0)
    assert got["seen"] is True
    assert len(got["seen_from"]) >= 3
    assert np.linalg.norm(world.at("bin") - truth["bin"]) < 0.05


def test_both_things_are_placed_from_the_same_pictures(scene):
    """One mechanism, two items, nothing in it named either of them."""
    body, _viewer, _groups, truth = scene
    world = L.World()
    for item in ("block", "bin"):
        world.anchor(body, item, _picks(scene, item), now=0.0)
    assert set(world.belief) == {"block", "bin"}
    for item in ("block", "bin"):
        assert np.linalg.norm(world.at(item) - truth[item]) < 0.05


def test_looking_again_corrects_a_sloppy_pick(scene):
    """The imagination is compared with the world and moves toward it.

    A pick fifteen pixels off puts the anchor around 20 mm out. Tracking with
    the learned look is far more precise than the pointing was, so the belief
    should end up closer to the truth than the picks that started it.
    """
    body, _viewer, _groups, truth = scene
    world = L.World()
    world.anchor(body, "block", _picks(scene, "block", jitter=15.0), now=0.0)
    started = float(np.linalg.norm(world.at("block") - truth["block"]))
    for tick in range(3):
        report = world.refresh(body, now=0.1 * (tick + 1))
    ended = float(np.linalg.norm(world.at("block") - truth["block"]))
    assert report["block"]["seen_now"] is True
    assert report["block"]["measurement_disagreed_by_m"] is not None
    assert ended < started
    assert ended < 0.01


def test_a_look_that_finds_the_bench_is_refused(scene):
    """The expensive failure, pinned.

    A pick that lands beside the object samples the bench behind it, and the
    tracker then follows the bench at 30 Hz while the belief drifts a third of
    a metre with nothing reporting a problem. The patch's own coherence does
    not catch this -- measured, it read 0.09 on a good look and 0.95 on a bad
    one. Tracking with the candidate and checking where it lands does.
    """
    body, _viewer, _groups, truth = scene
    world = L.World()
    honest = _picks(scene, "block")
    astray = {camera: [uv[0] + 60.0, uv[1] + 60.0]
              for camera, uv in honest.items()}
    got = world.anchor(body, "block", astray, now=0.0)
    if "block" in world.looks:
        # A look was kept, so it must genuinely lead back to the block rather
        # than to the bench -- which is the only thing being claimed here.
        assert got["look_lands_off_by_m"] <= L.LOOK_MUST_AGREE_M
    else:
        assert got["learned"] is None
        assert "why_no_look" in got
        # The position from the picks survives; only the appearance is dropped.
        assert world.at("block") is not None
        assert world.refresh(body, now=0.1) == {}


def test_one_camera_is_a_direction_and_not_a_position(scene):
    body, _viewer, _groups, _truth = scene
    world = L.World()
    only = dict(list(_picks(scene, "block").items())[:1])
    got = world.anchor(body, "block", only, now=0.0)
    assert got["seen"] is False
    assert world.at("block") is None


def test_disagreeing_picks_report_a_large_miss(scene):
    """Confidence has to be earned by agreement, not asserted.

    Picks that are not of the same thing still cross somewhere -- that is what
    least squares does -- so the position alone carries no evidence. The miss
    is what separates a fix from a guess.
    """
    body, _viewer, _groups, _truth = scene
    world = L.World()
    tight = world.anchor(body, "block", _picks(scene, "block"), now=0.0)
    scattered = {}
    for index, (camera, uv) in enumerate(_picks(scene, "block").items()):
        scattered[camera] = [uv[0] + (60 if index % 2 else -60), uv[1]]
    loose = L.World().anchor(body, "block", scattered, now=0.0)
    assert loose["rays_missed_by_m"] > tight["rays_missed_by_m"]
    assert world.confidence("block") > 0.4


def test_an_item_nobody_can_see_keeps_its_stale_belief_and_says_so(scene):
    """Not visible is not the same as not there, and neither is a position."""
    body, _viewer, _groups, _truth = scene
    world = L.World()
    world.anchor(body, "block", _picks(scene, "block"), now=0.0)
    # A look for a colour nothing in the room has.
    world.looks["ghost"] = L.Look(name="ghost", chroma_r=0.9, chroma_g=0.05,
                                  lum=200.0)
    world.belief["ghost"] = np.asarray([1.0, 1.0, 1.0])
    world.last_seen_s["ghost"] = 0.0
    report = world.refresh(body, now=4.0)
    assert report["ghost"]["seen_now"] is False
    assert report["ghost"]["last_seen_s_ago"] == pytest.approx(4.0)
    assert report["ghost"]["believed_xyz"] == [1.0, 1.0, 1.0]
