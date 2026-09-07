"""The grasp solver holds any block it can reach, not the corpus's one.

The seat is built from the hand's own measurements and the block's box, so
nothing about `grasp-block-table` (block at (0, 1.05, 0.29), 6 x 8 x 6 cm,
right hand) is written into it. This sweeps the block over positions, sizes
and both hands on the default scene and asks the same questions of every clip
that the contact audit and the ROM gate ask of the corpus case: the clip
compiles, the skinned hand is never inside the block or the table once it has
left the idle pose, the wrist stays inside its enforced range, the block rises
with the hand, and at the hold the palm and fingers are on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_poc.analysis.anatomy.rom import rom_checks
from rigby_poc.compiler import compile_motion
from rigby_poc.hand_mesh import box_signed_distance, hand_mesh
from rigby_poc.models import (
    CompileRequest,
    Hand,
    PlanRequest,
    PrimitiveKind,
    Vec3,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner

pytestmark = pytest.mark.medium

#: The contact audit's touch tolerance and the number of leading frames the
#: shared idle pose is allowed (its little finger starts 3 mm in the table).
TOUCH_TOLERANCE_M = 0.002
IDLE_FRAMES = 6
HELD_DISTANCE_M = 0.004

#: Blocks shorter than the hand's own geometry allows. The fingers are pitched
#: toward the forearm to keep the wrist inside its deviation range, so the open
#: little fingertip hangs below the knuckle line; keeping it out of the table
#: pushes the hand up a short block until only that finger is on it. The fix
#: is a different grasp family for short blocks (a pronated hand, from
#: above), not a tuning of this one; strict so the pin flips the day it lands.
SHORT_BLOCK = pytest.mark.xfail(
    reason="a block under ~6 cm ends up held by the little finger alone: side grasp with pitched fingers",
    strict=True,
)

PLACEMENTS = [
    pytest.param(Hand.RIGHT, (0.0, 0.29), (0.06, 0.08, 0.06), id="right-corpus"),
    pytest.param(Hand.RIGHT, (-0.15, 0.24), (0.06, 0.08, 0.06), id="right-outboard"),
    pytest.param(Hand.RIGHT, (-0.06, 0.32), (0.06, 0.08, 0.06), id="right-far"),
    pytest.param(Hand.RIGHT, (0.05, 0.22), (0.06, 0.08, 0.06), id="right-across"),
    pytest.param(Hand.RIGHT, (0.0, 0.29), (0.05, 0.05, 0.05), id="right-small-cube", marks=SHORT_BLOCK),
    pytest.param(Hand.RIGHT, (0.0, 0.29), (0.08, 0.06, 0.05), id="right-wide-low"),
    pytest.param(Hand.RIGHT, (0.0, 0.29), (0.04, 0.10, 0.04), id="right-tall-thin"),
    pytest.param(Hand.LEFT, (0.0, 0.29), (0.06, 0.08, 0.06), id="left-corpus"),
    pytest.param(Hand.LEFT, (0.15, 0.24), (0.06, 0.08, 0.06), id="left-outboard"),
    pytest.param(Hand.LEFT, (-0.05, 0.22), (0.05, 0.05, 0.05), id="left-across-small", marks=SHORT_BLOCK),
]


def _scene_with_block(position: tuple[float, float], size: tuple[float, float, float]):
    scene = default_scene()
    table = scene.support_surface()
    assert table is not None
    top = table.transform.translation.y + table.dimensions_m.y / 2.0
    objects = []
    for item in scene.objects:
        if item.id != "block":
            objects.append(item)
            continue
        objects.append(
            item.model_copy(
                update={
                    "transform": item.transform.model_copy(
                        update={"translation": Vec3(x=position[0], y=top + size[1] / 2.0, z=position[1])}
                    ),
                    "dimensions_m": Vec3(x=size[0], y=size[1], z=size[2]),
                }
            )
        )
    return scene.model_copy(update={"objects": objects})


@pytest.mark.parametrize(("hand", "position", "size"), PLACEMENTS)
def test_the_seat_holds_any_reachable_block(hand: Hand, position, size) -> None:
    scene = _scene_with_block(position, size)
    program = OfflinePlanner().plan(
        PlanRequest(
            text=f"Grab the block with your {hand.value} hand.",
            scene=scene,
            provider="offline",
        )
    ).program
    assert program.hand == hand
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success, clip.failure

    mesh = hand_mesh(hand.value)
    block = scene.object_by_id("block")
    table = scene.support_surface()
    assert block is not None and table is not None
    table_centre = np.asarray(table.transform.translation.as_list())
    table_half = np.asarray(table.dimensions_m.as_list()) / 2.0
    block_half = np.asarray(block.dimensions_m.as_list()) / 2.0

    worst_block, worst_table = 0.0, 0.0
    for frame in clip.frames[IDLE_FRAMES:]:
        points = mesh.world(frame.bones)
        pose = frame.objects["block"]
        centre = np.asarray(pose.translation.as_list())
        rotation = Rotation.from_quat(pose.rotation.as_list()).as_matrix()
        worst_block = max(worst_block, -float(box_signed_distance(points, centre, rotation, block_half).min()))
        worst_table = max(worst_table, -float(box_signed_distance(points, table_centre, np.eye(3), table_half).min()))
    assert worst_block <= TOUCH_TOLERANCE_M, f"hand {worst_block * 1000:.1f} mm inside the block"
    assert worst_table <= TOUCH_TOLERANCE_M, f"hand {worst_table * 1000:.1f} mm inside the table"

    wrist_failures = [
        result.id
        for result in rom_checks(clip.frames, fps=clip.fps)
        if result.status == "fail" and f"{hand.value}Hand." in result.id
    ]
    assert not wrist_failures, wrist_failures

    lift = next(item.parameters.lift_height_m for item in program.primitives if item.kind == PrimitiveKind.LIFT)
    start_y = clip.frames[0].objects["block"].translation.y
    end_y = clip.frames[-1].objects["block"].translation.y
    assert end_y - start_y >= 0.9 * lift, f"block rose {end_y - start_y:.3f} m of {lift:.3f}"

    last = clip.frames[-1]
    points = mesh.world(last.bones)
    pose = last.objects["block"]
    centre = np.asarray(pose.translation.as_list())
    rotation = Rotation.from_quat(pose.rotation.as_list()).as_matrix()
    on_block = [
        name
        for name, rows in mesh.groups.items()
        if float(box_signed_distance(points[rows], centre, rotation, block_half).min()) <= HELD_DISTANCE_M
    ]
    assert "Thumb" in on_block and len(on_block) >= 3, f"only {on_block} on the block at the hold"
