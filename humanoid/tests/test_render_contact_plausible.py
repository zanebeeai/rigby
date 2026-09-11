"""The renderer and the physics must agree that a carried object was touched.

Three things are asserted here, and they are deliberately different in kind.

1. The check *can* pass. A synthetic clip whose hand sits on the object clears
   the bound. Without this the other two assertions would be satisfied by a
   check that always fails, which ``docs/testing.md`` names as its own defect.
2. The check *skips* rather than passes when no object is lifted. A clip that
   never made a contact claim has not demonstrated agreement.
3. A **ledger**, in the shape of ``test_no_hardcoded_thresholds.py``: the number
   of untouched carried frames that the shipped pickup produces. It is recorded
   rather than tolerated, and it may only fall. Today it is every carried frame,
   which is the finding that motivated this module.

The third is the one that will change. When the grasp is repaired so the
rendered hand reaches the block, this number drops and the ledger is lowered
with it. It must never be raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.analysis import (
    carried_object_divergence_checks,
    carried_object_divergence_metrics,
    validate_clip,
)
from rigby_poc.compiler import compile_motion
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.models import (
    ClipFrame,
    CompileRequest,
    PlanRequest,
    Transform,
    Vec3,
    default_scene,
)
from rigby_poc.planner import plan_motion

#: no browser, no API key, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast




def _pickup():
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text="pick up box", scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    return scene, program, clip


def test_the_check_passes_when_the_rendered_hand_is_on_the_object() -> None:
    """A hand actually holding the block clears the bound.

    The clip is the shipped pickup with the block moved to the rendered palm on
    every frame, so the only thing that changed is the thing being measured.
    """
    _scene, program, clip = _pickup()
    kinematics = rig_kinematics()
    side = program.hand.value
    held = []
    for frame in clip.frames:
        positions = kinematics.canonical_positions(frame.bones)
        palm = np.mean(
            [
                positions[f"{side}Hand"],
                positions[f"{side}IndexProximal"],
                positions[f"{side}MiddleProximal"],
                positions[f"{side}RingProximal"],
                positions[f"{side}LittleProximal"],
            ],
            axis=0,
        )
        objects = dict(frame.objects)
        objects["block"] = Transform(
            translation=Vec3(x=float(palm[0]), y=float(palm[1]), z=float(palm[2])),
            rotation=objects["block"].rotation,
        )
        held.append(
            ClipFrame(time_s=frame.time_s, bones=frame.bones, objects=objects)
        )

    metrics = carried_object_divergence_metrics(held, program, clip.metrics)
    assert metrics["carried_frame_count"] > 0, "fixture must lift the block"
    assert metrics["carried_object_untouched_frame_ratio"] == pytest.approx(0.0)
    (check,) = carried_object_divergence_checks(metrics)
    assert check.status == "pass"


def test_the_check_skips_when_nothing_is_lifted() -> None:
    """No contact claim, no verdict -- and explicitly not a pass."""
    _scene, program, clip = _pickup()
    grounded = [
        ClipFrame(
            time_s=frame.time_s,
            bones=frame.bones,
            objects={**frame.objects, "block": clip.frames[0].objects["block"]},
        )
        for frame in clip.frames
    ]
    metrics = carried_object_divergence_metrics(grounded, program, clip.metrics)
    assert metrics["carried_frame_count"] == 0
    (check,) = carried_object_divergence_checks(metrics)
    assert check.status == "skip"


def test_the_check_is_reachable_from_validate() -> None:
    """Wired, not dark.

    ``validate()`` is the surface ``evals/mutations`` and plan 10 read. A check
    that exists but no caller reaches is the shape 04c shipped ROM in and had to
    correct later.
    """
    _scene, program, clip = _pickup()
    ids = {check.id for check in validate_clip(clip, program)}
    assert "physics.contact.render_plausible" in ids


def test_the_pickup_compiles_and_the_rendered_hand_reaches_the_block() -> None:
    """The ledger this replaces recorded 59 untouched airborne frames.

    Under the Cartesian proxy the block rose 0.142 m while the nearest rendered
    landmark stayed 0.078 m away. The grasp solver that now ships seats the
    hand on the block before it closes and carries the block by forward
    kinematics, so the rendered hand and the object the physics says it holds
    agree: the check that was written to catch the floating block passes on
    the shipped pickup, and this test pins that it stays passed. If the number
    below ever rises above zero, the renderer and the physics have started
    disagreeing again, which is the defect this module exists for.
    """
    _scene, program, clip = _pickup()
    assert clip.success, clip.failure

    metrics = carried_object_divergence_metrics(clip.frames, program, clip.metrics)
    assert metrics["carried_frame_count"] > 0, "the pickup lifts the block"
    assert metrics["carried_object_untouched_frame_ratio"] == 0.0, (
        f"{len(metrics['carried_object_untouched_frames'])} carried frames have no "
        "rendered hand landmark within the block's contact radius"
    )
