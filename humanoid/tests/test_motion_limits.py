"""How fast a joint may move, and what happens when a clip asks for more.

The range-of-motion envelope bounds where a bone may be and says nothing about
how fast it gets there. Those are different constraints, and the gap between
them is not academic: a clip can step a bone across its whole legal range in one
frame, every pose along the way inside the envelope, and the result is a strike.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_poc.models import BonePose, ClipFrame, Quat
from rigby_poc.motion_limits import limit_angular_rate, limit_for, rate_limits

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def _quat(degrees: float) -> Quat:
    xyzw = Rotation.from_euler("x", degrees, degrees=True).as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def _clip(bone: str, angles: list[float], fps: float = 30.0) -> list[ClipFrame]:
    return [
        ClipFrame(
            time_s=index / fps,
            bones={bone: BonePose(rotation=_quat(angle))},
            objects={},
        )
        for index, angle in enumerate(angles)
    ]


def test_every_joint_class_declares_a_rate() -> None:
    limits = rate_limits()
    assert limits["__default__"] > 0.0
    for name, value in limits.items():
        assert 0.0 < value <= 1000.0, f"{name} has an implausible ceiling"
    # Proximal joints carry more limb and move slower. That ordering is the
    # reason the values differ at all, so it is worth asserting.
    assert limits["spine"] < limits["shoulder"] < limits["elbow"] < limits["wrist"]
    assert limits["wrist"] < limits["thumb"] <= limits["digit"]


def test_a_bone_is_limited_by_its_own_joint_class() -> None:
    assert limit_for("rightIndexProximal") == rate_limits()["digit"]
    assert limit_for("rightThumbMetacarpal") == rate_limits()["thumb"]
    # An unknown bone falls back rather than going unbounded.
    assert limit_for("notABone") == rate_limits()["__default__"]


def test_a_step_faster_than_the_ceiling_is_clamped() -> None:
    """One frame, 60 degrees, at 30 fps -- 1800 deg/s asked of the wrist."""
    frames = _clip("rightHand", [0.0, 60.0])
    limited, report = limit_angular_rate(frames)
    ceiling = rate_limits()["wrist"]
    assert report.clamped_frames == 1
    assert report.peak_before_deg_per_s == pytest.approx(1800.0, rel=1e-3)
    assert report.peak_after_deg_per_s == pytest.approx(ceiling, rel=1e-6)

    delivered = Rotation.from_quat(
        [limited[1].bones["rightHand"].rotation.x, limited[1].bones["rightHand"].rotation.y,
         limited[1].bones["rightHand"].rotation.z, limited[1].bones["rightHand"].rotation.w]
    ).magnitude()
    assert math.degrees(delivered) == pytest.approx(ceiling / 30.0, abs=1e-6)


def test_a_motion_already_inside_the_ceiling_is_left_alone() -> None:
    """The limiter must not smooth, ease, or otherwise edit a legal motion."""
    slow = [0.0, 1.0, 2.0, 3.0]
    frames = _clip("rightHand", slow)
    limited, report = limit_angular_rate(frames)
    assert report.clamped_frames == 0
    assert not report.unreached_bones
    for before, after in zip(frames, limited):
        a = before.bones["rightHand"].rotation
        b = after.bones["rightHand"].rotation
        assert (a.x, a.y, a.z, a.w) == (b.x, b.y, b.z, b.w)


def test_a_target_the_clip_never_reaches_is_reported_not_hidden() -> None:
    """Slowing a motion can truncate it, and that has to be visible.

    A limiter that quietly delivers a pose short of the authored one turns a
    violent motion into a wrong one, which is harder to find.
    """
    frames = _clip("rightHand", [0.0, 90.0])
    _limited, report = limit_angular_rate(frames)
    assert report.unreached_bones == ("rightHand",)
    assert report.worst_residual_deg == pytest.approx(90.0 - rate_limits()["wrist"] / 30.0, abs=1e-3)


def test_object_transforms_are_not_rate_limited() -> None:
    """Objects are the simulation's account of what happened, not a command."""
    from rigby_poc.models import Transform, Vec3

    frames = [
        ClipFrame(
            time_s=index / 30.0,
            bones={"rightHand": BonePose(rotation=_quat(index * 60.0))},
            objects={"block": Transform(translation=Vec3(x=float(index), y=0.0, z=0.0))},
        )
        for index in range(3)
    ]
    limited, report = limit_angular_rate(frames)
    assert report.clamped_frames > 0, "the bone must actually have been clamped"
    assert [f.objects["block"].translation.x for f in limited] == [0.0, 1.0, 2.0]


def test_a_single_frame_clip_is_returned_unchanged() -> None:
    frames = _clip("rightHand", [0.0])
    limited, report = limit_angular_rate(frames)
    assert limited == frames
    assert report.clamped_frames == 0
