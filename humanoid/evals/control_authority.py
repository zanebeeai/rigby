"""Which controls can move which numbers, and in which direction.

Twice a primitive has held the whole loop while achieving nothing: orient_palm
for eleven of twenty-six decisions in run 000488, thumb_oppose_index for ten of
twenty-six in 000489. Both times the model's reasoning was sound and read
correctly off the sensors. The control simply had no authority over the number
it was chosen to move, and the model has no way to discover that -- it sees a
name, a description, and a metric that will not budge, so it tries again.

Direction is the half that matters and the half an unsigned audit misses. A
control that moves a number hard in the wrong direction looks identical to one
that moves it hard in the right direction, and the thumb_oppose family scores
well on magnitude while pushing the fingertip rays APART -- which is the blade
instead of a C, the failure that has been visible in the renders all along.

Run this after touching any primitive. A vocabulary that cannot express the
task is not a controller problem or a prompting problem, and no amount of either
will fix it.
"""

from __future__ import annotations

import numpy as np

from rigby_poc.closed_loop import (
    METRICS,
    SuperPrimitiveSelector,
    _register_metrics,
    generate,
    observe,
    sense,
    super_primitives,
)
from rigby_poc.models import BonePose, Hand, default_scene
from rigby_poc.talmy import interpret

#: What a grasp needs each number to become. A metric with no entry here is one
#: nothing is steering toward, which is its own kind of finding.
WANTED = {
    "c_closure": -1.0,
    "grip_closure": -1.0,
    "thumb_opposition": +1.0,
    "object_in_grasp_m": 0.0,
    "tips_to_object_m": 0.0,
    "ray_dot": -1.0,
    "rays_toward": +1.0,
}

MAGNITUDES = (0.3, 0.7, 1.0)
#: Below this a control is doing nothing the loop can build on.
_MEANINGFUL = 0.005


def reference_pose(hand: Hand = Hand.RIGHT):
    """A pose partway through a reach, which is where these choices are made.

    Auditing from rest would flatter every control that only matters once the
    hand is near the object, and the failures being hunted all happen close in.
    """
    scene = default_scene()
    situation = interpret("pick up box", scene)
    frames, _ = generate(
        situation, scene, hand,
        selector=SuperPrimitiveSelector(
            hand=hand, shortlist=("reach_to", "look_at_object")),
        max_seconds=1.0,
    )
    frame = frames[-1]
    block = frame.objects["block"].translation
    position = np.array([block.x, block.y, block.z])
    return observe(frame.bones, hand, position, position, {}, False, 1.0)


def authority(base, hand: Hand, metric: str, want: float):
    """Signed progress each control can make toward ``want``.

    Positive is a control that closes the gap; negative is one that widens it.
    """
    _register_metrics()
    here = float(METRICS[metric](base, hand))
    helps: list[tuple[float, str]] = []
    hurts: list[tuple[float, str]] = []
    for primitive in super_primitives(hand):
        best = 0.0
        for amount in MAGNITUDES:
            try:
                rotations = primitive.solve(base, hand, amount)
            except Exception:  # noqa: BLE001
                continue
            if not rotations:
                continue
            trial = dict(base.bones)
            for bone, rotation in rotations.items():
                trial[bone] = BonePose(rotation=rotation)
            probe = sense(trial, hand, base.object_position,
                          base.contact_force_n, base.opposed, base.time_s)
            try:
                value = float(METRICS[metric](probe, hand))
            except Exception:  # noqa: BLE001
                continue
            gain = abs(here - want) - abs(value - want)
            if abs(gain) > abs(best):
                best = gain
        if best > _MEANINGFUL:
            helps.append((best, primitive.name))
        elif best < -_MEANINGFUL:
            hurts.append((-best, primitive.name))
    helps.sort(reverse=True)
    hurts.sort(reverse=True)
    return here, helps, hurts


def main() -> None:
    hand = Hand.RIGHT
    base = reference_pose(hand)
    out = float(np.linalg.norm(base.convergence - base.object_position))
    print(f"pose: fingertips {out * 100:.1f} cm from the object\n")
    unreachable = []
    for metric, want in WANTED.items():
        here, helps, hurts = authority(base, hand, metric, want)
        print(f"{metric:<18}{here:+7.3f} -> {want:+5.1f}")
        print("    HELPS: " + (
            ", ".join(f"{n} (+{v:.3f})" for v, n in helps[:5]) or "NOTHING"))
        print("    HURTS: " + (
            ", ".join(f"{n} (-{v:.3f})" for v, n in hurts[:4]) or "none"))
        if not helps:
            unreachable.append(metric)
    if unreachable:
        print("\nNO CONTROL CAN IMPROVE: " + ", ".join(unreachable))
        print("The vocabulary cannot express the task. This is not something a "
              "better prompt or a longer run will reach.")


if __name__ == "__main__":
    main()
