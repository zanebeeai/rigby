"""Where does it grasp, and where does it fail? Both hands, ten placements.

One success is a configuration, not a capability. Every constant in the grasp
was fitted against a 6 cm block at one position with the right hand, so this
varies the object and the side and asks what survives -- which is the difference
between having solved the task and having solved the fixture.

It is also how the left hand's failure was found. It had the entire vocabulary,
an identical pre-grasp shape and identical grip metrics, and could not pick
anything up, because the palm's outward normal was read from an axis that flips
with handedness. Nothing but a sweep like this would have shown it: the single
authored scene is right-handed, and it passed.

Run it after touching a primitive, a threshold, or the physics.
"""

from __future__ import annotations

import numpy as np

from rigby_poc.closed_loop import (
    generate,
    object_in_grasp,
    observe,
    palm_facing,
)
from rigby_poc.models import Hand, default_scene
from rigby_poc.scripted_grasp import ScriptedGrasp, lift_achieved
from rigby_poc.talmy import interpret

#: (label, translation offset in metres, size in metres or None to keep it).
CASES: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float] | None], ...] = (
    ("as fitted",            (0.0, 0.0, 0.0),   None),
    ("10 cm nearer",         (0.0, 0.0, -0.10), None),
    ("10 cm further",        (0.0, 0.0, 0.10),  None),
    ("8 cm right",           (0.08, 0.0, 0.0),  None),
    ("8 cm left",            (-0.08, 0.0, 0.0), None),
    ("15 cm left",           (-0.15, 0.0, 0.0), None),
    ("narrow block 4 cm",    (0.0, 0.0, 0.0),   (0.04, 0.08, 0.04)),
    ("wide block 8 cm",      (0.0, 0.0, 0.0),   (0.08, 0.08, 0.08)),
    ("tall block 12 cm",     (0.0, 0.0, 0.0),   (0.06, 0.12, 0.06)),
    ("flat block 4 cm tall", (0.0, 0.0, 0.0),   (0.06, 0.04, 0.06)),
)

#: A lift that is still up at the end. A peak that falls back is the object
#: being knocked upward, which reads the same in a maximum and is not a grasp.
_HELD_M = 0.05


def run_case(label, shift, size, hand, seconds: float = 9.0) -> dict:
    scene = default_scene()
    block = scene.object_by_id("block")
    block.transform.translation.x += shift[0]
    block.transform.translation.y += shift[1]
    block.transform.translation.z += shift[2]
    if size:
        block.dimensions_m.x, block.dimensions_m.y, block.dimensions_m.z = size
    half = np.asarray([block.dimensions_m.x, block.dimensions_m.y,
                       block.dimensions_m.z]) / 2.0
    selector = ScriptedGrasp(hand=hand)
    frames, _ = generate(interpret("pick up box", scene), scene, hand,
                         selector=selector, max_seconds=seconds)
    achieved = lift_achieved(frames)
    # What the hand had achieved around the moment it would be closing.
    moment = min(frames, key=lambda f: abs(f.time_s - 2.0))
    where = moment.objects["block"].translation
    position = np.asarray([where.x, where.y, where.z])
    reading = observe(moment.bones, hand, position, position, {}, False, 2.0,
                      object_half_m=half)
    return {
        "label": label,
        "hand": hand.value,
        "held_m": achieved["final_lift_m"],
        "peak_m": achieved["peak_lift_m"],
        "pairs": selector.pairs_seen,
        "in_grasp_m": object_in_grasp(reading, hand),
        "facing": palm_facing(reading, hand),
        "lifted": achieved["final_lift_m"] > _HELD_M,
    }


def main() -> None:
    print(f"{'case':<22}{'hand':<7}{'held cm':>9}{'pairs':>7}"
          f"{'in_grasp':>10}{'facing':>8}  verdict")
    print("-" * 74)
    wins = 0
    rows = []
    for label, shift, size in CASES:
        for hand in (Hand.RIGHT, Hand.LEFT):
            try:
                row = run_case(label, shift, size, hand)
            except Exception as error:  # noqa: BLE001
                print(f"{label:<22}{hand.value:<7}{'--':>9}{'--':>7}{'--':>10}"
                      f"{'--':>8}  failed: {str(error)[:24]}")
                continue
            rows.append(row)
            wins += row["lifted"]
            verdict = ("LIFTED" if row["lifted"]
                       else "raised" if row["peak_m"] > 0.02 else "no")
            print(f"{row['label']:<22}{row['hand']:<7}{row['held_m'] * 100:>9.2f}"
                  f"{row['pairs']:>7}{row['in_grasp_m'] * 100:>9.1f}c"
                  f"{row['facing']:>8.2f}  {verdict}")
    print(f"\nlifted and held: {wins} of {len(rows)}")
    if rows:
        # The discriminator, stated rather than left to be noticed: every
        # success has sat under about 3 cm of grasp-line error and every
        # failure above it, while palm_facing is good almost everywhere.
        good = [r["in_grasp_m"] for r in rows if r["lifted"]]
        bad = [r["in_grasp_m"] for r in rows if not r["lifted"]]
        if good and bad:
            print(f"in_grasp when it worked : {min(good) * 100:.1f} to "
                  f"{max(good) * 100:.1f} cm")
            print(f"in_grasp when it did not: {min(bad) * 100:.1f} to "
                  f"{max(bad) * 100:.1f} cm")


if __name__ == "__main__":
    main()
