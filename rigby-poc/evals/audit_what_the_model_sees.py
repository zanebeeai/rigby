"""Every field the model is handed, and where that field actually comes from.

A claim that a machine is not cheating is worth nothing unless it is checked
against the payload rather than against the intention. This builds the `sensed`
block of a real decision payload -- readable(body, seen), which is where every
number the model may drive comes from -- and classifies each field by where it
actually comes from:

  PERCEIVED       from the cameras, through the belief. Wrong when the cameras
                  are wrong, which is the point.
  PROPRIOCEPTION  joint encoders and the forward kinematics they imply. Exact,
                  and every real arm has it.
  MEASURED        a real sensor reading: fingertip load cells, the rangefinder.
  DECLARED        known furniture, calibration, the mechanism's own dimensions.
                  Legitimate a priori knowledge, but not free -- someone had to
                  measure it once, and it is wrong if the world moves.
  MODEL           something the model itself said earlier in the run.
  SIMULATOR       read out of MuJoCo's state. This is the cheating category.
                  Anything here is knowledge no real machine of this class
                  would have.

WHAT THIS DOES NOT COVER, and how those parts were checked instead. The payload
carries more than `sensed`, and the rest is small enough to audit by reading:

  scene()          the task string, bin_centre_xyz and bin_rim_height_m from the
                   manifest, plus the sentence "the block's position is not
                   known and must be seen". DECLARED, and no block position.
  machine()        joint ranges and fractions -- PROPRIOCEPTION and the
                   mechanism's declared limits. Contains no call to body.block()
                   or to data.qpos.
  your_imagination PERCEIVED, plus the declared bin beside it for comparison.
  aim()            derived from the belief and the arm's own pose.
  step_partitioned Talmy's FIGURE/GROUND/PATH labels. Words, no coordinates --
                   the SceneManifest it is built from does carry positions, but
                   Talmy runs locally and only its labels are sent.

Run it as a script; it prints a table and exits non-zero if anything lands in
SIMULATOR or cannot be traced, so the question cannot quietly stop being asked.
A non-zero exit today is CORRECT and expected: object_in_target is the grader
and it is currently visible to the thing being graded.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from rigby_poc.gripper.decision.goals import (  # noqa: E402
    OUTCOMES, READABLE, readable,
)
from rigby_poc.gripper.physics.model import make  # noqa: E402
from rigby_poc.gripper.sensing.gripper_camera import Senses, sense  # noqa: E402

#: Provenance of every metric name, traced by reading the function that
#: computes it. The evidence is cited so a reader can check rather than trust.
WHERE_FROM = {
    # --- the arm knows where its own joints are -------------------------------
    "hand_x_m": ("PROPRIOCEPTION", "forward kinematics from the encoders"),
    "hand_y_m": ("PROPRIOCEPTION", "forward kinematics from the encoders"),
    "hand_z_m": ("PROPRIOCEPTION", "forward kinematics from the encoders"),
    "hand_pointing_down": ("PROPRIOCEPTION", "the plate's orientation"),
    "base_deg": ("PROPRIOCEPTION", "encoder"),
    "segment_1_deg": ("PROPRIOCEPTION", "encoder"),
    "segment_2_deg": ("PROPRIOCEPTION", "encoder"),
    "segment_3_deg": ("PROPRIOCEPTION", "encoder"),
    "grip_tip_spread_m": ("PROPRIOCEPTION", "finger encoders"),
    # --- object-relative: all of these go through the BELIEF ------------------
    "palm_to_object_m": ("PERCEIVED", "seen.object_at, seen.object_size"),
    "object_in_grasp_m": ("PERCEIVED", "seen.object_at"),
    "palm_facing": ("PERCEIVED", "chosen_face() from seen.object_at/size"),
    "pointing_at_object": ("PERCEIVED", "the belief, via aim()"),
    "object_seen": ("PERCEIVED", "whether a camera has a fix"),
    "object_in_hand_view": ("PERCEIVED", "the hand camera's frame"),
    "object_between_jaws": ("PERCEIVED", "seen.object_at against the pads"),
    # --- belief plus declared furniture ---------------------------------------
    "object_over_target_m": ("DECLARED", "seen.object_at vs bin_of()['centre']"),
    "object_above_rim_m": ("DECLARED", "seen.object_at vs bin_of()['rim']"),
    # --- real instruments ------------------------------------------------------
    "tip_force_left_n": ("MEASURED", "fingertip load cell"),
    "tip_force_right_n": ("MEASURED", "fingertip load cell"),
    "range_ahead_m": ("MEASURED", "rangefinder along the grasp axis"),
    "beam_finds_object": ("MEASURED", "the same rangefinder"),
    "holding": ("MEASURED", "both load cells under a commanded squeeze"),
    # --- the grader -------------------------------------------------------------
    "object_in_target": ("SIMULATOR", "body.block() -- MuJoCo's true block pose"),
}

TIP = ("segment_1_tip_", "segment_2_tip_", "segment_3_tip_")

#: A field can be sourced from a percept and still be a lie, if the percept
#: itself is a constant. This check exists because the first version of this
#: audit missed exactly that: palm_to_object_m traces to seen.object_size, was
#: classified PERCEIVED, and seen.object_size turned out to be the literal
#: `(point, 0.025, 0.03)` on the corner path -- the block's true half-extents,
#: typed in. Tracing one hop and stopping is how a leak survives an audit.
def size_is_estimated(body, seen) -> tuple[bool, str]:
    """Is the believed object size measured, or is it a typed-in constant?

    Built two bodies with different blocks and asks the sensing path how big
    each one is. A path that measures returns two different answers. A path
    that knows returns the truth twice, and a path with a constant returns the
    same wrong number twice.
    """
    from rigby_poc.gripper.sensing.gripper_camera import Senses, sense
    from rigby_poc.gripper.physics.model import make as _make

    answers = []
    for half in (np.asarray([0.025, 0.025, 0.03]),
                 np.asarray([0.045, 0.045, 0.02])):
        other = _make(half, np.asarray([0.0, 0.30, 0.76]), table_top=0.72)
        eyes = Senses()
        look = sense(other, eyes, np.asarray(other.q()), 0.0, 0.0, 0.72)
        answers.append((half, None if look.object_size is None
                        else np.asarray(look.object_size)))
    (h1, s1), (h2, s2) = answers
    if s1 is None or s2 is None:
        return True, "no size was produced, so nothing is asserted"
    if np.allclose(s1, s2, atol=1e-6):
        return False, (f"the same size {np.round(s1, 4).tolist()} is returned "
                       f"for two different blocks -- it is a constant, not a "
                       f"measurement")
    exact = np.allclose(s1, h1, atol=1e-6) and np.allclose(s2, h2, atol=1e-6)
    if exact:
        return False, ("the size returned is EXACTLY the true half-extents for "
                       "both blocks, which no camera can do")
    return True, (f"two blocks give two answers, off by "
                  f"{np.abs(s1 - h1).max() * 1000:.1f} and "
                  f"{np.abs(s2 - h2).max() * 1000:.1f} mm -- estimated")


def classify(name: str):
    if name in WHERE_FROM:
        return WHERE_FROM[name]
    if name.startswith(TIP):
        return ("PROPRIOCEPTION", "forward kinematics from the encoders")
    return ("UNCLASSIFIED", "not traced -- audit is incomplete for this field")


def main() -> int:
    body = make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=0.72)
    eyes = Senses()
    seen = sense(body, eyes, np.asarray(body.q()), 0.0, 0.0, 0.72)
    numbers = readable(body, seen)

    buckets: dict[str, list] = {}
    for name in sorted(numbers):
        kind, why = classify(name)
        buckets.setdefault(kind, []).append((name, why))

    print("WHAT THE MODEL IS HANDED, BY PROVENANCE")
    print()
    for kind in ("PERCEIVED", "PROPRIOCEPTION", "MEASURED", "DECLARED",
                 "MODEL", "SIMULATOR", "UNCLASSIFIED"):
        rows = buckets.get(kind) or []
        if not rows:
            continue
        print(f"  {kind}  ({len(rows)})")
        for name, why in rows:
            askable = "askable" if name in READABLE else "reading only"
            print(f"    {name:<24} {askable:<13} {why}")
        print()

    honest, note = size_is_estimated(body, seen)
    print("  IS THE BELIEVED OBJECT SIZE MEASURED, OR KNOWN?")
    print(f"    {'estimated' if honest else 'NOT MEASURED'}: {note}")
    print()

    leaks = [n for n, _ in buckets.get("SIMULATOR", [])]
    if not honest:
        leaks.append("seen.object_size (a constant behind a PERCEIVED field)")
    unknown = [n for n, _ in buckets.get("UNCLASSIFIED", [])]
    print(f"  fields shown to the model: {len(numbers)}")
    print(f"  askable as goals:          {len(set(READABLE) & set(numbers))}")
    print(f"  reading-only (OUTCOMES):   {len(set(OUTCOMES) & set(numbers))}")
    print()
    if unknown:
        print("  AUDIT INCOMPLETE -- untraced fields:", ", ".join(unknown))
    if leaks:
        print("  SIMULATOR TRUTH REACHES THE MODEL:", ", ".join(leaks))
        for name in leaks:
            print(f"    {name}: askable={name in READABLE}")
        return 1
    print("  no simulator truth in the payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
