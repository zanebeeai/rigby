"""Which named gate a gripper run trips, scored from the metrics it produced.

Ported from the humanoid corpus, and for the same reason it exists there: a run
reports its verdict as a success flag and some prose, which is fine for a human
reading a report and useless as a test assertion. A case that is SUPPOSED to
fail has to name the gate it was built to fail, because a sentence is not a
name and a reworded message breaks a substring match.

Two properties are load-bearing here, as in the original.

**Nothing here is a second definition of success.** `placed` is
`object_in_target`, read from the simulator, and every other gate is a
necessary condition for it that can be scored on its own. A gate that
disagreed with the run's own outcome would be a second opinion about the same
question, which is worse than having no gate.

**A gate must be able to fail for a stated reason.** `no_drop` is only scored
when the object was actually lifted, because "did not drop it" is not an
achievement for a run that never picked anything up, and a gate that passes
vacuously reports the same thing whether the machine worked or did nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """Everything a scored run reports. Nothing derived, nothing judged."""

    placed: bool
    ever_held: bool
    peak_lift_m: float
    deepest_penetration_mm: float
    ever_seen_close: bool
    held_at_end: bool
    final_block_z: float

#: A grasp counts as a lift once the block is clear of the bench by this much.
LIFT_M = 0.05
#: Deeper than this and the pads are inside the block rather than holding it.
PENETRATION_MM = 3.0


def scored(outcome: Outcome) -> dict[str, bool | None]:
    """Every gate, by name. None means the gate does not apply to this run."""
    lifted = outcome.peak_lift_m >= LIFT_M
    return {
        # The task itself.
        "placed": outcome.placed,
        # Necessary conditions, each failable on its own.
        "grasped": outcome.ever_held,
        "lifted": lifted,
        "jaws_intact": outcome.deepest_penetration_mm <= PENETRATION_MM,
        "looked_closely": outcome.ever_seen_close,
        # Only meaningful for a run that had something to drop.
        "no_drop": (outcome.held_at_end or outcome.placed) if lifted else None,
    }


def failures(outcome: Outcome) -> list[str]:
    """The gates this run tripped, in a fixed order so diffs are readable."""
    marks = scored(outcome)
    return [name for name in
            ("placed", "grasped", "lifted", "jaws_intact", "looked_closely",
             "no_drop")
            if marks.get(name) is False]
