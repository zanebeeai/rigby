"""The signal-quality layer against the real corpus. Plan 10 §3.2.

Two of §3.2's seven checks ship. The deferrals are in §3.2's own table with the
measurement behind each; the shortest form is that ``min_jerk``'s criterion is a
*single point-to-point reach* and these clips are multi-submovement, and the
other four need event or gait structure this corpus does not carry.

The headline number is ``dead_limb``: **38 of 47**, and always both legs.
"""

from __future__ import annotations

import pytest

from rigby_poc.analysis import validate_clip
from rigby_poc.analysis.contract import SIGNAL, CheckResult
from rigby_poc.analysis.signal_quality import LIMB_CHAINS, bone_activity

#: One corpus compile per case in a module fixture; see test_corpus_compile_budget.py.
pytestmark = pytest.mark.medium

SIGNAL_IDS = frozenset({"signal.activity.dead_limb", "signal.smoothness.sparc"})

NO_FRAMES_CASE = "knownbad-eigenvalues-unsupported"

#: A shrug moves the shoulders and neither limb chain, so there is no active body
#: against which a limb could be dead. Named because it is the one case that
#: exercises the body-activity guard, and a silent change to that guard would
#: otherwise turn it into a 39th failure without anyone noticing.
NO_LIMB_MOTION_CASE = "fullbody-shrug-shoulders"

#: The seven cases whose legs actually move. Measured on e71a6f5.
LEG_MOVING_CASES = frozenset(
    {
        "fullbody-burpee-cycle",
        "fullbody-cartwheel",
        "fullbody-dance",
        "fullbody-ladder-climb",
        "fullbody-run-forward",
        "fullbody-walk-forward",
        "sequence-wave-then-pushup",
    }
)


@pytest.fixture(scope="module")
def verdicts(compile_whole_corpus, corpus_by_id) -> dict[str, tuple[list[CheckResult], object]]:
    return {
        case_id: (validate_clip(clip, corpus_by_id[case_id].program), clip)
        for case_id, clip in compile_whole_corpus().items()
    }


def test_the_probe_saw_a_real_corpus(verdicts) -> None:
    assert len(verdicts) >= 47, f"the corpus produced {len(verdicts)} cases"
    assert NO_FRAMES_CASE in verdicts
    assert NO_LIMB_MOTION_CASE in verdicts


def test_every_case_emits_every_signal_id(verdicts) -> None:
    """Equality, on every case, including the ones that measured nothing."""

    for case_id, (checks, _clip) in verdicts.items():
        emitted = {
            c.id
            for c in checks
            if c.id.startswith(("signal.activity", "signal.smoothness"))
        }
        assert emitted == SIGNAL_IDS, case_id
        for check in checks:
            if check.id in SIGNAL_IDS:
                assert check.layer == SIGNAL, f"{case_id}: {check.id}"


def test_the_dead_limb_check_fires_on_the_upper_body_only_clips(verdicts) -> None:
    """38 of 47, and the shape is the finding rather than the count.

    The compiler leaves the legs at **literally their rest rotation** for the
    whole clip on every gesture, composite, strike and object path. Reported at
    38 of 47 rather than tuned around: the criterion is right and the subject is
    bad, which is what plan 10 §10.5 says to expect.
    """

    passing = {
        case_id
        for case_id, (checks, _clip) in verdicts.items()
        if any(
            c.id == "signal.activity.dead_limb" and c.status == "pass" for c in checks
        )
    }
    assert passing == LEG_MOVING_CASES

    failing = [
        case_id
        for case_id, (checks, _clip) in verdicts.items()
        if any(c.id == "signal.activity.dead_limb" and c.failed for c in checks)
    ]
    assert len(failing) == 38, f"{len(failing)} cases failed dead_limb, expected 38"


def test_the_rigid_chain_is_usually_a_leg_and_the_legs_are_exactly_rigid(
    verdicts,
) -> None:
    """The precise shape, after a first version of this claim was wrong.

    I reported "always both legs, never an arm" from a probe that tested for
    *exactly* 0.0. At the check's actual 0.01 rad bound that is false: the
    quietest chain is a leg on 37 of 46 cases and an **arm** on 9 — the idle arm
    of a one-handed gesture, which `composite-beckon-right` shows as
    ``left_arm ~ 0`` while ``right_arm`` does 5.7 rad of work.

    Both halves are worth pinning because they are different facts:

    - **The legs are at *exactly* zero rotation on 35 cases.** Not small: zero.
      The compiler never writes a leg rotation at all on the gesture, composite,
      strike and object paths.
    - **Arms reach ~1e-7 but never exactly zero**, so a rigid arm is a limb that
      was driven and barely moved, not one that was never addressed.

    A rigid idle arm is still a finding rather than a false positive — §3.2's
    "frozen mannequin limbs" is exactly the one-handed-gesture case, and a real
    human's idle arm is not rigid. But it is a *different* defect from a leg the
    compiler never animates, and folding them into one sentence lost that.
    """

    exactly_zero = {name: 0 for name in LIMB_CHAINS}
    quietest_kind = {"arm": 0, "leg": 0}
    smallest_nonzero = float("inf")

    for _checks, clip in verdicts.values():
        if not clip.frames:
            continue
        bones = tuple(b for chain in LIMB_CHAINS.values() for b in chain)
        activity = bone_activity(clip.frames, bones)
        totals = {
            name: sum(activity[b] for b in chain) for name, chain in LIMB_CHAINS.items()
        }
        for name, value in totals.items():
            if value == 0.0:
                exactly_zero[name] += 1
            else:
                smallest_nonzero = min(smallest_nonzero, value)
        quiet = min(totals, key=lambda name: totals[name])
        quietest_kind["leg" if quiet.endswith("_leg") else "arm"] += 1

    assert exactly_zero["left_arm"] == exactly_zero["right_arm"] == 0, (
        f"an arm reached exactly zero rotation: {exactly_zero}. Arms are driven "
        "and barely move; legs are never written at all, and the distinction is "
        "the finding."
    )
    assert exactly_zero["left_leg"] >= 30 and exactly_zero["right_leg"] >= 30, (
        f"the legs are no longer exactly rigid on most cases: {exactly_zero}"
    )
    assert quietest_kind["leg"] > quietest_kind["arm"], quietest_kind
    assert smallest_nonzero < 0.01, (
        f"the smallest non-zero chain activity is {smallest_nonzero}, above the "
        "bound; the rigid-arm cases have stopped being rigid"
    )


def test_a_clip_with_no_limb_motion_skips_rather_than_failing(verdicts) -> None:
    """A shrug is not a clip with a dead limb; it is a clip with no limbs moving.

    Folding those together would make "nothing to compare against" indistinguishable
    from "a limb is frozen while the rest moves" — the not-measured/measured-negative
    collapse. This case is why the guard exists and it is the only one that reaches it.
    """

    checks, _clip = verdicts[NO_LIMB_MOTION_CASE]
    dead = next(c for c in checks if c.id == "signal.activity.dead_limb")
    assert dead.status == "skip"
    assert dead.headroom is None


def test_the_zero_frame_clip_skips_both_signal_checks(verdicts) -> None:
    checks, _clip = verdicts[NO_FRAMES_CASE]
    for check in checks:
        if check.id in SIGNAL_IDS:
            assert check.status == "skip", check.id
            assert check.headroom is None, check.id


def test_sparc_is_an_outlier_bound_and_behaves_like_one(verdicts) -> None:
    """It must fire rarely, and what it fires on must be the extreme.

    This is not a naturalness gate and the test says so in the only way a test
    can: by pinning that almost nothing fails it. Against the corpus's own labels
    SPARC separates structurally-valid from known-bad at **AUC 0.546, n = 205
    pairs, baseline 0.500** — chance — so a bound that failed a meaningful
    fraction would be asserting a quality claim the evidence does not support.
    """

    measured = {}
    for case_id, (checks, _clip) in verdicts.items():
        for check in checks:
            if check.id == "signal.smoothness.sparc" and check.status != "skip":
                measured[case_id] = float(check.measured)
    assert len(measured) >= 45, f"only {len(measured)} cases produced a SPARC value"

    failing = {
        case_id
        for case_id, (checks, _clip) in verdicts.items()
        if any(c.id == "signal.smoothness.sparc" and c.failed for c in checks)
    }
    assert failing == {"fullbody-burpee-cycle"}
    assert min(measured.values()) == measured["fullbody-burpee-cycle"], (
        "the only failing case is not the corpus minimum, so the bound has "
        "stopped being an outlier bound"
    )


def test_the_signal_layer_is_live_under_a_frame_mutation(verdicts, corpus_by_id) -> None:
    """Frames-derived, so a transformed clip moves it. Same property as physics.

    Freezing every bone to its first-frame rotation is the mutation ``dead_limb``
    exists to catch, and it must move the verdict on a clip that currently passes.
    """

    checks, clip = verdicts["fullbody-walk-forward"]
    before = next(c for c in checks if c.id == "signal.activity.dead_limb")
    assert before.status == "pass", "the control case stopped passing dead_limb"

    case = corpus_by_id["fullbody-walk-forward"]
    first = clip.frames[0].bones
    frozen = clip.model_copy(
        update={
            "frames": [
                frame.model_copy(
                    update={
                        "bones": {
                            name: first.get(name, bone)
                            for name, bone in frame.bones.items()
                        }
                    }
                )
                for frame in clip.frames
            ]
        }
    )
    assert frozen.metrics == clip.metrics, (
        "this test needs the mutated clip to carry the original metrics; if that "
        "changes, re-derive rather than delete"
    )
    after = next(
        c
        for c in validate_clip(frozen, case.program)
        if c.id == "signal.activity.dead_limb"
    )
    assert after.status != "pass", (
        "freezing every bone left dead_limb passing; the layer is not reading frames"
    )
