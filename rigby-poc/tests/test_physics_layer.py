"""The physics layer against the real corpus. Plan 10 §3.1.

``test_physics_primitives.py`` covers the primitives with synthetic poses. This
file is about what the layer says on 47 real clips, and about the one property
that distinguishes it from most of the existing check surface: it reads frames,
so it is live under mutation.

**Two of §3.1's eight checks ship.** Each deferral was decided by measurement
and the numbers are in plan 10 §3.1 and TRACKING; the shortest form is that
``balance`` has the wrong criterion (static CoM-inside-support fails on walk,
run, turn, kick, dance, hurdle, cartwheel and burpee, and Hof's extrapolated CoM
makes every one of them *worse*), ``ballistic`` and ``momentum`` need a flight
phase that 5 of 46 cases have and which is a ladder climb and a pushup in three
of those, and the two ``gait`` checks need an applicability rule this PR does
not build.
"""

from __future__ import annotations

import dataclasses

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from rigby_poc.analysis import validate_clip
from rigby_poc.analysis.contract import PHYSICS, CheckResult

#: One corpus compile per case in a module fixture; see test_corpus_compile_budget.py.
pytestmark = pytest.mark.medium

PHYSICS_IDS = frozenset({"physics.ground.penetration", "physics.contact.foot_skate"})

#: The clip that compiles to no frames, so every physics check skips.
NO_FRAMES_CASE = "knownbad-eigenvalues-unsupported"

#: Measured on d58b389. Named rather than counted so a change has to be a
#: deliberate edit with a reason, not a silently updated number.
PENETRATING_CASES = frozenset(
    {"fullbody-burpee-cycle", "fullbody-cartwheel", "sequence-wave-then-pushup"}
)
SKATING_CASES = frozenset(
    {
        "fullbody-burpee-cycle",
        "fullbody-cartwheel",
        "fullbody-dance",
        "fullbody-ladder-climb",
        "fullbody-run-forward",
        "fullbody-step-over-hurdle",
        "fullbody-turn-left",
        "fullbody-walk-forward",
    }
)


@pytest.fixture(scope="module")
def verdicts() -> dict[str, list[CheckResult]]:
    collected: dict[str, list[CheckResult]] = {}
    for case in load_corpus():
        clip = compile_case(case)
        collected[case.entry.id] = validate_clip(clip, case.program)
    return collected


def test_the_probe_saw_a_real_corpus(verdicts) -> None:
    assert len(verdicts) >= 47, f"the corpus produced {len(verdicts)} cases"
    assert NO_FRAMES_CASE in verdicts


def test_every_case_emits_every_physics_id(verdicts) -> None:
    """Equality, and on every case including the one that measured nothing.

    A check that emits nothing on an inapplicable clip is indistinguishable from
    a check that does not exist, and the mutation registry counts emissions by
    equality -- so a vanishing id and a skipping id are different facts. This is
    the same rule ``rom_checks`` follows with 156 skips on the zero-frame case.
    """

    for case_id, checks in verdicts.items():
        emitted = {c.id for c in checks if c.id.startswith("physics.")}
        assert emitted == PHYSICS_IDS, case_id


def test_the_physics_layer_declares_its_layer(verdicts) -> None:
    for case_id, checks in verdicts.items():
        for check in checks:
            if check.id.startswith("physics."):
                assert check.layer == PHYSICS, f"{case_id}: {check.id}"


def test_the_zero_frame_clip_skips_rather_than_passing(verdicts) -> None:
    """The absence-becoming-a-value shape, at the layer boundary.

    A clip with no frames has no physics. Reporting a clean pass would put a
    verdict in the output that is present and not derived from anything.
    """

    physics = {c.id: c for c in verdicts[NO_FRAMES_CASE] if c.id.startswith("physics.")}
    assert set(physics) == PHYSICS_IDS
    for check in physics.values():
        assert check.status == "skip", check.id
        assert check.headroom is None, check.id


def test_ground_penetration_fires_on_the_hand_plant_clips_and_only_those(
    verdicts,
) -> None:
    """3 of 47, named. And the interesting half is *which bone*.

    On all three the lowest joint is a **finger** -- ``leftRingDistal`` on the
    cartwheel, ``leftMiddleDistal`` on the burpee and the pushup -- because those
    are the motions that put a hand on the floor. On the other 42 measured cases
    the lowest joint is a toe. A fingertip 167 mm below the floor plane is not a
    hand resting on the ground; it is a hand through it.

    Reported at 3 of 47 rather than scoped away to make the rate nicer. Plan 10
    §10.5 says to expect the first honest numbers to look worse.
    """

    failing = {
        case_id
        for case_id, checks in verdicts.items()
        if any(c.id == "physics.ground.penetration" and c.failed for c in checks)
    }
    assert failing == PENETRATING_CASES


def test_foot_skate_fires_on_the_locomotion_clips(verdicts) -> None:
    """8 of 47, against a bound that had never been able to fail before this.

    ``physics.foot_drift_max_m`` is marked ``UNMEASURED_GATE`` in
    ``thresholds.v1.json`` and its own rationale says why: ``safety_metrics``
    emits the literal 0.0 for ``foot_drift_m`` and the evidence layer compares
    that literal against the bound. This check is its first real producer.
    """

    failing = {
        case_id
        for case_id, checks in verdicts.items()
        if any(c.id == "physics.contact.foot_skate" and c.failed for c in checks)
    }
    assert failing == SKATING_CASES


def test_the_foot_skate_bound_sits_in_a_gap_rather_than_on_the_data(
    verdicts,
) -> None:
    """A margin guard: assert the constant covers the data *and* by how much.

    Measured over the corpus: 35 of 46 measured cases skate **exactly 0.00 mm**,
    the highest passing value is 0.0859 mm and the lowest failing one is
    3.79 mm. The 1 mm bound sits in a 44x gap. If a future clip lands near the
    bound this fails, which is the point -- that would be news about the
    compiler rather than a number to absorb.
    """

    passing, failing = [], []
    for checks in verdicts.values():
        for check in checks:
            if check.id != "physics.contact.foot_skate" or check.status == "skip":
                continue
            (failing if check.failed else passing).append(float(check.measured))
    assert passing and failing, "the corpus did not straddle the bound"
    assert max(passing) * 10 < min(failing), (
        f"the foot-skate bound no longer sits in a wide gap: highest passing "
        f"{max(passing) * 1000:.4f} mm, lowest failing {min(failing) * 1000:.4f} mm"
    )


def test_the_repository_has_exactly_one_definition_of_the_floor() -> None:
    """``AnalysisContext.ground_height`` and the physics layer's must be one thing.

    They were briefly two implementations of the same derivation, held equal by
    intention. Lane `groundtruth` pushed back on that and was right:
    deliberateness is not a mechanism, and a copy that agrees on the day it is
    written is unfalsifiable afterwards. The context now delegates to
    ``physics.ground_height_of``, and this fails if anyone re-inlines it -- the
    same treatment the rig-profile limits copy already gets.

    Lives in this file rather than beside the other primitive tests because it
    needs a real ``AnalysisContext``, which needs a compiled case. Classify by
    input, never by duration.
    """

    from rigby_poc.analysis.context import AnalysisContext
    from rigby_poc.analysis.physics import ground_height

    case = next(iter(load_corpus()))
    clip = compile_case(case)
    context = AnalysisContext.from_clip(clip, case.program, case.scene)
    assert context.ground_height == ground_height()
    assert context.ground_height > 0.0, "the floor collapsed onto the origin"


def test_the_physics_layer_is_live_under_a_frame_mutation(verdicts) -> None:
    """The property that separates this layer from most of the check surface.

    ``MutationSpec.apply`` transforms **frames only**, so a mutated clip carries
    the compiler's pre-mutation ``metrics``. Every metric-derived check therefore
    reads the unmutated clip and returns a confident, stable, wrong answer --
    measured by lane `groundtruth` on four full-body cases. Because ROM is 96%
    of the surface, a fold over it looks like it is responding while the
    remaining checks are frozen.

    This layer reads frames, so it moves. Asserted by lifting the whole skeleton
    a metre into the air on a clip that currently passes: the feet leave the
    ground, and the check that decides contact from toe height must notice.
    """

    case = next(
        (case for case in load_corpus() if case.entry.id == "fullbody-walk-forward"),
        None,
    )
    assert case is not None, "fullbody-walk-forward left the corpus"
    clip = compile_case(case)
    assert clip.frames, "the case compiled to no frames"

    def physics_of(target) -> dict[str, CheckResult]:
        return {
            c.id: c
            for c in validate_clip(target, case.program)
            if c.id.startswith("physics.")
        }

    before = physics_of(clip)

    # The rig carries root translation on `hips.position`; every other bone is
    # rotation-only. Lifting the root is therefore the smallest edit that moves
    # every world position, which is exactly what a frames-derived check must see.
    lifted_frames = []
    for frame in clip.frames:
        hips = frame.bones["hips"]
        assert hips.position is not None, (
            "fullbody-walk-forward stopped carrying a root translation on hips; "
            "this test lifts the root and needs one"
        )
        raised = hips.model_copy(
            update={
                "position": hips.position.model_copy(
                    update={"y": hips.position.y + 1.0}
                )
            }
        )
        lifted_frames.append(
            frame.model_copy(update={"bones": {**frame.bones, "hips": raised}})
        )

    lifted = clip.model_copy(update={"frames": lifted_frames})

    assert lifted.metrics == clip.metrics, (
        "this test is only meaningful while the mutated clip carries the "
        "original metrics; if metrics now move, the stale-source hazard is gone "
        "and this assertion should be re-derived rather than deleted"
    )

    after = physics_of(lifted)
    moved = [
        cid
        for cid in PHYSICS_IDS
        if (before[cid].status, before[cid].headroom)
        != (after[cid].status, after[cid].headroom)
    ]
    assert moved, (
        "lifting the whole skeleton one metre changed no physics verdict; the "
        "layer is not reading frames"
    )
