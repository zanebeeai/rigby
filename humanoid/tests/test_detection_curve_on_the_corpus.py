"""The detection curve against inputs this lane did not choose.

`test_detection_curve.py` pins the arithmetic on constructed inputs. That is the
half a wrong constant survives: docs/testing.md's "second dataset" rule says an
assumption true when it was written is caught by running against data the author
does not own, and the corpus is that data here.

It earns its cost immediately. The first sweep this module was pointed at —
`rom_sweep("rightLowerArm", "abduction")`, chosen because the elbow is the bone
04c's bound is about — produced **zero** threshold points on every case, because
`rom_guard` correctly refuses a clip whose target DOF is already `beyond_max`
before the mutation. Before the humeral-roll fix that removed the arm entirely:
46 of 47 cases were beyond_max unmutated. After the fix the arm has a corpus-wide
denominator again (37 of 47 cases are within_typical on rightLowerArm.abduction),
but on this file's four-case sample the one admitted case has a *static* elbow,
so the sweep still yields no threshold points here — pinned below rather than
worked around, so a successor picking that DOF meets the fact as a test rather
than as an empty curve.
"""

from __future__ import annotations

import pytest

from evals.calibration.detection import (
    DetectionError,
    detection_curve,
    detection_threshold,
    scored_clip,
    skip_ledger,
    sweep_outcomes,
    target_detected,
    unmutated_baseline,
)
from evals.mutations.anatomy import rom_sweep
from evals.mutations.family import MutationFamily, Tier
from evals.mutations.inject import add_dof
from evals.mutations.spec import MutationSpec
from evals.mutations.structural import gates_tripped

#: Compiles corpus cases.
pytestmark = pytest.mark.medium

#: Full-body cases move the most bones, so they carry threshold points rather
#: than static targets. Four is enough to exercise every branch and keeps the
#: compile cost of this file bounded.
CASE_IDS = (
    "fullbody-burpee-cycle",
    "fullbody-dance",
    "fullbody-kick-right",
    "fullbody-run-forward",
)


@pytest.fixture(scope="module")
def compiled(compile_corpus_case, corpus_by_id) -> list[tuple[str, object, object, object]]:
    missing = [case_id for case_id in CASE_IDS if case_id not in corpus_by_id]
    assert not missing, f"corpus no longer carries {missing}"
    return [
        (
            case_id,
            compile_corpus_case(case_id),
            corpus_by_id[case_id].program,
            corpus_by_id[case_id].scene,
        )
        for case_id in CASE_IDS
    ]


def test_the_probe_saw_real_compiled_clips(compiled) -> None:
    # The instrument test. Every assertion below is over a collection this file
    # did not construct, and an empty one would turn each of them green.
    assert len(compiled) == len(CASE_IDS)
    for case_id, clip, _program, _scene in compiled:
        assert clip.frames, f"{case_id} compiled to no frames"


def test_a_moving_in_band_dof_produces_a_real_curve_and_a_threshold(compiled) -> None:
    outcomes = sweep_outcomes(rom_sweep("rightUpperArm", "flexion"), compiled)
    curve = detection_curve(outcomes)
    assert curve, "the sweep produced no levels"

    measured = [level for level in curve if level.threshold_points.n > 0]
    assert measured, "no level carried a single threshold point"

    # Detection is monotone in severity for a deterministic band check -- an
    # excursion that leaves the band at severity s leaves it at every s' > s --
    # *until the injection wraps*. rightUpperArm.flexion's typical width is 240
    # degrees, so the severity-1.0 level injects a full 240: measured after the
    # humeral-roll fix, fullbody-run-forward's base extremum is -28.28 degrees,
    # the signed -240 lands at -268, wraps past -180 to +92, re-enters the band
    # and scores undetected (2/3 at 1.0 against 3/3 at 0.68). So monotonicity is
    # asserted over the non-wrapping levels, and the wrap itself is pinned.
    no_wrap = [level for level in measured if level.severity <= 0.68]
    rates = [level.threshold_points.estimate for level in no_wrap]
    assert rates == sorted(rates), f"detection fell as severity rose: {rates}"
    assert rates[-1] == 1.0, rates
    top = measured[-1]
    assert top.severity == 1.0
    assert top.threshold_points.estimate < 1.0, (
        "the severity-1.0 level no longer wraps; re-measure and restore the "
        "whole-curve monotonicity assertion"
    )

    # The bottom of the sweep must be plausibly sub-perceptual (plan 06 §3.2). A
    # sweep detecting everything at its mildest level measures nothing.
    assert measured[0].threshold_points.estimate < 1.0

    interval = detection_threshold(curve)
    assert interval is not None, "a curve that reaches 100% must locate a threshold"
    assert interval.lower_bound_95 <= interval.estimate <= interval.upper_bound_95
    assert interval.n == measured[0].threshold_points.n


def test_static_targets_never_enter_the_threshold_population(compiled) -> None:
    outcomes = sweep_outcomes(rom_sweep("rightHand", "flexion"), compiled)
    curve = detection_curve(outcomes)
    statics = [level for level in curve if level.static_target.n > 0]
    assert statics, "this DOF no longer has a static-target case; pick another"
    for level in curve:
        assert (
            level.threshold_points.n + level.static_target.n + level.skipped
            == len(CASE_IDS)
        ), "a pair went missing between the three populations"


def test_the_elbow_arm_still_has_no_threshold_points_on_this_sample(compiled) -> None:
    # The finding this file was written to carry has moved. Before the
    # humeral-roll fix, left/rightLowerArm.abduction was `beyond_max` on 46 of
    # the 47 corpus cases unmutated, so rom_guard refused every pair and the arm
    # vanished. After the fix the arm is un-vanished corpus-wide -- 37 of 47
    # cases sit within_typical on rightLowerArm.abduction -- but on this file's
    # four cases the measured split is: burpee, dance and run-forward are still
    # beyond_max (refused), and kick-right is admitted with a *static* elbow.
    # So every admitted pair lands in the static-target population and the
    # threshold population is still empty here.
    outcomes = sweep_outcomes(rom_sweep("rightLowerArm", "abduction"), compiled)
    curve = detection_curve(outcomes)

    assert outcomes, "the sweep produced no outcomes at all"
    assert all(level.threshold_points.n == 0 for level in curve)
    assert all(level.static_target.n == 1 for level in curve)
    assert all(level.skipped == len(CASE_IDS) - 1 for level in curve)

    ledger = skip_ledger(curve)
    assert ledger, "pairs were skipped but no reason was recorded"
    assert any("beyond_max" in reason for reason in ledger)

    # And it refuses rather than reporting a threshold over statics alone.
    with pytest.raises(DetectionError, match="capability and not a threshold"):
        detection_threshold(curve)


def test_every_skip_carries_a_reason_on_real_data(compiled) -> None:
    outcomes = sweep_outcomes(rom_sweep("rightLowerArm", "abduction"), compiled)
    skips = [item for item in outcomes if not item.applicable]
    assert skips, "expected refusals on this DOF and got none"
    assert all(item.reason for item in skips)
    assert all(item.detected is None for item in skips)


def test_the_rom_baseline_is_zero_because_the_guard_excludes_already_failing_clips(
    compiled,
) -> None:
    # `rom_guard` refuses a clip whose target DOF is already out of band, so for
    # this family the unmutated detection rate over *applicable* cases must be
    # zero. Measured rather than asserted from the guard's source: the point of a
    # baseline is that it is a measurement, and a family whose guard did not
    # exclude such clips would show a non-zero number here instead of silently
    # inflating its curve.
    specs = rom_sweep("rightUpperArm", "flexion")
    baseline = unmutated_baseline(specs, compiled)
    assert baseline.n > 0, "no applicable case, so the baseline measured nothing"
    assert baseline.successes == 0
    assert baseline.estimate == 0.0


def test_the_baseline_is_measured_once_per_case_not_once_per_level(compiled) -> None:
    # A per-level baseline would restate one measurement seven times and inflate
    # its n sevenfold, which is a denominator no measurement had.
    specs = rom_sweep("rightUpperArm", "flexion")
    baseline = unmutated_baseline(specs, compiled)
    assert baseline.n <= len(CASE_IDS)
    assert len(specs) == 7


def test_a_permissive_guard_shows_the_baseline_really_measures(compiled) -> None:
    # The anti-tautology test for the baseline. Every ROM baseline is zero
    # *because `rom_guard` excludes the clips that would make it non-zero*, so a
    # `unmutated_baseline` that simply returned zero would pass every other
    # assertion in this file.
    #
    # Same target, guard removed. After the humeral-roll fix,
    # rightLowerArm.abduction is `beyond_max` unmutated on three of this file's
    # four cases (burpee, dance, run-forward) and clean on kick-right, so with
    # nothing excluding them the unmutated detection rate must be exactly 3/4
    # -- and a hardcoded zero, or a function reading the mutated clip, cannot
    # produce that. (Before the fix it was 4/4; the un-vanishing of the clean
    # case is the fix showing up in this baseline.)
    from dataclasses import replace

    template = rom_sweep("rightLowerArm", "abduction")[0]
    permissive = replace(template, guard=None)
    baseline = unmutated_baseline([permissive], compiled)

    assert baseline.n == len(CASE_IDS), "the permissive guard should admit every case"
    assert baseline.successes == 3
    assert baseline.estimate == pytest.approx(0.75)

    # And this is exactly what rom_guard exists to keep out of a curve: with the
    # real guard the same sweep admits only the clean case (kick-right, whose
    # static elbow contributes no detection), so the baseline over admitted
    # cases stays zero rather than scoring a perfect detection of the bound.
    guarded = unmutated_baseline(rom_sweep("rightLowerArm", "abduction"), compiled)
    assert guarded.n == 1
    assert guarded.successes == 0


def test_the_structural_channel_reads_the_analysed_clip_not_the_compilers_metrics(
    compiled,
) -> None:
    """The defect the structural routing exists to avoid, shown on real clips.

    `MutationSpec.apply` transforms `frames` and nothing in `evals.mutations`
    touches `metrics`, so a mutated clip carries the **compiler's pre-mutation**
    `structural_failures` list. Every gate in the namespace has a measured base
    rate of zero, so that stale list is empty on a clean corpus case and stays
    empty after a mutation that trips six gates.

    Scoring from it would therefore return `False` at every severity for the whole
    contact and balance surface -- not an error, not an empty curve, but a smooth
    zero-detection result indistinguishable from a namespace of dead gates. This
    file exists for defects that read as conservative results, and that is one.

    Reuses the module fixture, so it adds no compile to this file's budget.
    """
    moved: list[str] = []
    for case_id, clip, program, scene in compiled:
        if "support_constraints" not in clip.metrics:
            continue
        base = scored_clip(clip, program, scene)
        if gates_tripped(base.metrics):
            continue  # only attribute against a clean base

        mutated = add_dof(clip, "rightLowerLeg", "flexion", 40.0)
        analysed = gates_tripped(scored_clip(mutated, program, scene).metrics)
        stale = gates_tripped(mutated.metrics)
        if not analysed:
            continue
        moved.append(case_id)

        assert not stale, (
            f"{case_id}: the compiler-written metrics now report {sorted(stale)}, so "
            f"this test can no longer tell the two sources apart -- re-derive the "
            f"claim rather than deleting the assertion"
        )
        assert analysed - stale == analysed

        # And the routing reads the one that moved.
        spec = MutationSpec(
            id=f"structural-probe/{case_id}",
            family=MutationFamily.ANATOMY,
            targets=(sorted(analysed)[0],),
            severity=0.5,
            tier=Tier.MODERATE,
        )
        scored = scored_clip(mutated, program, scene)
        assert target_detected(scored.results, spec, metrics=scored.metrics) is True
        assert target_detected({}, spec, metrics=mutated.metrics) is False

    assert moved, (
        "no compiled case tripped a structural gate under the probe injection, so "
        "every assertion above ran zero times"
    )
