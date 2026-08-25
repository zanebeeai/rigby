"""06b's four graded families, and the instrument tests that pin each one.

Plan 06 section 5's cheapest mitigation for the whole section 6.1 class: *every test
that measures something also pins its own instrument*. Four defects across 03b, 04a
and 04b shared one shape -- plausible output, no exception, no NaN, and a curve that
still rises monotonically, just wrongly -- and none was caught by a monotonicity
test. So each family here gets both: the curve, and a separate assertion that the
harness delivered what it said it did.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.anatomy import (
    HINGE_RATIO,
    base_band,
    is_hinge,
    rom_sweep,
    typical_width_deg,
)
from evals.mutations.checks import rom_detected
from evals.mutations.family import MutationFamily
from evals.mutations.clipping import limb_through_torso_sweep
from evals.mutations.inject import bone_dof_series
from evals.mutations.signal import expected_frame_delta_sigma_deg, jitter_sweep
from evals.mutations.timing import freeze_sweep, snap_sweep
from rigby_poc.analysis import analyze
from rigby_poc.models import ClipResult
from rigby_poc.analysis.anatomy.rom import rom_checks
from rigby_poc.analysis.safety import safety_metrics
from rigby_poc.thresholds import value_of

#: Compiles corpus cases.
pytestmark = pytest.mark.medium

#: A full-body case: moves the knee, and reaches the three universal contract checks.
BODY_CASE = "fullbody-dance"
#: A gesture case: the only path that emits the anatomy and signal axes.
GESTURE_CASE = "gesture-fist-right"
#: A gesture case whose arm never passes near the torso ellipse.
UNREACHED_GESTURE_CASE = "gesture-peace-left"


@pytest.fixture(scope="module")
def body_clip():
    return compile_case(load_case(BODY_CASE))


@pytest.fixture(scope="module")
def gesture_case():
    return load_case(GESTURE_CASE)


@pytest.fixture(scope="module")
def gesture_clip(gesture_case):
    return compile_case(gesture_case)


def _analyze(case, clip):
    request = case.compile_request()
    return analyze(clip, request.program, request.scene)


def _collisions(case, clip) -> int:
    """``self_collision_frames``, refusing a clip that never emitted it.

    ``.get(...) or 0`` here would turn "this path emits no anatomy metrics" into a
    measured zero, which is the not-measured / measured-negative conflation this
    push has six instances of. Only the 14 gesture-path cases emit it, so a caller
    reaching this on one of the other 33 has a bug rather than a clean clip.
    """
    value = _analyze(case, clip).get("self_collision_frames")
    if value is None:
        raise AssertionError(
            "this case emits no self_collision_frames at all, so it has no detector "
            "here; that is a gap to report, not a zero to score"
        )
    return int(value)


# --------------------------------------------------------------------- anatomy


def test_the_knee_is_a_moving_in_band_target(body_clip) -> None:
    """Instrument test. Every anatomy assertion below is void without this."""
    series = bone_dof_series(body_clip, "rightLowerLeg", "flexion")
    assert series, "the knee carries no pose, so nothing below measures a mutation"
    assert np.std(series) > 1e-6, "the knee is static here; this is a capability case"
    assert base_band(body_clip, "rightLowerLeg", "flexion") == "within_typical"


def test_the_sweep_top_is_the_authored_typical_width() -> None:
    """Instrument test: the magnitude is derived from the config, not authored."""
    specs = rom_sweep("rightLowerLeg", "flexion")
    assert specs[-1].params["magnitude_deg"] == pytest.approx(
        typical_width_deg("rightLowerLeg", "flexion")
    )
    assert specs[-1].params["reported_unit"] == "deg"


def test_a_rom_sweep_leaves_the_band_monotonically(body_clip) -> None:
    detected = []
    for spec in rom_sweep("rightLowerLeg", "flexion"):
        assert spec.applies_to(body_clip)
        mutated = spec.apply(body_clip)
        result = {
            check.id: check for check in rom_checks(mutated.frames, fps=30.0)
        }[spec.targets[0]]
        detected.append(rom_detected(result.measured))
    # Monotone: once out of band, never back in. A sweep that re-enters the band is
    # the shape of a wrapped or cancelling injection.
    assert detected == sorted(detected)
    assert detected[0] is False, "the sweep floor must be sub-perceptual"
    assert detected[-1] is True, "the sweep top must be a violation by construction"


def test_a_dof_already_out_of_band_is_refused_rather_than_scored(body_clip) -> None:
    # The state 06a did not have. Scoring this pair would record a perfect detection
    # at the sub-perceptual floor, and the number would be a property of the bound.
    assert base_band(body_clip, "rightLowerArm", "abduction") != "within_typical"
    verdict = rom_sweep("rightLowerArm", "abduction")[0].applies_to(body_clip)
    assert not verdict
    assert "already beyond" in verdict.reason


def test_hinges_are_the_hinges_and_nothing_else() -> None:
    assert is_hinge("rightLowerLeg") and is_hinge("rightLowerArm")
    # The wrist is condyloid and the shoulder is a ball joint. A rule that answers
    # "everything is a hinge" or "nothing is" would pass a smoke test and mean
    # nothing, so both directions are pinned.
    assert not is_hinge("rightHand")
    assert not is_hinge("rightUpperArm")
    assert typical_width_deg("rightLowerLeg", "flexion") / typical_width_deg(
        "rightLowerLeg", "abduction"
    ) >= HINGE_RATIO


# ---------------------------------------------------------------------- timing


def test_a_snap_delivers_between_theta_and_theta_plus_the_base_peak(body_clip) -> None:
    """Instrument test for the snap injector.

    The 04b defect delivered 43.2 degrees for a 30-degree request. That breaches the
    upper bound here at every level, which a monotonicity test would not notice.
    """
    base_peak = math.degrees(safety_metrics(body_clip.frames)["max_frame_rotation_delta_rad"])
    for spec in snap_sweep():
        requested = spec.params["magnitude_deg"]
        mutated = spec.apply(body_clip)
        delivered = math.degrees(
            safety_metrics(mutated.frames)["max_frame_rotation_delta_rad"]
        )
        assert delivered >= min(requested, base_peak) - 1e-9
        assert delivered <= requested + base_peak + 1e-9


def test_a_snap_crosses_the_stated_discontinuity_threshold_once(body_clip) -> None:
    threshold_deg = math.degrees(value_of("signal.discontinuity_rad"))
    fired = [
        safety_metrics(spec.apply(body_clip).frames)["discontinuities"] > 0
        for spec in snap_sweep()
    ]
    assert fired == sorted(fired), "detection must not switch back off as severity rises"
    assert any(fired) and not all(fired), "the sweep must straddle the threshold"
    # The crossing lands in the one sweep interval containing the stated threshold,
    # which is what makes this a calibration against a known bound rather than a
    # discovery of an unknown one.
    crossed_at = next(
        spec.params["magnitude_deg"] for spec, hit in zip(snap_sweep(), fired) if hit
    )
    below = max(
        spec.params["magnitude_deg"]
        for spec, hit in zip(snap_sweep(), fired)
        if not hit
    )
    assert below < threshold_deg < crossed_at


def test_a_freeze_publishes_a_whole_number_of_frames() -> None:
    # A count is published as a count: the transform rounds, so a fractional
    # published magnitude would disagree with the mutation actually applied.
    for spec in freeze_sweep():
        assert isinstance(spec.params["frames"], int)
        assert spec.params["magnitude_unit"] == "frames"


def test_a_freeze_holds_the_pose_and_keeps_the_frame_count(body_clip) -> None:
    """Instrument test: the mutation must not shorten the clip."""
    spec = freeze_sweep()[-1]
    mutated = spec.apply(body_clip)
    assert len(mutated.frames) == len(body_clip.frames)
    series = bone_dof_series(mutated, "rightLowerArm", "flexion")
    held = max(
        len(list(group))
        for group in _runs(round(value, 9) for value in series)
    )
    assert held >= spec.params["frames"], "the held run is shorter than requested"


def _runs(values):
    current: list[float] = []
    previous = object()
    for value in values:
        if value != previous:
            if current:
                yield current
            current = []
            previous = value
        current.append(value)
    if current:
        yield current


# ---------------------------------------------------------------------- signal


def test_jitter_delivers_the_standard_deviation_it_declares(body_clip) -> None:
    """Instrument test: the magnitude is a per-frame standard deviation.

    Asserted against the closed form rather than against "it got bigger": two
    independent draws of standard deviation s differ with standard deviation
    s*sqrt(2), and a jitter that does not scale that way is not adding independent
    per-frame noise however monotone its curve is.
    """
    base = np.degrees(bone_dof_series(body_clip, "rightLowerArm", "flexion"))
    for spec in jitter_sweep():
        requested = spec.params["magnitude_deg"]
        noise = np.degrees(
            bone_dof_series(spec.apply(body_clip), "rightLowerArm", "flexion")
        ) - base
        assert float(np.std(noise)) == pytest.approx(requested, rel=0.15)
        assert float(np.std(np.diff(noise))) == pytest.approx(
            expected_frame_delta_sigma_deg(requested), rel=0.15
        )


def test_jitter_is_byte_identical_for_one_seed(body_clip) -> None:
    spec = jitter_sweep()[3]
    first, second = spec.apply(body_clip), spec.apply(body_clip)
    assert [
        pose.rotation.as_list() for frame in first.frames for pose in frame.bones.values()
    ] == [
        pose.rotation.as_list() for frame in second.frames for pose in frame.bones.values()
    ]


def test_jitter_raises_the_discontinuity_count_monotonically(body_clip) -> None:
    counts = [
        safety_metrics(spec.apply(body_clip).frames)["discontinuities"]
        for spec in jitter_sweep()
    ]
    assert counts == sorted(counts)
    assert counts[0] == 0 and counts[-1] > 0


# -------------------------------------------------------------------- clipping


def test_the_self_collision_gate_can_be_tripped_at_all(gesture_case, gesture_clip) -> None:
    """The finding this family exists to establish.

    TRACKING records zero self-collisions over 517 swept compiles and zero over 20
    ordinary ones. That leaves two possibilities with different responses -- a dead
    gate to delete, or a live gate guarding a region generated motion never reaches.
    This is the second, and it is the assertion that says so.
    """
    assert _collisions(gesture_case, gesture_clip) == 0
    worst = limb_through_torso_sweep()[-1]
    assert _collisions(gesture_case, worst.apply(gesture_clip)) > 0


def test_the_clipping_sweep_is_monotone(gesture_case, gesture_clip) -> None:
    counts = [
        _collisions(gesture_case, spec.apply(gesture_clip))
        for spec in limb_through_torso_sweep()
    ]
    assert counts == sorted(counts)
    assert counts[0] == 0, "the sweep floor must not already be a collision"


def test_a_case_the_mutation_cannot_reach_is_a_zero_not_a_miss() -> None:
    """Per-(spec, case) applicability, plan 06 section 6.1.

    Two of five gesture cases never collide at any severity, because their arm never
    passes near the torso ellipse. That is "this mutation cannot reach this check on
    this case", not a detection failure, and the matrix must render the two apart.
    """
    case = load_case(UNREACHED_GESTURE_CASE)
    clip = compile_case(case)
    counts = [
        _collisions(case, spec.apply(clip)) for spec in limb_through_torso_sweep()
    ]
    assert set(counts) == {0}


# -------------------------------------------------------------------- contract

# Plan 06 section 5's `test_mutation_preserves_contract` applies to every family, not
# only to the ported legacy specs. `tests/test_mutation_contract.py` covers
# `legacy_specs()` and stops there, so 06b's four families would otherwise ship
# without ever being checked for the thing that makes a mutated clip usable
# downstream: capture and export must fail on the mutation's *content*, never on its
# shape.


def _body_families():
    """Every 06b spec applicable to the full-body case, labelled by family."""
    return [
        *rom_sweep("rightLowerLeg", "flexion"),
        *snap_sweep(),
        *freeze_sweep(),
        *jitter_sweep(),
    ]


def _gesture_families():
    """Every 06b spec applicable to the gesture case."""
    return [
        *rom_sweep("rightIndexProximal", "flexion"),
        *limb_through_torso_sweep(),
        *snap_sweep(),
    ]


@pytest.fixture(scope="module")
def body_mutants(body_clip):
    specs = _body_families()
    assert specs, "no specs reached the contract checks"
    return [(spec, spec.apply(body_clip)) for spec in specs]


@pytest.fixture(scope="module")
def gesture_mutants(gesture_clip):
    specs = _gesture_families()
    assert specs, "no specs reached the contract checks"
    return [(spec, spec.apply(gesture_clip)) for spec in specs]


def test_the_contract_fixtures_cover_all_four_families(body_mutants, gesture_mutants) -> None:
    """Instrument test: an empty or single-family collection passes everything below.

    Four judge guards passed vacuously on an empty collection this push. The rule
    that separates a real case from noise is to assert non-emptiness for a collection
    the test did not construct -- and here, to assert the collection is actually the
    four families it claims to be.
    """
    families = {spec.family for spec, _ in [*body_mutants, *gesture_mutants]}
    assert families == {
        MutationFamily.ANATOMY,
        MutationFamily.TIMING,
        MutationFamily.SIGNAL,
        MutationFamily.CLIPPING,
    }
    assert len(body_mutants) + len(gesture_mutants) >= 40


@pytest.mark.parametrize("fixture", ["body_mutants", "gesture_mutants"])
def test_a_mutated_clip_is_still_schema_valid(fixture, request) -> None:
    for spec, mutated in request.getfixturevalue(fixture):
        # `model_validate` raises on an invalid payload, so the round trip is the
        # assertion. Named in the failure via `pytest.fail` rather than a bare
        # expression, which would discard the label silently.
        try:
            ClipResult.model_validate(mutated.model_dump(mode="json"))
        except Exception as error:  # noqa: BLE001 - re-raised with the spec id
            pytest.fail(f"{spec.id} produced a clip that no longer validates: {error}")


@pytest.mark.parametrize("fixture", ["body_mutants", "gesture_mutants"])
def test_a_mutated_clip_stays_finite_and_normalised(fixture, request) -> None:
    for spec, mutated in request.getfixturevalue(fixture):
        for frame in mutated.frames:
            for name, pose in frame.bones.items():
                values = pose.rotation.as_list()
                assert all(math.isfinite(v) for v in values), f"{spec.id}/{name}"
                norm = math.sqrt(sum(v * v for v in values))
                # Bit-tight rather than approximate: a renormalisation bug that
                # drifts a quaternion by 1e-6 per frame is invisible at a loose
                # tolerance and visible in the render.
                assert norm == pytest.approx(1.0, abs=1e-9), f"{spec.id}/{name}"


@pytest.mark.parametrize("fixture", ["body_mutants", "gesture_mutants"])
def test_a_mutated_clip_keeps_its_frame_count_and_timebase(fixture, request, body_clip, gesture_clip) -> None:
    base = body_clip if fixture == "body_mutants" else gesture_clip
    for spec, mutated in request.getfixturevalue(fixture):
        assert len(mutated.frames) == len(base.frames), spec.id
        assert [f.time_s for f in mutated.frames] == [f.time_s for f in base.frames], spec.id


@pytest.mark.parametrize("fixture", ["body_mutants", "gesture_mutants"])
def test_a_mutated_clip_does_not_announce_that_it_is_mutated(
    fixture, request, body_clip, gesture_clip
) -> None:
    """Contract rule 1, the one `evals/corruptions.py` breaks three ways.

    A clip that labels itself lets a scorer read the label instead of the grader's
    verdict, which is how every rate the legacy suite produced stopped being a
    function of the judge at all (plan 06 section 6.5).

    The first draft of this test asserted `not getattr(mutated, "failures", None)`.
    `ClipResult` has no `failures` field -- it is `failure`, singular -- so that
    assertion read a default of `None` on every clip and passed unconditionally. It
    is the bug class this push has catalogued six times, written into a guard against
    it, and it is why the checks below compare against the *base clip* rather than
    against a literal: a comparison to a value the base actually carries cannot be
    satisfied by an attribute that does not exist.
    """
    base = body_clip if fixture == "body_mutants" else gesture_clip
    for spec, mutated in request.getfixturevalue(fixture):
        assert mutated.success == base.success, spec.id
        assert mutated.failure == base.failure, spec.id
        assert "deliberate_corruption" not in mutated.metrics, spec.id
        blob = mutated.model_dump_json()
        assert spec.id not in blob, f"{spec.id} leaks its own id into the clip"
        assert "corruption" not in blob.lower(), f"{spec.id} leaks the word corruption"


def test_applying_a_mutation_does_not_mutate_its_input(body_clip) -> None:
    before = [
        pose.rotation.as_list() for frame in body_clip.frames for pose in frame.bones.values()
    ]
    for spec in _body_families():
        spec.apply(body_clip)
    after = [
        pose.rotation.as_list() for frame in body_clip.frames for pose in frame.bones.values()
    ]
    assert before == after
