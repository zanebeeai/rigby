"""The two ways to build a plausible, monotonic, wrong detection curve.

Both defects this file pins produce a curve that rises with severity and reads as
a *conservative* result, which is why neither would be questioned on inspection.
`test_mutation_severity_monotonic` cannot see either of them.

The anti-tautology demonstration for this file is structural rather than a stash:
each of the two central tests constructs the input that the wrong reading scores
differently, and asserts the number the wrong reading would produce is **not**
what comes back. `test_reading_status_would_score_an_unenforced_dof_undetected`
and `test_a_refusal_is_not_a_zero` both fail against a detector that reads
`status`, or that folds a skip into the denominator, respectively.
"""

from __future__ import annotations

import pytest

from evals.calibration.detection import (
    BAND_READ_PREFIX,
    DETECTION_LEVEL,
    STRUCTURAL_READ_PREFIX,
    DetectionError,
    PairOutcome,
    capability_rate,
    clopper_pearson_upper,
    detection_curve,
    detection_threshold,
    skip_ledger,
    target_detected,
)
from evals.mutations.family import MutationFamily, Tier
from evals.mutations.spec import MutationSpec
from evals.mutations.structural import STRUCTURAL_GATES
from rigby_poc.analysis.contract import CheckResult, CheckStatus, skipped

pytestmark = pytest.mark.fast

ROM_TARGET = "anatomy.rom.leftLowerArm.abduction"
STATUS_TARGET = "anatomy.arm.self_collision"
STRUCTURAL_TARGET = "structural.support_foot.planted_target"


def _structural_spec(severity: float = 0.5) -> MutationSpec:
    return MutationSpec(
        id=f"structural@{severity:g}",
        family=MutationFamily.ANATOMY,
        targets=(STRUCTURAL_TARGET,),
        severity=severity,
        tier=Tier.MODERATE,
    )


def _metrics_reporting(*gate_ids: str) -> dict[str, list[str]]:
    """An analysis metrics dict whose failure list trips exactly `gate_ids`.

    Built from the registered prefixes rather than from hand-typed strings, so a
    gate whose message is reworded moves this fixture with it instead of leaving a
    test that passes against a message nothing emits any more.
    """
    return {"structural_failures": [STRUCTURAL_GATES[gate] for gate in gate_ids]}


def _rom_spec(severity: float = 0.5, target: str = ROM_TARGET) -> MutationSpec:
    return MutationSpec(
        id=f"rom@{severity:g}",
        family=MutationFamily.ANATOMY,
        targets=(target,),
        severity=severity,
        tier=Tier.MODERATE,
    )


def _rom_result(band: str, status: CheckStatus = "pass") -> CheckResult:
    return CheckResult(
        id=ROM_TARGET,
        layer="anatomy",
        status=status,
        measured={"bone": "leftLowerArm", "dof": "abduction", "band": band},
        # This file is about the band router, not about ranking, so the headroom
        # is only whatever the contract requires for the status. It is 0.0 on the
        # failing branch because these fixtures carry no severity.
        headroom=0.0 if status != "skip" else None,
    )


def _outcome(
    severity: float,
    *,
    detected: bool | None,
    applicable: bool = True,
    static: bool = False,
    reason: str = "",
    case_id: str = "case",
) -> PairOutcome:
    return PairOutcome(
        case_id=case_id,
        spec_id=f"spec@{severity:g}",
        severity=severity,
        applicable=applicable,
        static_target=static,
        detected=detected,
        reason=reason,
    )


# --- trap 1: band, not status -------------------------------------------------


def test_reading_status_would_score_an_unenforced_dof_undetected() -> None:
    # The whole point. Since 04c only 82 of 156 DOFs are enforced, and rom_checks
    # sets status="fail" only for beyond_max on an *enforced* DOF. An excursion on
    # an unenforced DOF is a real, measured detection that carries status="pass".
    results = {ROM_TARGET: _rom_result("beyond_typical", status="pass")}
    assert target_detected(results, _rom_spec()) is True
    # The number the wrong reading produces, asserted against explicitly.
    assert results[ROM_TARGET].status == "pass"


def test_beyond_max_on_an_enforced_dof_is_detected_too() -> None:
    results = {ROM_TARGET: _rom_result("beyond_max", status="fail")}
    assert target_detected(results, _rom_spec()) is True


def test_within_typical_is_the_only_clean_band() -> None:
    results = {ROM_TARGET: _rom_result("within_typical")}
    assert target_detected(results, _rom_spec()) is False


def test_a_non_rom_target_is_still_read_through_status() -> None:
    # The band rule is scoped to anatomy.rom.*; everything else has a live status
    # and reading `band` there would find nothing.
    spec = _rom_spec(target=STATUS_TARGET)
    failing = CheckResult(
        id=STATUS_TARGET,
        layer="anatomy",
        status="fail",
        measured=1.0,
        severity=0.4,
        headroom=-0.4,
    )
    passing = CheckResult(
        id=STATUS_TARGET, layer="anatomy", status="pass", measured=0.0, headroom=1.0
    )
    assert target_detected({STATUS_TARGET: failing}, spec) is True
    assert target_detected({STATUS_TARGET: passing}, spec) is False


def test_the_shape_rom_checks_really_emits_for_an_unmeasured_bone_raises() -> None:
    # The instrument half: `rom_checks` emits `skipped(...)` for a bone the clip
    # carries no pose for, and `skipped` sets `measured=0.0` -- a float, not a
    # mapping with a missing band. Asserting against a hand-built dict here would
    # pin a shape the code never produces.
    result = skipped(ROM_TARGET, "anatomy", detail="carries no pose in this clip")
    assert result.measured == 0.0
    assert result.status == "skip"
    with pytest.raises(ValueError, match="carries a mapping"):
        target_detected({ROM_TARGET: result}, _rom_spec())


def test_a_rom_mapping_with_no_band_raises_rather_than_reading_as_clean() -> None:
    # The other unmeasured shape: a mapping that carries no band at all. Returning
    # False would fold "not measured" into "measured negative".
    results = {
        ROM_TARGET: CheckResult(
            id=ROM_TARGET,
            layer="anatomy",
            status="skip",
            measured={"bone": "leftLowerArm", "dof": "abduction"},
        )
    }
    with pytest.raises(ValueError, match="not measured"):
        target_detected(results, _rom_spec())


def test_a_skipped_non_rom_target_raises_rather_than_scoring_a_miss() -> None:
    # The fold this module refuses on the ROM path, arriving through `status`.
    # `contract.clip.root_drift` (04e) skips on whole-body and sequence programs,
    # which enable root motion and have no bound to apply. Reading `status !=
    # "fail"` scores that as undetected at every severity -- a manufactured false
    # negative on exactly the clips the check declined to judge.
    spec = _rom_spec(target=STATUS_TARGET)
    skipped_result = skipped(STATUS_TARGET, "anatomy", detail="root motion is enabled")
    assert skipped_result.status == "skip"
    with pytest.raises(DetectionError, match="not measured"):
        target_detected({STATUS_TARGET: skipped_result}, spec)


def test_a_passing_non_rom_target_is_still_a_measured_negative() -> None:
    # The other half: `pass` really is "measured and clean", and must NOT raise.
    spec = _rom_spec(target=STATUS_TARGET)
    passing = CheckResult(
        id=STATUS_TARGET, layer="anatomy", status="pass", measured=0.0, headroom=1.0
    )
    assert target_detected({STATUS_TARGET: passing}, spec) is False


def test_a_target_nothing_emitted_raises_rather_than_returning_false() -> None:
    # A missing target and an undetected mutation produce the same matrix cell.
    with pytest.raises(DetectionError, match="not measured"):
        target_detected({}, _rom_spec())


# --- trap 2: a refusal is not a zero ------------------------------------------


def test_a_refusal_is_not_a_zero() -> None:
    # Two detections and three refusals. Counting the refusals as undetected
    # would report 2/5 = 0.4 and put this level below the 50% detection line;
    # the honest denominator is 2 and the rate is 1.0.
    outcomes = [
        _outcome(0.16, detected=True, case_id="a"),
        _outcome(0.16, detected=True, case_id="b"),
        *[
            _outcome(
                0.16,
                detected=None,
                applicable=False,
                reason="leftLowerArm carries no pose in this clip",
                case_id=name,
            )
            for name in ("c", "d", "e")
        ],
    ]
    (level,) = detection_curve(outcomes)
    assert level.threshold_points.n == 2
    assert level.threshold_points.estimate == 1.0
    assert level.threshold_points.estimate != pytest.approx(0.4)
    assert level.skipped == 3
    assert level.skip_reasons == ("leftLowerArm carries no pose in this clip",)


def test_a_skip_must_carry_a_reason() -> None:
    with pytest.raises(DetectionError, match="cannot tell"):
        _outcome(0.16, detected=None, applicable=False, reason="")


def test_an_applicable_pair_must_record_a_verdict() -> None:
    with pytest.raises(DetectionError, match="must record a verdict"):
        _outcome(0.16, detected=None, applicable=True)


def test_an_inapplicable_pair_may_not_record_one() -> None:
    with pytest.raises(DetectionError, match="carries no verdict"):
        _outcome(0.16, detected=True, applicable=False, reason="declined")


# --- static targets are scored apart, never pooled ----------------------------


def test_static_targets_are_kept_out_of_the_threshold_population() -> None:
    # A static target separates perfectly at every severity, so pooling it drags
    # the curve up and moves the threshold down -- an inflated sweep that looks
    # like a better detector.
    outcomes = [
        _outcome(0.04, detected=False, case_id="moving"),
        *[
            _outcome(0.04, detected=True, static=True, case_id=f"static{index}")
            for index in range(4)
        ],
    ]
    (level,) = detection_curve(outcomes)
    assert level.threshold_points.n == 1
    assert level.threshold_points.estimate == 0.0
    assert level.static_target.n == 4
    assert level.static_target.estimate == 1.0
    # Pooled it would read 4/5 = 0.8 and clear the detection line.
    assert level.threshold_points.estimate < DETECTION_LEVEL


def test_capability_pools_static_targets_across_levels_and_only_those() -> None:
    outcomes = [
        _outcome(0.04, detected=True, static=True),
        _outcome(0.48, detected=False, static=True),
        _outcome(0.48, detected=True),
    ]
    curve = detection_curve(outcomes)
    pooled = capability_rate(curve)
    assert (pooled.successes, pooled.n) == (1, 2)


# --- the threshold itself -----------------------------------------------------


def _level_outcomes(severity: float, detected: int, total: int) -> list[PairOutcome]:
    return [
        _outcome(severity, detected=index < detected, case_id=f"c{index}")
        for index in range(total)
    ]


def test_the_threshold_is_the_mildest_level_reaching_fifty_percent() -> None:
    outcomes = [
        *_level_outcomes(0.04, 0, 20),
        *_level_outcomes(0.16, 4, 20),
        *_level_outcomes(0.48, 18, 20),
        *_level_outcomes(1.00, 20, 20),
    ]
    interval = detection_threshold(detection_curve(outcomes))
    assert interval is not None
    assert interval.estimate == 0.48
    assert interval.n == 20
    assert interval.lower_bound_95 <= interval.estimate <= interval.upper_bound_95


def test_a_sweep_that_never_detects_returns_none_rather_than_a_number() -> None:
    # None is what drives _detection_criterion to FAILED. A large finite number
    # would read as a poor detector that the gate could still be tuned to accept.
    outcomes = [*_level_outcomes(0.04, 0, 20), *_level_outcomes(1.00, 3, 20)]
    assert detection_threshold(detection_curve(outcomes)) is None


def test_an_unbounded_threshold_reports_infinity_not_the_top_level() -> None:
    # Detection is observed but never *established*: no level's lower bound clears
    # 50%. Reporting the top severity would claim a bound the data did not carry.
    outcomes = [*_level_outcomes(0.04, 0, 4), *_level_outcomes(1.00, 2, 4)]
    interval = detection_threshold(detection_curve(outcomes))
    assert interval is not None
    assert interval.estimate == 1.00
    assert interval.upper_bound_95 == float("inf")


def test_a_sweep_with_no_threshold_points_refuses_rather_than_scoring_capability() -> (
    None
):
    outcomes = [_outcome(0.16, detected=True, static=True)]
    with pytest.raises(DetectionError, match="capability and not a threshold"):
        detection_threshold(detection_curve(outcomes))


def test_an_empty_sweep_refuses() -> None:
    with pytest.raises(DetectionError, match="empty sweep"):
        detection_threshold([])


def test_the_skip_ledger_makes_a_mostly_skipped_sweep_visible() -> None:
    outcomes = [
        _outcome(0.04, detected=True, case_id="a"),
        _outcome(
            0.04,
            detected=None,
            applicable=False,
            reason="already beyond_typical",
            case_id="b",
        ),
        _outcome(
            0.48,
            detected=None,
            applicable=False,
            reason="already beyond_typical",
            case_id="b",
        ),
    ]
    assert skip_ledger(detection_curve(outcomes)) == {"already beyond_typical": 2}


# --- the bound helper ---------------------------------------------------------


def test_the_upper_bound_is_the_mirror_of_the_lower() -> None:
    from evals.calibration_stats import clopper_pearson_lower

    assert clopper_pearson_upper(0, 20) == pytest.approx(
        1.0 - clopper_pearson_lower(20, 20)
    )
    assert clopper_pearson_upper(20, 20) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0


def test_the_bounds_bracket_the_estimate() -> None:
    for successes in range(21):
        lower_of_complement = clopper_pearson_upper(successes, 20)
        assert 0.0 <= lower_of_complement <= 1.0
        assert lower_of_complement >= successes / 20


# --- no emitted state reaches the router without a branch ---------------------


def test_the_router_handles_every_member_of_checkstatus() -> None:
    """Derived from the type, not from what I remember the analyzer emitting.

    The `skip`-scored-as-a-miss defect was latent for exactly as long as no
    emitted non-ROM check skipped; a test over hand-written results could not see
    it, and a test over a 4-case corpus sample would only have seen it once
    `contract.clip.root_drift` happened to land in that sample.

    `CheckStatus` is a `Literal`, so the state space is **closed**: every status a
    check can carry is enumerable without compiling anything. This asserts each
    one reaches an explicit branch, so adding a fourth status turns this red
    rather than silently scoring as "not detected" -- which is the direction the
    missing branch always fails in. Lane `analysis` proposed the corpus-walk form
    of this; the type is the same instrument with complete coverage and no
    compiles.
    """
    from typing import get_args

    statuses = set(get_args(CheckStatus))
    assert statuses == {"pass", "fail", "skip"}, (
        f"CheckStatus gained or lost a member: {sorted(statuses)}. Add a branch to "
        f"`target_detected` for it, or it scores as an undetected mutation."
    )

    spec = _rom_spec(target=STATUS_TARGET)
    handled: dict[str, str] = {}
    for status in sorted(statuses):
        result = CheckResult(
            id=STATUS_TARGET,
            layer="anatomy",
            status=status,
            measured=1.0 if status == "fail" else 0.0,
            severity=0.4 if status == "fail" else 0.0,
            headroom={"fail": -0.4, "pass": 1.0, "skip": None}[status],
        )
        try:
            handled[status] = (
                "detected"
                if target_detected({STATUS_TARGET: result}, spec)
                else "clean"
            )
        except DetectionError:
            handled[status] = "raised"

    # Each status maps to a *different* meaning. Two statuses collapsing onto one
    # outcome is how "not measured" became "measured negative" in the first place.
    assert handled == {"fail": "detected", "pass": "clean", "skip": "raised"}, handled
    assert len(set(handled.values())) == len(statuses), (
        f"two statuses share an outcome, so one of them is being folded: {handled}"
    )


def test_a_new_check_family_forces_a_routing_decision() -> None:
    """The status dimension is closed by the type; the *family* dimension is not.

    `test_the_router_handles_every_member_of_checkstatus` proves no status is
    unhandled. It cannot prove a new check *family* is routed correctly, and that
    is the live risk: `anatomy.rom.*` carries its signal in `measured["band"]`
    while `status` stays `pass` on the 74 unenforced DOFs, so a family that
    likewise reports through `measured` would be silently mis-scored by the
    status branch -- the same defect as reading `status` for ROM, arriving
    through a family nobody classified.

    Pinned from `known_check_ids()`, which is itself measured against what the
    analyzer emits (`test_mutation_check_registry`), so this cannot drift into a
    hand-maintained wish list. A new family turns this red and the reviewer has
    to answer one question: does it report through `status`, or through
    `measured`? Red for a family that routes correctly is the intended cost --
    the decision is what is being guarded, not the outcome.
    """
    from evals.mutations.checks import known_check_ids

    families = {".".join(check_id.split(".")[:2]) for check_id in known_check_ids()}
    assert families == {
        "anatomy.arm",
        "anatomy.forearm",
        "anatomy.rom",  # reads `measured["band"]` -- see BAND_READ_PREFIX
        "anatomy.travel_wheel",
        "anatomy.wrist",
        "contract.camera",
        "contract.clip",
        # 10b's physics layer. The routing decision this guard demands, answered:
        # through `status`. Both are ordinary `CheckResult`s with pass/fail/skip and
        # a plain float `measured`, so they take the default branch -- not
        # `measured["band"]` like anatomy.rom.*, whose band exists only because 04c
        # enforces 82 of 156 DOFs and an unenforced excursion must still say
        # `status="pass"`. Physics has no per-DOF enforcement, so `status` means what
        # it says. No new channel and no `target_detected` change.
        "physics.contact",
        "physics.ground",
        # 10c's signal-quality layer. Same routing answer as physics: through
        # `status`. Ordinary CheckResults, plain float `measured`, no band.
        "signal.activity",
        "signal.smoothness",
        "signal.angular",
        "signal.semantic_cycle",
        "signal.travel_wheel",
        # The seven below are 06c's structural-gate namespace, and the answer this
        # guard demanded is **neither** of the two it offered: they emit no
        # `CheckResult`, so they have no `status` and no `measured` to read. They
        # report by appending a string to `metrics["structural_failures"]` and are
        # routed through STRUCTURAL_READ_PREFIX -- a third channel, added here
        # rather than folded onto one of the first two.
        "structural.balance",
        "structural.ground",
        "structural.pushup",
        "structural.recovery_foot",
        "structural.root",
        "structural.semantic",
        "structural.support_foot",
    }, (
        f"check families changed: {sorted(families)}. Decide how the new one "
        f"reports -- through `status`, or through `measured` like anatomy.rom.* "
        f"-- and route it in `target_detected` before updating this set."
    )

    # The band-read set is exactly one family, and it is one of the above.
    assert BAND_READ_PREFIX.rstrip(".") in families

    # Every routing channel is non-empty and they partition the families rather
    # than overlapping. A prefix that matched nothing would leave this guard
    # asserting a set nobody routes, which is the failure mode it exists for.
    band = {name for name in families if f"{name}.".startswith(BAND_READ_PREFIX)}
    structural = {
        name for name in families if f"{name}.".startswith(STRUCTURAL_READ_PREFIX)
    }
    assert band, "the band-read prefix matches no family"
    assert structural, "the structural prefix matches no family"
    assert not band & structural, f"a family is routed two ways: {band & structural}"


def test_a_structural_target_is_read_from_the_failure_list() -> None:
    """The third channel, and the two directions it has to separate.

    These ids are absent from `results` by construction -- they emit no
    `CheckResult` -- so the missing-target branch would raise "was not measured"
    on every one of them if the structural branch did not come first.
    """
    spec = _structural_spec()

    tripped = _metrics_reporting(STRUCTURAL_TARGET)
    assert target_detected({}, spec, metrics=tripped) is True

    # A different gate firing is not this spec's target firing. Detection has to
    # be attributable to the declared target, or a family scores itself detected
    # off its neighbour's gate.
    other = _metrics_reporting("structural.balance.upright")
    assert target_detected({}, spec, metrics=other) is False

    # Evaluated and clean. Distinct from both of the above and from "no key".
    assert target_detected({}, spec, metrics={"structural_failures": []}) is False


def test_a_structural_target_without_metrics_raises_rather_than_scoring_clean() -> None:
    """The whole namespace has a measured base rate of zero on every corpus case.

    So `False` is precisely what a working detector returns, and a caller that
    forgot to pass `metrics` would read as a family whose gates never fire --
    at every severity, monotonically, exactly like a real negative result.
    """
    with pytest.raises(DetectionError, match="no metrics were passed"):
        target_detected({}, _structural_spec())


def test_a_path_that_evaluates_no_structural_gate_raises_rather_than_scoring_clean() -> (
    None
):
    """ "No detector on this path" and "the detectors ran and found nothing".

    A metrics dict with no `structural_failures` key at all means the clip took a
    compile path that evaluates none of these gates. That is a gap to report, not
    a clean measurement, and folding it to `False` is the same not-measured /
    measured-negative conflation the ROM and skip branches each refuse.
    """
    with pytest.raises(DetectionError, match="no detector on this clip's path"):
        target_detected({}, _structural_spec(), metrics={})
