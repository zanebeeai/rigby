"""Plan 10 §3.3 — the deterministic composite, and the four traps it has to clear.

Selection regret and repair efficacy (plan 10 §6) both need a per-clip score
that the judge did not produce. `evals/trajectory.py` ships both with an
injected `oracle` and no default because the producer did not exist. This file
guards the producer, and every test in it corresponds to a way the obvious
implementation is wrong:

1. a fold over severities cannot rank two clips that both pass;
2. a flat mean is 93-97% range-of-motion, so one bone's defect is the score;
3. a clip that measured almost nothing scores top marks on what little it did;
4. the judge's own score makes regret zero by construction.

The corpus numbers quoted here were measured on this branch over all 47 cases.
"""

from __future__ import annotations

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from rigby_poc.analysis import validate_clip
from rigby_poc.analysis.contract import (
    ANATOMY,
    CONTRACT,
    SIGNAL,
    CheckResult,
    binary_check,
    count_check,
    lower_bound_check,
    upper_bound_check,
)
from rigby_poc.analysis.score import (
    MINIMUM_COVERAGE,
    check_family,
    composite_score,
)

#: One corpus compile per case in a module fixture; see test_corpus_compile_budget.py.
pytestmark = pytest.mark.medium

#: The case that compiles to no frames at all, so all 156 ROM checks skip. Named
#: rather than discovered: this file's coverage test is about this exact shape,
#: and it must fail loudly if the case stops being the degenerate one.
NO_FRAMES_CASE = "knownbad-eigenvalues-unsupported"


@pytest.fixture(scope="module")
def scored() -> dict[str, tuple[list[CheckResult], object]]:
    """``case id -> (checks, CompositeScore)`` over the whole corpus."""

    collected = {}
    for case in load_corpus():
        clip = compile_case(case)
        checks = validate_clip(clip, case.program)
        collected[case.entry.id] = (checks, composite_score(checks))
    return collected


def test_the_probe_saw_a_real_corpus(scored) -> None:
    # Every corpus assertion below iterates a collection this file did not
    # construct and passes vacuously on an empty one. docs/testing.md.
    assert len(scored) >= 47, f"the corpus produced {len(scored)} cases"
    assert NO_FRAMES_CASE in scored


# -- trap 1: severity ranks only the failing half ---------------------------------------


def test_a_severity_fold_cannot_rank_two_clips_that_both_pass() -> None:
    """The measurement that rules out the obvious oracle.

    ``CheckResult`` forbids a non-zero severity on anything but a fail, so two
    clips that pass everything are indistinguishable to any function of
    severity -- and selection regret ranks candidates for one prompt, most of
    which pass most checks. Headroom separates them.
    """

    comfortable = upper_bound_check("signal.x.y", SIGNAL, 0.1, 1.0, scale=1.0)
    marginal = upper_bound_check("signal.x.y", SIGNAL, 0.99, 1.0, scale=1.0)

    assert comfortable.severity == marginal.severity == 0.0
    assert comfortable.headroom > marginal.headroom
    assert composite_score([comfortable]).score > composite_score([marginal]).score


def test_a_failures_headroom_is_exactly_its_negated_severity() -> None:
    """One ruler, not two. Pinned because the two halves are computed apart."""

    failing = upper_bound_check("signal.x.y", SIGNAL, 3.0, 1.0, scale=2.0)
    assert failing.failed and failing.severity == pytest.approx(1.0)
    assert failing.headroom == -failing.severity

    with pytest.raises(ValueError, match="exactly -severity"):
        CheckResult(
            id="signal.x.y",
            layer=SIGNAL,
            status="fail",
            measured=3.0,
            severity=0.5,
            headroom=0.5,
        )


def test_a_check_cannot_be_built_without_saying_which_way_is_bad() -> None:
    """Direction is known where the check is built and nowhere else.

    A consumer holding a finished result cannot recover it: ``measured`` may be
    a dict and ``threshold`` may be ``None``. Over the corpus that is 96.2% of
    the verdicts. So the contract refuses a graded result that omits it, rather
    than defaulting it to ``None`` and letting the fold read the check as having
    abstained.
    """

    with pytest.raises(ValueError, match="without a headroom"):
        CheckResult(id="signal.x.y", layer=SIGNAL, status="pass", measured=0.0)

    with pytest.raises(ValueError, match="no headroom"):
        CheckResult(
            id="signal.x.y",
            layer=SIGNAL,
            status="skip",
            measured=0.0,
            headroom=1.0,
        )


def test_every_bound_helper_reports_a_direction() -> None:
    """The helpers are the reason the rule above is cheap to obey."""

    built = [
        upper_bound_check("signal.a.b", SIGNAL, 0.5, 1.0, scale=1.0),
        lower_bound_check("signal.a.c", SIGNAL, 1.5, 1.0, scale=1.0),
        count_check("contract.a.d", CONTRACT, 0),
        binary_check("contract.a.e", CONTRACT, passed=True, measured=1.0),
    ]
    assert all(c.headroom is not None and c.headroom > 0.0 for c in built)
    assert all(c.status == "pass" for c in built)


# -- trap 2: one defective DOF must not be the score ------------------------------------


def test_the_rom_family_influences_the_score_by_its_stratum_not_its_count(
    scored,
) -> None:
    """The measurement that separates this fold from a flat mean.

    156 of ~162 verdicts are ``anatomy.rom.*`` -- **96.3%** of the check
    surface -- so an unweighted mean is that one family wearing a composite's
    name. Measured as an influence coefficient, ``d(score)/d(rom headroom)``:

    ==================  ===============
    flat mean            0.9341 - 0.9811
    balanced (this PR)   0.0833 - 0.5000
    ==================  ===============

    over 46 scoreable cases, with **no overlap**. The assertion is the exact
    structural relation rather than the range: a family's influence is
    ``1 / (layers x families in its layer)`` and has nothing to do with how many
    checks are in it. That is what makes ``leftLowerArm.abduction`` -- a compiler
    defect failing on 46 of 47 cases, and 91.1% of the corpus's severity mass --
    one voice among 156 rather than the score itself.

    An earlier version of this test bounded how far the score moved when the two
    elbow DOFs were dropped, and a flat mean **passed** it: two checks out of 167
    move a flat mean by less than the bound too. A guard that cannot fire for the
    design it exists to reject is ``docs/testing.md``'s central shape, and this
    one was caught by mutating the source rather than by review.
    """

    import dataclasses

    delta = 0.10
    checked = 0
    for case_id, (checks, score) in scored.items():
        if score.score is None:
            continue
        moved = [
            dataclasses.replace(
                check, headroom=max(0.0, min(1.0, check.headroom - delta))
            )
            if (
                check.id.startswith("anatomy.rom.")
                and check.status == "pass"
                and check.headroom is not None
            )
            else check
            for check in checks
        ]
        before = [
            c.headroom
            for c in checks
            if c.id.startswith("anatomy.rom.") and c.headroom is not None
        ]
        after = [
            c.headroom
            for c in moved
            if c.id.startswith("anatomy.rom.") and c.headroom is not None
        ]
        if not before:
            continue
        realised = (sum(before) - sum(after)) / len(before)
        if realised <= 0.0:
            continue

        shifted = composite_score(moved)
        assert shifted.score is not None, case_id
        influence = (score.score - shifted.score) / realised

        families = [
            f
            for layer in score.layers
            if layer.layer == ANATOMY
            for f in layer.families
        ]
        expected = 1.0 / (len(score.layers) * len(families))
        assert influence == pytest.approx(expected, abs=1e-6), (
            f"{case_id}: anatomy.rom moved the score with influence "
            f"{influence:.4f}; its stratum entitles it to {expected:.4f}. A flat "
            "mean over the checks would give it 0.93 or more."
        )
        # And the contrast is not hypothetical: state what the rejected design does.
        flat_members = [c.headroom for c in checks if c.headroom is not None]
        flat_moved = [c.headroom for c in moved if c.headroom is not None]
        flat_influence = (
            sum(flat_members) / len(flat_members) - sum(flat_moved) / len(flat_moved)
        ) / realised
        assert flat_influence > 0.9 > influence, (
            f"{case_id}: the flat mean this fold replaces gives anatomy.rom "
            f"{flat_influence:.4f} influence against this fold's {influence:.4f}"
        )
        checked += 1

    assert checked >= 46, f"only {checked} cases were exercised"


def test_a_check_family_never_spans_two_layers(scored) -> None:
    """The fold's denominators are only well defined if this holds.

    Verified over the corpus rather than assumed: 176 distinct ids fall into 10
    families and each family is single-layer. Note the family prefix is not the
    layer -- ``contract.camera.active_hand_visibility`` declares
    ``layer="signal"`` -- which is why the fold keys the layer level off the
    declared field.
    """

    seen: dict[str, set[str]] = {}
    for checks, _score in scored.values():
        for check in checks:
            seen.setdefault(check_family(check.id), set()).add(check.layer)
    assert seen, "no checks were collected"
    spanning = {f: sorted(v) for f, v in seen.items() if len(v) != 1}
    assert spanning == {}, f"families spanning more than one layer: {spanning}"


# -- trap 3: a clip nothing looked at is not a good clip ---------------------------------


def test_a_clip_that_measured_almost_nothing_is_refused_not_scored(scored) -> None:
    """The regression this file exists for, and it was a real defect here.

    ``knownbad-eigenvalues-unsupported`` compiles to zero frames, so all 156 ROM
    checks skip and only four categorical contract passes remain. The first
    draft of the composite scored it **+1.0000, the best in the corpus**, ahead
    of all 41 known-good cases -- an absence becoming a value, one level below
    the layer check written to catch exactly that.
    """

    _checks, score = scored[NO_FRAMES_CASE]
    assert score.coverage < MINIMUM_COVERAGE
    assert score.score is None, (
        "the zero-frame clip was given a comparable score; it measured "
        f"{score.contributing} of {score.contributing + score.unmeasured} checks"
    )
    assert score.normalised is None
    # The strata it did measure are still reported -- the refusal is of the
    # top-level number, not of the evidence.
    assert score.layers


def test_the_coverage_floor_sits_in_a_gap_rather_than_on_the_data(scored) -> None:
    """A margin guard: assert the constant covers the data *and* by how much.

    Coverage takes exactly three values over the corpus -- 0.0250, 0.99375 and
    1.0 -- so the floor is 20x above the degenerate case and 2x below the
    lowest real one. If a future clip lands near the floor this fails, which is
    the point: that would be news about the corpus, not a number to absorb.
    """

    values = sorted({round(s.coverage, 6) for _checks, s in scored.values()})
    assert values, "no coverage values"
    below = [v for v in values if v < MINIMUM_COVERAGE]
    above = [v for v in values if v >= MINIMUM_COVERAGE]
    assert below and above, f"coverage did not straddle the floor: {values}"
    assert max(below) * 4 < MINIMUM_COVERAGE < min(above) / 1.5, (
        f"the coverage floor {MINIMUM_COVERAGE} no longer sits in a wide gap: "
        f"highest refused {max(below)}, lowest accepted {min(above)}"
    )


def test_an_absent_layer_is_named_rather_than_scored(scored) -> None:
    """Plan 10's physics layer is unbuilt, so this is the ordinary case.

    A mean over an empty collection is the failure ``docs/testing.md`` opens
    with. The layer is left out of the fold and named, so a score cannot rise
    because a layer stopped being emitted.
    """

    for case_id, (_checks, score) in scored.items():
        assert "physics" in score.missing_layers, case_id
        assert all(layer.contributing > 0 for layer in score.layers), case_id


def test_a_clip_with_no_checks_at_all_scores_none_rather_than_zero() -> None:
    """0.0 is a real score meaning "everything sat exactly on its bound"."""

    empty = composite_score([])
    assert empty.score is None
    assert empty.normalised is None
    assert empty.coverage == 0.0
    assert empty.missing_layers == ("contract", "anatomy", "physics", "signal")


# -- trap 4: the oracle must not be the selector ----------------------------------------


def test_the_composite_reads_nothing_the_judge_produced() -> None:
    """Structural, because the failure is silent and looks like success.

    ``flywheel._candidate_score`` reads ``judgment.overall``, the quantity the
    winner is the argmax of, so regret computed with it is identically zero and
    reads as a perfect selector. Walked as a syntax tree rather than grepped:
    the first version of this test searched the source text and tripped over the
    word "judgement" in a comment, which is a guard failing in its own
    vocabulary -- ``docs/testing.md``'s most expensive shape. Names in code are
    the subject; prose is not.
    """

    import ast
    from pathlib import Path

    import rigby_poc.analysis.score as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.Name):
            attributes.add(node.id)

    assert imported, "the module parsed to no imports at all"
    forbidden_modules = {"judge", "flywheel", "llm_graders", "judge_claims"}
    reached = {
        name
        for name in imported
        for part in name.split(".")
        if part in forbidden_modules
    }
    assert reached == set(), (
        f"the composite imports {sorted(reached)}; the oracle must not be "
        "derived from the selector it is used to audit"
    )
    for name in ("judgment", "overall", "_candidate_score"):
        assert name not in attributes, (
            f"the composite reads {name!r}; plan 10 §3.3 exists so that "
            "selection regret is not the judge marking its own work"
        )


def test_the_composite_is_deterministic(scored) -> None:
    """Plan 10 §6 requires it, and a fold over a dict is where order leaks in."""

    for case_id, (checks, score) in scored.items():
        again = composite_score(list(reversed(checks)))
        assert again.score == score.score, case_id
        assert again.coverage == score.coverage, case_id


def test_the_composite_is_not_degenerate_across_the_corpus(scored) -> None:
    """10a's gate inspects predictors; an all-equal *label* is invisible to it.

    Measured: a severity fold takes 12 distinct values over the 47 cases and
    91.1% of its mass is one defect. The composite has to do better than that or
    it is the same instrument with more steps.
    """

    values = [s.score for _c, s in scored.values() if s.score is not None]
    assert len(values) >= 46, f"only {len(values)} cases were scoreable"
    assert len(set(values)) > 12, (
        f"the composite takes only {len(set(values))} distinct values over "
        f"{len(values)} cases; a severity fold takes 12"
    )
    assert min(values) < max(values)


# -- the point of the exercise: the two blocked §6 metrics now have a producer -----------


def test_the_composite_satisfies_the_trajectory_oracle_contract(scored) -> None:
    """Plan 10 §6's selection regret runs on this, which is why §3.3 exists.

    ``evals/trajectory.py`` ships selection regret and repair efficacy with an
    injected ``oracle`` and no default, and names them in ``unmeasured``,
    because §3.3's producer did not exist. This is the end-to-end demonstration
    that it does now: real corpus clips, scored by the composite, driven through
    the real ``selection_regret``.

    Asserted through ``evals.trajectory``'s public functions rather than by
    reimplementing the fold here -- plan 10 §8.1's rule, that a harness which
    reaches past the production path measures the harness.
    """

    from evals.trajectory import selection_regret

    ranked = sorted(
        (
            (case_id, score.normalised)
            for case_id, (_checks, score) in scored.items()
            if score.normalised is not None
        ),
        key=lambda pair: pair[1],
    )
    assert len(ranked) >= 46

    # A trace whose winner is deliberately not the best candidate. Regret must
    # see that; the judge's own score never could, because the winner is its
    # argmax.
    worst_id, worst_score = ranked[0]
    best_id, best_score = ranked[-1]
    by_id = dict(ranked)
    trace = {
        "rounds": [{"candidates": [{"result_id": worst_id}, {"result_id": best_id}]}],
        "winner_result_id": worst_id,
    }
    result = selection_regret(
        trace,
        oracle=lambda clip: by_id[clip["id"]],
        clip_of=lambda rid: {"id": rid},
    )
    assert result.candidates_scored == 2
    assert result.regret == pytest.approx(best_score - worst_score)
    assert result.regret > 0.0
    assert result.winner_was_best is False

    # And it reports no regret when the winner really was the best available.
    trace["winner_result_id"] = best_id
    happy = selection_regret(
        trace,
        oracle=lambda clip: by_id[clip["id"]],
        clip_of=lambda rid: {"id": rid},
    )
    assert happy.regret == pytest.approx(0.0)
    assert happy.winner_was_best is True


def test_an_unscoreable_clip_is_dropped_rather_than_given_a_number(scored) -> None:
    """The contract ``normalised`` returning ``None`` puts on the caller.

    Substituting 0.0 for an unmeasured candidate ranks it worst available and
    1.0 ranks it best; both are claims about a clip nothing looked at. The
    caller drops it in ``clip_of``, which ``selection_regret`` already skips.
    """

    from evals.trajectory import selection_regret

    _checks, unscoreable = scored[NO_FRAMES_CASE]
    assert unscoreable.normalised is None

    scoreable = {
        case_id: score.normalised
        for case_id, (_c, score) in scored.items()
        if score.normalised is not None
    }
    good_id = next(iter(scoreable))
    trace = {
        "rounds": [
            {"candidates": [{"result_id": NO_FRAMES_CASE}, {"result_id": good_id}]}
        ],
        "winner_result_id": good_id,
    }
    result = selection_regret(
        trace,
        oracle=lambda clip: scoreable[clip["id"]],
        clip_of=lambda rid: None if rid == NO_FRAMES_CASE else {"id": rid},
    )
    assert result.candidates_scored == 1, (
        "the unscoreable clip entered the candidate set; a clip nothing measured "
        "must not be ranked against clips that were measured"
    )
    assert result.regret == pytest.approx(0.0)
