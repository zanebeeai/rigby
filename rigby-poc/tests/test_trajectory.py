"""Plan 10 section 6 — trajectory evals.

Two tests here carry the PR and the rest support them.

`test_the_declared_stage_list_matches_the_literals_in_flywheel` parses `flywheel.py` and
compares the stages it actually emits against `trajectory.PIPELINE_STAGES`, by **equality**.
The existing `test_the_seven_declared_stages_are_all_recorded` asserts a hand-listed set with
`issubset` against a fixture that emits six of seven, so it is a restatement of its fixture
and cannot fail. This one fails when a stage is added, renamed, or dropped.

`test_a_repair_that_only_jitters_is_not_significant` is the control plan 10 section 6 calls
mandatory. A repair loop that perturbs parameters at random produces a positive mean delta
and looks effective; only the matched random control separates that from a real choice.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

import pytest

from evals.trajectory import (
    PIPELINE_STAGES,
    RepairOutcome,
    cost_report,
    diversity_relevance,
    escalation_report,
    perturbation_of,
    rejection_attribution,
    repair_efficacy,
    selection_regret,
    stage_attribution,
    trajectory_report,
)
from rigby_poc.transcript import Transcript, load


pytestmark = pytest.mark.fast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLYWHEEL = PROJECT_ROOT / "evals" / "flywheel.py"


def _emitted_stage_literals() -> set[str]:
    """Every `stage` argument passed to `_progress(callback, event, stage, message, ...)`."""
    tree = ast.parse(FLYWHEEL.read_text(encoding="utf-8"))
    stages: set[str] = set()
    dynamic = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_progress" or len(node.args) < 3:
            continue
        stage = node.args[2]
        if isinstance(stage, ast.Constant) and isinstance(stage.value, str):
            stages.add(stage.value)
        else:
            dynamic += 1
    assert dynamic == 0, (
        f"{dynamic} _progress call sites pass a non-literal stage; this scan would miss "
        f"them and the equality assertion below would be vacuously narrow"
    )
    assert stages, "no stage literals found; the scan is broken, not the pipeline"
    return stages


def test_the_declared_stage_list_matches_the_literals_in_flywheel() -> None:
    """Equality, not subset. A subset assertion cannot detect a stage that goes missing."""
    assert _emitted_stage_literals() == set(PIPELINE_STAGES)


def test_repair_is_one_of_the_declared_stages() -> None:
    """Pinned separately because it is the one no existing test causes to run.

    `tests/test_run_transcript.py::_cli_run` passes `max_rounds=1` and repair happens
    between rounds, so `repair` is emitted by no test in the suite. Plan 10 section 6 needs
    it for both stage attribution and repair efficacy, so its absence would be silent
    exactly where it matters most.
    """

    assert "repair" in _emitted_stage_literals()
    assert "repair" in PIPELINE_STAGES


# -- transcript-backed metrics ----------------------------------------------------------


def _span(**overrides: Any) -> dict[str, Any]:
    record = {
        "schema_version": "1.0",
        "span_id": overrides.pop("span_id", "01AAA"),
        "parent_id": None,
        "run_id": "run-1",
        "kind": "stage",
        "name": "planning",
        "status": "ok",
        "duration_ms": 10.0,
    }
    record.update(overrides)
    return record


def _transcript(records: list[dict[str, Any]], tmp_path: Path) -> Transcript:
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return load(path.parent)


def test_a_stage_that_never_ran_reports_missing_not_zero(tmp_path: Path) -> None:
    """The distinction the whole dataclass exists for."""
    document = _transcript(
        [
            _span(span_id="01ROOT", kind="run", name="pipeline.run", duration_ms=100.0),
            _span(span_id="01A", parent_id="01ROOT", name="planning", duration_ms=40.0),
        ],
        tmp_path,
    )
    attribution = stage_attribution(document)
    assert attribution.durations_ms["planning"] == 40.0
    assert "repair" in attribution.missing
    assert "repair" not in attribution.durations_ms
    assert attribution.occurrences["planning"] == 1


def test_cost_reports_no_per_clip_figure_when_nothing_was_accepted(tmp_path: Path) -> None:
    document = _transcript(
        [_span(span_id="01ROOT", kind="run", name="pipeline.run", duration_ms=100.0)],
        tmp_path,
    )
    assert cost_report(document, accepted_clips=0).tokens_per_accepted_clip is None


def test_usage_coverage_reports_the_share_of_attempts_that_carried_usage(tmp_path: Path) -> None:
    """Plan 01 §7 item 4: a dispatch that raised has no usage, so a token total is a floor."""
    document = _transcript(
        [
            _span(span_id="01ROOT", kind="run", name="pipeline.run", duration_ms=100.0),
            _span(
                span_id="01C",
                parent_id="01ROOT",
                kind="model_call",
                name="judge.call",
            ),
            _span(
                span_id="01A1",
                parent_id="01C",
                kind="attempt",
                name="judge.dispatch",
                attrs={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
                refs={"model": "primary"},
            ),
            _span(
                span_id="01A2",
                parent_id="01C",
                kind="attempt",
                name="judge.dispatch",
                status="error",
                refs={"model": "fallback"},
            ),
        ],
        tmp_path,
    )
    report = cost_report(document, accepted_clips=1)
    assert report.attempts == 2
    assert report.attempts_with_usage == 1
    assert report.usage_coverage == 0.5


def test_escalation_precision_is_none_when_nothing_escalated(tmp_path: Path) -> None:
    """None, not 1.0 — 1.0 would claim escalation always works on no evidence."""
    document = _transcript(
        [
            _span(span_id="01ROOT", kind="run", name="pipeline.run"),
            _span(span_id="01C", parent_id="01ROOT", kind="model_call", name="judge.call"),
            _span(
                span_id="01A1",
                parent_id="01C",
                kind="attempt",
                name="judge.dispatch",
                refs={"model": "primary"},
            ),
        ],
        tmp_path,
    )
    assert escalation_report(document).precision is None


def test_an_escalation_that_succeeded_is_counted(tmp_path: Path) -> None:
    document = _transcript(
        [
            _span(span_id="01ROOT", kind="run", name="pipeline.run"),
            _span(span_id="01C", parent_id="01ROOT", kind="model_call", name="judge.call"),
            _span(
                span_id="01A1",
                parent_id="01C",
                kind="attempt",
                name="judge.dispatch",
                status="error",
                refs={"model": "primary"},
            ),
            _span(
                span_id="01A2",
                parent_id="01C",
                kind="attempt",
                name="judge.dispatch",
                refs={"model": "fallback"},
            ),
        ],
        tmp_path,
    )
    report = escalation_report(document)
    assert report.escalated_calls == 1
    assert report.escalations_that_succeeded == 1
    assert report.precision == 1.0


# -- trace-backed metrics ---------------------------------------------------------------


def _trace(candidates: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"rounds": [{"candidates": candidates}], **extra}


def test_a_rejection_with_no_named_failure_is_counted_as_undiagnosed() -> None:
    """A by-check histogram that silently omits these would look complete."""
    trace = _trace(
        [
            {"result_id": "a", "structural_valid": False, "structural_failures": ["elbow"]},
            {"result_id": "b", "structural_valid": False, "structural_failures": []},
            {"result_id": "c", "structural_valid": True},
        ]
    )
    attribution = rejection_attribution(trace)
    assert attribution.total_candidates == 3
    assert attribution.rejected_candidates == 2
    assert attribution.by_check == {"elbow": 1}
    assert attribution.undiagnosed == 1
    assert sum(attribution.by_check.values()) < attribution.rejected_candidates


def test_a_batch_where_every_recipe_agrees_is_flagged_degenerate() -> None:
    trace = _trace(
        [
            {"result_id": "a", "recipe": "one", "structural_failures": ["x"]},
            {"result_id": "b", "recipe": "two", "structural_failures": ["x"]},
        ]
    )
    diversity = diversity_relevance(trace)
    assert diversity.distinct_outcomes == 1
    assert diversity.degenerate is True
    assert diversity.severity_spread == 0.0


def test_a_batch_with_differing_outcomes_is_not_degenerate() -> None:
    trace = _trace(
        [
            {"result_id": "a", "recipe": "one", "structural_failures": []},
            {"result_id": "b", "recipe": "two", "structural_failures": ["x", "y"]},
        ]
    )
    diversity = diversity_relevance(trace)
    assert diversity.distinct_outcomes == 2
    assert diversity.degenerate is False
    assert diversity.severity_spread > 0.0


# -- selection regret -------------------------------------------------------------------


def test_selection_regret_is_positive_when_a_better_candidate_was_passed_over() -> None:
    trace = _trace(
        [{"result_id": "a"}, {"result_id": "b"}],
        winner_result_id="a",
    )
    scores = {"a": 0.4, "b": 0.9}
    regret = selection_regret(
        trace, oracle=lambda clip: scores[clip["id"]], clip_of=lambda rid: {"id": rid}
    )
    assert regret.best_result_id == "b"
    assert regret.winner_score == 0.4
    assert regret.regret == pytest.approx(0.5)
    assert regret.winner_was_best is False


def test_selection_regret_is_zero_when_the_winner_was_best() -> None:
    trace = _trace([{"result_id": "a"}, {"result_id": "b"}], winner_result_id="b")
    scores = {"a": 0.4, "b": 0.9}
    regret = selection_regret(
        trace, oracle=lambda clip: scores[clip["id"]], clip_of=lambda rid: {"id": rid}
    )
    assert regret.regret == pytest.approx(0.0)
    assert regret.winner_was_best is True


def test_scoring_with_the_selector_itself_makes_regret_identically_zero() -> None:
    """Why §3.3 requires a deterministic oracle rather than the judge's own score.

    If the oracle *is* the function that picked the winner, the winner is by construction
    the argmax and regret is always zero — a perfect selector, measured by itself. This test
    exists so the tautology is demonstrated rather than described in a docstring.
    """

    judge_scores = {"a": 0.9, "b": 0.4}
    winner = max(judge_scores, key=lambda k: judge_scores[k])
    trace = _trace([{"result_id": "a"}, {"result_id": "b"}], winner_result_id=winner)
    regret = selection_regret(
        trace, oracle=lambda clip: judge_scores[clip["id"]], clip_of=lambda rid: {"id": rid}
    )
    assert regret.regret == 0.0
    assert regret.winner_was_best is True


# -- repair efficacy, the mandatory control ---------------------------------------------


def _repairs() -> list[dict[str, Any]]:
    return [{"after_round": 1, "kind": "deterministic_prefilter_safety"}]


def test_repair_efficacy_refuses_to_run_without_a_control() -> None:
    """Plan 10 §6: the random control *is* the test. Zero controls is not a measurement."""
    with pytest.raises(ValueError, match="at least one random control"):
        repair_efficacy(
            _repairs(),
            score_source=lambda e: 0.5,
            score_repaired=lambda e: 0.7,
            score_control=lambda e, p: 0.5,
            magnitudes_of=lambda e: {"x": 0.1},
            controls=0,
            seed=1,
        )


def test_an_outcome_without_controls_reports_none_not_success() -> None:
    outcome = RepairOutcome(after_round=1, kind="k", source_score=0.5, repaired_score=0.9)
    assert outcome.repair_delta == pytest.approx(0.4)
    assert outcome.empirical_p is None


def test_a_repair_that_only_jitters_is_not_significant() -> None:
    """The test this PR exists for.

    `score_repaired` improves on the source, so `repair_delta` is positive and the repair
    looks effective. But the repair is drawn from the same distribution as the control, so
    an arbitrary perturbation of matched magnitude does just as well about half the time and
    the empirical p-value is far from significant. Without the control this is indis-
    tinguishable from a repair that chose well.
    """

    jitter = random.Random(20260825)

    def score_control(entry: dict[str, Any], perturbation: dict[str, float]) -> float:
        return 0.5 + jitter.uniform(-0.2, 0.2)

    efficacy = repair_efficacy(
        _repairs(),
        score_source=lambda e: 0.5,
        # A jitter repair: same distribution as the control, one sample of it.
        score_repaired=lambda e: 0.5 + 0.05,
        score_control=score_control,
        magnitudes_of=lambda e: {"x": 0.1},
        controls=200,
        seed=7,
    )
    outcome = efficacy.outcomes[0]
    assert outcome.repair_delta > 0.0, "the jitter repair does look like an improvement"
    assert outcome.empirical_p is not None
    assert outcome.empirical_p > 0.2, (
        f"a jitter repair must not look significant; p={outcome.empirical_p}"
    )
    assert efficacy.measured is True


def test_a_repair_that_genuinely_chose_well_is_significant() -> None:
    """The other direction, so the previous test is not passing for want of power."""
    jitter = random.Random(20260825)

    efficacy = repair_efficacy(
        _repairs(),
        score_source=lambda e: 0.5,
        score_repaired=lambda e: 0.95,
        score_control=lambda e, p: 0.5 + jitter.uniform(-0.2, 0.2),
        magnitudes_of=lambda e: {"x": 0.1},
        controls=200,
        seed=7,
    )
    outcome = efficacy.outcomes[0]
    assert outcome.empirical_p is not None
    assert outcome.empirical_p < 0.05, (
        f"a decisive repair must separate from the control; p={outcome.empirical_p}"
    )


def test_the_control_matches_the_repair_magnitude_and_randomises_only_sign() -> None:
    """A control drawn from a fixed range would measure repair size, not repair choice."""
    magnitudes = {"wrist_delta": 0.3, "hold_delta": 0.05}
    rng = random.Random(1)
    for _ in range(20):
        perturbation = perturbation_of(magnitudes, rng=rng)
        assert set(perturbation) == set(magnitudes)
        for name, value in perturbation.items():
            assert abs(value) == pytest.approx(magnitudes[name])


# -- the assembled report ---------------------------------------------------------------


def test_the_report_names_the_metrics_it_could_not_compute(tmp_path: Path) -> None:
    """A five-metric report that reads as seven is the composition failure to avoid."""
    document = _transcript(
        [_span(span_id="01ROOT", kind="run", name="pipeline.run", duration_ms=10.0)], tmp_path
    )
    report = trajectory_report(document, _trace([]), accepted_clips=0)
    assert report.unmeasured == ("selection_regret", "repair_efficacy")


def test_the_report_reports_nothing_unmeasured_when_both_oracles_ran(tmp_path: Path) -> None:
    document = _transcript(
        [_span(span_id="01ROOT", kind="run", name="pipeline.run", duration_ms=10.0)], tmp_path
    )
    trace = _trace([{"result_id": "a"}], winner_result_id="a")
    selection = selection_regret(
        trace, oracle=lambda clip: 1.0, clip_of=lambda rid: {"id": rid}
    )
    repair = repair_efficacy(
        _repairs(),
        score_source=lambda e: 0.5,
        score_repaired=lambda e: 0.6,
        score_control=lambda e, p: 0.5,
        magnitudes_of=lambda e: {"x": 0.1},
        controls=4,
        seed=3,
    )
    report = trajectory_report(
        document, trace, accepted_clips=1, selection=selection, repair=repair
    )
    assert report.unmeasured == ()
