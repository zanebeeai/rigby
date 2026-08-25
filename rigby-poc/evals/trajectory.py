"""Plan 10 section 6 — trajectory evals over a run's own decision path.

Neither the planner nor the judge is agentic, so "trajectory" here means the pipeline's
decisions: which candidates it made, which it rejected and why, whether repair helped,
whether escalation changed anything, and what the whole thing cost.

Everything is a pure function over a `Transcript` (plan 01 section 4.2) or over the
flywheel trace. Nothing here calls a model.

**Two of the seven metrics cannot be computed today and this module refuses to pretend
otherwise.** Selection regret and repair efficacy both need a per-clip score, and plan 10
section 3.3 specifies that oracle as a *weighted deterministic composite* over check
severities. That composite does not exist yet — it is 10b/10c's work, unstarted at the time
of writing; `analysis/composite.py` is composite *motion*, not a composite *score*. So both
functions take an `oracle` argument and there is no default. A caller must supply one and
therefore must know what it is measuring with.

The one oracle that must never be used is the judge's own scores. `flywheel._candidate_score`
reads `judgment.overall`, so scoring selection regret with it would ask the judge whether the
judge chose correctly. Section 3.3 exists precisely to avoid that: "which is how selection
regret gets measured without a human."
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rigby_poc.transcript import Transcript


#: A per-clip score in [0, 1] where higher is better. Plan 10 section 3.3's composite is
#: the intended implementation; anything else must say so at the call site.
Oracle = Callable[[Mapping[str, Any]], float]

#: The stages the pipeline declares. Derived by test from the literals in `flywheel.py`
#: rather than restated here, so this list cannot drift from what is emitted.
PIPELINE_STAGES = (
    "planning",
    "candidates",
    "structural_checks",
    "visual_evidence",
    "vlm_judge",
    "repair",
    "finalize",
)


# -- stage attribution ------------------------------------------------------------------


@dataclass(frozen=True)
class StageAttribution:
    """Where a run spent its wall clock, and which stages never ran at all."""

    durations_ms: dict[str, float]
    occurrences: dict[str, int]
    missing: tuple[str, ...]

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms.values())

    def share(self, stage: str) -> float:
        total = self.total_ms
        return self.durations_ms.get(stage, 0.0) / total if total else 0.0


def stage_attribution(document: Transcript) -> StageAttribution:
    """Duration and occurrence count per stage.

    `missing` is the point of the type. A stage that never ran reports as absent rather
    than as zero, because those are different facts: `repair` at 0 ms means every round
    passed first time, while `repair` missing means nothing ever reached the repair path.
    A dict of durations alone cannot tell them apart.
    """

    durations = document.duration_by_stage()
    occurrences: dict[str, int] = {}
    for span in document.by_kind("stage"):
        occurrences[span.name] = occurrences.get(span.name, 0) + 1
    missing = tuple(stage for stage in PIPELINE_STAGES if stage not in durations)
    return StageAttribution(
        durations_ms=dict(durations), occurrences=occurrences, missing=missing
    )


# -- cost -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostReport:
    """Cost of a run, with the denominator that makes it interpretable."""

    model_calls: int
    attempts: int
    attempts_with_usage: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    wall_clock_ms: float
    accepted_clips: int

    #: `TokenUsage.total_tokens` is reported by the provider and is not always the sum
    #: of the parts, so it is carried through rather than recomputed here.
    total_tokens: int = 0

    @property
    def tokens_per_accepted_clip(self) -> float | None:
        """None rather than zero when nothing was accepted.

        A run that accepted nothing has no cost *per clip*; reporting 0 would say the run
        was free and reporting the raw total would say one clip cost the whole run. Plan 01
        section 7 item 4 makes the same distinction for usage that does not exist.
        """

        return self.total_tokens / self.accepted_clips if self.accepted_clips else None

    @property
    def usage_coverage(self) -> float:
        """Fraction of attempts that carried usage — the caveat that travels with the cost.

        Plan 01 section 7 item 4: a dispatch that raises carries no usage, because the
        provider never answered. So a token total is a floor, and this number says how much
        of a floor. Publishing the cost without it would state two populations as one.
        """

        return self.attempts_with_usage / self.attempts if self.attempts else 1.0


def cost_report(document: Transcript, *, accepted_clips: int) -> CostReport:
    attempts = document.attempts()
    usage = document.total_tokens()
    root = document.root()
    return CostReport(
        model_calls=len(document.model_calls()),
        attempts=len(attempts),
        attempts_with_usage=sum(1 for span in attempts if span.has_usage),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.total_tokens,
        wall_clock_ms=root.duration_ms or 0.0,
        accepted_clips=accepted_clips,
    )


# -- escalation -------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationReport:
    """Whether falling back to a second model changed anything."""

    escalated_calls: int
    escalations_that_succeeded: int
    escalations_that_failed_too: int
    models_used: tuple[str, ...]
    #: Calls where no fallback was configured, so escalation could not occur at all.
    #: `precision is None` otherwise conflates "the primary always sufficed" with "the
    #: safety net was switched off", which are opposite facts about a run (lane `judge`).
    unavailable_calls: int = 0

    @property
    def precision(self) -> float | None:
        """Share of escalations that produced a usable answer. None when none occurred.

        None rather than 1.0: a run that never escalated has no precision to report, and
        1.0 would read as "escalation always works" on evidence that does not exist. Read
        it beside `unavailable_calls` — see that field.
        """

        return (
            self.escalations_that_succeeded / self.escalated_calls
            if self.escalated_calls
            else None
        )


def escalation_report(document: Transcript) -> EscalationReport:
    """Escalation, read from what the judge recorded rather than inferred from models.

    `judge.py:1242` stamps its whole routing dict onto the `model_call` span, so
    `escalated` is stated rather than reconstructed. Reading it matters for two reasons
    the model-set heuristic gets wrong:

    * A **transient retry** re-dispatches the *same* model, so it never looked like an
      escalation — but `transient_retry_count` is recorded and the heuristic could not
      have reported it either way.
    * `VLMJudge` resolves `fallback_model` to the primary when a `model=` is passed with
      no explicit fallback (`judge.py:1263-1266`), and the escalation guard then requires
      `fallback_model != selected_primary` (`judge.py:1205-1206`). So escalation is
      **disabled**, not invisible — zero is the truth, but it is a different zero from
      "the primary always sufficed". `unavailable_calls` separates them.

    The model-set heuristic stays as the fallback for spans with no routing attrs, so a
    transcript written before this was recorded still reports something rather than zero.
    """

    escalated = 0
    succeeded = 0
    failed = 0
    unavailable = 0
    models: set[str] = set()
    for call in document.model_calls():
        attempts = [span for span in document.children(call.span_id) if span.kind == "attempt"]
        used = [str(span.refs.get("model")) for span in attempts if span.refs.get("model")]
        models.update(used)
        primary = call.attrs.get("primary_model")
        fallback = call.attrs.get("fallback_model")
        if primary is not None and (fallback is None or fallback == primary):
            unavailable += 1
        recorded = call.attrs.get("escalated")
        if isinstance(recorded, bool):
            did_escalate = recorded
        else:
            did_escalate = len(set(used)) > 1
        if not did_escalate:
            continue
        escalated += 1
        # The last attempt decides the call: escalation succeeded when the fallback
        # returned without an error, and "failed too" when even the fallback errored.
        if attempts and attempts[-1].status == "ok":
            succeeded += 1
        else:
            failed += 1
    return EscalationReport(
        escalated_calls=escalated,
        escalations_that_succeeded=succeeded,
        escalations_that_failed_too=failed,
        models_used=tuple(sorted(models)),
        unavailable_calls=unavailable,
    )


# -- rejection attribution --------------------------------------------------------------


@dataclass(frozen=True)
class RejectionAttribution:
    """Structural rejections grouped by the check that caused them."""

    by_check: dict[str, int]
    rejected_candidates: int
    total_candidates: int
    undiagnosed: int

    @property
    def rejection_rate(self) -> float | None:
        return self.rejected_candidates / self.total_candidates if self.total_candidates else None


def rejection_attribution(trace: Mapping[str, Any]) -> RejectionAttribution:
    """Group structural rejections by check id.

    `undiagnosed` counts candidates that were rejected while naming no failure. That is not
    a rounding error: it means the rejection has no attributable cause, which is exactly the
    per-family attribution gap `docs/evaluation.md:64` has recorded since the beginning.
    Reporting it as a number keeps it visible instead of letting a by-check histogram that
    sums to less than the rejection count look complete.
    """

    by_check: dict[str, int] = {}
    rejected = 0
    total = 0
    undiagnosed = 0
    for round_record in trace.get("rounds", []) or []:
        if not isinstance(round_record, Mapping):
            continue
        for candidate in round_record.get("candidates", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            total += 1
            if candidate.get("structural_valid") is not False:
                continue
            rejected += 1
            failures = candidate.get("structural_failures") or []
            named = [str(item) for item in failures if item]
            if not named:
                undiagnosed += 1
            for check_id in named:
                by_check[check_id] = by_check.get(check_id, 0) + 1
    return RejectionAttribution(
        by_check=by_check,
        rejected_candidates=rejected,
        total_candidates=total,
        undiagnosed=undiagnosed,
    )


# -- diversity relevance ----------------------------------------------------------------


@dataclass(frozen=True)
class DiversityRelevance:
    """Did the axes the batch varied matter for this prompt?"""

    recipes: tuple[str, ...]
    severity_spread: float
    distinct_outcomes: int
    degenerate: bool


def diversity_relevance(trace: Mapping[str, Any]) -> DiversityRelevance:
    """Spread of structural outcomes across the recipes a round generated.

    `degenerate` is the finding, not the number. When every recipe produces the same
    structural outcome the batch varied nothing that this prompt is sensitive to, and a
    diversity score computed over it is describing the recipe list rather than the motion.
    """

    recipes: list[str] = []
    signatures: list[tuple[str, ...]] = []
    severities: list[float] = []
    for round_record in trace.get("rounds", []) or []:
        if not isinstance(round_record, Mapping):
            continue
        for candidate in round_record.get("candidates", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            recipes.append(str(candidate.get("recipe", "")))
            failures = tuple(sorted(str(f) for f in (candidate.get("structural_failures") or [])))
            signatures.append(failures)
            severities.append(float(len(failures)))
    spread = 0.0
    if severities:
        mean = sum(severities) / len(severities)
        spread = (sum((value - mean) ** 2 for value in severities) / len(severities)) ** 0.5
    distinct = len(set(signatures))
    return DiversityRelevance(
        recipes=tuple(recipes),
        severity_spread=spread,
        distinct_outcomes=distinct,
        degenerate=bool(signatures) and distinct <= 1,
    )


# -- selection regret -------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRegret:
    """How much worse the chosen winner was than the best available candidate."""

    winner_result_id: str | None
    winner_score: float | None
    best_result_id: str | None
    best_score: float | None
    regret: float | None
    candidates_scored: int

    @property
    def winner_was_best(self) -> bool | None:
        if self.regret is None:
            return None
        return self.regret <= 0.0


def selection_regret(
    trace: Mapping[str, Any],
    *,
    oracle: Oracle,
    clip_of: Callable[[str], Mapping[str, Any] | None],
) -> SelectionRegret:
    """Score every candidate with `oracle` and compare the winner against the best.

    `oracle` must be deterministic and must not be the judge. Passing
    `flywheel._candidate_score` here would ask the judge to mark its own work — the winner
    is chosen by that score, so regret would be identically zero by construction and would
    look like a perfect selector.
    """

    scored: list[tuple[str, float]] = []
    for round_record in trace.get("rounds", []) or []:
        if not isinstance(round_record, Mapping):
            continue
        for candidate in round_record.get("candidates", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            result_id = candidate.get("result_id")
            if not result_id:
                continue
            clip = clip_of(str(result_id))
            if clip is None:
                continue
            scored.append((str(result_id), float(oracle(clip))))
    if not scored:
        return SelectionRegret(None, None, None, None, None, 0)

    winner_id = trace.get("winner_result_id")
    winner_id = str(winner_id) if winner_id else None
    by_id = dict(scored)
    best_id, best_score = max(scored, key=lambda pair: pair[1])
    winner_score = by_id.get(winner_id) if winner_id else None
    regret = None if winner_score is None else best_score - winner_score
    return SelectionRegret(
        winner_result_id=winner_id,
        winner_score=winner_score,
        best_result_id=best_id,
        best_score=best_score,
        regret=regret,
        candidates_scored=len(scored),
    )


# -- repair efficacy --------------------------------------------------------------------


@dataclass(frozen=True)
class RepairOutcome:
    """One repair, measured against a matched random control."""

    after_round: int
    kind: str
    source_score: float
    repaired_score: float
    control_scores: tuple[float, ...] = field(default=())

    @property
    def repair_delta(self) -> float:
        return self.repaired_score - self.source_score

    @property
    def control_deltas(self) -> tuple[float, ...]:
        return tuple(score - self.source_score for score in self.control_scores)

    @property
    def controls_beating_repair(self) -> int:
        return sum(1 for delta in self.control_deltas if delta >= self.repair_delta)

    @property
    def empirical_p(self) -> float | None:
        """Share of random controls that did at least as well as the repair.

        This is the whole measurement. A repair loop that merely jitters parameters
        produces a positive `repair_delta` and looks effective; it is only distinguishable
        from noise by how often an arbitrary jitter of the same size does as well. Small p
        means the repair chose better than chance. There is deliberately no default control
        count, and `None` here means no control was run — never "it passed".
        """

        controls = self.control_deltas
        if not controls:
            return None
        return (self.controls_beating_repair + 1) / (len(controls) + 1)


@dataclass(frozen=True)
class RepairEfficacy:
    outcomes: tuple[RepairOutcome, ...]

    @property
    def measured(self) -> bool:
        return bool(self.outcomes) and all(o.empirical_p is not None for o in self.outcomes)

    @property
    def mean_repair_delta(self) -> float | None:
        if not self.outcomes:
            return None
        return sum(o.repair_delta for o in self.outcomes) / len(self.outcomes)


def perturbation_of(magnitudes: Mapping[str, float], *, rng: random.Random) -> dict[str, float]:
    """A random delta set with the same per-field magnitudes as a real repair.

    Matching magnitude is what makes this a control rather than a strawman. A control drawn
    from a fixed range would be easy for any repair to beat when the repair happens to be
    large, and impossible when it is small, so the comparison would measure repair
    *magnitude* instead of repair *choice*. Sign is randomised; size is copied.
    """

    return {
        field_name: rng.choice((-1.0, 1.0)) * abs(magnitude)
        for field_name, magnitude in magnitudes.items()
    }


def repair_efficacy(
    repairs: Sequence[Mapping[str, Any]],
    *,
    score_source: Callable[[Mapping[str, Any]], float],
    score_repaired: Callable[[Mapping[str, Any]], float],
    score_control: Callable[[Mapping[str, Any], dict[str, float]], float],
    magnitudes_of: Callable[[Mapping[str, Any]], Mapping[str, float]],
    controls: int,
    seed: int,
) -> RepairEfficacy:
    """Measure each repair against `controls` random perturbations of matched magnitude.

    `controls` has no default on purpose. Plan 10 section 6 is explicit that the random
    control *is* the test, so a caller that omits it should get a `TypeError` rather than a
    silent zero-control run that reports `repair_delta` as though it meant something.
    """

    if controls < 1:
        raise ValueError("repair efficacy requires at least one random control; see plan 10 §6")
    rng = random.Random(seed)
    outcomes: list[RepairOutcome] = []
    for entry in repairs:
        if not isinstance(entry, Mapping):
            continue
        magnitudes = magnitudes_of(entry)
        control_scores = tuple(
            score_control(entry, perturbation_of(magnitudes, rng=rng)) for _ in range(controls)
        )
        outcomes.append(
            RepairOutcome(
                after_round=int(entry.get("after_round", 0)),
                kind=str(entry.get("kind", "")),
                source_score=float(score_source(entry)),
                repaired_score=float(score_repaired(entry)),
                control_scores=control_scores,
            )
        )
    return RepairEfficacy(outcomes=tuple(outcomes))


# -- the whole report -------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryReport:
    stages: StageAttribution
    cost: CostReport
    escalation: EscalationReport
    rejections: RejectionAttribution
    diversity: DiversityRelevance
    selection: SelectionRegret | None
    repair: RepairEfficacy | None

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """Metrics plan 10 section 6 names that this report could not compute.

        Named rather than omitted. A report that silently drops the two metrics needing an
        oracle reads as complete, and a reader summing seven metrics from a five-metric
        report is the composition failure this push has been recording all night.
        """

        missing = []
        if self.selection is None:
            missing.append("selection_regret")
        if self.repair is None:
            missing.append("repair_efficacy")
        return tuple(missing)


def trajectory_report(
    document: Transcript,
    trace: Mapping[str, Any],
    *,
    accepted_clips: int,
    selection: SelectionRegret | None = None,
    repair: RepairEfficacy | None = None,
) -> TrajectoryReport:
    """The five transcript-and-trace metrics, plus whichever oracle-backed ones were run."""

    return TrajectoryReport(
        stages=stage_attribution(document),
        cost=cost_report(document, accepted_clips=accepted_clips),
        escalation=escalation_report(document),
        rejections=rejection_attribution(trace),
        diversity=diversity_relevance(trace),
        selection=selection,
        repair=repair,
    )
