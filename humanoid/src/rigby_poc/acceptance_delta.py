"""Measure what changing the acceptance rule actually does, on one corpus, with n.

07d changes acceptance behaviour, and plan 07 §6.4 warns that any calibration
frozen before it describes a different instrument.  The obligation that follows
is to report accept rates **before and after on the same corpus**, not to assert
that the new path is better.

This module is the instrument, and it is deliberately separate from `judge.py`
so it can be pointed at recorded judge records without a model, a browser, or a
render.  Three of the four things 07d changes are measurable that way, because
they are pure functions of a recorded payload:

1. `accept` ceasing to be the model's self-report and becoming the published
   rule applied by the decision layer (§3.3).
2. The split graders' claim aggregation replacing a self-reported score (§3.2).
3. An unjudged gating dimension ceasing to be acceptable.

The fourth — removing diagnostics from the prompt — cannot be measured without
model calls, because it changes what the model *sees*, not how its answer is
processed.  That one belongs to 10f.  Reporting a single before/after delta
without saying which of the four it attributes to would be exactly the
unattributable number this file exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from evals.calibration_stats import proportion

from .judge import ACCEPTANCE_MINIMUM_SCORES, meets_acceptance_thresholds


@dataclass(frozen=True)
class AcceptanceComparison:
    """One clip's acceptance under both rules, with the disagreement classified."""

    case_id: str
    self_reported: bool
    rule_derived: bool
    #: `deterministic_valid` from the structural layer, which §3.3 makes part of
    #: the combined decision.  `None` when the record does not carry it.
    deterministic_valid: bool | None = None
    #: `render_provenance.render_mode` from PR 05.  A judged render and a
    #: deterministic render are not comparable: anti-aliasing changes the
    #: magnitude of the pixel difference a given pose delta produces, not just
    #: its noise floor, so a before/after that crosses modes measures the mode.
    render_mode: str | None = None

    @property
    def changed(self) -> bool:
        return self.self_reported != self.rule_derived

    @property
    def direction(self) -> str:
        if not self.changed:
            return "unchanged"
        # A clip the model accepted and the rule rejects is the case 07a's
        # escalation was firing on; a clip the rule accepts and the model
        # rejected is the rarer and more surprising direction.
        return "newly_rejected" if self.self_reported else "newly_accepted"


def compare_record(case_id: str, record: Mapping[str, Any]) -> AcceptanceComparison:
    """Read one **combined-path** judge record both ways. Pure; no model, no render.

    Refuses a `split_motion_judgment`, and the reason is a defect this module was
    written to prevent and then contained.

    On the split path `assemble_split_score` already sets
    `accept = fully_judged and meets_acceptance_thresholds(scores)` and
    materialises it at `call.parsed.accept` so existing consumers keep working.
    This function then reads that value as `self_reported` and recomputes
    `rule_derived` as `meets_acceptance_thresholds(parsed)` and not-unjudged —
    which is the *same predicate over the same dict*. Both sides are equal by
    construction, so the delta is a structural zero.

    A zero that means "wrong record kind" and a zero that means "the rule changed
    nothing" are the not-measured / measured-negative conflation, in the file
    written to stop it. Found by lane `groundtruth`'s independent check, which
    swept 20000 synthetic split records over random claim verdicts and got zero
    disagreements with both accept values present.

    The measurement 07d owes is over **combined** records, where `accept` is the
    model's own boolean and the rule is genuinely a second opinion.
    """
    kind = record.get("kind")
    if kind == "split_motion_judgment":
        raise ValueError(
            f"{case_id}: the split path already derives accept from the published rule, "
            "so before and after are equal by construction; 07d's delta needs "
            "combined-path records"
        )
    parsed = record.get("call", {}).get("parsed", {})
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{case_id}: judge record carries no parsed score")
    self_reported = parsed.get("accept")
    if not isinstance(self_reported, bool):
        raise ValueError(f"{case_id}: judge record carries no self-reported accept")
    unjudged = set(record.get("insufficient_evidence_dimensions", []))
    # An unjudged gating dimension is never acceptable: you cannot accept what
    # could not be judged. Checked here as well as in the decision layer so the
    # measurement does not depend on the code under measurement.
    rule_derived = meets_acceptance_thresholds(parsed) and not (
        unjudged & set(ACCEPTANCE_MINIMUM_SCORES)
    )
    deterministic = record.get("deterministic_valid")
    provenance = record.get("render_provenance")
    render_mode = (
        provenance.get("render_mode") if isinstance(provenance, Mapping) else None
    )
    return AcceptanceComparison(
        case_id=case_id,
        self_reported=self_reported,
        rule_derived=rule_derived,
        deterministic_valid=deterministic if isinstance(deterministic, bool) else None,
        render_mode=str(render_mode) if render_mode is not None else None,
    )


def acceptance_delta(comparisons: Sequence[AcceptanceComparison]) -> dict[str, Any]:
    """Both rates, their n, the disagreement count, and which way it went.

    Reported as two rates with a paired disagreement breakdown rather than one
    delta. A delta of zero is consistent with perfect agreement *and* with equal
    numbers of clips moving in each direction, which are entirely different
    findings about the judge.
    """
    if not comparisons:
        raise ValueError("acceptance_delta needs at least one comparison")
    # `deterministic_valid` is recorded but deliberately NOT part of `rule_derived`.
    # §3.3's decision is `deterministic_valid and grader_verdict and score_threshold`,
    # and folding the first term in here would make 07d's delta absorb whatever the
    # deterministic layer did that week. 04c is about to reject 100% of the corpus on
    # the elbow bound alone; a combined rate would then read 0% after and would be
    # entirely 04c's doing.
    deterministic = {item.deterministic_valid for item in comparisons}
    modes = {item.render_mode for item in comparisons}
    if len(modes) > 1:
        # Refused rather than reported with a caveat. A pooled rate across
        # render modes is a measurement of the mode wearing a rate's clothes,
        # and a caveat on it would be read past.
        raise ValueError(
            f"cannot pool acceptance across render modes: {sorted(str(mode) for mode in modes)}"
        )
    n = len(comparisons)
    newly_rejected = [item for item in comparisons if item.direction == "newly_rejected"]
    newly_accepted = [item for item in comparisons if item.direction == "newly_accepted"]
    return {
        "before": proportion(sum(item.self_reported for item in comparisons), n),
        "after": proportion(sum(item.rule_derived for item in comparisons), n),
        "n": n,
        "changed": len(newly_rejected) + len(newly_accepted),
        "newly_rejected": len(newly_rejected),
        "newly_accepted": len(newly_accepted),
        "newly_rejected_case_ids": sorted(item.case_id for item in newly_rejected),
        "newly_accepted_case_ids": sorted(item.case_id for item in newly_accepted),
        #: Named so a reader cannot take the delta as the effect of 07d in full.
        #: `claim_aggregation` was listed here and is not measurable by this
        #: instrument: it changes the dimension scores themselves, and both sides
        #: of the comparison are derived from one already-aggregated `parsed`.
        #: Measuring it needs a combined and a split judgment of the *same* clip,
        #: paired, and `compare_record` takes one record with no paired entry
        #: point. Moved to the deferred list rather than left implying coverage.
        "attributes_to": [
            "decision_layer_authority",
            "unjudged_dimension_rejection",
        ],
        "does_not_attribute_to": ["diagnostics_removal", "claim_aggregation"],
        "render_mode": next(iter(modes)),
        # A gate that fires on everything carries no information about anything,
        # however correct it is. Named so a reader cannot mistake a constant label
        # for a measured one -- the degenerate-label problem, arriving as a
        # correctly-derived anatomical bound rather than as a bug.
        #
        # Tri-state, because `None` means "not recorded" and not "recorded the
        # same everywhere". Discarding the Nones and calling the remainder
        # constant would claim the gate never varied across clips that never
        # reported -- the not-measured / measured-negative conflation, in the
        # one file written to stop it.
        "deterministic_valid_is_constant": (
            None if None in deterministic else len(deterministic) <= 1
        ),
        "deterministic_valid_values": sorted(
            str(value) for value in deterministic
        ),
        "measures": "grader_and_decision_layer_only",
    }


def compare_records(records: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    return acceptance_delta([compare_record(case_id, record) for case_id, record in records])
