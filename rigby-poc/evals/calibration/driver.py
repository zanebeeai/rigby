"""Drive graders across recorded evidence and score the result. No model calls.

Plan 10 §5. Every arm here consumes ground truth someone else authored — the
corpus for unmutated clips, `evals.mutations` for perturbed ones — and reports
what it can measure with the data that exists rather than what the plan hoped
for.

**Two arms cannot be measured on the data available today, and both say so
rather than returning a number.**

*Detection threshold.* §5.2 makes it the headline: severity at 50% detection with
a confidence interval. That needs a curve. 06a shipped 28 legacy specs all at
`Tier.SEVERE` — one severity, so one point — and 06b/06c, which add the graded
families, did not land. `CalibrationEvidence.detection_threshold=None` already
makes the gate *fail* rather than skip, which is the correct outcome: a grader
whose detection cannot be measured has not been shown to detect.

*Static-target results are not threshold points.* 676 of 964 applicable
(spec, case) pairs land on a bone the case never moves. The base holds one pose
and the mutated clip holds another, so a grader that can see the bone separates
them perfectly — a **capability** measurement, with no real motion for a
threshold to sit inside. Pooling them inflates the sweep, so they are tagged by
`Applicability.static_target`, scored, and reported apart.

**`evals.corruptions` is not used here and must not be.** `corrupt_clip` labels
its own output — it sets `success=False`, attaches a `Failure`, and writes
`metrics["deliberate_corruption"]` — and `calibrate_judge` scored the conjunction
of the grader's accept flag with `structural_valid`, which the corruption forces
false. Every rate that suite produced was therefore not a function of the
grader's verdict at all. `evals.mutations.legacy` strips all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evals.calibration.replay import ReplayStore, prompt_versions_agree
from evals.calibration_stats import (
    UnaryJudgment,
    UnaryScores,
    majority_class_baseline,
    score_unary,
)
from rigby_poc.judge import assemble_split_score


class CalibrationDataError(RuntimeError):
    """The arm cannot be scored honestly on the data it was given."""


@dataclass(frozen=True)
class ArmResult:
    """One calibration arm, or an explicit statement that it has no data.

    `scores` is `None` when the arm could not be measured. That is not a gap to
    be filled in by the reader: `reason` says which of "no data", "no variation"
    or "not applicable" applies, because a driver that records a skip with no
    reason cannot tell "no detector exists here" from "the harness declined".
    """

    name: str
    scores: UnaryScores | None
    n: int
    reason: str = ""
    #: Capability results, held apart from threshold results by
    #: `Applicability.static_target`. Never pooled into the arm's own numbers.
    static_target_n: int = 0

    @property
    def measured(self) -> bool:
        return self.scores is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "measured": self.measured,
            "n": self.n,
            "reason": self.reason,
            "static_target_n": self.static_target_n,
            "scores": self.scores.to_dict() if self.scores else None,
        }


def score_arm(
    name: str,
    records: Sequence[UnaryJudgment],
    *,
    static_target_n: int = 0,
) -> ArmResult:
    """Score one arm, refusing rather than reporting a rate over one class.

    Sensitivity over no good clips and specificity over no bad ones are both
    `0/0`. A proportion there is not a low score, it is an absent measurement,
    and reporting it as 0.0 would read as a grader that never accepts.
    """
    if not records:
        return ArmResult(name, None, 0, "no records", static_target_n)
    labels = [record.is_good for record in records]
    if len(set(labels)) < 2:
        only = "good" if labels[0] else "bad"
        return ArmResult(
            name,
            None,
            len(records),
            f"no variation: every clip is {only}, so one of the two rates has no denominator",
            static_target_n,
        )
    return ArmResult(name, score_unary(records), len(records), "", static_target_n)


def replay_accept(store: ReplayStore, case_id: str, *, intent: str | None) -> bool:
    """What the split path decides for one recorded case.

    Calls `judge.assemble_split_score` — the identical function a live run calls.
    Deliberately not wrapped in this module's replay layer; see `replay.py`.
    """
    score, _ = assemble_split_score(store.parts(case_id), intent=intent)
    return score.accept


def sensitivity_arm(
    store: ReplayStore,
    *,
    intents: Mapping[str, str | None],
    deterministically_valid: Mapping[str, bool],
) -> ArmResult:
    """Accept rate on unmutated clips that pass every deterministic gate.

    §5.2 calls this non-circular because selection is by deterministic gates and
    never by the judge. That property only holds while the gates *can* select:
    with lane `anatomy`'s 5° elbow bound rejecting 46 of 46 corpus cases, the
    selector admits nothing and this arm has no denominator. It reports that
    rather than an accept rate over an empty set.
    """
    eligible = [
        case_id for case_id in store.cases() if deterministically_valid.get(case_id, False)
    ]
    if not eligible:
        return ArmResult(
            "sensitivity",
            None,
            0,
            "no corpus case passes every deterministic gate, so the arm has no denominator",
        )
    records = [
        UnaryJudgment(
            clip_id=case_id,
            is_good=True,
            accepted=replay_accept(store, case_id, intent=intents.get(case_id)),
        )
        for case_id in eligible
    ]
    return score_arm("sensitivity", records)


def prompt_version_report(store: ReplayStore) -> dict[str, Any]:
    """Which prompt each grader was recorded against, and whether it is one.

    A calibration result is only meaningful against a stated prompt version
    (07 §3.5). More than one sha for a grader means the recordings span a prompt
    change and pooling them reports one calibration for two instruments.
    """
    seen = prompt_versions_agree(store, store.cases())
    return {
        "prompt_sha256_by_grader": {name: sorted(shas) for name, shas in seen.items()},
        "single_prompt_version": all(len(shas) <= 1 for shas in seen.values()),
        "graders_spanning_a_prompt_change": sorted(
            name for name, shas in seen.items() if len(shas) > 1
        ),
    }


def baseline_for(records: Sequence[UnaryJudgment]) -> float | None:
    """The majority-class baseline every reported rate must be stated against.

    Returned as `None` for an empty set rather than as a number, so a caller
    cannot emit a rate with a fabricated baseline beside it.
    """
    if not records:
        return None
    return majority_class_baseline([record.is_good for record in records])
