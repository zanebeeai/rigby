"""Text-and-JSON graders that judge the parts of the pipeline no image can show.

Plan 10 §4. Nothing in this category exists today: planner evaluation is fixture
classification against expected intent, which measures routing, not fidelity.
These four graders read text and JSON only — no pixels, no renders — and each has
a ground-truth source that already exists or is built in [06](06-mutation-library):

    clause fidelity    prompt + MotionProgram      <- semantic mutations
    rejection auditor  prompt + grammar + reason   <- the 60 supported / 20
                                                     unsupported planner fixtures
    repair soundness   critique + RepairPatch      <- deliberately irrelevant patches
    failure honesty    typed failure + gate fired  <- known-bad corpus cases

The rejection auditor is the highest-value of the four: it measures the
false-rejection rate, which nothing measures today, and a system that over-rejects
looks flawless on every other metric in the suite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import PydanticUndefined

from evals.calibration_stats import majority_class_baseline, proportion

from .judge import RepairPatch, RoutedModelClient, _never_escalate
from .models import (
    BodyAction,
    Digit,
    HandShape,
    Intent,
    ObjectAction,
    ObjectInteractionStyle,
    StrikeType,
    TrajectoryKind,
    TrajectoryPlane,
)


# Plan 10 §4: grade with a different model family than the planner uses. An LLM
# grading a program written by an LLM will happily ratify a misreading it would
# have made itself. `planner.DEFAULT_PRIMARY_MODEL` is the gpt family, so the
# graders default to a different one and the mismatch is asserted by a test
# rather than left to a comment.
DEFAULT_LLM_GRADER_MODEL = "claude-sonnet-4-6"
DEFAULT_LLM_GRADER_FALLBACK_MODEL = "claude-opus-4-6"


def capability_grammar() -> dict[str, list[str]]:
    """The executable capability surface, derived from the enums rather than prose.

    A hand-written grammar would drift the moment someone adds a `BodyAction`,
    and a rejection auditor reading a stale grammar would call a genuine
    capability miss "out of scope" — the exact error the grader exists to catch.
    """
    return {
        "intents": [value for value in Intent if value != Intent.UNSUPPORTED],
        "hand_shapes": list(HandShape),
        "digits": list(Digit),
        "strike_types": list(StrikeType),
        "object_actions": list(ObjectAction),
        "object_styles": list(ObjectInteractionStyle),
        "body_actions": list(BodyAction),
        "trajectory_kinds": list(TrajectoryKind),
        "trajectory_planes": list(TrajectoryPlane),
    }


# --------------------------------------------------------------- output shapes


class ClauseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause: str = Field(min_length=1, max_length=160)
    present: bool
    correct_order: bool
    count_matches: bool
    laterality_matches: bool
    note: str = Field(min_length=3, max_length=200)


class ClauseFidelityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clauses: list[ClauseVerdict] = Field(min_length=1, max_length=12)
    hallucinated_additions: list[str] = Field(max_length=8)
    summary: str = Field(min_length=5, max_length=300)


class RejectionAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: `out_of_scope` — the request genuinely is not in the grammar.
    #: `capability_miss` — it IS in the grammar and the planner failed to route it.
    #: `ambiguous` — the request could go either way.
    verdict: Literal["out_of_scope", "capability_miss", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    nearest_capability: str = Field(max_length=80)
    rationale: str = Field(min_length=5, max_length=300)


class RepairSoundnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responsive: bool
    directionally_correct: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unaddressed_failures: list[str] = Field(max_length=8)
    rationale: str = Field(min_length=5, max_length=300)


class FailureHonestyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consistent: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=5, max_length=300)


# -------------------------------------------------------------------- prompts

CLAUSE_FIDELITY_PROMPT = """You audit whether a compiled motion program says what the user's request said.
You are given the request and the program as JSON. You are NOT judging whether the motion looks good; you are
judging whether the program encodes the request. Break the request into its clauses — each requested action,
each modifier that changes what is performed, each count, each named hand or side. For every clause report
whether the program contains it, whether it appears in the requested order relative to the other clauses,
whether any requested repetition or beat count matches, and whether the named hand or side matches. Then list
anything the program adds that the request did not ask for. Quote clauses from the request verbatim. Judge only
the program; absence of a field is absence of the clause, not a reason to assume a default."""

REJECTION_AUDIT_PROMPT = """You audit a rejection. The planner refused a request as UNSUPPORTED and gave a
reason. You are given the request, the system's executable capability grammar, and that reason.

Answer one question: was the refusal correct? Three verdicts, and the distinction between the first two is the
whole point of this grader.

- out_of_scope: the request genuinely asks for something the grammar cannot express.
- capability_miss: the grammar CAN express this request and the planner failed to route it. This is a false
  rejection. A system that over-rejects looks flawless on every other metric, so do not be generous here.
- ambiguous: the request is genuinely unclear about which capability it wants.

Name the nearest capability in the grammar, or "none" if there truly is none. Judge against the grammar you are
given, not against what a humanoid animation system might plausibly support."""

REPAIR_SOUNDNESS_PROMPT = """You audit a proposed repair. You are given a visual judge's critique of a rejected
motion and the bounded parameter patch proposed in response. Answer two questions.

responsive: does the patch change parameters that could plausibly affect the failures the critique actually
cited? A patch that adjusts unrelated parameters is not responsive, however reasonable it looks.

directionally_correct: for each failure it does address, does the patch move in the direction that would fix it
rather than worsen it?

List any cited failure the patch leaves unaddressed. A patch of all-zero or near-zero deltas addresses nothing.
Do not reward a patch for being cautious; judge it against the critique it was given."""

FAILURE_HONESTY_PROMPT = """You audit whether a reported failure matches reality. You are given the typed
failure the system reported and the deterministic gate that actually fired, with its measured values.

Answer one question: is the reported failure a truthful description of the gate that fired? A run that failed a
ground-contact gate and reported a timing error is inconsistent, even if a timing error is also present. A
report that names the wrong subsystem, the wrong limb, or a failure the gate output does not support is
inconsistent. Being vaguely compatible is not consistency; the report must describe what actually fired."""


class LLMGraders(RoutedModelClient):
    """The four text-only graders, routed and observable like every other model call."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        *,
        fallback_model: str | None = None,
        reasoning_effort: str | None = None,
        max_model_calls: int | None = None,
    ) -> None:
        super().__init__(
            client=client,
            model=model or os.getenv("OPENAI_LLM_GRADER_MODEL", DEFAULT_LLM_GRADER_MODEL),
            fallback_model=(
                fallback_model
                or os.getenv(
                    "OPENAI_LLM_GRADER_FALLBACK_MODEL", DEFAULT_LLM_GRADER_FALLBACK_MODEL
                )
            ),
            reasoning_effort=reasoning_effort,
            max_model_calls=max_model_calls,
        )

    def _grade(
        self,
        *,
        kind: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        text_format: type[BaseModel],
    ) -> dict[str, Any]:
        response, parsed, attempts, routing = self._routed_parse(
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        }
                    ],
                },
            ],
            text_format=text_format,
            escalation_reason=_never_escalate,
        )
        from .judge import _call_record

        return {
            "schema_version": "1.0",
            "kind": kind,
            "call": _call_record(response, parsed, self.model),
            "attempts": attempts,
            "routing": routing,
        }

    def clause_fidelity(self, *, prompt: str, program: Mapping[str, Any]) -> dict[str, Any]:
        return self._grade(
            kind="clause_fidelity_audit",
            system_prompt=CLAUSE_FIDELITY_PROMPT,
            payload={"user_motion_request": prompt, "motion_program": dict(program)},
            text_format=ClauseFidelityReport,
        )

    def rejection_audit(self, *, prompt: str, unsupported_reason: str) -> dict[str, Any]:
        return self._grade(
            kind="rejection_audit",
            system_prompt=REJECTION_AUDIT_PROMPT,
            payload={
                "user_motion_request": prompt,
                "capability_grammar": capability_grammar(),
                "planner_unsupported_reason": unsupported_reason,
            },
            text_format=RejectionAuditReport,
        )

    def repair_soundness(
        self, *, critique: Mapping[str, Any], patch: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._grade(
            kind="repair_soundness_audit",
            system_prompt=REPAIR_SOUNDNESS_PROMPT,
            payload={"judge_critique": dict(critique), "repair_patch": dict(patch)},
            text_format=RepairSoundnessReport,
        )

    def failure_honesty(
        self, *, reported_failure: Mapping[str, Any], gate_output: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._grade(
            kind="failure_honesty_audit",
            system_prompt=FAILURE_HONESTY_PROMPT,
            payload={"reported_failure": dict(reported_failure), "gate_that_fired": dict(gate_output)},
            text_format=FailureHonestyReport,
        )


# ------------------------------------------------------------------- scoring

@dataclass(frozen=True)
class RejectionOutcome:
    """One fixture put through the planner, with the auditor's verdict if it refused."""

    case_id: str
    #: Ground truth: the prompt came from the 60 supported fixtures, not the 20
    #: unsupported ones.  This is the label, not a prediction.
    supported: bool
    #: The planner returned `Intent.UNSUPPORTED`.
    refused: bool
    #: The auditor's verdict, present only when `refused`.
    verdict: str | None = None


def expected_rejection_verdict(supported: bool) -> str:
    """A refusal of a supported prompt is a capability miss by construction."""
    return "capability_miss" if supported else "out_of_scope"


def score_rejection_audit(outcomes: Sequence[RejectionOutcome]) -> dict[str, Any]:
    """Two different numbers that are easy to confuse, so both are reported.

    `false_rejection` is a property of the **planner**: how often it refuses a
    request the grammar can express.  Nothing in the suite measures it today, and
    a system that over-rejects looks flawless on every other metric.  It needs no
    model at all — the fixtures carry the label.

    `auditor_agreement` is a property of the **grader**: how often the LLM
    auditor's verdict matches the label on the refusals.  It is what licenses
    using the auditor beyond the 80 fixtures, and it is meaningless unless
    reported against its baseline — with 20 unsupported and 60 supported
    fixtures, always answering `out_of_scope` scores well if few supported
    prompts are refused.
    """
    supported = [outcome for outcome in outcomes if outcome.supported]
    refusals = [outcome for outcome in outcomes if outcome.refused]
    false_rejections = [outcome for outcome in supported if outcome.refused]
    agreed = [
        outcome
        for outcome in refusals
        if outcome.verdict == expected_rejection_verdict(outcome.supported)
    ]
    labels = [expected_rejection_verdict(outcome.supported) for outcome in refusals]
    return {
        "false_rejection": proportion(len(false_rejections), len(supported)),
        "false_rejection_baseline": 0.0,
        "auditor_agreement": proportion(len(agreed), len(refusals)),
        "auditor_agreement_baseline": majority_class_baseline(labels) if labels else None,
        "n_supported": len(supported),
        "n_unsupported": len(outcomes) - len(supported),
        "n_refusals": len(refusals),
        "false_rejection_case_ids": sorted(outcome.case_id for outcome in false_rejections),
    }


# ------------------------------------------- ground truth for repair soundness

#: Which `RepairPatch` fields could plausibly address which cited failure. This
#: is the ground truth plan 10 §4 calls for — "patches synthesized to be
#: deliberately irrelevant" needs a definition of relevant, and it has to be
#: written down rather than left to the grader being audited.
FAILURE_RELEVANT_DELTAS: Mapping[str, tuple[str, ...]] = {
    "wrist_contortion": ("wrist_pitch_delta", "wrist_yaw_delta", "wrist_roll_delta"),
    "arm_contortion": ("arm_height_delta", "arm_depth_delta", "elbow_swivel_delta"),
    "finger_shape_error": ("finger_splay_delta", "thumb_curl_delta", "little_curl_delta"),
    "hand_cropped": ("arm_height_delta", "arm_depth_delta", "lateral_offset_delta"),
    "timing_error": (
        "present_duration_scale",
        "hold_duration_scale",
        "shake_duration_scale",
        "recover_duration_scale",
        "easing_delta",
    ),
    "wrong_strike_path": ("path_arc_delta", "torso_participation_delta"),
    "weak_lift": ("object_apex_scale", "object_contact_height_delta"),
    "object_slip": ("object_contact_height_delta", "object_contact_depth_delta"),
}

def _neutral_patch() -> dict[str, Any]:
    """A schema-valid `RepairPatch` that changes nothing.

    Derived from the model's own fields rather than hand-listed, so a new delta
    cannot leave the control patches silently invalid — a control that fails
    validation is a control that never runs.
    """
    patch: dict[str, Any] = {}
    for name, field in RepairPatch.model_fields.items():
        if field.default is not PydanticUndefined:
            patch[name] = field.default
        elif field.annotation is str:
            patch[name] = "Synthesized control patch."
        else:
            patch[name] = 1.0 if name.endswith("_scale") else 0.0
    return patch


def _neutral_value(field: str) -> float:
    return 1.0 if field.endswith("_scale") else 0.0


def relevant_patch(failure_tag: str, *, magnitude: float = 0.08) -> dict[str, Any]:
    """A patch that moves parameters which could address `failure_tag`."""
    patch = _neutral_patch()
    for field in FAILURE_RELEVANT_DELTAS.get(failure_tag, ()):
        patch[field] = _neutral_value(field) + magnitude
    return patch


def irrelevant_patch(failure_tag: str, *, magnitude: float = 0.08) -> dict[str, Any]:
    """A patch that moves only parameters which cannot address `failure_tag`.

    The negative control for repair soundness.  Without it, a grader that answers
    `responsive: true` unconditionally scores perfectly on any set of real repair
    attempts, because real attempts are mostly responsive.
    """
    relevant = set(FAILURE_RELEVANT_DELTAS.get(failure_tag, ()))
    candidates = [
        field
        for fields in FAILURE_RELEVANT_DELTAS.values()
        for field in fields
        if field not in relevant
    ]
    if not candidates:
        raise ValueError(f"no irrelevant deltas available for {failure_tag}")
    patch = _neutral_patch()
    for field in sorted(set(candidates))[:3]:
        patch[field] = _neutral_value(field) + magnitude
    return patch


def inert_patch() -> dict[str, Any]:
    """A schema-valid patch of all-neutral values. Addresses nothing, by construction.

    A distinct control from `irrelevant_patch`: a grader may well recognise that
    an all-zero patch does nothing while still being fooled by one that changes
    the wrong parameters confidently.
    """
    return _neutral_patch()


@dataclass(frozen=True)
class RepairSoundnessOutcome:
    """One synthesized patch, its label, and what the grader said about it."""

    case_id: str
    #: Ground truth: the patch was built to address the cited failure.
    is_relevant: bool
    #: What the grader answered for `responsive`.
    judged_responsive: bool


def score_repair_soundness(outcomes: Sequence[RepairSoundnessOutcome]) -> dict[str, Any]:
    """How well the grader separates relevant patches from synthesized irrelevant ones.

    Reported as sensitivity and specificity rather than one accuracy figure: a
    grader that answers `responsive: true` unconditionally has perfect
    sensitivity and zero specificity, and a single pooled number hides that
    completely.  The baseline is the majority class, which is what such a grader
    actually scores.
    """
    relevant = [outcome for outcome in outcomes if outcome.is_relevant]
    irrelevant = [outcome for outcome in outcomes if not outcome.is_relevant]
    labels = [outcome.is_relevant for outcome in outcomes]
    return {
        "sensitivity": proportion(
            sum(outcome.judged_responsive for outcome in relevant), len(relevant)
        ),
        "specificity": proportion(
            sum(not outcome.judged_responsive for outcome in irrelevant), len(irrelevant)
        ),
        "baseline": majority_class_baseline(labels) if labels else None,
        "n_relevant": len(relevant),
        "n_irrelevant": len(irrelevant),
    }
