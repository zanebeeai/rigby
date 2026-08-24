"""The carry-over contract a mutated clip has to obey.

A mutation perturbs a compiled clip and recompiles nothing, so ``clip.metrics``
afterwards is part fresh measurement, part authoring intent, and part stale
observation. These pin the boundary between the three, because getting it wrong
is silent: a stale number is a plausible number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_poc import analysis
from rigby_poc.analysis.equivalence import (
    CARRIED_AUTHORING_INTENT,
    UNWRITTEN_BASE_DEFAULTS,
    MutationSafeMetrics,
    StaleMetricError,
    stale_after_mutation,
)
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, MotionProgram, SceneManifest


# Every analysis test compiles or reads a compiled clip (plan 09 §3.3 tiering).
pytestmark = pytest.mark.medium


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "analysis_equivalence"


def _load(case_id: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def _compile(case_id: str):
    case = _load(case_id)
    scene = SceneManifest.model_validate(case["scene"])
    program = MotionProgram.model_validate(case["program"])
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    return scene, program, clip


class TestTheThreeKinds:
    def test_a_recomputed_key_is_readable(self) -> None:
        """Anything the analysis layer owns is recomputed, so it is never stale."""

        _, program, clip = _compile("full_body_walk")
        safe = MutationSafeMetrics(clip.metrics, program)

        for key in analysis.analyze(clip, program, _compile("full_body_walk")[0]):
            assert safe[key] == clip.metrics[key]

    def test_commanded_ik_targets_survive_a_mutation(self) -> None:
        """The point of persisting them: a mutation must not move what was asked for.

        If ``support_constraints`` were invalidated alongside the frames, a
        mutation that drags a foot off its commanded target would have nothing
        left to be measured against, and the defect would become undetectable.
        """

        _, program, clip = _compile("full_body_walk")
        safe = MutationSafeMetrics(clip.metrics, program)

        assert safe["support_constraints"] == clip.metrics["support_constraints"]
        assert "support_constraints" not in safe.stale

    def test_a_deferred_observation_raises_rather_than_returning(self) -> None:
        """An object clip's lifecycle metrics describe motion that has moved on."""

        _, program, clip = _compile("object_throw_forward")
        safe = MutationSafeMetrics(clip.metrics, program)

        assert "object_flight_distance_m" in safe.stale
        with pytest.raises(StaleMetricError, match="object_flight_distance_m"):
            safe["object_flight_distance_m"]


class TestTheFalseNegativeThisPrevents:
    @pytest.mark.parametrize("case_id", ["object_throw_forward", "sequence_catch_then_turn"])
    def test_the_structural_verdict_itself_is_stale(self, case_id: str) -> None:
        """The case that would have poisoned the detection matrix.

        ``structural_valid`` is deferred to the compiler for these two intents,
        so a mutated clip inherits the clean compile's verdict. A matrix scores
        that as "the targeted check did not fire" — the exact signature of a
        genuine coverage gap. The harness would manufacture the finding it
        exists to detect.
        """

        _, program, clip = _compile(case_id)
        safe = MutationSafeMetrics(clip.metrics, program)

        assert "structural_valid" in clip.metrics
        assert "structural_valid" in safe.stale
        with pytest.raises(StaleMetricError):
            safe["structural_valid"]

    def test_a_ported_path_keeps_its_verdict_readable(self) -> None:
        """Whole body was ported in 02b, so its verdict is recomputed, not cached."""

        _, program, clip = _compile("full_body_walk")
        safe = MutationSafeMetrics(clip.metrics, program)

        assert "structural_valid" not in safe.stale
        assert safe["structural_valid"] == clip.metrics["structural_valid"]


class TestTheLedgerCannotRot:
    def test_the_stale_set_is_derived_from_what_analysis_owns(self) -> None:
        """Not a hand-maintained list — it shrinks on its own as paths are ported.

        A listed set would drift out of step with the registry silently, and in
        the dangerous direction: re-admitting a key nobody recomputes.
        """

        _, program, clip = _compile("object_throw_forward")

        stale = stale_after_mutation(clip.metrics, program)
        owned = analysis.owned_metric_keys(program)

        assert not stale & owned
        assert not stale & CARRIED_AUTHORING_INTENT
        assert stale | owned | (CARRIED_AUTHORING_INTENT & set(clip.metrics)) == set(
            clip.metrics
        )

    def test_porting_a_path_withholds_only_the_unwritten_base_defaults(self) -> None:
        """Whole body is fully ported, so nothing it *measures* is withheld.

        What remains is a separate problem this ledger surfaced rather than
        caused: ``compiler._base_metrics`` seeds every clip with nine defaults,
        and a whole-body compile writes none of them. They are not stale — they
        were never measured at all. Withholding them is right for a mutated
        clip, but they are equally meaningless on an unmutated one, which is a
        live defect belonging to plan 08 rather than to this PR.
        """

        _, program, clip = _compile("full_body_climb")

        stale = stale_after_mutation(clip.metrics, program)

        assert stale <= UNWRITTEN_BASE_DEFAULTS
        assert not stale & analysis.owned_metric_keys(program)

    def test_the_withheld_set_is_reportable(self) -> None:
        """A detection matrix must say "no detector exists", not "zero detections".

        Those are different claims and only one of them is true. Exposing the
        withheld keys is what lets that region be rendered honestly rather than
        as a row of zeros.
        """

        _, program, clip = _compile("object_throw_forward")
        safe = MutationSafeMetrics(clip.metrics, program)

        assert safe.stale, "an unported path must report what it cannot measure"
        assert set(safe) == set(clip.metrics) - safe.stale
        assert len(safe) == len(clip.metrics) - len(safe.stale)


def test_the_base_metric_defaults_are_published_unmeasured() -> None:
    """``_base_metrics`` seeds nine values that most compile paths never write.

    ``max_penetration_m: 0.0`` on a whole-body clip reads as "verified no
    penetration"; nothing looked. ``lost_table_contact: True`` reads as a
    finding; there is no table. ``foot_drift_m: 0.0`` is the one lane `infra`
    found comparing against an acceptance criterion of 0.001 — a gate that has
    never been able to fail.

    It is not one key. It is one constructor, and which of its nine are real
    depends on the compile path. Pinned here so the scope of the problem is a
    number rather than an anecdote; the fix is plan 08's, not this PR's.
    """

    _, program, clip = _compile("full_body_climb")

    untouched = {
        key: clip.metrics[key]
        for key in UNWRITTEN_BASE_DEFAULTS
        if key in clip.metrics
    }

    assert untouched == {
        "finger_assertions": {},
        "hold_duration_s": 0.0,
        "lift_height_m": 0.0,
        "lost_table_contact": True,
        "max_penetration_m": 0.0,
        "opposing_contacts": False,
        "palm_relative_slip_m": 0.0,
        "unresolved_non_hand_collisions": 0,
        "vertical_drift_m": 0.0,
        "weld_used": False,
    }
