"""The repair stage, actually driven — plan 10 section 6.

No test in this suite has ever caused a repair round to run. `tests/test_run_transcript.py`
passes `max_rounds=1` and repair happens *between* rounds, so the `repair` stage span is
emitted by nothing, and `EXPECTED_STAGES` there lists the six that do fire. That set is
therefore a restatement of its fixture rather than a claim about the pipeline, and it is
asserted with `issubset`, so it cannot detect the seventh going missing either.

That matters here specifically: plan 10 section 6 names **repair efficacy** as a metric and
**repair-exhaustion** as a stage-attribution category, so the one stage with no coverage
anywhere is the one this PR depends on most.

This file drives a real repair round with a rejecting judge and zero model calls, and
asserts all seven stages appear. Separate from `test_trajectory.py` because it runs live
compiles and so is `medium`, and the tier guard allows one `pytestmark` per file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

import evals.flywheel as flywheel
from evals.trajectory import PIPELINE_STAGES, stage_attribution
from rigby_poc.observability import Tracer
from rigby_poc.transcript import load


#: live compiles, no browser, no model calls -- see docs/testing.md
pytestmark = pytest.mark.medium

PROMPT = "Throw a right jab."

_REJECTED = {
    "label": "A",
    "semantic_match": 2,
    "gesture_recognizability": 2,
    "anatomical_naturalness": 2,
    "temporal_readability": 2,
    "egocentric_visibility": 2,
    "overall": 2,
    "accept": False,
    "failure_tags": ["anatomy"],
    "summary": "Rejected so the round falls through to repair.",
    "suggested_adjustment": "Lower the arm.",
}

#: A minimal in-bounds patch. Only `arm_height_delta` is non-zero, so the repaired program
#: differs from its source in exactly one axis and the round is reproducible.
_PATCH = {
    "rationale": "Lower the arm to clear the shoulder limit.",
    "arm_height_delta": 0.05,
    "arm_depth_delta": 0.0,
    "lateral_offset_delta": 0.0,
    "wrist_pitch_delta": 0.0,
    "wrist_yaw_delta": 0.0,
    "wrist_roll_delta": 0.0,
    "elbow_swivel_delta": 0.0,
    "finger_splay_delta": 0.0,
    "thumb_curl_delta": 0.0,
    "little_curl_delta": 0.0,
    "wrist_shake_amplitude_delta": 0.0,
    "present_duration_scale": 1.0,
    "hold_duration_scale": 1.0,
    "shake_duration_scale": 1.0,
    "recover_duration_scale": 1.0,
    "easing_delta": 0.0,
}


class _RejectingJudge:
    """Rejects every candidate and proposes a fixed patch. Spends no model call."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.model = "fake"

    def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict[str, Any]:
        # Every index must be present: `_run_best_of_five` raises on a ranking that is
        # missing a candidate, so an empty ranking cannot be used to force this path.
        return {
            "kind": "five_way_motion_judgment",
            "mapped_scores": {index: dict(_REJECTED, label="ABCDE"[index]) for index in range(5)},
            "mapped_winner_index": None,
            "call": {"response_id": "fake-ranking"},
            "routing": {"selected_model": "fake"},
        }

    def recommend_repair(self, **_: Any) -> dict[str, Any]:
        return {
            "kind": "repair",
            "call": {"response_id": "fake-repair", "parsed": _PATCH},
            "routing": {"selected_model": "fake"},
        }


def _fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "evidence-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return manifest


@pytest.fixture(scope="module")
def repaired_run() -> Any:
    """A two-round run whose first round is rejected outright, so repair must fire."""
    original = flywheel.VLMJudge
    flywheel.VLMJudge = _RejectingJudge  # type: ignore[misc]
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracer = Tracer.open(root / "runs")
            with tracer.span("run", "pipeline.run", prompt=PROMPT, launched_by="cli"):
                flywheel.run_best_of_five(
                    PROMPT,
                    root / "artifacts",
                    provider="offline",
                    selection_mode="five_way",
                    max_rounds=2,
                    capture_fn=_fake_capture,
                    tracer=tracer,
                )
            yield load(tracer.run_dir)
    finally:
        flywheel.VLMJudge = original  # type: ignore[misc]


def test_a_rejected_round_emits_the_repair_stage(repaired_run: Any) -> None:
    """The assertion no existing test could make, because none reached the repair path."""
    emitted = set(repaired_run.duration_by_stage())
    assert "repair" in emitted, (
        f"a rejected round must emit the repair stage; emitted {sorted(emitted)}"
    )


def test_all_seven_declared_stages_appear_in_one_run(repaired_run: Any) -> None:
    """Equality against the declared list, which is itself pinned to flywheel's literals.

    `test_run_transcript.py` asserts six with `issubset`; this asserts all seven are
    present, which is only reachable on a run that repairs.
    """

    emitted = set(repaired_run.duration_by_stage())
    assert set(PIPELINE_STAGES) == emitted, (
        f"declared {sorted(PIPELINE_STAGES)} but emitted {sorted(emitted)}"
    )


def test_stage_attribution_reports_nothing_missing_on_a_repaired_run(repaired_run: Any) -> None:
    """`missing` is empty only when every declared stage actually ran."""
    attribution = stage_attribution(repaired_run)
    assert attribution.missing == ()
    assert attribution.occurrences["repair"] >= 1
    assert attribution.total_ms > 0.0
