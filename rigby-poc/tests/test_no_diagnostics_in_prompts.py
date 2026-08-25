"""No deterministic diagnostic reaches any model prompt. Plan 07 §1.3, §3.3.

The judge received `motion_diagnostics` alongside the images on every path —
unary, pairwise and five-way — and the prompts instructed the model on how to
treat reference bounds. That leaks the deterministic layer's verdict into a
supposedly independent perceptual signal: it encourages rubber-stamping, and it
makes the VLM's actual perceptual ability unmeasurable, because you cannot tell
whether it saw a bent wrist or read a number about one.

Asserted at string level against the *assembled payload*, for every path, so a
reintroduction has to defeat the assertion rather than slip past a review.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rigby_poc.judge import (
    FIVE_WAY_SYSTEM_PROMPT,
    PAIRWISE_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    UNARY_SYSTEM_PROMPT,
    VLMJudge,
)

pytestmark = pytest.mark.medium


SENTINEL_KEY = "max_forearm_twist_rad"
SENTINEL_VALUE = 0.987654321


def _manifest(tmp_path: Path, result_id: str = "diag-result") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(7 * index, 15, 33)).save(buffer, format="PNG")
        image = buffer.getvalue()
        (tmp_path / f"{view}.png").write_bytes(image)
        snapshots.append(
            {
                "id": f"01-{view}",
                "phase": "hold",
                "label": "presented_pose",
                "view": view,
                "requested_time_s": 0.5,
                "rendered_time_s": 0.5,
                "path": f"{view}.png",
                "sha256": hashlib.sha256(image).hexdigest(),
                "width_px": 1600,
                "height_px": 900,
                "camera": {
                    "view": view,
                    "width_px": 1600,
                    "height_px": 900,
                    "vertical_fov_deg": 94 if view == "ego" else 46,
                },
            }
        )
    path = tmp_path / "evidence-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "result_id": result_id,
                "intent": "gesture",
                "prompt": "Throw up a hang-ten sign.",
                # Present in the manifest, and must not reach any prompt.
                "motion_diagnostics": {SENTINEL_KEY: SENTINEL_VALUE},
                "capture_contract": {
                    "raw_canvas_only": True,
                    "width_px": 1600,
                    "height_px": 900,
                    "egocentric_vertical_fov_deg": 94.0,
                    "device_pixel_ratio": 1,
                    "ui_overlay_included": False,
                },
                "snapshots": snapshots,
            }
        ),
        encoding="utf-8",
    )
    return path


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["text_format"]
        system = kwargs["input"][0]["content"][0]["text"]
        payload = _stub_for(model, system)
        return SimpleNamespace(
            id="resp", model=kwargs["model"], usage=None, output_parsed=payload
        )


def _stub_for(model: type, system: str = ""):
    from rigby_poc.judge import (
        FiveWayJudgeDecision,
        GraderClaims,
        MotionJudgeScore,
        PairwiseJudgeDecision,
        SemanticGraderClaims,
    )
    from rigby_poc.judge_claims import claim_specs
    from rigby_poc.judge_prompts import GRADER_NAMES

    if model in (GraderClaims, SemanticGraderClaims):
        # Keyed off the system prompt: `GraderClaims` validates any claim list,
        # so returning the first grader's answers for all five would replay one
        # grader's claims against another's questions -- the exact defect the
        # replay store's own test rules out.
        from rigby_poc.judge_prompts import GRADER_CORES

        name = next(n for n in GRADER_NAMES if system.startswith(GRADER_CORES[n][:60]))
        body: dict[str, object] = {
            "claims": [
                {"id": spec.id, "verdict": "yes", "confidence": 0.9, "snapshot_id": "01-ego"}
                for spec in claim_specs(name, intent="gesture")
            ],
            "summary": "Nothing notable.",
        }
        if model is SemanticGraderClaims:
            body["suggested_adjustment"] = "Preserve the pose."
        return model.model_validate(body)
    if model is MotionJudgeScore:
        return MotionJudgeScore.model_validate(
            {
                "semantic_match": 4, "gesture_recognizability": 4,
                "anatomical_naturalness": 4, "temporal_readability": 4,
                "egocentric_visibility": 4, "cross_view_consistency": 4,
                "overall": 4, "accept": True, "confidence": 0.9,
                "failure_tags": ["none"],
                "evidence": [
                    {"snapshot_id": "01-ego", "observation": "Visible."},
                    {"snapshot_id": "01-orbit", "observation": "Natural."},
                ],
                "summary": "Fine.", "suggested_adjustment": "None.",
            }
        )
    if model is PairwiseJudgeDecision:
        return PairwiseJudgeDecision.model_validate(
            {
                "winner": "A", "confidence": 0.9, "semantic_winner": "A",
                "anatomy_winner": "A", "timing_winner": "tie", "visibility_winner": "tie",
                "evidence": [
                    {"snapshot_id": "A-01-ego", "observation": "A reads better."},
                    {"snapshot_id": "B-01-ego", "observation": "B is flatter."},
                ],
                "rationale": "A is clearer.",
            }
        )
    return FiveWayJudgeDecision.model_validate(
        {
            "candidates": [
                {
                    "label": label, "semantic_match": 4, "gesture_recognizability": 4,
                    "anatomical_naturalness": 4, "temporal_readability": 4,
                    "egocentric_visibility": 4, "overall": 4, "accept": True,
                    "failure_tags": ["none"], "summary": f"{label} is fine.",
                    "suggested_adjustment": "None.",
                }
                for label in "ABCDE"
            ],
            "winner": "C", "confidence": 0.9,
            "evidence": [
                {"snapshot_id": "A-01-ego", "observation": "A is visible."},
                {"snapshot_id": "C-01-ego", "observation": "C is natural."},
                {"snapshot_id": "E-01-ego", "observation": "E is readable."},
            ],
            "rationale": "C wins.",
        }
    )


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


def _payload_text(client: _Client) -> str:
    return json.dumps([call["input"] for call in client.responses.calls])


# ------------------------------------------------------- the assembled payloads


def test_the_split_path_ships_no_diagnostics(tmp_path: Path) -> None:
    client = _Client()
    VLMJudge(client=client, model="m", grader_mode="split").score(_manifest(tmp_path))
    text = _payload_text(client)
    assert SENTINEL_KEY not in text
    assert str(SENTINEL_VALUE) not in text


def test_the_combined_path_ships_no_diagnostics(tmp_path: Path) -> None:
    client = _Client()
    VLMJudge(client=client, model="m").score_combined(_manifest(tmp_path))
    text = _payload_text(client)
    assert SENTINEL_KEY not in text
    assert str(SENTINEL_VALUE) not in text


def test_the_pairwise_path_ships_no_diagnostics(tmp_path: Path) -> None:
    client = _Client()
    VLMJudge(client=client, model="m").compare(
        _manifest(tmp_path / "a", "a"), _manifest(tmp_path / "b", "b"), random_seed=1
    )
    text = _payload_text(client)
    assert SENTINEL_KEY not in text
    assert str(SENTINEL_VALUE) not in text


def test_the_five_way_path_ships_no_diagnostics(tmp_path: Path) -> None:
    client = _Client()
    VLMJudge(client=client, model="m").rank_five(
        [_manifest(tmp_path / str(i), f"c{i}") for i in range(5)], random_seed=1
    )
    text = _payload_text(client)
    assert SENTINEL_KEY not in text
    assert str(SENTINEL_VALUE) not in text


# ------------------------------------------------------------- the prompt text


@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("unary", UNARY_SYSTEM_PROMPT),
        ("pairwise", PAIRWISE_SYSTEM_PROMPT),
        ("five_way", FIVE_WAY_SYSTEM_PROMPT),
        ("repair", REPAIR_SYSTEM_PROMPT),
    ],
)
def test_no_prompt_instructs_the_model_on_diagnostics(name: str, prompt: str) -> None:
    """A prompt that explains how to weigh a measurement the model no longer
    receives is worse than one that never mentioned it: it invites the model to
    look for evidence that is not there.

    The five-way prompt keeps one mention — that a preference should be visible
    to a human *without access to* parameters or diagnostics — which reinforces
    the blinding rather than instructing on their use.
    """
    instructing = re.findall(r"[^.]*\bdiagnostics\b[^.]*\.", prompt, flags=re.I)
    allowed = "without access to parameters or diagnostics"
    offending = [s for s in instructing if allowed not in s]
    assert offending == [], (name, offending)


def test_the_manifest_still_carries_diagnostics_for_the_decision_layer(tmp_path: Path) -> None:
    """Removed from the *prompt*, not from the evidence.

    `deterministic_valid` is still the authority in §3.3's combination; what
    changed is that the graders no longer read it. Deleting it from the manifest
    would remove the term the decision layer depends on.
    """
    manifest = json.loads(_manifest(tmp_path).read_text(encoding="utf-8"))
    assert manifest["motion_diagnostics"][SENTINEL_KEY] == SENTINEL_VALUE
