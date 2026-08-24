"""The split-grader path: blinding, per-family fragments, and aggregation.

Plan 07 §3.1 splits one 115-line mega-prompt into five single-purpose graders so
each dimension can be calibrated on its own, and §3.5 gives every grader prompt a
version and a content hash.  This file is `test_grader_isolation.py` and
`test_prompt_version_recorded.py` from the plan's §5 test list, plus the
aggregation guard for the provisional score merge that 07c replaces.

The isolation assertions are string-level on the assembled payload on purpose:
prompt or diagnostic leakage is the failure mode that makes the graders
unmeasurable, and it creeps back in silently.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rigby_poc.judge import (
    GRADER_OUTPUT_MODELS,
    OVERALL_SOURCE_DIMENSIONS,
    UNARY_SYSTEM_PROMPT,
    VLMJudge,
    aggregate_grader_dimensions,
    assemble_split_score,
)
from rigby_poc.judge_claims import claim_specs
from rigby_poc.judge_prompts import (
    FALLBACK_FAMILY,
    GRADER_CORES,
    FAMILY_FRAGMENTS,
    FAMILY_NAMES,
    GRADER_NAMES,
    GRADER_SPECS,
    family_for_intent,
    grader_prompt,
)
from rigby_poc.models import Intent


PROMPT_TEXT = "Throw a left hook then step over the box."
DIAGNOSTIC_SENTINEL = 0.1234567
SENTINEL_KEY = "max_wrist_twist_rad"


class FakeUsage:
    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def _grader_payload(
    grader: str, *, intent: str, verdict: str = "yes", confidence: float = 0.9
) -> dict:
    payload: dict[str, object] = {
        "claims": [
            {
                "id": spec.id,
                "verdict": verdict,
                "confidence": confidence,
                "snapshot_id": "01-ego",
            }
            for spec in claim_specs(grader, intent=intent)
        ],
        "summary": "Nothing notable in this dimension.",
    }
    if grader == "semantic":
        payload["suggested_adjustment"] = "Preserve the pose."
    return payload


class FakeResponses:
    """Answers as whichever grader the system prompt belongs to."""

    def __init__(
        self,
        *,
        verdicts: dict[str, str] | None = None,
        intent: str = "strike",
    ) -> None:
        self.calls: list[dict] = []
        self.verdicts = verdicts or {}
        self.intent = intent

    def _grader_of(self, kwargs: dict) -> str | None:
        system = kwargs["input"][0]["content"][0]["text"]
        for name in GRADER_NAMES:
            if system.startswith(GRADER_CORES[name][:60]):
                return name
        return None

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        grader = self._grader_of(kwargs)
        if grader is None:  # combined path
            from rigby_poc.judge import MotionJudgeScore

            output = MotionJudgeScore.model_validate(
                {
                    "semantic_match": 4,
                    "gesture_recognizability": 4,
                    "anatomical_naturalness": 4,
                    "temporal_readability": 4,
                    "egocentric_visibility": 4,
                    "cross_view_consistency": 4,
                    "overall": 4,
                    "accept": True,
                    "confidence": 0.9,
                    "failure_tags": ["none"],
                    "evidence": [
                        {"snapshot_id": "01-ego", "observation": "Visible hand."},
                        {"snapshot_id": "01-orbit", "observation": "Natural arm."},
                    ],
                    "summary": "Combined path judgement.",
                    "suggested_adjustment": "Preserve the pose.",
                }
            )
        else:
            output = GRADER_OUTPUT_MODELS[grader].model_validate(
                _grader_payload(
                    grader,
                    intent=self.intent,
                    verdict=self.verdicts.get(grader, "yes"),
                )
            )
        return SimpleNamespace(
            id=f"resp_{grader or 'combined'}",
            model=kwargs["model"],
            usage=FakeUsage(),
            output_parsed=output,
        )


class FakeClient:
    def __init__(
        self, *, verdicts: dict[str, str] | None = None, intent: str = "strike"
    ) -> None:
        self.responses = FakeResponses(verdicts=verdicts, intent=intent)


def _manifest(tmp_path: Path, *, intent: str = "strike") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(20 * index, 40, 80)).save(buffer, format="PNG")
        image = buffer.getvalue()
        image_path = tmp_path / f"{view}.png"
        image_path.write_bytes(image)
        snapshots.append(
            {
                "id": f"01-{view}",
                "phase": "hold",
                "label": "presented_pose",
                "view": view,
                "requested_time_s": 0.5,
                "rendered_time_s": 0.5,
                "path": image_path.name,
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
                "schema_version": "1.0",
                "result_id": "split-result",
                "intent": intent,
                "prompt": PROMPT_TEXT,
                "motion_diagnostics": {SENTINEL_KEY: DIAGNOSTIC_SENTINEL},
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


def _text_of(call: dict) -> str:
    return "\n".join(
        item["text"]
        for message in call["input"]
        for item in message["content"]
        if item["type"] == "input_text"
    )


# ---------------------------------------------------------------- the flag


def test_combined_path_is_still_the_default(tmp_path: Path) -> None:
    client = FakeClient()
    record = VLMJudge(client=client, model="test-vlm").score(_manifest(tmp_path))
    assert record["kind"] == "unary_motion_judgment"
    assert len(client.responses.calls) == 1
    assert UNARY_SYSTEM_PROMPT in _text_of(client.responses.calls[0])


def test_split_mode_runs_one_call_per_grader(tmp_path: Path) -> None:
    client = FakeClient()
    record = VLMJudge(client=client, model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    assert record["kind"] == "split_motion_judgment"
    assert len(client.responses.calls) == len(GRADER_NAMES)
    assert set(record["graders"]) == set(GRADER_NAMES)
    assert UNARY_SYSTEM_PROMPT not in json.dumps(client.responses.calls[0]["input"])


def test_split_mode_reads_the_environment_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RIGBY_JUDGE_GRADER_MODE", "split")
    judge = VLMJudge(client=FakeClient(), model="test-vlm")
    assert judge.grader_mode == "split"


def test_unknown_grader_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="grader mode"):
        VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="hybrid")


# ------------------------------------------------------------- isolation


@pytest.mark.parametrize("grader", sorted(GRADER_NAMES))
def test_no_grader_payload_contains_deterministic_diagnostics(
    tmp_path: Path, grader: str
) -> None:
    client = FakeClient()
    VLMJudge(client=client, model="test-vlm", grader_mode="split").score(_manifest(tmp_path))
    call = client.responses.calls[list(GRADER_NAMES).index(grader)]
    serialized = json.dumps(call["input"])
    assert SENTINEL_KEY not in serialized
    assert str(DIAGNOSTIC_SENTINEL) not in serialized
    assert "DIAGNOSTIC" not in serialized.upper()


#: Hardcoded on purpose.  Deriving the blinded set from `GRADER_SPECS` would make
#: unblinding a grader *delete* its test instead of failing it — the same silent
#: class of bug as `semantic_match` never gating (plan 07 §1.1).
EXPECTED_SEES_PROMPT: dict[str, bool] = {
    "semantic": True,
    "anatomy": False,
    "artifact": False,
    "timing": False,
    "crossview": True,
}
BLINDED_GRADERS: tuple[str, ...] = ("anatomy", "artifact", "timing")


def test_grader_blinding_policy_matches_the_plan() -> None:
    assert {name: GRADER_SPECS[name].sees_prompt for name in GRADER_NAMES} == EXPECTED_SEES_PROMPT
    assert BLINDED_GRADERS == tuple(
        name for name in GRADER_NAMES if not EXPECTED_SEES_PROMPT[name]
    )
    # G4: no grader, blinded or not, ever sees the deterministic layer's verdict.
    assert not any(GRADER_SPECS[name].sees_diagnostics for name in GRADER_NAMES)


@pytest.mark.parametrize("grader", BLINDED_GRADERS)
def test_blinded_graders_never_receive_the_motion_request(tmp_path: Path, grader: str) -> None:
    client = FakeClient()
    VLMJudge(client=client, model="test-vlm", grader_mode="split").score(_manifest(tmp_path))
    call = client.responses.calls[list(GRADER_NAMES).index(grader)]
    serialized = json.dumps(call["input"])
    assert PROMPT_TEXT not in serialized
    for word in ("hook", "step over", "box"):
        assert word not in serialized.lower()


def test_timing_grader_receives_timelines_only(tmp_path: Path) -> None:
    # `gesture` keys on `presented_pose`, which this fixture carries, so the
    # single-pose views really are available to the graders that take them.
    client = FakeClient(intent="gesture")
    VLMJudge(client=client, model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path, intent="gesture")
    )
    by_grader = dict(zip(GRADER_NAMES, client.responses.calls))
    timing = _text_of(by_grader["timing"])
    assert "CHRONOLOGICAL" in timing
    assert "SNAPSHOT " not in timing
    for name in ("semantic", "anatomy", "artifact"):
        assert "SNAPSHOT " in _text_of(by_grader[name])
    crossview = _text_of(by_grader["crossview"])
    assert "SNAPSHOT " in crossview
    assert "CHRONOLOGICAL" not in crossview


def test_result_id_never_reaches_any_grader(tmp_path: Path) -> None:
    client = FakeClient()
    VLMJudge(client=client, model="test-vlm", grader_mode="split").score(_manifest(tmp_path))
    for call in client.responses.calls:
        assert "split-result" not in json.dumps(call["input"])


# ------------------------------------------------- per-family fragments


def test_every_executable_intent_has_its_own_family_fragment() -> None:
    executable = {intent.value for intent in Intent} - {Intent.UNSUPPORTED.value}
    assert set(FAMILY_NAMES) == executable
    assert set(FAMILY_FRAGMENTS) == executable | {FALLBACK_FAMILY}


@pytest.mark.parametrize("family", sorted(FAMILY_NAMES))
def test_semantic_prompt_carries_only_its_own_family_fragment(family: str) -> None:
    assembled = grader_prompt("semantic", intent=family)
    assert assembled.family == family
    assert FAMILY_FRAGMENTS[family] in assembled.text
    for other, fragment in FAMILY_FRAGMENTS.items():
        if other != family:
            assert fragment not in assembled.text


def test_unknown_intent_falls_back_rather_than_guessing() -> None:
    assert family_for_intent("interpretive_dance") == FALLBACK_FAMILY
    assert family_for_intent(None) == FALLBACK_FAMILY
    assert FAMILY_FRAGMENTS[FALLBACK_FAMILY] in grader_prompt("semantic", intent=None).text


def test_only_the_semantic_grader_is_family_specific() -> None:
    for name in GRADER_NAMES:
        assembled = grader_prompt(name, intent="full_body")
        if name == "semantic":
            assert assembled.family == "full_body"
        else:
            assert assembled.family is None
            assert FAMILY_FRAGMENTS["full_body"] not in assembled.text


def test_each_grader_prompt_is_far_shorter_than_the_mega_prompt() -> None:
    # §1.2: attention dilution across unrelated questions is the point of the
    # split. Measured over every family, so the worst case is the one asserted
    # (`full_body`, which carries the longest fragment): semantic tops out at
    # 0.51 of the mega-prompt and every blinded grader at 0.26 of it.
    mega = len(UNARY_SYSTEM_PROMPT)
    for name in GRADER_NAMES:
        longest = max(
            len(grader_prompt(name, intent=family).text) for family in FAMILY_NAMES
        )
        limit = 0.55 if name == "semantic" else 0.30
        assert longest < mega * limit, (name, longest, mega)


# -------------------------------------------------------- prompt version


def test_every_grader_record_carries_a_prompt_version_and_hash(tmp_path: Path) -> None:
    record = VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    for name in GRADER_NAMES:
        prompt = record["graders"][name]["prompt"]
        assert prompt["grader"] == name
        assert prompt["version"]
        assert len(prompt["sha256"]) == 64
        assert prompt["sha256"] == hashlib.sha256(
            grader_prompt(name, intent="strike").text.encode("utf-8")
        ).hexdigest()
    assert record["prompt_versions"] == {
        name: record["graders"][name]["prompt"] for name in GRADER_NAMES
    }


def test_the_semantic_prompt_hash_moves_with_the_family(tmp_path: Path) -> None:
    strike = grader_prompt("semantic", intent="strike")
    full_body = grader_prompt("semantic", intent="full_body")
    assert strike.sha256 != full_body.sha256
    assert strike.version != full_body.version


# ----------------------------------------------------------- end to end


def test_split_record_is_a_drop_in_for_existing_consumers(tmp_path: Path) -> None:
    # `flywheel.py:2290` and four other call sites read exactly this path.
    record = VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    parsed = record["call"]["parsed"]
    assert parsed["accept"] is True
    assert parsed["overall"] == 5
    assert set(parsed) >= {
        "semantic_match",
        "gesture_recognizability",
        "anatomical_naturalness",
        "temporal_readability",
        "egocentric_visibility",
        "cross_view_consistency",
        "overall",
        "accept",
    }


def test_a_failing_grader_makes_the_derived_score_reject(tmp_path: Path) -> None:
    client = FakeClient(verdicts={"anatomy": "no"})
    record = VLMJudge(client=client, model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    parsed = record["call"]["parsed"]
    assert parsed["anatomical_naturalness"] == 1
    assert parsed["overall"] == 1
    assert parsed["accept"] is False
    assert "wrist_contortion" in parsed["failure_tags"]


def test_dimension_verdicts_are_recorded_alongside_the_legacy_score(tmp_path: Path) -> None:
    record = VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    verdicts = record["dimension_verdicts"]
    assert set(verdicts) == {
        "semantic_match",
        "gesture_recognizability",
        "anatomical_naturalness",
        "temporal_readability",
        "egocentric_visibility",
        "cross_view_consistency",
        "overall",
    }
    assert verdicts["anatomical_naturalness"]["yes"] == len(claim_specs("anatomy"))
    assert record["insufficient_evidence_dimensions"] == []
    assert record["unexpected_claim_ids"] == {}


def test_an_unjudgeable_clip_is_not_accepted_and_says_so(tmp_path: Path) -> None:
    # Every claim `cannot_tell`: distinguishable from a failure, and not accepted.
    client = FakeClient(verdicts=dict.fromkeys(GRADER_NAMES, "cannot_tell"))
    record = VLMJudge(client=client, model="test-vlm", grader_mode="split").score(
        _manifest(tmp_path)
    )
    assert record["call"]["parsed"]["accept"] is False
    assert record["call"]["parsed"]["semantic_match"] == 3
    assert record["dimension_verdicts"]["semantic_match"]["score"] is None
    assert record["dimension_verdicts"]["semantic_match"]["insufficient_evidence"] is True
    assert "semantic_match" in record["insufficient_evidence_dimensions"]


def test_an_unjudgeable_clip_escalates_to_the_stronger_model(tmp_path: Path) -> None:
    client = FakeClient(verdicts=dict.fromkeys(GRADER_NAMES, "cannot_tell"))
    record = VLMJudge(
        client=client,
        model="test-vlm",
        fallback_model="test-vlm-large",
        grader_mode="split",
    ).score(_manifest(tmp_path))
    for name in GRADER_NAMES:
        routing = record["graders"][name]["routing"]
        assert routing["escalation_reason"] == "insufficient_evidence"
        assert routing["escalated"] is True
