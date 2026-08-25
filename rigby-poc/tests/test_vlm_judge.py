from __future__ import annotations

import hashlib
import io
import json
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rigby_poc.judge import (
    FiveWayJudgeDecision,
    MotionJudgeScore,
    PairwiseJudgeDecision,
    VLMJudge,
)

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


class FakeUsage:
    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["text_format"] is FiveWayJudgeDecision:
            output = FiveWayJudgeDecision(
                candidates=[
                    {
                        "label": label,
                        "semantic_match": 4,
                        "gesture_recognizability": 4,
                        "anatomical_naturalness": 4,
                        "temporal_readability": 4,
                        "egocentric_visibility": 5,
                        "overall": 4,
                        "accept": True,
                        "failure_tags": ["none"],
                        "summary": f"Candidate {label} is acceptable.",
                        "suggested_adjustment": "Preserve the pose.",
                    }
                    for label in "ABCDE"
                ],
                winner="C",
                confidence=0.84,
                evidence=[
                    {"snapshot_id": "A-01-ego", "observation": "Candidate A is visible."},
                    {"snapshot_id": "C-01-orbit", "observation": "Candidate C is more natural."},
                    {"snapshot_id": "E-01-ego", "observation": "Candidate E is readable."},
                ],
                rationale="Candidate C has the best combined silhouette and anatomy.",
            )
            return SimpleNamespace(
                id="resp_rank_five",
                model=kwargs["model"],
                usage=FakeUsage(),
                output_parsed=output,
            )
        if kwargs["text_format"] is PairwiseJudgeDecision:
            pair_call = sum(call["text_format"] is PairwiseJudgeDecision for call in self.calls)
            output = PairwiseJudgeDecision(
                winner="A" if pair_call == 1 else "B",
                confidence=0.88,
                semantic_winner="A" if pair_call == 1 else "B",
                anatomy_winner="A" if pair_call == 1 else "B",
                timing_winner="tie",
                visibility_winner="tie",
                evidence=[
                    {"snapshot_id": "A-01-ego", "observation": "Candidate A is readable."},
                    {"snapshot_id": "B-01-orbit", "observation": "Candidate B is less natural."},
                ],
                rationale="The same candidate wins after order reversal.",
            )
            return SimpleNamespace(
                id=f"resp_pair_{pair_call}",
                model=kwargs["model"],
                usage=FakeUsage(),
                output_parsed=output,
            )
        output = MotionJudgeScore(
            semantic_match=4,
            gesture_recognizability=4,
            anatomical_naturalness=4,
            temporal_readability=4,
            egocentric_visibility=5,
            cross_view_consistency=4,
            overall=4,
            accept=True,
            confidence=0.82,
            failure_tags=["none"],
            evidence=[
                {"snapshot_id": "01-ego", "observation": "The active hand is fully visible."},
                {"snapshot_id": "02-orbit", "observation": "The arm remains anatomically readable."},
            ],
            summary="The gesture is recognizable and natural.",
            suggested_adjustment="Preserve the current pose.",
        )
        return SimpleNamespace(
            id="resp_test",
            model=kwargs["model"],
            usage=FakeUsage(),
            output_parsed=output,
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class LowConfidenceLunaResponses(FakeResponses):
    def parse(self, **kwargs):
        response = super().parse(**kwargs)
        if kwargs["model"] == "gpt-5.6-luna":
            response.output_parsed.confidence = 0.40
        return response


class LowConfidenceLunaClient:
    def __init__(self) -> None:
        self.responses = LowConfidenceLunaResponses()


class APIConnectionError(RuntimeError):
    pass


class TransientOnceResponses(FakeResponses):
    def parse(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            raise APIConnectionError("temporary provider connection failure")
        return super().parse(**kwargs)


class TransientOnceClient:
    def __init__(self) -> None:
        self.responses = TransientOnceResponses()


def _manifest(tmp_path: Path, *, bad_fov: bool = False, result_id: str = "test-result") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        image_buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(20 * index, 40, 80)).save(
            image_buffer, format="PNG"
        )
        image = image_buffer.getvalue()
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
                    "vertical_fov_deg": 80 if view == "ego" and bad_fov else (94 if view == "ego" else 46),
                },
            }
        )
    path = tmp_path / "evidence-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result_id": result_id,
                "prompt": "Throw up a hang-ten sign.",
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


def test_unary_judge_sends_resized_uncropped_full_fov_images_and_keeps_usage(tmp_path: Path) -> None:
    client = FakeClient()
    record = VLMJudge(client=client, model="test-vlm").score(_manifest(tmp_path))
    assert record["call"]["usage"]["input_tokens"] == 123
    assert record["call"]["parsed"]["accept"] is True
    call = client.responses.calls[0]
    image_items = [
        item
        for message in call["input"]
        for item in message["content"]
        if item["type"] == "input_image"
    ]
    assert len(image_items) == 4
    assert all(item["detail"] == "high" for item in image_items)
    for item in image_items[:2]:
        encoded = item["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as payload:
            assert payload.size == (960, 540)
    assert record["judge_evidence_contract"]["full_frame_uncropped"] is True
    assert record["judge_evidence_contract"]["payloads"][0]["payload_dimensions_px"] == [960, 540]
    assert record["judge_evidence_contract"]["payloads"][2]["full_frame_uncropped_tiles"] is True
    assert call["reasoning"] == {"effort": "low"}
    user_payload = json.dumps(call["input"][1])
    assert "variation_seed" not in user_payload.lower()
    assert "result_id" not in user_payload.lower()
    assert "test-result" not in user_payload.lower()


def test_judge_refuses_egocentric_evidence_with_reduced_fov(tmp_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="complete 94-degree FOV"):
        VLMJudge(client=client, model="test-vlm").score(_manifest(tmp_path, bad_fov=True))
    assert client.responses.calls == []


def test_pairwise_reverse_order_maps_to_the_same_candidate(tmp_path: Path) -> None:
    client = FakeClient()
    first = _manifest(tmp_path / "first", result_id="first-result")
    second = _manifest(tmp_path / "second", result_id="second-result")
    record = VLMJudge(client=client, model="test-vlm").compare(
        first,
        second,
        random_seed=42,
        reverse_check=True,
    )
    assert record["order_consistent"] is True
    assert record["calls"][0]["mapped_winner"] == record["calls"][1]["mapped_winner"]
    for call in client.responses.calls:
        serialized = json.dumps(call["input"])
        assert "first-result" not in serialized
        assert "second-result" not in serialized


def test_five_way_judge_randomizes_labels_and_maps_winner_back(tmp_path: Path) -> None:
    client = FakeClient()
    manifests = [
        _manifest(tmp_path / str(index), result_id=f"candidate-{index}")
        for index in range(5)
    ]
    record = VLMJudge(client=client, model="test-vlm").rank_five(manifests, random_seed=99)
    winner = record["mapped_winner_index"]
    assert isinstance(winner, int) and 0 <= winner < 5
    assert set(record["mapped_scores"]) == set(range(5))
    call = client.responses.calls[0]
    image_items = [
        item
        for message in call["input"]
        for item in message["content"]
        if item["type"] == "input_image"
    ]
    assert len(image_items) == 20
    assert all(item["detail"] == "high" for item in image_items)


def test_low_confidence_luna_judgment_escalates_once_to_terra(tmp_path: Path) -> None:
    client = LowConfidenceLunaClient()
    record = VLMJudge(
        client=client,
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
    ).score(_manifest(tmp_path))
    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    assert record["routing"]["escalated"] is True
    assert record["routing"]["escalation_reason"] == "low_confidence"
    assert record["routing"]["attempt_count"] == 2
    assert record["call"]["model"] == "gpt-5.6-terra"


def test_hard_model_call_budget_skips_fallback_without_losing_primary(tmp_path: Path) -> None:
    client = LowConfidenceLunaClient()
    judge = VLMJudge(
        client=client,
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
        max_model_calls=1,
    )
    record = judge.score(_manifest(tmp_path))
    assert [call["model"] for call in client.responses.calls] == ["gpt-5.6-luna"]
    assert record["routing"]["escalated"] is False
    assert record["routing"]["escalation_requested_reason"] == "low_confidence"
    assert record["routing"]["escalation_skipped_reason"] == "model_call_budget_exhausted"
    assert record["routing"]["attempt_count"] == 1
    assert judge.model_calls_made == 1
    assert judge.remaining_model_calls == 0


def test_transient_provider_error_retries_lightweight_judge_once(tmp_path: Path) -> None:
    client = TransientOnceClient()
    judge = VLMJudge(
        client=client,
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
        max_model_calls=2,
    )

    record = judge.score(_manifest(tmp_path))

    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-luna",
        "gpt-5.6-luna",
    ]
    assert record["routing"]["dispatch_count"] == 2
    assert record["routing"]["transient_retry_count"] == 1
    assert record["routing"]["escalated"] is False
    assert judge.remaining_model_calls == 0


def test_pairwise_lightweight_route_escalates_each_low_confidence_order(tmp_path: Path) -> None:
    client = LowConfidenceLunaClient()
    first = _manifest(tmp_path / "first", result_id="first-result")
    second = _manifest(tmp_path / "second", result_id="second-result")
    record = VLMJudge(
        client=client,
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
    ).compare(
        first,
        second,
        random_seed=7,
        reverse_check=True,
        routing_policy="lightweight_routed",
    )
    assert record["routing_policy"] == "lightweight_routed"
    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    assert all(call["routing"]["escalated"] for call in record["calls"])


def test_pairwise_lightweight_only_has_exactly_one_call_per_order(tmp_path: Path) -> None:
    client = LowConfidenceLunaClient()
    first = _manifest(tmp_path / "first", result_id="first-result")
    second = _manifest(tmp_path / "second", result_id="second-result")
    record = VLMJudge(
        client=client,
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
        max_model_calls=2,
    ).compare(
        first,
        second,
        random_seed=7,
        reverse_check=True,
        routing_policy="lightweight_only",
    )
    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-luna",
        "gpt-5.6-luna",
    ]
    assert all(call["routing"]["escalated"] is False for call in record["calls"])
    assert all(call["routing"]["attempt_count"] == 1 for call in record["calls"])
