from __future__ import annotations

import json
import time
from pathlib import Path

from evals import flywheel
from rigby_poc.models import PipelineRunRequest, default_scene
from rigby_poc.pipeline import PipelineRunStore


def test_pipeline_run_persists_live_events_and_winner(tmp_path: Path, monkeypatch) -> None:
    def fake_best_of_five(prompt: str, output_dir: Path, **kwargs) -> Path:
        callback = kwargs["progress_callback"]
        callback(
            {
                "event": "plan_ready",
                "stage": "planning",
                "message": "Plan ready.",
                "data": {"intent": "gesture"},
            }
        )
        callback(
            {
                "event": "pipeline_finished",
                "stage": "finalize",
                "message": "Final animation ready.",
                "data": {"winner_result_id": "000001-test-motion"},
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "flywheel-trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "status": "winner_selected",
                    "winner_result_id": "000001-test-motion",
                    "selection_mode": "five_way",
                    "rounds": [{"round": 1, "candidates": [{}, {}, {}, {}, {}]}],
                    "repairs": [],
                }
            ),
            encoding="utf-8",
        )
        return trace_path

    monkeypatch.setattr(flywheel, "run_best_of_five", fake_best_of_five)
    store = PipelineRunStore(tmp_path / "pipeline-runs")
    started = store.start(
        PipelineRunRequest(
            text="Throw up a hang-ten sign.",
            scene=default_scene(),
            provider="offline",
            max_rounds=1,
        ),
        base_url="http://127.0.0.1:8000",
    )
    deadline = time.monotonic() + 3.0
    detail = store.get(started["run_id"])
    while detail and detail["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        detail = store.get(started["run_id"])

    assert detail is not None
    assert detail["status"] == "completed"
    assert detail["winner_result_id"] == "000001-test-motion"
    assert [event["event"] for event in detail["events"]] == [
        "run_queued",
        "plan_ready",
        "pipeline_finished",
    ]
    assert detail["summary"]["candidate_count"] == 5
    assert detail["trace"]["status"] == "winner_selected"


def test_pipeline_surfaces_unsupported_motion_without_crashing(tmp_path: Path, monkeypatch) -> None:
    reason = "unsupported motion: kick"

    def fake_best_of_five(prompt: str, output_dir: Path, **kwargs) -> Path:
        kwargs["progress_callback"](
            {
                "event": "pipeline_unsupported",
                "stage": "finalize",
                "message": reason,
                "data": {"reason": reason, "model_calls": 0},
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "flywheel-trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "status": "unsupported_motion",
                    "unsupported_reason": reason,
                    "winner_result_id": None,
                    "selection_mode": "five_way",
                    "rounds": [],
                    "repairs": [],
                }
            ),
            encoding="utf-8",
        )
        return trace_path

    monkeypatch.setattr(flywheel, "run_best_of_five", fake_best_of_five)
    store = PipelineRunStore(tmp_path / "pipeline-runs")
    started = store.start(
        PipelineRunRequest(
            text="do a flying kick",
            scene=default_scene(),
            provider="offline",
            max_rounds=1,
        ),
        base_url="http://127.0.0.1:8000",
    )
    deadline = time.monotonic() + 3.0
    detail = store.get(started["run_id"])
    while detail and detail["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        detail = store.get(started["run_id"])

    assert detail is not None
    assert detail["status"] == "unsupported"
    assert detail["error"] == {"type": "UnsupportedMotion", "message": reason}
    assert detail["summary"]["candidate_count"] == 0
    assert detail["trace"]["status"] == "unsupported_motion"


def test_right_jab_reaches_generalized_five_way_judging_and_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    persisted: list[str] = []
    captured: list[str] = []
    ranked_batches: list[list[Path]] = []
    ranking_seeds: list[int] = []

    class FakeStore:
        def persist(self, request, clip) -> str:  # noqa: ANN001
            result_id = f"candidate-{len(persisted) + 1}"
            persisted.append(result_id)
            return result_id

    class FakeJudge:
        def __init__(self, **_: object) -> None:
            pass

        def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict:
            assert len(manifests) == 5
            ranking_seeds.append(random_seed)
            ranked_batches.append(manifests)
            labels = ("A", "B", "C", "D", "E")
            scores = {
                index: {
                    "label": labels[index],
                    "semantic_match": 5,
                    "gesture_recognizability": 5,
                    "anatomical_naturalness": 5,
                    "temporal_readability": 5,
                    "egocentric_visibility": 5,
                    "overall": 5,
                    "accept": True,
                    "failure_tags": ["none"],
                    "summary": "Recognizable and structurally sound motion.",
                    "suggested_adjustment": "No adjustment needed.",
                }
                for index in range(5)
            }
            return {
                "kind": "five_way_motion_judgment",
                "mapped_scores": scores,
                "mapped_winner_index": 0,
                "call": {"response_id": "fake-ranking"},
                "routing": {"selected_model": "fake"},
            }

    def fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
        captured.append(result_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "evidence-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    monkeypatch.setattr(flywheel, "ResultStore", FakeStore)
    monkeypatch.setattr(flywheel, "VLMJudge", FakeJudge)
    trace_path = flywheel.run_best_of_five(
        "throw a right jab",
        tmp_path / "jab-run",
        provider="offline",
        selection_mode="five_way",
        max_rounds=2,
        capture_fn=fake_capture,
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    round_one = trace["rounds"][0]
    judged = [
        candidate
        for candidate in round_one["candidates"]
        if candidate.get("perceptually_rankable")
    ]

    assert trace["status"] == "winner_selected"
    assert trace["winner_result_id"] in {candidate["result_id"] for candidate in judged}
    assert len(judged) == 5
    assert len(captured) == 5
    assert len(ranked_batches) == 1
    assert ranking_seeds == [
        flywheel.blinding_seed(
            "throw a right jab", [candidate["result_id"] for candidate in judged]
        )
    ]
    assert round_one["batch_selection"]["all_pairs_above_threshold"] is True
    assert all(candidate.get("judgment", {}).get("accept") for candidate in judged)
    assert len(persisted) > 5  # The pool replenished globally after structural rejects.


def test_rejected_sequence_reaches_repair_and_second_round_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    persisted: list[str] = []
    repair_payloads: list[dict[str, object]] = []

    class FakeStore:
        def persist(self, request, clip) -> str:  # noqa: ANN001
            result_id = f"sequence-candidate-{len(persisted) + 1}"
            persisted.append(result_id)
            return result_id

    class FakeJudge:
        def __init__(self, **_: object) -> None:
            self.ranking_calls = 0

        def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict:
            assert len(manifests) == 5
            self.ranking_calls += 1
            accepted = self.ranking_calls == 2
            labels = ("A", "B", "C", "D", "E")
            return {
                "kind": "five_way_motion_judgment",
                "mapped_scores": {
                    index: {
                        "label": labels[index],
                        "semantic_match": 4 if accepted else 3,
                        "gesture_recognizability": 4 if accepted else 3,
                        "anatomical_naturalness": 4 if accepted else 3,
                        "temporal_readability": 4 if accepted else 3,
                        "egocentric_visibility": 5,
                        "overall": 4 if accepted else 3,
                        "accept": accepted,
                        "failure_tags": ["none" if accepted else "timing"],
                        "summary": "Second round succeeds." if accepted else "Timing needs repair.",
                        "suggested_adjustment": "None." if accepted else "Improve timing.",
                    }
                    for index in range(5)
                },
                "mapped_winner_index": 0 if accepted else None,
                "call": {"response_id": f"fake-ranking-{self.ranking_calls}"},
                "routing": {"selected_model": "fake"},
            }

        def recommend_repair(self, **kwargs) -> dict:  # noqa: ANN003
            repair_payloads.append(kwargs["parameters"])
            return {
                "kind": "bounded_motion_repair",
                "call": {
                    "parsed": {
                        "arm_height_delta": 0.0,
                        "arm_depth_delta": 0.0,
                        "lateral_offset_delta": 0.0,
                        "wrist_pitch_delta": 0.0,
                        "wrist_yaw_delta": 0.0,
                        "wrist_roll_delta": 0.0,
                        "elbow_swivel_delta": 0.0,
                        "torso_participation_delta": 0.0,
                        "path_arc_delta": 0.0,
                        "finger_splay_delta": 0.0,
                        "thumb_curl_delta": 0.0,
                        "little_curl_delta": 0.0,
                        "wrist_shake_amplitude_delta": 0.0,
                        "trajectory_amplitude_m_delta": 0.0,
                        "axial_rotation_amplitude_delta": 0.0,
                        "present_duration_scale": 0.95,
                        "hold_duration_scale": 1.05,
                        "shake_duration_scale": 1.0,
                        "recover_duration_scale": 1.0,
                        "easing_delta": 0.0,
                        "pose_root_scale": 1.0,
                        "pose_directional_scale": 1.0,
                        "object_distance_scale": 1.0,
                        "object_apex_scale": 1.0,
                        "object_contact_height_delta": 0.0,
                        "object_contact_depth_delta": 0.0,
                        "rationale": "Make the ordered phases more readable.",
                    }
                },
            }

    def fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "evidence-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    monkeypatch.setattr(flywheel, "ResultStore", FakeStore)
    monkeypatch.setattr(flywheel, "VLMJudge", FakeJudge)
    trace_path = flywheel.run_best_of_five(
        "turn left then throw a right jab",
        tmp_path / "sequence-run",
        provider="offline",
        selection_mode="five_way",
        max_rounds=2,
        capture_fn=fake_capture,
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert trace["status"] == "winner_selected"
    assert len(trace["rounds"]) == 2
    assert len(trace["rankings"]) == 2
    assert len(trace["repairs"]) == 1
    assert repair_payloads[0]["intent"] == "sequence"
    assert [step["intent"] for step in repair_payloads[0]["steps"]] == [
        "full_body",
        "strike",
    ]
