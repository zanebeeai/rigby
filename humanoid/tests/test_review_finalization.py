from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals.criteria import load_criteria as real_load_criteria
from evals import review

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _rotation(angle: float) -> dict[str, float]:
    return {"x": math.sin(angle / 2), "y": 0.0, "z": 0.0, "w": math.cos(angle / 2)}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "rigby-poc"
    acceptance = root / "results" / "acceptance-runs"
    (root / "frontend" / "dist").mkdir(parents=True)
    (root / "frontend" / "dist" / "review.js").write_text("fixed viewer", encoding="utf-8")
    avatar = root / "assets" / "avatar.glb"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"avatar")
    records = []
    for index, clip_id in enumerate(review.EXPECTED_CLIPS, start=1):
        result_id = f"{index:06d}-blind"
        result_root = root / "results" / result_id
        result_root.mkdir(parents=True)
        hand = "right" if index <= 10 else "left"
        prefix = hand
        bones = {
            f"{prefix}ThumbMetacarpal": {"rotation": _rotation(0.2)},
            f"{prefix}IndexProximal": {"rotation": _rotation(1.0)},
            f"{prefix}MiddleProximal": {"rotation": _rotation(1.1)},
            f"{prefix}RingProximal": {"rotation": _rotation(1.0)},
            f"{prefix}LittleProximal": {"rotation": _rotation(0.2)},
        }
        clip = {
            "schema_version": "1.0", "fps": 30, "duration_s": 1.0,
            "frames": [{"time_s": 0.0, "bones": bones, "objects": {}}], "contacts": [],
        }
        _write_json(result_root / "clip.json", clip)
        _write_json(result_root / "program.json", {"source_text": f"Show prompt {index}", "intent": "gesture", "hand": hand})
        _write_json(result_root / "scene.json", {"rig": {"asset_uri": "assets/avatar.glb"}})
        (result_root / "animation.glb").write_bytes(f"glTF-{clip_id}".encode())
        records.append({"clip_id": clip_id, "result_id": result_id, "available": True})
    monkeypatch.setattr(review, "POC_ROOT", root)
    monkeypatch.setattr(review, "ACCEPTANCE_ROOT", acceptance)
    criteria = real_load_criteria()
    monkeypatch.setattr(review, "load_criteria", lambda: {"gesture": criteria["gesture"], "runs_required": 2})
    return root, records


def _report(run_id: str) -> dict[str, object]:
    return {
        "schema_version": "1.0", "run_id": run_id, "started_at": "start", "completed_at": "end",
        "source": "live-public-api", "synthetic": False, "passed": False, "artifacts": [],
        "gates": [
            {"gate": "planner", "status": "pass", "summary": "unchanged", "measured": {"value": 1}, "required": {}, "evidence": [], "failures": []},
            {"gate": "semantic_forward_space", "status": "pass", "summary": "forward", "measured": {}, "required": {}, "evidence": [], "failures": []},
            {"gate": "egocentric_camera_orientation", "status": "pass", "summary": "camera", "measured": {}, "required": {}, "evidence": [], "failures": []},
            {"gate": "gesture_diversity", "status": "pass", "summary": "diverse", "measured": {}, "required": {}, "evidence": [], "failures": []},
            {"gate": "gesture", "status": "unverified", "summary": "manual review missing", "measured": {}, "required": {}, "evidence": [], "failures": []},
        ],
    }


def _raw_review(path: Path, run_id: str = "000002") -> Path:
    manifest = json.loads(
        (review.ACCEPTANCE_ROOT / run_id / "blinded-gesture-manifest.json").read_text(encoding="utf-8")
    )
    value = {
        "schema_version": "1.1",
        "acceptance_run_id": run_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [
            {"clip_id": clip_id, "egocentric_ratings": [5], "orbit_ratings": [5], "reviewer_notes": []}
            for clip_id in review.EXPECTED_CLIPS
        ],
    }
    _write_json(path, value)
    return path


def test_exact_run_review_finalization_is_append_only_and_recomputes_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _workspace(tmp_path, monkeypatch)
    acceptance = review.ACCEPTANCE_ROOT
    run_root = acceptance / "000002"
    run_root.mkdir(parents=True)
    review.write_review_bundle(run_root, records)
    manifest = json.loads((run_root / "blinded-gesture-manifest.json").read_text(encoding="utf-8"))
    assert manifest["blinded"] is False
    assert manifest["prompt_visible"] is True
    assert manifest["records"][0]["prompt"] == "Show prompt 1"
    original_report = _report("000002")
    _write_json(run_root / "report.json", original_report)
    original_bytes = (run_root / "report.json").read_bytes()
    _write_json(
        acceptance / "index.json",
        {
            "schema_version": "1.0", "next_sequence": 3, "certified": False, "consecutive_passes": 1,
            "runs": [
                {"id": "000001", "passed": True, "synthetic": False, "report": "000001/report.json", "html": "000001/report.html"},
                {"id": "000002", "passed": False, "synthetic": False, "report": "000002/report.json", "html": "000002/report.html"},
            ],
        },
    )
    bound = review.bind_review("000002", _raw_review(tmp_path / "download.json"))
    revised_path = review.finalize_review("000002", bound)
    revised = json.loads(revised_path.read_text(encoding="utf-8"))
    assert (run_root / "report.json").read_bytes() == original_bytes
    assert revised["passed"] is True
    assert revised["gates"][0] == original_report["gates"][0]
    assert revised["gates"][-1]["status"] == "pass"
    assert (revised_path.parent / "audit.json").is_file()
    assert (revised_path.parent / "submitted-review.json").is_file()
    ledger = json.loads((acceptance / "index.json").read_text(encoding="utf-8"))
    assert ledger["certified"] is True
    assert ledger["consecutive_passes"] == 2
    assert ledger["runs"][-1]["original_report"] == "000002/report.json"
    with pytest.raises(ValueError, match="already has finalized"):
        review.finalize_review("000002", bound)


def test_binding_rejects_motion_changed_after_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, records = _workspace(tmp_path, monkeypatch)
    run_root = review.ACCEPTANCE_ROOT / "000002"
    run_root.mkdir(parents=True)
    review.write_review_bundle(run_root, records)
    clip = root / "results" / records[0]["result_id"] / "clip.json"
    clip.write_text(clip.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="clip artifact changed"):
        review.bind_review("000002", _raw_review(tmp_path / "download.json"))


def test_bound_review_cannot_finalize_a_different_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, records = _workspace(tmp_path, monkeypatch)
    first = review.ACCEPTANCE_ROOT / "000002"
    second = review.ACCEPTANCE_ROOT / "000003"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    review.write_review_bundle(first, records)
    review.write_review_bundle(second, records)
    bound = review.bind_review("000002", _raw_review(tmp_path / "download.json"))
    with pytest.raises(ValueError, match="only a bound review archived under this run"):
        review._load_verified_bound_review(second, bound, review._load_verified_manifest(second))


def test_downloaded_review_cannot_bind_to_a_different_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, records = _workspace(tmp_path, monkeypatch)
    first = review.ACCEPTANCE_ROOT / "000002"
    second = review.ACCEPTANCE_ROOT / "000003"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    review.write_review_bundle(first, records)
    review.write_review_bundle(second, records)
    downloaded = _raw_review(tmp_path / "download.json", "000002")
    with pytest.raises(ValueError, match="different acceptance run"):
        review.bind_review("000003", downloaded)
