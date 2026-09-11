from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rigby_v2.calibration import (
    HumanStudyEvidenceError,
    construct_calibration_blueprint,
    export_human_study_blueprint,
    import_completed_human_study,
    validate_calibration_study,
)
from rigby_v2.flywheel.schemas import RubricScoresV1
from rigby_core.hashing import content_hash

pytestmark = pytest.mark.fast


def _export(tmp_path):
    blueprint = construct_calibration_blueprint(seed=431)
    public = tmp_path / "public-blueprint.json"
    private = tmp_path / "private-ground-truth.json"
    evidence = export_human_study_blueprint(
        blueprint,
        study_id="human-calibration-2026-01",
        public_blueprint_path=public,
        private_ground_truth_path=private,
    )
    return blueprint, public, private, evidence


def _completed_fixture(blueprint, evidence):
    """Build a contract fixture; this is never presented as collected evidence."""

    rubric = RubricScoresV1(
        semantic_fidelity=4.0,
        physical_plausibility=4.0,
        contact_quality=4.0,
        timing_energy=4.0,
        whole_body_quality=4.0,
        visual_clarity=4.0,
    ).model_dump(mode="json")
    ratings = []
    for assignment in blueprint.assignments:
        preferred = blueprint.ground_truth[assignment.pair_id].preferred_record_id
        for rater in assignment.raters:
            screen_left = (
                assignment.left_record_id
                if rater.presentation_order == "left_right"
                else assignment.right_record_id
            )
            ratings.append(
                {
                    "pair_id": assignment.pair_id,
                    "rater_id_hash": rater.rater_id_hash,
                    "presentation_order": rater.presentation_order,
                    "verdict": "left" if preferred == screen_left else "right",
                    "rubric": rubric,
                    "recorded_at": datetime(2026, 8, 10, 12, tzinfo=UTC).isoformat(),
                    "source_response_sha256": content_hash(
                        {"pair": assignment.pair_id, "rater": rater.rater_id_hash, "source": "fixture"}
                    ),
                    "rater_attestation_sha256": content_hash(
                        {"rater": rater.rater_id_hash, "attestation": "fixture"}
                    ),
                }
            )
    return {
        "schema_version": "1.0",
        "study_id": evidence.study_id,
        "collection_mode": "human",
        "public_blueprint_sha256": evidence.public_blueprint_sha256,
        "private_ground_truth_sha256": evidence.private_ground_truth_sha256,
        "ratings": ratings,
    }


def test_blueprint_export_blinds_ground_truth_and_binds_private_evidence(tmp_path) -> None:
    blueprint, public, private, evidence = _export(tmp_path)
    public_document = json.loads(public.read_text(encoding="utf-8"))
    private_document = json.loads(private.read_text(encoding="utf-8"))
    assert public_document["collection_mode"] == "human"
    assert "ground_truth" not in public_document
    assert "preferred_record_id" not in public.read_text(encoding="utf-8")
    assert private_document["public_blueprint_sha256"] == evidence.public_blueprint_sha256
    assert evidence.pair_count == 200
    assert evidence.expected_rating_count == 600
    assert len(blueprint.assignments) == 200
    with pytest.raises(FileExistsError):
        export_human_study_blueprint(
            blueprint,
            study_id=evidence.study_id,
            public_blueprint_path=public,
            private_ground_truth_path=tmp_path / "other-private.json",
        )


def test_completed_human_rating_import_requires_every_assigned_evidenced_slot(tmp_path) -> None:
    blueprint, public, private, evidence = _export(tmp_path)
    completed = tmp_path / "completed-ratings.json"
    completed.write_text(
        json.dumps(_completed_fixture(blueprint, evidence)), encoding="utf-8"
    )
    imported = import_completed_human_study(
        completed,
        public_blueprint_path=public,
        private_ground_truth_path=private,
    )
    validation = validate_calibration_study(imported.study)
    assert validation.pair_count == 200
    assert validation.rating_count == 600
    assert imported.completed_ratings_sha256
    assert len(imported.source_response_sha256s) == 600
    assert len(imported.rater_attestation_sha256s) == 600
    assert imported.release_grade_files_verified is False


@pytest.mark.parametrize("failure", ("synthetic", "missing", "wrong_order", "bad_hash"))
def test_completed_import_fails_closed_on_unverifiable_collection(tmp_path, failure) -> None:
    blueprint, public, private, evidence = _export(tmp_path)
    document = _completed_fixture(blueprint, evidence)
    if failure == "synthetic":
        document["collection_mode"] = "synthetic"
    elif failure == "missing":
        document["ratings"].pop()
    elif failure == "wrong_order":
        document["ratings"][0]["presentation_order"] = (
            "right_left"
            if document["ratings"][0]["presentation_order"] == "left_right"
            else "left_right"
        )
    else:
        document["ratings"][0]["source_response_sha256"] = "not-a-digest"
    completed = tmp_path / "invalid-ratings.json"
    completed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HumanStudyEvidenceError):
        import_completed_human_study(
            completed,
            public_blueprint_path=public,
            private_ground_truth_path=private,
        )
