from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rigby_v2.calibration import (
    HumanStudyEvidenceError,
    construct_calibration_blueprint,
    export_human_study_blueprint,
    import_verified_human_study_collection,
)
from rigby_v2.flywheel.schemas import RubricScoresV1
from rigby_v2.hashing import canonical_json_bytes, hash_file
from rigby_v2.calibration.study import (
    CalibrationStudyBlueprint,
    ComparisonAssignment,
    RaterAssignment,
)

pytestmark = pytest.mark.fast


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _retained_collection(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Create test-only artifacts; these are never represented as human evidence."""

    generated = construct_calibration_blueprint(seed=907)
    global_raters = tuple(
        value.rater_id_hash for value in generated.assignments[0].raters
    )
    assignments = tuple(
        ComparisonAssignment(
            pair_id=assignment.pair_id,
            action_family=assignment.action_family,
            left_record_id=assignment.left_record_id,
            right_record_id=assignment.right_record_id,
            defects=assignment.defects,
            raters=tuple(
                RaterAssignment(
                    rater_id_hash=global_raters[index],
                    presentation_order=rater.presentation_order,
                )
                for index, rater in enumerate(assignment.raters)
            ),  # type: ignore[arg-type]
        )
        for assignment in generated.assignments
    )
    blueprint = CalibrationStudyBlueprint(
        seed=generated.seed,
        assignments=assignments,
        ground_truth=generated.ground_truth,
    )
    public = tmp_path / "review" / "public-blueprint.json"
    private = tmp_path / "review" / "private-ground-truth.json"
    exported = export_human_study_blueprint(
        blueprint,
        study_id="human-calibration-retained-test",
        public_blueprint_path=public,
        private_ground_truth_path=private,
    )
    root = tmp_path / "retained"
    rubric = RubricScoresV1(
        semantic_fidelity=4.0,
        physical_plausibility=4.0,
        contact_quality=4.0,
        timing_energy=4.0,
        whole_body_quality=4.0,
        visual_clarity=4.0,
    ).model_dump(mode="json")
    indexed: list[dict[str, object]] = []
    responses_by_rater: dict[str, dict[str, str]] = {}
    for assignment in blueprint.assignments:
        preferred = blueprint.ground_truth[assignment.pair_id].preferred_record_id
        for rater in assignment.raters:
            screen_left = (
                assignment.left_record_id
                if rater.presentation_order == "left_right"
                else assignment.right_record_id
            )
            relative = (
                Path("responses")
                / rater.rater_id_hash[:12]
                / f"{assignment.pair_id}.json"
            ).as_posix()
            response = {
                "schema_version": "1.0",
                "artifact_kind": "human_rating_response",
                "study_id": exported.study_id,
                "collection_mode": "human",
                "public_blueprint_sha256": exported.public_blueprint_sha256,
                "pair_id": assignment.pair_id,
                "rater_id_hash": rater.rater_id_hash,
                "presentation_order": rater.presentation_order,
                "verdict": "left" if preferred == screen_left else "right",
                "rubric": rubric,
                "recorded_at": datetime(2026, 8, 11, 12, tzinfo=UTC).isoformat(),
            }
            response_path = root.joinpath(*Path(relative).parts)
            _write_json(response_path, response)
            digest = hash_file(response_path)
            responses_by_rater.setdefault(rater.rater_id_hash, {})[
                assignment.pair_id
            ] = digest
            indexed.append(
                {
                    "pair_id": assignment.pair_id,
                    "rater_id_hash": rater.rater_id_hash,
                    "presentation_order": rater.presentation_order,
                    "source_response_path": relative,
                    "source_response_sha256": digest,
                }
            )

    attestation_ref: dict[str, tuple[str, str]] = {}
    for rater_id, responses in responses_by_rater.items():
        relative = f"attestations/{rater_id}.json"
        path = root.joinpath(*Path(relative).parts)
        _write_json(
            path,
            {
                "schema_version": "1.0",
                "artifact_kind": "human_rater_attestation",
                "study_id": exported.study_id,
                "collection_mode": "human",
                "public_blueprint_sha256": exported.public_blueprint_sha256,
                "rater_id_hash": rater_id,
                "attested_at": datetime(2026, 8, 11, 15, tzinfo=UTC).isoformat(),
                "affirmations": {
                    "independent_human_rater": True,
                    "reviewed_assigned_media": True,
                    "responses_are_own": True,
                    "no_automated_or_synthetic_ratings": True,
                },
                "response_sha256s": responses,
            },
        )
        attestation_ref[rater_id] = (relative, hash_file(path))
    for item in indexed:
        relative, digest = attestation_ref[str(item["rater_id_hash"])]
        item["rater_attestation_path"] = relative
        item["rater_attestation_sha256"] = digest

    manifest = tmp_path / "collection-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2.0",
            "artifact_kind": "retained_human_study_collection",
            "study_id": exported.study_id,
            "collection_mode": "human",
            "public_blueprint_sha256": exported.public_blueprint_sha256,
            "private_ground_truth_sha256": exported.private_ground_truth_sha256,
            "ratings": indexed,
        },
    )
    return root, manifest, public, private


def _import(root: Path, manifest: Path, public: Path, private: Path):  # type: ignore[no-untyped-def]
    return import_verified_human_study_collection(
        manifest,
        evidence_root=root,
        public_blueprint_path=public,
        private_ground_truth_path=private,
    )


def test_release_grade_import_verifies_all_600_files_and_three_attestations(
    tmp_path: Path,
) -> None:
    root, manifest, public, private = _retained_collection(tmp_path)

    evidence = _import(root, manifest, public, private)

    assert evidence.release_grade_files_verified
    assert len(evidence.study.pairs) == 200
    assert len(evidence.source_response_paths) == 600
    assert len(set(evidence.source_response_paths)) == 600
    assert len(set(evidence.source_response_sha256s)) == 600
    assert len(evidence.rater_attestation_paths) == 600
    assert len(set(evidence.rater_attestation_paths)) == 3
    assert len(set(evidence.rater_attestation_sha256s)) == 3
    assert evidence.collection_manifest_sha256 == hash_file(manifest)
    assert evidence.retained_evidence_set_sha256

    original_manifest = manifest.read_bytes()
    document = json.loads(original_manifest)
    first = document["ratings"][0]
    response_path = root.joinpath(*Path(first["source_response_path"]).parts)
    original_response = response_path.read_bytes()

    first["source_response_sha256"] = "a" * 64
    _write_json(manifest, document)
    with pytest.raises(HumanStudyEvidenceError):
        _import(root, manifest, public, private)
    manifest.write_bytes(original_manifest)

    document = json.loads(original_manifest)
    document["ratings"][0]["source_response_path"] = "../outside.json"
    _write_json(manifest, document)
    with pytest.raises(HumanStudyEvidenceError, match="safe relative"):
        _import(root, manifest, public, private)
    manifest.write_bytes(original_manifest)

    response_path.write_bytes(b"tampered")
    with pytest.raises(HumanStudyEvidenceError, match="hash mismatch"):
        _import(root, manifest, public, private)
    response_path.write_bytes(original_response)

    document = json.loads(original_manifest)
    rater = document["ratings"][0]["rater_id_hash"]
    relative = document["ratings"][0]["rater_attestation_path"]
    attestation_path = root.joinpath(*Path(relative).parts)
    original_attestation = attestation_path.read_bytes()
    attestation = json.loads(original_attestation)
    attestation["response_sha256s"].pop(next(iter(attestation["response_sha256s"])))
    _write_json(attestation_path, attestation)
    new_hash = hash_file(attestation_path)
    for item in document["ratings"]:
        if item["rater_id_hash"] == rater:
            item["rater_attestation_sha256"] = new_hash
    _write_json(manifest, document)
    with pytest.raises(HumanStudyEvidenceError, match="exact assigned pair set"):
        _import(root, manifest, public, private)
    attestation_path.write_bytes(original_attestation)
    manifest.write_bytes(original_manifest)

    document = json.loads(original_manifest)
    attestation = json.loads(original_attestation)
    attestation["affirmations"]["no_automated_or_synthetic_ratings"] = False
    _write_json(attestation_path, attestation)
    new_hash = hash_file(attestation_path)
    for item in document["ratings"]:
        if item["rater_id_hash"] == rater:
            item["rater_attestation_sha256"] = new_hash
    _write_json(manifest, document)
    with pytest.raises(HumanStudyEvidenceError, match="affirmations"):
        _import(root, manifest, public, private)
