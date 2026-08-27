"""Auditable export/import boundary for real human calibration studies."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from rigby_v2.flywheel.schemas import CalibrationRatingV1, DefectKind, HumanComparisonPairV1
from rigby_core.hashing import (
    canonical_json_bytes,
    content_hash,
    hash_file,
    sha256_bytes,
    validate_sha256,
)

from .study import (
    CalibrationStudy,
    CalibrationStudyBlueprint,
    ComparisonAssignment,
    PairGroundTruth,
    RaterAssignment,
    validate_calibration_study,
)


@dataclass(frozen=True, slots=True)
class HumanStudyBlueprintEvidence:
    study_id: str
    public_blueprint_path: Path
    public_blueprint_sha256: str
    private_ground_truth_path: Path
    private_ground_truth_sha256: str
    pair_count: int
    expected_rating_count: int


@dataclass(frozen=True, slots=True)
class CompletedHumanStudyEvidence:
    study_id: str
    study: CalibrationStudy
    public_blueprint_sha256: str
    private_ground_truth_sha256: str
    completed_ratings_sha256: str
    source_response_sha256s: tuple[str, ...]
    rater_attestation_sha256s: tuple[str, ...]
    release_grade_files_verified: bool = False
    collection_manifest_sha256: str | None = None
    retained_evidence_set_sha256: str | None = None
    source_response_paths: tuple[str, ...] = ()
    rater_attestation_paths: tuple[str, ...] = ()


class HumanStudyEvidenceError(ValueError):
    pass


def _write_new(path: Path, payload: bytes) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            raise FileExistsError(path) from None
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def export_human_study_blueprint(
    blueprint: CalibrationStudyBlueprint,
    *,
    study_id: str,
    public_blueprint_path: Path,
    private_ground_truth_path: Path,
) -> HumanStudyBlueprintEvidence:
    """Export a blinded assignment file and a separately held ground-truth file."""

    if not study_id.strip():
        raise HumanStudyEvidenceError("study ID is required")
    assignments = []
    for value in blueprint.assignments:
        assignments.append(
            {
                "pair_id": value.pair_id,
                "action_family": value.action_family,
                "left_record_id": value.left_record_id,
                "right_record_id": value.right_record_id,
                "defects": [defect.value for defect in value.defects],
                "raters": [
                    {
                        "rater_id_hash": rater.rater_id_hash,
                        "presentation_order": rater.presentation_order,
                    }
                    for rater in value.raters
                ],
            }
        )
    public = {
        "schema_version": "1.0",
        "study_id": study_id,
        "study_kind": "blinded_human_comparison",
        "collection_mode": "human",
        "seed": blueprint.seed,
        "assignments": assignments,
    }
    public_bytes = canonical_json_bytes(public)
    _write_new(public_blueprint_path, public_bytes)
    public_sha256 = hash_file(public_blueprint_path)
    truth = {
        "schema_version": "1.0",
        "study_id": study_id,
        "public_blueprint_sha256": public_sha256,
        "ground_truth": [
            {
                "pair_id": value.pair_id,
                "preferred_record_id": value.preferred_record_id,
                "critical_reject_record_ids": list(value.critical_reject_record_ids),
            }
            for value in sorted(blueprint.ground_truth.values(), key=lambda item: item.pair_id)
        ],
    }
    _write_new(private_ground_truth_path, canonical_json_bytes(truth))
    return HumanStudyBlueprintEvidence(
        study_id=study_id,
        public_blueprint_path=public_blueprint_path.resolve(),
        public_blueprint_sha256=public_sha256,
        private_ground_truth_path=private_ground_truth_path.resolve(),
        private_ground_truth_sha256=hash_file(private_ground_truth_path),
        pair_count=len(blueprint.assignments),
        expected_rating_count=len(blueprint.assignments) * 3,
    )


def _load_object(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HumanStudyEvidenceError(f"invalid human-study evidence file: {path.name}") from error
    if not isinstance(value, dict):
        raise HumanStudyEvidenceError("human-study evidence root must be an object")
    return value, sha256_bytes(payload)


def _load_retained_object(
    path: Path, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    value, observed = _load_object(path)
    if observed != expected_sha256:
        raise HumanStudyEvidenceError(f"{label} changed while it was being verified")
    return value


def _load_blueprint(
    public_path: Path, private_path: Path
) -> tuple[str, CalibrationStudyBlueprint, str, str]:
    public, public_hash = _load_object(public_path)
    expected_public_fields = {
        "schema_version",
        "study_id",
        "study_kind",
        "collection_mode",
        "seed",
        "assignments",
    }
    if set(public) != expected_public_fields:
        raise HumanStudyEvidenceError("public blueprint has unknown or missing fields")
    if (
        public["schema_version"] != "1.0"
        or public["study_kind"] != "blinded_human_comparison"
        or public["collection_mode"] != "human"
    ):
        raise HumanStudyEvidenceError("public blueprint is not a real-human collection blueprint")
    assignments = []
    try:
        for item in public["assignments"]:
            if set(item) != {
                "pair_id",
                "action_family",
                "left_record_id",
                "right_record_id",
                "defects",
                "raters",
            }:
                raise HumanStudyEvidenceError("assignment has unknown or missing fields")
            raters = tuple(
                RaterAssignment(
                    rater_id_hash=str(rater["rater_id_hash"]),
                    presentation_order=str(rater["presentation_order"]),
                )
                for rater in item["raters"]
            )
            if len(raters) != 3:
                raise HumanStudyEvidenceError("every blueprint pair requires exactly three raters")
            assignments.append(
                ComparisonAssignment(
                    pair_id=str(item["pair_id"]),
                    action_family=str(item["action_family"]),
                    left_record_id=str(item["left_record_id"]),
                    right_record_id=str(item["right_record_id"]),
                    defects=tuple(DefectKind(value) for value in item["defects"]),
                    raters=raters,  # type: ignore[arg-type]
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, HumanStudyEvidenceError):
            raise
        raise HumanStudyEvidenceError("public blueprint assignment is invalid") from error

    private, private_hash = _load_object(private_path)
    if set(private) != {
        "schema_version",
        "study_id",
        "public_blueprint_sha256",
        "ground_truth",
    }:
        raise HumanStudyEvidenceError("private ground truth has unknown or missing fields")
    if private["schema_version"] != "1.0" or private["study_id"] != public["study_id"]:
        raise HumanStudyEvidenceError("blueprint and ground-truth identity mismatch")
    if private["public_blueprint_sha256"] != public_hash:
        raise HumanStudyEvidenceError("ground truth is not bound to the public blueprint")
    truths = {}
    try:
        for item in private["ground_truth"]:
            truth = PairGroundTruth(
                pair_id=str(item["pair_id"]),
                preferred_record_id=str(item["preferred_record_id"]),
                critical_reject_record_ids=tuple(item["critical_reject_record_ids"]),
            )
            if truth.pair_id in truths:
                raise HumanStudyEvidenceError("ground truth contains duplicate pair IDs")
            truths[truth.pair_id] = truth
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, HumanStudyEvidenceError):
            raise
        raise HumanStudyEvidenceError("private ground truth is invalid") from error
    if {value.pair_id for value in assignments} != set(truths):
        raise HumanStudyEvidenceError("ground truth does not exactly cover blueprint assignments")
    return (
        str(public["study_id"]),
        CalibrationStudyBlueprint(
            seed=int(public["seed"]), assignments=tuple(assignments), ground_truth=truths
        ),
        public_hash,
        private_hash,
    )


def import_completed_human_study(
    completed_ratings_path: Path,
    *,
    public_blueprint_path: Path,
    private_ground_truth_path: Path,
) -> CompletedHumanStudyEvidence:
    """Compatibility import for v1 bundles containing digest declarations only.

    This preserves the original contract boundary, but the returned evidence is
    explicitly not release grade because no retained file bytes are verified.
    Use :func:`import_verified_human_study_collection` for release evidence.
    """

    study_id, blueprint, public_hash, private_hash = _load_blueprint(
        public_blueprint_path, private_ground_truth_path
    )
    document, ratings_hash = _load_object(completed_ratings_path)
    if set(document) != {
        "schema_version",
        "study_id",
        "collection_mode",
        "public_blueprint_sha256",
        "private_ground_truth_sha256",
        "ratings",
    }:
        raise HumanStudyEvidenceError("completed rating bundle has unknown or missing fields")
    if document["schema_version"] != "1.0" or document["study_id"] != study_id:
        raise HumanStudyEvidenceError("completed ratings use the wrong study identity")
    if document["collection_mode"] != "human":
        raise HumanStudyEvidenceError("synthetic or unspecified ratings cannot be imported as human")
    if document["public_blueprint_sha256"] != public_hash:
        raise HumanStudyEvidenceError("completed ratings are not bound to the public blueprint")
    if document["private_ground_truth_sha256"] != private_hash:
        raise HumanStudyEvidenceError("completed ratings are not bound to the private ground truth")

    expected_slots = {
        (assignment.pair_id, rater.rater_id_hash): (assignment, rater)
        for assignment in blueprint.assignments
        for rater in assignment.raters
    }
    observed: dict[tuple[str, str], CalibrationRatingV1] = {}
    source_hashes = []
    attestation_hashes = []
    for item in document["ratings"]:
        required = {
            "pair_id",
            "rater_id_hash",
            "presentation_order",
            "verdict",
            "rubric",
            "recorded_at",
            "source_response_sha256",
            "rater_attestation_sha256",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise HumanStudyEvidenceError("rating has unknown or missing fields")
        slot = (str(item["pair_id"]), str(item["rater_id_hash"]))
        if slot not in expected_slots:
            raise HumanStudyEvidenceError("rating was not assigned in the public blueprint")
        if slot in observed:
            raise HumanStudyEvidenceError("completed ratings contain a duplicate rater slot")
        _, assigned_rater = expected_slots[slot]
        if item["presentation_order"] != assigned_rater.presentation_order:
            raise HumanStudyEvidenceError("rating presentation order differs from its assignment")
        try:
            recorded_at = datetime.fromisoformat(str(item["recorded_at"]).replace("Z", "+00:00"))
            if recorded_at.tzinfo is None:
                raise ValueError("naive time")
            source_hashes.append(validate_sha256(str(item["source_response_sha256"])))
            attestation_hashes.append(validate_sha256(str(item["rater_attestation_sha256"])))
            observed[slot] = CalibrationRatingV1.model_validate(
                {
                    "rater_id_hash": slot[1],
                    "presentation_order": item["presentation_order"],
                    "verdict": item["verdict"],
                    "rubric": item["rubric"],
                }
            )
        except (ValueError, TypeError) as error:
            raise HumanStudyEvidenceError("rating evidence or contract is invalid") from error
    if set(observed) != set(expected_slots):
        missing = len(set(expected_slots) - set(observed))
        raise HumanStudyEvidenceError(f"completed rating bundle is missing {missing} assigned slots")

    pairs = []
    for assignment in blueprint.assignments:
        ratings = tuple(
            observed[(assignment.pair_id, rater.rater_id_hash)] for rater in assignment.raters
        )
        pairs.append(
            HumanComparisonPairV1(
                pair_id=assignment.pair_id,
                action_family=assignment.action_family,
                left_record_id=assignment.left_record_id,
                right_record_id=assignment.right_record_id,
                defects=assignment.defects,
                ratings=ratings,
            )
        )
    study = CalibrationStudy(
        seed=blueprint.seed, pairs=tuple(pairs), ground_truth=blueprint.ground_truth
    )
    validate_calibration_study(study)
    return CompletedHumanStudyEvidence(
        study_id=study_id,
        study=study,
        public_blueprint_sha256=public_hash,
        private_ground_truth_sha256=private_hash,
        completed_ratings_sha256=ratings_hash,
        source_response_sha256s=tuple(source_hashes),
        rater_attestation_sha256s=tuple(attestation_hashes),
        release_grade_files_verified=False,
    )


_RESPONSE_FIELDS = {
    "schema_version",
    "artifact_kind",
    "study_id",
    "collection_mode",
    "public_blueprint_sha256",
    "pair_id",
    "rater_id_hash",
    "presentation_order",
    "verdict",
    "rubric",
    "recorded_at",
}
_ATTESTATION_FIELDS = {
    "schema_version",
    "artifact_kind",
    "study_id",
    "collection_mode",
    "public_blueprint_sha256",
    "rater_id_hash",
    "attested_at",
    "affirmations",
    "response_sha256s",
}
_ATTESTATION_AFFIRMATIONS = {
    "independent_human_rater",
    "reviewed_assigned_media",
    "responses_are_own",
    "no_automated_or_synthetic_ratings",
}


def _aware_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise HumanStudyEvidenceError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise HumanStudyEvidenceError(f"{label} timestamp must be timezone-aware")
    return parsed


def _retained_file(
    root: Path,
    relative_path: object,
    expected_sha256: object,
    *,
    label: str,
) -> tuple[Path, str, str]:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise HumanStudyEvidenceError(f"{label} path must be a safe relative POSIX path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.name:
        raise HumanStudyEvidenceError(f"{label} path must be a safe relative POSIX path")
    root = root.resolve()
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor /= part
        if cursor.is_symlink():
            raise HumanStudyEvidenceError(f"{label} path cannot contain symbolic links")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise HumanStudyEvidenceError(f"{label} path escaped the evidence root") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise HumanStudyEvidenceError(f"{label} retained artifact is missing or empty")
    expected = validate_sha256(str(expected_sha256))
    observed = hash_file(resolved)
    if observed != expected:
        raise HumanStudyEvidenceError(f"{label} retained artifact hash mismatch")
    return resolved, pure.as_posix(), expected


def _study_from_observed(
    blueprint: CalibrationStudyBlueprint,
    observed: dict[tuple[str, str], CalibrationRatingV1],
) -> CalibrationStudy:
    pairs = []
    for assignment in blueprint.assignments:
        ratings = tuple(
            observed[(assignment.pair_id, rater.rater_id_hash)]
            for rater in assignment.raters
        )
        pairs.append(
            HumanComparisonPairV1(
                pair_id=assignment.pair_id,
                action_family=assignment.action_family,
                left_record_id=assignment.left_record_id,
                right_record_id=assignment.right_record_id,
                defects=assignment.defects,
                ratings=ratings,
            )
        )
    study = CalibrationStudy(
        seed=blueprint.seed, pairs=tuple(pairs), ground_truth=blueprint.ground_truth
    )
    validate_calibration_study(study)
    return study


def import_verified_human_study_collection(
    collection_manifest_path: Path,
    *,
    evidence_root: Path,
    public_blueprint_path: Path,
    private_ground_truth_path: Path,
) -> CompletedHumanStudyEvidence:
    """Verify retained bytes for every slot before producing release-grade evidence.

    The collection manifest is an index only.  Verdicts and rubric values come
    from the retained response artifacts, and every rater attestation binds the
    exact response hash for all of that rater's assigned slots.
    """

    study_id, blueprint, public_hash, private_hash = _load_blueprint(
        public_blueprint_path, private_ground_truth_path
    )
    document, manifest_hash = _load_object(collection_manifest_path)
    if set(document) != {
        "schema_version",
        "artifact_kind",
        "study_id",
        "collection_mode",
        "public_blueprint_sha256",
        "private_ground_truth_sha256",
        "ratings",
    }:
        raise HumanStudyEvidenceError(
            "verified collection manifest has unknown or missing fields"
        )
    if (
        document["schema_version"] != "2.0"
        or document["artifact_kind"] != "retained_human_study_collection"
        or document["collection_mode"] != "human"
        or document["study_id"] != study_id
    ):
        raise HumanStudyEvidenceError("manifest is not a retained real-human collection")
    if document["public_blueprint_sha256"] != public_hash:
        raise HumanStudyEvidenceError("collection is not bound to the public blueprint")
    if document["private_ground_truth_sha256"] != private_hash:
        raise HumanStudyEvidenceError("collection is not bound to the private ground truth")
    if not isinstance(document["ratings"], list):
        raise HumanStudyEvidenceError("collection ratings index must be a list")

    expected_slots = {
        (assignment.pair_id, rater.rater_id_hash): (assignment, rater)
        for assignment in blueprint.assignments
        for rater in assignment.raters
    }
    expected_pairs_by_rater: dict[str, set[str]] = {}
    for pair_id, rater_id in expected_slots:
        expected_pairs_by_rater.setdefault(rater_id, set()).add(pair_id)
    observed: dict[tuple[str, str], CalibrationRatingV1] = {}
    response_hash_by_rater: dict[str, dict[str, str]] = {
        rater: {} for rater in expected_pairs_by_rater
    }
    response_paths: list[str] = []
    response_hashes: list[str] = []
    attestation_ref_by_rater: dict[str, tuple[str, str]] = {}
    attestation_paths_for_slots: list[str] = []
    attestation_hashes_for_slots: list[str] = []
    seen_response_paths: set[str] = set()
    seen_response_hashes: set[str] = set()
    attestation_documents: dict[tuple[str, str], dict[str, Any]] = {}
    retained_attestations: dict[tuple[str, str], tuple[Path, str, str]] = {}
    index_fields = {
        "pair_id",
        "rater_id_hash",
        "presentation_order",
        "source_response_path",
        "source_response_sha256",
        "rater_attestation_path",
        "rater_attestation_sha256",
    }
    for item in document["ratings"]:
        if not isinstance(item, dict) or set(item) != index_fields:
            raise HumanStudyEvidenceError("collection slot has unknown or missing fields")
        slot = (str(item["pair_id"]), str(item["rater_id_hash"]))
        if slot not in expected_slots or slot in observed:
            raise HumanStudyEvidenceError("collection has an unassigned or duplicate slot")
        _, assigned_rater = expected_slots[slot]
        if item["presentation_order"] != assigned_rater.presentation_order:
            raise HumanStudyEvidenceError("collection presentation order differs from assignment")

        response_path, response_relative, response_hash = _retained_file(
            evidence_root,
            item["source_response_path"],
            item["source_response_sha256"],
            label="source response",
        )
        if response_relative in seen_response_paths or response_hash in seen_response_hashes:
            raise HumanStudyEvidenceError(
                "each assigned slot requires a distinct retained response artifact"
            )
        seen_response_paths.add(response_relative)
        seen_response_hashes.add(response_hash)
        response_document = _load_retained_object(
            response_path, response_hash, label="source response"
        )
        if set(response_document) != _RESPONSE_FIELDS:
            raise HumanStudyEvidenceError("retained response has unknown or missing fields")
        if (
            response_document["schema_version"] != "1.0"
            or response_document["artifact_kind"] != "human_rating_response"
            or response_document["collection_mode"] != "human"
            or response_document["study_id"] != study_id
            or response_document["public_blueprint_sha256"] != public_hash
            or response_document["pair_id"] != slot[0]
            or response_document["rater_id_hash"] != slot[1]
            or response_document["presentation_order"]
            != assigned_rater.presentation_order
        ):
            raise HumanStudyEvidenceError("retained response identity or assignment mismatch")
        _aware_datetime(response_document["recorded_at"], label="response")
        try:
            observed[slot] = CalibrationRatingV1.model_validate(
                {
                    "rater_id_hash": slot[1],
                    "presentation_order": assigned_rater.presentation_order,
                    "verdict": response_document["verdict"],
                    "rubric": response_document["rubric"],
                }
            )
        except (TypeError, ValueError) as error:
            raise HumanStudyEvidenceError("retained response contract is invalid") from error
        response_hash_by_rater[slot[1]][slot[0]] = response_hash
        response_paths.append(response_relative)
        response_hashes.append(response_hash)

        raw_attestation_ref = (
            str(item["rater_attestation_path"]),
            str(item["rater_attestation_sha256"]),
        )
        retained_attestation = retained_attestations.get(raw_attestation_ref)
        if retained_attestation is None:
            retained_attestation = _retained_file(
                evidence_root,
                item["rater_attestation_path"],
                item["rater_attestation_sha256"],
                label="rater attestation",
            )
            retained_attestations[raw_attestation_ref] = retained_attestation
        attestation_path, attestation_relative, attestation_hash = retained_attestation
        attestation_ref = (attestation_relative, attestation_hash)
        prior_ref = attestation_ref_by_rater.setdefault(slot[1], attestation_ref)
        if prior_ref != attestation_ref:
            raise HumanStudyEvidenceError("one rater must use one retained attestation")
        if attestation_ref not in attestation_documents:
            attestation_documents[attestation_ref] = _load_retained_object(
                attestation_path, attestation_hash, label="rater attestation"
            )
        attestation_paths_for_slots.append(attestation_relative)
        attestation_hashes_for_slots.append(attestation_hash)

    if set(observed) != set(expected_slots):
        missing = len(set(expected_slots) - set(observed))
        raise HumanStudyEvidenceError(
            f"verified collection is missing {missing} assigned response artifacts"
        )
    if len(set(attestation_ref_by_rater.values())) != len(expected_pairs_by_rater):
        raise HumanStudyEvidenceError("each rater requires a distinct attestation artifact")

    for rater_id, reference in attestation_ref_by_rater.items():
        attestation = attestation_documents[reference]
        if set(attestation) != _ATTESTATION_FIELDS:
            raise HumanStudyEvidenceError("rater attestation has unknown or missing fields")
        if (
            attestation["schema_version"] != "1.0"
            or attestation["artifact_kind"] != "human_rater_attestation"
            or attestation["collection_mode"] != "human"
            or attestation["study_id"] != study_id
            or attestation["public_blueprint_sha256"] != public_hash
            or attestation["rater_id_hash"] != rater_id
        ):
            raise HumanStudyEvidenceError("rater attestation identity mismatch")
        _aware_datetime(attestation["attested_at"], label="attestation")
        affirmations = attestation["affirmations"]
        if (
            not isinstance(affirmations, dict)
            or set(affirmations) != _ATTESTATION_AFFIRMATIONS
            or any(value is not True for value in affirmations.values())
        ):
            raise HumanStudyEvidenceError("rater attestation affirmations are incomplete")
        declared = attestation["response_sha256s"]
        if not isinstance(declared, dict) or set(declared) != expected_pairs_by_rater[rater_id]:
            raise HumanStudyEvidenceError(
                "rater attestation does not cover its exact assigned pair set"
            )
        try:
            normalized = {
                str(pair_id): validate_sha256(str(digest))
                for pair_id, digest in declared.items()
            }
        except ValueError as error:
            raise HumanStudyEvidenceError(
                "rater attestation contains an invalid response hash"
            ) from error
        if normalized != response_hash_by_rater[rater_id]:
            raise HumanStudyEvidenceError(
                "rater attestation is not bound to the retained response artifacts"
            )

    study = _study_from_observed(blueprint, observed)
    evidence_set_hash = content_hash(
        {
            "collection_manifest_sha256": manifest_hash,
            "responses": [
                {"path": path, "sha256": digest}
                for path, digest in zip(response_paths, response_hashes, strict=True)
            ],
            "attestations": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(set(attestation_ref_by_rater.values()))
            ],
        }
    )
    return CompletedHumanStudyEvidence(
        study_id=study_id,
        study=study,
        public_blueprint_sha256=public_hash,
        private_ground_truth_sha256=private_hash,
        completed_ratings_sha256=manifest_hash,
        source_response_sha256s=tuple(response_hashes),
        rater_attestation_sha256s=tuple(attestation_hashes_for_slots),
        release_grade_files_verified=True,
        collection_manifest_sha256=manifest_hash,
        retained_evidence_set_sha256=evidence_set_hash,
        source_response_paths=tuple(response_paths),
        rater_attestation_paths=tuple(attestation_paths_for_slots),
    )
