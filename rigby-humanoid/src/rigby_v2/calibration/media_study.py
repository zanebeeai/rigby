"""Source-backed, blinded review packages for genuine human calibration."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rigby_v2.flywheel.schemas import DefectKind
from rigby_v2.hashing import canonical_json_bytes, hash_file, validate_sha256

from .io import HumanStudyBlueprintEvidence, export_human_study_blueprint
from .study import (
    MINIMUM_PAIR_COUNT,
    MINIMUM_PAIRS_PER_FAMILY,
    SUPPORTED_ACTION_FAMILIES,
    CalibrationStudyBlueprint,
    ComparisonAssignment,
    PairGroundTruth,
    RaterAssignment,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MEDIA_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
_MAX_MEDIA_BYTES = 512 * 1024 * 1024


class HumanReviewPackageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceMediaRecord:
    record_id: str
    source_path: Path
    sha256: str
    media_type: str


@dataclass(frozen=True, slots=True)
class SourceComparison:
    pair_id: str
    action_family: str
    candidate_a_id: str
    candidate_b_id: str
    preferred_record_id: str
    defects: tuple[DefectKind, ...]
    critical_reject_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HumanReviewPackageEvidence:
    study_id: str
    root: Path
    blueprint: HumanStudyBlueprintEvidence
    media_manifest_path: Path
    media_manifest_sha256: str
    private_source_map_path: Path
    private_source_map_sha256: str
    rater_packet_paths: tuple[Path, ...]
    rater_packet_sha256s: tuple[str, ...]
    pair_count: int
    expected_rating_count: int
    distinct_media_artifacts: int


def _write_new(path: Path, value: object | bytes) -> None:
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(path) from None
    finally:
        temporary.unlink(missing_ok=True)


def _blind_id(study_id: str, record: SourceMediaRecord) -> str:
    digest = hashlib.sha256(
        f"{study_id}\0{record.record_id}\0{record.sha256}".encode()
    ).hexdigest()
    return f"blind_{digest[:20]}"


def _validate_sources(
    records: tuple[SourceMediaRecord, ...], comparisons: tuple[SourceComparison, ...]
) -> dict[str, SourceMediaRecord]:
    if len(comparisons) < MINIMUM_PAIR_COUNT:
        raise HumanReviewPackageError(
            f"human review package requires at least {MINIMUM_PAIR_COUNT} pairs"
        )
    by_id: dict[str, SourceMediaRecord] = {}
    for record in records:
        if not _SAFE_ID.fullmatch(record.record_id) or record.record_id in by_id:
            raise HumanReviewPackageError("media record IDs must be unique and safe")
        if record.media_type not in _MEDIA_TYPES:
            raise HumanReviewPackageError(
                f"unsupported review media type: {record.media_type}"
            )
        source = record.source_path.resolve()
        if not source.is_file() or source.is_symlink():
            raise HumanReviewPackageError(
                f"review source is not a regular file: {record.record_id}"
            )
        size = source.stat().st_size
        if size <= 0 or size > _MAX_MEDIA_BYTES:
            raise HumanReviewPackageError("review media size is outside the safe range")
        expected = validate_sha256(record.sha256)
        if hash_file(source) != expected:
            raise HumanReviewPackageError(
                f"review media hash mismatch: {record.record_id}"
            )
        by_id[record.record_id] = record

    pair_ids: set[str] = set()
    family_counts = Counter()
    defect_counts = Counter()
    reversal_groups: dict[tuple[str, str], list[SourceComparison]] = defaultdict(list)
    for comparison in comparisons:
        if not _SAFE_ID.fullmatch(comparison.pair_id) or comparison.pair_id in pair_ids:
            raise HumanReviewPackageError("comparison pair IDs must be unique and safe")
        pair_ids.add(comparison.pair_id)
        if comparison.action_family not in SUPPORTED_ACTION_FAMILIES:
            raise HumanReviewPackageError(
                f"unsupported action family: {comparison.action_family}"
            )
        family_counts[comparison.action_family] += 1
        if (
            comparison.candidate_a_id == comparison.candidate_b_id
            or comparison.candidate_a_id not in by_id
            or comparison.candidate_b_id not in by_id
        ):
            raise HumanReviewPackageError("comparison references invalid media records")
        candidates = {comparison.candidate_a_id, comparison.candidate_b_id}
        if comparison.preferred_record_id not in candidates:
            raise HumanReviewPackageError("preferred record is not in its comparison")
        if not comparison.defects:
            raise HumanReviewPackageError("every comparison needs a defect classification")
        defect_counts.update(comparison.defects)
        if not set(comparison.critical_reject_record_ids).issubset(candidates):
            raise HumanReviewPackageError("critical reject is not in its comparison")
        if DefectKind.ORDER_REVERSAL in comparison.defects:
            reversal_groups[tuple(sorted(candidates))].append(comparison)
    if any(
        family_counts[family] < MINIMUM_PAIRS_PER_FAMILY
        for family in SUPPORTED_ACTION_FAMILIES
    ):
        raise HumanReviewPackageError(
            f"each family requires at least {MINIMUM_PAIRS_PER_FAMILY} pairs"
        )
    if any(defect_counts[defect] == 0 for defect in DefectKind):
        raise HumanReviewPackageError("review package must cover every defect class")
    if not reversal_groups or any(len(group) != 2 for group in reversal_groups.values()):
        raise HumanReviewPackageError(
            "order-reversal comparisons must form exact two-pair groups"
        )
    return by_id


def build_human_review_package(
    *,
    study_id: str,
    seed: int,
    records: tuple[SourceMediaRecord, ...],
    comparisons: tuple[SourceComparison, ...],
    rater_id_hashes: tuple[str, str, str],
    output_root: Path,
) -> HumanReviewPackageEvidence:
    """Create a non-overwriting review package with no source/truth leakage."""

    if not _SAFE_ID.fullmatch(study_id):
        raise HumanReviewPackageError("study ID is unsafe")
    normalized_raters = tuple(validate_sha256(value) for value in rater_id_hashes)
    if len(set(normalized_raters)) != 3:
        raise HumanReviewPackageError("three independent rater hashes are required")
    by_id = _validate_sources(records, comparisons)
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    blind_by_source = {
        record_id: _blind_id(study_id, record) for record_id, record in by_id.items()
    }
    rng = random.Random(seed)
    reversal_seen: dict[tuple[str, str], int] = defaultdict(int)
    assignments: list[ComparisonAssignment] = []
    truths: dict[str, PairGroundTruth] = {}
    for index, comparison in enumerate(comparisons):
        pair_key = tuple(
            sorted((comparison.candidate_a_id, comparison.candidate_b_id))
        )
        if DefectKind.ORDER_REVERSAL in comparison.defects:
            left_is_a = reversal_seen[pair_key] % 2 == 0
            reversal_seen[pair_key] += 1
        else:
            left_is_a = bool(rng.getrandbits(1))
        left_source = (
            comparison.candidate_a_id if left_is_a else comparison.candidate_b_id
        )
        right_source = (
            comparison.candidate_b_id if left_is_a else comparison.candidate_a_id
        )
        orders = (
            ["left_right", "right_left", "left_right"]
            if index % 2 == 0
            else ["right_left", "left_right", "right_left"]
        )
        rng.shuffle(orders)
        assignments.append(
            ComparisonAssignment(
                pair_id=comparison.pair_id,
                action_family=comparison.action_family,
                left_record_id=blind_by_source[left_source],
                right_record_id=blind_by_source[right_source],
                defects=comparison.defects,
                raters=tuple(
                    RaterAssignment(
                        rater_id_hash=rater,
                        presentation_order=orders[rater_index],
                    )
                    for rater_index, rater in enumerate(normalized_raters)
                ),  # type: ignore[arg-type]
            )
        )
        truths[comparison.pair_id] = PairGroundTruth(
            pair_id=comparison.pair_id,
            preferred_record_id=blind_by_source[comparison.preferred_record_id],
            critical_reject_record_ids=tuple(
                blind_by_source[value]
                for value in comparison.critical_reject_record_ids
            ),
        )

    blueprint = CalibrationStudyBlueprint(
        seed=seed, assignments=tuple(assignments), ground_truth=truths
    )
    blueprint_evidence = export_human_study_blueprint(
        blueprint,
        study_id=study_id,
        public_blueprint_path=root / "public" / "blueprint.json",
        private_ground_truth_path=root / "private" / "ground_truth.json",
    )

    media_entries = []
    private_entries = []
    copied: set[str] = set()
    for source_id in sorted(by_id):
        record = by_id[source_id]
        suffix = _MEDIA_TYPES[record.media_type]
        relative = Path("media") / f"{record.sha256}{suffix}"
        destination = root / "public" / relative
        if record.sha256 not in copied:
            _write_new(destination, record.source_path.resolve().read_bytes())
            copied.add(record.sha256)
        blind_id = blind_by_source[source_id]
        media_entries.append(
            {
                "blind_record_id": blind_id,
                "sha256": record.sha256,
                "size_bytes": record.source_path.stat().st_size,
                "media_type": record.media_type,
                "relative_path": relative.as_posix(),
            }
        )
        private_entries.append(
            {
                "blind_record_id": blind_id,
                "source_record_id": source_id,
                "source_sha256": record.sha256,
                "source_filename": record.source_path.name,
            }
        )
    media_manifest = {
        "schema_version": "1.0",
        "study_id": study_id,
        "public_blueprint_sha256": blueprint_evidence.public_blueprint_sha256,
        "records": media_entries,
    }
    media_manifest_path = root / "public" / "media_manifest.json"
    _write_new(media_manifest_path, media_manifest)
    media_manifest_hash = hash_file(media_manifest_path)
    private_map_path = root / "private" / "source_map.json"
    _write_new(
        private_map_path,
        {
            "schema_version": "1.0",
            "study_id": study_id,
            "public_blueprint_sha256": blueprint_evidence.public_blueprint_sha256,
            "media_manifest_sha256": media_manifest_hash,
            "records": private_entries,
        },
    )

    media_by_blind = {item["blind_record_id"]: item for item in media_entries}
    packet_paths = []
    packet_hashes = []
    for rater in normalized_raters:
        packet_assignments = []
        for assignment in assignments:
            rater_assignment = next(
                value for value in assignment.raters if value.rater_id_hash == rater
            )
            screen_left = (
                assignment.left_record_id
                if rater_assignment.presentation_order == "left_right"
                else assignment.right_record_id
            )
            screen_right = (
                assignment.right_record_id
                if rater_assignment.presentation_order == "left_right"
                else assignment.left_record_id
            )
            packet_assignments.append(
                {
                    "pair_id": assignment.pair_id,
                    "presentation_order": rater_assignment.presentation_order,
                    "screen_left": media_by_blind[screen_left],
                    "screen_right": media_by_blind[screen_right],
                }
            )
        packet = {
            "schema_version": "1.0",
            "study_id": study_id,
            "collection_mode": "human",
            "rater_id_hash": rater,
            "public_blueprint_sha256": blueprint_evidence.public_blueprint_sha256,
            "media_manifest_sha256": media_manifest_hash,
            "rubric_dimensions": [
                "semantic_fidelity",
                "physical_plausibility",
                "contact_quality",
                "timing_energy",
                "whole_body_quality",
                "visual_clarity",
            ],
            "allowed_verdicts": ["left", "right", "tie", "abstain", "both_fail"],
            "assignments": packet_assignments,
        }
        packet_path = root / "public" / "rater_packets" / f"{rater}.json"
        _write_new(packet_path, packet)
        packet_paths.append(packet_path)
        packet_hashes.append(hash_file(packet_path))

    return HumanReviewPackageEvidence(
        study_id=study_id,
        root=root,
        blueprint=blueprint_evidence,
        media_manifest_path=media_manifest_path,
        media_manifest_sha256=media_manifest_hash,
        private_source_map_path=private_map_path,
        private_source_map_sha256=hash_file(private_map_path),
        rater_packet_paths=tuple(packet_paths),
        rater_packet_sha256s=tuple(packet_hashes),
        pair_count=len(comparisons),
        expected_rating_count=len(comparisons) * 3,
        distinct_media_artifacts=len(copied),
    )


def verify_human_review_package(evidence: HumanReviewPackageEvidence) -> None:
    """Fail closed if any public/private/media package byte has drifted."""

    if hash_file(evidence.blueprint.public_blueprint_path) != (
        evidence.blueprint.public_blueprint_sha256
    ):
        raise HumanReviewPackageError("public blueprint hash drifted")
    if hash_file(evidence.blueprint.private_ground_truth_path) != (
        evidence.blueprint.private_ground_truth_sha256
    ):
        raise HumanReviewPackageError("private ground truth hash drifted")
    if hash_file(evidence.media_manifest_path) != evidence.media_manifest_sha256:
        raise HumanReviewPackageError("media manifest hash drifted")
    if hash_file(evidence.private_source_map_path) != evidence.private_source_map_sha256:
        raise HumanReviewPackageError("private source map hash drifted")
    if tuple(hash_file(path) for path in evidence.rater_packet_paths) != (
        evidence.rater_packet_sha256s
    ):
        raise HumanReviewPackageError("rater packet hash drifted")
    manifest = json.loads(evidence.media_manifest_path.read_text(encoding="utf-8"))
    for item in manifest["records"]:
        relative_value = item["relative_path"]
        if not isinstance(relative_value, str) or "\\" in relative_value:
            raise HumanReviewPackageError("media artifact path is not safe")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise HumanReviewPackageError("media artifact path is not safe")
        public_root = (evidence.root / "public").resolve()
        unresolved = public_root.joinpath(*relative.parts)
        cursor = public_root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise HumanReviewPackageError("media artifact path contains a symbolic link")
        path = unresolved.resolve()
        try:
            path.relative_to(public_root)
        except ValueError as error:
            raise HumanReviewPackageError("media artifact path escaped the package") from error
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or hash_file(path) != item["sha256"]
        ):
            raise HumanReviewPackageError(
                f"blinded media artifact drifted: {item['blind_record_id']}"
            )
