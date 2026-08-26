from __future__ import annotations

import json

import pytest

from rigby_v2.calibration import (
    SUPPORTED_ACTION_FAMILIES,
    HumanReviewPackageError,
    SourceComparison,
    SourceMediaRecord,
    build_human_review_package,
    verify_human_review_package,
)
from rigby_v2.flywheel.schemas import DefectKind
from rigby_v2.hashing import content_hash, hash_file


def _sources(tmp_path):  # type: ignore[no-untyped-def]
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"verified-left-evidence")
    second.write_bytes(b"verified-right-evidence")
    records = {}
    comparisons = []
    defect_cycle = (
        DefectKind.REALISTIC,
        DefectKind.NEAR_TIE,
        DefectKind.LEFT_RIGHT,
        DefectKind.TIMING,
        DefectKind.MISSING_VIEW,
        DefectKind.HAND,
        DefectKind.CONTACT,
    )
    for family in SUPPORTED_ACTION_FAMILIES:
        for index in range(40):
            group = index // 2 if index < 8 else index
            left_id = f"{family}-source-{group:03d}-a"
            right_id = f"{family}-source-{group:03d}-b"
            records.setdefault(
                left_id,
                SourceMediaRecord(
                    record_id=left_id,
                    source_path=first,
                    sha256=hash_file(first),
                    media_type="video/mp4",
                ),
            )
            records.setdefault(
                right_id,
                SourceMediaRecord(
                    record_id=right_id,
                    source_path=second,
                    sha256=hash_file(second),
                    media_type="video/mp4",
                ),
            )
            defects = (
                (DefectKind.ORDER_REVERSAL, DefectKind.REALISTIC)
                if index < 8
                else (defect_cycle[(index - 8) % len(defect_cycle)],)
            )
            critical = (
                (right_id,)
                if any(
                    defect
                    in {
                        DefectKind.LEFT_RIGHT,
                        DefectKind.MISSING_VIEW,
                        DefectKind.HAND,
                        DefectKind.CONTACT,
                    }
                    for defect in defects
                )
                else ()
            )
            comparisons.append(
                SourceComparison(
                    pair_id=f"{family}-pair-{index:03d}",
                    action_family=family,
                    candidate_a_id=left_id,
                    candidate_b_id=right_id,
                    preferred_record_id=left_id,
                    defects=defects,
                    critical_reject_record_ids=critical,
                )
            )
    return tuple(records.values()), tuple(comparisons)


def test_real_review_package_binds_media_without_rater_leakage(tmp_path) -> None:
    records, comparisons = _sources(tmp_path)
    raters = tuple(content_hash({"genuine_rater_slot": index}) for index in range(3))
    evidence = build_human_review_package(
        study_id="rigby-human-calibration-2026-01",
        seed=811,
        records=records,
        comparisons=comparisons,
        rater_id_hashes=raters,  # type: ignore[arg-type]
        output_root=tmp_path / "review-package",
    )
    verify_human_review_package(evidence)
    assert evidence.pair_count == 200
    assert evidence.expected_rating_count == 600
    assert evidence.distinct_media_artifacts == 2
    assert len(evidence.rater_packet_paths) == 3

    packet_text = evidence.rater_packet_paths[0].read_text(encoding="utf-8")
    packet = json.loads(packet_text)
    assert len(packet["assignments"]) == 200
    assert "defects" not in packet_text
    assert "preferred_record_id" not in packet_text
    assert "source_record_id" not in packet_text
    assert "action_family" not in packet_text
    private_text = evidence.private_source_map_path.read_text(encoding="utf-8")
    assert records[0].record_id in private_text
    assert records[0].record_id not in packet_text

    public_blueprint = json.loads(
        evidence.blueprint.public_blueprint_path.read_text(encoding="utf-8")
    )
    reversal = [
        item
        for item in public_blueprint["assignments"]
        if "order_reversal" in item["defects"]
    ]
    assert len(reversal) == 40
    assert reversal[0]["left_record_id"] == reversal[1]["right_record_id"]
    assert reversal[0]["right_record_id"] == reversal[1]["left_record_id"]


def test_review_package_rejects_unverified_media_and_detects_tampering(tmp_path) -> None:
    records, comparisons = _sources(tmp_path)
    raters = tuple(content_hash({"rater": index}) for index in range(3))
    bad = list(records)
    bad[0] = SourceMediaRecord(
        record_id=bad[0].record_id,
        source_path=bad[0].source_path,
        sha256="0" * 64,
        media_type=bad[0].media_type,
    )
    with pytest.raises(HumanReviewPackageError, match="hash mismatch"):
        build_human_review_package(
            study_id="bad-media-study",
            seed=1,
            records=tuple(bad),
            comparisons=comparisons,
            rater_id_hashes=raters,  # type: ignore[arg-type]
            output_root=tmp_path / "bad-package",
        )

    evidence = build_human_review_package(
        study_id="tamper-study",
        seed=2,
        records=records,
        comparisons=comparisons,
        rater_id_hashes=raters,  # type: ignore[arg-type]
        output_root=tmp_path / "tamper-package",
    )
    media = next((evidence.root / "public" / "media").iterdir())
    media.write_bytes(b"tampered")
    with pytest.raises(HumanReviewPackageError, match="media artifact drifted"):
        verify_human_review_package(evidence)


def test_review_package_requires_three_real_independent_rater_hashes(tmp_path) -> None:
    records, comparisons = _sources(tmp_path)
    repeated = content_hash({"rater": "same"})
    with pytest.raises(HumanReviewPackageError, match="independent"):
        build_human_review_package(
            study_id="duplicate-rater-study",
            seed=3,
            records=records,
            comparisons=comparisons,
            rater_id_hashes=(repeated, repeated, repeated),
            output_root=tmp_path / "duplicate-rater-package",
        )
