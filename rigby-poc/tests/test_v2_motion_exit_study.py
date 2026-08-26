from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from rigby_v2.calibration.motion_exit_study import (
    EXIT_DIMENSIONS,
    MatchedMotionPairSource,
    MotionExitStudyError,
    MotionVariantSource,
    build_motion_exit_study_package,
    import_verified_motion_exit_study,
)
from rigby_v2.hashing import canonical_json_bytes, hash_file, sha256_bytes


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(
        output, format="PNG", pnginfo=None, optimize=False
    )
    return output.getvalue()


def _frame_archive(path: Path, color: tuple[int, int, int]) -> None:
    frames = []
    entries = {}
    for index in range(30):
        payload = _png(((color[0] + index) % 256, color[1], color[2]))
        name = f"frames/{index:06d}.png"
        entries[name] = payload
        frames.append(
            {
                "path": name,
                "timestamp_s": index / 30.0,
                "sha256": sha256_bytes(payload),
            }
        )
    playback = {
        "schema_version": "1.0",
        "artifact_kind": "motion_frame_archive_source",
        "fps": 30,
        "camera": {
            "projection": "perspective",
            "width_px": 64,
            "height_px": 64,
            "horizontal_fov_deg": 60.0,
        },
        "frames": frames,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("playback.json", canonical_json_bytes(playback))
        for name, payload in entries.items():
            archive.writestr(name, payload)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sources(root: Path, count: int = 30) -> tuple[MatchedMotionPairSource, ...]:
    root.mkdir(parents=True, exist_ok=True)
    v2_media = root / "v2-motion.zip"
    poc_media = root / "poc-motion.zip"
    _frame_archive(v2_media, (120, 30, 20))
    _frame_archive(poc_media, (20, 30, 120))
    pairs = []
    for index in range(count):
        case_id = f"held_out_{index:03d}"
        variants = {}
        for variant, media in (("v2", v2_media), ("poc", poc_media)):
            trace = root / f"{case_id}-{variant}.trace"
            trace.write_bytes(f"authoritative {case_id} {variant}".encode())
            trace_hash = hash_file(trace)
            summary = root / f"{case_id}-{variant}-summary.json"
            _json(
                summary,
                {
                    "schema_version": "1.0",
                    "artifact_kind": "motion_trace_summary",
                    "case_id": case_id,
                    "variant": variant,
                    "split": "held_out",
                    "authoritative_trace_sha256": trace_hash,
                    "duration_s": 1.0,
                    "phase_timing_s": {"approach": 0.3, "action": 0.7},
                    "peak_speed": 2.0 if variant == "v2" else 3.0,
                    "jerk_distribution": {
                        "p50": 1.0,
                        "p95": 2.0,
                        "maximum": 3.0,
                    },
                    "hand_shape": {"aperture": 0.04, "curl": 0.6},
                },
            )
            certification = None
            certification_hash = None
            if variant == "v2":
                certification = root / f"{case_id}-certification.json"
                _json(
                    certification,
                    {
                        "schema_version": "1.0",
                        "artifact_kind": "rigby_v2_certification_evidence",
                        "case_id": case_id,
                        "certified": True,
                        "authoritative_trace_sha256": trace_hash,
                        "repeat_count": 3,
                        "replay_exact": True,
                        "export_reimport_passed": True,
                    },
                )
                certification_hash = hash_file(certification)
            variants[variant] = MotionVariantSource(
                media_path=media,
                media_sha256=hash_file(media),
                trace_path=trace,
                trace_sha256=trace_hash,
                trace_summary_path=summary,
                trace_summary_sha256=hash_file(summary),
                certification_path=certification,
                certification_sha256=certification_hash,
            )
        pairs.append(
            MatchedMotionPairSource(
                pair_id=f"pair_{index:03d}",
                held_out_case_id=case_id,
                split="held_out",
                v2=variants["v2"],
                poc=variants["poc"],
            )
        )
    return tuple(pairs)


def _completed_collection(package: Path, retained: Path) -> Path:
    manifest = json.loads((package / "package-manifest.json").read_text())
    private = json.loads((package / "private/variant_map.json").read_text())
    v2_ids = {pair["v2_blind_record_id"] for pair in private["pairs"]}
    response_refs = []
    responses_by_rater = {}
    for packet_ref in manifest["rater_packets"]:
        packet = json.loads((package / packet_ref["path"]).read_text())
        rater = packet["rater_id_hash"]
        per_rater = {}
        for assignment in packet["assignments"]:
            v2_choice = (
                "left"
                if assignment["left"]["blind_record_id"] in v2_ids
                else "right"
            )
            response = {
                "schema_version": "1.0",
                "artifact_kind": "human_motion_exit_response",
                "study_id": manifest["study_id"],
                "collection_mode": "human",
                "rater_id_hash": rater,
                "presentation_id": assignment["presentation_id"],
                "recorded_at": "2026-08-11T12:00:00Z",
                "choices": {dimension: v2_choice for dimension in EXIT_DIMENSIONS},
            }
            relative = f"responses/{rater}/{assignment['presentation_id']}.json"
            path = retained / relative
            _json(path, response)
            digest = hash_file(path)
            per_rater[assignment["presentation_id"]] = digest
            response_refs.append(
                {
                    "rater_id_hash": rater,
                    "presentation_id": assignment["presentation_id"],
                    "response_path": relative,
                    "response_sha256": digest,
                }
            )
        responses_by_rater[rater] = per_rater
    attestation_refs = {}
    for rater, response_hashes in responses_by_rater.items():
        attestation = {
            "schema_version": "1.0",
            "artifact_kind": "human_motion_exit_attestation",
            "study_id": manifest["study_id"],
            "collection_mode": "human",
            "rater_id_hash": rater,
            "attested_at": "2026-08-11T13:00:00Z",
            "affirmations": {
                "independent_human_rater": True,
                "reviewed_both_reversed_presentations": True,
                "responses_are_own": True,
                "no_automated_or_synthetic_ratings": True,
            },
            "response_sha256s": response_hashes,
        }
        relative = f"attestations/{rater}.json"
        path = retained / relative
        _json(path, attestation)
        attestation_refs[rater] = (relative, hash_file(path))
    indexed = []
    for item in response_refs:
        relative, digest = attestation_refs[item["rater_id_hash"]]
        indexed.append(
            {
                **item,
                "attestation_path": relative,
                "attestation_sha256": digest,
            }
        )
    collection = {
        "schema_version": "1.0",
        "artifact_kind": "retained_motion_exit_collection",
        "study_id": manifest["study_id"],
        "collection_mode": "human",
        "package_manifest_sha256": hash_file(package / "package-manifest.json"),
        "responses": indexed,
    }
    path = retained / "collection-manifest.json"
    _json(path, collection)
    return path


@pytest.fixture(scope="module")
def completed_study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("motion-exit")
    pairs = _sources(root / "sources")
    package = root / "package"
    build_motion_exit_study_package(
        study_id="motion_exit_fixture",
        seed=919,
        pairs=pairs,
        rater_id_hashes=tuple(
            sha256_bytes(f"rater-{index}".encode()) for index in range(3)
        ),  # type: ignore[arg-type]
        output_root=package,
        study_mode="synthetic_fixture",
    )
    retained = root / "retained"
    collection = _completed_collection(package, retained)
    return package, retained, collection


def test_verified_synthetic_study_exercises_statistics_without_release(
    completed_study: tuple[Path, Path, Path],
) -> None:
    package, retained, collection = completed_study
    result = import_verified_motion_exit_study(
        package_root=package,
        collection_manifest_path=collection,
        retained_response_root=retained,
    )
    assert result.pair_count == 30
    assert result.response_count == 180
    assert result.rater_count == 3
    assert result.reversal_consistent
    assert result.certification_and_trace_evidence_verified
    assert all(item.v2_majority_wins == 30 for item in result.dimensions)
    assert all(item.statistically_better for item in result.dimensions)
    assert result.study_mode == "synthetic_fixture"
    assert not result.release_allowed


def test_public_packets_expose_playable_motion_without_variant_leakage(
    completed_study: tuple[Path, Path, Path],
) -> None:
    package, _, _ = completed_study
    manifest = json.loads((package / "package-manifest.json").read_text())
    for packet_ref in manifest["rater_packets"]:
        payload = (package / packet_ref["path"]).read_text().lower()
        assert '"variant"' not in payload
        assert '"v2"' not in payload
        assert '"poc"' not in payload
        packet = json.loads(payload)
        for assignment in packet["assignments"]:
            for side in ("left", "right"):
                motion = assignment[side]["motion_evidence"]
                assert motion["fps"] == 30
                assert motion["frame_count"] >= 30
                assert (package / motion["path"]).suffix == ".zip"


def test_image_only_evidence_is_rejected(tmp_path: Path) -> None:
    pairs = _sources(tmp_path / "sources")
    image = tmp_path / "still.png"
    image.write_bytes(_png((1, 2, 3)))
    first = pairs[0]
    invalid = MatchedMotionPairSource(
        pair_id=first.pair_id,
        held_out_case_id=first.held_out_case_id,
        split="held_out",
        v2=MotionVariantSource(
            media_path=image,
            media_sha256=hash_file(image),
            trace_path=first.v2.trace_path,
            trace_sha256=first.v2.trace_sha256,
            trace_summary_path=first.v2.trace_summary_path,
            trace_summary_sha256=first.v2.trace_summary_sha256,
            certification_path=first.v2.certification_path,
            certification_sha256=first.v2.certification_sha256,
        ),
        poc=first.poc,
    )
    with pytest.raises(MotionExitStudyError, match="frame archive"):
        build_motion_exit_study_package(
            study_id="image_only",
            seed=1,
            pairs=(invalid, *pairs[1:]),
            rater_id_hashes=tuple(
                sha256_bytes(f"rater-{index}".encode()) for index in range(3)
            ),  # type: ignore[arg-type]
            output_root=tmp_path / "invalid-package",
            study_mode="synthetic_fixture",
        )


def test_tampered_nested_motion_archive_is_rejected(
    completed_study: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    package, retained, collection = completed_study
    copied = tmp_path / "package"
    shutil.copytree(package, copied)
    manifest = json.loads((copied / "package-manifest.json").read_text())
    packet = json.loads((copied / manifest["rater_packets"][0]["path"]).read_text())
    archive = copied / packet["assignments"][0]["left"]["motion_evidence"]["path"]
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(MotionExitStudyError, match="hash mismatch"):
        import_verified_motion_exit_study(
            package_root=copied,
            collection_manifest_path=collection,
            retained_response_root=retained,
        )


def test_reversal_disagreement_fails_the_gate_without_hiding_the_result(
    completed_study: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    package, retained, collection = completed_study
    copied = tmp_path / "retained"
    shutil.copytree(retained, copied)
    document = json.loads(collection.read_text())
    target = document["responses"][0]
    response_path = copied / target["response_path"]
    response = json.loads(response_path.read_text())
    response["choices"] = {
        dimension: "right" if choice == "left" else "left"
        for dimension, choice in response["choices"].items()
    }
    _json(response_path, response)
    target["response_sha256"] = hash_file(response_path)
    rater = target["rater_id_hash"]
    attestation_path = copied / target["attestation_path"]
    attestation = json.loads(attestation_path.read_text())
    attestation["response_sha256s"][target["presentation_id"]] = target[
        "response_sha256"
    ]
    _json(attestation_path, attestation)
    attestation_hash = hash_file(attestation_path)
    for item in document["responses"]:
        if item["rater_id_hash"] == rater:
            item["attestation_sha256"] = attestation_hash
    updated_collection = copied / "collection-manifest.json"
    _json(updated_collection, document)
    result = import_verified_motion_exit_study(
        package_root=package,
        collection_manifest_path=updated_collection,
        retained_response_root=copied,
    )
    assert not result.reversal_consistent
    assert not result.release_allowed


def test_hash_only_response_claim_cannot_masquerade_as_evidence(
    completed_study: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    package, retained, collection = completed_study
    document = json.loads(collection.read_text())
    document["responses"][0]["response_path"] = "responses/missing.json"
    document["responses"][0]["response_sha256"] = "f" * 64
    invalid = tmp_path / "collection.json"
    _json(invalid, document)
    with pytest.raises(MotionExitStudyError, match="file is missing"):
        import_verified_motion_exit_study(
            package_root=package,
            collection_manifest_path=invalid,
            retained_response_root=retained,
        )


def test_fewer_than_thirty_pairs_is_rejected(tmp_path: Path) -> None:
    pairs = _sources(tmp_path / "sources", count=29)
    with pytest.raises(MotionExitStudyError, match="at least 30"):
        build_motion_exit_study_package(
            study_id="too_small",
            seed=7,
            pairs=pairs,
            rater_id_hashes=tuple(
                sha256_bytes(f"rater-{index}".encode()) for index in range(3)
            ),  # type: ignore[arg-type]
            output_root=tmp_path / "package",
            study_mode="synthetic_fixture",
        )
