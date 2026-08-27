"""Release-grade blinded v2-versus-POC motion-quality exit study."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image
from scipy.stats import binomtest

from rigby_v2.hashing import (
    canonical_json_bytes,
    content_hash,
    hash_file,
    sha256_bytes,
    validate_sha256,
)


EXIT_DIMENSIONS = (
    "phase_timing",
    "peak_speed",
    "jerk_distribution",
    "hand_shape",
)
MINIMUM_EXIT_PAIRS = 30
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_AFFIRMATIONS = {
    "independent_human_rater",
    "reviewed_both_reversed_presentations",
    "responses_are_own",
    "no_automated_or_synthetic_ratings",
}


class MotionExitStudyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MotionVariantSource:
    """Hash-bound 30-FPS frame archive plus authoritative trace evidence.

    ``media_path`` is deliberately not an image. It must be a strict Rigby frame
    archive containing ``playback.json`` and metadata-free PNG frames. A contact
    sheet may be supplied only as supplemental evidence.
    """

    media_path: Path
    media_sha256: str
    trace_path: Path
    trace_sha256: str
    trace_summary_path: Path
    trace_summary_sha256: str
    certification_path: Path | None = None
    certification_sha256: str | None = None
    contact_sheet_path: Path | None = None
    contact_sheet_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class MatchedMotionPairSource:
    pair_id: str
    held_out_case_id: str
    split: Literal["held_out"]
    v2: MotionVariantSource
    poc: MotionVariantSource


@dataclass(frozen=True, slots=True)
class MotionExitPackageEvidence:
    study_id: str
    root: Path
    package_manifest_path: Path
    package_manifest_sha256: str
    pair_count: int
    expected_response_count: int
    study_mode: Literal["release", "synthetic_fixture"]


@dataclass(frozen=True, slots=True)
class ExactDimensionResult:
    dimension: str
    v2_majority_wins: int
    poc_majority_wins: int
    ties_or_inconsistent: int
    total_pairs: int
    v2_win_rate: float
    exact_p_value: float
    ci95_low: float
    ci95_high: float
    statistically_better: bool


@dataclass(frozen=True, slots=True)
class MotionExitStudyResult:
    study_id: str
    pair_count: int
    response_count: int
    rater_count: int
    reversal_checks: int
    reversal_consistent: bool
    certification_and_trace_evidence_verified: bool
    retained_files_verified: bool
    study_mode: Literal["release", "synthetic_fixture"]
    dimensions: tuple[ExactDimensionResult, ...]
    release_allowed: bool
    package_manifest_sha256: str
    collection_manifest_sha256: str
    evidence_set_sha256: str


def _write_new(path: Path, value: object | bytes) -> None:
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _json_bytes(path: Path, expected_sha256: str, *, label: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        observed = sha256_bytes(payload)
        if observed != validate_sha256(expected_sha256):
            raise MotionExitStudyError(f"{label} hash mismatch")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, MotionExitStudyError):
            raise
        raise MotionExitStudyError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise MotionExitStudyError(f"{label} must be a JSON object")
    return value


def _source_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MotionExitStudyError(f"{label} is not a retained regular file")
    payload = resolved.read_bytes()
    if not payload or sha256_bytes(payload) != validate_sha256(expected_sha256):
        raise MotionExitStudyError(f"{label} hash mismatch or empty file")
    return payload


def _normalized_image(payload: bytes, *, label: str) -> tuple[bytes, int, int]:
    """Re-encode a frame to PNG so source metadata cannot leak a variant."""

    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.width < 64 or source.height < 64:
                raise MotionExitStudyError(f"{label} is too small for review")
            image = source.convert("RGB")
    except (OSError, ValueError) as error:
        if isinstance(error, MotionExitStudyError):
            raise
        raise MotionExitStudyError(f"{label} must be a decodable image") from error
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue(), image.width, image.height


def _safe_archive_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise MotionExitStudyError(f"{label} is not a safe archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise MotionExitStudyError(f"{label} is not a safe archive path")
    return path.as_posix()


def _deterministic_zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, entries[name])
    return output.getvalue()


def _normalized_playback_archive(
    payload: bytes, *, label: str
) -> tuple[bytes, dict[str, object]]:
    """Validate and sanitize a playable, time-resolved 30-FPS frame archive."""

    if len(payload) > 512 * 1024 * 1024:
        raise MotionExitStudyError(f"{label} frame archive is too large")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
            infos = source.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or "playback.json" not in names:
                raise MotionExitStudyError(
                    f"{label} needs one playback.json and unique entries"
                )
            if any(
                item.is_dir()
                or item.flag_bits & 0x1
                or _safe_archive_name(item.filename, label=label) != item.filename
                for item in infos
            ):
                raise MotionExitStudyError(f"{label} has unsafe archive entries")
            if sum(item.file_size for item in infos) > 512 * 1024 * 1024:
                raise MotionExitStudyError(f"{label} expands beyond the safe limit")
            playback = json.loads(source.read("playback.json").decode("utf-8"))
            if not isinstance(playback, dict) or set(playback) != {
                "schema_version",
                "artifact_kind",
                "fps",
                "camera",
                "frames",
            }:
                raise MotionExitStudyError(
                    f"{label} playback metadata has unknown or missing fields"
                )
            if (
                playback["schema_version"] != "1.0"
                or playback["artifact_kind"]
                not in {"motion_frame_archive_source", "blinded_motion_frame_archive"}
                or playback["fps"] != 30
            ):
                raise MotionExitStudyError(
                    f"{label} must declare an exact 30-FPS playback stream"
                )
            camera = playback["camera"]
            if not isinstance(camera, dict) or set(camera) != {
                "projection",
                "width_px",
                "height_px",
                "horizontal_fov_deg",
            }:
                raise MotionExitStudyError(f"{label} camera metadata is incomplete")
            if camera["projection"] not in {"perspective", "orthographic"}:
                raise MotionExitStudyError(f"{label} camera projection is unsupported")
            width = int(_positive_number(camera["width_px"], label="camera width"))
            height = int(_positive_number(camera["height_px"], label="camera height"))
            fov = _positive_number(camera["horizontal_fov_deg"], label="camera FOV")
            if width < 64 or height < 64 or not 1.0 <= fov <= 179.0:
                raise MotionExitStudyError(f"{label} camera metadata is invalid")
            frames = playback["frames"]
            if not isinstance(frames, list) or len(frames) < 30:
                raise MotionExitStudyError(
                    f"{label} requires at least one second of 30-FPS motion"
                )
            expected_names = {"playback.json"}
            normalized_frames: list[dict[str, object]] = []
            normalized_entries: dict[str, bytes] = {}
            for index, frame in enumerate(frames):
                if not isinstance(frame, dict) or set(frame) != {
                    "path",
                    "timestamp_s",
                    "sha256",
                }:
                    raise MotionExitStudyError(f"{label} frame metadata is invalid")
                frame_name = _safe_archive_name(frame["path"], label="frame path")
                expected_name = f"frames/{index:06d}.png"
                timestamp = _positive_number(
                    frame["timestamp_s"], label="frame timestamp"
                )
                if frame_name != expected_name or not math.isclose(
                    timestamp, index / 30.0, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise MotionExitStudyError(
                        f"{label} must use contiguous 30-FPS timestamps"
                    )
                raw_frame = source.read(frame_name)
                if sha256_bytes(raw_frame) != validate_sha256(str(frame["sha256"])):
                    raise MotionExitStudyError(f"{label} frame hash mismatch")
                normalized, frame_width, frame_height = _normalized_image(
                    raw_frame, label=f"{label} frame"
                )
                if (frame_width, frame_height) != (width, height):
                    raise MotionExitStudyError(f"{label} frame/camera size mismatch")
                normalized_entries[frame_name] = normalized
                normalized_frames.append(
                    {
                        "path": frame_name,
                        "timestamp_s": index / 30.0,
                        "sha256": sha256_bytes(normalized),
                    }
                )
                expected_names.add(frame_name)
            if set(names) != expected_names:
                raise MotionExitStudyError(f"{label} contains undeclared archive entries")
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        if isinstance(error, MotionExitStudyError):
            raise
        raise MotionExitStudyError(f"{label} is not a valid frame archive") from error
    public_playback = {
        "schema_version": "1.0",
        "artifact_kind": "blinded_motion_frame_archive",
        "fps": 30,
        "camera": {
            "projection": camera["projection"],
            "width_px": width,
            "height_px": height,
            "horizontal_fov_deg": fov,
        },
        "frames": normalized_frames,
    }
    normalized_entries["playback.json"] = canonical_json_bytes(public_playback)
    return _deterministic_zip(normalized_entries), public_playback


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionExitStudyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MotionExitStudyError(f"{label} must be finite and nonnegative")
    return result


def _trace_summary(
    source: MotionVariantSource,
    *,
    case_id: str,
    variant: Literal["v2", "poc"],
) -> tuple[dict[str, object], bytes]:
    trace_payload = _source_file(
        source.trace_path, source.trace_sha256, label=f"{variant} trace"
    )
    summary_payload = _source_file(
        source.trace_summary_path,
        source.trace_summary_sha256,
        label=f"{variant} trace summary",
    )
    summary = _json_bytes(
        source.trace_summary_path,
        source.trace_summary_sha256,
        label=f"{variant} trace summary",
    )
    required = {
        "schema_version",
        "artifact_kind",
        "case_id",
        "variant",
        "split",
        "authoritative_trace_sha256",
        "duration_s",
        "phase_timing_s",
        "peak_speed",
        "jerk_distribution",
        "hand_shape",
    }
    if set(summary) != required:
        raise MotionExitStudyError("trace summary has unknown or missing fields")
    if (
        summary["schema_version"] != "1.0"
        or summary["artifact_kind"] != "motion_trace_summary"
        or summary["case_id"] != case_id
        or summary["variant"] != variant
        or summary["split"] != "held_out"
        or summary["authoritative_trace_sha256"] != sha256_bytes(trace_payload)
    ):
        raise MotionExitStudyError("trace summary identity/evidence binding mismatch")
    duration = _positive_number(summary["duration_s"], label="duration")
    if duration <= 0:
        raise MotionExitStudyError("trace duration must be positive")
    phase_timing = summary["phase_timing_s"]
    jerk = summary["jerk_distribution"]
    hand = summary["hand_shape"]
    if not isinstance(phase_timing, dict) or not phase_timing:
        raise MotionExitStudyError("phase timing summary is missing")
    if not isinstance(jerk, dict) or set(jerk) != {"p50", "p95", "maximum"}:
        raise MotionExitStudyError("jerk distribution must contain p50/p95/maximum")
    if not isinstance(hand, dict) or not hand:
        raise MotionExitStudyError("hand-shape summary is missing")
    normalized_phases = {
        str(name): _positive_number(value, label=f"phase {name}")
        for name, value in phase_timing.items()
    }
    normalized_jerk = {
        name: _positive_number(jerk[name], label=f"jerk {name}")
        for name in ("p50", "p95", "maximum")
    }
    if not (
        normalized_jerk["p50"]
        <= normalized_jerk["p95"]
        <= normalized_jerk["maximum"]
    ):
        raise MotionExitStudyError("jerk quantiles are not monotone")
    normalized_hand = {
        str(name): _positive_number(value, label=f"hand shape {name}")
        for name, value in hand.items()
    }
    public = {
        "schema_version": "1.0",
        "artifact_kind": "blinded_motion_trace_summary",
        "duration_s": duration,
        "phase_timing_s": normalized_phases,
        "peak_speed": _positive_number(summary["peak_speed"], label="peak speed"),
        "jerk_distribution": normalized_jerk,
        "hand_shape": normalized_hand,
    }
    # Reading both here prevents a summary-only shortcut.
    if sha256_bytes(summary_payload) != validate_sha256(source.trace_summary_sha256):
        raise MotionExitStudyError("trace summary changed during verification")
    return public, trace_payload


def _certification(
    source: MotionVariantSource, *, case_id: str, trace_sha256: str
) -> bytes:
    if source.certification_path is None or source.certification_sha256 is None:
        raise MotionExitStudyError("v2 source requires retained certification evidence")
    payload = _source_file(
        source.certification_path,
        source.certification_sha256,
        label="v2 certification evidence",
    )
    document = _json_bytes(
        source.certification_path,
        source.certification_sha256,
        label="v2 certification evidence",
    )
    if set(document) != {
        "schema_version",
        "artifact_kind",
        "case_id",
        "certified",
        "authoritative_trace_sha256",
        "repeat_count",
        "replay_exact",
        "export_reimport_passed",
    }:
        raise MotionExitStudyError("certification evidence has unknown or missing fields")
    if (
        document["schema_version"] != "1.0"
        or document["artifact_kind"] != "rigby_v2_certification_evidence"
        or document["case_id"] != case_id
        or document["certified"] is not True
        or document["authoritative_trace_sha256"] != trace_sha256
        or not isinstance(document["repeat_count"], int)
        or document["repeat_count"] < 3
        or document["replay_exact"] is not True
        or document["export_reimport_passed"] is not True
    ):
        raise MotionExitStudyError("v2 certification evidence is not release-grade")
    return payload


def _blind_id(study_id: str, pair_id: str, variant: str, digest: str) -> str:
    value = hashlib.sha256(
        f"{study_id}\0{pair_id}\0{variant}\0{digest}".encode()
    ).hexdigest()
    return f"blind_{value[:24]}"


def build_motion_exit_study_package(
    *,
    study_id: str,
    seed: int,
    pairs: tuple[MatchedMotionPairSource, ...],
    rater_id_hashes: tuple[str, str, str],
    output_root: Path,
    study_mode: Literal["release", "synthetic_fixture"] = "release",
) -> MotionExitPackageEvidence:
    if not _SAFE_ID.fullmatch(study_id) or seed < 0:
        raise MotionExitStudyError("study identity or seed is invalid")
    if study_mode not in {"release", "synthetic_fixture"}:
        raise MotionExitStudyError("unknown study mode")
    if len(pairs) < MINIMUM_EXIT_PAIRS:
        raise MotionExitStudyError(f"exit study requires at least {MINIMUM_EXIT_PAIRS} pairs")
    raters = tuple(validate_sha256(value) for value in rater_id_hashes)
    if len(set(raters)) != 3:
        raise MotionExitStudyError("exit study requires three independent rater IDs")
    pair_ids = [item.pair_id for item in pairs]
    case_ids = [item.held_out_case_id for item in pairs]
    if (
        len(pair_ids) != len(set(pair_ids))
        or len(case_ids) != len(set(case_ids))
        or any(not _SAFE_ID.fullmatch(value) for value in (*pair_ids, *case_ids))
        or any(item.split != "held_out" for item in pairs)
    ):
        raise MotionExitStudyError("matched pairs must be unique, safe, and held out")
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    rng = random.Random(seed)
    private_pairs = []
    public_records: dict[str, dict[str, object]] = {}
    evidence_entries = []
    pair_tokens: dict[str, str] = {}
    for pair in pairs:
        if (
            pair.v2.trace_sha256 == pair.poc.trace_sha256
            or pair.v2.media_sha256 == pair.poc.media_sha256
        ):
            raise MotionExitStudyError("matched variants need distinct motion evidence")
        pair_token = "pair_" + content_hash(
            {"study": study_id, "pair": pair.pair_id}
        )[:20]
        pair_tokens[pair.pair_id] = pair_token
        variant_records: dict[str, str] = {}
        pair_camera: dict[str, object] | None = None
        for variant, source in (("v2", pair.v2), ("poc", pair.poc)):
            media = _source_file(
                source.media_path,
                source.media_sha256,
                label=f"{variant} motion frame archive",
            )
            normalized_media, playback = _normalized_playback_archive(
                media, label=f"{variant} motion frame archive"
            )
            camera = playback["camera"]
            if pair_camera is None:
                pair_camera = camera  # type: ignore[assignment]
            elif camera != pair_camera:
                raise MotionExitStudyError(
                    "matched variants must use identical camera metadata"
                )
            summary, trace_payload = _trace_summary(
                source,
                case_id=pair.held_out_case_id,
                variant=variant,  # type: ignore[arg-type]
            )
            motion_duration = len(playback["frames"]) / 30.0  # type: ignore[arg-type]
            if not math.isclose(
                float(summary["duration_s"]),
                motion_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise MotionExitStudyError(
                    "trace summary duration does not match playable motion"
                )
            blind_id = _blind_id(
                study_id, pair.pair_id, variant, source.trace_summary_sha256
            )
            variant_records[variant] = blind_id
            media_relative = f"public/motion/{blind_id}.zip"
            summary_relative = f"public/summaries/{blind_id}.json"
            _write_new(root / media_relative, normalized_media)
            public_summary = {**summary, "blind_record_id": blind_id}
            _write_new(root / summary_relative, public_summary)
            contact_sheet_relative = None
            contact_sheet_hash = None
            if (source.contact_sheet_path is None) != (
                source.contact_sheet_sha256 is None
            ):
                raise MotionExitStudyError(
                    "contact-sheet path and hash must be supplied together"
                )
            if source.contact_sheet_path is not None:
                sheet_payload = _source_file(
                    source.contact_sheet_path,
                    str(source.contact_sheet_sha256),
                    label=f"{variant} supplemental contact sheet",
                )
                normalized_sheet, _, _ = _normalized_image(
                    sheet_payload, label=f"{variant} supplemental contact sheet"
                )
                contact_sheet_relative = f"public/contact_sheets/{blind_id}.png"
                _write_new(root / contact_sheet_relative, normalized_sheet)
                contact_sheet_hash = hash_file(root / contact_sheet_relative)
            trace_relative = f"evidence/traces/{source.trace_sha256}.bin"
            if not (root / trace_relative).exists():
                _write_new(root / trace_relative, trace_payload)
            certification_relative = None
            certification_hash = None
            if variant == "v2":
                certification = _certification(
                    source,
                    case_id=pair.held_out_case_id,
                    trace_sha256=source.trace_sha256,
                )
                certification_hash = validate_sha256(str(source.certification_sha256))
                certification_relative = (
                    f"evidence/certifications/{certification_hash}.json"
                )
                if not (root / certification_relative).exists():
                    _write_new(root / certification_relative, certification)
            elif source.certification_path is not None or source.certification_sha256 is not None:
                raise MotionExitStudyError("POC source must not claim v2 certification")
            public_records[blind_id] = {
                "blind_record_id": blind_id,
                "motion_evidence": {
                    "media_type": "application/vnd.rigby.frame-archive+zip",
                    "path": media_relative,
                    "sha256": hash_file(root / media_relative),
                    "fps": 30,
                    "frame_count": len(playback["frames"]),  # type: ignore[arg-type]
                    "duration_s": motion_duration,
                    "camera": camera,
                    "playback_manifest_sha256": sha256_bytes(
                        canonical_json_bytes(playback)
                    ),
                },
                "supplemental_contact_sheet": (
                    {
                        "path": contact_sheet_relative,
                        "sha256": contact_sheet_hash,
                    }
                    if contact_sheet_relative is not None
                    else None
                ),
                "trace_summary_path": summary_relative,
                "trace_summary_sha256": hash_file(root / summary_relative),
            }
            evidence_entries.append(
                {
                    "pair_id": pair.pair_id,
                    "held_out_case_id": pair.held_out_case_id,
                    "variant": variant,
                    "blind_record_id": blind_id,
                    "source_motion_media_sha256": validate_sha256(
                        source.media_sha256
                    ),
                    "source_trace_summary_sha256": validate_sha256(
                        source.trace_summary_sha256
                    ),
                    "trace_path": trace_relative,
                    "trace_sha256": validate_sha256(source.trace_sha256),
                    "certification_path": certification_relative,
                    "certification_sha256": certification_hash,
                }
            )
        private_pairs.append(
            {
                "pair_id": pair.pair_id,
                "pair_token": pair_token,
                "held_out_case_id": pair.held_out_case_id,
                "v2_blind_record_id": variant_records["v2"],
                "poc_blind_record_id": variant_records["poc"],
            }
        )

    packet_entries = []
    for rater_index, rater in enumerate(raters):
        assignments = []
        for pair in pairs:
            private = next(item for item in private_pairs if item["pair_id"] == pair.pair_id)
            v2_id = str(private["v2_blind_record_id"])
            poc_id = str(private["poc_blind_record_id"])
            v2_first = bool(rng.getrandbits(1))
            orders = ((v2_id, poc_id), (poc_id, v2_id))
            if not v2_first:
                orders = tuple(reversed(orders))
            for reversal_index, (left_id, right_id) in enumerate(orders):
                presentation_id = "presentation_" + content_hash(
                    {
                        "study": study_id,
                        "pair": pair.pair_id,
                        "rater": rater,
                        "reversal": reversal_index,
                    }
                )[:24]
                assignments.append(
                    {
                        "presentation_id": presentation_id,
                        "pair_token": private["pair_token"],
                        "left": public_records[left_id],
                        "right": public_records[right_id],
                        "dimensions": list(EXIT_DIMENSIONS),
                        "allowed_choices": ["left", "right", "tie"],
                    }
                )
        rng.shuffle(assignments)
        packet = {
            "schema_version": "1.0",
            "artifact_kind": "blinded_motion_exit_rater_packet",
            "study_id": study_id,
            "rater_id_hash": rater,
            "assignments": assignments,
        }
        relative = f"public/rater_packets/{rater}.json"
        _write_new(root / relative, packet)
        packet_entries.append(
            {"rater_id_hash": rater, "path": relative, "sha256": hash_file(root / relative)}
        )

    private_map = {
        "schema_version": "1.0",
        "artifact_kind": "motion_exit_private_variant_map",
        "study_id": study_id,
        "pairs": private_pairs,
    }
    private_path = root / "private/variant_map.json"
    _write_new(private_path, private_map)
    evidence_manifest = {
        "schema_version": "1.0",
        "artifact_kind": "motion_exit_source_evidence",
        "study_id": study_id,
        "records": evidence_entries,
    }
    evidence_path = root / "evidence/source_evidence.json"
    _write_new(evidence_path, evidence_manifest)
    package_manifest = {
        "schema_version": "1.0",
        "artifact_kind": "blinded_motion_exit_package",
        "study_id": study_id,
        "study_mode": study_mode,
        "seed": seed,
        "pair_count": len(pairs),
        "dimensions": list(EXIT_DIMENSIONS),
        "private_variant_map": {
            "path": "private/variant_map.json",
            "sha256": hash_file(private_path),
        },
        "source_evidence": {
            "path": "evidence/source_evidence.json",
            "sha256": hash_file(evidence_path),
        },
        "rater_packets": packet_entries,
    }
    manifest_path = root / "package-manifest.json"
    _write_new(manifest_path, package_manifest)
    return MotionExitPackageEvidence(
        study_id=study_id,
        root=root,
        package_manifest_path=manifest_path,
        package_manifest_sha256=hash_file(manifest_path),
        pair_count=len(pairs),
        expected_response_count=len(pairs) * 3 * 2,
        study_mode=study_mode,
    )


def _relative_file(
    root: Path, relative_value: object, expected_hash: object, *, label: str
) -> tuple[Path, str, str, bytes]:
    if not isinstance(relative_value, str) or not relative_value or "\\" in relative_value:
        raise MotionExitStudyError(f"{label} path is not a safe relative POSIX path")
    pure = PurePosixPath(relative_value)
    if pure.is_absolute() or ".." in pure.parts or not pure.name:
        raise MotionExitStudyError(f"{label} path is not a safe relative POSIX path")
    root = root.resolve()
    cursor = root
    for part in pure.parts:
        cursor /= part
        if cursor.is_symlink():
            raise MotionExitStudyError(f"{label} path contains a symlink")
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MotionExitStudyError(f"{label} path escaped its root") from error
    if not path.is_file():
        raise MotionExitStudyError(f"{label} file is missing")
    payload = path.read_bytes()
    digest = validate_sha256(str(expected_hash))
    if not payload or sha256_bytes(payload) != digest:
        raise MotionExitStudyError(f"{label} retained file hash mismatch")
    return path, pure.as_posix(), digest, payload


def _load_package(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    root = root.resolve()
    manifest_path = root / "package-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise MotionExitStudyError("package manifest is missing")
    manifest_payload = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_payload)
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionExitStudyError("invalid package manifest") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "artifact_kind",
        "study_id",
        "study_mode",
        "seed",
        "pair_count",
        "dimensions",
        "private_variant_map",
        "source_evidence",
        "rater_packets",
    }:
        raise MotionExitStudyError("invalid package manifest")
    if (
        manifest["schema_version"] != "1.0"
        or manifest["artifact_kind"] != "blinded_motion_exit_package"
        or manifest["study_mode"] not in {"release", "synthetic_fixture"}
        or not _SAFE_ID.fullmatch(str(manifest["study_id"]))
        or not isinstance(manifest["seed"], int)
        or not isinstance(manifest["pair_count"], int)
        or not isinstance(manifest["private_variant_map"], dict)
        or not isinstance(manifest["source_evidence"], dict)
        or not isinstance(manifest["rater_packets"], list)
    ):
        raise MotionExitStudyError("package manifest identity/schema is invalid")
    for label in ("private_variant_map", "source_evidence"):
        if set(manifest[label]) != {"path", "sha256"}:  # type: ignore[arg-type]
            raise MotionExitStudyError(f"invalid {label} reference")
    _, _, _, private_payload = _relative_file(
        root,
        manifest["private_variant_map"]["path"],  # type: ignore[index]
        manifest["private_variant_map"]["sha256"],  # type: ignore[index]
        label="private variant map",
    )
    _, _, _, evidence_payload = _relative_file(
        root,
        manifest["source_evidence"]["path"],  # type: ignore[index]
        manifest["source_evidence"]["sha256"],  # type: ignore[index]
        label="source evidence manifest",
    )
    try:
        private = json.loads(private_payload.decode("utf-8"))
        evidence = json.loads(evidence_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionExitStudyError("package private evidence is invalid") from error
    if (
        not isinstance(private, dict)
        or set(private) != {"schema_version", "artifact_kind", "study_id", "pairs"}
        or private["schema_version"] != "1.0"
        or private["artifact_kind"] != "motion_exit_private_variant_map"
        or private["study_id"] != manifest["study_id"]
        or not isinstance(private["pairs"], list)
        or not isinstance(evidence, dict)
        or set(evidence) != {"schema_version", "artifact_kind", "study_id", "records"}
        or evidence["schema_version"] != "1.0"
        or evidence["artifact_kind"] != "motion_exit_source_evidence"
        or evidence["study_id"] != manifest["study_id"]
        or not isinstance(evidence["records"], list)
    ):
        raise MotionExitStudyError("package private evidence is invalid")
    return manifest, private, evidence, manifest_hash


def _verify_public_record(root: Path, record: object) -> str:
    if not isinstance(record, dict) or set(record) != {
        "blind_record_id",
        "motion_evidence",
        "supplemental_contact_sheet",
        "trace_summary_path",
        "trace_summary_sha256",
    }:
        raise MotionExitStudyError("public motion record is invalid")
    blind_id = str(record["blind_record_id"])
    if not blind_id.startswith("blind_") or not _SAFE_ID.fullmatch(blind_id):
        raise MotionExitStudyError("public blind record ID is invalid")
    motion = record["motion_evidence"]
    if not isinstance(motion, dict) or set(motion) != {
        "media_type",
        "path",
        "sha256",
        "fps",
        "frame_count",
        "duration_s",
        "camera",
        "playback_manifest_sha256",
    }:
        raise MotionExitStudyError("playable motion evidence is missing")
    if (
        motion["media_type"] != "application/vnd.rigby.frame-archive+zip"
        or motion["fps"] != 30
        or not isinstance(motion["frame_count"], int)
        or motion["frame_count"] < 30
    ):
        raise MotionExitStudyError("public evidence is not playable 30-FPS motion")
    _, _, _, archive_payload = _relative_file(
        root, motion["path"], motion["sha256"], label="playable motion archive"
    )
    normalized, playback = _normalized_playback_archive(
        archive_payload, label="playable motion archive"
    )
    if normalized != archive_payload:
        raise MotionExitStudyError("playable motion archive is not sanitized/canonical")
    if (
        motion["frame_count"] != len(playback["frames"])  # type: ignore[arg-type]
        or motion["camera"] != playback["camera"]
        or not math.isclose(
            float(motion["duration_s"]),
            len(playback["frames"]) / 30.0,  # type: ignore[arg-type]
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or validate_sha256(str(motion["playback_manifest_sha256"]))
        != sha256_bytes(canonical_json_bytes(playback))
    ):
        raise MotionExitStudyError("playback camera/timestamp binding mismatch")
    sheet = record["supplemental_contact_sheet"]
    if sheet is not None:
        if not isinstance(sheet, dict) or set(sheet) != {"path", "sha256"}:
            raise MotionExitStudyError("supplemental contact sheet reference is invalid")
        _, _, _, sheet_payload = _relative_file(
            root, sheet["path"], sheet["sha256"], label="supplemental contact sheet"
        )
        normalized_sheet, _, _ = _normalized_image(
            sheet_payload, label="supplemental contact sheet"
        )
        if normalized_sheet != sheet_payload:
            raise MotionExitStudyError("supplemental contact sheet is not sanitized")
    _, _, _, summary_payload = _relative_file(
        root,
        record["trace_summary_path"],
        record["trace_summary_sha256"],
        label="blinded trace summary",
    )
    try:
        summary = json.loads(summary_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionExitStudyError("blinded trace summary is invalid") from error
    if not isinstance(summary, dict) or set(summary) != {
        "schema_version",
        "artifact_kind",
        "blind_record_id",
        "duration_s",
        "phase_timing_s",
        "peak_speed",
        "jerk_distribution",
        "hand_shape",
    }:
        raise MotionExitStudyError("blinded trace summary schema is invalid")
    if (
        summary["schema_version"] != "1.0"
        or summary["artifact_kind"] != "blinded_motion_trace_summary"
        or summary["blind_record_id"] != blind_id
        or not math.isclose(
            float(summary["duration_s"]),
            float(motion["duration_s"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise MotionExitStudyError("blinded trace summary identity is invalid")
    return blind_id


def _verified_packet(
    root: Path,
    manifest: dict[str, object],
    packet_ref: object,
) -> tuple[str, dict[str, dict[str, object]]]:
    if not isinstance(packet_ref, dict) or set(packet_ref) != {
        "rater_id_hash",
        "path",
        "sha256",
    }:
        raise MotionExitStudyError("rater packet reference is invalid")
    rater = validate_sha256(str(packet_ref["rater_id_hash"]))
    _, _, _, payload = _relative_file(
        root, packet_ref["path"], packet_ref["sha256"], label="rater packet"
    )
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionExitStudyError("rater packet is invalid") from error
    serialized = json.dumps(document, sort_keys=True).lower()
    if '"variant"' in serialized or '"v2"' in serialized or '"poc"' in serialized:
        raise MotionExitStudyError("public rater packet leaks variant identity")
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "artifact_kind",
        "study_id",
        "rater_id_hash",
        "assignments",
    }:
        raise MotionExitStudyError("rater packet schema is invalid")
    assignments = document["assignments"]
    if (
        document["schema_version"] != "1.0"
        or document["artifact_kind"] != "blinded_motion_exit_rater_packet"
        or document["study_id"] != manifest["study_id"]
        or document["rater_id_hash"] != rater
        or not isinstance(assignments, list)
        or len(assignments) != int(manifest["pair_count"]) * 2
    ):
        raise MotionExitStudyError("rater packet identity/count is invalid")
    by_presentation: dict[str, dict[str, object]] = {}
    pair_orders: dict[str, list[tuple[str, str]]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {
            "presentation_id",
            "pair_token",
            "left",
            "right",
            "dimensions",
            "allowed_choices",
        }:
            raise MotionExitStudyError("rater assignment schema is invalid")
        presentation = str(assignment["presentation_id"])
        pair_token = str(assignment["pair_token"])
        if (
            not _SAFE_ID.fullmatch(presentation)
            or not _SAFE_ID.fullmatch(pair_token)
            or presentation in by_presentation
            or tuple(assignment["dimensions"]) != EXIT_DIMENSIONS  # type: ignore[arg-type]
            or assignment["allowed_choices"] != ["left", "right", "tie"]
        ):
            raise MotionExitStudyError("rater assignment identity/options are invalid")
        left_id = _verify_public_record(root, assignment["left"])
        right_id = _verify_public_record(root, assignment["right"])
        if left_id == right_id:
            raise MotionExitStudyError("rater assignment compares one record to itself")
        pair_orders.setdefault(pair_token, []).append((left_id, right_id))
        by_presentation[presentation] = assignment
    if len(pair_orders) != int(manifest["pair_count"]) or any(
        len(orders) != 2 or orders[0] != tuple(reversed(orders[1]))
        for orders in pair_orders.values()
    ):
        raise MotionExitStudyError("rater packet reversal presentations are invalid")
    return rater, by_presentation


def _verify_package_artifacts(
    root: Path,
    manifest: dict[str, object],
    private: dict[str, object],
    evidence: dict[str, object],
) -> bool:
    if int(manifest.get("pair_count", 0)) < MINIMUM_EXIT_PAIRS:
        raise MotionExitStudyError("package contains too few held-out pairs")
    if tuple(manifest.get("dimensions", ())) != EXIT_DIMENSIONS:
        raise MotionExitStudyError("package dimensions differ from the exit gate")
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != int(manifest["pair_count"]) * 2:
        raise MotionExitStudyError("source evidence does not cover both variants")
    seen_records: set[tuple[str, str]] = set()
    evidence_identity: dict[tuple[str, str], tuple[str, str]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "pair_id",
            "held_out_case_id",
            "variant",
            "blind_record_id",
            "source_motion_media_sha256",
            "source_trace_summary_sha256",
            "trace_path",
            "trace_sha256",
            "certification_path",
            "certification_sha256",
        }:
            raise MotionExitStudyError("source evidence record is invalid")
        identity = (str(record["pair_id"]), str(record["variant"]))
        if identity in seen_records:
            raise MotionExitStudyError("duplicate source evidence record")
        seen_records.add(identity)
        if (
            not _SAFE_ID.fullmatch(identity[0])
            or not _SAFE_ID.fullmatch(str(record["held_out_case_id"]))
            or not str(record["blind_record_id"]).startswith("blind_")
        ):
            raise MotionExitStudyError("source evidence identity is invalid")
        try:
            validate_sha256(str(record["source_motion_media_sha256"]))
            validate_sha256(str(record["source_trace_summary_sha256"]))
        except ValueError as error:
            raise MotionExitStudyError("source evidence hash is invalid") from error
        evidence_identity[identity] = (
            str(record["held_out_case_id"]),
            str(record["blind_record_id"]),
        )
        _, _, trace_hash, _ = _relative_file(
            root, record["trace_path"], record["trace_sha256"], label="source trace"
        )
        if record["variant"] == "v2":
            _, _, _, cert_payload = _relative_file(
                root,
                record["certification_path"],
                record["certification_sha256"],
                label="v2 certification",
            )
            try:
                cert = json.loads(cert_payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MotionExitStudyError("v2 certification is invalid") from error
            if (
                not isinstance(cert, dict)
                or set(cert)
                != {
                    "schema_version",
                    "artifact_kind",
                    "case_id",
                    "certified",
                    "authoritative_trace_sha256",
                    "repeat_count",
                    "replay_exact",
                    "export_reimport_passed",
                }
                or cert.get("schema_version") != "1.0"
                or cert.get("artifact_kind") != "rigby_v2_certification_evidence"
                or cert.get("case_id") != record["held_out_case_id"]
                or cert.get("certified") is not True
                or cert.get("authoritative_trace_sha256") != trace_hash
                or not isinstance(cert.get("repeat_count"), int)
                or isinstance(cert.get("repeat_count"), bool)
                or int(cert["repeat_count"]) < 3
                or cert.get("replay_exact") is not True
                or cert.get("export_reimport_passed") is not True
            ):
                raise MotionExitStudyError("v2 certification no longer verifies")
        elif record["variant"] == "poc":
            if (
                record["certification_path"] is not None
                or record["certification_sha256"] is not None
            ):
                raise MotionExitStudyError("POC evidence cannot claim v2 certification")
        else:
            raise MotionExitStudyError("unknown private variant")
    pairs = private.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != int(manifest["pair_count"]):
        raise MotionExitStudyError("private variant map pair count is invalid")
    private_pair_ids: set[str] = set()
    private_pair_tokens: set[str] = set()
    private_blind_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "pair_id",
            "pair_token",
            "held_out_case_id",
            "v2_blind_record_id",
            "poc_blind_record_id",
        }:
            raise MotionExitStudyError("private variant-map record is invalid")
        pair_id = str(pair["pair_id"])
        pair_token = str(pair["pair_token"])
        case_id = str(pair["held_out_case_id"])
        v2_id = str(pair["v2_blind_record_id"])
        poc_id = str(pair["poc_blind_record_id"])
        if (
            pair_id in private_pair_ids
            or pair_token in private_pair_tokens
            or v2_id == poc_id
            or v2_id in private_blind_ids
            or poc_id in private_blind_ids
            or evidence_identity.get((pair_id, "v2")) != (case_id, v2_id)
            or evidence_identity.get((pair_id, "poc")) != (case_id, poc_id)
        ):
            raise MotionExitStudyError("private/source evidence identity mismatch")
        private_pair_ids.add(pair_id)
        private_pair_tokens.add(pair_token)
        private_blind_ids.update((v2_id, poc_id))
    packet_refs = manifest.get("rater_packets")
    if not isinstance(packet_refs, list) or len(packet_refs) != 3:
        raise MotionExitStudyError("exactly three independent rater packets are required")
    verified_packets = [_verified_packet(root, manifest, ref) for ref in packet_refs]
    raters = [item[0] for item in verified_packets]
    if len(set(raters)) != 3:
        raise MotionExitStudyError("rater packets are not independent")
    for _, assignments in verified_packets:
        packet_blind_ids = {
            str(assignment[side]["blind_record_id"])  # type: ignore[index]
            for assignment in assignments.values()
            for side in ("left", "right")
        }
        packet_tokens = {
            str(assignment["pair_token"]) for assignment in assignments.values()
        }
        if packet_blind_ids != private_blind_ids or packet_tokens != private_pair_tokens:
            raise MotionExitStudyError("rater packet does not cover the private pair map")
    return True


def _aware_timestamp(value: object, *, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise MotionExitStudyError(f"invalid {label} timestamp") from error
    if parsed.tzinfo is None:
        raise MotionExitStudyError(f"{label} timestamp must be timezone-aware")


def _exact_dimension(
    dimension: str, outcomes: list[str]
) -> ExactDimensionResult:
    total = len(outcomes)
    v2_wins = outcomes.count("v2")
    poc_wins = outcomes.count("poc")
    ties = total - v2_wins - poc_wins
    test = binomtest(v2_wins, total, p=0.5, alternative="greater")
    interval = test.proportion_ci(confidence_level=0.95, method="exact")
    statistically_better = float(interval.low) > 0.5 and float(test.pvalue) < 0.05
    return ExactDimensionResult(
        dimension=dimension,
        v2_majority_wins=v2_wins,
        poc_majority_wins=poc_wins,
        ties_or_inconsistent=ties,
        total_pairs=total,
        v2_win_rate=v2_wins / total,
        exact_p_value=float(test.pvalue),
        ci95_low=float(interval.low),
        ci95_high=float(interval.high),
        statistically_better=statistically_better,
    )


def import_verified_motion_exit_study(
    *,
    package_root: Path,
    collection_manifest_path: Path,
    retained_response_root: Path,
) -> MotionExitStudyResult:
    manifest, private, evidence, package_hash = _load_package(package_root)
    evidence_verified = _verify_package_artifacts(
        package_root, manifest, private, evidence
    )
    study_id = str(manifest["study_id"])
    packet_by_rater: dict[str, dict[str, dict[str, object]]] = {}
    for packet_ref in manifest["rater_packets"]:  # type: ignore[index]
        rater, assignments = _verified_packet(
            package_root, manifest, packet_ref
        )
        packet_by_rater[rater] = assignments
    if len(packet_by_rater) != 3:
        raise MotionExitStudyError("package does not have three independent raters")
    expected_slots = {
        (rater, presentation): assignment
        for rater, assignments in packet_by_rater.items()
        for presentation, assignment in assignments.items()
    }
    collection_manifest_path = collection_manifest_path.resolve()
    if not collection_manifest_path.is_file() or collection_manifest_path.is_symlink():
        raise MotionExitStudyError("collection manifest is not a retained regular file")
    collection_payload = collection_manifest_path.read_bytes()
    collection_hash = sha256_bytes(collection_payload)
    try:
        collection = json.loads(collection_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionExitStudyError("collection manifest is invalid") from error
    if not isinstance(collection, dict) or set(collection) != {
        "schema_version",
        "artifact_kind",
        "study_id",
        "collection_mode",
        "package_manifest_sha256",
        "responses",
    }:
        raise MotionExitStudyError("collection manifest has unknown or missing fields")
    if (
        collection["schema_version"] != "1.0"
        or collection["artifact_kind"] != "retained_motion_exit_collection"
        or collection["collection_mode"] != "human"
        or collection["study_id"] != study_id
        or collection["package_manifest_sha256"] != package_hash
        or not isinstance(collection["responses"], list)
    ):
        raise MotionExitStudyError("collection does not bind the human exit-study package")
    response_fields = {
        "rater_id_hash",
        "presentation_id",
        "response_path",
        "response_sha256",
        "attestation_path",
        "attestation_sha256",
    }
    observed: dict[tuple[str, str], dict[str, object]] = {}
    response_hash_by_rater: dict[str, dict[str, str]] = {
        rater: {} for rater in packet_by_rater
    }
    attestation_ref_by_rater: dict[str, tuple[str, str]] = {}
    attestation_documents: dict[tuple[str, str], dict[str, object]] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    response_refs = []
    for item in collection["responses"]:
        if not isinstance(item, dict) or set(item) != response_fields:
            raise MotionExitStudyError("response index has unknown or missing fields")
        slot = (str(item["rater_id_hash"]), str(item["presentation_id"]))
        if slot not in expected_slots or slot in observed:
            raise MotionExitStudyError("response is duplicate or outside assigned slots")
        path, relative, digest, payload = _relative_file(
            retained_response_root,
            item["response_path"],
            item["response_sha256"],
            label="retained response",
        )
        del path
        if relative in seen_paths or digest in seen_hashes:
            raise MotionExitStudyError("every response slot needs distinct retained bytes")
        seen_paths.add(relative)
        seen_hashes.add(digest)
        try:
            response = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MotionExitStudyError("retained response is invalid") from error
        if not isinstance(response, dict) or set(response) != {
            "schema_version",
            "artifact_kind",
            "study_id",
            "collection_mode",
            "rater_id_hash",
            "presentation_id",
            "recorded_at",
            "choices",
        }:
            raise MotionExitStudyError("retained response has unknown or missing fields")
        if (
            response["schema_version"] != "1.0"
            or response["artifact_kind"] != "human_motion_exit_response"
            or response["collection_mode"] != "human"
            or response["study_id"] != study_id
            or response["rater_id_hash"] != slot[0]
            or response["presentation_id"] != slot[1]
        ):
            raise MotionExitStudyError("retained response identity mismatch")
        _aware_timestamp(response["recorded_at"], label="response")
        choices = response["choices"]
        if (
            not isinstance(choices, dict)
            or set(choices) != set(EXIT_DIMENSIONS)
            or any(value not in {"left", "right", "tie"} for value in choices.values())
        ):
            raise MotionExitStudyError("response choices do not cover exact exit dimensions")
        observed[slot] = response
        response_hash_by_rater[slot[0]][slot[1]] = digest
        response_refs.append({"path": relative, "sha256": digest})

        _, att_relative, att_digest, att_payload = _relative_file(
            retained_response_root,
            item["attestation_path"],
            item["attestation_sha256"],
            label="retained attestation",
        )
        if att_relative in seen_paths:
            raise MotionExitStudyError("response and attestation paths must be distinct")
        reference = (att_relative, att_digest)
        if attestation_ref_by_rater.setdefault(slot[0], reference) != reference:
            raise MotionExitStudyError("one rater must use one retained attestation")
        if reference not in attestation_documents:
            try:
                value = json.loads(att_payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MotionExitStudyError("attestation must be valid JSON") from error
            if not isinstance(value, dict):
                raise MotionExitStudyError("attestation must be an object")
            attestation_documents[reference] = value
    if set(observed) != set(expected_slots):
        raise MotionExitStudyError(
            f"collection is missing {len(set(expected_slots) - set(observed))} responses"
        )
    if len(set(attestation_ref_by_rater.values())) != 3:
        raise MotionExitStudyError("three distinct retained attestations are required")
    if seen_paths.intersection(
        path for path, _ in attestation_ref_by_rater.values()
    ):
        raise MotionExitStudyError("response and attestation files must be distinct")
    for rater, reference in attestation_ref_by_rater.items():
        attestation = attestation_documents[reference]
        if set(attestation) != {
            "schema_version",
            "artifact_kind",
            "study_id",
            "collection_mode",
            "rater_id_hash",
            "attested_at",
            "affirmations",
            "response_sha256s",
        }:
            raise MotionExitStudyError("attestation has unknown or missing fields")
        if (
            attestation["schema_version"] != "1.0"
            or attestation["artifact_kind"] != "human_motion_exit_attestation"
            or attestation["collection_mode"] != "human"
            or attestation["study_id"] != study_id
            or attestation["rater_id_hash"] != rater
        ):
            raise MotionExitStudyError("attestation identity mismatch")
        _aware_timestamp(attestation["attested_at"], label="attestation")
        affirmations = attestation["affirmations"]
        if (
            not isinstance(affirmations, dict)
            or set(affirmations) != _AFFIRMATIONS
            or any(value is not True for value in affirmations.values())
        ):
            raise MotionExitStudyError("attestation human/reversal affirmations are incomplete")
        declared = attestation["response_sha256s"]
        if not isinstance(declared, dict) or {
            str(key): validate_sha256(str(value)) for key, value in declared.items()
        } != response_hash_by_rater[rater]:
            raise MotionExitStudyError("attestation does not bind every retained response")

    blind_variant = {}
    pair_by_token = {}
    for pair in private["pairs"]:  # type: ignore[index]
        blind_variant[str(pair["v2_blind_record_id"])] = "v2"
        blind_variant[str(pair["poc_blind_record_id"])] = "poc"
        pair_by_token[str(pair["pair_token"])] = str(pair["pair_id"])
    rater_choices: dict[tuple[str, str, str], list[str]] = {}
    for (rater, presentation), response in observed.items():
        assignment = expected_slots[(rater, presentation)]
        left_variant = blind_variant[str(assignment["left"]["blind_record_id"])]  # type: ignore[index]
        right_variant = blind_variant[str(assignment["right"]["blind_record_id"])]  # type: ignore[index]
        pair_id = pair_by_token[str(assignment["pair_token"])]
        for dimension in EXIT_DIMENSIONS:
            choice = response["choices"][dimension]  # type: ignore[index]
            normalized = (
                left_variant if choice == "left" else right_variant if choice == "right" else "tie"
            )
            rater_choices.setdefault((pair_id, rater, dimension), []).append(normalized)
    reversal_consistent = all(
        len(values) == 2 and values[0] == values[1]
        for values in rater_choices.values()
    )
    pair_ids = sorted(pair_by_token.values())
    outcomes_by_dimension = {dimension: [] for dimension in EXIT_DIMENSIONS}
    for pair_id in pair_ids:
        for dimension in EXIT_DIMENSIONS:
            collapsed = []
            for rater in packet_by_rater:
                values = rater_choices[(pair_id, rater, dimension)]
                collapsed.append(values[0] if len(values) == 2 and values[0] == values[1] else "tie")
            v2_count = collapsed.count("v2")
            poc_count = collapsed.count("poc")
            outcomes_by_dimension[dimension].append(
                "v2" if v2_count >= 2 else "poc" if poc_count >= 2 else "tie"
            )
    dimensions = tuple(
        _exact_dimension(dimension, outcomes_by_dimension[dimension])
        for dimension in EXIT_DIMENSIONS
    )
    study_mode = str(manifest["study_mode"])
    if study_mode not in {"release", "synthetic_fixture"}:
        raise MotionExitStudyError("unknown study mode")
    evidence_set_hash = content_hash(
        {
            "package_manifest_sha256": package_hash,
            "collection_manifest_sha256": collection_hash,
            "responses": sorted(response_refs, key=lambda item: item["path"]),
            "attestations": sorted(
                (
                    {"path": path, "sha256": digest}
                    for path, digest in set(attestation_ref_by_rater.values())
                ),
                key=lambda item: item["path"],
            ),
        }
    )
    release_allowed = bool(
        study_mode == "release"
        and evidence_verified
        and reversal_consistent
        and all(item.statistically_better for item in dimensions)
    )
    return MotionExitStudyResult(
        study_id=study_id,
        pair_count=len(pair_ids),
        response_count=len(observed),
        rater_count=3,
        reversal_checks=len(rater_choices),
        reversal_consistent=reversal_consistent,
        certification_and_trace_evidence_verified=evidence_verified,
        retained_files_verified=True,
        study_mode=study_mode,  # type: ignore[arg-type]
        dimensions=dimensions,
        release_allowed=release_allowed,
        package_manifest_sha256=package_hash,
        collection_manifest_sha256=collection_hash,
        evidence_set_sha256=evidence_set_hash,
    )
