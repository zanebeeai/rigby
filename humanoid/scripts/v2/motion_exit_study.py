from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from rigby_v2.calibration import (
    MatchedMotionPairSource,
    MotionVariantSource,
    build_motion_exit_study_package,
    import_verified_motion_exit_study,
)


def _relative_source(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.name:
        raise ValueError(f"{label} must be a relative POSIX path")
    root = root.resolve()
    candidate = root.joinpath(*pure.parts).resolve()
    if root != candidate.parent and root not in candidate.parents:
        raise ValueError(f"{label} escaped the specification directory")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} is not a retained regular file")
    return candidate


def _variant(root: Path, value: object, *, v2: bool) -> MotionVariantSource:
    fields = {
        "motion_archive_path",
        "motion_archive_sha256",
        "trace_path",
        "trace_sha256",
        "trace_summary_path",
        "trace_summary_sha256",
        "certification_path",
        "certification_sha256",
        "supplemental_contact_sheet_path",
        "supplemental_contact_sheet_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("variant source has unknown or missing fields")
    certification_path = value["certification_path"]
    certification_hash = value["certification_sha256"]
    if v2 != (certification_path is not None and certification_hash is not None):
        raise ValueError("only v2 must carry retained certification evidence")
    sheet_path = value["supplemental_contact_sheet_path"]
    sheet_hash = value["supplemental_contact_sheet_sha256"]
    if (sheet_path is None) != (sheet_hash is None):
        raise ValueError("supplemental contact-sheet path/hash mismatch")
    return MotionVariantSource(
        media_path=_relative_source(
            root, value["motion_archive_path"], label="motion archive"
        ),
        media_sha256=str(value["motion_archive_sha256"]),
        trace_path=_relative_source(root, value["trace_path"], label="trace"),
        trace_sha256=str(value["trace_sha256"]),
        trace_summary_path=_relative_source(
            root, value["trace_summary_path"], label="trace summary"
        ),
        trace_summary_sha256=str(value["trace_summary_sha256"]),
        certification_path=(
            _relative_source(
                root, certification_path, label="certification evidence"
            )
            if certification_path is not None
            else None
        ),
        certification_sha256=(
            str(certification_hash) if certification_hash is not None else None
        ),
        contact_sheet_path=(
            _relative_source(root, sheet_path, label="supplemental contact sheet")
            if sheet_path is not None
            else None
        ),
        contact_sheet_sha256=str(sheet_hash) if sheet_hash is not None else None,
    )


def _prepare(arguments: argparse.Namespace) -> int:
    specification_path = arguments.specification.resolve()
    if not specification_path.is_file() or specification_path.is_symlink():
        raise ValueError("specification is not a retained regular file")
    document = json.loads(specification_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "study_id",
        "seed",
        "rater_id_hashes",
        "pairs",
    }:
        raise ValueError("exit-study specification has unknown or missing fields")
    if document["schema_version"] != "1.0" or not isinstance(
        document["pairs"], list
    ):
        raise ValueError("exit-study specification schema is invalid")
    root = specification_path.parent
    pairs = []
    for value in document["pairs"]:
        if not isinstance(value, dict) or set(value) != {
            "pair_id",
            "held_out_case_id",
            "split",
            "v2",
            "poc",
        }:
            raise ValueError("matched-pair source has unknown or missing fields")
        pairs.append(
            MatchedMotionPairSource(
                pair_id=str(value["pair_id"]),
                held_out_case_id=str(value["held_out_case_id"]),
                split=value["split"],
                v2=_variant(root, value["v2"], v2=True),
                poc=_variant(root, value["poc"], v2=False),
            )
        )
    raters = tuple(document["rater_id_hashes"])
    if len(raters) != 3:
        raise ValueError("exit study requires exactly three independent raters")
    evidence = build_motion_exit_study_package(
        study_id=str(document["study_id"]),
        seed=int(document["seed"]),
        pairs=tuple(pairs),
        rater_id_hashes=raters,  # type: ignore[arg-type]
        output_root=arguments.output_root,
        study_mode=("synthetic_fixture" if arguments.synthetic_fixture else "release"),
    )
    print(json.dumps(asdict(evidence), indent=2, sort_keys=True, default=str))
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    result = import_verified_motion_exit_study(
        package_root=arguments.package_root,
        collection_manifest_path=arguments.collection_manifest,
        retained_response_root=arguments.retained_response_root,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True, default=str))
    return 0 if result.release_allowed else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the blinded v2-versus-POC motion exit study."
    )
    subparsers = parser.add_subparsers(required=True)
    prepare = subparsers.add_parser(
        "prepare", help="build a hash-bound blinded review package"
    )
    prepare.add_argument("specification", type=Path)
    prepare.add_argument("output_root", type=Path)
    prepare.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="mark logic-test data as permanently ineligible for release",
    )
    prepare.set_defaults(handler=_prepare)
    verify = subparsers.add_parser(
        "verify", help="verify retained human responses and evaluate the gate"
    )
    verify.add_argument("package_root", type=Path)
    verify.add_argument("collection_manifest", type=Path)
    verify.add_argument("retained_response_root", type=Path)
    verify.set_defaults(handler=_verify)
    arguments = parser.parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
