from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_v2.calibration import (
    SourceComparison,
    SourceMediaRecord,
    build_human_review_package,
)
from rigby_v2.flywheel.schemas import DefectKind


def _relative_source(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("media source paths must be confined relative paths")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("media source path escaped its input directory")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-backed blinded package for genuine human review."
    )
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    specification_path = arguments.specification.resolve()
    document = json.loads(specification_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "study_id",
        "seed",
        "rater_id_hashes",
        "records",
        "comparisons",
    }
    if set(document) != required or document["schema_version"] != "1.0":
        raise ValueError("human review specification has unknown or missing fields")
    source_root = specification_path.parent
    records = tuple(
        SourceMediaRecord(
            record_id=value["record_id"],
            source_path=_relative_source(source_root, value["source_path"]),
            sha256=value["sha256"],
            media_type=value["media_type"],
        )
        for value in document["records"]
    )
    comparisons = tuple(
        SourceComparison(
            pair_id=value["pair_id"],
            action_family=value["action_family"],
            candidate_a_id=value["candidate_a_id"],
            candidate_b_id=value["candidate_b_id"],
            preferred_record_id=value["preferred_record_id"],
            defects=tuple(DefectKind(item) for item in value["defects"]),
            critical_reject_record_ids=tuple(
                value.get("critical_reject_record_ids", ())
            ),
        )
        for value in document["comparisons"]
    )
    rater_hashes = tuple(document["rater_id_hashes"])
    if len(rater_hashes) != 3:
        raise ValueError("human review specification requires exactly three raters")
    evidence = build_human_review_package(
        study_id=document["study_id"],
        seed=int(document["seed"]),
        records=records,
        comparisons=comparisons,
        rater_id_hashes=rater_hashes,  # type: ignore[arg-type]
        output_root=arguments.output_root,
    )
    print(
        json.dumps(
            {
                "study_id": evidence.study_id,
                "root": str(evidence.root),
                "pair_count": evidence.pair_count,
                "expected_rating_count": evidence.expected_rating_count,
                "public_blueprint_sha256": (
                    evidence.blueprint.public_blueprint_sha256
                ),
                "media_manifest_sha256": evidence.media_manifest_sha256,
                "private_ground_truth_sha256": (
                    evidence.blueprint.private_ground_truth_sha256
                ),
                "private_source_map_sha256": evidence.private_source_map_sha256,
                "rater_packet_sha256s": evidence.rater_packet_sha256s,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
