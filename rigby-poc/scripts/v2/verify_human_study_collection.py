from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_v2.calibration import import_verified_human_study_collection


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Noninteractively verify all retained response and rater-attestation "
            "files for a completed human calibration study."
        )
    )
    parser.add_argument("collection_manifest", type=Path)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("public_blueprint", type=Path)
    parser.add_argument("private_ground_truth", type=Path)
    arguments = parser.parse_args()

    evidence = import_verified_human_study_collection(
        arguments.collection_manifest,
        evidence_root=arguments.evidence_root,
        public_blueprint_path=arguments.public_blueprint,
        private_ground_truth_path=arguments.private_ground_truth,
    )
    print(
        json.dumps(
            {
                "study_id": evidence.study_id,
                "release_grade_files_verified": (
                    evidence.release_grade_files_verified
                ),
                "pair_count": len(evidence.study.pairs),
                "rating_slot_count": len(evidence.source_response_paths),
                "distinct_response_artifacts": len(
                    set(evidence.source_response_sha256s)
                ),
                "distinct_rater_attestations": len(
                    set(evidence.rater_attestation_sha256s)
                ),
                "collection_manifest_sha256": (
                    evidence.collection_manifest_sha256
                ),
                "retained_evidence_set_sha256": (
                    evidence.retained_evidence_set_sha256
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
