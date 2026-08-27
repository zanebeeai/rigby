from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.benchmark import run_adversarial_benchmark_100
from rigby_v2.release_ops import seal_release_evidence

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and seal the deterministic 100-case adversarial firewall."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts-v2",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "artifacts-v2" / "release-evidence",
    )
    arguments = parser.parse_args()

    artifacts = ContentAddressedArtifactStore(arguments.artifact_root)
    attempt = run_adversarial_benchmark_100(artifacts=artifacts)
    if not attempt.sealed or attempt.seal is None or attempt.seal_artifact is None:
        print(
            json.dumps(
                {
                    "passed": False,
                    "critical_false_accepts": attempt.critical_false_accepts,
                    "typed_outcome_mismatches": attempt.typed_outcome_mismatches,
                    "model_calls": attempt.model_calls,
                },
                indent=2,
            )
        )
        return 1

    release_row = seal_release_evidence(
        arguments.evidence_root.resolve(),
        requirement_id="adversarial_benchmark_100",
        passed=True,
        payload={
            "case_count": len(attempt.results),
            "critical_deterministic_false_accepts": (
                attempt.critical_false_accepts
            ),
            "typed_outcome_mismatches": attempt.typed_outcome_mismatches,
            "model_calls": attempt.model_calls,
            "unique_evidence_artifacts": len(
                {reference.sha256 for reference in attempt.evidence}
            ),
            "benchmark_manifest_hash": attempt.seal.benchmark_manifest_hash,
            "firewall_version": attempt.seal.firewall_version,
            "evidence_set_hash": attempt.seal.evidence_set_hash,
            "seal_hash": attempt.seal.seal_hash,
            "seal_artifact_sha256": attempt.seal_artifact.sha256,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "release_evidence_sha256": release_row.artifact_sha256,
                "seal_hash": attempt.seal.seal_hash,
                "seal_artifact_sha256": attempt.seal_artifact.sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
