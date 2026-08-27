from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rigby_core.errors import ArtifactIntegrityError
from rigby_core.hashing import content_hash, validate_sha256

from .models import (
    BenchmarkCaseKind,
    BenchmarkCaseV1,
    BenchmarkManifestV1,
    BenchmarkSpecV1,
    ExpectedOutcome,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "assets" / "v2" / "benchmark" / "benchmark_spec.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_spec_artifact(path: Path, checksum_path: Path) -> str:
    try:
        declared = validate_sha256(checksum_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(
            "Benchmark checksum is missing or invalid",
            details={"checksum_path": str(checksum_path)},
        ) from error
    actual = _file_hash(path)
    if actual != declared:
        raise ArtifactIntegrityError(
            "Benchmark specification failed immutable checksum verification",
            details={"expected_sha256": declared, "actual_sha256": actual},
        )
    return actual


def expand_benchmark_spec(
    spec: BenchmarkSpecV1, *, spec_artifact_hash: str
) -> BenchmarkManifestV1:
    cases: list[BenchmarkCaseV1] = []
    for grid in sorted(spec.supported_grids, key=lambda item: item.family.value):
        sequence = 0
        for action in grid.actions:
            for entity in grid.entities:
                for modifier in grid.modifiers:
                    sequence += 1
                    cases.append(
                        BenchmarkCaseV1(
                            case_id=f"supported-{grid.family.value.replace('_', '-')}-{sequence:03d}",
                            kind=BenchmarkCaseKind.SUPPORTED,
                            family=grid.family,
                            prompt=f"{action.format(entity=entity)} {modifier}",
                            expected_outcome=ExpectedOutcome.CERTIFIED,
                            seed=spec.seed + len(cases) * 7919,
                            tags=("supported", grid.family.value, f"grid-{sequence:03d}"),
                        )
                    )
        if sequence != spec.supported_per_family:
            raise ArtifactIntegrityError(
                "Supported benchmark grid does not expand to 75 cases",
                details={"family": grid.family.value, "count": sequence},
            )
    for template in sorted(spec.adversarial_templates, key=lambda item: item.category):
        for index in range(1, 11):
            cases.append(
                BenchmarkCaseV1(
                    case_id=f"adversarial-{template.category}-{index:03d}",
                    kind=BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL,
                    prompt=template.prompt_template.format(index=index),
                    expected_outcome=template.expected_outcome,
                    seed=spec.seed + len(cases) * 7919,
                    tags=("adversarial", template.category, *template.tags),
                )
            )
    expanded_hash = content_hash([case.model_dump(mode="json") for case in cases])
    if expanded_hash != spec.expanded_cases_hash:
        raise ArtifactIntegrityError(
            "Expanded benchmark cases do not match the sealed case hash",
            details={
                "expected_sha256": spec.expanded_cases_hash,
                "actual_sha256": expanded_hash,
            },
        )
    return BenchmarkManifestV1(
        benchmark_id=spec.benchmark_id,
        generator_version=spec.generator_version,
        seed=spec.seed,
        spec_artifact_hash=spec_artifact_hash,
        expanded_cases_hash=expanded_hash,
        cases=tuple(cases),
    )


def load_benchmark_manifest(path: Path = DEFAULT_SPEC_PATH) -> BenchmarkManifestV1:
    path = path.resolve()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    artifact_hash = _verify_spec_artifact(path, checksum_path)
    try:
        spec = BenchmarkSpecV1.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as error:
        raise ArtifactIntegrityError(
            "Benchmark specification is not a valid frozen v1 contract",
            details={"spec_path": str(path), "error": str(error)},
        ) from error
    return expand_benchmark_spec(spec, spec_artifact_hash=artifact_hash)
