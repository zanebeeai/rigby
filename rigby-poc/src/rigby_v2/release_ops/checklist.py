"""Machine-readable release evidence gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ReleaseRequirement:
    requirement_id: str
    description: str
    required: bool


@dataclass(frozen=True, slots=True)
class ReleaseEvidence:
    requirement_id: str
    status: str
    artifact_path: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    can_ship: bool
    checklist_version: str
    verified_evidence: Mapping[str, str]
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verified_evidence", MappingProxyType(dict(self.verified_evidence))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "can_ship": self.can_ship,
            "checklist_version": self.checklist_version,
            "verified_evidence": dict(self.verified_evidence),
            "blockers": list(self.blockers),
        }


class ReleaseChecklistError(ValueError):
    pass


def load_release_checklist(path: Path) -> tuple[str, tuple[ReleaseRequirement, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"schema_version", "checklist_version", "requirements"}:
        raise ReleaseChecklistError("release checklist has unknown or missing fields")
    if data["schema_version"] != "1.0" or not isinstance(data["checklist_version"], str):
        raise ReleaseChecklistError("release checklist version is invalid")
    requirements = tuple(
        ReleaseRequirement(
            requirement_id=str(value["id"]),
            description=str(value["description"]),
            required=bool(value["required"]),
        )
        for value in data["requirements"]
    )
    identifiers = [value.requirement_id for value in requirements]
    if not requirements or len(identifiers) != len(set(identifiers)):
        raise ReleaseChecklistError("release requirement IDs must be nonempty and unique")
    return data["checklist_version"], requirements


def _safe_evidence_path(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ReleaseChecklistError("evidence paths must use relative POSIX syntax")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.name:
        raise ReleaseChecklistError("evidence path is unsafe")
    resolved_root = root.resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ReleaseChecklistError("evidence path escapes the evidence root") from error
    return path


def evaluate_release_checklist(
    checklist_path: Path,
    evidence: tuple[ReleaseEvidence, ...],
    *,
    evidence_root: Path,
) -> ReleaseDecision:
    version, requirements = load_release_checklist(checklist_path)
    by_id: dict[str, ReleaseEvidence] = {}
    duplicate_ids: set[str] = set()
    for value in evidence:
        if value.requirement_id in by_id:
            duplicate_ids.add(value.requirement_id)
        by_id[value.requirement_id] = value
    blockers = [f"duplicate_evidence:{value}" for value in sorted(duplicate_ids)]
    known = {value.requirement_id for value in requirements}
    blockers.extend(f"unknown_evidence:{value}" for value in sorted(set(by_id) - known))
    verified: dict[str, str] = {}
    for requirement in requirements:
        item = by_id.get(requirement.requirement_id)
        if item is None:
            if requirement.required:
                blockers.append(f"missing:{requirement.requirement_id}")
            continue
        if item.status != "pass":
            if requirement.required:
                blockers.append(f"failed:{requirement.requirement_id}")
            continue
        try:
            artifact = _safe_evidence_path(evidence_root, item.artifact_path)
        except ReleaseChecklistError:
            blockers.append(f"unsafe_evidence:{requirement.requirement_id}")
            continue
        if not artifact.is_file():
            blockers.append(f"missing_artifact:{requirement.requirement_id}")
            continue
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != item.artifact_sha256:
            blockers.append(f"hash_mismatch:{requirement.requirement_id}")
            continue
        verified[requirement.requirement_id] = actual
    return ReleaseDecision(
        can_ship=not blockers,
        checklist_version=version,
        verified_evidence=verified,
        blockers=tuple(blockers),
    )
