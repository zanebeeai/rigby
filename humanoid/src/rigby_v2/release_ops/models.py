"""Typed, JSON-safe release operation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    check_id: str
    status: CheckStatus
    summary: str
    required: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    kind: str
    checks: tuple[DiagnosticCheck, ...]
    offline: bool = True
    read_only: bool = True

    @property
    def passed(self) -> bool:
        return not any(
            check.required and check.status is CheckStatus.FAIL for check in self.checks
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            check.check_id
            for check in self.checks
            if check.required and check.status is CheckStatus.FAIL
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "offline": self.offline,
            "read_only": self.read_only,
            "blockers": list(self.blockers),
            "checks": [
                {
                    "check_id": check.check_id,
                    "status": check.status.value,
                    "summary": check.summary,
                    "required": check.required,
                    "details": dict(check.details),
                }
                for check in self.checks
            ],
        }
