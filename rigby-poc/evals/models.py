from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"
    ERROR = "error"


@dataclass
class GateResult:
    gate: str
    status: Status
    summary: str
    measured: dict[str, Any] = field(default_factory=dict)
    required: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class AcceptanceReport:
    run_id: str
    started_at: str
    completed_at: str
    source: str
    synthetic: bool
    gates: list[GateResult]
    artifacts: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            not self.synthetic
            and bool(self.gates)
            and all(gate.status is Status.PASS for gate in self.gates)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source": self.source,
            "synthetic": self.synthetic,
            "passed": self.passed,
            "gates": [gate.to_dict() for gate in self.gates],
            "artifacts": self.artifacts,
        }
