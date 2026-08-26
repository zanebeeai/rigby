"""Exception-safe startup readiness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .models import CheckStatus, DiagnosticCheck, DiagnosticReport


ReadinessProbe = Callable[[], tuple[bool, str, Mapping[str, object]]]


@dataclass(frozen=True, slots=True)
class StartupProbeSet:
    database: ReadinessProbe
    artifact_store: ReadinessProbe
    api_health: ReadinessProbe
    worker_job_store: ReadinessProbe


def _safe_probe(check_id: str, probe: ReadinessProbe) -> DiagnosticCheck:
    try:
        ready, summary, details = probe()
        return DiagnosticCheck(
            check_id,
            CheckStatus.PASS if ready else CheckStatus.FAIL,
            summary,
            details=details,
        )
    except Exception as error:  # A diagnostic must report failures, not crash startup.
        return DiagnosticCheck(
            check_id,
            CheckStatus.FAIL,
            f"readiness probe raised {type(error).__name__}",
            details={"error": str(error)},
        )


def diagnose_startup_readiness(
    preflight: DiagnosticReport,
    probes: StartupProbeSet,
) -> DiagnosticReport:
    checks = [
        DiagnosticCheck(
            "preflight",
            CheckStatus.PASS if preflight.passed else CheckStatus.FAIL,
            "fresh-machine preflight passed" if preflight.passed else "fresh-machine preflight has blockers",
            details={"blockers": list(preflight.blockers)},
        )
    ]
    checks.extend(
        (
            _safe_probe("database", probes.database),
            _safe_probe("artifact_store", probes.artifact_store),
            _safe_probe("api_health", probes.api_health),
            _safe_probe("worker_job_store", probes.worker_job_store),
        )
    )
    return DiagnosticReport(
        kind="startup_readiness",
        checks=tuple(checks),
        offline=False,
        read_only=True,
    )

