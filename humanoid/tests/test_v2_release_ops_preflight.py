from __future__ import annotations

from pathlib import Path

from rigby_v2.release_ops import (
    CheckStatus,
    DiagnosticCheck,
    DiagnosticReport,
    PreflightProbes,
    StartupProbeSet,
    audit_dependency_locks,
    diagnose_startup_readiness,
    run_fresh_machine_preflight,
)
import pytest

pytestmark = pytest.mark.fast


ROOT = Path(__file__).resolve().parents[1]


def _successful_probes(*, postgres_reachable: bool = False) -> PreflightProbes:
    def which(name: str) -> str | None:
        return f"C:/tools/{name}.exe" if name in {"ffmpeg", "docker"} else None

    def run(command: tuple[str, ...]) -> tuple[int, str]:
        return 0, "26.1.0" if "docker" in command[0] else "ffmpeg 8.0"

    return PreflightProbes(
        which=which,
        run_command=run,
        tcp_probe=lambda _host, _port, _timeout: postgres_reachable,
        package_version=lambda name: "3.11.0" if name == "mujoco" else "unknown",
        python_version=(3, 12, 11),
    )


def test_current_dependency_lock_audit_is_fully_release_pinned() -> None:
    report = audit_dependency_locks(ROOT)
    by_id = {check.check_id: check for check in report.checks}
    assert by_id["python_lock"].status is CheckStatus.PASS
    assert by_id["runtime_dependencies_exact"].status is CheckStatus.PASS
    assert by_id["runtime_lock_matches"].status is CheckStatus.PASS
    assert by_id["build_backend_exact"].status is CheckStatus.PASS
    assert report.blockers == ()


def test_preflight_is_offline_read_only_and_accepts_docker_or_reachable_postgres() -> None:
    report = run_fresh_machine_preflight(
        project_root=ROOT,
        env={
            "RIGBY_V2_DATABASE_URL": "postgresql://rigby:redacted@127.0.0.1:54329/rigby",
            "RIGBY_V2_ARTIFACT_DIR": "artifacts-v2",
        },
        probes=_successful_probes(postgres_reachable=False),
    )
    by_id = {check.check_id: check for check in report.checks}
    assert report.offline and report.read_only
    assert by_id["python_3_12"].status is CheckStatus.PASS
    assert by_id["mujoco_3_11"].status is CheckStatus.PASS
    assert by_id["ffmpeg"].status is CheckStatus.PASS
    assert by_id["postgres_runtime"].status is CheckStatus.PASS
    assert by_id["required_assets"].status is CheckStatus.PASS
    assert report.blockers == ()
    assert report.to_dict()["blockers"] == []

    external = run_fresh_machine_preflight(
        project_root=ROOT,
        env={"RIGBY_V2_DATABASE_URL": "postgresql://user:redacted@db.internal:5432/rigby"},
        probes=PreflightProbes(
            which=lambda name: "C:/ffmpeg.exe" if name == "ffmpeg" else None,
            run_command=lambda _command: (0, "ffmpeg 8.0"),
            tcp_probe=lambda host, port, _timeout: (host, port) == ("db.internal", 5432),
            package_version=lambda _name: "3.11.0",
            python_version=(3, 12, 0),
        ),
    )
    assert {check.check_id: check for check in external.checks}[
        "postgres_runtime"
    ].status is CheckStatus.PASS


def test_preflight_reports_missing_tools_and_bad_runtime_without_mutating() -> None:
    report = run_fresh_machine_preflight(
        project_root=ROOT,
        env={"RIGBY_V2_DATABASE_URL": "not-a-database"},
        probes=PreflightProbes(
            which=lambda _name: None,
            run_command=lambda _command: (1, "unavailable"),
            tcp_probe=lambda _host, _port, _timeout: False,
            package_version=lambda _name: "3.10.0",
            python_version=(3, 11, 9),
        ),
    )
    assert {"python_3_12", "mujoco_3_11", "ffmpeg", "postgres_runtime"} <= set(
        report.blockers
    )
    assert not report.passed


def test_readiness_probe_failures_are_reported_without_raising() -> None:
    preflight = DiagnosticReport(
        kind="test",
        checks=(DiagnosticCheck("preflight", CheckStatus.PASS, "ok"),),
    )
    def ready() -> tuple[bool, str, dict[str, object]]:
        return True, "ready", {"latency_ms": 1}

    def crashes() -> tuple[bool, str, dict[str, object]]:
        raise RuntimeError("database unavailable")

    report = diagnose_startup_readiness(
        preflight,
        StartupProbeSet(
            database=ready,
            artifact_store=ready,
            api_health=ready,
            worker_job_store=crashes,
        ),
    )
    assert not report.passed
    assert report.blockers == ("worker_job_store",)
    failed = {check.check_id: check for check in report.checks}["worker_job_store"]
    assert failed.status is CheckStatus.FAIL
    assert "RuntimeError" in failed.summary
