from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rigby_v2.hashing import hash_file
from rigby_v2.release_ops.evidence import (
    seal_release_evidence,
    write_release_evidence_manifest,
)
from rigby_v2.release_ops.operator_dry_run import (
    OperatorDrillError,
    OperatorDryRunConfig,
    run_operator_dry_run,
)

pytestmark = pytest.mark.fast


class _Handle:
    def __init__(self, label: str) -> None:
        self.label = label
        self.stopped = False

    def poll(self) -> int | None:
        return 0 if self.stopped else None


class _Processes:
    def __init__(self) -> None:
        self.handles: list[_Handle] = []

    def start_api(self, specification):  # type: ignore[no-untyped-def]
        handle = _Handle(specification.label)
        self.handles.append(handle)
        specification.log_path.parent.mkdir(parents=True, exist_ok=True)
        specification.log_path.write_text(
            "database=postgresql://operator_canary:operator-canary-secret@127.0.0.1/x",
            encoding="utf-8",
        )
        return handle

    def stop(self, process):  # type: ignore[no-untyped-def]
        process.stopped = True


class _Probes:
    def __init__(self, snapshots: list[dict[str, int]] | None = None) -> None:
        self.snapshots = snapshots or [
            {"jobs": 7, "releases": 2, "legacy": 6354},
            {"jobs": 7, "releases": 2, "legacy": 6354},
        ]

    def wait_health(self, process, port, *, expected_status, timeout_s):  # type: ignore[no-untyped-def]
        del port, timeout_s
        if process.label == "bad-canary":
            assert expected_status == "degraded"
            return {
                "status": "degraded",
                "database_error": (
                    "could not connect postgresql://operator_canary:"
                    "operator-canary-secret@127.0.0.1/unreachable"
                ),
            }
        assert expected_status == "ok"
        return {"status": "ok", "database": {"postgres_version": "16"}}

    def database_counts(self, database_url):  # type: ignore[no-untyped-def]
        del database_url
        return self.snapshots.pop(0)

    def artifact_readiness(self, artifact_root):  # type: ignore[no-untyped-def]
        del artifact_root
        return {
            "ready": True,
            "content_addressed_objects": 11,
            "inventory_sha256": "a" * 64,
            "read_only_probe": True,
        }


def _fixture(tmp_path: Path) -> OperatorDryRunConfig:
    project = tmp_path / "project"
    evidence_root = project / "evidence"
    artifact_root = project / "artifacts"
    (project / "docs" / "v2").mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    (project / "uv.lock").write_text("exact-lock", encoding="utf-8")
    (project / "docs" / "v2" / "operator-runbook.md").write_text(
        "# Backup and restore\n# Upgrade and rollback\n"
        "Rollback is forward-restorative.\n# Incident response\nDatabase unavailable.\n",
        encoding="utf-8",
    )
    lock_hash = hash_file(project / "uv.lock")
    backup = seal_release_evidence(
        evidence_root,
        requirement_id="database_backup_restore",
        passed=True,
        payload={
            "restored_table_counts_identical": True,
            "content_addressed_canary_restored": True,
            "disposable_database_cleaned": True,
            "dump_size_bytes": 100,
            "dump_sha256": "b" * 64,
            "table_counts": {"animation_records": 6354},
        },
    )
    replay = seal_release_evidence(
        evidence_root,
        requirement_id="deterministic_replay",
        passed=True,
        payload={
            "api_restarted_in_new_process": True,
            "live_replay_endpoint": True,
            "authoritative_trace_sha256": "c" * 64,
            "replay_trace_sha256": "c" * 64,
            "attempts": 2,
            "dependency_lock_sha256": lock_hash,
        },
    )
    write_release_evidence_manifest(evidence_root / "manifest.json", (backup, replay))
    return OperatorDryRunConfig(
        project_root=project,
        evidence_root=evidence_root,
        artifact_root=artifact_root,
        database_url="postgresql://real_user:supersecret@127.0.0.1:5432/rigby",
    )


def test_dry_run_degrades_redacts_rolls_back_and_seals_only_after_unchanged_state(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    processes = _Processes()
    evidence = run_operator_dry_run(
        config, process_runner=processes, probe_runner=_Probes()
    )
    assert evidence.requirement_id == "operator_dry_run"
    assert all(handle.stopped for handle in processes.handles)
    document = json.loads((config.evidence_root / evidence.artifact_path).read_text())
    payload = document["payload"]
    assert payload["bad_canary_degraded"] is True
    assert payload["protected_counts_before"] == payload["protected_counts_after"]
    assert payload["protected_counts_unchanged"] is True
    incident = config.evidence_root / payload["incident_bundle"]["path"]
    encoded = incident.read_text(encoding="utf-8")
    assert "supersecret" not in encoded
    assert "operator-canary-secret" not in encoded
    assert "operator_canary" not in encoded
    assert "[REDACTED]" in encoded


def test_dry_run_refuses_to_seal_if_rollback_mutates_protected_counts(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    processes = _Processes()
    probes = _Probes(
        [
            {"jobs": 7, "releases": 2, "legacy": 6354},
            {"jobs": 8, "releases": 2, "legacy": 6354},
        ]
    )
    with pytest.raises(OperatorDrillError, match="mutated"):
        run_operator_dry_run(config, process_runner=processes, probe_runner=probes)
    assert all(handle.stopped for handle in processes.handles)
    assert not (config.evidence_root / "operator_dry_run.json").exists()


def test_dry_run_refuses_tampered_prior_evidence(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = config.evidence_root / "deterministic_replay.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(OperatorDrillError, match="hash"):
        run_operator_dry_run(
            config, process_runner=_Processes(), probe_runner=_Probes()
        )


@pytest.mark.skipif(
    os.getenv("RIGBY_OPERATOR_DRY_RUN_LIVE") != "1",
    reason="set RIGBY_OPERATOR_DRY_RUN_LIVE=1 with the sealed local DB/artifact checkpoint",
)
def test_guarded_live_operator_dry_run() -> None:
    from rigby_v2.config import RuntimeSettings

    settings = RuntimeSettings.from_env()
    evidence = run_operator_dry_run(
        OperatorDryRunConfig(
            project_root=settings.project_root,
            evidence_root=settings.project_root / "artifacts-v2" / "release-evidence",
            artifact_root=settings.artifact_root,
            database_url=settings.database_url,
        )
    )
    assert evidence.status == "pass"
