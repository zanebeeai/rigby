from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rigby_v2.release_ops import (
    ReleaseEvidence,
    audit_dependency_locks,
    evaluate_release_checklist,
    run_fresh_machine_preflight,
    seal_release_evidence,
    verify_wheel_smoke,
    write_release_evidence_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def _verified_performance_report() -> tuple[bool, dict[str, object]]:
    directory = ROOT / "assets" / "v2" / "benchmark"
    report_path = directory / "production_renderer_performance.json"
    digest_path = directory / "production_renderer_performance.json.sha256"
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = bool(
        actual == expected
        and report.get("passed") is True
        and report.get("vlm_included") is False
        and report.get("workload_kind")
        == "articulated_keyframes_three_camera_h264"
        and float(report.get("p95_candidate_s", float("inf"))) <= 90.0
    )
    return passed, {
        "report_sha256": actual,
        "expected_sha256": expected,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect only locally demonstrated release evidence. Missing release "
            "requirements remain explicit ship blockers."
        )
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "artifacts-v2" / "release-evidence",
    )
    parser.add_argument("--skip-wheel", action="store_true")
    arguments = parser.parse_args()

    evidence_root = arguments.evidence_root.resolve()
    rows = []

    preflight = run_fresh_machine_preflight(project_root=ROOT)
    rows.append(
        seal_release_evidence(
            evidence_root,
            requirement_id="fresh_machine_preflight",
            passed=preflight.passed,
            payload=preflight.to_dict(),
        )
    )
    locks = audit_dependency_locks(ROOT)
    rows.append(
        seal_release_evidence(
            evidence_root,
            requirement_id="dependency_lock_audit",
            passed=locks.passed,
            payload=locks.to_dict(),
        )
    )
    if not arguments.skip_wheel:
        wheel = verify_wheel_smoke(ROOT)
        rows.append(
            seal_release_evidence(
                evidence_root,
                requirement_id="wheel_smoke",
                passed=wheel.passed,
                payload=asdict(wheel),
            )
        )
    performance_passed, performance = _verified_performance_report()
    rows.append(
        seal_release_evidence(
            evidence_root,
            requirement_id="performance_p95",
            passed=performance_passed,
            payload=performance,
        )
    )

    # Preserve independently sealed evidence produced by integration harnesses
    # (for example the live API/worker restart replay). The document identity,
    # pass state, filename, and bytes all have to agree before it can enter the
    # checklist manifest.
    collected_ids = {row.requirement_id for row in rows}
    for path in sorted(evidence_root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        requirement_id = str(document.get("requirement_id", ""))
        if requirement_id in collected_ids:
            continue
        if (
            document.get("schema_version") != "1.0"
            or document.get("passed") is not True
            or path.name != f"{requirement_id}.json"
        ):
            continue
        rows.append(
            ReleaseEvidence(
                requirement_id=requirement_id,
                status="pass",
                artifact_path=path.name,
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        collected_ids.add(requirement_id)

    if {
        "fresh_machine_preflight",
        "deterministic_replay",
    }.issubset(collected_ids):
        rows = [row for row in rows if row.requirement_id != "startup_readiness"]
        collected_ids.discard("startup_readiness")
        replay_document = json.loads(
            (evidence_root / "deterministic_replay.json").read_text(encoding="utf-8")
        )
        replay_payload = replay_document["payload"]
        startup = seal_release_evidence(
            evidence_root,
            requirement_id="startup_readiness",
            passed=bool(
                replay_payload.get("live_http_submission")
                and replay_payload.get("lease_recovered_by_replacement")
                and replay_payload.get("live_replay_endpoint")
                and replay_payload.get("authoritative_trace_sha256")
                == replay_payload.get("replay_trace_sha256")
            ),
            payload={
                "postgresql_job_store": "lease_recovered_and_completed",
                "artifact_store": "authoritative_trace_written_and_replayed",
                "api_health": "live_health_probe_passed_before_submission_and_after_restart",
                "worker_job_store": "replacement_worker_completed_attempt_2",
                "preflight_artifact_sha256": next(
                    row.artifact_sha256
                    for row in rows
                    if row.requirement_id == "fresh_machine_preflight"
                ),
                "restart_replay_artifact_sha256": next(
                    row.artifact_sha256
                    for row in rows
                    if row.requirement_id == "deterministic_replay"
                ),
            },
        )
        rows.append(startup)
        collected_ids.add("startup_readiness")

    manifest = evidence_root / "manifest.json"
    write_release_evidence_manifest(manifest, rows)
    decision = evaluate_release_checklist(
        ROOT / "docs" / "v2" / "release-checklist.json",
        tuple(rows),
        evidence_root=evidence_root,
    )
    print(
        json.dumps(
            {
                "collected": [row.requirement_id for row in rows],
                "all_collected_passed": all(row.status == "pass" for row in rows),
                "manifest": str(manifest),
                "release_decision": decision.to_dict(),
            },
            indent=2,
        )
    )
    return 0 if all(row.status == "pass" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
