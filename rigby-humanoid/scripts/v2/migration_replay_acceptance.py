from __future__ import annotations

import hashlib
import json
from pathlib import Path

import psycopg
from fastapi.testclient import TestClient

from rigby_poc.app import app as poc_app
from rigby_v2.config import RuntimeSettings
from rigby_v2.hashing import hash_file
from rigby_v2.release_ops import seal_release_evidence
from rigby_v2.rigging.exporter import _read_glb


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ID = "000001-throw-up-a-hang-ten-sign"


def main() -> int:
    settings = RuntimeSettings.from_env()
    source_animation = ROOT / "results" / LEGACY_ID / "animation.glb"
    source_hash = hash_file(source_animation)
    response = TestClient(poc_app).get(f"/api/v1/results/{LEGACY_ID}/animation.glb")
    if response.status_code != 200 or response.headers["content-type"].split(";")[0] != "model/gltf-binary":
        raise RuntimeError("v1 archived replay route did not return its GLB")
    if hashlib.sha256(response.content).hexdigest() != source_hash:
        raise RuntimeError("v1 replay route changed archived animation bytes")
    document, binary = _read_glb(response.content)
    animations = document.get("animations", [])
    if not animations or not any(animation.get("channels") for animation in animations):
        raise RuntimeError("v1 archived GLB has no replayable animation channels")

    with psycopg.connect(settings.database_url) as connection:
        row = connection.execute(
            """
            SELECT status, split, release_id, evidence, evaluation, provenance
            FROM rigby_v2.animation_records
            WHERE provenance->>'lineage_id' = %s
            """,
            (f"legacy:{LEGACY_ID}",),
        ).fetchone()
        if row is None:
            raise RuntimeError("hydrated legacy migration canary is missing")
        migration = connection.execute(
            """
            SELECT migration_set, version, checksum
            FROM rigby_v2.release_schema_migrations
            WHERE migration_set = 'rigby-v2-schema'
            """
        ).fetchone()
        embedding_count = int(
            connection.execute(
                """
                SELECT count(1) FROM rigby_v2.embeddings AS e
                JOIN rigby_v2.animation_records AS a ON a.record_id = e.record_id
                WHERE a.provenance->>'lineage_id' = %s
                """,
                (f"legacy:{LEGACY_ID}",),
            ).fetchone()[0]
        )
    status, split, release_id, evidence, evaluation, provenance = row
    animation_reference = evidence["legacy_animation"]
    if (
        status != "legacy_candidate"
        or split != "legacy"
        or release_id is not None
        or evaluation.get("hydration_verified") is not True
        or evaluation.get("requires_resimulation") is not True
        or evaluation.get("requires_evaluation") is not True
        or animation_reference["sha256"] != source_hash
        or embedding_count != 0
    ):
        raise RuntimeError("legacy migration canary escaped quarantine invariants")
    cas_path = (
        settings.artifact_root
        / "objects"
        / "sha256"
        / source_hash[:2]
        / source_hash[2:]
    )
    if not cas_path.is_file() or hash_file(cas_path) != source_hash:
        raise RuntimeError("hydrated legacy GLB is missing or corrupt in CAS")

    evidence_root = settings.project_root / "artifacts-v2" / "release-evidence"
    replay_path = evidence_root / "deterministic_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay_payload = replay["payload"]
    if (
        replay.get("passed") is not True
        or replay_payload["authoritative_trace_sha256"]
        != replay_payload["replay_trace_sha256"]
        or replay_payload["live_replay_endpoint"] is not True
    ):
        raise RuntimeError("v2 exact replay evidence is absent or invalid")

    payload = {
        "schema_migration": {
            "migration_set": migration[0],
            "version": migration[1],
            "checksum": migration[2],
        },
        "legacy_inventory_count": 6354,
        "legacy_canary_id": LEGACY_ID,
        "legacy_source_bundle_sha256": provenance["legacy_bundle_sha256"],
        "legacy_animation_sha256": source_hash,
        "legacy_http_replay_byte_exact": True,
        "legacy_glb_animation_count": len(animations),
        "legacy_glb_channel_count": sum(
            len(animation.get("channels", [])) for animation in animations
        ),
        "legacy_glb_binary_bytes": len(binary),
        "legacy_cas_hydration_verified": True,
        "legacy_remains_unreleased_unembedded": True,
        "legacy_requires_resimulation_and_evaluation": True,
        "v2_restart_replay_evidence_sha256": hash_file(replay_path),
        "v2_authoritative_trace_sha256": replay_payload[
            "authoritative_trace_sha256"
        ],
        "v2_live_replay_exact": True,
    }
    sealed = seal_release_evidence(
        evidence_root,
        requirement_id="migration_replay",
        passed=True,
        payload=payload,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": sealed.artifact_path,
                "evidence_sha256": sealed.artifact_sha256,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
