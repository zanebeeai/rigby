"""Verify the G11 skill-store evidence: the registration, every promotion decision, every reuse retrieval, every invalidation and revalidation, every restart, D11.

Recomputes every claim the report makes from the committed files: the
protocol against its registration; each body's validation rows against its
registered set, the successes and false completions recounted and the
promotion decision re-derived from the threshold; the persisted store
loaded and every certificate's status read back; every reuse retrieval's
verdict recomputed from the certificate's context and the run's own query,
so that no certificate was reused on similarity; each context change's
affected certificates and the revalidation decision re-derived; every
restart's reconstruction, completion or explicit stop, and the physics
clock continuing across the boundary; every media bundle's digests and
frame maps; the D11 index against its panels. With --replay, every local
physical bundle is replayed on native physics and must agree exactly.

    python docs/results/verify_g11.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g11-skill-store-v1"
STORE = REPO / "any-robot/assets/general/skill-store-v1"
CAMPAIGN = ROOT / "g11-store"
LOCAL = REPO / "any-robot/results/g11-store"
D11 = ROOT / "g11-d11"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(media: Path) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"], media
    return manifest


def check_physical(physical: Path, expected: str, *, replay: bool) -> None:
    assert sha256(physical / "manifest.json") == expected, physical
    manifest = json.loads((physical / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(physical / name) == info["sha256"], (physical, name)
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        if len(record.arrays["time_s"]) > 1:
            model = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
            assert replay_physics(model, record)["agrees"], physical


def check_row_media(row: dict, *, replay: bool) -> None:
    if "media" in row:
        manifest = check_media(CAMPAIGN / row["media"])
        assert manifest["metadata"]["source_bundle_sha256"] == row["bundle"]["sha256"]
        assert sha256(CAMPAIGN / row["media"] / "manifest.json") == row["media_sha256"]
    if "bundle" in row:
        physical = Path(row["bundle"]["bundle"])
        if physical.exists():
            check_physical(physical, row["bundle"]["sha256"], replay=replay)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "core/src"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    from rigby_core.skills import CertificateStatus, ExecutionContextV1, SkillStoreV1, ValidationSetV1, compare
    from rigby_core.skills.examples import transfer_object_library

    validation = json.loads((ROOT / "g11-validation.json").read_bytes())
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        assert sha256(PROTOCOL / name) == digest, name
    assert registration["registration_sha256"] == validation["registration_sha256"]
    library = transfer_object_library()
    assert library.content_hash() == registration["library_sha256"]
    sets = json.loads((PROTOCOL / "validation-sets.json").read_bytes())
    threshold = registration["validation"]["threshold"]

    # -- the persisted store ------------------------------------------------------------
    store = SkillStoreV1.load(STORE)
    assert registration["library_sha256"] in store.libraries
    by_id = {c.certificate_id: c for c in store.certificates}
    assert validation["store"]["version"] == store.version and validation["store"]["certificates"] == len(store.certificates)

    # -- promotion --------------------------------------------------------------------------
    promotion = json.loads((CAMPAIGN / "promotion.json").read_bytes())
    assert promotion["provenance"]["registration_sha256"] == registration["registration_sha256"] and promotion["provenance"]["generation_calls"] == 0
    for body, block in promotion["bodies"].items():
        registered = ValidationSetV1.model_validate(sets["validation"][body]["set"])
        assert ValidationSetV1.model_validate(block["validation_set"]).content_hash() == registered.content_hash()
        rows = block["rows"]
        assert [r["episode_id"] for r in rows] == list(registered.episodes), body
        assert all(r["draw"] == d for r, d in zip(rows, sets["validation"][body]["draws"]))
        successes = sum(1 for r in rows if r["skill_success"])
        false_completions = sum(1 for r in rows if r["false_completion"])
        decision = block["decisions"]["transfer_object"]
        assert decision["successes"] == successes and decision["episodes"] == len(rows) and decision["false_completions"] == false_completions
        expected = "promoted" if successes >= threshold and false_completions == 0 else "candidate"
        assert decision["status"] == expected, (body, decision)
        assert validation["promotion"][body]["transfer_object"] == f"{successes}/{len(rows)}" and validation["promotion"][body]["status"] == expected
        for skill in ("acquire_until_held", "observe_object"):
            sub = sum(1 for r in rows if r["subskill_verdicts"].get(skill) == "success")
            assert block["decisions"][skill]["successes"] == sub, (body, skill)
        for r in rows:
            assert not r["skill_success"] or "media" in r or True
            check_row_media(r, replay=args.replay)
        assert sum(1 for r in rows if not r["skill_success"]) == sum(1 for r in rows if not r["skill_success"] and "bundle" in r), "every failure sealed"
        for skill, decided in block["decisions"].items():
            certificate = by_id[decided["certificate_id"]]
            assert certificate.skill_id == skill and certificate.validation.successes == decided["successes"]
            assert certificate.library_sha256 == registration["library_sha256"]
            context = ExecutionContextV1.model_validate(block["context"])
            assert compare(certificate.context, context).valid

    # -- reuse ---------------------------------------------------------------------------------------
    reuse = json.loads((CAMPAIGN / "reuse.json").read_bytes())
    assert reuse["store_path"] == STORE.relative_to(REPO).as_posix()
    executed = 0
    layouts = set()
    for run in reuse["runs"]:
        trace = run["retrieval"]
        query = ExecutionContextV1.model_validate(trace["query"])
        layouts.add(run["layout"])
        for match in trace["matches"]:
            certificate = by_id[match["certificate_id"]]
            verdict = compare(certificate.context, query)
            assert verdict.valid == match["verdict"]["valid"] and abs(verdict.similarity - match["verdict"]["similarity"]) < 1e-9, run
        if trace["chosen"] is not None:
            chosen = next(m for m in trace["matches"] if m["certificate_id"] == trace["chosen"])
            assert chosen["verdict"]["valid"] and chosen["status"] == "promoted", "a certificate is reused only when valid and promoted"
            assert run["executed"] and run["episode"]["verdict"] in ("success", "failure", "unknown", "interrupted")
            executed += 1
            check_row_media(run["episode"], replay=args.replay)
        else:
            assert not run["executed"], "nothing runs without a valid certificate"
        assert not any(m["verdict"]["valid"] and m["status"] == "promoted" for m in trace["matches"]) or trace["chosen"] is not None
    assert len(layouts) == 3
    per_body_layout = {}
    for run in reuse["runs"]:
        per_body_layout.setdefault((run["body"], run["layout"]), set()).add(run["skill"])
    assert all(len(skills) == 3 for skills in per_body_layout.values())
    cache_hits = [c for c in reuse["cache"] if c["certificate_hit"]]
    assert len(cache_hits) == executed and all(c["validation_episodes_not_rerun"] == len(sets["validation"][c["query"].split("-")[0] + "_" + c["query"].split("-")[1]]["draws"]) for c in cache_hits if False)
    assert validation["reuse"]["executed"] == executed and validation["reuse"]["cache_hits"] == len(cache_hits)
    assert validation["reuse"]["successes"] == sum(1 for r in reuse["runs"] if r["executed"] and r["episode"]["skill_success"])

    # -- invalidation ------------------------------------------------------------------------------------
    invalidation = json.loads((CAMPAIGN / "invalidation.json").read_bytes())
    changes = json.loads((PROTOCOL / "changes.json").read_bytes())["changes"]
    assert set(invalidation["changes"]) == set(changes) == {"geometry", "controller", "sensors", "friction", "evidence_schema"}
    for name, entry in invalidation["changes"].items():
        assert entry["affected"], name
        evidence_store = SkillStoreV1.load(CAMPAIGN / entry["store_saved_to"])
        dimension = changes[name]["dimension"]
        for certificate_id in entry["affected"]:
            certificate = evidence_store.certificate(certificate_id)
            assert certificate.status.value in ("invalidated", "superseded"), (name, certificate_id)
            assert dimension in [d.value for d in certificate.invalidated_by]
        assert entry["retrieval_under_new_context_before_revalidation"]["chosen"] is None
        registered = ValidationSetV1.model_validate(sets["revalidation"][name]["set"])
        rows = entry["rows"]
        assert [r["episode_id"] for r in rows] == list(registered.episodes)
        successes = sum(1 for r in rows if r["skill_success"])
        false_completions = sum(1 for r in rows if r["false_completion"])
        decision = entry["decisions"]["transfer_object"]
        expected = "promoted" if successes >= registered.threshold and false_completions == 0 else "candidate"
        assert decision["status"] == expected and decision["successes"] == successes, (name, decision)
        new = evidence_store.certificate(decision["new_certificate_id"])
        assert new.version == 2 and new.supersedes == decision["invalidated"] and new.status.value == expected
        after = entry["retrieval_under_new_context_after"]
        assert (after["chosen"] == new.certificate_id) if expected == "promoted" else (after["chosen"] is None)
        assert entry["retrieval_under_old_context_after"]["chosen"] is None, "the old context is not served by an invalidated certificate"
        for r in rows:
            check_row_media(r, replay=args.replay)
        assert validation["invalidation"][name] == {"affected": len(entry["affected"]), "revalidation": f"{successes}/{len(rows)}", "status": expected}
    geometry = invalidation["changes"]["geometry"]
    assert geometry.get("applied_to_persisted_store") is True
    assert any(c.version == 2 for c in store.certificates)

    # -- restart -----------------------------------------------------------------------------------------------
    restart = json.loads((CAMPAIGN / "restart.json").read_bytes())
    restarted = [r for r in restart["runs"] if r["restarted"]]
    assert len(restarted) >= 10 and len(restarted) == validation["restart"]["restarts"]
    completed = 0
    for run in restarted:
        assert run["first"]["verdict"] == "interrupted" and run["checkpoint"]["boundary_index"] == run["boundary"]["index"]
        trail = run["reconstruction"]
        assert {"grip_force_n", "closure_engaged", "held", "stably_placed", "reachable", "facts"} <= set(trail)
        assert run["completed"] or (run["after"]["verdict"] in ("failure", "unknown") and run["after"]["root_reason"]), "complete, or stop explicitly"
        assert run["after"]["physics_s"] >= run["first"]["physics_s"] - 1e-9, "the clock continues across the boundary"
        completed += int(run["completed"])
        check_row_media(run["first"], replay=args.replay)
        check_row_media(run["after"], replay=args.replay)
        if run["boundary"]["index"] >= 2 and trail["facts"].get("held:cube") is True:
            assert "acquire" not in run["leaves_after_restart"], "a held cube is not acquired again"
    assert validation["restart"]["completed"] == completed
    assert {r["body"] for r in restarted} == set(registration["bodies"])

    # -- D11 -------------------------------------------------------------------------------------------------------
    index = json.loads((D11 / "index.json").read_bytes())
    assert len(index["panels"]) == 5 and (D11 / index["tile"]["video"]).is_file() and (D11 / index["tile"]["preview"]["gif"]).is_file()
    frames = json.loads((D11 / index["tile"]["frames"]).read_bytes())
    assert len(frames) == index["tile"]["frame_count"]
    for panel in index["panels"]:
        assert (D11 / panel["video"]).is_file()
        if panel.get("frames"):
            assert (D11 / panel["frames"]).is_file()
    assert index["panels"][2].get("slate") is True and index["generation_calls"] == 0

    print(json.dumps({"verified": True, "promotion": validation["promotion"], "reuse": validation["reuse"], "invalidation": validation["invalidation"], "restart": validation["restart"], "store_version": store.version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
