"""Verify the G17 evidence: the registered locomotion protocol, the three bodies' trial records, the summaries, the validation, D17.

Recomputes every claim from the committed files: the protocol hashes
to its registration and the course and bodies hash to what it names;
each body's trial record holds exactly the registered trials, run under
that registration, with the outcome rule applied as written (success
means every waypoint reached, stable at the end, no fall, within the
release radius); every summary count is recomputed from the rows and
the validation's numbers agree; the controller provenance in every row
is analytic and hand-authored, with no root write and no artificial
support; the D17 index's media bundles hash whole and each clip's
source is a sealed row of the record. With --replay, every sealed run
behind a D17 clip is replayed on native physics and must agree exactly;
its recorded user input (a push) is the trial's declared one.

    python docs/results/verify_g17.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g17-locomotion-v1"
RESULTS = ROOT / "g17-locomotion"
D17 = ROOT / "g17-d17"
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(media: Path) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"], media
    return manifest


def check_run(bundle: Path, expected: str, row: dict, *, replay: bool) -> None:
    assert sha256(bundle / "manifest.json") == expected, bundle
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(bundle / name) == info["sha256"], (bundle, name)
    task = json.loads((bundle / "task.json").read_bytes())
    assert task["trial_id"] == row["trial_id"] and task["perturbation"] == row["perturbation"], bundle
    if replay:
        import mujoco
        import numpy as np
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        record = PhysicsRecord.from_bytes((bundle / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
        assert replay_physics(model, record)["agrees"], bundle
        user = record.arrays["user_input"]
        pushed = row["perturbation"] is not None and row["perturbation"]["kind"] in ("push", "support") and not str(row["perturbation"].get("target", "")).startswith("wheel:")
        assert (float(np.abs(user).max()) > 0.0) == pushed, (bundle, "external input recorded only for a push or a fought limb")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "core/src"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    from rigby_general.mobility import load_mobile_body
    from rigby_general.mobility.world import course_v1

    validation = json.loads((ROOT / "g17-validation.json").read_bytes())
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    protocol = json.loads((PROTOCOL / "protocol.json").read_bytes())
    for name, digest in registration["files"].items():
        assert sha256(PROTOCOL / name) == digest, name
    body = {k: v for k, v in registration.items() if k not in ("registered_at_utc", "registration_sha256")}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() == registration["registration_sha256"]
    assert validation["registration_sha256"] == registration["registration_sha256"]
    course = course_v1()
    assert course.sha256() == protocol["course_sha256"] == registration["course_sha256"]
    for body_id in BODIES:
        assert load_mobile_body(REPO / "any-robot/assets/general/mobile" / body_id).model_sha256 == protocol["bodies"][body_id]["model_sha256"]
    # -- the trial records ------------------------------------------------------------------------------
    replayed = 0
    d17 = json.loads((D17 / "index.json").read_bytes())
    for body_id in BODIES:
        rows = json.loads((RESULTS / body_id / "trials.json").read_bytes())
        provenance = json.loads((RESULTS / body_id / "provenance.json").read_bytes())
        summary = json.loads((RESULTS / body_id / "summary.json").read_bytes())
        assert provenance["registration_sha256"] == registration["registration_sha256"] and provenance["scored"] and provenance["generation_calls"] == 0
        registered = {t["trial_id"]: t for t in protocol["trials"][body_id]}
        assert {r["trial_id"] for r in rows} == set(registered), body_id
        for r in rows:
            t = registered[r["trial_id"]]
            assert r["seed"] == t["seed"] and r["kind"] == t["kind"] and r["perturbation"] == t["perturbation"] and r["cap_s"] == t["cap_s"], r["trial_id"]
            expected = (r["waypoints_reached"] == len(t["waypoints"])) and r["stable_at_end"] and not r["fell"] and r["final_distance_m"] <= protocol["route"]["release_m"]
            assert r["success"] == expected, (r["trial_id"], "the outcome rule")
            assert r["actuation"]["provenance"].startswith("analytic, hand-authored") and r["actuation"]["root_writes"] == "none after placement" and r["actuation"]["artificial_support"] == "none"
            assert r["duration_s"] <= r["cap_s"] + 1e-6
            if not r["success"]:
                assert "sealed" in r, (r["trial_id"], "every failure is sealed")
        for kind, cell in summary["kinds"].items():
            mine = [r for r in rows if r["kind"] == kind]
            assert cell["trials"] == len(mine) and cell["successes"] == sum(1 for r in mine if r["success"]) and cell["falls"] == sum(1 for r in mine if r["fell"]), (body_id, kind)
        perturbation = [r for r in rows if r["kind"] != "travel"]
        travel = [r for r in rows if r["kind"] == "travel"]
        assert summary["travel"]["successes"] == sum(1 for r in travel if r["success"]) and summary["travel"]["trials"] == len(travel) == protocol["targets"]["travel_of"]
        assert summary["perturbation"]["successes"] == sum(1 for r in perturbation if r["success"]) and summary["perturbation"]["trials"] == len(perturbation) == protocol["targets"]["perturbation_of"]
        assert summary["travel"]["met"] == (summary["travel"]["successes"] >= protocol["targets"]["travel_min"])
        assert summary["perturbation"]["met"] == (summary["perturbation"]["successes"] >= protocol["targets"]["perturbation_min"])
        v = validation["bodies"][body_id]
        assert v["travel"] == [summary["travel"]["successes"], summary["travel"]["trials"]] and v["perturbation"] == [summary["perturbation"]["successes"], summary["perturbation"]["trials"]]
        assert v["falls"] == sum(1 for r in rows if r["fell"]) and v["by_kind"] == summary["perturbation"]["by_kind"]
        # -- D17: every clip's source is a sealed row, its media hashes whole ------------------------------
        by_id = {r["trial_id"]: r for r in rows}
        for clip in d17["bodies"][body_id]:
            row = by_id[clip["trial_id"]]
            assert row["success"] == clip["success"] and row["sealed"]["sha256"] == clip["source_sha256"], clip["clip"]
            media = check_media(D17 / body_id / clip["clip"] / "media")
            assert media["metadata"]["source_bundle_sha256"] == clip["source_sha256"] and media["metadata"]["full_episode"] and media["metadata"]["fps"] == 12
            assert media["metadata"]["playback_duration_s"] >= media["metadata"]["simulation_duration_s"] - 0.2, clip["clip"]
            check_run(REPO / row["sealed"]["bundle"], row["sealed"]["sha256"], row, replay=args.replay)
            replayed += args.replay
        assert {c["clip"] for c in d17["bodies"][body_id]} >= {"travel"}, body_id
    if "synchronized" in d17:
        media = check_media(D17 / "synchronized" / "media")
        for body_id, source in d17["synchronized"]["sources"].items():
            assert media["metadata"]["sources"][body_id]["bundle_sha256"] == source["source_sha256"]
    print(json.dumps({"verified": True, "registration_sha256": registration["registration_sha256"][:12], "bodies": {b: validation["bodies"][b]["travel"] + validation["bodies"][b]["perturbation"] for b in BODIES}, "replayed": replayed, "goal_met": validation["all_targets_met"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
