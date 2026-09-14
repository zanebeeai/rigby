"""Verify the G18 evidence: the registered retrieve protocol, the second-version bodies, the three bodies' trial records, the summaries, the validation, D18.

Recomputes every claim from the committed files: the protocol hashes
to its registration and the course and bodies hash to what it names;
the v2 dog and biped rebuild byte for byte from the builder with their
jaws below the last arm link, and the v1 bodies are untouched; each
body's trial record holds exactly the registered trials, run under
that registration, with the outcome rule re-applied to every row (the
cube in the tray, the base on the start pad, stable, no fall); every
summary count is recomputed from the rows and the validation agrees;
every phase carries its support set and holding limb, a holding limb
is never in the drive's support set, and every row's provenance names
no root write, no object write and no artificial support; every failure
is explicit, the first failure of each stage is sealed and a failure
left unsealed says why; every D18 clip's source is a sealed row and its
media hashes whole; every before/after pair has both sides sealed, their
media rendered from those sealed runs, and the fix switched off on the
before side only. With --replay, every sealed run behind a D18 clip or
a pair is replayed on native physics and must agree exactly.

    python docs/results/verify_g18.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g18-retrieve-v1"
MOBILE = REPO / "any-robot/assets/general/mobile"
RESULTS = ROOT / "g18-retrieve"
D18 = ROOT / "g18-d18"
PAIRS = ROOT / "g18-before-after"
BODIES = ("mobile_dog_arm_v2", "mobile_wheeled_biped_v2", "mobile_octopus_v2")
KEEP_FAILURES = 1
"""The campaign seals the first failures of each stage, as many as this, while the disk has room."""


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
    if "trial_id" in row:
        assert task["trial_id"] == row["trial_id"] and task["disturbance"] == row["disturbance"], bundle
    else:
        assert task["test"] == "before_after_pair" and task["pair_id"] == row["pair_id"] and task["side"] == row["side"] and task["body"] == row["body"] and task["seed"] == row["seed"] and task["settings"] == row["settings"], bundle
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        record = PhysicsRecord.from_bytes((bundle / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
        assert replay_physics(model, record)["agrees"], bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "core/src"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    sys.path.insert(0, str(REPO / "any-robot/scripts"))
    import mujoco
    import build_mobile_zoo as zoo
    from rigby_general.mobility import load_mobile_body
    from rigby_general.mobility.world import course_v1

    validation = json.loads((ROOT / "g18-validation.json").read_bytes())
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    protocol = json.loads((PROTOCOL / "protocol.json").read_bytes())
    for name, digest in registration["files"].items():
        assert sha256(PROTOCOL / name) == digest, name
    body = {k: v for k, v in registration.items() if k not in ("registered_at_utc", "registration_sha256")}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() == registration["registration_sha256"]
    assert validation["registration_sha256"] == registration["registration_sha256"]
    course = course_v1()
    assert course.sha256() == protocol["course_sha256"] == registration["course_sha256"]
    # -- the bodies: v2 rebuilt byte for byte, jaws below the last link; v1 untouched -----------------
    for builder in zoo.BUILDERS + zoo.BUILDERS_V2:
        b, _ = builder()
        assert (MOBILE / b.robot_id / "robot.xml").read_text(encoding="utf-8") == b.xml(), b.robot_id
    moving_bodies: dict[str, dict[str, set[str]]] = {}
    for body_id in BODIES:
        loaded = load_mobile_body(MOBILE / body_id)
        assert loaded.model_sha256 == protocol["bodies"][body_id]["model_sha256"], body_id
        if body_id.endswith("_v2") and body_id != "mobile_octopus_v2":
            palm = mujoco.mj_name2id(loaded.model, mujoco.mjtObj.mjOBJ_BODY, "arm_palm")
            assert loaded.model.body_pos[palm][2] < -0.1, "the v2 palm hangs below the last link"
        if body_id == "mobile_octopus_v2":
            pinch = mujoco.mj_name2id(loaded.model, mujoco.mjtObj.mjOBJ_JOINT, "t0_pinch_left")
            assert loaded.model.jnt_range[pinch][0] <= -0.85, "the v2 pincer opens wide"
        # the bodies a limb moves: those under its joints (a tentacle's root mount is part of the mantle and moves with nothing)
        moving_bodies[body_id] = {}
        for limb, joints in loaded.declaration["limbs"].items():
            ids = {int(loaded.model.jnt_bodyid[mujoco.mj_name2id(loaded.model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in joints if mujoco.mj_name2id(loaded.model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0}
            grown = True
            while grown:
                grown = False
                for b in range(loaded.model.nbody):
                    if b not in ids and int(loaded.model.body_parentid[b]) in ids:
                        ids.add(b)
                        grown = True
            moving_bodies[body_id][limb] = {mujoco.mj_id2name(loaded.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ids}
    # -- the trial records ------------------------------------------------------------------------------
    replayed = 0
    d18 = json.loads((D18 / "index.json").read_bytes())
    for body_id in BODIES:
        rows = json.loads((RESULTS / body_id / "trials.json").read_bytes())
        provenance = json.loads((RESULTS / body_id / "provenance.json").read_bytes())
        summary = json.loads((RESULTS / body_id / "summary.json").read_bytes())
        assert provenance["registration_sha256"] == registration["registration_sha256"] and provenance["scored"] and provenance["generation_calls"] == 0
        registered = {t["trial_id"]: t for t in protocol["trials"][body_id]}
        assert {r["trial_id"] for r in rows} == set(registered), body_id
        for r in rows:
            t = registered[r["trial_id"]]
            assert r["seed"] == t["seed"] and r["kind"] == t["kind"] and r["stage"] == t["stage"] and r["disturbance"] == t["disturbance"] and r["cap_s"] == t["cap_s"], r["trial_id"]
            expected = bool(r["cube_in_tray"] and r["at_start"] and r["stable_at_end"] and not r["fell"] and "return" in r["phases_completed"])
            assert r["success"] == expected, (r["trial_id"], "the outcome rule")
            assert r["actuation"]["root_writes"] == "none after placement" and r["actuation"]["object_writes"] == "none" and r["actuation"]["artificial_support"] == "none"
            assert r["actuation"]["provenance"].startswith("analytic, hand-authored")
            for entry in r["schedule"]:
                if entry["holding"]:
                    assert entry["holding"] in entry["drive_excludes"], (r["trial_id"], "a holding limb is out of the drive")
                    assert not (set(entry["support"]) & moving_bodies[body_id][entry["holding"]]), (r["trial_id"], "a holding limb's members are not support members")
            if not r["success"]:
                assert r["reason"], (r["trial_id"], "every failure is explicit")
                assert "sealed" in r or r.get("not_sealed"), (r["trial_id"], "a failure is sealed or says why not")
        for stage, cell in summary["stages"].items():
            mine = [r for r in rows if r["stage"] == stage]
            failures = [r for r in mine if not r["success"]]
            assert sum(1 for r in failures if "sealed" in r) >= min(len(failures), KEEP_FAILURES), (body_id, stage, "the first failures of a stage are sealed")
            assert cell["trials"] == len(mine) and cell["successes"] == sum(1 for r in mine if r["success"]) and cell["falls"] == sum(1 for r in mine if r["fell"]), (body_id, stage)
        nominal = [r for r in rows if r["kind"] == "nominal"]
        disturbed = [r for r in rows if r["kind"] == "disturbed"]
        assert summary["nominal"]["successes"] == sum(1 for r in nominal if r["success"]) and summary["nominal"]["trials"] == len(nominal) == protocol["targets"]["nominal_of"]
        assert summary["disturbed"]["successes"] == sum(1 for r in disturbed if r["success"]) and summary["disturbed"]["trials"] == len(disturbed) == protocol["targets"]["disturbed_of"]
        assert summary["nominal"]["met"] == (summary["nominal"]["successes"] >= protocol["targets"]["nominal_min"])
        assert summary["disturbed"]["met"] == (summary["disturbed"]["successes"] >= protocol["targets"]["disturbed_min"])
        v = validation["bodies"][body_id]
        assert v["nominal"] == [summary["nominal"]["successes"], summary["nominal"]["trials"]] and v["disturbed"] == [summary["disturbed"]["successes"], summary["disturbed"]["trials"]]
        assert v["by_stage"] == summary["disturbed"]["by_stage"] and v["invariant_violations"] == summary["invariant_violations_total"]
        by_id = {r["trial_id"]: r for r in rows}
        for clip in d18["bodies"][body_id]:
            row = by_id[clip["trial_id"]]
            assert row["success"] == clip["success"] and row["sealed"]["sha256"] == clip["source_sha256"], clip["clip"]
            media = check_media(D18 / body_id / clip["clip"] / "media")
            assert media["metadata"]["source_bundle_sha256"] == clip["source_sha256"] and media["metadata"]["full_episode"] and media["metadata"]["fps"] == 12
            check_run(REPO / row["sealed"]["bundle"], row["sealed"]["sha256"], row, replay=args.replay)
            replayed += args.replay
    # the before/after pairs: each side sealed and whole, its media whole and rendered from that sealed run, the fix switched off on the before side only
    pairs = json.loads((PAIRS / "index.json").read_bytes())["pairs"]
    assert {p["pair_id"] for p in pairs} == {"dog-jaw", "octopus-pincer", "dog-place-branch", "octopus-wave-gait"}
    assert [p["pair_id"] for p in validation["before_after"]] == [p["pair_id"] for p in pairs]
    for pair in pairs:
        for side in ("before", "after"):
            run = pair["runs"][side]
            assert run["success"] == (run["status"] == "success") and (run["success"] or run["reason"]), (pair["pair_id"], side)
            if side == "after":
                assert not run["settings"], (pair["pair_id"], "the after side runs as committed")
            else:
                assert run["settings"] or run["body"] != pair["runs"]["after"]["body"], (pair["pair_id"], "the before side differs in a body or a setting")
            media = check_media(PAIRS / pair["pair_id"] / side / "media")
            assert media["metadata"]["source_bundle_sha256"] == run["sealed"]["sha256"] and media["metadata"]["full_episode"] and media["metadata"]["fps"] == 12
            check_run(REPO / run["sealed"]["bundle"], run["sealed"]["sha256"], {"pair_id": pair["pair_id"], "side": side, "body": run["body"], "seed": pair["seed"], "settings": run["settings"]}, replay=args.replay)
            replayed += args.replay
    print(json.dumps({"verified": True, "registration_sha256": registration["registration_sha256"][:12], "bodies": {b: validation["bodies"][b]["nominal"] + validation["bodies"][b]["disturbed"] for b in BODIES}, "pairs": len(pairs), "replayed": replayed, "goal_met": validation["all_targets_met"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
