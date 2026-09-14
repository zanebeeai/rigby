"""Verify the G16 evidence: the three mobile bodies, their integrity and manifests, the settling and recovery tests, the frozen course and feasibility map, D16.

Recomputes every claim from the committed files: the bodies rebuild
byte for byte from the builder; each passes the floating-base integrity
rules; the manifest committed for each body is what ingest measures now;
every stance measurement and recovery trial in the report is the one in
the body's test record, and the record's verdicts agree with the summary;
the course hashes to its registration and the feasibility map is what
the analysis gives for the committed manifests; the D16 index's media
bundles hash whole. With --replay, every sealed run behind the D16 clips
is replayed on native physics and must agree exactly.

    python docs/results/verify_g16.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MOBILE = REPO / "any-robot/assets/general/mobile"
COURSE = MOBILE / "course-v1"
BODIES_OUT = ROOT / "g16-bodies"
COURSE_OUT = ROOT / "g16-course"
D16 = ROOT / "g16-d16"
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


def check_run(bundle: Path, expected: str, *, replay: bool) -> None:
    assert sha256(bundle / "manifest.json") == expected, bundle
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(bundle / name) == info["sha256"], (bundle, name)
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
    import build_mobile_zoo as zoo
    from rigby_general.mobility import check_mobile_integrity, load_mobile_body, measure_mobile_body
    from rigby_general.mobility.contracts import MobileBodyManifestV1, StanceMeasurementV1
    from rigby_general.mobility.world import course_v1, feasibility
    import g16_course

    validation = json.loads((ROOT / "g16-validation.json").read_bytes())
    summary = json.loads((BODIES_OUT / "summary.json").read_bytes())
    # -- bodies: rebuilt byte for byte, integrity, manifests ------------------------------------------
    for builder in zoo.BUILDERS:
        b, declaration = builder()
        committed = (MOBILE / b.robot_id / "robot.xml").read_text(encoding="utf-8")
        assert committed == b.xml(), f"{b.robot_id}: the committed model is not what the builder writes"
        assert sha256(MOBILE / b.robot_id / "robot.xml") == (MOBILE / b.robot_id / "robot.xml.sha256").read_text().strip()
        provenance = json.loads((MOBILE / b.robot_id / "provenance.json").read_bytes())
        assert provenance["origin"].startswith("procedural") and provenance["holdout"] is False and "licence" in provenance
    for body_id in BODIES:
        body = load_mobile_body(MOBILE / body_id)
        report = check_mobile_integrity(body)
        assert report.passed, (body_id, report.violations)
        committed = MobileBodyManifestV1.model_validate_json((BODIES_OUT / body_id / "manifest.json").read_bytes())
        measured = measure_mobile_body(body)
        assert committed.model_dump(mode="json") == measured.model_dump(mode="json"), f"{body_id}: the committed manifest is not what ingest measures"
        tests = json.loads((BODIES_OUT / body_id / "tests.json").read_bytes())
        integrity = json.loads((BODIES_OUT / body_id / "integrity.json").read_bytes())
        assert integrity["passed"] and integrity["robot_id"] == body_id
        cell = validation["bodies"][body_id]
        assert cell["integrity"] is True and cell["mass_kg"] == round(report.mass_kg, 3)
        for stance in tests["stances"]:
            assert stance["statically_stable"] == body.declaration["stances"][stance["stance"]]["statically_stable"], (body_id, stance["stance"], "measured stability disagrees with the declaration")
            assert cell["stances"][stance["stance"]] == stance["status"]
            if args.replay:
                check_run(Path(stance["sealed"]["bundle"]), stance["sealed"]["sha256"], replay=True)
        supported = [r for r in tests["recoveries"] if r["supported"]]
        assert cell["recoveries"]["supported_recovered"] == sum(1 for r in supported if r["recovered"]) and cell["recoveries"]["supported_total"] == len(supported)
        assert cell["recoveries"]["unsupported_total"] == sum(1 for r in tests["recoveries"] if not r["supported"])
        for trial in tests["recoveries"]:
            if args.replay:
                check_run(Path(trial["sealed"]["bundle"]), trial["sealed"]["sha256"], replay=True)
        if args.replay:
            check_run(Path(tests["inspection"]["sealed"]["bundle"]), tests["inspection"]["sealed"]["sha256"], replay=True)
        assert tests["inspection"]["upright"] is True
        assert all(any(e["joint"] == j.name for e in tests["inspection"]["schedule"]) for j in measured.joints), "every joint was swept"
    # -- the course: frozen, and the feasibility map recomputed ----------------------------------------
    registration = json.loads((COURSE / "registration.json").read_bytes())
    course = course_v1()
    assert registration["course_sha256"] == course.sha256() == validation["course_sha256"]
    for name, digest in registration["files"].items():
        assert sha256(COURSE / name) == digest, name
    committed_course = json.loads((COURSE / "course.json").read_bytes())
    assert committed_course == course.as_json()
    fmap = json.loads((COURSE / "feasibility.json").read_bytes())
    labels = json.loads((COURSE / "expert-labels.json").read_bytes())
    assert labels["labeller"].startswith("internal")
    for body_id in BODIES:
        manifest = MobileBodyManifestV1.model_validate_json((BODIES_OUT / body_id / "manifest.json").read_bytes())
        tests = json.loads((BODIES_OUT / body_id / "tests.json").read_bytes())
        stances = tuple(StanceMeasurementV1.model_validate({k: v for k, v in s.items() if k not in ("status", "sealed")}) for s in tests["stances"])
        recomputed = feasibility(course, manifest, stances, **g16_course.GEOMETRY[body_id])
        assert [v["feasible"] for v in recomputed["verdicts"]] == [v["feasible"] for v in fmap["bodies"][body_id]["verdicts"]], body_id
        for branch, entry in fmap["agreement"][body_id].items():
            assert entry["expert"] == labels["labels"][body_id][branch] and entry["agree"] == (entry["analysis"] == entry["expert"])
        assert validation["feasibility"][body_id] == {v["branch_id"]: v["feasible"] for v in recomputed["verdicts"]}
    settle = json.loads((COURSE_OUT / "course-settle.json").read_bytes())
    assert settle["provenance"]["registration_sha256"] == registration["registration_sha256"]
    for body_id in BODIES:
        assert settle["bodies"][body_id]["stable"] is True
        if args.replay:
            check_run(Path(settle["bodies"][body_id]["sealed"]["bundle"]), settle["bodies"][body_id]["sealed"]["sha256"], replay=True)
    # -- D16 ---------------------------------------------------------------------------------------------------
    index = json.loads((D16 / "index.json").read_bytes())
    assert index["generation_calls"] == 0 and set(index["bodies"]) == set(BODIES)
    for body_id, clips in index["bodies"].items():
        kinds = {c["clip"] for c in clips}
        assert "inspection" in kinds and "course-start" in kinds and any(k.startswith("stance-") for k in kinds) and any(k.startswith("recovery-") for k in kinds)
        assert any(c.get("supported") is False for c in clips), f"{body_id}: an unsupported manoeuvre is shown"
        for clip in clips:
            manifest = check_media(D16 / Path(clip["video"]).parent)
            assert sha256(D16 / Path(clip["video"]).parent / "manifest.json") == clip["media_sha256"]
            assert manifest["metadata"]["frame_count"] == clip["frame_count"] and manifest["metadata"]["source_bundle_sha256"] == clip["source_sha256"]
    print(json.dumps({"verified": True, "bodies": {b: validation["bodies"][b]["stances"] for b in BODIES}, "feasibility": validation["feasibility"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
