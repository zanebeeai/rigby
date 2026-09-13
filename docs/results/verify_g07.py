"""Verify the G07 skill-contract evidence: schemas, the nested example, its records, and the tree runs on physics.

Recomputes every claim the report makes from the committed files. The
schemas on disk are the models' own; the example library round-trips to the
same hash and expands to the committed tree; every scripted record reaches
the verdict the index says, and is itself a valid record; every physical
run's bundle is integrity-checked (and replayed with --replay), its media
checked against real-time playback, and its recorded verdict matched to the
one the report claims for it.

    python docs/results/verify_g07.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TREES = ROOT / "g07-trees"
EXPECTED_PHYSICAL = {"dual-selector": "success", "hand-budget": "failure", "jaw-interrupt": "interrupted", "compact-refused": "failure"}
EXPECTED_SCRIPTED = {
    "success-first-alternative": "success", "success-second-alternative": "success", "failure-budget-exhausted": "failure",
    "failure-no-progress": "failure", "unknown-nothing-observed": "unknown", "interrupted-mid-leaf": "interrupted",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bundle(physical: Path, expected_sha256: str, *, replay: bool) -> dict:
    manifest = json.loads((physical / "manifest.json").read_bytes())
    assert sha256(physical / "manifest.json") == expected_sha256, physical
    for name, info in manifest["files"].items():
        assert sha256(physical / name) == info["sha256"], (physical, name)
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        model = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
        record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        assert replay_physics(model, record)["agrees"], physical
    return manifest


def check_media(media: Path, source_sha256: str) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    assert manifest["metadata"]["source_bundle_sha256"] == source_sha256
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"]
    if manifest["metadata"].get("refusal_slate"):
        assert manifest["metadata"]["playback_duration_s"] > 0
    else:
        assert abs(manifest["metadata"]["playback_duration_s"] - manifest["metadata"]["simulation_duration_s"]) <= 2.0 / manifest["metadata"]["fps"]
    for name in ("episode.mp4", "preview.gif", "frames.json"):
        assert (media / name).is_file(), (media, name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g07-validation.json").read_bytes())
    index = json.loads((TREES / "index.json").read_bytes())
    assert index["commit"] == validation["evidence_commit"] and index["generation_calls"] == 0
    for name, digest in index["files"].items():
        assert sha256(TREES / name) == digest, name

    # -- the contract itself, from the installed package ---------------------
    from rigby_core.skills import ExecutionRecordV1, SkillLibraryV1, TaskTreeV1
    from rigby_core.skills.examples import ARGUMENTS, clear_bench_library

    for model in (SkillLibraryV1, TaskTreeV1, ExecutionRecordV1):
        on_disk = json.loads((TREES / "schema" / f"{model.__name__}.schema.json").read_bytes())
        assert on_disk == model.model_json_schema(), f"{model.__name__} schema drifted"
    library = SkillLibraryV1.model_validate_json((TREES / "example" / "clear_bench_v1.library.json").read_bytes())
    assert library == clear_bench_library() and library.content_hash() == index["library"]["sha256"]
    tree = TaskTreeV1.model_validate_json((TREES / "example" / "clear_bench.tree.json").read_bytes())
    assert tree == library.expand("clear_bench", ARGUMENTS) and tree.content_hash() == index["tree"]["sha256"]
    assert tree.max_depth >= 4 and index["tree"]["max_depth"] == tree.max_depth == validation["example_depth"]
    assert set(index["tree"]["kinds"]) == {"sequence", "selector", "repeat_until", "observe", "primitive"}
    assert all(index["round_trip"].values())

    # -- scripted records ---------------------------------------------------
    seen = {}
    for entry in index["scripted"]:
        record = ExecutionRecordV1.model_validate_json((TREES / "scripted" / f"{entry['name']}.record.json").read_bytes())
        assert record.content_hash() == entry["record_sha256"]
        assert record.verdict.value == entry["verdict"] == EXPECTED_SCRIPTED[entry["name"]], entry["name"]
        assert record.library_id == library.library_id
        seen[entry["name"]] = record.verdict.value
    assert seen == EXPECTED_SCRIPTED
    assert {"success", "failure", "unknown", "interrupted"} <= set(seen.values())

    # -- physics ------------------------------------------------------------
    physical_seen = {}
    for run in index["physical"]:
        name = run["name"]
        physical = TREES / "physical" / name / "physical"
        manifest = check_bundle(physical, run["bundle"]["sha256"], replay=args.replay)
        outcome = json.loads((physical / "outcome.json").read_bytes())
        assert outcome["verdict"] == run["verdict"] == EXPECTED_PHYSICAL[name], name
        record = ExecutionRecordV1.model_validate_json(json.dumps(json.loads((physical / "execution.json").read_bytes())["record"]))
        assert record.verdict.value == run["verdict"] and record.tree_sha256 == run["tree_sha256"]
        on_disk_tree = TaskTreeV1.model_validate_json((physical / "tree.json").read_bytes())
        assert on_disk_tree.content_hash() == run["tree_sha256"] and on_disk_tree.max_depth == run["max_depth"] >= 4
        task = json.loads((physical / "task.json").read_bytes())
        assert task["library_sha256"] == library.content_hash() and task["attempts"] == 1 and task["retry_limit"] == 0
        assert task["clock_disclosure"]["phase_timing_on_native_physics_time"] is True
        world = json.loads((physical / "world.json").read_bytes())
        policy = world["collision_policy"]
        assert policy["equality_constraints"] == 0 and policy["object_actuators"] == 0 and policy["contact_exclusions_with_object"] == [] and policy["object_geoms_collidable"]
        repeats = json.loads((physical / "repeats.json").read_bytes())
        assert repeats["recorded_control_replay"]["agrees"] is True
        check_media(TREES / "physical" / name / "media", run["bundle"]["sha256"])
        if name == "dual-selector":
            leaves = [r for r in record.root.walk() if r.kind.value == "primitive"]
            assert [r.verdict.value for r in leaves] == ["failure", "success"] and leaves[0].reason in ("self_collision_path", "unreachable_path")
            assert outcome["leaves"][0]["executed"] is False and outcome["leaves"][1]["certified"] is True
            assert manifest["metadata"]["outcome"] == "success"
        if name == "jaw-interrupt":
            assert record.interrupted and record.record("0.1.0.0.0").reason == "interrupted"
            assert 6.0 <= outcome["leaves"][0]["phases"][-1]["end_s"] < 6.01
            assert manifest["metadata"]["outcome"] == "interrupted"
        if name == "hand-budget":
            assert record.record("0.1").reason in ("budget_exhausted", "no_progress")
            assert manifest["metadata"]["outcome"] == "runtime_failure"
        if name == "compact-refused":
            assert outcome["physical_steps"] == 0 and manifest["metadata"]["outcome"] == "pre_execution_refusal"
        physical_seen[name] = run["verdict"]
    assert physical_seen == EXPECTED_PHYSICAL
    assert validation["physical"] == physical_seen and validation["scripted"] == seen
    print(json.dumps({"verified": True, "example_depth": tree.max_depth, "scripted": seen, "physical": physical_seen, "replayed": args.replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
