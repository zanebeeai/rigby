"""Verify the G06 transfer evidence: registration, campaign outcomes, bundles, media, D06.

Recomputes every claim the report makes from the committed files. The frozen
fixture, goal, feasibility map and roster are checked against their
registration; per-body success counts, the failure taxonomy and the enabled
set are re-derived from the per-trial records rather than read from the
summary; every committed physical bundle is integrity-checked and, with
--replay, its recorded controls are replayed through the physics; every
rendered episode's frame map is checked against real-time playback; the D06
reels and before/after pairs are checked against their index.

    python docs/results/verify_g06.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g06-transfer-v1"
CAMPAIGN = ROOT / "g06-campaign"
D06 = ROOT / "g06-d06"
SUCCESS_TARGET = 90
TRIALS_PER_BODY = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bundle(physical: Path, expected_sha256: str, *, replay: bool) -> dict:
    manifest = json.loads((physical / "manifest.json").read_bytes())
    assert sha256(physical / "manifest.json") == expected_sha256, physical
    for name, info in manifest["files"].items():
        assert sha256(physical / name) == info["sha256"], (physical, name)
    assert manifest["metadata"]["reference_clock_matches_physics"] is True
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
    assert manifest["metadata"]["full_episode"] and manifest["metadata"]["preview_is_summary"]
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"]
    assert abs(len(frames) / manifest["metadata"]["fps"] - manifest["metadata"]["playback_duration_s"]) < 1e-9
    # Real-time playback: the frame count is the ceiling of duration times fps
    # plus the final held state, so playback never differs from the simulated
    # duration by more than two frame periods.
    assert abs(manifest["metadata"]["playback_duration_s"] - manifest["metadata"]["simulation_duration_s"]) <= 2.0 / manifest["metadata"]["fps"]
    for name in ("episode.mp4", "preview.gif", "frames.json"):
        assert (media / name).is_file(), (media, name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="also replay every committed physical bundle's recorded controls (needs the workspace environment)")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g06-validation.json").read_bytes())

    # -- the registration -----------------------------------------------------
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        assert sha256(PROTOCOL / name) == digest, name
    files = json.dumps(registration["files"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert hashlib.sha256(files.encode("utf-8")).hexdigest() == registration["registration_sha256"] == validation["registration_sha256"]
    roster = json.loads((PROTOCOL / "roster.json").read_bytes())
    feasibility = {row["zoo_id"]: row for row in json.loads((PROTOCOL / "feasibility-map.json").read_bytes())["bodies"]}
    draws = [body["seeds"] for body in roster["bodies"]]
    assert all(d == draws[0] for d in draws) and len(draws[0]) == TRIALS_PER_BODY
    assert roster["scored_runs_started"] is False and roster["generation_calls"] == 0
    for zoo_id, row in feasibility.items():
        if row["class"] == "infeasible":
            assert row["reasons"] and all(r.startswith("necessary condition violated: ") for r in row["reasons"])
            assert any(not row["conditions"][r.split(": ")[1]]["ok"] for r in row["reasons"])
        elif row["class"] == "feasible":
            assert row["witness_configurations"]["source"] and row["witness_configurations"]["destination"]
        elif row["class"] == "unsupported_by_structure":
            assert row["reasons"] == ["no grasping effector"]
        else:
            raise AssertionError(f"{zoo_id}: unexpected class {row['class']}")
    assert set(registration["feasible_bodies"]) == {z for z, r in feasibility.items() if r["class"] == "feasible"}

    # -- the campaign ---------------------------------------------------------
    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    assert summary["registration_sha256"] == registration["registration_sha256"]
    assert summary["provenance"]["commit"] == validation["campaign_commit"]
    assert summary["provenance"]["generation_calls"] == 0 and summary["provenance"]["scored"] is True
    assert summary["provenance"]["seed_limit"] is None
    bodies = {}
    enabled = []
    taxonomy = {}
    rendered = {"successes": 0, "failures": 0}
    for entry in summary["bodies"]:
        zoo_id = entry["zoo_id"]
        trials = json.loads((CAMPAIGN / zoo_id / "trials.json").read_bytes())
        assert trials["feasibility_class"] == feasibility[zoo_id]["class"] == entry["feasibility_class"]
        assert trials["rig_id"] == entry["rig_id"]
        if entry["feasibility_class"] == "unsupported_by_structure":
            assert not trials["normalized"]["attempted"] and not trials["fixed"]["attempted"]
            bodies[zoo_id] = {"class": "unsupported_by_structure"}
            continue
        for track in ("normalized", "fixed-canonical"):
            record = trials["normalized"] if track == "normalized" else trials["fixed"]["canonical"]
            physical = CAMPAIGN / zoo_id / track / "physical"
            manifest = check_bundle(physical, record["bundle"]["sha256"], replay=args.replay)
            expected = "success" if record["certified"] else ("pre_execution_refusal" if not record["phases"] else "runtime_failure")
            assert manifest["metadata"]["outcome"] == expected == record["bundle"]["outcome"], (zoo_id, track)
            outcome = json.loads((physical / "outcome.json").read_bytes())
            assert outcome["certified"] == record["certified"] and outcome["failed_gate"] == record["failed_gate"]
            task = json.loads((physical / "task.json").read_bytes())
            assert task["clock_disclosure"]["phase_timing_on_native_physics_time"] is True
            assert task["attempts"] == 1 and task["retry_limit"] == 0 and task["interventions"] == []
            if record["certified"]:
                assert not outcome["violations"] and outcome["placement_success"] and outcome["released"] and outcome["placed_inside"]
                assert [p["name"] for p in json.loads((physical / "execution.json").read_bytes())["phases"]] == list(validation["phases"])
            else:
                assert outcome["violations"] and outcome["violations"][0]["code"] == record["failed_gate"]
            repeats = json.loads((physical / "repeats.json").read_bytes())
            assert repeats["recorded_control_replay"]["agrees"] is True
            check_media(CAMPAIGN / zoo_id / track / "media", record["bundle"]["sha256"])
        assert trials["normalized"]["certified"] == entry["normalized_certified"]
        fixed = trials["fixed"]
        rows = fixed["trials"]
        expected_seeds = [d["seed"] for d in draws[0]] if entry["feasibility_class"] == "feasible" else [draws[0][0]["seed"]]
        assert [r["seed"] for r in rows] == expected_seeds, zoo_id
        successes = 0
        body_taxonomy: dict[str, int] = {}
        for row, draw in zip(rows, draws[0]):
            assert row["draw"] == draw
            assert row["trace_file_sha256"] or not row["phases"], (zoo_id, row["seed"])
            if row["certified"]:
                assert row["failed_gate"] is None and not row["violations"]
                assert row["placement_success"] and row["released"] and row["placed_inside"] and row["opposition_achieved"]
                assert row["max_penetration_m"] <= 0.004 + 1e-12
                successes += 1
            else:
                assert row["failed_gate"] and row["violations"][0]["code"] == row["failed_gate"]
                body_taxonomy[row["failed_gate"]] = body_taxonomy.get(row["failed_gate"], 0) + 1
            if "bundle" in row:
                media = CAMPAIGN / zoo_id / "fixed" / f"seed-{row['seed']:03d}" / "media"
                manifest = check_media(media, row["bundle"]["sha256"])
                assert manifest["metadata"]["outcome"] == row["bundle"]["outcome"]
                assert (row["certified"] and manifest["metadata"]["outcome"] == "success") or (not row["certified"] and manifest["metadata"]["outcome"] != "success")
                rendered["successes" if row["certified"] else "failures"] += 1
        assert successes == fixed["successes"] == entry["fixed_successes"]
        assert len(rows) == fixed["count"] == entry["fixed_count"]
        assert body_taxonomy == fixed["failure_taxonomy"] == entry["fixed_failure_taxonomy"]
        for gate in body_taxonomy:
            assert any("bundle" in r for r in rows if not r["certified"] and r["failed_gate"] == gate), f"{zoo_id}: {gate} has no rendered clip"
        assert fixed["canonical"]["certified"] == entry["fixed_canonical_certified"]
        meets = entry["feasibility_class"] == "feasible" and len(rows) == TRIALS_PER_BODY and successes >= SUCCESS_TARGET
        if meets and entry["fixed_canonical_certified"] and entry["normalized_certified"]:
            enabled.append(zoo_id)
        for gate, count in body_taxonomy.items():
            taxonomy[gate] = taxonomy.get(gate, 0) + count
        bodies[zoo_id] = {"class": entry["feasibility_class"], "successes": successes, "count": len(rows), "canonical": entry["fixed_canonical_certified"],
                          "normalized": entry["normalized_certified"], "taxonomy": body_taxonomy}
    assert set(bodies) == set(feasibility)
    assert sorted(enabled) == sorted(validation["enabled_set"]), (enabled, validation["enabled_set"])
    assert len(enabled) >= 3
    assert {bodies[z]["class"] for z in enabled} == {"feasible"}
    assert validation["failure_taxonomy"] == taxonomy, (taxonomy, validation["failure_taxonomy"])
    assert validation["bodies"] == bodies, "the validation record disagrees with the per-trial records"

    # -- D06 ------------------------------------------------------------------
    index = json.loads((D06 / "index.json").read_bytes())
    assert index["registration_sha256"] == registration["registration_sha256"]
    for name, digest in index["files"].items():
        assert sha256(D06 / name) == digest, name
    assert len(index["five_body"]["bodies"]) == 5
    assert index["five_body"]["frames"] >= max(b["frames"] for b in index["five_body"]["bodies"])
    for body in index["five_body"]["bodies"]:
        media = CAMPAIGN / body["zoo_id"] / "fixed-canonical" / "media"
        assert sha256(media / "manifest.json") == body["media_sha256"]
    assert sorted(b["zoo_id"] for b in index["enabled_set"]["bodies"]) == sorted(enabled)
    assert all(b["outcome"] == "success" for b in index["enabled_set"]["bodies"])
    assert {p["repair"] for p in index["repairs"]} >= {"facing", "standoff"}
    for pair in index["repairs"]:
        before, after = pair["before"], pair["after"]
        assert before["outcome"] != "success"
        assert (before["outcome"], before["failed_gate"]) != (after["outcome"], after["failed_gate"])
        if pair["after_certifies"]:
            assert after["outcome"] == "success" and after["certified"]
        for label in ("before", "after"):
            physical = D06 / pair["repair"] / label / "physical"
            check_bundle(physical, pair[label]["physical_sha256"], replay=args.replay)
            check_media(D06 / pair["repair"] / label / "media", pair[label]["physical_sha256"])
        assert pair["side_by_side"]["frames"] >= max(before["frames"], after["frames"])
    assert validation["repairs"] == {p["repair"]: {"body": p["body"], "before": p["before"]["failed_gate"], "after": p["after"]["failed_gate"]} for p in index["repairs"]}

    print(json.dumps({"verified": True, "enabled_set": enabled, "bodies": bodies, "failure_taxonomy": taxonomy,
                      "rendered_seeded_trials": rendered, "replayed": args.replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
