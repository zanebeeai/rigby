"""Verify the G05 composition evidence: roster, campaign outcomes, bundles, media.

Recomputes every claim the report makes from the committed files. Physical
bundles are integrity-checked and their recorded-control physics is replayed;
the six canonical authored durations are compared with the recorded baseline;
per-body trial counts, refusal typing and the 19-of-20 gate are re-derived
from the per-trial records rather than read from the summary.

    python docs/results/verify_g05.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ROSTER = REPO / "any-robot/assets/general/research-protocols/g05-composition-v1"
CAMPAIGN = ROOT / "g05-campaign"
D05 = ROOT / "g05-d05"
BASELINE_DURATIONS = {
    # docs/results/g05-curves-report.md, authored durations at 5a6c0b3 (legacy intake,
    # native clock, bounded curves). The dual arm's is the refused, self-colliding program.
    "zoo_compact_arm": 30.509918, "zoo_dual_arm": 60.591332, "zoo_hand_arm": 49.245976,
    "zoo_jaw_arm": 47.844854, "zoo_long_arm": 67.454452, "zoo_tool_arm": 52.437660,
}
REFUSED_BASELINE = {"zoo_dual_arm"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="also replay every physical bundle's recorded controls (needs the workspace environment)")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g05-validation.json").read_bytes())

    registration = json.loads((ROSTER / "registration.json").read_bytes())
    roster = json.loads((ROSTER / "roster.json").read_bytes())
    assert sha256(ROSTER / "roster.json") == registration["roster_sha256"] == validation["roster_sha256"]
    assert registration["feasible_trials"] == sum(len(b["feasible_trials"]) for b in roster["bodies"]) == 120
    assert registration["invalid_requests"] == 18 and registration["not_constructible"] == 0
    for body in roster["bodies"]:
        assert len(body["starts"]) == 4 and len(body["feasible_trials"]) == 20
        assert sorted(t["pace_ordinal"] for t in body["feasible_trials"]) == sorted([-2, -1, 0, 1, 2] * 4)
        assert body["starts"][0]["kind"] == "measured_rest" and not body["starts"][0]["offsets"]
        for start in body["starts"][1:]:
            assert start["kind"] == "displaced" and start["offsets"] and start["redraws"] == 0
            assert len(start["qpos"]) == body["dof_count"]

    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    assert summary["roster_sha256"] == registration["roster_sha256"]
    assert summary["provenance"]["commit"] == validation["campaign_commit"]
    assert summary["provenance"]["generation_calls"] == 0
    zoo = {b["zoo_id"]: b for b in roster["bodies"]}
    seen = set()
    totals = {"feasible": 0, "successes": 0, "invalid": 0, "invalid_correct": 0}
    durations = {}
    for entry in summary["bodies"]:
        zoo_id = entry["zoo_id"]
        seen.add(zoo_id)
        trials = json.loads((CAMPAIGN / zoo_id / "trials.json").read_bytes())
        assert trials["rig_id"] == zoo[zoo_id]["rig_id"] and trials["package_sha256"] == zoo[zoo_id]["package_sha256"]
        rows = trials["feasible_trials"]
        assert [r["trial_id"] for r in rows] == [t["trial_id"] for t in zoo[zoo_id]["feasible_trials"]]
        successes = 0
        for row in rows:
            if row["accepted"]:
                assert row["certified"] and row["replay_agreement"] and len(row["replay_hashes"]) == 3
                assert row["region_substitutions"] == 0 and row["fresh_leaves_certified"] == 2
                assert not row["violations"] and not row["unexpected_contacts"]
                assert abs(row["actual_physics_duration_s"] - row["authored_duration_s"]) <= 0.002 + 1e-9
                successes += 1
            else:
                assert row["failure"]["code"], "a failed trial must carry a typed failure"
        assert successes == entry["feasible_successes"] == trials["feasible_successes"]
        assert entry["meets_19_of_20"] == (successes >= 19 and len(rows) >= 20) == trials["meets_19_of_20"]
        invalid = trials["invalid_requests"]
        assert [c["case_id"] for c in invalid] == [c["case_id"] for c in zoo[zoo_id]["invalid_requests"]]
        correct = sum(1 for c in invalid if c.get("correct_typed_refusal"))
        assert correct == entry["invalid_correctly_refused"]
        for case in invalid:
            assert not case.get("accepted", False)
        totals["feasible"] += len(rows)
        totals["successes"] += successes
        totals["invalid"] += len(invalid)
        totals["invalid_correct"] += correct

        physical = CAMPAIGN / zoo_id / "canonical" / "physical"
        manifest = json.loads((physical / "manifest.json").read_bytes())
        assert sha256(physical / "manifest.json") == entry["canonical_sha256"] == trials["canonical"]["sha256"]
        for name, info in manifest["files"].items():
            assert sha256(physical / name) == info["sha256"], name
        assert manifest["metadata"]["outcome"] == "success" == entry["canonical_outcome"]
        outcome = json.loads((physical / "outcome.json").read_bytes())
        assert outcome["region_substitutions"] == 0 and outcome["leaf_duration_scales"] == [1.0, 1.0]
        assert not outcome["violations"] and not outcome["unexpected_contacts"]
        task = json.loads((physical / "task.json").read_bytes())
        assert task["start_is_measured_rest"] and task["prompt"] == roster["canonical_prompt"]
        canonical_row = next(r for r in rows if r["trial_id"] == "pace+0_start_0")
        assert abs(canonical_row["authored_duration_s"] - manifest["metadata"]["reference_duration_s"]) < 1e-9
        durations[zoo_id] = manifest["metadata"]["reference_duration_s"]
        media = CAMPAIGN / zoo_id / "canonical" / "media"
        media_manifest = json.loads((media / "manifest.json").read_bytes())
        for name, info in media_manifest["files"].items():
            assert sha256(media / name) == info["sha256"], name
        assert media_manifest["metadata"]["source_bundle_sha256"] == entry["canonical_sha256"]
        assert media_manifest["metadata"]["full_episode"] and media_manifest["metadata"]["preview_is_summary"]
        frames = json.loads((media / "frames.json").read_bytes())
        assert len(frames) == media_manifest["metadata"]["frame_count"]
        assert abs(len(frames) / media_manifest["metadata"]["fps"] - media_manifest["metadata"]["playback_duration_s"]) < 1e-9
        # Real-time playback: the frame count is the ceiling of duration times
        # fps plus the final held state, so playback never differs from the
        # simulated duration by more than two frame periods.
        assert abs(media_manifest["metadata"]["playback_duration_s"] - media_manifest["metadata"]["simulation_duration_s"]) <= 2.0 / media_manifest["metadata"]["fps"]
        if args.replay:
            from rigby_general.evidence.capture import replay_bundle

            assert replay_bundle(physical, entry["canonical_sha256"])["agrees"], zoo_id
    assert seen == set(BASELINE_DURATIONS) == set(zoo)
    assert totals == validation["totals"], (totals, validation["totals"])
    assert summary["all_bodies_meet_19_of_20"] == all(b["meets_19_of_20"] for b in summary["bodies"])
    assert summary["all_canonical_success"]

    # The two pilot runs are retained with their per-trial records; each was
    # run on the same registered roster at an earlier commit, and each failed
    # exactly the trials the report says it failed.
    for name, expected_commit, expected_failures in validation["pilots"]:
        pilot = json.loads((ROOT / name / "summary.json").read_bytes())
        assert pilot["roster_sha256"] == registration["roster_sha256"]
        assert pilot["provenance"]["commit"] == expected_commit
        failed = {}
        for entry in pilot["bodies"]:
            rows = json.loads((ROOT / name / entry["zoo_id"] / "trials.json").read_bytes())["feasible_trials"]
            assert len(rows) == 20 and sum(1 for r in rows if r["accepted"]) == entry["feasible_successes"]
            lost = sorted(r["trial_id"] for r in rows if not r["accepted"])
            if lost:
                failed[entry["zoo_id"]] = lost
            assert entry["canonical_outcome"] == "success"
        assert failed == {k: sorted(v) for k, v in expected_failures.items()}, (name, failed)

    for zoo_id, duration in durations.items():
        baseline = BASELINE_DURATIONS[zoo_id]
        if zoo_id in REFUSED_BASELINE:
            assert duration < baseline, (zoo_id, duration, baseline)
        else:
            assert abs(duration - baseline) < 1e-6, (zoo_id, duration, baseline)
    assert durations == validation["canonical_durations_s"]

    index = json.loads((D05 / "index.json").read_bytes())
    for name, digest in index["files"].items():
        assert sha256(D05 / name) == digest, name
    assert index["six_body"]["frames"] >= max(b["frames"] for b in index["six_body"]["bodies"])
    assert index["dual_arm"]["before"]["outcome"] == "runtime_failure"
    assert index["dual_arm"]["after"]["outcome"] == "success"
    assert index["dual_arm"]["before_matches_frozen_fixture"] is True
    assert {v["code"] for v in index["dual_arm"]["before"]["violations"]} >= {"self_collision"}
    assert index["dual_arm"]["after"]["unexpected_contacts"] == []
    for label in ("before", "after"):
        physical = D05 / "dual-arm" / label / "physical"
        manifest = json.loads((physical / "manifest.json").read_bytes())
        assert sha256(physical / "manifest.json") == index["dual_arm"][label]["physical_sha256"]
        for name, info in manifest["files"].items():
            assert sha256(physical / name) == info["sha256"], name
        if args.replay:
            from rigby_core.simulation.recording import PhysicsRecord, replay_physics
            import mujoco

            model = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
            record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
            assert replay_physics(model, record)["agrees"], label
    print(json.dumps({"verified": True, "totals": totals, "canonical_durations_s": durations,
                      "dual_arm_before_after": [index["dual_arm"]["before"]["outcome"], index["dual_arm"]["after"]["outcome"]],
                      "replayed": args.replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
