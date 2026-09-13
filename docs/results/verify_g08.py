"""Verify the G08 transition evidence: the registered corpus, the campaign, the bundles, the media, D08.

Recomputes every claim the report makes from the committed files: the corpus
against its registration and the G06 fixture it composes against; the
per-case rows against the corpus (every case run once, in order); the
feasible tally and the injected handling re-derived from the rows; every
committed bundle's digests (and its physics replay with --replay); every
rendered episode's frame map; the D08 pairs against their index.

    python docs/results/verify_g08.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g08-transitions-v1"
G06 = REPO / "any-robot/assets/general/research-protocols/g06-transfer-v1"
CAMPAIGN = ROOT / "g08-campaign"
LOCAL = REPO / "any-robot/results/g08-campaign"
D08 = ROOT / "g08-d08"
FEASIBLE_TARGET = 95


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_bundle(group: str, case_id: str) -> Path | None:
    """A campaign case's replayable bundle lives in the local results tree
    (untracked); when it is present it is checked in full, and when it is
    not, its digest in the row and its rendered media stand for it."""

    physical = LOCAL / group / case_id / "physical"
    return physical if (physical / "manifest.json").is_file() else None


def check_bundle(physical: Path, expected: str, *, replay: bool) -> dict:
    manifest = json.loads((physical / "manifest.json").read_bytes())
    assert sha256(physical / "manifest.json") == expected, physical
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


def check_media(media: Path, source: str) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    assert manifest["metadata"]["source_bundle_sha256"] == source
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"]
    if manifest["metadata"].get("refusal_slate"):
        assert manifest["metadata"]["playback_duration_s"] > 0
    else:
        assert abs(manifest["metadata"]["playback_duration_s"] - manifest["metadata"]["simulation_duration_s"]) <= 2.0 / manifest["metadata"]["fps"]
    for name in ("episode.mp4", "preview.gif", "frames.json"):
        assert (media / name).is_file(), (media, name)
    return manifest


def handled(row: dict) -> tuple[bool, str]:
    if row["kind"] == "feasible":
        return row["composed_success"], "composed_success" if row["composed_success"] else "composed_failure"
    if row["rejected"] and row["second"] is None:
        return True, "rejected"
    if row["repairs"] and row["re_verified"] and row["verdicts"] and row["verdicts"][-1]["compatible"]:
        return True, "repaired_then_reverified"
    return False, "not_handled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g08-validation.json").read_bytes())

    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    assert sha256(PROTOCOL / "corpus.json") == registration["files"]["corpus.json"]
    assert sha256(G06 / "environment.json") == registration["files"]["g06/environment.json"] and sha256(G06 / "goal.json") == registration["files"]["g06/goal.json"]
    files = json.dumps(registration["files"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert hashlib.sha256(files.encode("utf-8")).hexdigest() == registration["registration_sha256"] == validation["registration_sha256"]
    corpus = json.loads((PROTOCOL / "corpus.json").read_bytes())
    assert corpus["scored_runs_started"] is False and corpus["generation_calls"] == 0
    feasible_cases, injected_cases = corpus["feasible_cases"], corpus["injected_cases"]
    assert len(feasible_cases) == 100 and len(injected_cases) >= 40
    assert {c["zoo_id"] for c in feasible_cases} == set(corpus["bodies"]) == {"zoo_dual_arm", "zoo_jaw_arm", "zoo_long_arm"}
    assert any(c["case_id"].startswith("zoo_dual_arm-reviewed_dual_arm_failure") for c in injected_cases)
    reviewed = next(c for c in injected_cases if c["case_id"].startswith("zoo_dual_arm-reviewed_dual_arm_failure"))
    assert sha256(REPO / reviewed["fixture"]) == reviewed["fixture_sha256"]
    for case in feasible_cases:
        assert case["witness"]["straight_joint_path_clear"] and case["witness"]["transfer_path_seed"]

    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    assert summary["registration_sha256"] == registration["registration_sha256"]
    assert summary["provenance"]["commit"] == validation["campaign_commit"] and summary["provenance"]["scored"] is True and summary["provenance"]["generation_calls"] == 0
    rows = json.loads((CAMPAIGN / "trials.json").read_bytes())
    assert [r["case_id"] for r in rows["feasible"]] == [c["case_id"] for c in feasible_cases]
    assert [r["case_id"] for r in rows["injected"]] == [c["case_id"] for c in injected_cases]
    successes, per_body, taxonomy = 0, {}, {}
    for row in rows["feasible"]:
        ok, how = handled(row)
        assert ok == row["handled_correctly"]
        per_body.setdefault(row["zoo_id"], [0, 0])
        per_body[row["zoo_id"]][1] += 1
        if ok:
            successes += 1
            per_body[row["zoo_id"]][0] += 1
            assert row["first"]["certified"] and row["second"]["certified"] and not row["gate_violations"] and row["verdicts"][-1]["compatible"]
        else:
            code = (row["second"]["gate"] if row["second"] else None) or row["rejection"] or (row["gate_violations"][0]["code"] if row["gate_violations"] else None) or row["first"]["gate"] or "unknown"
            taxonomy[code] = taxonomy.get(code, 0) + 1
        assert row["trace_file_sha256"] or not row["first"]["executed"]
        if "bundle" in row:
            check_media(CAMPAIGN / "feasible" / row["case_id"] / "media", row["bundle"]["sha256"])
            physical = local_bundle("feasible", row["case_id"])
            if physical is not None:
                check_bundle(physical, row["bundle"]["sha256"], replay=args.replay)
    assert successes == summary["feasible"]["successes"] == validation["feasible_successes"]
    assert successes >= FEASIBLE_TARGET, f"{successes} of 100 feasible compositions succeeded"
    assert {k: v[0] for k, v in per_body.items()} == {k: v["successes"] for k, v in summary["feasible"]["per_body"].items()}
    assert taxonomy == validation["feasible_failure_taxonomy"]
    for body, (ok, count) in per_body.items():
        assert count >= 33 and ok >= 1, body

    injected_ok, per_kind, kinds_seen = 0, {}, {}
    for row in rows["injected"]:
        ok, how = handled(row)
        assert ok == row["handled_correctly"], row["case_id"]
        assert ok, f"{row['case_id']}: {row['handling']}"
        injected_ok += 1
        per_kind[row["kind"]] = per_kind.get(row["kind"], 0) + 1
        kinds_seen.setdefault(row["kind"], set()).add(how)
        expect = next(c["expect"] for c in injected_cases if c["case_id"] == row["case_id"])
        if expect == "rejected":
            assert how == "rejected" and row["second"] is None and row["repairs"] == [], row["case_id"]
        else:
            assert how == "repaired_then_reverified" and row["repairs"] and row["verdicts"][-1]["compatible"], row["case_id"]
            assert row["second"] is not None, "the second skill ran only after the boundary was verified again"
        assert "bundle" in row
        check_media(CAMPAIGN / "injected" / row["case_id"] / "media", row["bundle"]["sha256"])
        physical = local_bundle("injected", row["case_id"])
        if physical is not None:
            check_bundle(physical, row["bundle"]["sha256"], replay=args.replay)
            outcome = json.loads((physical / "outcome.json").read_bytes())
            assert outcome["handled_correctly"] and outcome["case_id"] == row["case_id"]
    assert injected_ok == len(injected_cases) == summary["injected"]["handled_correctly"] == validation["injected_handled"]
    assert per_kind == validation["injected_per_kind"]
    # A contact-mode or ownership mismatch is never bridged by a path.
    for row in rows["injected"]:
        if row["kind"] in ("holding_into_free", "free_into_holding", "resource_conflict"):
            assert row["rejected"] and row["repairs"] == [] and row["second"] is None, row["case_id"]
    # The reviewed dual-arm failure was repaired and verified again before the transfer ran.
    reviewed_row = next(r for r in rows["injected"] if r["case_id"] == reviewed["case_id"])
    assert reviewed_row["repairs"][0]["kind"] == "joint_move" and reviewed_row["re_verified"] and reviewed_row["second"] is not None

    # The retained pilot: the first scored pass, run whole on the corpus as
    # first registered, before the witness checked the world. Its corpus and
    # registration are kept beside its records; it failed what the report
    # says it failed, and every failure is explained by a pose the refined
    # witness now rejects.
    for name, expected_commit, expected in validation["pilots"]:
        pilot_dir = ROOT / name
        pilot = json.loads((pilot_dir / "summary.json").read_bytes())
        pilot_registration = json.loads((pilot_dir / "protocol" / "registration.json").read_bytes())
        assert sha256(pilot_dir / "protocol" / "corpus.json") == pilot_registration["files"]["corpus.json"]
        assert pilot["registration_sha256"] == pilot_registration["registration_sha256"]
        assert pilot["provenance"]["commit"] == expected_commit and pilot["provenance"]["scored"] is True
        pilot_rows = json.loads((pilot_dir / "trials.json").read_bytes())
        tallies = {"feasible_successes": sum(1 for r in pilot_rows["feasible"] if r["handled_correctly"]), "feasible_count": len(pilot_rows["feasible"]),
                   "injected_handled": sum(1 for r in pilot_rows["injected"] if r["handled_correctly"]), "injected_count": len(pilot_rows["injected"]),
                   "failed_cases": sorted(r["case_id"] for r in pilot_rows["feasible"] if not r["handled_correctly"])}
        assert tallies == expected, (name, tallies)
        for row in pilot_rows["feasible"] + pilot_rows["injected"]:
            if "bundle" in row:
                group = "feasible" if row["kind"] == "feasible" else "injected"
                check_media(pilot_dir / group / row["case_id"] / "media", row["bundle"]["sha256"])

    index = json.loads((D08 / "index.json").read_bytes())
    assert index["registration_sha256"] == registration["registration_sha256"]
    for name, digest in index["files"].items():
        assert sha256(D08 / name) == digest, name
    names = {p["name"] for p in index["pairs"]}
    assert names == {"joint-limit-approach", "carry-to-place", "contact-mode-change"}
    for pair in index["pairs"]:
        before, after = pair["before"], pair["after"]
        assert not before["validated"] and after["validated"]
        for label in ("before", "after"):
            check_bundle(D08 / pair["name"] / label / "physical", pair[label]["physical_sha256"], replay=args.replay)
            check_media(D08 / pair["name"] / label / "media", pair[label]["physical_sha256"])
        if pair["name"] == "joint-limit-approach":
            # Without the check the transfer is attempted from the wrist at
            # its limit and does not certify; with it the wrist is moved
            # inside the margin, verified, and the transfer certifies.
            assert not before["composed_success"] and before["repairs"] == []
            assert after["repairs"] and after["repairs"][0]["kind"] == "joint_move" and after["composed_success"]
        if pair["name"] == "carry-to-place":
            assert before["verdicts"][0]["violations"] and all(v["code"] == "velocity_too_high" for v in before["verdicts"][0]["violations"])
            assert before["repairs"] == [], "without the check the placement begins from the moving arm"
            assert after["repairs"] and after["repairs"][0]["kind"] == "settle" and after["composed_success"]
            assert after["boundary"]["contact_mode"] == "holding" and after["repairs"][0]["compatible_after"]
        if pair["name"] == "contact-mode-change":
            assert after["rejected"] and after["rejection"] == "contact_mode_mismatch" and after["second"] is None
            assert before["second"] is not None and before["boundary"]["contact_mode"] == "holding", "without the check the return began with the cube in hand"
            assert before["second"]["certified"] and before["composed_success"], "and the return's own certificate calls the composition a success"
            assert before["final_boundary"]["held"] == {}, "while the cube left the hand without any placement skill having run"
            assert pair["inserted"]["acquire_to_place_success"] and pair["inserted"]["place_to_return_success"]
            check_bundle(D08 / pair["name"] / "inserted" / "physical", pair["inserted"]["physical_sha256"], replay=args.replay)
        assert pair["side_by_side"]["frames"] >= max(before["frames"], after["frames"])
    assert validation["d08"] == {p["name"]: {"before": p["before"]["outcome"], "after": p["after"]["outcome"]} for p in index["pairs"]}
    print(json.dumps({"verified": True, "feasible_successes": successes, "per_body": per_body, "feasible_failure_taxonomy": taxonomy, "injected_handled": injected_ok,
                      "injected_per_kind": per_kind, "replayed": args.replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
