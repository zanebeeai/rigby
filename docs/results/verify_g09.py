"""Verify the G09 conditionals evidence: the registration, the scored rows, the negatives, the leakage check, D09.

Recomputes every claim the report makes from the committed files: the
policy and corpus against their registration and the G06 fixture; the
per-query rows re-tallied into the summary's confusion counts, agreement,
false-success counts and decided-episode counts; the engineered negatives
(at least sixty, none decided pass); the leakage check (no query whose
verdict changed with a privileged sensor present); the D09 overlays (a
media bundle per case, its frame map, real-time playback, the verdict
record with one entry per frame); and, with --recompute, a sample of
the rows decided again from the local bundles where they are present.

    python docs/results/verify_g09.py [--recompute N]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g09-conditionals-v1"
G06 = REPO / "any-robot/assets/general/research-protocols/g06-transfer-v1"
CAMPAIGN = ROOT / "g09-conditionals"
D09 = ROOT / "g09-d09"
PREDICATES = ("reachable", "opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear")
MIN_EPISODES_PER_PREDICATE = 20
MIN_NEGATIVES = 60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(media: Path, *, overlay: bool) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    meta = manifest["metadata"]
    assert len(frames) == meta["frame_count"]
    assert abs(meta["playback_duration_s"] - meta["simulation_duration_s"]) <= 2.0 / meta["fps"], media
    for name in ("episode.mp4", "preview.gif", "frames.json"):
        assert (media / name).is_file(), (media, name)
    if overlay:
        verdicts = json.loads((media / "verdicts.json").read_bytes())
        assert len(verdicts) == meta["frame_count"]
        for entry, frame in zip(verdicts, frames):
            assert entry["frame"] == frame["frame"] and entry["simulation_time_s"] == frame["simulation_time_s"]
            for name in meta["overlay"]["configurations"]:
                assert set(entry["configurations"][name]) == set(PREDICATES)
                for verdict in entry["configurations"][name].values():
                    assert verdict["decision"] in ("pass", "fail", "unknown")
                    assert (verdict["decision"] == "unknown") == bool(verdict["reason"])
            assert set(entry["labels"]) == set(PREDICATES)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", type=int, default=0, help="decide this many registered queries again from local bundles")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g09-validation.json").read_bytes())

    # -- registration ---------------------------------------------------------------
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    assert sha256(PROTOCOL / "corpus.json") == registration["files"]["corpus.json"]
    assert sha256(PROTOCOL / "policy.json") == registration["files"]["policy.json"]
    assert sha256(G06 / "goal.json") == registration["files"]["g06/goal.json"] and sha256(G06 / "environment.json") == registration["files"]["g06/environment.json"]
    listing = json.dumps(registration["files"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert hashlib.sha256(listing.encode("utf-8")).hexdigest() == registration["registration_sha256"] == validation["registration_sha256"]
    corpus = json.loads((PROTOCOL / "corpus.json").read_bytes())
    policy = json.loads((PROTOCOL / "policy.json").read_bytes())
    assert corpus["policy_sha256"] == registration["files"]["policy.json"] and corpus["generation_calls"] == 0
    evaluation = [e for e in corpus["episodes"] if e["kind"] == "evaluation"]
    excluded = {e["trace_sha256"] for e in corpus["episodes"] if e["kind"] == "calibration_excluded"}
    assert {e["trace_sha256"] for e in policy["calibration"]["episodes"]} <= excluded, "every calibration episode is excluded from the evaluation"
    assert not ({e["trace_sha256"] for e in evaluation} & {e["trace_sha256"] for e in policy["calibration"]["episodes"]})
    assert len(evaluation) == validation["evaluation_episodes"]
    for key in ("contact_force_n", "held", "moving_with_robot", "stably_placed", "reachable"):
        assert any(k.endswith(key) for k in policy["calibration"]["results"]), key
    assert set(corpus["configurations"]) == set(validation["configurations"])
    for name, configuration in corpus["configurations"].items():
        oracle = [s for s in configuration["sensors"] if s["oracle"]]
        assert bool(oracle) == name.endswith("with_oracle"), name

    # -- the scored rows -------------------------------------------------------------
    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    assert summary["registration_sha256"] == registration["registration_sha256"]
    assert summary["provenance"]["scored"] is True and summary["provenance"]["generation_calls"] == 0 and summary["provenance"]["commit"] == validation["campaign_commit"]
    assert summary["episodes_scored"] == len(evaluation)
    rows = []
    with gzip.open(CAMPAIGN / "queries.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    assert len(rows) == summary["queries"]
    registered = {e["trace_sha256"] for e in evaluation}
    assert {r["trace_sha256"] for r in rows} == registered, "every registered episode has rows and nothing else does"
    tally: dict[str, dict[str, dict]] = {}
    for row in rows:
        assert row["predicate"] in PREDICATES and isinstance(row["label"], bool)
        for config, (decision, reason) in row["verdicts"].items():
            cell = tally.setdefault(row["predicate"], {}).setdefault(config, {"queries": 0, "episodes": set(), "decided": set(), "pass_true": 0, "pass_false": 0, "fail_true": 0, "fail_false": 0, "unknown_true": 0, "unknown_false": 0})
            cell["queries"] += 1
            cell["episodes"].add(row["trace_sha256"])
            cell[f"{decision}_{'true' if row['label'] else 'false'}"] += 1
            if decision != "unknown":
                cell["decided"].add(row["trace_sha256"])
            assert (decision == "unknown") == bool(reason), row
        primary, oracle = row["verdicts"][summary["primary_configuration"]], row["verdicts"][summary["primary_configuration"] + "_with_oracle"]
        assert primary == oracle, ("a verdict changed with the oracle present", row)
    for predicate in PREDICATES:
        for config, cell in tally[predicate].items():
            claimed = summary["per_predicate"][predicate][config]
            for key in ("queries", "pass_true", "pass_false", "fail_true", "fail_false", "unknown_true", "unknown_false"):
                assert claimed[key] == cell[key], (predicate, config, key)
            assert claimed["episodes"] == len(cell["episodes"]) and claimed["decided_episodes"] == len(cell["decided"])
            assert claimed["false_success"] == cell["pass_false"]
        primary = summary["per_predicate"][predicate][summary["primary_configuration"]]
        assert primary["decided_episodes"] >= MIN_EPISODES_PER_PREDICATE, (predicate, primary["decided_episodes"])
        assert primary["positives"] > 0 and primary["negatives"] > 0, predicate
    assert summary["leakage_mismatches"] == 0
    leakage = json.loads((CAMPAIGN / "leakage.json").read_bytes())
    assert leakage == []
    assert validation["per_predicate_primary"] == {p: {k: summary["per_predicate"][p][summary["primary_configuration"]][k] for k in ("decided_episodes", "agreement", "false_success")} for p in PREDICATES}

    # -- the engineered negatives -----------------------------------------------------
    negatives = json.loads((CAMPAIGN / "negatives.json").read_bytes())
    assert len(negatives) >= MIN_NEGATIVES and len(negatives) == summary["negatives"]["count"] == validation["negatives"]["count"]
    kinds = {n["kind"] for n in negatives}
    assert {"occlusion", "stale", "absent", "sparse", "stale_after_lost_hold"} <= kinds
    for negative in negatives:
        assert negative["verdict"]["decision"] in ("unknown", "fail"), negative
        assert negative["false_success"] is False
        if negative["verdict"]["decision"] == "unknown":
            assert negative["verdict"]["fallback"] in ("re_observe", "fail", "abstain")
    assert summary["negatives"]["false_success"] == 0 == validation["negatives"]["false_success"]
    assert summary["negatives"]["naive_would_pass"] == sum(1 for n in negatives if n["naive_would_pass"]) == validation["negatives"]["naive_would_pass"]

    # -- D09 ----------------------------------------------------------------------------
    index = json.loads((D09 / "index.json").read_bytes())
    assert index["policy_sha256"] == registration["files"]["policy.json"] and index["generation_calls"] == 0
    assert [c["name"] for c in index["cases"]] == ["visible-grasp", "hidden-slip", "occluded-placement"]
    for case in index["cases"]:
        media = check_media(D09 / case["name"] / "media", overlay=False)
        assert media["metadata"]["source_bundle_sha256"] == case["physical_sha256"]
        overlay = check_media(D09 / case["name"] / "overlay", overlay=True)
        assert overlay["metadata"]["source_media_sha256"] == sha256(D09 / case["name"] / "media" / "manifest.json")
        assert overlay["metadata"]["overlay"]["policy_id"] == policy["policy_id"]
        assert sha256(D09 / case["video"]) == overlay["files"]["episode.mp4"]["sha256"]
        assert sha256(D09 / case["preview"]) == overlay["files"]["preview.gif"]["sha256"]
        assert sha256(D09 / case["frames"]) == overlay["files"]["frames.json"]["sha256"]
        assert case["overlay"]["verdict_frames"] == overlay["metadata"]["verdict_frames"]
    expectations = validation["d09"]
    for case in index["cases"]:
        frames = case["overlay"]["verdict_frames"]
        for configuration_name, predicate, decision in expectations[case["name"]]:
            assert frames[configuration_name][predicate][decision] > 0, (case["name"], configuration_name, predicate, decision)

    # -- recompute a sample from local bundles ----------------------------------------------
    if args.recompute:
        import random
        import sys

        sys.path.insert(0, str(REPO / "any-robot/scripts"))
        import g09_calibrate as calibrate
        from rigby_general.sensing import Episode, EvidenceStreams, OracleLabeler, configuration, conditionals_for, decide

        goal = calibrate.registered_goal()
        by_id = {e["trace_sha256"]: e for e in evaluation}
        candidates = [r for r in rows if (Path(by_id[r["trace_sha256"]]["tree"]) / by_id[r["trace_sha256"]]["relative"] / "manifest.json").is_file()]
        sample = random.Random(9).sample(candidates, min(args.recompute, len(candidates)))
        cache = {}
        for row in sample:
            entry = by_id[row["trace_sha256"]]
            bundle = Path(entry["tree"]) / entry["relative"]
            if bundle not in cache:
                episode = Episode.load(bundle, default_goal=goal)
                assert episode.trace_sha256 == entry["trace_sha256"]
                cache[bundle] = (episode, EvidenceStreams.build(episode, configuration(summary["primary_configuration"], episode.effectors)), OracleLabeler(episode))
            episode, streams, labeler = cache[bundle]
            effector = next(e for e in episode.effectors if e.chain_id == row["effector"])
            conditional = conditionals_for(episode, effector, policy)[row["predicate"]]
            verdict = decide(conditional, streams, row["time_s"])
            assert [verdict.decision.value, verdict.reason] == row["verdicts"][summary["primary_configuration"]], row
            assert labeler.label(row["predicate"], effector, row["time_s"], conditional.window.duration_s).value == row["label"], row
        print(json.dumps({"recomputed": len(sample), "bundles": len(cache)}))

    print(json.dumps({"verified": True, "episodes": len(evaluation), "queries": len(rows), "negatives": len(negatives), "leakage_mismatches": 0,
                      "decided_episodes": {p: summary["per_predicate"][p][summary["primary_configuration"]]["decided_episodes"] for p in PREDICATES}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
