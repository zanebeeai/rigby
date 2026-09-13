"""Score the six conditionals on the registered episodes, engineer the negatives, check for leakage.

Three parts, all from sealed records and declared sensors:

- Labeled evaluation. Every registered episode is observed under every
  sensor configuration; at every query instant each conditional is decided
  and the oracle labels the same instant from the full state. The rows
  keep every verdict, decided or abstained, against its label: agreement,
  false success (pass against a false label), misses and abstentions per
  predicate and configuration, and how many episodes each predicate was
  decided on.
- Engineered negatives. From instants the primary configuration decided
  pass in agreement with the label, the evidence is degraded four ways --
  a screen placed between the camera and the fixtures, the required
  sensor frozen before the window, the required sensor absent, the
  required sensor too sparse for the window -- and from the instants a
  hold ended, released or lost, a frozen sensor that still shows the hold.
  Each case records the
  verdict, which must not be pass, and what a monitor that trusted its
  last frame would have said.
- Leakage. Every query is also decided under the configuration that
  carries a privileged oracle sensor; the verdict must be identical to the
  one without it, in decision, reason and detail.

    python any-robot/scripts/g09_conditionals_campaign.py --out docs/results/g09-conditionals
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.skills import Decision, EvidenceKind, TemporalWindowV1, evaluate

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import CONFIGURATIONS, Episode, EvidenceStreams, OracleLabeler, configuration, conditionals_for, decide, load_policy, policy_digest
from rigby_general.sensing.corrupt import absent, occluded_by_screen, sensors_of, sparse, stale
from rigby_general.sensing.sensors import CAMERAS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g09_calibrate as calibrate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / "assets/general/research-protocols/g09-conditionals-v1"
PREDICATES = ("reachable", "opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear")
PRIMARY = "front_contact"
ORACLE_CONFIGURATION = "front_contact_with_oracle"
QUERY_STEP_S = 1.0
REACH_QUERIES = 3
NEGATIVES_PER_KIND = 5
SCREEN_CENTRE = (-0.08, 1.0, 0.41)
SCREEN_HALF = (0.3, 0.005, 0.3)
"""A 60 by 60 centimetre screen halfway between the front camera and the fixtures."""

REQUIRED_KIND = {
    "reachable": EvidenceKind.OBJECT_POSE, "moving_with_robot": EvidenceKind.OBJECT_POSE, "stably_placed": EvidenceKind.OBJECT_POSE, "area_clear": EvidenceKind.OBJECT_POSE,
    "opposition_established": EvidenceKind.CONTACT_FORCE, "held": EvidenceKind.CONTACT_FORCE,
}


def load_registration() -> tuple[dict, dict, dict]:
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name in ("corpus.json", "policy.json"):
        if hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() != registration["files"][name]:
            raise SystemExit(f"{name} does not match its registration; refusing to run")
    for name in ("goal.json", "environment.json"):
        if hashlib.sha256((calibrate.G06 / name).read_bytes()).hexdigest() != registration["files"][f"g06/{name}"]:
            raise SystemExit(f"g06/{name} does not match the registration; refusing to run")
    return registration, json.loads((PROTOCOL / "corpus.json").read_bytes()), load_policy(PROTOCOL / "policy.json")


def verdict_row(verdict) -> dict:
    return {"decision": verdict.decision.value, "reason": verdict.reason, "sensors": list(verdict.sensors_used), "fallback": verdict.fallback,
            "detail": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in verdict.detail.items()}}


def naive_verdict(conditional, streams: EvidenceStreams, now_s: float):
    """What a monitor that trusts its newest frame, however old, would say:
    the window is moved back to end at the newest valid sample of each
    required kind, and no age limit applies."""

    newest = None
    for requirement in conditional.evidence:
        if not requirement.required:
            continue
        entity = conditional.entities[requirement.role]
        for stream in streams.streams.values():
            if stream.sensor.kind is not requirement.kind or stream.sensor.oracle or (stream.sensor.entity and stream.sensor.entity != entity):
                continue
            valid = [float(t) for t, q in zip(stream.times, stream.quality) if q.value == "valid" and t <= now_s + 1e-9]
            if valid:
                newest = valid[-1] if newest is None else min(newest, valid[-1])
    if newest is None:
        return None
    relaxed = conditional.model_copy(update={"window": TemporalWindowV1(duration_s=conditional.window.duration_s, max_age_s=1e9),
                                             "evidence": tuple(e.model_copy(update={"min_samples": 1}) for e in conditional.evidence)})
    sensors = tuple(s.model_copy(update={"max_age_s": 1e9}) for s in streams.configuration.sensors)
    relaxed_configuration = streams.configuration.model_copy(update={"sensors": sensors})
    return evaluate(relaxed, relaxed_configuration, streams.samples(newest - conditional.window.duration_s, newest), newest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="score only the first N registered episodes (engineering smoke; never a scored run)")
    args = parser.parse_args()
    registration, corpus, policy = load_registration()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a campaign writes only to a fresh destination")
    args.out.mkdir(parents=True)
    goal = calibrate.registered_goal()
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "registration_sha256": registration["registration_sha256"], "policy_sha256": corpus["policy_sha256"], "policy_digest": policy_digest(policy),
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "limit": args.limit, "scored": args.limit is None, "generation_calls": 0}
    (args.out / "provenance.json").write_bytes(json_bytes(provenance))
    roster = [r for r in corpus["episodes"] if r["kind"] == "evaluation"]
    if args.limit is not None:
        roster = roster[: args.limit]

    rows: list[dict] = []
    leakage: list[dict] = []
    negatives: list[dict] = []
    positives_for_negatives: dict[str, list[tuple]] = {name: [] for name in PREDICATES}
    hold_losses: list[tuple] = []
    started = time.perf_counter()
    for number, entry in enumerate(roster):
        bundle = Path(entry["tree"]) / entry["relative"]
        episode = Episode.load(bundle, default_goal=goal)
        if episode.trace_sha256 != entry["trace_sha256"]:
            raise SystemExit(f"{bundle}: trace hash does not match the registered roster")
        wall = time.perf_counter()
        streams = {name: EvidenceStreams.build(episode, configuration(name, episode.effectors)) for name in CONFIGURATIONS}
        labeler = OracleLabeler(episode)
        if episode.refusal:
            times = np.array([episode.start_s + 0.1])
        else:
            times = np.arange(episode.start_s + 0.5, episode.end_s - 1e-6, QUERY_STEP_S)
        reach_indices = sorted({0, len(times) // 2, len(times) - 1})[:REACH_QUERIES]
        for effector in episode.effectors:
            conditionals = conditionals_for(episode, effector, policy)
            held_track = []
            for k, t in enumerate(times):
                now = float(t)
                for name in PREDICATES:
                    if name == "reachable" and k not in reach_indices:
                        continue
                    conditional = conditionals[name]
                    label = labeler.label(name, effector, now, conditional.window.duration_s)
                    verdicts = {config: decide(conditional, s, now) for config, s in streams.items()}
                    primary, oracle = verdicts[PRIMARY], verdicts[ORACLE_CONFIGURATION]
                    if (primary.decision, primary.reason, primary.detail, primary.sensors_used) != (oracle.decision, oracle.reason, oracle.detail, oracle.sensors_used):
                        leakage.append({"episode_id": episode.episode_id, "trace_sha256": episode.trace_sha256, "effector": effector.chain_id, "predicate": name, "time_s": now, "without": verdict_row(primary), "with_oracle": verdict_row(oracle)})
                    rows.append({"episode_id": episode.episode_id, "trace_sha256": episode.trace_sha256, "tree": Path(entry["tree"]).name, "zoo_id": episode.zoo_id,
                                 "source_goal": episode.metadata.get("goal"), "outcome": episode.metadata.get("outcome"),
                                 "effector": effector.chain_id, "predicate": name, "time_s": now, "label": bool(label.value),
                                 "label_detail": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in label.detail.items()},
                                 "verdicts": {config: [v.decision.value, v.reason] for config, v in verdicts.items()}})
                    if primary.decision is Decision.PASS and label.value:
                        positives_for_negatives[name].append((episode, effector, conditional, streams[PRIMARY], now))
                    if name == "held":
                        held_track.append((now, bool(label.value), primary.decision))
            # A natural lost hold: held was true and decided pass, then false.
            for (t1, l1, d1), (t2, l2, _) in zip(held_track, held_track[1:]):
                if l1 and d1 is Decision.PASS and not l2:
                    hold_losses.append((episode, effector, conditionals["held"], streams[PRIMARY], t1, t2))
        print(json.dumps({"episode": episode.episode_id, "n": number + 1, "of": len(roster), "queries": int(len(times)), "wall": round(time.perf_counter() - wall, 1)}), flush=True)

    # -- engineered negatives ------------------------------------------------------
    rng = np.random.default_rng(9)
    for name in PREDICATES:
        candidates = positives_for_negatives[name]
        if not candidates:
            continue
        chosen = [candidates[i] for i in sorted(rng.choice(len(candidates), size=min(NEGATIVES_PER_KIND, len(candidates)), replace=False))]
        kind = REQUIRED_KIND[name]
        for episode, effector, conditional, streams, now in chosen:
            ids = {s for s in sensors_of(streams, kind) if not streams.streams[s].sensor.entity or streams.streams[s].sensor.entity == effector.chain_id}
            variants = []
            if kind is EvidenceKind.OBJECT_POSE:
                variants.append(("occlusion", occluded_by_screen(streams, "camera:front", SCREEN_CENTRE, SCREEN_HALF), f"a screen {2 * SCREEN_HALF[0]:.1f} m square between the front camera and the fixtures"))
            frozen_at = now - conditional.window.max_age_s - 0.25
            variants.append(("stale", stale(streams, ids, frozen_at), f"{sorted(ids)} frozen at {frozen_at:.3f} s, {now - frozen_at:.2f} s before the decision"))
            variants.append(("absent", absent(streams, kind), f"no sensor of kind {kind.value}"))
            minimum = conditional.evidence[0].min_samples
            if minimum >= 2:
                # The window is closed at both ends, so a sensor at exactly
                # (minimum - 1) / duration can land minimum samples in it.
                rate = max(0.5, (minimum - 1.5) / conditional.window.duration_s)
                variants.append(("sparse", sparse(streams, ids, rate), f"{sorted(ids)} thinned to {rate:g} Hz, at most {minimum - 1} samples in {conditional.window.duration_s:g} s"))
            for kind_name, corrupted, description in variants:
                verdict = decide(conditional, corrupted, now)
                naive = naive_verdict(conditional, corrupted, now)
                negatives.append({"predicate": name, "kind": kind_name, "episode_id": episode.episode_id, "trace_sha256": episode.trace_sha256, "effector": effector.chain_id, "time_s": now, "label": True,
                                  "description": description, "configuration": corrupted.configuration.configuration_id, "uncorrupted": "pass",
                                  "verdict": verdict_row(verdict), "naive": None if naive is None else verdict_row(naive),
                                  "false_success": verdict.decision is Decision.PASS, "naive_would_pass": naive is not None and naive.decision is Decision.PASS})
    for episode, effector, conditional, streams, t1, t2 in hold_losses[:NEGATIVES_PER_KIND * 2]:
        ids = {f"contact:{effector.chain_id}"}
        corrupted = stale(streams, ids, t1)
        verdict = decide(conditional, corrupted, t2)
        naive = naive_verdict(conditional, corrupted, t2)
        negatives.append({"predicate": "held", "kind": "stale_after_lost_hold", "episode_id": episode.episode_id, "trace_sha256": episode.trace_sha256, "effector": effector.chain_id, "time_s": t2, "label": False,
                          "description": f"the contact sensor frozen at {t1:.2f} s while the hold was real; by {t2:.2f} s the hold had ended (released or lost)", "configuration": corrupted.configuration.configuration_id,
                          "uncorrupted": decide(conditional, streams, t2).decision.value, "verdict": verdict_row(verdict), "naive": None if naive is None else verdict_row(naive),
                          "false_success": verdict.decision is Decision.PASS, "naive_would_pass": naive is not None and naive.decision is Decision.PASS})

    # -- tallies -----------------------------------------------------------------
    per = {}
    for row in rows:
        for config, (decision, reason) in row["verdicts"].items():
            cell = per.setdefault(row["predicate"], {}).setdefault(config, {"queries": 0, "episodes": set(), "decided_episodes": set(), "positives": 0, "negatives": 0,
                                                                            "pass_true": 0, "pass_false": 0, "fail_true": 0, "fail_false": 0, "unknown_true": 0, "unknown_false": 0, "unknown_reasons": {}})
            cell["queries"] += 1
            cell["episodes"].add(row["trace_sha256"])
            cell["positives" if row["label"] else "negatives"] += 1
            key = f"{decision}_{'true' if row['label'] else 'false'}"
            cell[key] += 1
            if decision != "unknown":
                cell["decided_episodes"].add(row["trace_sha256"])
            else:
                head = reason.split(":")[0]
                cell["unknown_reasons"][head] = cell["unknown_reasons"].get(head, 0) + 1
    summary_cells = {}
    for predicate, configs in per.items():
        summary_cells[predicate] = {}
        for config, cell in configs.items():
            decided = cell["pass_true"] + cell["pass_false"] + cell["fail_true"] + cell["fail_false"]
            tpr = cell["pass_true"] / (cell["pass_true"] + cell["fail_true"]) if cell["pass_true"] + cell["fail_true"] else None
            tnr = cell["fail_false"] / (cell["fail_false"] + cell["pass_false"]) if cell["fail_false"] + cell["pass_false"] else None
            summary_cells[predicate][config] = {**{k: v for k, v in cell.items() if k not in ("episodes", "decided_episodes")},
                                                "episodes": len(cell["episodes"]), "decided_episodes": len(cell["decided_episodes"]), "decided": decided,
                                                "agreement": (cell["pass_true"] + cell["fail_false"]) / decided if decided else None,
                                                "false_success": cell["pass_false"], "true_positive_rate": tpr, "true_negative_rate": tnr,
                                                "abstained_fraction": (cell["unknown_true"] + cell["unknown_false"]) / cell["queries"]}
    summary = {"goal": "G09", "provenance": provenance, "registration_sha256": registration["registration_sha256"], "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "episodes_scored": len(roster), "queries": len(rows), "query_step_s": QUERY_STEP_S, "primary_configuration": PRIMARY, "configurations": list(CONFIGURATIONS),
               "per_predicate": summary_cells, "leakage_mismatches": len(leakage),
               "negatives": {"count": len(negatives), "false_success": sum(1 for n in negatives if n["false_success"]), "naive_would_pass": sum(1 for n in negatives if n["naive_would_pass"]),
                             "per_kind": {k: sum(1 for n in negatives if n["kind"] == k) for k in sorted({n["kind"] for n in negatives})},
                             "verdicts": {k: sum(1 for n in negatives if n["verdict"]["decision"] == k) for k in ("pass", "fail", "unknown")}},
               "wall_seconds": time.perf_counter() - started}
    (args.out / "summary.json").write_bytes(json_bytes(summary))
    (args.out / "negatives.json").write_bytes(json_bytes(negatives))
    (args.out / "leakage.json").write_bytes(json_bytes(leakage))
    with gzip.open(args.out / "queries.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"queries": len(rows), "negatives": summary["negatives"], "leakage": len(leakage), "wall": round(summary["wall_seconds"], 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
