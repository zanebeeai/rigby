"""Calibrate the conditionals' thresholds on labeled episodes, and write the policy.

Every threshold a rule uses is either a task policy declared here with its
reason, or a value chosen on a calibration set of recorded episodes that
the scored evaluation never sees: the G06 pilot bundles, successes and
failures alike. For each calibrated quantity the same rule is decided over
a grid of candidate values against the oracle labels, and the grid point
with the best balanced accuracy is taken, and when several tie, the middle
of the tied run, so the threshold sits away from the data that chose it.
Nothing is read from a robot description. No API or model calls.

    python any-robot/scripts/g09_calibrate.py --calibration <pilot bundle tree> \
        --out any-robot/assets/general/research-protocols/g09-conditionals-v1/policy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rigby_core.skills import Decision

from rigby_general.contact.placement import PlacementGoal
from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import CONFIGURATIONS, Episode, EvidenceStreams, OracleLabeler, configuration, conditionals_for, decide
from rigby_general.sensing.sensors import CAMERAS


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
G06 = ROOT / "assets/general/research-protocols/g06-transfer-v1"
QUERY_STEP_S = 0.5

DECLARED = {
    "abstention": {"min_valid_fraction": 0.8, "why": "a window in which a fifth of a sensor's samples are occluded or missing is not a window that sensor observed"},
    "windows": {
        "reachable": {"duration_s": 0.5, "min_samples": 1, "max_age_s": 0.2, "why": "one fresh position and the calibrated shell decide reach; a fifth of a second is six camera frames"},
        "opposition_established": {"duration_s": 0.5, "min_samples": 5, "max_age_s": 0.05, "why": "the closure's own hysteresis is a tenth of a second; half a second of contact at 200 Hz is a hundred samples, five the least that can show a fraction"},
        "held": {"duration_s": 0.5, "min_samples": 5, "max_age_s": 0.05, "why": "as opposition; a hold is opposition that lasts"},
        "moving_with_robot": {"duration_s": 0.5, "min_samples": 5, "max_age_s": 0.2, "why": "half a second of travel at the carry's speed is centimetres, well above the camera's noise"},
        "stably_placed": {"duration_s": 2.0, "min_samples": 10, "max_age_s": 0.2, "why": "the registered placement dwell is two seconds; the predicate asks for the same stillness the goal does"},
        "area_clear": {"duration_s": 0.5, "min_samples": 3, "max_age_s": 0.2, "why": "the region is clear when nothing was seen in it for half a second of frames"},
    },
    "task_policy": {
        "stably_placed.max_speed_mps": {"value": None, "why": "the registered placement goal's maximum linear speed; the predicate holds the object to the goal's own stillness"},
        "position_tolerance_m": {"value": None, "why": "three standard deviations of the declared camera noise; containment and clearance are asked to that tolerance"},
        "moving_with_robot.pairing_tolerance_s": {"value": 0.05, "why": "the camera's frame period plus its latency: the nearest effector sample to a frame"},
        "opposition_established.min_fraction": {"value": 0.9, "why": "opposition on nine tenths of the window's contact samples; the closure controller tolerates a tenth of a second of lost contact and so does the predicate"},
        "held.min_fraction": {"value": 0.95, "why": "a hold tolerates less than a grasp being formed"},
    },
}


def registered_goal() -> PlacementGoal:
    g = json.loads((G06 / "goal.json").read_bytes())
    return PlacementGoal(region_minimum_m=tuple(g["region_minimum_m"]), region_maximum_m=tuple(g["region_maximum_m"]), dwell_s=g["dwell_s"],
                         maximum_linear_speed_mps=g["maximum_linear_speed_mps"], maximum_angular_speed_radps=g["maximum_angular_speed_radps"])


def discover(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        found.extend(sorted(p.parent for p in root.rglob("physical/manifest.json")))
    return found


def balanced_accuracy(pairs: list[tuple[bool, bool]]) -> float:
    positives = [v for label, v in pairs if label]
    negatives = [v for label, v in pairs if not label]
    tpr = sum(positives) / len(positives) if positives else 1.0
    tnr = sum(not v for v in negatives) / len(negatives) if negatives else 1.0
    return 0.5 * (tpr + tnr)


def with_rule(policy: dict, rule: str, **values) -> dict:
    out = json.loads(json.dumps(policy))
    if rule == "contact_force_n":
        out["rules"]["contact_force_n"] = values["value"]
    else:
        out["rules"][rule].update(values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    goal = registered_goal()
    camera_noise = CAMERAS["front"].noise_m
    base = {
        "policy_id": "g09-conditionals-v1",
        "abstention": {"min_valid_fraction": DECLARED["abstention"]["min_valid_fraction"]},
        "windows": {k: {kk: vv for kk, vv in v.items() if kk != "why"} for k, v in DECLARED["windows"].items()},
        "rules": {
            "contact_force_n": 0.05,
            "position_tolerance_m": 3.0 * camera_noise,
            "reachable": {"margin_fraction": 0.0},
            "opposition_established": {"min_fraction": DECLARED["task_policy"]["opposition_established.min_fraction"]["value"]},
            "held": {"min_fraction": DECLARED["task_policy"]["held.min_fraction"]["value"], "lift_fraction_of_height": 0.1},
            "moving_with_robot": {"min_speed_mps": 0.01, "max_mismatch_fraction": 0.5, "pairing_tolerance_s": DECLARED["task_policy"]["moving_with_robot.pairing_tolerance_s"]["value"]},
            "stably_placed": {"max_speed_mps": goal.maximum_linear_speed_mps, "max_spread_m": 0.008},
        },
    }
    DECLARED["task_policy"]["stably_placed.max_speed_mps"]["value"] = goal.maximum_linear_speed_mps
    DECLARED["task_policy"]["position_tolerance_m"]["value"] = 3.0 * camera_noise

    # -- the calibration episodes, their streams and labels ---------------------
    episodes = []
    for root in discover(args.calibration):
        episode = Episode.load(root, default_goal=goal)
        if episode.track != "strict_fixed_world" or episode.refusal:
            continue
        episodes.append(episode)
    print(json.dumps({"calibration_episodes": len(episodes)}), flush=True)
    cache = []
    for episode in episodes:
        streams = EvidenceStreams.build(episode, configuration("front_contact", episode.effectors))
        labeler = OracleLabeler(episode)
        times = np.arange(episode.start_s + QUERY_STEP_S, episode.end_s - 1e-6, QUERY_STEP_S)
        for effector in episode.effectors:
            labels = {}
            for name in ("opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear"):
                window = base["windows"][name]["duration_s"]
                labels[name] = [(float(t), labeler.label(name, effector, float(t), window).value) for t in times]
            reach_times = [float(times[0]), float(times[len(times) // 2]), float(times[-1])]
            labels["reachable"] = [(t, labeler.label("reachable", effector, t, 0.5).value) for t in reach_times]
            cache.append((episode, effector, streams, labels))
        print(json.dumps({"episode": episode.episode_id, "queries": int(len(times))}), flush=True)

    def score(policy: dict, name: str) -> tuple[float, dict]:
        pairs = []
        for episode, effector, streams, labels in cache:
            conditional = conditionals_for(episode, effector, policy)[name]
            for t, label in labels[name]:
                verdict = decide(conditional, streams, t)
                if verdict.decision is Decision.UNKNOWN:
                    continue
                pairs.append((label, verdict.decision is Decision.PASS))
        return balanced_accuracy(pairs), {"decided": len(pairs), "positives": sum(1 for l, _ in pairs if l)}

    calibration: dict[str, dict] = {}

    def choose(name: str, rule: str, grid: list[dict], conservative_key) -> None:
        nonlocal base
        results = []
        for point in grid:
            candidate = with_rule(base, rule, **point)
            accuracy, detail = score(candidate, name)
            results.append({"point": point, "balanced_accuracy": accuracy, **detail})
        best = max(r["balanced_accuracy"] for r in results)
        tied = sorted([r for r in results if abs(r["balanced_accuracy"] - best) < 1e-12], key=conservative_key)
        # The middle of the plateau: a threshold at either end of the run of
        # equally good values sits against the data that chose it.
        chosen = tied[len(tied) // 2]
        base = with_rule(base, rule, **chosen["point"])
        calibration[f"{name}.{rule}"] = {"grid": results, "chosen": chosen["point"], "balanced_accuracy": chosen["balanced_accuracy"], "tied": len(tied)}
        print(json.dumps({"calibrated": f"{name}.{rule}", "chosen": chosen["point"], "balanced_accuracy": round(chosen["balanced_accuracy"], 4)}), flush=True)

    # The contact threshold: the same value serves opposition, held and the
    # released test of a placement; chosen on opposition, the rule with the
    # most transitions in the calibration set.
    choose("opposition_established", "contact_force_n", [{"value": v} for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)], lambda r: r["point"]["value"])
    choose("held", "held", [{"lift_fraction_of_height": v} for v in (0.05, 0.1, 0.2, 0.3, 0.5)], lambda r: r["point"]["lift_fraction_of_height"])
    choose("moving_with_robot", "moving_with_robot", [{"min_speed_mps": s, "max_mismatch_fraction": f} for s in (0.005, 0.01, 0.02, 0.04) for f in (0.25, 0.5, 0.75)],
           lambda r: (r["point"]["min_speed_mps"], r["point"]["max_mismatch_fraction"]))
    choose("stably_placed", "stably_placed", [{"max_spread_m": v} for v in (0.004, 0.006, 0.008, 0.01, 0.015)], lambda r: r["point"]["max_spread_m"])
    choose("reachable", "reachable", [{"margin_fraction": v} for v in (0.0, 0.02, 0.05, 0.1)], lambda r: r["point"]["margin_fraction"])

    accuracies = {name: score(base, name) for name in ("reachable", "opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear")}
    policy = dict(base)
    policy["calibration"] = {
        "episodes": [{"episode_id": e.episode_id, "trace_sha256": e.trace_sha256, "outcome": e.outcome.get("status"), "failed_gate": e.outcome.get("failed_gate"), "root": e.root.as_posix()} for e in episodes],
        "configuration": "front_contact", "query_step_s": QUERY_STEP_S, "results": calibration,
        "balanced_accuracy_on_calibration": {k: {"balanced_accuracy": v[0], **v[1]} for k, v in accuracies.items()},
        "declared": DECLARED, "camera_noise_m": camera_noise,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json_bytes(policy))
    print(json.dumps({"policy": args.out.as_posix(), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(), "rules": policy["rules"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
