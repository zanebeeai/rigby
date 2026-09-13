"""Register the G09 evaluation roster: which recorded episodes the conditionals are scored on.

The roster is every replayable bundle in the given trees that ran in the
strict fixed world -- the G06 scored campaign and its canonical trials,
the D06 pairs, the G07 tree runs, the G08 campaign, its pilot and the D08
pairs -- successes, failures, refusals and interruptions alike, each by
its trace hash. The calibration set (the G06 pilot) is excluded by hash.
The registration hashes the roster, the policy and the sensor
configurations, and the campaign refuses to run against anything else.
No API or model calls.

    python any-robot/scripts/g09_corpus.py --roots <tree> [<tree> ...] --exclude <policy.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import CONFIGURATIONS, Episode, configuration
from rigby_general.sensing.sensors import CAMERAS

import g09_calibrate as calibrate


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / "assets/general/research-protocols/g09-conditionals-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, default=PROTOCOL / "policy.json")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_bytes())
    excluded = {e["trace_sha256"] for e in policy["calibration"]["episodes"]}
    goal = calibrate.registered_goal()
    roster, seen = [], set()
    for root in args.roots:
        for bundle in calibrate.discover([root]):
            episode = Episode.load(bundle, default_goal=goal)
            if episode.track != "strict_fixed_world":
                continue
            if episode.trace_sha256 in seen:
                continue
            seen.add(episode.trace_sha256)
            kind = "calibration_excluded" if episode.trace_sha256 in excluded else "evaluation"
            roster.append({"episode_id": episode.episode_id, "zoo_id": episode.zoo_id, "goal": episode.metadata.get("goal"), "protocol": episode.task.get("protocol"),
                           "trace_sha256": episode.trace_sha256, "bundle_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
                           "outcome": episode.metadata.get("outcome"), "failed_gate": episode.outcome.get("failed_gate"), "refusal": episode.refusal,
                           "duration_s": episode.end_s - episode.start_s, "effectors": [e.chain_id for e in episode.effectors],
                           "tree": root.as_posix(), "relative": bundle.relative_to(root).as_posix(), "kind": kind})
            print(json.dumps({"episode": episode.episode_id, "kind": kind, "outcome": episode.metadata.get("outcome")}), flush=True)
    evaluation = [r for r in roster if r["kind"] == "evaluation"]
    reference = next(Episode.load(Path(r["tree"]) / r["relative"], default_goal=goal) for r in evaluation if r["zoo_id"] == "zoo_jaw_arm")
    configurations = {name: configuration(name, reference.effectors).model_dump(mode="json") for name in CONFIGURATIONS}
    corpus = {
        "schema": "g09.conditionals-corpus.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "cameras": {k: {"position_m": v.position_m, "rate_hz": v.rate_hz, "latency_s": v.latency_s, "max_age_s": v.max_age_s, "visible_fraction": v.visible_fraction, "noise_m": v.noise_m} for k, v in CAMERAS.items()},
        "configurations": configurations, "configuration_reference_body": reference.zoo_id,
        "episodes": roster, "evaluation_count": len(evaluation), "excluded_count": len(roster) - len(evaluation),
        "scored_runs_started": False, "generation_calls": 0,
    }
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    (PROTOCOL / "corpus.json").write_bytes(json_bytes(corpus))
    files = {name: hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() for name in ("corpus.json", "policy.json")}
    files["g06/goal.json"] = hashlib.sha256((calibrate.G06 / "goal.json").read_bytes()).hexdigest()
    files["g06/environment.json"] = hashlib.sha256((calibrate.G06 / "environment.json").read_bytes()).hexdigest()
    listing = json.dumps(files, indent=2, sort_keys=True, allow_nan=False) + "\n"
    registration = {"schema": "g09.registration.v1", "registered_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                    "registration_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(), "evaluation_episodes": len(evaluation), "note": "registered before any scored evaluation; the campaign refuses a corpus or policy that does not hash to this"}
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"evaluation": len(evaluation), "excluded": len(roster) - len(evaluation), "registration_sha256": registration["registration_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
