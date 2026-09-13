"""Register the G06 transfer roster before any scored trial is run.

One hundred seeds per gripper-bearing public body on the frozen fixture, each
seed a numerically recorded draw: the cube displaced by up to 20 mm in the
bench plane, its mass and its sliding friction scaled by up to 20 percent,
drawn independently and uniformly -- the same perturbation family the G02
transfer protocol registers -- and identical across bodies, since the world is
the thing that must not change when the body does. The capability-normalized
track's reference is recorded here too. The registration hashes the frozen
environment, goal, feasibility map and this roster together; the runner
refuses any of them that no longer matches. No API or model calls.

    python any-robot/scripts/g06_roster.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "assets/general/research-protocols/g06-transfer-v1"
SEED = 20260913
TRIALS_PER_BODY = 100
TRANSLATION_HALF_WIDTH_M = (0.02, 0.02, 0.0)
MASS_MULTIPLIER_RANGE = (0.8, 1.2)
FRICTION_MULTIPLIER_RANGE = (0.8, 1.2)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    if (PROTOCOL_DIR / "roster.json").exists():
        raise SystemExit("the roster is registered; it is never overwritten")
    feasibility = json.loads((PROTOCOL_DIR / "feasibility-map.json").read_bytes())
    rng = np.random.default_rng(SEED)
    draws = []
    for seed in range(TRIALS_PER_BODY):
        translation = [float(rng.uniform(-w, w)) if w > 0 else 0.0 for w in TRANSLATION_HALF_WIDTH_M]
        draws.append({"seed": seed, "translation_m": translation,
                      "mass_multiplier": float(rng.uniform(*MASS_MULTIPLIER_RANGE)),
                      "friction_multiplier": float(rng.uniform(*FRICTION_MULTIPLIER_RANGE))})
    bodies = []
    for body in feasibility["bodies"]:
        bodies.append({"zoo_id": body["zoo_id"], "rig_id": body["rig_id"], "source_urdf": f"any-robot/assets/general/zoo/{body['zoo_id']}/robot.urdf",
                       "feasibility_class": body["class"], "seeds": draws})
    roster = {
        "schema": "g06.transfer-roster.v1", "goal": "G06", "environment_id": "g06_transfer_v1",
        "draw": {"seed": SEED, "generator": "numpy.random.default_rng(seed), one stream, draw order translation xyz, mass, friction per seed",
                 "trials_per_body": TRIALS_PER_BODY, "translation_half_width_m": list(TRANSLATION_HALF_WIDTH_M),
                 "mass_multiplier_range": list(MASS_MULTIPLIER_RANGE), "friction_multiplier_range": list(FRICTION_MULTIPLIER_RANGE),
                 "identical_across_bodies": True},
        "normalization": {"rule": "EnvironmentV1.scaled_to: lengths by reach over reference reach, object size by aperture over reference aperture, mass by volume, capped at payload",
                          "reference_body": "zoo_jaw_arm",
                          "reference_reach_m": 1.32,
                          "reference_aperture_m": 0.087,
                          "note": "The reference is the jaw arm's whole-body reach and aperture, so its normalized world coincides with the fixed one to within the environment's own scaling rule."},
        "success": {"definition": "attempt_transfer certifies: hidden-weld/object-actuator/exclusion policy clean, opposition established, lifted at least 0.8 object heights, held continuously 2.0 s, carried within 2.5 apertures of the grasp centre, released with no robot contact or normal force, whole geometry inside the destination region and still for 2.0 s, penetration at most 4 mm, no self-collision",
                    "target": "at least 90 of 100 registered trials per feasible body", "attempts_per_trial": 1,
                    "enabled_set_rule": "bodies the independent feasibility map classes feasible; an infeasible body is attempted once and its typed refusal recorded; a body without a grasping effector is recorded as unsupported by structure"},
        "budgets": {"simulation_s_per_trial": 40.0, "wall_s_per_trial": 300.0},
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "scored_runs_started": False, "generation_calls": 0,
        "bodies": bodies,
    }
    (PROTOCOL_DIR / "roster.json").write_bytes(json_bytes(roster))
    files = {name: hashlib.sha256((PROTOCOL_DIR / name).read_bytes()).hexdigest() for name in ("environment.json", "goal.json", "feasibility-map.json", "roster.json")}
    registration = {"schema": "g06.registration.v1", "files": files,
                    "registration_sha256": hashlib.sha256(json_bytes(files)).hexdigest(),
                    "registered_at_utc": datetime.now(timezone.utc).isoformat(), "scored_runs_started": False,
                    "feasible_bodies": [b["zoo_id"] for b in feasibility["bodies"] if b["class"] == "feasible"],
                    "infeasible_bodies": [b["zoo_id"] for b in feasibility["bodies"] if b["class"] == "infeasible"],
                    "unsupported_by_structure": [b["zoo_id"] for b in feasibility["bodies"] if b["class"] == "unsupported_by_structure"],
                    "scope": "Public engineering registration for G06; the G02 transfer benchmark's sealed identities are untouched."}
    (PROTOCOL_DIR / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({k: registration[k] for k in ("registration_sha256", "feasible_bodies", "infeasible_bodies", "unsupported_by_structure")}, indent=1))


if __name__ == "__main__":
    main()
