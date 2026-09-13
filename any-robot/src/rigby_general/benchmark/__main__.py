"""Run a registered sensor-policy episode or an explicitly unscored probe."""

import argparse
import json
from pathlib import Path

from .runner import run_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("strict_fixed_world", "capability_normalized"), required=True)
    parser.add_argument("--policy", required=True, help="Importable trusted policy module:function")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-steps", type=int, help="Unscored infrastructure probe; never a manipulation success claim")
    args = parser.parse_args()
    report = run_trial(registration=args.registration, expected_registration_sha256=args.expected_registration_sha256,
                       robot_source=args.robot, split_id=args.split, seed=args.seed, mode=args.mode,
                       policy=args.policy, destination=args.out, probe_steps=args.probe_steps)
    print(json.dumps({key: report[key] for key in ("status", "scored", "root_success", "physical_steps", "elapsed_simulation_s", "world_sha256")}))


if __name__ == "__main__":
    main()
