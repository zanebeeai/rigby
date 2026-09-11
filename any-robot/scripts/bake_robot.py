"""Bake the certified primitive library for one robot, or for the whole zoo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rigby_general.bake import bake_robot
from rigby_general.config import GeneralSettings
from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveLibrary
from rigby_general.schema.inventory import load_inventory


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="*", help="robot ids; default is the whole zoo")
    parser.add_argument("--root", type=Path, default=ZOO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    settings = GeneralSettings.from_env()
    library = PrimitiveLibrary(arguments.out or settings.robot_root)
    inventory = load_inventory()

    ids = arguments.robot or sorted(
        directory.name
        for directory in arguments.root.iterdir()
        if (directory / "robot.urdf").is_file()
    )

    failures = 0
    for robot_id in ids:
        source = arguments.root / robot_id / "robot.urdf"
        started = time.perf_counter()
        robot = ingest_robot(source, robot_id=robot_id)

        def progress(candidate, outcome, done, total):
            if arguments.verbose:
                ok = not hasattr(outcome, "stage")
                print(
                    f"    [{done:>3}/{total}] {candidate.label:<34} "
                    f"{'ok' if ok else 'x ' + outcome.failure_code}"
                )

        result = bake_robot(
            robot.manifest,
            robot.finalized.model,
            inventory,
            budget_seconds=arguments.budget,
            progress=progress,
        )
        library.write(robot_id, list(result.records), list(result.failures), result.summary)

        summary = result.summary
        print(
            f"{robot_id:<18} {summary.certified:>3}/{summary.attempted:<3} certified"
            f"  yield={summary.yield_fraction:.0%}"
            f"  coverage={summary.schema_coverage:.0%}"
            f"  {time.perf_counter() - started:5.1f}s"
            f"  {'complete' if summary.complete else 'BUDGET EXHAUSTED'}"
        )
        if summary.schema_coverage < 1.0:
            missing = sorted(
                set(summary.afforded_entry_ids) - set(summary.covered_entry_ids)
            )
            print(f"{'':18} uncovered schemas: {missing}")
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
