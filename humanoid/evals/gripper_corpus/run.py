"""Run the gripper corpus and report which named gates each case tripped.

This replaces a shell one-liner that lived in a terminal. That mattered: the
one-liner had its thresholds typed inline, no record of what was expected, and
no way to be wrong on purpose -- so when a rename dropped the suite from 12/12
to 7/12 the only reason it was noticed was that the number happened to be read
that day.

Two things this does that the one-liner could not.

It scores NAMED gates, so a failure says `grasped` rather than a count, and a
regression names the thing that broke.

It runs KNOWN-BAD cases, which is the half that keeps the suite honest. A case
declaring `must_fail: [grasped]` is a claim about the harness, not the machine:
if a block 110 mm across is ever reported as grasped by jaws that open to 86,
then `grasped` is measuring something other than a grip, and every green result
above it is worth nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, "src")

from rigby_poc.gripper.body.manifest import spec  # noqa: E402
from rigby_poc.gripper.decision.pick_and_place import (  # noqa: E402
    advance, decide, object_in_target,
)
from rigby_poc.gripper.physics.model import (  # noqa: E402
    JOINTS, computed_torque, make,
)
from rigby_poc.gripper.sensing.gripper_camera import Senses, sense  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates import Outcome, failures, scored  # noqa: E402

HERE = Path(__file__).resolve().parent


def one(case: dict, seconds: float, table_top: float) -> Outcome:
    body = make(np.asarray(case["half"], dtype=float),
                np.asarray(case["at"], dtype=float), table_top=table_top)
    limits = spec().get("rate_limits", {})
    per_joint = limits.get("joints_deg_per_s", {})
    ceiling = np.asarray(
        [float(np.radians(per_joint.get(n, 120.0))) for n in JOINTS[:4]]
        + [float(limits.get("finger_m_per_s", 0.07)) / 2.0] * 2)

    eyes = Senses()
    held = np.asarray(body.q())
    squeeze = 0.0
    phase = 0
    fps = 30
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    start_z = float(body.block()[2])
    ever_held = False
    peak = 0.0
    worst = 0.0
    seen_close = False
    holding_now = False

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = sense(body, eyes, held, squeeze, now, table_top)
        phase = advance(body, seen, phase, now)
        command = decide(body, seen, phase, table_top, now)
        squeeze = command.squeeze_n
        if not command.hold_station:
            held = held + np.clip(command.target - held,
                                  -ceiling / fps, ceiling / fps)
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)
        holding_now = bool(seen.holding())
        ever_held = ever_held or holding_now
        seen_close = seen_close or bool(getattr(seen, "fine_fix", False))
        peak = max(peak, float(body.block()[2]) - start_z)
        worst = max(worst, float(body.penetration_mm()))

    return Outcome(
        placed=bool(object_in_target(body)),
        ever_held=ever_held,
        peak_lift_m=round(peak, 4),
        deepest_penetration_mm=round(worst, 3),
        ever_seen_close=seen_close,
        held_at_end=holding_now,
        final_block_z=round(float(body.block()[2]), 4),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--only", default="", help="substring of a case id")
    args = parser.parse_args()

    document = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    seconds = float(document.get("seconds", 24.0))
    table_top = float(document.get("table_top_m", 0.72))
    cases = [c for c in document["cases"] if args.only in c["id"]]

    good = [c for c in cases if c.get("expect") == "placed"]
    bad = [c for c in cases if c.get("expect") == "fails"]
    placed = 0
    honest = 0
    broken: list[str] = []

    print(f"  {len(good)} cases expected to place, "
          f"{len(bad)} expected to fail a named gate")
    print()
    for case in good:
        outcome = one(case, seconds, table_top)
        tripped = failures(outcome)
        placed += bool(outcome.placed)
        mark = "PLACED" if outcome.placed else "MISS  "
        print(f"  {mark} {case['id']:<24} lift={outcome.peak_lift_m:5.3f} "
              f"pen={outcome.deepest_penetration_mm:5.2f}mm"
              + (f"  tripped: {', '.join(tripped)}" if tripped else ""))

    print()
    for case in bad:
        outcome = one(case, seconds, table_top)
        marks = scored(outcome)
        wanted = list(case.get("must_fail") or [])
        missed = [g for g in wanted if marks.get(g) is not False]
        if missed:
            broken.append(f"{case['id']}: expected to fail {missed}, did not")
            print(f"  HARNESS {case['id']:<24} did NOT fail {missed} "
                  f"-- the gate is not measuring what it claims")
        else:
            honest += 1
            print(f"  as-designed {case['id']:<20} failed {wanted}")

    print()
    print(f"  placed        {placed}/{len(good)}")
    print(f"  known-bad ok  {honest}/{len(bad)}")
    if broken:
        print()
        for line in broken:
            print(f"  HARNESS PROBLEM: {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
