"""Does the honest-sensing gripper work anywhere, or only where it was tuned?

Every run here uses the same controller with the same instrument list: joint
encoders, one gripper camera, contact inferred from the encoders. The block moves
around the bench and changes size, and nothing tells the machine that it has.

The sweep is the point. A single scripted success at one placement proves the
numbers were tuned to that placement; the interesting question is how much of
the bench the same vocabulary covers, and where it stops.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from rigby_poc.gripper.runs.pick_and_place import run  # noqa: E402

#: Placement and size, in MuJoCo coordinates. The bench is the region the scan
#: grid covers; two of these sit outside it on purpose.
CASES: tuple[tuple[str, list[float], list[float]], ...] = (
    ("centre, as tuned", [0.00, 0.30, 0.76], [0.030, 0.040, 0.030]),
    ("near, centred", [0.00, 0.16, 0.76], [0.030, 0.040, 0.030]),
    ("far, centred", [0.00, 0.44, 0.76], [0.030, 0.040, 0.030]),
    ("left", [-0.18, 0.30, 0.76], [0.030, 0.040, 0.030]),
    ("right", [0.18, 0.30, 0.76], [0.030, 0.040, 0.030]),
    ("left and far", [-0.16, 0.42, 0.76], [0.030, 0.040, 0.030]),
    ("right and near", [0.16, 0.17, 0.76], [0.030, 0.040, 0.030]),
    ("narrow block", [0.00, 0.30, 0.75], [0.018, 0.040, 0.018]),
    ("wide block", [0.00, 0.30, 0.77], [0.042, 0.040, 0.042]),
    ("tall thin block", [0.00, 0.30, 0.79], [0.020, 0.065, 0.020]),
    ("off the scan grid", [0.28, 0.30, 0.76], [0.030, 0.040, 0.030]),
    ("behind the reach", [0.00, 0.56, 0.76], [0.030, 0.040, 0.030]),
)


def main(seconds: float = 26.0) -> int:
    print(f"{'case':<22} {'placed':<8} {'lift_cm':>8} {'pen_mm':>7}  outcome")
    print("-" * 62)
    wins = 0
    for label, at, half in CASES:
        got = run(block_half=half, block_at=at, seconds=seconds,
                  name=f"gripper-{label.replace(' ', '-').replace(',', '')}",
                  verbose=False)
        placed = bool(got["in_target"])
        wins += placed
        print(f"{label:<22} {str(placed):<8} {got['peak_lift_m'] * 100:8.2f} "
              f"{got['deepest_penetration_mm']:7.3f}  "
              f"{'PLACED' if placed else 'no'}")
    print("-" * 62)
    print(f"{wins}/{len(CASES)} placed in the bin")
    return wins


if __name__ == "__main__":
    main()
