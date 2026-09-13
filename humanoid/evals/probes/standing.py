"""Does the canonical humanoid stand under the existing runtime, and what does it cost?

Plan 14 rests on two claims this probe measures directly: that the v2 simulation
runtime already works on the canonical human without any new code, and that a
corpus-sized pass is cheap enough that cost is not an argument in the design.

It also demonstrates the split plan 14 §3 turns on. Driven by joint targets alone
the body falls over; with :class:`WholeBodyStandingController` owning the free
root it holds station. The clip commands joints and says nothing about balance,
so the second configuration is the one a verifier must use.

**Read the loaded model, never the XML.** ``canonical_human.xml`` gives all 67
actuators a uniform +-150 N.m ctrlrange; ``rigging.canonical_human`` rewrites
them anatomically at load. This probe uses ``xml_for_profile`` so the numbers
describe the model the runtime actually receives.
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np
from rigby_core.simulation.controller import (
    ConstantTarget,
    ControlTarget,
    StandingControlConfig,
)
from rigby_v2.rigging.canonical_human import load_canonical_human, xml_for_profile
from rigby_v2.simulation.runtime import (
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
)


def actuator_limit_groups(profile: str = "medium") -> dict[float, int]:
    """How many actuators share each ctrlrange ceiling, on the *loaded* model."""

    model = load_canonical_human(profile)
    counts: collections.Counter[float] = collections.Counter()
    for index in range(model.nu):
        counts[float(model.actuator_ctrlrange[index, 1])] += 1
    return dict(sorted(counts.items()))


def run(
    profile: str = "medium", duration_s: float = 3.5
) -> dict[str, dict[str, float]]:
    xml = xml_for_profile(profile)
    model = load_canonical_human(profile)
    target = ControlTarget.stationary(np.asarray(model.qpos0, dtype=float), model.nv)
    runtime = NativeMujocoRuntime()

    configurations = {
        "joint targets only": None,
        "with standing controller": StandingControlConfig(
            support_foot_bodies=("left_foot", "right_foot"),
            pelvis_body="pelvis",
            torso_body="torso",
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for label, standing in configurations.items():
        request = SimulationRequest(
            model_xml=xml,
            trajectory=ConstantTarget(target),
            config=SimulationConfig(
                duration_s=duration_s,
                standing=standing,
                free_root_joint_name="pelvis_free",
            ),
            initial_qpos=np.asarray(model.qpos0, dtype=float),
            request_id=label,
        )
        started = time.perf_counter()
        result = runtime.simulate(request)
        wall_s = time.perf_counter() - started
        if not result.completed:
            raise RuntimeError(f"{label}: {result.failure}")
        qpos = result.trace.qpos
        root = qpos[:, :3]
        forces = result.trace.actuator_force
        results[label] = {
            "steps": float(qpos.shape[0]),
            "wall_s": wall_s,
            "realtime_factor": duration_s / wall_s,
            "pelvis_z_start_m": float(root[0, 2]),
            "pelvis_z_end_m": float(root[-1, 2]),
            "pelvis_z_min_m": float(root[:, 2].min()),
            "pelvis_drift_m": float(
                np.linalg.norm(root[:, :2] - root[0, :2], axis=1).max()
            ),
            "peak_actuator_force_nm": float(np.abs(forces).max())
            if forces.size
            else 0.0,
            "mean_actuator_force_nm": float(np.abs(forces).mean())
            if forces.size
            else 0.0,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--duration", type=float, default=3.5)
    args = parser.parse_args()

    model = load_canonical_human(args.profile)
    print(
        f"canonical_human {args.profile}: nq {model.nq} nv {model.nv} nu {model.nu} "
        f"mass {model.body_mass.sum():.2f} kg"
    )
    print(
        f"actuator ctrlrange groups (N.m -> count): {actuator_limit_groups(args.profile)}"
    )
    print()

    for label, row in run(args.profile, args.duration).items():
        print(f"--- {label} ---")
        print(
            f"  {int(row['steps'])} steps in {row['wall_s']:.2f} s "
            f"({row['realtime_factor']:.1f}x realtime)"
        )
        print(
            f"  pelvis z {row['pelvis_z_start_m']:.4f} -> {row['pelvis_z_end_m']:.4f} m "
            f"(min {row['pelvis_z_min_m']:.4f})"
        )
        print(f"  pelvis horizontal drift {row['pelvis_drift_m']:.4f} m")
        print(
            f"  actuator force peak {row['peak_actuator_force_nm']:.1f} N.m, "
            f"mean {row['mean_actuator_force_nm']:.2f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
