"""Which clip bones can the physical rig not represent, and by how much?

``VisualPhysicalPoseAdapter.visual_to_physical`` raises at the first bone whose
rotation its ordered hinge chain cannot reproduce, so on a real clip it reports
one bone and stops. That is correct for a strict round trip and useless as a
diagnosis: it hides how many bones are affected and by how much.

This probe runs the same per-bone least-squares solve the adapter runs, but for
every bone, and reports the residual instead of raising. The residual is the
product — "this clip asks the index PIP for 80 degrees about an axis no finger
joint has" is a finding about the clip, or about the conversion, and either way
it is a number rather than a traceback.

Sampling is strided rather than exhaustive: the residual is per-pose, not
cumulative, and a stride of six frames per case covers every case at a fraction
of the cost. Pass ``--stride 1`` to check that conclusion.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import mujoco
import numpy as np
from rigby_v2.rigging.coordinates import GltfQuaternionXYZW, gltf_quaternion_to_mujoco
from rigby_v2.rigging.pose_adapter import VisualPhysicalPoseAdapter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .corpus_snapshot import iter_committed_clips


def bone_residual(
    model: mujoco.MjModel,
    joint_ids: tuple[int, ...],
    quaternion_xyzw: tuple[float, ...],
) -> tuple[float, float]:
    """Return (residual, demanded) in radians for one bone's ordered hinge chain.

    ``demanded`` is the magnitude of the rotation the clip asked for. When the two
    are equal the solver achieved nothing, which means the demanded axis lies
    outside the span of the chain rather than merely beyond its limits.
    """

    converted = gltf_quaternion_to_mujoco(GltfQuaternionXYZW(*quaternion_xyzw))
    target = Rotation.from_quat(
        (converted.x, converted.y, converted.z, converted.w)
    ).as_matrix()
    axes = [np.asarray(model.jnt_axis[j], dtype=float) for j in joint_ids]
    lower = np.asarray([model.jnt_range[j, 0] for j in joint_ids])
    upper = np.asarray([model.jnt_range[j, 1] for j in joint_ids])
    rest = np.asarray(
        [model.qpos0[int(model.jnt_qposadr[j])] for j in joint_ids], dtype=float
    )

    def residual(values: np.ndarray) -> np.ndarray:
        predicted = np.eye(3, dtype=float)
        for axis, value, rest_value in zip(axes, values, rest, strict=True):
            predicted = (
                predicted
                @ Rotation.from_rotvec(axis * (value - rest_value)).as_matrix()
            )
        return Rotation.from_matrix(target.T @ predicted).as_rotvec()

    solved = least_squares(
        residual,
        x0=np.clip(rest, lower, upper),
        bounds=(lower, upper),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=200,
    )
    demanded = float(np.linalg.norm(Rotation.from_matrix(target).as_rotvec()))
    return float(np.linalg.norm(residual(solved.x))), demanded


def survey(profile: str = "medium", stride: int = 6) -> dict[str, dict[str, float]]:
    adapter = VisualPhysicalPoseAdapter(profile)
    model = adapter.model
    worst: dict[str, float] = collections.defaultdict(float)
    demanded: dict[str, float] = collections.defaultdict(float)
    infeasible: collections.Counter[str] = collections.Counter()
    seen: collections.Counter[str] = collections.Counter()

    for _case_id, clip in iter_committed_clips():
        frames = clip["frames"]
        if not frames:
            continue
        step = max(1, len(frames) // stride)
        for frame in frames[::step]:
            for bone, joint_ids in adapter._bone_joint_ids.items():
                if (
                    not joint_ids
                    or model.jnt_type[joint_ids[0]] == mujoco.mjtJoint.mjJNT_FREE
                ):
                    continue
                pose = frame["bones"].get(bone)
                if pose is None:
                    continue
                rotation = pose["rotation"]
                residual, asked = bone_residual(
                    model,
                    joint_ids,
                    (rotation["x"], rotation["y"], rotation["z"], rotation["w"]),
                )
                seen[bone] += 1
                if residual > 1e-7:
                    infeasible[bone] += 1
                worst[bone] = max(worst[bone], residual)
                demanded[bone] = max(demanded[bone], asked)

    return {
        bone: {
            "dofs": float(len(adapter._bone_joint_ids[bone])),
            "frames_sampled": float(seen[bone]),
            "frames_infeasible": float(infeasible[bone]),
            "worst_residual_deg": float(np.degrees(worst[bone])),
            "max_demanded_deg": float(np.degrees(demanded[bone])),
        }
        for bone in sorted(worst)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--stride", type=int, default=6, help="frames sampled per case")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = survey(args.profile, args.stride)
    sampled = int(max((row["frames_sampled"] for row in rows.values()), default=0))
    print(f"{sampled} frames sampled per bone across the corpus\n")
    header = f"{'bone':28s}{'dof':>4}{'infeasible':>12}{'worst resid deg':>17}{'max asked deg':>15}"
    print(header)
    for bone, row in sorted(rows.items(), key=lambda kv: -kv[1]["worst_residual_deg"]):
        if row["worst_residual_deg"] <= 1e-5:
            continue
        print(
            f"{bone:28s}{int(row['dofs']):4d}{int(row['frames_infeasible']):12d}"
            f"{row['worst_residual_deg']:17.2f}{row['max_demanded_deg']:15.2f}"
        )
    clean = [b for b, r in rows.items() if r["worst_residual_deg"] <= 1e-5]
    print(
        f"\n{len(clean)} of {len(rows)} bones are exactly representable on every sampled frame"
    )
    if clean:
        print(f"  {', '.join(sorted(clean))}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", newline="\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
