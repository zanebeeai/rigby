"""Is the retarget failure an infeasibility, or a frame convention?

The distinction decides plan 14. If a clip genuinely demands rotations no human
joint can make, the verifier's first finding is about the compiler. If the two
rigs merely disagree about which way a bone's local axes point, the verifier
would be an expensive way to rediscover one conversion bug.

Two measurements separate them.

**Axis constancy.** A 1-DOF bone driven through a real range sweeps one axis and
varies only the angle. If the clip's demanded axis is constant to within a
fraction of a degree over hundreds of samples and is *orthogonal* to the joint's
hinge, that is not a body doing something impossible — that is two conventions
disagreeing. The knee is the control: its local frame is world-aligned in both
rigs, so if the theory holds the knee matches exactly while the fingers do not.

**The conjugation spike.** If the disagreement is a fixed frame rotation ``A``,
then conjugating the target by it (``A R Aᵀ``) should collapse the residual to
zero. This applies one candidate per hand and reports what survives. What
survives is the real rig-model gap: a knuckle given one hinge where the clip
poses two DOFs, and a thumb saddle modelled as two hinges.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
from rigby_v2.rigging.coordinates import GltfQuaternionXYZW, gltf_quaternion_to_mujoco
from rigby_v2.rigging.pose_adapter import VisualPhysicalPoseAdapter
from scipy.spatial.transform import Rotation

from .corpus_snapshot import iter_committed_clips
from .retarget_residual import bone_residual

#: One candidate alignment per hand, mirrored. Not fitted — read off the measured
#: axis disagreement (clip x, hinge y) and then tested against every sampled frame.
LEFT_ALIGNMENT = Rotation.from_euler("z", 90, degrees=True).as_matrix()
RIGHT_ALIGNMENT = Rotation.from_euler("z", -90, degrees=True).as_matrix()

#: Below this the bone is not moving and its axis is numerical noise.
MINIMUM_ANGLE_RAD = np.radians(2.0)


def axis_constancy(profile: str = "medium", stride: int = 7) -> dict[str, dict]:
    """For each 1-DOF bone: is the demanded rotation axis the same every frame?"""

    adapter = VisualPhysicalPoseAdapter(profile)
    model = adapter.model
    single = [b for b, j in adapter._bone_joint_ids.items() if len(j) == 1]
    samples: dict[str, list[np.ndarray]] = {bone: [] for bone in single}

    for _case_id, clip in iter_committed_clips():
        for frame in clip["frames"][::stride]:
            for bone in single:
                pose = frame["bones"].get(bone)
                if pose is None:
                    continue
                rotation = pose["rotation"]
                converted = gltf_quaternion_to_mujoco(
                    GltfQuaternionXYZW(
                        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
                    )
                )
                rotvec = Rotation.from_quat(
                    (converted.x, converted.y, converted.z, converted.w)
                ).as_rotvec()
                angle = float(np.linalg.norm(rotvec))
                if angle > MINIMUM_ANGLE_RAD:
                    samples[bone].append(rotvec / angle)

    rows: dict[str, dict] = {}
    for bone, vectors in samples.items():
        if len(vectors) < 5:
            continue
        stacked = np.asarray(vectors)
        mean = stacked.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        spread = float(np.degrees(np.arccos(np.clip(stacked @ mean, -1.0, 1.0))).max())
        joint = adapter._bone_joint_ids[bone][0]
        rows[bone] = {
            "samples": len(vectors),
            "axis_spread_deg": spread,
            "mean_clip_axis": np.round(mean, 3).tolist(),
            "hinge_axis": np.round(model.jnt_axis[joint], 3).tolist(),
        }
    return rows


def conjugation_spike(profile: str = "medium", stride: int = 20) -> dict[str, dict]:
    """Does one constant mirrored rotation make the hand bones feasible?"""

    adapter = VisualPhysicalPoseAdapter(profile)
    model = adapter.model
    hand_tokens = ("Thumb", "Index", "Middle", "Ring", "Little")
    bones = [b for b in adapter._bone_joint_ids if any(t in b for t in hand_tokens)]
    before: dict[str, float] = collections.defaultdict(float)
    after: dict[str, float] = collections.defaultdict(float)

    for _case_id, clip in iter_committed_clips():
        for frame in clip["frames"][::stride]:
            for bone in bones:
                joint_ids = adapter._bone_joint_ids[bone]
                pose = frame["bones"].get(bone)
                if not joint_ids or pose is None:
                    continue
                rotation = pose["rotation"]
                quaternion = (
                    rotation["x"],
                    rotation["y"],
                    rotation["z"],
                    rotation["w"],
                )
                raw, _ = bone_residual(model, joint_ids, quaternion)
                before[bone] = max(before[bone], raw)

                converted = gltf_quaternion_to_mujoco(GltfQuaternionXYZW(*quaternion))
                target = Rotation.from_quat(
                    (converted.x, converted.y, converted.z, converted.w)
                ).as_matrix()
                alignment = (
                    LEFT_ALIGNMENT if bone.startswith("left") else RIGHT_ALIGNMENT
                )
                aligned = Rotation.from_matrix(
                    alignment @ target @ alignment.T
                ).as_quat()
                # bone_residual re-applies the gltf->mujoco conversion, so undo it
                # here and hand it a quaternion in the source basis.
                source = _mujoco_quat_to_gltf_xyzw(aligned)
                fixed, _ = bone_residual(model, joint_ids, source)
                after[bone] = max(after[bone], fixed)

    return {
        bone: {
            "worst_before_deg": float(np.degrees(before[bone])),
            "worst_after_deg": float(np.degrees(after[bone])),
        }
        for bone in sorted(before)
    }


def _mujoco_quat_to_gltf_xyzw(quat_xyzw: np.ndarray) -> tuple[float, ...]:
    """Invert :func:`gltf_quaternion_to_mujoco` for a quaternion already in MuJoCo basis."""

    from rigby_v2.rigging.coordinates import (
        MujocoQuaternionWXYZ,
        mujoco_quaternion_to_gltf,
    )

    x, y, z, w = (float(v) for v in quat_xyzw)
    back = mujoco_quaternion_to_gltf(MujocoQuaternionWXYZ(w, x, y, z))
    return (back.x, back.y, back.z, back.w)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--skip-spike", action="store_true")
    args = parser.parse_args()

    print("=== axis constancy, 1-DOF bones ===")
    print(
        f"{'bone':26s}{'n':>6}{'spread deg':>12}{'mean clip axis':>24}{'hinge axis':>18}"
    )
    rows = axis_constancy(args.profile)
    for bone, row in rows.items():
        print(
            f"{bone:26s}{row['samples']:6d}{row['axis_spread_deg']:12.2f}"
            f"{row['mean_clip_axis']!s:>24}{row['hinge_axis']!s:>18}"
        )

    if args.skip_spike:
        return 0

    print("\n=== conjugation spike, hand bones ===")
    spike = conjugation_spike(args.profile)
    print(f"{'bone':26s}{'worst before deg':>18}{'worst after deg':>17}")
    for bone, row in spike.items():
        print(
            f"{bone:26s}{row['worst_before_deg']:18.2f}{row['worst_after_deg']:17.2f}"
        )
    fixed = sum(1 for row in spike.values() if row["worst_after_deg"] < 1e-5)
    print(
        f"\nexactly feasible after one constant mirrored rotation: {fixed} of {len(spike)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
