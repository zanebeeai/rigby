"""Real-contact acceptance demonstrations for the two bimanual packs.

The routines in this module author controller targets only.  During an
acceptance simulation MuJoCo owns every free-root and object coordinate after
the single initialization write; neither object has an actuator or constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    ArticulatedCompletionPredicate,
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationResult,
    GraspPredicate,
    HoldPredicate,
    JointComparator,
    LiftPredicate,
    ReleasePredicate,
    policy_from_manifests,
    predicates_from_manifest,
)
from rigby_v2.contracts import (
    ContactEdgeV2,
    CoordinateFrame,
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
    Vec3,
)
from rigby_v2.motion import TimingProfile, compile_motion_program
from rigby_v2.rigging import (
    stage_canonical_rig,
    validate_simulated_glb_round_trip,
    xml_for_profile,
)
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import (
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    TargetProvider,
    canonical_standing_config,
    project_qpos_to_rig,
)


_LEFT_ARM = (
    "left_shoulder_flex",
    "left_shoulder_abduct",
    "left_shoulder_twist",
    "left_elbow_flex",
    "left_forearm_twist",
    "left_wrist_flex",
    "left_wrist_deviation",
)
_RIGHT_ARM = tuple(name.replace("left_", "right_") for name in _LEFT_ARM)
@dataclass(frozen=True)
class BimanualAcceptanceReport:
    pack_id: str
    neutral: CertificationResult
    demonstration: CertificationResult


@dataclass(frozen=True)
class _AcceptanceCase:
    pack_id: str
    request: CandidateCertificationRequest
    neutral_request: CandidateCertificationRequest


@dataclass(frozen=True)
class _PoseObjective:
    site: str
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class _SmoothTrajectory:
    model: mujoco.MjModel
    times_s: np.ndarray
    qpos: np.ndarray

    def sample(self, time_s: float) -> ControlTarget:
        t = float(np.clip(time_s, self.times_s[0], self.times_s[-1]))
        right = min(
            max(int(np.searchsorted(self.times_s, t, side="right")), 1),
            len(self.times_s) - 1,
        )
        left = right - 1
        duration = float(self.times_s[right] - self.times_s[left])
        u = float(np.clip((t - self.times_s[left]) / duration, 0.0, 1.0))
        position_scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        velocity_scale = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
        acceleration_scale = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
        difference = self.qpos[right] - self.qpos[left]
        qpos = self.qpos[left] + position_scale * difference
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        qacc = np.zeros(self.model.nv, dtype=np.float64)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            qvel[dof_address] = velocity_scale * difference[qpos_address]
            qacc[dof_address] = acceleration_scale * difference[qpos_address]
        return ControlTarget(qpos=qpos, qvel=qvel, qacc=qacc)


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _joint_address(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"acceptance model is missing joint {name!r}")
    return int(model.jnt_qposadr[joint_id]), int(joint_id)


def _solve_pose(
    model: mujoco.MjModel,
    seed: np.ndarray,
    joint_names: tuple[str, ...],
    objectives: tuple[_PoseObjective, ...],
) -> np.ndarray:
    qpos_addresses: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        address, joint_id = _joint_address(model, name)
        qpos_addresses.append(address)
        if bool(model.jnt_limited[joint_id]):
            low, high = (float(value) for value in model.jnt_range[joint_id])
        else:
            low, high = -np.pi, np.pi
        lower.append(low + 1e-6)
        upper.append(high - 1e-6)
    site_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, objective.site)
        for objective in objectives
    )
    if any(site_id < 0 for site_id in site_ids):
        raise ValueError("acceptance IK references a missing palm site")
    data = mujoco.MjData(model)
    original = seed[qpos_addresses].copy()

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[:] = seed
        data.qpos[qpos_addresses] = values
        mujoco.mj_forward(model, data)
        result: list[float] = []
        for site_id, objective in zip(site_ids, objectives, strict=True):
            position_error = (data.site_xpos[site_id] - objective.position) / 0.004
            current = data.site_xmat[site_id].reshape(3, 3)
            orientation_error = 0.5 * sum(
                (
                    np.cross(current[:, axis], objective.rotation[:, axis])
                    for axis in range(3)
                ),
                start=np.zeros(3),
            )
            result.extend(position_error)
            result.extend(orientation_error / 0.06)
        result.extend((values - original) * 0.01)
        return np.asarray(result, dtype=np.float64)

    solved = least_squares(
        residual,
        np.clip(original, lower, upper),
        bounds=(lower, upper),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=350,
    )
    output = seed.copy()
    output[qpos_addresses] = solved.x
    data.qpos[:] = output
    mujoco.mj_forward(model, data)
    errors = [
        float(np.linalg.norm(data.site_xpos[site_id] - objective.position))
        for site_id, objective in zip(site_ids, objectives, strict=True)
    ]
    if max(errors, default=0.0) > 0.012:
        raise ValueError(f"physical bimanual pose is unreachable; site errors={errors}")
    return output


def _standing_seed(model: mujoco.MjModel, *, root_y: float = -0.05) -> np.ndarray:
    qpos = model.qpos0.copy()
    root_address, _ = _joint_address(model, "pelvis_free")
    qpos[root_address + 1] = root_y
    return qpos


def _palm_site_for_geom_center(
    side: str, center: tuple[float, float, float], rotation: np.ndarray
) -> np.ndarray:
    offset = np.asarray((-0.005 if side == "left" else 0.005, 0.043, 0.0))
    return np.asarray(center, dtype=np.float64) - rotation @ offset


_WORLD_ABSOLUTE_WXYZ = QuaternionConvention(
    order=QuaternionOrder.WXYZ,
    frame=CoordinateFrame.WORLD,
    meaning=QuaternionMeaning.ABSOLUTE,
)


def _task_keyframe(
    time_s: float,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    hard: bool = True,
) -> MotionKeyframeV2:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=np.float64).reshape(9))
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        ),
        rotation=Quaternion(
            values=tuple(float(value) for value in quaternion),
            convention=_WORLD_ABSOLUTE_WXYZ,
        ),
        hard=hard,
    )


def _task_position_keyframe(
    time_s: float, position: np.ndarray, *, hard: bool = True
) -> MotionKeyframeV2:
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        ),
        hard=hard,
    )


def _site_pose(
    model: mujoco.MjModel,
    site_name: str,
    *,
    joint_name: str | None = None,
    joint_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    if joint_name is not None:
        address, _ = _joint_address(model, joint_name)
        data.qpos[address] = joint_value
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"acceptance model is missing site {site_name!r}")
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def _compile_task_program(
    *,
    model: mujoco.MjModel,
    rig,  # type: ignore[no-untyped-def]
    program_id: str,
    scene_id: str,
    duration_s: float,
    tracks: tuple[MotionTrackV2, ...],
    contacts: tuple[ContactEdgeV2, ...],
    allowed_contact_pairs: tuple[tuple[str, str], ...],
):  # type: ignore[no-untyped-def]
    rest = _standing_seed(model)
    scene_rig = rig.manifest.model_copy(
        update={"rest_qpos": tuple(float(value) for value in rest)}
    )
    program = MotionProgramV2(
        program_id=program_id,
        source_text="Certified physical task-space acceptance trajectory.",
        duration_s=duration_s,
        rig_id=scene_rig.rig_id,
        scene_id=scene_id,
        seed=20260811,
        phases=(
            MotionPhaseV2(
                phase_id="approach",
                kind=PhaseKind.SETUP,
                start_s=0.0,
                end_s=duration_s * 0.24,
                energy=0.25,
            ),
            MotionPhaseV2(
                phase_id="interaction",
                kind=PhaseKind.ACTION,
                start_s=duration_s * 0.24,
                end_s=duration_s * 0.64,
                energy=0.45,
            ),
            MotionPhaseV2(
                phase_id="hold",
                kind=PhaseKind.HOLD,
                start_s=duration_s * 0.64,
                end_s=duration_s * 0.82,
                energy=0.2,
            ),
            MotionPhaseV2(
                phase_id="finish",
                kind=PhaseKind.RECOVERY,
                start_s=duration_s * 0.82,
                end_s=duration_s,
                energy=0.2,
            ),
        ),
        tracks=tracks,
        contacts=contacts,
    )
    return compile_motion_program(
        program,
        model,
        scene_rig,
        # The runtime remains fixed at 240 Hz; two task-space knots per second
        # keep the global nonlinear refinement bounded and are interpolated by
        # the runtime trajectory provider at every physics step.
        sample_hz=2,
        timing_profile=TimingProfile.NEUTRAL,
        allowed_contact_pairs=allowed_contact_pairs,
    )


def _configured_pack_source(
    xml: str, pack_id: str
) -> tuple[bytes, mujoco.MjModel]:
    root = ET.fromstring(xml)
    if pack_id == "container_lid":
        hinge = root.find(".//joint[@name='obj__container__lid_hinge']")
        support = root.find(".//geom[@name='support__container_table']")
        object_body = root.find(".//body[@name='obj__container__base']")
        if hinge is None or support is None or object_body is None:
            raise ValueError("container scene omitted acceptance geometry")
        # The panel lies at negative local Y, so -X makes the declared
        # positive 0..125 degree range open upward rather than into the box.
        hinge.set("axis", "-1 0 0")
        # A waist-height compact work table is reachable without moving or
        # welding the canonical human, and remains a realistic declared box.
        support.set("pos", "0 -0.44 0.85")
        support.set("size", "0.7 0.16 0.04")
        object_body.set("pos", "0 -0.34 0.905")
    elif pack_id == "two_handed_object":
        support = root.find(".//geom[@name='support__bar_rack']")
        object_body = root.find(".//body[@name='obj__carry_bar__bar']")
        if support is None or object_body is None:
            raise ValueError("carry-bar scene omitted acceptance geometry")
        # The forward handles move the rigid body's centre of mass toward the
        # operator, so the passive rack must support that declared footprint.
        # It remains well below the grips and does not constrain the free body.
        support.set("pos", "0 -0.45 1.02")
        support.set("size", "0.25 0.13 0.025")
        object_body.set("pos", "0 -0.45 1.08")
        worldbody = root.find("./worldbody")
        if worldbody is None:
            raise ValueError("carry-bar scene omitted its world body")
        for suffix, y in (("operator", -0.41), ("rear", -0.49)):
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"support__bar_cradle_{suffix}",
                    "type": "box",
                    "pos": f"0 {y:.3f} 1.055",
                    "size": "0.25 0.01 0.01",
                    "friction": "0.7 0.02 0.01",
                    "rgba": "0.25 0.25 0.27 1",
                },
            )
    else:
        raise ValueError(f"unsupported bimanual acceptance pack {pack_id!r}")
    corrected = ET.tostring(root, encoding="unicode")
    spec = mujoco.MjSpec.from_string(corrected)
    model = spec.compile()
    archive = BytesIO()
    mujoco.to_zip(spec, archive)
    return archive.getvalue(), model


def _export_validator(artifacts, rig, scene_model):  # type: ignore[no-untyped-def]
    source_glb = artifacts.read_bytes(rig.manifest.visual_asset)
    rig_model = mujoco.MjModel.from_xml_string(xml_for_profile("medium"))

    def validate(result):  # type: ignore[no-untyped-def]
        return validate_simulated_glb_round_trip(
            source_glb=source_glb,
            model_xml=xml_for_profile("medium"),
            rig=rig.manifest,
            times_s=result.trace.times_s,
            qpos=project_qpos_to_rig(scene_model, rig_model, result.trace.qpos),
        )

    return validate


def _request(
    *,
    request_id: str,
    source: bytes,
    model: mujoco.MjModel,
    trajectory: TargetProvider,
    duration_s: float,
) -> SimulationRequest:
    return SimulationRequest(
        model_xml=None,
        model_mjz=source,
        trajectory=trajectory,
        config=SimulationConfig(
            duration_s=duration_s,
            free_root_joint_name="pelvis_free",
            standing=canonical_standing_config(model),
        ),
        initial_qpos=trajectory.qpos[0],
        initial_qvel=np.zeros(model.nv),
        request_id=request_id,
    )


def _neutral_request(
    source: bytes, model: mujoco.MjModel, duration_s: float, request_id: str
) -> SimulationRequest:
    qpos = np.vstack((model.qpos0, model.qpos0))
    return _request(
        request_id=request_id,
        source=source,
        model=model,
        trajectory=_SmoothTrajectory(model, np.asarray((0.0, duration_s)), qpos),
        duration_s=duration_s,
    )


def build_two_handed_object_acceptance(
    artifacts: ContentAddressedArtifactStore,
) -> _AcceptanceCase:
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "two_handed_object", profile="medium", rig_reference=rig.reference, artifacts=artifacts
    )
    source, model = _configured_pack_source(scene.compiled.xml, "two_handed_object")
    left_rotation = right_rotation = np.eye(3)
    left_handle, _ = _site_pose(
        model, "obj__carry_bar__left_forward_grasp"
    )
    right_handle, _ = _site_pose(
        model, "obj__carry_bar__right_forward_grasp"
    )
    # Opposed front/back palm normals generate a real grip without routing
    # either lower arm through the long central capsule.
    contact_offset = 0.025 + 0.042 - 0.001
    approach_clearance = 0.045
    left_contact_center = left_handle + np.asarray((0.0, contact_offset, 0.0))
    right_contact_center = right_handle - np.asarray((0.0, contact_offset, 0.0))
    left_approach_center = left_contact_center + np.asarray(
        (0.0, approach_clearance, 0.0)
    )
    right_approach_center = right_contact_center - np.asarray(
        (0.0, approach_clearance, 0.0)
    )
    lift = np.asarray((0.0, 0.0, 0.20))
    times = (0.0, 1.2, 2.4, 3.2, 4.2, 5.0)
    contacts = (
        ContactEdgeV2(
            contact_id="left-bar-grasp",
            body_a="left_hand",
            body_b="obj__carry_bar__left_forward_handle",
            start_s=1.18,
            end_s=4.22,
        ),
        ContactEdgeV2(
            contact_id="right-bar-grasp",
            body_a="right_hand",
            body_b="obj__carry_bar__right_forward_handle",
            start_s=1.18,
            end_s=4.22,
        ),
    )
    tracks = tuple(
        MotionTrackV2(
            track_id=f"{side}-forward-handle",
            target=f"{side}_palm",
            owner=f"{side}_arm",
            interpolation=InterpolationKind.QUINTIC,
            keyframes=tuple(
                _task_position_keyframe(
                    time_s,
                    _palm_site_for_geom_center(side, tuple(center), rotation),
                )
                for time_s, center in zip(
                    times,
                    (
                        approach,
                        contact,
                        contact + lift,
                        contact + lift,
                        contact,
                        approach,
                    ),
                    strict=True,
                )
            ),
        )
        for side, rotation, approach, contact in (
            ("left", left_rotation, left_approach_center, left_contact_center),
            ("right", right_rotation, right_approach_center, right_contact_center),
        )
    )
    trajectory = _compile_task_program(
        model=model,
        rig=rig,
        program_id="physical-forward-handle-lift",
        scene_id=scene.manifest.scene_id,
        duration_s=5.0,
        tracks=tracks,
        contacts=contacts,
        allowed_contact_pairs=(
            ("left_hand", "obj__carry_bar__left_forward_handle"),
            ("right_hand", "obj__carry_bar__right_forward_handle"),
        ),
    )
    request = _request(
        request_id="accept-two-handed-object",
        source=source,
        model=model,
        trajectory=trajectory,
        duration_s=5.0,
    )
    object_geoms = frozenset(
        {
            "obj__carry_bar__bar__bar_capsule",
            "obj__carry_bar__left_forward_handle__left_handle_stem",
            "obj__carry_bar__left_forward_handle__left_handle_grip",
            "obj__carry_bar__right_forward_handle__right_handle_stem",
            "obj__carry_bar__right_forward_handle__right_handle_grip",
        }
    )
    effector_geoms = frozenset({"left_palm_collision", "right_palm_collision"})
    predicates = predicates_from_manifest(scene.manifest) + (
        GraspPredicate(object_geoms, effector_geoms, 1.1, 4.3, name="bilateral_grasp"),
        LiftPredicate("obj__carry_bar__bar", 0.18, complete_by_s=2.8),
        HoldPredicate("obj__carry_bar__bar", 2.5, 3.2, 0.16, max_vertical_drift_m=0.035),
        ReleasePredicate(object_geoms, effector_geoms, 4.6, settle_duration_s=0.25),
    )
    policy = policy_from_manifests(rig.manifest, scene.manifest)
    export = _export_validator(artifacts, rig, model)
    return _AcceptanceCase(
        "two_handed_object",
        CandidateCertificationRequest(request, policy, predicates, export, contacts),
        CandidateCertificationRequest(
            _neutral_request(source, model, 1.0, "neutral-two-handed-object"),
            policy,
            predicates_from_manifest(scene.manifest),
            export,
        ),
    )


def build_container_lid_acceptance(
    artifacts: ContentAddressedArtifactStore,
) -> _AcceptanceCase:
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "container_lid", profile="medium", rig_reference=rig.reference, artifacts=artifacts
    )
    source, model = _configured_pack_source(scene.compiled.xml, "container_lid")
    duration_s = 4.5
    times = (0.0, 1.1, 2.0, 2.9, 3.8, duration_s)
    contacts = (
        ContactEdgeV2(
            contact_id="left-container-stabilize",
            body_a="left_hand",
            body_b="obj__container__base",
            start_s=1.08,
            end_s=duration_s,
        ),
        ContactEdgeV2(
            contact_id="right-lid-knob-open",
            body_a="right_hand",
            body_b="obj__container__lid_knob",
            start_s=1.08,
            end_s=duration_s,
        ),
    )
    left_rotation = np.eye(3)
    left_centers = tuple(
        np.asarray((0.08, y, 1.005), dtype=np.float64)
        for y in (-0.08, -0.129, -0.129, -0.129, -0.129, -0.129)
    )
    right_targets: list[tuple[np.ndarray, np.ndarray]] = []
    for index, angle_degrees in enumerate((0.0, 0.0, 35.0, 70.0, 95.0, 95.0)):
        knob_position, knob_rotation = _site_pose(
            model,
            "obj__container__lid_knob",
            joint_name="obj__container__lid_hinge",
            joint_value=np.deg2rad(angle_degrees),
        )
        normal = knob_rotation[:, 2]
        separation = 0.025 + 0.018 - 0.001
        if index == 0:
            separation += 0.05
        right_center = knob_position - separation * normal
        right_targets.append(
            (
                _palm_site_for_geom_center(
                    "right", tuple(right_center), knob_rotation
                ),
                knob_rotation,
            )
        )
    tracks = (
        MotionTrackV2(
            track_id="left-digit-base-stabilize",
            target="left_palm",
            owner="left_arm",
            interpolation=InterpolationKind.QUINTIC,
            keyframes=tuple(
                _task_position_keyframe(
                    time_s,
                    _palm_site_for_geom_center(
                        "left", tuple(center), left_rotation
                    ),
                )
                for time_s, center in zip(times, left_centers, strict=True)
            ),
        ),
        MotionTrackV2(
            track_id="right-knob-open",
            target="right_palm",
            owner="right_arm",
            interpolation=InterpolationKind.QUINTIC,
            keyframes=tuple(
                _task_position_keyframe(time_s, position)
                for time_s, (position, rotation) in zip(
                    times, right_targets, strict=True
                )
            ),
        ),
    )
    trajectory = _compile_task_program(
        model=model,
        rig=rig,
        program_id="physical-lid-knob-open",
        scene_id=scene.manifest.scene_id,
        duration_s=duration_s,
        tracks=tracks,
        contacts=contacts,
        allowed_contact_pairs=(
            ("left_hand", "obj__container__base"),
            ("right_hand", "obj__container__lid_knob"),
        ),
    )
    request = _request(
        request_id="accept-container-lid",
        source=source,
        model=model,
        trajectory=trajectory,
        duration_s=duration_s,
    )
    base_geoms = frozenset(
        {
            "obj__container__base__bottom",
            "obj__container__base__left_wall",
            "obj__container__base__right_wall",
            "obj__container__base__front_wall",
            "obj__container__base__back_wall",
        }
    )
    lid_geoms = frozenset(
        {
            "obj__container__lid__lid_panel",
            "obj__container__lid_knob__knob_neck",
            "obj__container__lid_knob__knob_grip",
        }
    )
    left_effectors = frozenset(
        {
            "left_palm_collision",
            "left_thumb_distal_collision",
            "left_index_distal_collision",
            "left_middle_distal_collision",
            "left_ring_distal_collision",
            "left_little_distal_collision",
        }
    )
    right_effectors = frozenset(
        {
            "right_palm_collision",
            "right_thumb_distal_collision",
            "right_index_distal_collision",
            "right_middle_distal_collision",
            "right_ring_distal_collision",
            "right_little_distal_collision",
        }
    )
    predicates = predicates_from_manifest(scene.manifest) + (
        GraspPredicate(
            base_geoms,
            left_effectors,
            1.0,
            duration_s,
            min_distinct_effectors=1,
            name="left_base_stabilize",
        ),
        GraspPredicate(
            lid_geoms,
            right_effectors,
            1.0,
            duration_s,
            min_distinct_effectors=1,
            name="right_lid_knob_contact",
        ),
        ArticulatedCompletionPredicate(
            "obj__container__lid_hinge",
            np.deg2rad(80.0),
            np.deg2rad(5.0),
            comparator=JointComparator.AT_LEAST,
            name="lid_completion",
        ),
    )
    policy = policy_from_manifests(rig.manifest, scene.manifest)
    export = _export_validator(artifacts, rig, model)
    return _AcceptanceCase(
        "container_lid",
        CandidateCertificationRequest(request, policy, predicates, export, contacts),
        CandidateCertificationRequest(
            _neutral_request(source, model, 1.0, "neutral-container-lid"),
            policy,
            predicates_from_manifest(scene.manifest),
            export,
        ),
    )


def certify_acceptance(
    case: _AcceptanceCase,
    *,
    engine: CertificationEngine | None = None,
) -> BimanualAcceptanceReport:
    certification = engine or CertificationEngine(NativeMujocoRuntime())
    neutral = certification.certify(case.neutral_request)
    demonstration = certification.certify(case.request)
    return BimanualAcceptanceReport(case.pack_id, neutral, demonstration)


def authoritative_source_sha256(case: _AcceptanceCase) -> str:
    source = case.request.simulation.model_mjz
    if source is None:
        raise ValueError("acceptance case is not bound to MJZ")
    return sha256(source).hexdigest()
