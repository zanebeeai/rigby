from __future__ import annotations

import math
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from .analysis import _angular_kinematics, arm_landmarks
from .analysis.composite import composite_metrics as _composite_metrics
from .analysis.context import (
    AnalysisContext as _AnalysisContext,
)
from .analysis.context import (
    root_drift_policy as _root_drift_policy,
)
from .analysis.full_body import full_body_metrics as _full_body_metrics
from .analysis.geometry import line_segment_distance as _line_segment_distance
from .analysis.hand import hand_metrics as _hand_metrics
from .analysis.objects import handoff_metrics as _handoff_metrics
from .analysis.rig import (
    EGO_NEUTRAL_GAZE as _EGO_NEUTRAL_GAZE,
)
from .analysis.rig import (
    RIG_PROFILE,
)
from .analysis.rig import (
    identity_pose as _identity_pose,
)
from .analysis.safety import (
    clip_contract_violations as _clip_contract_violations,
)
from .analysis.safety import (
    safety_metrics as _safety_metrics,
)
from .analysis.semantic import semantic_cycle_assertion as _semantic_cycle_assertion
from .arm_plane import twist_about_local_y
from .clearance import (
    ForbiddenBody,
    basis_with_first_axis,
    bend_pole,
    bone_world_position,
    clear_wrist_target,
    hand_clearance_m,
    hinted_arm_solve,
    pushing_front_m,
)
from .grasp import (
    Box,
    Seat,
    arc_path,
    attach_offset,
    carried_transform,
    close_to_contact,
    elbow_direction,
    elbow_hang,
    grasp_finger_axis,
    grasp_palm_normal,
    hand_world,
    hand_world as _grasp_hand_world,
    rotate_toward,
    rotation_toward,
    solve_hand_along,
    solve_seat,
)
from .hand_mesh import hand_mesh
from .kinematics import rig_kinematics
from .models import (
    AffordanceRole,
    BodyAction,
    BodyClimbDirection,
    BodyObstacleMode,
    BodyRotationAxis,
    BodyRotationMode,
    BodySupportMode,
    BodyTarget,
    BonePose,
    ClipFrame,
    ClipResult,
    CompileRequest,
    ContactEvent,
    EffectorTarget,
    Failure,
    FailureCode,
    Hand,
    HandShape,
    Intent,
    MotionProgram,
    ObjectAction,
    ObjectInteractionStyle,
    PrimitiveKind,
    PrimitiveParameters,
    Provenance,
    Quat,
    SceneManifest,
    SceneObject,
    TrajectoryKind,
    TrajectoryPlane,
    Transform,
    Vec3,
)
from .physics import PhysicsOutcome, simulate_grasp
from .primitives import (
    MAX_WRIST_TWIST_RAD,
    TRUNK_YAW_DISTRIBUTION,
    arm_pose_from_target,
    arm_reach_m,
    counter_rotate_about_trunk,
    finger_assertions,
    forearm_shake_amplitude_rad,
    gesture_target,
    hand_pose,
    presentation_arc_amplitude_rad,
    shoulder_position,
    smoothstep,
    strike_path_target,
    strike_target,
    strike_trunk_yaw,
    subdivision_frames,
    thumb_to_fingertip_pose,
    trunk_yaw_poses,
    wrist_flourish_amplitude_rad,
)
from .thresholds import value_of

#: Nlerp is not constant-angular-speed near 180 degrees, so every compile path
#: raises its frame count until no inter-frame delta exceeds this.  Deliberately
#: stricter than the validator's discontinuity limit -- see
#: config/thresholds.v1.json.  Bound once here because the same rule appears on
#: four compile paths and they must not drift apart.
NLERP_SUBDIVISION_RAD = value_of("signal.discontinuity_generator_target_rad")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILER_VERSION = "rigby-compiler-0.4.0"
FULL_BODY_STANDING_ROOT_HEIGHT_M = -0.075


def _gesture_idle_pose() -> dict[str, Quat]:
    """A relaxed, symmetric base pose used instead of the rig's T-pose."""
    pose = _identity_pose()
    parameters = PrimitiveParameters(
        arm_height=0.0,
        arm_depth=0.0,
        elbow_swivel=0.10,
        wrist_pitch=0.0,
        wrist_yaw=0.0,
        wrist_roll=0.0,
        torso_participation=0.0,
    )
    for hand in (Hand.LEFT, Hand.RIGHT):
        side = 1.0 if hand == Hand.LEFT else -1.0
        # Keep the relaxed hands low and forward, as they would be in a
        # first-person interaction stance.  Unlike the source T-pose, this is
        # both anatomically relaxed and completely inside the 94-degree ego
        # camera frustum when a presentation begins.
        target = Vec3(x=side * 0.27, y=1.14, z=0.24)
        arm, _ = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            parameters,
            present_hand=False,
        )
        pose.update(arm)
        pose.update(hand_pose(hand, HandShape.OPEN, parameters))
    return pose


def _full_body_idle_pose() -> dict[str, Quat]:
    """A relaxed standing pose whose arms can swing beside the torso.

    The interaction idle intentionally keeps both hands high and forward so a
    first-person hand gesture is immediately visible.  Reusing it for walking
    made the elbows stay in a tabletop pose and caused the hands to sweep
    across the abdomen.  Whole-body skills start lower and wider instead,
    while retaining the same open-hand defaults and exact rig mapping.
    """

    pose = _identity_pose()
    parameters = PrimitiveParameters(
        elbow_swivel=-0.04,
        wrist_pitch=0.0,
        wrist_yaw=0.0,
        wrist_roll=0.0,
        torso_participation=0.0,
    )
    for hand in (Hand.LEFT, Hand.RIGHT):
        side = 1.0 if hand == Hand.LEFT else -1.0
        target = Vec3(x=side * 0.27, y=0.94, z=0.08)
        arm, _ = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            parameters,
            present_hand=False,
        )
        pose.update(arm)
        pose.update(hand_pose(hand, HandShape.OPEN, parameters))
    return pose


def _inactive_guard_pose(active_hand: Hand) -> dict[str, Quat]:
    """Keep the non-striking hand in a readable defensive guard."""
    hand = Hand.RIGHT if active_hand == Hand.LEFT else Hand.LEFT
    side = 1.0 if hand == Hand.LEFT else -1.0
    parameters = PrimitiveParameters(
        elbow_swivel=0.28,
        wrist_pitch=-0.04,
        finger_curl=0.18,
        thumb_opposition=0.92,
    )
    target = Vec3(x=side * 0.12, y=1.43, z=0.17)
    arm, _ = arm_pose_from_target(
        hand,
        shoulder_position(hand),
        target,
        parameters,
        present_hand=False,
    )
    arm.update(hand_pose(hand, HandShape.FIST, parameters))
    return arm


def _nlerp(first: Quat, second: Quat, alpha: float) -> Quat:
    a = np.asarray(first.as_list(), dtype=float)
    b = np.asarray(second.as_list(), dtype=float)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    value = a * (1.0 - alpha) + b * alpha
    value /= max(float(np.linalg.norm(value)), 1e-12)
    return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))


def _slerp(first: Quat, second: Quat, alpha: float) -> Quat:
    """Constant-rate geodesic blend between two local rotations.

    ``_nlerp`` traverses the same geodesic but speeds up mid-path by a factor
    that grows with the endpoint angle (tan(theta/2)/(theta/2) at the
    midpoint).  For the large reorientations the humeral-roll solver produces
    between an arms-down and an arms-overhead solution (2.4-2.6 rad), that
    warp alone pushes per-frame deltas past the discontinuity gate even when
    the constant-rate path fits with margin, so blends covering such spans
    must advance at constant angular rate.
    """
    a = Rotation.from_quat(first.as_list())
    b = Rotation.from_quat(second.as_list())
    relative = a.inv() * b
    value = a * Rotation.from_rotvec(relative.as_rotvec() * float(alpha))
    xyzw = value.as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


#: The bones whose local deltas carry the humeral roll and its forearm
#: pronation compensation.  Their longitudinal twists are authored as a
#: cancelling pair, so interpolation must keep them synchronized.
_ARM_TWIST_BLEND_BONES = frozenset(
    {"leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm"}
)


def _swing_twist_blend(first: Quat, second: Quat, alpha: float) -> Quat:
    """Blend an arm bone's local delta with its longitudinal twist unmixed.

    The hinge-in-plane solver pairs every humeral roll with a forearm
    pronation that restores the pre-roll hand world orientation -- but only
    at solved keyframes.  A quaternion geodesic between two solutions whose
    rolls differ mixes each bone's twist into its swing on its own schedule,
    so mid-blend the pair no longer cancels: the hand world orientation
    transiently swings by up to the roll difference, which a carried object
    amplifies by its grip lever arm.  Decomposing each endpoint into swing
    and twist about the bone's long axis and interpolating the twist angles
    linearly keeps the upper-arm roll and its forearm compensation advancing
    in lockstep, so the blended hand tracks the calibrated pre-roll path the
    endpoints were compensated toward.
    """
    a = Rotation.from_quat(first.as_list())
    b = Rotation.from_quat(second.as_list())
    twist_a = twist_about_local_y(a)
    twist_b = twist_about_local_y(b)
    swing_a = a * Rotation.from_rotvec([0.0, -twist_a, 0.0])
    swing_b = b * Rotation.from_rotvec([0.0, -twist_b, 0.0])
    relative = swing_a.inv() * swing_b
    swing = swing_a * Rotation.from_rotvec(relative.as_rotvec() * float(alpha))
    twist = twist_a + float(alpha) * math.remainder(twist_b - twist_a, math.tau)
    value = swing * Rotation.from_rotvec([0.0, twist, 0.0])
    xyzw = value.as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def _local_rotation_offset(value: Quat, rotation_vector: list[float]) -> Quat:
    rotation = Rotation.from_quat(value.as_list()) * Rotation.from_rotvec(rotation_vector)
    xyzw = rotation.as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def _hips_world_euler_quat(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Quat:
    world_delta = Rotation.from_euler("xyz", [x, y, z]).as_matrix()
    return rig_kinematics().world_delta_quat("hips", world_delta)


def apply_overrides(request: CompileRequest) -> tuple[SceneManifest, MotionProgram]:
    raw = request.parameter_overrides.model_dump(exclude_none=True)
    scene = request.scene.model_copy(deep=True)
    program = request.program.model_copy(deep=True)
    if program.intent == Intent.SEQUENCE:
        next_steps: list[MotionProgram] = []
        for step in program.steps:
            scene, next_step = apply_overrides(
                CompileRequest(
                    scene=scene,
                    program=step,
                    parameter_overrides=request.parameter_overrides,
                    persist=False,
                )
            )
            next_steps.append(next_step)
        program.steps = next_steps
        program.hands = list(
            dict.fromkeys(
                hand
                for step in next_steps
                for hand in (step.hands or [step.hand])
            )
        )
        if program.hands:
            program.hand = program.hands[0]
        return scene, program
    if "hand" in raw:
        next_hand = raw.pop("hand")
        if program.intent == Intent.COMPOSITE and len(program.hands) == 1:
            previous_hand = program.hands[0]
            if next_hand != previous_hand:
                program.hand = next_hand
                program.hands = [next_hand]
                for primitive in program.primitives:
                    primitive.effectors = [
                        target.model_copy(
                            update={
                                "hand": next_hand,
                                "target_x": -target.target_x,
                                "elbow_swivel": -target.elbow_swivel,
                                "wrist_yaw": -target.wrist_yaw,
                            }
                        )
                        if target.hand == previous_hand
                        else target
                        for target in primitive.effectors
                    ]
        elif program.intent != Intent.COMPOSITE and next_hand != program.hand:
            program.hand = next_hand
            # Lateral target and yaw are authored in hand-relative space.
            # Preserve inward/outward semantics when swapping effectors.
            for primitive in program.primitives:
                parameters = primitive.parameters
                primitive.parameters = parameters.model_copy(
                    update={
                        "lateral_offset": -parameters.lateral_offset,
                        "wrist_yaw": -parameters.wrist_yaw,
                    }
                )

    scene_keys = {
        "block_x",
        "block_y",
        "block_z",
        "block_width_m",
        "block_height_m",
        "block_depth_m",
        "block_mass_kg",
        "block_friction",
    }
    scene_raw = {key: raw.pop(key) for key in list(raw) if key in scene_keys}
    if scene_raw:
        target_ids = {item.object_id for item in program.primitives if item.object_id}
        target = next((item for item in scene.objects if item.id in target_ids), scene.objects[0])
        position = target.transform.translation.model_copy(
            update={
                "x": scene_raw.get("block_x", target.transform.translation.x),
                "y": scene_raw.get("block_y", target.transform.translation.y),
                "z": scene_raw.get("block_z", target.transform.translation.z),
            }
        )
        target.transform = target.transform.model_copy(update={"translation": position})
        target.dimensions_m = target.dimensions_m.model_copy(
            update={
                "x": scene_raw.get("block_width_m", target.dimensions_m.x),
                "y": scene_raw.get("block_height_m", target.dimensions_m.y),
                "z": scene_raw.get("block_depth_m", target.dimensions_m.z),
            }
        )
        if "block_mass_kg" in scene_raw:
            target.mass_kg = scene_raw["block_mass_kg"]
        if "block_friction" in scene_raw:
            target.friction = scene_raw["block_friction"]
    body_key_map = {
        "body_distance_m": "distance_m",
        "body_turn_degrees": "turn_degrees",
        "body_height_m": "height_m",
        "body_cycles": "cycles",
        "body_intensity": "intensity",
    }
    body_raw = {
        body_key_map[key]: raw.pop(key)
        for key in list(raw)
        if key in body_key_map
    }
    if body_raw:
        for primitive in program.primitives:
            if primitive.kind == PrimitiveKind.BODY and primitive.body is not None:
                primitive.body = primitive.body.model_copy(update=body_raw)
    object_key_map = {
        "object_distance_m": "distance_m",
        "object_apex_height_m": "apex_height_m",
        "object_spin_turns": "spin_turns",
        "object_contact_height_m": "contact_height_m",
        "object_contact_depth_m": "contact_depth_m",
        "object_landing_height_m": "landing_height_m",
    }
    object_raw = {
        object_key_map[key]: raw.pop(key)
        for key in list(raw)
        if key in object_key_map
    }
    if object_raw and program.object_motion is not None:
        program.object_motion = program.object_motion.model_copy(update=object_raw)
    if raw:
        for primitive in program.primitives:
            primitive.parameters = primitive.parameters.model_copy(update=raw)
    return scene, program


def _program_params(program: MotionProgram) -> Any:
    return program.primitives[0].parameters if program.primitives else None


def _slider_observables(scene: SceneManifest, program: MotionProgram) -> dict[str, float]:
    params = _program_params(program)
    result: dict[str, float] = {}
    if params is not None:
        for key, value in params.model_dump().items():
            result[key] = float(value)
        result["duration_s"] = float(sum(item.parameters.duration_s for item in program.primitives))
    shake = next((item.parameters for item in program.primitives if item.kind == PrimitiveKind.SHAKE), None)
    if shake is not None:
        result["forearm_rotation_amplitude"] = float(shake.wrist_shake_amplitude)
        result["forearm_rotation_cycles"] = float(shake.wrist_shake_cycles)
        # Retain the serialized POC names so older result manifests remain editable.
        result["wrist_shake_amplitude"] = float(shake.wrist_shake_amplitude)
        result["wrist_shake_cycles"] = float(shake.wrist_shake_cycles)
        result["shake_duration_s"] = float(shake.duration_s)
    body = next(
        (item.body for item in program.primitives if item.kind == PrimitiveKind.BODY and item.body),
        None,
    )
    if body is not None:
        result.update(
            {
                "body_distance_m": float(body.distance_m),
                "body_turn_degrees": float(body.turn_degrees),
                "body_height_m": float(body.height_m),
                "body_cycles": float(body.cycles),
                "body_intensity": float(body.intensity),
            }
        )
    result["handedness"] = -1.0 if program.hand.value == "left" else 1.0
    target_ids = {item.object_id for item in program.primitives if item.object_id}
    block = next((item for item in scene.objects if item.id in target_ids), scene.objects[0])
    result.update(
        {
            "block_x": block.transform.translation.x,
            "block_y": block.transform.translation.y,
            "block_z": block.transform.translation.z,
            "block_width_m": block.dimensions_m.x,
            "block_height_m": block.dimensions_m.y,
            "block_depth_m": block.dimensions_m.z,
            "block_mass_kg": block.mass_kg,
            "block_friction": block.friction,
        }
    )
    return result


#: Fraction of a reach spent travelling to the point above the target; the
#: rest is the descent onto it.
_REACH_OVER_FRACTION = 0.7


def _reach_over_path(start: Vec3, end: Vec3, alpha: float, rise_m: float) -> Vec3:
    """A reach that arrives from above: over to a point ``rise_m`` above the
    target, then straight down onto it.

    A straight line from the resting hand to a target beside a block on a
    table sweeps the fingers through the block's side, where the clearance
    has to throw the wrist about between one face and the next. Coming down
    from above meets only the top, once.
    """

    over = Vec3(x=end.x, y=end.y + rise_m, z=end.z)
    if alpha < _REACH_OVER_FRACTION:
        blend = smoothstep(alpha / _REACH_OVER_FRACTION, 0.0)
        return Vec3(
            x=start.x + (over.x - start.x) * blend,
            y=start.y + (over.y - start.y) * blend,
            z=start.z + (over.z - start.z) * blend,
        )
    blend = smoothstep((alpha - _REACH_OVER_FRACTION) / (1.0 - _REACH_OVER_FRACTION), 0.0)
    return Vec3(x=over.x, y=over.y + (end.y - over.y) * blend, z=over.z)



def _push_destination_wall(
    original_position: np.ndarray,
    original_rotation: Rotation,
    object_half_extents: np.ndarray,
    push_direction: np.ndarray,
    distance_m: float,
) -> ForbiddenBody:
    """The pushed block's trailing face at its destination, extruded forward.

    Only as tall and as wide as the block, so only skin that would be pushing
    the block is held at it; fingers above the block pass over it the way
    they pass over the block itself.
    """

    rotation_matrix = original_rotation.as_matrix()
    trailing_extent = float(np.dot(np.abs(push_direction @ rotation_matrix), object_half_extents))
    vertical_extent = float(np.dot(np.abs(rotation_matrix[1]), object_half_extents))
    lateral = np.cross(np.asarray([0.0, 1.0, 0.0]), push_direction)
    lateral_extent = float(np.dot(np.abs(lateral @ rotation_matrix), object_half_extents))
    destination = original_position + push_direction * distance_m
    centre = destination - push_direction * trailing_extent + push_direction * 1.0
    centre[1] = destination[1]
    basis = basis_with_first_axis(push_direction)
    half = np.asarray([1.0, lateral_extent, lateral_extent], dtype=float)
    half[1 + int(np.argmax(np.abs(basis[1, 1:])))] = vertical_extent
    return ForbiddenBody(
        name="push_destination",
        centre=centre,
        half_extents=half,
        rotation=basis,
        escape_along=-push_direction,
        margin_m=0.0,
    )


def _fk_arm_landmarks(
    frame: ClipFrame, hand: Hand
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Rotation]:
    """Shoulder, elbow, wrist and hand orientation from the rig's own FK.

    ``arm_landmarks`` rebuilds the arm from a fixed rest shoulder, so it cannot
    see the trunk. An object attached through it rides a wrist that is not
    where the rendered wrist is whenever the chest turns, and slides through
    the rendered hand by the difference. The rig's world matrices are the
    rendered truth, so a carried object is attached to those.
    """

    kinematics = rig_kinematics()
    matrices = kinematics.world_matrices(frame.bones)
    node = kinematics.node_by_canonical
    prefix = hand.value
    hand_matrix = matrices[node[f"{prefix}Hand"]]
    return (
        matrices[node[f"{prefix}UpperArm"]][:3, 3].copy(),
        matrices[node[f"{prefix}LowerArm"]][:3, 3].copy(),
        hand_matrix[:3, 3].copy(),
        Rotation.from_matrix(hand_matrix[:3, :3]),
    )



#: Object-interaction phases in which the hand is travelling to or from the
#: object through free space, where a straight wrist path with per-frame
#: clearance replaces the joint-space blend. Everything else either carries
#: the object (it rides the wrist) or is a swing whose arc is authored.
_INTERACTION_PATH_PHASES = frozenset(
    {
        PrimitiveKind.REACH,
        PrimitiveKind.PRESHAPE,
        PrimitiveKind.CONTACT,
        PrimitiveKind.CLOSE,
    }
)


def _interaction_forbidden_bodies(
    scene: SceneManifest,
    target_object: SceneObject,
    kind: PrimitiveKind,
    object_position: np.ndarray,
    object_rotation: Rotation,
    attached: bool,
    approach: np.ndarray | None = None,
) -> list[ForbiddenBody]:
    """What the hand must stay out of at one moment of an object interaction.

    The support surface, always, left through its top. The object only while
    it is free: attached, it rides the wrist and clearing against it would
    chase itself. Touching is allowed at contact and close, nowhere else.
    """

    bodies: list[ForbiddenBody] = []
    support = scene.support_surface()
    if support is not None:
        top = next(
            (socket for socket in support.sockets if socket.role == AffordanceRole.SUPPORT),
            None,
        )
        bodies.append(
            ForbiddenBody.from_scene_object(
                support, escape_along=top.approach_normal if top is not None else None
            )
        )
    if not attached and approach is not None and kind != PrimitiveKind.RECOVER:
        # Arriving to push: the hand may reach the block's trailing face and
        # nothing past it at block height, whatever part of the hand gets
        # there first. The box alone lets fingers over the top count as
        # "clear" and then shove the block by their whole length.
        rotation_matrix = object_rotation.as_matrix()
        half = np.asarray(target_object.dimensions_m.as_list(), dtype=float) * 0.5
        trailing_extent = float(np.dot(np.abs(approach @ rotation_matrix), half))
        vertical_extent = float(np.dot(np.abs(rotation_matrix[1]), half))
        lateral = np.cross(np.asarray([0.0, 1.0, 0.0]), approach)
        lateral_extent = float(np.dot(np.abs(lateral @ rotation_matrix), half))
        basis = basis_with_first_axis(approach)
        # Full width, not the block's: the hand sweeps in from the side, and a
        # wall only as wide as the block would let it get ahead of the
        # trailing plane beside the block and then throw it back by a hand's
        # length the frame it lines up.
        extents = np.asarray([1.0, 1.0, 1.0], dtype=float)
        extents[1 + int(np.argmax(np.abs(basis[1, 1:])))] = vertical_extent
        centre = object_position - approach * trailing_extent + approach * 1.0
        centre[1] = object_position[1]
        bodies.append(
            ForbiddenBody(
                name=target_object.id,
                centre=centre,
                half_extents=extents,
                rotation=basis,
                escape_along=-approach,
                margin_m=0.0 if kind in (PrimitiveKind.CONTACT, PrimitiveKind.CLOSE) else None,
            )
        )
        return bodies
    if not attached:
        quat = object_rotation.as_quat()
        # Withdrawing, the hand goes up and over -- world up, whatever way a
        # rolled block is lying, expressed here in the block's own frame
        # because that is the frame the escape is read in. A nearest-face
        # escape would flip between the top and a side as the hand crosses
        # the edge, and the arm would jerk with it.
        world_up_local = object_rotation.inv().apply([0.0, 1.0, 0.0])
        bodies.append(
            ForbiddenBody.from_scene_object(
                target_object,
                escape_along=(
                    Vec3(x=float(world_up_local[0]), y=float(world_up_local[1]), z=float(world_up_local[2]))
                    if kind == PrimitiveKind.RECOVER
                    else None
                ),
                transform=Transform(
                    translation=Vec3(
                        x=float(object_position[0]),
                        y=float(object_position[1]),
                        z=float(object_position[2]),
                    ),
                    rotation=Quat(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
                ),
                margin_m=0.0 if kind in (PrimitiveKind.CONTACT, PrimitiveKind.CLOSE) else None,
            )
        )
    return bodies


def _physics_for(program: MotionProgram, scene: SceneManifest) -> PhysicsOutcome:
    object_id = next(item.object_id for item in program.primitives if item.object_id is not None)
    block = scene.object_by_id(object_id)
    if block is None:
        raise KeyError(object_id)
    close_params = next(item.parameters for item in program.primitives if item.kind == PrimitiveKind.CLOSE)
    lift_params = next(item.parameters for item in program.primitives if item.kind == PrimitiveKind.LIFT)
    hold_params = next(item.parameters for item in program.primitives if item.kind == PrimitiveKind.HOLD)
    attempts = (
        (close_params.grip_force, close_params.thumb_opposition),
        (min(1.0, close_params.grip_force + 0.12), close_params.thumb_opposition),
        (min(1.0, close_params.grip_force + 0.22), min(1.0, close_params.thumb_opposition + 0.12)),
        (1.0, 1.0),
        (0.88, 0.92),
        (0.95, 0.82),
        (0.82, 1.0),
        (0.9, 0.9),
    )
    outcome: PhysicsOutcome | None = None
    for grip_force, thumb_opposition in attempts:
        outcome = simulate_grasp(
            block=block,
            hand=program.hand,
            lift_height_m=lift_params.lift_height_m,
            hold_duration_s=hold_params.hold_duration_s,
            grip_force=grip_force,
            thumb_opposition=thumb_opposition,
            digit_curl_adjustments={
                "thumb": close_params.thumb_curl,
                "index": close_params.index_curl,
                "middle": close_params.middle_curl,
                "ring": close_params.ring_curl,
                "little": close_params.little_curl,
            },
            fps=scene.fps,
            support=scene.support_surface(),
        )
        if outcome.success:
            break
    assert outcome is not None
    outcome.metrics["grasp_attempts"] = attempts.index((grip_force, thumb_opposition)) + 1
    return outcome



def _failure_result(
    scene: SceneManifest,
    program: MotionProgram,
    code: FailureCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> ClipResult:
    metrics = _base_metrics()
    observables = _slider_observables(scene, program)
    return ClipResult(
        success=False,
        fps=scene.fps,
        duration_s=0.0,
        frames=[],
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=Failure(code=code, message=message, details=details or {}, recoverable=False),
    )


def _base_metrics() -> dict[str, Any]:
    return {
        "finger_assertions": {},
        "lift_height_m": 0.0,
        "lost_table_contact": True,
        "opposing_contacts": False,
        "hold_duration_s": 0.0,
        "vertical_drift_m": 0.0,
        "palm_relative_slip_m": 0.0,
        "weld_used": False,
        "joint_limit_violations": 0,
        "root_drift_m": 0.0,
        "foot_drift_m": 0.0,
        "unresolved_non_hand_collisions": 0,
        "nan_count": 0,
        "discontinuities": 0,
        "max_penetration_m": 0.0,
    }


def _provenance(program: MotionProgram) -> Provenance:
    return Provenance(
        rig_id="mesh2motion-human-vrm1",
        rig_asset="assets/models/human-male.glb",
        compiler_version=COMPILER_VERSION,
        physics_engine="MuJoCo",
        physics_version=version("mujoco"),
        planner_provider="compile-only",
        planner_model="none",
        model_calls=0,
        seed=program.seed,
        physics_model={
            "type": "cartesian-palm articulated five-digit contact proxy",
            "parallel_gripper_proxy": False,
            "free_block_joint": True,
            "independent_contact_digits": ["thumb", "index", "middle", "ring", "little"],
            "digits": {
                digit: {"independently_actuated": True, "contactable": True}
                for digit in ("thumb", "index", "middle", "ring", "little")
            },
            "visual_proxy_relation": "visual wrist target follows identical object-relative lift path",
        },
        coordinate_frames={
            "application": "glTF Y-up, forward +Z, meters",
            "simulation": "MuJoCo Z-up, forward -Y, meters",
            "app_to_sim_position": ["x", "-z", "y"],
            "sim_to_app_position": ["x", "z", "-y"],
            "measured_roundtrip_error_m": 0.0,
            "app_up_axis": "Y",
            "simulation_up_axis": "Z",
            "explicit_trajectory_conversion": True,
            "trajectory_roundtrip_error_m": 0.0,
        },
    )


def _composite_workspace_target(target: EffectorTarget) -> Vec3:
    """Map a normalized body-relative target into the calibrated arm workspace."""

    # Face- and chest-height gestures use the original visible interaction
    # workspace. Explicit overhead skills opt into a taller mapping; keeping
    # that distinction typed prevents a generic word such as "high" from
    # pushing salutes or beckoning gestures above the egocentric frustum.
    target_y = (
        1.30 + target.target_y * 0.54
        if target.reach_overhead and target.target_y >= 0.0
        else 1.30 + target.target_y * 0.22
    )
    hand_target = np.asarray(
        [
            target.target_x * 0.34,
            target_y,
            0.28 + target.target_z * 0.15,
        ],
        dtype=float,
    )
    shoulder = np.asarray(shoulder_position(target.hand).as_list(), dtype=float)
    offset = hand_target - shoulder
    maximum_reach = arm_reach_m(target.hand) - 0.018
    distance = float(np.linalg.norm(offset))
    if distance > maximum_reach:
        hand_target = shoulder + offset * (maximum_reach / max(distance, 1e-8))
    return Vec3(x=float(hand_target[0]), y=float(hand_target[1]), z=float(hand_target[2]))


def _composite_parameters(
    primitive_parameters: PrimitiveParameters,
    target: EffectorTarget,
) -> PrimitiveParameters:
    return primitive_parameters.model_copy(
        update={
            "elbow_swivel": target.elbow_swivel,
            "wrist_pitch": target.wrist_pitch,
            "wrist_yaw": target.wrist_yaw,
            "wrist_roll": target.wrist_roll,
        }
    )


def _trajectory_target(
    start: Vec3,
    center: Vec3,
    target: EffectorTarget,
    primitive_kind: TrajectoryKind,
    plane: TrajectoryPlane,
    progress: float,
    alpha: float,
    amplitude_m: float,
    cycles: float,
) -> Vec3:
    start_value = np.asarray(start.as_list(), dtype=float)
    center_value = np.asarray(center.as_list(), dtype=float)
    linear = start_value * (1.0 - alpha) + center_value * alpha
    if primitive_kind == TrajectoryKind.HOLD:
        value = center_value
    elif primitive_kind == TrajectoryKind.ARC:
        curve = math.sin(math.pi * progress) * amplitude_m
        axis = {
            TrajectoryPlane.FRONTAL: np.asarray([0.0, 0.0, 1.0]),
            TrajectoryPlane.SAGITTAL: np.asarray([1.0, 0.0, 0.0]),
            TrajectoryPlane.HORIZONTAL: np.asarray([0.0, 1.0, 0.0]),
        }[plane]
        value = linear + axis * curve
    elif primitive_kind in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}:
        envelope = math.sin(math.pi * progress) ** 2
        angle = 2.0 * math.pi * (
            cycles * progress + target.phase_offset_cycles
        )
        if primitive_kind == TrajectoryKind.CIRCLE:
            first, second = {
                TrajectoryPlane.FRONTAL: (
                    np.asarray([1.0, 0.0, 0.0]),
                    np.asarray([0.0, 1.0, 0.0]),
                ),
                TrajectoryPlane.SAGITTAL: (
                    np.asarray([0.0, 1.0, 0.0]),
                    np.asarray([0.0, 0.0, 1.0]),
                ),
                TrajectoryPlane.HORIZONTAL: (
                    np.asarray([1.0, 0.0, 0.0]),
                    np.asarray([0.0, 0.0, 1.0]),
                ),
            }[plane]
            offset = (first * math.cos(angle) + second * math.sin(angle)) * amplitude_m
        else:
            axis = {
                TrajectoryPlane.FRONTAL: np.asarray([1.0, 0.0, 0.0]),
                TrajectoryPlane.SAGITTAL: np.asarray([0.0, 1.0, 0.0]),
                TrajectoryPlane.HORIZONTAL: np.asarray([0.0, 0.0, 1.0]),
            }[plane]
            offset = axis * math.sin(angle) * amplitude_m
        value = center_value + offset * envelope
    else:
        value = linear
    return Vec3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _smooth_composite_joint_paths(
    frames: list[ClipFrame],
    hands: list[Hand],
    *,
    passes: int = 2,
    include_legs: bool = False,
) -> None:
    """Remove analytic-IK chatter without changing the authored wrist path.

    The two-bone solve is deterministic but can change its pole solution
    sharply near a workspace singularity.  A short, sign-aware quaternion
    filter gives the same kind of joint-space continuity a runtime animation
    system would normally apply after Cartesian IK.  Endpoints are retained so
    the relaxed start and recovery pose remain exact.
    """

    if len(frames) < 5 or passes <= 0:
        return
    bone_names = ["chest"] + [
        f"{hand.value}{suffix}"
        for hand in hands
        for suffix in ("UpperArm", "LowerArm", "Hand")
    ]
    if include_legs:
        bone_names.extend(
            f"{side}{suffix}"
            for side in (Hand.LEFT.value, Hand.RIGHT.value)
            for suffix in ("UpperLeg", "LowerLeg", "Foot")
        )
    for bone_name in bone_names:
        values = np.asarray(
            [frame.bones[bone_name].rotation.as_list() for frame in frames],
            dtype=float,
        )
        for index in range(1, len(values)):
            if float(np.dot(values[index - 1], values[index])) < 0.0:
                values[index] *= -1.0
        original_first = values[0].copy()
        original_last = values[-1].copy()
        for _ in range(passes):
            filtered = values.copy()
            filtered[1:-1] = (
                values[:-2] * 0.25
                + values[1:-1] * 0.50
                + values[2:] * 0.25
            )
            norms = np.linalg.norm(filtered, axis=1, keepdims=True)
            values = filtered / np.maximum(norms, 1e-12)
            values[0] = original_first
            values[-1] = original_last
        for frame, value in zip(frames, values, strict=True):
            frame.bones[bone_name] = BonePose(
                rotation=Quat(
                    x=float(value[0]),
                    y=float(value[1]),
                    z=float(value[2]),
                    w=float(value[3]),
                )
            )


def _body_action_pose(
    base: dict[str, Quat],
    target: BodyTarget,
    progress: float,
    alpha: float,
    root_yaw_rad: float,
    ingress: float = 1.0,
) -> dict[str, Quat]:
    """Evaluate one procedural whole-body skill in local joint space.

    ``ingress`` is the caller's phase entry blend.  Since the caller composes
    this articulation on top of its start-pose ease rather than attenuating
    it, any articulation term that is non-zero at ``progress == 0`` (a
    constant lean, a guard arm pose, a support-chain reset) must scale itself
    by ``ingress`` so the skill still enters from the base pose.
    """

    pose = base.copy()
    intensity = target.intensity
    phase_scale = math.pi if target.action in {BodyAction.WALK, BodyAction.RUN} else 2.0 * math.pi
    phase = phase_scale * target.cycles * progress
    pelvis_pitch = 0.0
    pelvis_yaw = root_yaw_rad
    pelvis_roll = 0.0

    if target.action in {BodyAction.WALK, BodyAction.RUN}:
        run_scale = 1.30 if target.action == BodyAction.RUN else 1.0
        swing = math.sin(phase) * 0.52 * intensity * run_scale
        left_knee = max(0.0, -math.sin(phase)) * 0.82 * intensity * run_scale
        right_knee = max(0.0, math.sin(phase)) * 0.82 * intensity * run_scale
        pose["leftUpperLeg"] = _local_rotation_offset(pose["leftUpperLeg"], [swing, 0.0, 0.0])
        pose["rightUpperLeg"] = _local_rotation_offset(pose["rightUpperLeg"], [-swing, 0.0, 0.0])
        pose["leftLowerLeg"] = _local_rotation_offset(pose["leftLowerLeg"], [left_knee, 0.0, 0.0])
        pose["rightLowerLeg"] = _local_rotation_offset(pose["rightLowerLeg"], [right_knee, 0.0, 0.0])
        pose["leftFoot"] = _local_rotation_offset(pose["leftFoot"], [-left_knee * 0.42, 0.0, 0.0])
        pose["rightFoot"] = _local_rotation_offset(pose["rightFoot"], [-right_knee * 0.42, 0.0, 0.0])
        if target.action == BodyAction.RUN:
            # A runner's bent, contralateral arm drive is one of the strongest
            # visual cues separating a run from a sped-up walk.  Driving only
            # the upper-arm joint left the elbows nearly straight and made the
            # flight phase read as a compact hop.  Solve both arms to close
            # wrist targets instead, so elbow flexion emerges from the same
            # anatomical IK used by hand-authored gestures.
            stride_phase = math.sin(phase) * min(1.15, intensity)
            for hand in (Hand.LEFT, Hand.RIGHT):
                side = 1.0 if hand == Hand.LEFT else -1.0
                advance = -stride_phase if hand == Hand.LEFT else stride_phase
                wrist_target = Vec3(
                    x=side * (0.30 + 0.025 * abs(advance)),
                    y=1.08 + 0.13 * max(0.0, advance) - 0.035 * max(0.0, -advance),
                    z=0.10 + 0.19 * advance,
                )
                arm_parameters = PrimitiveParameters(
                    elbow_swivel=-0.10,
                    wrist_pitch=0.0,
                    wrist_yaw=0.0,
                    wrist_roll=0.0,
                    torso_participation=0.0,
                )
                arm_pose, _ = arm_pose_from_target(
                    hand,
                    shoulder_position(hand),
                    wrist_target,
                    arm_parameters,
                    present_hand=False,
                    # A runner's elbow points down-back-out through the whole
                    # stride.  The default pole puts it forward, which at the
                    # back-swing extreme demands more humeral twist than the
                    # (-95, 95) shoulder band allows and saturates the forearm
                    # pronation budget; this pole keeps every stride sample
                    # inside both with margin.
                    elbow_pole=Vec3(x=side * 0.6, y=-0.55, z=-0.5),
                )
                # The guard is a full arm pose from the first stride sample,
                # so it owns its entry ramp now that the caller no longer
                # attenuates articulation with the phase ingress.
                for bone_name, rotation in arm_pose.items():
                    pose[bone_name] = _nlerp(pose[bone_name], rotation, ingress)
        else:
            pose["leftUpperArm"] = _local_rotation_offset(
                pose["leftUpperArm"], [-swing * 0.58, 0.0, 0.0]
            )
            pose["rightUpperArm"] = _local_rotation_offset(
                pose["rightUpperArm"], [swing * 0.58, 0.0, 0.0]
            )
        pelvis_roll = math.sin(phase) * 0.045 * intensity
        pelvis_pitch = -0.05 * intensity * run_scale
        pose["chest"] = _local_rotation_offset(
            pose["chest"], [0.04 * intensity * run_scale * ingress, 0.0, -pelvis_roll]
        )
    elif target.action == BodyAction.STEP:
        envelope = math.sin(math.pi * progress)
        swing = envelope * 0.62 * intensity
        side = target.lead_side.value
        other = Hand.LEFT.value if target.lead_side == Hand.RIGHT else Hand.RIGHT.value
        pose[f"{side}UpperLeg"] = _local_rotation_offset(pose[f"{side}UpperLeg"], [swing, 0.0, 0.0])
        pose[f"{side}LowerLeg"] = _local_rotation_offset(
            pose[f"{side}LowerLeg"], [envelope * 0.72 * intensity, 0.0, 0.0]
        )
        pose[f"{side}Foot"] = _local_rotation_offset(
            pose[f"{side}Foot"], [-envelope * 0.28 * intensity, 0.0, 0.0]
        )
        pose[f"{other}UpperLeg"] = _local_rotation_offset(
            pose[f"{other}UpperLeg"], [-swing * 0.18, 0.0, 0.0]
        )
        pelvis_roll = (
            (1.0 if target.lead_side == Hand.LEFT else -1.0)
            * envelope
            * 0.06
            * intensity
        )
    elif target.action == BodyAction.TURN:
        step = math.sin(2.0 * math.pi * progress) * 0.22 * intensity
        spread = 0.10 * math.sin(math.pi * progress)
        pose["leftUpperLeg"] = _local_rotation_offset(pose["leftUpperLeg"], [step, 0.0, spread])
        pose["rightUpperLeg"] = _local_rotation_offset(pose["rightUpperLeg"], [-step, 0.0, -spread])
    elif target.action == BodyAction.CROUCH:
        bend = alpha * intensity
        pose["leftUpperLeg"] = _local_rotation_offset(pose["leftUpperLeg"], [-0.78 * bend, 0.0, 0.08 * bend])
        pose["rightUpperLeg"] = _local_rotation_offset(pose["rightUpperLeg"], [-0.78 * bend, 0.0, -0.08 * bend])
        pose["leftLowerLeg"] = _local_rotation_offset(pose["leftLowerLeg"], [1.30 * bend, 0.0, 0.0])
        pose["rightLowerLeg"] = _local_rotation_offset(pose["rightLowerLeg"], [1.30 * bend, 0.0, 0.0])
        pose["leftFoot"] = _local_rotation_offset(pose["leftFoot"], [-0.48 * bend, 0.0, 0.0])
        pose["rightFoot"] = _local_rotation_offset(pose["rightFoot"], [-0.48 * bend, 0.0, 0.0])
        pelvis_pitch = 0.16 * bend
        pose["chest"] = _local_rotation_offset(pose["chest"], [-0.18 * bend, 0.0, 0.0])
    elif target.action == BodyAction.JUMP:
        local_progress = (target.cycles * progress) % 1.0 if progress < 1.0 else 1.0
        bend = math.sin(2.0 * math.pi * local_progress) ** 2 * 0.52 * intensity
        flight = math.sin(math.pi * local_progress) * intensity
        pose["leftUpperLeg"] = _local_rotation_offset(pose["leftUpperLeg"], [-0.62 * bend, 0.0, 0.05 * bend])
        pose["rightUpperLeg"] = _local_rotation_offset(pose["rightUpperLeg"], [-0.62 * bend, 0.0, -0.05 * bend])
        pose["leftLowerLeg"] = _local_rotation_offset(pose["leftLowerLeg"], [1.05 * bend, 0.0, 0.0])
        pose["rightLowerLeg"] = _local_rotation_offset(pose["rightLowerLeg"], [1.05 * bend, 0.0, 0.0])
        jumping_jack = bool(
            abs(target.pose.left_foot_shift_x_m)
            + abs(target.pose.right_foot_shift_x_m)
            > 1e-8
        )
        if jumping_jack or target.raise_arms_overhead:
            # Squared rather than |sin|: the overhead arm pose sits a large
            # rotation (including a near-half-turn of humeral roll) away from
            # the arms-down pose, and |sin| pushes its steepest slope into the
            # phase's first frames, on top of the whole-pose ingress blend.
            # sin**2 starts the raise at zero rate, so the reorientation
            # spreads over the blend window instead of the first three frames
            # deciding the clip's rotational-discontinuity peak.
            jack_envelope = (
                math.sin(math.pi * target.cycles * progress) ** 2
            )
            for hand in (Hand.LEFT, Hand.RIGHT):
                side = 1.0 if hand == Hand.LEFT else -1.0
                overhead_pose, _ = arm_pose_from_target(
                    hand,
                    shoulder_position(hand),
                    Vec3(x=side * 0.18, y=1.84, z=0.12),
                    PrimitiveParameters(elbow_swivel=side * 0.12),
                    present_hand=False,
                )
                for bone_name, rotation in overhead_pose.items():
                    # Constant-rate on purpose: the down->overhead span is
                    # 2.4-2.6 rad per arm bone (the humeral roll and its
                    # forearm compensation dominate), and nlerp's mid-path
                    # speed-up at that angle is what pushed each raise's
                    # fastest frames past the discontinuity gate.
                    pose[bone_name] = _slerp(
                        pose[bone_name],
                        rotation,
                        jack_envelope,
                    )
        else:
            pose["leftUpperArm"] = _local_rotation_offset(
                pose["leftUpperArm"], [-0.42 * flight, 0.0, 0.0]
            )
            pose["rightUpperArm"] = _local_rotation_offset(
                pose["rightUpperArm"], [-0.42 * flight, 0.0, 0.0]
            )
    elif target.action == BodyAction.ROTATE:
        rotation_angle = math.radians(target.rotation_degrees) * alpha
        rotation_envelope = math.sin(math.pi * progress) ** 0.65
        if target.rotation_axis == BodyRotationAxis.PITCH:
            pelvis_pitch = rotation_angle
        elif target.rotation_axis == BodyRotationAxis.ROLL:
            pelvis_roll = rotation_angle
        else:
            pelvis_yaw += rotation_angle
        if target.rotation_mode == BodyRotationMode.FLOOR:
            # Compact tuck keeps a grounded somersault inside a plausible
            # shoulder/hip radius while the root completes the full turn.
            tuck = rotation_envelope * target.intensity
            pose["leftUpperLeg"] = _local_rotation_offset(
                pose["leftUpperLeg"], [-1.10 * tuck, 0.0, 0.10 * tuck]
            )
            pose["rightUpperLeg"] = _local_rotation_offset(
                pose["rightUpperLeg"], [-1.10 * tuck, 0.0, -0.10 * tuck]
            )
            pose["leftLowerLeg"] = _local_rotation_offset(
                pose["leftLowerLeg"], [1.62 * tuck, 0.0, 0.0]
            )
            pose["rightLowerLeg"] = _local_rotation_offset(
                pose["rightLowerLeg"], [1.62 * tuck, 0.0, 0.0]
            )
            pose["leftUpperArm"] = _local_rotation_offset(
                pose["leftUpperArm"], [-0.72 * tuck, 0.0, 0.18 * tuck]
            )
            pose["rightUpperArm"] = _local_rotation_offset(
                pose["rightUpperArm"], [-0.72 * tuck, 0.0, -0.18 * tuck]
            )
            pose["chest"] = _local_rotation_offset(
                pose["chest"], [0.24 * tuck, 0.0, 0.0]
            )
        elif target.rotation_mode == BodyRotationMode.CARTWHEEL:
            # Extend both arms along the body's local vertical axis so one
            # hand becomes the ground support as the hips pass sideways over
            # the shoulders. Opposed hip roll keeps the legs visibly split.
            extension = rotation_envelope
            for hand in (Hand.LEFT, Hand.RIGHT):
                side = 1.0 if hand == Hand.LEFT else -1.0
                overhead_pose, _ = arm_pose_from_target(
                    hand,
                    shoulder_position(hand),
                    Vec3(x=side * 0.18, y=2.06, z=0.10),
                    PrimitiveParameters(elbow_swivel=side * 0.12),
                    present_hand=False,
                )
                for bone_name, rotation in overhead_pose.items():
                    pose[bone_name] = _nlerp(
                        pose[bone_name],
                        rotation,
                        extension,
                    )
            pose["leftUpperLeg"] = _local_rotation_offset(
                pose["leftUpperLeg"], [0.0, 0.0, 0.62 * extension]
            )
            pose["rightUpperLeg"] = _local_rotation_offset(
                pose["rightUpperLeg"], [0.0, 0.0, -0.62 * extension]
            )
        elif target.rotation_mode == BodyRotationMode.AIRBORNE:
            # Pitch/roll flips need a compact moment of inertia and a clear
            # visual silhouette. A yaw spin remains long-bodied so it reads
            # as a vertical jump turn rather than a somersault.
            if target.rotation_axis in {
                BodyRotationAxis.PITCH,
                BodyRotationAxis.ROLL,
            }:
                tuck = rotation_envelope * target.intensity
                pose["leftUpperLeg"] = _local_rotation_offset(
                    pose["leftUpperLeg"], [-1.02 * tuck, 0.0, 0.10 * tuck]
                )
                pose["rightUpperLeg"] = _local_rotation_offset(
                    pose["rightUpperLeg"], [-1.02 * tuck, 0.0, -0.10 * tuck]
                )
                pose["leftLowerLeg"] = _local_rotation_offset(
                    pose["leftLowerLeg"], [1.54 * tuck, 0.0, 0.0]
                )
                pose["rightLowerLeg"] = _local_rotation_offset(
                    pose["rightLowerLeg"], [1.54 * tuck, 0.0, 0.0]
                )
                pose["leftUpperArm"] = _local_rotation_offset(
                    pose["leftUpperArm"], [-0.62 * tuck, 0.0, 0.14 * tuck]
                )
                pose["rightUpperArm"] = _local_rotation_offset(
                    pose["rightUpperArm"], [-0.62 * tuck, 0.0, -0.14 * tuck]
                )
                pose["chest"] = _local_rotation_offset(
                    pose["chest"], [0.20 * tuck, 0.0, 0.0]
                )
    elif target.action == BodyAction.DANCE:
        rhythm = math.sin(math.pi * target.cycles * progress)
        counter_rhythm = math.sin(
            0.5 * math.pi * target.cycles * progress
        )
        pelvis_roll = 0.11 * rhythm * intensity
        pelvis_yaw += 0.09 * counter_rhythm * intensity
        pose["chest"] = _local_rotation_offset(
            pose["chest"],
            [
                0.035 * abs(rhythm) * intensity,
                -0.14 * counter_rhythm * intensity,
                -0.10 * rhythm * intensity,
            ],
        )
        pose["head"] = _local_rotation_offset(
            pose["head"],
            [0.0, 0.05 * counter_rhythm * intensity, 0.04 * rhythm * intensity],
        )
    elif target.action == BodyAction.CLIMB:
        stride = math.sin(2.0 * math.pi * target.cycles * progress)
        pelvis_pitch = -0.12 * intensity
        pelvis_roll = 0.045 * stride * intensity
        pose["chest"] = _local_rotation_offset(
            pose["chest"],
            [-0.18 * intensity * ingress, -0.05 * stride * intensity, 0.0],
        )
        pose["head"] = _local_rotation_offset(
            pose["head"],
            [0.08 * intensity * ingress, 0.0, -0.025 * stride * intensity],
        )
    elif target.action == BodyAction.KICK:
        envelope = math.sin(math.pi * progress)
        side = target.lead_side.value
        support = Hand.LEFT.value if target.lead_side == Hand.RIGHT else Hand.RIGHT.value
        pose[f"{side}UpperLeg"] = _local_rotation_offset(
            pose[f"{side}UpperLeg"], [1.08 * envelope * intensity, 0.0, 0.0]
        )
        pose[f"{side}LowerLeg"] = _local_rotation_offset(
            pose[f"{side}LowerLeg"], [0.34 * envelope * intensity, 0.0, 0.0]
        )
        pose[f"{side}Foot"] = _local_rotation_offset(
            pose[f"{side}Foot"], [-0.22 * envelope * intensity, 0.0, 0.0]
        )
        pose[f"{support}UpperLeg"] = _local_rotation_offset(
            pose[f"{support}UpperLeg"], [-0.12 * envelope * intensity, 0.0, 0.0]
        )
        pelvis_roll = (
            (1.0 if target.lead_side == Hand.LEFT else -1.0)
            * 0.10
            * envelope
            * intensity
        )
        pose["chest"] = _local_rotation_offset(pose["chest"], [0.0, 0.0, -pelvis_roll * 0.8])
    elif target.action == BodyAction.POSE:
        authored = target.pose
        if authored.support_mode == BodySupportMode.PLANK:
            # The standing idle is produced by planted-foot IK and therefore
            # contains a useful knee bend. A toe-supported plank must start
            # from the rig's straight neutral leg chain instead of inheriting
            # that standing bend.
            for bone_name in (
                "leftUpperLeg",
                "leftLowerLeg",
                "leftFoot",
                "rightUpperLeg",
                "rightLowerLeg",
                "rightFoot",
            ):
                pose[bone_name] = _nlerp(pose[bone_name], Quat(), ingress)
        degrees = math.pi / 180.0
        quadruped_swing = 0.0
        if (
            authored.support_mode == BodySupportMode.QUADRUPED
            and abs(authored.root_shift_x_m) + abs(authored.root_shift_z_m) > 1e-8
        ):
            quadruped_swing = (
                math.sin(2.0 * math.pi * target.cycles * progress)
                * math.sin(math.pi * progress) ** 2
            )
        head_alpha = alpha
        if (
            authored.has_head_motion()
            and not authored.has_non_head_motion()
            and target.cycles > 0.0
        ):
            head_alpha = math.sin(
                2.0 * math.pi * target.cycles * smoothstep(progress, 0.78)
            )
        pelvis_pitch = authored.pelvis_pitch_deg * degrees * alpha
        pelvis_yaw += authored.pelvis_yaw_deg * degrees * alpha
        pelvis_roll = authored.pelvis_roll_deg * degrees * alpha
        pose["chest"] = _local_rotation_offset(
            pose["chest"],
            [
                authored.torso_pitch_deg * degrees * alpha,
                authored.torso_yaw_deg * degrees * alpha,
                authored.torso_roll_deg * degrees * alpha,
            ],
        )
        pose["head"] = _local_rotation_offset(
            pose["head"],
            [
                authored.head_pitch_deg * degrees * head_alpha,
                authored.head_yaw_deg * degrees * head_alpha,
                authored.head_roll_deg * degrees * head_alpha,
            ],
        )
        pose["leftShoulder"] = _local_rotation_offset(
            pose["leftShoulder"],
            [
                0.0,
                0.0,
                -authored.left_shoulder_elevation_deg * degrees * alpha,
            ],
        )
        pose["rightShoulder"] = _local_rotation_offset(
            pose["rightShoulder"],
            [
                0.0,
                0.0,
                authored.right_shoulder_elevation_deg * degrees * alpha,
            ],
        )
        pose["leftUpperLeg"] = _local_rotation_offset(
            pose["leftUpperLeg"],
            [
                authored.left_hip_pitch_deg * degrees * alpha,
                0.0,
                authored.left_hip_roll_deg * degrees * alpha,
            ],
        )
        pose["rightUpperLeg"] = _local_rotation_offset(
            pose["rightUpperLeg"],
            [
                authored.right_hip_pitch_deg * degrees * alpha,
                0.0,
                authored.right_hip_roll_deg * degrees * alpha,
            ],
        )
        pose["leftLowerLeg"] = _local_rotation_offset(
            pose["leftLowerLeg"],
            [authored.left_knee_flexion_deg * degrees * alpha, 0.0, 0.0],
        )
        pose["rightLowerLeg"] = _local_rotation_offset(
            pose["rightLowerLeg"],
            [authored.right_knee_flexion_deg * degrees * alpha, 0.0, 0.0],
        )
        pose["leftFoot"] = _local_rotation_offset(
            pose["leftFoot"],
            [authored.left_ankle_pitch_deg * degrees * alpha, 0.0, 0.0],
        )
        pose["rightFoot"] = _local_rotation_offset(
            pose["rightFoot"],
            [authored.right_ankle_pitch_deg * degrees * alpha, 0.0, 0.0],
        )
        if abs(quadruped_swing) > 1e-8:
            pose["leftUpperLeg"] = _local_rotation_offset(
                pose["leftUpperLeg"],
                [0.16 * quadruped_swing, 0.0, 0.0],
            )
            pose["rightUpperLeg"] = _local_rotation_offset(
                pose["rightUpperLeg"],
                [-0.16 * quadruped_swing, 0.0, 0.0],
            )
            pose["leftLowerLeg"] = _local_rotation_offset(
                pose["leftLowerLeg"],
                [-0.12 * quadruped_swing, 0.0, 0.0],
            )
            pose["rightLowerLeg"] = _local_rotation_offset(
                pose["rightLowerLeg"],
                [0.12 * quadruped_swing, 0.0, 0.0],
            )

    pose["hips"] = _hips_world_euler_quat(pelvis_pitch, pelvis_yaw, pelvis_roll)
    return pose


def _gait_step_state(
    target: BodyTarget,
    progress: float,
) -> tuple[dict[str, float], dict[str, float], str | None, float]:
    """Return planted-foot offsets, swing clearances, and support side.

    ``cycles`` is interpreted as the requested number of footfalls.  The final
    foot placement straddles the requested root destination, so the root ends
    at ``distance_m`` without foot skating during any stance interval.
    """

    steps = max(1, int(round(target.cycles)))
    stride = target.distance_m / max(0.5, steps - 0.5)
    scaled = min(float(steps), max(0.0, progress) * steps)
    step_index = min(steps - 1, int(math.floor(scaled)))
    local = 1.0 if progress >= 1.0 else scaled - step_index
    lead = target.lead_side.value
    alternate = Hand.LEFT.value if target.lead_side == Hand.RIGHT else Hand.RIGHT.value

    offsets = {Hand.LEFT.value: 0.0, Hand.RIGHT.value: 0.0}
    for completed in range(step_index):
        side = lead if completed % 2 == 0 else alternate
        offsets[side] = (completed + 1) * stride

    swing = lead if step_index % 2 == 0 else alternate
    support = alternate if swing == lead else lead
    start = offsets[swing]
    end = (step_index + 1) * stride
    # A full minimum-jerk transfer makes position, velocity, and acceleration
    # continuous when the next foot takes over at a footfall boundary.
    swing_alpha = smoothstep(local, 1.0)
    offsets[swing] = start * (1.0 - swing_alpha) + end * swing_alpha
    clearance = (0.115 if target.action == BodyAction.RUN else 0.075) * target.intensity
    lifts = {
        Hand.LEFT.value: 0.0,
        Hand.RIGHT.value: 0.0,
    }
    # This sixth-order bump is one at mid-swing and has zero first and second
    # derivatives at both contacts.  The previous fractional-sine envelope
    # produced a visible ankle/knee jerk at every support change.
    lifts[swing] = 64.0 * local**3 * (1.0 - local) ** 3 * clearance
    flight_envelope = 0.0
    if target.action == BodyAction.RUN:
        # A run is not merely a faster planted walk.  Lift the trailing foot
        # through mid-stride so both feet briefly clear the floor.  The C2
        # bump retains continuous takeoff and landing.
        flight_local = float(np.clip((local - 0.35) / 0.50, 0.0, 1.0))
        flight_envelope = (
            64.0 * flight_local**3 * (1.0 - flight_local) ** 3
            if 0.0 < flight_local < 1.0
            else 0.0
        )
        lifts[support] = 0.085 * target.intensity * flight_envelope
        if lifts[support] > 0.020 and lifts[swing] > 0.020:
            support = None
    return offsets, lifts, support, flight_envelope


def _climb_limb_fraction(
    progress: float,
    cycles: float,
    phase_offset: float,
) -> float:
    """Advance one ladder-contact limb through smooth alternating rungs."""

    def staircase(value: float) -> float:
        whole = math.floor(value)
        local = value - whole
        transfer = smoothstep(
            float(np.clip((local - 0.14) / 0.72, 0.0, 1.0)),
            0.82,
        )
        return whole + transfer

    start = staircase(phase_offset)
    end = staircase(cycles + phase_offset)
    current = staircase(max(0.0, min(1.0, progress)) * cycles + phase_offset)
    return float(np.clip((current - start) / max(end - start, 1e-8), 0.0, 1.0))


def _leg_reach_height_adjustment(
    kinematics: Any,
    bones: dict[str, BonePose],
    ankle_targets: dict[str, np.ndarray],
    *,
    bend_margin_m: float,
) -> float:
    positions = kinematics.canonical_positions(bones)
    adjustments: list[float] = []
    for side in (Hand.LEFT.value, Hand.RIGHT.value):
        hip = positions[f"{side}UpperLeg"]
        ankle = ankle_targets[side]
        horizontal_reach = float(np.linalg.norm((ankle - hip)[[0, 2]]))
        usable_reach = max(0.05, kinematics.leg_reach(side) - bend_margin_m)
        vertical_reach = math.sqrt(
            max(0.0, usable_reach * usable_reach - horizontal_reach * horizontal_reach)
        )
        current_vertical = float(hip[1] - ankle[1])
        adjustments.append(vertical_reach - current_vertical)
    # A hard min changes the limiting leg abruptly when support transfers.
    # A small soft minimum stays conservatively below every reach limit while
    # giving the pelvis a continuous height path and a slight natural knee
    # bend even in a nominally straight standing pose.
    values = np.asarray([0.0, *adjustments], dtype=float)
    temperature = 0.0035
    minimum = float(np.min(values))
    soft_minimum = minimum - temperature * math.log(
        float(np.sum(np.exp(-(values - minimum) / temperature)))
    )
    return soft_minimum


def _compile_full_body(scene: SceneManifest, program: MotionProgram) -> ClipResult:
    """Compile sequential root/leg smart primitives into one mobile clip."""

    kinematics = rig_kinematics()
    base = _full_body_idle_pose()
    # The source rig's exact rest pose has locked, fully extended knees.  A
    # small standing root offset gives both legs enough reach margin for a
    # normal split stance without making the pelvis sink at every footfall.
    neutral_bones = {
        key: BonePose(
            rotation=value,
            position=(Vec3(x=0.0, y=0.0, z=0.0) if key == "hips" else None),
        )
        for key, value in base.items()
    }
    neutral_positions = kinematics.canonical_positions(neutral_bones)
    neutral_ground_height = min(
        float(neutral_positions["leftToes"][1]),
        float(neutral_positions["rightToes"][1]),
    )
    neutral_foot_rotations = {
        side: kinematics.canonical_world_rotation(neutral_bones, f"{side}Foot")
        for side in (Hand.LEFT.value, Hand.RIGHT.value)
    }
    standing_bones = {
        key: BonePose(
            rotation=value,
            position=(
                Vec3(x=0.0, y=FULL_BODY_STANDING_ROOT_HEIGHT_M, z=0.0)
                if key == "hips"
                else None
            ),
        )
        for key, value in base.items()
    }
    for side in (Hand.LEFT.value, Hand.RIGHT.value):
        solved = kinematics.solve_leg(
            standing_bones,
            side,
            neutral_positions[f"{side}Foot"],
            foot_world_rotation=neutral_foot_rotations[side],
        )
        for bone_name, rotation in solved.items():
            base[bone_name] = rotation
            standing_bones[bone_name] = BonePose(rotation=rotation)
    current_pose = base.copy()
    current_root = np.asarray(
        [0.0, FULL_BODY_STANDING_ROOT_HEIGHT_M, 0.0],
        dtype=float,
    )
    current_yaw = 0.0
    frames: list[ClipFrame] = []
    phase_ranges: list[dict[str, float | str]] = []
    elapsed = 0.0
    fps = scene.fps
    actions: list[str] = []
    support_constraints: list[dict[str, Any]] = []
    climb_support_constraints: list[dict[str, Any]] = []
    arm_idle_targets = {
        Hand.LEFT: Vec3(x=0.27, y=0.94, z=0.08),
        Hand.RIGHT: Vec3(x=-0.27, y=0.94, z=0.08),
    }
    last_arm_targets = arm_idle_targets.copy()
    previous_target: BodyTarget | None = None
    pose_return_positions: dict[str, np.ndarray] = {}
    pose_return_rotations: dict[str, np.ndarray] = {}
    pose_return_root: np.ndarray | None = None
    pose_sequence_has_foot_motion = False

    for primitive in program.primitives:
        target = primitive.body or BodyTarget(
            action=BodyAction.HOLD,
            cycles=0.0,
            intensity=0.0,
        )
        recover = primitive.kind == PrimitiveKind.RECOVER
        previous_action = actions[-1] if actions else None
        actions.append(target.action.value)
        start_pose = current_pose.copy()
        start_root = current_root.copy()
        start_yaw = current_yaw
        requested_frames = max(2, int(round(primitive.parameters.duration_s * fps)))
        if target.action in {BodyAction.WALK, BodyAction.RUN}:
            requested_frames = max(
                requested_frames,
                int(math.ceil(max(1.0, target.cycles) * 18.0)) + 1,
            )
        elif target.action == BodyAction.JUMP:
            requested_frames = max(
                requested_frames,
                int(math.ceil(max(1.0, target.cycles) * 24.0)) + 1,
            )
        elif target.action == BodyAction.ROTATE:
            requested_frames = max(
                requested_frames,
                int(math.ceil(abs(target.rotation_degrees) / 5.0)) + 1,
            )
        elif target.action == BodyAction.DANCE:
            requested_frames = max(
                requested_frames,
                int(math.ceil(max(1.0, target.cycles) * 18.0)) + 1,
            )
        elif target.action == BodyAction.CLIMB:
            requested_frames = max(
                requested_frames,
                int(math.ceil(max(1.0, target.cycles) * 30.0)) + 1,
            )
        arm_trajectory = primitive.trajectory or TrajectoryKind.LINEAR
        arm_cycles = primitive.parameters.trajectory_cycles
        if primitive.effectors and arm_trajectory in {
            TrajectoryKind.CIRCLE,
            TrajectoryKind.OSCILLATE,
        }:
            requested_frames = max(
                requested_frames,
                int(math.ceil(max(1.0, arm_cycles) * 24.0)) + 1,
            )
        if (
            not recover
            and target.action == BodyAction.POSE
            and target.pose.support_mode == BodySupportMode.QUADRUPED
            and abs(target.pose.root_shift_x_m) + abs(target.pose.root_shift_z_m)
            > 1e-8
        ):
            # Crawl articulation occupies the post-acquisition portion of the
            # phase. Retain at least 24 rendered samples per active limb cycle
            # after that temporal compression.
            requested_frames = max(
                requested_frames,
                int(
                    math.ceil(
                        max(1.0, target.cycles)
                        * (64.0 if target.cycles <= 2.0 else 52.0)
                    )
                )
                + 1,
            )
        if recover and previous_action in {
            BodyAction.WALK.value,
            BodyAction.RUN.value,
            BodyAction.STEP.value,
            BodyAction.TURN.value,
            BodyAction.CROUCH.value,
            BodyAction.JUMP.value,
            BodyAction.DANCE.value,
            BodyAction.CLIMB.value,
            BodyAction.ROTATE.value,
            BodyAction.KICK.value,
            BodyAction.POSE.value,
        }:
            requested_frames = max(requested_frames, 31)
        frame_count = requested_frames
        phase_duration_s = max(
            primitive.parameters.duration_s,
            (frame_count - 1) / fps,
        )
        phase_ranges.append(
            {
                "kind": primitive.kind.value,
                "label": primitive.label or target.action.value,
                "action": target.action.value,
                "start_s": elapsed,
                "end_s": elapsed + phase_duration_s,
            }
        )
        direction = np.asarray(
            [target.direction_x, 0.0, target.direction_z],
            dtype=float,
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-8:
            direction /= direction_norm
        arm_centers = {
            effector.hand: _composite_workspace_target(effector)
            for effector in primitive.effectors
        }
        arm_parameters = {
            effector.hand: _composite_parameters(primitive.parameters, effector)
            for effector in primitive.effectors
        }
        gait_start_positions: dict[str, np.ndarray] = {}
        gait_foot_rotations: dict[str, np.ndarray] = {}
        gait_root_spline: CubicSpline | None = None
        obstacle_object = (
            scene.object_by_id(target.obstacle_object_id)
            if target.obstacle_object_id
            else None
        )
        support_object = (
            scene.object_by_id(target.support_object_id)
            if target.support_object_id
            else None
        )
        climb_limb_start_positions: dict[str, np.ndarray] = {}
        climb_face_center = np.zeros(3, dtype=float)
        climb_side_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
        obstacle_swing_lift_m = 0.0
        lateral_direction = np.asarray(
            [direction[2], 0.0, -direction[0]],
            dtype=float,
        )
        gait_turn_center = np.asarray([start_root[0], 0.0, start_root[2]], dtype=float)
        if not recover and target.action != BodyAction.HOLD:
            start_bones = {
                key: BonePose(
                    rotation=value,
                    position=(
                        Vec3(
                            x=float(start_root[0]),
                            y=float(start_root[1]),
                            z=float(start_root[2]),
                        )
                        if key == "hips"
                        else None
                    ),
                )
                for key, value in start_pose.items()
            }
            start_positions = kinematics.canonical_positions(start_bones)
            gait_turn_center = start_positions["hips"].copy()
            gait_turn_center[1] = 0.0
            gait_start_positions = {
                side: start_positions[f"{side}Foot"].copy()
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            }
            gait_foot_rotations = {
                side: kinematics.canonical_world_rotation(start_bones, f"{side}Foot")
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            }
            if target.action == BodyAction.CLIMB:
                climb_limb_start_positions = {
                    f"{side}{limb}": start_positions[f"{side}{limb}"].copy()
                    for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    for limb in ("Hand", "Foot")
                }
                if support_object is not None:
                    support_rotation = Rotation.from_quat(
                        support_object.transform.rotation.as_list()
                    )
                    climb_side_axis = support_rotation.apply(
                        np.asarray([1.0, 0.0, 0.0], dtype=float)
                    )
                    climb_face_center = np.asarray(
                        support_object.transform.translation.as_list(),
                        dtype=float,
                    ) + support_rotation.apply(
                        np.asarray(
                            [
                                0.0,
                                0.0,
                                -0.5 * support_object.dimensions_m.z - 0.018,
                            ],
                            dtype=float,
                        )
                    )
            if target.action == BodyAction.POSE:
                if previous_target is None or previous_target.action != BodyAction.POSE:
                    pose_return_root = start_root.copy()
                    pose_return_positions = {
                        side: value.copy() for side, value in gait_start_positions.items()
                    }
                    pose_return_rotations = {
                        side: value.copy() for side, value in gait_foot_rotations.items()
                    }
                    pose_sequence_has_foot_motion = False
                pose_sequence_has_foot_motion = (
                    pose_sequence_has_foot_motion or target.pose.has_foot_motion()
                )
            if target.action in {BodyAction.WALK, BodyAction.RUN, BodyAction.STEP}:
                step_count = (
                    1
                    if target.action == BodyAction.STEP
                    else max(1, int(round(target.cycles)))
                )
                knots = np.linspace(0.0, 1.0, step_count + 1)
                if step_count == 1:
                    root_fractions = np.asarray([0.0, 1.0], dtype=float)
                else:
                    root_fractions = np.asarray(
                        [0.0]
                        + [
                            (index - 0.5) / (step_count - 0.5)
                            for index in range(1, step_count + 1)
                        ],
                        dtype=float,
                    )
                # The spline passes through the midpoint of the planted feet
                # at every support exchange while retaining continuous root
                # velocity and acceleration.  This avoids both foot-driven
                # stop/start motion and a pelvis that outruns its support leg.
                gait_root_spline = CubicSpline(
                    knots,
                    root_fractions,
                    bc_type=((1, 0.0), (1, 0.0)),
                )
                if (
                    target.obstacle_mode == BodyObstacleMode.OVER
                    and obstacle_object is not None
                ):
                    object_top = (
                        obstacle_object.transform.translation.y
                        + 0.5 * obstacle_object.dimensions_m.y
                    )
                    lead_foot_height = float(
                        gait_start_positions[target.lead_side.value][1]
                    )
                    obstacle_swing_lift_m = max(
                        0.0,
                        object_top
                        + target.obstacle_clearance_m
                        # The IK target is the ankle joint while the gate
                        # measures the lowest foot/toe point. Add the rig's
                        # shoe/ankle offset so the visible sole, not merely
                        # the ankle origin, clears the obstacle.
                        + 0.13
                        - lead_foot_height,
                    )
        gait_recovery = recover and previous_action in {
            BodyAction.WALK.value,
            BodyAction.RUN.value,
            BodyAction.STEP.value,
            BodyAction.TURN.value,
            BodyAction.CROUCH.value,
            BodyAction.JUMP.value,
            BodyAction.DANCE.value,
            BodyAction.CLIMB.value,
            BodyAction.ROTATE.value,
            BodyAction.KICK.value,
            BodyAction.POSE.value,
        } and not (
            previous_target is not None
            and previous_target.action == BodyAction.POSE
            and not previous_target.pose.lock_feet
        )
        free_pose_recovery = bool(
            recover
            and previous_target is not None
            and previous_target.action == BodyAction.POSE
            and not previous_target.pose.lock_feet
        )
        pose_foot_recovery = bool(
            gait_recovery
            and previous_target is not None
            and previous_target.action == BodyAction.POSE
            and pose_sequence_has_foot_motion
            and pose_return_positions
        )
        recovery_start_positions: dict[str, np.ndarray] = {}
        recovery_foot_rotations: dict[str, np.ndarray] = {}
        if gait_recovery:
            start_bones = {
                key: BonePose(
                    rotation=value,
                    position=(
                        Vec3(x=float(start_root[0]), y=float(start_root[1]), z=float(start_root[2]))
                        if key == "hips"
                        else None
                    ),
                )
                for key, value in start_pose.items()
            }
            start_positions = kinematics.canonical_positions(start_bones)
            recovery_start_positions = {
                side: start_positions[f"{side}Foot"].copy()
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            }
            recovery_foot_rotations = {
                side: kinematics.canonical_world_rotation(start_bones, f"{side}Foot")
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            }

        plank_hand_targets: dict[str, np.ndarray] | None = None
        plank_hand_rotations: dict[str, np.ndarray] = {}
        plank_root_height: float | None = None
        plank_ankle_targets: dict[str, np.ndarray] = {}
        plank_foot_rotations: dict[str, np.ndarray] = {}

        for local_index in range(frame_count):
            if frames and local_index == 0:
                continue
            progress = local_index / (frame_count - 1)
            alpha = smoothstep(progress, primitive.parameters.easing)
            action_alpha = alpha
            push_up_drop = 0.0
            moving_quadruped = bool(
                not recover
                and target.action == BodyAction.POSE
                and target.pose.support_mode == BodySupportMode.QUADRUPED
                and (
                    abs(target.pose.root_shift_x_m)
                    + abs(target.pose.root_shift_z_m)
                    > 1e-8
                )
            )
            if (
                not recover
                and target.action == BodyAction.POSE
                and target.pose.support_mode == BodySupportMode.PLANK
            ):
                # Reach the stable high-plank early, then perform the exact
                # requested number of smooth down/up cycles before recovery.
                action_alpha = smoothstep(min(1.0, progress / 0.20), 0.78)
                cycle_progress = float(
                    np.clip((progress - 0.40) / 0.50, 0.0, 1.0)
                )
                # ``height_m`` describes the semantic strength of the push-up
                # pose, but the actual chest excursion is bounded by this
                # rig's arm length and floor clearance.  A 28 cm root drop
                # forces the shoulders through the ground; 11 cm produces a
                # full, readable repetition while retaining joint margin.
                push_up_drop = min(target.height_m or 0.11, 0.11) * (
                    math.sin(math.pi * target.cycles * cycle_progress) ** 2
                )
            elif moving_quadruped:
                action_alpha = smoothstep(min(1.0, progress / 0.20), 0.78)
            articulation_progress = (
                float(np.clip((progress - 0.28) / 0.62, 0.0, 1.0))
                if moving_quadruped
                else progress
            )
            ankle_targets: dict[str, np.ndarray] = {}
            leg_foot_rotations: dict[str, np.ndarray] = {}
            climb_limb_targets: dict[str, np.ndarray] = {}
            climb_solve_alpha = 0.0
            support_side: str | None = None
            leg_ik_active = False
            if recover:
                target_pose = base.copy()
                target_pose["hips"] = _hips_world_euler_quat(y=start_yaw)
                pose = {
                    key: _nlerp(start_pose[key], target_pose[key], alpha)
                    for key in base
                }
                root = start_root.copy()
                root[1] = (
                    start_root[1] * (1.0 - alpha)
                    + FULL_BODY_STANDING_ROOT_HEIGHT_M * alpha
                )
                if (
                    previous_target is not None
                    and previous_target.action == BodyAction.POSE
                    and pose_return_root is not None
                    and not (
                        previous_target.pose.support_mode
                        == BodySupportMode.QUADRUPED
                        and (
                            abs(previous_target.pose.root_shift_x_m)
                            + abs(previous_target.pose.root_shift_z_m)
                            > 1e-8
                        )
                    )
                ):
                    root[0] = start_root[0] * (1.0 - alpha) + pose_return_root[0] * alpha
                    root[2] = start_root[2] * (1.0 - alpha) + pose_return_root[2] * alpha
                if gait_recovery:
                    if pose_foot_recovery:
                        assert previous_target is not None
                        first = (
                            Hand.LEFT.value
                            if previous_target.lead_side == Hand.RIGHT
                            else Hand.RIGHT.value
                        )
                        second = previous_target.lead_side.value
                        ordered_sides = [
                            side
                            for side in (first, second)
                            if float(
                                np.linalg.norm(
                                    recovery_start_positions[side]
                                    - pose_return_positions[side]
                                )
                            )
                            > 0.005
                        ]
                        ankle_targets = {}
                        moving_now: str | None = None
                        for side in (Hand.LEFT.value, Hand.RIGHT.value):
                            if side not in ordered_sides:
                                local_foot_progress = 1.0
                            elif len(ordered_sides) == 1:
                                local_foot_progress = progress
                            else:
                                stage = ordered_sides.index(side)
                                local_foot_progress = float(
                                    np.clip(progress * 2.0 - stage, 0.0, 1.0)
                                )
                            foot_alpha = smoothstep(local_foot_progress, 0.78)
                            foot_lift = (
                                math.sin(math.pi * local_foot_progress) ** 1.5
                                if side in ordered_sides
                                and 0.0 < local_foot_progress < 1.0
                                else 0.0
                            )
                            if (
                                side in ordered_sides
                                and 0.0 < local_foot_progress < 1.0
                            ):
                                moving_now = side
                            ankle_targets[side] = (
                                recovery_start_positions[side] * (1.0 - foot_alpha)
                                + pose_return_positions[side] * foot_alpha
                                + np.asarray([0.0, 0.055 * foot_lift, 0.0], dtype=float)
                            )
                        support_side = (
                            Hand.LEFT.value
                            if moving_now == Hand.RIGHT.value
                            else Hand.RIGHT.value
                            if moving_now == Hand.LEFT.value
                            else Hand.LEFT.value
                        )
                    else:
                        # Locomotion recovery restores torso and arm posture
                        # without silently adding extra footfalls.  Both feet
                        # remain planted in the terminal stance authored by the
                        # body skill.
                        ankle_targets = {
                            side: recovery_start_positions[side].copy()
                            for side in (Hand.LEFT.value, Hand.RIGHT.value)
                        }
                        support_side = Hand.LEFT.value
                    preliminary = {
                        key: BonePose(
                            rotation=value,
                            position=(
                                Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                                if key == "hips"
                                else None
                            ),
                        )
                        for key, value in pose.items()
                    }
                    root[1] += _leg_reach_height_adjustment(
                        kinematics,
                        preliminary,
                        ankle_targets,
                        bend_margin_m=0.008,
                    )
                    leg_foot_rotations = (
                        pose_return_rotations
                        if pose_foot_recovery
                        else recovery_foot_rotations
                    )
                    leg_ik_active = True
            else:
                yaw = start_yaw + (
                    math.radians(target.turn_degrees) * alpha
                    if target.action == BodyAction.TURN
                    else 0.0
                )
                chained_pose = bool(
                    target.action == BodyAction.POSE
                    and previous_target is not None
                    and previous_target.action == BodyAction.POSE
                )
                # A chained pose carries the actual prior endpoint. Driving
                # its next target through the standing base creates a hidden
                # extra motion (for example supine -> upright -> curl).
                # Interpolate directly between the two absolute poses.
                ingress = (
                    action_alpha
                    if chained_pose
                    else smoothstep(min(1.0, progress / 0.18), 0.78)
                )
                generated_pose = _body_action_pose(
                    base,
                    target,
                    articulation_progress,
                    1.0 if chained_pose else action_alpha,
                    yaw,
                    ingress=1.0 if chained_pose else ingress,
                )
                if chained_pose or ingress >= 1.0:
                    pose = {
                        key: _nlerp(start_pose[key], generated_pose[key], ingress)
                        for key in base
                    }
                else:
                    # The entry blend eases only the start-pose mismatch onto
                    # the base; the skill's own articulation rides on top at
                    # full rate.  Blending the whole generated pose from the
                    # start pose multiplied the ingress ramp into the skill's
                    # articulation rate, and for an articulation covering a
                    # large rotation inside the ingress window (a jumping
                    # jack's first overhead raise) the compounded rate alone
                    # crossed the discontinuity gate.  The hips stay on the
                    # absolute blend: their generated value is authored in
                    # world terms (start yaw included), not as a base offset.
                    pose = {}
                    for key in base:
                        if key == "hips":
                            pose[key] = _nlerp(
                                start_pose[key], generated_pose[key], ingress
                            )
                            continue
                        eased = Rotation.from_quat(
                            _nlerp(start_pose[key], base[key], ingress).as_list()
                        )
                        offset = (
                            Rotation.from_quat(base[key].as_list()).inv()
                            * Rotation.from_quat(generated_pose[key].as_list())
                        )
                        xyzw = (eased * offset).as_quat()
                        pose[key] = Quat(
                            x=float(xyzw[0]),
                            y=float(xyzw[1]),
                            z=float(xyzw[2]),
                            w=float(xyzw[3]),
                        )
                root = start_root.copy()
                if target.action in {BodyAction.WALK, BodyAction.RUN, BodyAction.STEP}:
                    gait_target = (
                        target.model_copy(update={"cycles": 1.0})
                        if target.action == BodyAction.STEP
                        else target
                    )
                    offsets, lifts, support_side, flight_envelope = _gait_step_state(
                        gait_target, progress
                    )
                    if target.obstacle_mode == BodyObstacleMode.OVER:
                        swing_bump = 64.0 * progress**3 * (1.0 - progress) ** 3
                        lifts[target.lead_side.value] = max(
                            lifts[target.lead_side.value],
                            min(0.72, obstacle_swing_lift_m) * swing_bump,
                        )
                    assert gait_root_spline is not None
                    root_path_fraction = float(gait_root_spline(progress))
                    root += direction * (target.distance_m * root_path_fraction)
                    detour_path_fraction = float(
                        np.clip(root_path_fraction, 0.0, 1.0)
                    )
                    root += lateral_direction * (
                        target.path_lateral_offset_m
                        * math.sin(math.pi * detour_path_fraction) ** 1.35
                    )
                    if target.action == BodyAction.RUN:
                        root[1] += 0.055 * target.intensity * flight_envelope
                    ankle_targets = {
                        side: (
                            gait_start_positions[side]
                            + direction * offsets[side]
                            + lateral_direction
                            * (
                                target.path_lateral_offset_m
                                * math.sin(
                                    math.pi
                                    * float(
                                        np.clip(
                                            offsets[side]
                                            / max(target.distance_m, 1e-8),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                )
                                ** 1.35
                            )
                            + np.asarray([0.0, lifts[side], 0.0], dtype=float)
                        )
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    preliminary = {
                        key: BonePose(
                            rotation=value,
                            position=(
                                Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                                if key == "hips"
                                else None
                            ),
                        )
                        for key, value in pose.items()
                    }
                    root[1] += _leg_reach_height_adjustment(
                        kinematics,
                        preliminary,
                        ankle_targets,
                        bend_margin_m=0.008,
                    )
                    leg_foot_rotations = gait_foot_rotations
                    leg_ik_active = True
                elif target.action == BodyAction.TURN:
                    center = gait_turn_center
                    full_turn = Rotation.from_euler(
                        "y", math.radians(target.turn_degrees)
                    ).as_matrix()
                    final_targets = {
                        side: center
                        + full_turn @ (gait_start_positions[side] - center)
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    first = (
                        Hand.RIGHT.value
                        if target.turn_degrees >= 0.0
                        else Hand.LEFT.value
                    )
                    second = Hand.LEFT.value if first == Hand.RIGHT.value else Hand.RIGHT.value
                    if progress < 0.5:
                        moving = first
                        support_side = second
                        local_turn = progress * 2.0
                        completed: set[str] = set()
                    else:
                        moving = second
                        support_side = first
                        local_turn = (progress - 0.5) * 2.0
                        completed = {first}
                    move_alpha = smoothstep(local_turn, 0.78)
                    ankle_targets = {}
                    for side in (Hand.LEFT.value, Hand.RIGHT.value):
                        if side in completed:
                            ankle_targets[side] = final_targets[side].copy()
                        elif side == moving:
                            ankle_targets[side] = (
                                gait_start_positions[side] * (1.0 - move_alpha)
                                + final_targets[side] * move_alpha
                            )
                            ankle_targets[side][1] += (
                                math.sin(math.pi * local_turn) ** 1.35 * 0.035
                            )
                        else:
                            ankle_targets[side] = gait_start_positions[side].copy()
                    yaw_rotation = Rotation.from_euler("y", yaw - start_yaw).as_matrix()
                    leg_foot_rotations = {
                        side: yaw_rotation @ gait_foot_rotations[side]
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    preliminary = {
                        key: BonePose(
                            rotation=value,
                            position=(
                                Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                                if key == "hips"
                                else None
                            ),
                        )
                        for key, value in pose.items()
                    }
                    root[1] += _leg_reach_height_adjustment(
                        kinematics,
                        preliminary,
                        ankle_targets,
                        bend_margin_m=0.004,
                    )
                    leg_ik_active = True
                elif target.action == BodyAction.CROUCH:
                    # ``height_m`` is the requested physical pelvis drop, not
                    # a stylistic amplitude.  Multiplying it by intensity made
                    # an ordinary 24 cm crouch only move 15.6 cm at the neutral
                    # default, which read as a slight knee bend.  Intensity
                    # still shapes torso/limb participation above.
                    target_root_y = (
                        FULL_BODY_STANDING_ROOT_HEIGHT_M
                        - (target.height_m or 0.24)
                    )
                    root[1] = (
                        start_root[1] * (1.0 - alpha)
                        + target_root_y * alpha
                    )
                    ankle_targets = {
                        side: gait_start_positions[side].copy()
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    leg_foot_rotations = gait_foot_rotations
                    support_side = Hand.LEFT.value
                    leg_ik_active = True
                elif target.action == BodyAction.JUMP:
                    requested_height = target.height_m or 0.30
                    # Height is ground/standing referenced.  When a jump
                    # follows a crouch, measuring only from the compressed
                    # pelvis makes the apex merely return to standing height.
                    # Retain the crouched takeoff and landing, but extend the
                    # airborne excursion enough to reach standing height plus
                    # the requested jump magnitude.
                    apex_root_y = max(
                        start_root[1] + requested_height,
                        FULL_BODY_STANDING_ROOT_HEIGHT_M + requested_height,
                    )
                    root[1] += (apex_root_y - start_root[1]) * abs(
                        math.sin(math.pi * target.cycles * progress)
                    )
                    flight_envelope = abs(
                        math.sin(math.pi * target.cycles * progress)
                    )
                    flight_height = root[1] - start_root[1]
                    foot_rise = requested_height * 1.22 * flight_envelope
                    ankle_targets = {}
                    for side in (Hand.LEFT.value, Hand.RIGHT.value):
                        local_spread = np.asarray(
                            [
                                (
                                    target.pose.left_foot_shift_x_m
                                    if side == Hand.LEFT.value
                                    else target.pose.right_foot_shift_x_m
                                )
                                * flight_envelope,
                                foot_rise,
                                0.0,
                            ],
                            dtype=float,
                        )
                        ankle_targets[side] = gait_start_positions[side] + (
                            Rotation.from_euler("y", start_yaw).apply(local_spread)
                        )
                    leg_foot_rotations = gait_foot_rotations
                    support_side = (
                        Hand.LEFT.value if flight_height <= 0.015 else None
                    )
                    leg_ik_active = True
                elif target.action == BodyAction.ROTATE:
                    root += direction * (target.distance_m * action_alpha)
                    rotation_envelope = math.sin(math.pi * progress) ** 0.75
                    if target.rotation_mode == BodyRotationMode.FLOOR:
                        root[1] -= (0.24 + 0.12 * target.intensity) * rotation_envelope
                    elif target.rotation_mode == BodyRotationMode.AIRBORNE:
                        root[1] += (target.height_m or 0.45) * rotation_envelope
                    else:
                        root[1] += 0.10 * rotation_envelope
                    preliminary = {
                        key: BonePose(
                            rotation=value,
                            position=(
                                Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                                if key == "hips"
                                else None
                            ),
                        )
                        for key, value in pose.items()
                    }
                    preliminary_positions = kinematics.canonical_positions(preliminary)
                    support_bones = (
                        "hips",
                        "chest",
                        "upperChest",
                        "head",
                        "leftHand",
                        "rightHand",
                        "leftFoot",
                        "rightFoot",
                        "leftToes",
                        "rightToes",
                    )
                    lowest_body = min(
                        float(preliminary_positions[name][1])
                        for name in support_bones
                    )
                    if target.rotation_mode != BodyRotationMode.AIRBORNE:
                        root[1] += (
                            neutral_ground_height + 0.004 - lowest_body
                        )
                elif target.action == BodyAction.DANCE:
                    beat_position = target.cycles * progress
                    beat_index = min(
                        max(0, int(math.ceil(target.cycles)) - 1),
                        int(math.floor(beat_position)),
                    )
                    local_beat = (
                        1.0
                        if progress >= 1.0
                        else beat_position - math.floor(beat_position)
                    )
                    tap_envelope = math.sin(math.pi * local_beat) ** 1.5
                    active_side = (
                        Hand.LEFT.value
                        if beat_index % 2 == 0
                        else Hand.RIGHT.value
                    )
                    support_side = (
                        Hand.RIGHT.value
                        if active_side == Hand.LEFT.value
                        else Hand.LEFT.value
                    )
                    lateral_sign = 1.0 if active_side == Hand.LEFT.value else -1.0
                    ankle_targets = {
                        side: gait_start_positions[side].copy()
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    local_tap = np.asarray(
                        [
                            0.075 * lateral_sign * tap_envelope,
                            0.060 * tap_envelope,
                            0.025 * tap_envelope,
                        ],
                        dtype=float,
                    )
                    ankle_targets[active_side] += Rotation.from_euler(
                        "y", start_yaw
                    ).apply(local_tap)
                    local_sway = np.asarray(
                        [
                            -0.040 * lateral_sign * tap_envelope,
                            0.0,
                            0.0,
                        ],
                        dtype=float,
                    )
                    root += Rotation.from_euler("y", start_yaw).apply(local_sway)
                    root[1] -= 0.025 * tap_envelope
                    preliminary = {
                        key: BonePose(
                            rotation=value,
                            position=(
                                Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                                if key == "hips"
                                else None
                            ),
                        )
                        for key, value in pose.items()
                    }
                    root[1] += _leg_reach_height_adjustment(
                        kinematics,
                        preliminary,
                        ankle_targets,
                        bend_margin_m=0.008,
                    )
                    leg_foot_rotations = gait_foot_rotations
                    leg_ik_active = True
                elif target.action == BodyAction.CLIMB:
                    direction_sign = (
                        -1.0
                        if target.climb_direction == BodyClimbDirection.DOWN
                        else 1.0
                    )
                    entry_offset = (
                        target.height_m
                        if target.climb_direction == BodyClimbDirection.DOWN
                        else 0.0
                    )
                    climb_progress = smoothstep(
                        float(np.clip((progress - 0.30) / 0.70, 0.0, 1.0)),
                        0.76,
                    )
                    root[1] = (
                        start_root[1]
                        + entry_offset
                        + direction_sign * target.height_m * climb_progress
                    )
                    contact_alpha = (
                        1.0
                        if target.climb_direction == BodyClimbDirection.DOWN
                        else smoothstep(min(1.0, progress / 0.30), 0.80)
                    )
                    climb_solve_alpha = contact_alpha
                    horizontal_offset = (
                        min(0.18, support_object.dimensions_m.x * 0.36)
                        if support_object is not None
                        else 0.16
                    )
                    phase_offsets = {
                        "leftHand": 0.0,
                        "rightFoot": 0.0,
                        "rightHand": 0.5,
                        "leftFoot": 0.5,
                    }
                    for side in (Hand.LEFT.value, Hand.RIGHT.value):
                        side_sign = 1.0 if side == Hand.LEFT.value else -1.0
                        for limb in ("Hand", "Foot"):
                            name = f"{side}{limb}"
                            start_position = climb_limb_start_positions.get(name)
                            if start_position is None:
                                continue
                            limb_fraction = _climb_limb_fraction(
                                climb_progress,
                                target.cycles,
                                phase_offsets[name],
                            )
                            ladder_target = climb_face_center.copy()
                            ladder_target += climb_side_axis * (
                                side_sign * horizontal_offset
                            )
                            ladder_target[1] = (
                                start_position[1]
                                + (0.44 if limb == "Hand" else 0.12)
                                + entry_offset
                                + direction_sign * target.height_m * limb_fraction
                            )
                            climb_limb_targets[name] = (
                                start_position * (1.0 - contact_alpha)
                                + ladder_target * contact_alpha
                            )
                    ankle_targets = {
                        side: climb_limb_targets[f"{side}Foot"]
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                        if f"{side}Foot" in climb_limb_targets
                    }
                    if len(ankle_targets) == 2:
                        leg_foot_rotations = gait_foot_rotations
                        leg_ik_active = True
                elif target.action == BodyAction.KICK:
                    kick_side = target.lead_side.value
                    support_side = (
                        Hand.LEFT.value
                        if target.lead_side == Hand.RIGHT
                        else Hand.RIGHT.value
                    )
                    envelope = math.sin(math.pi * progress)
                    ankle_targets = {
                        side: gait_start_positions[side].copy()
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    kick_reach = 0.42 + 0.28 * target.intensity
                    kick_lift = 0.28 + 0.28 * target.intensity
                    ankle_targets[kick_side] += (
                        direction * (kick_reach * envelope)
                        + np.asarray(
                            [0.0, kick_lift * envelope, 0.0],
                            dtype=float,
                        )
                    )
                    leg_foot_rotations = gait_foot_rotations
                    leg_ik_active = True
                elif target.action == BodyAction.POSE:
                    authored = target.pose
                    local_shift = np.asarray(
                        [
                            authored.root_shift_x_m,
                            0.0,
                            authored.root_shift_z_m,
                        ],
                        dtype=float,
                    )
                    world_shift = Rotation.from_euler("y", start_yaw).apply(local_shift)
                    root_shift_alpha = alpha if moving_quadruped else action_alpha
                    root += world_shift * root_shift_alpha
                    # Locked-foot leg IK necessarily owns the final hip/knee
                    # rotations. Convert an authored bilateral knee bend into
                    # the compatible pelvis drop before solving, otherwise the
                    # explicit knee controls are silently overwritten and a
                    # request such as "soften both knees" looks motionless.
                    if (
                        authored.left_foot_lift_m > 1e-8
                        and authored.right_foot_lift_m <= 1e-8
                    ):
                        knee_flexion_deg = authored.right_knee_flexion_deg
                    elif (
                        authored.right_foot_lift_m > 1e-8
                        and authored.left_foot_lift_m <= 1e-8
                    ):
                        knee_flexion_deg = authored.left_knee_flexion_deg
                    else:
                        knee_flexion_deg = 0.5 * (
                            authored.left_knee_flexion_deg
                            + authored.right_knee_flexion_deg
                        )
                    knee_implied_drop = (
                        0.22 * math.sin(math.radians(knee_flexion_deg))
                        if authored.lock_feet
                        else 0.0
                    )
                    effective_root_drop = max(
                        authored.root_drop_m,
                        knee_implied_drop,
                    )
                    if authored.support_mode == BodySupportMode.PLANK:
                        standing_origin_y = (
                            float(pose_return_root[1])
                            if pose_return_root is not None
                            else FULL_BODY_STANDING_ROOT_HEIGHT_M
                        )
                        target_root_y = (
                            plank_root_height
                            if plank_root_height is not None
                            else standing_origin_y - effective_root_drop
                        )
                        root[1] = (
                            start_root[1] * (1.0 - action_alpha)
                            + target_root_y * action_alpha
                            - push_up_drop
                        )
                    else:
                        standing_origin_y = (
                            float(pose_return_root[1])
                            if pose_return_root is not None
                            else FULL_BODY_STANDING_ROOT_HEIGHT_M
                        )
                        target_root_y = standing_origin_y - effective_root_drop
                        root[1] = (
                            start_root[1] * (1.0 - action_alpha)
                            + target_root_y * action_alpha
                        )
                    if authored.lock_feet:
                        local_foot_shifts = {
                            Hand.LEFT.value: np.asarray(
                                [
                                    authored.left_foot_shift_x_m,
                                    authored.left_foot_lift_m,
                                    authored.left_foot_shift_z_m,
                                ],
                                dtype=float,
                            ),
                            Hand.RIGHT.value: np.asarray(
                                [
                                    authored.right_foot_shift_x_m,
                                    authored.right_foot_lift_m,
                                    authored.right_foot_shift_z_m,
                                ],
                                dtype=float,
                            ),
                        }
                        world_foot_shifts = {
                            side: Rotation.from_euler("y", start_yaw).apply(shift)
                            for side, shift in local_foot_shifts.items()
                        }
                        moving_sides = [
                            side
                            for side, shift in world_foot_shifts.items()
                            if float(np.linalg.norm(shift)) > 1e-8
                        ]
                        if len(moving_sides) == 2:
                            first = target.lead_side.value
                            second = (
                                Hand.LEFT.value
                                if first == Hand.RIGHT.value
                                else Hand.RIGHT.value
                            )
                            ordered_sides = [first, second]
                        else:
                            ordered_sides = moving_sides
                        ankle_targets = {}
                        moving_now: str | None = None
                        for side in (Hand.LEFT.value, Hand.RIGHT.value):
                            if side not in ordered_sides:
                                foot_alpha = 0.0
                                foot_lift = 0.0
                            elif len(ordered_sides) == 1:
                                local_foot_progress = progress
                                foot_alpha = smoothstep(local_foot_progress, 0.78)
                                foot_lift = math.sin(math.pi * local_foot_progress) ** 1.5
                                moving_now = side if progress < 1.0 else None
                            else:
                                stage = ordered_sides.index(side)
                                local_foot_progress = float(
                                    np.clip(progress * 2.0 - stage, 0.0, 1.0)
                                )
                                foot_alpha = smoothstep(local_foot_progress, 0.78)
                                foot_lift = (
                                    math.sin(math.pi * local_foot_progress) ** 1.5
                                    if 0.0 < local_foot_progress < 1.0
                                    else 0.0
                                )
                                if 0.0 < local_foot_progress < 1.0:
                                    moving_now = side
                            ankle_targets[side] = (
                                gait_start_positions[side]
                                + world_foot_shifts[side] * foot_alpha
                                + np.asarray([0.0, 0.055 * foot_lift, 0.0], dtype=float)
                            )
                        leg_foot_rotations = gait_foot_rotations
                        support_side = (
                            Hand.LEFT.value
                            if moving_now == Hand.RIGHT.value
                            else Hand.RIGHT.value
                            if moving_now == Hand.LEFT.value
                            else Hand.RIGHT.value
                            if authored.left_foot_lift_m > 1e-8
                            and authored.right_foot_lift_m <= 1e-8
                            else Hand.LEFT.value
                            if authored.right_foot_lift_m > 1e-8
                            and authored.left_foot_lift_m <= 1e-8
                            else Hand.LEFT.value
                        )
                        leg_ik_active = True
                    elif (
                        authored.support_mode == BodySupportMode.PLANK
                        and plank_ankle_targets
                    ):
                        # Once the high plank is established, both feet are
                        # exact world-space supports while the torso descends.
                        ankle_targets = {
                            side: value.copy()
                            for side, value in plank_ankle_targets.items()
                        }
                        leg_foot_rotations = plank_foot_rotations
                        leg_ik_active = True
            if primitive.effectors and target.action != BodyAction.CLIMB:
                for effector in primitive.effectors:
                    path_target = _trajectory_target(
                        last_arm_targets.get(
                            effector.hand,
                            arm_idle_targets[effector.hand],
                        ),
                        arm_centers[effector.hand],
                        effector,
                        arm_trajectory,
                        primitive.trajectory_plane or TrajectoryPlane.FRONTAL,
                        articulation_progress,
                        action_alpha,
                        primitive.parameters.trajectory_amplitude_m,
                        arm_cycles,
                    )
                    arm, _ = arm_pose_from_target(
                        effector.hand,
                        shoulder_position(effector.hand),
                        path_target,
                        arm_parameters[effector.hand],
                        present_hand=False,
                    )
                    axial_amplitude = (
                        primitive.parameters.axial_rotation_amplitude * 0.72
                    )
                    if axial_amplitude > 1e-8 and arm_trajectory in {
                        TrajectoryKind.CIRCLE,
                        TrajectoryKind.OSCILLATE,
                    }:
                        axial_angle = 2.0 * math.pi * (
                            arm_cycles * articulation_progress
                            + effector.phase_offset_cycles
                        )
                        lower_arm = f"{effector.hand.value}LowerArm"
                        arm[lower_arm] = _local_rotation_offset(
                            arm[lower_arm],
                            [0.0, math.sin(axial_angle) * axial_amplitude, 0.0],
                        )
                    pose.update(arm)
                    pose.update(
                        hand_pose(
                            effector.hand,
                            effector.hand_shape,
                            arm_parameters[effector.hand],
                        )
                    )
            elif target.action == BodyAction.CLIMB:
                for effector in primitive.effectors:
                    pose.update(
                        hand_pose(
                            effector.hand,
                            effector.hand_shape,
                            arm_parameters[effector.hand],
                        )
                    )
            if (
                (not recover and target.action == BodyAction.POSE and not target.pose.lock_feet)
                or free_pose_recovery
            ):
                # Free-leg poses (a kneel or seated posture) author explicit
                # joint flexion rather than solving both ankles to the standing
                # contacts.  Lift the root only as much as needed to keep the
                # lowest toe on the floor throughout the transition/recovery.
                preliminary_bones = {
                    key: BonePose(
                        rotation=value,
                        position=(
                            Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2]))
                            if key == "hips"
                            else None
                        ),
                    )
                    for key, value in pose.items()
                }
                preliminary_positions = kinematics.canonical_positions(preliminary_bones)
                horizontal_support = bool(
                    previous_target is not None
                    and free_pose_recovery
                    and (
                        previous_target.pose.support_mode != BodySupportMode.FEET
                        or abs(previous_target.pose.pelvis_pitch_deg) >= 80.0
                    )
                    or (
                        not recover
                        and target.action == BodyAction.POSE
                        and (
                            target.pose.support_mode != BodySupportMode.FEET
                            or abs(target.pose.pelvis_pitch_deg) >= 80.0
                        )
                    )
                )
                if horizontal_support:
                    support_is_exactly_planted = bool(
                        not recover
                        and target.action == BodyAction.POSE
                        and target.pose.support_mode == BodySupportMode.PLANK
                        and plank_hand_targets is not None
                    )
                    support_bones = (
                        "hips",
                        "chest",
                        "upperChest",
                        "head",
                        "leftUpperArm",
                        "leftLowerArm",
                        "leftHand",
                        "rightUpperArm",
                        "rightLowerArm",
                        "rightHand",
                        "leftUpperLeg",
                        "leftLowerLeg",
                        "leftFoot",
                        "leftToes",
                        "rightUpperLeg",
                        "rightLowerLeg",
                        "rightFoot",
                        "rightToes",
                    )
                    if not support_is_exactly_planted:
                        lowest_body = min(
                            float(preliminary_positions[name][1])
                            for name in support_bones
                        )
                        horizontal_amount = (
                            1.0 - alpha
                            if recover
                            else min(
                                1.0,
                                max(
                                    abs(target.pose.pelvis_pitch_deg) / 90.0,
                                    abs(target.pose.pelvis_roll_deg) / 75.0,
                                    1.0
                                    if target.pose.support_mode != BodySupportMode.FEET
                                    else 0.0,
                                )
                                * alpha,
                            )
                        )
                        desired_clearance = (
                            neutral_ground_height + 0.055 * horizontal_amount
                        )
                        root[1] += max(0.0, desired_clearance - lowest_body)
                else:
                    lowest_toe = min(
                        float(preliminary_positions["leftToes"][1]),
                        float(preliminary_positions["rightToes"][1]),
                    )
                    root[1] += max(0.0, neutral_ground_height - lowest_toe)
            bones = {
                key: BonePose(
                    rotation=value,
                    position=(
                        Vec3(
                            x=float(root[0]),
                            y=float(root[1]),
                            z=float(root[2]),
                        )
                        if key == "hips"
                        else None
                    ),
                )
                for key, value in pose.items()
            }
            if leg_ik_active:
                for side in (Hand.LEFT.value, Hand.RIGHT.value):
                    desired_ankle = ankle_targets[side]
                    solve_target = desired_ankle.copy()
                    original_leg_rotations = {
                        bone_name: bones[bone_name].rotation
                        for bone_name in (
                            f"{side}UpperLeg",
                            f"{side}LowerLeg",
                            f"{side}Foot",
                        )
                    }
                    iterations = (
                        2
                        if target.action == BodyAction.POSE
                        and target.pose.support_mode == BodySupportMode.PLANK
                        and plank_ankle_targets
                        else 1
                    )
                    for _ in range(iterations):
                        solved = kinematics.solve_leg(
                            bones,
                            side,
                            solve_target,
                            foot_world_rotation=leg_foot_rotations[side],
                        )
                        for bone_name, rotation in solved.items():
                            bones[bone_name] = BonePose(rotation=rotation)
                        if iterations > 1:
                            actual_ankle = kinematics.canonical_positions(bones)[
                                f"{side}Foot"
                            ]
                            solve_target += desired_ankle - actual_ankle
                    if iterations > 1:
                        support_solve_alpha = smoothstep(
                            float(
                                np.clip(
                                    (progress - 0.20) / 0.18,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.78,
                        )
                        for bone_name, original_rotation in original_leg_rotations.items():
                            bones[bone_name] = BonePose(
                                rotation=_nlerp(
                                    original_rotation,
                                    bones[bone_name].rotation,
                                    support_solve_alpha,
                                )
                            )
            if target.action == BodyAction.CLIMB and climb_limb_targets:
                for effector in primitive.effectors:
                    hand_name = f"{effector.hand.value}Hand"
                    desired_hand = climb_limb_targets.get(hand_name)
                    if desired_hand is None:
                        continue
                    solved = kinematics.solve_arm(
                        bones,
                        effector.hand.value,
                        desired_hand,
                        bend_hint_world=np.asarray(
                            [
                                0.45
                                if effector.hand == Hand.LEFT
                                else -0.45,
                                0.15,
                                -0.35,
                            ],
                            dtype=float,
                        ),
                    )
                    for bone_name, rotation in solved.items():
                        bones[bone_name] = BonePose(
                            rotation=_nlerp(
                                bones[bone_name].rotation,
                                rotation,
                                climb_solve_alpha,
                            )
                        )
            if (
                not recover
                and target.action == BodyAction.POSE
                and target.pose.support_mode == BodySupportMode.PLANK
                and progress >= 0.20
            ):
                if plank_hand_targets is None:
                    support_positions = kinematics.canonical_positions(bones)
                    plank_root_height = float(root[1])
                    plank_hand_targets = {
                        side: support_positions[f"{side}Hand"].copy()
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    plank_hand_rotations = {
                        side: kinematics.canonical_world_rotation(
                            bones,
                            f"{side}Hand",
                        )
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    plank_ankle_targets = {
                        side: support_positions[f"{side}Foot"].copy()
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                    plank_foot_rotations = {
                        side: kinematics.canonical_world_rotation(
                            bones,
                            f"{side}Foot",
                        )
                        for side in (Hand.LEFT.value, Hand.RIGHT.value)
                    }
                else:
                    support_solve_alpha = smoothstep(
                        float(np.clip((progress - 0.20) / 0.18, 0.0, 1.0)),
                        0.78,
                    )
                    for side in (Hand.LEFT.value, Hand.RIGHT.value):
                        solved = kinematics.solve_arm(
                            bones,
                            side,
                            plank_hand_targets[side],
                            hand_world_rotation=plank_hand_rotations[side],
                            bend_hint_world=np.asarray(
                                [
                                    1.0 if side == Hand.LEFT.value else -1.0,
                                    0.35,
                                    -0.10,
                                ],
                                dtype=float,
                            ),
                        )
                        for bone_name, rotation in solved.items():
                            bones[bone_name] = BonePose(
                                rotation=_nlerp(
                                    bones[bone_name].rotation,
                                    rotation,
                                    support_solve_alpha,
                                )
                            )
            frames.append(
                ClipFrame(
                    time_s=elapsed + progress * phase_duration_s,
                    bones=bones,
                    objects={item.id: item.transform for item in scene.objects},
                )
            )
            if target.action == BodyAction.CLIMB and climb_limb_targets:
                climb_support_constraints.append(
                    {
                        "frame_index": len(frames) - 1,
                        "object_id": target.support_object_id,
                        "targets": {
                            name: value.copy()
                            for name, value in climb_limb_targets.items()
                        },
                        "settled": climb_solve_alpha >= 0.98,
                    }
                )
            if leg_ik_active and support_side is not None:
                constrained_sides = (
                    (Hand.LEFT.value, Hand.RIGHT.value)
                    if gait_recovery and not pose_foot_recovery
                    or (
                        target.action == BodyAction.POSE
                        and target.pose.lock_feet
                        and not target.pose.has_foot_motion()
                    )
                    else (support_side,)
                )
                for constrained_side in constrained_sides:
                    support_constraints.append(
                        {
                            "frame_index": len(frames) - 1,
                            "side": constrained_side,
                            "ankle_target": ankle_targets[constrained_side].copy(),
                        }
                    )

        current_pose = {
            name: bone_pose.rotation
            for name, bone_pose in frames[-1].bones.items()
        }
        current_root = np.asarray(
            frames[-1].bones["hips"].position.as_list(),
            dtype=float,
        )
        if target.action == BodyAction.TURN and not recover:
            current_yaw = start_yaw + math.radians(target.turn_degrees)
        if recover:
            # Recovery restores the actual arm pose to the calibrated idle
            # configuration.  Keeping a stale floor/overhead effector target
            # here makes the next authored arm phase snap back to the prior
            # support point on its first frame.
            last_arm_targets = arm_idle_targets.copy()
        else:
            for hand, center in arm_centers.items():
                last_arm_targets[hand] = center
        previous_target = target
        elapsed += phase_duration_s

    if any(
        primitive.body is not None
        and primitive.body.action == BodyAction.CLIMB
        for primitive in program.primitives
    ):
        _smooth_composite_joint_paths(
            frames,
            [Hand.LEFT, Hand.RIGHT],
            passes=3,
            include_legs=True,
        )

    metrics = _base_metrics()
    metrics["phase_ranges_s"] = phase_ranges
    metrics["body_actions"] = actions
    metrics["root_motion_enabled"] = True
    metrics["support_constraints"] = [
        {
            "frame_index": int(item["frame_index"]),
            "side": str(item["side"]),
            "ankle_target": [float(value) for value in item["ankle_target"]],
        }
        for item in support_constraints
    ]
    metrics["climb_support_constraints"] = [
        {
            "frame_index": int(item["frame_index"]),
            "object_id": item["object_id"],
            "targets": {
                name: [float(value) for value in target]
                for name, target in item["targets"].items()
            },
            "settled": bool(item["settled"]),
        }
        for item in climb_support_constraints
    ]
    # The measurement pass lives in ``analysis.full_body`` from 02b onwards.
    # Compiling and analysing therefore run the same code rather than two
    # implementations that agree by inspection, and the same pass can re-derive
    # these numbers from a stored clip with no recompilation. The two records
    # above are its only inputs that frames cannot supply: they are the ankle
    # and limb positions the IK solver was *commanded* to reach, which is
    # authoring intent and not observable in the result.
    metrics.update(
        _full_body_metrics(
            _AnalysisContext.from_frames(
                frames,
                program,
                scene,
                metrics,
                kinematics=kinematics,
            )
        )
    )
    structural_failures = list(metrics["structural_failures"])
    success = bool(frames and not structural_failures)
    observables = _slider_observables(scene, program)
    observables.update(
        {
            "root_path_length_m": float(metrics["root_path_length_m"]),
            "root_displacement_m": float(metrics["root_displacement_m"]),
            "root_turn_degrees": float(metrics["final_root_yaw_deg"]),
            "root_vertical_amplitude_m": float(
                metrics["root_vertical_max_m"]
                - metrics["root_vertical_min_m"]
            ),
        }
    )
    return ClipResult(
        success=success,
        fps=fps,
        duration_s=elapsed,
        frames=frames,
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=(
            None
            if success
            else Failure(
                code=FailureCode.JOINT_LIMIT_EXCEEDED,
                message="; ".join(structural_failures)
                or "Full-body movement failed structural checks",
                details=metrics,
                recoverable=True,
            )
        ),
    )


def _travel_wheel_target(
    center: Vec3,
    hand: Hand,
    progress: float,
    cycles: float,
    orbit_radius_m: float,
) -> Vec3:
    """Put each fist across the chest on one of two orbiting parallel rods.

    Each fist stays near the opposite elbow. The two complete forearms rotate
    as a phase-locked pair about their shared cross-body axis, exchanging
    over/under and front/back order while retaining physical clearance.
    """

    side = 1.0 if hand == Hand.LEFT else -1.0
    angle = 2.0 * math.pi * cycles * progress
    half_forearm_span = 0.135
    depth_radius = min(orbit_radius_m, 0.055)
    return Vec3(
        x=center.x - side * half_forearm_span,
        y=center.y + side * orbit_radius_m * math.cos(angle),
        z=center.z + side * depth_radius * math.sin(angle),
    )


def _travel_wheel_elbow_hint(
    center: Vec3,
    hand: Hand,
    progress: float,
    cycles: float,
    orbit_radius_m: float,
) -> Vec3:
    side = 1.0 if hand == Hand.LEFT else -1.0
    angle = 2.0 * math.pi * cycles * progress
    half_forearm_span = 0.135
    depth_radius = min(orbit_radius_m, 0.055)
    return Vec3(
        x=center.x + side * half_forearm_span,
        y=center.y + side * orbit_radius_m * math.cos(angle),
        z=center.z + side * depth_radius * math.sin(angle),
    )


def _head_gaze_rotation(target: Vec3) -> Quat:
    """Aim the canonical egocentric gaze at a world-space target."""

    kinematics = rig_kinematics()
    neutral = {
        name: BonePose(rotation=value)
        for name, value in _identity_pose().items()
    }
    head = kinematics.canonical_positions(neutral)["head"]
    direction = np.asarray(target.as_list(), dtype=float) - head
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    rotation, _ = Rotation.align_vectors(
        np.asarray([direction]),
        np.asarray([_EGO_NEUTRAL_GAZE]),
    )
    return kinematics.world_delta_quat("head", rotation.as_matrix())


def _compile_composite(scene: SceneManifest, program: MotionProgram) -> ClipResult:
    """Compile concurrent, phase-based upper-body end-effector trajectories."""

    base = _gesture_idle_pose()
    dexterous_program = any(
        primitive.intra_hand_contact is not None
        for primitive in program.primitives
    )
    current = base.copy()
    idle_targets = {
        Hand.LEFT: Vec3(x=0.27, y=1.14, z=0.24),
        Hand.RIGHT: Vec3(x=-0.27, y=1.14, z=0.24),
    }
    last_targets = {hand: idle_targets[hand] for hand in program.hands}
    frames: list[ClipFrame] = []
    phase_ranges: list[dict[str, float | str]] = []
    presentation_ranges: list[tuple[float, float]] = []
    elapsed = 0.0
    fps = scene.fps

    for primitive in program.primitives:
        start_pose = current.copy()
        recover = primitive.kind == PrimitiveKind.RECOVER
        target_pose = base.copy() if recover else current.copy()
        travel_setup = primitive.label == "parallel_forearm_travel_setup"
        travel_cycle = primitive.label == "parallel_forearm_travel_cycle"
        centers: dict[Hand, Vec3] = {}
        effector_parameters: dict[Hand, PrimitiveParameters] = {}
        for effector in primitive.effectors:
            center = (
                idle_targets[effector.hand]
                if recover
                else _composite_workspace_target(effector)
            )
            centers[effector.hand] = center
            parameters = _composite_parameters(primitive.parameters, effector)
            effector_parameters[effector.hand] = parameters
        travel_shared_center: Vec3 | None = None
        if (travel_setup or travel_cycle) and len(centers) == 2:
            travel_shared_center = Vec3(
                x=sum(value.x for value in centers.values()) / 2.0,
                y=sum(value.y for value in centers.values()) / 2.0,
                z=sum(value.z for value in centers.values()) / 2.0,
            )
            wheel_radius = (
                float(
                    np.clip(
                        primitive.parameters.trajectory_amplitude_m,
                        0.080,
                        0.100,
                    )
                )
                if travel_cycle
                else 0.090
            )
            centers = {
                hand: _travel_wheel_target(
                    travel_shared_center,
                    hand,
                    0.0,
                    primitive.parameters.trajectory_cycles,
                    wheel_radius,
                )
                for hand in centers
            }
        if not recover:
            for effector in primitive.effectors:
                elbow_hint = (
                    _travel_wheel_elbow_hint(
                        travel_shared_center,
                        effector.hand,
                        0.0,
                        primitive.parameters.trajectory_cycles,
                        wheel_radius,
                    )
                    if travel_shared_center is not None
                    else None
                )
                arm, _ = arm_pose_from_target(
                    effector.hand,
                    shoulder_position(effector.hand),
                    centers[effector.hand],
                    effector_parameters[effector.hand],
                    present_hand=False,
                    elbow_hint=elbow_hint,
                )
                target_pose.update(arm)
                target_pose.update(
                    hand_pose(
                        effector.hand,
                        effector.hand_shape,
                        effector_parameters[effector.hand],
                    )
                )
            if primitive.intra_hand_contact is not None:
                target_pose.update(
                    thumb_to_fingertip_pose(
                        primitive.intra_hand_contact.hand,
                        primitive.intra_hand_contact.target_digit,
                    )
                )
            if primitive.gaze_target is not None:
                if primitive.gaze_target.hand is not None:
                    gaze_position = centers[primitive.gaze_target.hand]
                else:
                    gaze_object = scene.object_by_id(
                        primitive.gaze_target.object_id or ""
                    )
                    if gaze_object is None:
                        raise ValueError("gaze target object is not in the scene")
                    gaze_position = gaze_object.transform.translation
                target_pose["head"] = _head_gaze_rotation(gaze_position)
        if not recover:
            target_pose["chest"] = Quat(
                y=math.sin(primitive.parameters.torso_participation * 0.08),
                w=math.cos(primitive.parameters.torso_participation * 0.08),
            )

        trajectory = primitive.trajectory or TrajectoryKind.LINEAR
        plane = primitive.trajectory_plane or TrajectoryPlane.FRONTAL
        cycles = primitive.parameters.trajectory_cycles
        requested_frames = max(2, int(round(primitive.parameters.duration_s * fps)))
        if trajectory in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}:
            requested_frames = max(requested_frames, int(math.ceil(cycles * 24.0)) + 1)
        target_delta = max(
            2.0
            * math.acos(
                float(
                    np.clip(
                        abs(np.dot(start_pose[key].as_list(), target_pose[key].as_list())),
                        0.0,
                        1.0,
                    )
                )
            )
            for key in base
        )
        frame_count = max(requested_frames, subdivision_frames(target_delta, NLERP_SUBDIVISION_RAD))
        phase_duration_s = max(primitive.parameters.duration_s, (frame_count - 1) / fps)
        phase_ranges.append(
            {
                "kind": primitive.kind.value,
                "label": primitive.label or primitive.kind.value,
                "start_s": elapsed,
                "end_s": elapsed + phase_duration_s,
            }
        )
        if not recover:
            # A relaxed rest pose is intentionally allowed to sit below the
            # egocentric frame.  Visibility is enforced once the authored
            # movement has visibly begun, matching the capture sampler's
            # first meaningful phase rather than treating frame zero as a
            # failed presentation.
            visibility_start = elapsed + (
                phase_duration_s * 0.50
                if travel_setup
                else min(0.15, phase_duration_s * 0.25)
            )
            presentation_ranges.append((visibility_start, elapsed + phase_duration_s))

        for local_index in range(frame_count):
            if frames and local_index == 0:
                continue
            progress = local_index / (frame_count - 1)
            alpha = smoothstep(progress, primitive.parameters.easing)
            pose = {
                key: BonePose(rotation=_nlerp(start_pose[key], target_pose[key], alpha))
                for key in base
            }
            if not recover:
                for effector in primitive.effectors:
                    if travel_cycle and travel_shared_center is not None:
                        path_target = _travel_wheel_target(
                            travel_shared_center,
                            effector.hand,
                            progress,
                            cycles,
                            float(
                                np.clip(
                                    primitive.parameters.trajectory_amplitude_m,
                                    0.080,
                                    0.100,
                                )
                            ),
                        )
                    else:
                        path_target = _trajectory_target(
                            last_targets.get(
                                effector.hand,
                                idle_targets[effector.hand],
                            ),
                            centers[effector.hand],
                            effector,
                            trajectory,
                            plane,
                            progress,
                            alpha,
                            primitive.parameters.trajectory_amplitude_m,
                            cycles,
                        )
                    arm, _ = arm_pose_from_target(
                        effector.hand,
                        shoulder_position(effector.hand),
                        path_target,
                        effector_parameters[effector.hand],
                        present_hand=False,
                        elbow_hint=(
                            _travel_wheel_elbow_hint(
                                travel_shared_center,
                                effector.hand,
                                progress,
                                cycles,
                                float(
                                    np.clip(
                                        primitive.parameters.trajectory_amplitude_m,
                                        0.080,
                                        0.100,
                                    )
                                ),
                            )
                            if travel_shared_center is not None
                            else None
                        ),
                        # The setup phase carries the hand from idle to the
                        # wheel along this alpha; applied at full strength
                        # from frame zero, the wheel's elbow hint demands a
                        # bend plane the idle-adjacent targets can only reach
                        # through out-of-band humeral twist.  Carrying the
                        # hint in on the same blend keeps the plane with the
                        # hand it belongs to.
                        elbow_hint_weight=alpha if travel_setup else 1.0,
                    )
                    axial_amplitude = primitive.parameters.axial_rotation_amplitude * 0.72
                    if axial_amplitude > 1e-8 and trajectory in {
                        TrajectoryKind.CIRCLE,
                        TrajectoryKind.OSCILLATE,
                    }:
                        axial_angle = 2.0 * math.pi * (
                            cycles * progress + effector.phase_offset_cycles
                        )
                        lower_arm = f"{effector.hand.value}LowerArm"
                        arm[lower_arm] = _local_rotation_offset(
                            arm[lower_arm],
                            [
                                0.0,
                                # Integer-cycle phases already cross zero at
                                # both boundaries. A whole-phase envelope
                                # erased the first and last visible reversals.
                                math.sin(axial_angle) * axial_amplitude,
                                0.0,
                            ],
                        )
                    pose.update(
                        {name: BonePose(rotation=rotation) for name, rotation in arm.items()}
                    )
                    if not dexterous_program:
                        pose.update(
                            {
                                name: BonePose(rotation=rotation)
                                for name, rotation in hand_pose(
                                    effector.hand,
                                    effector.hand_shape,
                                    effector_parameters[effector.hand],
                                ).items()
                            }
                        )
            now = elapsed + local_index / fps
            frames.append(
                ClipFrame(
                    time_s=now,
                    bones=pose,
                    objects={item.id: item.transform for item in scene.objects},
                )
            )
        current = target_pose
        for hand, center in centers.items():
            last_targets[hand] = center
        elapsed += phase_duration_s

    _smooth_composite_joint_paths(frames, program.hands)

    metrics = _base_metrics()
    metrics["phase_ranges_s"] = phase_ranges
    # Authoring intent, not observation, and not recoverable from phase_ranges_s.
    # The window opens part-way into each phase, at a fraction of the phase's
    # *authored* duration -- but phase_ranges_s stores start and end as running
    # sums, so ``end_s - start_s`` carries whatever rounding ``elapsed`` has
    # accumulated by then. On a ten-phase finger count that is enough to move
    # the window by one ulp, which is enough to move a metric derived from it.
    # Persist the interval rather than have a post-hoc pass re-derive it wrong.
    metrics["presentation_ranges_s"] = [
        [float(start), float(end)] for start, end in presentation_ranges
    ]
    metrics.update(
        _composite_metrics(
            _AnalysisContext.from_frames(frames, program, scene, metrics)
        )
    )
    structural_failures = list(metrics["structural_failures"])
    success = bool(
        frames
        and not structural_failures
        and metrics["nan_count"] == 0
        and _clip_contract_violations(metrics, policy="fixed") == 0
    )
    failure = None
    if not success:
        failure = Failure(
            code=(
                FailureCode.COLLISION_UNRESOLVED
                if metrics.get("self_collision_frames", 0)
                else FailureCode.JOINT_LIMIT_EXCEEDED
            ),
            message="; ".join(structural_failures) or "Composite movement failed structural checks",
            details=metrics,
            recoverable=True,
        )
    observables = _slider_observables(scene, program)
    observables.update(
        {
            "active_hand_count": float(len(program.hands)),
            "trajectory_cycles": float(metrics["trajectory_cycles"]),
            "trajectory_amplitude_m": float(metrics["trajectory_amplitude_m"]),
            "axial_rotation_amplitude": float(metrics["axial_rotation_amplitude"]),
        }
    )
    return ClipResult(
        success=success,
        fps=fps,
        duration_s=elapsed,
        frames=frames,
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=failure,
    )


def _object_interaction_arm_target(
    program: MotionProgram,
    target_object: Any,
    kind: PrimitiveKind,
) -> Vec3 | None:
    """Return a wrist target for one semantic object-lifecycle phase."""

    motion = program.object_motion
    action = program.object_action
    if motion is None or action is None:
        return None
    side = 1.0 if program.hand == Hand.LEFT else -1.0
    center = target_object.transform.translation
    grasp = Vec3(x=center.x, y=center.y, z=center.z - 0.085)
    if action == ObjectAction.PLACE:
        direction = np.asarray(
            [motion.direction_x, 0.0, motion.direction_z],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        placed = Vec3(
            x=center.x + float(direction[0]) * motion.distance_m,
            y=motion.landing_height_m,
            z=center.z + float(direction[2]) * motion.distance_m - 0.085,
        )
        lifted = Vec3(x=grasp.x, y=grasp.y + 0.12, z=grasp.z)
        carried = Vec3(x=placed.x, y=max(placed.y, grasp.y) + 0.12, z=placed.z)
        if kind in {
            PrimitiveKind.REACH,
            PrimitiveKind.PRESHAPE,
            PrimitiveKind.CONTACT,
            PrimitiveKind.CLOSE,
        }:
            return grasp
        if kind == PrimitiveKind.LIFT:
            return lifted
        if kind == PrimitiveKind.MOVE:
            return carried
        if kind == PrimitiveKind.RELEASE:
            return placed
        return None
    if action == ObjectAction.DROP:
        lifted = Vec3(x=grasp.x, y=grasp.y + 0.12, z=grasp.z)
        if kind in {
            PrimitiveKind.REACH,
            PrimitiveKind.PRESHAPE,
            PrimitiveKind.CONTACT,
            PrimitiveKind.CLOSE,
        }:
            return grasp
        if kind in {
            PrimitiveKind.LIFT,
            PrimitiveKind.HOLD,
            PrimitiveKind.RELEASE,
            PrimitiveKind.FLIGHT,
        }:
            return lifted
        return None
    if action in {
        ObjectAction.PUSH,
        ObjectAction.PULL,
        ObjectAction.ROLL,
        ObjectAction.SPIN,
    }:
        direction = np.asarray(
            [motion.direction_x, 0.0, motion.direction_z],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        guided_end = (
            grasp
            if action == ObjectAction.SPIN
            else Vec3(
                x=grasp.x + float(direction[0]) * motion.distance_m,
                y=grasp.y,
                z=grasp.z + float(direction[2]) * motion.distance_m,
            )
        )
        if kind in {
            PrimitiveKind.REACH,
            PrimitiveKind.PRESHAPE,
            PrimitiveKind.CONTACT,
            PrimitiveKind.CLOSE,
        }:
            return grasp
        if kind in {PrimitiveKind.MOVE, PrimitiveKind.HOLD}:
            return guided_end
        return None
    if action == ObjectAction.CATCH:
        contact = Vec3(
            x=side * 0.20,
            y=motion.contact_height_m,
            z=motion.contact_depth_m,
        )
        receive = Vec3(x=contact.x, y=contact.y, z=contact.z - 0.085)
        if kind in {
            PrimitiveKind.RECEIVE,
            PrimitiveKind.FLIGHT,
            PrimitiveKind.CONTACT,
            PrimitiveKind.CLOSE,
        }:
            return receive
        if kind == PrimitiveKind.ABSORB:
            return Vec3(x=side * 0.19, y=contact.y - 0.10, z=max(0.16, contact.z - 0.12))
        return Vec3(x=side * 0.18, y=1.17, z=0.19)
    if kind in {
        PrimitiveKind.REACH,
        PrimitiveKind.PRESHAPE,
        PrimitiveKind.CONTACT,
        PrimitiveKind.CLOSE,
    }:
        return grasp
    if kind in {PrimitiveKind.LIFT, PrimitiveKind.HOLD}:
        return Vec3(x=grasp.x, y=grasp.y + 0.12, z=grasp.z)
    windup_release = {
        ObjectInteractionStyle.OVERHAND: (
            Vec3(x=side * 0.28, y=1.53, z=0.02),
            Vec3(x=side * 0.19, y=1.50, z=0.39),
        ),
        ObjectInteractionStyle.UNDERHAND: (
            Vec3(x=side * 0.25, y=1.00, z=0.08),
            Vec3(x=side * 0.17, y=1.23, z=0.43),
        ),
        ObjectInteractionStyle.SIDEARM: (
            Vec3(x=side * 0.40, y=1.25, z=0.08),
            Vec3(x=side * 0.17, y=1.28, z=0.44),
        ),
        ObjectInteractionStyle.TOSS: (
            Vec3(x=side * 0.22, y=1.09, z=0.19),
            Vec3(x=side * 0.16, y=1.37, z=0.39),
        ),
    }[motion.style]
    windup, release = windup_release
    if kind == PrimitiveKind.WINDUP:
        return windup
    if kind == PrimitiveKind.RELEASE:
        return release
    if kind == PrimitiveKind.FLIGHT:
        direction = np.asarray([motion.direction_x, 0.0, motion.direction_z], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        return Vec3(
            x=release.x + float(direction[0]) * 0.07,
            y=release.y - 0.04,
            z=release.z + float(direction[2]) * 0.08,
        )
    return None


def _ballistic_flight(
    start: np.ndarray,
    end: np.ndarray,
    apex_height_m: float,
) -> tuple[float, np.ndarray]:
    """Solve the unique gravity trajectory that clears the requested apex."""

    gravity = 9.81
    apex_y = max(float(start[1]), float(end[1])) + apex_height_m
    vertical_speed = math.sqrt(max(0.0, 2.0 * gravity * (apex_y - float(start[1]))))
    discriminant = vertical_speed * vertical_speed - 2.0 * gravity * (
        float(end[1]) - float(start[1])
    )
    duration = (vertical_speed + math.sqrt(max(0.0, discriminant))) / gravity
    duration = max(duration, 0.20)
    velocity = (end - start) / duration
    velocity[1] = vertical_speed
    return duration, velocity


def _compile_object_handoff(
    scene: SceneManifest,
    program: MotionProgram,
    target_object: Any,
) -> ClipResult:
    """Compile a continuous source-hand to receiver-hand ownership transfer."""

    source = program.hand
    receiver = next((hand for hand in program.hands if hand != source), None)
    if receiver is None:
        return _failure_result(
            scene,
            program,
            FailureCode.INVALID_PROGRAM,
            "Object handoff requires a distinct receiver hand",
        )

    base = _gesture_idle_pose()
    current = base.copy()
    frames: list[ClipFrame] = []
    contacts: list[ContactEvent] = []
    phase_ranges: list[dict[str, float | str]] = []
    elapsed = 0.0
    center = target_object.transform.translation
    source_side = 1.0 if source == Hand.LEFT else -1.0
    receiver_side = 1.0 if receiver == Hand.LEFT else -1.0
    grasp = Vec3(x=center.x, y=center.y, z=center.z - 0.085)
    transfer_y = max(1.20, center.y + 0.15)
    transfer_z = max(0.30, center.z + 0.02)
    source_transfer = Vec3(
        x=source_side * 0.045,
        y=transfer_y,
        z=transfer_z,
    )
    receiver_transfer = Vec3(
        x=receiver_side * 0.045,
        y=transfer_y,
        z=transfer_z,
    )
    receiver_hold = Vec3(
        x=receiver_side * 0.15,
        y=1.20,
        z=0.25,
    )

    current_object_position = np.asarray(center.as_list(), dtype=float)
    current_object_rotation = Rotation.from_quat(
        target_object.transform.rotation.as_list()
    )
    owner: Hand | None = None
    owner_local_offsets: dict[Hand, np.ndarray] = {}
    owner_local_rotations: dict[Hand, Rotation] = {}
    source_attachment_time_s: float | None = None
    receiver_contact_time_s: float | None = None
    transfer_time_s: float | None = None
    object_steps: list[float] = []
    attachment_offsets: dict[Hand, list[np.ndarray]] = {
        source: [],
        receiver: [],
    }
    lifecycle: list[dict[str, float | str]] = [
        {"state": "free", "time_s": 0.0},
    ]

    def update_hand_target(
        pose: dict[str, Quat],
        hand: Hand,
        target: Vec3,
        shape: HandShape,
        parameters: PrimitiveParameters,
    ) -> None:
        arm, _ = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            parameters,
            present_hand=False,
        )
        pose.update(arm)
        pose.update(hand_pose(hand, shape, parameters))

    def reset_hand(pose: dict[str, Quat], hand: Hand) -> None:
        prefix = hand.value
        for name, rotation in base.items():
            if name.startswith(prefix):
                pose[name] = rotation

    def add_contacts(time_s: float, hand: Hand) -> None:
        for digit, x_offset in (("thumb", -0.018), ("index", 0.018)):
            contacts.append(
                ContactEvent(
                    time_s=time_s,
                    hand=hand,
                    object_id=target_object.id,
                    digit=digit,
                    position=Vec3(
                        x=float(current_object_position[0] + x_offset),
                        y=float(current_object_position[1]),
                        z=float(current_object_position[2]),
                    ),
                    normal_force_n=4.0,
                )
            )

    for primitive in program.primitives:
        start = current.copy()
        target_pose = current.copy()
        kind = primitive.kind
        if kind in {
            PrimitiveKind.REACH,
            PrimitiveKind.PRESHAPE,
            PrimitiveKind.CONTACT,
            PrimitiveKind.CLOSE,
        }:
            update_hand_target(
                target_pose,
                source,
                grasp,
                primitive.hand_shape or HandShape.OPEN,
                primitive.parameters,
            )
        elif kind == PrimitiveKind.LIFT:
            update_hand_target(
                target_pose,
                source,
                source_transfer,
                HandShape.FIST,
                primitive.parameters,
            )
        elif kind == PrimitiveKind.RECEIVE:
            update_hand_target(
                target_pose,
                source,
                source_transfer,
                HandShape.FIST,
                primitive.parameters,
            )
            update_hand_target(
                target_pose,
                receiver,
                receiver_transfer,
                HandShape.OPEN,
                primitive.parameters.model_copy(
                    update={"elbow_swivel": -primitive.parameters.elbow_swivel}
                ),
            )
        elif kind == PrimitiveKind.RELEASE:
            update_hand_target(
                target_pose,
                source,
                source_transfer,
                HandShape.OPEN,
                primitive.parameters.model_copy(
                    update={"finger_curl": 0.0, "grip_force": 0.0}
                ),
            )
            update_hand_target(
                target_pose,
                receiver,
                receiver_transfer,
                HandShape.FIST,
                primitive.parameters.model_copy(
                    update={
                        "finger_curl": 0.62,
                        "thumb_opposition": 0.90,
                        "grip_force": 0.74,
                        "elbow_swivel": -primitive.parameters.elbow_swivel,
                    }
                ),
            )
        elif kind in {PrimitiveKind.HOLD, PrimitiveKind.RECOVER}:
            reset_hand(target_pose, source)
            update_hand_target(
                target_pose,
                receiver,
                receiver_hold,
                HandShape.FIST,
                primitive.parameters.model_copy(
                    update={
                        "finger_curl": 0.62,
                        "thumb_opposition": 0.90,
                        "grip_force": 0.74,
                        "elbow_swivel": -primitive.parameters.elbow_swivel,
                    }
                ),
            )
        else:
            return _failure_result(
                scene,
                program,
                FailureCode.INVALID_PROGRAM,
                f"Unsupported handoff phase: {kind.value}",
            )

        requested_duration = primitive.parameters.duration_s
        frame_count = max(2, int(round(requested_duration * scene.fps)) + 1)
        target_delta = max(
            2.0
            * math.acos(
                float(
                    np.clip(
                        abs(np.dot(start[name].as_list(), target_pose[name].as_list())),
                        0.0,
                        1.0,
                    )
                )
            )
            for name in base
        )
        frame_count = max(frame_count, subdivision_frames(target_delta, NLERP_SUBDIVISION_RAD))
        phase_duration_s = max(requested_duration, (frame_count - 1) / scene.fps)
        phase_ranges.append(
            {
                "kind": kind.value,
                "label": primitive.label or kind.value,
                "start_s": elapsed,
                "end_s": elapsed + phase_duration_s,
            }
        )

        for local_index in range(frame_count):
            if frames and local_index == 0:
                continue
            progress = local_index / (frame_count - 1)
            alpha = smoothstep(progress, primitive.parameters.easing)
            bones = {
                name: BonePose(rotation=_nlerp(start[name], target_pose[name], alpha))
                for name in base
            }
            now = elapsed + progress * phase_duration_s
            provisional = ClipFrame(time_s=now, bones=bones, objects={})
            landmarks: dict[Hand, tuple[np.ndarray, Rotation]] = {}
            for active_hand in (source, receiver):
                _, _, wrist, hand_world = arm_landmarks(
                    provisional,
                    active_hand,
                )
                landmarks[active_hand] = (wrist, hand_world)

            if kind == PrimitiveKind.CLOSE and owner is None:
                wrist, hand_world = landmarks[source]
                owner = source
                owner_local_offsets[source] = hand_world.inv().apply(
                    current_object_position - wrist
                )
                owner_local_rotations[source] = (
                    hand_world.inv() * current_object_rotation
                )
                source_attachment_time_s = now
                lifecycle.append({"state": "source_attached", "time_s": now})
                add_contacts(now, source)

            if (
                kind == PrimitiveKind.RECEIVE
                and receiver_contact_time_s is None
                and progress >= 0.72
            ):
                receiver_contact_time_s = now
                lifecycle.append({"state": "dual_contact", "time_s": now})
                add_contacts(now, receiver)

            if (
                kind == PrimitiveKind.RELEASE
                and owner == source
                and receiver_contact_time_s is not None
                and progress >= 0.55
            ):
                wrist, hand_world = landmarks[receiver]
                owner_local_offsets[receiver] = hand_world.inv().apply(
                    current_object_position - wrist
                )
                owner_local_rotations[receiver] = (
                    hand_world.inv() * current_object_rotation
                )
                owner = receiver
                transfer_time_s = now
                lifecycle.append({"state": "receiver_attached", "time_s": now})

            if owner is not None:
                wrist, hand_world = landmarks[owner]
                current_object_position = wrist + hand_world.apply(
                    owner_local_offsets[owner]
                )
                current_object_rotation = hand_world * owner_local_rotations[owner]
                attachment_offsets[owner].append(
                    hand_world.inv().apply(current_object_position - wrist)
                )

            objects = {item.id: item.transform for item in scene.objects}
            quat = current_object_rotation.as_quat()
            objects[target_object.id] = Transform(
                translation=Vec3(
                    x=float(current_object_position[0]),
                    y=float(current_object_position[1]),
                    z=float(current_object_position[2]),
                ),
                rotation=Quat(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3]),
                ),
            )
            if frames:
                previous = np.asarray(
                    frames[-1].objects[target_object.id].translation.as_list(),
                    dtype=float,
                )
                object_steps.append(
                    float(np.linalg.norm(current_object_position - previous))
                )
            frames.append(ClipFrame(time_s=now, bones=bones, objects=objects))

        current = target_pose
        elapsed += phase_duration_s

    metrics = _base_metrics()
    metrics["phase_ranges_s"] = phase_ranges
    # Carried, not measured. The lifecycle transitions fire on authored phase
    # kinds at fixed progress fractions, so rebuilding them post-hoc would mean
    # recomputing ``elapsed + progress * phase_duration_s`` from phase_ranges_s
    # -- running sums, which is the arithmetic that moved the composite
    # presentation window by an ulp in 02c. It is already a published key, so
    # writing it before the measurement pass costs nothing.
    metrics["object_lifecycle"] = lifecycle
    metrics.update(
        _handoff_metrics(
            _AnalysisContext.from_frames(frames, program, scene, metrics)
        )
    )
    structural_failures = list(metrics["structural_failures"])
    success = not structural_failures
    return ClipResult(
        success=success,
        fps=scene.fps,
        duration_s=elapsed,
        frames=frames,
        contacts=contacts,
        metrics=metrics,
        slider_observables=_slider_observables(scene, program),
        parametric_observables=_slider_observables(scene, program),
        provenance=_provenance(program),
        failure=(
            None
            if success
            else Failure(
                code=(
                    FailureCode.JOINT_LIMIT_EXCEEDED
                    if _clip_contract_violations(metrics, policy="fixed")
                    else FailureCode.INVALID_PROGRAM
                ),
                message="; ".join(structural_failures),
                details=metrics,
                recoverable=True,
            )
        ),
    )


def _compile_object_interaction(scene: SceneManifest, program: MotionProgram) -> ClipResult:
    """Compile contact, attachment, free flight, and interception as one lifecycle."""

    object_id = next((item.object_id for item in program.primitives if item.object_id), None)
    target_object = scene.object_by_id(object_id or "")
    if target_object is None:
        return _failure_result(
            scene,
            program,
            FailureCode.UNKNOWN_OBJECT,
            f"Unknown object: {object_id}",
        )
    if target_object is scene.support_surface():
        return _failure_result(
            scene,
            program,
            FailureCode.UNKNOWN_OBJECT,
            f"Support surfaces are not interaction targets: {object_id}",
        )
    if program.object_action is None or program.object_motion is None:
        return _failure_result(
            scene,
            program,
            FailureCode.INVALID_PROGRAM,
            "Object interaction is missing its lifecycle target",
        )
    if program.object_action == ObjectAction.HANDOFF:
        return _compile_object_handoff(scene, program, target_object)
    if program.object_action in {
        ObjectAction.THROW,
        ObjectAction.PUSH,
        ObjectAction.PULL,
        ObjectAction.ROLL,
        ObjectAction.SPIN,
        ObjectAction.PLACE,
        ObjectAction.DROP,
    }:
        shoulder = np.asarray(shoulder_position(program.hand).as_list(), dtype=float)
        center = np.asarray(target_object.transform.translation.as_list(), dtype=float)
        distance = float(np.linalg.norm(center - shoulder))
        effective_reach_m = min(scene.reachable_radius_m, arm_reach_m(program.hand) - 1e-4)
        if center[2] <= shoulder[2] or distance > effective_reach_m:
            return _failure_result(
                scene,
                program,
                FailureCode.UNREACHABLE_TARGET,
                "Source object is outside the reachable forward workspace",
                {"distance_m": distance, "effective_limit_m": effective_reach_m},
            )

    physics = (
        _physics_for(program, scene)
        if program.object_action in {ObjectAction.THROW, ObjectAction.DROP}
        else None
    )
    base = _gesture_idle_pose()
    current = base.copy()
    frames: list[ClipFrame] = []
    phase_ranges: list[dict[str, float | str]] = []
    elapsed = 0.0
    side = 1.0 if program.hand == Hand.LEFT else -1.0
    original_position = np.asarray(target_object.transform.translation.as_list(), dtype=float)
    original_rotation = Rotation.from_quat(target_object.transform.rotation.as_list())
    current_object_position = original_position.copy()
    current_object_rotation = original_rotation
    object_half_extents = np.asarray(
        [
            target_object.dimensions_m.x,
            target_object.dimensions_m.y,
            target_object.dimensions_m.z,
        ],
        dtype=float,
    ) * 0.5
    original_vertical_extent = float(
        np.dot(np.abs(original_rotation.as_matrix()[1]), object_half_extents)
    )
    object_support_plane_y = float(original_position[1] - original_vertical_extent)
    attachment_local_offset: np.ndarray | None = None
    attachment_local_rotation: Rotation | None = None
    attachment_world_offset: np.ndarray | None = None
    attachment_started_s: float | None = None
    release_time_s: float | None = None
    catch_time_s: float | None = None
    release_position: np.ndarray | None = None
    release_rotation: Rotation | None = None
    flight_start_s: float | None = None
    flight_duration_s: float | None = None
    flight_velocity: np.ndarray | None = None
    flight_end = original_position.copy()
    landed_position: np.ndarray | None = None
    landed_rotation: Rotation | None = None
    object_steps: list[float] = []
    attachment_offsets: list[np.ndarray] = []
    guided_front_m = -float("inf")
    spin_hold_target: Vec3 | None = None
    rolling_radius_m = max(
        0.01,
        min(
            target_object.dimensions_m.x,
            target_object.dimensions_m.y,
            target_object.dimensions_m.z,
        )
        * 0.5,
    )
    rolling_angle_rad = 0.0
    support_spin_angle_rad = 0.0
    lifecycle: list[dict[str, float | str]] = [
        {"state": "free", "time_s": 0.0},
    ]
    contacts: list[ContactEvent] = []

    catch_contact_center = np.asarray(
        [
            side * 0.20,
            program.object_motion.contact_height_m,
            program.object_motion.contact_depth_m,
        ],
        dtype=float,
    )
    if program.object_action == ObjectAction.CATCH:
        flight_end = catch_contact_center
        flight_duration_s, flight_velocity = _ballistic_flight(
            original_position,
            flight_end,
            program.object_motion.apex_height_m,
        )

    push_approach: np.ndarray | None = None
    if program.object_action in {ObjectAction.PUSH, ObjectAction.ROLL} and program.object_motion is not None:
        push_approach = np.asarray(
            [program.object_motion.direction_x, 0.0, program.object_motion.direction_z], dtype=float
        )
        push_approach /= max(float(np.linalg.norm(push_approach)), 1e-8)
    # A throw or a place takes the block the way a grab does: the hand's own
    # frame is put around it once (the seat) and the four phases that bring
    # the hand to it are solved by orientation along a straight wrist path,
    # not placed by a pivot and then cleared. The seat is solved against the
    # trunk the first of those phases renders with.
    seat_action = program.object_action in {ObjectAction.THROW, ObjectAction.PLACE}
    grasp_block: Box | None = None
    grasp_boxes: list[Box] = []
    grasp_seat: Seat | None = None
    grasp_hint: np.ndarray | None = None
    if seat_action:
        grasp_block = Box.from_scene_object(target_object)
        grasp_boxes = [grasp_block]
        support = scene.support_surface()
        if support is not None:
            grasp_boxes.append(Box.from_scene_object(support))
        grasp_hint = elbow_hang(program.hand.value)
    for primitive in program.primitives:
        guided_action = program.object_action in {
            ObjectAction.PUSH,
            ObjectAction.PULL,
            ObjectAction.ROLL,
            ObjectAction.SPIN,
        }
        if (
            guided_action
            and primitive.kind == PrimitiveKind.RECOVER
            and release_time_s is None
            and attachment_local_offset is not None
        ):
            release_time_s = elapsed
            landed_position = current_object_position.copy()
            landed_rotation = current_object_rotation
            lifecycle.append({"state": "released", "time_s": release_time_s})
        start = current.copy()
        target = _object_interaction_arm_target(program, target_object, primitive.kind)
        shape = primitive.hand_shape or HandShape.OPEN
        seated = seat_action and primitive.kind in _INTERACTION_PATH_PHASES
        if seated and grasp_block is not None and grasp_hint is not None:
            trunk = current.copy()
            trunk["chest"] = Quat(
                y=math.sin(side * primitive.parameters.torso_participation * 0.08),
                w=math.cos(primitive.parameters.torso_participation * 0.08),
            )
            grasp_trunk = {name: BonePose(rotation=rotation) for name, rotation in trunk.items()}
            if grasp_seat is None:
                _, grasp_seat, grasp_hint = solve_seat(
                    grasp_trunk,
                    program.hand.value,
                    grasp_block,
                    grasp_finger_axis(target_object, primitive),
                    grasp_palm_normal(program.hand, target_object, primitive),
                    grasp_hint,
                    primitive.parameters,
                )
            target = Vec3(
                x=float(grasp_seat.wrist[0]), y=float(grasp_seat.wrist[1]), z=float(grasp_seat.wrist[2])
            )
            arm = solve_hand_along(
                grasp_trunk, program.hand.value, grasp_seat.wrist, grasp_seat.rotation, grasp_hint
            )
            fingers = hand_pose(program.hand, shape, primitive.parameters)
            target_pose = trunk.copy()
            target_pose.update(arm)
            target_pose.update(fingers)
        elif primitive.kind == PrimitiveKind.RECOVER and program.object_action in {
            ObjectAction.THROW,
            ObjectAction.PUSH,
            ObjectAction.PULL,
            ObjectAction.ROLL,
            ObjectAction.SPIN,
            ObjectAction.PLACE,
            ObjectAction.DROP,
        }:
            target_pose = base.copy()
        else:
            if target is None:
                return _failure_result(
                    scene,
                    program,
                    FailureCode.INVALID_PROGRAM,
                    f"No arm target for object phase {primitive.kind.value}",
                )
            keyframe_solve = lambda point: arm_pose_from_target(  # noqa: E731
                program.hand,
                shoulder_position(program.hand),
                point,
                primitive.parameters,
                present_hand=False,
            )[0]
            arm = keyframe_solve(target)
            fingers = hand_pose(program.hand, shape, primitive.parameters)
            # The authored target is a pivot at the object's centre; move it
            # until the hand's skin is out of the table and, while the object
            # is still free, out of the object too.
            keyframe_object_position = current_object_position
            keyframe_object_attached = (
                program.object_action == ObjectAction.CATCH
                or (attachment_local_offset is not None and release_time_s is None)
            )
            if program.object_action == ObjectAction.PLACE and primitive.kind == PrimitiveKind.RELEASE:
                # Set down: the block parts from the hand and ends at the
                # landing point, so that is the body the open hand clears.
                place_direction = np.asarray(
                    [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
                    dtype=float,
                )
                place_direction /= max(float(np.linalg.norm(place_direction)), 1e-8)
                keyframe_object_position = original_position + place_direction * program.object_motion.distance_m
                keyframe_object_position[1] = program.object_motion.landing_height_m
                keyframe_object_attached = False
            forbidden = _interaction_forbidden_bodies(
                scene,
                target_object,
                primitive.kind,
                keyframe_object_position,
                current_object_rotation,
                keyframe_object_attached,
                push_approach,
            )
            if (
                program.object_action in {ObjectAction.PUSH, ObjectAction.ROLL}
                and primitive.kind in {PrimitiveKind.MOVE, PrimitiveKind.HOLD}
            ):
                # The pushed block stops where the program says; its trailing
                # face there is a wall the hand's front may reach, not cross.
                push_direction = np.asarray(
                    [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
                    dtype=float,
                )
                push_direction /= max(float(np.linalg.norm(push_direction)), 1e-8)
                forbidden.append(
                    _push_destination_wall(
                        original_position,
                        original_rotation,
                        object_half_extents,
                        push_direction,
                        program.object_motion.distance_m,
                    )
                )
            cleared_keyframe = clear_wrist_target(
                program.hand,
                target,
                keyframe_solve,
                current,
                fingers,
                forbidden,
            )
            target, arm = cleared_keyframe.target, cleared_keyframe.arm
            if program.object_action == ObjectAction.SPIN:
                # A spin turns the block under a hand that stays put: every
                # phase after contact keeps contact's cleared wrist, so the
                # hand does not creep as each phase's hand shape is cleared
                # against the table on its own.
                if primitive.kind == PrimitiveKind.CONTACT:
                    spin_hold_target = target
                elif spin_hold_target is not None:
                    target = spin_hold_target
                    arm = arm_pose_from_target(
                        program.hand,
                        shoulder_position(program.hand),
                        target,
                        primitive.parameters,
                        present_hand=False,
                    )[0]
            if (
                program.object_action in {ObjectAction.PUSH, ObjectAction.ROLL}
                and primitive.kind in {PrimitiveKind.MOVE, PrimitiveKind.HOLD}
            ):
                # The authored target moves the wrist by the push distance,
                # but it is the palm that pushes and the palm falls behind the
                # wrist as the arm pitches. Advance the wrist until the skin at
                # block height reaches the destination's trailing face; the
                # wall in the clearance keeps it from going past.
                wall = next(body for body in forbidden if body.name == "push_destination")
                face_along = float(np.dot(wall.centre - push_direction * 1.0, push_direction))
                block_at_destination = original_position + push_direction * program.object_motion.distance_m
                # Solve the way the move's per-frame path will render it --
                # elbow steered from where the arm is now, on its branch --
                # so the reach measured here is the reach the frames have.
                pushing_hint = bone_world_position(current, f"{program.hand.value}LowerArm")
                pushing_pole = bend_pole(program.hand, current, primitive.parameters)
                pushing_solve = lambda point, _hint=pushing_hint, _pole=pushing_pole: hinted_arm_solve(  # noqa: E731
                    program.hand, point, primitive.parameters, _hint, _pole
                )
                arm = pushing_solve(target)
                for _ in range(8):
                    pose_now = dict(current)
                    pose_now.update(arm)
                    pose_now.update(fingers)
                    skin = hand_mesh(program.hand.value).world(
                        {name: BonePose(rotation=rotation) for name, rotation in pose_now.items()}
                    )
                    front = pushing_front_m(
                        skin,
                        block_at_destination,
                        original_rotation.as_matrix(),
                        object_half_extents,
                        push_direction,
                    )
                    if front is None:
                        break
                    shortfall = face_along - front
                    if shortfall <= hand_clearance_m():
                        break
                    advanced = np.asarray(target.as_list(), dtype=float) + push_direction * shortfall
                    cleared_keyframe = clear_wrist_target(
                        program.hand,
                        Vec3(x=float(advanced[0]), y=float(advanced[1]), z=float(advanced[2])),
                        pushing_solve,
                        current,
                        fingers,
                        forbidden,
                    )
                    target, arm = cleared_keyframe.target, cleared_keyframe.arm
            target_pose = current.copy()
            target_pose.update(arm)
            target_pose.update(fingers)
            target_pose["chest"] = Quat(
                y=math.sin(side * primitive.parameters.torso_participation * 0.08),
                w=math.cos(primitive.parameters.torso_participation * 0.08),
            )

        if primitive.kind == PrimitiveKind.FLIGHT:
            if program.object_action == ObjectAction.THROW:
                if release_position is None:
                    return _failure_result(
                        scene,
                        program,
                        FailureCode.INVALID_PROGRAM,
                        "Throw reached flight before release",
                    )
                direction = np.asarray(
                    [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
                    dtype=float,
                )
                direction /= max(float(np.linalg.norm(direction)), 1e-8)
                flight_end = release_position + direction * program.object_motion.distance_m
                flight_end[1] = program.object_motion.landing_height_m
                flight_duration_s, flight_velocity = _ballistic_flight(
                    release_position,
                    flight_end,
                    program.object_motion.apex_height_m,
                )
            elif program.object_action == ObjectAction.DROP:
                if release_position is None:
                    return _failure_result(
                        scene,
                        program,
                        FailureCode.INVALID_PROGRAM,
                        "Drop reached flight before release",
                    )
                flight_end = release_position.copy()
                flight_end[1] = program.object_motion.landing_height_m
                fall_height = max(0.0, release_position[1] - flight_end[1])
                flight_duration_s = max(0.20, math.sqrt(2.0 * fall_height / 9.81))
                flight_velocity = np.zeros(3, dtype=float)
            flight_start_s = elapsed
            lifecycle.append({"state": "ballistic", "time_s": elapsed})

        requested_duration = primitive.parameters.duration_s
        if primitive.kind == PrimitiveKind.FLIGHT and flight_duration_s is not None:
            requested_duration = flight_duration_s
        frame_count = max(2, int(round(requested_duration * scene.fps)) + 1)
        target_delta = max(
            2.0
            * math.acos(
                float(
                    np.clip(
                        abs(np.dot(start[key].as_list(), target_pose[key].as_list())),
                        0.0,
                        1.0,
                    )
                )
            )
            for key in base
        )
        frame_count = max(frame_count, subdivision_frames(target_delta, NLERP_SUBDIVISION_RAD))
        phase_duration_s = max(requested_duration, (frame_count - 1) / scene.fps)
        phase_object_start = current_object_position.copy()
        phase_ranges.append(
            {
                "kind": primitive.kind.value,
                "start_s": elapsed,
                "end_s": elapsed + phase_duration_s,
            }
        )
        # A free hand travelling to or from the object follows a straight
        # wrist path cleared frame by frame; the joint-space blend would swing
        # it through the table's edge and the object's face on the way.
        place_release = (
            program.object_action == ObjectAction.PLACE
            and primitive.kind == PrimitiveKind.RELEASE
            and attachment_world_offset is not None
        )
        path_phase = not seated and (
            (primitive.kind in _INTERACTION_PATH_PHASES and target is not None)
            or place_release
            or (
                # A push's blend between two clear keyframes sags the hand
                # into the table; the wrist goes straight, cleared each frame.
                push_approach is not None
                and primitive.kind in (PrimitiveKind.MOVE, PrimitiveKind.HOLD)
                and target is not None
            )
            or (
                primitive.kind == PrimitiveKind.RECOVER
                and (attachment_local_offset is None or release_time_s is not None)
            )
        )
        path_end: Vec3 | None = None
        path_start: Vec3 | None = None
        path_elbow: Vec3 | None = None
        path_pole: Vec3 | None = None
        if path_phase:
            path_end = (
                target
                if target is not None and primitive.kind != PrimitiveKind.RECOVER
                else bone_world_position(target_pose, f"{program.hand.value}Hand")
            )
            path_start = bone_world_position(start, f"{program.hand.value}Hand")
            path_elbow = bone_world_position(start, f"{program.hand.value}LowerArm")
            path_pole = bend_pole(program.hand, start, primitive.parameters)
        grasp_path: tuple[np.ndarray, Rotation, np.ndarray] | None = None
        if seated:
            start_bones = {name: BonePose(rotation=rotation) for name, rotation in start.items()}
            start_wrist, start_rotation = _grasp_hand_world(start_bones, program.hand.value)
            grasp_path = (start_wrist, start_rotation, elbow_direction(start_bones, program.hand.value))
        phase_object_start_rotation = current_object_rotation
        # A placed object parts from the hand during release, on its own
        # path to the landing point; from then on it is a body to stay out of.
        # A catch is the exception to "stay out of a free object": the ball
        # arrives into the hand by design, and a hand that dodged it would
        # never catch it.
        object_free_now = lambda: (  # noqa: E731
            program.object_action != ObjectAction.CATCH
            and (
                attachment_local_offset is None
                or release_time_s is not None
                or (program.object_action == ObjectAction.PLACE and primitive.kind == PrimitiveKind.RELEASE)
            )
        )
        for local_index in range(frame_count):
            if frames and local_index == 0:
                continue
            progress = local_index / (frame_count - 1)
            alpha = smoothstep(progress, primitive.parameters.easing)
            bones = {
                key: BonePose(
                    rotation=(
                        # Keep the humeral roll and its forearm pronation
                        # compensation advancing as a synchronized twist pair,
                        # so the hand world orientation between solver
                        # keyframes tracks the compensated (pre-roll) path.
                        # An object carried on that hand follows the grip
                        # lever arm, so an unsynchronized blend turns the
                        # transient hand swing directly into carried-object
                        # position steps.
                        _swing_twist_blend(start[key], target_pose[key], alpha)
                        if key in _ARM_TWIST_BLEND_BONES
                        else _nlerp(start[key], target_pose[key], alpha)
                    )
                )
                for key in base
            }
            now = elapsed + progress * phase_duration_s
            if seated and grasp_seat is not None and grasp_path is not None and grasp_hint is not None:
                # As the grab: the wrist travels a straight path (the reach is
                # one eased arc up and over to the seat), the hand turns from
                # where it started to the seat's orientation, and the arm is
                # solved fresh each frame against the trunk being rendered,
                # elbow pinned to its branch. At alpha=1 this is the
                # keyframe's own solve, so the next phase blends from it.
                start_wrist, start_rotation, start_elbow = grasp_path
                if primitive.kind == PrimitiveKind.REACH:
                    wrist_now = arc_path(
                        start_wrist, grasp_seat.wrist, alpha, target_object.dimensions_m.y
                    )
                    hint_now = rotate_toward(start_elbow, grasp_hint, alpha)
                else:
                    wrist_now = start_wrist + (grasp_seat.wrist - start_wrist) * alpha
                    hint_now = grasp_hint
                rotation_now = rotation_toward(
                    start_rotation, Rotation.from_matrix(grasp_seat.rotation), alpha
                ).as_matrix()
                path_arm = solve_hand_along(
                    bones, program.hand.value, wrist_now, rotation_now, hint_now
                )
                bones.update({name: BonePose(rotation=rotation) for name, rotation in path_arm.items()})
                if local_index == frame_count - 1:
                    target_pose.update(path_arm)
            elif path_phase and path_start is not None and path_end is not None:
                if place_release:
                    # Setting down: the block goes to its landing point on its
                    # own schedule, and the hand keeps the hold it had on it
                    # all the way there, opening as it goes.
                    place_direction = np.asarray(
                        [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
                        dtype=float,
                    )
                    place_direction /= max(float(np.linalg.norm(place_direction)), 1e-8)
                    landing = original_position + place_direction * program.object_motion.distance_m
                    landing[1] = program.object_motion.landing_height_m
                    block_on_path = phase_object_start * (1.0 - alpha) + landing * alpha
                    hold_offset = np.asarray(path_start.as_list(), dtype=float) - phase_object_start
                    path_point = block_on_path + hold_offset
                    path_target = Vec3(x=float(path_point[0]), y=float(path_point[1]), z=float(path_point[2]))
                elif primitive.kind == PrimitiveKind.REACH and push_approach is None:
                    path_target = _reach_over_path(
                        path_start, path_end, progress, target_object.dimensions_m.y
                    )
                else:
                    path_target = Vec3(
                        x=path_start.x + (path_end.x - path_start.x) * alpha,
                        y=path_start.y + (path_end.y - path_start.y) * alpha,
                        z=path_start.z + (path_end.z - path_start.z) * alpha,
                    )
                hint = path_elbow

                def _solve_path_arm(
                    point: Vec3, _hint: Vec3 | None = hint, _pole: Vec3 | None = path_pole
                ) -> dict[str, Quat]:
                    return hinted_arm_solve(program.hand, point, primitive.parameters, _hint, _pole)

                frame_pose = {name: bone.rotation for name, bone in bones.items()}
                frame_bodies = _interaction_forbidden_bodies(
                    scene,
                    target_object,
                    primitive.kind,
                    current_object_position,
                    current_object_rotation,
                    not object_free_now(),
                    push_approach,
                )
                if push_approach is not None and primitive.kind in (PrimitiveKind.MOVE, PrimitiveKind.HOLD):
                    frame_bodies.append(
                        _push_destination_wall(
                            original_position,
                            original_rotation,
                            object_half_extents,
                            push_approach,
                            program.object_motion.distance_m,
                        )
                    )
                cleared = clear_wrist_target(
                    program.hand, path_target, _solve_path_arm, frame_pose, {}, frame_bodies
                )
                bones.update(
                    {name: BonePose(rotation=rotation) for name, rotation in cleared.arm.items()}
                )
                frame_pose.update(cleared.arm)
                path_elbow = bone_world_position(frame_pose, f"{program.hand.value}LowerArm")
                if local_index == frame_count - 1:
                    target_pose.update(cleared.arm)
            provisional = ClipFrame(time_s=now, bones=bones, objects={})
            _, _, wrist, hand_world = _fk_arm_landmarks(provisional, program.hand)

            attach_now = bool(
                program.object_action in {
                    ObjectAction.THROW,
                    ObjectAction.CATCH,
                    ObjectAction.DROP,
                    ObjectAction.PLACE,
                }
                and primitive.kind == PrimitiveKind.CLOSE
                or program.object_action == ObjectAction.PUSH
                and primitive.kind == PrimitiveKind.CONTACT
                or program.object_action == ObjectAction.ROLL
                and primitive.kind == PrimitiveKind.CONTACT
                or program.object_action == ObjectAction.SPIN
                and primitive.kind == PrimitiveKind.CONTACT
                or program.object_action == ObjectAction.PULL
                and primitive.kind == PrimitiveKind.CLOSE
            )
            if (
                attach_now
                and program.object_action != ObjectAction.CATCH
                and local_index < frame_count - 1
            ):
                # A grasp takes hold when the fingers have closed, at the end
                # of the phase, not at its first frame while the hand is still
                # open at the preshape. Attaching early drags the object off
                # the table with an open hand and then curls the fingers into
                # it. A catch is the exception: the ball arrives into the hand.
                attach_now = False
            if attach_now and attachment_local_offset is None:
                attachment_local_offset = hand_world.inv().apply(current_object_position - wrist)
                attachment_local_rotation = hand_world.inv() * current_object_rotation
                attachment_world_offset = current_object_position - wrist
                attachment_started_s = now
                lifecycle.append(
                    {
                        "state": (
                            "guided_contact"
                            if program.object_action == ObjectAction.PUSH
                            else "attached"
                        ),
                        "time_s": now,
                    }
                )
                for digit, x_offset in (("thumb", -0.02), ("index", 0.02)):
                    contacts.append(
                        ContactEvent(
                            time_s=now,
                            hand=program.hand,
                            object_id=target_object.id,
                            digit=digit,
                            position=Vec3(
                                x=float(current_object_position[0] + x_offset),
                                y=float(current_object_position[1]),
                                z=float(current_object_position[2]),
                            ),
                            normal_force_n=4.0,
                        )
                    )
                if program.object_action == ObjectAction.CATCH:
                    catch_time_s = now

            attached = attachment_local_offset is not None and (
                program.object_action == ObjectAction.CATCH
                or release_time_s is None
            )
            if attached:
                if program.object_action == ObjectAction.PLACE:
                    if primitive.kind == PrimitiveKind.RELEASE:
                        place_direction = np.asarray(
                            [
                                program.object_motion.direction_x,
                                0.0,
                                program.object_motion.direction_z,
                            ],
                            dtype=float,
                        )
                        place_direction /= max(
                            float(np.linalg.norm(place_direction)),
                            1e-8,
                        )
                        placement_target = original_position + (
                            place_direction * program.object_motion.distance_m
                        )
                        placement_target[1] = program.object_motion.landing_height_m
                        current_object_position = (
                            phase_object_start * (1.0 - alpha)
                            + placement_target * alpha
                        )
                        # Set down upright: the carried tilt eases out over
                        # the release rather than snapping at its first frame.
                        tilt = phase_object_start_rotation * original_rotation.inv()
                        current_object_rotation = (
                            Rotation.from_rotvec(tilt.as_rotvec() * (1.0 - alpha)) * original_rotation
                        )
                    else:
                        # Carried the way it was grasped: rigid in the hand, so
                        # the fingers that closed on it stay on it however the
                        # wrist turns on the way.
                        current_object_position = wrist + hand_world.apply(attachment_local_offset)
                        current_object_rotation = hand_world * (
                            attachment_local_rotation or Rotation.identity()
                        )
                    if primitive.kind != PrimitiveKind.RELEASE:
                        attachment_offsets.append(
                            hand_world.inv().apply(current_object_position - wrist)
                        )
                elif guided_action:
                    current_object_position = wrist + (
                        attachment_world_offset
                        if attachment_world_offset is not None
                        else np.zeros(3, dtype=float)
                    )
                    current_object_position[1] = original_position[1]
                    if program.object_action in {ObjectAction.PUSH, ObjectAction.ROLL}:
                        # A pushed block is wherever the front of the hand has
                        # pushed it to: its trailing face sits on the hand's
                        # foremost skin vertex along the push direction, and
                        # it never comes back toward the body.
                        push_direction = np.asarray(
                            [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
                            dtype=float,
                        )
                        push_direction /= max(float(np.linalg.norm(push_direction)), 1e-8)
                        skin = hand_mesh(program.hand.value).world(bones)
                        rotation_matrix = current_object_rotation.as_matrix()
                        trailing_extent = float(
                            np.dot(np.abs(push_direction @ rotation_matrix), object_half_extents)
                        )
                        front = pushing_front_m(
                            skin, current_object_position, rotation_matrix, object_half_extents, push_direction
                        )
                        pushed_to = front + trailing_extent if front is not None else -float("inf")
                        along_now = float(np.dot(current_object_position, push_direction))
                        destination_along = float(
                            np.dot(original_position, push_direction)
                        ) + program.object_motion.distance_m
                        pushed_to = min(
                            max(pushed_to, float(np.dot(original_position, push_direction)), guided_front_m),
                            destination_along,
                        )
                        guided_front_m = pushed_to
                        current_object_position += push_direction * (pushed_to - along_now)
                    if program.object_action == ObjectAction.SPIN:
                        support_spin_angle_rad = 2.0 * math.pi * program.object_motion.spin_turns * (
                            alpha if primitive.kind == PrimitiveKind.MOVE else 1.0
                        )
                        current_object_position = original_position.copy()
                        current_object_rotation = Rotation.from_rotvec(
                            np.asarray([0.0, support_spin_angle_rad, 0.0], dtype=float)
                        ) * original_rotation
                        current_vertical_extent = float(
                            np.dot(
                                np.abs(current_object_rotation.as_matrix()[1]),
                                object_half_extents,
                            )
                        )
                        current_object_position[1] = (
                            object_support_plane_y + current_vertical_extent
                        )
                    elif program.object_action == ObjectAction.ROLL:
                        guided_delta_now = current_object_position - original_position
                        direction = np.asarray(
                            [
                                program.object_motion.direction_x,
                                0.0,
                                program.object_motion.direction_z,
                            ],
                            dtype=float,
                        )
                        direction /= max(float(np.linalg.norm(direction)), 1e-8)
                        projected_distance = float(np.dot(guided_delta_now, direction))
                        rolling_angle_rad = projected_distance / rolling_radius_m
                        rolling_axis = np.asarray(
                            [direction[2], 0.0, -direction[0]],
                            dtype=float,
                        )
                        current_object_rotation = Rotation.from_rotvec(
                            rolling_axis * rolling_angle_rad
                        ) * original_rotation
                        current_vertical_extent = float(
                            np.dot(
                                np.abs(current_object_rotation.as_matrix()[1]),
                                object_half_extents,
                            )
                        )
                        current_object_position[1] = (
                            object_support_plane_y + current_vertical_extent
                        )
                    else:
                        current_object_rotation = original_rotation
                    guided_offset = current_object_position - wrist
                    guided_offset[1] = 0.0
                    if program.object_action in {ObjectAction.PUSH, ObjectAction.ROLL}:
                        # Along the push the block is pinned to the hand's
                        # front by construction; only lateral drift is slip.
                        guided_offset -= push_direction * float(np.dot(guided_offset, push_direction))
                    attachment_offsets.append(guided_offset)
                else:
                    current_object_position = wrist + hand_world.apply(attachment_local_offset)
                    current_object_rotation = hand_world * (
                        attachment_local_rotation or Rotation.identity()
                    )
                    attachment_offsets.append(
                        hand_world.inv().apply(current_object_position - wrist)
                    )
            elif primitive.kind == PrimitiveKind.FLIGHT and flight_start_s is not None:
                assert flight_duration_s is not None and flight_velocity is not None
                flight_t = min(flight_duration_s, max(0.0, now - flight_start_s))
                start_position = release_position if release_position is not None else original_position
                current_object_position = (
                    start_position
                    + flight_velocity * flight_t
                    + np.asarray([0.0, -4.905 * flight_t * flight_t, 0.0])
                )
                spin_alpha = flight_t / max(flight_duration_s, 1e-8)
                base_rotation = release_rotation or original_rotation
                spin = Rotation.from_rotvec(
                    np.asarray([1.0, 0.2, 0.15], dtype=float)
                    / np.linalg.norm([1.0, 0.2, 0.15])
                    * (2.0 * math.pi * program.object_motion.spin_turns * spin_alpha)
                )
                current_object_rotation = spin * base_rotation
                if flight_t >= flight_duration_s - 1e-8:
                    first_landing = landed_position is None
                    current_object_position = flight_end.copy()
                    landed_position = current_object_position.copy()
                    landed_rotation = current_object_rotation
                    if first_landing and program.object_action == ObjectAction.DROP:
                        lifecycle.append({"state": "landed", "time_s": now})
            elif landed_position is not None:
                current_object_position = landed_position.copy()
                current_object_rotation = landed_rotation or current_object_rotation

            objects = {item.id: item.transform for item in scene.objects}
            quat = current_object_rotation.as_quat()
            objects[target_object.id] = Transform(
                translation=Vec3(
                    x=float(current_object_position[0]),
                    y=float(current_object_position[1]),
                    z=float(current_object_position[2]),
                ),
                rotation=Quat(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
            )
            if frames:
                previous = frames[-1].objects[target_object.id].translation
                previous_position = np.asarray(previous.as_list(), dtype=float)
                object_steps.append(float(np.linalg.norm(current_object_position - previous_position)))
            frames.append(ClipFrame(time_s=now, bones=bones, objects=objects))

        current = target_pose
        elapsed += phase_duration_s
        if guided_action and primitive.kind == PrimitiveKind.MOVE:
            lifecycle.append({"state": "guided", "time_s": elapsed})
        if (
            program.object_action
            in {ObjectAction.THROW, ObjectAction.DROP, ObjectAction.PLACE}
            and primitive.kind == PrimitiveKind.RELEASE
        ):
            release_time_s = frames[-1].time_s
            release_position = current_object_position.copy()
            release_rotation = current_object_rotation
            lifecycle.append(
                {
                    "state": (
                        "placed"
                        if program.object_action == ObjectAction.PLACE
                        else "released"
                    ),
                    "time_s": release_time_s,
                }
            )

    safety = _safety_metrics(frames)
    prefix = program.hand.value
    angular_velocity, angular_acceleration, angular_jerk = _angular_kinematics(
        frames,
        [f"{prefix}UpperArm", f"{prefix}LowerArm", f"{prefix}Hand"],
    )
    max_step = max(object_steps, default=0.0)
    terminal_speed = 0.0
    if flight_velocity is not None and flight_duration_s is not None:
        terminal_velocity = flight_velocity + np.asarray(
            [0.0, -9.81 * flight_duration_s, 0.0], dtype=float
        )
        terminal_speed = max(
            float(np.linalg.norm(flight_velocity)),
            float(np.linalg.norm(terminal_velocity)),
        )
    step_reference = max(
        0.08,
        terminal_speed / scene.fps + 0.5 * 9.81 / (scene.fps * scene.fps) + 0.01,
    )
    attachment_slip = (
        max(
            float(np.linalg.norm(offset - attachment_offsets[0]))
            for offset in attachment_offsets
        )
        if attachment_offsets
        else float("inf")
    )
    catch_error = (
        float(np.linalg.norm((landed_position if landed_position is not None else current_object_position) - flight_end))
        if program.object_action == ObjectAction.CATCH
        else 0.0
    )
    landing_error = (
        abs(float(current_object_position[1]) - program.object_motion.landing_height_m)
        if program.object_action
        in {ObjectAction.THROW, ObjectAction.DROP, ObjectAction.PLACE}
        else 0.0
    )
    guided_direction = np.asarray(
        [program.object_motion.direction_x, 0.0, program.object_motion.direction_z],
        dtype=float,
    )
    guided_direction /= max(float(np.linalg.norm(guided_direction)), 1e-8)
    guided_delta = current_object_position - original_position
    placement_horizontal_distance = (
        float(np.linalg.norm(guided_delta[[0, 2]]))
        if program.object_action == ObjectAction.PLACE
        else 0.0
    )
    guided_projected_distance = (
        float(np.dot(guided_delta, guided_direction))
        if program.object_action
        in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL}
        else 0.0
    )
    guided_lateral_error = (
        float(
            np.linalg.norm(
                guided_delta
                - guided_direction * guided_projected_distance
                - np.asarray([0.0, guided_delta[1], 0.0], dtype=float)
            )
        )
        if program.object_action
        in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL}
        else 0.0
    )
    guided_support_height_error = (
        abs(
            float(
                current_object_position[1]
                - np.dot(
                    np.abs(current_object_rotation.as_matrix()[1]),
                    object_half_extents,
                )
                - object_support_plane_y
            )
        )
        if program.object_action in {ObjectAction.ROLL, ObjectAction.SPIN}
        else abs(float(current_object_position[1] - original_position[1]))
        if program.object_action
        in {ObjectAction.PUSH, ObjectAction.PULL}
        else 0.0
    )
    expected_roll_turns = (
        program.object_motion.distance_m / (2.0 * math.pi * rolling_radius_m)
        if program.object_action == ObjectAction.ROLL
        else 0.0
    )
    measured_roll_turns = (
        abs(rolling_angle_rad) / (2.0 * math.pi)
        if program.object_action == ObjectAction.ROLL
        else 0.0
    )
    measured_support_spin_turns = (
        abs(support_spin_angle_rad) / (2.0 * math.pi)
        if program.object_action == ObjectAction.SPIN
        else 0.0
    )
    structural_failures: list[str] = []
    if physics is not None and not physics.success:
        structural_failures.append("object was not physically secured before the throw")
    if attachment_started_s is None:
        structural_failures.append("object never entered the attached state")
    if (
        program.object_action
        in {ObjectAction.THROW, ObjectAction.DROP, ObjectAction.PLACE}
        and release_time_s is None
    ):
        structural_failures.append("object never entered the released state")
    if (
        program.object_action == ObjectAction.PLACE
        and placement_horizontal_distance < program.object_motion.distance_m * 0.75
    ):
        structural_failures.append("placed object under-traveled the requested distance")
    if (
        program.object_action
        in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL, ObjectAction.SPIN}
        and release_time_s is None
    ):
        structural_failures.append("guided object never returned to the free state")
    if (
        program.object_action in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL}
        and guided_projected_distance < program.object_motion.distance_m * 0.75
    ):
        structural_failures.append("guided object under-traveled the requested distance")
    if (
        program.object_action in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL}
        and guided_lateral_error > 0.06
    ):
        structural_failures.append("guided object drifted away from the requested direction")
    if (
        program.object_action
        in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL, ObjectAction.SPIN}
        and guided_support_height_error > 0.01
    ):
        structural_failures.append("guided object left its support plane")
    if (
        program.object_action == ObjectAction.ROLL
        and measured_roll_turns < expected_roll_turns * 0.75
    ):
        structural_failures.append("rolling object under-rotated for its travel distance")
    if (
        program.object_action == ObjectAction.SPIN
        and measured_support_spin_turns < abs(program.object_motion.spin_turns) * 0.90
    ):
        structural_failures.append("spinning object did not complete its requested turns")
    if program.object_action == ObjectAction.CATCH and catch_error > 0.015:
        structural_failures.append("hand did not intercept the object flight")
    if attachment_slip > 0.005:
        structural_failures.append("attached object slipped relative to the palm")
    if max_step > step_reference:
        structural_failures.append("object trajectory contains a teleport-sized step")
    if landing_error > 0.01:
        structural_failures.append("object did not finish on its target surface")
    if safety["nan_count"]:
        structural_failures.append("clip contains non-finite transforms")
    if _clip_contract_violations(safety, policy="fixed"):
        structural_failures.append("clip exceeds a joint limit")
    metrics = _base_metrics()
    if physics is not None:
        metrics.update(physics.metrics)
    metrics.update(safety)
    metrics.update(
        {
            "max_angular_velocity_rad_s": angular_velocity,
            "max_angular_acceleration_rad_s2": angular_acceleration,
            "max_angular_jerk_rad_s3": angular_jerk,
            "phase_ranges_s": phase_ranges,
            "object_action": program.object_action.value,
            "object_lifecycle": lifecycle,
            "attachment_started_s": attachment_started_s,
            "release_time_s": release_time_s,
            "catch_time_s": catch_time_s,
            "object_flight_duration_s": flight_duration_s,
            "object_flight_distance_m": program.object_motion.distance_m,
            "object_guided_distance_m": (
                program.object_motion.distance_m
                if program.object_action
                in {ObjectAction.PUSH, ObjectAction.PULL, ObjectAction.ROLL}
                else 0.0
            ),
            "object_guided_projected_distance_m": guided_projected_distance,
            "object_guided_lateral_error_m": guided_lateral_error,
            "object_guided_support_height_error_m": guided_support_height_error,
            "object_expected_roll_turns": expected_roll_turns,
            "object_measured_roll_turns": measured_roll_turns,
            "object_measured_support_spin_turns": measured_support_spin_turns,
            "object_placement_horizontal_distance_m": placement_horizontal_distance,
            "object_apex_height_m": program.object_motion.apex_height_m,
            "object_spin_turns": program.object_motion.spin_turns,
            "object_max_step_m": max_step,
            "object_max_step_reference_m": step_reference,
            "palm_relative_object_slip_m": attachment_slip,
            "catch_intercept_error_m": catch_error,
            "landing_height_error_m": landing_error,
            "weld_used": False,
            "structural_failures": structural_failures,
            "structural_valid": not structural_failures,
        }
    )
    success = not structural_failures
    failure = None
    if not success:
        failure = Failure(
            code=(
                FailureCode.GRASP_UNSTABLE
                if physics is not None and not physics.success
                else FailureCode.JOINT_LIMIT_EXCEEDED
                if _clip_contract_violations(safety, policy="fixed")
                else FailureCode.INVALID_PROGRAM
            ),
            message="; ".join(structural_failures),
            details=metrics,
            recoverable=False,
        )
    observables = _slider_observables(scene, program)
    observables.update(
        {
            "object_flight_distance_m": program.object_motion.distance_m,
            "object_apex_height_m": program.object_motion.apex_height_m,
            "object_spin_turns": program.object_motion.spin_turns,
            "object_contact_height_m": program.object_motion.contact_height_m,
            "object_contact_depth_m": program.object_motion.contact_depth_m,
        }
    )
    return ClipResult(
        success=success,
        fps=scene.fps,
        duration_s=elapsed,
        frames=frames,
        contacts=contacts,
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=failure,
    )


def _compose_quat(parent: Quat, local: Quat) -> Quat:
    value = (
        Rotation.from_quat(parent.as_list())
        * Rotation.from_quat(local.as_list())
    ).as_quat()
    return Quat(
        x=float(value[0]),
        y=float(value[1]),
        z=float(value[2]),
        w=float(value[3]),
    )


def _rotate_horizontal(value: np.ndarray, yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        [
            cosine * float(value[0]) + sine * float(value[2]),
            float(value[1]),
            -sine * float(value[0]) + cosine * float(value[2]),
        ],
        dtype=float,
    )


def _sequence_frame_in_world(
    frame: ClipFrame,
    *,
    world_root_start: np.ndarray,
    local_root_start: np.ndarray,
    base_yaw_rad: float,
) -> ClipFrame:
    """Rebase one independently compiled step into accumulated actor space."""

    bones = {
        name: pose.model_copy(deep=True)
        for name, pose in frame.bones.items()
    }
    hips = bones["hips"]
    local_root = np.asarray(
        hips.position.as_list() if hips.position is not None else local_root_start,
        dtype=float,
    )
    world_root = world_root_start + _rotate_horizontal(
        local_root - local_root_start,
        base_yaw_rad,
    )
    base_rotation = _hips_world_euler_quat(y=base_yaw_rad)
    bones["hips"] = BonePose(
        rotation=_compose_quat(base_rotation, hips.rotation),
        position=Vec3(
            x=float(world_root[0]),
            y=float(world_root[1]),
            z=float(world_root[2]),
        ),
    )

    yaw_rotation = Rotation.from_euler("y", base_yaw_rad)
    objects: dict[str, Transform] = {}
    for object_id, transform in frame.objects.items():
        local_position = np.asarray(transform.translation.as_list(), dtype=float)
        horizontal = _rotate_horizontal(
            np.asarray([local_position[0], 0.0, local_position[2]], dtype=float),
            base_yaw_rad,
        )
        world_object_rotation = (
            yaw_rotation * Rotation.from_quat(transform.rotation.as_list())
        ).as_quat()
        objects[object_id] = Transform(
            translation=Vec3(
                x=float(world_root_start[0] + horizontal[0]),
                y=float(local_position[1]),
                z=float(world_root_start[2] + horizontal[2]),
            ),
            rotation=Quat(
                x=float(world_object_rotation[0]),
                y=float(world_object_rotation[1]),
                z=float(world_object_rotation[2]),
                w=float(world_object_rotation[3]),
            ),
        )
    return ClipFrame(time_s=frame.time_s, bones=bones, objects=objects)


def _sequence_attach_object_to_hand(
    frame: ClipFrame,
    *,
    object_id: str,
    hand: Hand,
    local_offset: np.ndarray,
    local_rotation: Rotation,
) -> ClipFrame:
    """Carry an attached scene object through an independently compiled step."""

    wrist, hand_world = _sequence_world_wrist(frame, hand)
    position = wrist + hand_world.apply(local_offset)
    rotation = (hand_world * local_rotation).as_quat()
    objects = {
        key: value.model_copy(deep=True)
        for key, value in frame.objects.items()
    }
    objects[object_id] = Transform(
        translation=Vec3(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        ),
        rotation=Quat(
            x=float(rotation[0]),
            y=float(rotation[1]),
            z=float(rotation[2]),
            w=float(rotation[3]),
        ),
    )
    return frame.model_copy(update={"objects": objects})


def _sequence_world_wrist(
    frame: ClipFrame,
    hand: Hand,
) -> tuple[np.ndarray, Rotation]:
    """Apply authored hips translation/yaw to the calibrated arm landmarks."""

    _, _, local_wrist, local_hand_world = arm_landmarks(frame, hand)
    hips = frame.bones.get("hips", BonePose())
    root = np.asarray(
        (hips.position or Vec3()).as_list(),
        dtype=float,
    )
    root_rotation = Rotation.from_quat(hips.rotation.as_list())
    return root + root_rotation.apply(local_wrist), root_rotation * local_hand_world


def _sequence_adapt_attached_release(
    frames: list[ClipFrame],
    *,
    program: MotionProgram,
    object_id: str,
    hand: Hand,
    local_offset: np.ndarray,
    local_rotation: Rotation,
    base_yaw_rad: float,
    phase_ranges: list[dict[str, Any]],
) -> tuple[list[ClipFrame], list[dict[str, Any]], float, float, float]:
    """Release an already-held object without replaying a table pickup.

    Standalone throw/drop skills include reach, contact, and close phases so
    they are complete in isolation.  Inside an ordered sequence those phases
    would reset an object that is already attached to the palm.  This adapter
    begins at the held-object phase, preserves the incoming palm weld through
    release, and computes flight from the actual accumulated world position.
    """

    if program.object_action not in {ObjectAction.DROP, ObjectAction.THROW}:
        raise ValueError("attached release adapter requires a drop or throw")
    if program.object_motion is None:
        raise ValueError("attached release adapter requires object motion")
    start_kind = (
        PrimitiveKind.WINDUP.value
        if program.object_action == ObjectAction.THROW
        else PrimitiveKind.LIFT.value
    )
    start_range = next(
        (item for item in phase_ranges if item.get("kind") == start_kind),
        None,
    )
    release_range = next(
        (item for item in phase_ranges if item.get("kind") == PrimitiveKind.RELEASE.value),
        None,
    )
    flight_range = next(
        (item for item in phase_ranges if item.get("kind") == PrimitiveKind.FLIGHT.value),
        None,
    )
    if start_range is None or release_range is None or flight_range is None:
        raise ValueError("attached release lifecycle is missing a required phase")
    requested_start_s = float(start_range.get("start_s", 0.0))
    retained = [
        frame.model_copy(deep=True)
        for frame in frames
        if frame.time_s >= requested_start_s - 1e-8
    ]
    if not retained:
        raise ValueError("attached release lifecycle produced no retained frames")
    start_s = retained[0].time_s
    release_s = float(release_range.get("end_s", start_s))
    flight_start_s = float(flight_range.get("start_s", release_s))
    flight_end_s = float(flight_range.get("end_s", flight_start_s))
    flight_span_s = max(1e-8, flight_end_s - flight_start_s)
    adapted: list[ClipFrame] = []
    release_position: np.ndarray | None = None
    release_rotation: Rotation | None = None
    landed_position: np.ndarray | None = None
    landed_rotation: Rotation | None = None
    direction = _rotate_horizontal(
        np.asarray(
            [
                program.object_motion.direction_x,
                0.0,
                program.object_motion.direction_z,
            ],
            dtype=float,
        ),
        base_yaw_rad,
    )
    horizontal_norm = float(np.linalg.norm(direction[[0, 2]]))
    if horizontal_norm > 1e-8:
        direction /= horizontal_norm

    for frame in retained:
        old_time_s = frame.time_s
        rebased = frame.model_copy(update={"time_s": old_time_s - start_s})
        if old_time_s <= release_s + 1e-8:
            rebased = _sequence_attach_object_to_hand(
                rebased,
                object_id=object_id,
                hand=hand,
                local_offset=local_offset,
                local_rotation=local_rotation,
            )
            transform = rebased.objects[object_id]
            release_position = np.asarray(transform.translation.as_list(), dtype=float)
            release_rotation = Rotation.from_quat(transform.rotation.as_list())
        else:
            if release_position is None or release_rotation is None:
                raise ValueError("attached release did not establish a release transform")
            progress = min(
                1.0,
                max(0.0, (old_time_s - flight_start_s) / flight_span_s),
            )
            landing = release_position.copy()
            landing[1] = program.object_motion.landing_height_m
            if program.object_action == ObjectAction.THROW:
                landing += direction * program.object_motion.distance_m
                landing[1] = program.object_motion.landing_height_m
                physical_duration_s, velocity = _ballistic_flight(
                    release_position,
                    landing,
                    program.object_motion.apex_height_m,
                )
                flight_t = progress * physical_duration_s
                position = (
                    release_position
                    + velocity * flight_t
                    + np.asarray([0.0, -4.905 * flight_t * flight_t, 0.0])
                )
            else:
                # Normalized gravity curve reaches the support plane exactly
                # at the authored end of the flight phase.
                position = release_position.copy()
                position[1] = release_position[1] + (
                    program.object_motion.landing_height_m - release_position[1]
                ) * progress * progress
            spin_axis = np.asarray([1.0, 0.2, 0.15], dtype=float)
            spin_axis /= float(np.linalg.norm(spin_axis))
            rotation = Rotation.from_rotvec(
                spin_axis
                * (2.0 * math.pi * program.object_motion.spin_turns * progress)
            ) * release_rotation
            if progress >= 1.0 - 1e-8:
                position = landing
                landed_position = landing.copy()
                landed_rotation = rotation
            elif landed_position is not None:
                position = landed_position.copy()
                rotation = landed_rotation or rotation
            objects = {
                key: value.model_copy(deep=True)
                for key, value in rebased.objects.items()
            }
            quaternion = rotation.as_quat()
            objects[object_id] = Transform(
                translation=Vec3(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                ),
                rotation=Quat(
                    x=float(quaternion[0]),
                    y=float(quaternion[1]),
                    z=float(quaternion[2]),
                    w=float(quaternion[3]),
                ),
            )
            rebased = rebased.model_copy(update={"objects": objects})
        adapted.append(rebased)

    adapted_ranges: list[dict[str, Any]] = []
    for phase in phase_ranges:
        phase_end_s = float(phase.get("end_s", 0.0))
        if phase_end_s < start_s - 1e-8:
            continue
        adapted_ranges.append(
            {
                **phase,
                "start_s": max(0.0, float(phase.get("start_s", 0.0)) - start_s),
                "end_s": max(0.0, phase_end_s - start_s),
                "stateful_attachment": True,
            }
        )
    return (
        adapted,
        adapted_ranges,
        adapted[-1].time_s,
        max(0.0, release_s - start_s),
        max(0.0, flight_end_s - start_s),
    )


def _blend_sequence_boundary(
    previous: ClipFrame,
    target: ClipFrame,
    *,
    start_time_s: float,
    duration_s: float,
    fps: int,
) -> list[ClipFrame]:
    """Insert a minimum-jerk-like neutral bridge between recovered skills."""

    count = max(2, int(round(duration_s * fps)))
    frames: list[ClipFrame] = []
    bone_names = set(previous.bones) | set(target.bones)
    for index in range(1, count + 1):
        linear = index / count
        alpha = smoothstep(linear, 0.82)
        bones: dict[str, BonePose] = {}
        for name in bone_names:
            first = previous.bones.get(name, BonePose())
            second = target.bones.get(name, BonePose())
            position = None
            if first.position is not None or second.position is not None:
                first_position = np.asarray(
                    (first.position or Vec3()).as_list(), dtype=float
                )
                second_position = np.asarray(
                    (second.position or Vec3()).as_list(), dtype=float
                )
                blended = first_position * (1.0 - alpha) + second_position * alpha
                position = Vec3(
                    x=float(blended[0]),
                    y=float(blended[1]),
                    z=float(blended[2]),
                )
            bones[name] = BonePose(
                rotation=_nlerp(first.rotation, second.rotation, alpha),
                position=position,
            )
        objects = {
            object_id: transform.model_copy(deep=True)
            for object_id, transform in (
                target.objects if linear >= 1.0 else previous.objects
            ).items()
        }
        frames.append(
            ClipFrame(
                time_s=start_time_s + duration_s * linear,
                bones=bones,
                objects=objects,
            )
        )
    return frames


#: Largest per-frame step a carried object may take through a sequence, in
#: metres; a throw's release is allowed a little more. Shared by the gate and
#: by the boundary-blend retime that keeps carried objects under it.
CARRIED_OBJECT_STEP_REFERENCE_M = 0.08
_CARRIED_OBJECT_THROW_STEP_M = 0.10


def _sequence_hand_relative(frame: ClipFrame, object_id: str, hand: Hand) -> np.ndarray:
    """An object's position in one hand's frame, for one sequence frame."""

    wrist, hand_world = _sequence_world_wrist(frame, hand)
    position = np.asarray(frame.objects[object_id].translation.as_list(), dtype=float)
    return hand_world.inv().apply(position - wrist)


def _compile_sequence(scene: SceneManifest, program: MotionProgram) -> ClipResult:
    """Validate child skills independently, then compose them continuously."""

    fps = scene.fps
    child_results: list[tuple[MotionProgram, ClipResult]] = []
    for index, step in enumerate(program.steps, start=1):
        result = compile_motion(
            CompileRequest(scene=scene, program=step, persist=False)
        )
        if not result.success:
            failure = result.failure
            return _failure_result(
                scene,
                program,
                failure.code if failure is not None else FailureCode.UNSUPPORTED_MOTION,
                f"Sequence step {index} ({step.intent.value}) failed: "
                + (failure.message if failure is not None else "unknown compiler failure"),
                {
                    "failed_step_index": index,
                    "failed_step_intent": step.intent.value,
                    "child_metrics": result.metrics,
                },
            )
        child_results.append((step, result))

    frames: list[ClipFrame] = []
    contacts: list[ContactEvent] = []
    phase_ranges: list[dict[str, Any]] = []
    step_ranges: list[dict[str, Any]] = []
    elapsed = 0.0
    accumulated_yaw = 0.0
    current_world_root: np.ndarray | None = None
    boundary_duration_s = 0.30
    carried_object_id: str | None = None
    carried_hand: Hand | None = None
    carried_local_offset: np.ndarray | None = None
    carried_local_rotation: Rotation | None = None
    carried_object_ids: set[str] = set()
    carried_release_times_s: dict[str, float] = {}
    stateful_object_transitions: list[dict[str, Any]] = []

    for step_index, (step, result) in enumerate(child_results, start=1):
        if not result.frames:
            continue
        first_hips = result.frames[0].bones["hips"]
        local_root_start = np.asarray(
            first_hips.position.as_list()
            if first_hips.position is not None
            else [0.0, 0.0, 0.0],
            dtype=float,
        )
        world_root_start = (
            local_root_start.copy()
            if current_world_root is None
            else current_world_root.copy()
        )
        transformed = [
            _sequence_frame_in_world(
                frame,
                world_root_start=world_root_start,
                local_root_start=local_root_start,
                base_yaw_rad=accumulated_yaw,
            )
            for frame in result.frames
        ]
        has_incoming_attachment = bool(
            carried_object_id is not None
            and carried_hand is not None
            and carried_local_offset is not None
            and carried_local_rotation is not None
        )
        stateful_release = bool(
            has_incoming_attachment
            and step.intent == Intent.OBJECT_INTERACTION
            and step.object_action in {ObjectAction.DROP, ObjectAction.THROW}
        )
        if (
            has_incoming_attachment
            and step.intent == Intent.OBJECT_INTERACTION
            and not stateful_release
        ):
            return _failure_result(
                scene,
                program,
                FailureCode.INVALID_PROGRAM,
                "Sequence object transition requires an explicit release or handoff lifecycle",
                {
                    "failed_step_index": step_index,
                    "object_action": (
                        step.object_action.value if step.object_action is not None else None
                    ),
                    "carried_object_id": carried_object_id,
                },
            )
        if stateful_release and step.hand != carried_hand:
            return _failure_result(
                scene,
                program,
                FailureCode.INVALID_PROGRAM,
                "A carried object must be handed off before release by the other hand",
                {
                    "failed_step_index": step_index,
                    "carried_hand": carried_hand.value if carried_hand is not None else None,
                    "release_hand": step.hand.value,
                },
            )
        effective_phase_ranges = [
            dict(item)
            for item in result.metrics.get("phase_ranges_s", [])
            if isinstance(item, dict)
        ]
        effective_contacts = list(result.contacts)
        effective_duration_s = result.duration_s
        release_local_time_s: float | None = None
        landing_local_time_s: float | None = None
        if stateful_release:
            assert carried_object_id is not None
            assert carried_hand is not None
            assert carried_local_offset is not None
            assert carried_local_rotation is not None
            (
                transformed,
                effective_phase_ranges,
                effective_duration_s,
                release_local_time_s,
                landing_local_time_s,
            ) = (
                _sequence_adapt_attached_release(
                    transformed,
                    program=step,
                    object_id=carried_object_id,
                    hand=carried_hand,
                    local_offset=carried_local_offset,
                    local_rotation=carried_local_rotation,
                    base_yaw_rad=accumulated_yaw,
                    phase_ranges=effective_phase_ranges,
                )
            )
            # Contact was established by an earlier child.  Replaying the
            # standalone skill's contact events would falsely report a second
            # grasp at the old table location.
            effective_contacts = []
        elif has_incoming_attachment:
            assert carried_object_id is not None
            assert carried_hand is not None
            assert carried_local_offset is not None
            assert carried_local_rotation is not None
            transformed = [
                _sequence_attach_object_to_hand(
                    frame,
                    object_id=carried_object_id,
                    hand=carried_hand,
                    local_offset=carried_local_offset,
                    local_rotation=carried_local_rotation,
                )
                for frame in transformed
            ]
        transition = 0.0
        if frames:
            transition = 0.50 if has_incoming_attachment else boundary_duration_s
            # A step may end on a solved arm the next step's authored idle
            # does not share; give the bridge the frames that rotation needs,
            # by the same subdivision rule every phase applies to itself.
            boundary_delta = max(
                (
                    2.0
                    * math.acos(
                        float(
                            np.clip(
                                abs(
                                    np.dot(
                                        np.asarray(frames[-1].bones[key].rotation.as_list()),
                                        np.asarray(transformed[0].bones[key].rotation.as_list()),
                                    )
                                ),
                                0.0,
                                1.0,
                            )
                        )
                    )
                    for key in transformed[0].bones
                    if key in frames[-1].bones
                ),
                default=0.0,
            )
            transition = max(
                transition,
                subdivision_frames(boundary_delta, NLERP_SUBDIVISION_RAD) / fps,
            )
            for _ in range(4):
                bridge = _blend_sequence_boundary(
                    frames[-1],
                    transformed[0],
                    start_time_s=elapsed,
                    duration_s=transition,
                    fps=fps,
                )
                if not has_incoming_attachment:
                    break
                assert carried_object_id is not None
                assert carried_hand is not None
                assert carried_local_offset is not None
                assert carried_local_rotation is not None
                bridge = [
                    _sequence_attach_object_to_hand(
                        frame,
                        object_id=carried_object_id,
                        hand=carried_hand,
                        local_offset=carried_local_offset,
                        local_rotation=carried_local_rotation,
                    )
                    for frame in bridge
                ]
                # A carried object swings on the hand's lever through the
                # boundary blend. Retime the blend, as every phase is retimed
                # for its rotation rate, so the object never steps further
                # per frame than the carried-object gate below allows.
                worst_step = max(
                    (
                        float(
                            np.linalg.norm(
                                _sequence_hand_relative(later, carried_object_id, carried_hand)
                                - _sequence_hand_relative(earlier, carried_object_id, carried_hand)
                            )
                        )
                        for earlier, later in zip([frames[-1], *bridge], bridge)
                        if carried_object_id in earlier.objects and carried_object_id in later.objects
                    ),
                    default=0.0,
                )
                if worst_step <= CARRIED_OBJECT_STEP_REFERENCE_M:
                    break
                transition *= worst_step / CARRIED_OBJECT_STEP_REFERENCE_M
            frames.extend(bridge)
        step_start_s = elapsed + transition
        for frame_index, frame in enumerate(transformed):
            if frames and frame_index == 0:
                continue
            frames.append(
                frame.model_copy(update={"time_s": step_start_s + frame.time_s})
            )
        for contact in effective_contacts:
            local_position = np.asarray(contact.position.as_list(), dtype=float)
            rotated = _rotate_horizontal(
                np.asarray([local_position[0], 0.0, local_position[2]], dtype=float),
                accumulated_yaw,
            )
            contacts.append(
                contact.model_copy(
                    update={
                        "time_s": step_start_s + contact.time_s,
                        "position": Vec3(
                            x=float(world_root_start[0] + rotated[0]),
                            y=float(local_position[1]),
                            z=float(world_root_start[2] + rotated[2]),
                        ),
                    }
                )
            )
        for phase in effective_phase_ranges:
            phase_ranges.append(
                {
                    **phase,
                    "start_s": step_start_s + float(phase.get("start_s", 0.0)),
                    "end_s": step_start_s + float(phase.get("end_s", 0.0)),
                    "step_index": step_index,
                    "step_intent": step.intent.value,
                }
            )
        step_end_s = step_start_s + effective_duration_s
        step_ranges.append(
            {
                "step_index": step_index,
                "intent": step.intent.value,
                "source_text": step.source_text,
                "start_s": step_start_s,
                "end_s": step_end_s,
                "child_structural_valid": bool(
                    result.metrics.get("structural_valid", result.success)
                ),
                "horizontal_pose_requested": bool(
                    result.metrics.get("horizontal_pose_requested", False)
                ),
                "horizontal_pose_variant": result.metrics.get(
                    "horizontal_pose_variant"
                ),
                "horizontal_pose_contact_point_count": result.metrics.get(
                    "horizontal_pose_contact_point_count"
                ),
            }
        )
        final_hips = transformed[-1].bones["hips"]
        current_world_root = np.asarray(
            final_hips.position.as_list()
            if final_hips.position is not None
            else world_root_start,
            dtype=float,
        )
        accumulated_yaw += math.radians(
            float(result.metrics.get("final_root_yaw_deg", 0.0) or 0.0)
        )
        attachment_source = step.intent == Intent.GRAB or (
            step.intent == Intent.OBJECT_INTERACTION
            and step.object_action == ObjectAction.CATCH
        )
        if attachment_source:
            object_id = next(
                (item.object_id for item in step.primitives if item.object_id),
                None,
            )
            if object_id is not None and object_id in transformed[-1].objects:
                attachment_hand = step.hand
                wrist, hand_world = _sequence_world_wrist(
                    transformed[-1], attachment_hand
                )
                object_transform = transformed[-1].objects[object_id]
                object_position = np.asarray(
                    object_transform.translation.as_list(), dtype=float
                )
                carried_object_id = object_id
                carried_hand = attachment_hand
                carried_local_offset = hand_world.inv().apply(
                    object_position - wrist
                )
                carried_local_rotation = hand_world.inv() * Rotation.from_quat(
                    object_transform.rotation.as_list()
                )
                carried_object_ids.add(object_id)
        elif step.intent == Intent.OBJECT_INTERACTION:
            if stateful_release and carried_object_id is not None:
                absolute_release_s = step_start_s + float(release_local_time_s or 0.0)
                carried_release_times_s[carried_object_id] = absolute_release_s
                stateful_object_transitions.append(
                    {
                        "object_id": carried_object_id,
                        "hand": carried_hand.value if carried_hand is not None else step.hand.value,
                        "action": (
                            step.object_action.value if step.object_action is not None else None
                        ),
                        "step_index": step_index,
                        "release_time_s": absolute_release_s,
                        "landing_time_s": step_start_s
                        + float(landing_local_time_s or effective_duration_s),
                        "landing_height_m": (
                            step.object_motion.landing_height_m
                            if step.object_motion is not None
                            else None
                        ),
                    }
                )
            carried_object_id = None
            carried_hand = None
            carried_local_offset = None
            carried_local_rotation = None
        elapsed = step_end_s

    metrics = _base_metrics()
    metrics.update(_safety_metrics(frames, policy="free"))
    metrics["phase_ranges_s"] = phase_ranges
    metrics["sequence_step_ranges_s"] = step_ranges
    metrics["sequence_step_count"] = len(step_ranges)
    metrics["sequence_intents"] = [step.intent.value for step, _ in child_results]
    horizontal_children = [
        result.metrics
        for _, result in child_results
        if result.metrics.get("horizontal_pose_requested")
    ]
    metrics["horizontal_pose_requested"] = bool(horizontal_children)
    if horizontal_children:
        metrics["horizontal_pose_variants"] = [
            value.get("horizontal_pose_variant") for value in horizontal_children
        ]
        metrics["horizontal_pose_contact_point_count"] = min(
            int(value.get("horizontal_pose_contact_point_count", 0) or 0)
            for value in horizontal_children
        )
        metrics["horizontal_pose_minimum_clearance_m"] = min(
            float(value.get("horizontal_pose_minimum_clearance_m", 0.0) or 0.0)
            for value in horizontal_children
        )
        metrics["horizontal_body_axis_vertical_fraction"] = max(
            float(value.get("horizontal_body_axis_vertical_fraction", 1.0) or 1.0)
            for value in horizontal_children
        )
        push_up_children = [
            value
            for value in horizontal_children
            if value.get("horizontal_pose_variant") == "plank"
        ]
        if push_up_children:
            metrics["requested_push_up_cycles"] = sum(
                float(value.get("requested_push_up_cycles", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["measured_push_up_cycles"] = sum(
                int(value.get("measured_push_up_cycles", 0) or 0)
                for value in push_up_children
            )
            metrics["push_up_vertical_excursion_m"] = min(
                float(value.get("push_up_vertical_excursion_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["plank_hold_vertical_range_m"] = max(
                float(value.get("plank_hold_vertical_range_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["push_up_palm_height_range_m"] = max(
                float(value.get("push_up_palm_height_range_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["push_up_toe_height_range_m"] = max(
                float(value.get("push_up_toe_height_range_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["push_up_toe_position_range_m"] = max(
                float(value.get("push_up_toe_position_range_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["push_up_toe_support_clearance_m"] = max(
                float(value.get("push_up_toe_support_clearance_m", 0.0) or 0.0)
                for value in push_up_children
            )
            metrics["push_up_min_knee_extension_deg"] = min(
                float(value.get("push_up_min_knee_extension_deg", 0.0) or 0.0)
                for value in push_up_children
            )
    metrics["root_motion_enabled"] = any(
        bool(result.metrics.get("root_motion_enabled"))
        for _, result in child_results
    )
    metrics["root_path_length_m"] = 0.0
    metrics["root_displacement_m"] = 0.0
    if frames:
        root_positions = np.asarray(
            [
                (frame.bones["hips"].position or Vec3()).as_list()
                for frame in frames
            ],
            dtype=float,
        )
        if len(root_positions) > 1:
            metrics["root_path_length_m"] = float(
                np.sum(np.linalg.norm(np.diff(root_positions[:, [0, 2]], axis=0), axis=1))
            )
        metrics["root_displacement_m"] = float(
            np.linalg.norm(root_positions[-1, [0, 2]] - root_positions[0, [0, 2]])
        )
    metrics["final_root_yaw_deg"] = math.degrees(accumulated_yaw)
    # A carried object teleports when it moves relative to the hand that
    # carries it, not when that hand swings it fast: a throw's windup takes a
    # block on the fingers past three metres a second and is still one rigid
    # motion. So the step is measured in the hand's own frame, and against
    # whichever hand it is stiller in, which covers a handoff.
    carried_object_max_step = 0.0
    for object_id in carried_object_ids:
        attachment_end_s = carried_release_times_s.get(object_id, float("inf"))
        relative: list[dict[Hand, np.ndarray]] = []
        for frame in frames:
            if object_id not in frame.objects or frame.time_s > attachment_end_s + 1e-8:
                continue
            position = np.asarray(frame.objects[object_id].translation.as_list(), dtype=float)
            per_hand: dict[Hand, np.ndarray] = {}
            for hand in (Hand.LEFT, Hand.RIGHT):
                wrist, hand_world = _sequence_world_wrist(frame, hand)
                per_hand[hand] = hand_world.inv().apply(position - wrist)
            relative.append(per_hand)
        for first, second in zip(relative, relative[1:]):
            carried_object_max_step = max(
                carried_object_max_step,
                min(
                    float(np.linalg.norm(second[hand] - first[hand]))
                    for hand in (Hand.LEFT, Hand.RIGHT)
                ),
            )
    metrics["carried_object_ids"] = sorted(carried_object_ids)
    metrics["carried_object_max_step_m"] = carried_object_max_step
    carried_object_step_reference_m = (
        _CARRIED_OBJECT_THROW_STEP_M
        if any(item.get("action") == ObjectAction.THROW.value for item in stateful_object_transitions)
        else CARRIED_OBJECT_STEP_REFERENCE_M
    )
    metrics["carried_object_max_step_reference_m"] = carried_object_step_reference_m
    metrics["carried_object_release_times_s"] = carried_release_times_s
    stateful_landing_errors: list[float] = []
    stateful_release_displacements: list[float] = []
    for transition in stateful_object_transitions:
        object_id = str(transition["object_id"])
        object_frames = [frame for frame in frames if object_id in frame.objects]
        if not object_frames:
            continue
        release_frame = min(
            object_frames,
            key=lambda frame: abs(frame.time_s - float(transition["release_time_s"])),
        )
        landing_frame = min(
            object_frames,
            key=lambda frame: abs(frame.time_s - float(transition["landing_time_s"])),
        )
        initial_position = np.asarray(
            object_frames[0].objects[object_id].translation.as_list(), dtype=float
        )
        release_position = np.asarray(
            release_frame.objects[object_id].translation.as_list(), dtype=float
        )
        landing_position = np.asarray(
            landing_frame.objects[object_id].translation.as_list(), dtype=float
        )
        release_displacement = float(
            np.linalg.norm(release_position[[0, 2]] - initial_position[[0, 2]])
        )
        landing_height_error = abs(
            float(landing_position[1]) - float(transition["landing_height_m"])
        )
        transition["release_position_m"] = release_position.tolist()
        transition["landing_position_m"] = landing_position.tolist()
        transition["release_displacement_m"] = release_displacement
        transition["landing_height_error_m"] = landing_height_error
        stateful_release_displacements.append(release_displacement)
        stateful_landing_errors.append(landing_height_error)
    metrics["stateful_object_transitions"] = stateful_object_transitions
    metrics["stateful_object_transition_count"] = len(stateful_object_transitions)
    metrics["stateful_object_release_displacement_m"] = max(
        stateful_release_displacements, default=0.0
    )
    metrics["stateful_object_landing_height_error_m"] = max(
        stateful_landing_errors, default=0.0
    )
    body_bones = [
        "hips", "chest", "leftUpperLeg", "leftLowerLeg", "rightUpperLeg",
        "rightLowerLeg", "leftUpperArm", "leftLowerArm", "rightUpperArm",
        "rightLowerArm",
    ]
    angular_velocity, angular_acceleration, angular_jerk = _angular_kinematics(
        frames, body_bones
    )
    metrics["max_angular_velocity_rad_s"] = angular_velocity
    metrics["max_angular_acceleration_rad_s2"] = angular_acceleration
    metrics["max_angular_jerk_rad_s3"] = angular_jerk
    structural_failures: list[str] = []
    if metrics["nan_count"]:
        structural_failures.append("sequence contains non-finite transforms")
    if metrics["discontinuities"]:
        structural_failures.append(
            f"sequence contains {metrics['discontinuities']} rotational discontinuities"
        )
    if carried_object_max_step > carried_object_step_reference_m:
        structural_failures.append("carried object contains a teleport-sized step")
    if metrics["stateful_object_landing_height_error_m"] > 0.01:
        structural_failures.append("stateful object release missed its support height")
    metrics["structural_failures"] = structural_failures
    metrics["structural_valid"] = not structural_failures
    observables = {
        "duration_s": float(elapsed),
        "sequence_step_count": float(len(step_ranges)),
        "root_displacement_m": float(metrics["root_displacement_m"]),
        "root_turn_degrees": float(metrics["final_root_yaw_deg"]),
        "handedness": -1.0 if program.hand == Hand.LEFT else 1.0,
    }
    return ClipResult(
        success=bool(frames and not structural_failures),
        fps=fps,
        duration_s=elapsed,
        frames=frames,
        contacts=contacts,
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=(
            None
            if frames and not structural_failures
            else Failure(
                code=FailureCode.JOINT_LIMIT_EXCEEDED,
                message="; ".join(structural_failures)
                or "Ordered sequence produced no frames",
                details=metrics,
                recoverable=True,
            )
        ),
    )


def compile_motion(request: CompileRequest) -> ClipResult:
    scene, program = apply_overrides(request)
    if program.intent == Intent.UNSUPPORTED:
        return _failure_result(
            scene,
            program,
            FailureCode.UNSUPPORTED_MOTION,
            program.unsupported_reason or "unsupported program",
        )
    if program.intent == Intent.SEQUENCE:
        return _compile_sequence(scene, program)
    if program.intent == Intent.COMPOSITE:
        return _compile_composite(scene, program)
    if program.intent == Intent.FULL_BODY:
        return _compile_full_body(scene, program)
    if program.intent == Intent.OBJECT_INTERACTION:
        return _compile_object_interaction(scene, program)

    target_object = None
    if program.intent == Intent.GRAB:
        object_id = next((item.object_id for item in program.primitives if item.object_id), None)
        target_object = scene.object_by_id(object_id or "")
        if target_object is None:
            return _failure_result(scene, program, FailureCode.UNKNOWN_OBJECT, f"Unknown object: {object_id}")
        if target_object is scene.support_surface():
            return _failure_result(
                scene,
                program,
                FailureCode.UNKNOWN_OBJECT,
                f"Support surfaces are not grasp targets: {object_id}",
            )
        shoulder = shoulder_position(program.hand)
        forward_offset = target_object.transform.translation.z - shoulder.z
        if forward_offset <= 0.0:
            return _failure_result(
                scene,
                program,
                FailureCode.UNREACHABLE_TARGET,
                "Target is behind the humanoid's +Z forward plane",
                {"forward_offset_m": forward_offset, "forward_axis": "+Z"},
            )
        distance = float(
            np.linalg.norm(
                np.asarray(target_object.transform.translation.as_list()) - np.asarray(shoulder.as_list())
            )
        )
        effective_reach_m = min(scene.reachable_radius_m, arm_reach_m(program.hand) - 1e-4)
        if distance > effective_reach_m:
            return _failure_result(
                scene,
                program,
                FailureCode.UNREACHABLE_TARGET,
                f"Target is {distance:.3f} m from the shoulder; effective limit is {effective_reach_m:.3f} m",
                {
                    "distance_m": distance,
                    "declared_scene_limit_m": scene.reachable_radius_m,
                    "effective_limit_m": effective_reach_m,
                    "analytic_arm_reach_m": arm_reach_m(program.hand),
                },
            )

    physics = _physics_for(program, scene) if program.intent == Intent.GRAB else None
    fps = scene.fps
    # Both authored motion families begin from a relaxed humanoid stance. The
    # identity pose is the rig's calibration T-pose and should never leak into
    # a rendered pickup through the inactive arm.
    base = _gesture_idle_pose() if program.intent in {Intent.GESTURE, Intent.GRAB, Intent.STRIKE} else _identity_pose()
    current = base.copy()
    frames: list[ClipFrame] = []
    elapsed = 0.0
    final_shape = HandShape.OPEN
    side = 1.0 if program.hand == Hand.LEFT else -1.0
    # A grab is placed by orientation: the hand's own frame is put around the
    # block once (the seat), the fingers are closed onto its skin at contact
    # and close, and from the end of close the block rides the rendered hand
    # through the rig's FK. Nothing below re-solves a wrist against a point.
    grasp_block: Box | None = None
    grasp_boxes: list[Box] = []
    grasp_seat: Seat | None = None
    grasp_curls: dict[str, float] | None = None
    grasp_hold: tuple[np.ndarray, Rotation] | None = None
    grasp_hint: np.ndarray | None = None
    if program.intent == Intent.GRAB and target_object is not None:
        grasp_block = Box.from_scene_object(target_object)
        grasp_boxes = [grasp_block]
        support = scene.support_surface()
        if support is not None:
            grasp_boxes.append(Box.from_scene_object(support))
        grasp_hint = elbow_hang(program.hand.value)
    # Root-space leg posture, strikes only (2026-08-29 ruling). Authored as
    # per-phase hips offsets -- crouch_depth_m / weight_shift_m in metres --
    # and the legs follow by exact IK on ankles planted at the idle stance's
    # own ankle positions, the same solver the whole-body path uses. Gated on
    # any nonzero value so a program authored before this capability compiles
    # byte-identically: every stored corpus program carries the 0.0 defaults
    # and takes none of this path.
    strike_root_enabled = program.intent == Intent.STRIKE and any(
        item.parameters.crouch_depth_m != 0.0 or item.parameters.weight_shift_m != 0.0
        for item in program.primitives
    )
    current_root = np.zeros(3)
    target_root = np.zeros(3)
    planted_ankles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    kinematics = rig_kinematics()
    if strike_root_enabled:
        stance = {name: BonePose(rotation=value) for name, value in base.items()}
        stance_world = kinematics.world_matrices(stance)
        for leg_side in ("left", "right"):
            matrix = stance_world[kinematics.node_by_canonical[f"{leg_side}Foot"]]
            planted_ankles[leg_side] = (matrix[:3, 3].copy(), matrix[:3, :3].copy())
    last_target = (
        strike_target(
            program.hand,
            program.strike_type,
            program.primitives[0].kind,
            program.primitives[0].parameters,
        )
        if program.intent == Intent.STRIKE and program.strike_type is not None
        else gesture_target(program.hand, _program_params(program))
    )
    observable_target = last_target
    observable_params = _program_params(program)
    presentation_ranges: list[tuple[float, float]] = []
    phase_ranges: list[dict[str, float | str]] = []
    shake_primitive = next(
        (item for item in program.primitives if item.kind == PrimitiveKind.SHAKE),
        None,
    )
    forearm_twist_reserve_rad = 0.0
    if shake_primitive is not None:
        forearm_twist_reserve_rad = (
            forearm_shake_amplitude_rad(
                shake_primitive.parameters.duration_s,
                shake_primitive.parameters.wrist_shake_cycles,
            )
            * shake_primitive.parameters.wrist_shake_amplitude
        )

    for primitive in program.primitives:
        start = current.copy()
        start_root = current_root
        if strike_root_enabled:
            target_root = np.asarray(
                [
                    primitive.parameters.weight_shift_m,
                    -primitive.parameters.crouch_depth_m,
                    0.0,
                ],
                dtype=float,
            )
        shape = primitive.hand_shape or final_shape
        grasp_lifted = primitive.kind in (PrimitiveKind.LIFT, PrimitiveKind.HOLD, PrimitiveKind.RECOVER)
        grasp_trunk: dict[str, BonePose] | None = None
        if program.intent == Intent.GRAB and target_object is not None and grasp_block is not None:
            # The keyframe is solved against the trunk it will be rendered
            # with, so the seat is where the rendered hand ends up.
            trunk = current.copy()
            trunk["chest"] = Quat(
                y=math.sin(primitive.parameters.torso_participation * 0.08),
                w=math.cos(primitive.parameters.torso_participation * 0.08),
            )
            grasp_trunk = {name: BonePose(rotation=rotation) for name, rotation in trunk.items()}
            if grasp_seat is None:
                _, grasp_seat, grasp_hint = solve_seat(
                    grasp_trunk,
                    program.hand.value,
                    grasp_block,
                    grasp_finger_axis(target_object, primitive),
                    grasp_palm_normal(program.hand, target_object, primitive),
                    grasp_hint,
                    primitive.parameters,
                )
            wrist_key = grasp_seat.wrist.copy()
            if grasp_lifted:
                wrist_key[1] += primitive.parameters.lift_height_m
            target = Vec3(x=float(wrist_key[0]), y=float(wrist_key[1]), z=float(wrist_key[2]))
        elif program.intent == Intent.STRIKE and program.strike_type is not None:
            target = strike_target(
                program.hand,
                program.strike_type,
                primitive.kind,
                primitive.parameters,
            )
        else:
            target = gesture_target(program.hand, primitive.parameters)
        if program.intent == Intent.GESTURE and primitive.kind == PrimitiveKind.HOLD:
            observable_target = target
            observable_params = primitive.parameters
        if program.intent in {Intent.GESTURE, Intent.STRIKE} and primitive.kind == PrimitiveKind.RECOVER:
            # Recover to a true relaxed pose.  The former implementation
            # solved another presentation target, so clips never visibly
            # completed the gesture and often ended in a strained pose.
            target_pose = base.copy()
            shape = HandShape.OPEN
        else:
            # A strike yaws the whole trunk, and the arm chain rides it: the
            # authored target describes where the fist should land in the body
            # frame, so the arm is solved against the target counter-rotated
            # about the trunk axis and the trunk rotation carries it back.
            keyframe_trunk_yaw = (
                strike_trunk_yaw(program.hand, primitive.kind, primitive.parameters)
                if program.intent == Intent.STRIKE
                else 0.0
            )
            if grasp_seat is not None and grasp_trunk is not None:
                arm = solve_hand_along(
                    grasp_trunk,
                    program.hand.value,
                    np.asarray(target.as_list(), dtype=float),
                    grasp_seat.rotation,
                    grasp_hint,
                )
                if not grasp_lifted:
                    # Each digit goes to the curl nearest this phase's shape
                    # at which its skin touches neither block nor table: an
                    # open finger lifts off the table, a closing one stops at
                    # the block. The curls found at close are what the hand
                    # lifts with.
                    body = dict(trunk)
                    body.update(arm)
                    curls = close_to_contact(
                        program.hand, shape, primitive.parameters, body, grasp_boxes
                    )
                    if primitive.kind == PrimitiveKind.CLOSE:
                        grasp_curls = curls
                    fingers = hand_pose(program.hand, shape, primitive.parameters, curl_overrides=curls)
                elif grasp_lifted and grasp_curls is not None:
                    fingers = hand_pose(
                        program.hand, shape, primitive.parameters, curl_overrides=grasp_curls
                    )
                else:
                    fingers = hand_pose(program.hand, shape, primitive.parameters)
            else:
                arm, _ = arm_pose_from_target(
                    program.hand,
                    shoulder_position(program.hand),
                    counter_rotate_about_trunk(target, keyframe_trunk_yaw),
                    primitive.parameters,
                    present_hand=program.intent == Intent.GESTURE,
                    forearm_twist_reserve_rad=(
                        forearm_twist_reserve_rad
                        if program.intent == Intent.GESTURE
                        else 0.0
                    ),
                )
                fingers = hand_pose(program.hand, shape, primitive.parameters)
            target_pose = current.copy()
            target_pose.update(arm)
            target_pose.update(fingers)
            if program.intent == Intent.STRIKE:
                target_pose.update(_inactive_guard_pose(program.hand))
                target_pose.update(trunk_yaw_poses(keyframe_trunk_yaw))
            else:
                target_pose["chest"] = Quat(
                    y=math.sin(primitive.parameters.torso_participation * 0.08),
                    w=math.cos(primitive.parameters.torso_participation * 0.08),
                )
        frame_count = max(2, int(round(primitive.parameters.duration_s * fps)))
        # Retime only when a short model-authored phase would otherwise step
        # more than 0.30 rad in a single 30 Hz frame. This preserves the
        # semantic pose while enforcing a continuous animation trajectory.
        target_delta = max(
            2.0
            * math.acos(
                float(
                    np.clip(
                        abs(np.dot(start[key].as_list(), target_pose[key].as_list())),
                        0.0,
                        1.0,
                    )
                )
            )
            for key in base
        )
        # Nlerp is not constant-angular-speed near 180 degrees, so use a
        # conservative 0.14-rad subdivision target.  This is stricter than the
        # old discontinuity-only rule because it also keeps acceleration under
        # the human-reference envelope when a model requests a very short
        # phase.
        frame_count = max(frame_count, subdivision_frames(target_delta, NLERP_SUBDIVISION_RAD))
        phase_duration_s = max(primitive.parameters.duration_s, (frame_count - 1) / fps)
        phase_ranges.append(
            {
                "kind": primitive.kind.value,
                "start_s": elapsed,
                "end_s": elapsed + phase_duration_s,
            }
        )
        if program.intent == Intent.GESTURE and primitive.kind in (
            PrimitiveKind.PRESENT,
            PrimitiveKind.HOLD,
            PrimitiveKind.SHAKE,
        ):
            presentation_ranges.append((elapsed, elapsed + phase_duration_s))
        if program.intent == Intent.STRIKE and primitive.kind in (
            PrimitiveKind.GUARD,
            PrimitiveKind.LOAD,
            PrimitiveKind.STRIKE,
            PrimitiveKind.FOLLOW_THROUGH,
        ):
            presentation_ranges.append((elapsed, elapsed + phase_duration_s))
        grasp_path: tuple[np.ndarray, Rotation, np.ndarray] | None = None
        if grasp_seat is not None:
            start_bones = {name: BonePose(rotation=rotation) for name, rotation in start.items()}
            start_wrist, start_rotation = hand_world(start_bones, program.hand.value)
            grasp_path = (start_wrist, start_rotation, elbow_direction(start_bones, program.hand.value))
        for local_index in range(frame_count):
            if frames and local_index == 0:
                continue
            alpha = smoothstep(local_index / (frame_count - 1), primitive.parameters.easing)
            pose = {key: BonePose(rotation=_nlerp(start[key], target_pose[key], alpha)) for key in base}
            if (
                program.intent == Intent.STRIKE
                and program.strike_type is not None
                and primitive.kind == PrimitiveKind.STRIKE
            ):
                path_target = strike_path_target(
                    program.hand,
                    program.strike_type,
                    last_target,
                    target,
                    alpha,
                    primitive.parameters,
                )
                # The interpolated trunk yaw at this frame, read from the pose
                # actually being rendered: the trunk quats are pure local-Y
                # rotations and nlerp keeps them pure, so the yaw sum is exact.
                frame_trunk_yaw = sum(
                    2.0 * math.atan2(pose[bone].rotation.y, pose[bone].rotation.w)
                    for bone, _ in TRUNK_YAW_DISTRIBUTION
                )
                path_arm, _ = arm_pose_from_target(
                    program.hand,
                    shoulder_position(program.hand),
                    counter_rotate_about_trunk(path_target, frame_trunk_yaw),
                    primitive.parameters,
                    present_hand=False,
                )
                pose.update(
                    {name: BonePose(rotation=rotation) for name, rotation in path_arm.items()}
                )
            if grasp_seat is not None and grasp_path is not None and grasp_hint is not None:
                # The wrist travels a straight path (the reach comes in over
                # the block) and the hand turns from where it started to the
                # seat's orientation; the arm is solved fresh on each frame
                # against the trunk being rendered, elbow pinned to its branch
                # and eased from where it hung to where a grab hangs it. At
                # alpha=1 this is the keyframe's own solve.
                start_wrist, start_rotation, start_elbow = grasp_path
                path_end = np.asarray(target.as_list(), dtype=float)
                if primitive.kind == PrimitiveKind.REACH:
                    # One eased arc up and over to the seat, on the phase's
                    # own clock: the hand comes at the block's side from above
                    # and behind, palm side leading, so it crosses neither the
                    # block nor the table's edge, and the arm never has to
                    # start faster than the phase's easing asks.
                    wrist_now = arc_path(
                        start_wrist, path_end, alpha, target_object.dimensions_m.y
                    )
                    hint_now = rotate_toward(start_elbow, grasp_hint, alpha)
                else:
                    wrist_now = start_wrist + (path_end - start_wrist) * alpha
                    hint_now = grasp_hint
                rotation_now = rotation_toward(
                    start_rotation, Rotation.from_matrix(grasp_seat.rotation), alpha
                ).as_matrix()
                path_arm = solve_hand_along(
                    pose, program.hand.value, wrist_now, rotation_now, hint_now
                )
                pose.update({name: BonePose(rotation=rotation) for name, rotation in path_arm.items()})
                if local_index == frame_count - 1:
                    # The next phase interpolates from the pose that was
                    # rendered, which is this solve.
                    target_pose.update(path_arm)
                if primitive.kind == PrimitiveKind.CLOSE and local_index == frame_count - 1:
                    # The fingers have closed: from here the block is carried
                    # by the rendered hand, through the rig's own FK.
                    wrist_fk, hand_fk = hand_world(pose, program.hand.value)
                    grasp_hold = attach_offset(hand_fk, wrist_fk, grasp_block)
            if (
                program.intent == Intent.GESTURE
                and primitive.kind == PrimitiveKind.PRESENT
                and (
                    abs(primitive.parameters.path_arc) > 1e-8
                    or abs(primitive.parameters.wrist_flourish) > 1e-8
                )
            ):
                # Apply a bounded joint-space arc around the normal trajectory.
                # The sin² envelope has zero value and velocity at both ends,
                # so the path returns exactly to the requested hold pose.  The
                # offsets scale continuously with their authoring parameters;
                # unlike blending an independent IK solve, an infinitesimal
                # requested arc cannot create a finite kinematic jump.
                curve_weight = math.sin(math.pi * alpha) ** 2
                arc = primitive.parameters.path_arc
                flourish = primitive.parameters.wrist_flourish
                arc_amplitude = presentation_arc_amplitude_rad(
                    primitive.parameters.duration_s
                )
                flourish_amplitude = wrist_flourish_amplitude_rad(
                    primitive.parameters.duration_s
                )
                upper_arm = f"{program.hand.value}UpperArm"
                lower_arm = f"{program.hand.value}LowerArm"
                pose[upper_arm] = BonePose(
                    rotation=_local_rotation_offset(
                        pose[upper_arm].rotation,
                        [0.0, 0.0, side * arc * arc_amplitude * curve_weight],
                    )
                )
                pose[lower_arm] = BonePose(
                    rotation=_local_rotation_offset(
                        pose[lower_arm].rotation,
                        [
                            0.0,
                            flourish * flourish_amplitude * curve_weight,
                            -side * arc * arc_amplitude * 0.55 * curve_weight,
                        ],
                    )
                )
            if program.intent == Intent.GESTURE and primitive.kind == PrimitiveKind.SHAKE:
                progress = local_index / (frame_count - 1)
                attack = smoothstep(min(progress / 0.15, 1.0), 1.0)
                release = smoothstep(min((1.0 - progress) / 0.15, 1.0), 1.0)
                envelope = attack * release
                cycles = primitive.parameters.wrist_shake_cycles
                amplitude = (
                    forearm_shake_amplitude_rad(phase_duration_s, cycles)
                    * primitive.parameters.wrist_shake_amplitude
                )
                shake_angle = amplitude * envelope * math.sin(2.0 * math.pi * cycles * progress)
                # A hang-ten shake is forearm pronation/supination about the
                # lower arm's longitudinal local +Y axis.  The hand joint must
                # remain stable; using it here reads as flexion/deviation.
                lower_arm = f"{program.hand.value}LowerArm"
                pose[lower_arm] = BonePose(
                    rotation=_local_rotation_offset(
                        pose[lower_arm].rotation,
                        [0.0, shake_angle, 0.0],
                    )
                )
            if strike_root_enabled:
                # The hips ride the same eased alpha as every rotation channel,
                # so the root arrives exactly at each phase keyframe and the
                # cross-phase lerp endpoints are shared. The legs are then
                # solved fresh per frame against the planted stance ankles --
                # bent knees under a dropped pelvis, feet never leaving the
                # floor -- with the foot's world rotation held so the sole
                # stays flat.
                root = start_root * (1.0 - alpha) + target_root * alpha
                pose["hips"] = BonePose(
                    rotation=pose["hips"].rotation,
                    position=Vec3(x=float(root[0]), y=float(root[1]), z=float(root[2])),
                )
                for leg_side, (ankle_target, foot_rotation) in planted_ankles.items():
                    solved = kinematics.solve_leg(
                        pose, leg_side, ankle_target, foot_world_rotation=foot_rotation
                    )
                    for name, rotation in solved.items():
                        pose[name] = BonePose(rotation=rotation)
            now = elapsed + local_index / fps
            objects: dict[str, Transform] = {}
            for item in scene.objects:
                # The grasped block rides the rendered hand once it is held;
                # every other body -- the table it rested on included -- and
                # the block before that stay where the scene put them.
                if target_object is not None and item.id == target_object.id and grasp_hold is not None:
                    wrist_fk, hand_fk = hand_world(pose, program.hand.value)
                    objects[item.id] = carried_transform(hand_fk, wrist_fk, *grasp_hold)
                else:
                    objects[item.id] = item.transform
            frames.append(ClipFrame(time_s=now, bones=pose, objects=objects))
        current = target_pose
        if strike_root_enabled:
            current_root = target_root
        last_target = target
        final_shape = shape
        elapsed += phase_duration_s

    metrics = _base_metrics()
    metrics["phase_ranges_s"] = phase_ranges
    if physics is not None:
        metrics.update(physics.metrics)
    # The measurement pass lives in ``analysis.hand`` from 02c onwards. What
    # stays here is the part that cannot: ``physics.metrics`` above comes from a
    # MuJoCo grasp simulation, and the GRAB branch below reads its verdict.
    # ``analysis`` must not import ``rigby_poc.physics`` -- a check has to run
    # where MuJoCo is absent -- so those keys are deferred by design.
    metrics.update(
        _hand_metrics(
            _AnalysisContext.from_frames(frames, program, scene, metrics)
        )
    )
    assertions = metrics["finger_assertions"]
    actual_curls = metrics["normalized_finger_curls"]
    # Gesture and grab stay "fixed"; strike is "bounded" per the 2026-08-29
    # ruling -- a real whole-clip gate against the strike envelope rather than
    # the fixed-root epsilon or no gate at all. Derived from the program so
    # this path and the post-hoc analyzer cannot disagree.
    root_policy = _root_drift_policy(program)
    if program.intent == Intent.GRAB:
        structural_failures: list[str] = []
        if physics is None or not physics.success:
            structural_failures.append("physical grasp gates did not pass")
        if metrics["nan_count"]:
            structural_failures.append("clip contains non-finite transforms")
        if _clip_contract_violations(metrics, policy=root_policy):
            structural_failures.append("clip exceeds a joint limit")
        if not all(assertions.values()):
            structural_failures.append("hand-shape assertion failed")
        metrics["structural_failures"] = structural_failures
        metrics["structural_valid"] = not structural_failures
    success = bool(
        metrics["nan_count"] == 0
        and _clip_contract_violations(metrics, policy=root_policy) == 0
        and all(assertions.values())
        and (physics is None or physics.success)
        and metrics.get("structural_valid", True)
    )
    failure = None
    if not success:
        if physics is not None and not physics.success:
            code = FailureCode.GRASP_UNSTABLE
        elif metrics.get("self_collision_frames", 0):
            code = FailureCode.COLLISION_UNRESOLVED
        elif _clip_contract_violations(metrics, policy=root_policy) or metrics.get("structural_failures"):
            code = FailureCode.JOINT_LIMIT_EXCEEDED
        else:
            code = FailureCode.INVALID_PROGRAM
        structural_message = "; ".join(metrics.get("structural_failures", []))
        failure = Failure(
            code=code,
            message=(
                "Physical pickup did not satisfy the contact gates"
                if physics
                else structural_message or "Gesture assertions failed"
            ),
            details=metrics,
            recoverable=False,
        )
    observables = _slider_observables(scene, program)
    observables.update(
        {
            "wrist_target_lateral_m": observable_target.x,
            "wrist_target_height_m": observable_target.y,
            "wrist_target_depth_m": observable_target.z,
            "wrist_lateral_m": observable_target.x,
            "wrist_height_m": observable_target.y,
            "wrist_depth_m": observable_target.z,
            "wrist_pitch_rad": observable_params.wrist_pitch * 0.35,
            "wrist_yaw_rad": observable_params.wrist_yaw * 0.28,
            # User-facing wrist roll means pronation/supination of the hand,
            # which is anatomically produced by the forearm.  Preserve that
            # semantic observable while exposing the small residual hand-joint
            # twist separately for structural inspection.
            "wrist_roll_rad": observable_params.wrist_roll * 0.40,
            "forearm_roll_offset_rad": observable_params.wrist_roll * 0.40,
            "wrist_joint_twist_rad": observable_params.wrist_roll * MAX_WRIST_TWIST_RAD,
            "elbow_swivel_rad": observable_params.elbow_swivel * 0.6,
            "torso_rotation_rad": observable_params.torso_participation * 0.16,
            **{f"{digit}_curl_normalized": value for digit, value in actual_curls.items()},
            **{f"{digit}_curl": value for digit, value in actual_curls.items()},
        }
    )
    if physics is not None:
        observables["lift_height_m"] = float(metrics["lift_height_m"])
        observables["hold_duration_s"] = float(metrics["hold_duration_s"])
    return ClipResult(
        success=success,
        fps=fps,
        duration_s=elapsed,
        frames=frames,
        contacts=physics.contacts if physics else [],
        metrics=metrics,
        slider_observables=observables,
        parametric_observables=observables,
        provenance=_provenance(program),
        failure=failure,
    )
