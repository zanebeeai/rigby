from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class Intent(StrEnum):
    GESTURE = "gesture"
    GRAB = "grab"
    STRIKE = "strike"
    COMPOSITE = "composite"
    FULL_BODY = "full_body"
    OBJECT_INTERACTION = "object_interaction"
    SEQUENCE = "sequence"
    UNSUPPORTED = "unsupported"


class PrimitiveKind(StrEnum):
    GUARD = "guard"
    LOAD = "load"
    STRIKE = "strike"
    FOLLOW_THROUGH = "follow_through"
    PRESENT = "present"
    SHAKE = "shake"
    REACH = "reach"
    PRESHAPE = "preshape"
    CONTACT = "contact"
    CLOSE = "close"
    LIFT = "lift"
    WINDUP = "windup"
    RELEASE = "release"
    FLIGHT = "flight"
    RECEIVE = "receive"
    ABSORB = "absorb"
    HOLD = "hold"
    MOVE = "move"
    CYCLE = "cycle"
    BODY = "body"
    RECOVER = "recover"


class HandShape(StrEnum):
    OPEN = "open"
    FIST = "fist"
    POINT = "point"
    PINCH = "pinch"
    HANG_TEN = "hang_ten"
    THUMBS_UP = "thumbs_up"
    PEACE = "peace"


class StrikeType(StrEnum):
    HOOK = "hook"
    JAB = "jab"
    CROSS = "cross"
    UPPERCUT = "uppercut"


class ObjectAction(StrEnum):
    """Lifecycle that connects an animated hand with a free scene object."""

    THROW = "throw"
    CATCH = "catch"
    PUSH = "push"
    PULL = "pull"
    ROLL = "roll"
    SPIN = "spin"
    PLACE = "place"
    DROP = "drop"
    HANDOFF = "handoff"


class ObjectInteractionStyle(StrEnum):
    OVERHAND = "overhand"
    UNDERHAND = "underhand"
    SIDEARM = "sidearm"
    TOSS = "toss"


class TrajectoryKind(StrEnum):
    LINEAR = "linear"
    ARC = "arc"
    CIRCLE = "circle"
    OSCILLATE = "oscillate"
    HOLD = "hold"


class TrajectoryPlane(StrEnum):
    FRONTAL = "frontal"
    SAGITTAL = "sagittal"
    HORIZONTAL = "horizontal"


class BodyAction(StrEnum):
    """Procedural whole-body skills that may be composed sequentially."""

    HOLD = "hold"
    STEP = "step"
    WALK = "walk"
    RUN = "run"
    TURN = "turn"
    CROUCH = "crouch"
    JUMP = "jump"
    KICK = "kick"
    DANCE = "dance"
    CLIMB = "climb"
    ROTATE = "rotate"
    POSE = "pose"


class BodyRotationAxis(StrEnum):
    """Avatar-local axis for a continuous whole-body rotation."""

    PITCH = "pitch"
    ROLL = "roll"
    YAW = "yaw"


class BodyRotationMode(StrEnum):
    """Support strategy used while the root completes a rotation."""

    FLOOR = "floor"
    CARTWHEEL = "cartwheel"
    AIRBORNE = "airborne"


class BodyObstacleMode(StrEnum):
    """How a locomotion primitive relates to one scene obstacle."""

    NONE = "none"
    OVER = "over"
    AROUND = "around"


class BodyClimbDirection(StrEnum):
    """Direction of travel along a vertical support affordance."""

    UP = "up"
    DOWN = "down"


class BodySupportMode(StrEnum):
    """Authored support relationship for a grounded whole-body pose."""

    FEET = "feet"
    BROAD_FLOOR = "broad_floor"
    QUADRUPED = "quadruped"
    PLANK = "plank"


class GestureStyle(StrEnum):
    RELAXED = "relaxed"
    NEUTRAL = "neutral"
    ENERGETIC = "energetic"
    PRECISE = "precise"
    PLAYFUL = "playful"


class GestureTiming(StrEnum):
    QUICK = "quick"
    BALANCED = "balanced"
    SLOW = "slow"


class GestureHeight(StrEnum):
    LOW = "low"
    CHEST = "chest"
    HIGH = "high"


class GestureDepth(StrEnum):
    CLOSE = "close"
    NATURAL = "natural"
    EXTENDED = "extended"


class GestureLateral(StrEnum):
    INWARD = "inward"
    CENTER = "center"
    OUTWARD = "outward"


class WristPitch(StrEnum):
    DOWN = "down"
    NEUTRAL = "neutral"
    UP = "up"


class WristYaw(StrEnum):
    INWARD = "inward"
    NEUTRAL = "neutral"
    OUTWARD = "outward"


class WristRoll(StrEnum):
    COUNTERCLOCKWISE = "counterclockwise"
    NEUTRAL = "neutral"
    CLOCKWISE = "clockwise"


class GestureMotionProfile(Contract):
    """Bounded semantic choices authored by the planner for one gesture."""

    style: GestureStyle = GestureStyle.NEUTRAL
    timing: GestureTiming = GestureTiming.BALANCED
    height: GestureHeight = GestureHeight.CHEST
    depth: GestureDepth = GestureDepth.NATURAL
    lateral: GestureLateral = GestureLateral.CENTER
    wrist_pitch: WristPitch = WristPitch.NEUTRAL
    wrist_yaw: WristYaw = WristYaw.NEUTRAL
    wrist_roll: WristRoll = WristRoll.NEUTRAL


class FailureCode(StrEnum):
    UNSUPPORTED_MOTION = "unsupported_motion"
    UNKNOWN_OBJECT = "unknown_object"
    CONTRADICTORY_PROMPT = "contradictory_prompt"
    UNREACHABLE_TARGET = "unreachable_target"
    GRASP_UNSTABLE = "grasp_unstable"
    COLLISION_UNRESOLVED = "collision_unresolved"
    JOINT_LIMIT_EXCEEDED = "joint_limit_exceeded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_PROGRAM = "invalid_program"
    EXPORT_FAILED = "export_failed"


class Vec3(Contract):
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class Quat(Contract):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]


class Transform(Contract):
    translation: Vec3
    rotation: Quat = Field(default_factory=Quat)


class AffordanceRole(StrEnum):
    """Physical relationship a socket advertises to the motion planner."""

    GRASP = "grasp"
    CLIMB_CONTACT = "climb_contact"
    SUPPORT = "support"


class AffordanceSocket(Contract):
    id: str = Field(min_length=1, max_length=64)
    transform: Transform
    approach_normal: Vec3
    grasp_span_m: Annotated[float, Field(gt=0.0, le=0.12)]
    role: AffordanceRole = AffordanceRole.GRASP
    supports_body_weight: bool = False


class SceneObject(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    kind: str = Field(default="block", pattern=r"^[a-z][a-z0-9_-]*$")
    transform: Transform
    dimensions_m: Vec3
    mass_kg: Annotated[float, Field(ge=0.01, le=250.0)] = 0.25
    friction: Annotated[float, Field(ge=0.1, le=3.0)] = 1.2
    sockets: list[AffordanceSocket] = Field(min_length=1)

    @model_validator(mode="after")
    def dimensions_supported(self) -> "SceneObject":
        dims = self.dimensions_m.as_list()
        if min(dims) < 0.02 or max(dims) > 4.0:
            raise ValueError("scene-object dimensions must be within 0.02..4.0 m")
        if self.kind == "block":
            if min(dims) < 0.04 or max(dims) > 0.14:
                raise ValueError("block dimensions must be within 0.04..0.14 m")
            if not 0.1 <= self.mass_kg <= 0.5:
                raise ValueError("block mass must be within 0.1..0.5 kg")
        if self.kind == "ladder":
            if not (
                0.25 <= self.dimensions_m.x <= 1.5
                and 0.8 <= self.dimensions_m.y <= 3.5
                and 0.02 <= self.dimensions_m.z <= 0.35
            ):
                raise ValueError("ladder dimensions are outside the supported body-scale range")
            climb_contacts = [
                socket
                for socket in self.sockets
                if socket.role == AffordanceRole.CLIMB_CONTACT
                and socket.supports_body_weight
            ]
            if len(climb_contacts) < 2:
                raise ValueError("a ladder requires at least two weight-bearing climb contacts")
        return self


class RigReference(Contract):
    id: str = "mesh2motion-human-vrm1"
    asset_uri: str = "assets/models/human-male.glb"
    profile_uri: str = "config/rig_profiles/mesh2motion-human-vrm1.json"
    fixed_root: bool = True


class SceneManifest(Contract):
    schema_version: str = "1.0"
    rig: RigReference = Field(default_factory=RigReference)
    objects: list[SceneObject] = Field(min_length=1)
    fps: Annotated[int, Field(ge=24, le=60)] = 30
    reachable_radius_m: Annotated[float, Field(gt=0.3, le=1.0)] = 0.72

    @model_validator(mode="after")
    def unique_objects(self) -> "SceneManifest":
        ids = [item.id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        return self

    def object_by_id(self, object_id: str) -> SceneObject | None:
        return next((item for item in self.objects if item.id == object_id), None)


class PrimitiveParameters(Contract):
    duration_s: Annotated[float, Field(ge=0.05, le=4.0)] = 0.35
    easing: Annotated[float, Field(ge=0.0, le=1.0)] = 0.65
    arm_height: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    arm_depth: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    lateral_offset: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_pitch: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_yaw: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_roll: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    elbow_swivel: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    torso_participation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    path_arc: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_flourish: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_shake_amplitude: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    wrist_shake_cycles: Annotated[float, Field(ge=0.0, le=6.0)] = 0.0
    finger_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    thumb_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    index_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    middle_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    ring_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    little_curl: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    finger_splay: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    thumb_opposition: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    grip_force: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    lift_height_m: Annotated[float, Field(ge=0.0, le=0.80)] = 0.10
    hold_duration_s: Annotated[float, Field(ge=0.0, le=3.0)] = 1.0
    trajectory_amplitude_m: Annotated[float, Field(ge=0.0, le=0.20)] = 0.0
    trajectory_cycles: Annotated[float, Field(ge=0.0, le=8.0)] = 0.0
    axial_rotation_amplitude: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class EffectorTarget(Contract):
    """Body-relative target for one hand inside a concurrent motion phase.

    Coordinates are normalized to the humanoid interaction workspace:
    target_x is avatar-right (-1) to avatar-left (+1), target_y is low (-1) to high
    (+1), and target_z is close (-1) to extended (+1).
    """

    hand: Hand
    target_x: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    target_y: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    target_z: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    reach_overhead: bool = False
    elbow_swivel: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_pitch: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_yaw: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    wrist_roll: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    phase_offset_cycles: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    hand_shape: HandShape = HandShape.OPEN


class BodyPoseTarget(Contract):
    """Continuous controls for an arbitrary grounded whole-body pose.

    Angles are authored in degrees because that is the most reliable unit for
    a language model and for a human inspecting a saved semantic plan.  The
    compiler converts them to the rig's local joint convention.
    """

    root_drop_m: Annotated[float, Field(ge=0.0, le=0.85)] = 0.0
    root_shift_x_m: Annotated[float, Field(ge=-0.45, le=0.45)] = 0.0
    root_shift_z_m: Annotated[float, Field(ge=-0.45, le=0.45)] = 0.0
    pelvis_pitch_deg: Annotated[float, Field(ge=-105.0, le=105.0)] = 0.0
    pelvis_yaw_deg: Annotated[float, Field(ge=-90.0, le=90.0)] = 0.0
    pelvis_roll_deg: Annotated[float, Field(ge=-60.0, le=60.0)] = 0.0
    torso_pitch_deg: Annotated[float, Field(ge=-100.0, le=100.0)] = 0.0
    torso_yaw_deg: Annotated[float, Field(ge=-100.0, le=100.0)] = 0.0
    torso_roll_deg: Annotated[float, Field(ge=-75.0, le=75.0)] = 0.0
    head_pitch_deg: Annotated[float, Field(ge=-55.0, le=55.0)] = 0.0
    head_yaw_deg: Annotated[float, Field(ge=-80.0, le=80.0)] = 0.0
    head_roll_deg: Annotated[float, Field(ge=-45.0, le=45.0)] = 0.0
    left_shoulder_elevation_deg: Annotated[float, Field(ge=-20.0, le=25.0)] = 0.0
    right_shoulder_elevation_deg: Annotated[float, Field(ge=-20.0, le=25.0)] = 0.0
    left_hip_pitch_deg: Annotated[float, Field(ge=-120.0, le=120.0)] = 0.0
    right_hip_pitch_deg: Annotated[float, Field(ge=-120.0, le=120.0)] = 0.0
    left_hip_roll_deg: Annotated[float, Field(ge=-55.0, le=55.0)] = 0.0
    right_hip_roll_deg: Annotated[float, Field(ge=-55.0, le=55.0)] = 0.0
    left_knee_flexion_deg: Annotated[float, Field(ge=0.0, le=145.0)] = 0.0
    right_knee_flexion_deg: Annotated[float, Field(ge=0.0, le=145.0)] = 0.0
    left_ankle_pitch_deg: Annotated[float, Field(ge=-65.0, le=65.0)] = 0.0
    right_ankle_pitch_deg: Annotated[float, Field(ge=-65.0, le=65.0)] = 0.0
    left_foot_shift_x_m: Annotated[float, Field(ge=-0.45, le=0.45)] = 0.0
    left_foot_lift_m: Annotated[float, Field(ge=0.0, le=0.65)] = 0.0
    left_foot_shift_z_m: Annotated[float, Field(ge=-0.55, le=0.55)] = 0.0
    right_foot_shift_x_m: Annotated[float, Field(ge=-0.45, le=0.45)] = 0.0
    right_foot_lift_m: Annotated[float, Field(ge=0.0, le=0.65)] = 0.0
    right_foot_shift_z_m: Annotated[float, Field(ge=-0.55, le=0.55)] = 0.0
    support_mode: BodySupportMode = BodySupportMode.FEET
    lock_feet: bool = True

    def has_authored_change(self) -> bool:
        values = self.model_dump(
            mode="python",
            exclude={"lock_feet", "support_mode"},
        ).values()
        return (
            any(abs(float(value)) > 1e-6 for value in values)
            or self.support_mode != BodySupportMode.FEET
        )

    def has_foot_motion(self) -> bool:
        return any(
            abs(value) > 1e-6
            for value in (
                self.left_foot_shift_x_m,
                self.left_foot_lift_m,
                self.left_foot_shift_z_m,
                self.right_foot_shift_x_m,
                self.right_foot_lift_m,
                self.right_foot_shift_z_m,
            )
        )

    def has_head_motion(self) -> bool:
        return any(
            abs(value) > 1e-6
            for value in (
                self.head_pitch_deg,
                self.head_yaw_deg,
                self.head_roll_deg,
            )
        )

    def has_non_head_motion(self) -> bool:
        values = self.model_dump(
            mode="python",
            exclude={
                "lock_feet",
                "support_mode",
                "head_pitch_deg",
                "head_yaw_deg",
                "head_roll_deg",
            },
        ).values()
        return (
            any(abs(float(value)) > 1e-6 for value in values)
            or self.support_mode != BodySupportMode.FEET
        )


class BodyTarget(Contract):
    """Bounded parameters for one procedural whole-body skill phase."""

    action: BodyAction
    direction_x: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    direction_z: Annotated[float, Field(ge=-1.0, le=1.0)] = 1.0
    distance_m: Annotated[float, Field(ge=0.0, le=3.0)] = 0.0
    turn_degrees: Annotated[float, Field(ge=-180.0, le=180.0)] = 0.0
    rotation_degrees: Annotated[float, Field(ge=-720.0, le=720.0)] = 0.0
    rotation_axis: BodyRotationAxis = BodyRotationAxis.PITCH
    rotation_mode: BodyRotationMode = BodyRotationMode.FLOOR
    obstacle_mode: BodyObstacleMode = BodyObstacleMode.NONE
    obstacle_object_id: str | None = Field(default=None, max_length=64)
    obstacle_clearance_m: Annotated[float, Field(ge=0.0, le=0.25)] = 0.0
    path_lateral_offset_m: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    support_object_id: str | None = Field(default=None, max_length=64)
    climb_direction: BodyClimbDirection = BodyClimbDirection.UP
    height_m: Annotated[float, Field(ge=0.0, le=1.5)] = 0.0
    cycles: Annotated[float, Field(ge=0.0, le=8.0)] = 1.0
    intensity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.65
    lead_side: Hand = Hand.RIGHT
    raise_arms_overhead: bool = False
    pose: BodyPoseTarget = Field(default_factory=BodyPoseTarget)

    @model_validator(mode="after")
    def action_parameters_are_consistent(self) -> "BodyTarget":
        if self.raise_arms_overhead and self.action != BodyAction.JUMP:
            raise ValueError("overhead jump arms are only valid for jump actions")
        if self.action in {BodyAction.STEP, BodyAction.WALK, BodyAction.RUN}:
            if self.distance_m <= 0.0:
                raise ValueError("translating body actions require positive distance")
            if abs(self.direction_x) + abs(self.direction_z) <= 1e-6:
                raise ValueError("translating body actions require a direction")
        if self.action == BodyAction.TURN and abs(self.turn_degrees) <= 1e-6:
            raise ValueError("turn actions require a non-zero angle")
        if self.action == BodyAction.ROTATE:
            if abs(self.rotation_degrees) <= 1e-6:
                raise ValueError("rotate actions require a non-zero rotation")
            if (
                self.rotation_mode
                in {BodyRotationMode.FLOOR, BodyRotationMode.CARTWHEEL}
                and self.distance_m <= 0.0
            ):
                raise ValueError(
                    "grounded rotate actions require positive root travel"
                )
            if self.distance_m > 0.0 and (
                abs(self.direction_x) + abs(self.direction_z) <= 1e-6
            ):
                raise ValueError("rotate actions require a travel direction")
        elif abs(self.rotation_degrees) > 1e-6:
            raise ValueError("rotation_degrees is only valid for rotate actions")
        if self.obstacle_mode == BodyObstacleMode.NONE:
            if self.obstacle_object_id is not None:
                raise ValueError(
                    "obstacle_object_id requires an obstacle traversal mode"
                )
            if abs(self.obstacle_clearance_m) > 1e-6:
                raise ValueError(
                    "obstacle_clearance_m requires an obstacle traversal mode"
                )
            if abs(self.path_lateral_offset_m) > 1e-6:
                raise ValueError(
                    "path_lateral_offset_m requires an obstacle traversal mode"
                )
        else:
            if not self.obstacle_object_id:
                raise ValueError("obstacle traversal requires a scene object id")
            if self.action not in {
                BodyAction.STEP,
                BodyAction.WALK,
                BodyAction.RUN,
            }:
                raise ValueError("obstacle traversal requires locomotion")
            if self.obstacle_mode == BodyObstacleMode.OVER:
                if self.action != BodyAction.STEP:
                    raise ValueError("stepping over an obstacle uses one step action")
                if self.obstacle_clearance_m <= 0.0:
                    raise ValueError(
                        "stepping over an obstacle requires positive clearance"
                    )
                if abs(self.path_lateral_offset_m) > 1e-6:
                    raise ValueError(
                        "over-obstacle steps cannot also use a lateral detour"
                    )
            elif abs(self.path_lateral_offset_m) < 0.10:
                raise ValueError(
                    "moving around an obstacle requires a lateral detour"
                )
        if self.action in {BodyAction.WALK, BodyAction.RUN} and self.cycles <= 0.0:
            raise ValueError("cyclic locomotion requires positive cycles")
        if self.action == BodyAction.DANCE and self.cycles <= 0.0:
            raise ValueError("dance actions require at least one beat")
        if self.action == BodyAction.CLIMB:
            if not self.support_object_id:
                raise ValueError("climb actions require a support object id")
            if self.height_m <= 0.0:
                raise ValueError("climb actions require positive vertical travel")
            if self.cycles <= 0.0:
                raise ValueError("climb actions require at least one contact cycle")
        elif self.support_object_id is not None:
            raise ValueError("support_object_id is only valid for climb actions")
        if self.action == BodyAction.POSE and not self.pose.has_authored_change():
            raise ValueError("pose actions require at least one authored pose control")
        jump_spread_only = bool(
            self.action == BodyAction.JUMP
            and (
                abs(self.pose.left_foot_shift_x_m)
                + abs(self.pose.right_foot_shift_x_m)
                > 1e-8
            )
            and not self.pose.model_copy(
                update={
                    "left_foot_shift_x_m": 0.0,
                    "right_foot_shift_x_m": 0.0,
                }
            ).has_authored_change()
        )
        if (
            self.action != BodyAction.POSE
            and self.pose.has_authored_change()
            and not jump_spread_only
        ):
            raise ValueError("pose controls are only valid for pose actions")
        return self


class ObjectMotionTarget(Contract):
    """Bounded, world-aware controls for a throw or catch trajectory.

    X follows Rigby's avatar-relative convention (+left/-right), Z is
    +forward/-back, and Y is world-up.  The compiler derives a physically
    consistent release velocity and flight duration from these semantic
    controls instead of asking the planner to invent frame-by-frame motion.
    """

    style: ObjectInteractionStyle = ObjectInteractionStyle.OVERHAND
    direction_x: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    direction_z: Annotated[float, Field(ge=-1.0, le=1.0)] = 1.0
    distance_m: Annotated[float, Field(ge=0.08, le=2.50)] = 0.85
    apex_height_m: Annotated[float, Field(ge=0.05, le=1.20)] = 0.35
    spin_turns: Annotated[float, Field(ge=-3.0, le=3.0)] = 0.35
    contact_height_m: Annotated[float, Field(ge=0.85, le=1.65)] = 1.25
    contact_depth_m: Annotated[float, Field(ge=0.18, le=0.58)] = 0.38
    landing_height_m: Annotated[float, Field(ge=0.02, le=1.30)] = 0.04

    @model_validator(mode="after")
    def direction_is_defined(self) -> "ObjectMotionTarget":
        if abs(self.direction_x) + abs(self.direction_z) <= 1e-6:
            raise ValueError("object motion requires a non-zero horizontal direction")
        return self


class MotionPrimitive(Contract):
    kind: PrimitiveKind
    label: str | None = Field(default=None, min_length=1, max_length=64)
    hand_shape: HandShape | None = None
    object_id: str | None = None
    socket_id: str | None = None
    trajectory: TrajectoryKind | None = None
    trajectory_plane: TrajectoryPlane | None = None
    effectors: list[EffectorTarget] = Field(default_factory=list, max_length=2)
    body: BodyTarget | None = None
    parameters: PrimitiveParameters = Field(default_factory=PrimitiveParameters)

    @model_validator(mode="after")
    def effectors_are_unique(self) -> "MotionPrimitive":
        hands = [target.hand for target in self.effectors]
        if len(hands) != len(set(hands)):
            raise ValueError("a primitive cannot target the same hand twice")
        if (
            self.kind in {PrimitiveKind.MOVE, PrimitiveKind.CYCLE}
            and not self.effectors
            and self.object_id is None
        ):
            raise ValueError("compositional movement primitives require effectors")
        if self.kind == PrimitiveKind.BODY and self.body is None:
            raise ValueError("body primitives require a body target")
        if self.kind != PrimitiveKind.BODY and self.body is not None and self.kind != PrimitiveKind.RECOVER:
            raise ValueError("body targets are only valid on body or recovery primitives")
        return self


class AssertionSpec(Contract):
    name: str = Field(min_length=1, max_length=64)
    threshold: float | None = None


class MotionProgram(Contract):
    schema_version: str = "1.0"
    source_text: str = Field(min_length=1, max_length=500)
    intent: Intent
    hand: Hand = Hand.RIGHT
    hands: list[Hand] = Field(default_factory=list, max_length=2)
    strike_type: StrikeType | None = None
    object_action: ObjectAction | None = None
    object_motion: ObjectMotionTarget | None = None
    steps: list["MotionProgram"] = Field(default_factory=list, max_length=8)
    motion_profile: GestureMotionProfile | None = None
    primitives: list[MotionPrimitive] = Field(default_factory=list, max_length=48)
    assertions: list[AssertionSpec] = Field(default_factory=list, max_length=32)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    unsupported_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def intent_consistent(self) -> "MotionProgram":
        if self.intent == Intent.UNSUPPORTED:
            if self.primitives:
                raise ValueError("unsupported programs cannot contain primitives")
            if self.steps:
                raise ValueError("unsupported programs cannot contain sequence steps")
            if not self.unsupported_reason:
                raise ValueError("unsupported programs require a reason")
        elif self.intent == Intent.SEQUENCE:
            if self.primitives:
                raise ValueError("sequence programs contain steps, not direct primitives")
            if len(self.steps) < 2:
                raise ValueError("sequence programs require at least two steps")
            if any(
                step.intent in {Intent.SEQUENCE, Intent.UNSUPPORTED}
                for step in self.steps
            ):
                raise ValueError("sequence steps must be executable, non-nested programs")
        elif not self.primitives:
            raise ValueError("supported programs require primitives")
        if self.intent != Intent.SEQUENCE and self.steps:
            raise ValueError("steps are only valid for sequence programs")
        if self.intent == Intent.STRIKE and self.strike_type is None:
            raise ValueError("strike programs require strike_type")
        if self.intent != Intent.STRIKE and self.strike_type is not None:
            raise ValueError("strike_type is only valid for strike programs")
        if self.intent == Intent.OBJECT_INTERACTION:
            if self.object_action is None or self.object_motion is None:
                raise ValueError("object interaction programs require an action and motion target")
            object_ids = {primitive.object_id for primitive in self.primitives if primitive.object_id}
            if len(object_ids) != 1:
                raise ValueError("object interaction programs require exactly one target object")
        elif self.object_action is not None or self.object_motion is not None:
            raise ValueError("object action and motion target are only valid for object interactions")
        if self.intent == Intent.COMPOSITE:
            if not self.hands:
                raise ValueError("composite programs require at least one active hand")
            if len(self.hands) != len(set(self.hands)):
                raise ValueError("composite program hands must be unique")
            if any(
                primitive.kind not in {PrimitiveKind.MOVE, PrimitiveKind.CYCLE, PrimitiveKind.RECOVER}
                for primitive in self.primitives
            ):
                raise ValueError("composite programs may only contain move, cycle, and recover primitives")
            targeted = {
                target.hand for primitive in self.primitives for target in primitive.effectors
            }
            if not set(self.hands) <= targeted:
                raise ValueError("every active composite hand must have an effector target")
        if self.intent == Intent.FULL_BODY:
            if any(
                primitive.kind not in {PrimitiveKind.BODY, PrimitiveKind.RECOVER}
                for primitive in self.primitives
            ):
                raise ValueError("full-body programs may only contain body and recover primitives")
            if not any(primitive.body is not None for primitive in self.primitives):
                raise ValueError("full-body programs require at least one body target")
        return self


class Failure(Contract):
    code: FailureCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = False


class BonePose(Contract):
    rotation: Quat = Field(default_factory=Quat)
    position: Vec3 | None = None


class ClipFrame(Contract):
    time_s: float = Field(ge=0.0)
    bones: dict[str, BonePose]
    objects: dict[str, Transform] = Field(default_factory=dict)


class ContactEvent(Contract):
    time_s: float = Field(ge=0.0)
    hand: Hand
    object_id: str
    digit: str
    position: Vec3
    normal_force_n: float = Field(ge=0.0)


class Provenance(Contract):
    rig_id: str
    rig_asset: str
    compiler_version: str
    physics_engine: str
    physics_version: str
    planner_provider: str
    planner_model: str
    model_calls: int = Field(ge=0)
    seed: int
    physics_model: dict[str, Any]
    coordinate_frames: dict[str, Any]


class ClipResult(Contract):
    schema_version: str = "1.0"
    success: bool
    fps: int
    duration_s: float = Field(ge=0.0)
    frames: list[ClipFrame]
    contacts: list[ContactEvent] = Field(default_factory=list)
    metrics: dict[str, Any]
    slider_observables: dict[str, float]
    parametric_observables: dict[str, float]
    provenance: Provenance
    failure: Failure | None = None

    @model_validator(mode="after")
    def result_consistent(self) -> "ClipResult":
        if self.success and self.failure is not None:
            raise ValueError("successful result cannot contain failure")
        if not self.success and self.failure is None:
            raise ValueError("failed result requires failure")
        return self


class PlanRequest(Contract):
    text: str = Field(min_length=1, max_length=500)
    scene: SceneManifest
    provider: str = Field(default="auto", pattern=r"^(auto|openai|offline)$")


class ParameterOverrides(Contract):
    hand: Hand | None = None
    duration_s: float | None = Field(default=None, ge=0.05, le=4.0)
    easing: float | None = Field(default=None, ge=0.0, le=1.0)
    arm_height: float | None = Field(default=None, ge=-1.0, le=1.0)
    arm_depth: float | None = Field(default=None, ge=-1.0, le=1.0)
    lateral_offset: float | None = Field(default=None, ge=-1.0, le=1.0)
    wrist_pitch: float | None = Field(default=None, ge=-1.0, le=1.0)
    wrist_yaw: float | None = Field(default=None, ge=-1.0, le=1.0)
    wrist_roll: float | None = Field(default=None, ge=-1.0, le=1.0)
    elbow_swivel: float | None = Field(default=None, ge=-1.0, le=1.0)
    torso_participation: float | None = Field(default=None, ge=0.0, le=1.0)
    path_arc: float | None = Field(default=None, ge=-1.0, le=1.0)
    wrist_flourish: float | None = Field(default=None, ge=-1.0, le=1.0)
    wrist_shake_amplitude: float | None = Field(default=None, ge=0.0, le=1.0)
    wrist_shake_cycles: float | None = Field(default=None, ge=0.0, le=6.0)
    trajectory_amplitude_m: float | None = Field(default=None, ge=0.0, le=0.20)
    trajectory_cycles: float | None = Field(default=None, ge=0.0, le=8.0)
    axial_rotation_amplitude: float | None = Field(default=None, ge=0.0, le=1.0)
    body_distance_m: float | None = Field(default=None, ge=0.0, le=3.0)
    body_turn_degrees: float | None = Field(default=None, ge=-180.0, le=180.0)
    body_height_m: float | None = Field(default=None, ge=0.0, le=0.65)
    body_cycles: float | None = Field(default=None, ge=0.0, le=8.0)
    body_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    object_distance_m: float | None = Field(default=None, ge=0.20, le=2.50)
    object_apex_height_m: float | None = Field(default=None, ge=0.05, le=1.20)
    object_spin_turns: float | None = Field(default=None, ge=-3.0, le=3.0)
    object_contact_height_m: float | None = Field(default=None, ge=0.85, le=1.65)
    object_contact_depth_m: float | None = Field(default=None, ge=0.18, le=0.58)
    object_landing_height_m: float | None = Field(default=None, ge=0.02, le=1.30)
    finger_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    thumb_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    index_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    middle_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    ring_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    little_curl: float | None = Field(default=None, ge=-1.0, le=1.0)
    finger_splay: float | None = Field(default=None, ge=-1.0, le=1.0)
    thumb_opposition: float | None = Field(default=None, ge=0.0, le=1.0)
    grip_force: float | None = Field(default=None, ge=0.0, le=1.0)
    lift_height_m: float | None = Field(default=None, ge=0.0, le=0.25)
    hold_duration_s: float | None = Field(default=None, ge=0.0, le=3.0)
    block_x: float | None = Field(default=None, ge=-0.6, le=0.6)
    block_y: float | None = Field(default=None, ge=0.7, le=1.35)
    block_z: float | None = Field(default=None, ge=0.15, le=0.8)
    block_width_m: float | None = Field(default=None, ge=0.04, le=0.09)
    block_height_m: float | None = Field(default=None, ge=0.05, le=0.14)
    block_depth_m: float | None = Field(default=None, ge=0.04, le=0.14)
    block_mass_kg: float | None = Field(default=None, ge=0.1, le=0.5)
    block_friction: float | None = Field(default=None, ge=0.4, le=2.0)


class CompileRequest(Contract):
    scene: SceneManifest
    program: MotionProgram
    parameter_overrides: ParameterOverrides = Field(default_factory=ParameterOverrides)
    persist: bool = True


class ExportRequest(Contract):
    result_id: str = Field(pattern=r"^\d{6}-[a-z0-9-]+$")


class ResultSummary(Contract):
    id: str
    created_at: str
    prompt: str
    intent: Intent
    success: bool
    duration_s: float = Field(default=0.0, ge=0.0)
    metrics: dict[str, Any]


class CompileResponse(Contract):
    result_id: str | None = None
    clip: ClipResult


class PlanResponse(Contract):
    program: MotionProgram
    provider: str
    model: str
    model_calls: int


class PipelineRunRequest(Contract):
    """One autonomous best-of-five authoring request."""

    text: str = Field(min_length=1, max_length=500)
    scene: SceneManifest
    provider: str = Field(default="auto", pattern=r"^(auto|openai|offline)$")
    max_rounds: int = Field(default=2, ge=1, le=2)


def default_scene() -> SceneManifest:
    block = SceneObject(
        id="block",
        transform=Transform(translation=Vec3(x=0.0, y=1.05, z=0.29)),
        dimensions_m=Vec3(x=0.06, y=0.08, z=0.06),
        mass_kg=0.25,
        friction=1.2,
        sockets=[
            AffordanceSocket(
                id="front_center",
                transform=Transform(translation=Vec3(x=0.0, y=0.0, z=-0.03)),
                approach_normal=Vec3(x=0.0, y=0.0, z=-1.0),
                grasp_span_m=0.06,
            )
        ],
    )
    ladder = SceneObject(
        id="ladder",
        kind="ladder",
        transform=Transform(translation=Vec3(x=0.0, y=1.05, z=0.58)),
        dimensions_m=Vec3(x=0.46, y=2.10, z=0.08),
        mass_kg=18.0,
        friction=0.9,
        sockets=[
            AffordanceSocket(
                id=f"rung_{index + 1}",
                transform=Transform(
                    translation=Vec3(
                        x=0.0,
                        y=-0.82 + index * 0.205,
                        z=-0.05,
                    )
                ),
                approach_normal=Vec3(x=0.0, y=0.0, z=-1.0),
                grasp_span_m=0.04,
                role=AffordanceRole.CLIMB_CONTACT,
                supports_body_weight=True,
            )
            for index in range(9)
        ],
    )
    hurdle = SceneObject(
        id="hurdle",
        kind="hurdle",
        transform=Transform(translation=Vec3(x=0.0, y=0.07, z=0.55)),
        dimensions_m=Vec3(x=0.22, y=0.10, z=0.10),
        mass_kg=4.0,
        friction=1.2,
        sockets=[
            AffordanceSocket(
                id="top_center",
                transform=Transform(translation=Vec3(x=0.0, y=0.05, z=0.0)),
                approach_normal=Vec3(x=0.0, y=1.0, z=0.0),
                grasp_span_m=0.10,
                role=AffordanceRole.SUPPORT,
            )
        ],
    )
    return SceneManifest(objects=[block, ladder, hurdle])
