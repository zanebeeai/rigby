from __future__ import annotations

import os
import hashlib
import math
import random
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    AssertionSpec,
    BodyAction,
    BodyClimbDirection,
    BodyObstacleMode,
    BodyPoseTarget,
    BodyRotationAxis,
    BodyRotationMode,
    BodySupportMode,
    BodyTarget,
    EffectorTarget,
    GestureDepth,
    GestureHeight,
    GestureLateral,
    GestureMotionProfile,
    GestureStyle,
    GestureTiming,
    Hand,
    HandShape,
    Intent,
    MotionPrimitive,
    MotionProgram,
    ObjectAction,
    ObjectInteractionStyle,
    ObjectMotionTarget,
    PlanRequest,
    PrimitiveKind,
    PrimitiveParameters,
    SceneManifest,
    StrikeType,
    TrajectoryKind,
    TrajectoryPlane,
    WristPitch,
    WristRoll,
    WristYaw,
)


DEFAULT_PRIMARY_MODEL = "gpt-5.6-luna"
DEFAULT_REPAIR_MODEL = "gpt-5.6-terra"


@dataclass(frozen=True)
class PlannerOutcome:
    program: MotionProgram
    provider: str
    model: str
    model_calls: int
    response_ids: tuple[str, ...] = ()


class GenericEffectorSelection(BaseModel):
    """One normalized hand target authored by the semantic planner."""

    model_config = ConfigDict(extra="forbid")

    hand: Hand
    target_x: float = Field(ge=-1.0, le=1.0)
    target_y: float = Field(ge=-1.0, le=1.0)
    target_z: float = Field(ge=-1.0, le=1.0)
    reach_overhead: bool = False
    elbow_swivel: float = Field(default=0.0, ge=-1.0, le=1.0)
    wrist_pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    wrist_yaw: float = Field(default=0.0, ge=-1.0, le=1.0)
    wrist_roll: float = Field(default=0.0, ge=-1.0, le=1.0)
    phase_offset_cycles: float = Field(default=0.0, ge=0.0, le=1.0)
    hand_shape: HandShape = HandShape.OPEN


class GenericSegmentSelection(BaseModel):
    """Concurrent upper-body segment produced from unconstrained language."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=64)
    duration_s: float = Field(ge=0.15, le=4.0)
    easing: float = Field(default=0.7, ge=0.0, le=1.0)
    trajectory: TrajectoryKind = TrajectoryKind.LINEAR
    trajectory_plane: TrajectoryPlane = TrajectoryPlane.FRONTAL
    trajectory_amplitude_m: float = Field(default=0.0, ge=0.0, le=0.20)
    trajectory_cycles: float = Field(default=0.0, ge=0.0, le=8.0)
    axial_rotation_amplitude: float = Field(default=0.0, ge=0.0, le=1.0)
    torso_participation: float = Field(default=0.2, ge=0.0, le=1.0)
    effectors: list[GenericEffectorSelection] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def segment_consistent(self) -> "GenericSegmentSelection":
        hands = [target.hand for target in self.effectors]
        if len(hands) != len(set(hands)):
            raise ValueError("a segment cannot target the same hand twice")
        cyclic = self.trajectory in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}
        if cyclic and (self.trajectory_cycles <= 0.0 or self.trajectory_amplitude_m <= 0.0):
            raise ValueError("cyclic trajectories require positive cycles and amplitude")
        return self


class GenericBodySegmentSelection(BaseModel):
    """One bounded procedural whole-body skill selected from language."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=64)
    duration_s: float = Field(ge=0.25, le=4.0)
    easing: float = Field(default=0.72, ge=0.0, le=1.0)
    action: BodyAction
    direction_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    direction_z: float = Field(default=1.0, ge=-1.0, le=1.0)
    distance_m: float = Field(default=0.0, ge=0.0, le=3.0)
    turn_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    rotation_degrees: float = Field(default=0.0, ge=-720.0, le=720.0)
    rotation_axis: BodyRotationAxis = BodyRotationAxis.PITCH
    rotation_mode: BodyRotationMode = BodyRotationMode.FLOOR
    obstacle_mode: BodyObstacleMode = BodyObstacleMode.NONE
    obstacle_object_id: str | None = Field(default=None, max_length=64)
    obstacle_clearance_m: float = Field(default=0.0, ge=0.0, le=0.25)
    path_lateral_offset_m: float = Field(default=0.0, ge=-1.0, le=1.0)
    support_object_id: str | None = Field(default=None, max_length=64)
    climb_direction: BodyClimbDirection = BodyClimbDirection.UP
    height_m: float = Field(default=0.0, ge=0.0, le=1.5)
    cycles: float = Field(default=1.0, ge=0.0, le=8.0)
    intensity: float = Field(default=0.65, ge=0.0, le=1.0)
    lead_side: Hand = Hand.RIGHT
    raise_arms_overhead: bool = False
    pose: BodyPoseTarget = Field(default_factory=BodyPoseTarget)
    trajectory: TrajectoryKind = TrajectoryKind.LINEAR
    trajectory_plane: TrajectoryPlane = TrajectoryPlane.FRONTAL
    trajectory_amplitude_m: float = Field(default=0.0, ge=0.0, le=0.20)
    trajectory_cycles: float = Field(default=0.0, ge=0.0, le=8.0)
    axial_rotation_amplitude: float = Field(default=0.0, ge=0.0, le=1.0)
    torso_participation: float = Field(default=0.2, ge=0.0, le=1.0)
    effectors: list[GenericEffectorSelection] = Field(default_factory=list, max_length=2)
    recovery_transition: bool = False

    @model_validator(mode="after")
    def concurrent_effectors_are_consistent(self) -> "GenericBodySegmentSelection":
        if self.raise_arms_overhead and self.action != BodyAction.JUMP:
            raise ValueError("overhead jump arms are only valid for jump actions")
        if self.action == BodyAction.ROTATE:
            if abs(self.rotation_degrees) <= 1e-6:
                raise ValueError("rotate segments require a non-zero rotation")
            if (
                self.rotation_mode
                in {BodyRotationMode.FLOOR, BodyRotationMode.CARTWHEEL}
                and self.distance_m <= 0.0
            ):
                raise ValueError(
                    "grounded rotate segments require positive root travel"
                )
            if self.distance_m > 0.0 and (
                abs(self.direction_x) + abs(self.direction_z) <= 1e-6
            ):
                raise ValueError("rotate segments require a travel direction")
        elif abs(self.rotation_degrees) > 1e-6:
            raise ValueError("rotation_degrees is only valid for rotate segments")
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
        if self.action == BodyAction.DANCE and self.cycles <= 0.0:
            raise ValueError("dance segments require at least one beat")
        if self.action == BodyAction.CLIMB:
            if not self.support_object_id:
                raise ValueError("climb segments require a support object id")
            if self.height_m <= 0.0:
                raise ValueError("climb segments require positive vertical travel")
            if self.cycles <= 0.0:
                raise ValueError("climb segments require contact cycles")
        elif self.support_object_id is not None:
            raise ValueError("support_object_id is only valid for climb segments")
        hands = [target.hand for target in self.effectors]
        if len(hands) != len(set(hands)):
            raise ValueError("a body segment cannot target the same hand twice")
        cyclic = self.trajectory in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}
        if self.effectors and cyclic and (
            self.trajectory_cycles <= 0.0 or self.trajectory_amplitude_m <= 0.0
        ):
            raise ValueError("cyclic body-segment arm trajectories require cycles and amplitude")
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
            raise ValueError("pose controls are only valid for pose body segments")
        if self.recovery_transition and (
            self.action != BodyAction.HOLD
            or self.effectors
            or self.pose.has_authored_change()
        ):
            raise ValueError(
                "a body recovery transition must be an unposed hold without effectors"
            )
        return self


class PlannerSelection(BaseModel):
    """Bounded semantic decisions delegated to the model."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    hand: Hand = Hand.RIGHT
    hand_shape: HandShape | None = None
    strike_type: StrikeType | None = None
    object_id: str | None = None
    object_action: ObjectAction | None = None
    object_motion: ObjectMotionTarget | None = None
    motion_profile: GestureMotionProfile = Field(default_factory=GestureMotionProfile)
    composition_segments: list[GenericSegmentSelection] = Field(
        default_factory=list,
        max_length=12,
    )
    body_segments: list[GenericBodySegmentSelection] = Field(
        default_factory=list,
        max_length=40,
    )
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def composite_selection_consistent(self) -> "PlannerSelection":
        if self.intent == Intent.SEQUENCE:
            raise ValueError(
                "ordered mixed-family sequences are decomposed deterministically before model selection"
            )
        if self.intent == Intent.COMPOSITE and not self.composition_segments:
            raise ValueError("composite selections require composition_segments")
        if self.intent != Intent.COMPOSITE and self.composition_segments:
            raise ValueError("composition_segments are only valid for composite selections")
        if self.intent == Intent.FULL_BODY and not self.body_segments:
            raise ValueError("full-body selections require body_segments")
        if self.intent != Intent.FULL_BODY and self.body_segments:
            raise ValueError("body_segments are only valid for full-body selections")
        if self.intent == Intent.OBJECT_INTERACTION:
            if self.object_action is None or self.object_motion is None:
                raise ValueError("object interactions require object_action and object_motion")
            if self.object_id is None:
                raise ValueError("object interactions require an exact object_id")
        elif self.object_action is not None or self.object_motion is not None:
            raise ValueError("object_action and object_motion are only valid for object interactions")
        return self


def _composite_program(
    text: str,
    segments: list[GenericSegmentSelection],
    *,
    seed: int = 0,
) -> MotionProgram:
    hands: list[Hand] = []
    primitives: list[MotionPrimitive] = []
    for segment in segments:
        effectors = [
            EffectorTarget.model_validate(target.model_dump(mode="python"))
            for target in segment.effectors
        ]
        for target in effectors:
            if target.hand not in hands:
                hands.append(target.hand)
        primitives.append(
            MotionPrimitive(
                kind=(
                    PrimitiveKind.CYCLE
                    if segment.trajectory in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}
                    else PrimitiveKind.MOVE
                ),
                label=segment.label,
                trajectory=segment.trajectory,
                trajectory_plane=segment.trajectory_plane,
                effectors=effectors,
                parameters=PrimitiveParameters(
                    duration_s=segment.duration_s,
                    easing=segment.easing,
                    torso_participation=segment.torso_participation,
                    trajectory_amplitude_m=segment.trajectory_amplitude_m,
                    trajectory_cycles=segment.trajectory_cycles,
                    axial_rotation_amplitude=segment.axial_rotation_amplitude,
                ),
            )
        )
    if not hands:
        raise ValueError("composite program requires at least one active hand")
    recover_targets = [
        EffectorTarget(
            hand=hand,
            target_x=0.78 if hand == Hand.LEFT else -0.78,
            target_y=-0.72,
            target_z=-0.25,
            elbow_swivel=0.10,
            hand_shape=HandShape.OPEN,
        )
        for hand in hands
    ]
    primitives.append(
        MotionPrimitive(
            kind=PrimitiveKind.RECOVER,
            label="return_to_relaxed",
            trajectory=TrajectoryKind.LINEAR,
            effectors=recover_targets,
            parameters=PrimitiveParameters(duration_s=0.75, easing=0.82),
        )
    )
    return MotionProgram(
        source_text=text,
        intent=Intent.COMPOSITE,
        hand=Hand.RIGHT if Hand.RIGHT in hands else hands[0],
        hands=hands,
        primitives=primitives,
        assertions=[
            AssertionSpec(name="fixed_root"),
            AssertionSpec(name="full_fov_visibility"),
            AssertionSpec(name="coordinated_effectors"),
        ],
        seed=seed,
    )


def _full_body_program(
    text: str,
    segments: list[GenericBodySegmentSelection],
    *,
    seed: int = 0,
) -> MotionProgram:
    primitives: list[MotionPrimitive] = []
    hands: list[Hand] = []
    for segment in segments:
        if segment.action == BodyAction.POSE and not segment.pose.has_authored_change():
            # Models sometimes restate "return to neutral" as an all-zero
            # pose. Recovery is appended below, so this segment is redundant.
            continue
        effectors = [
            EffectorTarget.model_validate(target.model_dump(mode="python"))
            for target in segment.effectors
        ]
        for target in effectors:
            if target.hand not in hands:
                hands.append(target.hand)
        primitives.append(
            MotionPrimitive(
                kind=(
                    PrimitiveKind.RECOVER
                    if segment.recovery_transition
                    else PrimitiveKind.BODY
                ),
                label=segment.label,
                body=BodyTarget(
                    action=segment.action,
                    direction_x=segment.direction_x,
                    direction_z=segment.direction_z,
                    distance_m=segment.distance_m,
                    turn_degrees=segment.turn_degrees,
                    rotation_degrees=segment.rotation_degrees,
                    rotation_axis=segment.rotation_axis,
                    rotation_mode=segment.rotation_mode,
                    obstacle_mode=segment.obstacle_mode,
                    obstacle_object_id=segment.obstacle_object_id,
                    obstacle_clearance_m=segment.obstacle_clearance_m,
                    path_lateral_offset_m=segment.path_lateral_offset_m,
                    support_object_id=segment.support_object_id,
                    climb_direction=segment.climb_direction,
                    height_m=segment.height_m,
                    cycles=segment.cycles,
                    intensity=segment.intensity,
                    lead_side=segment.lead_side,
                    raise_arms_overhead=segment.raise_arms_overhead,
                    pose=segment.pose,
                ),
                trajectory=segment.trajectory if effectors else None,
                trajectory_plane=segment.trajectory_plane if effectors else None,
                effectors=effectors,
                parameters=PrimitiveParameters(
                    duration_s=segment.duration_s,
                    easing=segment.easing,
                    torso_participation=segment.torso_participation,
                    trajectory_amplitude_m=segment.trajectory_amplitude_m,
                    trajectory_cycles=segment.trajectory_cycles,
                    axial_rotation_amplitude=segment.axial_rotation_amplitude,
                ),
            )
        )
    if not primitives:
        raise ValueError("full-body program requires one substantive body segment")
    terminal_climb = bool(
        segments and segments[-1].action == BodyAction.CLIMB
    )
    if not terminal_climb:
        primitives.append(
            MotionPrimitive(
                kind=PrimitiveKind.RECOVER,
                label="settle_to_balanced_stance",
                body=BodyTarget(action=BodyAction.HOLD, cycles=0.0, intensity=0.0),
                parameters=PrimitiveParameters(duration_s=0.60, easing=0.84),
            )
        )
    return MotionProgram(
        source_text=text,
        intent=Intent.FULL_BODY,
        hands=hands,
        primitives=primitives,
        assertions=(
            [
                AssertionSpec(name="bounded_root_motion"),
                AssertionSpec(name="climb_support_contacts"),
                AssertionSpec(name="vertical_travel_completion"),
            ]
            if terminal_climb
            else [
                AssertionSpec(name="bounded_root_motion"),
                AssertionSpec(name="balanced_support"),
                AssertionSpec(name="ground_clearance"),
            ]
        ),
        seed=seed,
    )


def _sequence_program(
    text: str,
    steps: list[MotionProgram],
    *,
    seed: int = 0,
) -> MotionProgram:
    """Preserve an ordered list of independently executable motion skills.

    A sequence is intentionally hierarchical rather than a flattened bag of
    primitives.  Strike, object, locomotion, and gesture primitives reuse
    names such as ``recover`` but have different state machines; retaining the
    child program boundary lets the compiler validate each lifecycle before
    composing the resulting clips.
    """

    if len(steps) < 2:
        raise ValueError("an ordered sequence requires at least two executable steps")
    hands: list[Hand] = []
    for step in steps:
        active = step.hands or [step.hand]
        for hand in active:
            if hand not in hands:
                hands.append(hand)
    return MotionProgram(
        source_text=text,
        intent=Intent.SEQUENCE,
        hand=hands[0] if hands else Hand.RIGHT,
        hands=hands,
        steps=steps,
        assertions=[
            AssertionSpec(name="ordered_step_completion"),
            AssertionSpec(name="continuous_step_boundaries"),
            AssertionSpec(name="preserve_all_requested_actions"),
        ],
        seed=seed,
    )


def _object_interaction_program(
    text: str,
    action: ObjectAction,
    hand: Hand,
    object_id: str,
    motion: ObjectMotionTarget,
    *,
    seed: int = 0,
) -> MotionProgram:
    """Expand one object lifecycle into inspectable, adjustable phases."""

    if action == ObjectAction.HANDOFF:
        raise ValueError("handoff expansion requires an explicit receiver hand")

    base = PrimitiveParameters(arm_depth=0.35, elbow_swivel=0.18)
    if action == ObjectAction.THROW:
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.55}),
            (PrimitiveKind.PRESHAPE, HandShape.PINCH, {"duration_s": 0.24, "finger_curl": 0.24}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.18, "finger_curl": 0.36, "grip_force": 0.25}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.28, "finger_curl": 0.62, "thumb_opposition": 0.90, "grip_force": 0.72}),
            (PrimitiveKind.LIFT, HandShape.FIST, {"duration_s": 0.42, "finger_curl": 0.62, "grip_force": 0.72, "lift_height_m": 0.12}),
            (PrimitiveKind.HOLD, HandShape.FIST, {"duration_s": 0.22, "finger_curl": 0.62, "grip_force": 0.72, "lift_height_m": 0.12, "hold_duration_s": 1.0}),
            (PrimitiveKind.WINDUP, HandShape.FIST, {"duration_s": 0.44, "finger_curl": 0.62, "grip_force": 0.72, "torso_participation": 0.35}),
            (PrimitiveKind.RELEASE, HandShape.OPEN, {"duration_s": 0.18, "finger_curl": 0.0, "grip_force": 0.0, "torso_participation": 0.42}),
            (PrimitiveKind.FLIGHT, HandShape.OPEN, {"duration_s": 0.90, "finger_curl": 0.0, "grip_force": 0.0, "torso_participation": 0.28}),
            (PrimitiveKind.RECOVER, HandShape.OPEN, {"duration_s": 0.62, "finger_curl": 0.0, "grip_force": 0.0}),
        )
        assertions = [
            AssertionSpec(name="contact_before_attachment"),
            AssertionSpec(name="release_before_ballistic_flight"),
            AssertionSpec(name="continuous_object_trajectory"),
        ]
    elif action == ObjectAction.CATCH:
        phase_specs = (
            (PrimitiveKind.RECEIVE, HandShape.OPEN, {"duration_s": 0.55, "finger_splay": 0.30}),
            (PrimitiveKind.FLIGHT, HandShape.OPEN, {"duration_s": 0.75, "finger_splay": 0.30}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.14, "finger_curl": 0.30, "grip_force": 0.30}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.22, "finger_curl": 0.62, "thumb_opposition": 0.90, "grip_force": 0.74}),
            (PrimitiveKind.ABSORB, HandShape.FIST, {"duration_s": 0.36, "finger_curl": 0.62, "grip_force": 0.74, "torso_participation": 0.20}),
            (PrimitiveKind.HOLD, HandShape.FIST, {"duration_s": 0.45, "finger_curl": 0.62, "grip_force": 0.74, "hold_duration_s": 0.45}),
            (PrimitiveKind.RECOVER, HandShape.FIST, {"duration_s": 0.62, "finger_curl": 0.62, "grip_force": 0.74}),
        )
        assertions = [
            AssertionSpec(name="hand_object_intercept"),
            AssertionSpec(name="contact_before_attachment"),
            AssertionSpec(name="continuous_object_trajectory"),
        ]
    elif action in {ObjectAction.PUSH, ObjectAction.ROLL, ObjectAction.SPIN}:
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.48}),
            (PrimitiveKind.PRESHAPE, HandShape.OPEN, {"duration_s": 0.22, "finger_splay": 0.18}),
            (PrimitiveKind.CONTACT, HandShape.OPEN, {"duration_s": 0.18, "grip_force": 0.18}),
            (PrimitiveKind.MOVE, HandShape.OPEN, {"duration_s": 0.78, "torso_participation": 0.34}),
            (PrimitiveKind.HOLD, HandShape.OPEN, {"duration_s": 0.24, "hold_duration_s": 0.24}),
            (PrimitiveKind.RECOVER, HandShape.OPEN, {"duration_s": 0.58}),
        )
        assertions = [
            AssertionSpec(name="contact_before_guided_motion"),
            AssertionSpec(name="continuous_object_trajectory"),
            AssertionSpec(name="release_after_guided_motion"),
            *(
                [AssertionSpec(name="support_plane_rolling")]
                if action == ObjectAction.ROLL
                else []
            ),
            *(
                [AssertionSpec(name="support_plane_spin_completion")]
                if action == ObjectAction.SPIN
                else []
            ),
        ]
    elif action == ObjectAction.PLACE:
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.48}),
            (PrimitiveKind.PRESHAPE, HandShape.PINCH, {"duration_s": 0.22, "finger_curl": 0.24}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.18, "finger_curl": 0.36, "grip_force": 0.24}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.26, "finger_curl": 0.62, "grip_force": 0.72}),
            (PrimitiveKind.LIFT, HandShape.FIST, {"duration_s": 0.38, "finger_curl": 0.62, "grip_force": 0.72}),
            (PrimitiveKind.MOVE, HandShape.FIST, {"duration_s": 0.68, "finger_curl": 0.62, "grip_force": 0.72, "torso_participation": 0.22}),
            (PrimitiveKind.RELEASE, HandShape.OPEN, {"duration_s": 0.34, "finger_curl": 0.0, "grip_force": 0.0}),
            (PrimitiveKind.RECOVER, HandShape.OPEN, {"duration_s": 0.58, "grip_force": 0.0}),
        )
        assertions = [
            AssertionSpec(name="contact_before_attachment"),
            AssertionSpec(name="continuous_object_trajectory"),
            AssertionSpec(name="supported_guided_placement"),
            AssertionSpec(name="release_after_placement"),
        ]
    elif action == ObjectAction.PULL:
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.48}),
            (PrimitiveKind.PRESHAPE, HandShape.PINCH, {"duration_s": 0.22, "finger_curl": 0.24}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.18, "finger_curl": 0.36, "grip_force": 0.24}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.26, "finger_curl": 0.62, "grip_force": 0.70}),
            (PrimitiveKind.MOVE, HandShape.FIST, {"duration_s": 0.72, "finger_curl": 0.62, "grip_force": 0.70, "torso_participation": 0.28}),
            (PrimitiveKind.HOLD, HandShape.FIST, {"duration_s": 0.24, "finger_curl": 0.62, "grip_force": 0.70, "hold_duration_s": 0.24}),
            (PrimitiveKind.RECOVER, HandShape.OPEN, {"duration_s": 0.58, "grip_force": 0.0}),
        )
        assertions = [
            AssertionSpec(name="contact_before_attachment"),
            AssertionSpec(name="continuous_object_trajectory"),
            AssertionSpec(name="release_after_guided_motion"),
        ]
    else:  # drop
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.48}),
            (PrimitiveKind.PRESHAPE, HandShape.PINCH, {"duration_s": 0.22, "finger_curl": 0.24}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.18, "finger_curl": 0.36, "grip_force": 0.24}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.26, "finger_curl": 0.62, "grip_force": 0.72}),
            (PrimitiveKind.LIFT, HandShape.FIST, {"duration_s": 0.42, "finger_curl": 0.62, "grip_force": 0.72, "lift_height_m": 0.12}),
            (PrimitiveKind.HOLD, HandShape.FIST, {"duration_s": 0.24, "finger_curl": 0.62, "grip_force": 0.72, "lift_height_m": 0.12, "hold_duration_s": 0.24}),
            (PrimitiveKind.RELEASE, HandShape.OPEN, {"duration_s": 0.18, "grip_force": 0.0}),
            (PrimitiveKind.FLIGHT, HandShape.OPEN, {"duration_s": 0.55, "grip_force": 0.0}),
            (PrimitiveKind.RECOVER, HandShape.OPEN, {"duration_s": 0.58, "grip_force": 0.0}),
        )
        assertions = [
            AssertionSpec(name="contact_before_attachment"),
            AssertionSpec(name="release_before_ballistic_flight"),
            AssertionSpec(name="continuous_object_trajectory"),
            AssertionSpec(name="supported_landing_height"),
        ]
    return MotionProgram(
        source_text=text,
        intent=Intent.OBJECT_INTERACTION,
        hand=hand,
        object_action=action,
        object_motion=motion,
        seed=seed,
        primitives=[
            MotionPrimitive(
                kind=kind,
                hand_shape=shape,
                object_id=object_id,
                socket_id="front_center",
                parameters=base.model_copy(update=updates),
            )
            for kind, shape, updates in phase_specs
        ],
        assertions=assertions,
    )


def _object_handoff_program(
    text: str,
    source_hand: Hand,
    target_hand: Hand,
    object_id: str,
    *,
    seed: int = 0,
) -> MotionProgram:
    """Author an inspectable two-hand transfer with an overlap interval."""

    phase_specs = (
        (PrimitiveKind.REACH, HandShape.OPEN, 0.48, "source_reach"),
        (PrimitiveKind.PRESHAPE, HandShape.PINCH, 0.22, "source_preshape"),
        (PrimitiveKind.CONTACT, HandShape.PINCH, 0.18, "source_contact"),
        (PrimitiveKind.CLOSE, HandShape.FIST, 0.28, "source_secure"),
        (PrimitiveKind.LIFT, HandShape.FIST, 0.42, "present_for_handoff"),
        (PrimitiveKind.RECEIVE, HandShape.OPEN, 0.45, "receiver_contact"),
        (PrimitiveKind.RELEASE, HandShape.OPEN, 0.32, "ownership_transfer"),
        (PrimitiveKind.HOLD, HandShape.FIST, 0.45, "receiver_retains"),
        (PrimitiveKind.RECOVER, HandShape.OPEN, 0.55, "source_recovers"),
    )
    motion = ObjectMotionTarget(
        direction_x=1.0 if target_hand == Hand.LEFT else -1.0,
        direction_z=0.15,
        distance_m=0.20,
        apex_height_m=0.05,
        contact_height_m=1.22,
        contact_depth_m=0.30,
        landing_height_m=0.04,
        spin_turns=0.0,
    )
    return MotionProgram(
        source_text=text,
        intent=Intent.OBJECT_INTERACTION,
        hand=source_hand,
        hands=[source_hand, target_hand],
        object_action=ObjectAction.HANDOFF,
        object_motion=motion,
        primitives=[
            MotionPrimitive(
                kind=kind,
                label=label,
                hand_shape=shape,
                object_id=object_id,
                socket_id="front_center",
                parameters=PrimitiveParameters(
                    duration_s=duration,
                    finger_curl=(
                        0.62
                        if shape == HandShape.FIST
                        else 0.20
                        if shape == HandShape.PINCH
                        else 0.0
                    ),
                    thumb_opposition=0.90 if shape == HandShape.FIST else 0.75,
                    grip_force=(
                        0.74
                        if shape == HandShape.FIST
                        else 0.25
                        if shape == HandShape.PINCH
                        else 0.0
                    ),
                    elbow_swivel=0.18,
                    torso_participation=0.16,
                ),
            )
            for kind, shape, duration, label in phase_specs
        ],
        assertions=[
            AssertionSpec(name="source_contact_before_attachment"),
            AssertionSpec(name="dual_hand_overlap_before_transfer"),
            AssertionSpec(name="receiver_retains_after_source_release"),
            AssertionSpec(name="continuous_object_trajectory"),
        ],
        seed=seed,
    )


def _repetition_count(text: str, *, default: float = 3.0) -> float:
    lower = text.lower()
    words = {
        "once": 1.0,
        "twice": 2.0,
        "thrice": 3.0,
        "three times": 3.0,
        "four times": 4.0,
        "five times": 5.0,
        "six times": 6.0,
    }
    for phrase, value in words.items():
        if phrase in lower:
            return value
    modified_exercise_number = re.search(
        r"\b(one|two|three|four|five|six|seven|eight)\b"
        r"(?:\s+(?:deep|low|full|shallow|small|slight|slow|controlled|fast|quick|forward|backward|backwards|side|lateral|alternating)){0,3}"
        r"\s+(?:squats?|lunges?|sit[- ]?ups?)\b",
        lower,
    )
    if modified_exercise_number:
        return {
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
        }[modified_exercise_number.group(1)]
    word_number = re.search(
        r"\b(one|two|three|four|five|six|seven|eight)\s+"
        r"(?:times|cycles|repetitions|reps|push[- ]?ups?|burpees?|squats?|steps?|rungs?|beats?|counts?|nods?|shakes?|hops?|jumps?|jumping[- ]?jacks?|waves?)\b",
        lower,
    )
    if word_number:
        return {
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
        }[word_number.group(1)]
    numeric = re.search(r"\b([1-8])\s*(?:times|cycles|repetitions|reps)\b", lower)
    if numeric is None:
        numeric = re.search(
            r"\b([1-8])\s*(?:push[- ]?ups?|burpees?|squats?|steps?|rungs?|beats?|counts?|nods?|shakes?|hops?|jumps?|jumping[- ]?jacks?|waves?)\b",
            lower,
        )
    return float(numeric.group(1)) if numeric else default


def _gesture_phase_parameters(
    profile: GestureMotionProfile,
    hand: Hand,
    variation_seed: int = 0,
) -> tuple[PrimitiveParameters, PrimitiveParameters, PrimitiveParameters]:
    style = {
        GestureStyle.RELAXED: (0.90, 0.55, 0.08),
        GestureStyle.NEUTRAL: (0.65, 0.20, 0.20),
        GestureStyle.ENERGETIC: (0.25, -0.45, 0.70),
        GestureStyle.PRECISE: (0.55, 0.00, 0.02),
        GestureStyle.PLAYFUL: (0.75, 0.75, 0.45),
    }[profile.style]
    timing = {
        GestureTiming.QUICK: (0.32, 0.65, 0.38),
        GestureTiming.BALANCED: (0.55, 1.00, 0.50),
        GestureTiming.SLOW: (0.90, 1.45, 0.75),
    }[profile.timing]
    side = 1.0 if hand == Hand.LEFT else -1.0
    rng = random.Random(variation_seed)

    def accent(scale: float) -> float:
        return rng.uniform(-scale, scale) if variation_seed else 0.0

    def bounded(value: float, minimum: float, maximum: float) -> float:
        return min(maximum, max(minimum, value))

    lateral_relative = {
        GestureLateral.INWARD: -0.55,
        GestureLateral.CENTER: 0.0,
        GestureLateral.OUTWARD: 0.55,
    }[profile.lateral]
    present = PrimitiveParameters(
        duration_s=timing[0] * (1.0 + accent(0.035)),
        easing=bounded(style[0] + accent(0.025), 0.0, 1.0),
        arm_height=bounded({GestureHeight.LOW: -0.75, GestureHeight.CHEST: 0.05, GestureHeight.HIGH: 0.75}[profile.height] + accent(0.035), -1.0, 1.0),
        arm_depth=bounded({GestureDepth.CLOSE: -0.55, GestureDepth.NATURAL: 0.0, GestureDepth.EXTENDED: 0.65}[profile.depth] + accent(0.035), -1.0, 1.0),
        lateral_offset=bounded(side * lateral_relative + accent(0.030), -1.0, 1.0),
        wrist_pitch=bounded({WristPitch.DOWN: -0.55, WristPitch.NEUTRAL: 0.0, WristPitch.UP: 0.55}[profile.wrist_pitch] + accent(0.035), -1.0, 1.0),
        wrist_yaw=bounded(side * {WristYaw.INWARD: -0.50, WristYaw.NEUTRAL: 0.0, WristYaw.OUTWARD: 0.50}[profile.wrist_yaw] + accent(0.035), -1.0, 1.0),
        wrist_roll=bounded({WristRoll.COUNTERCLOCKWISE: -0.55, WristRoll.NEUTRAL: 0.0, WristRoll.CLOCKWISE: 0.55}[profile.wrist_roll] + accent(0.035), -1.0, 1.0),
        elbow_swivel=bounded(style[1] + accent(0.040), -1.0, 1.0),
        torso_participation=bounded(style[2] + accent(0.030), 0.0, 1.0),
        thumb_opposition=0.0,
    )
    hold = present.model_copy(
        update={
            "duration_s": timing[1] * (1.0 + accent(0.025)),
            "hold_duration_s": timing[1] * (1.0 + accent(0.025)),
            "easing": 0.0,
        }
    )
    recover = PrimitiveParameters(
        duration_s=timing[2] * (1.0 + accent(0.035)),
        easing=bounded(style[0] + accent(0.025), 0.0, 1.0),
    )
    return present, hold, recover


def _requested_shake(text: str) -> tuple[float, float] | None:
    """Return (cycles, duration) for a requested repeated forearm oscillation."""
    lower = text.lower()
    if not re.search(r"\b(shak(?:e|es|ing)|wiggl(?:e|es|ing)|waggl(?:e|es|ing))\b|back and forth", lower):
        return None
    if re.search(r"\b(once|one time|one shake)\b", lower):
        cycles = 1.0
    elif re.search(r"\b(twice|two times|two shakes?)\b", lower):
        cycles = 2.0
    elif re.search(r"\b(thrice|three times|three shakes?)\b", lower):
        cycles = 3.0
    elif re.search(r"\b(four times|four shakes?)\b", lower):
        cycles = 4.0
    elif re.search(r"\b(five times|five shakes?)\b", lower):
        cycles = 5.0
    elif re.search(r"\b(six times|six shakes?)\b", lower):
        cycles = 6.0
    else:
        cycles = 3.0
    frequency_hz = 2.6 if re.search(r"\b(rapid|rapidly|fast|quick|quickly|swift|swiftly)\b", lower) else 1.8
    return cycles, max(0.95, cycles / frequency_hz)


def _gesture_primitives(
    text: str,
    shape: HandShape,
    profile: GestureMotionProfile,
    hand: Hand,
    variation_seed: int = 0,
) -> list[MotionPrimitive]:
    """Compile linguistic gesture modifiers into typed, editable phases."""
    present, hold, recover = _gesture_phase_parameters(profile, hand, variation_seed)
    lower = text.lower()
    maximum_middle_curl = bool(
        re.search(
            r"middle three fingers?.{0,45}(?:contracted|curled|closed).{0,30}(?:possible|maximum|maximally|fully)",
            lower,
        )
        or re.search(
            r"(?:contract|curl|close).{0,45}middle three fingers?.{0,30}(?:possible|maximum|maximally|fully)",
            lower,
        )
        or re.search(
            r"middle three fingers?.{0,25}(?:fully|maximally).{0,12}(?:contracted|curled|closed)",
            lower,
        )
        or re.search(
            r"(?:fully|maximally).{0,12}(?:contract|curl|close)(?:ed)?.{0,35}middle three fingers?",
            lower,
        )
    )
    if re.search(r"\b(swift|swiftly)\b", lower):
        present = present.model_copy(update={"duration_s": min(present.duration_s, 0.32)})
    if maximum_middle_curl:
        curl_updates = {"index_curl": 1.0, "middle_curl": 1.0, "ring_curl": 1.0}
        present = present.model_copy(update=curl_updates)
        hold = hold.model_copy(update=curl_updates)

    primitives = [MotionPrimitive(kind=PrimitiveKind.PRESENT, hand_shape=shape, parameters=present)]
    requested_shake = _requested_shake(text)
    if requested_shake is None:
        primitives.append(MotionPrimitive(kind=PrimitiveKind.HOLD, hand_shape=shape, parameters=hold))
    else:
        cycles, shake_duration = requested_shake
        settle = hold.model_copy(update={"duration_s": 0.22, "hold_duration_s": 0.22})
        shake = hold.model_copy(
            update={
                "duration_s": shake_duration,
                "hold_duration_s": shake_duration,
                "easing": 1.0,
                "wrist_shake_amplitude": 0.90,
                "wrist_shake_cycles": cycles,
            }
        )
        primitives.extend(
            [
                MotionPrimitive(kind=PrimitiveKind.HOLD, hand_shape=shape, parameters=settle),
                MotionPrimitive(kind=PrimitiveKind.SHAKE, hand_shape=shape, parameters=shake),
            ]
        )
    primitives.append(MotionPrimitive(kind=PrimitiveKind.RECOVER, hand_shape=HandShape.OPEN, parameters=recover))
    return primitives


def _strike_primitives(
    text: str,
    strike_type: StrikeType,
    hand: Hand,
) -> list[MotionPrimitive]:
    """Expand a semantic strike into editable guard/load/impact/recovery phases."""
    lower = text.lower()
    quick = bool(re.search(r"\b(quick|quickly|fast|rapid|swift|snappy)\b", lower))
    slow = bool(re.search(r"\b(slow|slowly|deliberate|deliberately)\b", lower))
    timing_scale = 0.94 if quick else 1.15 if slow else 1.0
    power = 0.78 if re.search(r"\b(hard|heavy|powerful|power)\b", lower) else 0.62
    elbow_side = 1.0 if hand == Hand.LEFT else -1.0
    strike_family_scale = 1.30 if strike_type == StrikeType.CROSS else 1.0
    if strike_type == StrikeType.HOOK:
        specs = (
            (PrimitiveKind.GUARD, 0.65, 0.08, -0.08, 0.02, 0.26, 0.18, 0.0),
            (PrimitiveKind.LOAD, 0.60, 0.02, 0.00, 0.35, 0.54, 0.38, 0.45),
            (PrimitiveKind.STRIKE, 0.80, 0.08, 0.25, 0.00, 0.66, power, 0.82),
            (PrimitiveKind.FOLLOW_THROUGH, 0.50, -0.02, 0.22, 0.20, 0.62, power * 0.88, 0.52),
            (PrimitiveKind.RECOVER, 0.85, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0),
        )
    elif strike_type == StrikeType.UPPERCUT:
        specs = (
            (PrimitiveKind.GUARD, 0.65, 0.08, -0.08, 0.02, 0.26, 0.18, 0.0),
            (PrimitiveKind.LOAD, 0.58, -0.45, 0.05, 0.18, 0.34, 0.34, 0.32),
            (PrimitiveKind.STRIKE, 0.78, 0.48, 0.32, -0.12, 0.44, power, 0.72),
            (PrimitiveKind.FOLLOW_THROUGH, 0.46, 0.58, 0.22, -0.18, 0.36, power * 0.82, 0.42),
            (PrimitiveKind.RECOVER, 0.85, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0),
        )
    else:
        cross_power = min(0.88, power + (0.10 if strike_type == StrikeType.CROSS else 0.0))
        specs = (
            (PrimitiveKind.GUARD, 0.65, 0.08, -0.08, 0.02, 0.26, 0.18, 0.0),
            (PrimitiveKind.LOAD, 0.52, 0.04, 0.02, 0.10, 0.32, 0.30, 0.14),
            (PrimitiveKind.STRIKE, 0.76, 0.06, 0.64, -0.04, 0.24, cross_power, 0.18),
            (PrimitiveKind.FOLLOW_THROUGH, 0.42, 0.02, 0.52, -0.06, 0.22, cross_power * 0.82, 0.10),
            (PrimitiveKind.RECOVER, 0.82, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0),
        )
    return [
        MotionPrimitive(
            kind=kind,
            hand_shape=HandShape.OPEN if kind == PrimitiveKind.RECOVER else HandShape.FIST,
            parameters=PrimitiveParameters(
                duration_s=duration * timing_scale * strike_family_scale,
                easing=0.82 if kind in {PrimitiveKind.GUARD, PrimitiveKind.RECOVER} else 0.62,
                arm_height=height,
                arm_depth=depth,
                lateral_offset=lateral,
                wrist_pitch=-0.08 if kind in {PrimitiveKind.STRIKE, PrimitiveKind.FOLLOW_THROUGH} else 0.0,
                elbow_swivel=elbow * elbow_side,
                torso_participation=torso,
                path_arc=arc,
                finger_curl=0.18 if kind != PrimitiveKind.RECOVER else 0.0,
                thumb_opposition=0.92 if kind != PrimitiveKind.RECOVER else 0.75,
            ),
        )
        for kind, duration, height, depth, lateral, elbow, torso, arc in specs
    ]


def load_environment() -> Path | None:
    package_root = Path(__file__).resolve().parents[2]
    candidates = (package_root / ".env", package_root.parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def provider_status() -> dict[str, object]:
    loaded_from = load_environment()
    return {
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "dotenv_loaded": loaded_from is not None,
        "dotenv_scope": "poc" if loaded_from and loaded_from.parent.name == "rigby-poc" else "workspace",
        "primary_model": os.getenv("OPENAI_PLANNER_MODEL", DEFAULT_PRIMARY_MODEL),
        "repair_model": os.getenv("OPENAI_REPAIR_MODEL", DEFAULT_REPAIR_MODEL),
    }


class OfflinePlanner:
    _unsupported = re.compile(r"(?!)")
    _body_action = re.compile(
        r"\b(walk|run|jog|side[- ]?step|shuffle|crawl|lunge|step|turn|pivot|crouch|squat|jump|hop|kick|bend|bow|lean|kneel|sit|lie|lying|recline|prone|supine|all[- ]?fours|push[- ]?up|plank|roll|look|nod|shake|tilt|shrug)\w*\b",
        re.IGNORECASE,
    )
    _strike = re.compile(r"\b(punch|hook|jab|uppercut|cross|strike)\b", re.IGNORECASE)
    _grab = re.compile(r"\b(grab|grasp|grip|pick(?: up)?|lift|take|collect)\b", re.IGNORECASE)
    _object_throw = re.compile(r"\b(throw|toss|lob|hurl)\b", re.IGNORECASE)
    _object_catch = re.compile(r"\b(catch|receive)\b", re.IGNORECASE)
    _object_push = re.compile(r"\b(push|shove|slide)\w*\b", re.IGNORECASE)
    _object_pull = re.compile(r"\b(pull|draw|drag)\w*\b", re.IGNORECASE)
    _object_roll = re.compile(r"\broll\w*\b", re.IGNORECASE)
    _object_spin = re.compile(r"\bspin\w*\b", re.IGNORECASE)
    _object_place = re.compile(r"\b(?:place|set)\w*\b", re.IGNORECASE)
    _object_drop = re.compile(r"\b(drop|release|let go of)\w*\b", re.IGNORECASE)
    _gesture_shapes: tuple[tuple[HandShape, re.Pattern[str]], ...] = (
        (
            HandShape.HANG_TEN,
            re.compile(
                r"\b(hang[- ]?ten|hang loose|shaka|surfer(?:'s)? sign|thumb and (?:pinky|little finger)(?: (?:out|extended))?)\b",
                re.I,
            ),
        ),
        (HandShape.PINCH, re.compile(r"\b(pinch|precision grip)\b", re.I)),
        (
            HandShape.THUMBS_UP,
            re.compile(r"\b(?:thumbs?[- ]?up|thumb pointing up)\b", re.I),
        ),
        (
            HandShape.PEACE,
            re.compile(r"\b(?:peace sign|victory sign|v[- ]sign)\b", re.I),
        ),
        (HandShape.POINT, re.compile(r"\b(point(?:ing)?|index finger)\b", re.I)),
        (HandShape.FIST, re.compile(r"\b(fist|clench)\b", re.I)),
        (
            HandShape.OPEN,
            re.compile(r"\b(open (?:your )?(?:right |left )?hand|open (?:right |left )?palm|palm with all fingers open|spread (?:your )?fingers|wave)\b", re.I),
        ),
    )
    _bilateral = re.compile(
        r"\b(both(?:\s+\w+){0,2}\s+(?:hands|arms|forearms|fists)|two[- ]handed|"
        r"each ?other|your (?:hands|forearms|arms)|clap(?:ping)? (?:your )?hands)\b",
        re.IGNORECASE,
    )
    _composite_action = re.compile(
        r"\b(move|put|place|hold|reach|stretch|raise|lower|lift|drop|sweep|circle|rotate|roll|wind|twirl|wave|"
        r"push|pull|cross|uncross|clap|alternate|oscillate|swing)\w*\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _sequence_parts(text: str) -> list[str] | None:
        """Split only explicit temporal connectors, never ambiguous ``and``.

        ``while`` and a plain conjunction usually describe concurrency and are
        handled by full-body effectors/composite segments.  Explicit ``then``
        or semicolons establish an auditable order.  A trailing return-to-rest
        clause remains inside its action because every skill already owns its
        calibrated recovery.
        """

        # "the wrist should then shake" is an internal phase modifier of one
        # gesture, not a second independent skill.
        if re.search(r"\b(?:wrist|hand|arm|forearm)\s+should\s+then\b", text, re.I):
            return None
        # Review-style prompts often enumerate pose/orientation attributes
        # with "then" even though they still describe one presented gesture:
        # "keep pitch neutral, then roll it clockwise".  Splitting the suffix
        # would turn a bounded wrist modifier into an unsupported action.
        if (
            re.search(r"\b(?:and\s+)?then\s+(?:\w+ly\s+)?(?:roll|pitch|yaw|shake)\b", text, re.I)
            and re.search(
                r"\b(?:shaka|hang[- ]?ten|gesture|palm|fist|point|pinch|hand)\b",
                text,
                re.I,
            )
        ):
            return None
        if not re.search(
            r"\b(?:and\s+)?then\b|\bafter\s+that\b|\bafterwards\b|;",
            text,
            flags=re.IGNORECASE,
        ):
            return None
        parts = [
            re.sub(r"^(?:first|next|finally)\s*[:,]?\s*", "", part.strip(" ,.;"), flags=re.I)
            for part in re.split(
                r"\b(?:and\s+)?then\b|\bafter\s+that\b|\bafterwards\b|;",
                text,
                flags=re.IGNORECASE,
            )
            if part.strip(" ,.;")
        ]
        if len(parts) < 2:
            return None
        recovery_only = re.compile(
            r"^(?:(?:cleanly|smoothly|gently|slowly|quickly)\s+)?"
            r"(?:(?:return|recover|come|go|settle)\b.*\b"
            r"(?:default|neutral|rest|resting|start|starting|standing)"
            r"(?:\s+(?:pose|position|stance))?"
            r"|stand(?:\s+back)?\s+up(?:\s+again)?)\.?$",
            flags=re.I,
        )
        if any(recovery_only.fullmatch(part) for part in parts[1:]):
            return None
        return parts

    @classmethod
    def _scene_semantic_guard(
        cls,
        text: str,
        scene: SceneManifest,
    ) -> str | None:
        """Reject scene-dependent requests that a body keyword would misroute.

        A transparent unsupported result is safer than silently animating
        only ``turn`` in "turn the doorknob" or only ``step`` in "carry the
        block forward two steps". The scene schema currently contains small
        graspable blocks; other affordances must be present before they can
        be planned physically.
        """

        lower = text.lower()
        absent_environment = re.search(
            r"\b(door|doorknob|door knob|chair|ladder|rope|lever|button|drawer|window)\b",
            lower,
        )
        if absent_environment and not any(
            item.id.lower() == absent_environment.group(1).replace(" ", "")
            or item.kind.lower() == absent_environment.group(1).replace(" ", "")
            for item in scene.objects
        ):
            return f"scene does not contain requested {absent_environment.group(1)} affordance"

        explicit_scene_object = bool(
            re.search(r"\b(block|cube|box|object in front)\b", lower)
            or any(
                re.search(rf"\b{re.escape(item.id.lower())}\b", lower)
                for item in scene.objects
            )
        )
        pronoun_manipulation = bool(
            re.search(
                r"\b(?:grab|grasp|pick|lift|throw|toss|catch|receive|push|pull|carry|drop|place|pass|slide|drag|kick|stack)\w*\b",
                lower,
            )
        )
        if not explicit_scene_object and not pronoun_manipulation:
            return None
        object_id = cls._resolve_object(lower, scene)
        if object_id is None:
            return None
        if re.search(
            r"\b(?:step|stride|walk|run|go|move)\s+(?:over|around)\b",
            lower,
        ):
            obstacle = scene.object_by_id(object_id)
            if obstacle is not None:
                bottom_y = (
                    obstacle.transform.translation.y
                    - obstacle.dimensions_m.y * 0.5
                )
                if bottom_y > 0.12:
                    return (
                        f"requested object '{object_id}' is not a traversable "
                        "floor-level obstacle"
                    )
        unsupported_action = re.search(
            r"\b(?:(kick|stack)\w*|"
            r"(open|close)(?:d|s|ing)?)\b",
            lower,
        )
        if unsupported_action:
            action_word = unsupported_action.group(1) or unsupported_action.group(2)
            return (
                f"object action '{action_word}' requires a persistent "
                "contact lifecycle that is not yet supported"
            )
        if (
            re.search(r"\b(?:grab|grasp|pick|lift|throw|toss|catch|receive)\w*\b", lower)
            and cls._body_action.search(lower)
            and not re.search(r"\b(?:and\s+)?then\b|\bafter\s+that\b|\bafterwards\b|;", lower)
        ):
            return "concurrent object manipulation and whole-body travel is not yet statefully coupled"
        if cls._object_place.search(text) and re.search(
            r"\b(?:ground|floor)\b",
            lower,
        ):
            return "ground placement requires coupled body reach and support transfer"
        return None

    @classmethod
    def _unilateral_action_segments(
        cls,
        text: str,
    ) -> list[GenericSegmentSelection] | None:
        """Map common single-arm actions to the generic effector vocabulary."""

        lower = text.lower()
        hand = Hand.LEFT if re.search(r"\bleft\b", lower) else Hand.RIGHT
        side = 1.0 if hand == Hand.LEFT else -1.0
        if re.search(r"\bsalut(?:e|ing)\b", lower):
            target = GenericEffectorSelection(
                hand=hand,
                target_x=side * 0.18,
                target_y=0.82,
                target_z=0.12,
                elbow_swivel=side * 0.52,
                wrist_pitch=0.08,
                wrist_roll=-side * 0.24,
                hand_shape=HandShape.OPEN,
            )
            return [
                GenericSegmentSelection(
                    label="salute_to_brow",
                    duration_s=0.55,
                    easing=0.78,
                    effectors=[target],
                ),
                GenericSegmentSelection(
                    label="salute_hold",
                    duration_s=0.55,
                    easing=0.0,
                    effectors=[target],
                ),
            ]
        if re.search(r"\b(?:beckon|come closer|come here)\b", lower):
            target = GenericEffectorSelection(
                hand=hand,
                target_x=side * 0.34,
                target_y=0.48,
                target_z=0.72,
                elbow_swivel=side * 0.22,
                wrist_pitch=0.26,
                hand_shape=HandShape.OPEN,
            )
            cycles = _repetition_count(text, default=3.0)
            return [
                GenericSegmentSelection(
                    label="beckon_setup",
                    duration_s=0.45,
                    effectors=[target],
                ),
                GenericSegmentSelection(
                    label="beckon_cycles",
                    duration_s=max(1.0, cycles / 1.8),
                    trajectory=TrajectoryKind.OSCILLATE,
                    trajectory_plane=TrajectoryPlane.SAGITTAL,
                    trajectory_amplitude_m=0.08,
                    trajectory_cycles=cycles,
                    effectors=[target],
                ),
            ]
        if re.search(r"\b(?:snap|snapping)\b.{0,32}\b(?:finger|fingers)\b", lower):
            target_base = {
                "hand": hand,
                "target_x": side * 0.30,
                "target_y": 0.30,
                "target_z": 0.52,
                "elbow_swivel": side * 0.18,
            }
            segments: list[GenericSegmentSelection] = []
            for index, shape in enumerate(
                (HandShape.OPEN, HandShape.PINCH, HandShape.OPEN, HandShape.PINCH),
                start=1,
            ):
                segments.append(
                    GenericSegmentSelection(
                        label=f"finger_snap_{index}",
                        duration_s=0.20 if index > 1 else 0.40,
                        easing=0.82,
                        effectors=[
                            GenericEffectorSelection(
                                **target_base,
                                hand_shape=shape,
                            )
                        ],
                    )
                )
            return segments
        if re.search(r"\bhigh[- ]?five\b", lower):
            return [
                GenericSegmentSelection(
                    label="high_five_present",
                    duration_s=0.55,
                    easing=0.78,
                    effectors=[
                        GenericEffectorSelection(
                            hand=hand,
                            target_x=side * 0.42,
                            target_y=0.78,
                            target_z=0.76,
                            elbow_swivel=side * 0.18,
                            wrist_pitch=-0.08,
                            hand_shape=HandShape.OPEN,
                        )
                    ],
                )
            ]
        return None

    def _carry_program(
        self,
        text: str,
        scene: SceneManifest,
    ) -> MotionProgram | None:
        """Compose secure pickup and locomotion while retaining one attachment."""

        lower = text.lower()
        if not re.search(r"\b(?:carry|transport)\w*\b", lower):
            return None
        object_id = self._resolve_object(lower, scene)
        if object_id is None:
            return None
        hand = (
            Hand.LEFT
            if re.search(r"\b(?:with|using)\s+(?:your\s+)?left\s+hand\b", lower)
            else Hand.RIGHT
        )
        grab = self.plan(
            PlanRequest(
                text=f"grab the {object_id} with your {hand.value} hand",
                scene=scene,
                provider="offline",
            )
        ).program
        travel_text = re.sub(
            r"\b(?:carry|transport)\w*\b(?:\s+the)?\s+(?:block|cube|box|object|thing|it)?",
            "walk",
            text,
            count=1,
            flags=re.I,
        )
        travel_segments = self._body_segments(travel_text, scene)
        if not travel_segments:
            travel_segments = self._body_segments("walk forward two steps", scene)
        assert travel_segments
        side = 1.0 if hand == Hand.LEFT else -1.0
        carry_effector = GenericEffectorSelection(
            hand=hand,
            target_x=side * 0.30,
            target_y=0.10,
            target_z=0.42,
            elbow_swivel=side * 0.22,
            wrist_pitch=-0.08,
            hand_shape=HandShape.FIST,
        )
        travel_segments = [
            segment.model_copy(
                update={
                    "label": f"carry_{segment.label}",
                    "effectors": [carry_effector],
                    "trajectory": TrajectoryKind.LINEAR,
                    "trajectory_amplitude_m": 0.0,
                    "trajectory_cycles": 0.0,
                }
            )
            for segment in travel_segments
        ]
        transport = _full_body_program(
            travel_text,
            travel_segments,
        )
        transport.primitives[-1] = transport.primitives[-1].model_copy(
            update={
                "trajectory": TrajectoryKind.LINEAR,
                "trajectory_plane": TrajectoryPlane.FRONTAL,
                "effectors": [
                    EffectorTarget.model_validate(
                        carry_effector.model_dump(mode="python")
                    )
                ],
            }
        )
        transport.hands = [hand]
        return _sequence_program(text, [grab, transport])

    @classmethod
    def _body_segments(
        cls,
        text: str,
        scene: SceneManifest | None = None,
    ) -> list[GenericBodySegmentSelection] | None:
        """Map common locomotion/posture language to audited body skills."""

        lower_text = text.lower()
        if (cls._object_roll.search(text) or cls._object_spin.search(text)) and re.search(
            r"\b(block|cube|box|object in front)\b",
            lower_text,
        ):
            # Object rolling owns a continuous hand-contact lifecycle; do not
            # misclassify the verb as the avatar performing a floor roll.
            return None
        climb_match = re.search(
            r"\b(?:climb|climbing|scale|scaling|ascend|ascending|descend|descending)\b",
            lower_text,
        )
        if climb_match and not re.search(r"\bthen\b|;", lower_text):
            support_object = (
                next(
                    (
                        item
                        for item in scene.objects
                        if item.kind == "ladder"
                        and (
                            re.search(rf"\b{re.escape(item.id.lower())}\b", lower_text)
                            or "ladder" in lower_text
                        )
                    ),
                    None,
                )
                if scene is not None
                else None
            )
            if support_object is not None:
                downward = bool(
                    re.search(r"\b(?:down|descend|descending|downward)\b", lower_text)
                )
                distance_match = re.search(
                    r"\b(0(?:\.\d+)?|1(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b",
                    lower_text,
                )
                word_meter = re.search(
                    r"\b(one|half)(?:\s+of\s+a)?\s+(?:meter|metre)\b",
                    lower_text,
                )
                explicit_height = (
                    float(distance_match.group(1))
                    if distance_match
                    else 1.0
                    if word_meter and word_meter.group(1) == "one"
                    else 0.5
                    if word_meter
                    else None
                )
                rung_requested = bool(re.search(r"\brungs?\b", lower_text))
                rung_match = re.search(r"\b([1-8])\s*(?:rungs?|cycles?)\b", lower_text)
                cycles = (
                    float(rung_match.group(1))
                    if rung_match
                    else _repetition_count(text, default=3.0)
                    if re.search(r"\b(?:rungs?|cycles?)\b", lower_text)
                    else max(
                        2.0,
                        min(6.0, round((explicit_height or 0.62) / 0.20)),
                    )
                )
                height_m = (
                    explicit_height
                    if explicit_height is not None
                    else cycles * 0.205
                    if rung_requested
                    else min(0.62, support_object.dimensions_m.y * 0.34)
                )
                height_m = max(0.20, min(1.20, height_m))
                duration_s = min(
                    4.0,
                    max(1.8, cycles * (0.82 if "slow" in lower_text else 0.68)),
                )
                return [
                    GenericBodySegmentSelection(
                        label=(
                            "climb_ladder_down"
                            if downward
                            else "climb_ladder_up"
                        ),
                        duration_s=duration_s,
                        action=BodyAction.CLIMB,
                        support_object_id=support_object.id,
                        climb_direction=(
                            BodyClimbDirection.DOWN
                            if downward
                            else BodyClimbDirection.UP
                        ),
                        height_m=height_m,
                        cycles=cycles,
                        intensity=0.72,
                        lead_side=(
                            Hand.LEFT
                            if re.search(r"\bleft\b", lower_text)
                            else Hand.RIGHT
                        ),
                        torso_participation=0.55,
                        effectors=[
                            GenericEffectorSelection(
                                hand=Hand.LEFT,
                                target_x=0.42,
                                target_y=0.18,
                                target_z=0.78,
                                elbow_swivel=-0.18,
                                hand_shape=HandShape.FIST,
                            ),
                            GenericEffectorSelection(
                                hand=Hand.RIGHT,
                                target_x=-0.42,
                                target_y=0.18,
                                target_z=0.78,
                                elbow_swivel=0.18,
                                hand_shape=HandShape.FIST,
                            ),
                        ],
                    )
                ]
        dance_match = re.search(
            # Keep the named dance "two-step" distinct from a locomotion
            # quantity such as "walk forward two steps".
            r"\b(?:danc\w*|boogie\w*|groov\w*|two-step)\b",
            lower_text,
        )
        if dance_match and not re.search(r"\bthen\b|;", lower_text):
            beat_match = re.search(
                r"\b([1-8])\s*(?:beats?|counts?|steps?)\b",
                lower_text,
            )
            beats = int(beat_match.group(1)) if beat_match else int(
                round(_repetition_count(text, default=4.0))
            )
            beats = max(1, min(8, beats))
            beat_duration = (
                0.82
                if re.search(r"\b(?:slow|slowly|gentle|gently)\b", lower_text)
                else 0.42
                if re.search(
                    r"\b(?:fast|quick|quickly|rapid|energetic)\b",
                    lower_text,
                )
                else 0.58
            )
            # One continuous primitive owns every beat, keeping planted-foot
            # targets and rhythmic phase continuous across the whole phrase.
            return [
                GenericBodySegmentSelection(
                    label="dance_rhythmic_two_step",
                    duration_s=min(4.0, beat_duration * beats),
                    easing=0.74,
                    action=BodyAction.DANCE,
                    cycles=float(beats),
                    intensity=0.66,
                    lead_side=Hand.LEFT,
                    trajectory=TrajectoryKind.OSCILLATE,
                    trajectory_plane=TrajectoryPlane.FRONTAL,
                    trajectory_amplitude_m=0.12,
                    trajectory_cycles=beats / 2.0,
                    torso_participation=0.48,
                    effectors=[
                        GenericEffectorSelection(
                            hand=Hand.LEFT,
                            target_x=0.56,
                            target_y=0.42,
                            target_z=0.44,
                            elbow_swivel=0.24,
                            wrist_roll=-0.10,
                            phase_offset_cycles=0.0,
                            hand_shape=HandShape.OPEN,
                        ),
                        GenericEffectorSelection(
                            hand=Hand.RIGHT,
                            target_x=-0.56,
                            target_y=0.42,
                            target_z=0.44,
                            elbow_swivel=-0.24,
                            wrist_roll=0.10,
                            phase_offset_cycles=0.5,
                            hand_shape=HandShape.OPEN,
                        ),
                    ],
                )
            ]
        step_over_match = re.search(
            r"\b(?:step|stride|walk)\s+over\s+(?:the\s+|an?\s+)?(?:block|cube|box|obstacle|[a-z][a-z0-9_-]*)\b",
            lower_text,
        )
        around_match = re.search(
            r"\b(?:walk|run|go|move|step)\s+around\s+(?:the\s+|an?\s+)?(?:block|cube|box|obstacle|[a-z][a-z0-9_-]*)\b",
            lower_text,
        )
        if (step_over_match or around_match) and scene is not None:
            object_id = cls._resolve_object(lower_text, scene)
            if object_id is None and re.search(r"\bobstacle\b", lower_text):
                object_id = scene.objects[0].id if len(scene.objects) == 1 else None
            obstacle = scene.object_by_id(object_id) if object_id else None
            if obstacle is not None:
                object_x = obstacle.transform.translation.x
                object_z = obstacle.transform.translation.z
                object_distance = math.hypot(object_x, object_z)
                if object_distance <= 1e-6:
                    direction_x, direction_z = 0.0, 1.0
                else:
                    direction_x, direction_z = (
                        object_x / object_distance,
                        object_z / object_distance,
                    )
                if step_over_match:
                    lead_side = (
                        Hand.LEFT
                        if re.search(r"\bleft\s+(?:leg|foot)\b", lower_text)
                        else Hand.RIGHT
                    )
                    approach_distance = max(0.0, object_distance - 0.55)
                    segments: list[GenericBodySegmentSelection] = []
                    if approach_distance > 0.12:
                        segments.append(
                            GenericBodySegmentSelection(
                                label=f"obstacle_approach_{object_id}",
                                duration_s=max(0.85, approach_distance / 0.55),
                                action=BodyAction.WALK,
                                direction_x=direction_x,
                                direction_z=direction_z,
                                distance_m=min(3.0, approach_distance),
                                cycles=max(
                                    1.0,
                                    min(8.0, round(approach_distance / 0.34)),
                                ),
                                intensity=0.62,
                                lead_side=lead_side,
                            )
                        )
                    segments.append(
                        GenericBodySegmentSelection(
                            label=f"obstacle_over_{object_id}",
                            duration_s=1.55,
                            action=BodyAction.STEP,
                            direction_x=direction_x,
                            direction_z=direction_z,
                            distance_m=min(0.68, max(0.46, object_distance - approach_distance)),
                            cycles=1.0,
                            intensity=0.88,
                            lead_side=lead_side,
                            obstacle_mode=BodyObstacleMode.OVER,
                            obstacle_object_id=object_id,
                            obstacle_clearance_m=0.065,
                        )
                    )
                    return segments
                detour_left = not bool(
                    re.search(
                        r"\b(?:around|pass|veer)\b.{0,20}\b(?:right|clockwise)\b",
                        lower_text,
                    )
                )
                object_radius = 0.5 * math.hypot(
                    obstacle.dimensions_m.x,
                    obstacle.dimensions_m.z,
                )
                travel_distance = min(3.0, max(1.0, object_distance + 0.78))
                return [
                    GenericBodySegmentSelection(
                        label=f"obstacle_around_{object_id}",
                        duration_s=2.25,
                        action=(
                            BodyAction.RUN
                            if re.search(r"\brun\b", lower_text)
                            else BodyAction.WALK
                        ),
                        direction_x=direction_x,
                        direction_z=direction_z,
                        distance_m=travel_distance,
                        cycles=4.0,
                        intensity=0.72,
                        lead_side=Hand.RIGHT,
                        obstacle_mode=BodyObstacleMode.AROUND,
                        obstacle_object_id=object_id,
                        path_lateral_offset_m=(
                            0.34 + object_radius
                            if detour_left
                            else -(0.34 + object_radius)
                        ),
                    )
                ]
        cartwheel_match = re.search(r"\bcartwheels?\b", lower_text)
        airborne_flip_match = re.search(
            r"\b(front|forward|back|backward)\s*[- ]?flips?\b"
            r"|\b(front|back)flips?\b",
            lower_text,
        )
        airborne_spin_match = re.search(
            r"\b(?:(?:jump|leap|hop)(?:\s+\w+){0,4}\s+(?:spin|twirl)|"
            r"(?:spin|twirl)(?:\s+\w+){0,4}\s+(?:air|jump)|"
            r"(?:do|perform)\s+(?:a\s+)?(?:180|360|540|720|three[- ]sixty))\b",
            lower_text,
        )
        floor_roll_match = re.search(
            r"\b(?:(forward|front|backward|back)\s+(?:roll|somersault)|(?:roll|somersault)\s+(forward|front|backward|back)|somersault)\b",
            lower_text,
        )
        if airborne_flip_match and not re.search(r"\b(?:and|then)\b|;", lower_text):
            direction_word = next(
                (
                    value
                    for value in airborne_flip_match.groups()
                    if value is not None
                ),
                "front",
            )
            backward = direction_word in {"back", "backward"}
            return [
                GenericBodySegmentSelection(
                    label="airborne_back_flip" if backward else "airborne_front_flip",
                    duration_s=1.65,
                    action=BodyAction.ROTATE,
                    direction_x=0.0,
                    direction_z=-1.0 if backward else 1.0,
                    distance_m=0.28,
                    rotation_degrees=-360.0 if backward else 360.0,
                    rotation_axis=BodyRotationAxis.PITCH,
                    rotation_mode=BodyRotationMode.AIRBORNE,
                    height_m=0.58,
                    cycles=1.0,
                    intensity=0.88,
                )
            ]
        if airborne_spin_match and not re.search(r"\bthen\b|;", lower_text):
            degree_match = re.search(r"\b(180|360|540|720)\b", lower_text)
            degrees = float(degree_match.group(1)) if degree_match else 360.0
            clockwise = bool(re.search(r"\b(?:right|clockwise)\b", lower_text))
            return [
                GenericBodySegmentSelection(
                    label=(
                        "airborne_spin_clockwise"
                        if clockwise
                        else "airborne_spin_counterclockwise"
                    ),
                    duration_s=1.35 + 0.35 * (degrees / 360.0),
                    action=BodyAction.ROTATE,
                    direction_x=0.0,
                    direction_z=0.0,
                    distance_m=0.0,
                    rotation_degrees=-degrees if clockwise else degrees,
                    rotation_axis=BodyRotationAxis.YAW,
                    rotation_mode=BodyRotationMode.AIRBORNE,
                    height_m=0.42,
                    cycles=1.0,
                    intensity=0.78,
                )
            ]
        if cartwheel_match and not re.search(r"\b(?:and|then)\b|;", lower_text):
            to_right = bool(re.search(r"\b(?:right|rightward|rightwards)\b", lower_text))
            direction_x = -1.0 if to_right else 1.0
            return [
                GenericBodySegmentSelection(
                    label="cartwheel_right" if to_right else "cartwheel_left",
                    duration_s=2.40,
                    action=BodyAction.ROTATE,
                    direction_x=direction_x,
                    direction_z=0.0,
                    distance_m=0.90,
                    rotation_degrees=360.0 if to_right else -360.0,
                    rotation_axis=BodyRotationAxis.ROLL,
                    rotation_mode=BodyRotationMode.CARTWHEEL,
                    height_m=0.55,
                    cycles=1.0,
                    intensity=0.78,
                    lead_side=Hand.RIGHT if to_right else Hand.LEFT,
                )
            ]
        if floor_roll_match and not re.search(r"\b(?:and|then)\b|;", lower_text):
            direction_word = next(
                (
                    value
                    for value in floor_roll_match.groups()
                    if value is not None
                ),
                "forward",
            )
            backward = direction_word in {"back", "backward"}
            return [
                GenericBodySegmentSelection(
                    label="floor_roll_backward" if backward else "floor_roll_forward",
                    duration_s=2.10,
                    action=BodyAction.ROTATE,
                    direction_x=0.0,
                    direction_z=-1.0 if backward else 1.0,
                    distance_m=0.78,
                    rotation_degrees=-360.0 if backward else 360.0,
                    rotation_axis=BodyRotationAxis.PITCH,
                    rotation_mode=BodyRotationMode.FLOOR,
                    height_m=0.18,
                    cycles=1.0,
                    intensity=0.82,
                )
            ]

        balance_match = re.search(
            r"\b(?:stand|balance)\s+on\s+(?:your\s+)?(?:(left|right)\s+|one\s+)?(?:leg|foot)\b"
            r"|\blift\s+(?:your\s+)?(left|right)\s+(?:knee|foot|leg)\b",
            text,
            re.I,
        )
        if balance_match and not re.search(r"\b(?:and|then)\b|;", text, re.I):
            lower = text.lower()
            lifted_side_token = balance_match.group(2)
            support_side_token = balance_match.group(1)
            if lifted_side_token:
                raised_hand = Hand.LEFT if lifted_side_token.lower() == "left" else Hand.RIGHT
            elif support_side_token:
                raised_hand = Hand.RIGHT if support_side_token.lower() == "left" else Hand.LEFT
            else:
                raised_hand = Hand.RIGHT
            support_hand = Hand.LEFT if raised_hand == Hand.RIGHT else Hand.RIGHT
            side = 1.0 if support_hand == Hand.LEFT else -1.0
            high_raise = bool(re.search(r"\b(?:high|chest|waist)\b", lower))
            foot_lift = 0.36 if high_raise else 0.27
            raised_left = raised_hand == Hand.LEFT
            pose = BodyPoseTarget(
                root_drop_m=0.035,
                root_shift_x_m=0.10 * side,
                pelvis_roll_deg=-4.0 * side,
                torso_roll_deg=5.0 * side,
                left_hip_pitch_deg=-18.0 if raised_left else -4.0,
                right_hip_pitch_deg=-4.0 if raised_left else -18.0,
                left_hip_roll_deg=8.0 if raised_left else 0.0,
                right_hip_roll_deg=-8.0 if not raised_left else 0.0,
                left_knee_flexion_deg=78.0 if raised_left else 8.0,
                right_knee_flexion_deg=8.0 if raised_left else 78.0,
                left_ankle_pitch_deg=-18.0 if raised_left else 0.0,
                right_ankle_pitch_deg=0.0 if raised_left else -18.0,
                left_foot_lift_m=foot_lift if raised_left else 0.0,
                right_foot_lift_m=0.0 if raised_left else foot_lift,
                left_foot_shift_z_m=0.10 if raised_left else 0.0,
                right_foot_shift_z_m=0.0 if raised_left else 0.10,
                lock_feet=True,
            )
            support_name = support_hand.value
            return [
                GenericBodySegmentSelection(
                    label=f"single_leg_balance_{support_name}_support",
                    duration_s=2.0,
                    action=BodyAction.POSE,
                    cycles=0.0,
                    intensity=0.68,
                    lead_side=raised_hand,
                    pose=pose,
                )
            ]

        sit_up_exercise = bool(
            re.search(r"\bsit-ups?\b|\bsit\s+ups\b", text, re.I)
            or re.search(
                r"\b(?:do|perform)\s+(?:(?:a|one|two|three|four|five|six|seven|eight|[1-8])\s+)?sit\s+up\b",
                text,
                re.I,
            )
        )
        if sit_up_exercise and not re.search(r"\b(?:and|then)\b|;", text, re.I):
            repetitions = int(round(_repetition_count(text, default=1.0)))
            lower = text.lower()
            phase_duration = (
                2.15
                if re.search(r"\b(?:slow|slowly|controlled)\b", lower)
                else 1.55
                if re.search(r"\b(?:fast|quick|quickly|rapid)\b", lower)
                else 1.80
            )
            supine = BodyPoseTarget(
                root_drop_m=0.80,
                root_shift_z_m=0.06,
                pelvis_pitch_deg=-90.0,
                torso_pitch_deg=-4.0,
                left_hip_pitch_deg=-5.0,
                right_hip_pitch_deg=-5.0,
                left_knee_flexion_deg=8.0,
                right_knee_flexion_deg=8.0,
                left_ankle_pitch_deg=-6.0,
                right_ankle_pitch_deg=-6.0,
                support_mode=BodySupportMode.BROAD_FLOOR,
                lock_feet=False,
            )
            curl = BodyPoseTarget(
                root_drop_m=0.78,
                root_shift_z_m=0.06,
                pelvis_pitch_deg=-90.0,
                torso_pitch_deg=95.0,
                left_hip_pitch_deg=-28.0,
                right_hip_pitch_deg=-28.0,
                left_knee_flexion_deg=72.0,
                right_knee_flexion_deg=72.0,
                left_ankle_pitch_deg=-12.0,
                right_ankle_pitch_deg=-12.0,
                support_mode=BodySupportMode.BROAD_FLOOR,
                lock_feet=False,
            )
            segments = [
                GenericBodySegmentSelection(
                    label="sit_up_setup_supine",
                    duration_s=phase_duration,
                    action=BodyAction.POSE,
                    cycles=0.0,
                    intensity=0.70,
                    pose=supine,
                )
            ]
            for repetition in range(1, repetitions + 1):
                segments.extend(
                    [
                        GenericBodySegmentSelection(
                            label=f"sit_up_{repetition}_curl",
                            duration_s=phase_duration,
                            action=BodyAction.POSE,
                            cycles=1.0,
                            intensity=0.75,
                            pose=curl,
                        ),
                        GenericBodySegmentSelection(
                            label=f"sit_up_{repetition}_return",
                            duration_s=phase_duration,
                            action=BodyAction.POSE,
                            cycles=1.0,
                            intensity=0.70,
                            pose=supine,
                        ),
                    ]
                )
            return segments

        if (
            re.search(r"\bsquats?\b", text, re.I)
            and not re.search(r"\b(?:and|then)\b|;", text, re.I)
        ):
            repetitions = int(round(_repetition_count(text, default=1.0)))
            lower = text.lower()
            depth = (
                0.28
                if re.search(r"\b(?:deep|low|full)\b", lower)
                else 0.16
                if re.search(r"\b(?:shallow|small|slight)\b", lower)
                else 0.22
            )
            descent_duration = (
                0.92
                if re.search(r"\b(?:slow|slowly|controlled)\b", lower)
                else 0.56
                if re.search(r"\b(?:fast|quick|quickly|rapid)\b", lower)
                else 0.72
            )
            recovery_duration = max(0.48, descent_duration * 0.88)
            segments: list[GenericBodySegmentSelection] = []
            for repetition in range(1, repetitions + 1):
                segments.append(
                    GenericBodySegmentSelection(
                        label=f"squat_{repetition}_descent",
                        duration_s=descent_duration,
                        action=BodyAction.CROUCH,
                        height_m=depth,
                        cycles=1.0,
                        intensity=0.76,
                    )
                )
                if repetition < repetitions:
                    segments.append(
                        GenericBodySegmentSelection(
                            label=f"squat_{repetition}_stance_reset",
                            duration_s=recovery_duration,
                            action=BodyAction.HOLD,
                            cycles=0.0,
                            intensity=0.0,
                            recovery_transition=True,
                        )
                    )
            return segments

        if (
            re.search(r"\blunges?\b", text, re.I)
            and not re.search(r"\b(?:and|then)\b|;", text, re.I)
        ):
            repetitions = int(round(_repetition_count(text, default=1.0)))
            lower = text.lower()
            explicit_left = bool(re.search(r"\bleft\s+(?:leg|foot)\b", lower))
            explicit_right = bool(re.search(r"\bright\s+(?:leg|foot)\b", lower))
            backward = bool(re.search(r"\b(?:back|backward|backwards|reverse)\b", lower))
            lateral = bool(re.search(r"\b(?:side|lateral|sideways)\b", lower))
            depth_scale = (
                1.15
                if re.search(r"\b(?:deep|low|full)\b", lower)
                else 0.72
                if re.search(r"\b(?:shallow|small|slight)\b", lower)
                else 1.0
            )
            phase_duration = (
                1.0
                if re.search(r"\b(?:slow|slowly|controlled)\b", lower)
                else 0.72
                if re.search(r"\b(?:fast|quick|quickly)\b", lower)
                else 0.84
            )
            recovery_duration = max(0.52, phase_duration * 0.82)
            segments: list[GenericBodySegmentSelection] = []
            for repetition in range(1, repetitions + 1):
                lunging_left = (
                    True
                    if explicit_left
                    else False
                    if explicit_right
                    else repetition % 2 == 0
                )
                side = 1.0 if lunging_left else -1.0
                forward = -1.0 if backward else 1.0
                pose = BodyPoseTarget(
                    root_drop_m=min(0.34, 0.22 * depth_scale),
                    root_shift_x_m=(0.16 * side * depth_scale if lateral else 0.0),
                    root_shift_z_m=(0.16 * forward * depth_scale if not lateral else 0.0),
                    pelvis_pitch_deg=(0.0 if lateral else -6.0 * forward),
                    pelvis_roll_deg=(8.0 * side if lateral else 0.0),
                    torso_pitch_deg=(0.0 if lateral else -10.0 * forward),
                    torso_roll_deg=(-10.0 * side if lateral else 0.0),
                    left_hip_pitch_deg=-58.0 if lunging_left else -16.0,
                    right_hip_pitch_deg=-16.0 if lunging_left else -58.0,
                    left_hip_roll_deg=(-12.0 * side if lateral and lunging_left else 0.0),
                    right_hip_roll_deg=(-12.0 * side if lateral and not lunging_left else 0.0),
                    left_knee_flexion_deg=82.0 if lunging_left else 24.0,
                    right_knee_flexion_deg=24.0 if lunging_left else 82.0,
                    left_ankle_pitch_deg=-24.0 if lunging_left else 8.0,
                    right_ankle_pitch_deg=8.0 if lunging_left else -24.0,
                    left_foot_shift_x_m=(0.34 * side if lateral and lunging_left else -0.10 * side if lateral else 0.0),
                    right_foot_shift_x_m=(0.34 * side if lateral and not lunging_left else -0.10 * side if lateral else 0.0),
                    left_foot_shift_z_m=(0.34 * forward if lunging_left and not lateral else -0.14 * forward if not lateral else 0.0),
                    right_foot_shift_z_m=(0.34 * forward if not lunging_left and not lateral else -0.14 * forward if not lateral else 0.0),
                    lock_feet=True,
                )
                lead_name = "left" if lunging_left else "right"
                segments.append(
                    GenericBodySegmentSelection(
                        label=f"lunge_{repetition}_{lead_name}",
                        duration_s=phase_duration,
                        action=BodyAction.POSE,
                        cycles=1.0,
                        intensity=0.72,
                        lead_side=Hand.LEFT if lunging_left else Hand.RIGHT,
                        pose=pose,
                    )
                )
                if repetition < repetitions:
                    segments.append(
                        GenericBodySegmentSelection(
                            label=f"lunge_{repetition}_stance_reset",
                            duration_s=recovery_duration,
                            action=BodyAction.HOLD,
                            cycles=0.0,
                            intensity=0.0,
                            recovery_transition=True,
                        )
                    )
            return segments

        if re.search(r"\bburpees?\b", text, re.I):
            repetitions = int(round(_repetition_count(text, default=1.0)))
            palm_targets = [
                GenericEffectorSelection(
                    hand=Hand.LEFT,
                    target_x=0.32,
                    target_y=-0.90,
                    target_z=0.80,
                    elbow_swivel=0.20,
                    hand_shape=HandShape.OPEN,
                ),
                GenericEffectorSelection(
                    hand=Hand.RIGHT,
                    target_x=-0.32,
                    target_y=-0.90,
                    target_z=0.80,
                    elbow_swivel=-0.20,
                    hand_shape=HandShape.OPEN,
                ),
            ]
            plank = BodyPoseTarget(
                root_drop_m=0.76,
                pelvis_pitch_deg=82.0,
                torso_pitch_deg=-6.0,
                left_hip_pitch_deg=-10.0,
                right_hip_pitch_deg=-10.0,
                left_knee_flexion_deg=6.0,
                right_knee_flexion_deg=6.0,
                left_ankle_pitch_deg=-15.0,
                right_ankle_pitch_deg=-15.0,
                support_mode=BodySupportMode.PLANK,
                lock_feet=False,
            )
            segments: list[GenericBodySegmentSelection] = []
            for repetition in range(1, repetitions + 1):
                segments.extend(
                    [
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_crouch",
                            duration_s=0.70,
                            action=BodyAction.CROUCH,
                            height_m=0.28,
                            cycles=1.0,
                            intensity=0.75,
                        ),
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_push_up",
                            duration_s=2.0,
                            action=BodyAction.POSE,
                            height_m=0.28,
                            cycles=1.0,
                            intensity=0.78,
                            pose=plank,
                            effectors=[target.model_copy(deep=True) for target in palm_targets],
                        ),
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_stand_return",
                            duration_s=1.0,
                            action=BodyAction.HOLD,
                            cycles=0.0,
                            intensity=0.0,
                            recovery_transition=True,
                        ),
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_takeoff_crouch",
                            duration_s=0.70,
                            action=BodyAction.CROUCH,
                            height_m=0.28,
                            cycles=1.0,
                            intensity=0.78,
                        ),
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_jump",
                            # The synchronized overhead arm arc needs a few
                            # more samples than a hands-down hop to remain
                            # below the rig's per-frame rotation bound.
                            duration_s=1.25,
                            action=BodyAction.JUMP,
                            height_m=0.30,
                            cycles=1.0,
                            intensity=0.82,
                            raise_arms_overhead=True,
                        ),
                    ]
                )
                if repetition < repetitions:
                    # A burpee repetition ends in an overhead jump. Restore
                    # an ordinary landed stance before the next crouch so the
                    # next floor reach begins from real carried state instead
                    # of snapping the arms from overhead to neutral.
                    segments.append(
                        GenericBodySegmentSelection(
                            label=f"burpee_{repetition}_cycle_reset",
                            duration_s=0.65,
                            action=BodyAction.HOLD,
                            cycles=0.0,
                            intensity=0.0,
                            recovery_transition=True,
                        )
                    )
            return segments

        def body_match(value: str) -> re.Match[str] | None:
            for candidate in cls._body_action.finditer(value):
                token = candidate.group(1).lower()
                window = value[
                    max(0, candidate.start() - 32) : min(len(value), candidate.end() + 24)
                ].lower()
                if token in {"bend", "turn"} and re.search(
                    r"\b(?:wrist|finger|hand|elbow|forearm)\b.{0,24}\b(?:bend|turn)"
                    r"|\b(?:bend|turn)\w*\b.{0,20}\b(?:wrist|finger|hand|elbow|forearm)\b",
                    window,
                ):
                    continue
                if token == "roll":
                    lateral_floor_roll = bool(
                        re.search(
                            r"\b(?:roll\s+(?:onto|on|to)\s+(?:your\s+|the\s+)?(?:left\s+|right\s+)?side|roll\s+(?:left|right))\b",
                            window,
                        )
                    )
                    if not lateral_floor_roll:
                        continue
                if token in {"look", "nod", "shake", "tilt"}:
                    head_context = bool(re.search(r"\b(?:head|gaze|chin|shoulder)\b", window))
                    directional_look = token == "look" and bool(
                        re.search(r"\b(?:left|right|up|down|around|back|behind)\b", window)
                    )
                    if not (head_context or directional_look):
                        continue
                return candidate
            return None

        if body_match(text) is None:
            return None
        parts = [
            part.strip(" ,.;")
            for part in re.split(r"\b(?:and\s+)?then\b|[;]", text, flags=re.IGNORECASE)
            if part.strip(" ,.;")
        ]
        segments: list[GenericBodySegmentSelection] = []
        for index, part in enumerate(parts, start=1):
            match = body_match(part)
            if match is None:
                continue
            token = match.group(1).lower()
            jumping_jack = bool(
                token in {"jump", "hop"}
                and re.search(r"\bjumping[- ]?jacks?\b", part, re.I)
            )
            action = {
                "walk": BodyAction.WALK,
                "run": BodyAction.RUN,
                "jog": BodyAction.RUN,
                "sidestep": BodyAction.STEP,
                "side-step": BodyAction.STEP,
                "side step": BodyAction.STEP,
                "shuffle": BodyAction.WALK,
                "crawl": BodyAction.POSE,
                "lunge": BodyAction.POSE,
                "step": BodyAction.STEP,
                "turn": BodyAction.TURN,
                "pivot": BodyAction.TURN,
                "crouch": BodyAction.CROUCH,
                "squat": BodyAction.CROUCH,
                "jump": BodyAction.JUMP,
                "hop": BodyAction.JUMP,
                "kick": BodyAction.KICK,
                "bend": BodyAction.POSE,
                "bow": BodyAction.POSE,
                "lean": BodyAction.POSE,
                "kneel": BodyAction.POSE,
                "sit": BodyAction.POSE,
                "lie": BodyAction.POSE,
                "lying": BodyAction.POSE,
                "recline": BodyAction.POSE,
                "prone": BodyAction.POSE,
                "supine": BodyAction.POSE,
                "all-fours": BodyAction.POSE,
                "all fours": BodyAction.POSE,
                "allfours": BodyAction.POSE,
                "push-up": BodyAction.POSE,
                "push up": BodyAction.POSE,
                "pushup": BodyAction.POSE,
                "plank": BodyAction.POSE,
                "roll": BodyAction.POSE,
                "look": BodyAction.POSE,
                "nod": BodyAction.POSE,
                "shake": BodyAction.POSE,
                "tilt": BodyAction.POSE,
                "shrug": BodyAction.POSE,
            }[token]
            lower = part.lower()
            left_limb = re.search(r"\bleft\s+(?:leg|foot|knee)\b", lower)
            right_limb = re.search(r"\bright\s+(?:leg|foot|knee)\b", lower)
            lead_side = Hand.LEFT if left_limb else Hand.RIGHT
            lateral_left = re.search(r"\b(?:leftward|leftwards|to\s+(?:the\s+)?left)\b", lower)
            lateral_right = re.search(r"\b(?:rightward|rightwards|to\s+(?:the\s+)?right)\b", lower)
            if action in {BodyAction.STEP, BodyAction.WALK, BodyAction.RUN} and not (left_limb or right_limb):
                lateral_left = lateral_left or re.search(r"\bleft\b", lower)
                lateral_right = lateral_right or re.search(r"\bright\b", lower)
            direction_x = 1.0 if lateral_left else -1.0 if lateral_right else 0.0
            direction_z = -1.0 if re.search(r"\b(back|backward|backwards|reverse)\b", lower) else 1.0
            if direction_x and not re.search(r"\b(forward|back|backward|backwards|diagonal)\b", lower):
                direction_z = 0.0
            count = _repetition_count(
                part,
                default=(
                    2.0
                    if token in {"nod", "shake", "crawl"}
                    else 3.0
                    if token in {"push-up", "push up", "pushup"}
                    else 0.0
                    if token == "plank"
                    else 3.0
                    if action == BodyAction.RUN
                    else 2.0
                    if action == BodyAction.WALK
                    else 1.0
                ),
            )
            step_count = re.search(r"\b([1-8])\s+steps?\b", lower)
            if step_count:
                count = float(step_count.group(1))
            else:
                word_step_count = next(
                    (
                        value
                        for word, value in {
                            "one": 1.0,
                            "two": 2.0,
                            "three": 3.0,
                            "four": 4.0,
                            "five": 5.0,
                            "six": 6.0,
                            "seven": 7.0,
                            "eight": 8.0,
                        }.items()
                        if re.search(rf"\b{word}\s+steps?\b", lower)
                    ),
                    None,
                )
                if word_step_count is not None:
                    count = word_step_count
            distance_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", lower)
            distance = float(distance_match.group(1)) if distance_match else (
                # The procedural run currently treats each requested cycle as
                # one footfall.  About 0.55 m per footfall remains inside the
                # rig's planted-leg workspace while still reading as a run;
                # the former 0.68 m default forced knee flips on this avatar.
                0.55 * count if action == BodyAction.RUN else
                0.48 * count if action == BodyAction.WALK else
                0.42 if action == BodyAction.STEP else 0.0
            )
            angle_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:degree|degrees|deg)\b", lower)
            angle = float(angle_match.group(1)) if angle_match else (
                180.0 if re.search(r"\b(around|half(?:\s+turn)?)\b", lower) else
                90.0 if re.search(r"\bquarter(?:\s+turn)?\b", lower) else 90.0
            )
            if action == BodyAction.TURN:
                angle *= -1.0 if re.search(r"\b(right|clockwise)\b", lower) else 1.0
            else:
                angle = 0.0
            height_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b", lower)
            height = float(height_match.group(1)) / 100.0 if height_match else (
                0.32 if action == BodyAction.JUMP else
                0.24 if action == BodyAction.CROUCH else 0.0
            )
            intensity = 0.82 if re.search(r"\b(fast|quick|hard|high|powerful)\b", lower) else 0.48 if re.search(r"\b(slow|gentle|small|shallow)\b", lower) else 0.65
            pose = BodyPoseTarget()
            if action == BodyAction.JUMP and jumping_jack:
                pose = BodyPoseTarget(
                    left_foot_shift_x_m=0.26,
                    right_foot_shift_x_m=-0.26,
                )
                height = 0.24
            if action == BodyAction.POSE:
                pose_scale = (
                    0.55
                    if re.search(r"\b(slight|slightly|small|subtle|shallow)\b", lower)
                    else 1.15
                    if re.search(r"\b(deep|deeply|far)\b", lower)
                    else 1.0
                )
                if token in {"bend", "bow"}:
                    backward = bool(re.search(r"\b(back|backward|backwards)\b", lower))
                    pitch = -42.0 if backward else 62.0
                    pose = BodyPoseTarget(
                        pelvis_pitch_deg=(8.0 if backward else 0.0) * pose_scale,
                        torso_pitch_deg=pitch * pose_scale,
                    )
                elif token == "lean":
                    lateral = bool(
                        re.search(r"\b(left|right|leftward|rightward|sideways)\b", lower)
                    )
                    pose = BodyPoseTarget(
                        root_shift_x_m=(
                            (0.10 if re.search(r"\b(left|leftward)\b", lower) else -0.10)
                            * pose_scale
                            if lateral
                            else 0.0
                        ),
                        root_shift_z_m=(
                            (-0.08 if re.search(r"\b(back|backward|backwards)\b", lower) else 0.08)
                            * pose_scale
                            if not lateral
                            else 0.0
                        ),
                        pelvis_roll_deg=(
                            (8.0 if re.search(r"\b(left|leftward)\b", lower) else -8.0)
                            * pose_scale
                            if lateral
                            else 0.0
                        ),
                        torso_pitch_deg=(
                            (24.0 if re.search(r"\b(back|backward|backwards)\b", lower) else -24.0)
                            * pose_scale
                            if not lateral
                            else 0.0
                        ),
                        torso_roll_deg=(
                            (24.0 if re.search(r"\b(left|leftward)\b", lower) else -24.0)
                            * pose_scale
                            if lateral
                            else 0.0
                        ),
                    )
                elif token == "kneel":
                    kneeling_left = lead_side == Hand.LEFT
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.62, 0.48 * pose_scale),
                        pelvis_pitch_deg=8.0,
                        torso_pitch_deg=-8.0,
                        left_hip_pitch_deg=-68.0 if kneeling_left else -38.0,
                        right_hip_pitch_deg=-38.0 if kneeling_left else -68.0,
                        left_knee_flexion_deg=122.0 if kneeling_left else 78.0,
                        right_knee_flexion_deg=78.0 if kneeling_left else 122.0,
                        left_ankle_pitch_deg=-42.0 if kneeling_left else -18.0,
                        right_ankle_pitch_deg=-18.0 if kneeling_left else -42.0,
                        left_foot_shift_z_m=-0.20 if kneeling_left else 0.26,
                        right_foot_shift_z_m=0.26 if kneeling_left else -0.20,
                        lock_feet=True,
                    )
                elif token == "lunge":
                    lunging_left = bool(
                        left_limb
                        or re.search(r"\b(?:to (?:the )?)?left\b", lower)
                    )
                    forward_sign = (
                        -1.0
                        if re.search(r"\b(back|backward|backwards)\b", lower)
                        else 1.0
                    )
                    lateral_sign = (
                        1.0
                        if re.search(r"\b(?:to (?:the )?)?left\b", lower)
                        else -1.0
                        if re.search(r"\b(?:to (?:the )?)?right\b", lower)
                        else 0.0
                    )
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.32, 0.22 * pose_scale),
                        root_shift_x_m=0.16 * lateral_sign * pose_scale,
                        root_shift_z_m=0.16 * forward_sign * pose_scale,
                        pelvis_pitch_deg=-6.0 * forward_sign,
                        torso_pitch_deg=-10.0 * forward_sign,
                        left_hip_pitch_deg=-58.0 if lunging_left else -16.0,
                        right_hip_pitch_deg=-16.0 if lunging_left else -58.0,
                        left_knee_flexion_deg=82.0 if lunging_left else 24.0,
                        right_knee_flexion_deg=24.0 if lunging_left else 82.0,
                        left_ankle_pitch_deg=-24.0 if lunging_left else 8.0,
                        right_ankle_pitch_deg=8.0 if lunging_left else -24.0,
                        left_foot_shift_x_m=0.22 * lateral_sign if lunging_left else 0.0,
                        right_foot_shift_x_m=0.22 * lateral_sign if not lunging_left else 0.0,
                        left_foot_shift_z_m=0.34 * forward_sign if lunging_left else -0.14 * forward_sign,
                        right_foot_shift_z_m=-0.14 * forward_sign if lunging_left else 0.34 * forward_sign,
                        lock_feet=True,
                    )
                elif token in {"look", "nod", "shake", "tilt"}:
                    if token == "nod":
                        pose = BodyPoseTarget(head_pitch_deg=28.0 * pose_scale)
                    elif token == "shake":
                        pose = BodyPoseTarget(head_yaw_deg=36.0 * pose_scale)
                    elif token == "tilt":
                        pose = BodyPoseTarget(
                            head_roll_deg=(
                                24.0
                                if re.search(r"\bleft\b", lower)
                                else -24.0
                            )
                            * pose_scale
                        )
                    else:
                        pose = BodyPoseTarget(
                            torso_yaw_deg=(
                                14.0
                                if re.search(r"\bleft\b", lower)
                                else -14.0
                                if re.search(r"\bright\b", lower)
                                else 0.0
                            ),
                            head_pitch_deg=(
                                -32.0
                                if re.search(r"\bup\b", lower)
                                else 32.0
                                if re.search(r"\bdown\b", lower)
                                else 0.0
                            ),
                            head_yaw_deg=(
                                62.0
                                if re.search(r"\bleft\b", lower)
                                else -62.0
                                if re.search(r"\bright\b", lower)
                                else 48.0
                            ),
                        )
                elif token == "shrug":
                    pose = BodyPoseTarget(
                        left_shoulder_elevation_deg=16.0 * pose_scale,
                        right_shoulder_elevation_deg=16.0 * pose_scale,
                        head_pitch_deg=-4.0 * pose_scale,
                    )
                elif token in {"all-fours", "all fours", "allfours"}:
                    # Transfer support from the feet to both palms and knees.
                    # The full-body compiler rotates the articulated trunk;
                    # paired effectors below place the open palms under it.
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.78, 0.68 * pose_scale),
                        pelvis_pitch_deg=82.0,
                        torso_pitch_deg=-6.0,
                        left_hip_pitch_deg=-35.0,
                        right_hip_pitch_deg=-35.0,
                        left_knee_flexion_deg=105.0,
                        right_knee_flexion_deg=105.0,
                        left_ankle_pitch_deg=-20.0,
                        right_ankle_pitch_deg=-20.0,
                        support_mode=BodySupportMode.QUADRUPED,
                        lock_feet=False,
                    )
                elif token == "crawl":
                    crawl_distance = min(0.45, 0.16 * count)
                    crawl_left = bool(re.search(r"\b(?:to (?:the )?)?left\b", lower))
                    crawl_right = bool(re.search(r"\b(?:to (?:the )?)?right\b", lower))
                    crawl_backward = bool(
                        re.search(r"\b(?:back|backward|backwards|reverse)\b", lower)
                    )
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.78, 0.68 * pose_scale),
                        root_shift_x_m=(
                            crawl_distance
                            if crawl_left
                            else -crawl_distance
                            if crawl_right
                            else 0.0
                        ),
                        root_shift_z_m=(
                            0.0
                            if crawl_left or crawl_right
                            else -crawl_distance
                            if crawl_backward
                            else crawl_distance
                        ),
                        pelvis_pitch_deg=82.0,
                        torso_pitch_deg=-6.0,
                        left_hip_pitch_deg=-35.0,
                        right_hip_pitch_deg=-35.0,
                        left_knee_flexion_deg=105.0,
                        right_knee_flexion_deg=105.0,
                        left_ankle_pitch_deg=-20.0,
                        right_ankle_pitch_deg=-20.0,
                        support_mode=BodySupportMode.QUADRUPED,
                        lock_feet=False,
                    )
                elif token in {"push-up", "push up", "pushup", "plank"}:
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.82, 0.76 * pose_scale),
                        pelvis_pitch_deg=82.0,
                        torso_pitch_deg=-6.0,
                        left_hip_pitch_deg=-10.0,
                        right_hip_pitch_deg=-10.0,
                        left_knee_flexion_deg=6.0,
                        right_knee_flexion_deg=6.0,
                        left_ankle_pitch_deg=-15.0,
                        right_ankle_pitch_deg=-15.0,
                        support_mode=BodySupportMode.PLANK,
                        lock_feet=False,
                    )
                    height = 0.0 if token == "plank" else 0.28
                elif token == "roll":
                    onto_right = bool(re.search(r"\bright(?:\s+side)?\b", lower))
                    roll_sign = -1.0 if onto_right else 1.0
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.82, 0.74 * pose_scale),
                        pelvis_roll_deg=60.0 * roll_sign,
                        torso_roll_deg=25.0 * roll_sign,
                        left_hip_pitch_deg=-80.0,
                        right_hip_pitch_deg=-80.0,
                        left_knee_flexion_deg=60.0,
                        right_knee_flexion_deg=60.0,
                        left_ankle_pitch_deg=-10.0,
                        right_ankle_pitch_deg=-10.0,
                        support_mode=BodySupportMode.BROAD_FLOOR,
                        lock_feet=False,
                    )
                elif token in {"lie", "lying", "recline", "prone", "supine"}:
                    face_down = token == "prone" or bool(
                        re.search(
                            r"\b(?:face\s*down|facedown|on (?:your|the) (?:stomach|belly|front)|prone)\b",
                            lower,
                        )
                    )
                    face_up = token == "supine" or bool(
                        re.search(
                            r"\b(?:face\s*up|faceup|on (?:your|the) back|supine)\b",
                            lower,
                        )
                    )
                    # Unqualified "lie down" defaults to a face-up resting
                    # pose. Root pitch rotates the complete articulated body;
                    # unlocked feet deliberately transfer support from the
                    # standing contacts to the back/front contact set.
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.85, 0.78 * pose_scale),
                        root_shift_z_m=(-0.06 if face_down else 0.06),
                        pelvis_pitch_deg=(90.0 if face_down else -90.0),
                        torso_pitch_deg=(6.0 if face_down else -4.0),
                        left_hip_pitch_deg=(-5.0 if face_up else 5.0),
                        right_hip_pitch_deg=(-5.0 if face_up else 5.0),
                        left_knee_flexion_deg=8.0,
                        right_knee_flexion_deg=8.0,
                        left_ankle_pitch_deg=-6.0,
                        right_ankle_pitch_deg=-6.0,
                        support_mode=BodySupportMode.BROAD_FLOOR,
                        lock_feet=False,
                    )
                else:  # sit
                    pose = BodyPoseTarget(
                        root_drop_m=min(0.68, 0.54 * pose_scale),
                        pelvis_pitch_deg=6.0,
                        torso_pitch_deg=7.0,
                        left_hip_pitch_deg=-72.0,
                        right_hip_pitch_deg=-72.0,
                        left_knee_flexion_deg=108.0,
                        right_knee_flexion_deg=108.0,
                        left_ankle_pitch_deg=-34.0,
                        right_ankle_pitch_deg=-34.0,
                        left_foot_shift_z_m=0.36,
                        right_foot_shift_z_m=0.36,
                        lock_feet=True,
                    )
            if token in {"look", "tilt"}:
                count = 0.0
            duration = (
                max(1.4, min(4.0, count * 0.90))
                if jumping_jack
                else max(0.85, min(4.0, count * (0.48 if action == BodyAction.RUN else 0.72)))
                if action in {BodyAction.WALK, BodyAction.RUN}
                else max(0.8, min(4.0, count * 0.85))
                if action == BodyAction.JUMP
                else max(2.0, min(4.0, 1.2 + 0.65 * count))
                if token in {"push-up", "push up", "pushup"}
                else max(
                    1.5,
                    min(
                        6.0,
                        float(
                            (
                                re.search(
                                    r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b",
                                    lower,
                                )
                                or [None, "2.5"]
                            )[1]
                        ),
                    ),
                )
                if token == "plank"
                else max(2.2, min(4.0, 1.2 + 0.72 * count))
                if token == "crawl"
                else 1.80
                if action == BodyAction.POSE
                and token
                in {
                    "lie",
                    "lying",
                    "recline",
                    "prone",
                    "supine",
                    "all-fours",
                    "all fours",
                    "allfours",
                    "roll",
                }
                else max(1.0, min(3.0, count / 1.15))
                if token in {"nod", "shake"}
                else 1.25
                if action in {BodyAction.CROUCH, BodyAction.KICK, BodyAction.TURN, BodyAction.POSE}
                else 0.95
            )
            arm_effectors: list[GenericEffectorSelection] = []
            arm_trajectory = TrajectoryKind.LINEAR
            arm_plane = TrajectoryPlane.FRONTAL
            arm_trajectory_amplitude = 0.0
            arm_trajectory_cycles = 0.0
            if token == "crawl":
                arm_trajectory = TrajectoryKind.OSCILLATE
                arm_plane = TrajectoryPlane.HORIZONTAL
                arm_trajectory_amplitude = 0.09
                arm_trajectory_cycles = count
                arm_effectors = [
                    GenericEffectorSelection(
                        hand=Hand.LEFT,
                        target_x=0.32,
                        target_y=-0.90,
                        target_z=0.80,
                        elbow_swivel=0.20,
                        phase_offset_cycles=0.0,
                        hand_shape=HandShape.OPEN,
                    ),
                    GenericEffectorSelection(
                        hand=Hand.RIGHT,
                        target_x=-0.32,
                        target_y=-0.90,
                        target_z=0.80,
                        elbow_swivel=-0.20,
                        phase_offset_cycles=0.5,
                        hand_shape=HandShape.OPEN,
                    ),
                ]
            elif token in {
                "all-fours",
                "all fours",
                "allfours",
                "push-up",
                "push up",
                "pushup",
                "plank",
            }:
                arm_effectors = [
                    GenericEffectorSelection(
                        hand=Hand.LEFT,
                        target_x=0.32,
                        target_y=-0.90,
                        target_z=0.80,
                        elbow_swivel=0.20,
                        hand_shape=HandShape.OPEN,
                    ),
                    GenericEffectorSelection(
                        hand=Hand.RIGHT,
                        target_x=-0.32,
                        target_y=-0.90,
                        target_z=0.80,
                        elbow_swivel=-0.20,
                        hand_shape=HandShape.OPEN,
                    ),
                ]
            elif re.search(r"\bwav(?:e|ing)\w*\b", lower):
                if re.search(r"\b(both|two)\b", lower):
                    wave_hands = (Hand.LEFT, Hand.RIGHT)
                elif re.search(r"\bleft\s+hand\b", lower):
                    wave_hands = (Hand.LEFT,)
                else:
                    wave_hands = (Hand.RIGHT,)
                wave_count_match = re.search(
                    r"\bwav(?:e|ing)\w*\b[^.;]{0,36}\b([1-8])\s*(?:times|cycles|waves)\b",
                    lower,
                )
                arm_trajectory_cycles = (
                    float(wave_count_match.group(1)) if wave_count_match else 3.0
                )
                arm_trajectory = TrajectoryKind.OSCILLATE
                arm_trajectory_amplitude = 0.10
                arm_effectors = [
                    GenericEffectorSelection(
                        hand=wave_hand,
                        target_x=0.72 if wave_hand == Hand.LEFT else -0.72,
                        target_y=0.68,
                        target_z=0.52,
                        elbow_swivel=0.18 if wave_hand == Hand.LEFT else -0.18,
                        phase_offset_cycles=0.0,
                        hand_shape=HandShape.OPEN,
                    )
                    for wave_hand in wave_hands
                ]
            else:
                strike_match = cls._strike.search(part)
                if strike_match is not None:
                    strike_token = strike_match.group(1).lower()
                    strike_hand = (
                        Hand.LEFT
                        if re.search(
                            r"\bleft\s+(?:handed\s+)?(?:punch|hook|jab|uppercut|cross|strike)\b",
                            lower,
                        )
                        else Hand.RIGHT
                    )
                    side = 1.0 if strike_hand == Hand.LEFT else -1.0
                    if strike_token == "hook":
                        target_x = -side * 0.42
                        target_y = 0.24
                        target_z = 0.48
                        arm_trajectory_amplitude = 0.14
                        arm_plane = TrajectoryPlane.HORIZONTAL
                    elif strike_token == "uppercut":
                        target_x = side * 0.18
                        target_y = 0.78
                        target_z = 0.42
                        arm_trajectory_amplitude = 0.12
                        arm_plane = TrajectoryPlane.SAGITTAL
                    else:
                        target_x = side * 0.16
                        target_y = 0.24
                        target_z = 0.92
                        arm_trajectory_amplitude = 0.06
                        arm_plane = TrajectoryPlane.HORIZONTAL
                    arm_trajectory = TrajectoryKind.ARC
                    arm_effectors = [
                        GenericEffectorSelection(
                            hand=strike_hand,
                            target_x=target_x,
                            target_y=target_y,
                            target_z=target_z,
                            elbow_swivel=(
                                -0.28 if strike_hand == Hand.RIGHT else 0.28
                            ),
                            wrist_pitch=-0.08,
                            hand_shape=HandShape.FIST,
                        )
                    ]
                else:
                    concurrent_shape = next(
                        (
                            shape
                            for shape, pattern in cls._gesture_shapes
                            if pattern.search(part)
                        ),
                        None,
                    )
                    if concurrent_shape is None and re.search(
                        r"\bopen\s+(?:left\s+|right\s+|both\s+)?hands?\b",
                        lower,
                    ):
                        concurrent_shape = HandShape.OPEN
                    if concurrent_shape is not None:
                        if re.search(r"\b(both|two)\b", lower):
                            gesture_hands = (Hand.LEFT, Hand.RIGHT)
                        elif re.search(r"\bleft\s+(?:hand|palm|fist|finger)\b", lower):
                            gesture_hands = (Hand.LEFT,)
                        else:
                            gesture_hands = (Hand.RIGHT,)
                        profile = cls._motion_profile(part)
                        target_y = {
                            GestureHeight.LOW: -0.45,
                            GestureHeight.CHEST: 0.18,
                            GestureHeight.HIGH: 0.72,
                        }[profile.height]
                        target_z = {
                            GestureDepth.CLOSE: -0.20,
                            GestureDepth.NATURAL: 0.52,
                            GestureDepth.EXTENDED: 0.88,
                        }[profile.depth]
                        if concurrent_shape == HandShape.POINT and not re.search(
                            r"\b(close|near|inward)\b", lower
                        ):
                            target_z = 0.88
                        arm_trajectory = TrajectoryKind.ARC
                        arm_trajectory_amplitude = 0.07
                        arm_plane = TrajectoryPlane.FRONTAL
                        arm_effectors = [
                            GenericEffectorSelection(
                                hand=gesture_hand,
                                target_x=(
                                    0.44 if gesture_hand == Hand.LEFT else -0.44
                                ),
                                target_y=target_y,
                                target_z=target_z,
                                elbow_swivel=(
                                    0.16 if gesture_hand == Hand.LEFT else -0.16
                                ),
                                hand_shape=concurrent_shape,
                            )
                            for gesture_hand in gesture_hands
                        ]
                    else:
                        arm_plane = TrajectoryPlane.FRONTAL
            if not arm_effectors and not re.search(r"\b(?:snap|snapping)\b", lower):
                concurrent_segments = cls._unilateral_action_segments(part)
                if concurrent_segments:
                    selected_arm_segment = next(
                        (
                            item
                            for item in concurrent_segments
                            if item.trajectory
                            in {TrajectoryKind.CIRCLE, TrajectoryKind.OSCILLATE}
                        ),
                        concurrent_segments[0],
                    )
                    arm_effectors = list(selected_arm_segment.effectors)
                    arm_trajectory = selected_arm_segment.trajectory
                    arm_plane = selected_arm_segment.trajectory_plane
                    arm_trajectory_amplitude = (
                        selected_arm_segment.trajectory_amplitude_m
                    )
                    arm_trajectory_cycles = selected_arm_segment.trajectory_cycles
            segments.append(
                GenericBodySegmentSelection(
                    label=(
                        f"all_fours_{index}"
                        if token in {"all-fours", "all fours", "allfours"}
                        else f"crawl_{index}"
                        if token == "crawl"
                        else f"push_up_{index}"
                        if token in {"push-up", "push up", "pushup"}
                        else f"static_plank_{index}"
                        if token == "plank"
                        else f"jumping_jack_{index}"
                        if jumping_jack
                        else f"side_roll_{index}"
                        if token == "roll"
                        else f"{action.value}_{index}"
                    ),
                    duration_s=duration,
                    action=action,
                    direction_x=direction_x,
                    direction_z=direction_z,
                    distance_m=min(3.0, distance),
                    turn_degrees=angle,
                    height_m=min(0.65, height),
                    cycles=count,
                    intensity=intensity,
                    lead_side=lead_side,
                    pose=pose,
                    trajectory=arm_trajectory,
                    trajectory_plane=arm_plane,
                    trajectory_amplitude_m=arm_trajectory_amplitude,
                    trajectory_cycles=arm_trajectory_cycles,
                    effectors=arm_effectors,
                )
            )
        return segments or None

    @classmethod
    def _object_motion(
        cls,
        text: str,
        scene: SceneManifest,
    ) -> tuple[ObjectAction, Hand, str, ObjectMotionTarget] | None:
        """Recognize physical object lifecycles without confusing gesture idioms."""

        lower = text.lower()
        action = (
            ObjectAction.CATCH
            if cls._object_catch.search(text)
            else ObjectAction.THROW
            if cls._object_throw.search(text)
            else ObjectAction.PUSH
            if cls._object_push.search(text)
            else ObjectAction.PULL
            if cls._object_pull.search(text)
            else ObjectAction.ROLL
            if cls._object_roll.search(text)
            else ObjectAction.SPIN
            if cls._object_spin.search(text)
            else ObjectAction.PLACE
            if cls._object_place.search(text)
            else ObjectAction.DROP
            if cls._object_drop.search(text)
            else None
        )
        if action is None:
            return None
        # These phrases are resolved by the earlier gesture/strike branches,
        # but keeping the exclusion here makes the object parser safe to reuse.
        if re.search(r"\b(?:hook|jab|cross|uppercut|punch|hang[- ]?ten|shaka)\b", lower):
            return None
        object_id = cls._resolve_object(lower, scene)
        if object_id is None:
            return None
        explicit_left = re.search(
            r"\b(?:with|using|use)\s+(?:your\s+)?left\s+hand\b|\bleft[- ]handed\b|\bleft\s+hand\s+(?:throw|catch|push|pull)",
            lower,
        )
        explicit_right = re.search(
            r"\b(?:with|using|use)\s+(?:your\s+)?right\s+hand\b|\bright[- ]handed\b|\bright\s+hand\s+(?:throw|catch|push|pull)",
            lower,
        )
        hand = Hand.LEFT if explicit_left else Hand.RIGHT if explicit_right else Hand.RIGHT
        style = (
            ObjectInteractionStyle.UNDERHAND
            if re.search(r"\bunderhand\b", lower)
            else ObjectInteractionStyle.SIDEARM
            if re.search(r"\bsidearm\b|\bside-arm\b", lower)
            else ObjectInteractionStyle.TOSS
            if re.search(r"\b(?:toss|lob|gentl(?:e|y)|softly)\b", lower)
            else ObjectInteractionStyle.OVERHAND
        )
        to_left = re.search(r"\b(?:to|toward|towards)\s+(?:the\s+)?left\b", lower)
        to_right = re.search(r"\b(?:to|toward|towards)\s+(?:the\s+)?right\b", lower)
        direction_x = 1.0 if to_left else -1.0 if to_right else 0.0
        direction_z = (
            -1.0
            if action in {ObjectAction.PULL}
            or re.search(r"\b(?:back|backward|backwards|behind|toward you|towards you|closer)\b", lower)
            else 1.0
        )
        if direction_x and not re.search(r"\b(?:forward|ahead|back|behind|diagonal)\b", lower):
            direction_z = 0.15
        distance = (
            0.08
            if action in {ObjectAction.DROP, ObjectAction.SPIN}
            else 0.14
            if action == ObjectAction.PULL
            else 0.18
            if action in {ObjectAction.PUSH, ObjectAction.ROLL, ObjectAction.PLACE}
            else 1.60
            if re.search(r"\b(?:far|hard|powerful|across)\b", lower)
            else 0.45
            if re.search(r"\b(?:near|nearby|short|gentl(?:e|y)|softly)\b", lower)
            else 0.85
        )
        explicit_distance = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", lower)
        if explicit_distance:
            distance = min(2.5, max(0.08, float(explicit_distance.group(1))))
        apex = (
            0.05
            if action == ObjectAction.DROP
            else 0.65
            if re.search(r"\b(?:high|loft|arc(?:ing|ed)?)\b", lower)
            else 0.18
            if re.search(r"\b(?:flat|low)\b", lower)
            else 0.30
            if style in {ObjectInteractionStyle.UNDERHAND, ObjectInteractionStyle.TOSS}
            else 0.38
        )
        contact_height = (
            1.52
            if re.search(r"\b(?:high|overhead|above your head)\b", lower)
            else 0.98
            if re.search(r"\b(?:low|near the ground|below your waist)\b", lower)
            else 1.25
        )
        spin = (
            -1.0
            if action == ObjectAction.SPIN and "counterclockwise" in lower
            else 1.0
            if action == ObjectAction.SPIN
            else -0.75
            if "counterclockwise" in lower
            else 0.75
            if "clockwise" in lower
            else 0.30
        )
        target_object = scene.object_by_id(object_id)
        landing_height = (
            target_object.transform.translation.y
            if action == ObjectAction.PLACE and target_object is not None
            else max(
                0.02,
                (target_object.dimensions_m.y / 2.0) if target_object else 0.04,
            )
        )
        return (
            action,
            hand,
            object_id,
            ObjectMotionTarget(
                style=style,
                direction_x=direction_x,
                direction_z=direction_z,
                distance_m=distance,
                apex_height_m=apex,
                spin_turns=spin,
                contact_height_m=contact_height,
                landing_height_m=landing_height,
            ),
        )

    @classmethod
    def _composite_segments(cls, text: str) -> list[GenericSegmentSelection] | None:
        """Deterministically decompose common coordinated upper-body language.

        This is a semantic fallback for offline operation. The OpenAI planner
        can author the same normalized segment schema for motions not covered
        by these lexical cues.
        """

        lower = text.lower()
        explicit_sides = "left" in lower and "right" in lower
        if not (cls._bilateral.search(lower) or explicit_sides) or not cls._composite_action.search(lower):
            return None
        hands = (Hand.LEFT, Hand.RIGHT)
        shape = (
            HandShape.FIST
            if re.search(
                r"\b(fist|closed|clench|forearms?|travel(?:ing)?[\"']?\s+foul)\b",
                lower,
            )
            else HandShape.OPEN
        )
        target_y = (
            0.68
            if re.search(r"\b(overhead|above (?:the )?head|high)\b", lower)
            else -0.55
            if re.search(r"\b(low|down|waist)\b", lower)
            else 0.10
        )
        reach_overhead = bool(
            re.search(r"\b(overhead|above (?:the )?head)\b", lower)
        )
        target_z = (
            0.78
            if re.search(r"\b(extend|forward|away)\b", lower)
            else -0.45
            if re.search(r"\b(close|toward (?:the )?(?:body|chest))\b", lower)
            else 0.35
        )
        around_each_other = bool(re.search(r"\b(around|over) each ?other\b", lower))
        hands_on_hips = bool(
            re.search(r"\b(?:put|place|hold|rest)\b.*\bboth\s+hands\b.*\bhips?\b", lower)
            or re.search(r"\bboth\s+hands\b.*\b(?:on|at)\s+(?:your\s+)?hips?\b", lower)
        )
        forward_reach = bool(
            re.search(r"\b(?:reach|extend|push)\w*\b.*\b(?:forward|ahead|out)\b", lower)
        )
        if hands_on_hips:
            # Keep the palms just forward of the iliac crest. On this avatar
            # a literal negative-depth target hides both hands below/behind
            # the head camera and is visually indistinguishable from missing
            # arms in the egocentric evidence.
            target_y = -0.28
            target_z = 0.14
        elif forward_reach:
            target_y = 0.16
            target_z = 0.92
        if around_each_other and target_z == 0.35:
            # A travel signal lives close to the chest. Pushing both hands far
            # forward makes the two forearms form a deep V and intersect at
            # their circular base pose. This target preserves egocentric
            # visibility while keeping the visible forearm axes nearly level.
            target_y = 0.60
            target_z = -0.20
        # Keep generic bilateral targets away from the shoulder-aligned IK
        # singularity.  Around-each-other motions intentionally sit nearer
        # the midline with a flexed elbow.
        center_x = (
            0.36
            if around_each_other
            else 0.42
            if hands_on_hips
            else 0.30
            if forward_reach
            else 0.68
        )
        direction = -1.0 if "counterclockwise" in lower else 1.0
        base_effectors = [
            GenericEffectorSelection(
                hand=hand,
                target_x=(center_x if hand == Hand.LEFT else -center_x),
                target_y=target_y,
                target_z=target_z,
                reach_overhead=reach_overhead,
                elbow_swivel=(
                    0.72
                    if hands_on_hips and hand == Hand.LEFT
                    else -0.72
                    if hands_on_hips
                    else 0.34
                    if hand == Hand.LEFT
                    else -0.34
                ),
                wrist_roll=(
                    -0.34
                    if hands_on_hips and hand == Hand.LEFT
                    else 0.34
                    if hands_on_hips
                    else 0.0
                ),
                phase_offset_cycles=(0.0 if hand == Hand.LEFT else 0.5),
                hand_shape=shape,
            )
            for hand in hands
        ]
        setup = GenericSegmentSelection(
            label=(
                "parallel_forearm_travel_setup"
                if around_each_other
                else "coordinated_setup"
            ),
            duration_s=0.90 if around_each_other else 0.65,
            easing=0.78,
            trajectory=TrajectoryKind.LINEAR,
            trajectory_plane=TrajectoryPlane.FRONTAL,
            trajectory_amplitude_m=0.0,
            torso_participation=0.16,
            effectors=base_effectors,
        )
        if re.search(r"\b(circle|rotate|roll|wind|twirl)\w*\b", lower):
            cycles = _repetition_count(text)
            cycle_effectors = [
                target.model_copy(
                    update={
                        "phase_offset_cycles": (
                            target.phase_offset_cycles
                            if direction > 0
                            else (1.0 - target.phase_offset_cycles) % 1.0
                        )
                    }
                )
                for target in base_effectors
            ]
            return [
                setup,
                GenericSegmentSelection(
                    label=(
                        "parallel_forearm_travel_cycle"
                        if around_each_other
                        else "coordinated_circles"
                    ),
                    duration_s=min(4.0, max(1.8, cycles / 0.65)),
                    easing=0.62,
                    trajectory=TrajectoryKind.CIRCLE,
                    trajectory_plane=(
                        TrajectoryPlane.FRONTAL
                        if around_each_other
                        or re.search(r"\b(in front|frontal|sideways)\b", lower)
                        else TrajectoryPlane.HORIZONTAL
                        if re.search(r"\b(horizontal|around the body)\b", lower)
                        else TrajectoryPlane.SAGITTAL
                    ),
                    # For the travel signal this is the clearance radius of
                    # two parallel cross-body forearms orbiting one shared
                    # sternum-centered axis, not two independent hand circles.
                    trajectory_amplitude_m=0.090 if around_each_other else 0.10,
                    trajectory_cycles=cycles,
                    axial_rotation_amplitude=0.68 if around_each_other else 0.0,
                    torso_participation=0.22,
                    effectors=cycle_effectors,
                ),
            ]
        if re.search(r"\b(alternate|oscillate|back and forth|side to side|swing)\w*\b", lower):
            cycles = _repetition_count(text)
            return [
                setup,
                GenericSegmentSelection(
                    label="coordinated_oscillation",
                    duration_s=max(1.5, cycles / 1.25),
                    trajectory=TrajectoryKind.OSCILLATE,
                    trajectory_plane=TrajectoryPlane.FRONTAL,
                    trajectory_amplitude_m=0.10,
                    trajectory_cycles=cycles,
                    effectors=base_effectors,
                ),
            ]
        if re.search(r"\bthen\s+(?:slowly\s+|quickly\s+)?(?:lower|drop|bring)\b", lower):
            lowered = [
                target.model_copy(update={"target_y": -0.62})
                for target in base_effectors
            ]
            return [
                setup,
                GenericSegmentSelection(
                    label="coordinated_lowering",
                    duration_s=0.75,
                    easing=0.78,
                    trajectory=TrajectoryKind.LINEAR,
                    trajectory_plane=TrajectoryPlane.FRONTAL,
                    torso_participation=0.12,
                    effectors=lowered,
                ),
            ]
        return [setup]

    @staticmethod
    def _motion_profile(text: str) -> GestureMotionProfile:
        lower = text.lower()
        return GestureMotionProfile(
            style=(
                GestureStyle.RELAXED if "relaxed" in lower else
                GestureStyle.ENERGETIC if "energetic" in lower else
                GestureStyle.PRECISE if "precise" in lower else
                GestureStyle.PLAYFUL if "playful" in lower else GestureStyle.NEUTRAL
            ),
            timing=(
                GestureTiming.SLOW if re.search(r"\b(slow|slowly|unhurried)\b", lower) else
                GestureTiming.QUICK if re.search(r"\b(quick|quickly|snappy|brisk|snap|swift|swiftly|rapid|rapidly)\b", lower) else GestureTiming.BALANCED
            ),
            height=(
                GestureHeight.HIGH if re.search(r"\b(high|above shoulder)\b", lower) else
                GestureHeight.LOW if re.search(r"\b(low|below chest)\b", lower) else GestureHeight.CHEST
            ),
            depth=(
                GestureDepth.CLOSE if re.search(r"\b(close|near the body)\b", lower) else
                GestureDepth.EXTENDED
                if re.search(
                    r"\b(far forward|fully forward|(?:arm|reach) (?:fully )?extended)\b"
                    r"|\b(?:shaka|hang[- ]?ten)\b[^.]{0,80}\band (?:fully )?extended\b",
                    lower,
                )
                else GestureDepth.NATURAL
            ),
            lateral=(
                GestureLateral.INWARD if re.search(r"\b(drawn inward|pulled inward|toward the midline)\b", lower) else
                GestureLateral.OUTWARD if re.search(r"\b(held outward|positioned outward|away from the body)\b", lower) else GestureLateral.CENTER
            ),
            wrist_pitch=(
                WristPitch.UP if "pitch the wrist up" in lower else
                WristPitch.DOWN if "pitch the wrist down" in lower else WristPitch.NEUTRAL
            ),
            wrist_yaw=(
                WristYaw.INWARD if "yaw it inward" in lower else
                WristYaw.OUTWARD if "yaw it outward" in lower else WristYaw.NEUTRAL
            ),
            wrist_roll=(
                WristRoll.COUNTERCLOCKWISE if "counterclockwise" in lower else
                WristRoll.CLOCKWISE if "clockwise" in lower else WristRoll.NEUTRAL
            ),
        )

    def plan(self, request: PlanRequest) -> PlannerOutcome:
        text = " ".join(request.text.strip().split())
        lower = text.lower()
        has_left = bool(re.search(r"\bleft\b", lower))
        has_right = bool(re.search(r"\bright\b", lower))
        hand = Hand.LEFT if has_left else Hand.RIGHT
        contradiction = re.search(
            r"do not touch.+(?:pick|lift|grab)|(?:pick|lift).+leav(?:e|ing).+table|"
            r"fist.+(?:extend|open).+finger|both hands closed.+open (?:the )?(?:right|left) palm|"
            r"only (?:your )?left hand.+only (?:your )?right hand|"
            r"only (?:your )?right hand.+only (?:your )?left hand",
            lower,
        )
        if contradiction:
            return self._unsupported_result(text, "contradictory request", hand)
        handoff_verb = re.search(
            r"\b(?:pass|transfer|hand(?:\s+over)?|give|switch)\w*\b",
            lower,
        )
        if handoff_verb:
            explicit_handoff = re.search(
                r"\bfrom\s+(?:your\s+)?(left|right)\s+hand\b"
                r".*\b(?:to|into)\s+(?:your\s+)?(left|right)\s+hand\b",
                lower,
            )
            target_only = re.search(
                r"\b(?:to|into)\s+(?:your\s+)?(left|right)\s+hand\b",
                lower,
            )
            if explicit_handoff or target_only:
                target_hand = Hand(
                    explicit_handoff.group(2)
                    if explicit_handoff
                    else target_only.group(1)
                )
                source_hand = (
                    Hand(explicit_handoff.group(1))
                    if explicit_handoff
                    else Hand.LEFT
                    if target_hand == Hand.RIGHT
                    else Hand.RIGHT
                )
                if source_hand == target_hand:
                    return self._unsupported_result(
                        text,
                        "object handoff requires two different hands",
                        source_hand,
                    )
                object_id = self._resolve_object(lower, request.scene)
                if object_id is None:
                    return self._unsupported_result(
                        text,
                        "unknown or ambiguous handoff object",
                        source_hand,
                    )
                return PlannerOutcome(
                    program=_object_handoff_program(
                        text,
                        source_hand,
                        target_hand,
                        object_id,
                    ),
                    provider="offline",
                    model="rule-planner-v7",
                    model_calls=0,
                )
        semantic_guard = self._scene_semantic_guard(text, request.scene)
        if semantic_guard:
            return self._unsupported_result(text, semantic_guard, hand)
        sequence_parts = self._sequence_parts(text)
        if sequence_parts:
            steps: list[MotionProgram] = []
            active_object_id: str | None = None

            def manipulation_object_ids(program: MotionProgram) -> set[str]:
                ids = {
                    primitive.object_id
                    for primitive in program.primitives
                    if primitive.object_id
                }
                for child in program.steps:
                    ids.update(manipulation_object_ids(child))
                return ids

            for index, part in enumerate(sequence_parts, start=1):
                planning_part = (
                    re.sub(
                        r"\b(?:it|that object|the object|the same object)\b",
                        active_object_id,
                        part,
                        flags=re.IGNORECASE,
                    )
                    if active_object_id is not None
                    and re.search(
                        r"\b(?:it|that object|the object|the same object)\b",
                        part,
                        re.IGNORECASE,
                    )
                    else part
                )
                outcome = self.plan(
                    PlanRequest(
                        text=planning_part,
                        scene=request.scene,
                        provider="offline",
                    )
                )
                if planning_part != part and outcome.program.intent != Intent.UNSUPPORTED:
                    outcome = PlannerOutcome(
                        program=outcome.program.model_copy(update={"source_text": part}),
                        provider=outcome.provider,
                        model=outcome.model,
                        model_calls=outcome.model_calls,
                        response_ids=outcome.response_ids,
                    )
                if outcome.program.intent == Intent.UNSUPPORTED:
                    full_composite = self._composite_segments(text)
                    if full_composite:
                        return PlannerOutcome(
                            program=_composite_program(text, full_composite),
                            provider="offline",
                            model="rule-planner-v5",
                            model_calls=0,
                        )
                    reason = outcome.program.unsupported_reason or "unrecognized action"
                    return self._unsupported_result(
                        text,
                        f"unsupported sequence step {index}: {reason}",
                        hand,
                    )
                if outcome.program.intent == Intent.SEQUENCE:
                    steps.extend(
                        child.model_copy(deep=True)
                        for child in outcome.program.steps
                    )
                else:
                    steps.append(outcome.program)
                resolved_ids = manipulation_object_ids(outcome.program)
                if len(resolved_ids) == 1:
                    active_object_id = next(iter(resolved_ids))

            # The full-body compiler already owns continuous planted support
            # across multiple body phases. Keep a purely whole-body sequence
            # in that state machine; the hierarchical sequence contract is
            # for boundaries between different executable skill families.
            contains_free_horizontal_pose = any(
                primitive.body is not None
                and primitive.body.action == BodyAction.POSE
                and not primitive.body.pose.lock_feet
                and abs(primitive.body.pose.pelvis_pitch_deg) >= 80.0
                for step in steps
                for primitive in step.primitives
            )
            if (
                steps
                and all(step.intent == Intent.FULL_BODY for step in steps)
                and not contains_free_horizontal_pose
            ):
                merged_primitives = [
                    primitive.model_copy(deep=True)
                    for step in steps
                    for primitive in step.primitives
                    if primitive.kind != PrimitiveKind.RECOVER
                ]
                merged_primitives.append(
                    steps[-1].primitives[-1].model_copy(deep=True)
                )
                merged_hands = list(
                    dict.fromkeys(
                        active_hand
                        for step in steps
                        for active_hand in step.hands
                    )
                )
                return PlannerOutcome(
                    program=MotionProgram(
                        source_text=text,
                        intent=Intent.FULL_BODY,
                        hand=merged_hands[0] if merged_hands else hand,
                        hands=merged_hands,
                        primitives=merged_primitives,
                        assertions=[
                            AssertionSpec(name="bounded_root_motion"),
                            AssertionSpec(name="balanced_support"),
                            AssertionSpec(name="ground_clearance"),
                        ],
                    ),
                    provider="offline",
                    model="rule-planner-v5",
                    model_calls=0,
                )

            # Manipulation lifecycles already contain their required reach,
            # contact, and (where needed) secure hold. Avoid replaying a
            # standalone grab that would reset the same object's state at the
            # child-program boundary.
            normalized: list[MotionProgram] = []
            for step in steps:
                if (
                    normalized
                    and normalized[-1].intent == Intent.GRAB
                    and step.intent == Intent.GRAB
                    and normalized[-1].hand == step.hand
                ):
                    previous_ids = {
                        item.object_id
                        for item in normalized[-1].primitives
                        if item.object_id is not None
                    }
                    current_ids = {
                        item.object_id
                        for item in step.primitives
                        if item.object_id is not None
                    }
                    if previous_ids == current_ids:
                        # A carry clause expands to ``grab + travel``.  If an
                        # explicit preceding clause already secured the same
                        # object with the same hand, retain that ownership
                        # instead of replaying the pickup lifecycle.
                        continue
                if (
                    normalized
                    and normalized[-1].intent == Intent.GRAB
                    and step.intent == Intent.OBJECT_INTERACTION
                    and step.object_action != ObjectAction.CATCH
                ):
                    grabbed_ids = {
                        item.object_id
                        for item in normalized[-1].primitives
                        if item.object_id is not None
                    }
                    thrown_ids = {
                        item.object_id for item in step.primitives if item.object_id is not None
                    }
                    if grabbed_ids == thrown_ids:
                        normalized.pop()
                normalized.append(step)
            if len(normalized) == 1:
                only = normalized[0].model_copy(deep=True)
                only.source_text = text
                return PlannerOutcome(
                    program=only,
                    provider="offline",
                    model="rule-planner-v5",
                    model_calls=0,
                )
            return PlannerOutcome(
                program=_sequence_program(text, normalized),
                provider="offline",
                model="rule-planner-v5",
                model_calls=0,
            )
        carry_program = self._carry_program(text, request.scene)
        if carry_program is not None:
            return PlannerOutcome(
                program=carry_program,
                provider="offline",
                model="rule-planner-v6",
                model_calls=0,
            )
        body_segments = self._body_segments(text, request.scene)
        if body_segments:
            return PlannerOutcome(
                program=_full_body_program(text, body_segments),
                provider="offline",
                model="rule-planner-v3",
                model_calls=0,
            )
        unilateral_segments = self._unilateral_action_segments(text)
        if unilateral_segments:
            return PlannerOutcome(
                program=_composite_program(text, unilateral_segments),
                provider="offline",
                model="rule-planner-v5",
                model_calls=0,
            )
        composite_segments = self._composite_segments(text)
        if composite_segments:
            program = _composite_program(text, composite_segments)
            return PlannerOutcome(
                program=program,
                provider="offline",
                model="rule-planner-v2",
                model_calls=0,
            )
        strike = self._strike.search(text)
        if strike:
            token = strike.group(1).lower()
            strike_type = {
                "hook": StrikeType.HOOK,
                "jab": StrikeType.JAB,
                "uppercut": StrikeType.UPPERCUT,
                "cross": StrikeType.CROSS,
                "punch": StrikeType.CROSS,
                "strike": StrikeType.CROSS,
            }[token]
            profile = self._motion_profile(text)
            return PlannerOutcome(
                program=MotionProgram(
                    source_text=text,
                    intent=Intent.STRIKE,
                    hand=hand,
                    strike_type=strike_type,
                    motion_profile=profile,
                    primitives=_strike_primitives(text, strike_type, hand),
                    assertions=[
                        AssertionSpec(name="closed_fist_at_impact"),
                        AssertionSpec(name="curved_or_direct_strike_path"),
                        AssertionSpec(name="fixed_root"),
                    ],
                ),
                provider="offline",
                model="rule-planner-v1",
                model_calls=0,
            )
        if re.search(r"thumb.+index fingertip", lower):
            shape = HandShape.PINCH
        else:
            shape = next((item for item, pattern in self._gesture_shapes if pattern.search(text)), None)
        if shape is not None and not re.search(r"\b(block|cube|box)\b", lower):
            profile = self._motion_profile(text)
            program = MotionProgram(
                source_text=text,
                intent=Intent.GESTURE,
                hand=hand,
                motion_profile=profile,
                primitives=_gesture_primitives(text, shape, profile, hand),
                assertions=[AssertionSpec(name="finger_assertions"), AssertionSpec(name="fixed_root")],
            )
            return PlannerOutcome(program=program, provider="offline", model="rule-planner-v1", model_calls=0)

        object_motion = self._object_motion(text, request.scene)
        if object_motion is not None:
            action, object_hand, object_id, motion = object_motion
            return PlannerOutcome(
                program=_object_interaction_program(
                    text,
                    action,
                    object_hand,
                    object_id,
                    motion,
                ),
                provider="offline",
                model="rule-planner-v4",
                model_calls=0,
            )

        unsupported = self._unsupported.search(text)
        if unsupported:
            return self._unsupported_result(text, f"unsupported motion: {unsupported.group(1)}", hand)

        if self._grab.search(text):
            object_id = self._resolve_object(lower, request.scene)
            if object_id is None:
                return self._unsupported_result(text, "unknown or ambiguous object", hand)
            overhead_lift = bool(
                re.search(
                    r"\b(?:overhead|over (?:your |the )?head|above (?:your |the )?head)\b",
                    lower,
                )
            )
            lift_height_m = 0.70 if overhead_lift else 0.12
            params = PrimitiveParameters(lift_height_m=lift_height_m)
            phases = (
                (PrimitiveKind.REACH, HandShape.OPEN, 0.55),
                (PrimitiveKind.PRESHAPE, HandShape.OPEN, 0.25),
                (PrimitiveKind.CONTACT, HandShape.PINCH, 0.20),
                (PrimitiveKind.CLOSE, HandShape.FIST, 0.35),
                (PrimitiveKind.LIFT, HandShape.FIST, 0.85 if overhead_lift else 0.60),
                (PrimitiveKind.HOLD, HandShape.FIST, 1.20 if overhead_lift else 1.00),
                (PrimitiveKind.RECOVER, HandShape.FIST, 0.55),
            )
            program = MotionProgram(
                source_text=text,
                intent=Intent.GRAB,
                hand=hand,
                primitives=[
                    MotionPrimitive(
                        kind=kind,
                        hand_shape=hand_shape,
                        object_id=object_id,
                        socket_id="front_center",
                        parameters=params.model_copy(
                            update={
                                "duration_s": duration,
                                "hold_duration_s": (
                                    duration
                                    if kind == PrimitiveKind.HOLD
                                    else params.hold_duration_s
                                ),
                            }
                        ),
                    )
                    for kind, hand_shape, duration in phases
                ],
                assertions=[
                    AssertionSpec(name="free_body_pickup"),
                    AssertionSpec(name="opposing_contacts"),
                    AssertionSpec(name="fixed_root"),
                ],
            )
            return PlannerOutcome(program=program, provider="offline", model="rule-planner-v1", model_calls=0)
        return self._unsupported_result(text, "no supported gesture or pickup intent", hand)

    @staticmethod
    def _resolve_object(text: str, scene: SceneManifest) -> str | None:
        mentions = [item.id for item in scene.objects if re.search(rf"\b{re.escape(item.id)}\b", text)]
        if len(mentions) == 1:
            return mentions[0]
        block_mentions = bool(re.search(r"\b(block|cube|box|object in front)\b", text))
        blocks = [item.id for item in scene.objects if item.kind == "block"]
        if block_mentions and len(blocks) == 1:
            return blocks[0]
        if len(scene.objects) == 1 and re.search(r"\b(it|object|thing)\b", text):
            return scene.objects[0].id
        return None

    @staticmethod
    def _unsupported_result(text: str, reason: str, hand: Hand = Hand.RIGHT) -> PlannerOutcome:
        return PlannerOutcome(
            program=MotionProgram(
                source_text=text,
                intent=Intent.UNSUPPORTED,
                hand=hand,
                unsupported_reason=reason,
            ),
            provider="offline",
            model="rule-planner-v1",
            model_calls=0,
        )


class OpenAIPlanner:
    def __init__(self, client: OpenAI | None = None) -> None:
        load_environment()
        self.client = client or OpenAI()

    def plan(self, request: PlanRequest) -> PlannerOutcome:
        # Deterministic logical contradictions are authoritative and cost no
        # model call. Capability classification is left to the structured
        # planner so unfamiliar upper-body movement can use composition.
        preflight = OfflinePlanner().plan(
            request.model_copy(update={"provider": "offline"})
        )
        preflight_reason = preflight.program.unsupported_reason or ""
        if preflight.program.intent == Intent.UNSUPPORTED and preflight_reason.startswith(
            (
                "contradictory",
                "scene does not contain",
                "object action",
                "concurrent object manipulation",
            )
        ):
            return preflight
        if preflight.program.intent == Intent.SEQUENCE:
            return preflight
        if preflight.program.intent == Intent.FULL_BODY and any(
            str(primitive.label or "").startswith(
                (
                    "burpee_",
                    "squat_",
                    "lunge_",
                    "single_leg_balance_",
                    "sit_up_",
                    "floor_roll_",
                    "cartwheel_",
                    "airborne_",
                    "obstacle_",
                    "dance_",
                    "climb_",
                )
            )
            for primitive in preflight.program.primitives
        ):
            # Calibrated exercise/support graphs rely on exact mid-program
            # state transitions. Keep them deterministic instead of asking
            # the general selector to rediscover or flatten the lifecycle.
            return preflight
        if preflight.program.intent == Intent.GRAB and re.search(
            r"\b(?:overhead|over (?:your |the )?head|above (?:your |the )?head)\b",
            request.text,
            re.I,
        ):
            return preflight
        calls = 0
        prompt = self._prompt(request)
        primary_model = os.getenv("OPENAI_PLANNER_MODEL", DEFAULT_PRIMARY_MODEL)
        repair_model = os.getenv("OPENAI_REPAIR_MODEL", DEFAULT_REPAIR_MODEL)
        try:
            calls += 1
            response = self.client.responses.parse(
                model=primary_model,
                input=prompt,
                text_format=PlannerSelection,
            )
            selection = response.output_parsed
            if selection is None:
                raise ValueError("provider returned no parsed planner selection")
            program = self._expand(selection, request, self._response_seed(response))
            self._validate_semantics(program, request.scene)
            return PlannerOutcome(
                program=program,
                provider="openai",
                model=primary_model,
                model_calls=calls,
                response_ids=(str(response.id),),
            )
        except Exception as primary_error:
            primary_response_id = str(getattr(locals().get("response"), "id", ""))
            calls += 1
            response = self.client.responses.parse(
                model=repair_model,
                input=f"{prompt}\n\nThe first attempt failed validation: {type(primary_error).__name__}: {primary_error}. Repair it once.",
                text_format=PlannerSelection,
            )
            selection = response.output_parsed
            if selection is None:
                raise ValueError("repair returned no parsed planner selection") from primary_error
            program = self._expand(selection, request, self._response_seed(response))
            self._validate_semantics(program, request.scene)
            response_ids = tuple(
                value for value in (primary_response_id, str(response.id)) if value
            )
            return PlannerOutcome(
                program=program,
                provider="openai",
                model=repair_model,
                model_calls=calls,
                response_ids=response_ids,
            )

    @staticmethod
    def _prompt(request: PlanRequest) -> str:
        catalog = (
            "Select intent, hand, hand_shape, strike_type, exact scene object id, object_action/object_motion, composition_segments, or body_segments "
            "when needed. Supported intents are gesture, strike, single-block grab, composite upper-body movement, "
            "full_body procedural movement, object_interaction for physical throws, catches, pushes, pulls, support-plane rolls, spins, or placements, drops, or explicit hand-to-hand object transfers, and deterministic ordered sequences. "
            "Supported strikes are hook, jab, "
            "cross, and uppercut, each expanded into guard, load, strike, follow-through, and recovery. "
            "A grab is the complete reach, contact, close, lift, and "
            "hold pipeline, so requests to pick up, raise, lift, or hold the grabbed block aloft are supported "
            "single-block grabs, not multi-action sequences. Shapes: open, fist, point, pinch, hang_ten, thumbs_up, peace. "
            "The idiom 'throw up a hang-ten/shaka sign' means present that hand gesture; it is not throwing. "
            "For a physical throw/catch, select the exact object, throw or catch action, style, avatar-relative horizontal "
            "direction (+X left, -X right, +Z forward), travel distance, height above the endpoints at the apex, optional "
            "spin turns, hand-contact height/depth, and final landing height. Throw/catch uses a free-object lifecycle with "
            "contact, attachment, release/flight or intercept/absorption; do not encode it as a generic arm gesture. "
            "Push and pull use continuous guided contact rather than flight: choose push with a forward/away direction and pull "
            "with a toward-avatar direction, plus the requested guided distance. "
            "Roll uses the same continuous guided contact while the object rotates by travel distance divided by its physical radius. "
            "Spin keeps the object center fixed on its support and applies the requested yaw turns under continuous contact. "
            "Place secures and lifts the object, translates it above the same support surface, lowers it to support, then opens the hand before recovery. "
            "Drop secures and lifts the object, opens the hand, and lets gravity carry it vertically to its support height. "
            "A handoff requires distinct source and receiver hands, dual contact before ownership transfer, source release, and receiver retention. "
            "A carry is deterministically decomposed into a secure pickup followed by locomotion with a persistent hand attachment. "
            "A gesture may include a swift presentation, an emphasized finger shape, a repeated shake/back-and-forth "
            "forearm oscillation, a hold, and return to default; these are modifiers inside one supported gesture, "
            "not a multi-action sequence. Use composite for any other arm, hand, forearm, shoulder, or torso motion, "
            "including coordinated bilateral movement and multi-phase upper-body sequences. Composite segments run "
            "sequentially, while all effectors inside one segment move concurrently. Use normalized body coordinates: "
            "target_x -1 is the avatar's right and +1 is the avatar's left; target_y -1 is low and +1 high; "
            "target_z -1 is close and +1 extended. "
            "Set reach_overhead=true only when a hand must clear the head; ordinary face-, salute-, and chest-height targets keep it false. "
            "Use linear for direct repositioning, arc for a curved transition, circle for repeated circular paths, "
            "oscillate for back-and-forth motion, and hold for a stationary beat. For opposed bilateral cycles, give "
            "the two effectors phase offsets 0 and 0.5. Use frontal plane for X/Y paths, sagittal for Y/Z paths, and "
            "horizontal for X/Z paths. Use two to eight readable segments; recovery is appended deterministically. "
            "Use axial_rotation_amplitude when the wording asks a forearm to roll, pronate, supinate, or rotate about "
            "its own long axis; use about 0.5-0.75 for a clearly visible repeated roll and zero when no axial roll is requested. "
            "Use full_body for locomotion, turns, crouches, jumps, kicks, continuous body rotations, and grounded postures. Decompose sequences into "
            "body_segments using hold/step/walk/run/turn/crouch/jump/kick/rotate/pose. Rotate is a continuous root rotation: choose pitch for forward/backward flips or rolls, roll for cartwheels or side flips, and yaw for airborne spins; "
            "choose floor, cartwheel, or airborne support mode, set signed rotation_degrees, and provide direction/distance for rotations that travel. Use pose for bends, leans, bows, kneels, "
            "seated postures, face-up supine, face-down prone, or lateral side-lying poses, palm-and-knee all-fours and crawling poses, palm-and-toe push-up cycles, and unfamiliar grounded joint configurations. "
            "Burpees use a calibrated crouch, plank/push-up, foot-support recovery, takeoff crouch, jump, and landing graph and are handled deterministically. "
            "For a jump whose wording requires raised hands, set raise_arms_overhead=true so the arm peak is synchronized to the airborne apex. "
            "For a lie-down pose rotate pelvis pitch to about +/-90 degrees, lower the root, and set lock_feet=false so support "
            "can transfer from the feet to the back or front, with support_mode=broad_floor. For all fours, use support_mode=quadruped, "
            "lock_feet=false, a near-horizontal pelvis, deeply flexed knees, and two low forward open-palm effectors. For push-ups, "
            "use support_mode=plank, straight legs, two low forward open-palm effectors, cycles for the requested repetitions, and height_m for lowering depth. Its pose object directly controls root drop/shift, "
            "For crawling, use quadruped support with a bounded root shift and opposed horizontal hand oscillations; cycles is the requested crawl steps. "
            "pelvis, torso, and head pitch/yaw/roll, bilateral hip pitch/roll, knee flexion, ankle pitch, and whether the feet stay "
            "locked to their world contacts. left/right_foot_shift_x/z_m author a staggered grounded stance with sequential "
            "support transfers; left/right_foot_lift_m raises one foot while the opposite foot stays planted for single-leg balance or knee-raise poses. "
            "Angles are degrees; keep lock_feet=true for standing, kneeling, and seated poses unless the feet "
            "must intentionally leave their contacts. A balanced return to standing is appended automatically: do not author an "
            "all-zero pose just to represent the requested return/default/neutral phase. direction_x +1 is avatar-left and -1 avatar-right; "
            "direction_z +1 is forward and -1 backward. distance_m is total root travel, turn_degrees is signed yaw "
            "(positive left/counterclockwise). For scene-aware locomotion, set obstacle_mode=over only on a single step over one exact obstacle_object_id and give the requested vertical obstacle_clearance_m; "
            "set obstacle_mode=around on walk/run with the exact object id and a signed path_lateral_offset_m of at least 0.1 m to choose the detour side. Never invent an object id. "
            "height_m controls jump or crouch magnitude, cycles is the requested steps or "
            "repetitions, and lead_side selects the stepping/kicking side. For a static look, set head angles and cycles=0; "
            "for repeated nods or head shakes, author only head pose controls and set cycles to the requested repetitions. "
            "A body_segment may also carry zero to two effectors "
            "plus trajectory/plane/amplitude/cycles/axial rotation, using the same normalized arm-target semantics as composite; "
            "use these when an arm or hand action happens concurrently with walking or another body action (for example walking "
            "while waving). Do not drop either the body action or the concurrent effectors. Return unsupported only when the request requires "
            "a body skill outside that library, an unknown object, or contradictory instructions. Default an "
            "unspecified hand to right. For every gesture, populate motion_profile from the requested words. "
            "Allowed profile values: style relaxed/neutral/energetic/precise/playful; timing quick/balanced/slow; "
            "height low/chest/high; depth close/natural/extended; lateral inward/center/outward; wrist_pitch "
            "down/neutral/up; wrist_yaw inward/neutral/outward; wrist_roll counterclockwise/neutral/clockwise. "
            "Use neutral, balanced, chest, natural, or center only when that aspect is unspecified. Do not invent ids."
        )
        scene = request.scene.model_dump_json()
        return f"{catalog}\nSCENE={scene}\nREQUEST={request.text}"

    @staticmethod
    def _response_seed(response: object) -> int:
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            digest = hashlib.sha256(response_id.encode("utf-8")).digest()
            return int.from_bytes(digest[:4], "big") & (2**31 - 1)
        return secrets.randbelow(2**31 - 1) + 1

    @staticmethod
    def _expand(
        selection: PlannerSelection,
        request: PlanRequest,
        variation_seed: int = 0,
    ) -> MotionProgram:
        known = OfflinePlanner().plan(
            request.model_copy(update={"provider": "offline"})
        ).program
        if known.intent == Intent.SEQUENCE:
            known.seed = variation_seed
            return known
        if selection.intent == Intent.COMPOSITE:
            if known.intent == Intent.COMPOSITE:
                # Recognized compositions are executable motion knowledge, not
                # merely a classification hint.  Prefer their calibrated
                # geometry and use free-form model segments only for language
                # the local decomposer does not understand.
                known.seed = variation_seed
                return known
            return _composite_program(
                request.text,
                selection.composition_segments,
                seed=variation_seed,
            )
        if selection.intent == Intent.FULL_BODY:
            if known.intent == Intent.FULL_BODY:
                known.seed = variation_seed
                return known
            return _full_body_program(
                request.text,
                selection.body_segments,
                seed=variation_seed,
            )
        if selection.intent == Intent.OBJECT_INTERACTION:
            if known.intent == Intent.OBJECT_INTERACTION:
                known.seed = variation_seed
                return known
            object_id = selection.object_id or OfflinePlanner._resolve_object(
                request.text.lower(), request.scene
            )
            if object_id is None or object_id not in {item.id for item in request.scene.objects}:
                raise ValueError("object interaction requires an exact scene object id")
            if selection.object_action is None or selection.object_motion is None:
                raise ValueError("object interaction requires action and motion target")
            return _object_interaction_program(
                request.text,
                selection.object_action,
                selection.hand,
                object_id,
                selection.object_motion,
                seed=variation_seed,
            )
        if selection.intent == Intent.UNSUPPORTED:
            # The structured model selects semantic categories, while the
            # deterministic parser is the executable capability authority.
            # Do not let a stale model intuition reject a phrase that maps
            # unambiguously to an implemented gesture or pickup pipeline.
            if known.intent != Intent.UNSUPPORTED:
                known.seed = variation_seed
                return known
            return MotionProgram(
                source_text=request.text,
                intent=Intent.UNSUPPORTED,
                hand=selection.hand,
                unsupported_reason=selection.unsupported_reason or "unsupported request",
            )
        if known.intent != Intent.UNSUPPORTED and known.intent != selection.intent:
            known.seed = variation_seed
            return known
        if selection.intent == Intent.GESTURE:
            if known.intent == Intent.UNSUPPORTED and (
                (known.unsupported_reason or "").startswith("unsupported motion:")
                or (known.unsupported_reason or "").startswith("contradictory")
            ):
                known.seed = variation_seed
                return known
            if selection.hand_shape is None:
                raise ValueError("gesture selection requires hand_shape")
            # Numeric staging is derived from explicit words by audited local
            # code. The model chooses the semantic category but cannot invent
            # an unrequested high pose, wrist direction, or expressive style.
            profile = OfflinePlanner._motion_profile(request.text)
            return MotionProgram(
                source_text=request.text,
                intent=Intent.GESTURE,
                hand=selection.hand,
                motion_profile=profile,
                seed=variation_seed,
                primitives=_gesture_primitives(
                    request.text,
                    selection.hand_shape,
                    profile,
                    selection.hand,
                    variation_seed,
                ),
                assertions=[AssertionSpec(name="finger_assertions"), AssertionSpec(name="fixed_root")],
            )
        if selection.intent == Intent.STRIKE:
            strike_type = selection.strike_type
            if known.intent == Intent.STRIKE:
                strike_type = known.strike_type
            if strike_type is None:
                raise ValueError("strike selection requires strike_type")
            profile = OfflinePlanner._motion_profile(request.text)
            return MotionProgram(
                source_text=request.text,
                intent=Intent.STRIKE,
                hand=selection.hand,
                strike_type=strike_type,
                motion_profile=profile,
                seed=variation_seed,
                primitives=_strike_primitives(request.text, strike_type, selection.hand),
                assertions=[
                    AssertionSpec(name="closed_fist_at_impact"),
                    AssertionSpec(name="curved_or_direct_strike_path"),
                    AssertionSpec(name="fixed_root"),
                ],
            )
        if selection.intent != Intent.GRAB:
            raise ValueError(f"unhandled planner intent: {selection.intent}")
        object_id = selection.object_id or OfflinePlanner._resolve_object(request.text.lower(), request.scene)
        if object_id is None or object_id not in {item.id for item in request.scene.objects}:
            raise ValueError("grab selection requires an exact scene object id")
        base = PrimitiveParameters(arm_depth=0.50, elbow_swivel=0.20)
        phase_specs = (
            (PrimitiveKind.REACH, HandShape.OPEN, {"duration_s": 0.65}),
            (PrimitiveKind.PRESHAPE, HandShape.PINCH, {"duration_s": 0.30, "finger_curl": 0.25, "thumb_opposition": 0.65}),
            (PrimitiveKind.CONTACT, HandShape.PINCH, {"duration_s": 0.20, "finger_curl": 0.35, "thumb_opposition": 0.80, "grip_force": 0.20}),
            (PrimitiveKind.CLOSE, HandShape.FIST, {"duration_s": 0.30, "finger_curl": 0.55, "thumb_opposition": 0.90, "grip_force": 0.65}),
            (PrimitiveKind.LIFT, HandShape.FIST, {"duration_s": 0.65, "finger_curl": 0.55, "thumb_opposition": 0.90, "grip_force": 0.65, "lift_height_m": 0.12}),
            (PrimitiveKind.HOLD, HandShape.FIST, {"duration_s": 1.20, "finger_curl": 0.55, "thumb_opposition": 0.90, "grip_force": 0.65, "lift_height_m": 0.12, "hold_duration_s": 1.20}),
            (PrimitiveKind.RECOVER, HandShape.FIST, {"duration_s": 0.60, "finger_curl": 0.55, "thumb_opposition": 0.90, "grip_force": 0.65, "lift_height_m": 0.12}),
        )
        return MotionProgram(
            source_text=request.text,
            intent=Intent.GRAB,
            hand=selection.hand,
            seed=variation_seed,
            primitives=[
                MotionPrimitive(
                    kind=kind,
                    hand_shape=shape,
                    object_id=object_id,
                    socket_id="front_center",
                    parameters=base.model_copy(update=updates),
                )
                for kind, shape, updates in phase_specs
            ],
            assertions=[
                AssertionSpec(name="free_body_pickup"),
                AssertionSpec(name="opposing_contacts"),
                AssertionSpec(name="fixed_root"),
            ],
        )

    @staticmethod
    def _validate_semantics(program: MotionProgram, scene: SceneManifest) -> None:
        if program.intent == Intent.SEQUENCE:
            if len(program.steps) < 2:
                raise ValueError("ordered sequences require at least two executable steps")
            for step in program.steps:
                OpenAIPlanner._validate_semantics(step, scene)
            return
        ids = {item.id for item in scene.objects}
        for primitive in program.primitives:
            if primitive.object_id is not None and primitive.object_id not in ids:
                raise ValueError(f"unknown object id: {primitive.object_id}")
        if program.intent == Intent.COMPOSITE:
            if len(program.primitives) > 13:
                raise ValueError("composite program exceeds the bounded segment count")
            if program.primitives[-1].kind != PrimitiveKind.RECOVER:
                raise ValueError("composite program must end in recovery")
            if any(
                primitive.kind in {PrimitiveKind.MOVE, PrimitiveKind.CYCLE}
                and not primitive.effectors
                for primitive in program.primitives
            ):
                raise ValueError("every composite movement segment requires effectors")
        if program.intent == Intent.FULL_BODY:
            if len(program.primitives) > 13:
                raise ValueError("full-body program exceeds the bounded segment count")
            if program.primitives[-1].kind != PrimitiveKind.RECOVER:
                raise ValueError("full-body program must end in a balanced recovery")
            if any(
                primitive.kind == PrimitiveKind.BODY and primitive.body is None
                for primitive in program.primitives
            ):
                raise ValueError("every full-body phase requires a body target")
        if program.intent == Intent.GRAB:
            expected = [
                PrimitiveKind.REACH,
                PrimitiveKind.PRESHAPE,
                PrimitiveKind.CONTACT,
                PrimitiveKind.CLOSE,
                PrimitiveKind.LIFT,
                PrimitiveKind.HOLD,
                PrimitiveKind.RECOVER,
            ]
            if [item.kind for item in program.primitives] != expected:
                raise ValueError("pickup primitives must use the canonical phase order")
            phases = {item.kind: item.parameters for item in program.primitives}
            if phases[PrimitiveKind.CLOSE].grip_force < 0.5:
                raise ValueError("close grip_force must be at least 0.5")
            if phases[PrimitiveKind.LIFT].lift_height_m < 0.10:
                raise ValueError("lift height must be at least 0.10 m")
            hold = phases[PrimitiveKind.HOLD]
            if hold.hold_duration_s < 1.0 or hold.duration_s < 1.0:
                raise ValueError("hold phase and physical hold_duration_s must both be at least 1.0 s")
        if program.intent == Intent.OBJECT_INTERACTION:
            expected_by_action = {
                ObjectAction.THROW: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.LIFT,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.WINDUP,
                    PrimitiveKind.RELEASE,
                    PrimitiveKind.FLIGHT,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.CATCH: [
                    PrimitiveKind.RECEIVE,
                    PrimitiveKind.FLIGHT,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.ABSORB,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.PUSH: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.MOVE,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.ROLL: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.MOVE,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.SPIN: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.MOVE,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.PLACE: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.LIFT,
                    PrimitiveKind.MOVE,
                    PrimitiveKind.RELEASE,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.PULL: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.MOVE,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.DROP: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.LIFT,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RELEASE,
                    PrimitiveKind.FLIGHT,
                    PrimitiveKind.RECOVER,
                ],
                ObjectAction.HANDOFF: [
                    PrimitiveKind.REACH,
                    PrimitiveKind.PRESHAPE,
                    PrimitiveKind.CONTACT,
                    PrimitiveKind.CLOSE,
                    PrimitiveKind.LIFT,
                    PrimitiveKind.RECEIVE,
                    PrimitiveKind.RELEASE,
                    PrimitiveKind.HOLD,
                    PrimitiveKind.RECOVER,
                ],
            }
            expected = expected_by_action.get(program.object_action)
            if expected is None:
                raise ValueError("object interaction action has no lifecycle")
            if [item.kind for item in program.primitives] != expected:
                raise ValueError("object interaction primitives do not match the action lifecycle")
            if any(item.object_id is None for item in program.primitives):
                raise ValueError("every object interaction phase requires an object id")
            if program.object_action == ObjectAction.THROW:
                phases = {item.kind: item.parameters for item in program.primitives}
                if phases[PrimitiveKind.CLOSE].grip_force < 0.5:
                    raise ValueError("throw must secure the object before windup")
                if phases[PrimitiveKind.RELEASE].grip_force > 0.05:
                    raise ValueError("throw release must open the grip")
            if program.object_action == ObjectAction.PULL:
                phases = {item.kind: item.parameters for item in program.primitives}
                if phases[PrimitiveKind.CLOSE].grip_force < 0.5:
                    raise ValueError("pull must secure the object before guided motion")
            if program.object_action == ObjectAction.DROP:
                phases = {item.kind: item.parameters for item in program.primitives}
                if phases[PrimitiveKind.CLOSE].grip_force < 0.5:
                    raise ValueError("drop must secure the object before release")
                if phases[PrimitiveKind.RELEASE].grip_force > 0.05:
                    raise ValueError("drop release must open the grip")
            if program.object_action == ObjectAction.PLACE:
                phases = {item.kind: item.parameters for item in program.primitives}
                if phases[PrimitiveKind.CLOSE].grip_force < 0.5:
                    raise ValueError("place must secure the object before transport")
                if phases[PrimitiveKind.RELEASE].grip_force > 0.05:
                    raise ValueError("place release must open the grip")
        if program.intent == Intent.STRIKE:
            expected = [
                PrimitiveKind.GUARD,
                PrimitiveKind.LOAD,
                PrimitiveKind.STRIKE,
                PrimitiveKind.FOLLOW_THROUGH,
                PrimitiveKind.RECOVER,
            ]
            if [item.kind for item in program.primitives] != expected:
                raise ValueError("strike primitives must use the canonical phase order")
            if any(
                item.hand_shape != HandShape.FIST
                for item in program.primitives
                if item.kind != PrimitiveKind.RECOVER
            ):
                raise ValueError("strike phases require a closed fist")


def plan_motion(request: PlanRequest) -> PlannerOutcome:
    status = provider_status()
    if request.provider == "offline":
        return OfflinePlanner().plan(request)
    if request.provider == "openai" or (request.provider == "auto" and status["openai_available"]):
        return OpenAIPlanner().plan(request)
    return OfflinePlanner().plan(request)
