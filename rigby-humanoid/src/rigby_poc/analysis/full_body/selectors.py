"""Which primitives gate which full-body metric block, computed once.

``_compile_full_body`` decides what to measure by filtering ``program.primitives``
— sometimes on ``BodyAction``, more often on the *label* the planner assigned or
on ``pose.support_mode``. Squats are ``CROUCH`` primitives labelled ``squat_*``;
lunges, sit-ups, crawls, push-ups and single-leg balances are all ``POSE``. That
is why the metric blocks do not map one-to-one onto ``BodyAction`` members: five
of the twelve members gate no block of their own and feed only the shared root,
ground, support and balance measurements.

Every selector here is a pure function of the program, so the analysis layer
reproduces them exactly rather than needing them carried over from compilation.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from ...models import (
    BodyAction,
    BodyObstacleMode,
    BodySupportMode,
    ClipFrame,
    Hand,
    MotionProgram,
    MotionPrimitive,
    SceneManifest,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..context import AnalysisContext


class FullBodyPass:
    """One full-body analysis pass: the clip, its program, and the selectors.

    Holds exactly the derived values ``_compile_full_body`` had in scope when it
    ran its measurement block, so each moved block reads the same names it read
    inside the compiler.
    """

    def __init__(self, ctx: AnalysisContext) -> None:
        self.ctx = ctx

    # --- clip and program -------------------------------------------------

    @property
    def frames(self) -> list[ClipFrame]:
        return self.ctx.frames

    @property
    def program(self) -> MotionProgram:
        return self.ctx.program

    @property
    def scene(self) -> SceneManifest:
        return self.ctx.scene

    @property
    def phase_ranges(self) -> list[dict[str, float | str]]:
        return self.ctx.phase_ranges

    @property
    def world_positions(self) -> list[dict[str, np.ndarray]]:
        return self.ctx.world_positions

    @property
    def ground_height(self) -> float:
        return self.ctx.ground_height

    @property
    def neutral_ground_height(self) -> float:
        """The ground height the *generation* pass used.

        ``_compile_full_body`` derives this twice by two different routes: once
        from the calibrated idle pose with the hips pinned to the origin, and
        once from the plain identity pose. They agree bit-for-bit because the
        idle pose leaves every leg bone and the hips at identity — it only moves
        the arms and hands. ``tests/test_full_body_analysis.py`` pins that, so a
        future idle pose that rotates a leg fails loudly instead of silently
        shifting every clearance metric.
        """

        return self.ctx.ground_height

    @property
    def toe_clearances(self) -> dict[str, np.ndarray]:
        return self.ctx.toe_clearances

    @cached_property
    def hips_positions(self) -> np.ndarray:
        return np.asarray(
            [frame.bones["hips"].position.as_list() for frame in self.frames],
            dtype=float,
        )

    # --- carried-over authoring intent ------------------------------------

    @property
    def support_constraints(self) -> list[dict[str, Any]]:
        """Commanded ankle position per constrained frame.

        Persisted by the compiler because it is authoring intent: the analyzer
        can see where the foot *went*, never where it was *told* to go.
        """

        return self._checked_constraints("support_constraints")

    @property
    def climb_support_constraints(self) -> list[dict[str, Any]]:
        """Commanded per-limb support targets while climbing. Same reasoning."""

        return self._checked_constraints("climb_support_constraints")

    def _checked_constraints(self, key: str) -> list[dict[str, Any]]:
        """Read a carried constraint record, refusing one that does not fit.

        Every record names the frame it constrains. A record pointing past the
        end of the clip means the carried metrics came from a different
        compilation than these frames — the mutation and corpus work in plans 03
        and 06 will hand this pass clips and metrics from separate sources, and
        a bare IndexError three call frames down would send whoever hits it
        looking in the wrong place.
        """

        value = self.ctx.compiled_metric(key, [])
        records = list(value) if isinstance(value, list) else []
        frame_count = len(self.frames)
        for record in records:
            index = int(record["frame_index"])
            if not 0 <= index < frame_count:
                raise ValueError(
                    f"{key} references frame {index} of a {frame_count}-frame "
                    "clip; the carried metrics do not belong to these frames"
                )
        return records

    # --- primitive selectors ----------------------------------------------

    @cached_property
    def rotation_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.ROTATE
        ]

    @cached_property
    def dance_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.DANCE
        ]

    @cached_property
    def climb_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.CLIMB
        ]

    @cached_property
    def terminal_climb(self) -> bool:
        return bool(
            self.program.primitives
            and self.program.primitives[-1].body is not None
            and self.program.primitives[-1].body.action == BodyAction.CLIMB
        )

    @cached_property
    def obstacle_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.obstacle_mode != BodyObstacleMode.NONE
        ]

    @cached_property
    def jumping_jack_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.JUMP
            and (
                abs(primitive.body.pose.left_foot_shift_x_m)
                + abs(primitive.body.pose.right_foot_shift_x_m)
                > 1e-8
            )
        ]

    @cached_property
    def burpee_jump_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.JUMP
            and str(primitive.label or "").startswith("burpee_")
        ]

    @cached_property
    def burpee_floor_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and primitive.body.pose.support_mode == BodySupportMode.PLANK
            and str(primitive.label or "").startswith("burpee_")
        ]

    @cached_property
    def squat_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.CROUCH
            and str(primitive.label or "").startswith("squat_")
        ]

    @cached_property
    def lunge_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and str(primitive.label or "").startswith("lunge_")
            and not str(primitive.label or "").endswith("_stance_reset")
        ]

    @cached_property
    def single_leg_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and str(primitive.label or "").startswith("single_leg_balance_")
        ]

    @cached_property
    def sit_up_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and str(primitive.label or "").startswith("sit_up_")
            and str(primitive.label or "").endswith("_curl")
        ]

    @cached_property
    def crawl_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and primitive.body.pose.support_mode == BodySupportMode.QUADRUPED
            and (
                abs(primitive.body.pose.root_shift_x_m)
                + abs(primitive.body.pose.root_shift_z_m)
                > 1e-8
            )
        ]

    @cached_property
    def push_up_primitives(self) -> list[MotionPrimitive]:
        return [
            primitive
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and primitive.body.pose.support_mode == BodySupportMode.PLANK
        ]

    @cached_property
    def horizontal_pose_requested(self) -> bool:
        return any(
            primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and (
                primitive.body.pose.support_mode != BodySupportMode.FEET
                or (
                    not primitive.body.pose.lock_feet
                    and abs(primitive.body.pose.pelvis_pitch_deg) >= 80.0
                )
            )
            for primitive in self.program.primitives
        )

    @cached_property
    def horizontal_pose_labels(self) -> set[str]:
        return {
            primitive.label or BodyAction.POSE.value
            for primitive in self.program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and (
                primitive.body.pose.support_mode != BodySupportMode.FEET
                or (
                    not primitive.body.pose.lock_feet
                    and abs(primitive.body.pose.pelvis_pitch_deg) >= 80.0
                )
            )
        }


SIDES = (Hand.LEFT.value, Hand.RIGHT.value)


__all__ = ["SIDES", "FullBodyPass"]
