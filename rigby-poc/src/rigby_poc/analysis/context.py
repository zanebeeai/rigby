"""Everything expensive an analysis pass derives, computed once.

Checks read the context rather than recomputing forward kinematics per check.
Every derivation here is lazy, so a check that never asks for world positions
never pays for them.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np

from ..kinematics import RigKinematics, rig_kinematics
from ..models import (
    BonePose,
    ClipFrame,
    ClipResult,
    Hand,
    Intent,
    MotionProgram,
    ObjectAction,
    PrimitiveKind,
    SceneManifest,
    Transform,
)
from .rig import rig_profile


# Phase kinds whose interval counts as an active presentation, per intent.
# ``compile_motion`` appends exactly these while it emits frames; reading them
# back off ``phase_ranges_s`` reproduces the same list.
_PRESENTATION_KINDS: dict[Intent, frozenset[str]] = {
    Intent.GESTURE: frozenset(
        {PrimitiveKind.PRESENT.value, PrimitiveKind.HOLD.value, PrimitiveKind.SHAKE.value}
    ),
    Intent.STRIKE: frozenset(
        {
            PrimitiveKind.GUARD.value,
            PrimitiveKind.LOAD.value,
            PrimitiveKind.STRIKE.value,
            PrimitiveKind.FOLLOW_THROUGH.value,
        }
    ),
}

_TRAVEL_SETUP_LABEL = "parallel_forearm_travel_setup"


class AnalysisContext:
    """A finished clip plus its effective program and scene.

    Construct it with :meth:`from_clip`. The clip must be the *effective* one —
    that is, compiled from the post-override program. Stored artifacts persist
    the pre-override program (``store.py`` writes ``request.program``), so an
    analyzer reading from disk must go through
    :func:`rigby_poc.analysis.artifacts.load_analysis_inputs`, which applies the
    overrides first.
    """

    def __init__(
        self,
        frames: list[ClipFrame],
        program: MotionProgram,
        scene: SceneManifest,
        carried_metrics: Mapping[str, Any] | None = None,
        *,
        kinematics: RigKinematics | None = None,
    ) -> None:
        self.frames = list(frames)
        self.program = program
        self.scene = scene
        self.carried_metrics: Mapping[str, Any] = carried_metrics or {}
        self.kinematics = kinematics or rig_kinematics()

    @classmethod
    def from_clip(
        cls,
        clip: ClipResult,
        program: MotionProgram,
        scene: SceneManifest,
        *,
        kinematics: RigKinematics | None = None,
    ) -> "AnalysisContext":
        return cls(
            clip.frames, program, scene, clip.metrics, kinematics=kinematics
        )

    @classmethod
    def from_frames(
        cls,
        frames: list[ClipFrame],
        program: MotionProgram,
        scene: SceneManifest,
        carried_metrics: Mapping[str, Any],
        *,
        kinematics: RigKinematics | None = None,
    ) -> "AnalysisContext":
        """Build a context before a :class:`ClipResult` exists.

        ``compile_motion`` needs this: it has the frames and the carry-over
        metrics in hand but has not assembled the result yet, and the whole
        point of the extraction is that the compiler and a post-hoc analyzer run
        the *same* code rather than two implementations that agree by
        inspection.
        """

        return cls(
            frames, program, scene, carried_metrics, kinematics=kinematics
        )

    @property
    def intent(self) -> Intent:
        return self.program.intent

    @property
    def object_action(self) -> ObjectAction | None:
        return self.program.object_action

    @property
    def allow_root_motion(self) -> bool:
        """Whether the clip is permitted to translate its root.

        Matches the ``allow_root_motion`` argument each compile path passes to
        the safety metrics: only the whole-body and sequence paths enable it.
        """

        return self.intent in {Intent.FULL_BODY, Intent.SEQUENCE}

    @cached_property
    def phase_ranges(self) -> list[dict[str, float | str]]:
        """Authored phase intervals, read from the clip, never re-derived.

        Phase timing is not ``primitive.parameters.duration_s``: the compiler
        inflates each phase's frame count with per-action rules before writing
        ``phase_ranges_s``. Re-deriving it from the program would silently
        disagree with the clip.
        """

        value = self.carried_metrics.get("phase_ranges_s")
        return list(value) if isinstance(value, list) else []

    def ranges_for_kind(self, kind: str) -> list[tuple[float, float]]:
        return [
            (float(item["start_s"]), float(item["end_s"]))
            for item in self.phase_ranges
            if item.get("kind") == kind
        ]

    def ranges_for_label(self, label: str) -> list[tuple[float, float]]:
        return [
            (float(item["start_s"]), float(item["end_s"]))
            for item in self.phase_ranges
            if item.get("label") == label
        ]

    @cached_property
    def presentation_ranges(self) -> list[tuple[float, float]]:
        """Intervals over which the active hand must read as presented.

        Reproduces what each compile path appends while emitting frames. The
        composite path starts its window part-way into the phase so a relaxed
        rest pose is not judged as a failed presentation.
        """

        if self.intent == Intent.COMPOSITE:
            # Read, never re-derived. The composite window opens at a fraction
            # of each phase's *authored* duration, and ``phase_ranges_s`` cannot
            # supply that: its bounds are running sums, so ``end_s - start_s``
            # differs from the authored duration by whatever rounding the
            # running total has accumulated. Measured on the finger-count case,
            # re-deriving moved two of nine windows by one ulp -- small, and
            # more than enough to move every metric computed over them.
            #
            # Plan 02 §1.5 called this "reconstructible from kind and label".
            # It is reconstructible approximately, which for this layer is the
            # same as not reconstructible.
            persisted = self.carried_metrics.get("presentation_ranges_s")
            if persisted is None:
                raise ValueError(
                    "composite clips carry presentation_ranges_s; this clip has "
                    "none, so it was compiled before 02c and cannot be analysed "
                    "without re-deriving a window that would not match"
                )
            return [(float(start), float(end)) for start, end in persisted]
        kinds = _PRESENTATION_KINDS.get(self.intent)
        if kinds is None:
            return []
        return [
            (float(item["start_s"]), float(item["end_s"]))
            for item in self.phase_ranges
            if item.get("kind") in kinds
        ]

    @cached_property
    def world_positions(self) -> list[dict[str, np.ndarray]]:
        return [
            self.kinematics.canonical_positions(frame.bones) for frame in self.frames
        ]

    @cached_property
    def ground_height(self) -> float:
        neutral_bones = {name: BonePose() for name in rig_profile()["bone_map"]}
        neutral_positions = self.kinematics.canonical_positions(neutral_bones)
        return min(
            float(neutral_positions["leftToes"][1]),
            float(neutral_positions["rightToes"][1]),
        )

    @cached_property
    def toe_clearances(self) -> dict[str, np.ndarray]:
        ground_height = self.ground_height
        return {
            side: np.asarray(
                [
                    float(position[f"{side}Toes"][1]) - ground_height
                    for position in self.world_positions
                ],
                dtype=float,
            )
            for side in (Hand.LEFT.value, Hand.RIGHT.value)
        }

    @cached_property
    def object_tracks(self) -> dict[str, list[Transform]]:
        """Per-object transform history, for every object present in every frame."""

        object_ids = sorted({key for frame in self.frames for key in frame.objects})
        return {
            object_id: [
                frame.objects[object_id]
                for frame in self.frames
                if object_id in frame.objects
            ]
            for object_id in object_ids
        }

    def frames_in(self, ranges: list[tuple[float, float]]) -> list[ClipFrame]:
        return [
            frame
            for frame in self.frames
            if any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in ranges)
        ]

    def indices_in(self, ranges: list[tuple[float, float]]) -> list[int]:
        return [
            index
            for index, frame in enumerate(self.frames)
            if any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in ranges)
        ]

    def compiled_metric(self, key: str, default: Any = None) -> Any:
        """Read a metric the analysis layer does not own yet.

        Every use is a pointer at work still living in ``compiler.py``. As
        02b–02e land, these disappear.
        """

        return self.carried_metrics.get(key, default)
