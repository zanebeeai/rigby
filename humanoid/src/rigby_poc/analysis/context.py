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

from ..kinematics import FINGERTIP_SOURCE_STEMS, RigKinematics, rig_kinematics
from ..models import (
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
from .rig import identity_bones

# Phase kinds whose interval counts as an active presentation, per intent.
# ``compile_motion`` appends exactly these while it emits frames; reading them
# back off ``phase_ranges_s`` reproduces the same list.
_PRESENTATION_KINDS: dict[Intent, frozenset[str]] = {
    Intent.GESTURE: frozenset(
        {
            PrimitiveKind.PRESENT.value,
            PrimitiveKind.HOLD.value,
            PrimitiveKind.SHAKE.value,
        }
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


def root_motion_allowed(program: MotionProgram) -> bool:
    """Whether this program's clip may translate its root.

    Matches the ``allow_root_motion`` argument each compile path passes to the
    safety metrics: only the whole-body and sequence paths enable it. One
    definition, because :func:`rigby_poc.analysis.validate` needs the same
    answer without an :class:`AnalysisContext` to hand, and two copies of a rule
    that decides whether a gate applies is how the gate stops applying.
    """

    return program.intent in {Intent.FULL_BODY, Intent.SEQUENCE}


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
        self._world_matrices: dict[int, list[np.ndarray]] = {}

    @classmethod
    def from_clip(
        cls,
        clip: ClipResult,
        program: MotionProgram,
        scene: SceneManifest,
        *,
        kinematics: RigKinematics | None = None,
    ) -> AnalysisContext:
        return cls(clip.frames, program, scene, clip.metrics, kinematics=kinematics)

    @classmethod
    def from_frames(
        cls,
        frames: list[ClipFrame],
        program: MotionProgram,
        scene: SceneManifest,
        carried_metrics: Mapping[str, Any],
        *,
        kinematics: RigKinematics | None = None,
    ) -> AnalysisContext:
        """Build a context before a :class:`ClipResult` exists.

        ``compile_motion`` needs this: it has the frames and the carry-over
        metrics in hand but has not assembled the result yet, and the whole
        point of the extraction is that the compiler and a post-hoc analyzer run
        the *same* code rather than two implementations that agree by
        inspection.
        """

        return cls(frames, program, scene, carried_metrics, kinematics=kinematics)

    @property
    def intent(self) -> Intent:
        return self.program.intent

    @property
    def object_action(self) -> ObjectAction | None:
        return self.program.object_action

    @property
    def allow_root_motion(self) -> bool:
        """Whether the clip is permitted to translate its root."""

        return root_motion_allowed(self.program)

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

    def world_matrices(self, index: int) -> list[np.ndarray]:
        """Every node's world matrix for one frame, evaluated at most once.

        The single forward-kinematics evaluation for this frame. Positions,
        rotations and fingertip pivots are all slices of it, so a check that
        needs an orientation costs no pass of its own — before this, a check
        wanting a world *rotation* had to call
        :meth:`RigKinematics.canonical_world_rotation`, which walks the whole
        hierarchy again for one 3x3 block.

        Memoised lazily rather than filled eagerly for every frame: the strike,
        gesture, grab, object and sequence paths evaluate no forward kinematics
        at all today, and the checks that want orientations want them over a
        phase window, not the whole clip. An eager list would charge every one
        of those paths a full pass per frame whether or not anything reads it.

        The memo lives here and not on :class:`RigKinematics` because this
        object's lifetime is one clip and ``self.frames`` is already a snapshot.
        ``rig_kinematics()`` is process-wide and its ``lru_cache`` carries a
        load-bearing provenance contract, so it must not grow a per-pose memo.

        Returned unowned: the list and its matrices are the cache itself, so a
        caller keeping a slice copies it, exactly as ``kinematics`` does.
        """

        cached = self._world_matrices.get(index)
        if cached is None:
            cached = self.kinematics.world_matrices(self.frames[index].bones)
            self._world_matrices[index] = cached
        return cached

    def world_rotation(self, index: int, canonical: str) -> np.ndarray:
        """One canonical bone's world rotation, sliced from the frame's one pass."""

        node = self.kinematics.node_by_canonical[canonical]
        return self.world_matrices(index)[node][:3, :3].copy()

    def fingertip_positions(self, index: int, hand: str) -> dict[str, np.ndarray]:
        """The five fingertip pivots of one hand, sliced from the frame's one pass."""

        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        suffix = "l" if hand == "left" else "r"
        matrices = self.world_matrices(index)
        node_by_name = self.kinematics.node_by_name
        return {
            digit: matrices[node_by_name[f"{stem}_{suffix}"]][:3, 3].copy()
            for digit, stem in FINGERTIP_SOURCE_STEMS.items()
        }

    @cached_property
    def rest_head_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """The head's world position and rotation at the identity pose.

        The datum every head-carried quantity is measured against: the ego
        camera's eye offset is the rest camera position minus this position, and
        a frame's head delta is its head rotation times this rotation's
        transpose. Both the gaze check and the visibility sampler need it, so it
        is derived once here rather than twice.

        The identity pose is not a clip frame, so it has no index to memoise
        under in :meth:`world_matrices`; it is one of the two non-frame
        evaluations the forward-kinematics guard allows per clip.
        """

        matrix = self.kinematics.world_matrices(identity_bones())[
            self.kinematics.node_by_canonical["head"]
        ]
        return matrix[:3, 3].copy(), matrix[:3, :3].copy()

    @cached_property
    def world_positions(self) -> list[dict[str, np.ndarray]]:
        # The slice arithmetic is character-identical to
        # ``RigKinematics.canonical_positions`` on purpose. These positions feed
        # every full-body and composite metric, so a last-ulp drift here moves
        # their corpus digests; taking the same slice of the same deterministic
        # matrices keeps the values bit-identical, which recomputing them any
        # other way would not.
        return [
            {
                canonical: self.world_matrices(index)[node_index][:3, 3].copy()
                for canonical, node_index in self.kinematics.node_by_canonical.items()
            }
            for index in range(len(self.frames))
        ]

    @cached_property
    def ground_height(self) -> float:
        """The floor, from the one definition of it. See :func:`physics.ground_height_of`.

        Delegated rather than duplicated: the physics layer needs the same
        number and two implementations of a datum agree on the day they are
        written and nothing holds them equal afterwards. The skeleton is passed
        in so this context's injected kinematics still decides.
        """

        from .physics import ground_height_of

        return ground_height_of(self.kinematics)

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
