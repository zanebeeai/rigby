"""Unit tests for the whole-body analysis pass extracted in PR 02b.

The equivalence harness proves the move preserved every number. These pin the
three things the harness cannot: that the re-derivations are re-derivations and
not lucky coincidences, that the persisted authoring intent is actually load
bearing, and that the fixture reaches every gated block.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rigby_poc import analysis
from rigby_poc.analysis import equivalence
from rigby_poc.analysis.context import AnalysisContext
from rigby_poc.analysis.full_body import full_body_metrics
from rigby_poc.analysis.full_body.root import commanded_root_yaw_rad
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    BodyAction,
    BodyTarget,
    CompileRequest,
    Intent,
    MotionPrimitive,
    MotionProgram,
    PrimitiveKind,
    PrimitiveParameters,
    SceneManifest,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "analysis_equivalence"
CASE_IDS: list[str] = json.loads(
    (FIXTURE_DIR / "index.json").read_text(encoding="utf-8")
)["cases"]


def _load(case_id: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def _full_body_cases() -> list[str]:
    return [case_id for case_id in CASE_IDS if _load(case_id)["intent"] == "full_body"]


def _compile(case: dict):
    scene = SceneManifest.model_validate(case["scene"])
    program = MotionProgram.model_validate(case["program"])
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    return scene, program, clip


def _turn_program(*turns: tuple[float, bool]) -> MotionProgram:
    """A whole-body program of turns, each flagged as authored or recovery."""

    return MotionProgram(
        intent=Intent.FULL_BODY,
        source_text="turn on the spot",
        primitives=[
            MotionPrimitive(
                kind=PrimitiveKind.RECOVER if is_recovery else PrimitiveKind.BODY,
                label=f"turn_{index}",
                parameters=PrimitiveParameters(duration_s=0.5),
                body=BodyTarget(
                    action=BodyAction.TURN,
                    cycles=1.0,
                    intensity=0.5,
                    turn_degrees=degrees,
                ),
            )
            for index, (degrees, is_recovery) in enumerate(turns)
        ],
    )


class TestCommandedRootYaw:
    """``final_root_yaw_deg`` is commanded, and the derivation must be exact."""

    def test_it_accumulates_authored_turns_in_program_order(self) -> None:
        program = _turn_program((90.0, False), (45.0, False), (-30.0, False))

        expected = math.radians(90.0)
        expected = expected + math.radians(45.0)
        expected = expected + math.radians(-30.0)

        assert commanded_root_yaw_rad(program) == expected

    def test_it_ignores_turns_inside_a_recovery_primitive(self) -> None:
        """The compiler advances ``current_yaw`` only when ``not recover``.

        No planner emits a recovery TURN today, so no fixture case covers this
        branch — but the compiler guards it, and an analyzer that dropped the
        guard would silently disagree the moment one appeared.
        """

        with_recovery = _turn_program((90.0, False), (180.0, True), (45.0, False))
        without = _turn_program((90.0, False), (45.0, False))

        assert commanded_root_yaw_rad(with_recovery) == commanded_root_yaw_rad(without)

    def test_a_program_with_no_turn_commands_no_yaw(self) -> None:
        walking = _load("full_body_walk")["program"]

        assert commanded_root_yaw_rad(MotionProgram.model_validate(walking)) == 0.0

    @pytest.mark.parametrize("case_id", _full_body_cases())
    def test_it_reproduces_the_compiler_exactly_on_every_case(
        self, case_id: str
    ) -> None:
        case = _load(case_id)
        _, program, clip = _compile(case)

        derived = math.degrees(commanded_root_yaw_rad(program))

        assert derived == clip.metrics["final_root_yaw_deg"]


def test_the_two_neutral_ground_heights_the_compiler_derives_agree() -> None:
    """``_compile_full_body`` computes the ground plane twice, two ways.

    Generation uses the calibrated idle pose with the hips pinned to the origin;
    the measurement pass used the plain identity pose. They agree bit-for-bit
    only because the idle pose moves nothing below the waist. The analysis layer
    keeps one value for both, so this is the invariant that keeps that honest: an
    idle pose that ever rotates a leg breaks here rather than shifting every
    clearance metric by a silent millimetre.
    """

    from rigby_poc.compiler import _full_body_idle_pose
    from rigby_poc.kinematics import rig_kinematics
    from rigby_poc.models import BonePose, Vec3
    from rigby_poc.analysis.rig import rig_profile

    kinematics = rig_kinematics()
    idle = _full_body_idle_pose()
    generation_bones = {
        name: BonePose(
            rotation=rotation,
            position=(Vec3(x=0.0, y=0.0, z=0.0) if name == "hips" else None),
        )
        for name, rotation in idle.items()
    }
    generation = kinematics.canonical_positions(generation_bones)
    generation_height = min(
        float(generation["leftToes"][1]), float(generation["rightToes"][1])
    )

    identity = kinematics.canonical_positions(
        {name: BonePose() for name in rig_profile()["bone_map"]}
    )
    measurement_height = min(
        float(identity["leftToes"][1]), float(identity["rightToes"][1])
    )

    assert generation_height == measurement_height


class TestPersistedSupportTargets:
    """The commanded IK targets are an input the frames cannot supply."""

    def test_the_compiler_persists_them_for_a_walk(self) -> None:
        _, _, clip = _compile(_load("full_body_walk"))

        constraints = clip.metrics["support_constraints"]

        assert constraints, "a walk constrains a support foot on most frames"
        record = constraints[0]
        assert set(record) == {"frame_index", "side", "ankle_target"}
        assert isinstance(record["frame_index"], int)
        assert record["side"] in {"left", "right"}
        assert len(record["ankle_target"]) == 3

    def test_the_compiler_persists_climb_targets_for_a_climb(self) -> None:
        _, _, clip = _compile(_load("full_body_climb"))

        constraints = clip.metrics["climb_support_constraints"]

        assert constraints, "a climb records per-limb support targets"
        record = constraints[0]
        assert set(record) == {"frame_index", "object_id", "targets", "settled"}
        assert record["targets"], "a climb constraint names at least one limb"

    def test_the_support_metrics_actually_read_them(self) -> None:
        """Move the commanded target, and the achieved-versus-commanded error moves.

        Without this the three support metrics could be reading the frames alone
        and the persisted record would be dead weight — which is exactly the
        failure mode a carried-over value invites.
        """

        scene, program, clip = _compile(_load("full_body_walk"))
        baseline = full_body_metrics(
            AnalysisContext.from_clip(clip, program, scene)
        )

        nudged = json.loads(json.dumps(clip.metrics))
        for record in nudged["support_constraints"]:
            record["ankle_target"][0] += 0.5
        shifted = full_body_metrics(
            AnalysisContext.from_frames(clip.frames, program, scene, nudged)
        )

        assert (
            shifted["max_support_foot_target_error_m"]
            > baseline["max_support_foot_target_error_m"] + 0.4
        )

    def test_analysis_of_a_stored_clip_needs_no_recompilation(self) -> None:
        """The point of the extraction: same numbers, no compile step."""

        scene, program, clip = _compile(_load("full_body_climb"))

        from_clip = analysis.analyze(clip, program, scene)

        for key, value in from_clip.items():
            assert json.dumps(value, sort_keys=True) == json.dumps(
                clip.metrics[key], sort_keys=True
            ), key


class TestCoverage:
    def test_every_body_action_has_a_ported_analyzer(self) -> None:
        unported = [
            f"BodyAction.{action.name}"
            for action, entry in analysis.BODY_ANALYZERS.items()
            if not entry.ported
        ]

        assert not unported, f"02b left these unported: {unported}"
        assert set(analysis.BODY_ANALYZERS) == set(BodyAction)

    def test_the_fixture_reaches_every_gated_full_body_block(self) -> None:
        """A block nobody executes is a block nobody has verified.

        Thirteen of the whole-body metric blocks are gated on a primitive
        selector, so a fixture that never triggers one would let the equivalence
        assertions pass over dead code.
        """

        seen: set[str] = set()
        for case_id in _full_body_cases():
            seen |= set(_load(case_id)["expected_metrics"])

        unreached = sorted(
            name
            for name, keys in equivalence.FULL_BODY_GATED_KEYS.items()
            if not keys & seen
        )

        assert not unreached, f"no fixture case triggers: {unreached}"

    def test_the_carried_keys_are_not_claimed_as_analysis_output(self) -> None:
        """Inputs to the pass must not be re-exported as if measured."""

        case = _load("full_body_climb")
        scene, program, clip = _compile(case)

        produced = set(analysis.analyze(clip, program, scene))

        assert not produced & equivalence.FULL_BODY_CARRIED_KEYS
        assert equivalence.FULL_BODY_CARRIED_KEYS <= set(clip.metrics)
