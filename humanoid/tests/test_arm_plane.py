"""The humeral-roll fix's invariants, asserted at the solver level.

The roll (``arm_plane.humeral_roll``) presents the elbow hinge inside the
chosen bend plane, and the downstream compensation
(``arm_plane.forearm_roll_compensation``) returns the hand to the world
orientation the pre-roll solve produced.  Four properties make that design
safe, and each is pinned here directly against ``arm_pose_from_target`` rather
than against a compiled corpus, so a regression names the invariant it broke:

1. elbow hinge purity — flexion is the non-negative interior bend and
   abduction stays near zero across the authored workspace;
2. the hand's world orientation at a solver keyframe is identical to a
   roll-off replica's, which is what keeps silhouette projection (and the
   hand-visibility gate) byte-equivalent to the pre-roll solve;
3. the bend plane moves continuously along authored target paths, so the roll
   can never contribute a rotational discontinuity of its own;
4. the humerus twist the solve presents stays inside the enforced (-95, 95)
   shoulder band, with the infeasible bend-plane branch rejected in favour of
   its mirror rather than clamped.

The two ``arm_plane`` constants that make the authoring path independent of
the analysis layer are pinned against their sources of truth (the rig-derived
anatomical frame and ``config/rom.v1.json``) so a rig re-export or a ROM edit
turns this file red instead of silently skewing the solver's decisions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import pairwise

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_poc import primitives
from rigby_poc.analysis.anatomy.frame import bone_anatomical_frame, decompose
from rigby_poc.analysis.anatomy.rom import rom_limit
from rigby_poc.arm_plane import (
    ELBOW_FLEXION_AXIS_LOCAL,
    UPPER_ARM_TWIST_BAND_RAD,
)
from rigby_poc.kinematics import arm_calibration
from rigby_poc.models import Hand, PrimitiveParameters, Quat, Vec3
from rigby_poc.primitives import (
    _arm_rest_rotations,
    arm_pose_from_target,
    arm_segment_lengths,
    shoulder_position,
)
from rigby_poc.thresholds import value_of

#: no compile, no pipeline, no corpus, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast

HANDS = (Hand.LEFT, Hand.RIGHT)


def _side(hand: Hand) -> float:
    return 1.0 if hand == Hand.LEFT else -1.0


def _authored_workspace(hand: Hand) -> Iterator[tuple[Vec3, PrimitiveParameters]]:
    """Targets shaped like what the generator actually authors.

    The rectangular block spans the gesture/strike/object workspace in front
    of the torso; the two landmarks are the jump's raise-arms-overhead target
    and the cartwheel's near-straight overhead extension — the most twist-
    hungry reaches any compile path requests, carried with the elbow swivel
    the compiler authors them with.  Deliberately excluded: targets behind
    the body or against the sternum, which no path authors with the default
    pole and where no bend-plane branch is anatomically clean.
    """
    side = _side(hand)
    for x in np.linspace(0.05, 0.45, 5):
        for y in np.linspace(0.95, 1.85, 6):
            for z in np.linspace(0.15, 0.55, 5):
                yield Vec3(x=side * float(x), y=float(y), z=float(z)), PrimitiveParameters()
    overhead = PrimitiveParameters(elbow_swivel=side * 0.12)
    yield Vec3(x=side * 0.18, y=1.84, z=0.12), overhead
    yield Vec3(x=side * 0.18, y=2.06, z=0.10), overhead


def _run_stride_targets(hand: Hand) -> list[Vec3]:
    """The run gait's wrist path, the deepest back-reach any path authors."""
    side = _side(hand)
    targets = []
    for phase in np.linspace(0.0, 2.0 * math.pi, 61):
        advance = math.sin(phase) * 1.15
        targets.append(
            Vec3(
                x=side * (0.30 + 0.025 * abs(advance)),
                y=1.08 + 0.13 * max(0.0, advance) - 0.035 * max(0.0, -advance),
                z=0.10 + 0.19 * advance,
            )
        )
    return targets


RUN_ELBOW_POLE = {hand: Vec3(x=_side(hand) * 0.6, y=-0.55, z=-0.5) for hand in HANDS}


def _arm_world_rotations(hand: Hand, poses: dict[str, Quat]) -> dict[str, Rotation]:
    """World rotations composed from the solver's own rest calibration."""
    rest_upper, rest_lower, rest_hand = _arm_rest_rotations(hand)
    upper = rest_upper * Rotation.from_quat(poses[f"{hand.value}UpperArm"].as_list())
    lower = upper * rest_lower * Rotation.from_quat(poses[f"{hand.value}LowerArm"].as_list())
    world_hand = lower * rest_hand * Rotation.from_quat(poses[f"{hand.value}Hand"].as_list())
    return {"upper": upper, "lower": lower, "hand": world_hand}


def _elbow_line(hand: Hand, target: Vec3) -> tuple[np.ndarray, np.ndarray, float]:
    """The two-link triangle: mid-line point, default-pole bend, bend height."""
    shoulder = np.asarray(shoulder_position(hand).as_list(), dtype=float)
    reach = np.asarray(target.as_list(), dtype=float) - shoulder
    distance = float(np.linalg.norm(reach))
    upper, lower = arm_segment_lengths(hand)
    clamped = float(np.clip(distance, abs(upper - lower) + 1e-4, upper + lower - 1e-4))
    direction = reach / max(distance, 1e-8)
    along = (upper**2 - lower**2 + clamped**2) / (2 * clamped)
    bend_height = math.sqrt(max(upper**2 - along**2, 0.0))
    side = _side(hand)
    pole = np.asarray([side, -0.55, 0.25], dtype=float)
    bend = pole - direction * float(np.dot(pole, direction))
    bend /= np.linalg.norm(bend)
    return shoulder + direction * along, bend, bend_height


def _decomposed(hand: Hand, poses: dict[str, Quat], bone: str):
    return decompose(
        np.asarray(poses[f"{hand.value}{bone}"].as_list()),
        bone_anatomical_frame(f"{hand.value}{bone}"),
    )


# --------------------------------------------------------------------------
# The two authoring-path constants track their sources of truth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ("left", "right"))
def test_the_elbow_flexion_axis_constant_tracks_the_rig(side: str) -> None:
    derived = bone_anatomical_frame(f"{side}LowerArm").flexion_axis
    assert np.allclose(derived, ELBOW_FLEXION_AXIS_LOCAL, atol=1e-9), derived


@pytest.mark.parametrize("side", ("left", "right"))
def test_the_twist_band_constant_tracks_the_rom_config(side: str) -> None:
    limit = rom_limit(f"{side}UpperArm", "twist")
    low, high = (math.radians(limit.to_rest_relative(v)) for v in limit.max_deg)
    assert abs(high - UPPER_ARM_TWIST_BAND_RAD) < 1e-6, high
    assert abs(low + UPPER_ARM_TWIST_BAND_RAD) < 1e-6, low


# --------------------------------------------------------------------------
# 1. Elbow hinge purity, and 4. the humerus twist band
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand", HANDS)
def test_elbow_hinge_purity_across_the_authored_workspace(hand: Hand) -> None:
    checked = 0
    for target, parameters in _authored_workspace(hand):
        poses, _ = arm_pose_from_target(
            hand, shoulder_position(hand), target, parameters
        )
        angles = _decomposed(hand, poses, "LowerArm")
        assert abs(angles.abduction_rad) < math.radians(3.0), (target, angles)
        # The interior bend is non-negative by construction.  The slack
        # covers the LowerArm's small rest offset, which reads as up to a
        # degree of nominal extension when the arm is pulled near-straight.
        assert angles.flexion_rad > -math.radians(1.5), (target, angles)
        checked += 1
    assert checked > 100


@pytest.mark.parametrize("hand", HANDS)
def test_upper_arm_twist_stays_inside_the_band(hand: Hand) -> None:
    run_parameters = PrimitiveParameters(elbow_swivel=-0.10)
    cases = [
        (target, parameters, None) for target, parameters in _authored_workspace(hand)
    ] + [
        (target, run_parameters, RUN_ELBOW_POLE[hand])
        for target in _run_stride_targets(hand)
    ]
    checked = 0
    for target, parameters, pole in cases:
        poses, _ = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            parameters,
            elbow_pole=pole,
        )
        twist = _decomposed(hand, poses, "UpperArm").twist_rad
        assert abs(twist) <= UPPER_ARM_TWIST_BAND_RAD + 1e-9, (target, twist)
        checked += 1
    assert checked > 150


# --------------------------------------------------------------------------
# 2. The hand's world orientation matches a roll-off replica at keyframes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("present_hand", (False, True))
def test_hand_world_orientation_matches_a_roll_off_replica(
    hand: Hand, present_hand: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The roll must change nothing about the hand a camera can see.

    The replica solver never rolls the humerus, so it reproduces the pre-roll
    solve this fix's compensation promises to preserve.  Points where the
    pronation budget saturates are excluded — there the remainder moves into
    the hand bone by design — and the exclusion is counted so the assertion
    cannot go vacuous.
    """
    budget = primitives.MAX_FOREARM_TWIST_RAD
    shoulder = np.asarray(shoulder_position(hand).as_list(), dtype=float)
    compared = 0
    skipped = 0
    for target, parameters in _authored_workspace(hand):
        poses, _ = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            parameters,
            present_hand=present_hand,
        )
        lower_twist = _decomposed(hand, poses, "LowerArm").twist_rad
        if abs(lower_twist) >= budget - 1e-6:
            skipped += 1
            continue
        # The replica must solve on the same bend-plane branch the fixed
        # solve chose, so the chosen elbow position is recovered from the
        # pose and handed back as a hint, and the branch selection is
        # disarmed (the band widened to infinity) — otherwise a target the
        # fixed solve mirrored would be compared against the other branch's
        # pre-roll hand, which is a different orientation by design.
        world = _arm_world_rotations(hand, poses)
        elbow = shoulder + world["upper"].apply([0.0, 1.0, 0.0]) * arm_segment_lengths(hand)[0]
        with monkeypatch.context() as patch:
            patch.setattr(primitives, "humeral_roll", lambda *args: 0.0)
            patch.setattr(primitives, "UPPER_ARM_TWIST_BAND_RAD", math.inf)
            replica, _ = arm_pose_from_target(
                hand,
                shoulder_position(hand),
                target,
                parameters,
                present_hand=present_hand,
                elbow_hint=Vec3(
                    x=float(elbow[0]), y=float(elbow[1]), z=float(elbow[2])
                ),
            )
        preroll = _arm_world_rotations(hand, replica)["hand"]
        assert (world["hand"].inv() * preroll).magnitude() < 1e-7, target
        compared += 1
    assert compared > 90
    assert skipped < compared // 3


# --------------------------------------------------------------------------
# 3. Bend-plane / roll continuity along authored target paths
# --------------------------------------------------------------------------


def _max_step_delta(sequences: list[dict[str, Quat]]) -> float:
    worst = 0.0
    for previous, current in pairwise(sequences):
        for key in previous:
            a = np.asarray(previous[key].as_list())
            b = np.asarray(current[key].as_list())
            worst = max(
                worst, 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))
            )
    return worst


@pytest.mark.parametrize("hand", HANDS)
def test_roll_continuity_along_the_run_stride(hand: Hand) -> None:
    """Sampled at the compiled gait's own frame density, the solver's output
    must stay well under the validator's rotational-discontinuity limit."""
    poses = [
        arm_pose_from_target(
            hand,
            shoulder_position(hand),
            target,
            PrimitiveParameters(elbow_swivel=-0.10),
            elbow_pole=RUN_ELBOW_POLE[hand],
        )[0]
        for target in _run_stride_targets(hand)
    ]
    assert _max_step_delta(poses) < value_of("signal.discontinuity_rad") * 0.6


@pytest.mark.parametrize("hand", HANDS)
def test_roll_continuity_around_the_travel_wheel(hand: Hand) -> None:
    """A full orbit of the travel wheel, elbow hint included: the bend plane
    follows the hint continuously and never jumps a branch mid-cycle."""
    side = _side(hand)
    center = Vec3(x=0.0, y=1.422, z=0.25)
    poses = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 121):
        target = Vec3(
            x=center.x - side * 0.135,
            y=center.y + side * 0.09 * math.cos(angle),
            z=center.z + side * 0.055 * math.sin(angle),
        )
        hint = Vec3(
            x=center.x + side * 0.135,
            y=center.y + side * 0.09 * math.cos(angle),
            z=center.z + side * 0.055 * math.sin(angle),
        )
        poses.append(
            arm_pose_from_target(
                hand,
                shoulder_position(hand),
                target,
                PrimitiveParameters(),
                elbow_hint=hint,
            )[0]
        )
    assert _max_step_delta(poses) < value_of("signal.discontinuity_rad") * 0.6


# --------------------------------------------------------------------------
# 4. The infeasible bend-plane branch is chosen away, never clamped
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand", HANDS)
def test_an_infeasible_elbow_hint_selects_the_mirrored_branch(hand: Hand) -> None:
    """A hint on the bend-plane branch the shoulder cannot twist to must make
    the solver choose the mirrored branch — which here is exactly the default
    pole's own solution, so the whole pose must match it."""
    side = _side(hand)
    target = Vec3(x=side * 0.25, y=1.30, z=0.30)
    line_point, bend, bend_height = _elbow_line(hand, target)
    wrong_side = line_point - bend * bend_height
    natural, _ = arm_pose_from_target(
        hand, shoulder_position(hand), target, PrimitiveParameters()
    )
    hinted, _ = arm_pose_from_target(
        hand,
        shoulder_position(hand),
        target,
        PrimitiveParameters(),
        elbow_hint=Vec3(x=float(wrong_side[0]), y=float(wrong_side[1]), z=float(wrong_side[2])),
    )
    for key, value in natural.items():
        agreement = abs(
            float(
                np.dot(
                    np.asarray(value.as_list()),
                    np.asarray(hinted[key].as_list()),
                )
            )
        )
        assert agreement > 1.0 - 1e-9, key
    twist = _decomposed(hand, hinted, "UpperArm").twist_rad
    assert abs(twist) <= UPPER_ARM_TWIST_BAND_RAD
    assert _decomposed(hand, hinted, "LowerArm").flexion_rad >= -math.radians(1.5)


@pytest.mark.parametrize("hand", HANDS)
def test_a_feasible_elbow_hint_is_still_honoured(hand: Hand) -> None:
    """The companion that keeps the mirror test non-vacuous: a hint the
    shoulder can reach must move the elbow, not be silently discarded."""
    side = _side(hand)
    target = Vec3(x=side * 0.25, y=1.30, z=0.30)
    line_point, bend, bend_height = _elbow_line(hand, target)
    direction = np.asarray(target.as_list()) - np.asarray(
        shoulder_position(hand).as_list()
    )
    direction = direction / np.linalg.norm(direction)
    swung = Rotation.from_rotvec(direction * side * 0.5).apply(bend)
    hint_point = line_point + swung * bend_height
    hinted, _ = arm_pose_from_target(
        hand,
        shoulder_position(hand),
        target,
        PrimitiveParameters(),
        elbow_hint=Vec3(
            x=float(hint_point[0]), y=float(hint_point[1]), z=float(hint_point[2])
        ),
    )
    natural, _ = arm_pose_from_target(
        hand, shoulder_position(hand), target, PrimitiveParameters()
    )
    upper_natural = np.asarray(natural[f"{hand.value}UpperArm"].as_list())
    upper_hinted = np.asarray(hinted[f"{hand.value}UpperArm"].as_list())
    moved = 2.0 * math.acos(
        float(np.clip(abs(np.dot(upper_natural, upper_hinted)), 0.0, 1.0))
    )
    assert moved > math.radians(10.0)
    assert abs(_decomposed(hand, hinted, "UpperArm").twist_rad) <= UPPER_ARM_TWIST_BAND_RAD


# ---------------------------------------------------------------------------
# arm calibration: rig-derived, replacing the six frozen literals
# ---------------------------------------------------------------------------

#: The retired constants, kept as the measurement of what the retirement
#: changed. They are the test's fixtures now, not anyone's source of truth.
_RETIRED_UPPER_REST_WORLD_XYZW = {
    Hand.LEFT: (0.009422436964241731, -0.009307101973217042, 0.7089223032569065, -0.7051622249379483),
    Hand.RIGHT: (-0.009422320458211458, -0.009307162762708104, 0.7089222775335169, 0.7051622515529192),
}
_RETIRED_LOWER_REST_LOCAL_XYZW = {
    Hand.LEFT: (0.02164141647517681, 0.0002870236639864743, -0.006738185882568359, 0.9997430443763733),
    Hand.RIGHT: (0.021641412749886513, -0.0002871047181542963, 0.006738179363310337, 0.9997430443763733),
}
_RETIRED_HAND_REST_LOCAL_XYZW = {
    Hand.LEFT: (-0.00840034894645214, -0.00005970777783659287, 0.009397653862833977, 0.9999206066131592),
    Hand.RIGHT: (-0.00840041134506464, 0.00005969807898509316, -0.009397652931511402, 0.9999205470085144),
}
_RETIRED_UPPER_LENGTH_M = 0.2966
_RETIRED_LOWER_LENGTH_M = 0.2798


def _rotation_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float((Rotation.from_quat(a) * Rotation.from_quat(b).inv()).magnitude())


def test_arm_calibration_matches_the_rig_and_names_the_change() -> None:
    """The rig-derived calibration, pinned against the literals it retired.

    Two different claims, deliberately separated. The rest ROTATIONS agree
    with the frozen copies to ~1e-15, so for orientation the retirement is a
    near-no-op and any visible motion change traces to the lengths. The
    LENGTHS move by the 5.3 / 11.9 um the 4-decimal rounding hid -- asserted
    both that the gap exists (the change is real, not a refactor) and that it
    stays micrometre-scale (a regression in the derivation cannot hide inside
    this test's tolerance).
    """

    for hand in (Hand.LEFT, Hand.RIGHT):
        calibration = arm_calibration(hand.value)
        assert _rotation_distance(
            calibration.upper_rest_world_xyzw, _RETIRED_UPPER_REST_WORLD_XYZW[hand]
        ) < 5e-15
        assert _rotation_distance(
            calibration.lower_rest_local_xyzw, _RETIRED_LOWER_REST_LOCAL_XYZW[hand]
        ) < 5e-15
        assert _rotation_distance(
            calibration.hand_rest_local_xyzw, _RETIRED_HAND_REST_LOCAL_XYZW[hand]
        ) < 5e-15
        upper_gap = abs(calibration.upper_length_m - _RETIRED_UPPER_LENGTH_M)
        lower_gap = abs(calibration.lower_length_m - _RETIRED_LOWER_LENGTH_M)
        assert 1e-6 < upper_gap < 2e-5, "the rounding gap this PR closes is gone or grew"
        assert 1e-5 < lower_gap < 3e-5, "the rounding gap this PR closes is gone or grew"
        upper, lower = arm_segment_lengths(hand)
        assert (upper, lower) == (calibration.upper_length_m, calibration.lower_length_m)


def test_the_rig_is_left_right_asymmetric_and_the_calibration_says_so() -> None:
    """~1.1e-7 m of upper-arm asymmetry is the asset's, reported not repaired.

    A symmetrised shared constant would manufacture an exactness the rig does
    not have; the 2026-08-29 ruling chose truth over imposed symmetry, so the
    asymmetry is pinned as a property. If a future rig IS exactly symmetric,
    this test should be updated with that measurement, not deleted.
    """

    left = arm_calibration("left")
    right = arm_calibration("right")
    upper_asymmetry = abs(left.upper_length_m - right.upper_length_m)
    assert 5e-8 < upper_asymmetry < 5e-7
    lower_asymmetry = abs(left.lower_length_m - right.lower_length_m)
    assert lower_asymmetry < 5e-7
