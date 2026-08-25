"""Published values in this layer that encode nothing, pinned as an inventory.

Six lanes independently found the same defect shape this cycle: a field that is
present, authoritative-looking, and not derived from the thing it claims to
summarise. A manifest header carried forward while every row disagreed. A
``determinism_class: portable`` asserting bit-equality that measurement
disproved. Ceilings computed and never compared. Gates compared but unreachable.
Metrics published without being measured.

Prose findings rot. These are the instances inside ``compiler.py`` and
``analysis/`` — the territory this lane owns — written as assertions so the
inventory is executable and so the day one of them stops being vacuous, the
test that pins it fails and says so.

None of these is fixed here. Each fix changes published numbers, so each wants
its own PR with before and after. What this file buys is that the scope is a
number rather than a story, and that nobody re-derives it from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_poc.analysis.equivalence import UNWRITTEN_BASE_DEFAULTS
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, MotionProgram, SceneManifest

pytestmark = pytest.mark.medium


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "analysis_equivalence"


def _compile(case_id: str):
    case = json.loads((FIXTURE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    scene = SceneManifest.model_validate(case["scene"])
    program = MotionProgram.model_validate(case["program"])
    return compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )


def test_ten_base_metrics_are_published_without_being_measured() -> None:
    """``_base_metrics`` seeds ten values most compile paths never write.

    ``max_penetration_m: 0.0`` on a whole-body climb reads as "verified no
    penetration"; nothing looked. ``lost_table_contact: True`` reads as a
    finding; there is no table in the scene. ``foot_drift_m`` — the one lane
    `infra` found compared against an acceptance criterion of 0.001 it can never
    breach — is a member of the same dict, so it is one constructor rather than
    one key.
    """

    clip = _compile("full_body_climb")

    untouched = {
        key: clip.metrics[key]
        for key in UNWRITTEN_BASE_DEFAULTS
        if key in clip.metrics
    }

    assert untouched == {
        "finger_assertions": {},
        "hold_duration_s": 0.0,
        "lift_height_m": 0.0,
        "lost_table_contact": True,
        "max_penetration_m": 0.0,
        "opposing_contacts": False,
        "palm_relative_slip_m": 0.0,
        "unresolved_non_hand_collisions": 0,
        "vertical_drift_m": 0.0,
        "weld_used": False,
    }


@pytest.mark.parametrize(
    "case_id",
    [
        "gesture_open_palm",
        "gesture_point_right",
        "strike_left_hook",
        "grab_block_right",
        "strike_shake_echo",
    ],
)
def test_the_hand_shape_gate_cannot_fail_for_most_shapes(case_id: str) -> None:
    """``finger_assertions`` is a tautology for every shape but HANG_TEN.

    The else branch is ``{"shape_defined": len(actual_curls) == 5}``, and
    ``curl_values_from_frame`` iterates a five-entry table — it returns exactly
    five values or raises ``KeyError``. The comparison cannot be False.

    It is load-bearing in appearance: ``compile_motion`` folds
    ``all(assertions.values())`` into ``success``, so a reader sees a hand-shape
    gate contributing to the verdict. Only HANG_TEN clips actually have one.
    """

    clip = _compile(case_id)

    assert clip.metrics["finger_assertions"] == {"shape_defined": True}


def test_hang_ten_is_the_one_shape_with_a_real_assertion() -> None:
    """The contrast that shows the others are not merely passing."""

    clip = _compile("gesture_shaka_right")

    assertions = clip.metrics["finger_assertions"]

    assert set(assertions) == {
        "thumb_extended",
        "little_extended",
        "index_curled",
        "middle_curled",
        "ring_curled",
        "all_finger_bones_present",
    }


@pytest.mark.parametrize(
    "case_id", ["full_body_walk", "full_body_run", "full_body_cartwheel"]
)
def test_the_whole_body_path_compares_its_kinematics_to_no_limit(
    case_id: str,
) -> None:
    """Three angular metrics computed, never compared to anything.

    Only gesture, strike and composite enforce a kinematic ceiling. The
    whole-body path emits velocity, acceleration and jerk and reads none of them
    back, so a clip can exceed every published ceiling and still report
    ``structural_valid: True``. Lane `groundtruth` measured five of ten
    whole-body corpus cases over the acceleration and jerk ceilings, by up to
    2.9x and 3.4x, two of them plain walk and run.

    Fixing this is plan 10 §3.1's physics layer, not a threshold edit — which is
    why it is pinned rather than patched.
    """

    clip = _compile(case_id)

    for key in (
        "max_angular_velocity_rad_s",
        "max_angular_acceleration_rad_s2",
        "max_angular_jerk_rad_s3",
    ):
        assert key in clip.metrics, f"{case_id} stopped emitting {key}"

    assert clip.metrics["structural_valid"] is True
    assert not any(
        "angular" in failure or "jerk" in failure
        for failure in clip.metrics["structural_failures"]
    ), "a kinematic ceiling now fires on the whole-body path; update this file"


def test_the_support_metrics_measure_solver_convergence_not_motion() -> None:
    """The three keys 02b unlocked read ~5000x below their own gates.

    Plan 02 §1.4 called persisting ``support_constraints`` "the highest-leverage
    edit in this PR" because it unblocks four metrics. It does — but on a
    generated clip those metrics measure how well the IK solver converged, not
    whether the motion is good. Across the whole-body corpus,
    ``max_support_foot_target_error_m`` sits at ~2e-6 m against a 0.012 gate and
    ``max_support_foot_slide_per_frame_m`` at ~1e-7 against 0.006. They cannot
    fire unless the solver fails.

    That does not make persisting them wrong: a mutation moves the achieved foot
    while the commanded target stays fixed, which is exactly what makes "the foot
    missed where it was told to go" detectable, and is why plan 06 carries them
    over unchanged. But the plan's framing oversold what they measure *today*,
    and this pins the distinction so nobody reads a passing gate as evidence the
    motion was checked.
    """

    clip = _compile("full_body_walk")

    assert clip.metrics["max_support_foot_target_error_m"] < 1e-5
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 1e-5


def test_a_climb_publishes_perfect_support_contact_having_measured_none() -> None:
    """The fourth instance, and it is in code this lane moved.

    The climb path records ``climb_support_constraints`` and never
    ``support_constraints``, so the support block runs over an empty list.
    ``max(..., default=0.0)`` gives two zeros that read as perfect tracking, and
    ``support_contacts / len(...) if ... else 1.0`` gives
    ``support_contact_fraction: 1.0`` — a positive claim of full support contact
    on a clip where nothing counted a single contact.

    Same shape as ``lost_table_contact: True`` with no table in the scene: not
    merely unmeasured, but asserted in the direction that passes.
    """

    clip = _compile("full_body_climb")

    assert clip.metrics["support_constraints"] == []
    assert clip.metrics["max_support_foot_target_error_m"] == 0.0
    assert clip.metrics["max_support_foot_slide_per_frame_m"] == 0.0
    assert clip.metrics["support_contact_fraction"] == 1.0
