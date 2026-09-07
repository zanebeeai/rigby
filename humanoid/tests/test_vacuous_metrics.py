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


def test_handoff_attachment_slip_measures_float_noise_not_slip() -> None:
    """The fifth instance, and plan 02 §6.1 named it as a *drift risk*.

    The handoff path defines the object's position as
    ``wrist + hand_world.apply(offset)`` and then measures slip as
    ``hand_world.inv().apply(position - wrist)`` compared against its own first
    sample. Algebraically that is the offset itself, so the object is rigidly
    attached by construction and the metric can only ever report the round-trip
    error of ``R.inv().apply(R.apply(x))``.

    Measured on the corpus: 1.4e-16 m against a 0.005 m gate — 2.9e-14 of the
    limit. "object slipped relative to an owning palm" cannot fire.

    Worth pinning because §6.1 lists ``attachment_slip`` as one of the two
    numeric-drift risks that make 02d hard. The drift risk is real; the metric
    it threatens measures nothing physical, so re-deriving it exactly is not
    worth buying with a persisted accumulator.
    """

    clip = _compile("object_handoff")

    slip = float(clip.metrics["handoff_attachment_slip_m"])

    assert slip < 1e-12, f"slip is {slip:.3e}; it used to be float noise"
    assert slip < 0.005 / 1e9, "the 0.005 gate is nine orders away from firing"


@pytest.fixture(scope="module")
def object_clips() -> dict[str, tuple[object, object]]:
    """Every corpus case carrying an object action, compiled once. ``id -> (case, clip)``.

    Scoped to the object cases rather than to the whole corpus. Only that
    compile path writes the two keys pinned below, so looping over all 47 to
    reach nine of them is precisely the corpus-wide loop
    ``tests/test_corpus_compile_budget.py`` exists to keep visible -- and this
    file is budgeted there.
    """

    from evals.corpus.loader import load_corpus
    from rigby_poc.compiler import compile_motion

    cases = [case for case in load_corpus() if case.program.object_action is not None]
    assert cases, "no object-action case in the corpus; these two pins stand on nothing"
    return {case.id: (case, compile_motion(case.compile_request())) for case in cases}


def test_measured_support_spin_turns_is_an_echo_of_the_request(
    object_clips: dict[str, tuple[object, object]],
) -> None:
    """``object_measured_support_spin_turns`` equals ``spin_turns``, by construction.

    The compiler sets ``support_spin_angle_rad = 2*pi * spin_turns * (alpha if
    MOVE else 1.0)`` and then reports ``abs(angle) / (2*pi)`` as the *measured*
    turn count. At the final frame of the move ``alpha`` is 1.0, so the measured
    value is the requested one with a multiply and a divide in between.

    Same class as the ``forearm_rotation_cycles`` echo in plan 02 §1.6: a metric
    named as a measurement whose value is the request. Unlike that one it has no
    branch that ever overwrites it with an observation, so it is an echo on
    every clip that spins rather than on some of them.
    """

    spun = [
        (case, clip)
        for case, clip in object_clips.values()
        if case.program.object_action.value == "spin"
    ]
    assert spun, "no spin case in the corpus; this pin has nothing to stand on"

    for case, clip in spun:
        requested = float(case.program.object_motion.spin_turns)
        measured = float(clip.metrics["object_measured_support_spin_turns"])
        assert measured == requested, (
            f"{case.id}: measured {measured} vs requested {requested} -- if these "
            "have diverged the metric has become a real measurement; delete this pin"
        )


def test_object_interaction_slip_is_float_noise_on_every_action(
    object_clips: dict[str, tuple[object, object]],
) -> None:
    """``palm_relative_object_slip_m`` cannot fire, on any of the eight actions.

    Measured across the corpus: 2.8e-17 to 1.3e-16 m against a ``> 0.005`` gate
    — thirteen to fourteen orders of magnitude away. Same mechanism as the
    handoff slip: while the object is held, its position is *defined* relative
    to the wrist, so the offset compared against its own first sample is the
    offset itself and the metric reports the round-trip error of a rotation.

    This is load-bearing for 02d's remaining scope rather than a curiosity. The
    three distinct attachment-offset branches in ``_compile_object_interaction``
    are the most intricate thing left to re-derive post-hoc, and this is the
    only metric they feed. Reproducing them byte-for-byte would buy an exact
    copy of a number that encodes nothing.
    """

    measured = {
        case_id: float(clip.metrics["palm_relative_object_slip_m"])
        for case_id, (_case, clip) in object_clips.items()
        if "palm_relative_object_slip_m" in clip.metrics
    }

    assert len(measured) >= 8, "the corpus lost its object-interaction cases"
    # Since 2026-09-05 the wrist a carried object is attached to is the rig's
    # own FK wrist, which moves with the chest. A spin pins the block to the
    # table and turns it under a hand whose wrist the chest carries a few
    # millimetres between phases, so its slip is now a real, small number
    # rather than round-trip noise. Every action that carries the object still
    # reports noise, because the object is defined relative to that wrist.
    real_slip = {case_id for case_id in measured if "spin" in case_id}
    noise = {case_id: value for case_id, value in measured.items() if case_id not in real_slip}
    worst = max(noise.values())
    assert worst < 1e-12, (
        f"worst slip is {worst:.3e}; it used to be float noise on every carried action"
    )
    assert worst < 0.005 / 1e9, "the 0.005 gate is nine orders from firing"
    assert all(measured[case_id] < 0.005 for case_id in real_slip), measured
