"""Provenance for the committed cases: which prompt each one was planned from.

Every corpus case is a *program*, and a program is opaque about where it came from.
This table records that, so a reader can see why these twelve exist and 03b can add
its cases the same way:

    python -m evals.corpus freeze --from-seed gesture-hangten-shake-right

The twelve were chosen by measured check coverage, not prompt variety. A greedy
set-cover over 81 offline-plannable candidates -- scored on distinct
``clip.metrics`` keys plus ``Intent``/``BodyAction``/``PrimitiveKind``/
``HandShape``/``StrikeType``/support-mode/rotation-mode/obstacle-mode members --
reaches 305 of 334 available features with these twelve. See plan 03 section 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gates import StructuralGate
from .models import DeterminismClass, Family, StoragePolicy


@dataclass(frozen=True)
class SeedCase:
    id: str
    family: Family
    prompt: str
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    #: ``ParameterOverrides`` kwargs applied at compile time.  03b's known-bad cases
    #: use these to drive a *real* program past a *real* gate, so the case stays a
    #: compiler output rather than a hand-edited clip.
    overrides: dict[str, Any] = field(default_factory=dict)
    #: See :mod:`evals.corpus.gates`.  Required for :attr:`Family.KNOWN_BAD`.
    must_fail: tuple[StructuralGate, ...] = ()
    determinism_class: DeterminismClass = DeterminismClass.PLATFORM_DEPENDENT
    storage: StoragePolicy = StoragePolicy.PROGRAM_AND_CLIP


#: Two prompts are used by more than one case -- a known-good case and the known-bad
#: case built by overriding it -- so they are named once.  A known-bad case whose
#: prompt drifted from its known-good twin would no longer isolate the override as
#: the cause of the failure.
HANGTEN_PROMPT = (
    'Throw up a "hang-ten" sign with your right hand, there should be a '
    "swift motion up to the main position wherein the middle three fingers "
    "are as contracted as possible, the wrist should then shake rapidly "
    "back and forth a few times, before returning to default"
)

TRAVEL_PROMPT = (
    "roll your forearms around eachother repeatedly, as if you are "
    'indicating the "travel" foul in basketball'
)


SEED_CASES_03A: tuple[SeedCase, ...] = (
    SeedCase(
        id="composite-travel-forearms",
        family=Family.COMPOSITE,
        prompt=TRAVEL_PROMPT,
        tags=["both-hands", "cycle", "parallel-forearm", "trajectory-circle"],
        notes="Richest metric block in the corpus: parallel-forearm, travel-wheel and semantic-cycle checks.",
    ),
    SeedCase(
        id="composite-wave-left",
        family=Family.COMPOSITE,
        prompt="wave hello with your left hand",
        tags=["left", "single-hand-composite"],
        notes="Single-hand composite path, distinct from the two-hand cycle case.",
    ),
    SeedCase(
        id="fullbody-burpee-cycle",
        family=Family.FULL_BODY,
        prompt="do three burpees",
        tags=["crouch", "jump", "plank", "pose", "multi-action"],
        notes="Four BodyActions and the plank support mode in one case; the slowest case to compile.",
    ),
    SeedCase(
        id="fullbody-cartwheel",
        family=Family.FULL_BODY,
        prompt="do a cartwheel",
        tags=["rotate", "rotation-cartwheel"],
        notes="Cartwheel rotation mode; floor and airborne modes are deferred to 03b.",
    ),
    SeedCase(
        id="fullbody-dance",
        family=Family.FULL_BODY,
        prompt="do a little dance",
        tags=["dance", "cycles"],
    ),
    SeedCase(
        id="fullbody-ladder-climb",
        family=Family.FULL_BODY,
        prompt="climb up the ladder",
        tags=["climb", "support-object", "affordance"],
        notes="Only case that consumes a climb-contact affordance socket.",
    ),
    SeedCase(
        id="fullbody-step-over-hurdle",
        family=Family.FULL_BODY,
        prompt="step over the hurdle with your right foot",
        tags=["step", "obstacle-over", "clearance"],
        notes="Only case exercising obstacle traversal.",
    ),
    SeedCase(
        id="gesture-hangten-shake-right",
        family=Family.GESTURE,
        prompt=HANGTEN_PROMPT,
        tags=["right", "hang-ten", "shake", "dexterity"],
        notes="The flagship demo prompt; only case with a SHAKE primitive and a forearm-twist reserve.",
    ),
    SeedCase(
        id="gesture-shaka-playful-left",
        family=Family.GESTURE,
        prompt=(
            "With your left hand, make a balanced relaxed shaka high and extended, "
            "drawn inward. Pitch the wrist down, yaw it outward, and roll it "
            "counterclockwise."
        ),
        tags=["left", "hang-ten", "review-fixture-s12", "human-rated"],
        notes="planner_supported.json s12; mean human score 4.0 in config/motion_quality_reference.json.",
    ),
    SeedCase(
        id="gesture-shaka-playful-right",
        family=Family.GESTURE,
        prompt=(
            "With your right hand, make a balanced playful shaka at chest height "
            "and a natural reach, drawn inward. Pitch the wrist up, yaw it outward, "
            "and roll it clockwise."
        ),
        tags=["right", "hang-ten", "review-fixture-s09", "human-rated"],
        notes="planner_supported.json s09; mean human score 4.5 in config/motion_quality_reference.json.",
    ),
    SeedCase(
        id="strike-jab-left",
        family=Family.STRIKE,
        prompt="throw a left jab",
        tags=["left", "jab", "linear-path"],
    ),
    SeedCase(
        id="strike-uppercut-right",
        family=Family.STRIKE,
        prompt="throw a right uppercut",
        tags=["right", "uppercut", "vertical-path"],
    ),
)

#: 03b's additions.  Kept in a separate tuple so the twelve 03a cases stay legible
#: as the set the greedy set-cover chose; ``SEED_CASES`` is the concatenation.
SEED_CASES_03B: tuple[SeedCase, ...] = (
    # -- gesture: the remaining hand shapes -------------------------------------
    SeedCase(
        id="gesture-fist-right",
        family=Family.GESTURE,
        prompt="clench your right hand into a fist",
        tags=["right", "fist"],
        notes="HandShape.FIST outside a strike, where it is a held shape rather than an impact pose.",
    ),
    SeedCase(
        id="gesture-peace-left",
        family=Family.GESTURE,
        prompt="make a peace sign with your left hand",
        tags=["left", "peace", "two-finger"],
        notes="HandShape.PEACE; the only two-finger extension in the corpus.",
    ),
    SeedCase(
        id="gesture-pinch-left",
        family=Family.GESTURE,
        prompt="pinch your thumb and index fingertip together with your left hand",
        tags=["left", "pinch", "precision-grip"],
        notes="HandShape.PINCH outside the grab lifecycle, where it is a transient contact pose.",
    ),
    SeedCase(
        id="gesture-point-right",
        family=Family.GESTURE,
        prompt="point straight ahead with your right index finger",
        tags=["right", "point", "index-extension"],
        notes="HandShape.POINT. Note the prompt must not name a scene object: planner.py:3940 excludes a block/cube/box mention from the gesture branch, so 'point at the block' resolves UNSUPPORTED.",
    ),
    SeedCase(
        id="gesture-thumbsup-right",
        family=Family.GESTURE,
        prompt="give a thumbs up with your right hand",
        tags=["right", "thumbs-up"],
        notes="HandShape.THUMBS_UP.",
    ),
    # -- strike: the remaining two paths -----------------------------------------
    SeedCase(
        id="strike-cross-right",
        family=Family.STRIKE,
        prompt="throw a right cross",
        tags=["right", "cross", "linear-path", "longest-strike"],
        notes="StrikeType.CROSS; the longest strike at 119 frames.",
    ),
    SeedCase(
        id="strike-hook-right",
        family=Family.STRIKE,
        prompt="throw a right hook",
        tags=["right", "hook", "curved-path"],
        notes="StrikeType.HOOK; completes StrikeType with jab, uppercut and cross.",
    ),
    # -- composite ---------------------------------------------------------------
    SeedCase(
        id="composite-beckon-right",
        family=Family.COMPOSITE,
        prompt="beckon someone over with your right hand",
        tags=["right", "cycle", "single-hand-composite"],
        notes="Right-hand mirror of composite-wave-left; the laterality pair plan 10 section 2.2 calls minimal distance.",
    ),
    SeedCase(
        id="composite-clap-twice",
        family=Family.COMPOSITE,
        prompt="clap your hands together twice",
        tags=["both-hands", "converging", "no-cycle-primitive"],
        notes="Two-hand composite built from MOVE alone: the only composite case with no CYCLE primitive.",
    ),
    # -- full body: the remaining four actions ------------------------------------
    SeedCase(
        id="fullbody-kick-right",
        family=Family.FULL_BODY,
        prompt="kick with your right foot",
        tags=["kick", "single-support"],
        notes="BodyAction.KICK; single-support balance with a free swinging leg.",
    ),
    SeedCase(
        id="fullbody-run-forward",
        family=Family.FULL_BODY,
        prompt="run forward",
        tags=["run", "gait", "flight-phase"],
        notes="BodyAction.RUN; the gait path at its highest cadence.",
    ),
    SeedCase(
        id="fullbody-shrug-shoulders",
        family=Family.FULL_BODY,
        prompt="shrug your shoulders",
        tags=["pose", "clavicle", "shoulder-elevation"],
        notes="The only case that moves leftShoulder/rightShoulder. Without it the clavicle joint class has n=0 and cannot be calibrated at all; planner.py's 'shrug' token is the only route to BodyPoseTarget.left_shoulder_elevation_deg.",
    ),
    SeedCase(
        id="fullbody-turn-left",
        family=Family.FULL_BODY,
        prompt="turn around to your left",
        tags=["turn", "yaw"],
        notes="BodyAction.TURN; the only case with a large final root yaw.",
    ),
    SeedCase(
        id="fullbody-walk-forward",
        family=Family.FULL_BODY,
        prompt="walk forward four steps",
        tags=["walk", "gait", "root-travel"],
        notes="BodyAction.WALK; the gait path with sustained root travel.",
    ),
    # -- grasp: the MuJoCo path ----------------------------------------------------
    SeedCase(
        id="grasp-block-left",
        family=Family.GRASP,
        prompt="grab the block with your left hand",
        tags=["left", "mujoco", "contact"],
        notes="Left-hand mirror of the grasp lifecycle; the solver runs per hand.",
    ),
    SeedCase(
        id="grasp-block-overhead",
        family=Family.GRASP,
        prompt="pick up the block and hold it above your head",
        tags=["right", "mujoco", "contact", "overhead-lift"],
        notes="0.70 m lift rather than 0.12 m, and 165 contacts against the table case's 85.",
    ),
    SeedCase(
        id="grasp-block-table",
        family=Family.GRASP,
        prompt="pick up the block on the table",
        tags=["right", "mujoco", "contact", "flagship"],
        notes="The pickup demo prompt; Intent.GRAB runs simulate_grasp inside compilation.",
    ),
    # -- object interaction: all nine ObjectAction members ---------------------------
    SeedCase(
        id="object-catch-left",
        family=Family.OBJECT_INTERACTION,
        prompt="catch the block with your left hand",
        tags=["left", "catch", "intercept"],
        notes="ObjectAction.CATCH; the only case with a catch_intercept_error_m metric.",
    ),
    SeedCase(
        id="object-drop-right",
        family=Family.OBJECT_INTERACTION,
        prompt="drop the block",
        tags=["right", "drop", "release"],
        notes="ObjectAction.DROP; the shortest object travel at 0.08 m. Runs simulate_grasp.",
    ),
    SeedCase(
        id="object-handoff-to-left",
        family=Family.OBJECT_INTERACTION,
        prompt="pass the block to your left hand",
        tags=["handoff", "both-hands", "dual-contact"],
        notes="ObjectAction.HANDOFF; the only case with two attached hands and an attachment-slip pair.",
    ),
    SeedCase(
        id="object-place-gently",
        family=Family.OBJECT_INTERACTION,
        prompt="place the block down gently",
        tags=["right", "place", "toss-style"],
        notes="ObjectAction.PLACE, and the only case resolving ObjectInteractionStyle.TOSS.",
    ),
    SeedCase(
        id="object-pull-toward",
        family=Family.OBJECT_INTERACTION,
        prompt="pull the block toward you",
        tags=["right", "pull", "negative-z"],
        notes="ObjectAction.PULL; the only case with a negative travel direction.",
    ),
    SeedCase(
        id="object-push-forward",
        family=Family.OBJECT_INTERACTION,
        prompt="push the block forward",
        tags=["right", "push", "support-plane"],
        notes="ObjectAction.PUSH; support-plane travel with the object never leaving the table.",
    ),
    SeedCase(
        id="object-roll-across",
        family=Family.OBJECT_INTERACTION,
        prompt="roll the block across the table",
        tags=["right", "roll", "rolling-angle"],
        notes="ObjectAction.ROLL; the only case with object_measured_roll_turns.",
    ),
    SeedCase(
        id="object-spin-in-place",
        family=Family.OBJECT_INTERACTION,
        prompt="spin the block on the table",
        tags=["right", "spin", "support-spin"],
        notes="ObjectAction.SPIN; the only case with object_measured_support_spin_turns.",
    ),
    SeedCase(
        id="object-throw-far",
        family=Family.OBJECT_INTERACTION,
        prompt="throw the block far and high with your right hand",
        tags=["right", "throw", "ballistic", "overhand"],
        notes="ObjectAction.THROW at the 1.60 m far distance and the 0.65 m lofted apex. Runs simulate_grasp.",
    ),
    # -- sequence -------------------------------------------------------------------
    SeedCase(
        id="sequence-grab-then-wave",
        family=Family.SEQUENCE,
        prompt="pick up the block and then wave",
        tags=["sequence", "grab-then-gesture", "cross-path"],
        notes="The only sequence bridging the MuJoCo grasp path into a composite step.",
    ),
    SeedCase(
        id="sequence-push-then-pull",
        family=Family.SEQUENCE,
        prompt="push the block forward then pull the block toward you",
        tags=["sequence", "object-continuity", "two-object-steps"],
        notes="Object state survives the step boundary: the block pushed in step one is the block pulled in step two.",
    ),
    SeedCase(
        id="sequence-wave-then-pushup",
        family=Family.SEQUENCE,
        prompt="wave hello and then do a push-up",
        tags=["sequence", "gesture-then-fullbody", "longest"],
        notes="Bridges the composite path into the full-body path; 242 frames, the longest case in the corpus.",
    ),
    # -- known-bad: each case fails the gate it names, and only that gate -------------
    SeedCase(
        id="knownbad-composite-overhead-out-of-view",
        family=Family.KNOWN_BAD,
        prompt="open both hands and raise them overhead",
        tags=["hand-visibility", "no-overrides", "natural-failure"],
        notes="Fails on the unmodified prompt with no overrides at all: raising both hands overhead leaves the 94-degree egocentric FOV.",
        must_fail=(StructuralGate.HAND_VISIBILITY,),
    ),
    SeedCase(
        id="knownbad-eigenvalues-unsupported",
        family=Family.KNOWN_BAD,
        prompt="compute the eigenvalues of this matrix",
        tags=["unsupported", "planner-refusal", "zero-frames"],
        notes="Intent.UNSUPPORTED: no motion to gate, so the check is the planner's refusal itself. The only case with zero frames.",
    ),
    SeedCase(
        id="knownbad-gesture-out-of-view",
        family=Family.KNOWN_BAD,
        prompt=HANGTEN_PROMPT,
        tags=["hand-visibility", "overrides"],
        notes="The flagship gesture driven low, near and inboard until the active hand leaves frame.",
        overrides={"arm_height": -1.0, "arm_depth": -1.0, "lateral_offset": -1.0},
        must_fail=(StructuralGate.HAND_VISIBILITY,),
    ),
    SeedCase(
        id="knownbad-sequence-throw-then-catch",
        family=Family.KNOWN_BAD,
        prompt="throw the block then catch the block",
        tags=["carried-object-step", "sequence", "no-overrides", "natural-failure"],
        notes="Both steps are individually valid; the bridge between them teleports the block 1.38 m against a 0.08 m reference. No overrides.",
        must_fail=(StructuralGate.CARRIED_OBJECT_STEP,),
    ),
    SeedCase(
        id="knownbad-strike-hyperfast",
        family=Family.KNOWN_BAD,
        prompt="throw a left jab",
        tags=["angular-acceleration", "angular-jerk", "overrides"],
        notes="A jab compressed into 0.05 s: over the acceleration and jerk ceilings but still under the velocity one.",
        overrides={"duration_s": 0.05, "path_arc": 1.0},
        must_fail=(StructuralGate.ANGULAR_ACCELERATION, StructuralGate.ANGULAR_JERK),
    ),
    SeedCase(
        id="knownbad-travel-hyperfast",
        family=Family.KNOWN_BAD,
        prompt=TRAVEL_PROMPT,
        tags=["angular-velocity", "angular-acceleration", "angular-jerk", "overrides", "both-hands"],
        notes="The only case over all three kinematic ceilings, and the only one failing them on both arms at once. "
        "Re-authored 2026-08-27 so the cycle itself breaches the velocity ceiling: the committed case files "
        "(scene fps 60, setup 4.0 s, cycle 0.05 s) supersede this row, which freeze_from_seed cannot express — "
        "regenerate from evals/corpus/cases/knownbad-travel-hyperfast/, not from these overrides.",
        overrides={"duration_s": 0.05, "trajectory_cycles": 8.0, "axial_rotation_amplitude": 1.0},
        must_fail=(
            StructuralGate.ANGULAR_VELOCITY,
            StructuralGate.ANGULAR_ACCELERATION,
            StructuralGate.ANGULAR_JERK,
        ),
    ),
)

SEED_CASES = SEED_CASES_03A + SEED_CASES_03B
SEED_CASES_BY_ID: dict[str, SeedCase] = {case.id: case for case in SEED_CASES}
