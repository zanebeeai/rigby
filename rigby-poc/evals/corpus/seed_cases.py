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

from .models import Family


@dataclass(frozen=True)
class SeedCase:
    id: str
    family: Family
    prompt: str
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


SEED_CASES: tuple[SeedCase, ...] = (
    SeedCase(
        id="composite-travel-forearms",
        family=Family.COMPOSITE,
        prompt=(
            "roll your forearms around eachother repeatedly, as if you are "
            'indicating the "travel" foul in basketball'
        ),
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
        prompt=(
            'Throw up a "hang-ten" sign with your right hand, there should be a '
            "swift motion up to the main position wherein the middle three fingers "
            "are as contracted as possible, the wrist should then shake rapidly "
            "back and forth a few times, before returning to default"
        ),
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

SEED_CASES_BY_ID: dict[str, SeedCase] = {case.id: case for case in SEED_CASES}
