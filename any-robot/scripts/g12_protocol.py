"""Freeze the G12 acquisition problems before any search runs.

Three problems in three body families, each a transfer the library cannot
certify as it stands: the jaw arm with the cube at half its registered
friction (the G11 friction test failed it six of six); the multifinger hand
in the fixed world (G06 recorded it failing every seed, the fingertip pinch
losing the cube as the lift decelerates); and the long-reach arm with a
grown, heavier cube, posed as a control the existing implementation is
expected to pass without search, so that an instantiation is recorded as
one. The admissible family is the contact transfer's own parameters: how
the closure advances, detects contact and squeezes, how the arm tracks, and
how much slower than the declared joint speeds the moving phases run. The
development and confirmation draws the search may see, the fifty held-out
draws it may not, the acceptance threshold and the ceiling are all fixed
here and hashed; the campaign refuses a protocol that no longer matches.

    python any-robot/scripts/g12_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rigby_core.skills import AcceptanceV1, AcquisitionProblemV1, CeilingV1, EffectV1, ParameterSpecV1, ProblemKind
from rigby_core.skills.examples import transfer_object_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import load_policy, policy_digest
from rigby_general.skills.skill_store import draw_validation_set

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "assets/general/research-protocols/g12-acquisition-v1"
DEVELOPMENT_SEEDS = (4000, 4001)
CONFIRMATION_SEEDS = (4002, 4003)
HOLDOUT_SEEDS = tuple(range(3000, 3050))
RNG_DEVELOPMENT = 20264444
RNG_HOLDOUT = 20263333
SEARCH_SEED = 20261212
CEILING = CeilingV1(attempts=200, worker_minutes=30.0)
ACCEPTANCE_THRESHOLD = 40

PARAMETERS = (
    ParameterSpecV1(name="closure.grip_safety_factor", low=10.0, high=400.0, default=80.0, units="ratio", scale="log", description="multiplier on the statically required grip force"),
    ParameterSpecV1(name="closure.closing_force_fraction", low=0.03, high=0.4, default=0.12, units="fraction", description="of the weakest closure actuator's limit, to move the members through the approach"),
    ParameterSpecV1(name="closure.actuator_ceiling_fraction", low=0.2, high=0.9, default=0.5, units="fraction", description="never more than this fraction of what the joint can produce"),
    ParameterSpecV1(name="closure.closure_rate_per_s", low=0.1, high=1.0, default=0.55, units="span/s", description="fraction of the joint's span advanced per second while closing"),
    ParameterSpecV1(name="closure.hold_duration_s", low=0.1, high=0.5, default=0.15, units="s", description="how long opposition must be held before the grip is called established"),
    ParameterSpecV1(name="arm.natural_frequency_hz", low=5.0, high=25.0, default=14.0, units="Hz", description="the computed-torque tracking bandwidth"),
    ParameterSpecV1(name="duration_scale", low=1.0, high=4.0, default=1.0, units="ratio", description="how much slower than the declared joint speeds the moving phases run"),
)

PROBLEMS = (
    AcquisitionProblemV1(
        problem_id="jaw-slick", body="zoo_jaw_arm", body_family="single-arm jaw gripper", effect=EffectV1(skill_id="transfer_object", arguments={"object": "cube", "destination": "platform"}),
        world_change="the cube at half its registered friction (0.7 against 1.4): the G11 friction revalidation failed 0/6 with the implementation as it stands",
        hypothesis=ProblemKind.DISCOVERY, controller_family="contact.transfer", parameters=PARAMETERS, development_seeds=DEVELOPMENT_SEEDS, confirmation_seeds=CONFIRMATION_SEEDS,
        acceptance=AcceptanceV1(set_id="g12-holdout-jaw-slick", trials=len(HOLDOUT_SEEDS), threshold=ACCEPTANCE_THRESHOLD), ceiling=CEILING,
        notes="new contact and timing parameters are the expected answer: more grip, a slower lift and carry",
    ),
    AcquisitionProblemV1(
        problem_id="hand-fixed-world", body="zoo_hand_arm", body_family="multifinger hand", effect=EffectV1(skill_id="transfer_object", arguments={"object": "cube", "destination": "platform"}),
        world_change="none: the registered fixed world, in which G06 recorded the hand failing all 100 seeds (94 hold_not_sustained, 6 object_not_lifted) with the fingertip pinch losing the cube as the lift decelerates",
        hypothesis=ProblemKind.DISCOVERY, controller_family="contact.transfer", parameters=PARAMETERS, development_seeds=DEVELOPMENT_SEEDS, confirmation_seeds=CONFIRMATION_SEEDS,
        acceptance=AcceptanceV1(set_id="g12-holdout-hand-fixed-world", trials=len(HOLDOUT_SEEDS), threshold=ACCEPTANCE_THRESHOLD), ceiling=CEILING,
        notes="G06's diagnosis names the inertial load at the top of the lift; a slower lift is inside the family, a wrap grasp or a compliant pad is not",
    ),
    AcquisitionProblemV1(
        problem_id="long-heavy-large", body="zoo_long_arm", body_family="single-arm long-reach jaw gripper", effect=EffectV1(skill_id="transfer_object", arguments={"object": "cube", "destination": "platform"}),
        world_change="the cube grown to 35 mm a side and made half again heavier than its volume would make it (about 20.6 g against 8.6 g)",
        hypothesis=ProblemKind.INSTANTIATION, controller_family="contact.transfer", parameters=PARAMETERS, development_seeds=DEVELOPMENT_SEEDS, confirmation_seeds=CONFIRMATION_SEEDS,
        acceptance=AcceptanceV1(set_id="g12-holdout-long-heavy-large", trials=len(HOLDOUT_SEEDS), threshold=ACCEPTANCE_THRESHOLD), ceiling=CEILING,
        notes="posed as a control: the defaults are expected to pass, and the outcome must then say instantiation, not discovery",
    ),
)
CHANGES = {
    "jaw-slick": {"friction_scale": 0.5},
    "hand-fixed-world": {},
    "long-heavy-large": {"object_half_size_m": 0.0175, "mass_scale": 1.5},
}
FRICTION_ASSUMPTIONS = {"jaw-slick": {"object_friction_range": [0.5, 0.9], "basis": "the G11 friction change: half the registered coefficient, with the draw's 0.8-1.2"}}


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    library = transfer_object_library()
    policy = load_policy(g10.G09 / "policy.json")
    environment = g10.environment()
    problems = {"schema": "g12.problems.v1", "problems": [p.model_dump(mode="json") for p in PROBLEMS], "changes": CHANGES, "friction_assumptions": FRICTION_ASSUMPTIONS,
                "search": {"algorithm": "one-plus-lambda-es", "seed": SEARCH_SEED, "offspring": 4, "sigma0": 0.25, "contraction": 0.85, "minimum_sigma": 0.02},
                "rule": "every evaluation is the G06 transfer primitive single-shot with every hard gate on; the search sees the development and confirmation draws only; the held-out draws run once with the settled vector; zero manual trajectory edits",
                "classification": "instantiation when attempt 0 (the defaults) passes and no parameter moved; discovery when a searched vector passes; budget_exhausted with a hypothesis about the limiting capability otherwise; a composition is a new arrangement of existing leaves and none is posed here (G11's reuse is the composition evidence)"}
    (PROTOCOL / "problems.json").write_bytes(json_bytes(problems))
    draws = {"schema": "g12.draws.v1", "rule": "the G06 draw rule from numpy.random.default_rng seeded apart from every development set, one stream per set, in seed order", "problems": {}}
    for problem in PROBLEMS:
        friction = CHANGES[problem.problem_id].get("friction_scale", 1.0)
        dev_set, dev_draws = draw_validation_set(problem.body, set_id=f"g12-development-{problem.problem_id}", seeds=DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS, rng_seed=RNG_DEVELOPMENT, threshold=len(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS),
                                                  independent_of=("g10-transfer-v1:nominal-seeds-0-99", "g06-transfer-v1:roster-seeds-0-99", "g11-skill-store-v1:validation-seeds-1000-1011"))
        hold_set, hold_draws = draw_validation_set(problem.body, set_id=problem.acceptance.set_id, seeds=HOLDOUT_SEEDS, rng_seed=RNG_HOLDOUT, threshold=problem.acceptance.threshold,
                                                    independent_of=("g10-transfer-v1:nominal-seeds-0-99", "g06-transfer-v1:roster-seeds-0-99", "g11-skill-store-v1:validation-seeds-1000-1011", f"g12-development-{problem.problem_id}"))
        draws["problems"][problem.problem_id] = {"development": {"set": dev_set.model_dump(mode="json"), "draws": dev_draws}, "holdout": {"set": hold_set.model_dump(mode="json"), "draws": hold_draws},
                                                 "note": "the friction draw multiplies the problem world's coefficient, so a slick problem's draws stay slick" if friction != 1.0 else ""}
    (PROTOCOL / "draws.json").write_bytes(json_bytes(draws))
    files = {name: hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() for name in ("problems.json", "draws.json")}
    registration = {"schema": "g12.registration.v1", "files": files, "library_id": library.library_id, "library_sha256": library.content_hash(), "policy_sha256": policy_digest(policy),
                    "environment_sha256": hashlib.sha256(json_bytes(environment.model_dump(mode="json"))).hexdigest(), "problems": [p.problem_id for p in PROBLEMS],
                    "ceiling": CEILING.model_dump(mode="json"), "acceptance_threshold": ACCEPTANCE_THRESHOLD, "holdout_trials": len(HOLDOUT_SEEDS), "search_seed": SEARCH_SEED,
                    "generation_calls": 0, "registered_at_utc": datetime.now(timezone.utc).isoformat(), "note": "registered before any search; the campaign refuses a protocol, library, policy or world that does not hash to this"}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "problems": registration["problems"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
