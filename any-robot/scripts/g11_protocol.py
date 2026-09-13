"""Register the G11 protocol before any scored run: validation sets, layouts, context changes, restart boundaries.

Every number a certificate will rest on is fixed here and hashed: the
independent validation set per enabled body (twelve seeds no development
set used, drawn by the G06 rule from a generator seeded apart, eleven
successes required and no false completion), the smaller revalidation sets
for each context change on the jaw arm, the three object and layout
instances the promoted skills are reused in, the five changes -- one per
dimension the catalog names -- and the leaf boundaries every body is
interrupted and restarted at. The campaign refuses a protocol that no
longer hashes to this registration.

    python any-robot/scripts/g11_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rigby_core.skills import safe_boundaries
from rigby_core.skills.examples import transfer_object_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import load_policy, policy_digest
from rigby_general.skills.skill_store import CLAIMED_RANGES, DEVELOPMENT_SETS, EVIDENCE_SCHEMA, FRICTION_ASSUMPTION, draw_validation_set, goal_for, layouts

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "assets/general/research-protocols/g11-skill-store-v1"
BODIES = ("zoo_dual_arm", "zoo_jaw_arm", "zoo_long_arm")
CHANGE_BODY = "zoo_jaw_arm"
VALIDATION_SEEDS = tuple(range(1000, 1012))
VALIDATION_THRESHOLD = 11
REVALIDATION_SEEDS = tuple(range(2000, 2006))
REVALIDATION_THRESHOLD = 5
RNG_VALIDATION = 20261111
RNG_REVALIDATION = 20262222
CONFIGURATION = "front_overhead_contact"
ARGUMENTS = {"object": "cube", "destination": "platform"}
SKILLS = ("transfer_object", "acquire_until_held", "observe_object")
"""The root and the two subskills promoted and reused from the store."""

CHANGES = {
    "geometry": {"dimension": "geometry", "declared": "the cube grows from 30 mm to 35 mm a side, its mass with its volume; every fixture as it was",
                 "object_half_size_m": 0.0175, "mass_scale": (0.0175 / 0.015) ** 3},
    "controller": {"dimension": "controller", "declared": "the arm's computed-torque tracking runs at half its natural frequency (7 Hz instead of 14 Hz), damping ratio unchanged",
                   "natural_frequency_hz": 7.0, "damping_ratio": 1.2},
    "sensors": {"dimension": "sensors", "declared": "the overhead camera is removed: the front camera and the gripper's contact sensor alone", "configuration": "front_contact"},
    "friction": {"dimension": "friction", "declared": "the friction assumption drops to 0.5-0.9 and the revalidation cubes are drawn at half the registered friction",
                 "object_friction_range": [0.5, 0.9], "friction_scale": 0.5},
    "evidence_schema": {"dimension": "evidence_schema", "declared": "the episode protocol is bumped to /2; nothing physical changes",
                        "episode_protocol": "rigby.transfer-object-episode/2"},
}


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    library = transfer_object_library()
    environment = g10.environment()
    goal = goal_for(environment)
    registered = g10.registered_goal()
    if tuple(round(v, 6) for v in goal.region_minimum_m) != tuple(round(v, 6) for v in registered.region_minimum_m) or tuple(round(v, 6) for v in goal.region_maximum_m) != tuple(round(v, 6) for v in registered.region_maximum_m):
        raise SystemExit("goal_for does not reproduce the registered G06 goal on the registered world")
    policy = load_policy(g10.G09 / "policy.json")

    sets = {"schema": "g11.validation-sets.v1", "development_sets": list(DEVELOPMENT_SETS), "rule": "the G06 draw rule (translation +-2 cm, mass and friction 0.8-1.2 of the registered cube) from numpy.random.default_rng seeded apart from every development set, one stream per set, seeds no development set used",
            "validation": {}, "revalidation": {}}
    for body in BODIES:
        validation, draws = draw_validation_set(body, set_id=f"g11-validation-{body}", seeds=VALIDATION_SEEDS, rng_seed=RNG_VALIDATION, threshold=VALIDATION_THRESHOLD)
        sets["validation"][body] = {"set": validation.model_dump(mode="json"), "draws": draws}
    for index, (name, change) in enumerate(CHANGES.items()):
        friction_range = tuple(change.get("object_friction_range", FRICTION_ASSUMPTION["object_friction_range"]))
        scale = change.get("friction_scale", 1.0)
        validation, draws = draw_validation_set(CHANGE_BODY, set_id=f"g11-revalidation-{name}", seeds=REVALIDATION_SEEDS, rng_seed=RNG_REVALIDATION + index, threshold=REVALIDATION_THRESHOLD,
                                                friction_multiplier_range=(0.8 * scale, 1.2 * scale))
        sets["revalidation"][name] = {"set": validation.model_dump(mode="json"), "draws": draws, "friction_assumption": {"object_friction_range": list(friction_range), "basis": change["declared"]}}
    (PROTOCOL / "validation-sets.json").write_bytes(json_bytes(sets))

    worlds = {"schema": "g11.layouts.v1", "base_environment_id": environment.environment_id, "base_environment_sha256": hashlib.sha256(json_bytes(environment.model_dump(mode="json"))).hexdigest(),
              "claimed_ranges": [r.model_dump(mode="json") for r in CLAIMED_RANGES], "layouts": {}}
    for name, world in layouts(environment).items():
        layout_goal = goal_for(world)
        worlds["layouts"][name] = {"environment": world.model_dump(mode="json"), "goal": {"region_minimum_m": list(layout_goal.region_minimum_m), "region_maximum_m": list(layout_goal.region_maximum_m),
                                                                                              "dwell_s": layout_goal.dwell_s, "maximum_linear_speed_mps": layout_goal.maximum_linear_speed_mps, "maximum_angular_speed_radps": layout_goal.maximum_angular_speed_radps}}
    (PROTOCOL / "layouts.json").write_bytes(json_bytes(worlds))
    (PROTOCOL / "changes.json").write_bytes(json_bytes({"schema": "g11.changes.v1", "body": CHANGE_BODY, "changes": CHANGES, "evidence_schema": EVIDENCE_SCHEMA,
                                                         "rule": "each change is applied to a copy of the promoted store from the same baseline, on its own, and followed by its revalidation set; the geometry change is also applied to the persisted store as D11's physical-context change"}))
    tree = library.expand("transfer_object", {**ARGUMENTS, "effector": "chain_placeholder"})
    boundaries = [{"index": i + 1, "node_id": node_id, "skill_id": tree.node(node_id).skill_id} for i, node_id in enumerate(safe_boundaries(tree))]
    (PROTOCOL / "restart.json").write_bytes(json_bytes({"schema": "g11.restart.v1", "bodies": list(BODIES), "boundaries": boundaries, "episode_draw": "the first validation draw of each body (seed 1000)",
                                                         "rule": "the tree is run with a checkpoint requested as the k-th leaf completes; a fresh session is opened on the checkpoint's physical state, reconstructs its belief from the sensors alone, and runs the tree from its root; it completes or stops with a typed reason",
                                                         "restarts": len(BODIES) * len(boundaries)}))
    files = {name: hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() for name in ("validation-sets.json", "layouts.json", "changes.json", "restart.json")}
    registration = {"schema": "g11.registration.v1", "files": files, "library_id": library.library_id, "library_sha256": library.content_hash(), "policy_sha256": policy_digest(policy),
                    "environment_sha256": worlds["base_environment_sha256"], "sensor_configuration": CONFIGURATION, "skills": list(SKILLS), "bodies": list(BODIES), "change_body": CHANGE_BODY,
                    "validation": {"seeds": list(VALIDATION_SEEDS), "threshold": VALIDATION_THRESHOLD, "rng_seed": RNG_VALIDATION}, "revalidation": {"seeds": list(REVALIDATION_SEEDS), "threshold": REVALIDATION_THRESHOLD, "rng_seed": RNG_REVALIDATION},
                    "generation_calls": 0, "registered_at_utc": datetime.now(timezone.utc).isoformat(), "note": "registered before any scored run; the campaign refuses a protocol, library, policy or world that does not hash to this"}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "validation_sets": len(sets["validation"]), "revalidation_sets": len(sets["revalidation"]), "layouts": list(worlds["layouts"]), "restarts": len(BODIES) * len(boundaries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
