"""Register the G15 protocol before any scored run: worlds, chain lengths, paired seeds, disturbances, caps, targets.

Three chain lengths on three frozen enabled bodies; fifty nominal seeds per
chain length, the same fifty on every body and every chain length (a seed
draws jitter, mass and friction for twelve objects, of which a world uses
its first three, five or ten); thirty disturbed seeds for five objects,
each naming the object and the kind of disturbance, run once through the
generated tree and once through its flat twin. The caps, the retry and
pass budgets, and the targets are fixed here and hashed; the campaign
refuses a protocol that no longer matches.

Version 3. The first registration (g15-clearance-v1, kept beside this one
for its baseline evidence) put neighbours eight centimetres apart, inside
the long arm's open jaw, and closed the clearing loop on the transfers'
own verdicts, so a cube a later placement knocked out of its cell stayed
"placed". The second (g15-clearance-v2, kept as registered, its run
stopped) put neighbours twelve centimetres apart at positions within
every body's reach envelope and ended every pass with a look; its run
found that the envelope is not the solver -- the dual arm's IK misses two
of those positions by four millimetres every time -- and that the arm
parked over the platform hides cubes from both cameras. This one keeps
the pitch, takes only positions every enabled body has transferred from
and to alone (the qualification table is registered with the layout),
and has the arm stand clear before every look. The draws are unchanged:
the same seed draws the same jitter, mass and friction under all three.

    python any-robot/scripts/g15_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rigby_core.skills.clearance import PASS_BUDGET, cell_names, clear_work_area_library, episode_cap_s, object_names
from rigby_core.skills.examples import RETRY_BUDGET

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import load_policy, policy_digest
from rigby_general.skills.clear_work_area import LAYOUT, build_clearance_world

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "assets/general/research-protocols/g15-clearance-v3"
PROTOCOL_V2 = ROOT / "assets/general/research-protocols/g15-clearance-v2"
PROTOCOL_V1 = ROOT / "assets/general/research-protocols/g15-clearance-v1"
QUALIFICATION = PROTOCOL / "qualification.json"
"""Every candidate position's single-object transfer on every enabled body: a slot with the reference cell, a cell with the reference slot; the layout takes the ones every body succeeded at."""
BODIES = ("zoo_dual_arm", "zoo_jaw_arm", "zoo_long_arm")
CHAIN_LENGTHS = (3, 5, 10)
NOMINAL_SEEDS = tuple(range(5000, 5050))
DISTURBED_SEEDS = tuple(range(6000, 6030))
DISTURBED_COUNT = 5
DISTURBANCE_KINDS = ("displaced", "slip", "occlusion")
RNG_NOMINAL = 20265555
RNG_DISTURBED = 20266666
OBJECTS_PER_DRAW = 12
TRANSLATION_HALF_WIDTH_M = 0.01
MASS_MULTIPLIER_RANGE = (0.9, 1.1)
FRICTION_MULTIPLIER_RANGE = (0.9, 1.1)
TARGETS = {3: 0.90, 5: 0.80, 10: 0.60}
RECOVERY_GAIN_TARGET_PP = 15.0
CONFIGURATION = "front_overhead_contact"


def draw_objects(rng: np.random.Generator) -> list[dict]:
    return [{"translation_m": [float(rng.uniform(-TRANSLATION_HALF_WIDTH_M, TRANSLATION_HALF_WIDTH_M)), float(rng.uniform(-TRANSLATION_HALF_WIDTH_M, TRANSLATION_HALF_WIDTH_M)), 0.0],
             "mass_multiplier": float(rng.uniform(*MASS_MULTIPLIER_RANGE)), "friction_multiplier": float(rng.uniform(*FRICTION_MULTIPLIER_RANGE))} for _ in range(OBJECTS_PER_DRAW)]


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    base = g10.environment()
    policy = load_policy(g10.G09 / "policy.json")
    rng = np.random.default_rng(RNG_NOMINAL)
    nominal = [{"seed": seed, "objects": draw_objects(rng)} for seed in NOMINAL_SEEDS]
    rng = np.random.default_rng(RNG_DISTURBED)
    disturbed = []
    for index, seed in enumerate(DISTURBED_SEEDS):
        objects = draw_objects(rng)
        disturbed.append({"seed": seed, "objects": objects, "kind": DISTURBANCE_KINDS[index % len(DISTURBANCE_KINDS)], "object_index": int(rng.integers(0, DISTURBED_COUNT))})
    worlds = {}
    libraries = {}
    for count in CHAIN_LENGTHS:
        world = build_clearance_world(base, count)
        worlds[count] = {"environment": world.environment.model_dump(mode="json"), "environment_sha256": hashlib.sha256(json_bytes(world.environment.model_dump(mode="json"))).hexdigest(),
                         "objects": list(world.objects), "cells": dict(world.cells), "offsets": {c: list(v) for c, v in world.offsets.items()},
                         "goals": {c: {"region_minimum_m": list(world.goal_for(c).region_minimum_m), "region_maximum_m": list(world.goal_for(c).region_maximum_m)} for c in world.cells.values()},
                         "cap_s": episode_cap_s(count)}
        for flat in (False, True):
            library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects), flat=flat)
            tree = library.expand("clear_work_area", {"effector": "chain_placeholder"})
            libraries[f"{count}-{'flat' if flat else 'tree'}"] = {"library_id": library.library_id, "sha256": library.content_hash(), "definitions": len(library.skills), "nodes": sum(1 for _ in tree.root.walk()),
                                                                   "max_depth": tree.max_depth, "transfer_instances": sum(1 for n in tree.root.walk() if n.skill_id == "transfer_object")}
    payload = {
        "schema": "g15.clearance-corpus.v3", "bodies": list(BODIES), "chain_lengths": list(CHAIN_LENGTHS), "sensor_configuration": CONFIGURATION,
        "layout": LAYOUT.as_json(), "qualification_sha256": hashlib.sha256(QUALIFICATION.read_bytes()).hexdigest(),
        "tree": {"observe_each_pass": True, "note": "every pass ends with the arm standing clear (its reference configuration) and a look at the area and the cells from the declared cameras; what they see decides which placements stand"},
        "worlds": {str(k): v for k, v in worlds.items()}, "libraries": libraries,
        "revision": {"supersedes": ["g15-clearance-v2", "g15-clearance-v1"],
                     "why": ["v1 put neighbours 8 cm apart, inside the long arm's open jaw (17.2 cm across the outer finger faces): its finger came down on the next cube",
                             "v1 closed the clearing loop on the transfers' verdicts, so a cube a later placement knocked out of its cell stayed placed and the root reported success",
                             "v1's far slot row lay beyond the dual arm's reach",
                             "v2 took positions from the reach envelope; the dual arm's IK missed two of them by four millimetres every time (its fifth slot failed in every five-object episode)",
                             "v2 looked with the arm parked over the platform, which hid cubes from both cameras: the long arm placed all ten and the look stayed undecided",
                             "v3 takes only positions every enabled body has transferred from and to alone, and stands clear before every look"],
                     "draws_unchanged": True},
        "draw": {"nominal_seed": RNG_NOMINAL, "disturbed_seed": RNG_DISTURBED, "objects_per_draw": OBJECTS_PER_DRAW, "translation_half_width_m": TRANSLATION_HALF_WIDTH_M, "mass_multiplier_range": list(MASS_MULTIPLIER_RANGE),
                 "friction_multiplier_range": list(FRICTION_MULTIPLIER_RANGE), "rule": "numpy.random.default_rng(seed), one stream, per seed twelve objects drawn in order (translation xy, mass, friction); a world of n objects uses the first n; paired: the same seed on every body and every chain length"},
        "nominal": nominal, "disturbed": disturbed, "disturbed_count": DISTURBED_COUNT, "disturbance_kinds": list(DISTURBANCE_KINDS),
        "budgets": {"retry_budget": RETRY_BUDGET, "pass_budget": PASS_BUDGET, "cap_s": {str(k): episode_cap_s(k) for k in CHAIN_LENGTHS}, "flat": "the same leaves with a leaf-level retry budget equal to the loop budget, one pass, no selector"},
        "targets": {"nominal": {str(k): v for k, v in TARGETS.items()}, "recovery_gain_pp": RECOVERY_GAIN_TARGET_PP},
        "success": "root verdict success from the executor (every object's placement decided pass from the sensors and the area observed clear), and the oracle agrees from the final poses; a success the oracle denies is a false completion",
        "generation_calls": 0,
    }
    (PROTOCOL / "corpus.json").write_bytes(json_bytes(payload))
    files = {"corpus.json": hashlib.sha256((PROTOCOL / "corpus.json").read_bytes()).hexdigest(), "g09/policy.json": hashlib.sha256((g10.G09 / "policy.json").read_bytes()).hexdigest()}
    files["qualification.json"] = hashlib.sha256(QUALIFICATION.read_bytes()).hexdigest()
    registration = {"schema": "g15.registration.v3", "files": files, "policy_sha256": policy_digest(policy), "base_environment_sha256": hashlib.sha256(json_bytes(base.model_dump(mode="json"))).hexdigest(),
                    "worlds": {str(k): v["environment_sha256"] for k, v in worlds.items()}, "layout": LAYOUT.name,
                    "libraries": {k: v["sha256"] for k, v in libraries.items()}, "episodes": {"nominal": len(BODIES) * len(CHAIN_LENGTHS) * len(NOMINAL_SEEDS), "disturbed": len(BODIES) * len(DISTURBED_SEEDS) * 2},
                    "generation_calls": 0, "registered_at_utc": datetime.now(timezone.utc).isoformat(), "note": "registered before any scored run; the campaign refuses a corpus, policy, world or library that does not hash to this"}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "episodes": registration["episodes"], "libraries": {k: (v["max_depth"], v["nodes"]) for k, v in libraries.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
