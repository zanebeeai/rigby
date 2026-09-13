"""Register the G10 protocol: the world, the bodies, the seeds, the disturbance schedule, the cap, the library and the policy.

Everything a scored run depends on is written here and hashed before any
scored run: the G06 fixture standing on a floor; the three enabled bodies
the G08 corpus certified transitions on; the G06 registered seed draws for
the nominal episodes (one hundred per body) and the first twenty for each
disturbance class; the disturbances themselves as declared (a push on the
object during the first approach, a pull on the held object early in the
first carry sized from the sensed grip force, an occluder between the front
camera and the fixtures for three seconds when the first placement
verification begins); the frozen 120 s episode cap; the sensor
configuration the skill runs with; the neutral library by content hash;
the G09 policy by hash. No API or model calls.

    python any-robot/scripts/g10_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rigby_core.skills.examples import EPISODE_CAP_S, RETRY_BUDGET, transfer_object_library

from rigby_general.contact.placement import PlacementGoal
from rigby_general.evidence.capture import json_bytes
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import configuration
from rigby_general.skills.transfer_object import FLOOR_TOP_M, SHUTTER_ACTIVE, SHUTTER_HALF, DisplacedObject, InducedSlip, TemporaryOcclusion, with_floor


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / "assets/general/research-protocols/g10-transfer-v1"
G06 = ROOT / "assets/general/research-protocols/g06-transfer-v1"
G09 = ROOT / "assets/general/research-protocols/g09-conditionals-v1"
BODIES = ("zoo_dual_arm", "zoo_jaw_arm", "zoo_long_arm")
CLASSES = ("nominal", "displaced", "slip", "occlusion")
NOMINAL_PER_BODY = 100
DISTURBED_PER_CLASS = 20
CONFIGURATION = "front_contact"


def registered_goal() -> PlacementGoal:
    g = json.loads((G06 / "goal.json").read_bytes())
    return PlacementGoal(region_minimum_m=tuple(g["region_minimum_m"]), region_maximum_m=tuple(g["region_maximum_m"]), dwell_s=g["dwell_s"],
                         maximum_linear_speed_mps=g["maximum_linear_speed_mps"], maximum_angular_speed_radps=g["maximum_angular_speed_radps"])


def environment() -> EnvironmentV1:
    return with_floor(EnvironmentV1.model_validate_json((G06 / "environment.json").read_bytes()))


def load_registration() -> dict:
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        path = PROTOCOL / name if not name.startswith(("g06/", "g09/")) else (G06 if name.startswith("g06/") else G09) / name.split("/", 1)[1]
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} does not match its registration; refusing to run")
    corpus = json.loads((PROTOCOL / "corpus.json").read_bytes())
    library = transfer_object_library()
    if library.content_hash() != corpus["library_sha256"]:
        raise SystemExit("the transfer_object library no longer hashes to the registered corpus; refusing to run")
    if hashlib.sha256(json_bytes(environment().model_dump(mode="json"))).hexdigest() != corpus["environment_sha256"]:
        raise SystemExit("the world no longer hashes to the registered corpus; refusing to run")
    return corpus


def main() -> int:
    env = environment()
    roster = json.loads((G06 / "roster.json").read_bytes())
    seeds = {b["zoo_id"]: b["seeds"] for b in roster["bodies"]}
    library = transfer_object_library()
    episodes = []
    for body in BODIES:
        draws = seeds[body]
        if len(draws) < NOMINAL_PER_BODY:
            raise SystemExit(f"{body}: the G06 roster registers only {len(draws)} seeds")
        for draw in draws[:NOMINAL_PER_BODY]:
            episodes.append({"episode_id": f"{body}-nominal-{draw['seed']:03d}", "zoo_id": body, "class": "nominal", "seed": draw["seed"], "draw": draw})
        for kind in CLASSES[1:]:
            for draw in draws[:DISTURBED_PER_CLASS]:
                episodes.append({"episode_id": f"{body}-{kind}-{draw['seed']:03d}", "zoo_id": body, "class": kind, "seed": draw["seed"], "draw": draw})
    disturbances = {
        "displaced": {**{k: v for k, v in DisplacedObject().__dict__.items() if not k.startswith("_") and k not in ("log", "done")},
                      "declared": "a constant force on the object, applied through the step hook while the first acquisition approaches, recorded as user input"},
        "slip": {**{k: v for k, v in InducedSlip().__dict__.items() if not k.startswith("_") and k not in ("log", "done")},
                 "declared": "a downward pull on the held object early in the first carry: gain x 2 x friction x the grip force the contact sensor reports, plus extra, capped"},
        "occlusion": {**{k: v for k, v in TemporaryOcclusion().__dict__.items() if not k.startswith("_") and k not in ("log", "done")},
                      "shutter_position_m": list(SHUTTER_ACTIVE), "shutter_half_m": list(SHUTTER_HALF),
                      "declared": "a mocap occluder between the front camera and the fixtures from the start of the first placement verification, parked again after"},
    }
    reference = configuration(CONFIGURATION, ())
    corpus = {
        "schema": "g10.transfer-corpus.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment_sha256": hashlib.sha256(json_bytes(env.model_dump(mode="json"))).hexdigest(), "environment_id": env.environment_id, "floor_top_m": FLOOR_TOP_M,
        "library_id": library.library_id, "library_sha256": library.content_hash(), "root_skill": "transfer_object", "retry_budget": RETRY_BUDGET, "episode_cap_s": EPISODE_CAP_S,
        "policy_sha256": hashlib.sha256((G09 / "policy.json").read_bytes()).hexdigest(), "sensor_configuration": CONFIGURATION,
        "sensor_configuration_description": reference.description, "bodies": list(BODIES), "classes": list(CLASSES),
        "nominal_per_body": NOMINAL_PER_BODY, "disturbed_per_class": DISTURBED_PER_CLASS, "disturbances": disturbances, "episodes": episodes,
        "success_targets": {"nominal_per_body": 90, "recovered_per_class_per_body": 16, "false_completions": 0},
        "scored_runs_started": False, "generation_calls": 0,
    }
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    (PROTOCOL / "environment.json").write_bytes(env.model_dump_json(indent=2).encode("utf-8"))
    (PROTOCOL / "library.json").write_bytes(library.model_dump_json(indent=2).encode("utf-8"))
    (PROTOCOL / "corpus.json").write_bytes(json_bytes(corpus))
    files = {name: hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() for name in ("corpus.json", "environment.json", "library.json")}
    files["g06/goal.json"] = hashlib.sha256((G06 / "goal.json").read_bytes()).hexdigest()
    files["g06/roster.json"] = hashlib.sha256((G06 / "roster.json").read_bytes()).hexdigest()
    files["g09/policy.json"] = hashlib.sha256((G09 / "policy.json").read_bytes()).hexdigest()
    listing = json.dumps(files, indent=2, sort_keys=True, allow_nan=False) + "\n"
    registration = {"schema": "g10.registration.v1", "registered_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                    "registration_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(), "episodes": len(episodes),
                    "note": "registered before any scored run; the campaign refuses a corpus, world, library or policy that does not hash to this"}
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"episodes": len(episodes), "registration_sha256": registration["registration_sha256"], "library_sha256": library.content_hash()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
