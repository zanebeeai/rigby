"""Register the G18 loco-manipulation protocol before any scored run: the bodies, the route, the seeds, the disturbances, the caps, the targets.

Every body runs the course's retrieve goal -- approach the cube on the
station, stabilise, acquire it, carry it to the tray, place it inside
the rim, return to the start pad -- from a hundred frozen seeds, each
drawing the start jitter (10 cm, 10 degrees), under a cap per body.
Thirty predetermined disturbances per body follow, on their own seeds,
spread over the three things that can go wrong on the way: ten during
navigation (pushes on the base, and the support disturbance the body's
mechanics call for), fourteen while carrying (pushes on the base, forces
on the held object, a support disturbance with the object in hand), six
during placement (pushes and a force on the object as it is set down).
The bodies are the dog and the biped in their second version (the jaw
hung below the last arm link, a correction found in G18: the first
version's wrist capsule ran between the fingers, so a jaw could take a
30 mm object by its top few millimetres only) and the octopus in its
second version too (pincers that open wide: the first version's parted
less than the object). The targets are the goal's: at least 80 of 100 nominal
trials complete; at least 24 of 30 disturbed trials recover and
complete; every other trial fails explicitly.

    python any-robot/scripts/g18_protocol.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body
from rigby_general.mobility.locomotion import make_locomotor
from rigby_general.mobility.manipulation import manipulators_of
from rigby_general.mobility.retrieve import PHASES, RetrieveSession, grasp_poses
from rigby_general.mobility.world import course_v1


ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "assets/general/mobile"
PROTOCOL = ROOT / "assets/general/research-protocols/g18-retrieve-v1"
BODIES = ("mobile_dog_arm_v2", "mobile_wheeled_biped_v2", "mobile_octopus_v2")
NOMINAL_SEEDS = tuple(range(1, 101))
DISTURBED_SEEDS = tuple(range(201, 231))
CAPS_S = {"mobile_dog_arm_v2": 360.0, "mobile_wheeled_biped_v2": 200.0, "mobile_octopus_v2": 800.0}
JITTER = {"xy_m": 0.10, "yaw_deg": 10.0}
SETTLE_S = 1.5
RETRY_BUDGET = 2
PUSH_N = {"mobile_dog_arm_v2": (40.0, 70.0), "mobile_wheeled_biped_v2": (30.0, 55.0), "mobile_octopus_v2": (100.0, 200.0)}
PUSH_AT_S = {"mobile_dog_arm_v2": 4.0, "mobile_wheeled_biped_v2": 4.0, "mobile_octopus_v2": 8.0}
CARRY_AT_S = {"mobile_dog_arm_v2": 6.0, "mobile_wheeled_biped_v2": 6.0, "mobile_octopus_v2": 12.0}
PLACE_AT_S = 4.0
DIRECTIONS = {"left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0), "fore": (1.0, 0.0, 0.0), "aft": (-1.0, 0.0, 0.0)}
OBJECT_FORCES = ((2.0, (0.0, 1.0, 0.0), "sideways"), (2.0, (0.0, 0.0, -1.0), "downward"), (4.0, (0.0, 1.0, 0.0), "sideways"), (4.0, (0.0, 0.0, -1.0), "downward"))
TARGETS = {"nominal_min": 80, "nominal_of": 100, "disturbed_min": 24, "disturbed_of": 30}


def support_case(body_id: str, phase: str, at: float) -> dict:
    if body_id == "mobile_dog_arm_v2":
        return {"kind": "support", "detail": f"leg_fl's servos fought by a 16 N m external torque at 3 Hz for 1.0 s, {at:.0f} s into {phase}", "phase": phase, "offset_s": at, "duration_s": 1.0, "magnitude": 16.0, "target": "leg_fl", "mechanics": "legged: a leg loses its servo authority"}
    if body_id == "mobile_wheeled_biped_v2":
        return {"kind": "support", "detail": f"the left wheel's drive cut by the controller for 0.15 s, {at:.0f} s into {phase}", "phase": phase, "offset_s": at, "duration_s": 0.15, "magnitude": 0.0, "target": "wheel:left_wheel_spin", "mechanics": "wheeled: a drive motor drops out"}
    return {"kind": "support", "detail": f"tentacle_3's servos fought by an 8 N m external torque at 3 Hz for 1.5 s, {at:.0f} s into {phase}", "phase": phase, "offset_s": at, "duration_s": 1.5, "magnitude": 8.0, "target": "tentacle_3", "mechanics": "crawling: a support tentacle loses its servo authority"}


def disturbance_plan(body_id: str) -> list[dict]:
    plan = []
    # navigation: 8 pushes and 2 support disturbances during the approach
    for name, direction in DIRECTIONS.items():
        for newton in PUSH_N[body_id]:
            plan.append({"kind": "push", "detail": f"{newton:.0f} N {name} push on the base for 0.3 s, {PUSH_AT_S[body_id]:.0f} s into the approach", "phase": "approach", "offset_s": PUSH_AT_S[body_id], "duration_s": 0.3, "magnitude": newton, "direction": list(direction), "target": "", "stage": "navigation"})
    plan.append({**support_case(body_id, "approach", PUSH_AT_S[body_id]), "stage": "navigation"})
    plan.append({**support_case(body_id, "approach", PUSH_AT_S[body_id] + 4.0), "stage": "navigation"})
    # carrying: 8 pushes, 4 forces on the held object, 2 support disturbances with the object in hand
    for name, direction in DIRECTIONS.items():
        for newton in PUSH_N[body_id]:
            plan.append({"kind": "push", "detail": f"{newton:.0f} N {name} push on the base for 0.3 s, {CARRY_AT_S[body_id]:.0f} s into the carry", "phase": "carry", "offset_s": CARRY_AT_S[body_id], "duration_s": 0.3, "magnitude": newton, "direction": list(direction), "target": "", "stage": "carrying"})
    for newton, direction, word in OBJECT_FORCES:
        plan.append({"kind": "object", "detail": f"{newton:.0f} N {word} force on the held object for 0.5 s, {CARRY_AT_S[body_id]:.0f} s into the carry", "phase": "carry", "offset_s": CARRY_AT_S[body_id], "duration_s": 0.5, "magnitude": newton, "direction": list(direction), "target": "", "stage": "carrying"})
    plan.append({**support_case(body_id, "carry", CARRY_AT_S[body_id]), "stage": "carrying"})
    plan.append({**support_case(body_id, "carry", CARRY_AT_S[body_id] + 6.0), "stage": "carrying"})
    # placement: 4 pushes and 2 forces on the object as it is set down
    for name in ("left", "aft"):
        for newton in PUSH_N[body_id]:
            plan.append({"kind": "push", "detail": f"{newton:.0f} N {name} push on the base for 0.3 s, {PLACE_AT_S:.0f} s into the placement", "phase": "place", "offset_s": PLACE_AT_S, "duration_s": 0.3, "magnitude": newton, "direction": list(DIRECTIONS[name]), "target": "", "stage": "placement"})
    for newton in (2.0, 4.0):
        plan.append({"kind": "object", "detail": f"{newton:.0f} N sideways force on the object for 0.5 s, {PLACE_AT_S + 4.0:.0f} s into the placement", "phase": "place", "offset_s": PLACE_AT_S + 4.0, "duration_s": 0.5, "magnitude": newton, "direction": [0.0, 1.0, 0.0], "target": "", "stage": "placement"})
    assert len(plan) == 30, (body_id, len(plan))
    return plan


def trials_for(body_id: str) -> list[dict]:
    trials = [{"trial_id": f"{body_id}-nominal-{seed:03d}", "body": body_id, "kind": "nominal", "stage": "nominal", "seed": seed, "cap_s": CAPS_S[body_id], "disturbance": None} for seed in NOMINAL_SEEDS]
    for seed, disturbance in zip(DISTURBED_SEEDS, disturbance_plan(body_id)):
        index = seed - DISTURBED_SEEDS[0] + 1
        trials.append({"trial_id": f"{body_id}-{disturbance['stage']}-{index:02d}", "body": body_id, "kind": "disturbed", "stage": disturbance["stage"], "seed": seed, "cap_s": CAPS_S[body_id], "disturbance": disturbance})
    return trials


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    course = course_v1()
    bodies = {}
    for body_id in BODIES:
        body = load_mobile_body(MOBILE / body_id)
        locomotor = make_locomotor(body, body.floor_model)
        poses = grasp_poses(body)
        manipulator = manipulators_of(body, body.floor_model)[0]
        bodies[body_id] = {"model_sha256": body.model_sha256, "base_kind": body.declaration["base_kind"], "working_stance": locomotor.working_stance(), "controller": type(locomotor).__name__, "provenance": locomotor.provenance,
                           "manipulator": manipulator.limb, "grasp_site": manipulator.declaration["grasp_site"], "manipulation_stance": poses.manipulation_stance, "grasp_radius_m": poses.grasp_radius_m,
                           "routes": {"station": poses.station_waypoints, "tray": poses.tray_waypoints, "return": poses.return_waypoints}, "cap_s": CAPS_S[body_id], "push_newtons": list(PUSH_N[body_id]),
                           "version_note": body.declaration.get("description", "")}
    trials = {body_id: trials_for(body_id) for body_id in BODIES}
    payload = {"schema": "g18.retrieve-protocol.v1", "course_id": course.course_id, "course_sha256": course.sha256(), "goal": next(g for g in course.goals if g["goal_id"] == "retrieve"),
               "phases": list(PHASES), "jitter": JITTER, "settle_s": SETTLE_S, "retry_budget": RETRY_BUDGET,
               "success": {"rule": "the cube at rest inside the tray's rim (within 19 cm of its centre, between the floor and 6 cm above it, moving under 2 cm/s), the base within 45 cm of the start pad's centre, stable for the window (speed under 0.05 m/s, tilt under 15 degrees), no fall (tilt over 55 degrees)", "grasp_tolerance_m": RetrieveSession.GRASP_TOLERANCE_M},
               "resource_rules": ["each phase names the limbs used for support and the limb used to hold; a limb holding the object is excluded from the drive while it holds (the octopus crawls on five tentacles) and its links must bear no ground contact while holding: a violation is recorded as a fault",
                                  "the root is placed once at the start and never written again; the object's state is never written; the object's pose is read from the simulator's state in place of perception (the mobile bodies carry no camera)",
                                  "a disturbance is a push on the base, a force on the held object or a limb fought by an external joint torque, recorded as user input and replayed exactly; a cut wheel drive is the controller's own command, in the record"],
               "targets": TARGETS, "bodies": bodies, "trials": trials, "counts": {b: {"nominal": len(NOMINAL_SEEDS), "disturbed": len(DISTURBED_SEEDS)} for b in BODIES}}
    (PROTOCOL / "protocol.json").write_bytes(json_bytes(payload))
    files = {"protocol.json": hashlib.sha256((PROTOCOL / "protocol.json").read_bytes()).hexdigest()}
    registration = {"schema": "g18.registration.v1", "files": files, "course_sha256": course.sha256(), "bodies": {b: v["model_sha256"] for b, v in bodies.items()}, "controllers": {b: v["provenance"] for b, v in bodies.items()},
                    "trials": {b: len(t) for b, t in trials.items()}, "targets": TARGETS, "registered_at_utc": datetime.now(timezone.utc).isoformat()}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "trials": registration["trials"], "course_sha256": course.sha256()[:12]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
