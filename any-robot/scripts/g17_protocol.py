"""Register the G17 locomotion protocol before any scored run: the course, the route, the seeds, the perturbations, the caps, the targets.

Every body runs the course's travel goal -- leave the start pad, pass
the corridor to the station, stop within 0.3 m of the route point for
2 s, turn around, return to the start pad and stop -- from a hundred
frozen seeds, each drawing the start jitter (10 cm, 10 degrees), under
a cap per body set from its top speed. Thirty predetermined perturbation
trials per body follow, on their own seeds: twelve pushes on the base
(four directions at three magnitudes, 0.3 s, landing when the body is
about a metre out and moving through the corridor: the dog at 6 s, the
biped at 4 s, the octopus at 8 s), six low-friction patches across the route
(three frictions at two places, one on the corridor and one on the
approach to the station, met on the way out and on the way back), and
twelve support disturbances appropriate to the declared mechanics: the
dog has a leg's servos fought by an external joint torque; the biped
has a wheel's drive cut by its own controller, or a leg's servos fought;
the octopus has a tentacle's servos fought, at the same per-body time
as the pushes. The push magnitudes bracket
what a pilot ladder on seed 1 (kept beside the registration) found each
body survives, so the ladder records recoveries and failures alike. The
targets are the goal's: at least 90 of 100 travel trials reach the goal
and stand stable for 2 s; at least 24 of 30 perturbation trials recover.

    python any-robot/scripts/g17_protocol.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body
from rigby_general.mobility.locomotion import make_locomotor
from rigby_general.mobility.trials import FALL_TILT_DEG, STABLE_SPEED_MPS, STABLE_WINDOW_S
from rigby_general.mobility.world import course_v1


ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "assets/general/mobile"
PROTOCOL = ROOT / "assets/general/research-protocols/g17-locomotion-v1"
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")
TRAVEL_SEEDS = tuple(range(1, 101))
PERTURBATION_SEEDS = tuple(range(201, 231))
CAPS_S = {"mobile_dog_arm": 120.0, "mobile_wheeled_biped": 90.0, "mobile_octopus": 300.0}
JITTER = {"xy_m": 0.10, "yaw_deg": 10.0}
SETTLE_S = 1.5
PUSH_N = {"mobile_dog_arm": (40.0, 70.0, 100.0), "mobile_wheeled_biped": (30.0, 55.0, 80.0), "mobile_octopus": (100.0, 200.0, 300.0)}
PUSH_DIRECTIONS = {"left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0), "fore": (1.0, 0.0, 0.0), "aft": (-1.0, 0.0, 0.0)}
PUSH_AT_S = {"mobile_dog_arm": 6.0, "mobile_wheeled_biped": 4.0, "mobile_octopus": 8.0}
"""When the push lands: each body is then about a metre out, mid-corridor, at its cruising speed (the biped would otherwise be at the station by 5 s)."""
PUSH_FOR_S = 0.3
PATCH_FRICTIONS = (0.15, 0.30, 0.50)
PATCH_PLACES = {"corridor": (1.5, 0.0), "approach": (2.5, 0.0)}
PATCH_HALF_M = (0.4, 0.8)
TARGETS = {"travel_min": 90, "travel_of": 100, "perturbation_min": 24, "perturbation_of": 30}


def support_plan(body_id: str) -> list[dict]:
    """The twelve support disturbances a body's mechanics call for."""

    plan = []
    at = PUSH_AT_S[body_id]
    if body_id == "mobile_dog_arm":
        for leg in ("leg_fl", "leg_fr", "leg_hl", "leg_hr"):
            for torque in (8.0, 16.0, 24.0):
                plan.append({"kind": "support", "detail": f"{leg}'s servos fought by a {torque:.0f} N m external torque at 3 Hz for 1.0 s at {at} s", "at_s": at, "duration_s": 1.0, "magnitude": torque, "target": leg, "mechanics": "legged: a leg loses its servo authority mid-stride"})
    elif body_id == "mobile_wheeled_biped":
        for side in ("left", "right"):
            for cut in (0.15, 0.30, 0.50):
                plan.append({"kind": "support", "detail": f"the {side} wheel's drive cut by the controller for {cut:.2f} s at {at} s", "at_s": at, "duration_s": cut, "magnitude": 0.0, "target": f"wheel:{side}_wheel_spin", "mechanics": "wheeled: a drive motor drops out while balancing"})
        for side in ("left", "right"):
            for torque in (6.0, 12.0, 18.0):
                plan.append({"kind": "support", "detail": f"leg_{side}'s servos fought by a {torque:.0f} N m external torque at 3 Hz for 1.0 s at {at} s", "at_s": at, "duration_s": 1.0, "magnitude": torque, "target": f"leg_{side}", "mechanics": "wheeled: a leg's hip and knee lose their hold under the wheel torque"})
    elif body_id == "mobile_octopus":
        for tentacle in range(6):
            for torque in (4.0, 8.0):
                plan.append({"kind": "support", "detail": f"tentacle_{tentacle}'s servos fought by a {torque:.0f} N m external torque at 3 Hz for 1.5 s at {at} s", "at_s": at, "duration_s": 1.5, "magnitude": torque, "target": f"tentacle_{tentacle}", "mechanics": "crawling: a tentacle in the tripod loses its servo authority"})
    assert len(plan) == 12, body_id
    return plan


def perturbation_plan(body_id: str) -> list[dict]:
    plan = []
    for name, direction in PUSH_DIRECTIONS.items():
        for newton in PUSH_N[body_id]:
            plan.append({"kind": "push", "detail": f"{newton:.0f} N {name} push on the base for {PUSH_FOR_S} s at {PUSH_AT_S[body_id]} s", "at_s": PUSH_AT_S[body_id], "duration_s": PUSH_FOR_S, "magnitude": newton, "direction": list(direction), "target": ""})
    for place, centre in PATCH_PLACES.items():
        for friction in PATCH_FRICTIONS:
            plan.append({"kind": "patch", "detail": f"a slick patch (friction {friction:.2f}) across the {place} at x = {centre[0]:.1f} m, met on the way out and on the way back", "patch_friction": friction, "patch_centre_m": list(centre), "patch_half_m": list(PATCH_HALF_M), "target": place})
    plan += support_plan(body_id)
    assert len(plan) == 30, body_id
    return plan


def trials_for(body_id: str, course) -> list[dict]:
    waypoints = [list(course.route[1]), list(course.route[0])]
    trials = [{"trial_id": f"{body_id}-travel-{seed:03d}", "body": body_id, "kind": "travel", "seed": seed, "waypoints": waypoints, "cap_s": CAPS_S[body_id], "perturbation": None} for seed in TRAVEL_SEEDS]
    for seed, perturbation in zip(PERTURBATION_SEEDS, perturbation_plan(body_id)):
        index = seed - PERTURBATION_SEEDS[0] + 1
        trials.append({"trial_id": f"{body_id}-{perturbation['kind']}-{index:02d}", "body": body_id, "kind": perturbation["kind"], "seed": seed, "waypoints": waypoints, "cap_s": CAPS_S[body_id], "perturbation": perturbation})
    return trials


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    course = course_v1()
    bodies = {}
    for body_id in BODIES:
        body = load_mobile_body(MOBILE / body_id)
        locomotor = make_locomotor(body, body.floor_model)
        bodies[body_id] = {"model_sha256": body.model_sha256, "base_kind": body.declaration["base_kind"], "working_stance": locomotor.working_stance(), "controller": type(locomotor).__name__, "provenance": locomotor.provenance,
                           "max_speed_mps": locomotor.max_speed_mps, "max_turn_radps": locomotor.max_turn_radps, "cap_s": CAPS_S[body_id], "push_newtons": list(PUSH_N[body_id])}
    trials = {body_id: trials_for(body_id, course) for body_id in BODIES}
    payload = {"schema": "g17.locomotion-protocol.v1", "course_id": course.course_id, "course_sha256": course.sha256(), "goal": next(g for g in course.goals if g["goal_id"] == "travel"),
               "route": {"waypoints": [list(course.route[1]), list(course.route[0])], "radius_m": 0.3, "release_m": 0.45, "dwell_s": 2.0}, "jitter": JITTER, "settle_s": SETTLE_S,
               "success": {"rule": "every waypoint reached in turn within the radius, stable for the window at the end (speed below the threshold, tilt under 15 degrees), no fall, final distance within the release radius", "stable_window_s": STABLE_WINDOW_S, "stable_speed_mps": STABLE_SPEED_MPS, "fall_tilt_deg": FALL_TILT_DEG},
               "targets": TARGETS, "bodies": bodies, "trials": trials, "counts": {b: {"travel": len(TRAVEL_SEEDS), "perturbation": len(PERTURBATION_SEEDS)} for b in BODIES},
               "rules": ["the controller commands only the body's own actuators; the root is placed once before the settle and never written again", "no artificial support, no hidden wrench: a push is the trial's declared external input, recorded and replayed with the physics",
                         "a support disturbance is an external joint torque on a declared limb (recorded input) or the controller cutting its own wheel drive (a command, in the record)", "a patch is a static geom of the world with its own friction; the world is compiled with it before the trial",
                         "the controller runs without the language planner; the navigator knows a base pose and a drive interface only"]}
    (PROTOCOL / "protocol.json").write_bytes(json_bytes(payload))
    files = {"protocol.json": hashlib.sha256((PROTOCOL / "protocol.json").read_bytes()).hexdigest()}
    if (PROTOCOL / "pilot-ladder.json").is_file():
        files["pilot-ladder.json"] = hashlib.sha256((PROTOCOL / "pilot-ladder.json").read_bytes()).hexdigest()
    registration = {"schema": "g17.registration.v1", "files": files, "course_sha256": course.sha256(), "bodies": {b: v["model_sha256"] for b, v in bodies.items()}, "controllers": {b: v["provenance"] for b, v in bodies.items()},
                    "trials": {b: len(t) for b, t in trials.items()}, "targets": TARGETS, "registered_at_utc": datetime.now(timezone.utc).isoformat()}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (PROTOCOL / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "trials": registration["trials"], "course_sha256": course.sha256()[:12]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
