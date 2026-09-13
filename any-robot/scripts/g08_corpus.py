"""Freeze the G08 transition corpus before any scored composition runs.

One hundred feasible boundary cases across the three bodies the G06 campaign
enabled, and forty-seven injected incompatible ones, each written out
numerically and hashed with the G06 fixture it composes against. A feasible
case is a terminal joint configuration for a free motion from rest -- drawn
uniformly from the middle sixty percent of every joint's range -- that the
self-collision guard clears along the straight joint path from rest and from
which the transfer's own guarded, facing-aware path solves; the composed
execution is that motion followed by the whole transfer, and its success is
the transfer's certificate plus the free-motion joint gates over the
composition. An injected case names its kind and what the checker must do
with it: reject it before execution, or repair it and verify the boundary
again before the second skill runs. The reviewed dual-arm failure is among
them: the left wrist of the bimanual body at the limit it folded to in the
pre-guard return, as recorded in the G05 fixture. No API or model calls.

    python any-robot/scripts/g08_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.contact.grasp import _hand_facing, _scene_rest_qpos
from rigby_general.contact.transfer import FACING_TOLERANCE_RAD, PHASES, _grasp_offset_m, _joint_path, _path, facing_angle, grasp_standoff_m, transfer_scene_from_environment
from rigby_general.contracts import SiteSemantic
from rigby_general.grounding import ik
from rigby_general.grounding.grounder import _collision_guard, figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import SUPPORT_PREFIX
from rigby_general.transitions import arm_joint_names

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g06_transfer_campaign as campaign  # noqa: E402

DESTINATION = ROOT / "assets/general/research-protocols/g08-transitions-v1"
G06 = ROOT / "assets/general/research-protocols/g06-transfer-v1"
FIXTURE = ROOT / "tests/fixtures/g05_composition/dual_arm_program_before_guard.json"
SEED = 20260914
BODIES = {"zoo_dual_arm": 34, "zoo_jaw_arm": 33, "zoo_long_arm": 33}
BAND = 0.2
"""Fraction of each joint's range kept clear at either end when drawing a feasible pose."""
GUARD_SAMPLES = 24
WORLD_CLEARANCE_M = 0.001
"""A robot geom deeper than this inside a fixture disqualifies a pose or a path; any contact with the object does."""
PATH_SUBSAMPLES = 4
"""Configurations checked against the world between consecutive solved rows of the transfer's path."""
INJECTED = {"joint_beyond_limit": 10, "joint_inside_margin": 6, "velocity_too_high": 8, "holding_into_free": 6, "free_into_holding": 6, "belief_stale": 5, "resource_conflict": 5}


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class Body:
    def __init__(self, zoo_id: str, env, goal):
        self.zoo_id = zoo_id
        self.source = REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf"
        self.robot = ingest_robot(self.source, robot_id=zoo_id)
        self.manifest, self.morphology = self.robot.manifest, self.robot.morphology
        self.effector = self.morphology.grasping_effectors[0]
        chain = next(c for c in self.morphology.chains if c.chain_id == self.effector.chain_id)
        self.frame = build_workspace_frame(self.robot.finalized.model, self.morphology, chain, figure_site=figure_site_for(self.manifest, self.effector.chain_id))
        self.scene = transfer_scene_from_environment(self.manifest, self.robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=goal, asset_root=self.source.parent)
        self.model = self.scene.model
        self.guard = _collision_guard(self.manifest, self.model)
        self.arm = arm_joint_names(self.model, self.effector, self.frame)
        self.dofs = {d.joint: d for d in self.manifest.dofs}
        self.arm_adr = np.array([int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in self.arm])
        self.rest = _scene_rest_qpos(self.model, self.manifest)

    def pose_from(self, targets: dict[str, float]) -> np.ndarray:
        qpos = np.array(self.rest, dtype=float)
        for index, name in enumerate(self.arm):
            qpos[self.arm_adr[index]] = targets[self.dofs[name].name]
        return qpos

    def world_contact(self, qpos: np.ndarray) -> str | None:
        """A robot geom inside a fixture or the object at ``qpos``: the guard
        keeps the body out of itself, not out of the world, and a free
        motion driven through the bench is not a certified skill."""

        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
            world = [n for n in names if n.startswith((SUPPORT_PREFIX, "scene_"))]
            if len(world) != 1:
                continue
            allowance = 0.0 if world[0].startswith("scene_") else WORLD_CLEARANCE_M
            if float(contact.dist) > -allowance:
                continue
            return f"{names[0] or 'geom'} against {names[1] or 'geom'} by {-float(contact.dist) * 1000:.1f} mm"
        return None

    def straight_path_clear(self, targets: dict[str, float]) -> tuple[bool, str]:
        end = self.pose_from(targets)
        for fraction in np.linspace(0.0, 1.0, GUARD_SAMPLES):
            sample = self.rest + fraction * (end - self.rest)
            inside = ik.penetrations(self.model, self.guard, sample)
            if inside:
                return False, f"{inside[0][0]} inside {inside[0][1]} at {fraction:.2f}"
            hit = self.world_contact(sample)
            if hit:
                return False, f"world: {hit} at {fraction:.2f}"
        return True, ""

    def transfer_path_solves(self, targets: dict[str, float]) -> tuple[bool, str, str]:
        """The transfer's own guarded, facing-aware path from this pose."""

        qpos = self.pose_from(targets)
        model, manifest, effector, frame = self.model, self.manifest, self.effector, self.frame
        site = next((s.name for s in manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id)), frame.figure_site)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        home = np.array(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)], dtype=float)
        facing = _hand_facing(manifest, effector)
        standoff = grasp_standoff_m(model, manifest, effector, site, self.scene.scene.block_half_extent_m)
        points, spans = _path(self.scene, frame, home, _grasp_offset_m(manifest, effector), standoff)
        downward = (facing, -np.asarray(frame.up, dtype=float)) if facing is not None else None
        # From a pose the arm actually stands in, the only seed is that pose:
        # the transfer resumed there solves from it and nothing else.
        seeds = [("pose", np.array(qpos, dtype=float))]
        try:
            path, marks, seed = _joint_path(model, site, self.arm, points, seeds, self.guard, 6, facing=downward)
        except ik.IkFailure as error:
            return False, "self_collision_path" if error.collision is not None else "unreachable_path", str(error)[:160]
        if downward is not None and "descend" in spans:
            worst = max(facing_angle(model, site, path[marks[index]], downward[0], downward[1]) for index in spans["descend"])
            if worst > FACING_TOLERANCE_RAD:
                return False, "facing_unmet", f"the planned hand is {np.degrees(worst):.1f} degrees from vertical at the hover or the grasp"
        low = np.array([self.dofs[n].minimum for n in self.arm])
        high = np.array([self.dofs[n].maximum for n in self.arm])
        arm_rows = path[:, self.arm_adr]
        if np.any(arm_rows < low - 1e-9) or np.any(arm_rows > high + 1e-9):
            return False, "path_outside_limits", "the solved path leaves a joint range"
        # The transfer's path keeps the body out of itself; the world it
        # crosses on the way to the hover is checked here, up to the hover,
        # between the solved rows as well as at them, and any contact at all
        # with the object counts: a finger that brushes the cube on the way
        # in moves it, and the descent then stops.
        rows = path[: marks[spans["turn"][1]] + 1] if "turn" in spans else path
        for index in range(len(rows) - 1):
            for fraction in np.linspace(0.0, 1.0, PATH_SUBSAMPLES, endpoint=False):
                hit = self.world_contact(rows[index] + fraction * (rows[index + 1] - rows[index]))
                if hit:
                    return False, "transfer_path_through_world", f"row {index} + {fraction:.2f}: {hit}"
        hit = self.world_contact(rows[-1])
        if hit:
            return False, "transfer_path_through_world", f"row {len(rows) - 1}: {hit}"
        return True, seed, ""


def feasible_cases(body: Body, count: int, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    cases, rejected = [], []
    draws = 0
    while len(cases) < count:
        draws += 1
        targets = {}
        for name in body.arm:
            dof = body.dofs[name]
            span = dof.maximum - dof.minimum
            targets[dof.name] = float(rng.uniform(dof.minimum + BAND * span, dof.maximum - BAND * span))
        clear, why = body.straight_path_clear(targets)
        if not clear:
            rejected.append({"draw": draws, "reason": "straight_joint_path_" + "self_collision", "detail": why})
            continue
        solves, seed_or_code, detail = body.transfer_path_solves(targets)
        if not solves:
            rejected.append({"draw": draws, "reason": seed_or_code, "detail": detail})
            continue
        cases.append({"case_id": f"{body.zoo_id}-feasible-{len(cases):03d}", "zoo_id": body.zoo_id, "kind": "feasible", "draw": draws,
                      "first": {"skill": "joint_move", "targets": targets}, "second": {"skill": "transfer"},
                      "witness": {"straight_joint_path_clear": True, "transfer_path_seed": seed_or_code}, "expect": "composed_success"})
    return cases, rejected


def injected_cases(bodies: dict[str, Body], rng: np.random.Generator) -> list[dict]:
    cases = []
    ids = sorted(bodies)

    def base_targets(body: Body) -> dict[str, float]:
        while True:
            targets = {}
            for name in body.arm:
                dof = body.dofs[name]
                span = dof.maximum - dof.minimum
                targets[dof.name] = float(rng.uniform(dof.minimum + BAND * span, dof.maximum - BAND * span))
            if body.straight_path_clear(targets)[0] and body.transfer_path_solves(targets)[0]:
                return targets

    def limit_case(kind: str, body: Body, index: int, fraction_past: float) -> dict | None:
        """One joint driven to ``fraction_past`` of its range past (positive) or
        inside (negative) its upper limit, the rest mid-band; the joint is the
        first, from the base outward, whose straight path stays clear."""

        targets = base_targets(body)
        for name in body.arm:
            dof = body.dofs[name]
            span = dof.maximum - dof.minimum
            trial = dict(targets)
            trial[dof.name] = dof.maximum + fraction_past * span
            if body.straight_path_clear(trial)[0]:
                return {"case_id": f"{body.zoo_id}-{kind}-{index:02d}", "zoo_id": body.zoo_id, "kind": kind, "first": {"skill": "joint_move", "targets": trial},
                        "second": {"skill": "transfer"}, "subject": dof.name, "expect": "repaired_then_reverified", "note": f"{dof.name} commanded to {trial[dof.name]:.4f} against a limit of {dof.maximum:.4f}"}
        return None

    counters = {kind: 0 for kind in INJECTED}
    for kind, fraction in (("joint_beyond_limit", 0.03), ("joint_inside_margin", -0.01)):
        while counters[kind] < INJECTED[kind]:
            body = bodies[ids[counters[kind] % len(ids)]]
            case = limit_case(kind, body, counters[kind], fraction)
            if case is None:
                continue
            cases.append(case)
            counters[kind] += 1
    while counters["velocity_too_high"] < INJECTED["velocity_too_high"]:
        body = bodies[ids[counters["velocity_too_high"] % len(ids)]]
        targets = base_targets(body)
        cases.append({"case_id": f"{body.zoo_id}-velocity_too_high-{counters['velocity_too_high']:02d}", "zoo_id": body.zoo_id, "kind": "velocity_too_high",
                      "first": {"skill": "joint_move", "targets": targets, "stop_fraction": float(rng.uniform(0.35, 0.55))}, "second": {"skill": "transfer"},
                      "expect": "repaired_then_reverified", "note": "the motion is cut part way, still moving"})
        counters["velocity_too_high"] += 1
    while counters["holding_into_free"] < INJECTED["holding_into_free"]:
        body = bodies[ids[counters["holding_into_free"] % len(ids)]]
        cases.append({"case_id": f"{body.zoo_id}-holding_into_free-{counters['holding_into_free']:02d}", "zoo_id": body.zoo_id, "kind": "holding_into_free",
                      "first": {"skill": "acquire_carry"}, "second": {"skill": "transfer"}, "expect": "rejected", "note": "ends holding the cube; the next skill begins free with the cube resting"})
        counters["holding_into_free"] += 1
    while counters["free_into_holding"] < INJECTED["free_into_holding"]:
        body = bodies[ids[counters["free_into_holding"] % len(ids)]]
        cases.append({"case_id": f"{body.zoo_id}-free_into_holding-{counters['free_into_holding']:02d}", "zoo_id": body.zoo_id, "kind": "free_into_holding",
                      "first": {"skill": "joint_move", "targets": base_targets(body)}, "second": {"skill": "place"}, "expect": "rejected", "note": "ends free; the next skill begins holding the cube"})
        counters["free_into_holding"] += 1
    while counters["belief_stale"] < INJECTED["belief_stale"]:
        body = bodies[ids[counters["belief_stale"] % len(ids)]]
        cases.append({"case_id": f"{body.zoo_id}-belief_stale-{counters['belief_stale']:02d}", "zoo_id": body.zoo_id, "kind": "belief_stale",
                      "first": {"skill": "joint_move", "targets": base_targets(body)}, "second": {"skill": "transfer"}, "belief_age_s": float(rng.uniform(8.0, 40.0)),
                      "expect": "repaired_then_reverified", "note": "the belief predates the motion by more than the next skill allows"})
        counters["belief_stale"] += 1
    while counters["resource_conflict"] < INJECTED["resource_conflict"]:
        body = bodies[ids[counters["resource_conflict"] % len(ids)]]
        cases.append({"case_id": f"{body.zoo_id}-resource_conflict-{counters['resource_conflict']:02d}", "zoo_id": body.zoo_id, "kind": "resource_conflict",
                      "first": {"skill": "joint_move", "targets": base_targets(body), "keeps_resources": [f"effector:{body.effector.chain_id}"]}, "second": {"skill": "transfer"},
                      "expect": "rejected", "note": "the first skill keeps owning the manipulator"})
        counters["resource_conflict"] += 1
    # The reviewed dual-arm failure: the left wrist at the limit it folded to.
    dual = bodies["zoo_dual_arm"]
    program = json.loads(FIXTURE.read_bytes())
    keyframes = program["tracks"][0]["keyframes"]
    chosen = None
    for keyframe in sorted(keyframes, key=lambda k: -max(abs(v) for v in k["joint_values"].values())):
        targets = {}
        for name in dual.arm:
            dof = dual.dofs[name]
            targets[dof.name] = float(keyframe["joint_values"].get(name, keyframe["joint_values"].get(dof.name, 0.0)))
        wrist = max(targets.values(), key=abs)
        inside_margin = any(abs(v) > 0.98 * dual.dofs[n].maximum for n, v in ((n, targets[dual.dofs[n].name]) for n in dual.arm))
        if inside_margin and dual.straight_path_clear(targets)[0] and dual.world_contact(dual.pose_from(targets)) is None:
            chosen = (keyframe, targets)
            break
    if chosen is None:
        raise SystemExit("no keyframe of the reviewed dual-arm failure is both inside a limit margin and reachable clear of the body")
    keyframe, targets = chosen
    at_limit, value = max(targets.items(), key=lambda item: abs(item[1]))
    cases.append({"case_id": "zoo_dual_arm-reviewed_dual_arm_failure-00", "zoo_id": "zoo_dual_arm", "kind": "joint_inside_margin", "first": {"skill": "joint_move", "targets": targets},
                  "second": {"skill": "transfer"}, "expect": "repaired_then_reverified", "subject": at_limit,
                  "note": (f"the reviewed dual-arm failure: the pre-guard program's keyframe at {keyframe['time_s']:.2f} s, {at_limit} at {value:.3f} rad against a limit of 2.85, "
                           "the latest keyframe of that program with a joint inside its limit margin that the body can be moved to clear of itself and of the world "
                           "(the keyframes with the wrist folded into the forearm are self-colliding and cannot be reached as a certified motion)"),
                  "fixture": FIXTURE.relative_to(REPO).as_posix(), "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()})
    return cases


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit(f"{DESTINATION} exists; a registered corpus is never overwritten")
    env, goal, feasibility, roster, registration = campaign.load_registration()
    rng = np.random.default_rng(SEED)
    bodies = {zoo_id: Body(zoo_id, env, goal) for zoo_id in BODIES}
    feasible, rejected = [], {}
    for zoo_id, count in BODIES.items():
        cases, lost = feasible_cases(bodies[zoo_id], count, rng)
        feasible.extend(cases)
        rejected[zoo_id] = lost
        print(json.dumps({"body": zoo_id, "feasible": len(cases), "rejected_draws": len(lost)}), flush=True)
    injected = injected_cases(bodies, rng)
    print(json.dumps({"injected": len(injected), "kinds": {k: sum(1 for c in injected if c["kind"] == k) for k in sorted({c["kind"] for c in injected})}}), flush=True)
    corpus = {
        "schema": "g08.transition-corpus.v1", "goal": "G08", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "g06_registration_sha256": registration["registration_sha256"], "environment_id": env.environment_id,
        "draw": {"seed": SEED, "generator": "numpy.random.default_rng(seed), one stream, bodies in the order listed, feasible then injected",
                 "feasible_band_fraction": BAND, "guard_samples_along_straight_path": GUARD_SAMPLES, "world_clearance_m": WORLD_CLEARANCE_M,
                 "witness": "the straight joint path from rest clear of the body and of the world; the transfer's guarded path from the pose alone solved, inside every joint range, the hand within the facing tolerance at the hover and the grasp, and clear of the world up to the hover"},
        "initiation": {"limit_margin_fraction": 0.02, "speed_fraction": 0.05, "max_belief_age_s": 5.0,
                       "note": "the transfer begins free with the cube resting; the placement begins holding the cube; every skill requires its manipulator and the object not owned elsewhere"},
        "success": {"feasible": "the first skill certified on its own gates, the boundary compatible (after any repair), the second skill certified, no joint gate violation on the transitions or the second skill; target at least 95 of 100",
                    "injected": "rejected before execution, or repaired and the boundary verified compatible before the second skill runs; every one of at least 40"},
        "repair_budget": 2, "bodies": list(BODIES), "feasible_cases": feasible, "injected_cases": injected,
        "rejected_draws": rejected, "scored_runs_started": False, "generation_calls": 0,
    }
    DESTINATION.mkdir(parents=True)
    (DESTINATION / "corpus.json").write_bytes(json_bytes(corpus))
    files = {"corpus.json": hashlib.sha256((DESTINATION / "corpus.json").read_bytes()).hexdigest(),
             "g06/environment.json": hashlib.sha256((G06 / "environment.json").read_bytes()).hexdigest(),
             "g06/goal.json": hashlib.sha256((G06 / "goal.json").read_bytes()).hexdigest()}
    reg = {"schema": "g08.registration.v1", "files": files, "registration_sha256": hashlib.sha256(json_bytes(files)).hexdigest(),
           "registered_at_utc": datetime.now(timezone.utc).isoformat(), "feasible_cases": len(feasible), "injected_cases": len(injected), "scored_runs_started": False}
    (DESTINATION / "registration.json").write_bytes(json_bytes(reg))
    print(json.dumps({"registration_sha256": reg["registration_sha256"], "feasible": len(feasible), "injected": len(injected)}, indent=1))


if __name__ == "__main__":
    main()
