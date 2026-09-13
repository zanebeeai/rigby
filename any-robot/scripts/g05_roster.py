"""Freeze the G05 composition roster before any scored trial is run.

Twenty predeclared feasible variants per public zoo body -- five paces of the
reviewed prompt crossed with four start states -- plus the invalid requests
whose typed refusals are counted separately. Start states are drawn once, from
a recorded seed, and written out numerically so the campaign executes what was
registered rather than re-drawing. A drawn start that the body cannot hold is
redrawn and the redraw is recorded; nothing is inspected or tuned after the
draw. No API or model calls.

    python any-robot/scripts/g05_roster.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.contracts import JointRole
from rigby_general.grounding import ik
from rigby_general.grounding.grounder import _collision_guard, self_clearance_m


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ZOO = ROOT / "assets/general/zoo"
DESTINATION = ROOT / "assets/general/research-protocols/g05-composition-v1"

SEED = 20260912
START_VARIANTS = 4
MAX_OFFSET_RAD = 0.15
"""Largest joint offset a displaced start may carry: about nine degrees on a
hinge, and the same fraction of a slide's range scaled by its range. Small on
purpose -- the question is whether the composition survives an ordinary
displacement of where the body happens to be, not whether it survives an
arbitrary pose."""
LIMIT_MARGIN_FRACTION = 0.01
"""A displaced start stays this fraction of the range inside every limit, so a
draw never sits exactly on a bound it would then be judged against."""
MAX_REDRAWS = 20

PACES = (
    {"ordinal": -2, "prompt": "very slowly reach out as far as you can and then very slowly come back"},
    {"ordinal": -1, "prompt": "slowly reach out as far as you can and then slowly come back"},
    {"ordinal": 0, "prompt": "reach out as far as you can and then come back"},
    {"ordinal": 1, "prompt": "quickly reach out as far as you can and then quickly come back"},
    {"ordinal": 2, "prompt": "flat out reach out as far as you can and then come back flat out"},
)
CANONICAL = PACES[2]["prompt"]


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def positioning_joints(robot) -> list:
    """Joints that place the effectors, in manifest order; closure joints never
    move in a reach and are left at rest."""

    closure = {name for effector in robot.morphology.effectors for name in effector.grip_joints}
    return [
        joint for joint in robot.morphology.joints
        if joint.name not in closure
        and joint.role not in (JointRole.GRIP, JointRole.IMMOBILE)
        and joint.maximum - joint.minimum > 1e-6
    ]


def draw_start(robot, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    model, manifest = robot.finalized.model, robot.manifest
    start = np.asarray(manifest.rest_qpos, dtype=float).copy()
    offsets = {}
    for joint in positioning_joints(robot):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        address = int(model.jnt_qposadr[joint_id])
        span = joint.maximum - joint.minimum
        scale = MAX_OFFSET_RAD if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else MAX_OFFSET_RAD * span
        offset = float(rng.uniform(-1.0, 1.0)) * scale
        low = joint.minimum + LIMIT_MARGIN_FRACTION * span
        high = joint.maximum - LIMIT_MARGIN_FRACTION * span
        value = float(np.clip(start[address] + offset, low, high))
        offsets[joint.name] = value - float(start[address])
        start[address] = value
    return start, offsets


def start_is_holdable(robot, start: np.ndarray) -> tuple[bool, str]:
    model, manifest = robot.finalized.model, robot.manifest
    for dof in manifest.dofs:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        value = float(start[int(model.jnt_qposadr[joint])])
        if value < dof.minimum or value > dof.maximum:
            return False, f"{dof.name} outside its range"
    guard = _collision_guard(manifest, model)
    if guard is not None:
        inside = ik.penetrations(model, guard, start)
        if inside:
            first, second, depth = inside[0]
            return False, f"{first} {depth * 1000:.1f} mm inside {second}"
    return True, "clear"


def self_colliding_start(robot) -> tuple[np.ndarray, str] | None:
    """A start state with the body inside itself, if one exists within limits.

    Searched by folding the most distal positioning joints toward their bounds
    -- the shape a wrist takes when it closes onto its own forearm -- and
    reported only if the guard finds an actual penetration. Bodies with no
    such pose simply have no invalid start of this kind, and say so.
    """

    model, manifest = robot.finalized.model, robot.manifest
    guard = _collision_guard(manifest, model)
    if guard is None:
        return None
    joints = positioning_joints(robot)
    base = np.asarray(manifest.rest_qpos, dtype=float)
    for count in (1, 2, 3):
        distal = joints[-count:]
        for signs in np.array(np.meshgrid(*([[1.0, -1.0]] * count))).T.reshape(-1, count):
            for fraction in (1.0, 0.85, 0.7):
                start = base.copy()
                for joint, sign in zip(distal, signs):
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
                    bound = joint.maximum if sign > 0 else joint.minimum
                    rest = float(base[int(model.jnt_qposadr[joint_id])])
                    start[int(model.jnt_qposadr[joint_id])] = rest + fraction * (bound - rest)
                inside = ik.penetrations(model, guard, start)
                if inside:
                    first, second, depth = inside[0]
                    return start, f"{first} {depth * 1000:.1f} mm inside {second}"
    return None


def body_entry(source: Path, rng: np.random.Generator) -> dict:
    zoo_id = source.parent.name
    body = ingest_capability_body(source)
    robot = body.robot
    model, manifest = robot.finalized.model, robot.manifest
    starts = [{"start_id": "start_0", "kind": "measured_rest", "qpos": [float(v) for v in manifest.rest_qpos], "offsets": {}, "redraws": 0}]
    for index in range(1, START_VARIANTS):
        for redraw in range(MAX_REDRAWS + 1):
            start, offsets = draw_start(robot, rng)
            holdable, why = start_is_holdable(robot, start)
            if holdable:
                starts.append({"start_id": f"start_{index}", "kind": "displaced", "qpos": start.tolist(),
                               "offsets": offsets, "redraws": redraw})
                break
            print(f"  {zoo_id} start_{index}: redraw {redraw + 1} ({why})", file=sys.stderr)
        else:
            raise RuntimeError(f"{zoo_id}: no holdable displaced start within {MAX_REDRAWS} redraws")

    joints = positioning_joints(robot)
    beyond = np.asarray(manifest.rest_qpos, dtype=float).copy()
    first = joints[0]
    first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, first.name)
    span = first.maximum - first.minimum
    beyond[int(model.jnt_qposadr[first_id])] = first.maximum + (0.5 if model.jnt_type[first_id] == mujoco.mjtJoint.mjJNT_HINGE else 0.5 * span)
    invalid = [
        {"case_id": "start_beyond_limit", "prompt": CANONICAL, "start_qpos": beyond.tolist(),
         "reason": f"{first.name} placed past its maximum",
         "expected": {"refusal_code": "ungroundable", "measurement": "start_state.joint_limit", "where": "leaf_bake_and_binding"}},
        {"case_id": "unafforded_request", "prompt": "grasp the cup and then come back", "start_qpos": [float(v) for v in manifest.rest_qpos],
         "reason": "asks for a contact schema this free-space campaign does not certify",
         "expected": {"refusal_code": "unafforded_schema", "measurement": None, "where": "planning"}},
    ]
    folded = self_colliding_start(robot)
    if folded is not None:
        start, why = folded
        invalid.append({"case_id": "start_self_collision", "prompt": CANONICAL, "start_qpos": start.tolist(),
                        "reason": why,
                        "expected": {"refusal_code": "ungroundable", "measurement": "start_state.self_collision", "where": "leaf_bake_and_binding"}})
    else:
        invalid.append({"case_id": "start_self_collision", "prompt": None, "start_qpos": None,
                        "reason": "no self-colliding start exists within this body's joint limits along the searched folds",
                        "expected": {"not_constructible": True}})

    trials = [
        {"trial_id": f"pace{pace['ordinal']:+d}_{start['start_id']}", "pace_ordinal": pace["ordinal"], "prompt": pace["prompt"], "start_id": start["start_id"]}
        for start in starts for pace in PACES
    ]
    return {
        "zoo_id": zoo_id, "source_urdf": source.relative_to(REPO).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "intake": "rigby_general.capabilities.intake.ingest_capability_body",
        "rig_id": manifest.rig_id, "package_sha256": body.manifest.package_sha256,
        "canonical_urdf_sha256": body.manifest.canonical_urdf_sha256,
        "dof_count": len(manifest.dofs), "positioning_joints": [j.name for j in joints],
        "reach_radius_m": manifest.morphology.scale.reach_radius_m,
        "self_clearance_m": self_clearance_m(manifest.morphology.scale.reach_radius_m),
        "guarded_pairs": len(_collision_guard(manifest, model).pairs) if _collision_guard(manifest, model) else 0,
        "collision_exclusions": [list(p) for p in manifest.adjacent_collision_exclusions],
        "starts": starts, "feasible_trials": trials, "invalid_requests": invalid,
    }


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit(f"{DESTINATION} exists; a registered roster is never overwritten")
    rng = np.random.default_rng(SEED)
    bodies = [body_entry(source, rng) for source in sorted(ZOO.glob("*/robot.urdf"))]
    roster = {
        "schema": "g05.composition-roster.v1", "goal": "G05",
        "title": "Six-body free-space composition: predeclared start/pace variants",
        "canonical_prompt": CANONICAL, "paces": list(PACES),
        "draw": {"seed": SEED, "generator": "numpy.random.default_rng(seed), one stream over bodies in sorted zoo order",
                 "start_variants_per_body": START_VARIANTS, "max_offset_rad": MAX_OFFSET_RAD,
                 "limit_margin_fraction": LIMIT_MARGIN_FRACTION, "max_redraws": MAX_REDRAWS,
                 "redraw_rule": "a drawn start outside a limit or inside the body's own links is redrawn; every redraw is recorded"},
        "feasibility_rule": "Every pace of the reviewed prompt is feasible by construction: pace is a preference the grounder clamps to the joint velocity limits, never a request that exceeds them. A start is feasible when the body can hold it: inside every limit and clear of self-penetration under the same guard the grounder applies. Invalid requests are predeclared and counted separately; a body that cannot construct one kind of invalid start records that instead of inventing one.",
        "success": {
            "definition": "The composed prompt is accepted by the pipeline: both leaves freshly baked at the exact requested region, zero region substitutions, the composition certified by all physical gates (joint position and velocity limits, actuator effort, base drift, self-collision, tracking) on three identical native-clock replays, with unchanged limits and thresholds.",
            "gate_policy": "rigby_general.gates.certify.GatePolicy() defaults",
            "leaf_pace_retry": "The bake's existing single pace retry (duration scale) is permitted for a leaf and recorded; it is not a region substitution.",
            "target": "at least 19 of 20 feasible trials per body",
        },
        "budgets": {"attempts_per_trial": 1, "repeats_per_certification": 3, "wall_seconds_per_trial": 300},
        "evidence": {
            "per_trial": "outcome, failure typing, authored and actual physics duration, tracking error, violations, replay hashes, leaf duration scales, wall time per stage",
            "physical_traces": "certification trace hashes recorded for every trial; full traces retained locally under any-robot/results/g05-campaign (untracked)",
            "full_video": "canonical trial of every body as a G01 evidence bundle with full-duration MP4 and labelled GIF summary (D05)",
        },
        "duration_criterion": {
            "baseline_reference": "docs/results/g05-curves-report.md canonical authored durations at commit 5a6c0b3bb17eb721da6ba6c144743c24ba872301 (legacy intake, native clock, bounded curves)",
            "rule": "The canonical (pace 0, measured rest) authored duration per body must be no greater than the baseline; the bimanual body's baseline was a self-colliding, physically refused program, so its comparison is reported against that refused duration and labelled as such.",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__,
        "scored_runs_started": False, "generation_calls": 0,
        "bodies": bodies,
    }
    DESTINATION.mkdir(parents=True)
    (DESTINATION / "roster.json").write_bytes(json_bytes(roster))
    digest = hashlib.sha256((DESTINATION / "roster.json").read_bytes()).hexdigest()
    registration = {
        "schema": "g05.composition-registration.v1", "roster_sha256": digest,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "feasible_trials": sum(len(b["feasible_trials"]) for b in bodies),
        "invalid_requests": sum(1 for b in bodies for c in b["invalid_requests"] if c["prompt"] is not None),
        "not_constructible": sum(1 for b in bodies for c in b["invalid_requests"] if c["prompt"] is None),
        "scored_runs_started": False,
        "scope": "Public engineering roster for G05; the confirmatory sealed identities of the transfer benchmark are untouched.",
    }
    (DESTINATION / "registration.json").write_bytes(json_bytes(registration))
    print(json.dumps({"roster_sha256": digest, **{k: registration[k] for k in ("feasible_trials", "invalid_requests", "not_constructible")}}, indent=2))


if __name__ == "__main__":
    main()
