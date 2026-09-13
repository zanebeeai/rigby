"""Freeze the G06 fixed fixture and its independent feasibility map.

One authored world, in absolute metres, that never learns which body enters
it: a cube on a low bench and an empty platform beside it. For every
gripper-bearing public body the map records, before any controller runs,
whether the task is mechanically feasible -- aperture against the cube's span,
payload against its mass, the source and destination and the hover above each
inside the measured reach envelope, the fixture clear of the body at rest, and
a kinematic witness: a self-collision-guarded joint configuration that puts the
grasp point on the cube and another that puts it on the destination, with the
body clear of the fixtures. A body missing a necessary condition is infeasible
with that condition named; a body with every condition and a witness is
feasible; anything else stays unknown. Bodies without a grasping effector are
recorded as unsupported by structure. No API or model calls.

    python any-robot/scripts/g06_fixture.py
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

from rigby_general.contact.grasp import _hand_facing, _scene_rest_qpos
from rigby_general.contact.placement import PlacementGoal
from rigby_general.contact.transfer import grasp_standoff_m, restart_seeds, transfer_scene_from_environment
from rigby_general.grounding import ik
from rigby_general.grounding.grounder import _collision_guard, figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1, FixtureV1, SceneObjectV1, SUPPORT_PREFIX
from rigby_general.contracts import SiteSemantic


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ZOO = ROOT / "assets/general/zoo"
DESTINATION = ROOT / "assets/general/research-protocols/g06-transfer-v1"

CUBE_HALF_M = 0.015
CUBE_DENSITY = 320.0
SOURCE_XY = (-0.15, 0.65)
DESTINATION_XY = (-0.01, 0.65)
FIXTURE_HALF_M = (0.05, 0.05, 0.015)
FIXTURE_TOP_M = 0.26
APERTURE_MARGIN_M = 0.004


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def environment() -> EnvironmentV1:
    height = FIXTURE_HALF_M[2]
    return EnvironmentV1(
        environment_id="g06_transfer_v1",
        description=(
            "A 30 mm cube on a low bench to the left of and slightly behind the mount, and an empty platform of the "
            "same height beside it, both in absolute metres. Nothing about the world depends on which body enters it."
        ),
        robot_mount_m=(0.0, 0.0, 0.0),
        fixtures=(
            FixtureV1(name="bench", size_m=FIXTURE_HALF_M, position_m=(SOURCE_XY[0], SOURCE_XY[1], FIXTURE_TOP_M - height)),
            FixtureV1(name="platform", size_m=FIXTURE_HALF_M, position_m=(DESTINATION_XY[0], DESTINATION_XY[1], FIXTURE_TOP_M - height), rgba=(0.30, 0.45, 0.75, 1.0)),
        ),
        objects=(
            SceneObjectV1(name="cube", size_m=(CUBE_HALF_M,) * 3, mass_kg=round(CUBE_DENSITY * (2 * CUBE_HALF_M) ** 3, 6),
                          position_m=(SOURCE_XY[0], SOURCE_XY[1], FIXTURE_TOP_M + CUBE_HALF_M)),
        ),
        authored_for_reach_m=None,
        authored_for_aperture_m=None,
    )


def goal() -> PlacementGoal:
    half = FIXTURE_HALF_M
    return PlacementGoal(
        region_minimum_m=(DESTINATION_XY[0] - half[0] - 0.01, DESTINATION_XY[1] - half[1] - 0.01, FIXTURE_TOP_M - 0.001),
        region_maximum_m=(DESTINATION_XY[0] + half[0] + 0.01, DESTINATION_XY[1] + half[1] + 0.01, FIXTURE_TOP_M + 0.09),
        dwell_s=2.0, maximum_linear_speed_mps=0.01, maximum_angular_speed_radps=0.1,
    )


def _witness(model, manifest, effector, frame, points, guard) -> dict:
    site = next((s.name for s in manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id)), frame.figure_site)
    joints = ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))
    rest = _scene_rest_qpos(model, manifest)
    facing = _hand_facing(manifest, effector)
    downward = (facing, -np.asarray(frame.up, dtype=float)) if facing is not None else None
    standoff = grasp_standoff_m(model, manifest, effector, site, CUBE_HALF_M)
    points = [np.asarray(p, dtype=float) + np.asarray(frame.up, dtype=float) * standoff for p in points]
    solution, seed_used, failure = None, None, None
    for label, seed in restart_seeds(model, frame, joints, rest, points[-1]):
        try:
            solution = ik.solve_site_path(model, site, joints, np.asarray(points), seed_qpos=seed, guard=guard, facing=downward)
            seed_used = label
            break
        except ik.IkFailure as error:
            failure = failure or error
    if solution is None:
        return {"found": False, "reason": str(failure)[:200], "collision": list(failure.collision) if failure.collision else None}
    data = mujoco.MjData(model)
    fixture_hits = []
    for index, qpos in enumerate(solution.qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            contact = data.contact[c]
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""]
            if any(n.startswith(SUPPORT_PREFIX) for n in names) and float(contact.dist) < -0.001:
                fixture_hits.append({"waypoint": index, "geoms": names, "depth_m": -float(contact.dist)})
    return {"found": not fixture_hits, "standoff_m": standoff, "seed": seed_used, "residuals_m": [float(r) for r in solution.residuals_m],
            "fixture_penetrations": fixture_hits, "configurations": [[float(v) for v in q] for q in solution.qpos]}


def assess(source: Path, env: EnvironmentV1, placement: PlacementGoal) -> dict:
    robot = ingest_robot(source, robot_id=source.parent.name)
    model, manifest, morphology = robot.finalized.model, robot.manifest, robot.morphology
    row = {"zoo_id": source.parent.name, "rig_id": manifest.rig_id, "conditions": {}, "class": "unknown", "reasons": []}
    if not morphology.grasping_effectors:
        row.update(**{"class": "unsupported_by_structure", "reasons": ["no grasping effector"]})
        return row
    effector = morphology.grasping_effectors[0]
    chain = next(c for c in morphology.chains if c.chain_id == effector.chain_id)
    frame = build_workspace_frame(model, morphology, chain, figure_site=figure_site_for(manifest, effector.chain_id))
    cube = env.objects[0]
    span = cube.span_m
    row["effector"] = {"kind": effector.kind.value, "max_aperture_m": effector.max_aperture_m, "min_aperture_m": effector.min_aperture_m, "chain_id": effector.chain_id}
    conditions = row["conditions"]
    conditions["aperture_spans_cube"] = {"ok": bool(effector.max_aperture_m is not None and effector.max_aperture_m >= span + APERTURE_MARGIN_M), "aperture_m": effector.max_aperture_m, "span_m": span}
    conditions["payload_carries_cube"] = {"ok": bool(morphology.scale.payload_kg >= cube.mass_kg), "payload_kg": morphology.scale.payload_kg, "mass_kg": cube.mass_kg}
    height = 2 * CUBE_HALF_M
    source_point = np.asarray(cube.position_m)
    destination_point = np.array([DESTINATION_XY[0], DESTINATION_XY[1], FIXTURE_TOP_M + CUBE_HALF_M])
    hover = np.array([0.0, 0.0, height + 1.75 * height])
    reach = {}
    for name, point in (("source", source_point), ("source_hover", source_point + hover), ("destination", destination_point), ("destination_hover", destination_point + hover)):
        azimuth, elevation = frame.bearing_of(point)
        reach[name] = {"contained": bool(frame.contains(point, margin=1.0)), "distance_m": float(np.linalg.norm(point - frame.origin)),
                       "inner_m": float(frame.inner_reach(azimuth, elevation)), "outer_m": float(frame.directional_reach(azimuth, elevation))}
    conditions["points_inside_measured_envelope"] = {"ok": all(v["contained"] for v in reach.values()), **reach}
    scene = transfer_scene_from_environment(manifest, robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=placement)
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.model.qpos0
    for dof in manifest.dofs:
        joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        data.qpos[int(scene.model.jnt_qposadr[joint])] = manifest.rest_qpos[int(robot.finalized.model.jnt_qposadr[mujoco.mj_name2id(robot.finalized.model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)])]
    mujoco.mj_forward(scene.model, data)
    rest_hits = []
    for c in range(data.ncon):
        contact = data.contact[c]
        names = [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "", mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""]
        world = [n for n in names if n.startswith((SUPPORT_PREFIX, "scene_"))]
        if world and len(world) == 1 and float(contact.dist) <= 0.0:
            rest_hits.append({"geoms": names, "dist_m": float(contact.dist)})
    conditions["fixture_clear_of_body_at_rest"] = {"ok": not rest_hits, "contacts": rest_hits}
    guard = _collision_guard(manifest, scene.model)
    witness = {
        "source": _witness(scene.model, manifest, effector, frame, [source_point + hover, source_point], guard),
        "destination": _witness(scene.model, manifest, effector, frame, [destination_point + hover, destination_point + np.array([0, 0, 0.002])], guard),
    }
    conditions["kinematic_witness"] = {"ok": witness["source"]["found"] and witness["destination"]["found"],
                                       "source": {k: v for k, v in witness["source"].items() if k != "configurations"},
                                       "destination": {k: v for k, v in witness["destination"].items() if k != "configurations"}}
    row["witness_configurations"] = {k: v.get("configurations") for k, v in witness.items()}
    necessary = ["aperture_spans_cube", "payload_carries_cube", "points_inside_measured_envelope", "fixture_clear_of_body_at_rest"]
    failed = [n for n in necessary if not conditions[n]["ok"]]
    if failed:
        row["class"] = "infeasible"
        row["reasons"] = [f"necessary condition violated: {n}" for n in failed]
    elif conditions["kinematic_witness"]["ok"]:
        row["class"] = "feasible"
        row["reasons"] = ["all necessary conditions hold and a guarded kinematic witness reaches the cube and the destination clear of the fixtures"]
    else:
        row["class"] = "unknown"
        row["reasons"] = ["necessary conditions hold but no kinematic witness was found; a failed solve is not an impossibility certificate"]
    return row


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit(f"{DESTINATION} exists; a registered fixture is never overwritten")
    env = environment()
    placement = goal()
    rows = [assess(source, env, placement) for source in sorted(ZOO.glob("*/robot.urdf"))]
    DESTINATION.mkdir(parents=True)
    (DESTINATION / "environment.json").write_bytes(json_bytes(env.model_dump(mode="json")))
    (DESTINATION / "goal.json").write_bytes(json_bytes({
        "schema": "g06.placement-goal.v1", "predicate": "object_placed_released_and_still",
        "object": "cube", "destination_fixture": "platform", "region_minimum_m": list(placement.region_minimum_m),
        "region_maximum_m": list(placement.region_maximum_m), "dwell_s": placement.dwell_s,
        "maximum_linear_speed_mps": placement.maximum_linear_speed_mps, "maximum_angular_speed_radps": placement.maximum_angular_speed_radps,
        "containment": "whole_geometry", "release_rule": "no_robot_touch_or_positive_normal_force",
        "hold_rule": "opposition held continuously for 2.0 s with the object raised at least half the required lift",
        "tolerances": {"max_penetration_m": 0.004, "required_lift_fraction_of_height": 0.8, "carry_offset_fraction_of_aperture": 2.5},
    }))
    (DESTINATION / "feasibility-map.json").write_bytes(json_bytes({
        "schema": "g06.feasibility-map.v1", "environment_id": env.environment_id,
        "environment_sha256": hashlib.sha256((DESTINATION / "environment.json").read_bytes()).hexdigest(),
        "goal_sha256": hashlib.sha256((DESTINATION / "goal.json").read_bytes()).hexdigest(),
        "rule": "independent_geometry_and_mechanics_before_scoring",
        "independence": "Assessed from the measured body and the authored world before any transfer controller ran; no controller outcome informs a class.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__,
        "bodies": rows,
    }))
    for row in rows:
        print(json.dumps({"body": row["zoo_id"], "class": row["class"], "reasons": row["reasons"],
                          "conditions": {k: v.get("ok") for k, v in row.get("conditions", {}).items()}}))


if __name__ == "__main__":
    main()
