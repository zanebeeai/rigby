"""Ingest a mobile body: compile it, check its integrity against the rules a floating base can meet, and measure its manifest.

The fixed-base ingest refuses a free joint. This path requires exactly one,
on the declared base body, and otherwise holds the model to the same
standards: no external asset paths, finite positive inertias that obey
the triangle inequality, valid limits on every hinge and slide (a wheel
may be declared unlimited), convex colliders on every moving body, a
collider on every member declared to bear weight, and every actuator
bound to a declared joint with a finite force range. What it measures it
reads off the compiled model -- masses, ranges, efforts, the fingers'
aperture, a limb's length, a manipulator's reach, the body's footprint --
never off a name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..ingest.integrity import scan_source_text
from .contracts import BaseKind, LimbV1, ManipulatorV1, MobileBodyManifestV1, MobileIntegrityReportV1, MobileJointV1, SupportMemberV1


FLOOR_XML = '<geom name="floor" type="plane" size="8 8 0.1" friction="1.0 0.005 0.0001" rgba="0.82 0.82 0.8 1" condim="3"/>'


@dataclass(frozen=True)
class MobileBody:
    robot_id: str
    folder: Path
    xml: str
    declaration: dict
    provenance: dict
    model: mujoco.MjModel
    """The body alone, compiled."""
    floor_model: mujoco.MjModel
    """The body on a level floor: what every settling test runs on."""
    floor_xml: str
    model_sha256: str


def with_floor(xml: str) -> str:
    """The body's MJCF with a level floor and a light added to its world."""

    return xml.replace("<worldbody>", "<worldbody>\n    " + FLOOR_XML + '\n    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>', 1)


def load_mobile_body(folder: Path) -> MobileBody:
    folder = Path(folder)
    xml = (folder / "robot.xml").read_text(encoding="utf-8")
    declaration = json.loads((folder / "mobility.json").read_bytes())
    provenance = json.loads((folder / "provenance.json").read_bytes()) if (folder / "provenance.json").exists() else {}
    digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    model = mujoco.MjSpec.from_string(xml).compile()
    floor_xml = with_floor(xml)
    floor_model = mujoco.MjSpec.from_string(floor_xml).compile()
    return MobileBody(robot_id=declaration["robot_id"], folder=folder, xml=xml, declaration=declaration, provenance=provenance, model=model, floor_model=floor_model, floor_xml=floor_xml, model_sha256=digest)


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"{kind.name.lower()}_{index}"


def check_mobile_integrity(body: MobileBody) -> MobileIntegrityReportV1:
    model = body.model
    declaration = body.declaration
    violations: list[str] = []
    notes: list[str] = []
    for violation in scan_source_text(body.xml.encode("utf-8")):
        if violation.rule.value != "asset.base.fixed.v1":
            violations.append(f"{violation.rule.value}: {violation.detail}")
    if body.model_sha256 != declaration.get("model_sha256"):
        violations.append("the model on disk does not hash to the declaration's model_sha256")
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, declaration["base_body"])
    if base < 0:
        violations.append(f"declared base body {declaration['base_body']!r} is not in the model")
    if len(free) != 1:
        violations.append(f"a mobile body needs exactly one free joint; found {len(free)}")
    elif base >= 0 and int(model.jnt_bodyid[free[0]]) != base:
        violations.append("the free joint is not on the declared base body")
    elif _name(model, mujoco.mjtObj.mjOBJ_JOINT, free[0]) != declaration["root_joint"]:
        violations.append("the free joint is not the declared root joint")
    declared_joints = {j["name"]: j for j in declaration["joints"]}
    for j in range(model.njnt):
        kind = int(model.jnt_type[j])
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            violations.append(f"joint {name!r} is neither a hinge nor a slide")
            continue
        if name not in declared_joints:
            violations.append(f"joint {name!r} is in the model but not in the declaration")
            continue
        limited = bool(model.jnt_limited[j])
        low, high = (float(v) for v in model.jnt_range[j])
        declared_range = declared_joints[name]["range"]
        if declared_range is None:
            if limited:
                violations.append(f"joint {name!r} is declared unlimited (a wheel) but the model limits it")
        else:
            if not limited or not (np.isfinite(low) and np.isfinite(high) and low < high):
                violations.append(f"joint {name!r} has no valid finite limit")
            elif abs(low - declared_range[0]) > 1e-6 or abs(high - declared_range[1]) > 1e-6:
                violations.append(f"joint {name!r} limits differ from the declaration")
    for name in declared_joints:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            violations.append(f"declared joint {name!r} is not in the model")
    # inertias: finite, positive, and the principal moments obey the triangle inequality
    checked = 0
    for b in range(1, model.nbody):
        mass = float(model.body_mass[b])
        inertia = np.asarray(model.body_inertia[b], dtype=float)
        if mass <= 0.0 or not np.isfinite(mass):
            violations.append(f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, b)!r} has no positive finite mass")
            continue
        checked += 1
        if not np.all(np.isfinite(inertia)) or np.any(inertia <= 0.0):
            violations.append(f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, b)!r} has a non-positive principal inertia")
        elif not (inertia[0] + inertia[1] >= inertia[2] * (1 - 1e-6) and inertia[1] + inertia[2] >= inertia[0] * (1 - 1e-6) and inertia[0] + inertia[2] >= inertia[1] * (1 - 1e-6)):
            violations.append(f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, b)!r} inertia breaks the triangle inequality")
    # colliders: every moving collider primitive or a mesh with a convex hull; every declared support member has one
    colliders = 0
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        colliders += 1
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH and int(model.mesh_graphadr[int(model.geom_dataid[g])]) < 0:
            violations.append(f"geom {_name(model, mujoco.mjtObj.mjOBJ_GEOM, g)!r} collides with a mesh that has no convex hull")
    for member in declaration["support_members"]:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member)
        if b < 0:
            violations.append(f"declared support member {member!r} is not a body")
        elif not any(model.geom_bodyid[g] == b and (model.geom_contype[g] or model.geom_conaffinity[g]) for g in range(model.ngeom)):
            violations.append(f"declared support member {member!r} carries no collider")
    # actuators: each on a declared joint, with a finite force range
    for a in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if int(model.actuator_trntype[a]) != mujoco.mjtTrn.mjTRN_JOINT:
            violations.append(f"actuator {name!r} does not drive a joint")
            continue
        joint = _name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0]))
        if joint not in declared_joints:
            violations.append(f"actuator {name!r} drives an undeclared joint {joint!r}")
        if not model.actuator_forcelimited[a] or not np.all(np.isfinite(model.actuator_forcerange[a])):
            violations.append(f"actuator {name!r} has no finite force range")
    if model.nu == 0:
        violations.append("the body has no actuators")
    if not declaration.get("stances"):
        violations.append("no stance is declared")
    return MobileIntegrityReportV1(robot_id=body.robot_id, passed=not violations, violations=tuple(violations), notes=tuple(notes), body_count=int(model.nbody) - 1,
                                   joint_count=int(model.njnt) - len(free), actuator_count=int(model.nu), collider_count=colliders, free_joints=len(free), mass_kg=float(model.body_mass.sum()),
                                   inertia_checked_bodies=checked)


def _subtree_bodies(model: mujoco.MjModel, root: int) -> list[int]:
    found = [root]
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for b in range(1, model.nbody):
            if int(model.body_parentid[b]) == parent and b not in found:
                found.append(b)
                frontier.append(b)
    return found


def _limb_geometry(model: mujoco.MjModel, joints: tuple[str, ...]) -> tuple[str, float]:
    """The limb's tip body (the deepest body under its first joint's body) and its length along the body offsets."""

    first = int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])])
    bodies = _subtree_bodies(model, first)
    depth = {first: 0}
    for b in bodies[1:]:
        depth[b] = depth[int(model.body_parentid[b])] + 1
    tip = max(bodies, key=lambda b: (depth[b], -b))
    length = 0.0
    b = tip
    while b != first:
        length += float(np.linalg.norm(model.body_pos[b]))
        b = int(model.body_parentid[b])
    return _name(model, mujoco.mjtObj.mjOBJ_BODY, tip), length


def _manipulator_measurements(body: MobileBody, declared: dict, stance: dict[str, float]) -> tuple[float, float]:
    """The fingers' aperture at the open end of the grip joints, and the grasp site's distance from the base with the limb's hinges at zero."""

    model = body.model
    data = mujoco.MjData(model)
    for name, value in stance.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    limb_joints = body.declaration["limbs"][declared["limb"]]
    for name in limb_joints:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if name in declared["grip_joints"]:
            low, high = model.jnt_range[j]
            # open end: for the slide jaws the low end (fingers apart at zero travel), for the hinge pincer the low end too
            data.qpos[model.jnt_qposadr[j]] = float(low)
        else:
            data.qpos[model.jnt_qposadr[j]] = 0.0
    mujoco.mj_forward(model, data)
    fingers = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in declared["fingers"]]
    finger_geoms = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == f] for f in fingers]
    centres = [np.mean([data.geom_xpos[g] for g in geoms], axis=0) for geoms in finger_geoms]
    halves = [float(np.min([min(model.geom_size[g][:2]) for g in geoms])) for geoms in finger_geoms]
    aperture = float(np.linalg.norm(centres[0] - centres[1]) - halves[0] - halves[1])
    # reach: the body offsets from the limb's first joint's body down to the grasp site's body, plus the site's own offset
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, declared["grasp_site"])
    first = int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, limb_joints[0])])
    b = int(model.site_bodyid[site])
    reach = float(np.linalg.norm(model.site_pos[site]))
    while b != first and b > 0:
        reach += float(np.linalg.norm(model.body_pos[b]))
        b = int(model.body_parentid[b])
    return max(aperture, 1e-4), reach


def measure_mobile_body(body: MobileBody, *, footprint_stance: str | None = None) -> MobileBodyManifestV1:
    model = body.model
    declaration = body.declaration
    joints = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        declared = next(d for d in declaration["joints"] if d["name"] == name)
        actuator = next(a for a in range(model.nu) if int(model.actuator_trnid[a][0]) == j)
        limited = bool(model.jnt_limited[j])
        joints.append(MobileJointV1(name=name, kind="hinge" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "slide", body=_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])),
                                    limb=declared["limb"], role=declared["role"], minimum=float(model.jnt_range[j][0]) if limited else None, maximum=float(model.jnt_range[j][1]) if limited else None,
                                    effort_limit=float(abs(model.actuator_forcerange[actuator][1])), velocity_limit=float(declared["velocity"]), actuator=_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                                    actuator_kind=declared["actuator_kind"], rest=float(declared["rest"])))
    role_of = {"leg": "leg", "wheel": "wheel_leg", "tentacle": "tentacle", "arm": "arm", "grip": "arm"}
    limbs = []
    for limb_id, names in declaration["limbs"].items():
        tip, length = _limb_geometry(model, tuple(names))
        roles = {next(d for d in declaration["joints"] if d["name"] == n)["role"] for n in names}
        role = "wheel_leg" if "wheel" in roles else ("tentacle" if "tentacle" in roles else ("arm" if roles <= {"arm", "grip"} else "leg"))
        limbs.append(LimbV1(limb_id=limb_id, role=role, joints=tuple(names), tip_body=tip, length_m=length))
    support = []
    for member in declaration["support_members"]:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member)
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and (model.geom_contype[g] or model.geom_conaffinity[g])]
        limb = next((limb_id for limb_id, names in declaration["limbs"].items() if b in _subtree_bodies(model, int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[0])]))), None)
        sensor = f"{member}_contact" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{member}_contact") >= 0 else None
        support.append(SupportMemberV1(body=member, limb=limb, geoms=tuple(_name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms), friction=float(min(model.geom_friction[g][0] for g in geoms)), touch_sensor=sensor))
    working = declaration["working_stance"]
    stance = declaration["stances"][working]["joints"]
    manipulators = []
    for declared in declaration.get("manipulators", []):
        aperture, reach = _manipulator_measurements(body, declared, stance)
        manipulators.append(ManipulatorV1(limb=declared["limb"], grasp_site=declared["grasp_site"], grip_joints=tuple(declared["grip_joints"]), fingers=tuple(declared["fingers"]), aperture_m=aperture, reach_m=reach))
    # footprint: the geoms' extent across x and y in the footprint stance (the working one by default)
    data = mujoco.MjData(model)
    for name, value in declaration["stances"][footprint_stance or working]["joints"].items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    xs, ys = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        r = float(model.geom_rbound[g])
        xs += [float(data.geom_xpos[g][0]) - r, float(data.geom_xpos[g][0]) + r]
        ys += [float(data.geom_xpos[g][1]) - r, float(data.geom_xpos[g][1]) + r]
    sensors = tuple(_name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(model.nsensor))
    return MobileBodyManifestV1(robot_id=body.robot_id, description=declaration["description"], base_kind=BaseKind(declaration["base_kind"]), base_body=declaration["base_body"], root_joint=declaration["root_joint"],
                                model_sha256=body.model_sha256, mass_kg=float(model.body_mass.sum()), joints=tuple(joints), limbs=tuple(limbs), support_members=tuple(support), manipulators=tuple(manipulators),
                                stances={name: dict(s["joints"]) for name, s in declaration["stances"].items()}, working_stance=working, expected_standing_height_m=float(declaration["expected_standing_height_m"]),
                                sensors=tuple(declaration.get("sensors", ())), sensor_names=sensors, footprint_m=(max(xs) - min(xs), max(ys) - min(ys)), traction=dict(declaration.get("traction", {})),
                                provenance=dict(body.provenance))


__all__ = ["FLOOR_XML", "MobileBody", "check_mobile_integrity", "load_mobile_body", "measure_mobile_body", "with_floor"]
