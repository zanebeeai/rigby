"""Compile a world first, attach a body, and compare resolved world properties.

MjSpec attachment keeps robot defaults/assets in the child namespace:
https://mujoco.readthedocs.io/en/stable/python.html#attachment
World hashes use resolved properties, not a robot-containing XML/MJB digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, Self
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from pydantic import Field, FiniteFloat, model_validator
from rigby_core.contracts import Contract
from rigby_core.hashing import content_hash

from ..contracts import RobotAssetManifestV1
from .contracts import BenchmarkWorldV1, Mode, Quaternion, Vector3

ROBOT_PREFIX = "robot/"
WORLD_PREFIX = "world/"


class BenchmarkRefusal(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


class BodyCameraMountV1(Contract):
    channel: str = Field(min_length=1)
    body: str = Field(min_length=1)
    position_m: Vector3
    quaternion_wxyz: Quaternion
    fovy_degrees: FiniteFloat = Field(gt=0, lt=180)
    provenance: Literal["source_model", "declared_simulated_sensor"]

    @model_validator(mode="after")
    def unit_rotation(self) -> Self:
        if abs(sum(x*x for x in self.quaternion_wxyz) - 1) > 1e-9:
            raise ValueError("Sensor mount quaternion must be unit length")
        return self


class BenchmarkBodyManifestV1(Contract):
    schema_version: Literal["benchmark.body.v1"] = "benchmark.body.v1"
    robot: RobotAssetManifestV1
    camera_mounts: tuple[BodyCameraMountV1, ...] = ()

    @model_validator(mode="after")
    def unique_mounts(self) -> Self:
        if len({m.channel for m in self.camera_mounts}) != len(self.camera_mounts):
            raise ValueError("Sensor channels must be unique")
        bodies = {self.robot.morphology.base_body}
        bodies.update(name for chain in self.robot.morphology.chains for name in chain.bodies)
        bodies.update(site.body for site in self.robot.morphology.sites)
        if any(m.body not in bodies for m in self.camera_mounts):
            raise ValueError("Sensor mount references an undeclared robot body")
        return self


@dataclass(frozen=True)
class CompiledBenchmarkWorld:
    model: mujoco.MjModel
    xml: str
    world: BenchmarkWorldV1
    manifest: dict
    world_sha256: str
    body_manifest: BenchmarkBodyManifestV1 | None
    compiled_model_sha256: str


def _numbers(values) -> str:
    return " ".join(format(float(x), ".17g") for x in values)


def model_digest(model: mujoco.MjModel) -> str:
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _world_xml(world: BenchmarkWorldV1) -> str:
    root = ET.Element("mujoco", model="benchmark-world")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    physics = world.physics
    ET.SubElement(root, "option", timestep=str(physics.timestep_s), gravity=_numbers(physics.gravity_mps2),
                  integrator=physics.integrator, solver=physics.solver, jacobian=physics.jacobian,
                  iterations=str(physics.iterations), tolerance=str(physics.tolerance))
    # Prevent automatic scene extent (which changes with robot size) from
    # changing clipping, lighting or camera presentation behind the world hash.
    ET.SubElement(root, "statistic", center="0 0 0", extent=str(world.floor_half_width_m))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(max(640, world.sensor_policy.width)), offheight=str(max(480, world.sensor_policy.height)))
    ET.SubElement(visual, "headlight", ambient="0 0 0", diffuse="0 0 0", specular="0 0 0")
    body = ET.SubElement(root, "worldbody")
    ET.SubElement(body, "geom", name=WORLD_PREFIX+"floor", type="plane", size=_numbers((world.floor_half_width_m, world.floor_half_width_m, 0.1)),
                  friction="1 0.005 0.0001", rgba="0.18 0.2 0.23 1", contype="1", conaffinity="1")
    light = world.lighting
    ET.SubElement(body, "light", name=WORLD_PREFIX+"key", pos=_numbers(light.position_m), dir=_numbers(light.direction),
                  diffuse=_numbers(light.diffuse), ambient=_numbers(light.ambient), specular=_numbers(light.specular), directional="true")
    camera = world.camera
    ET.SubElement(body, "camera", name=WORLD_PREFIX+"camera", pos=_numbers(camera.position_m), xyaxes=_numbers(camera.xyaxes), fovy=str(camera.fovy_degrees))
    for item in sorted((*world.environment.fixtures, *world.environment.objects), key=lambda x: x.name):
        node = ET.SubElement(body, "body", name=WORLD_PREFIX+item.name, pos=_numbers(item.position_m))
        dynamic = hasattr(item, "mass_kg")
        if dynamic:
            ET.SubElement(node, "freejoint", name=WORLD_PREFIX+item.name+"_free")
        attrs = dict(name=WORLD_PREFIX+item.name+"_geom", type="box", size=_numbers(item.size_m),
                     rgba=_numbers(item.rgba), contype="1", conaffinity="1", condim="4",
                     friction=_numbers((item.friction if dynamic else 1.0, 0.03, 0.001)),
                     solref=f"{physics.contact_timeconst_s} 1", solimp="0.95 0.99 0.001", priority="1")
        if dynamic:
            attrs["mass"] = str(item.mass_kg)
        ET.SubElement(node, "geom", **attrs)
    return ET.tostring(root, encoding="unicode")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, float, int)) or value is None:
        return value
    raise TypeError(f"Unsupported resolved field {type(value)}")


def _fields(value) -> dict:
    return {name: _plain(getattr(value, name)) for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))}


def _named(model, kind, index) -> str:
    if index < 0:
        return "none"
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed:{int(kind)}:{index}"


def _robot_object(model, kind, index) -> bool:
    if index < 0:
        return True  # Absent second endpoint in an internal joint constraint.
    if kind == mujoco.mjtObj.mjOBJ_GEOM:
        return _robot_object(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[index])
    if kind == mujoco.mjtObj.mjOBJ_JOINT:
        return _robot_object(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[index])
    if kind == mujoco.mjtObj.mjOBJ_SITE:
        return _robot_object(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[index])
    return _named(model, kind, index).startswith(ROBOT_PREFIX)


def _reject_world_actuation_and_constraints(model) -> None:
    for name in ("control", "passive", "sensor", "act_gain", "act_bias", "act_dyn", "contactfilter"):
        if getattr(mujoco, "get_mjcb_"+name)() is not None:
            raise BenchmarkRefusal("unregistered_callback", "External MuJoCo callbacks need separate provenance/state support")
    if model.nplugin or model.nflex:
        raise BenchmarkRefusal("unsupported_world_physics", "Plugins/flex need a separate resolved world contract")
    for i in range(model.neq):
        if not all(_robot_object(model, int(model.eq_objtype[i]), int(j)) for j in (model.eq_obj1id[i], model.eq_obj2id[i])):
            raise BenchmarkRefusal("world_constraint", "A constraint touches non-robot state")
    for i in range(model.npair):
        if not all(_robot_object(model, mujoco.mjtObj.mjOBJ_GEOM, int(j)) for j in (model.pair_geom1[i], model.pair_geom2[i])):
            raise BenchmarkRefusal("world_contact_override", "Explicit world contact pairs are not in this protocol")
    for signature in model.exclude_signature:
        if not all(_robot_object(model, mujoco.mjtObj.mjOBJ_BODY, int(j)) for j in (int(signature) >> 16, int(signature) & 65535)):
            raise BenchmarkRefusal("world_contact_exclusion", "A contact exclusion touches non-robot geometry")
    # Every tendon wrap must remain within the declared embodiment.
    kinds = {int(mujoco.mjtWrap.mjWRAP_JOINT): mujoco.mjtObj.mjOBJ_JOINT,
             int(mujoco.mjtWrap.mjWRAP_SITE): mujoco.mjtObj.mjOBJ_SITE,
             int(mujoco.mjtWrap.mjWRAP_SPHERE): mujoco.mjtObj.mjOBJ_GEOM,
             int(mujoco.mjtWrap.mjWRAP_CYLINDER): mujoco.mjtObj.mjOBJ_GEOM}
    for kind, index in zip(model.wrap_type, model.wrap_objid):
        if int(kind) in kinds and not _robot_object(model, kinds[int(kind)], int(index)):
            raise BenchmarkRefusal("world_tendon", "A tendon touches non-robot state")
    transmissions = {int(mujoco.mjtTrn.mjTRN_JOINT): mujoco.mjtObj.mjOBJ_JOINT,
                     int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): mujoco.mjtObj.mjOBJ_JOINT,
                     int(mujoco.mjtTrn.mjTRN_TENDON): mujoco.mjtObj.mjOBJ_TENDON,
                     int(mujoco.mjtTrn.mjTRN_SITE): mujoco.mjtObj.mjOBJ_SITE,
                     int(mujoco.mjtTrn.mjTRN_BODY): mujoco.mjtObj.mjOBJ_BODY}
    for kind, indices in zip(model.actuator_trntype, model.actuator_trnid):
        if int(kind) not in transmissions or not _robot_object(model, transmissions[int(kind)], int(indices[0])):
            raise BenchmarkRefusal("world_actuator", "Unsupported transmission or actuator targets non-robot state")
        if int(kind) == int(mujoco.mjtTrn.mjTRN_SITE) and not _robot_object(model, mujoco.mjtObj.mjOBJ_SITE, int(indices[1])):
            raise BenchmarkRefusal("world_actuator", "Actuator reference site belongs to the world")


def resolved_world_manifest(model: mujoco.MjModel, world: BenchmarkWorldV1) -> dict:
    """Fingerprint the compiled non-robot world, including effective defaults.

G02 worlds use analytic plane/box geometry; unknown world geometry/materials
are refused until their resolved assets can be included without omissions.
Robot mesh assets remain allowed and belong to the separate body manifest.
"""
    _reject_world_actuation_and_constraints(model)
    world_bodies = [i for i in range(1, model.nbody) if not _robot_object(model, mujoco.mjtObj.mjOBJ_BODY, i)]
    body_fields = ("pos", "quat", "ipos", "iquat", "mass", "inertia", "gravcomp", "mocapid")
    bodies = []
    for i in world_bodies:
        if model.body_mocapid[i] >= 0:
            raise BenchmarkRefusal("world_mocap", "Movable mocap world bodies are not in this protocol")
        bodies.append({"name": _named(model, mujoco.mjtObj.mjOBJ_BODY, i),
                       "parent": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[i]),
                       **{key: _plain(getattr(model, "body_"+key)[i]) for key in body_fields}})
    geoms = []
    for i in range(model.ngeom):
        if _robot_object(model, mujoco.mjtObj.mjOBJ_GEOM, i):
            continue
        if model.geom_type[i] not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX) or model.geom_matid[i] >= 0:
            raise BenchmarkRefusal("unsupported_world_geometry", "World geometry requires an analytic shape without an external material")
        keys = ("type", "size", "pos", "quat", "rgba", "contype", "conaffinity", "condim", "friction", "priority", "solmix", "solref", "solimp", "margin", "gap", "fluid", "adhesion", "surfacevel", "group")
        geoms.append({"name": _named(model, mujoco.mjtObj.mjOBJ_GEOM, i), "body": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]),
                      **{key: _plain(getattr(model, "geom_"+key)[i]) for key in keys}})
    joints = []
    for i in range(model.njnt):
        if not _robot_object(model, mujoco.mjtObj.mjOBJ_JOINT, i):
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                raise BenchmarkRefusal("unsupported_world_joint", "World objects must be free rigid bodies")
            q, v = int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])
            joints.append({"name": _named(model, mujoco.mjtObj.mjOBJ_JOINT, i), "qpos0": model.qpos0[q:q+7].tolist(),
                           **{name: getattr(model, "dof_"+name)[v:v+6].tolist() for name in ("damping", "armature", "frictionloss")}})
    lights = [{"name": _named(model, mujoco.mjtObj.mjOBJ_LIGHT, i),
               "body": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.light_bodyid[i]),
               "target_body": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.light_targetbodyid[i]),
               "texture": _named(model, mujoco.mjtObj.mjOBJ_TEXTURE, model.light_texid[i]),
               **{key: _plain(getattr(model, "light_"+key)[i]) for key in ("active", "type", "mode", "pos", "dir", "diffuse", "ambient", "specular", "castshadow", "attenuation", "cutoff", "exponent", "intensity", "range", "bulbradius")}}
              for i in range(model.nlight)]
    cameras = [{"name": _named(model, mujoco.mjtObj.mjOBJ_CAMERA, i),
                "body": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[i]),
                "target_body": _named(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_targetbodyid[i]),
                **{key: _plain(getattr(model, "cam_"+key)[i]) for key in ("pos", "quat", "fovy", "mode", "projection", "resolution", "sensorsize", "intrinsic", "ipd")}}
               for i in range(model.ncam) if not _robot_object(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[i])]
    return {"schema": "benchmark.resolved-world.v1", "mujoco": mujoco.__version__,
            "declared_world": world.model_dump(mode="json"), "physics": _fields(model.opt),
            "bodies": sorted(bodies, key=lambda x: x["name"]), "geoms": sorted(geoms, key=lambda x: x["name"]),
            "joints": sorted(joints, key=lambda x: x["name"]), "lights": sorted(lights, key=lambda x: x["name"]),
            "cameras": sorted(cameras, key=lambda x: x["name"]),
            "visual": {key: _fields(getattr(model.vis, key)) for key in ("global_", "quality", "headlight", "map", "scale", "rgba")},
            "scene_extent": float(model.stat.extent), "scene_center": model.stat.center.tolist()}


def compile_world(world: BenchmarkWorldV1, *, body: BenchmarkBodyManifestV1 | None = None,
                  robot_xml: str | None = None, asset_root: Path | None = None) -> CompiledBenchmarkWorld:
    world = BenchmarkWorldV1.model_validate_json(world.model_dump_json())
    spec = mujoco.MjSpec.from_string(_world_xml(world))
    reference_model = spec.compile()
    reference = resolved_world_manifest(reference_model, world)
    if body is not None:
        body = BenchmarkBodyManifestV1.model_validate_json(body.model_dump_json())
        if robot_xml is None:
            raise BenchmarkRefusal("missing_robot_model", "Body manifest needs its robot XML")
        raw = ET.fromstring(robot_xml)
        root = raw.find("worldbody")
        if root is None or root.findall("geom") or root.findall("light") or root.findall("camera"):
            raise BenchmarkRefusal("robot_contains_world", "Robot upload includes undeclared world geometry, lights or cameras")
        if hashlib.sha256(robot_xml.encode("utf-8")).hexdigest() != body.robot.mjcf.sha256:
            raise BenchmarkRefusal("robot_source_changed", "Robot XML differs from the ingested body manifest")
        child = mujoco.MjSpec.from_string(robot_xml)
        if asset_root is not None:
            child.meshdir = str(asset_root.resolve())
            child.texturedir = str(asset_root.resolve())
        source_model = child.compile()
        source_cameras = {_named(source_model, mujoco.mjtObj.mjOBJ_CAMERA, i): i for i in range(source_model.ncam)}
        declarations = {m.channel: m for m in body.camera_mounts}
        if set(source_cameras) - set(declarations):
            raise BenchmarkRefusal("undeclared_body_sensor", "Source cameras must be represented in the body sensor contract")
        for mount in body.camera_mounts:
            parent = child.body(mount.body)
            if parent is None:
                raise BenchmarkRefusal("missing_sensor_body", mount.body)
            if mount.channel in source_cameras:
                i = source_cameras[mount.channel]
                expected = BodyCameraMountV1(channel=mount.channel,
                    body=_named(source_model, mujoco.mjtObj.mjOBJ_BODY, source_model.cam_bodyid[i]),
                    position_m=tuple(source_model.cam_pos[i]), quaternion_wxyz=tuple(source_model.cam_quat[i]),
                    fovy_degrees=float(source_model.cam_fovy[i]), provenance="source_model")
                if content_hash(expected) != content_hash(mount):
                    raise BenchmarkRefusal("sensor_mount_mismatch", "Declared source sensor differs from compiled body sensor")
            elif mount.provenance != "declared_simulated_sensor":
                raise BenchmarkRefusal("sensor_source_missing", "A source-model sensor does not exist in the source")
            else:
                parent.add_camera(name=mount.channel, pos=mount.position_m, quat=mount.quaternion_wxyz, fovy=mount.fovy_degrees)
        frame = spec.worldbody.add_frame(pos=world.start_zone.base_position_m, quat=world.start_zone.base_quaternion_wxyz)
        spec.copy_during_attach = True
        spec.attach(child, frame=frame, prefix=ROBOT_PREFIX)
    elif robot_xml is not None:
        raise BenchmarkRefusal("missing_body_manifest", "Robot XML requires a body manifest")
    model = spec.compile()
    manifest = resolved_world_manifest(model, world)
    if content_hash(manifest) != content_hash(reference):
        raise BenchmarkRefusal("world_changed_by_body", "Compiled world differs after embodiment attachment")
    return CompiledBenchmarkWorld(model, spec.to_xml(), world, manifest, content_hash(manifest), body, model_digest(model))


def verify_world(compiled: CompiledBenchmarkWorld, expected_sha256: str) -> None:
    if content_hash(resolved_world_manifest(compiled.model, compiled.world)) != expected_sha256:
        raise BenchmarkRefusal("world_changed", "Resolved world differs from the registered world hash")
    if model_digest(compiled.model) != compiled.compiled_model_sha256:
        raise BenchmarkRefusal("body_or_model_changed", "Compiled body, mounting, sensor or model parameters changed after registration")


def normalize_world(world: BenchmarkWorldV1, length_factor: float) -> tuple[BenchmarkWorldV1, dict]:
    """Explicit capability probe: uniformly scale lengths and preserve density."""
    if not np.isfinite(length_factor) or length_factor <= 0:
        raise BenchmarkRefusal("invalid_normalization", "Length factor must be finite and positive")
    value = world.model_dump(mode="json")
    f = float(length_factor)
    for kind in ("fixtures", "objects"):
        for item in value["environment"][kind]:
            for field in ("size_m", "position_m"):
                item[field] = [x*f for x in item[field]]
            if kind == "objects":
                item["mass_kg"] *= f**3
    for key in ("minimum_m", "maximum_m"):
        value["goal"]["target"][key] = [x*f for x in value["goal"]["target"][key]]
    # Sensor/view geometry is part of the normalized-world disclosure too.
    value["camera"]["position_m"] = [x*f for x in value["camera"]["position_m"]]
    value["lighting"]["position_m"] = [x*f for x in value["lighting"]["position_m"]]
    value["floor_half_width_m"] *= f
    result = BenchmarkWorldV1.model_validate(value)
    return result, {"rule": "uniform_length_and_cubic_mass_preserve_density", "length_factor": f,
                    "mass_factor": f**3, "authored_world_sha256": content_hash(world), "resolved_world_sha256": content_hash(result)}


def validate_binding(*, mode: Mode, authored: BenchmarkWorldV1, resolved: BenchmarkWorldV1,
                     requested_semantics: dict, bound_semantics: dict, normalization: dict | None = None) -> None:
    if content_hash(requested_semantics) != content_hash(bound_semantics):
        raise BenchmarkRefusal("semantic_substitution", "Requested semantics cannot be substituted in either benchmark mode")
    if mode == "strict_fixed_world":
        if normalization is not None or content_hash(authored) != content_hash(resolved):
            raise BenchmarkRefusal("world_fitting_forbidden", "Strict mode must preserve the authored world")
    elif mode == "capability_normalized":
        if normalization is None or "length_factor" not in normalization:
            raise BenchmarkRefusal("missing_normalization", "Capability probes require a normalization record")
        expected, record = normalize_world(authored, normalization["length_factor"])
        if content_hash(expected) != content_hash(resolved) or record != normalization:
            raise BenchmarkRefusal("undocumented_normalization", "Resolved world must match the declared normalization exactly")
    else:
        raise BenchmarkRefusal("unknown_benchmark_mode", str(mode))
