"""Canonical, evidence-qualified intake used by new transfer experiments.

Legacy ingestion remains available for reproducing prior evidence. This profile
removes uploaded labels from all numerical measurement/grounding decisions and
retains a complete alias map for inspection and source-annotation matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET

import mujoco
from rigby_core.hashing import content_hash, hash_file

from ..errors import GeneralFailureCode, ModelIngestError, RigbyGeneralError
from ..pipeline import IngestedRobot, ingest_robot
from .canonical import CanonicalURDF, canonicalize_urdf
from .contracts import BodyCapabilityManifestV1, BodyCapabilityV1, CapabilityFactV1


@dataclass(frozen=True)
class CapabilityBody:
    robot: IngestedRobot
    manifest: BodyCapabilityManifestV1
    canonical: CanonicalURDF

    def require(self, capability: str, *, subject: str = "body") -> BodyCapabilityV1:
        found = next((c for c in self.manifest.capabilities if c.capability == capability and c.subject == subject), None)
        if found is None or not found.enabled:
            raise RigbyGeneralError(GeneralFailureCode.UNAFFORDED_SCHEMA,
                f"Capability {capability!r} is not established for {subject!r}",
                details={"capability": capability, "subject": subject,
                         "status": found.status if found else "unknown",
                         "condition": found.condition if found else "No evidence registered"})
        return found


def _facts(robot: IngestedRobot, source: ET.Element):
    facts, capabilities = [], []

    def fact(identifier, subject, value, basis, method, uncertainty, units=None):
        facts.append(CapabilityFactV1(fact_id=identifier, subject=subject, value=value, basis=basis,
                                     method=method, uncertainty=uncertainty, units=units))
        return identifier

    def capability(name, subject, status, enabled, evidence, condition):
        capabilities.append(BodyCapabilityV1(capability=name, subject=subject, status=status,
            enabled=enabled, evidence_ids=tuple(evidence), condition=condition))

    model, morphology = robot.finalized.model, robot.morphology
    links = {node.get("name"): node for node in source.findall("link")}
    source_joints = {node.get("name"): node for node in source.findall("joint")}
    mass = fact("body.mass", "body", float(model.body_mass.sum()), "compiled_model", "Sum compiled link masses",
                "Source values and geometry-derived inertias are not independently validated hardware measurements", "kg")
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        declared = links.get(name) is not None and links[name].find("inertial") is not None
        fact(f"inertia.{name}", name, {"mass_kg": float(model.body_mass[index]), "principal_inertia_kgm2": model.body_inertia[index].tolist()},
             "source_declared" if declared else "simulation_assumption", "Compiled positive inertia with source-presence attribution",
             "Declared inertia has not been measured independently" if declared else "MuJoCo inferred inertia from authored geometry/density")
    for joint in morphology.joints:
        node = source_joints[joint.name]
        limit = node.find("limit")
        velocity = float(limit.get("velocity", "0")) if limit is not None else 0
        continuous = node.get("type") == "continuous"
        fact(f"joint.{joint.name}", joint.name,
             {"kind": joint.kind.value, "source_position_range": None if continuous else [joint.minimum, joint.maximum],
              "measurement_interval": [joint.minimum, joint.maximum], "measurement_interval_is_assumed": continuous,
              "position_is_bounded": not continuous, "velocity_limit": joint.velocity_limit,
              "effort_value": joint.effort_limit, "effort_is_declared_limit": joint.effort_declared,
              "velocity_is_declared": velocity > 0, "axis": joint.axis.model_dump(mode="json")},
             "source_declared" if joint.effort_declared and velocity > 0 and not continuous else "simulation_assumption",
             "Source limit parsing and compiled joint measurement",
             "Declared limits are unverified. Continuous joints have no declared position bounds; their finite measurement interval is a sampling assumption. Missing effort values are model-load lower bounds, not actuator capacity; missing velocity uses a stated sweep assumption")
    drive = fact("body.actuation", "body",
        {"motor_count": int(model.nu), "model": "generated_direct_joint_torque_motors",
         "source_transmissions": len(source.findall("transmission")), "acceleration_limit": 500.0},
        "simulation_assumption", "Legacy finalize adds direct motors; source transmissions are descriptive inputs",
        "Does not validate hardware drives, motor dynamics, gear compliance, controller bandwidth or the default acceleration bound")
    geometry = fact("body.geometry", "body", {"colliders": int(robot.integrity.collider_count),
        "rest_qpos": list(robot.manifest.rest_qpos), "sites": [s.model_dump(mode="json") for s in morphology.sites]},
        "measured_kinematics", "Derived geometry sites and nonpenetrating-rest search",
        "Mesh collision is convexified by the existing intake; derived sites have no universal accuracy bound")
    fact("body.collision_exclusions", "body", [list(pair) for pair in robot.manifest.adjacent_collision_exclusions],
         "simulation_assumption", "Recorded source-overlap/IK exclusion policy",
         "Excluded or convexified geometry needs an independent contact check before task certification")
    frame = fact("body.frame", "body", morphology.intrinsic_frame.model_dump(mode="json"), "measured_kinematics",
        "Gravity, workspace asymmetry and geometric symmetry", "Proposed intrinsic front remains unconfirmed; confidence is a heuristic, not a calibrated probability")
    fact("body.legacy_payload_estimate", "body", morphology.scale.payload_kg, "simulation_assumption",
         "Legacy mass/effort heuristic retained for provenance", "Not an identified payload capacity and not an independent feasibility certificate", "kg")
    for chain in morphology.chains:
        witness = fact(f"chain.{chain.chain_id}", chain.chain_id, chain.model_dump(mode="json"), "measured_kinematics",
            "Seeded reachable-site sampling and joint sweeps", "Sampled workspace is not an exact reachable set or a contact/torque feasibility guarantee")
        capability("kinematic_positioning", chain.chain_id,
            "observed_in_model" if chain.positioning_dof >= 2 else "unsupported_by_runtime", chain.positioning_dof >= 2,
            (witness, drive, geometry, frame), "Goal binding still requires reachable geometry, limits and an appropriate reference frame")
    for effector in morphology.effectors:
        evidence = fact(f"effector.{effector.name}", effector.name, effector.model_dump(mode="json"), "measured_kinematics",
            "Surface closure sweep and measured opposition groups", "Geometric closure is not a measured force-closure grasp or a successful object lift")
        capability("kinematic_closure", effector.name,
            "observed_in_model" if effector.can_grasp else "unsupported_by_runtime", bool(effector.can_grasp), (evidence,),
            "Closing opposing surfaces observed in this model; object-specific fit/contact remains to be checked")
        capability("physical_grasp_transport", effector.name, "unknown", False, (evidence, drive, geometry),
            "Requires independently checked object fit, contact mechanics and physical lift/retention/release trials")
    sensor = fact("body.sensors", "body", {"declared_model_cameras": int(model.ncam), "declared_model_sensors": int(model.nsensor),
        "simulation_joint_position_and_velocity_available": True}, "compiled_model", "Compiled sensor/camera inventory",
        "URDF link names confer no sensor capability; simulated encoders do not prove hardware sensor availability")
    capability("simulation_joint_observation", "body", "observed_in_model", True, (sensor,), "A declared observation adapter must expose these channels")
    for name in ("hardware_rgb", "hardware_depth", "suction", "rolling", "dynamic_balance", "physical_payload_capacity"):
        gap = fact("gap."+name, "body", None, "unverified", "No independent operational evidence registered",
                   "Unknown; visual shape and uploaded identifier strings cannot establish this ability")
        capability(name, "body", "unknown", False, (gap,), "Requires source declarations and an appropriate independent mechanics/sensor evaluation")
    fixed = fact("body.base", "body", {"fixed_base": robot.manifest.fixed_base}, "compiled_model",
                 "Fixed-base admission gate", "No locomotion or base-balance controller is admitted by this profile")
    capability("locomotion_execution", "body", "unsupported_by_runtime", False, (fixed,), "A floating-base/mobility runtime and its gates are required")
    return tuple(facts), tuple(capabilities)


def ingest_capability_body(source_path: Path) -> CapabilityBody:
    source_path = source_path.resolve()
    canonical = canonicalize_urdf(source_path.read_bytes())
    tree = ET.fromstring(canonical.xml)
    mimics = tree.findall("joint/mimic")
    if mimics:
        # The current lower-level intake drops these constraints. Refusing here
        # preserves the upload's mechanics instead of certifying independent
        # motors as if they were a coupled mechanism.
        raise ModelIngestError(GeneralFailureCode.UNSUPPORTED_COUPLING,
            "This capability profile cannot yet enforce URDF mimic constraints during measurement and control",
            details={"constraints": len(mimics), "model_invalid": False})
    assets = {}
    for node in tree.iter():
        for key in ("filename", "file"):
            if key not in node.attrib:
                continue
            relative = node.attrib[key]
            path = (source_path.parent / relative).resolve()
            if not path.is_relative_to(source_path.parent) or not path.is_file():
                raise ModelIngestError(GeneralFailureCode.UNSAFE_ASSET, "Missing or escaping model asset", details={"asset": relative})
            assets[relative] = hash_file(path)
    package = content_hash({"canonical_urdf": canonical.canonical_sha256, "assets": assets})
    with tempfile.TemporaryDirectory(prefix="rigby-structural-intake-") as temporary:
        root = Path(temporary)
        for relative in assets:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((source_path.parent / relative).read_bytes())
            if hash_file(destination) != assets[relative]:
                raise ModelIngestError(GeneralFailureCode.UNSAFE_ASSET, "Model asset changed during intake", details={"asset": relative})
        model_source = root / "structural.urdf"
        if model_source.exists():
            raise ModelIngestError(GeneralFailureCode.UNSAFE_ASSET, "An asset collides with the normalized source filename")
        model_source.write_text(canonical.xml, encoding="utf-8", newline="\n")
        robot = ingest_robot(model_source, robot_id="body_"+package[:16])
    facts, capabilities = _facts(robot, tree)
    manifest = BodyCapabilityManifestV1(source_urdf_sha256=canonical.source_sha256,
        canonical_urdf_sha256=canonical.canonical_sha256, package_sha256=package, assets=assets,
        link_aliases=canonical.link_aliases, joint_aliases=canonical.joint_aliases,
        robot=robot.manifest, facts=facts, capabilities=capabilities,
        limitations=("Kinematic capability evidence only; no physical manipulation success is inferred.",
            "Rigid fixed-base URDF profile; unsupported extensions, couplings and ambiguous structural identities receive typed refusals.",
            "Legacy name-hint annotations are unavailable in this profile; actual sensor declarations need an explicit supported sensor format.",
            "Existing source transmissions, inferred actuator limits, default accelerations and collision assumptions are not verified hardware properties."))
    return CapabilityBody(robot, manifest, canonical)
