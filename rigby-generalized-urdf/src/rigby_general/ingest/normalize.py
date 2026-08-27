"""Second ingest pass: write back what the measurement decided.

The first pass compiles the model as delivered so morphology can be measured.
This pass takes those measurements and produces the model everything downstream
actually runs on -- one that carries named sites where the analyser found them
and one actuator per actuated joint.

Actuators are plain ``motor`` transmissions, not ``position`` servos. That is not
a style preference: ``rigby_v2.simulation.controller.InverseDynamicsPDController``
builds its motor map by rejecting anything with a bias term, a dynamics type, or
a multi-DOF transmission. A position actuator carries an affine bias and would be
refused, and the certified controller is worth more than the convenience.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np
from rigby_v2.contracts import ArtifactRefV1, CoordinateSystem

from ..contracts import RobotAssetManifestV1, RobotMorphologyV1
from ..errors import GeneralFailureCode, ModelIngestError
from ..morphology.measure import neutral_qpos
from .loader import LoadedModel


DEFAULT_ACCELERATION_LIMIT = 500.0


@dataclass(frozen=True, slots=True)
class FinalizedRobot:
    """A measured robot, plus the model that carries its sites and actuators."""

    manifest: RobotAssetManifestV1
    morphology: RobotMorphologyV1
    mjcf_xml: str
    model: mujoco.MjModel

    @property
    def mjcf_bytes(self) -> bytes:
        return self.mjcf_xml.encode("utf-8")


def _artifact_ref(payload: bytes, *, filename: str, media_type: str) -> ArtifactRefV1:
    return ArtifactRefV1(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type=media_type,
        filename=filename,
    )


def finalize(
    loaded: LoadedModel,
    morphology: RobotMorphologyV1,
    *,
    ungateable_pairs: tuple[tuple[str, str], ...] = (),
) -> FinalizedRobot:
    """Attach derived sites and one motor per joint, then recompile.

    ``ungateable_pairs`` are link pairs whose collision hulls are authored
    intersecting, found during ingest. They join the adjacency exclusions so the
    self-collision gate does not report them once per frame forever.
    """

    spec = loaded.spec
    existing_sites = {site.name for site in spec.sites}

    for site in morphology.sites:
        if site.name in existing_sites:
            continue
        try:
            body = spec.body(site.body)
        except (KeyError, ValueError) as error:  # pragma: no cover - defensive
            raise ModelIngestError(
                GeneralFailureCode.UNREADABLE_MODEL,
                f"Derived site {site.name!r} names a body the model does not "
                f"have: {site.body!r}",
                details={"site": site.name, "body": site.body},
            ) from error
        body.add_site(
            name=site.name,
            pos=[site.position_m.x, site.position_m.y, site.position_m.z],
        )

    existing_actuators = {actuator.target for actuator in spec.actuators}
    ordered: list[str] = []
    for joint in morphology.joints:
        ordered.append(joint.name)
        if joint.name in existing_actuators:
            continue
        actuator = spec.add_actuator()
        actuator.name = f"{joint.name}_motor"
        actuator.target = joint.name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_motor()
        actuator.gear[0] = 1.0
        # A joint whose source declared no torque limit is left unlimited rather
        # than clamped to a guess. Clamping to a guess produces an arm that sags
        # under its own weight, and a tracking failure that blames the motion.
        if joint.effort_declared:
            actuator.forcerange = [-joint.effort_limit, joint.effort_limit]
            actuator.forcelimited = 1
            actuator.ctrlrange = [-joint.effort_limit, joint.effort_limit]
            actuator.ctrllimited = 1
        else:
            actuator.forcelimited = 0
            actuator.ctrllimited = 0

    try:
        model = spec.compile()
    except ValueError as error:  # pragma: no cover - defensive
        raise ModelIngestError(
            GeneralFailureCode.UNREADABLE_MODEL,
            f"The model would not recompile after adding sites and actuators: {error}",
            details={"robot_id": morphology.robot_id},
        ) from error

    missing = [
        site.name
        for site in morphology.sites
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name) < 0
    ]
    if missing:  # pragma: no cover - defensive
        raise ModelIngestError(
            GeneralFailureCode.UNREADABLE_MODEL,
            f"Sites vanished during recompilation: {missing}",
            details={"sites": missing},
        )

    xml = spec.to_xml()
    rest = neutral_qpos(model)

    manifest = RobotAssetManifestV1(
        rig_id=morphology.robot_id,
        mjcf=_artifact_ref(
            xml.encode("utf-8"), filename="robot.xml", media_type="application/xml"
        ),
        source_asset=_artifact_ref(
            loaded.source_bytes,
            filename=loaded.source_path.name,
            media_type="application/xml",
        ),
        source_format=loaded.source_format,
        fixed_base=True,
        coordinate_system=_coordinate_system(model, morphology),
        dofs=tuple(
            joint.to_dof_spec(acceleration_limit=DEFAULT_ACCELERATION_LIMIT)
            for joint in morphology.joints
        ),
        actuator_order=tuple(ordered),
        rest_qpos=tuple(float(value) for value in rest),
        sites=morphology.sites,
        morphology=morphology,
        adjacent_collision_exclusions=(
            morphology.self_collision_pairs + ungateable_pairs
        ),
    )

    return FinalizedRobot(
        manifest=manifest, morphology=morphology, mjcf_xml=xml, model=model
    )


def _coordinate_system(
    model: mujoco.MjModel, morphology: RobotMorphologyV1
) -> CoordinateSystem:
    """Report the robot's own axes in the fixed vocabulary the v2 contract uses.

    The literal set only admits world-aligned axes, so a robot whose measured
    front sits at an angle is reported by its nearest axis. The exact direction
    is never lost -- ``morphology.intrinsic_frame`` keeps it -- and the grounder
    works from that, not from this summary.
    """

    up = _nearest_axis(np.array(-model.opt.gravity, dtype=float))
    frame = morphology.intrinsic_frame
    forward = _nearest_axis(
        np.array([frame.front.x, frame.front.y, frame.front.z], dtype=float)
    )
    if forward[-1] == up[-1]:
        forward = "+X" if up[-1] != "X" else "+Y"
    return CoordinateSystem(up_axis=up, forward_axis=forward)


def _nearest_axis(vector: np.ndarray) -> str:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return "+Z"
    unit = vector / norm
    index = int(np.argmax(np.abs(unit)))
    sign = "+" if unit[index] >= 0.0 else "-"
    return f"{sign}{'XYZ'[index]}"
