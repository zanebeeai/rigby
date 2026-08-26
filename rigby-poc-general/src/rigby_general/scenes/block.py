"""A graspable object, sized and placed from the robot's own measurements.

Every dimension here is derived, because a scene authored in absolute metres is
the same mistake as a motion authored in absolute metres. A block sized for a
1.3 m industrial arm is a boulder to a desktop arm and a pebble to a long-reach
one, and in both cases the grasp that follows tells you nothing.

So the block is sized from the measured gripper aperture, placed inside the
measured reach envelope at a height the effector can actually descend to, and
given a mass inside the measured payload. What stays fixed across robots is the
*relationship* -- an object that comfortably fits this gripper, within this arm's
working volume -- which is exactly the invariant the schema layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..errors import GeneralFailureCode, RigbyGeneralError
from ..grounding.workspace import WorkspaceFrame


# Fraction of the gripper's measured aperture. Comfortably inside it: a block at
# the full aperture cannot be approached without the jaws already touching it.
BLOCK_APERTURE_FRACTION = 0.55
BLOCK_DENSITY = 320.0
SUPPORT_MARGIN = 0.004

OBJECT_BODY = "scene_block"
OBJECT_GEOM = "scene_block_geom"
SUPPORT_GEOM = "scene_support_geom"
OBJECT_SITE = "scene_block_center"


@dataclass(frozen=True, slots=True)
class GraspScene:
    """A robot plus one block it can actually pick up."""

    model: mujoco.MjModel
    xml: str
    block_half_extent_m: float
    block_mass_kg: float
    block_position_m: np.ndarray
    support_height_m: float
    approach_height_m: float

    @property
    def block_body(self) -> str:
        return OBJECT_BODY


def _reachable_placement(
    frame: WorkspaceFrame, *, minimum_height_m: float
) -> np.ndarray:
    """A spot the effector can descend onto, inside the measured envelope.

    Searched over descending elevations rather than fixed at one, because the
    same downward angle lands at very different heights on arms of different
    length -- on a tall one it puts the block below the robot's own base, which
    is a placement no arm can reach and no scene should contain.
    """

    for elevation in (-32.0, -22.0, -12.0, -4.0, 4.0, 12.0):
        for radius in (0.55, 0.45, 0.65, 0.38):
            candidate = frame.point_at_reach_fraction(
                radius_fraction=radius, azimuth_deg=0.0, elevation_deg=elevation
            )
            if candidate[2] >= minimum_height_m:
                return candidate
    return frame.point_at_reach_fraction(
        radius_fraction=0.5, azimuth_deg=0.0, elevation_deg=0.0
    )


def build_grasp_scene(
    manifest: RobotAssetManifestV1,
    base_xml: str,
    effector: EffectorV1,
    frame: WorkspaceFrame,
) -> GraspScene:
    """Add a support surface and a graspable block to a robot's model."""

    if effector.max_aperture_m is None or not effector.can_grasp:
        raise RigbyGeneralError(
            GeneralFailureCode.UNAFFORDED_SCHEMA,
            f"{manifest.rig_id} has no gripper to grasp with",
            details={"effector": effector.name},
        )

    half_extent = 0.5 * BLOCK_APERTURE_FRACTION * effector.max_aperture_m
    mass = max(0.005, BLOCK_DENSITY * (2.0 * half_extent) ** 3)
    payload = manifest.morphology.scale.payload_kg
    if mass > payload:
        mass = max(0.005, payload * 0.5)

    # Never below the base's own footprint: a block under the robot is not a
    # grasp target, it is a modelling error.
    placement = _reachable_placement(
        frame, minimum_height_m=float(frame.origin[2]) * 0.25 + 0.02
    )
    support_top = float(placement[2])
    block_centre = np.array(
        [placement[0], placement[1], support_top + half_extent], dtype=float
    )

    root = ET.fromstring(base_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:  # pragma: no cover - every compiled model has one
        raise RigbyGeneralError(
            GeneralFailureCode.INTERNAL_ERROR, "model has no worldbody"
        )

    # A static plate rather than an infinite floor: an infinite plane at the
    # block's height would swallow the robot's own base.
    support = ET.SubElement(worldbody, "body", name="scene_support")
    support.set("pos", f"{block_centre[0]:.6f} {block_centre[1]:.6f} {support_top:.6f}")
    ET.SubElement(
        support,
        "geom",
        name=SUPPORT_GEOM,
        type="box",
        size=f"{half_extent * 6:.6f} {half_extent * 6:.6f} {SUPPORT_MARGIN:.6f}",
        solref="0.002 1",
        solimp="0.95 0.99 0.001",
        pos=f"0 0 {-SUPPORT_MARGIN:.6f}",
        rgba="0.35 0.35 0.4 1",
    )

    block = ET.SubElement(worldbody, "body", name=OBJECT_BODY)
    block.set(
        "pos", f"{block_centre[0]:.6f} {block_centre[1]:.6f} {block_centre[2]:.6f}"
    )
    ET.SubElement(block, "freejoint", name="scene_block_free")
    ET.SubElement(
        block,
        "geom",
        name=OBJECT_GEOM,
        type="box",
        size=f"{half_extent:.6f} {half_extent:.6f} {half_extent:.6f}",
        mass=f"{mass:.6f}",
        friction="1.4 0.03 0.001",
        rgba="0.85 0.45 0.15 1",
        condim="4",
        # Stiffer than MuJoCo's default contact, and deliberately so. The default
        # solver reference is soft enough that a firm grip sinks the jaws several
        # millimetres into the block -- which is not a grasp of a rigid object,
        # it is a grasp of a sponge, and the penetration gate correctly refuses
        # it. Tightening the reference time constant is the honest fix: it makes
        # the block behave like the rigid body it is meant to be, rather than
        # loosening the gate to accept a soft one.
        solref="0.002 1",
        solimp="0.95 0.99 0.001",
    )
    ET.SubElement(block, "site", name=OBJECT_SITE, pos="0 0 0", size="0.002")

    xml = ET.tostring(root, encoding="unicode")
    try:
        model = mujoco.MjSpec.from_string(xml).compile()
    except ValueError as error:  # pragma: no cover - defensive
        raise RigbyGeneralError(
            GeneralFailureCode.INTERNAL_ERROR,
            f"the grasp scene would not compile: {error}",
        ) from error

    return GraspScene(
        model=model,
        xml=xml,
        block_half_extent_m=half_extent,
        block_mass_kg=mass,
        block_position_m=block_centre,
        support_height_m=support_top,
        approach_height_m=support_top + half_extent * 4.0,
    )


def block_qpos_address(model: mujoco.MjModel) -> int:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")
    if joint < 0:  # pragma: no cover - defensive
        raise KeyError("scene block has no free joint")
    return int(model.jnt_qposadr[joint])


def block_height(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    return float(qpos[block_qpos_address(model) + 2])
