"""Attempt a named object in an authored world.

The derived probe and this differ in what they are allowed to change. The probe
sizes and places its block from the arm, so it can always construct a workable
approach. Here the object is where the file says, at the size the file says, and
the only thing that can adapt is the arm's path to it.

Getting the approach right needs to know how far the hand reaches past its own
grasp centre, and that has to be measured along the hand's pointing axis. Doing
it along world up at the rest pose -- where these arms stand upright with the
hand facing the ceiling -- measures the gripper backwards and reports every one
of them as having a palm that protrudes past its fingers. They do not: the
fingers lead by 30 to 110 mm on every generated arm. Only the Franka hand really
is palm-first, by 12 mm, which is a fact about that hand and not about all of
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..grounding.workspace import WorkspaceFrame
from ..scenes.environment import (
    OBJECT_PREFIX,
    EnvironmentV1,
    SceneObjectV1,
    object_qpos_address,
)


PHYSICS_HZ = 240

# Clearance above the object's top face at which the approach hovers, as a
# multiple of the object's own height. Far enough that a swing into position
# cannot clip it, close enough that the descent is short.
HOVER_FACTION = 2.0

# How far a bystanding object may be nudged before the attempt counts as having
# disturbed the world. A tenth of its own span: perceptible, but not so tight
# that settling contact registers as a shove.
BYSTANDER_TOLERANCE_FRACTION = 0.10

# Authored objects sit where their world puts them, generally low with the
# envelope to spare, so the lift can afford to clear the required height by a
# margin rather than only just. Measured: 1 of 13 held at 2.5, 3 of 13 at 5.0.
AUTHORED_LIFT_FRACTION = 5.0


@dataclass(frozen=True, slots=True)
class TaskGeometry:
    """The measured path to one object, in world coordinates."""

    hover_m: np.ndarray
    grasp_m: np.ndarray
    lift_m: np.ndarray
    palm_clearance_m: float
    finger_clearance_m: float


def _support_extent_below(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: tuple[str, ...],
    origin: np.ndarray,
    up: np.ndarray,
) -> float:
    """How far these bodies reach below a point, along -up.

    The support point of each geom's oriented bounding box, so a rotated jaw is
    measured by where it actually is rather than by its half-extents.
    """

    worst = -np.inf
    for name in body_names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) != body:
                continue
            centre = np.array(data.geom_xpos[geom], dtype=float)
            rotation = np.array(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            size = np.array(model.geom_size[geom], dtype=float)
            kind = int(model.geom_type[geom])
            if kind == 6:
                half = size[:3]
            elif kind == 2:
                half = np.array([size[0]] * 3)
            elif kind in (3, 5):
                half = np.array([size[0], size[0], size[1]])
            else:
                half = np.array([float(np.max(size))] * 3)
            reach = float(np.sum(np.abs(rotation.T @ up) * half))
            worst = max(worst, float((origin - centre) @ up) + reach)
    return 0.0 if worst == -np.inf else worst


def measure_clearances(
    model: mujoco.MjModel,
    manifest: RobotAssetManifestV1,
    effector: EffectorV1,
    grasp_site: str,
    up: "np.ndarray | None" = None,
) -> tuple[float, float]:
    """How far the palm and the fingers reach past the grasp centre.

    Measured along the *gripper's own* pointing axis -- the grasp-centre site's
    +z -- and not along world up. That distinction is the entire content of this
    function. At the rest pose these arms stand straight up with the hand facing
    the ceiling, so measuring along world -up measures the hand backwards: it
    reports the palm protruding past fingers that are in fact 30 to 110 mm ahead
    of it, and turns a working gripper into an apparently impossible one.

    ``up`` is accepted and ignored, so callers that pass a world axis get the
    right answer rather than a plausible wrong one.
    """

    data = mujoco.MjData(model)
    # The model may be a scene: robot joints first, then whatever the world adds.
    # Only the robot's own portion is being posed here.
    rest = np.asarray(manifest.rest_qpos, dtype=float)
    data.qpos[: min(len(rest), model.nq)] = rest[: model.nq]
    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site)
    origin = np.array(data.site_xpos[site], dtype=float)
    rotation = np.array(data.site_xmat[site], dtype=float).reshape(3, 3)
    axis = -rotation[:, 2]

    palm_body = next(
        (s.body for s in manifest.morphology.sites if s.name == grasp_site),
        effector.tip_body,
    )
    palm = _support_extent_below(model, data, (palm_body,), origin, axis)
    fingers = _support_extent_below(
        model, data, effector.member_bodies, origin, axis
    )
    return palm, fingers


def plan_geometry(
    model: mujoco.MjModel,
    manifest: RobotAssetManifestV1,
    effector: EffectorV1,
    grasp_site: str,
    frame: WorkspaceFrame,
    item: SceneObjectV1,
) -> TaskGeometry:
    """Where to hover, where to stop descending, and where to lift to."""

    up = np.asarray(frame.up, dtype=float)
    palm, fingers = measure_clearances(model, manifest, effector, grasp_site, up)

    centre = np.asarray(item.position_m, dtype=float)
    height = float(item.size_m[2])

    # Stop the descent before the palm reaches the object's top face. Whatever
    # hangs lowest decides -- on every gripper measured that is the palm.
    lowest = max(palm, fingers)
    standoff = max(lowest - height, 0.0)
    grasp = centre + up * standoff

    hover = centre + up * (standoff + HOVER_FACTION * height * 2.0)
    lift = hover + up * (2.0 * height)
    return TaskGeometry(hover, grasp, lift, palm, fingers)


@dataclass(frozen=True, slots=True)
class BystanderReport:
    name: str
    moved_m: float
    tolerance_m: float

    @property
    def disturbed(self) -> bool:
        return self.moved_m > self.tolerance_m


def bystander_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    environment: EnvironmentV1,
    target_name: str,
) -> tuple[BystanderReport, ...]:
    """How far every object that was not the target ended up moving.

    An attempt that lands the target but sweeps its neighbours off the table has
    not done the task, and no gate on the target alone can notice.
    """

    reports = []
    for item in environment.objects:
        if item.name == target_name:
            continue
        address = object_qpos_address(model, item.name)
        start = np.asarray(qpos[0, address : address + 3], dtype=float)
        end = np.asarray(qpos[-1, address : address + 3], dtype=float)
        reports.append(
            BystanderReport(
                item.name,
                float(np.linalg.norm(end - start)),
                BYSTANDER_TOLERANCE_FRACTION * item.span_m,
            )
        )
    return tuple(reports)


def object_geom_names(item: SceneObjectV1) -> frozenset[str]:
    return frozenset({f"{OBJECT_PREFIX}{item.name}_geom"})


def build_task_scene(
    manifest: RobotAssetManifestV1,
    mjcf_xml: str,
    environment: EnvironmentV1,
    target_name: str,
):
    """An authored world, presented to the existing grasp runner.

    The runner already knows how to approach, close and gate. What it assumes is
    that the thing being picked up carries the derived scene's canonical names,
    so the target object is emitted under those names and every other object
    keeps its own. Nothing about the world changes -- the same file produces the
    same geometry whichever object is named -- but the runner needs no knowledge
    of environments at all.
    """

    from ..scenes.block import GraspScene
    from ..scenes.environment import build_environment_model

    target = environment.object_by_name(target_name)
    model, xml = build_environment_model(manifest, mjcf_xml, environment)

    # Re-emit with the target renamed to what the runner looks for.
    xml = (
        xml.replace(f'"{OBJECT_PREFIX}{target_name}_free"', '"scene_block_free"')
        .replace(f'"{OBJECT_PREFIX}{target_name}_geom"', '"scene_block_geom"')
        .replace(f'"{OBJECT_PREFIX}{target_name}"', '"scene_block"')
        .replace(f'"{OBJECT_PREFIX}{target_name}_site"', '"scene_block_center"')
    )
    model = mujoco.MjSpec.from_string(xml).compile()

    half = float(target.size_m[2])
    support = float(target.position_m[2]) - half
    return GraspScene(
        model=model,
        xml=xml,
        block_half_extent_m=half,
        block_mass_kg=target.mass_kg,
        block_position_m=np.asarray(target.position_m, dtype=float),
        support_height_m=support,
        approach_height_m=support + half * 2.0 + HOVER_FACTION * half * 2.0,
        lift_fraction=AUTHORED_LIFT_FRACTION,
    )
