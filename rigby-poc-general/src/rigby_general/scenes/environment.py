"""Worlds that exist before any robot is asked to do anything in them.

The grasp probe builds its block out of the robot: side length a fraction of the
measured aperture, placed at a fraction of the measured reach. That is what makes
one probe meaningful across nine different bodies, and it is also its ceiling. A
block positioned by the reach envelope can never be somewhere the envelope says
is out of reach, so the probe can demonstrate grasping but never *failing to
reach*, and it can never present the same object to two robots.

An environment inverts that. It is authored in absolute metres, in world
coordinates, with no knowledge of which robot will be dropped into it -- and no
knowledge of what will be asked. The robot is mounted at a pose the environment
declares, and then two things can be measured that the derived scene cannot ask:

*Can this robot reach that?* The object is where it is. If it lies outside the
measured envelope, the answer is a typed refusal naming the shortfall in
millimetres, which is a result rather than a failure.

*Can this hand hold that?* The object is the size it is. A jaw that opens 32 mm
is not going to span a 70 mm crate, and saying so is more useful than resizing
the crate until it fits.

Everything here is data. Nothing in an environment file names a robot, and
nothing adapts to one.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from pydantic import Field
from xml.etree import ElementTree as ET

from ..contracts import Contract, RobotAssetManifestV1
from ..errors import GeneralFailureCode, RigbyGeneralError


ENVIRONMENT_ROOT = (
    Path(__file__).resolve().parents[3] / "assets" / "general" / "environments"
)

SUPPORT_PREFIX = "env_fixture_"
OBJECT_PREFIX = "env_object_"


class FixtureV1(Contract):
    """Immovable furniture: a bench, a shelf, a bin wall."""

    name: str = Field(min_length=1)
    size_m: tuple[float, float, float]
    """Half-extents, MuJoCo's convention."""

    position_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.35, 0.35, 0.40, 1.0)


class SceneObjectV1(Contract):
    """A free body a robot may be asked to manipulate."""

    name: str = Field(min_length=1)
    size_m: tuple[float, float, float]
    mass_kg: float = Field(gt=0.0)
    position_m: tuple[float, float, float]
    friction: float = Field(default=1.4, gt=0.0)
    rgba: tuple[float, float, float, float] = (0.85, 0.45, 0.15, 1.0)

    @property
    def span_m(self) -> float:
        """The width a gripper has to open to, across the narrowest face.

        A jaw approaching from above straddles the two smaller horizontal
        dimensions, so the span that matters is the smaller of them.
        """

        return 2.0 * min(self.size_m[0], self.size_m[1])

    @property
    def top_m(self) -> float:
        return self.position_m[2] + self.size_m[2]


class EnvironmentV1(Contract):
    """A world, authored once, in metres."""

    environment_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    robot_mount_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Where the robot's base is bolted. Absolute, like everything else."""

    fixtures: tuple[FixtureV1, ...] = ()
    objects: tuple[SceneObjectV1, ...] = Field(min_length=1)

    def object_by_name(self, name: str) -> SceneObjectV1:
        for item in self.objects:
            if item.name == name:
                return item
        raise KeyError(f"{self.environment_id} has no object named {name!r}")


def load_environment(environment_id: str, *, root: Path | None = None) -> EnvironmentV1:
    path = (root or ENVIRONMENT_ROOT) / f"{environment_id}.json"
    if not path.is_file():
        raise RigbyGeneralError(
            GeneralFailureCode.INTERNAL_ERROR,
            f"no environment named {environment_id!r} at {path}",
        )
    return EnvironmentV1.model_validate(json.loads(path.read_text(encoding="utf-8")))


def available_environments(root: Path | None = None) -> tuple[str, ...]:
    directory = root or ENVIRONMENT_ROOT
    if not directory.is_dir():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.json")))


def build_environment_model(
    manifest: RobotAssetManifestV1,
    mjcf_xml: str,
    environment: EnvironmentV1,
) -> tuple[mujoco.MjModel, str]:
    """Compile the robot into the world, with the world unchanged.

    The robot is translated to the environment's mount pose rather than the
    world being translated to the robot, because the world was authored first and
    is the thing that must not move.
    """

    root = ET.fromstring(mjcf_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:  # pragma: no cover - every compiled model has one
        raise RigbyGeneralError(
            GeneralFailureCode.INTERNAL_ERROR, "model has no worldbody"
        )

    mount = environment.robot_mount_m
    if any(abs(value) > 1e-9 for value in mount):
        for body in worldbody.findall("body"):
            existing = [float(v) for v in (body.get("pos") or "0 0 0").split()]
            body.set(
                "pos",
                " ".join(
                    f"{existing[i] + mount[i]:.6f}" for i in range(3)
                ),
            )

    for fixture in environment.fixtures:
        node = ET.SubElement(
            worldbody, "body", name=f"{SUPPORT_PREFIX}{fixture.name}"
        )
        node.set("pos", " ".join(f"{v:.6f}" for v in fixture.position_m))
        ET.SubElement(
            node,
            "geom",
            name=f"{SUPPORT_PREFIX}{fixture.name}_geom",
            type="box",
            size=" ".join(f"{v:.6f}" for v in fixture.size_m),
            rgba=" ".join(f"{v:.4f}" for v in fixture.rgba),
            solref="0.002 1",
            solimp="0.95 0.99 0.001",
        )

    for item in environment.objects:
        node = ET.SubElement(worldbody, "body", name=f"{OBJECT_PREFIX}{item.name}")
        node.set("pos", " ".join(f"{v:.6f}" for v in item.position_m))
        ET.SubElement(node, "freejoint", name=f"{OBJECT_PREFIX}{item.name}_free")
        ET.SubElement(
            node,
            "geom",
            name=f"{OBJECT_PREFIX}{item.name}_geom",
            type="box",
            size=" ".join(f"{v:.6f}" for v in item.size_m),
            mass=f"{item.mass_kg:.6f}",
            friction=f"{item.friction:.3f} 0.03 0.001",
            rgba=" ".join(f"{v:.4f}" for v in item.rgba),
            condim="4",
            # Same reasoning as the derived block scene: MuJoCo's default contact
            # is soft enough that a firm grip sinks the jaws millimetres into the
            # object, which is a grasp of a sponge rather than of a rigid body.
            solref="0.002 1",
            solimp="0.95 0.99 0.001",
        )
        ET.SubElement(
            node, "site", name=f"{OBJECT_PREFIX}{item.name}_site", pos="0 0 0", size="0.002"
        )

    xml = ET.tostring(root, encoding="unicode")
    try:
        model = mujoco.MjSpec.from_string(xml).compile()
    except ValueError as error:
        raise RigbyGeneralError(
            GeneralFailureCode.INTERNAL_ERROR,
            f"{environment.environment_id} would not compile with "
            f"{manifest.rig_id}: {error}",
        ) from error
    return model, xml


def object_qpos_address(model: mujoco.MjModel, object_name: str) -> int:
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{OBJECT_PREFIX}{object_name}_free"
    )
    if joint < 0:  # pragma: no cover - defensive
        raise KeyError(f"scene has no free joint for object {object_name!r}")
    return int(model.jnt_qposadr[joint])
