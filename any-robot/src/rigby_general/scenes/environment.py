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
import math
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

BLOCK_DENSITY = 320.0
"""Density of a graspable block, matching the derived probe scene."""

MIN_BLOCK_DENSITY = 30.0
"""The lightest a block may be made before it has to be made smaller instead.

Expanded polystyrene, roughly. Below this nothing real is that light and the
contact solver stops behaving. Swept against the authored worlds: 30 holds
thirty-three, 5 holds thirty-one -- the same as no floor at all -- and 120
twenty-nine."""

CONTACT_WIDTH_M = 0.001
"""How deep the contact softening zone runs before the constraint goes hard.

Swept in both directions and this is the peak: 0.003 holds thirty-three, 0.006
thirty-two, 0.012 twenty-eight, and going harder is no better -- 0.0005 and
0.0002 both hold thirty-four."""

CONTACT_TIMECONST_S = 0.002
"""Contact time constant, in seconds.

Below what MuJoCo asks for, knowingly. The models integrate at 2 ms, so the
documented floor is 4 ms, and these contacts are half that. Softening to the
floor costs holds -- 0.003 holds thirty-six, 0.004 thirty-five, 0.005
thirty-three, 0.0084 twenty-six and 0.03 two -- because a grip needs the
contact stiff enough that the object does not sink into the jaws. Stepping
faster instead, so the floor comes down to meet it, does not help either
(480 Hz holds twenty-eight, 960 Hz twenty-nine), and neither does an implicit
integrator, which leaves the one-gram block on the SO-ARM101 launched to
exactly the same millimetre. Stiff and technically under-resolved is measurably
the best of these, and the residual instability is real: a 1.5 g block between
the jaws of a 0.6 kg arm still takes 77 N and ends up at -9.3 m."""


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

    authored_for_reach_m: float | None = Field(default=None, gt=0.0)
    """The reach this world's distances were chosen against, if it declares one.

    A world written in metres is a world written for one size of robot. This
    fleet spans 0.21 m to 2.05 m of reach, so a 24 mm cube 165 mm out is a fair
    task for exactly the arm it was measured against and an impossibility either
    side of it: of 120 robot-and-object pairings only 33 were ever attempted,
    the rest turned away for reach or aperture before anything moved. Declaring
    what the numbers were chosen against is what lets the same *task* -- a block
    on a bench, at the edge of comfortable reach -- be put to any body."""

    authored_for_aperture_m: float | None = Field(default=None, gt=0.0)
    """The jaw opening this world's object sizes were chosen against."""

    def scaled_to(
        self, reach_m: float, aperture_m: float, payload_kg: float | None = None
    ) -> "EnvironmentV1":
        """This world as it would have been authored for a body of this size.

        Distances scale with reach and object sizes with jaw opening, because
        those are the two measurements that decide whether a task is possible:
        whether the robot can get there, and whether it can close on what it
        finds. Heights above the bench scale with the object so a block still
        rests on its surface, and mass follows volume so a scaled block is the
        same material rather than the same weight.

        ``payload_kg`` caps the object's mass at what the arm can actually hold.
        Scaling mass with volume is right for keeping a block the same material,
        and wrong once the block outgrows the gripper: the squeeze force is
        `min(required, half the actuator)`, so an object needing more than the
        hand has is gripped at less than it needs and slips. Measured through
        the lift, the Panda holds opposition on 0 of 433 steps and the compact
        arm on 59, while the EEZYbotARM -- whose block is within its payload --
        holds it on all 433 and carries the object. A world fitted to an arm
        should present something it can hold, not merely reach.

        A world that declares neither reference is returned unchanged -- it is
        an absolute world and means what it says.
        """

        if not self.authored_for_reach_m or not self.authored_for_aperture_m:
            return self

        span = reach_m / self.authored_for_reach_m
        grip = aperture_m / self.authored_for_aperture_m

        # Objects grow with the jaw and move apart with the reach, and nothing
        # was keeping those two consistent.
        #
        # A hand with a wide opening on a short arm scales its objects up faster
        # than the world spreads them out, and they grow into one another: the
        # Beetlebot's block ended up 60 mm inside a neighbouring crate before
        # anything had moved. A world cannot present a task it has already
        # broken, so the object scale gives way -- the arm still has to reach
        # what it is asked for, and that is what the distances are for.
        for index, first in enumerate(self.objects):
            for second in self.objects[index + 1 :]:
                gap = math.dist(first.position_m, second.position_m) * span
                reach_across = max((first.span_m + second.span_m) / 2.0, 1e-9)
                if reach_across * grip > gap:
                    grip = min(grip, gap / reach_across)
        grip = max(grip, 1e-6)

        # Scale the whole offset, direction intact. Scaling the horizontal
        # distance while deriving the height some other way tilts the bearing,
        # and reach is *directional*: a point moved to a steeper elevation can
        # leave the envelope even though the robot is bigger. Nine pairings were
        # still refused `object_out_of_reach` in worlds scaled to fit them.
        def place(position: tuple[float, float, float], height: float):
            return (position[0] * span, position[1] * span, position[2] * span)

        fixtures = tuple(
            fixture.model_copy(
                update={
                    "size_m": (
                        fixture.size_m[0] * span,
                        fixture.size_m[1] * span,
                        fixture.size_m[2] * span,
                    ),
                    "position_m": place(
                        fixture.position_m, fixture.position_m[2] * span
                    ),
                }
            )
            for fixture in self.fixtures
        )
        objects = []
        for item in self.objects:
            size = tuple(extent * grip for extent in item.size_m)
            # A block the hand cannot hold may be made lighter, but not
            # lighter than a real material.
            #
            # Capping the mass alone drove the density through the floor: the
            # EEZYbotARM was handed a 30 mm cube massing 0.1 g, about four
            # kilogrammes per cubic metre and three times lighter than air.
            # Nothing is made of that, and the contact solver will not integrate
            # it -- the block jittered out of the jaws and was reported as
            # excessive penetration on an object that weighed nothing. Down to
            # `MIN_BLOCK_DENSITY` the block is simply a lighter material, which
            # is honest; below it the block *shrinks* instead, which keeps it
            # made of something. Holding the density fixed at `BLOCK_DENSITY`
            # instead costs five holds, and only the absurd end is the defect.
            if payload_kg and payload_kg > 0.0:
                volume = 8.0 * size[0] * size[1] * size[2]
                if payload_kg < MIN_BLOCK_DENSITY * volume:
                    size = tuple(
                        extent
                        * (payload_kg / (MIN_BLOCK_DENSITY * volume)) ** (1.0 / 3.0)
                        for extent in size
                    )
            # Sit the object back down on whatever it was resting on: its
            # clearance above the bench top scales with the bench, its own
            # half-height with the jaw.
            # Sit it back down on whatever it was resting on: the support moves
            # with the world, the object's own half-height with the jaw.
            support = (item.position_m[2] - item.size_m[2]) * span
            objects.append(
                item.model_copy(
                    update={
                        "size_m": size,
                        # Same material the derived probe uses, then capped by
                        # what this hand can hold. Scaling the authored mass with
                        # volume keeps a block the same stuff as authored, but
                        # the authored stuff differs per world; `BLOCK_DENSITY`
                        # is the density this repository already chose for a
                        # graspable block, so a scaled world presents the same
                        # material to every arm. The cap is what keeps it liftable
                        # by the hand it was fitted to.
                        "mass_kg": max(
                            min(
                                BLOCK_DENSITY * 8.0 * size[0] * size[1] * size[2],
                                payload_kg if payload_kg else float("inf"),
                            ),
                            1e-4,
                        ),
                        "position_m": (
                            item.position_m[0] * span,
                            item.position_m[1] * span,
                            support + size[2],
                        ),
                    }
                )
            )
        return self.model_copy(
            update={"fixtures": fixtures, "objects": tuple(objects)}
        )

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
    asset_root: "Path | None" = None,
) -> tuple[mujoco.MjModel, str]:
    """Compile the robot into the world, with the world unchanged.

    The robot is translated to the environment's mount pose rather than the
    world being translated to the robot, because the world was authored first and
    is the thing that must not move.

    ``asset_root`` is where the robot's meshes live. A URDF names them relatively
    -- ``meshes/kr6_agilus/link_1.stl`` -- and relative to a string being compiled
    in memory means relative to whatever directory the process happens to be in.
    Ingest gets away with that because it runs beside the file; a world does not,
    and every mesh-bearing robot failed to enter one with a compile error naming
    the first mesh it could not open. Passing the root turns the reference back
    into something resolvable from anywhere.
    """

    root = ET.fromstring(mjcf_xml)
    # Same policy the derived scene applies: parts ingest measured as authored
    # inside one another are already excluded from the self-collision gate, so
    # they should not be generating contact forces either.
    from .block import _exclude_authored_overlaps

    _exclude_authored_overlaps(root, manifest)
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
            solref=f"{CONTACT_TIMECONST_S} 1",
            solimp=f"0.95 0.99 {CONTACT_WIDTH_M}",
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
            solref=f"{CONTACT_TIMECONST_S} 1",
            solimp=f"0.95 0.99 {CONTACT_WIDTH_M}",
            # ...and the object has to *win* the contact, or none of the above
            # applies. With equal priority MuJoCo resolves two geoms' solver
            # references by taking the softer of them, so every number here was
            # overridden by whatever the robot's own jaw geoms carried -- usually
            # the compiler default, an order of magnitude softer. The object was
            # authored rigid and simulated as a sponge, and the jaws sank 5 to
            # 22 mm into blocks 18 to 48 mm across. Priority makes the object's
            # own rigidity govern the contact on every robot it meets, which is
            # the only way one authored world means the same thing across a
            # fleet nobody tuned it against.
            priority="1",
        )
        ET.SubElement(
            node, "site", name=f"{OBJECT_PREFIX}{item.name}_site", pos="0 0 0", size="0.002"
        )

    if asset_root is not None:
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        if not compiler.get("meshdir"):
            compiler.set("meshdir", str(Path(asset_root).resolve()))
        if not compiler.get("texturedir"):
            compiler.set("texturedir", str(Path(asset_root).resolve()))

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
