"""Generate the URDF robot zoo used for development and acceptance.

These are real URDF files written to disk and then ingested through exactly the
same path an uploaded robot takes. They use primitive geometry only, so the whole
suite runs offline and deterministically with no mesh downloads or licence
questions.

The zoo is deliberately spread wide, because a morphology analyser that only ever
sees one shape of arm learns nothing:

    id                DOF     effector        reach    role
    zoo_tool_arm      5       tool tip        0.95 m   development
    zoo_jaw_arm       6 + 2   parallel jaw    1.05 m   development
    zoo_hand_arm      7 + 3   multifinger     1.15 m   development
    zoo_dual_arm      2x(5+2) two jaws        0.75 m   development
    zoo_compact_arm   4 + 2   parallel jaw    0.30 m   HELD OUT
    zoo_long_arm      6 + 2   jaw + camera    1.75 m   HELD OUT

Reach spans 0.30 m to 1.75 m -- a factor of nearly six. If any magnitude leaked
into the semantic layer, the small and large arms would need different programs
for the same prompt, and requirement ``schema_invariance`` would fail.

Ground truth is keyed on ``(semantic, body)`` pairs rather than site names,
because the names are the analyser's own output and checking them against itself
would prove nothing. What is being asserted is that a site of the right kind
landed on the right link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"

DEVELOPMENT = "development"
HELD_OUT = "holdout"


# --------------------------------------------------------------------------
# URDF construction
# --------------------------------------------------------------------------


@dataclass
class Link:
    name: str
    shape: str
    size: tuple[float, ...]
    mass: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float
    effort: float
    velocity: float
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Robot:
    robot_id: str
    split: str
    description: str
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    expected_sites: list[tuple[str, str]] = field(default_factory=list)
    expected_effectors: list[dict[str, object]] = field(default_factory=list)
    morphology_class: str = "fixed_base_arm"

    def add_link(self, link: Link) -> Link:
        self.links.append(link)
        return link

    def add_joint(self, joint: Joint) -> Joint:
        self.joints.append(joint)
        return joint


def _inertia_for(shape: str, size: tuple[float, ...], mass: float) -> tuple[float, float, float]:
    """Solid-body principal inertia, so no model needs an inertia repair pass."""

    if shape == "cylinder":
        radius, length = size
        ixx = mass * (3.0 * radius * radius + length * length) / 12.0
        return (ixx, ixx, 0.5 * mass * radius * radius)
    if shape == "box":
        x, y, z = size
        return (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )
    radius = size[0]
    value = 0.4 * mass * radius * radius
    return (value, value, value)


def _geometry(parent: ET.Element, shape: str, size: tuple[float, ...]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if shape == "cylinder":
        ET.SubElement(
            geometry, "cylinder", radius=f"{size[0]:.6f}", length=f"{size[1]:.6f}"
        )
    elif shape == "box":
        ET.SubElement(geometry, "box", size=" ".join(f"{v:.6f}" for v in size))
    else:
        ET.SubElement(geometry, "sphere", radius=f"{size[0]:.6f}")


def _vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def build_urdf(robot: Robot) -> str:
    root = ET.Element("robot", name=robot.robot_id)

    for link in robot.links:
        element = ET.SubElement(root, "link", name=link.name)
        ixx, iyy, izz = _inertia_for(link.shape, link.size, link.mass)

        inertial = ET.SubElement(element, "inertial")
        ET.SubElement(inertial, "mass", value=f"{link.mass:.6f}")
        ET.SubElement(
            inertial, "origin", xyz=_vector(link.origin), rpy=_vector(link.rpy)
        )
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{ixx:.8f}",
            ixy="0",
            ixz="0",
            iyy=f"{iyy:.8f}",
            iyz="0",
            izz=f"{izz:.8f}",
        )

        for tag in ("visual", "collision"):
            node = ET.SubElement(element, tag)
            ET.SubElement(
                node, "origin", xyz=_vector(link.origin), rpy=_vector(link.rpy)
            )
            _geometry(node, link.shape, link.size)

    for joint in robot.joints:
        element = ET.SubElement(root, "joint", name=joint.name, type=joint.kind)
        ET.SubElement(element, "parent", link=joint.parent)
        ET.SubElement(element, "child", link=joint.child)
        ET.SubElement(
            element, "origin", xyz=_vector(joint.origin), rpy=_vector(joint.rpy)
        )
        ET.SubElement(element, "axis", xyz=_vector(joint.axis))
        ET.SubElement(
            element,
            "limit",
            lower=f"{joint.lower:.6f}",
            upper=f"{joint.upper:.6f}",
            effort=f"{joint.effort:.3f}",
            velocity=f"{joint.velocity:.3f}",
        )

    ET.indent(root, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


# --------------------------------------------------------------------------
# Shared builders
# --------------------------------------------------------------------------


def add_base(robot: Robot, *, radius: float, height: float, mass: float) -> str:
    robot.add_link(
        Link("base", "cylinder", (radius, height), mass, origin=(0.0, 0.0, height / 2))
    )
    robot.expected_sites.append(("base", "base"))
    return "base"


def add_serial_arm(
    robot: Robot,
    *,
    prefix: str,
    parent: str,
    mount: tuple[float, float, float],
    mount_rpy: tuple[float, float, float],
    segment_lengths: list[float],
    axes: list[tuple[float, float, float]],
    radius: float,
    effort: float,
    velocity: float,
    density: float = 900.0,
) -> str:
    """Emit a serial revolute chain and return the name of its last link."""

    current = parent
    origin = mount
    rpy = mount_rpy

    for index, (length, axis) in enumerate(zip(segment_lengths, axes)):
        link_name = f"{prefix}link_{index}"
        mass = max(0.25, density * math.pi * radius * radius * length)
        robot.add_link(
            Link(
                link_name,
                "cylinder",
                (radius, length),
                mass,
                origin=(0.0, 0.0, length / 2),
            )
        )
        robot.add_joint(
            Joint(
                name=f"{prefix}joint_{index}",
                kind="revolute",
                parent=current,
                child=link_name,
                origin=origin,
                rpy=rpy,
                axis=axis,
                lower=-2.85,
                upper=2.85,
                effort=effort,
                velocity=velocity,
            )
        )
        current = link_name
        origin = (0.0, 0.0, length)
        rpy = (0.0, 0.0, 0.0)
        radius = max(0.022, radius * 0.88)

    return current


def add_parallel_jaw(
    robot: Robot,
    *,
    prefix: str,
    parent: str,
    mount_z: float,
    finger_length: float,
    stroke: float,
    effort: float = 60.0,
) -> None:
    palm = f"{prefix}palm"
    robot.add_link(
        Link(
            palm,
            "box",
            (0.09, 0.05, 0.05),
            0.35,
            origin=(0.0, 0.0, 0.025),
        )
    )
    robot.add_joint(
        Joint(
            name=f"{prefix}palm_fixed",
            kind="fixed",
            parent=parent,
            child=palm,
            origin=(0.0, 0.0, mount_z),
            axis=(0.0, 0.0, 1.0),
            lower=0.0,
            upper=0.0,
            effort=0.0,
            velocity=0.0,
        )
    )

    for side, direction in (("left", 1.0), ("right", -1.0)):
        finger = f"{prefix}finger_{side}"
        robot.add_link(
            Link(
                finger,
                "box",
                (0.014, 0.03, finger_length),
                0.06,
                origin=(0.0, 0.0, finger_length / 2),
            )
        )
        robot.add_joint(
            Joint(
                name=f"{prefix}grip_{side}",
                kind="prismatic",
                parent=palm,
                child=finger,
                origin=(direction * (stroke / 2 + 0.008), 0.0, 0.05),
                axis=(-direction, 0.0, 0.0),
                lower=0.0,
                upper=stroke / 2,
                effort=effort,
                velocity=0.4,
            )
        )

    members = sorted([f"{prefix}finger_left", f"{prefix}finger_right"])
    robot.expected_sites.append(("grasp_center", palm))
    robot.expected_sites.append(("tip", members[0]))
    robot.expected_sites.extend(("contact", member) for member in members)
    robot.expected_effectors.append(
        {
            "kind": "parallel_jaw",
            "member_bodies": members,
            "grip_joints": sorted([f"{prefix}grip_left", f"{prefix}grip_right"]),
        }
    )


def add_three_finger_hand(
    robot: Robot, *, prefix: str, parent: str, mount_z: float
) -> None:
    palm = f"{prefix}palm"
    robot.add_link(
        Link(palm, "cylinder", (0.05, 0.04), 0.4, origin=(0.0, 0.0, 0.02))
    )
    robot.add_joint(
        Joint(
            name=f"{prefix}palm_fixed",
            kind="fixed",
            parent=parent,
            child=palm,
            origin=(0.0, 0.0, mount_z),
            axis=(0.0, 0.0, 1.0),
            lower=0.0,
            upper=0.0,
            effort=0.0,
            velocity=0.0,
        )
    )

    members: list[str] = []
    grips: list[str] = []
    # One opposing digit plus two on the far side: the classic tripod, and the
    # arrangement the opposition-group search has to recover from geometry alone.
    for index, angle_deg in enumerate((0.0, 140.0, 220.0)):
        angle = math.radians(angle_deg)
        digit = f"{prefix}digit_{index}"
        robot.add_link(
            Link(digit, "box", (0.018, 0.018, 0.085), 0.05, origin=(0.0, 0.0, 0.0425))
        )
        robot.add_joint(
            Joint(
                name=f"{prefix}digit_{index}_flex",
                kind="revolute",
                parent=palm,
                child=digit,
                origin=(0.042 * math.cos(angle), 0.042 * math.sin(angle), 0.04),
                rpy=(0.0, 0.0, angle),
                # Negated so positive flex swings the tip radially inward toward
                # the palm axis. With +y the digits splay outward and nothing on
                # the hand would register as closing.
                axis=(0.0, -1.0, 0.0),
                lower=0.0,
                # Enough flex to bring the tips to the palm axis and no further.
                # A digit that swings past centre re-separates from its
                # neighbours, and the closure sweep correctly reads that as a
                # non-monotone approach rather than as a grip.
                upper=0.55,
                effort=18.0,
                velocity=2.5,
            )
        )
        members.append(digit)
        grips.append(f"{prefix}digit_{index}_flex")

    members = sorted(members)
    robot.expected_sites.append(("grasp_center", palm))
    robot.expected_sites.append(("tip", members[0]))
    robot.expected_sites.extend(("contact", member) for member in members)
    robot.expected_effectors.append(
        {
            "kind": "multifinger",
            "member_bodies": members,
            "grip_joints": sorted(grips),
        }
    )


# --------------------------------------------------------------------------
# The zoo
# --------------------------------------------------------------------------


def zoo_tool_arm() -> Robot:
    robot = Robot(
        "zoo_tool_arm",
        DEVELOPMENT,
        "Five-axis arm terminating in a rigid tool tip. No closure degree of "
        "freedom anywhere, so nothing on it may be classified as a gripper.",
    )
    base = add_base(robot, radius=0.10, height=0.14, mass=6.0)
    last = add_serial_arm(
        robot,
        prefix="",
        parent=base,
        mount=(0.0, 0.0, 0.14),
        mount_rpy=(0.0, 0.0, 0.0),
        segment_lengths=[0.24, 0.30, 0.24, 0.10, 0.07],
        axes=[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
        ],
        radius=0.052,
        effort=140.0,
        velocity=2.1,
    )
    robot.add_link(Link("tool", "cylinder", (0.012, 0.09), 0.18, origin=(0, 0, 0.045)))
    robot.add_joint(
        Joint(
            "tool_fixed",
            "fixed",
            last,
            "tool",
            (0.0, 0.0, 0.07),
            (0.0, 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )
    robot.expected_sites.append(("tip", "tool"))
    robot.expected_effectors.append(
        {"kind": "tool_tip", "member_bodies": [], "grip_joints": []}
    )
    return robot


def zoo_jaw_arm() -> Robot:
    robot = Robot(
        "zoo_jaw_arm",
        DEVELOPMENT,
        "Six-axis industrial arm with a two-finger parallel jaw.",
    )
    base = add_base(robot, radius=0.11, height=0.16, mass=8.0)
    last = add_serial_arm(
        robot,
        prefix="",
        parent=base,
        mount=(0.0, 0.0, 0.16),
        mount_rpy=(0.0, 0.0, 0.0),
        segment_lengths=[0.22, 0.34, 0.28, 0.12, 0.09, 0.06],
        axes=[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        radius=0.055,
        effort=160.0,
        velocity=2.2,
    )
    add_parallel_jaw(
        robot, prefix="", parent=last, mount_z=0.06, finger_length=0.075, stroke=0.085
    )
    return robot


def zoo_hand_arm() -> Robot:
    robot = Robot(
        "zoo_hand_arm",
        DEVELOPMENT,
        "Seven-axis redundant arm with a three-finger hand. The extra axis must "
        "be recognised as redundant rather than as another positioning joint.",
    )
    base = add_base(robot, radius=0.12, height=0.18, mass=9.0)
    last = add_serial_arm(
        robot,
        prefix="",
        parent=base,
        mount=(0.0, 0.0, 0.18),
        mount_rpy=(0.0, 0.0, 0.0),
        segment_lengths=[0.20, 0.30, 0.14, 0.28, 0.10, 0.08, 0.05],
        axes=[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        radius=0.058,
        effort=170.0,
        velocity=2.0,
    )
    add_three_finger_hand(robot, prefix="", parent=last, mount_z=0.05)
    return robot


def zoo_dual_arm() -> Robot:
    robot = Robot(
        "zoo_dual_arm",
        DEVELOPMENT,
        "Bimanual platform: two mirrored five-axis arms with parallel jaws on a "
        "shared torso. The mirror plane has to be measured, not assumed.",
        morphology_class="fixed_base_bimanual",
    )
    add_base(robot, radius=0.16, height=0.30, mass=22.0)
    robot.add_link(Link("torso", "box", (0.22, 0.46, 0.10), 7.0, origin=(0, 0, 0.05)))
    robot.add_joint(
        Joint(
            "torso_fixed",
            "fixed",
            "base",
            "torso",
            (0.0, 0.0, 0.30),
            (0.0, 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )

    for side, sign in (("left", 1.0), ("right", -1.0)):
        last = add_serial_arm(
            robot,
            prefix=f"{side}_",
            parent="torso",
            # Clear of the torso in both axes: the shoulders sit outboard of
            # its side faces and above its top, so nothing is in contact at
            # rest. An arm resting against the torso is not merely untidy --
            # it is jammed, and its base joint saturates at full torque
            # without moving at all.
            mount=(0.0, sign * 0.30, 0.14),
            # Negated: a positive roll about +x tilts the arm's +z toward -y,
            # i.e. toward its neighbour. Splaying them outward is what keeps the
            # two arms from starting the simulation already inside each other.
            mount_rpy=(-sign * 0.40, 0.0, 0.0),
            segment_lengths=[0.16, 0.24, 0.20, 0.09, 0.06],
            axes=[
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
            ],
            radius=0.042,
            effort=95.0,
            velocity=2.4,
        )
        add_parallel_jaw(
            robot,
            prefix=f"{side}_",
            parent=last,
            mount_z=0.06,
            finger_length=0.06,
            stroke=0.07,
            effort=45.0,
        )
    return robot


def zoo_compact_arm() -> Robot:
    robot = Robot(
        "zoo_compact_arm",
        HELD_OUT,
        "Desktop-scale four-axis arm with a small jaw. Roughly one sixth the "
        "reach of the longest arm in the zoo.",
    )
    base = add_base(robot, radius=0.045, height=0.05, mass=0.9)
    last = add_serial_arm(
        robot,
        prefix="",
        parent=base,
        mount=(0.0, 0.0, 0.05),
        mount_rpy=(0.0, 0.0, 0.0),
        segment_lengths=[0.07, 0.10, 0.08, 0.035],
        axes=[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        radius=0.018,
        effort=14.0,
        velocity=3.2,
        density=700.0,
    )
    add_parallel_jaw(
        robot, prefix="", parent=last, mount_z=0.035, finger_length=0.03, stroke=0.03,
        effort=8.0,
    )
    return robot


def zoo_long_arm() -> Robot:
    robot = Robot(
        "zoo_long_arm",
        HELD_OUT,
        "Long-reach six-axis arm with a jaw and a wrist camera. Mounted with a "
        "rotated base frame so the front direction cannot be assumed from the "
        "world axes.",
    )
    base = add_base(robot, radius=0.18, height=0.24, mass=26.0)
    last = add_serial_arm(
        robot,
        prefix="",
        parent=base,
        mount=(0.0, 0.0, 0.24),
        mount_rpy=(0.0, 0.0, math.radians(35.0)),
        segment_lengths=[0.34, 0.58, 0.46, 0.18, 0.12, 0.08],
        axes=[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        radius=0.085,
        effort=420.0,
        velocity=1.6,
    )
    robot.add_link(
        Link("wrist_camera", "box", (0.05, 0.05, 0.03), 0.2, origin=(0, 0, 0.015))
    )
    robot.add_joint(
        Joint(
            "wrist_camera_fixed",
            "fixed",
            last,
            "wrist_camera",
            (0.06, 0.0, 0.02),
            (0.0, 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )
    robot.expected_sites.append(("gaze", "wrist_camera"))
    robot.expected_effectors.append(
        {"kind": "sensor", "member_bodies": [], "grip_joints": []}
    )
    add_parallel_jaw(
        robot, prefix="", parent=last, mount_z=0.08, finger_length=0.11, stroke=0.13,
        effort=110.0,
    )
    return robot


ZOO_BUILDERS = (
    zoo_tool_arm,
    zoo_jaw_arm,
    zoo_hand_arm,
    zoo_dual_arm,
    zoo_compact_arm,
    zoo_long_arm,
)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def write_robot(robot: Robot, root: Path) -> dict[str, object]:
    directory = root / robot.robot_id
    directory.mkdir(parents=True, exist_ok=True)

    urdf = build_urdf(robot)
    urdf_path = directory / "robot.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    digest = hashlib.sha256(urdf.encode("utf-8")).hexdigest()
    (directory / "robot.urdf.sha256").write_text(
        f"{digest}  robot.urdf\n", encoding="ascii"
    )

    ground_truth = {
        "robot_id": robot.robot_id,
        "split": robot.split,
        "description": robot.description,
        "morphology_class": robot.morphology_class,
        "source_sha256": digest,
        "expected_sites": sorted(
            [{"semantic": semantic, "body": body} for semantic, body in robot.expected_sites],
            key=lambda item: (item["semantic"], item["body"]),
        ),
        "expected_effectors": sorted(
            robot.expected_effectors, key=lambda item: (item["kind"], str(item["member_bodies"]))
        ),
        "link_count": len(robot.links),
        "actuated_joint_count": sum(1 for joint in robot.joints if joint.kind != "fixed"),
    }
    truth_path = directory / "ground_truth.json"
    payload = json.dumps(ground_truth, indent=2, sort_keys=True) + "\n"
    truth_path.write_text(payload, encoding="utf-8")
    (directory / "ground_truth.json.sha256").write_text(
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  ground_truth.json\n",
        encoding="ascii",
    )
    return ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ZOO_ROOT)
    arguments = parser.parse_args()

    root: Path = arguments.root
    root.mkdir(parents=True, exist_ok=True)

    entries = []
    for builder in ZOO_BUILDERS:
        robot = builder()
        truth = write_robot(robot, root)
        entries.append(
            {
                "robot_id": truth["robot_id"],
                "split": truth["split"],
                "morphology_class": truth["morphology_class"],
                "source": f"{truth['robot_id']}/robot.urdf",
                "source_sha256": truth["source_sha256"],
                "actuated_joint_count": truth["actuated_joint_count"],
            }
        )
        print(
            f"{truth['robot_id']:<18} {truth['split']:<12} "
            f"{truth['actuated_joint_count']:>2} dof  {truth['link_count']:>2} links"
        )

    index = {
        "schema_version": "1.0",
        "development": [e for e in entries if e["split"] == DEVELOPMENT],
        "holdout": [e for e in entries if e["split"] == HELD_OUT],
    }
    index_payload = json.dumps(index, indent=2, sort_keys=True) + "\n"
    index_path = root.parent / "benchmark" / "robots.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_payload, encoding="utf-8")
    index_path.with_suffix(".json.sha256").write_text(
        f"{hashlib.sha256(index_payload.encode('utf-8')).hexdigest()}  robots.json\n",
        encoding="ascii",
    )
    print(f"\nindex -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
