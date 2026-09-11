"""Build the EEZYbotARM MK1 from its published kinematics, not from a sketch.

`assets/general/irl/eezybotarm_mk1` was first transcribed from
`inaciose/ebamk1_description`, a community ROS package. That file is a stick
figure: it carries no mass, no collision geometry, no gripper, and -- the part
that matters -- joint limits that do not describe this arm. Its elbow range does
not overlap the real one at any point, so every primitive certified against it
was certified against poses the machine cannot hold.

What is authoritative here
--------------------------
Link lengths and joint limits come from `meisben/easyEEZYbotARM` (MIT), whose
Mk1 class carries the numbers its author determined experimentally on the
physical arm:

    L1 = 61 mm   base plate to shoulder axis
    L2 = 80 mm   main arm
    L3 = 80 mm   horarm
    L4 = 57 mm   wrist to end effector, horizontal
    q1 in [-30, 30] deg
    q2 in [ 39, 120] deg
    q3 limits are a *function of q2*: q3_min = -0.6755*q2 - 70.768
                                      q3_max = -0.7165*q2 - 13.144

Three things are approximations, and each is marked in `references.json`.

**The elbow range is over-approximated.** A URDF joint limit is a constant and
the real one is not: at q2 = 39 the elbow may swing [-97, -41], at q2 = 120 it
may swing [-152, -99], and those two windows do not intersect. There is no
constant interval that is both non-empty and always legal, so this takes the
union and lets the firmware enforce the coupling -- which is what
`easyEEZYbotARM.checkErrorJointLimits` already does on the real arm. Motions
certified here may therefore include elbow/shoulder pairs the machine refuses.
That is the honest direction to err for a description, but it is a real gap and
not a rounding difference.

**The MK1 has no wrist servo.** Four servos: base, main arm, horarm, gripper.
The old description gave it a wrist pitch joint, which does not exist. The
parallelogram holds the gripper plate at a fixed attitude relative to the
*base*, which a URDF tree cannot express without a mimic joint that MuJoCo drops
anyway. So the plate is fixed to the horarm here, and its attitude diverges from
the real one by the horarm's own pitch. Under-promising: the model cannot be
asked for wrist orientation it does not have.

**The gripper is scaled, not measured.** No MK1 gripper mesh is published. The
jaw geometry is the MK2's `claw_l`/`claw_r` STL bounding boxes measured directly
(58.3 x 21.5 x 9.0 mm, pivots 17 mm apart, 0 to 0.65 rad) and scaled by the arm
ratio L2_mk1 / L2_mk2 = 80 / 135. The finger cross-section is narrowed to 3 mm
and each jaw canted inward, because a straight box standing in for a hooked claw
keeps its closest surfaces at the pivot and measures as never closing at all.

Masses are PLA at 25% infill over each link's own extent, plus 9 g point masses
where the SG90s actually sit -- three on the base assembly, one at the gripper.

Effort and velocity limits are declared, and declared honestly: 0.176 N*m is an
SG90's stall torque (1.8 kg*cm), 3.5 rad/s a loaded traverse rate. Declaring
real numbers is the point. The published SO-101 declares effort="10"
velocity="10" on all six joints and bakes 8 primitives of 72; the same file with
those limits removed bakes 71. A placeholder is worse than an omission, and both
are worse than a measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "general" / "irl"

# --- easyEEZYbotARM Mk1 class attributes, in metres ------------------------
L1 = 0.061
L2 = 0.080
L3 = 0.080
L4 = 0.057

Q1_MIN, Q1_MAX = math.radians(-30.0), math.radians(30.0)
Q2_MIN, Q2_MAX = math.radians(39.0), math.radians(120.0)


def q3_limits(q2_degrees: float) -> tuple[float, float]:
    """The elbow window at one shoulder angle, per easyEEZYbotARM."""

    return (
        -0.6755 * q2_degrees - 70.768,
        -0.7165 * q2_degrees - 13.144,
    )


# The union across the legal shoulder range. See the module docstring: the
# intersection is empty, so a constant limit has to over-approximate.
_LOW_AT_39, _HIGH_AT_39 = q3_limits(39.0)
_LOW_AT_120, _HIGH_AT_120 = q3_limits(120.0)
Q3_MIN = math.radians(min(_LOW_AT_39, _LOW_AT_120))
Q3_MAX = math.radians(max(_HIGH_AT_39, _HIGH_AT_120))

# --- gripper, scaled from the MK2 claw meshes ------------------------------
MK2_SCALE = L2 / 0.135
JAW_LENGTH = 0.0583 * MK2_SCALE
JAW_WIDTH = 0.003
JAW_THICK = 0.0090 * MK2_SCALE
JAW_PIVOT_SEPARATION = 0.017 * MK2_SCALE
JAW_OPEN_RAD = 0.65
# The MK2 claw is a hook, not a bar: 21.5 mm of lateral extent on a 58 mm jaw,
# so its tips meet on the centreline while the pivots stay 17 mm apart. A
# straight box loses exactly that, and loses it silently -- hinged at its base,
# a straight jaw keeps its closest surfaces at the pivot, where rotation moves
# nothing. Measured closure on that shape is 10.1 mm open, 10.1 mm closed: zero
# travel, no grip, reported as a rigid tool tip. Canting each jaw inward so the
# tips converge is the least a box can do and still be a claw.
JAW_INWARD_RAD = math.asin((0.017 * (0.080 / 0.135) / 2 - 0.002) / (0.0583 * (0.080 / 0.135)))
CLAW_BASE = (0.054 * MK2_SCALE, 0.025 * MK2_SCALE, 0.025 * MK2_SCALE)

# --- materials and actuators ----------------------------------------------
PLA_DENSITY = 1240.0
INFILL = 0.25
SG90_MASS = 0.009
SG90_STALL_NM = 0.176
SG90_SPEED_RAD_S = 3.5

COLUMN = (0.030, 0.030, L1 - 0.020)
ARM_SECTION = (0.010, 0.010)
BASE_RADIUS = 0.035
BASE_HEIGHT = 0.020


def _inertial(parent: ET.Element, size, origin, extra_mass: float = 0.0) -> None:
    """A solid box of PLA over ``size``, plus any servo bolted to the link."""

    x, y, z = size
    mass = PLA_DENSITY * INFILL * x * y * z + extra_mass
    node = ET.SubElement(parent, "inertial")
    ET.SubElement(node, "origin", {"xyz": origin, "rpy": "0 0 0"})
    ET.SubElement(node, "mass", {"value": f"{mass:.6f}"})
    ET.SubElement(
        node,
        "inertia",
        {
            "ixx": f"{mass * (y * y + z * z) / 12:.9f}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{mass * (x * x + z * z) / 12:.9f}",
            "iyz": "0",
            "izz": f"{mass * (x * x + y * y) / 12:.9f}",
        },
    )


def _box_link(root, name, size, origin, colour, extra_mass=0.0) -> ET.Element:
    link = ET.SubElement(root, "link", {"name": name})
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": origin, "rpy": "0 0 0"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(
            geometry, "box", {"size": f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}"}
        )
        if tag == "visual":
            ET.SubElement(node, "material", {"name": colour})
    _inertial(link, size, origin, extra_mass)
    return link


def _joint(root, name, parent, child, origin, axis, lower, upper) -> None:
    joint = ET.SubElement(root, "joint", {"name": name, "type": "revolute"})
    ET.SubElement(joint, "origin", {"xyz": origin, "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": f"{lower:.6f}",
            "upper": f"{upper:.6f}",
            "effort": f"{SG90_STALL_NM:.6f}",
            "velocity": f"{SG90_SPEED_RAD_S:.6f}",
        },
    )


def build() -> ET.ElementTree:
    root = ET.Element("robot", {"name": "eezybotarm_mk1"})
    for name, rgba in (("blue", "0 0 0.8 1"), ("grey", "0.4 0.4 0.4 1")):
        material = ET.SubElement(root, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": rgba})

    # Base plate. Carries the base-yaw servo.
    base = ET.SubElement(root, "link", {"name": "base_link"})
    for tag in ("visual", "collision"):
        node = ET.SubElement(base, tag)
        ET.SubElement(node, "origin", {"xyz": f"0 0 {BASE_HEIGHT / 2:.6f}"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(
            geometry,
            "cylinder",
            {"radius": f"{BASE_RADIUS:.6f}", "length": f"{BASE_HEIGHT:.6f}"},
        )
        if tag == "visual":
            ET.SubElement(node, "material", {"name": "grey"})
    _inertial(
        base,
        (BASE_RADIUS * 2, BASE_RADIUS * 2, BASE_HEIGHT),
        f"0 0 {BASE_HEIGHT / 2:.6f}",
        extra_mass=SG90_MASS,
    )

    # Rotating column. Carries the main-arm and horarm servos: the whole point
    # of the IRB460 linkage is that neither of them travels with the arm.
    _joint(
        root, "joint_1", "base_link", "link_1",
        f"0 0 {BASE_HEIGHT:.6f}", "0 0 1", Q1_MIN, Q1_MAX,
    )
    _box_link(
        root, "link_1", COLUMN, f"0 0 {COLUMN[2] / 2:.6f}", "blue",
        extra_mass=2 * SG90_MASS,
    )

    # Main arm. Zero is horizontal along +X; positive pitches the tip up.
    _joint(
        root, "joint_2", "link_1", "link_2",
        f"0 0 {COLUMN[2]:.6f}", "0 -1 0", Q2_MIN, Q2_MAX,
    )
    _box_link(
        root, "link_2", (L2, ARM_SECTION[0], ARM_SECTION[1]),
        f"{L2 / 2:.6f} 0 0", "blue",
    )

    # Horarm.
    _joint(
        root, "joint_3", "link_2", "link_3",
        f"{L2:.6f} 0 0", "0 -1 0", Q3_MIN, Q3_MAX,
    )
    _box_link(
        root, "link_3", (L3, ARM_SECTION[0], ARM_SECTION[1]),
        f"{L3 / 2:.6f} 0 0", "blue",
    )

    # Gripper plate. Fixed, because the servo that would move it does not
    # exist -- see the module docstring.
    plate = ET.SubElement(root, "joint", {"name": "joint_tool", "type": "fixed"})
    ET.SubElement(plate, "origin", {"xyz": f"{L3:.6f} 0 0", "rpy": "0 0 0"})
    ET.SubElement(plate, "parent", {"link": "link_3"})
    ET.SubElement(plate, "child", {"link": "claw_base"})
    _box_link(
        root, "claw_base", CLAW_BASE, f"{CLAW_BASE[0] / 2:.6f} 0 0", "grey",
        extra_mass=SG90_MASS,
    )

    # Two jaws. One SG90 drives both through a gear pair, which is what the
    # <mimic> records; MuJoCo does not implement it and will read them as two
    # independent joints, so closure has to be *measured* rather than declared.
    offset = JAW_PIVOT_SEPARATION / 2
    for side, sign, axis in (("left", 1.0, "0 0 1"), ("right", -1.0, "0 0 -1")):
        name = f"joint_gripper_{side}"
        joint = ET.SubElement(root, "joint", {"name": name, "type": "revolute"})
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": f"{CLAW_BASE[0]:.6f} {sign * offset:.6f} 0",
                "rpy": f"0 0 {-sign * JAW_INWARD_RAD:.6f}",
            },
        )
        ET.SubElement(joint, "parent", {"link": "claw_base"})
        ET.SubElement(joint, "child", {"link": f"jaw_{side}"})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": "0.0",
                "upper": f"{JAW_OPEN_RAD:.6f}",
                "effort": f"{SG90_STALL_NM:.6f}",
                "velocity": f"{SG90_SPEED_RAD_S:.6f}",
            },
        )
        if side == "right":
            ET.SubElement(
                joint,
                "mimic",
                {"joint": "joint_gripper_left", "multiplier": "1", "offset": "0"},
            )
        _box_link(
            root, f"jaw_{side}", (JAW_LENGTH, JAW_WIDTH, JAW_THICK),
            f"{JAW_LENGTH / 2:.6f} {sign * JAW_WIDTH / 2:.6f} 0", "blue",
        )

    return ET.ElementTree(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()

    directory = arguments.out / "eezybotarm_mk1"
    directory.mkdir(parents=True, exist_ok=True)
    tree = build()
    ET.indent(tree, space="  ")
    target = directory / "robot.urdf"
    target.write_bytes(
        b'<?xml version="1.0"?>\n' + ET.tostring(tree.getroot(), encoding="utf-8")
    )

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    print(f"  sha256 {digest}")
    print(f"  q1 {math.degrees(Q1_MIN):.0f}..{math.degrees(Q1_MAX):.0f} deg")
    print(f"  q2 {math.degrees(Q2_MIN):.0f}..{math.degrees(Q2_MAX):.0f} deg")
    print(
        f"  q3 {math.degrees(Q3_MIN):.0f}..{math.degrees(Q3_MAX):.0f} deg"
        " (union over q2; real limit is q2-dependent)"
    )
    print(f"  jaw {JAW_LENGTH * 1000:.1f} mm, 0..{JAW_OPEN_RAD} rad")

    references = arguments.out / "references.json"
    if references.is_file():
        data = json.loads(references.read_text(encoding="utf-8"))
        for record in data.get("robots", []):
            if record.get("robot_id") == "eezybotarm_mk1":
                record["sha256"] = digest
        references.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
