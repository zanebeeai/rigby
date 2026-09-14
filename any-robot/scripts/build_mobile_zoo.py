"""Generate the three exotic mobile bodies: a dog with an arm, a wheeled biped, and a rigid segmented octopus.

Procedural models in MJCF with primitive geometry only, written to disk and
then ingested through the mobility path the way an uploaded mobile model
would be. Nothing here is a research holdout: these are development and
showcase bodies. Every body carries a mobility declaration beside it (what
kind of base it is, which members are meant to bear its weight, its parked
and working stances) and a provenance record (procedural, this repository,
redistributable under the repository licence). Masses come from geometry
at declared densities so the compiler's inertias are the solid-body ones;
actuators are position servos on the limbs and the arm, velocity servos on
the wheels, with declared force limits.

    id                    base       limbs                            manipulator
    mobile_dog_arm        legged     4 legs x 3 hinges                4-dof arm + parallel jaw on the torso
    mobile_wheeled_biped  wheeled    2 legs x 2 hinges + 2 wheels     3-dof arm + parallel jaw on the torso, a tail skid to park on
    mobile_octopus        crawling   6 tentacles x 4 segments x 2     two pincer tentacles (hinge fingers)

    python any-robot/scripts/build_mobile_zoo.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


MOBILE_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "mobile"
STEEL = 2200.0
"""kg/m^3, a hollow-ish structural member: between plastic and solid aluminium."""
PLASTIC = 900.0
RUBBER = 1100.0


def fmt(*values: float) -> str:
    return " ".join(f"{v:.5f}" for v in values)


class Builder:
    """A small MJCF writer: bodies, geoms, joints and actuators by name."""

    def __init__(self, robot_id: str, description: str) -> None:
        self.robot_id = robot_id
        self.description = description
        self.root = ET.Element("mujoco", model=robot_id)
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true")
        ET.SubElement(self.root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
        default = ET.SubElement(self.root, "default")
        ET.SubElement(default, "geom", condim="3", friction="1.0 0.005 0.0001", solref="0.01 1", solimp="0.95 0.99 0.001", margin="0.0005")
        ET.SubElement(default, "joint", damping="0.5", armature="0.01", limited="true")
        ET.SubElement(default, "position", ctrllimited="true", forcelimited="true")
        ET.SubElement(default, "velocity", ctrllimited="true", forcelimited="true")
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")
        self.joints: list[dict] = []
        self.actuators: list[dict] = []
        self.support: list[str] = []
        self.bodies: dict[str, ET.Element] = {}

    def body(self, parent: ET.Element | None, name: str, pos: tuple[float, float, float], euler: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> ET.Element:
        element = ET.SubElement(parent if parent is not None else self.worldbody, "body", name=name, pos=fmt(*pos))
        if any(abs(v) > 1e-9 for v in euler):
            element.set("euler", fmt(*euler))
        self.bodies[name] = element
        return element

    def box(self, body: ET.Element, name: str, half: tuple[float, float, float], density: float, pos=(0.0, 0.0, 0.0), rgba="0.55 0.58 0.62 1", **extra) -> None:
        ET.SubElement(body, "geom", name=name, type="box", size=fmt(*half), density=f"{density:.1f}", pos=fmt(*pos), rgba=rgba, **extra)

    def capsule(self, body: ET.Element, name: str, radius: float, length: float, density: float, pos=(0.0, 0.0, 0.0), axis="z", rgba="0.55 0.58 0.62 1", **extra) -> None:
        half = length / 2.0
        if axis == "z":
            fromto = fmt(pos[0], pos[1], pos[2] - half, pos[0], pos[1], pos[2] + half)
        elif axis == "x":
            fromto = fmt(pos[0] - half, pos[1], pos[2], pos[0] + half, pos[1], pos[2])
        else:
            fromto = fmt(pos[0], pos[1] - half, pos[2], pos[0], pos[1] + half, pos[2])
        ET.SubElement(body, "geom", name=name, type="capsule", size=f"{radius:.5f}", fromto=fromto, density=f"{density:.1f}", rgba=rgba, **extra)

    def sphere(self, body: ET.Element, name: str, radius: float, density: float, pos=(0.0, 0.0, 0.0), rgba="0.2 0.2 0.22 1", **extra) -> None:
        ET.SubElement(body, "geom", name=name, type="sphere", size=f"{radius:.5f}", density=f"{density:.1f}", pos=fmt(*pos), rgba=rgba, **extra)

    def cylinder(self, body: ET.Element, name: str, radius: float, half_length: float, density: float, axis_euler: tuple[float, float, float], rgba="0.15 0.15 0.17 1", **extra) -> None:
        ET.SubElement(body, "geom", name=name, type="cylinder", size=fmt(radius, half_length), density=f"{density:.1f}", euler=fmt(*axis_euler), rgba=rgba, **extra)

    def hinge(self, body: ET.Element, name: str, axis: tuple[float, float, float], low: float, high: float, *, effort: float, velocity: float, kp: float, rest: float = 0.0, role: str, limb: str, damping: float | None = None) -> None:
        joint = ET.SubElement(body, "joint", name=name, type="hinge", axis=fmt(*axis), range=fmt(low, high))
        if damping is not None:
            joint.set("damping", f"{damping:.3f}")
        ET.SubElement(self.actuator, "position", name=f"{name}_servo", joint=name, kp=f"{kp:.1f}", ctrlrange=fmt(low, high), forcerange=fmt(-effort, effort))
        self.joints.append({"name": name, "kind": "hinge", "range": [low, high], "effort": effort, "velocity": velocity, "rest": rest, "role": role, "limb": limb, "actuator": f"{name}_servo", "actuator_kind": "position"})
        self.actuators.append({"name": f"{name}_servo", "joint": name, "kind": "position", "kp": kp, "forcerange": [-effort, effort]})

    def slide(self, body: ET.Element, name: str, axis: tuple[float, float, float], low: float, high: float, *, effort: float, velocity: float, kp: float, rest: float, role: str, limb: str) -> None:
        ET.SubElement(body, "joint", name=name, type="slide", axis=fmt(*axis), range=fmt(low, high))
        ET.SubElement(self.actuator, "position", name=f"{name}_servo", joint=name, kp=f"{kp:.1f}", ctrlrange=fmt(low, high), forcerange=fmt(-effort, effort))
        self.joints.append({"name": name, "kind": "slide", "range": [low, high], "effort": effort, "velocity": velocity, "rest": rest, "role": role, "limb": limb, "actuator": f"{name}_servo", "actuator_kind": "position"})
        self.actuators.append({"name": f"{name}_servo", "joint": name, "kind": "position", "kp": kp, "forcerange": [-effort, effort]})

    def wheel(self, body: ET.Element, name: str, axis: tuple[float, float, float], *, effort: float, velocity: float, kv: float, limb: str) -> None:
        ET.SubElement(body, "joint", name=name, type="hinge", axis=fmt(*axis), limited="false", damping="0.05")
        ET.SubElement(self.actuator, "velocity", name=f"{name}_drive", joint=name, kv=f"{kv:.2f}", ctrlrange=fmt(-velocity, velocity), forcerange=fmt(-effort, effort))
        self.joints.append({"name": name, "kind": "hinge", "range": None, "effort": effort, "velocity": velocity, "rest": 0.0, "role": "wheel", "limb": limb, "actuator": f"{name}_drive", "actuator_kind": "velocity"})
        self.actuators.append({"name": f"{name}_drive", "joint": name, "kind": "velocity", "kv": kv, "forcerange": [-effort, effort]})

    def site(self, body: ET.Element, name: str, pos=(0.0, 0.0, 0.0), size: float = 0.005) -> None:
        ET.SubElement(body, "site", name=name, pos=fmt(*pos), size=f"{size:.4f}", rgba="1 0.3 0.3 0.5")

    def imu(self, body_name: str) -> None:
        body = self.bodies[body_name]
        self.site(body, f"{body_name}_imu", size=0.004)
        ET.SubElement(self.sensor, "gyro", name=f"{body_name}_gyro", site=f"{body_name}_imu")
        ET.SubElement(self.sensor, "accelerometer", name=f"{body_name}_accel", site=f"{body_name}_imu")
        ET.SubElement(self.sensor, "framequat", name=f"{body_name}_orientation", objtype="site", objname=f"{body_name}_imu")

    def touch(self, body: ET.Element, body_name: str, pos=(0.0, 0.0, 0.0), size: float = 0.03) -> None:
        self.site(body, f"{body_name}_touch", pos=pos, size=size)
        ET.SubElement(self.sensor, "touch", name=f"{body_name}_contact", site=f"{body_name}_touch")

    def xml(self) -> str:
        ET.indent(self.root, space="  ")
        return ET.tostring(self.root, encoding="unicode") + "\n"


def parallel_jaw(b: Builder, parent: ET.Element, prefix: str, *, stroke: float, finger_length: float, effort: float, limb: str) -> None:
    """A two-finger slide jaw hanging from ``parent`` along its -z."""

    palm = b.body(parent, f"{prefix}palm", (0.0, 0.0, -0.03))
    b.box(palm, f"{prefix}palm_geom", (0.045, 0.025, 0.015), PLASTIC, rgba="0.35 0.35 0.4 1")
    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = b.body(palm, f"{prefix}finger_{side}", (sign * (stroke + 0.008), 0.0, -0.015 - finger_length / 2))
        b.box(finger, f"{prefix}finger_{side}_geom", (0.007, 0.015, finger_length / 2), PLASTIC, rgba="0.3 0.3 0.34 1", friction="1.4 0.005 0.0001")
        b.slide(finger, f"{prefix}grip_{side}", (-sign, 0.0, 0.0), 0.0, stroke, effort=effort, velocity=0.2, kp=400.0, rest=0.0, role="grip", limb=limb)
    b.site(palm, f"{prefix}grasp_centre", pos=(0.0, 0.0, -0.015 - finger_length / 2), size=0.006)


def serial_arm(b: Builder, parent: ET.Element, prefix: str, *, mount: tuple[float, float, float], lengths: tuple[float, ...], axes: tuple[tuple[float, float, float], ...], radius: float,
               efforts: tuple[float, ...], rests: tuple[float, ...], limb: str) -> ET.Element:
    """A serial hinge chain of capsules; returns the last link's body."""

    current = parent
    pos = mount
    for index, (length, axis, effort, rest) in enumerate(zip(lengths, axes, efforts, rests)):
        link = b.body(current, f"{prefix}link_{index}", pos)
        b.hinge(link, f"{prefix}joint_{index}", axis, -2.6, 2.6, effort=effort, velocity=3.0, kp=60.0 * effort / 20.0, rest=rest, role="arm", limb=limb)
        b.capsule(link, f"{prefix}link_{index}_geom", radius, length, PLASTIC, pos=(0.0, 0.0, -length / 2), rgba="0.75 0.72 0.65 1")
        current = link
        pos = (0.0, 0.0, -length)
        radius = max(0.014, radius * 0.85)
    return current


# --------------------------------------------------------------------------------------------------
# the dog with an arm
# --------------------------------------------------------------------------------------------------


def mobile_dog_arm() -> tuple[Builder, dict]:
    b = Builder("mobile_dog_arm", "A quadruped with three-hinge legs (hip roll, hip pitch, knee) on spherical feet, carrying a four-hinge arm and a parallel jaw on the front of its torso.")
    torso = b.body(None, "torso", (0.0, 0.0, 0.40))
    ET.SubElement(torso, "freejoint", name="root")
    b.box(torso, "torso_geom", (0.26, 0.12, 0.06), STEEL * 0.35, rgba="0.45 0.5 0.6 1")
    b.imu("torso")
    upper, lower = 0.20, 0.21
    stance = {}
    for name, (sx, sy) in (("fl", (1, 1)), ("fr", (1, -1)), ("hl", (-1, 1)), ("hr", (-1, -1))):
        hip = b.body(torso, f"{name}_hip", (sx * 0.20, sy * 0.13, -0.02))
        b.hinge(hip, f"{name}_hip_roll", (1.0, 0.0, 0.0), -0.6, 0.6, effort=30.0, velocity=8.0, kp=120.0, rest=0.0, role="leg", limb=f"leg_{name}")
        b.sphere(hip, f"{name}_hip_geom", 0.035, PLASTIC, rgba="0.35 0.38 0.45 1")
        thigh = b.body(hip, f"{name}_thigh", (0.0, sy * 0.03, 0.0))
        b.hinge(thigh, f"{name}_hip_pitch", (0.0, 1.0, 0.0), -1.6, 1.6, effort=40.0, velocity=8.0, kp=160.0, rest=0.55, role="leg", limb=f"leg_{name}")
        b.capsule(thigh, f"{name}_thigh_geom", 0.028, upper, PLASTIC, pos=(0.0, 0.0, -upper / 2), rgba="0.5 0.52 0.58 1")
        shank = b.body(thigh, f"{name}_shank", (0.0, 0.0, -upper))
        b.hinge(shank, f"{name}_knee", (0.0, 1.0, 0.0), -2.4, -0.15, effort=40.0, velocity=8.0, kp=160.0, rest=-1.15, role="leg", limb=f"leg_{name}")
        b.capsule(shank, f"{name}_shank_geom", 0.022, lower, PLASTIC, pos=(0.0, 0.0, -lower / 2), rgba="0.5 0.52 0.58 1")
        foot = b.body(shank, f"{name}_foot", (0.0, 0.0, -lower))
        b.sphere(foot, f"{name}_foot_geom", 0.026, RUBBER, friction="1.3 0.005 0.0001")
        b.touch(foot, f"{name}_foot", size=0.03)
        b.support.append(f"{name}_foot")
        stance[f"{name}_hip_roll"] = 0.0
        stance[f"{name}_hip_pitch"] = 0.55
        stance[f"{name}_knee"] = -1.15
    arm_base = b.body(torso, "arm_base", (0.20, 0.0, 0.06))
    b.hinge(arm_base, "arm_yaw", (0.0, 0.0, 1.0), -2.6, 2.6, effort=25.0, velocity=3.0, kp=75.0, rest=0.0, role="arm", limb="arm")
    b.cylinder(arm_base, "arm_base_geom", 0.045, 0.02, PLASTIC, (0.0, 0.0, 0.0), rgba="0.6 0.55 0.5 1")
    # the arm hangs forward and down: shoulder pitch, elbow pitch, wrist pitch; capsules point along -z of their links, so a rest of
    # -1.2 at the shoulder swings the upper arm forward and up out of the way while walking
    wrist = serial_arm(b, arm_base, "arm_", mount=(0.0, 0.0, 0.04), lengths=(0.22, 0.20, 0.08), axes=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), radius=0.03,
                       efforts=(30.0, 25.0, 15.0), rests=(-1.9, 1.6, 0.9), limb="arm")
    parallel_jaw(b, wrist, "arm_", stroke=0.035, finger_length=0.06, effort=40.0, limb="arm")
    stance.update({"arm_yaw": 0.0, "arm_joint_0": -1.9, "arm_joint_1": 1.6, "arm_joint_2": 0.9, "arm_grip_left": 0.0, "arm_grip_right": 0.0})
    declaration = {
        "base_kind": "legged", "base_body": "torso", "root_joint": "root", "support_members": list(b.support),
        "limbs": {**{f"leg_{n}": [f"{n}_hip_roll", f"{n}_hip_pitch", f"{n}_knee"] for n in ("fl", "fr", "hl", "hr")}, "arm": ["arm_yaw", "arm_joint_0", "arm_joint_1", "arm_joint_2", "arm_grip_left", "arm_grip_right"]},
        "manipulators": [{"limb": "arm", "grasp_site": "arm_grasp_centre", "grip_joints": ["arm_grip_left", "arm_grip_right"], "fingers": ["arm_finger_left", "arm_finger_right"]}],
        "stances": {"standing": {"joints": stance, "statically_stable": True, "note": "four feet on the ground, knees bent, the arm folded up over the torso"},
                    "parked": {"joints": {**stance, **{f"{n}_hip_pitch": 1.4 for n in ("fl", "fr", "hl", "hr")}, **{f"{n}_knee": -2.35 for n in ("fl", "fr", "hl", "hr")}}, "statically_stable": True, "note": "crouched low on folded legs, still on its four feet"}},
        "working_stance": "standing", "expected_standing_height_m": 0.38, "traction": {"floor_friction": 1.3, "members": "rubber spherical feet"},
        "sensors": ["joint encoders on every hinge and slide", "torso gyro, accelerometer and orientation", "touch on every foot"],
    }
    return b, declaration


# --------------------------------------------------------------------------------------------------
# the wheeled biped
# --------------------------------------------------------------------------------------------------


def mobile_wheeled_biped() -> tuple[Builder, dict]:
    b = Builder("mobile_wheeled_biped", "A two-legged wheeled body: hip and knee hinges on each leg, a driven wheel at each foot, knee pads to park on, and a three-hinge arm with a parallel jaw on the torso.")
    torso = b.body(None, "torso", (0.0, 0.0, 0.55))
    ET.SubElement(torso, "freejoint", name="root")
    b.box(torso, "torso_geom", (0.10, 0.14, 0.14), STEEL * 0.3, rgba="0.55 0.45 0.4 1")
    b.imu("torso")
    upper, lower, wheel_r = 0.20, 0.20, 0.08
    stance = {}
    parked = {}
    for name, sy in (("left", 1.0), ("right", -1.0)):
        hip = b.body(torso, f"{name}_hip", (0.0, sy * 0.19, -0.10))
        b.hinge(hip, f"{name}_hip_pitch", (0.0, 1.0, 0.0), -2.0, 2.0, effort=45.0, velocity=6.0, kp=180.0, rest=0.7, role="leg", limb=f"leg_{name}")
        b.cylinder(hip, f"{name}_hip_geom", 0.035, 0.02, PLASTIC, (math.pi / 2, 0.0, 0.0), rgba="0.4 0.35 0.32 1")
        thigh_body = b.body(hip, f"{name}_thigh", (0.0, sy * 0.03, 0.0))
        b.capsule(thigh_body, f"{name}_thigh_geom", 0.028, upper, PLASTIC, pos=(0.0, 0.0, -upper / 2), rgba="0.6 0.5 0.45 1")
        shank = b.body(thigh_body, f"{name}_shank", (0.0, 0.0, -upper))
        # the knee flexes backward (positive), so a thigh swung forward and a shank folded back put the wheel under the hip
        b.hinge(shank, f"{name}_knee", (0.0, 1.0, 0.0), 0.0, 2.6, effort=45.0, velocity=6.0, kp=180.0, rest=1.4, role="leg", limb=f"leg_{name}")
        b.capsule(shank, f"{name}_shank_geom", 0.024, lower, PLASTIC, pos=(0.0, 0.0, -lower / 2), rgba="0.6 0.5 0.45 1")
        b.sphere(shank, f"{name}_knee_pad", 0.035, RUBBER, pos=(0.0, 0.0, 0.0), rgba="0.2 0.2 0.22 1", friction="1.2 0.005 0.0001")
        wheel = b.body(shank, f"{name}_wheel", (0.0, sy * 0.04, -lower))
        b.wheel(wheel, f"{name}_wheel_spin", (0.0, 1.0, 0.0), effort=12.0, velocity=25.0, kv=1.5, limb=f"leg_{name}")
        b.cylinder(wheel, f"{name}_wheel_geom", wheel_r, 0.02, RUBBER, (math.pi / 2, 0.0, 0.0), rgba="0.12 0.12 0.14 1", friction="1.1 0.005 0.0001")
        b.touch(wheel, f"{name}_wheel", size=wheel_r + 0.005)
        b.support.append(f"{name}_wheel")
        stance[f"{name}_hip_pitch"] = -0.7
        stance[f"{name}_knee"] = 1.4
        parked[f"{name}_hip_pitch"] = -1.5
        parked[f"{name}_knee"] = 0.1
    # a tail skid under the back of the torso: with the legs stretched forward the body sits level on it and the two wheels
    b.sphere(torso, "tail_skid_geom", 0.03, RUBBER, pos=(-0.09, 0.0, -0.16), rgba="0.2 0.2 0.22 1", friction="1.2 0.005 0.0001")
    b.touch(torso, "tail_skid", pos=(-0.09, 0.0, -0.16), size=0.035)
    b.support.append("torso")
    arm_base = b.body(torso, "arm_base", (0.10, 0.0, 0.14))
    b.hinge(arm_base, "arm_yaw", (0.0, 0.0, 1.0), -2.6, 2.6, effort=20.0, velocity=3.0, kp=60.0, rest=0.0, role="arm", limb="arm")
    b.cylinder(arm_base, "arm_base_geom", 0.04, 0.02, PLASTIC, (0.0, 0.0, 0.0), rgba="0.6 0.55 0.5 1")
    wrist = serial_arm(b, arm_base, "arm_", mount=(0.0, 0.0, 0.04), lengths=(0.22, 0.20), axes=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), radius=0.028, efforts=(25.0, 18.0), rests=(-1.9, 1.7), limb="arm")
    parallel_jaw(b, wrist, "arm_", stroke=0.035, finger_length=0.06, effort=40.0, limb="arm")
    arm = {"arm_yaw": 0.0, "arm_joint_0": -1.9, "arm_joint_1": 1.7, "arm_grip_left": 0.0, "arm_grip_right": 0.0}
    stance.update(arm)
    parked.update(arm)
    declaration = {
        "base_kind": "wheeled", "base_body": "torso", "root_joint": "root", "support_members": list(b.support),
        "limbs": {"leg_left": ["left_hip_pitch", "left_knee", "left_wheel_spin"], "leg_right": ["right_hip_pitch", "right_knee", "right_wheel_spin"], "arm": ["arm_yaw", "arm_joint_0", "arm_joint_1", "arm_grip_left", "arm_grip_right"]},
        "manipulators": [{"limb": "arm", "grasp_site": "arm_grasp_centre", "grip_joints": ["arm_grip_left", "arm_grip_right"], "fingers": ["arm_finger_left", "arm_finger_right"]}],
        "stances": {"parked": {"joints": parked, "statically_stable": True, "note": "legs stretched forward, the body sitting level on its two wheels and the tail skid under its back; the stance it holds without a balance controller"},
                    "standing": {"joints": stance, "statically_stable": False, "note": "up on two wheels with the knees flexed; needs an active balance controller (G17), which is why released standing is an unsupported manoeuvre here"}},
        "working_stance": "standing", "expected_standing_height_m": 0.47, "traction": {"floor_friction": 1.1, "members": "rubber wheels; the tail skid when parked"},
        "sensors": ["joint encoders on every hinge and slide, wheel angle", "torso gyro, accelerometer and orientation", "touch on both wheels and the tail skid"],
    }
    return b, declaration


# --------------------------------------------------------------------------------------------------
# the segmented octopus
# --------------------------------------------------------------------------------------------------


def mobile_octopus() -> tuple[Builder, dict]:
    b = Builder("mobile_octopus", "A rigid segmented octopus-like body: a mantle resting on the ground and six tentacles of four capsule segments each, every segment on a pitch and a yaw hinge; the two front tentacles end in a pincer.")
    mantle = b.body(None, "mantle", (0.0, 0.0, 0.16))
    ET.SubElement(mantle, "freejoint", name="root")
    ET.SubElement(mantle, "geom", name="mantle_geom", type="ellipsoid", size=fmt(0.16, 0.14, 0.11), density=f"{PLASTIC * 0.6:.1f}", rgba="0.5 0.3 0.45 1", friction="0.8 0.005 0.0001")
    b.imu("mantle")
    b.touch(mantle, "mantle", pos=(0.0, 0.0, -0.10), size=0.06)
    b.support.append("mantle")
    lengths = (0.14, 0.13, 0.12, 0.11)
    radii = (0.03, 0.026, 0.022, 0.018)
    stance = {}
    limbs = {}
    manipulators = []
    for index in range(6):
        angle = math.pi / 6 + index * math.pi / 3
        prefix = f"t{index}_"
        root_pos = (0.15 * math.cos(angle), 0.13 * math.sin(angle), -0.06)
        parent = b.body(mantle, f"{prefix}root", root_pos, euler=(0.0, 0.0, angle))
        b.sphere(parent, f"{prefix}root_geom", 0.03, PLASTIC, rgba="0.45 0.28 0.4 1")
        b.support.append(f"{prefix}root")
        limbs[f"tentacle_{index}"] = []
        current = parent
        # segments extend along the local +x of the tentacle root; each carries a yaw (about z) and a pitch (about y) hinge at its base
        for seg, (length, radius) in enumerate(zip(lengths, radii)):
            segment = b.body(current, f"{prefix}seg_{seg}", (0.0, 0.0, 0.0) if seg == 0 else (lengths[seg - 1], 0.0, 0.0))
            b.hinge(segment, f"{prefix}yaw_{seg}", (0.0, 0.0, 1.0), -1.2, 1.2, effort=12.0, velocity=4.0, kp=30.0, rest=0.0, role="tentacle", limb=f"tentacle_{index}")
            b.hinge(segment, f"{prefix}pitch_{seg}", (0.0, 1.0, 0.0), -1.4, 1.4, effort=12.0, velocity=4.0, kp=30.0, rest=(0.35 if seg == 0 else -0.12), role="tentacle", limb=f"tentacle_{index}")
            b.capsule(segment, f"{prefix}seg_{seg}_geom", radius, length, PLASTIC, pos=(length / 2, 0.0, 0.0), axis="x", rgba="0.6 0.35 0.5 1", friction="1.0 0.005 0.0001")
            b.touch(segment, f"{prefix}seg_{seg}", pos=(length / 2, 0.0, 0.0), size=radius + 0.004)
            stance[f"{prefix}yaw_{seg}"] = 0.0
            stance[f"{prefix}pitch_{seg}"] = 0.35 if seg == 0 else -0.12
            limbs[f"tentacle_{index}"] += [f"{prefix}yaw_{seg}", f"{prefix}pitch_{seg}"]
            b.support.append(f"{prefix}seg_{seg}")
            current = segment
        if index in (0, 5):
            # a pincer at the tip: two hinge fingers closing across the tentacle's axis
            tip = b.body(current, f"{prefix}pincer", (lengths[-1], 0.0, 0.0))
            b.sphere(tip, f"{prefix}pincer_geom", 0.02, PLASTIC, rgba="0.4 0.25 0.35 1")
            for side, sign in (("left", 1.0), ("right", -1.0)):
                finger = b.body(tip, f"{prefix}finger_{side}", (0.01, sign * 0.022, 0.0))
                b.hinge(finger, f"{prefix}pinch_{side}", (0.0, 0.0, -sign), -0.05, 0.9, effort=6.0, velocity=4.0, kp=20.0, rest=0.0, role="grip", limb=f"tentacle_{index}")
                b.capsule(finger, f"{prefix}finger_{side}_geom", 0.008, 0.05, PLASTIC, pos=(0.025, 0.0, 0.0), axis="x", rgba="0.3 0.2 0.28 1", friction="1.4 0.005 0.0001")
                stance[f"{prefix}pinch_{side}"] = 0.0
                limbs[f"tentacle_{index}"].append(f"{prefix}pinch_{side}")
            b.site(tip, f"{prefix}grasp_centre", pos=(0.045, 0.0, 0.0), size=0.006)
            manipulators.append({"limb": f"tentacle_{index}", "grasp_site": f"{prefix}grasp_centre", "grip_joints": [f"{prefix}pinch_left", f"{prefix}pinch_right"], "fingers": [f"{prefix}finger_left", f"{prefix}finger_right"]})
    declaration = {
        "base_kind": "crawling", "base_body": "mantle", "root_joint": "root", "support_members": list(b.support),
        "limbs": limbs, "manipulators": manipulators,
        "stances": {"resting": {"joints": stance, "statically_stable": True, "note": "the mantle on the ground and the six tentacles splayed flat around it"},
                    "reaching": {"joints": {**stance, "t0_pitch_0": -0.9, "t0_pitch_1": -0.3, "t5_pitch_0": -0.9, "t5_pitch_1": -0.3}, "statically_stable": True, "note": "the two pincer tentacles raised while the other four and the mantle bear the weight"}},
        "working_stance": "resting", "expected_standing_height_m": 0.12, "traction": {"floor_friction": 1.0, "members": "tentacle segments and the mantle underside"},
        "sensors": ["joint encoders on every hinge", "mantle gyro, accelerometer and orientation", "touch on the mantle underside and every tentacle segment"],
    }
    return b, declaration


BUILDERS = (mobile_dog_arm, mobile_wheeled_biped, mobile_octopus)


def write_body(builder: Builder, declaration: dict, root: Path) -> dict:
    folder = root / builder.robot_id
    folder.mkdir(parents=True, exist_ok=True)
    xml = builder.xml()
    (folder / "robot.xml").write_text(xml, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    (folder / "robot.xml.sha256").write_text(digest + "\n", encoding="utf-8", newline="\n")
    mobility = {"schema": "rigby.mobility-declaration/1", "robot_id": builder.robot_id, "description": builder.description, "model": "robot.xml", "model_sha256": digest,
                "joints": builder.joints, "actuators": builder.actuators, **declaration}
    (folder / "mobility.json").write_text(json.dumps(mobility, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    provenance = {"schema": "rigby.asset-provenance/1", "robot_id": builder.robot_id, "origin": "procedural: generated by any-robot/scripts/build_mobile_zoo.py in this repository from primitive geometry; no external design, mesh or measurement was used",
                  "author": "the rigby repository", "licence": "the repository's own licence; redistributable with the repository", "redistribution": "permitted with the repository under its licence",
                  "preferred_external_design": None, "substitute_for": {"mobile_dog_arm": "a quadruped with a manipulator", "mobile_wheeled_biped": "a wheel-legged balancing robot", "mobile_octopus": "a soft multi-arm crawler, here rigid and segmented"}[builder.robot_id],
                  "holdout": False, "role": "development and showcase body"}
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"robot_id": builder.robot_id, "model_sha256": digest, "joints": len(builder.joints), "actuators": len(builder.actuators), "support_members": len(builder.support)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=MOBILE_ROOT)
    args = parser.parse_args()
    summary = [write_body(*builder(), args.root) for builder in BUILDERS]
    (args.root / "index.json").write_text(json.dumps({"schema": "rigby.mobile-zoo/1", "bodies": summary}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
