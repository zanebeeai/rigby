"""The gripper as a real mechanism: joints, actuators, and commanded torque.

Everything before this drove the machine by welding its parts to poses and
letting the solver argue with the object. That cannot transfer to hardware --
there is no actuator that holds a position against contact by out-stiffening it
-- and it produced grips made of overlap: 7 to 18 mm of finger inside a 60 mm
block, with the holding force being the penetration rather than causing it.

Here the arm has hinges and the fingers have slides, and nothing is welded. The
controller commands torque:

    tau = M(q) * (qacc_desired + 2*zeta*omega*e_dot + omega**2 * e) + C(q, qdot) + g(q)

M, C and g come from the model at the configuration the robot is *currently* in,
so the closed loop obeys the same second-order error response on every joint
whatever its inertia -- and the only parameters left, zeta and omega, are
dimensionless and describe how the error should decay rather than anything about
the body. That is the same move the manifest makes: state what does not depend
on the body and let a measurement supply the rest.

Grip force stops being a side effect of geometry. The fingers carry a
feed-forward squeeze on top of their tracking, so "close to 6 N" is a command
rather than a distance chosen in the hope that 6 N falls out.

Coordinates are MuJoCo's throughout: Z-up, and no app-frame conversion anywhere
inside. Converting mid-pipeline is how the block's orientation came out mirrored
and needed a hand-written axis swap to look right. The viewer converts, once, at
the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .gripper import spec

#: Damping ratio and natural frequency of the closed-loop error. Dimensionless
#: on purpose: they say how the error should decay, not what the arm weighs.
_ZETA = 1.0
_OMEGA = 12.0
#: The fingers track more sharply than the arm -- they are light and their job
#: is contact, where lag reads as a soft grip.
_FINGER_OMEGA = 26.0

#: Names in the order their joints appear, which is also the order of qpos.
JOINTS = ("yaw", "lift", "elbow", "wrist", "finger_left", "finger_right")


def _model_xml(block_half, block_at, table_top: float) -> str:
    """The arm as a jointed chain, the fingers as slides, nothing welded."""
    document = spec()
    kinematics = document["kinematics"]
    lengths = kinematics["segments_m"]
    radii = kinematics.get("link_radius_m", [0.022, 0.019, 0.016])
    finger = kinematics["finger"]
    base = kinematics["base"]["at"]
    limits = {j["name"]: j for j in kinematics["joints"]}
    bin_doc = document["scene"]["bin"]

    def deg(name: str) -> str:
        """The declared range, in the units MuJoCo is actually reading.

        These were being converted to radians and written into a compiler that
        defaults to angle="degree", so every range was read as its own numeric
        value in degrees: a declared 150 became 2.6, and every joint on the arm
        was clamped to about two degrees of travel. The controller was commanding
        correctly the whole time into joints that could not move, which looked
        exactly like a tracking failure and was not one.
        """
        low, high = limits[name]["range_deg"]
        return f"{float(low):.4f} {float(high):.4f}"

    reach = float(finger["length_m"])
    thick = float(finger["thickness_m"])
    pad = float(finger["pad_width_m"]) / 2.0
    travel = limits["finger_left"]["range_m"]

    # MuJoCo is Z-up; the manifest is authored Y-up, so the base is placed once
    # here and nothing downstream converts again.
    bx, by, bz = float(base[0]), float(base[2]), float(base[1])
    cx, cy, cz = (float(bin_doc["centre"][0]), float(bin_doc["centre"][2]),
                  float(bin_doc["centre"][1]))
    inner = bin_doc["inner_half_m"]
    wall = float(bin_doc["wall_m"])
    walls = "".join(
        f'<geom name="bin_wall{i}" contype="1" conaffinity="6" type="box" '
        f'pos="{cx + dx * (inner[0] + wall)} {cy + dz * (inner[2] + wall)} {cz}" '
        f'size="{wall if dx else inner[0] + wall * 2} '
        f'{inner[2] + wall * 2 if dx else wall} {inner[1]}" '
        f'rgba="0.5 0.55 0.63 1" friction="0.9 0.02 0.001"/>'
        for i, (dx, dz) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))))

    return f"""
<mujoco>
  <!-- Angles in degrees, which is MuJoCo's default and the units the
       manifest already declares. Stated rather than relied upon. -->
  <compiler angle="degree"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
  <!-- WHAT MAY TOUCH WHAT, stated once. Bit 1 is scenery, bit 2 the object,
       bit 4 the gripping surfaces. The arm links carry neither, so they collide
       with nothing: they are linkage, and letting them collide put the shoulder
       sphere inside its own pedestal reporting 4.4e18 N, which pinned the arm
       so hard the controller settled into a stable equilibrium 1.4 rad from its
       target and looked like a tracking failure. The two fingers do not collide
       with each other either -- both are surfaces, and a gripper closing on
       nothing should close. -->
  <default>
    <geom friction="0.9 0.02 0.001" solref="0.004 1" solimp="0.98 0.999 0.0005"
          contype="0" conaffinity="0"/>
    <joint damping="0.4" armature="0.01"/>
  </default>
  <worldbody>
    <light name="key" pos="0.6 -1.0 2.6" dir="-0.2 0.4 -1" diffuse="1 1 1"/>
    <geom name="table" contype="1" conaffinity="6" type="plane" pos="0 0 {table_top}" size="2 2 0.1"
          rgba="0.4 0.42 0.48 1"/>
    <geom name="bin_floor" contype="1" conaffinity="6" type="box"
          pos="{cx} {cy} {cz - inner[1] - wall}"
          size="{inner[0] + wall} {inner[2] + wall} {wall}"
          rgba="0.45 0.5 0.58 1"/>
    {walls}
    <geom name="bin_riser" contype="1" conaffinity="6" type="cylinder"
          pos="{cx} {cy} {(table_top + cz - inner[1]) / 2}"
          size="0.045 {max((cz - inner[1] - table_top) / 2, 0.01)}"
          rgba="0.4 0.44 0.52 1"/>
    <geom name="pedestal" contype="1" conaffinity="6" type="cylinder" pos="{bx} {by} {(table_top + bz) / 2}"
          size="0.05 {max((bz - table_top) / 2, 0.01)}" rgba="0.4 0.44 0.52 1"/>

    <body name="block" pos="{block_at[0]} {block_at[1]} {block_at[2]}">
      <freejoint name="block_free"/>
      <geom name="block_geom" contype="2" conaffinity="5" type="box"
            size="{block_half[0]} {block_half[1]} {block_half[2]}"
            mass="0.25" rgba="0.85 0.55 0.3 1"/>
    </body>

    <body name="base" pos="{bx} {by} {bz}">
      <joint name="yaw" type="hinge" axis="0 0 1" range="{deg('yaw')}"/>
      <geom name="shoulder" type="sphere" size="{radii[0] * 1.3}" mass="0.6"
            rgba="0.42 0.47 0.56 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="lift" type="hinge" axis="1 0 0" range="{deg('lift')}"/>
        <geom name="seg1" type="capsule" fromto="0 0 0 0 {lengths[0]} 0"
              size="{radii[0]}" mass="1.1" rgba="0.42 0.47 0.56 1"/>
        <body name="link2" pos="0 {lengths[0]} 0">
          <joint name="elbow" type="hinge" axis="1 0 0" range="{deg('elbow')}"/>
          <geom name="seg2" type="capsule" fromto="0 0 0 0 {lengths[1]} 0"
                size="{radii[1]}" mass="0.8" rgba="0.42 0.47 0.56 1"/>
          <body name="link3" pos="0 {lengths[1]} 0">
            <joint name="wrist" type="hinge" axis="1 0 0" range="{deg('wrist')}"/>
            <geom name="seg3" type="capsule" fromto="0 0 0 0 {lengths[2]} 0"
                  size="{radii[2]}" mass="0.4" rgba="0.42 0.47 0.56 1"/>
            <body name="plate" pos="0 {lengths[2]} 0">
              <geom name="plate_geom" contype="4" conaffinity="3" type="box" size="0.05 0.012 0.04"
                    mass="0.35" rgba="0.25 0.5 0.7 1"/>
              <body name="finger_left" pos="0 0 0">
                <joint name="finger_left" type="slide" axis="1 0 0"
                       range="{travel[0]} {travel[1]}"/>
                <geom name="left_geom" contype="4" conaffinity="3" type="box"
                      pos="0 {reach / 2 + 0.012} 0"
                      size="{thick} {reach / 2} {pad}" mass="0.06"
                      rgba="0.3 0.7 0.9 1"/>
              </body>
              <body name="finger_right" pos="0 0 0">
                <joint name="finger_right" type="slide" axis="-1 0 0"
                       range="{travel[0]} {travel[1]}"/>
                <geom name="right_geom" contype="4" conaffinity="3" type="box"
                      pos="0 {reach / 2 + 0.012} 0"
                      size="{thick} {reach / 2} {pad}" mass="0.06"
                      rgba="0.3 0.7 0.9 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="yaw" name="m_yaw" gear="1" ctrlrange="-80 80"/>
    <motor joint="lift" name="m_lift" gear="1" ctrlrange="-120 120"/>
    <motor joint="elbow" name="m_elbow" gear="1" ctrlrange="-90 90"/>
    <motor joint="wrist" name="m_wrist" gear="1" ctrlrange="-40 40"/>
    <motor joint="finger_left" name="m_left" gear="1" ctrlrange="-60 60"/>
    <motor joint="finger_right" name="m_right" gear="1" ctrlrange="-60 60"/>
  </actuator>
</mujoco>
"""


@dataclass
class Body:
    """A torque-driven gripper, and everything read from it."""

    model: Any
    data: Any
    block_half: np.ndarray

    def address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint])

    def q(self) -> np.ndarray:
        return np.asarray([self.data.qpos[self.address(n)] for n in JOINTS])

    def qd(self) -> np.ndarray:
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in JOINTS]
        return np.asarray([self.data.qvel[int(self.model.jnt_dofadr[j])]
                           for j in joints])

    def body_at(self, name: str) -> np.ndarray:
        return np.asarray(self.data.xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)])

    def geom_at(self, name: str) -> np.ndarray:
        return np.asarray(self.data.geom_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)])

    def pads(self) -> tuple[np.ndarray, np.ndarray]:
        return self.geom_at("left_geom"), self.geom_at("right_geom")

    def grasp_centre(self) -> np.ndarray:
        left, right = self.pads()
        return (left + right) / 2.0

    def opening(self) -> float:
        left, right = self.pads()
        thickness = float(spec()["kinematics"]["finger"]["thickness_m"])
        return float(max(0.0, np.linalg.norm(left - right) - 2.0 * thickness))

    def approach(self) -> np.ndarray:
        """The direction the gripper points, from the plate out past the pads."""
        plate = self.body_at("plate")
        span = self.grasp_centre() - plate
        size = float(np.linalg.norm(span))
        return span / size if size > 1e-9 else np.asarray([0.0, 1.0, 0.0])

    def block(self) -> np.ndarray:
        return self.body_at("block")

    def block_quat(self) -> np.ndarray:
        return np.asarray(self.data.xquat[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")])

    def forces(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            }
            if "block_geom" not in names:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            for geom, digit in (("left_geom", "finger_left"),
                                ("right_geom", "finger_right"),
                                ("plate_geom", "plate")):
                if geom in names:
                    out[digit] = max(out.get(digit, 0.0), float(abs(force[0])))
        return out

    def penetration_mm(self) -> float:
        """Deepest the gripper is inside the block, in millimetres.

        The number the whole exercise turns on. Under position control this ran
        to 18 mm because the holding force WAS the overlap; under torque control
        it should stay at the contact solver's own tolerance.
        """
        worst = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            }
            if "block_geom" not in names:
                continue
            if not names & {"left_geom", "right_geom", "plate_geom"}:
                continue
            worst = max(worst, -float(contact.dist))
        return worst * 1000.0


def computed_torque(body: Body, target: np.ndarray,
                    squeeze_n: float = 0.0) -> np.ndarray:
    """Torque that makes the error decay the same way on every joint.

    The mass matrix is asked for at the CURRENT configuration rather than
    measured once, because inertia is a property of a pose and not of a joint: a
    base yaw carrying a folded arm has almost none about its own axis and a
    great deal with the arm extended. Gains fixed at rest are hopeless
    everywhere else, and gains fixed at the worst case are unstable at rest.
    There is no single right value, so do not look for one.
    """
    model, data = body.model, body.data
    error = target - body.q()
    rate = -body.qd()
    omega = np.asarray([_OMEGA] * 4 + [_FINGER_OMEGA] * 2)
    wanted = 2.0 * _ZETA * omega * rate + omega ** 2 * error

    full = np.zeros(model.nv)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    dofs = [int(model.jnt_dofadr[j]) for j in joints]
    for dof, value in zip(dofs, wanted):
        full[dof] = value

    inertial = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, inertial, full)
    tau = inertial + data.qfrc_bias

    out = np.asarray([tau[dof] for dof in dofs])
    if squeeze_n:
        # A commanded grip, not a hoped-for one. Both fingers are pushed inward
        # along their own axis, so the force is what was asked for rather than
        # whatever a chosen position happens to produce against the object.
        out[4] -= squeeze_n
        out[5] -= squeeze_n
    return out


def make(block_half: np.ndarray, block_at: np.ndarray,
         table_top: float = 0.72) -> Body:
    xml = _model_xml(block_half, block_at, table_top)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body = Body(model=model, data=data, block_half=np.asarray(block_half))
    mujoco.mj_forward(model, data)
    return body
