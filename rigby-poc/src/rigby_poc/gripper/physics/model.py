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

from ..body.manifest import spec

#: Damping ratio and natural frequency of the closed-loop error. Dimensionless
#: on purpose: they say how the error should decay, not what the arm weighs.
_ZETA = 1.0
_OMEGA = 12.0
#: The fingers track more sharply than the arm -- they are light and their job
#: is contact, where lag reads as a soft grip.
_FINGER_OMEGA = 26.0

#: Names in the order their joints appear, which is also the order of qpos.
JOINTS = ("base", "segment_1", "segment_2", "segment_3", "finger_left", "finger_right")


#: The original single key light. The pick-and-place was tuned under it and its
#: camera reads the scene through it.
_BENCH_LIGHTING = (
    '<light name="key" pos="0.6 -1.0 2.6" dir="-0.2 0.4 -1" diffuse="1 1 1"/>')

#: A lit room, for the scene that has a camera standing in the corner of one.
_ROOM_LIGHTING = """
    <light name="key" pos="0.6 -1.0 2.6" dir="-0.2 0.4 -1"
           diffuse="0.75 0.75 0.75" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="-1.2 0.4 2.2" dir="0.5 -0.1 -1"
           diffuse="0.45 0.46 0.5" specular="0 0 0"/>
    <light name="back" pos="0.9 1.6 2.0" dir="-0.3 -0.7 -1"
           diffuse="0.3 0.31 0.35" specular="0 0 0"/>
    <geom name="wall_back" contype="0" conaffinity="0" type="box"
          pos="0 1.15 1.25" size="2.0 0.02 0.9" rgba="0.30 0.32 0.38 1"/>
    <geom name="wall_side" contype="0" conaffinity="0" type="box"
          pos="0.95 0 1.25" size="0.02 1.15 0.9" rgba="0.26 0.28 0.34 1"/>"""


def _room_camera(document) -> str:
    """The fixed camera in the corner, aimed by hand at the middle of the bench.

    MuJoCo can aim a camera at a body, but the interesting point here is a place
    rather than a thing, and a camera that tracks a moving body reframes itself
    every time the arm picks something up.
    """
    doc = document["scene"].get("cameras", {}).get("room")
    if doc is None:
        return ""
    eye = np.asarray([doc["at"][0], doc["at"][2], doc["at"][1]], dtype=float)
    at = np.asarray([doc["looks_at"][0], doc["looks_at"][2], doc["looks_at"][1]],
                    dtype=float)
    forward = at - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(-forward, right)
    return (f'<camera name="room" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}" '
            f'fovy="{float(doc.get("fovy_deg", 58.0)):.1f}" '
            f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} '
            f'{up[0]:.4f} {up[1]:.4f} {up[2]:.4f}"/>')


def _bin_furniture(document, table_top: float) -> str:
    """The elevated bin, for the pick-and-place task."""
    doc = document["scene"]["bin"]
    cx, cy, cz = (float(doc["centre"][0]), float(doc["centre"][2]),
                  float(doc["centre"][1]))
    inner = doc["inner_half_m"]
    wall = float(doc["wall_m"])
    walls = "".join(
        f'<geom name="bin_wall{i}" contype="1" conaffinity="6" type="box" '
        f'pos="{cx + dx * (inner[0] + wall)} {cy + dz * (inner[2] + wall)} {cz}" '
        f'size="{wall if dx else inner[0] + wall * 2} '
        f'{inner[2] + wall * 2 if dx else wall} {inner[1]}" '
        f'rgba="0.5 0.55 0.63 1" friction="0.9 0.02 0.001"/>'
        for i, (dx, dz) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))))
    return f"""
    <geom name="bin_floor" contype="1" conaffinity="6" type="box"
          pos="{cx} {cy} {cz - inner[1] - wall}"
          size="{inner[0] + wall} {inner[2] + wall} {wall}"
          rgba="0.45 0.5 0.58 1"/>
    {walls}
    <geom name="bin_riser" contype="1" conaffinity="6" type="cylinder"
          pos="{cx} {cy} {(table_top + cz - inner[1]) / 2}"
          size="0.045 {max((cz - inner[1] - table_top) / 2, 0.01)}"
          rgba="0.4 0.44 0.52 1"/>"""


def _fridge_furniture(document, table_top: float) -> str:
    """The cabinet: four fixed walls, and a door on a hinge with a handle."""
    doc = document["scene"]["fridge"]
    cx, cy = float(doc["centre"][0]), float(doc["centre"][2])
    ix, iz, iy = (float(doc["inner_half_m"][0]), float(doc["inner_half_m"][1]),
                  float(doc["inner_half_m"][2]))
    wall = float(doc["wall_m"])
    mid = table_top + iz
    hinge_x, hinge_y = (float(doc["door"]["hinge_at"][0]),
                        float(doc["door"]["hinge_at"][1]))
    width = float(doc["door"]["width_m"])
    low, high = doc["door"]["swing_deg"]
    handle = doc["handle"]
    hx = float(handle["at_from_hinge_m"])
    stand = float(handle["stands_off_m"])
    skin = 'contype="1" conaffinity="6"'
    stems = "".join(
        f'<geom name="door_stem{i}" contype="1" conaffinity="6" type="box" '
        f'pos="{hx} {-stand / 2:.4f} {z}" size="0.008 {stand / 2} 0.010" '
        f'mass="0.03" rgba="0.45 0.48 0.55 1"/>'
        for i, z in enumerate(handle.get("brackets_at_z", [0.0])))
    return f"""
    <geom name="fridge_back" {skin} type="box" rgba="0.62 0.65 0.70 1"
          pos="{cx} {cy + iy + wall / 2:.4f} {mid}"
          size="{ix + wall} {wall / 2} {iz}"/>
    <geom name="fridge_left" {skin} type="box" rgba="0.62 0.65 0.70 1"
          pos="{cx + ix + wall / 2:.4f} {cy} {mid}"
          size="{wall / 2} {iy + wall} {iz}"/>
    <geom name="fridge_right" {skin} type="box" rgba="0.62 0.65 0.70 1"
          pos="{cx - ix - wall / 2:.4f} {cy} {mid}"
          size="{wall / 2} {iy + wall} {iz}"/>
    <geom name="fridge_top" {skin} type="box" rgba="0.58 0.61 0.67 1"
          pos="{cx} {cy} {table_top + 2 * iz + wall / 2:.4f}"
          size="{ix + wall} {iy + wall} {wall / 2}"/>

    <body name="door" pos="{hinge_x} {hinge_y} {mid}">
      <joint name="door_hinge" type="hinge" axis="0 0 1"
             range="{low} {high}" damping="{float(doc['door']['damping'])}"/>
      <geom name="door_panel" contype="1" conaffinity="6" type="box"
            pos="{-width / 2:.4f} 0 0" size="{width / 2} {wall / 2} {iz}"
            mass="0.5" rgba="0.70 0.73 0.78 1"/>
      {stems}
      <geom name="handle" contype="1" conaffinity="6" type="cylinder"
            pos="{hx} {-stand:.4f} 0"
            size="{float(handle['radius_m'])} {float(handle['half_length_m'])}"
            mass="0.05" rgba="0.88 0.90 0.94 1" friction="1.4 0.02 0.001"/>
    </body>"""


def _model_xml(block_half, block_at, table_top: float,
               scene: str = "bin") -> str:
    """The arm as a jointed chain, the fingers as slides, nothing welded."""
    document = spec()
    kinematics = document["kinematics"]
    lengths = kinematics["segments_m"]
    radii = kinematics.get("link_radius_m", [0.022, 0.019, 0.016])
    finger = kinematics["finger"]
    base = kinematics["base"]["at"]
    limits = {j["name"]: j for j in kinematics["joints"]}

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
    furniture = (_fridge_furniture(document, table_top) if scene == "fridge"
                 else _bin_furniture(document, table_top))
    # The room -- three lights and two walls -- belongs to the CABINET scene,
    # which has a camera in the corner that needs somewhere to look and a sense
    # of scale. Adding it globally relit the pick-and-place too, and that scene
    # is read by a camera that finds the object by segmenting warm pixels: the
    # blob changed shape, the estimates moved, and placements that had a
    # millimetre of jaw clearance stopped landing. Lighting is scene dress for
    # one task and an input to perception for the other.
    lighting = (_ROOM_LIGHTING if scene == "fridge" else _BENCH_LIGHTING)
    room_camera = _room_camera(document)

    return f"""
<mujoco>
  <!-- Angles in degrees, which is MuJoCo's default and the units the
       manifest already declares. Stated rather than relied upon. -->
  <compiler angle="degree"/>
  <!-- The offscreen buffer defaults to 640x480, which is fine for the wrist
       camera and too small for a room view worth showing anyone. -->
  <visual><global offwidth="1280" offheight="960"/></visual>
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
    <!-- Finger travel limits stiff enough to be limits. A joint range in
         MuJoCo is a constraint like any other, and a constraint can be
         overpowered: an arm swinging into a door it had just opened pried the
         jaws to 14.4 cm, which is 6 cm wider than the mechanism can physically
         go. A limit that a hard enough shove walks through is not modelling a
         stop, it is modelling a strong spring. -->
    <default class="finger">
      <joint damping="0.4" armature="0.01"
             solreflimit="0.002 1" solimplimit="0.99 0.9999 0.0001"/>
    </default>
  </default>
  <worldbody>
    {lighting}
    <geom name="table" contype="1" conaffinity="6" type="plane"
          pos="0 0 {table_top}" size="2 2 0.1" rgba="0.4 0.42 0.48 1"/>
    {furniture}
    {room_camera}
    <geom name="pedestal" contype="1" conaffinity="6" type="cylinder" pos="{bx} {by} {(table_top + bz) / 2}"
          size="0.05 {max((bz - table_top) / 2, 0.01)}" rgba="0.4 0.44 0.52 1"/>

    <body name="block" pos="{block_at[0]} {block_at[1]} {block_at[2]}">
      <freejoint name="block_free"/>
      <geom name="block_geom" contype="2" conaffinity="5" type="box"
            size="{block_half[0]} {block_half[1]} {block_half[2]}"
            mass="0.25" rgba="0.85 0.55 0.3 1"/>
    </body>

    <body name="base" pos="{bx} {by} {bz}">
      <joint name="base" type="hinge" axis="0 0 1" range="{deg('base')}"/>
      <geom name="base_hub" type="sphere" size="{radii[0] * 1.3}" mass="0.6"
            rgba="0.42 0.47 0.56 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="segment_1" type="hinge" axis="1 0 0" range="{deg('segment_1')}"/>
        <geom name="seg1" type="capsule" fromto="0 0 0 0 {lengths[0]} 0"
              size="{radii[0]}" mass="1.1" rgba="0.42 0.47 0.56 1"/>
        <body name="link2" pos="0 {lengths[0]} 0">
          <joint name="segment_2" type="hinge" axis="1 0 0" range="{deg('segment_2')}"/>
          <geom name="seg2" type="capsule" fromto="0 0 0 0 {lengths[1]} 0"
                size="{radii[1]}" mass="0.8" rgba="0.42 0.47 0.56 1"/>
          <body name="link3" pos="0 {lengths[1]} 0">
            <joint name="segment_3" type="hinge" axis="1 0 0" range="{deg('segment_3')}"/>
            <geom name="seg3" type="capsule" fromto="0 0 0 0 {lengths[2]} 0"
                  size="{radii[2]}" mass="0.4" rgba="0.42 0.47 0.56 1"/>
            <body name="plate" pos="0 {lengths[2]} 0">
              <!-- The gripper camera. Mounted behind and above the plate looking
                   straight down the approach axis, so the fingers frame the
                   bottom of the view the way they do on a real wrist cam.
                   xyaxes gives right=+x and up=+z, which puts the view
                   direction along local +y -- the direction the hand reaches.
                   This is the ONLY exteroceptive instrument on the machine. -->
              <camera name="gripper" pos="0 -0.02 0.055" xyaxes="1 0 0 0 0 1"
                      fovy="70"/>
              <geom name="plate_geom" contype="4" conaffinity="3" type="box" size="0.05 0.012 0.04"
                    mass="0.35" rgba="0.25 0.5 0.7 1"/>
              <body name="finger_left" pos="0 0 0">
                <joint name="finger_left" class="finger" type="slide" axis="1 0 0"
                       range="{travel[0]} {travel[1]}"/>
                <geom name="left_geom" contype="4" conaffinity="3" type="box"
                      pos="0 {reach / 2 + 0.012} 0"
                      size="{thick} {reach / 2} {pad}" mass="0.06"
                      rgba="0.3 0.7 0.9 1"/>
              </body>
              <body name="finger_right" pos="0 0 0">
                <joint name="finger_right" class="finger" type="slide" axis="-1 0 0"
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
  <!-- THE JAWS ARE MECHANICALLY LINKED, as a parallel gripper's are: one
       motor, one screw, both jaws always the same distance from the centre.
       Without this they are two independent actuators that merely happen to be
       commanded alike, and an object between them is free to push one closed
       while the other opens. The APERTURE stays correct throughout -- which is
       why nothing downstream noticed -- while the CENTRE walks sideways, and a
       cabinet handle ended up pinned against one pad with the other shut on
       nothing at all. Coupling them is what makes an object self-centre in the
       jaws, and self-centring is most of what a parallel gripper is for. -->
  <equality>
    <joint joint1="finger_left" joint2="finger_right" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <motor joint="base" name="m_yaw" gear="1" ctrlrange="-80 80"/>
    <motor joint="segment_1" name="m_lift" gear="1" ctrlrange="-120 120"/>
    <motor joint="segment_2" name="m_elbow" gear="1" ctrlrange="-90 90"/>
    <motor joint="segment_3" name="m_wrist" gear="1" ctrlrange="-40 40"/>
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
    _eye: dict | None = None

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

    def view(self, width: int = 320, height: int = 240,
             camera: str = "gripper") -> np.ndarray:
        """What the gripper camera sees, as RGB pixels.

        A real render through a real camera in the model, not a geometric
        stand-in. The renderer is built once and kept: constructing one per
        frame costs a GL context each time.
        """
        # Cached per SIZE, not per width. Two cameras at two resolutions
        # alternating through one width-keyed slot rebuilt the renderer on every
        # single call, and a dozen abandoned GL contexts later every frame came
        # back black -- which reads exactly like a camera pointed at nothing.
        if self._eye is None:
            self._eye = {}
        key = (int(width), int(height))
        if key not in self._eye:
            self._eye[key] = mujoco.Renderer(self.model, height=height,
                                             width=width)
        viewer = self._eye[key]
        viewer.update_scene(self.data, camera=camera)
        return viewer.render()

    def camera_pose(self, name: str = "gripper"
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Where a camera is and how it is pointed, in the world.

        For the hand camera this is proprioception: the arm's joint encoders
        plus a fixed mounting. For the room camera it is calibration -- a fixed
        camera whose place in the room was measured once. Both are things a real
        machine may know without looking at anything.
        """
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return (np.asarray(self.data.cam_xpos[index]),
                np.asarray(self.data.cam_xmat[index]).reshape(3, 3))

    def camera_fovy(self, name: str = "gripper") -> float:
        """A camera's vertical field of view in degrees, as the model declares."""
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return float(self.model.cam_fovy[index])

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

    #: Every geom that is part of the hand itself.
    _OWN_GEOMS = ("left_geom", "right_geom", "plate_geom")

    def tip_forces(self) -> tuple[float, float]:
        """Normal force at each fingertip pad, in newtons. A real load cell.

        Deliberately BLIND to what it is touching. forces() above filters to
        contacts involving block_geom, which is fine for scoring a run from the
        outside but is ground truth: a load cell in a fingertip reports a
        number, not the identity of what pressed on it. This sums every contact
        on each pad, whatever produced it -- the block, the bench, the bin, the
        other finger.

        This is what replaces inferring a grip from the encoders. Fingers that
        have stopped closing are not necessarily fingers with something between
        them; they may simply have met each other. Force tells them apart.
        """
        left = right = 0.0
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            }
            # SELF-TOUCH IS NOT A GRIP. With nothing between them the jaws run
            # all the way shut and press on EACH OTHER, and a pad that reports
            # every force it feels reports that one too -- both cells reading
            # hard with the opening at 0.00 cm. Measuring force instead of
            # inferring it from the encoders removed one route to "holding
            # nothing" and left this one open.
            #
            # A machine can tell the difference without being told what it is
            # touching: it knows its own geometry, so a contact whose other side
            # is also part of the gripper is itself, not the world.
            if names <= set(self._OWN_GEOMS):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            size = float(abs(force[0]))
            if "left_geom" in names:
                left = max(left, size)
            if "right_geom" in names:
                right = max(right, size)
        return left, right

    def range_ahead(self, reach_m: float = 1.5) -> tuple[float, str]:
        """Distance from the wrist to the first surface along the approach axis.

        A time-of-flight rangefinder bolted beside the gripper camera and pointed
        the same way. It sits on the centre line, and the fingers slide out to
        either side of it, so the beam passes between the jaws rather than into
        them -- which is what makes the reading mean "how far to what is in
        front of the hand".

        It answers the one question the top-down camera cannot: HEIGHT. A camera
        looking straight down sees where a thing is across the bench and says
        nothing about how far below it is; that gap is why the jaws were opened
        wide and closed until they stalled. One number fixes it.

        Returns the distance and the name of what was struck. The name is for
        the log and for tests -- it is not offered to the planner, because a
        rangefinder returns a distance and nothing else.
        """
        origin, _ = self.camera_pose()
        direction = np.ascontiguousarray(self.approach(), dtype=np.float64)
        hit = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model, self.data,
            np.ascontiguousarray(origin, dtype=np.float64), direction,
            None, 1, -1, hit)
        if distance < 0.0 or distance > reach_m:
            return reach_m, ""
        struck = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0])) or ""
        return float(distance), struck

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
         table_top: float = 0.72, scene: str = "bin") -> Body:
    xml = _model_xml(block_half, block_at, table_top, scene)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body = Body(model=model, data=data, block_half=np.asarray(block_half))
    mujoco.mj_forward(model, data)
    return body
