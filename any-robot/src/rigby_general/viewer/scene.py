"""Export a compiled model as geometry a browser can draw and articulate.

The studio used to show an animated GIF, which is a picture of a robot rather
than a robot: you cannot pause it on the frame that matters, orbit to see whether
the wrist really cleared the base, or drive a joint yourself to find out what the
mechanism does. So the viewer gets the model itself.

What crosses the wire is deliberately *not* baked per-frame transforms. Those
would be simpler -- MuJoCo already computes them -- but they are roughly ten
times larger, and more importantly they are dead: a recording of poses cannot be
posed. Shipping the kinematic tree instead means the browser can run forward
kinematics itself, which is what lets a slider drive a joint on a robot nobody
simulated for that purpose.

The cost of that choice is that the browser's forward kinematics must agree with
MuJoCo's exactly, or every pose in the studio is subtly a lie.
:func:`forward_kinematics` is the same algorithm written in Python and checked
against ``mj_kinematics`` in the tests, so the JavaScript has a reference to be
correct against rather than a convention to guess at.

Meshes are sent whole. Measured across every robot here that is 360 KB, most of
it one industrial arm, which is cheaper than any decimation scheme worth writing.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# ``mjtGeom`` values the viewer draws. Anything else is skipped rather than
# approximated, because a height field rendered as a box is worse than a gap.
GEOM_PLANE = 0
GEOM_SPHERE = 2
GEOM_CAPSULE = 3
GEOM_ELLIPSOID = 4
GEOM_CYLINDER = 5
GEOM_BOX = 6
GEOM_MESH = 7

DRAWABLE = frozenset(
    {
        GEOM_PLANE,
        GEOM_SPHERE,
        GEOM_CAPSULE,
        GEOM_ELLIPSOID,
        GEOM_CYLINDER,
        GEOM_BOX,
        GEOM_MESH,
    }
)

JOINT_FREE = 0
JOINT_BALL = 1
JOINT_SLIDE = 2
JOINT_HINGE = 3


def _name(model: mujoco.MjModel, kind: int, index: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"{fallback}_{index}"


def _rgba(model: mujoco.MjModel, geom: int) -> list[float]:
    """Geom colour, preferring its material when it has one."""

    material = int(model.geom_matid[geom])
    if material >= 0:
        return [round(float(v), 4) for v in model.mat_rgba[material]]
    return [round(float(v), 4) for v in model.geom_rgba[geom]]


def build_scene(model: mujoco.MjModel) -> dict:
    """Everything needed to draw and pose this robot, and nothing else."""

    bodies = []
    for body in range(model.nbody):
        bodies.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body, "body"),
                "parent": int(model.body_parentid[body]),
                "pos": [round(float(v), 12) for v in model.body_pos[body]],
                "quat": [round(float(v), 12) for v in model.body_quat[body]],
            }
        )

    joints = []
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind == JOINT_FREE:
            # Not a robot joint: a loose object in the scene, such as the block a
            # grasp probe is trying to pick up. Its seven values are a world
            # position and a quaternion, and they replace the body's frame rather
            # than displacing it.
            joints.append(
                {
                    "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint, "joint"),
                    "body": int(model.jnt_bodyid[joint]),
                    "type": "free",
                    "qposadr": int(model.jnt_qposadr[joint]),
                }
            )
            continue
        if kind not in (JOINT_HINGE, JOINT_SLIDE):
            # A ball joint would need a different parameterisation, and nothing
            # that reaches the viewer has one.
            continue
        address = int(model.jnt_qposadr[joint])
        limited = bool(model.jnt_limited[joint])
        low, high = (float(v) for v in model.jnt_range[joint])
        joints.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint, "joint"),
                "body": int(model.jnt_bodyid[joint]),
                "type": "hinge" if kind == JOINT_HINGE else "slide",
                "axis": [round(float(v), 12) for v in model.jnt_axis[joint]],
                "pos": [round(float(v), 12) for v in model.jnt_pos[joint]],
                "qposadr": address,
                "ref": round(float(model.qpos0[address]), 12),
                "limited": limited,
                "range": [round(low, 6), round(high, 6)] if limited else None,
            }
        )

    meshes: list[dict] = []
    mesh_index: dict[int, int] = {}
    geoms = []
    for geom in range(model.ngeom):
        kind = int(model.geom_type[geom])
        if kind not in DRAWABLE:
            continue
        entry = {
            "body": int(model.geom_bodyid[geom]),
            "type": kind,
            "size": [round(float(v), 7) for v in model.geom_size[geom]],
            "pos": [round(float(v), 7) for v in model.geom_pos[geom]],
            "quat": [round(float(v), 7) for v in model.geom_quat[geom]],
            "rgba": _rgba(model, geom),
        }
        if kind == GEOM_MESH:
            source = int(model.geom_dataid[geom])
            if source not in mesh_index:
                mesh_index[source] = len(meshes)
                meshes.append(_mesh(model, source))
            entry["mesh"] = mesh_index[source]
        geoms.append(entry)

    return {
        "nq": int(model.nq),
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "meshes": meshes,
        "qpos0": [round(float(v), 12) for v in model.qpos0],
    }


def _mesh(model: mujoco.MjModel, mesh: int) -> dict:
    start = int(model.mesh_vertadr[mesh])
    count = int(model.mesh_vertnum[mesh])
    face_start = int(model.mesh_faceadr[mesh])
    face_count = int(model.mesh_facenum[mesh])

    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count], dtype=int
    )
    return {
        "vert": [round(float(v), 5) for v in vertices.reshape(-1)],
        "face": [int(v) for v in faces.reshape(-1)],
    }


# --------------------------------------------------------------------------
# The reference implementation the JavaScript has to match
# --------------------------------------------------------------------------


def _quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


@dataclass(frozen=True, slots=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray


def forward_kinematics(scene: dict, qpos) -> list[Pose]:
    """World pose of every body, from joint values alone.

    This is the algorithm the viewer runs, written once here so it can be
    checked against ``mj_kinematics``. A body's frame is its parent's frame
    composed with its own fixed offset, and then each of its joints displaces it:
    a hinge rotates the frame about an axis through its anchor, a slide
    translates along one. Both the axis and the anchor are expressed in the
    body's own frame, which is why they are applied after the fixed offset rather
    than before.

    A free joint is the exception: it does not displace a frame, it *is* the
    frame. Its seven values give a world position and orientation outright, which
    is what a loose object in a scene needs -- the block in a grasp probe is
    placed by the physics, not by the robot's tree.
    """

    values = np.asarray(qpos, dtype=float)
    poses: list[Pose] = []
    joints_by_body: dict[int, list[dict]] = {}
    for joint in scene["joints"]:
        joints_by_body.setdefault(int(joint["body"]), []).append(joint)

    for index, body in enumerate(scene["bodies"]):
        parent = int(body["parent"])
        if index == 0:
            poses.append(Pose(np.zeros(3), np.eye(3)))
            continue

        base = poses[parent]
        rotation = base.rotation @ _quat_to_mat(body["quat"])
        position = base.position + base.rotation @ np.asarray(body["pos"], dtype=float)

        for joint in joints_by_body.get(index, ()):
            if joint["type"] == "free":
                address = int(joint["qposadr"])
                position = np.asarray(values[address : address + 3], dtype=float)
                rotation = _quat_to_mat(values[address + 3 : address + 7])
                continue
            anchor = np.asarray(joint["pos"], dtype=float)
            axis = np.asarray(joint["axis"], dtype=float)
            value = float(values[int(joint["qposadr"])]) - float(joint["ref"])
            if joint["type"] == "hinge":
                spin = _axis_angle(axis, value)
                # Rotate about the anchor, not the body origin.
                position = position + rotation @ (anchor - spin @ anchor)
                rotation = rotation @ spin
            else:
                position = position + rotation @ (axis * value)

        poses.append(Pose(position, rotation))
    return poses
