"""The viewer's kinematics must be MuJoCo's, or every pose it shows is a lie.

The studio draws robots by shipping the kinematic tree to the browser and
running forward kinematics there, rather than shipping baked per-frame
transforms. That choice is what lets a slider pose a robot nobody simulated for
that purpose -- and it puts a second implementation of forward kinematics in the
system, in a language with no tests.

So the Python in :mod:`rigby_general.viewer.scene` is the reference: the same
algorithm, checked here against ``mj_kinematics``, which the JavaScript is a
direct port of. If the port drifts, this is the file that says what it drifted
from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_general.pipeline import ingest_robot
from rigby_general.viewer import build_scene, forward_kinematics, sample_track


ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "assets" / "general" / "zoo"
EXOTIC = ROOT / "assets" / "general" / "exotic"

# One of each shape of problem: a plain arm, a bimanual one, a many-fingered
# hand, and -- when it has been fetched -- a real arm whose geometry is meshes.
ZOO_ROBOTS = ("zoo_tool_arm", "zoo_jaw_arm", "zoo_hand_arm", "zoo_dual_arm")

# Sub-nanometre. The residual is float noise between MuJoCo's quaternion
# composition and this matrix composition, not a difference in the kinematics.
TOLERANCE_M = 1e-9


def robots():
    for robot_id in ZOO_ROBOTS:
        yield robot_id, ZOO / robot_id / "robot.urdf"
    mesh_arm = EXOTIC / "iiwa7" / "robot.urdf"
    if mesh_arm.is_file():
        yield "iiwa7", mesh_arm


@pytest.fixture(scope="module")
def ingested():
    return {
        robot_id: ingest_robot(source, robot_id=robot_id)
        for robot_id, source in robots()
    }


def random_qpos(model: mujoco.MjModel, rng) -> np.ndarray:
    qpos = np.array(model.qpos0, dtype=float)
    for joint in range(model.njnt):
        if model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        if model.jnt_limited[joint]:
            low, high = (float(v) for v in model.jnt_range[joint])
        else:
            low, high = -np.pi, np.pi
        qpos[int(model.jnt_qposadr[joint])] = rng.uniform(low, high)
    return qpos


def test_exported_kinematics_reproduce_mujoco(ingested) -> None:
    """Every body, every robot, across the whole joint range."""

    rng = np.random.default_rng(11)
    for robot_id, robot in ingested.items():
        model = robot.finalized.model
        scene = build_scene(model)
        data = mujoco.MjData(model)

        for _ in range(20):
            qpos = random_qpos(model, rng)
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            poses = forward_kinematics(scene, qpos)

            for body in range(model.nbody):
                assert poses[body].position == pytest.approx(
                    np.asarray(data.xpos[body]), abs=TOLERANCE_M
                ), f"{robot_id} body {body} position"
                assert poses[body].rotation == pytest.approx(
                    np.asarray(data.xmat[body]).reshape(3, 3), abs=TOLERANCE_M
                ), f"{robot_id} body {body} orientation"


def test_the_scene_carries_every_drawable_geom(ingested) -> None:
    """A silently dropped geom is a robot that renders with a piece missing."""

    for robot_id, robot in ingested.items():
        model = robot.finalized.model
        scene = build_scene(model)
        drawable = sum(
            1
            for geom in range(model.ngeom)
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_HFIELD)
        )
        assert len(scene["geoms"]) == drawable, robot_id
        for geom in scene["geoms"]:
            if "mesh" in geom:
                assert 0 <= geom["mesh"] < len(scene["meshes"])


def test_every_actuated_joint_can_be_driven(ingested) -> None:
    """The explorer's sliders come from this list; a missing joint is a dead one."""

    for robot_id, robot in ingested.items():
        model = robot.finalized.model
        scene = build_scene(model)
        expected = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for joint in range(model.njnt)
            if model.jnt_type[joint]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        }
        assert {joint["name"] for joint in scene["joints"]} == expected, robot_id


def test_a_sampled_track_keeps_its_ends(ingested) -> None:
    """Scrubbing to the end must land on the pose the gates actually judged."""

    times = np.linspace(0.0, 7.5, 1801)
    qpos = np.column_stack([np.sin(times), np.cos(times), times * 0.1])
    track = sample_track(times, qpos, max_frames=60)

    assert track["frames"] <= 61
    assert track["nq"] == 3
    assert track["duration_s"] == pytest.approx(7.5)
    assert track["times"][0] == pytest.approx(0.0)
    assert track["times"][-1] == pytest.approx(7.5)
    assert track["qpos"][-3:] == pytest.approx(list(qpos[-1]), abs=1e-5)
    assert len(track["qpos"]) == track["frames"] * track["nq"]


def test_the_javascript_is_a_port_of_this_module() -> None:
    """A cheap guard on the one invariant a Python test cannot otherwise reach.

    The browser runs its own forward kinematics. Nothing here can execute it, but
    the two implementations agreeing is load-bearing, so at least assert that the
    JavaScript still contains the algorithm and still points at its reference.
    """

    source = (ROOT / "scripts" / "studio_viewer.js").read_text(encoding="utf-8")

    assert "forwardKinematics" in source
    assert "rigby_general.viewer.scene.forward_kinematics" in source, (
        "the port lost the pointer to the module it must match"
    )
    for behaviour in ("hinge", "slide", "axisAngleMat", "quatToMat"):
        assert behaviour in source, behaviour


def test_the_studio_declares_the_viewer_payloads() -> None:
    """The builder must still inline both scripts and both data blocks."""

    builder = (ROOT / "scripts" / "build_studio.py").read_text(encoding="utf-8")
    for token in ("__VIEWER__", "__VIEWER_JS__", "studio_viewer.js", "studio_views.js"):
        assert token in builder, token
