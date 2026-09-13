"""ClearWorkArea bound to a body: one TransferObject session per object, the world carried exactly between them.

The generated clearance tree runs the shared TransferObject subskills once
per object. On physics each object is manipulated through its own session
-- the scene is compiled with that object under the grasp machinery's
canonical names and every other object standing in the world as a
bystander -- and when the tree moves to the next object the runtime closes
the session, reads every object's pose and the arm's joints from the last
recorded state, and opens the next session on exactly that state at the
same clock. Nothing is reset: an object knocked over stays knocked over,
one dropped on the table stays there, and the arm begins where it stood.
The clock is one clock across the chain, and every segment is a
replayable record.

The work area itself is observed, not believed: after every pass the arm
stands clear (its reference configuration, on a guarded path) and the
declared cameras cast rays at every object as it stands; the area is clear
only when every object is seen and none of them lies in it, and an
object's placement stands only while a camera sees it in its cell. A cube
a later placement knocked out of its cell loses its fact and the next pass
transfers it again. An object no camera can see leaves the observation
undecided.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rigby_core.skills import Interrupt, LeafContext, LeafOutcome, Verdict

from ..contact.closure import ClosureConfig
from ..contact.placement import PlacementGoal
from ..contact.transfer import TransferStart
from ..gates.control import ControllerConfig
from ..scenes.environment import OBJECT_PREFIX, EnvironmentV1, FixtureV1, SceneObjectV1
from ..sensing.sensors import CAMERAS, visible_fraction
from .transfer_object import Disturbance, TransferObjectRuntime, TransferObjectSession


@dataclass(frozen=True)
class Layout:
    """Where the slots, the cells and the platform lie: explicit positions,
    nearest the robot first, so a world of n objects uses the first n of
    each. The pitch between neighbours has to clear the widest enabled jaw
    standing open beside its target -- a finger that comes down on the next
    cube knocks it away -- and every position has to lie within every
    enabled body's reach envelope."""

    name: str
    pitch_m: float
    platform_half_m: tuple[float, float, float]
    platform_centre_m: tuple[float, float]
    slots_m: tuple[tuple[float, float], ...]
    """where the objects start, on the table, nearest first"""
    cell_offsets_m: tuple[tuple[float, float], ...]
    """the cells, as offsets from the platform centre, nearest first"""
    work_area_min_m: tuple[float, float]
    work_area_max_m: tuple[float, float]
    note: str
    table_half_m: tuple[float, float, float] | None = None
    """The table's half extents when the layout needs more than the G10 table gives it (None: the G10 table as it is)."""

    def as_json(self) -> dict:
        return {"name": self.name, "pitch_m": self.pitch_m, "platform_half_m": list(self.platform_half_m), "platform_centre_m": list(self.platform_centre_m), "table_half_m": None if self.table_half_m is None else list(self.table_half_m),
                "slots_m": [list(v) for v in self.slots_m], "cell_offsets_m": [list(v) for v in self.cell_offsets_m],
                "cells_m": [[round(self.platform_centre_m[0] + dx, 4), round(self.platform_centre_m[1] + dy, 4)] for dx, dy in self.cell_offsets_m],
                "work_area_m": {"min": list(self.work_area_min_m), "max": list(self.work_area_max_m)}, "note": self.note}


def _grid(columns: tuple[float, ...], rows: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    return tuple((x, y) for y in rows for x in columns)


LAYOUT_V1 = Layout(name="v1", pitch_m=0.08, platform_half_m=(0.14, 0.18, 0.015), platform_centre_m=(0.06, 0.65),
                   slots_m=_grid((-0.33, -0.25, -0.17), (0.55, 0.63, 0.71, 0.79)), cell_offsets_m=_grid((-0.08, 0.0, 0.08), (-0.12, -0.04, 0.04, 0.12)),
                   work_area_min_m=(-0.38, 0.50), work_area_max_m=(-0.12, 0.84),
                   note="eight centimetres between neighbours, a grid to the robot's left and a grid to its right: inside the long arm's open jaw (17 cm across the finger faces), so its finger came down on "
                        "the next cube; the far row beyond the dual arm's reach; kept for the baseline evidence only")
LAYOUT_V2 = Layout(name="v2", pitch_m=0.12, platform_half_m=(0.18, 0.24, 0.015), platform_centre_m=(0.18, 0.58),
                   slots_m=((-0.06, 0.40), (-0.18, 0.40), (-0.06, 0.52), (-0.18, 0.52), (-0.30, 0.40), (-0.30, 0.52), (-0.06, 0.64), (-0.18, 0.64), (-0.30, 0.64), (-0.06, 0.76), (-0.18, 0.76)),
                   cell_offsets_m=((-0.12, -0.18), (0.0, -0.18), (0.0, -0.06), (-0.12, -0.06), (0.12, -0.18), (0.12, -0.06), (0.0, 0.06), (-0.12, 0.06), (0.12, 0.06), (0.0, 0.18), (-0.12, 0.18), (0.12, 0.18)),
                   work_area_min_m=(-0.36, 0.34), work_area_max_m=(0.0, 0.82),
                   note="twelve centimetres between neighbours, the widest enabled jaw opening to 17.2 cm across the outer finger faces, so a finger beside its target clears the next cube by two "
                        "centimetres before the draws' jitter; eleven slots and twelve cells, every one within 95% of every enabled body's directional reach, nearest the robot first; the run "
                        "found the envelope is not the solver (the dual arm's IK missed two of them by four millimetres) and the arm parked over the platform hid cubes from the cameras; kept as registered")
LAYOUT_V3 = Layout(name="v3", pitch_m=0.10, platform_half_m=(0.21, 0.21, 0.015), platform_centre_m=(0.19, 0.55),
                   slots_m=((-0.16, 0.5), (-0.26, 0.4), (-0.26, 0.5), (-0.16, 0.6), (-0.36, 0.4), (-0.26, 0.6), (-0.36, 0.5), (-0.36, 0.6), (-0.46, 0.4), (-0.26, 0.7), (-0.46, 0.5)),
                   cell_offsets_m=((-0.05, -0.15), (-0.15, -0.05), (-0.05, -0.05), (-0.15, 0.05), (0.05, -0.05), (-0.05, 0.05), (0.05, 0.05), (0.15, -0.05), (-0.15, 0.15), (-0.05, 0.15), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15)),
                   work_area_min_m=(-0.51, 0.35), work_area_max_m=(-0.11, 0.75), table_half_m=(0.45, 0.30, 0.005),
                   note="ten centimetres between neighbours with the jaw opened to the object's width (the long arm's outer finger faces then 11.6 cm apart); every slot and cell a position every enabled body "
                        "transferred a cube from and to alone (the registered qualification table), nearest the robot first; the table widened to the left for the far slot column")
LAYOUT = LAYOUT_V3
"""Thirteen cells on an enlarged platform to the robot's right, eleven slots on the table to its left, every one qualified on every enabled body."""
CELL_HALF_M = 0.035
GOAL_HEIGHT_M = 0.09
CUBE_HALF_M = 0.015
CUBE_MASS_KG = 0.00864
CUBE_FRICTION = 1.4
STAND_BY_S = 0.5
OBSERVE_AREA_S = 0.3


@dataclass(frozen=True)
class ClearanceWorld:
    environment: EnvironmentV1
    objects: tuple[str, ...]
    cells: dict[str, str]
    """object name -> cell name"""
    offsets: dict[str, tuple[float, float]]
    """cell name -> offset on the platform"""
    layout: Layout = LAYOUT

    def goal_for(self, cell: str) -> PlacementGoal:
        platform = next(f for f in self.environment.fixtures if f.name == "platform")
        top = float(platform.position_m[2] + platform.size_m[2])
        dx, dy = self.offsets[cell]
        x, y = float(platform.position_m[0] + dx), float(platform.position_m[1] + dy)
        return PlacementGoal(region_minimum_m=(x - CELL_HALF_M, y - CELL_HALF_M, top - 0.001), region_maximum_m=(x + CELL_HALF_M, y + CELL_HALF_M, top + GOAL_HEIGHT_M),
                             dwell_s=2.0, maximum_linear_speed_mps=0.01, maximum_angular_speed_radps=0.1)

    def cell_of(self, name: str) -> str:
        return self.cells[name]


def build_clearance_world(base: EnvironmentV1, count: int, *, draw: dict | None = None, layout: Layout = LAYOUT) -> ClearanceWorld:
    """The G10 table and floor, an enlarged platform, and ``count`` cubes in
    the work-area slots, each jittered by its draw."""

    if not 1 <= count <= min(len(layout.slots_m), len(layout.cell_offsets_m)):
        raise ValueError(f"count must be 1..{min(len(layout.slots_m), len(layout.cell_offsets_m))}")
    table = next(f for f in base.fixtures if f.name == "table")
    if layout.table_half_m is not None:
        table = table.model_copy(update={"size_m": tuple(float(v) for v in layout.table_half_m)})
    floor = next(f for f in base.fixtures if f.name == "floor")
    top = float(table.position_m[2] + table.size_m[2])
    platform = FixtureV1(name="platform", size_m=layout.platform_half_m, position_m=(layout.platform_centre_m[0], layout.platform_centre_m[1], top + layout.platform_half_m[2]), rgba=(0.3, 0.45, 0.75, 1.0))
    slots = list(layout.slots_m)[:count]
    objects = []
    names = tuple(f"cube_{index:02d}" for index in range(1, count + 1))
    for index, (name, (x, y)) in enumerate(zip(names, slots)):
        jitter = (draw or {}).get("objects", [{}] * count)[index] if draw else {}
        translation = jitter.get("translation_m", [0.0, 0.0, 0.0])
        objects.append(SceneObjectV1(name=name, size_m=(CUBE_HALF_M, CUBE_HALF_M, CUBE_HALF_M), mass_kg=round(CUBE_MASS_KG * jitter.get("mass_multiplier", 1.0), 9),
                                     position_m=(x + translation[0], y + translation[1], top + CUBE_HALF_M), friction=round(CUBE_FRICTION * jitter.get("friction_multiplier", 1.0), 6),
                                     rgba=(0.85, 0.45 + 0.04 * index, 0.15, 1.0)))
    environment = EnvironmentV1(environment_id=f"g15_clearance_{count}" + ("" if layout is LAYOUT else f"_{layout.name}"),
                                description=f"{count} cubes in the work area on the table, neighbours {layout.pitch_m * 100:.0f} cm apart, an enlarged platform of twelve cells to the right; the G10 table and floor.",
                                robot_mount_m=base.robot_mount_m, fixtures=(platform, table, floor), objects=tuple(objects))
    cell_names = tuple(f"cell_{index:02d}" for index in range(1, count + 1))
    offsets = {f"cell_{index:02d}": (dx, dy) for index, (dx, dy) in enumerate(layout.cell_offsets_m, start=1)}
    return ClearanceWorld(environment=environment, objects=names, cells=dict(zip(names, cell_names)), offsets={c: offsets[c] for c in cell_names}, layout=layout)


def in_work_area(position_m, layout: Layout = LAYOUT) -> bool:
    x, y = float(position_m[0]), float(position_m[1])
    return layout.work_area_min_m[0] <= x <= layout.work_area_max_m[0] and layout.work_area_min_m[1] <= y <= layout.work_area_max_m[1]


# -- the runtime ----------------------------------------------------------------------------


class ChainClock:
    """The executor's clock: the current session's physics time, or the time
    the last session ended while none is open."""

    def __init__(self, runtime: "ClearWorkAreaRuntime") -> None:
        self.runtime = runtime

    def now(self) -> float:
        session = self.runtime.current
        return float(session.time_s) if session is not None else float(self.runtime.clock_s)


@dataclass
class Segment:
    index: int
    object_name: str
    session: TransferObjectSession
    runtime: TransferObjectRuntime
    started_s: float
    ended_s: float | None = None
    leaves: list[str] = field(default_factory=list)


@dataclass
class ClearWorkAreaRuntime:
    """A LeafRuntime over a chain of TransferObject sessions, one per object."""

    body: str
    source: Path
    robot: Any
    world: ClearanceWorld
    policy: dict
    interrupt: Interrupt
    configuration: str = "front_overhead_contact"
    controller_config: ControllerConfig | None = None
    closure_config: ClosureConfig | None = None
    duration_scale: float = 1.0
    disturbance: tuple[str, Disturbance] | None = None
    """The object whose session carries the protocol's disturbance, and the disturbance."""
    seed_label: str = ""
    report_placements: bool = True
    """Whether a look reports each object's placement from the cameras (protocol v2); False reproduces the first runtime, whose look reported the area alone."""
    current: TransferObjectSession | None = None
    current_object: str | None = None
    inner: TransferObjectRuntime | None = None
    clock_s: float = 0.0
    poses: dict[str, np.ndarray] = field(default_factory=dict)
    """object name -> its free joint's seven values as last recorded"""
    arm: dict[str, float] = field(default_factory=dict)
    """joint name -> position as last recorded, for every joint that is not an object's"""
    segments: list[Segment] = field(default_factory=list)
    switches: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    # -- sessions ----------------------------------------------------------------------------

    def session_for(self, name: str) -> TransferObjectSession:
        if self.current is not None and self.current_object == name:
            return self.current
        if self.current is not None:
            self.close_current()
        cell = self.world.cell_of(name)
        disturbance = self.disturbance[1] if self.disturbance is not None and self.disturbance[0] == name else None
        session = TransferObjectSession.open(self.body, self.source, self.environment_now(), self.world.goal_for(cell), self.policy, configuration_name=self.configuration,
                                             disturbance=disturbance, seed_label=f"{self.seed_label}|{name}", controller_config=self.controller_config, closure_config=self.closure_config,
                                             duration_scale=self.duration_scale, object_name=name, destination_fixture="platform", destination_offset_m=self.world.offsets[cell], robot=self.robot)
        if self.poses or self.arm:
            qpos = self.carried_qpos(session)
            session.adopt(TransferStart(qpos=qpos, qvel=np.zeros(session.model.nv), time_s=float(self.clock_s), state=None))
        else:
            session.time_s = float(self.clock_s)
        self.current, self.current_object = session, name
        self.inner = TransferObjectRuntime(session, self.interrupt)
        self.segments.append(Segment(index=len(self.segments), object_name=name, session=session, runtime=self.inner, started_s=float(self.clock_s)))
        self.switches += 1
        return session

    def environment_now(self) -> EnvironmentV1:
        """The authored world with every object at its last recorded position
        (the exact pose, orientation included, is carried through the joint
        values; the authored position only seeds the compile)."""

        if not self.poses:
            return self.world.environment
        objects = tuple(item.model_copy(update={"position_m": tuple(float(v) for v in self.poses[item.name][:3])}) if item.name in self.poses else item for item in self.world.environment.objects)
        return self.world.environment.model_copy(update={"objects": objects})

    def carried_qpos(self, session: TransferObjectSession) -> np.ndarray:
        model = session.model
        qpos = np.array(session.current_qpos(), dtype=float)
        for joint_name, value in self.arm.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint >= 0:
                qpos[int(model.jnt_qposadr[joint])] = value
        for name, pose in self.poses.items():
            joint_name = "scene_block_free" if name == session.object_name else f"{OBJECT_PREFIX}{name}_free"
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint >= 0:
                address = int(model.jnt_qposadr[joint])
                qpos[address: address + 7] = pose
        return qpos

    def close_current(self) -> None:
        session = self.current
        if session is None:
            return
        model = session.model
        qpos = np.array(session.state.qpos if session.state is not None else session.current_qpos(), dtype=float)
        for name in self.world.objects:
            joint_name = "scene_block_free" if name == session.object_name else f"{OBJECT_PREFIX}{name}_free"
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint >= 0:
                address = int(model.jnt_qposadr[joint])
                self.poses[name] = qpos[address: address + 7].copy()
        object_joints = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")} | {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{OBJECT_PREFIX}{n}_free") for n in self.world.objects}
        for joint in range(model.njnt):
            if joint in object_joints:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name and int(model.jnt_type[joint]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                self.arm[name] = float(qpos[int(model.jnt_qposadr[joint])])
        self.clock_s = float(session.time_s)
        segment = self.segments[-1]
        segment.ended_s = self.clock_s
        self.calls.extend({**c, "segment": segment.index, "object": segment.object_name} for c in self.inner.calls)
        self.verdicts.extend({**v, "segment": segment.index, "object": segment.object_name} for v in self.inner.verdicts)
        self.current, self.current_object, self.inner = None, None, None

    def finish(self) -> None:
        self.close_current()

    # -- leaves ------------------------------------------------------------------------------

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        node = context.node
        name = node.arguments.get("object")
        if name is None:
            # stand_clear is bound to the effector alone: it acts in the
            # session that is open, or the last object's.
            name = self.current_object if self.current is not None else (self.segments[-1].object_name if self.segments else self.world.objects[-1])
        session = self.session_for(name)
        self.segments[-1].leaves.append(node.skill_id)
        if node.skill_id == "stand_by":
            # The pass moves on: the arm holds where it is for a moment, the
            # closure as it stands (an object it still holds stays held).
            holding = bool(context.belief.get(f"held:{name}", False))
            spent = session.hold_still(STAND_BY_S, holding=holding, should_stop=lambda now: context.should_stop())
            self.calls.append({"node": node.node_id, "leaf": "stand_by", "object": name, "segment": self.segments[-1].index, "physics_s": spent})
            return LeafOutcome(Verdict.SUCCESS, evidence={"leaf": "stand_by", "object": name, "physics_s": spent})
        if node.skill_id == "stand_clear":
            # Out of the cameras' way before the look: the reference
            # configuration on a guarded path; where no path is found the
            # arm stays and the look does what it can.
            moved = session.reroute(holding=False)
            self.calls.append({"node": node.node_id, "leaf": "stand_clear", "object": name, "segment": self.segments[-1].index, **{k: moved.get(k) for k in ("executed", "refusal", "physics_s", "joint_travel_rad", "detail")}})
            return LeafOutcome(Verdict.SUCCESS, evidence={"leaf": "stand_clear", "executed": bool(moved.get("executed")), "refusal": moved.get("refusal"), "physics_s": moved.get("physics_s", 0.0)})
        return self.inner.run_primitive(context)

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        node = context.node
        if node.skill_id == "observe_work_area":
            return self.observe_work_area(context)
        session = self.session_for(node.arguments.get("object"))
        self.segments[-1].leaves.append(node.skill_id)
        return self.inner.observe(context)

    def observe_work_area(self, context: LeafContext) -> dict[str, Any] | None:
        """Every object as the declared cameras see it now: clear when all are
        seen and none lies in the area, undecided when one is hidden; and for
        every object seen, whether it stands in its cell, which is what its
        ``placed`` fact becomes. A cube a later placement knocked out of its
        cell loses the fact here, and the next pass transfers it again."""

        session = self.current if self.current is not None else self.session_for(self.world.objects[-1])
        self.segments[-1].leaves.append("observe_work_area")
        session.hold_still(OBSERVE_AREA_S, holding=False, should_stop=lambda now: context.should_stop())
        model = session.model
        data = mujoco.MjData(model)
        if session.state is not None and session.state.state is not None:
            from rigby_core.simulation.recording import STATE_SPEC

            mujoco.mj_setState(model, data, np.asarray(session.state.state, dtype=float), STATE_SPEC)
        else:
            data.qpos[:] = session.current_qpos()
        mujoco.mj_forward(model, data)
        cameras = {name: CAMERAS[name] for name in ("overhead", "front")}
        geomid = np.zeros(1, dtype=np.int32)
        rng = np.random.default_rng(int(hashlib.sha256(f"{self.seed_label}|{session.time_s:.3f}".encode("utf-8")).hexdigest()[:8], 16))
        report = []
        clear = True
        undecided = False
        in_cells: dict[str, bool] = {}
        for name in self.world.objects:
            geom_name = "scene_block_geom" if name == session.object_name else f"{OBJECT_PREFIX}{name}_geom"
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            half = np.array(model.geom_size[geom], dtype=float)
            # Either declared camera may see the object: the arm that has just
            # released it often stands between it and the overhead camera.
            views = {}
            for camera_name, camera in cameras.items():
                fraction, centre = visible_fraction(model, data, geom, half, np.array(camera.position_m, dtype=float), geomid)
                views[camera_name] = (float(fraction), centre, camera)
            seen_by = [n for n, (fraction, _, camera) in views.items() if fraction >= camera.visible_fraction]
            if seen_by:
                _, centre, camera = views[seen_by[0]]
                reported = centre + rng.normal(0.0, camera.noise_m, size=3)
            else:
                reported = None
            inside = bool(reported is not None and in_work_area(reported, self.world.layout))
            in_cell = None
            if reported is not None:
                goal = self.world.goal_for(self.world.cell_of(name))
                low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
                in_cell = bool(np.all(reported >= low) and np.all(reported <= high))
                in_cells[name] = in_cell
            report.append({"object": name, "visible_fraction": {n: round(v[0], 3) for n, v in views.items()}, "seen_by": seen_by, "reported_m": None if reported is None else [round(float(v), 4) for v in reported],
                           "in_work_area": inside, "in_cell": in_cell})
            undecided |= not seen_by
            clear &= not inside
        observation = {"time_s": float(session.time_s), "cameras": list(cameras), "objects": report, "clear": None if undecided else bool(clear), "in_cells": in_cells}
        self.observations.append(observation)
        self.calls.append({"node": context.node.node_id, "leaf": "observe_work_area", "segment": self.segments[-1].index, "observed": observation})
        if undecided:
            return None
        facts: dict[str, Any] = {"work_area_clear": bool(clear), "objects_in_work_area": [r["object"] for r in report if r["in_work_area"]]}
        if self.report_placements:
            for name, in_cell in in_cells.items():
                facts[f"placed:{name}:{self.world.cell_of(name)}"] = in_cell
        return facts


def predicates_for_clearance(world: ClearanceWorld):
    pairs = dict(world.cells)
    return {
        "object_known": lambda belief, args: bool(belief.get(f"known:{args[0]}", False)),
        "object_held": lambda belief, args: bool(belief.get(f"held:{args[0]}", False)),
        "object_placed": lambda belief, args: bool(belief.get(f"placed:{args[0]}:{args[1]}", False)),
        "object_in_reach": lambda belief, args: bool(belief.get(f"reach:{args[0]}", False)),
        "all_objects_placed": lambda belief, args: all(belief.get(f"placed:{o}:{c}", False) for o, c in pairs.items()),
        "work_area_clear": lambda belief, args: bool(belief.get("work_area_clear", False)),
        "area_cleared": lambda belief, args: all(belief.get(f"placed:{o}:{c}", False) for o, c in pairs.items()) and bool(belief.get("work_area_clear", False)),
    }


def final_object_poses(runtime: ClearWorkAreaRuntime) -> dict[str, np.ndarray]:
    """Every object's free-joint values as the chain left them."""

    runtime.finish()
    return {name: np.array(pose, dtype=float) for name, pose in runtime.poses.items()}


def oracle_clear(world: ClearanceWorld, poses: dict[str, np.ndarray]) -> dict:
    """The root goal judged from the final poses alone: every object's centre
    inside its cell's region, and nothing left in the work area."""

    per_object = {}
    for name in world.objects:
        goal = world.goal_for(world.cell_of(name))
        position = poses[name][:3]
        low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
        per_object[name] = {"in_cell": bool(np.all(position >= low) and np.all(position <= high)), "in_work_area": in_work_area(position, world.layout), "position_m": [round(float(v), 4) for v in position]}
    placed = sum(1 for v in per_object.values() if v["in_cell"])
    return {"objects": per_object, "placed": placed, "of": len(world.objects), "work_area_clear": not any(v["in_work_area"] for v in per_object.values()),
            "root_success": placed == len(world.objects) and not any(v["in_work_area"] for v in per_object.values())}


__all__ = ["CELL_HALF_M", "ChainClock", "ClearWorkAreaRuntime", "ClearanceWorld", "LAYOUT", "LAYOUT_V1", "LAYOUT_V2", "LAYOUT_V3", "Layout", "Segment", "build_clearance_world", "final_object_poses", "in_work_area", "oracle_clear",
           "predicates_for_clearance"]
