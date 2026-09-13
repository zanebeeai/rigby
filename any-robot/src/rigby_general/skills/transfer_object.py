"""The closed-loop TransferObject skill bound to a body: leaves on physics, verifications from sensors.

The neutral ``transfer_object`` library says what happens and in what
order; this binds its leaves. ``acquire``, ``transport`` and ``release``
are the G06 transfer primitive over contiguous phase ranges, each
continuing the world the last leaf left -- the arm where it stands, the
object where it lies, the clock where it stood -- and ``acquire`` plans
to where a sensor last saw the object, never to where the world was
authored to put it. ``observe_object``, ``verify_hold`` and
``verify_placement`` are observations: the arm holds still for the
window while the declared sensors sample, and the G09 conditionals
decide from those samples alone; an undecidable verdict is re-observed
within the conditional's fallback budget and otherwise left undecided,
never counted as a success. During the carry the ``held`` conditional is
decided every few hundredths of a second, and a hold the sensors lose
stops the carry where it is, typed, so the next attempt begins from the
object wherever it went.

A protocol may declare disturbances: an external force on the object, or
an occluder moved between the camera and the fixtures, applied through
the physics step hook so the record keeps them as user input and replays
them. Nothing here resets the world or writes the object's state.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecorder
from rigby_core.skills import ConditionalV1, Decision, Interrupt, LeafContext, LeafOutcome, Verdict

from ..contact.transfer import TransferResult, TransferStart, attempt_transfer, transfer_scene_from_environment
from ..contact.placement import PlacementGoal
from ..contact.closure import ClosureConfig
from ..contracts import EffectorV1
from ..gates.control import ControllerConfig
from ..grounding.grounder import figure_site_for
from ..grounding.workspace import WorkspaceFrame, build_workspace_frame
from ..pipeline import ingest_robot
from ..scenes.environment import EnvironmentV1, FixtureV1
from ..sensing import LiveSensing, bind_conditionals, configuration, decide_live, policy_digest
from ..grounding.grounder import _collision_guard
from ..transitions.boundary import arm_joint_names
from ..transitions.repair import joint_move, track


SHUTTER_BODY = "g10_shutter"
SHUTTER_PARKED = (0.0, -3.0, -3.0)
SHUTTER_ACTIVE = (-0.08, 1.0, 0.41)
SHUTTER_HALF = (0.3, 0.005, 0.3)
"""An occluder the protocol can move between the front camera and the
fixtures: a mocap body that collides with nothing and blocks rays."""

FLOOR_TOP_M = -0.12
"""Where the floor's top surface lies: under every enabled body's base
(the lowest base geom bound is 96 mm below the mount) and 38 cm below the
fixture tops, so an object that leaves the fixtures lands on something
the sensors can see and the arm may or may not reach."""


TABLE_TOP_M = 0.23
"""The table's top: the G06 bench and platform, three centimetres tall
with their tops at 0.26 m, stand on it. An object that leaves a fixture
lands on the table three centimetres lower, in every enabled body's reach
and the front camera's view, instead of on a floor 38 cm down."""
TABLE_HALF_M = (0.40, 0.30, 0.005)
TABLE_CENTRE_M = (-0.05, 0.65)


def with_floor(environment: EnvironmentV1) -> EnvironmentV1:
    """The world standing on a floor. The G06 world has none: an object
    dropped beside its fixtures fell forever. A floor is a fixture like
    the bench, two metres square, its top at ``FLOOR_TOP_M``."""

    floor = FixtureV1(name="floor", size_m=(1.0, 1.0, 0.01), position_m=(0.0, 0.5, FLOOR_TOP_M - 0.01), rgba=(0.42, 0.42, 0.45, 1.0))
    return environment.model_copy(update={"environment_id": f"{environment.environment_id}+floor", "fixtures": (*environment.fixtures, floor),
                                          "description": environment.description + " The fixtures stand on a floor twelve centimetres below the mount."})


def with_table(environment: EnvironmentV1) -> EnvironmentV1:
    """The G06 fixtures standing on a table. In the first scored pass
    (retained as pilot 1) an object that slipped from the grip fell from
    its lift height onto a slab one object-length across, bounced off it
    and landed on a floor 38 cm down, outside the dual arm's reach shell
    and beside slabs the other arms' descents collided with; thirteen of
    the dual arm's twenty induced slips ended there. A table under the
    fixtures is where such fixtures stand; nothing about the task -- the
    cube, its start, the region, the fixtures -- changes."""

    table = FixtureV1(name="table", size_m=TABLE_HALF_M, position_m=(TABLE_CENTRE_M[0], TABLE_CENTRE_M[1], TABLE_TOP_M - TABLE_HALF_M[2]), rgba=(0.55, 0.50, 0.42, 1.0))
    return environment.model_copy(update={"environment_id": f"{environment.environment_id}+table", "fixtures": (*environment.fixtures, table),
                                          "description": environment.description + " The bench and the platform stand on a table whose top is three centimetres below theirs."})


def g10_world(environment: EnvironmentV1) -> EnvironmentV1:
    """The registered G10 world: the G06 fixtures on a table, on a floor."""

    return with_floor(with_table(environment))


MONITOR_PERIOD_S = 0.05
"""How often the carry asks the held conditional."""
OBSERVE_WAIT_S = 1.0
"""How long an observation holds still before asking its sensors again."""
STILL_WINDOW_S = 1.0
STILL_SPEED_MPS = 0.02
"""An object is planned to only once the camera has seen it still: the
mean of its sensed positions over the last half second against the half
second before, under two centimetres a second. A falling or rolling
object is re-observed instead; the camera's noise, averaged over fifteen
frames a half, reads as a millimetre or two a second."""
PHASES_OF = {"acquire": ("approach", "hold"), "transport": ("carry", "carry"), "release": ("lower", "retreat")}
REROUTE_ON = ("facing_unmet", "self_collision_path", "unreachable_path")
"""Refusals that come from the posture the arm stands in rather than from
the world: the leaf moves the arm to its reference configuration on a
guarded path and plans once more, as the G08 composition did."""


def with_shutter(scene, xml: str):
    """The scene with the protocol's occluder added, parked out of every camera's view."""

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    body = ET.SubElement(worldbody, "body", name=SHUTTER_BODY, mocap="true", pos=" ".join(f"{v:.4f}" for v in SHUTTER_PARKED))
    ET.SubElement(body, "geom", name=f"{SHUTTER_BODY}_geom", type="box", size=" ".join(f"{v:.4f}" for v in SHUTTER_HALF), contype="0", conaffinity="0", rgba="0.25 0.25 0.28 0.9")
    new_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjSpec.from_string(new_xml).compile()
    return replace(scene, scene=replace(scene.scene, model=model, xml=new_xml))


# --------------------------------------------------------------------------
# disturbances a protocol declares
# --------------------------------------------------------------------------


@dataclass
class Disturbance:
    """An external input applied through the step hook and logged."""

    kind: str
    log: list[dict] = field(default_factory=list)
    done: bool = False

    def apply(self, session: "TransferObjectSession", data: mujoco.MjData) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class DisplacedObject(Disturbance):
    """A push on the object while the first acquisition approaches it, so
    it is no longer where the sensor saw it when the path was planned.

    The push is horizontal, across the support toward the support's
    centre -- the object is never pushed off its support by the protocol,
    whatever the fingers then do to it -- and lasts ``duration_s``; on the
    registered cube it moves the object two to three centimetres."""

    kind: str = "displaced"
    after_s: float = 0.4
    duration_s: float = 0.03
    push_n: float = 0.2
    support: str = "bench"
    force_n: tuple[float, float, float] | None = None
    """A fixed direction instead of the one toward the support's centre."""
    _until: float | None = None
    _force: np.ndarray | None = None

    def apply(self, session, data) -> None:
        body = session.object_body
        if session.current_leaf == "acquire" and session.leaf_count["acquire"] == 1 and data.time - session.leaf_started_s >= self.after_s and not self.done:
            if self.force_n is not None:
                self._force = np.asarray(self.force_n, dtype=float)
            else:
                centre = next(np.asarray(f.position_m[:2], dtype=float) for f in session.environment.fixtures if f.name == self.support)
                offset = centre - np.asarray(data.xpos[body][:2], dtype=float)
                direction = offset / max(float(np.linalg.norm(offset)), 1e-6)
                self._force = np.array([direction[0] * self.push_n, direction[1] * self.push_n, 0.0])
            self.done = True
            self._until = float(data.time) + self.duration_s
            self.log.append({"kind": self.kind, "start_s": float(data.time), "duration_s": self.duration_s, "force_n": self._force.tolist(),
                             "object_before_m": [float(v) for v in data.xpos[body]]})
        if self.done and self._until is not None and data.time <= self._until:
            data.xfrc_applied[body, :3] = self._force
        else:
            data.xfrc_applied[body, :] = 0.0
            if self.done and self._until is not None and "object_after_m" not in self.log[-1] and data.time > self._until + 0.5:
                self.log[-1]["object_after_m"] = [float(v) for v in data.xpos[body]]


@dataclass
class InducedSlip(Disturbance):
    """The held object drawn out of the fingers at a walking pace.

    A downward pull on the object at the start of the first carry, served
    by the protocol to a target slide speed: it rises from below the grip's
    friction capacity until the object begins to slide, then holds the
    object's descent relative to the grasp point at ``slide_speed_mps``
    until the object has slid clear of the fingertips, and ends. The object
    leaves the fingers at about that speed and falls from the lift height
    onto whatever is below. The capacity is the object's authored sliding
    friction (read from the model by the protocol, which authored it)
    against twice the grip force the contact sensor reports; the pull
    never exceeds ``high`` times it.
    """

    kind: str = "slip"
    after_s: float = 0.05
    """At the start of the carry, while the object is over the support it
    came from and a lift height above it."""
    slide_speed_mps: float = 0.25
    gain_per_s: float = 400.0
    """How fast the pull changes per unit of speed error: newtons per second per metre per second."""
    low: float = 0.5
    high: float = 1.5
    clearance_m: float = 0.01
    timeout_s: float = 1.5
    cap_n: float = 40.0
    _until: float | None = None
    _capacity: float = 0.0
    _pull: float = 0.0
    _released_s: float | None = None
    _offset0: float | None = None
    _slide_m: float = 0.0
    _last_t: float | None = None

    def _slid_m(self, session, data) -> float:
        site = session.sensing.sites[session.effector.chain_id]
        centre = data.xpos[session.object_body]
        offset = float(centre[2] - data.site_xpos[site][2])
        if self._offset0 is None:
            self._offset0 = offset
        return self._offset0 - offset

    def apply(self, session, data) -> None:
        body = session.object_body
        if session.current_leaf == "transport" and session.leaf_count["transport"] == 1 and data.time - session.leaf_started_s >= self.after_s and not self.done:
            from ..contact.transfer import grasp_standoff_m

            grip = session.sensing.grip_force_n(session.effector.chain_id, float(data.time))
            friction = float(session.model.geom_friction[session.sensing.object_geom][0])
            self._capacity = 2.0 * friction * grip
            half = float(session.sensing.half[2])
            site_name = mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_SITE, session.sensing.sites[session.effector.chain_id])
            below = grasp_standoff_m(session.model, session.robot.manifest, session.effector, site_name, half) + half - 0.006
            self._slide_m = max(0.02, below + half + self.clearance_m)
            self._pull = self.low * self._capacity
            self._last_t = float(data.time)
            self.done = True
            self._until = float(data.time) + self.timeout_s
            self.log.append({"kind": self.kind, "start_s": float(data.time), "grip_force_n": grip, "object_friction": friction, "capacity_n": self._capacity,
                             "pull_from_n": self._pull, "pull_cap_n": min(self.cap_n, self.high * self._capacity), "slide_speed_mps": self.slide_speed_mps, "slide_to_clear_m": self._slide_m})
        pulling = self.done and self._until is not None and data.time <= self._until and self._released_s is None
        if pulling:
            slid = self._slid_m(session, data)
            velocity = data.qvel[session.sensing.object_dof: session.sensing.object_dof + 3]
            if slid >= self._slide_m:
                self._released_s = float(data.time)
                self.log[-1]["pulled_for_s"] = self._released_s - self.log[-1]["start_s"]
                self.log[-1]["exit_speed_mps"] = float(np.linalg.norm(velocity))
                self.log[-1]["peak_pull_n"] = round(self.log[-1].get("peak_pull_n", 0.0), 4)
                pulling = False
            else:
                site = session.sensing.sites[session.effector.chain_id]
                grasp_velocity = np.zeros(6)
                mujoco.mj_objectVelocity(session.model, data, mujoco.mjtObj.mjOBJ_SITE, site, grasp_velocity, 0)
                descent = float(grasp_velocity[5] - velocity[2])  # how fast the object drops relative to the grasp point
                dt = float(data.time) - self._last_t
                self._last_t = float(data.time)
                self._pull = float(np.clip(self._pull + self.gain_per_s * (self.slide_speed_mps - descent) * dt, 0.0, min(self.cap_n, self.high * self._capacity)))
                self.log[-1]["peak_pull_n"] = max(self.log[-1].get("peak_pull_n", 0.0), self._pull)
        elif self.done and self._released_s is None and self._until is not None and data.time > self._until:
            self._released_s = float(data.time)
            self.log[-1]["pulled_for_s"] = self.timeout_s
            self.log[-1]["exit_speed_mps"] = None
            self.log[-1]["note"] = "the object did not slide clear of the fingers within the allowance"
        if pulling:
            data.xfrc_applied[body, :3] = (0.0, 0.0, -self._pull)
        else:
            data.xfrc_applied[body, :] = 0.0


@dataclass
class TemporaryOcclusion(Disturbance):
    """The occluder moved between the front camera and the fixtures when
    the first placement verification begins, and parked again after."""

    kind: str = "occlusion"
    duration_s: float = 3.0
    _until: float | None = None

    def apply(self, session, data) -> None:
        shutter = session.shutter_id
        if shutter < 0:
            return
        if session.current_leaf == "verify_placement" and session.leaf_count["verify_placement"] == 1 and not self.done:
            self.done = True
            self._until = float(data.time) + self.duration_s
            self.log.append({"kind": self.kind, "start_s": float(data.time), "duration_s": self.duration_s, "position_m": list(SHUTTER_ACTIVE)})
        if self.done and self._until is not None and data.time <= self._until:
            data.mocap_pos[shutter] = SHUTTER_ACTIVE
        else:
            data.mocap_pos[shutter] = SHUTTER_PARKED


def disturbance_named(kind: str) -> Disturbance | None:
    if kind == "nominal":
        return None
    if kind == "displaced":
        return DisplacedObject()
    if kind == "slip":
        return InducedSlip()
    if kind == "occlusion":
        return TemporaryOcclusion()
    raise KeyError(kind)


# --------------------------------------------------------------------------
# the session: one body, one world, one record, live sensors
# --------------------------------------------------------------------------


@dataclass
class TransferObjectSession:
    robot: Any
    source: Path
    environment: EnvironmentV1
    goal: PlacementGoal
    scene: Any
    effector: EffectorV1
    frame: WorkspaceFrame
    recorder: PhysicsRecorder
    sensing: LiveSensing
    conditionals: dict[str, ConditionalV1]
    policy_sha256: str
    configuration_id: str
    disturbance: Disturbance | None = None
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    """The arm controller every leaf and hold runs; part of a certificate's context."""
    closure_config: ClosureConfig = field(default_factory=ClosureConfig)
    """How the closure advances, detects contact and squeezes; part of the same context."""
    duration_scale: float = 1.0
    """How much slower than the declared joint speeds the transfer's moving phases run."""
    state: TransferStart | None = None
    time_s: float = 0.0
    results: list[tuple[str, TransferResult]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    current_leaf: str | None = None
    leaf_started_s: float = 0.0
    leaf_count: dict[str, int] = field(default_factory=dict)
    steps: int = 0

    @classmethod
    def open(cls, zoo_id: str, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, *, configuration_name: str = "front_overhead_contact",
             disturbance: Disturbance | None = None, seed_label: str = "", controller_config: ControllerConfig | None = None,
             closure_config: ClosureConfig | None = None, duration_scale: float = 1.0) -> "TransferObjectSession":
        robot = ingest_robot(source, robot_id=zoo_id)
        effectors = tuple(robot.morphology.grasping_effectors)
        if not effectors:
            raise ValueError(f"{zoo_id} has no grasping effector")
        effector = effectors[0]
        chain = next(c for c in robot.morphology.chains if c.chain_id == effector.chain_id)
        scene = transfer_scene_from_environment(robot.manifest, robot.mjcf_xml, environment, object_name="cube", destination_fixture="platform", goal=goal, asset_root=source.parent)
        scene = with_shutter(scene, scene.scene.xml)
        frame = build_workspace_frame(scene.model, robot.morphology, chain, figure_site=figure_site_for(robot.manifest, effector.chain_id))
        sensing = LiveSensing(scene.model, robot.manifest, effectors, configuration(configuration_name, effectors), goal, seed=f"{zoo_id}|{seed_label}")
        cube = environment.objects[0]
        support = next(f for f in environment.fixtures if abs(f.position_m[0] - cube.position_m[0]) <= f.size_m[0] + 0.05 and abs(f.position_m[1] - cube.position_m[1]) <= f.size_m[1] + 0.05)
        conditionals = bind_conditionals(policy, effector.chain_id, support_top_m=float(support.position_m[2] + support.size_m[2]), half_height_m=float(cube.size_m[2]), half_extent_m=float(max(cube.size_m)))
        return cls(robot=robot, source=source, environment=environment, goal=goal, scene=scene, effector=effector, frame=frame, recorder=PhysicsRecorder(scene.model),
                   sensing=sensing, conditionals=conditionals, policy_sha256=policy_digest(policy), configuration_id=configuration_name, disturbance=disturbance,
                   controller_config=controller_config or ControllerConfig(), closure_config=closure_config or ClosureConfig(), duration_scale=float(duration_scale))

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene.model

    @property
    def effectors(self) -> dict[str, EffectorV1]:
        return {self.effector.chain_id: self.effector}

    @property
    def object_body(self) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scene_block")

    @property
    def shutter_id(self) -> int:
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHUTTER_BODY)
        return int(self.model.body_mocapid[body]) if body >= 0 else -1

    def current_qpos(self) -> np.ndarray:
        if self.state is not None:
            return np.array(self.state.qpos, dtype=float)
        from ..contact.grasp import _scene_rest_qpos

        return np.array(_scene_rest_qpos(self.model, self.robot.manifest), dtype=float)

    def hook(self, data: mujoco.MjData) -> None:
        """After every physics step: the sensors sample, the protocol's disturbance applies."""

        self.steps += 1
        if self.disturbance is not None:
            self.disturbance.apply(self, data)
        self.sensing.observe(data)

    def begin_leaf(self, name: str) -> None:
        self.current_leaf = name
        self.leaf_count[name] = self.leaf_count.get(name, 0) + 1
        self.leaf_started_s = self.time_s

    def adopt(self, start: TransferStart | None) -> None:
        if start is not None:
            self.state = start
            self.time_s = float(start.time_s)

    def reroute(self, *, holding: bool) -> dict:
        """A guarded joint move to the reference configuration from where the arm stands."""

        from ..contact.grasp import _scene_rest_qpos

        if self.state is None:
            return {"executed": False, "refusal": "nothing_to_reroute"}
        model, manifest = self.model, self.robot.manifest
        arm = arm_joint_names(model, self.effector, self.frame)
        rest = np.asarray(_scene_rest_qpos(model, manifest), dtype=float)
        by_joint = {dof.joint: dof for dof in manifest.dofs}
        targets = {by_joint[n].name: float(rest[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])]) for n in arm}
        executed = joint_move(model, manifest, self.effector, arm, self.state, self.recorder, targets, _collision_guard(manifest, model), holding=holding, on_step=self.hook,
                              controller_config=self.controller_config, closure_config=self.closure_config)
        if executed.executed:
            self.adopt(executed.start)
        return {"executed": executed.executed, "refusal": executed.refusal, "physics_s": executed.physics_s, "joint_travel_rad": executed.joint_travel_rad, "detail": executed.detail}

    def hold_still(self, duration_s: float, *, holding: bool, stop_when=None, should_stop=None) -> float:
        """Hold the arm where it is for ``duration_s`` of physics, the closure
        squeezing if ``holding``, sensors sampling; returns the physics time spent."""

        if self.state is None:
            from ..contact.grasp import _scene_rest_qpos

            rest = np.asarray(_scene_rest_qpos(self.model, self.robot.manifest), dtype=float)
            data = mujoco.MjData(self.model)
            data.qpos[:] = rest
            mujoco.mj_forward(self.model, data)
            resume = TransferStart(qpos=rest, qvel=np.zeros(self.model.nv), time_s=0.0, state=None)
        else:
            resume = self.state
        arm = arm_joint_names(self.model, self.effector, self.frame)
        arm_adr = np.array([int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm])
        q0 = np.array(resume.qpos, dtype=float)[arm_adr]
        zero = np.zeros_like(q0)

        def planned(_elapsed: float):
            return q0, zero, zero

        executed = track(self.model, self.robot.manifest, self.effector, arm, resume, self.recorder, planned, duration_s, holding=holding,
                         stop_when=stop_when, should_stop=should_stop, on_step=self.hook, controller_config=self.controller_config, closure_config=self.closure_config)
        self.adopt(executed.start)
        return executed.physics_s

    def decide(self, name: str) -> Any:
        return decide_live(self.conditionals[name], self.sensing, self.time_s)


# --------------------------------------------------------------------------
# the runtime: leaves
# --------------------------------------------------------------------------


@dataclass
class TransferObjectRuntime:
    session: TransferObjectSession
    interrupt: Interrupt
    calls: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    def _stopper(self, context: LeafContext, monitor: bool = False):
        session = self.session
        last = {"at": -1.0, "reason": ""}

        def should_stop(now: float) -> bool:
            if context.should_stop():
                return True
            if now > context.deadline_s:
                self.interrupt.request(f"episode cap reached at {now:.2f} s")
                return True
            if monitor and now - last["at"] >= MONITOR_PERIOD_S:
                last["at"] = now
                verdict = decide_live(session.conditionals["held"], session.sensing, now)
                if verdict.decision is Decision.FAIL:
                    last["reason"] = "hold_lost"
                    session.events.append({"time_s": now, "event": "hold_lost", "detail": {k: round(v, 4) for k, v in verdict.detail.items()}})
                    return True
            return False

        return should_stop, last

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        node = context.node
        session = self.session
        leaf = node.skill_id
        session.begin_leaf(leaf)
        stopper, last = self._stopper(context, monitor=(leaf == "transport"))
        believed = context.belief.get("pose:cube")
        object_position = None if believed is None else np.asarray(believed, dtype=float)
        result = attempt_transfer(session.robot.manifest, session.scene, session.effector, session.frame, recorder=session.recorder, resume=session.state,
                                  should_stop=stopper, phase_range=PHASES_OF[leaf], object_position_m=object_position, on_step=session.hook, controller_config=session.controller_config,
                                  closure_config=session.closure_config, duration_scale=session.duration_scale)
        reroute = None
        if leaf == "acquire" and not result.executed and result.failed_gate in REROUTE_ON and session.state is not None:
            reroute = {"refusal": result.failed_gate, **session.reroute(holding=False)}
            session.events.append({"time_s": session.time_s, "event": "reroute", "detail": reroute})
            if reroute["executed"]:
                result = attempt_transfer(session.robot.manifest, session.scene, session.effector, session.frame, recorder=session.recorder, resume=session.state,
                                          should_stop=stopper, phase_range=PHASES_OF[leaf], object_position_m=object_position, on_step=session.hook, controller_config=session.controller_config,
                                          closure_config=session.closure_config, duration_scale=session.duration_scale)
        session.results.append((node.node_id, result))
        if result.executed:
            session.adopt(result.continuation())
        facts: dict[str, Any] = {}
        if last["reason"] == "hold_lost":
            verdict, reason = Verdict.FAILURE, "hold_lost"
            facts["held:cube"] = False
        elif result.interrupted:
            verdict, reason = Verdict.INTERRUPTED, self.interrupt.reason or "interrupted"
        elif result.certified:
            verdict, reason = Verdict.SUCCESS, ""
        else:
            verdict, reason = Verdict.FAILURE, result.failed_gate or "failed"
        if leaf in ("transport", "release") and verdict is not Verdict.SUCCESS:
            facts["held:cube"] = False
        if leaf == "release" and verdict is Verdict.SUCCESS:
            facts["held:cube"] = False
        self.calls.append({"node": node.node_id, "leaf": leaf, "attempt": session.leaf_count[leaf], "executed": result.executed, "certified": result.certified, "gate": result.failed_gate,
                           "phases": [p.name for p in result.phases], "physics_time_s": [float(result.times_s[0]), float(result.times_s[-1])] if result.executed else None,
                           "verdict": verdict.value, "reason": reason, "planned_to": None if object_position is None else [float(v) for v in object_position], "reroute": reroute,
                           "detail": result.violations[0].detail[:200] if result.violations else ""})
        evidence = {"leaf": leaf, "attempt": session.leaf_count[leaf], "executed": result.executed, "failed_gate": result.failed_gate, "phases": [p.name for p in result.phases],
                    "path_seed": result.path_seed, "planned_to_sensed_position": None if object_position is None else [float(v) for v in object_position], "reroute": reroute}
        return LeafOutcome(verdict, reason, facts=facts, evidence=evidence)

    def _observe_until(self, context: LeafContext, name: str, *, holding: bool, dwell_s: float, budget: int) -> tuple[Any, list[dict]]:
        """Hold still for the window, decide; re-observe within the budget while undecidable."""

        session = self.session
        stopper, _ = self._stopper(context)
        trail = []
        verdict = None
        for attempt in range(budget + 1):
            if attempt == 0:
                session.hold_still(dwell_s, holding=holding, should_stop=stopper)
            else:
                session.hold_still(OBSERVE_WAIT_S, holding=holding, should_stop=stopper)
            verdict = session.decide(name)
            trail.append({"time_s": session.time_s, "conditional": name, "decision": verdict.decision.value, "reason": verdict.reason, "sensors": list(verdict.sensors_used),
                          "detail": {k: round(v, 4) for k, v in verdict.detail.items()}, "re_observation": attempt, "cameras": session.sensing.camera_state(session.time_s)})
            self.verdicts.append(trail[-1])
            if verdict.decision is not Decision.UNKNOWN or context.should_stop() or self.interrupt.requested:
                break
        return verdict, trail

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        node = context.node
        session = self.session
        leaf = node.skill_id
        session.begin_leaf(leaf)
        holding = bool(context.belief.get("held:cube", False))
        if leaf == "observe_object":
            budget = session.conditionals["reachable"].fallback.budget
            stopper, _ = self._stopper(context)
            max_age = session.conditionals["reachable"].window.max_age_s
            for attempt in range(budget + 1):
                # Every observation spends physics time with the arm still
                # before it reads: a reading taken without advancing the
                # world would see the samples the last reading saw.
                session.hold_still(OBSERVE_WAIT_S if attempt > 0 else STILL_WINDOW_S + 0.1, holding=holding, should_stop=stopper)
                position = session.sensing.last_object_position(session.time_s, max_age_s=max_age)
                drift = session.sensing.object_drift_mps(session.time_s, STILL_WINDOW_S)
                if position is not None and (drift is None or drift > STILL_SPEED_MPS):
                    # Seen, but moving: a falling or rolling object is not a target.
                    self.verdicts.append({"time_s": session.time_s, "conditional": "object_still", "decision": "unknown", "reason": f"drift:{drift if drift is None else round(drift, 4)}",
                                          "sensors": list(session.sensing.cameras), "detail": {}, "re_observation": attempt, "cameras": session.sensing.camera_state(session.time_s)})
                    position = None
                if position is not None:
                    # Seen still: the plan goes to the mean of the frames that
                    # showed it still, not to the newest frame's noise. A single
                    # frame at two millimetres a side missed the cube's centre
                    # by enough to pinch it off-centre once in twenty.
                    averaged = session.sensing.mean_object_position(session.time_s, STILL_WINDOW_S)
                    if averaged is not None:
                        position = averaged
                reach = session.decide("reachable")
                self.verdicts.append({"time_s": session.time_s, "conditional": "reachable", "decision": reach.decision.value, "reason": reach.reason, "sensors": list(reach.sensors_used),
                                      "detail": {k: round(v, 4) for k, v in reach.detail.items()}, "re_observation": attempt, "cameras": session.sensing.camera_state(session.time_s)})
                if position is not None:
                    facts = {"known:cube": True, "pose:cube": [float(v) for v in position], "reach:cube": reach.decision is Decision.PASS, "reach_verdict:cube": reach.decision.value}
                    self.calls.append({"node": node.node_id, "leaf": leaf, "attempt": session.leaf_count[leaf], "re_observations": attempt, "observed": facts})
                    return facts
                if context.should_stop() or self.interrupt.requested:
                    break
            self.calls.append({"node": node.node_id, "leaf": leaf, "attempt": session.leaf_count[leaf], "re_observations": budget, "observed": None, "reason": "object_not_seen"})
            return None
        if leaf == "verify_hold":
            conditional = session.conditionals["held"]
            verdict, trail = self._observe_until(context, "held", holding=True, dwell_s=conditional.window.duration_s, budget=conditional.fallback.budget)
            self.calls.append({"node": node.node_id, "leaf": leaf, "attempt": session.leaf_count[leaf], "trail": trail})
            if verdict.decision is Decision.UNKNOWN:
                return None
            return {"held:cube": verdict.decision is Decision.PASS}
        if leaf == "verify_placement":
            conditional = session.conditionals["stably_placed"]
            verdict, trail = self._observe_until(context, "stably_placed", holding=False, dwell_s=conditional.window.duration_s, budget=conditional.fallback.budget)
            self.calls.append({"node": node.node_id, "leaf": leaf, "attempt": session.leaf_count[leaf], "trail": trail})
            if verdict.decision is Decision.UNKNOWN:
                return None
            return {"placed:cube:platform": verdict.decision is Decision.PASS, "held:cube": False}
        return None


def predicates_for_transfer():
    """Predicates as functions of the belief; a fact never established is
    false, never unknown: a loop asking whether the object is held before
    any verification has run is asking whether that was established."""

    return {
        "object_known": lambda belief, args: bool(belief.get(f"known:{args[0]}", False)),
        "object_held": lambda belief, args: bool(belief.get(f"held:{args[0]}", False)),
        "object_placed": lambda belief, args: bool(belief.get(f"placed:{args[0]}:{args[1]}", False)),
        "object_in_reach": lambda belief, args: bool(belief.get(f"reach:{args[0]}", False)),
    }
