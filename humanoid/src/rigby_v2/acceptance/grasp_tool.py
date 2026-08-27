"""Measured grasp/place and hand-tool acceptance.

This module deliberately separates an attempted controller demonstration from
acceptance.  A completed MuJoCo rollout is not a successful task.  Acceptance
requires the authored contact lifecycle, the selected pack state predicate,
the pack-specific terminal outcome, the existing global physics limits, three
identical repeats, and a self-contained MJZ reimport.

The current canonical-human controller cannot yet pass these probes.  The
returned reports preserve the measured blocker instead of turning a lift (or
merely a finite trace) into a false positive.
"""

from __future__ import annotations

import hashlib
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np

from rigby_v2.certification import CertificationPolicy
from rigby_v2.scenes import compile_scene
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    canonical_standing_config,
    compare_replays,
)


class PackTask(StrEnum):
    GRASP_PLACE_BLOCK = "grasp_place_block"
    HAND_TOOL = "hand_tool"


class AcceptanceGate(StrEnum):
    STRUCTURE = "structure"
    EXPORT_REIMPORT = "export_reimport"
    THREE_EXACT_REPEATS = "three_exact_repeats"
    GLOBAL_PHYSICS = "global_physics"
    SELECTED_STATE = "selected_state"
    CONTACT_LIFECYCLE = "contact_lifecycle"
    GRASP = "grasp"
    LIFT = "lift"
    CARRY = "carry"
    PLACE_AND_RELEASE = "place_and_release"
    TOOL_OUTCOME = "tool_outcome"


@dataclass(frozen=True, slots=True)
class AcceptanceMeasurement:
    gate: AcceptanceGate
    passed: bool
    observed: float | int | str
    required: float | int | str


@dataclass(frozen=True, slots=True)
class AcceptanceAttempt:
    task: PackTask
    request: SimulationRequest
    model_sha256: str
    object_joint: str
    object_geom: str
    support_geom: str
    selected_site: str
    selected_rise_m: float


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    task: PackTask
    accepted: bool
    model_sha256: str
    measurements: tuple[AcceptanceMeasurement, ...]
    runs: tuple[SimulationResult, ...]

    @property
    def failures(self) -> tuple[AcceptanceMeasurement, ...]:
        return tuple(item for item in self.measurements if not item.passed)


_ARM_NAMES = (
    "left_shoulder_flex",
    "left_shoulder_abduct",
    "left_shoulder_twist",
    "left_elbow_flex",
    "left_forearm_twist",
    "left_wrist_flex",
    "left_wrist_deviation",
)

# Audited arm-only IK solutions for the side workbench.  Torso and leg targets
# remain canonical-neutral; the free pelvis and objects remain unactuated.
_APPROACH = (
    -0.4942103127,
    -0.1280156188,
    -0.2838194005,
    1.503349366,
    -0.0405197044,
    -0.9069555931,
    0.2776934159,
)
_PREGRASP = (
    -0.1420546344,
    -0.0226850261,
    -0.632944444,
    1.089199361,
    -0.1022084878,
    -0.8712133122,
    0.5519707694,
)
_LIFT = (
    -0.5423975356,
    -0.2630177767,
    -0.5561886949,
    1.282072978,
    -0.1013321381,
    -0.6004741843,
    0.5467875827,
)
_CARRY = (
    -0.8212630922,
    -0.4994868211,
    -0.5372536235,
    2.00644542,
    -0.1717150536,
    -0.9859270831,
    0.5676081415,
)
_LOWER = (
    -0.3725160516,
    -0.0937430265,
    -0.8453590988,
    1.8287875233,
    -0.9772000505,
    -1.308996939,
    -0.133008948,
)
_RETREAT = (
    -0.661041486,
    0.1816078982,
    -0.4757712001,
    1.6668231045,
    -0.7109514065,
    -0.8954523432,
    -0.2997186053,
)
_EFFECTOR_GEOMS = frozenset(
    {
        "left_palm_collision",
        *(
            f"left_{finger}_{segment}_collision"
            for finger in ("thumb", "index", "middle", "ring", "little")
            for segment in ("proximal", "middle", "distal")
        ),
    }
)


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"required joint {name!r} is missing")
    return int(model.jnt_qposadr[joint_id])


def _set_joint(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    name: str,
    value: float,
) -> None:
    qpos[_joint_qpos_address(model, name)] = value


def _close_left_hand(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    closure_scale: float,
) -> None:
    for finger in ("index", "middle", "ring", "little"):
        for segment, degrees in (("mcp", 70), ("pip", 85), ("dip", 55)):
            _set_joint(
                model,
                qpos,
                f"left_{finger}_{segment}",
                closure_scale * np.deg2rad(degrees),
            )
    for name, degrees in (
        ("thumb_opposition", 35),
        ("thumb_mcp", 50),
        ("thumb_pip", 55),
        ("thumb_dip", 40),
    ):
        _set_joint(
            model,
            qpos,
            f"left_{name}",
            closure_scale * np.deg2rad(degrees),
        )


def _pose(
    model: mujoco.MjModel,
    initial: np.ndarray,
    arm: tuple[float, ...],
    *,
    closure_scale: float = 0.0,
) -> np.ndarray:
    value = initial.copy()
    for name, target in zip(_ARM_NAMES, arm, strict=True):
        _set_joint(model, value, name, target)
    if closure_scale:
        _close_left_hand(model, value, closure_scale)
    return value


def _calibrated_mjz(task: PackTask) -> tuple[bytes, str]:
    """Return the pack's self-contained audited workbench artifact."""

    scene = compile_scene(task.value)
    root = ET.fromstring(scene.xml)
    xml = ET.tostring(root, encoding="unicode")
    spec = mujoco.MjSpec.from_string(xml, assets=scene.assets)
    spec.compile()
    stream = io.BytesIO()
    spec.to_zip(stream)
    payload = stream.getvalue()
    # The artifact, rather than only the in-memory XML, is authoritative.
    mujoco.MjSpec.from_zip(io.BytesIO(payload)).compile()
    return payload, hashlib.sha256(payload).hexdigest()


def build_attempt(task: PackTask | str) -> AcceptanceAttempt:
    selected = PackTask(task)
    mjz, digest = _calibrated_mjz(selected)
    model = mujoco.MjSpec.from_zip(io.BytesIO(mjz)).compile()
    # The object remains at the authored pack pose and is never actuated.  Only
    # the arm/finger set-points are changed from canonical rest at t=0.
    initial = _pose(model, model.qpos0.copy(), _APPROACH)

    if selected is PackTask.GRASP_PLACE_BLOCK:
        object_joint = "obj__block__free"
        object_geom = "obj__block__body__box"
        support_geom = "support__work_table"
        selected_site = "obj__block__bottom_center"
        selected_rise = 0.12
    else:
        object_joint = "obj__hammer__free"
        object_geom = "obj__hammer__body__handle"
        support_geom = "support__tool_bench"
        selected_site = "obj__hammer__handle_grasp"
        selected_rise = 0.15

    # A deliberately slow, collision-free approach precedes partial closure.
    # Contact-onset sweeps select task-specific partial closure targets.  Full
    # kinematic closure crushes the objects and is not a physically valid
    # target; the long holds account for the canonical finger actuator rise
    # time without increasing gains or bypassing dynamics.
    if selected is PackTask.GRASP_PLACE_BLOCK:
        closure = 0.40
        times = np.asarray((0.0, 0.50, 1.50, 2.50, 4.50, 5.50, 6.20, 7.20, 8.20, 8.90, 9.70))
        keyframes = np.vstack(
            (
                _pose(model, initial, _APPROACH),
                _pose(model, initial, _APPROACH),
                _pose(model, initial, _PREGRASP),
                _pose(model, initial, _PREGRASP, closure_scale=closure),
                _pose(model, initial, _PREGRASP, closure_scale=closure),
                _pose(model, initial, _LIFT, closure_scale=closure),
                _pose(model, initial, _LIFT, closure_scale=closure),
                _pose(model, initial, _CARRY, closure_scale=closure),
                _pose(model, initial, _LOWER, closure_scale=closure),
                _pose(model, initial, _LOWER),
                _pose(model, initial, _RETREAT),
            )
        )
    else:
        # The hammer starts flat on the bench.  After lift/hold the same
        # pregrasp pose supplies the downward tool-head strike while closure is
        # maintained; opening happens only after the measured outcome window.
        closure = 0.75
        times = np.asarray((0.0, 0.50, 1.50, 2.50, 6.50, 7.50, 8.20, 9.20, 9.80, 10.50))
        keyframes = np.vstack(
            (
                _pose(model, initial, _APPROACH),
                _pose(model, initial, _APPROACH),
                _pose(model, initial, _PREGRASP),
                _pose(model, initial, _PREGRASP, closure_scale=closure),
                _pose(model, initial, _PREGRASP, closure_scale=closure),
                _pose(model, initial, _LIFT, closure_scale=closure),
                _pose(model, initial, _LIFT, closure_scale=closure),
                _pose(model, initial, _PREGRASP, closure_scale=closure),
                _pose(model, initial, _PREGRASP),
                _pose(model, initial, _RETREAT),
            )
        )
    trajectory = LinearKeyframeTrajectory(
        times_s=times,
        qpos=keyframes,
        qvel=np.zeros((len(times), model.nv), dtype=np.float64),
    )
    request = SimulationRequest(
        model_xml=None,
        model_mjz=mjz,
        trajectory=trajectory,
        config=SimulationConfig(
            duration_s=float(times[-1]),
            standing=canonical_standing_config(model),
            free_root_joint_name="pelvis_free",
        ),
        initial_qpos=initial,
        request_id=f"acceptance-{selected.value}",
    )
    return AcceptanceAttempt(
        task=selected,
        request=request,
        model_sha256=digest,
        object_joint=object_joint,
        object_geom=object_geom,
        support_geom=support_geom,
        selected_site=selected_site,
        selected_rise_m=selected_rise,
    )


def _pair(contact, first: frozenset[str], second: frozenset[str]) -> bool:
    names = {contact.geom1_name, contact.geom2_name}
    return bool(names & first) and bool(names & second)


def _object_site_positions(
    model: mujoco.MjModel,
    result: SimulationResult,
    site_name: str,
) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"required site {site_name!r} is missing")
    data = mujoco.MjData(model)
    positions = np.empty((len(result.trace.qpos), 3), dtype=np.float64)
    for index, qpos in enumerate(result.trace.qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[site_id]
    return positions


def _measure_run(
    attempt: AcceptanceAttempt,
    model: mujoco.MjModel,
    result: SimulationResult,
) -> tuple[AcceptanceMeasurement, ...]:
    if result.status is not SimulationStatus.COMPLETED or result.metrics is None:
        return (
            AcceptanceMeasurement(
                AcceptanceGate.GLOBAL_PHYSICS,
                False,
                result.failure.message if result.failure else result.status.value,
                "completed finite MuJoCo trace",
            ),
        )
    policy = CertificationPolicy()
    metrics = result.metrics
    nonfoot_ground = 0
    for frame in result.trace.contacts:
        for contact in frame.contacts:
            if (
                "ground" in {contact.geom1_name, contact.geom2_name}
                and not {
                    "left_foot_collision",
                    "right_foot_collision",
                }
                & {contact.geom1_name, contact.geom2_name}
                and contact.normal_force_n >= policy.min_contact_force_n
            ):
                nonfoot_ground += 1
    global_values = {
        "penetration_m": metrics.max_penetration_m,
        "speed": metrics.max_speed_rad_or_m_s,
        "root_drift_m": metrics.root_translation_drift_m,
        "nonfoot_ground_contacts": nonfoot_ground,
    }
    global_pass = (
        metrics.finite
        and metrics.max_penetration_m <= policy.max_penetration_m
        and metrics.max_speed_rad_or_m_s <= policy.default_joint_velocity_limit
        and metrics.root_translation_drift_m <= policy.max_pelvis_drift_m
        and nonfoot_ground == 0
        and result.diagnostics.get("qpos_writes_after_initialization") == 0
    )

    site_positions = _object_site_positions(model, result, attempt.selected_site)
    rise = float(np.max(site_positions[:, 2] - site_positions[0, 2]))
    xy = site_positions[:, :2]
    carry = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    object_names = frozenset({attempt.object_geom})
    support_names = frozenset({attempt.support_geom})
    contact_counts: list[int] = []
    effector_contact: list[bool] = []
    support_contact: list[bool] = []
    head_strike_force: list[float] = []
    for frame in result.trace.contacts:
        effectors: set[str] = set()
        support = False
        strike_force = 0.0
        for contact in frame.contacts:
            if _pair(contact, object_names, _EFFECTOR_GEOMS) and contact.normal_force_n >= 0.1:
                names = {contact.geom1_name, contact.geom2_name}
                effectors.update(names & _EFFECTOR_GEOMS)
            if _pair(contact, object_names, support_names) and contact.normal_force_n >= 0.05:
                support = True
            if attempt.task is PackTask.HAND_TOOL and _pair(
                contact,
                frozenset({"obj__hammer__body__head"}),
                support_names,
            ):
                strike_force = max(strike_force, contact.normal_force_n)
        contact_counts.append(len(effectors))
        effector_contact.append(bool(effectors))
        support_contact.append(support)
        head_strike_force.append(strike_force)

    times = result.trace.times_s
    if attempt.task is PackTask.GRASP_PLACE_BLOCK:
        grasp_window = (times >= 1.45) & (times <= 5.50)
        carry_window = (times >= 4.50) & (times <= 8.20)
        release_window = times >= 8.95
        final_support = times >= 8.95
        latest_first_contact = 5.50
    else:
        grasp_window = (times >= 1.45) & (times <= 7.50)
        carry_window = (times >= 6.50) & (times <= 9.20)
        release_window = times >= 9.85
        final_support = times >= 9.85
        latest_first_contact = 7.50
    pregrasp_window = times < 1.45
    counts = np.asarray(contact_counts)
    touching = np.asarray(effector_contact)
    supported = np.asarray(support_contact)
    max_effectors = int(np.max(counts[grasp_window])) if np.any(grasp_window) else 0
    first_contact = (
        float(times[np.flatnonzero(touching)[0]])
        if np.any(touching)
        else float("inf")
    )
    lifecycle_pass = (
        not np.any(touching[pregrasp_window])
        and 1.45 <= first_contact <= latest_first_contact
        and max_effectors >= 2
        and float(np.mean(touching[carry_window])) >= 0.90
        and not np.any(touching[release_window])
    )

    measurements = [
        AcceptanceMeasurement(
            AcceptanceGate.GLOBAL_PHYSICS,
            global_pass,
            str(global_values),
            (
                f"penetration<={policy.max_penetration_m}, speed<="
                f"{policy.default_joint_velocity_limit}, root_drift<="
                f"{policy.max_pelvis_drift_m}, no fall contacts/no qpos writes"
            ),
        ),
        AcceptanceMeasurement(
            AcceptanceGate.SELECTED_STATE,
            rise >= attempt.selected_rise_m,
            rise,
            attempt.selected_rise_m,
        ),
        AcceptanceMeasurement(AcceptanceGate.GRASP, max_effectors >= 2, max_effectors, 2),
        AcceptanceMeasurement(
            AcceptanceGate.CONTACT_LIFECYCLE,
            lifecycle_pass,
            f"first={first_contact:.6g}, carry_coverage={np.mean(touching[carry_window]):.6g}",
            "no early contact; 2-effectors; >=90% carry contact; released",
        ),
        AcceptanceMeasurement(
            AcceptanceGate.LIFT,
            rise >= attempt.selected_rise_m,
            rise,
            attempt.selected_rise_m,
        ),
    ]
    if attempt.task is PackTask.GRASP_PLACE_BLOCK:
        measurements.append(
            AcceptanceMeasurement(AcceptanceGate.CARRY, carry >= 0.10, carry, 0.10)
        )
        placed = (
            carry >= 0.10
            and bool(np.any(supported[final_support]))
            and not bool(np.any(touching[release_window]))
            and abs(float(site_positions[-1, 2] - site_positions[0, 2])) <= 0.04
        )
        measurements.append(
            AcceptanceMeasurement(
                AcceptanceGate.PLACE_AND_RELEASE,
                placed,
                (
                    f"final_support={bool(np.any(supported[final_support]))}, "
                    f"final_dz={site_positions[-1, 2] - site_positions[0, 2]:.6g}"
                ),
                "carried >=0.10 m, support contact, final dz<=0.04 m, no hand contact",
            )
        )
    else:
        strike_force = float(np.max(np.asarray(head_strike_force)[times >= 8.20]))
        strike_while_grasped = any(
            force >= 10.0 and touching[index]
            for index, force in enumerate(head_strike_force)
            if times[index] >= 8.20
        )
        measurements.append(
            AcceptanceMeasurement(
                AcceptanceGate.TOOL_OUTCOME,
                strike_while_grasped,
                strike_force,
                ">=10 N head/support strike while measured grasp remains active",
            )
        )
    return tuple(measurements)


def _structure_measurement(model: mujoco.MjModel) -> AcceptanceMeasurement:
    welds = sum(model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD for index in range(model.neq))
    object_actuators = 0
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        object_actuators += int(name.startswith("obj__"))
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    free_root = root_id >= 0 and model.jnt_type[root_id] == mujoco.mjtJoint.mjJNT_FREE
    passed = bool(free_root and not model.nmocap and not welds and not object_actuators)
    return AcceptanceMeasurement(
        AcceptanceGate.STRUCTURE,
        passed,
        f"free_root={free_root}, mocap={model.nmocap}, welds={welds}, object_actuators={object_actuators}",
        "free pelvis; zero mocap/weld/object actuators",
    )


def evaluate_attempt(
    attempt: AcceptanceAttempt,
    runs: tuple[SimulationResult, ...],
) -> AcceptanceReport:
    if len(runs) != 3:
        raise ValueError("pack acceptance requires exactly three repeats")
    assert attempt.request.model_mjz is not None
    try:
        model = mujoco.MjSpec.from_zip(io.BytesIO(attempt.request.model_mjz)).compile()
        reimport_ok = True
        reimport_observed = f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
    except (ValueError, RuntimeError, mujoco.FatalError) as error:
        reimport_ok = False
        reimport_observed = str(error)
        model = mujoco.MjModel.from_xml_string("<mujoco/>")
    comparisons = tuple(compare_replays(runs[0], item) for item in runs[1:])
    repeat_ok = all(item.matches for item in comparisons)
    measurements = [
        _structure_measurement(model),
        AcceptanceMeasurement(
            AcceptanceGate.EXPORT_REIMPORT,
            reimport_ok,
            reimport_observed,
            "self-contained MJZ compiles after reimport",
        ),
        AcceptanceMeasurement(
            AcceptanceGate.THREE_EXACT_REPEATS,
            repeat_ok,
            ", ".join(item.reason or "exact" for item in comparisons),
            "two exact comparisons against the first of three runs",
        ),
    ]
    if reimport_ok:
        per_run = tuple(_measure_run(attempt, model, run) for run in runs)
        gate_order = tuple(item.gate for item in per_run[0])
        for index, gate in enumerate(gate_order):
            values = tuple(items[index] for items in per_run)
            measurements.append(
                AcceptanceMeasurement(
                    gate,
                    all(item.passed for item in values),
                    " | ".join(str(item.observed) for item in values),
                    values[0].required,
                )
            )
    result = tuple(measurements)
    return AcceptanceReport(
        task=attempt.task,
        accepted=all(item.passed for item in result),
        model_sha256=attempt.model_sha256,
        measurements=result,
        runs=runs,
    )


def run_acceptance(
    task: PackTask | str,
    *,
    runtime: NativeMujocoRuntime | None = None,
) -> AcceptanceReport:
    attempt = build_attempt(task)
    simulator = runtime or NativeMujocoRuntime()
    runs = tuple(simulator.simulate(attempt.request) for _ in range(3))
    return evaluate_attempt(attempt, runs)
