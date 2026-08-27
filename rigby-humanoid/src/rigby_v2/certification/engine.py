"""Deterministic certification gates over authoritative MuJoCo traces."""

from __future__ import annotations

from math import acos, degrees

import mujoco
import numpy as np

from rigby_v2.simulation.replay import compare_replays
from rigby_v2.simulation.runtime import (
    NativeMujocoRuntime,
    SimulationResult,
    SimulationStatus,
    load_model_source,
    model_source_hash,
)

from .models import (
    CandidateCertificationRequest,
    CandidateSelectionResult,
    CertificationOutcome,
    CertificationPolicy,
    CertificationResult,
    ContactPair,
    ExportValidation,
    GateCode,
    GateViolation,
)
from .predicates import (
    KinematicTrace,
    ObjectStateContext,
    PredicateEvaluation,
    PredicateSupportError,
)


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"{object_type.name}_{index}"


def _kinematics(model: mujoco.MjModel, simulation: SimulationResult) -> KinematicTrace:
    if simulation.trace.qpos.ndim != 2 or simulation.trace.qpos.shape[1] != model.nq:
        raise ValueError("trace qpos shape does not match model")
    data = mujoco.MjData(model)
    body_names = tuple(_name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody))
    body_positions = {name: [] for name in body_names}
    body_rotations = {name: [] for name in body_names}
    subtree_com = {name: [] for name in body_names}
    joint_names = tuple(_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt))
    joint_positions = {name: [] for name in joint_names}
    site_names = tuple(_name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite))
    site_positions = {name: [] for name in site_names}
    for qpos in simulation.trace.qpos:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for body_id, name in enumerate(body_names, start=1):
            body_positions[name].append(data.xpos[body_id].copy())
            body_rotations[name].append(data.xmat[body_id].reshape(3, 3).copy())
            subtree_com[name].append(data.subtree_com[body_id].copy())
        for joint_id, name in enumerate(joint_names):
            adr = int(model.jnt_qposadr[joint_id])
            joint_positions[name].append(float(data.qpos[adr]))
        for site_id, name in enumerate(site_names):
            site_positions[name].append(data.site_xpos[site_id].copy())
    return KinematicTrace(
        times_s=simulation.trace.times_s.copy(),
        body_positions={key: np.asarray(value) for key, value in body_positions.items()},
        body_rotations={key: np.asarray(value) for key, value in body_rotations.items()},
        joint_positions={key: np.asarray(value) for key, value in joint_positions.items()},
        site_positions={key: np.asarray(value) for key, value in site_positions.items()},
        subtree_com={key: np.asarray(value) for key, value in subtree_com.items()},
    )


def _limit_violation(
    code: GateCode,
    label: str,
    actual: float,
    limit: float,
) -> GateViolation | None:
    if actual <= limit:
        return None
    return GateViolation(code, f"{label} exceeded its certified limit", actual, limit)


def _structural_violations(model: mujoco.MjModel, simulation: SimulationResult) -> list[GateViolation]:
    violations: list[GateViolation] = []
    if model.nmocap:
        violations.append(
            GateViolation(GateCode.MOCAP_BODY, "mocap bodies are forbidden in certified simulation", model.nmocap, 0)
        )
    weld_count = sum(
        model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD for index in range(model.neq)
    )
    if weld_count:
        violations.append(
            GateViolation(GateCode.HIDDEN_WELD, "weld equalities are forbidden", weld_count, 0)
        )
    writes = int(simulation.diagnostics.get("qpos_writes_after_initialization", -1))
    if writes != 0:
        violations.append(
            GateViolation(GateCode.TELEPORT, "runtime did not prove initialization-only state writes", writes, 0)
        )
    return violations


def _continuity_violations(
    model: mujoco.MjModel,
    simulation: SimulationResult,
    tolerance: float,
) -> list[GateViolation]:
    trace = simulation.trace
    if len(trace.times_s) < 2:
        return [GateViolation(GateCode.TRACE_MISSING, "trace has fewer than two samples")]
    violations: list[GateViolation] = []
    dt = np.diff(trace.times_s)
    if np.any(dt <= 0):
        return [GateViolation(GateCode.TELEPORT, "trace time is not strictly increasing")]
    worst = 0.0
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            delta = np.diff(trace.qpos[:, qpos_adr])
            integrated = trace.qvel[1:, dof_adr] * dt
            worst = max(worst, float(np.max(np.abs(delta - integrated))))
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            delta = np.diff(trace.qpos[:, qpos_adr : qpos_adr + 3], axis=0)
            integrated = trace.qvel[1:, dof_adr : dof_adr + 3] * dt[:, None]
            worst = max(worst, float(np.max(np.abs(delta - integrated))))
    if worst > tolerance:
        violations.append(
            GateViolation(GateCode.TELEPORT, "state discontinuity is inconsistent with integrated velocity", worst, tolerance)
        )
    return violations


def _rotation_violations(
    model: mujoco.MjModel,
    simulation: SimulationResult,
    tolerance: float = 1e-5,
) -> list[GateViolation]:
    """Require every authoritative free/ball quaternion to remain unit length."""

    violations: list[GateViolation] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr += 3
        elif joint_type != mujoco.mjtJoint.mjJNT_BALL:
            continue
        norms = np.linalg.norm(
            simulation.trace.qpos[:, qpos_adr : qpos_adr + 4], axis=1
        )
        error = float(np.max(np.abs(norms - 1.0)))
        if error > tolerance:
            name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append(
                GateViolation(
                    GateCode.ROTATION_NORMALIZATION,
                    f"joint {name} quaternion is not normalized",
                    error,
                    tolerance,
                )
            )
    return violations


def _qpos_rotation_violations(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    label: str,
    tolerance: float = 1e-5,
) -> list[GateViolation]:
    violations: list[GateViolation] = []
    if qpos.shape != (model.nq,) or np.any(~np.isfinite(qpos)):
        return violations
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr += 3
        elif joint_type != mujoco.mjtJoint.mjJNT_BALL:
            continue
        error = abs(float(np.linalg.norm(qpos[qpos_adr : qpos_adr + 4])) - 1.0)
        if error > tolerance:
            name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append(
                GateViolation(
                    GateCode.ROTATION_NORMALIZATION,
                    f"{label} joint {name} quaternion is not normalized",
                    error,
                    tolerance,
                )
            )
    return violations


def _dynamic_limit_violations(
    model: mujoco.MjModel,
    simulation: SimulationResult,
    policy: CertificationPolicy,
) -> list[GateViolation]:
    trace = simulation.trace
    violations: list[GateViolation] = []
    expected_shapes = (
        (trace.actuator_force, (len(trace.times_s), model.nu), "actuator force"),
        (trace.generalized_effort, (len(trace.times_s), model.nv), "generalized effort"),
    )
    for array, shape, label in expected_shapes:
        if array.shape != shape:
            violations.append(GateViolation(GateCode.TRACE_MISSING, f"{label} trace is missing or malformed"))
    if violations:
        return violations
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        width = 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
        dof = int(model.jnt_dofadr[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        if bool(model.jnt_limited[joint_id]) and joint_type in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            lower, upper = model.jnt_range[joint_id]
            observed_min = float(np.min(trace.qpos[:, qpos_address]))
            observed_max = float(np.max(trace.qpos[:, qpos_address]))
            if (
                observed_min < lower - policy.joint_position_tolerance
                or observed_max > upper + policy.joint_position_tolerance
            ):
                violations.append(
                    GateViolation(
                        GateCode.JOINT_POSITION_LIMIT,
                        f"joint {name} left its physical range",
                        f"[{observed_min:.6g}, {observed_max:.6g}]",
                        f"[{lower:.6g}, {upper:.6g}]",
                    )
                )
        velocity = float(np.max(np.abs(trace.qvel[:, dof : dof + width])))
        acceleration = float(
            np.max(
                np.abs(
                    np.diff(trace.qvel[:, dof : dof + width], axis=0)
                    / np.diff(trace.times_s)[:, None]
                )
            )
        )
        effort = float(np.max(np.abs(trace.generalized_effort[:, dof : dof + width])))
        power = float(
            np.max(
                np.abs(
                    trace.generalized_effort[:, dof : dof + width]
                    * trace.qvel[:, dof : dof + width]
                )
            )
        )
        checks = (
            (GateCode.VELOCITY_LIMIT, f"joint {name} velocity", velocity, policy.joint_velocity_limits.get(name, policy.default_joint_velocity_limit)),
            (GateCode.ACCELERATION_LIMIT, f"joint {name} acceleration", acceleration, policy.joint_acceleration_limits.get(name, policy.default_joint_acceleration_limit)),
            (GateCode.JOINT_EFFORT_LIMIT, f"joint {name} effort", effort, policy.joint_effort_limits.get(name, policy.default_joint_effort_limit)),
            (GateCode.JOINT_POWER_LIMIT, f"joint {name} power", power, policy.joint_power_limits.get(name, policy.default_joint_power_limit)),
        )
        for check in checks:
            violation = _limit_violation(*check)
            if violation:
                violations.append(violation)
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        effort = float(np.max(np.abs(trace.actuator_force[:, actuator_id])))
        limit = policy.actuator_effort_limits.get(name, policy.default_actuator_effort_limit)
        violation = _limit_violation(
            GateCode.ACTUATOR_EFFORT_LIMIT, f"actuator {name} effort", effort, limit
        )
        if violation:
            violations.append(violation)
    return violations


def _selector_matches_geom(
    model: mujoco.MjModel,
    selector: str,
    geom_id: int,
    geom_name: str,
) -> bool:
    if selector == geom_name:
        return True
    body_id = int(model.geom_bodyid[geom_id])
    while body_id >= 0:
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if selector == body_name:
            return True
        if body_id == 0:
            break
        body_id = int(model.body_parentid[body_id])
    return False


def _edge_contacts(
    model: mujoco.MjModel,
    frame,
    edge,
    minimum_force_n: float,
):
    return tuple(
        contact
        for contact in frame.contacts
        if contact.normal_force_n >= minimum_force_n
        and (
            (
                _selector_matches_geom(
                    model, edge.body_a, contact.geom1_id, contact.geom1_name
                )
                and _selector_matches_geom(
                    model, edge.body_b, contact.geom2_id, contact.geom2_name
                )
            )
            or (
                _selector_matches_geom(
                    model, edge.body_a, contact.geom2_id, contact.geom2_name
                )
                and _selector_matches_geom(
                    model, edge.body_b, contact.geom1_id, contact.geom1_name
                )
            )
        )
    )


def _planned_contact_violations(
    model: mujoco.MjModel,
    simulation: SimulationResult,
    policy: CertificationPolicy,
    planned_contacts,
) -> list[GateViolation]:
    if not planned_contacts:
        return []
    violations: list[GateViolation] = []
    tolerance = policy.contact_window_tolerance_s
    observed: dict[str, tuple[float, float]] = {}
    for edge in planned_contacts:
        active: list[tuple[int, tuple]] = []
        outside_samples: list[tuple[int, float]] = []
        for index, frame in enumerate(simulation.trace.contacts):
            contacts = _edge_contacts(model, frame, edge, policy.min_contact_force_n)
            if not contacts:
                continue
            if frame.time_s < edge.start_s - tolerance or frame.time_s > edge.end_s + tolerance:
                outside_samples.append((index, frame.time_s))
            else:
                active.append((index, contacts))
        persistent_outside: list[float] = []
        if outside_samples:
            sample_period = (
                float(np.median(np.diff(simulation.trace.times_s)))
                if len(simulation.trace.times_s) > 1
                else policy.contact_window_tolerance_s
            )
            _, run_start_time = outside_samples[0]
            prior_index, prior_time = outside_samples[0]
            for sample_index, sample_time in outside_samples[1:] + [(-1, float("nan"))]:
                if sample_index == prior_index + 1:
                    prior_index, prior_time = sample_index, sample_time
                    continue
                if prior_time - run_start_time + sample_period >= policy.unplanned_contact_persistence_s:
                    persistent_outside.append(prior_time)
                run_start_time = sample_time
                prior_index, prior_time = sample_index, sample_time
        if persistent_outside:
            violations.append(
                GateViolation(
                    GateCode.CONTACT_WINDOW,
                    f"planned contact {edge.contact_id} persisted outside its authored window",
                    max(persistent_outside),
                    f"[{edge.start_s},{edge.end_s}]",
                )
            )
        if not active:
            if edge.required:
                violations.append(
                    GateViolation(
                        GateCode.CONTACT_ORDER,
                        f"required planned contact {edge.contact_id} was never created",
                    )
                )
            continue
        first_time = simulation.trace.contacts[active[0][0]].time_s
        last_time = simulation.trace.contacts[active[-1][0]].time_s
        observed[edge.contact_id] = (first_time, last_time)
        if edge.required and first_time > edge.start_s + tolerance:
            violations.append(
                GateViolation(
                    GateCode.CONTACT_ORDER,
                    f"planned contact {edge.contact_id} was created late",
                    first_time,
                    edge.start_s + tolerance,
                )
            )
        expected_end_sample = max(
            (
                frame.time_s
                for frame in simulation.trace.contacts
                if frame.time_s <= edge.end_s + tolerance
            ),
            default=edge.end_s,
        )
        if edge.required and last_time < expected_end_sample - tolerance:
            violations.append(
                GateViolation(
                    GateCode.CONTACT_BREAK,
                    f"planned contact {edge.contact_id} broke before its authored end",
                    last_time,
                    edge.end_s,
                )
            )
        if edge.required:
            active_indices = {index for index, _ in active}
            dropout = next(
                (
                    frame.time_s
                    for index, frame in enumerate(simulation.trace.contacts)
                    if active[0][0] < index < active[-1][0]
                    and edge.start_s - tolerance <= frame.time_s <= edge.end_s + tolerance
                    and index not in active_indices
                ),
                None,
            )
            if dropout is not None:
                violations.append(
                    GateViolation(
                        GateCode.CONTACT_DROPOUT,
                        f"planned contact {edge.contact_id} dropped out inside its authored plateau",
                        time_s=dropout,
                    )
                )
        maximum_penetration = max(
            max(0.0, -contact.distance_m)
            for _, contacts in active
            for contact in contacts
        )
        if maximum_penetration > edge.max_penetration_m:
            violations.append(
                GateViolation(
                    GateCode.PENETRATION,
                    f"planned contact {edge.contact_id} exceeded its edge penetration limit",
                    maximum_penetration,
                    edge.max_penetration_m,
                )
            )
        data = mujoco.MjData(model)
        local_positions: list[tuple[np.ndarray, np.ndarray]] = []
        for frame_index, contacts in active:
            data.qpos[:] = simulation.trace.qpos[frame_index]
            mujoco.mj_forward(model, data)
            side_a: list[np.ndarray] = []
            side_b: list[np.ndarray] = []
            for contact in contacts:
                point = np.asarray(contact.position_m, dtype=np.float64)
                first_is_a = _selector_matches_geom(
                    model, edge.body_a, contact.geom1_id, contact.geom1_name
                )
                for geom_id, destination in (
                    (contact.geom1_id, side_a if first_is_a else side_b),
                    (contact.geom2_id, side_b if first_is_a else side_a),
                ):
                    body_id = int(model.geom_bodyid[geom_id])
                    rotation = data.xmat[body_id].reshape(3, 3)
                    destination.append(rotation.T @ (point - data.xpos[body_id]))
            local_positions.append((np.mean(side_a, axis=0), np.mean(side_b, axis=0)))
        positions_a = np.asarray([item[0] for item in local_positions], dtype=np.float64)
        positions_b = np.asarray([item[1] for item in local_positions], dtype=np.float64)
        slip = max(
            float(np.max(np.linalg.norm(positions_a - positions_a[0], axis=1))),
            float(np.max(np.linalg.norm(positions_b - positions_b[0], axis=1))),
        )
        if slip > edge.max_slip_m:
            violations.append(
                GateViolation(
                    GateCode.CONTACT_SLIP,
                    f"planned contact {edge.contact_id} exceeded its edge slip limit",
                    slip,
                    edge.max_slip_m,
                )
            )

    by_creation = sorted(planned_contacts, key=lambda item: (item.start_s, item.contact_id))
    by_break = sorted(planned_contacts, key=lambda item: (item.end_s, item.contact_id))
    for ordered, component, code, label in (
        (by_creation, 0, GateCode.CONTACT_ORDER, "creation"),
        (by_break, 1, GateCode.CONTACT_BREAK, "break"),
    ):
        prior_plan_time = -np.inf
        prior_observed = -np.inf
        for edge in ordered:
            if edge.contact_id not in observed:
                continue
            plan_time = edge.start_s if component == 0 else edge.end_s
            actual_time = observed[edge.contact_id][component]
            if plan_time > prior_plan_time + tolerance and actual_time < prior_observed - tolerance:
                violations.append(
                    GateViolation(
                        code,
                        f"planned contact {label} order was not observed at {edge.contact_id}",
                        actual_time,
                        plan_time,
                    )
                )
                break
            prior_plan_time = plan_time
            prior_observed = max(prior_observed, actual_time)
    return violations


def _contact_violations(
    model: mujoco.MjModel,
    simulation: SimulationResult,
    policy: CertificationPolicy,
    planned_contacts=(),
) -> list[GateViolation]:
    violations: list[GateViolation] = []
    maximum_penetration = max(
        (max(0.0, -contact.distance_m) for frame in simulation.trace.contacts for contact in frame.contacts),
        default=0.0,
    )
    if maximum_penetration > policy.max_penetration_m:
        violations.append(
            GateViolation(GateCode.PENETRATION, "contact penetration exceeds certified limit", maximum_penetration, policy.max_penetration_m)
        )
    if policy.allowed_contact_pairs is not None:
        unexpected: set[ContactPair] = set()
        for frame in simulation.trace.contacts:
            for contact in frame.contacts:
                if contact.normal_force_n < policy.min_contact_force_n:
                    continue
                pair = ContactPair.of(contact.geom1_name, contact.geom2_name)
                if pair not in policy.allowed_contact_pairs:
                    unexpected.add(pair)
        if unexpected:
            violations.append(
                GateViolation(
                    GateCode.UNEXPECTED_CONTACT,
                    "unapproved contact pairs were measured",
                    ", ".join(f"{pair.first}<->{pair.second}" for pair in sorted(unexpected)),
                )
            )
    first_contact_time: dict[ContactPair, float] = {}
    expected = set(policy.expected_contact_order)
    for frame in simulation.trace.contacts:
        for contact in frame.contacts:
            pair = ContactPair.of(contact.geom1_name, contact.geom2_name)
            if pair in expected and contact.normal_force_n >= policy.min_contact_force_n:
                first_contact_time.setdefault(pair, frame.time_s)
    previous_time = -np.inf
    for pair in policy.expected_contact_order:
        time = first_contact_time.get(pair)
        if time is None or time < previous_time:
            violations.append(
                GateViolation(GateCode.CONTACT_ORDER, "required measured contact order was not observed", f"missing/out-of-order:{pair.first}<->{pair.second}")
            )
            break
        previous_time = time
    violations.extend(
        _planned_contact_violations(
            model, simulation, policy, planned_contacts
        )
    )
    return violations


def _balance_violations(
    simulation: SimulationResult,
    kinematics: KinematicTrace,
    policy: CertificationPolicy,
) -> list[GateViolation]:
    try:
        pelvis = kinematics.body_positions[policy.pelvis_body]
        torso_rotation = kinematics.body_rotations[policy.torso_body]
    except KeyError as exc:
        raise PredicateSupportError(f"required balance body is missing: {exc.args[0]}") from exc
    violations: list[GateViolation] = []
    pelvis_drop = float(pelvis[0, 2] - np.min(pelvis[:, 2]))
    pelvis_drift = float(np.max(np.linalg.norm(pelvis - pelvis[0], axis=1)))
    initial_up = torso_rotation[0, :, 2]
    tilt = max(
        degrees(acos(float(np.clip(np.dot(rotation[:, 2], initial_up), -1.0, 1.0))))
        for rotation in torso_rotation
    )
    if pelvis_drop > policy.max_pelvis_drop_m or tilt > policy.max_torso_tilt_deg:
        violations.append(
            GateViolation(GateCode.FALL, "pelvis drop or torso tilt indicates a fall", f"drop={pelvis_drop:.6g}, tilt={tilt:.6g}", f"drop<={policy.max_pelvis_drop_m}, tilt<={policy.max_torso_tilt_deg}")
        )
    if pelvis_drift > policy.max_pelvis_drift_m:
        violations.append(
            GateViolation(GateCode.PELVIS_DRIFT, "pelvis drift exceeds limit", pelvis_drift, policy.max_pelvis_drift_m)
        )
    for foot in policy.support_feet:
        try:
            positions = kinematics.body_positions[foot.body_name]
        except KeyError as exc:
            raise PredicateSupportError(f"support foot body {foot.body_name!r} is missing") from exc
        drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        if drift > policy.max_foot_drift_m:
            violations.append(
                GateViolation(GateCode.FOOT_DRIFT, f"support foot {foot.body_name} drifted", drift, policy.max_foot_drift_m)
            )
        contact_indices: list[int] = []
        for index, frame in enumerate(simulation.trace.contacts):
            if any(
                {contact.geom1_name, contact.geom2_name} & foot.geom_names
                and {contact.geom1_name, contact.geom2_name} & policy.ground_geom_names
                and contact.normal_force_n >= policy.min_contact_force_n
                for contact in frame.contacts
            ):
                contact_indices.append(index)
        if contact_indices:
            planted = positions[contact_indices, :2]
            slip = float(np.max(np.linalg.norm(planted - planted[0], axis=1)))
            if slip > policy.max_foot_slip_m:
                violations.append(
                    GateViolation(GateCode.FOOT_SLIP, f"support foot {foot.body_name} slipped while contacting ground", slip, policy.max_foot_slip_m)
                )
    return violations


def _single_evaluation(
    request: CandidateCertificationRequest,
    simulation: SimulationResult,
) -> tuple[CertificationOutcome, tuple[GateViolation, ...], tuple[PredicateEvaluation, ...]]:
    if simulation.status is not SimulationStatus.COMPLETED:
        message = simulation.failure.message if simulation.failure else "simulation did not complete"
        return (
            CertificationOutcome.SIMULATION_FAILED,
            (GateViolation(GateCode.NONFINITE_TRACE, message),),
            (),
        )
    try:
        model, _ = load_model_source(request.simulation)
        kinematics = _kinematics(model, simulation)
        violations = _structural_violations(model, simulation)
        violations += _continuity_violations(
            model, simulation, request.policy.teleport_tolerance_m_or_rad
        )
        violations += _rotation_violations(model, simulation)
        violations += _dynamic_limit_violations(model, simulation, request.policy)
        violations += _contact_violations(
            model, simulation, request.policy, request.planned_contacts
        )
        violations += _balance_violations(simulation, kinematics, request.policy)
        context = ObjectStateContext(model=model, simulation=simulation, kinematics=kinematics)
        evaluations = tuple(predicate.evaluate(context) for predicate in request.predicates)
    except PredicateSupportError as exc:
        return (
            CertificationOutcome.UNSUPPORTED,
            (GateViolation(GateCode.UNSUPPORTED_PREDICATE, str(exc)),),
            (),
        )
    except (ValueError, mujoco.FatalError) as exc:
        return (
            CertificationOutcome.UNSUPPORTED,
            (GateViolation(GateCode.UNSUPPORTED_MODEL, str(exc)),),
            (),
        )
    violations.extend(
        GateViolation(GateCode.OBJECT_PREDICATE, evaluation.message, evaluation.observed, evaluation.required)
        for evaluation in evaluations
        if not evaluation.passed
    )
    return (
        CertificationOutcome.CERTIFIED if not violations else CertificationOutcome.INFEASIBLE,
        tuple(violations),
        evaluations,
    )


class CertificationEngine:
    def __init__(self, runtime: NativeMujocoRuntime | None = None) -> None:
        self._runtime = runtime or NativeMujocoRuntime()

    def certify(self, request: CandidateCertificationRequest) -> CertificationResult:
        binding_violations: list[GateViolation] = []
        if (
            request.expected_model_hash is not None
            and model_source_hash(request.simulation) != request.expected_model_hash
        ):
            binding_violations.append(
                GateViolation(
                    GateCode.ASSET_BINDING,
                    "authoritative MuJoCo payload does not match its certified asset hash",
                    model_source_hash(request.simulation),
                    request.expected_model_hash,
                )
            )
        if (
            request.rig_asset_hash is not None
            and request.scene_rig_asset_hash is not None
            and request.rig_asset_hash != request.scene_rig_asset_hash
        ):
            binding_violations.append(
                GateViolation(
                    GateCode.ASSET_BINDING,
                    "scene is bound to a different canonical rig asset",
                    request.scene_rig_asset_hash,
                    request.rig_asset_hash,
                )
            )
        if (
            request.rig_coordinate_system is not None
            and request.scene_coordinate_system is not None
            and request.rig_coordinate_system != request.scene_coordinate_system
        ):
            binding_violations.append(
                GateViolation(
                    GateCode.COORDINATE_BINDING,
                    "scene and rig coordinate systems do not match",
                    str(request.scene_coordinate_system),
                    str(request.rig_coordinate_system),
                )
            )
        try:
            boundary_model, _ = load_model_source(request.simulation)
            if request.simulation.initial_qpos is not None:
                binding_violations.extend(
                    _qpos_rotation_violations(
                        boundary_model,
                        np.asarray(request.simulation.initial_qpos, dtype=np.float64),
                        "initial state",
                    )
                )
            initial_target = request.simulation.trajectory.sample(0.0)
            binding_violations.extend(
                _qpos_rotation_violations(
                    boundary_model,
                    np.asarray(initial_target.qpos, dtype=np.float64),
                    "initial target",
                )
            )
        except Exception:
            # Invalid models/providers remain typed by the native runtime; this
            # preflight adds gates but does not replace runtime validation.
            pass
        if binding_violations:
            return CertificationResult(
                outcome=CertificationOutcome.INFEASIBLE,
                candidate_id=request.simulation.request_id,
                violations=tuple(binding_violations),
                predicate_evaluations=(),
                simulation_runs=(),
            )
        runs = tuple(
            self._runtime.simulate(request.simulation) for _ in range(request.policy.repeat_count)
        )
        evaluations = tuple(_single_evaluation(request, simulation) for simulation in runs)
        baseline_outcome, baseline_violations, baseline_predicates = evaluations[0]
        repeat_matches = all(
            outcome == baseline_outcome
            and violations == baseline_violations
            and predicates == baseline_predicates
            and compare_replays(runs[0], runs[index]).matches
            for index, (outcome, violations, predicates) in enumerate(evaluations[1:], start=1)
        )
        if not repeat_matches:
            return CertificationResult(
                outcome=CertificationOutcome.SIMULATION_FAILED,
                candidate_id=request.simulation.request_id,
                violations=(GateViolation(GateCode.REPEAT_DISAGREEMENT, "three-repeat state or outcome agreement failed"),),
                predicate_evaluations=baseline_predicates,
                simulation_runs=runs,
            )
        if baseline_outcome is not CertificationOutcome.CERTIFIED:
            return CertificationResult(
                outcome=baseline_outcome,
                candidate_id=request.simulation.request_id,
                violations=baseline_violations,
                predicate_evaluations=baseline_predicates,
                simulation_runs=runs,
            )
        if request.policy.require_export_reimport:
            if request.export_reimport is None:
                return CertificationResult(
                    outcome=CertificationOutcome.UNSUPPORTED,
                    candidate_id=request.simulation.request_id,
                    violations=(GateViolation(GateCode.EXPORT_REIMPORT, "no export/reimport validator was provided"),),
                    predicate_evaluations=baseline_predicates,
                    simulation_runs=runs,
                )
            try:
                export_result = request.export_reimport(runs[0])
                validation = export_result if isinstance(export_result, ExportValidation) else ExportValidation(bool(export_result))
            except Exception as exc:
                validation = ExportValidation(False, str(exc))
            if not validation.passed:
                return CertificationResult(
                    outcome=CertificationOutcome.SIMULATION_FAILED,
                    candidate_id=request.simulation.request_id,
                    violations=(GateViolation(GateCode.EXPORT_REIMPORT, validation.message or "export/reimport validation failed"),),
                    predicate_evaluations=baseline_predicates,
                    simulation_runs=runs,
                )
        return CertificationResult(
            outcome=CertificationOutcome.CERTIFIED,
            candidate_id=request.simulation.request_id,
            violations=(),
            predicate_evaluations=baseline_predicates,
            simulation_runs=runs,
        )

    def certify_many(
        self,
        requests: tuple[CandidateCertificationRequest, ...],
        *,
        max_workers: int | None = None,
    ) -> tuple[CertificationResult, ...]:
        """Run repeat simulations as ordered process-isolated candidate batches."""

        if not requests:
            return ()
        identifiers = [request.simulation.request_id for request in requests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("batched certification request IDs must be unique")
        repeat_counts = {request.policy.repeat_count for request in requests}
        if len(repeat_counts) != 1:
            raise ValueError("one certification batch requires one repeat count")
        runs: list[list[SimulationResult]] = [[] for _ in requests]
        for _ in range(repeat_counts.pop()):
            batch = self._runtime.simulate_batch(
                tuple(request.simulation for request in requests),
                max_workers=max_workers,
            )
            if len(batch) != len(requests):
                raise RuntimeError("simulation batch returned the wrong result count")
            for destination, simulation in zip(runs, batch, strict=True):
                destination.append(simulation)

        class _RecordedRuntime:
            def __init__(self, recorded: tuple[SimulationResult, ...]) -> None:
                self._recorded = iter(recorded)

            def simulate(self, simulation_request):  # type: ignore[no-untyped-def]
                del simulation_request
                return next(self._recorded)

        return tuple(
            CertificationEngine(_RecordedRuntime(tuple(recorded))).certify(request)  # type: ignore[arg-type]
            for request, recorded in zip(requests, runs, strict=True)
        )

    def certify_candidates(
        self,
        requests: tuple[CandidateCertificationRequest, ...],
    ) -> CandidateSelectionResult:
        results = tuple(self.certify(request) for request in requests)
        selected = next((result for result in results if result.certified), None)
        if selected is None:
            return CandidateSelectionResult(
                outcome=CertificationOutcome.ALL_CANDIDATES_REJECTED,
                selected_candidate_id=None,
                candidates=results,
            )
        return CandidateSelectionResult(
            outcome=CertificationOutcome.CERTIFIED,
            selected_candidate_id=selected.candidate_id,
            candidates=results,
        )
