"""Deterministic contact capture and simulation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, order=True)
class ContactRecord:
    geom1_id: int
    geom2_id: int
    geom1_name: str
    geom2_name: str
    distance_m: float
    position_m: tuple[float, float, float]
    normal_force_n: float


@dataclass(frozen=True)
class ContactFrame:
    time_s: float
    contacts: tuple[ContactRecord, ...]


@dataclass(frozen=True)
class SimulationMetrics:
    finite: bool
    physics_steps: int
    duration_s: float
    max_penetration_m: float
    max_speed_rad_or_m_s: float
    max_abs_control: float
    max_actuator_force: float
    max_generalized_effort: float
    max_joint_power: float
    root_translation_drift_m: float
    contact_frame_count: int
    contact_sample_count: int
    terminal_actuated_rmse: float | None


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def capture_contact_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> ContactFrame:
    """Copy current contacts into an immutable, stable ordering."""

    records: list[ContactRecord] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        records.append(
            ContactRecord(
                geom1_id=geom1_id,
                geom2_id=geom2_id,
                geom1_name=_geom_name(model, geom1_id),
                geom2_name=_geom_name(model, geom2_id),
                distance_m=float(contact.dist),
                position_m=tuple(float(value) for value in contact.pos),
                normal_force_n=abs(float(force[0])),
            )
        )
    records.sort(key=lambda item: (item.geom1_id, item.geom2_id, item.distance_m))
    return ContactFrame(time_s=float(data.time), contacts=tuple(records))


def extract_metrics(
    *,
    times_s: npt.NDArray[np.float64],
    qpos: npt.NDArray[np.float64],
    qvel: npt.NDArray[np.float64],
    ctrl: npt.NDArray[np.float64],
    actuator_force: npt.NDArray[np.float64],
    generalized_effort: npt.NDArray[np.float64],
    contacts: tuple[ContactFrame, ...],
    root_translation_adr: int,
    actuated_qpos_adrs: tuple[int, ...],
    terminal_target_qpos: npt.NDArray[np.float64] | None,
) -> SimulationMetrics:
    """Extract metrics using only copied authoritative simulation traces."""

    arrays = (times_s, qpos, qvel, ctrl, actuator_force, generalized_effort)
    finite = all(bool(np.all(np.isfinite(array))) for array in arrays)
    max_penetration = max(
        (max(0.0, -record.distance_m) for frame in contacts for record in frame.contacts),
        default=0.0,
    )
    max_speed = float(np.max(np.abs(qvel))) if qvel.size else 0.0
    max_control = float(np.max(np.abs(ctrl))) if ctrl.size else 0.0
    max_actuator_force = (
        float(np.max(np.abs(actuator_force))) if actuator_force.size else 0.0
    )
    max_generalized_effort = (
        float(np.max(np.abs(generalized_effort))) if generalized_effort.size else 0.0
    )
    max_joint_power = (
        float(np.max(np.abs(generalized_effort * qvel)))
        if generalized_effort.size and generalized_effort.shape == qvel.shape
        else 0.0
    )
    if len(qpos):
        start = qpos[0, root_translation_adr : root_translation_adr + 3]
        displacement = qpos[:, root_translation_adr : root_translation_adr + 3] - start
        root_drift = float(np.max(np.linalg.norm(displacement, axis=1)))
    else:
        root_drift = 0.0
    terminal_rmse: float | None = None
    if len(qpos) and terminal_target_qpos is not None and actuated_qpos_adrs:
        actual = qpos[-1, list(actuated_qpos_adrs)]
        target = terminal_target_qpos[list(actuated_qpos_adrs)]
        terminal_rmse = float(np.sqrt(np.mean(np.square(actual - target))))
    return SimulationMetrics(
        finite=finite,
        physics_steps=max(0, len(times_s) - 1),
        duration_s=float(times_s[-1] - times_s[0]) if len(times_s) else 0.0,
        max_penetration_m=float(max_penetration),
        max_speed_rad_or_m_s=max_speed,
        max_abs_control=max_control,
        max_actuator_force=max_actuator_force,
        max_generalized_effort=max_generalized_effort,
        max_joint_power=max_joint_power,
        root_translation_drift_m=root_drift,
        contact_frame_count=sum(bool(frame.contacts) for frame in contacts),
        contact_sample_count=sum(len(frame.contacts) for frame in contacts),
        terminal_actuated_rmse=terminal_rmse,
    )
