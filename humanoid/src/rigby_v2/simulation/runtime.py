"""Native MuJoCo 240 Hz runtime.

``COMPLETED`` means that MuJoCo produced a finite trace.  It never means that a
task succeeded; task predicates belong to the certification layer.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
import multiprocessing
import os
from typing import Any, Callable, Sequence

import mujoco
from mujoco import rollout as mj_rollout
import numpy as np
import numpy.typing as npt

from rigby_core.simulation.controller import (
    ControllerConfigurationError,
    InverseDynamicsPDController,
    PDGains,
    StandingControlConfig,
    TargetProvider,
    WholeBodyStandingController,
)
from .metrics import ContactFrame, SimulationMetrics, capture_contact_frame, extract_metrics


PHYSICS_HZ = 240
FIXED_TIMESTEP_S = 1.0 / PHYSICS_HZ


class SimulationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationFailureCode(StrEnum):
    INVALID_MODEL = "invalid_model"
    INVALID_REQUEST = "invalid_request"
    MISSING_FREE_ROOT = "missing_free_root"
    UNSUPPORTED_ACTUATOR = "unsupported_actuator"
    NONFINITE_STATE = "nonfinite_state"
    ENGINE_ERROR = "engine_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SimulationFailure:
    code: SimulationFailureCode
    message: str
    step: int | None = None


@dataclass(frozen=True)
class SimulationConfig:
    duration_s: float
    physics_hz: int = PHYSICS_HZ
    gains: PDGains = field(default_factory=PDGains)
    standing: StandingControlConfig | None = None
    free_root_joint_name: str = "root"

    def validate(self) -> None:
        if self.physics_hz != PHYSICS_HZ:
            raise ValueError(f"native certification simulation is fixed at {PHYSICS_HZ} Hz")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not self.free_root_joint_name:
            raise ValueError("free_root_joint_name cannot be empty")


@dataclass(frozen=True)
class SimulationRequest:
    model_xml: str | None
    trajectory: TargetProvider
    config: SimulationConfig
    initial_qpos: npt.ArrayLike | None = None
    initial_qvel: npt.ArrayLike | None = None
    request_id: str = ""
    model_mjz: bytes | None = None
    cancel_check: Callable[[], bool] | None = None
    precomputed_ctrl: npt.ArrayLike | None = None


@dataclass(frozen=True)
class SimulationTrace:
    times_s: npt.NDArray[np.float64]
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    ctrl: npt.NDArray[np.float64]
    contacts: tuple[ContactFrame, ...]
    actuator_force: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    generalized_effort: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )

    @classmethod
    def empty(cls) -> "SimulationTrace":
        return cls(
            times_s=np.empty(0, dtype=np.float64),
            qpos=np.empty((0, 0), dtype=np.float64),
            qvel=np.empty((0, 0), dtype=np.float64),
            ctrl=np.empty((0, 0), dtype=np.float64),
            contacts=(),
            actuator_force=np.empty((0, 0), dtype=np.float64),
            generalized_effort=np.empty((0, 0), dtype=np.float64),
        )


@dataclass(frozen=True)
class SimulationResult:
    status: SimulationStatus
    request_id: str
    model_hash: str
    trace: SimulationTrace
    metrics: SimulationMetrics | None
    failure: SimulationFailure | None
    diagnostics: dict[str, Any]

    @property
    def completed(self) -> bool:
        return self.status is SimulationStatus.COMPLETED


def model_source_bytes(request: SimulationRequest) -> bytes:
    """Return the exact authoritative model payload used for hashing."""

    if request.model_mjz is not None:
        return bytes(request.model_mjz)
    if request.model_xml is not None:
        return request.model_xml.encode("utf-8")
    return b""


def model_source_hash(request: SimulationRequest) -> str:
    return sha256(model_source_bytes(request)).hexdigest()


def load_model_source(request: SimulationRequest) -> tuple[mujoco.MjModel, str]:
    """Load exactly one XML or self-contained MJZ model source."""

    has_xml = request.model_xml is not None
    has_mjz = request.model_mjz is not None
    if has_xml == has_mjz:
        raise ValueError("provide exactly one authoritative model source: model_xml or model_mjz")
    if request.model_mjz is not None:
        spec = mujoco.MjSpec.from_zip(BytesIO(request.model_mjz))
        return spec.compile(), "mjz"
    assert request.model_xml is not None
    return mujoco.MjModel.from_xml_string(request.model_xml), "xml"


def _failure_result(
    request: SimulationRequest,
    code: SimulationFailureCode,
    message: str,
    *,
    step: int | None = None,
    trace: SimulationTrace | None = None,
) -> SimulationResult:
    return SimulationResult(
        status=SimulationStatus.FAILED,
        request_id=request.request_id,
        model_hash=model_source_hash(request),
        trace=trace or SimulationTrace.empty(),
        metrics=None,
        failure=SimulationFailure(code=code, message=message, step=step),
        diagnostics={"task_success_evaluated": False, "qpos_writes_after_initialization": 0},
    )


def _free_root_qpos_adr(model: mujoco.MjModel, preferred_name: str) -> int | None:
    preferred = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, preferred_name)
    if preferred >= 0 and model.jnt_type[preferred] == mujoco.mjtJoint.mjJNT_FREE:
        return int(model.jnt_qposadr[preferred])
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        return None
    return int(model.jnt_qposadr[int(free_joints[0])])


def _actuated_qpos_adrs(model: mujoco.MjModel) -> tuple[int, ...]:
    return tuple(
        int(model.jnt_qposadr[int(model.actuator_trnid[index, 0])])
        for index in range(model.nu)
    )


class NativeMujocoRuntime:
    """Execute independent closed-loop native MuJoCo simulations."""

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        if request.precomputed_ctrl is not None:
            return _simulate_precomputed_requests((request,), max_workers=1)[0]
        try:
            request.config.validate()
        except ValueError as exc:
            return _failure_result(request, SimulationFailureCode.INVALID_REQUEST, str(exc))
        if (request.model_xml is None) == (request.model_mjz is None):
            return _failure_result(
                request,
                SimulationFailureCode.INVALID_REQUEST,
                "provide exactly one authoritative model source: model_xml or model_mjz",
            )
        try:
            model, model_source_format = load_model_source(request)
        except Exception as exc:  # MuJoCo exposes several parser exception types.
            return _failure_result(request, SimulationFailureCode.INVALID_MODEL, str(exc))

        root_adr = _free_root_qpos_adr(model, request.config.free_root_joint_name)
        if root_adr is None:
            return _failure_result(
                request,
                SimulationFailureCode.MISSING_FREE_ROOT,
                "model must identify a free humanoid root joint; multiple unnamed free joints are ambiguous",
            )

        # Runtime authority: source XML cannot change certification frequency.
        model.opt.timestep = FIXED_TIMESTEP_S
        data = mujoco.MjData(model)
        try:
            # The only direct authoritative state assignment.  The integration
            # loop below modifies qpos/qvel solely through mj_step.
            if request.initial_qpos is not None:
                initial_qpos = np.asarray(request.initial_qpos, dtype=np.float64)
                if initial_qpos.shape != (model.nq,) or np.any(~np.isfinite(initial_qpos)):
                    raise ValueError(f"initial_qpos must be finite with shape ({model.nq},)")
                data.qpos[:] = initial_qpos
            if request.initial_qvel is not None:
                initial_qvel = np.asarray(request.initial_qvel, dtype=np.float64)
                if initial_qvel.shape != (model.nv,) or np.any(~np.isfinite(initial_qvel)):
                    raise ValueError(f"initial_qvel must be finite with shape ({model.nv},)")
                data.qvel[:] = initial_qvel
            mujoco.mj_forward(model, data)
            base_controller = InverseDynamicsPDController(model, request.config.gains)
            controller: InverseDynamicsPDController | WholeBodyStandingController
            if request.config.standing is None:
                controller = base_controller
            else:
                controller = WholeBodyStandingController(
                    model, data, base_controller, request.config.standing
                )
            initial_target = request.trajectory.sample(0.0)
            # Validate provider dimensions before taking a physics step.
            controller.compute(data, initial_target)
        except ControllerConfigurationError as exc:
            return _failure_result(
                request, SimulationFailureCode.UNSUPPORTED_ACTUATOR, str(exc)
            )
        except (ValueError, FloatingPointError) as exc:
            return _failure_result(request, SimulationFailureCode.INVALID_REQUEST, str(exc))

        times = [float(data.time)]
        qpos = [data.qpos.copy()]
        qvel = [data.qvel.copy()]
        controls = [np.zeros(model.nu, dtype=np.float64)]
        actuator_forces = [np.zeros(model.nu, dtype=np.float64)]
        generalized_efforts = [np.zeros(model.nv, dtype=np.float64)]
        contacts = [capture_contact_frame(model, data)]
        step_count = int(np.ceil(request.config.duration_s / FIXED_TIMESTEP_S))
        terminal_target = initial_target.qpos.copy()

        for step in range(1, step_count + 1):
            if (
                request.cancel_check is not None
                and (step == 1 or step % 24 == 0)
                and request.cancel_check()
            ):
                partial = SimulationTrace(
                    times_s=np.asarray(times, dtype=np.float64),
                    qpos=np.asarray(qpos, dtype=np.float64),
                    qvel=np.asarray(qvel, dtype=np.float64),
                    ctrl=np.asarray(controls, dtype=np.float64),
                    contacts=tuple(contacts),
                    actuator_force=np.asarray(actuator_forces, dtype=np.float64),
                    generalized_effort=np.asarray(generalized_efforts, dtype=np.float64),
                )
                return _failure_result(
                    request,
                    SimulationFailureCode.CANCELLED,
                    "simulation was cancelled",
                    step=step,
                    trace=partial,
                )
            try:
                target_time = min(step * FIXED_TIMESTEP_S, request.config.duration_s)
                target = request.trajectory.sample(target_time)
                command = controller.compute(data, target)
                data.ctrl[:] = command
                mujoco.mj_step(model, data)
            except (ValueError, FloatingPointError) as exc:
                partial = SimulationTrace(
                    times_s=np.asarray(times, dtype=np.float64),
                    qpos=np.asarray(qpos, dtype=np.float64),
                    qvel=np.asarray(qvel, dtype=np.float64),
                    ctrl=np.asarray(controls, dtype=np.float64),
                    contacts=tuple(contacts),
                    actuator_force=np.asarray(actuator_forces, dtype=np.float64),
                    generalized_effort=np.asarray(generalized_efforts, dtype=np.float64),
                )
                return _failure_result(
                    request,
                    SimulationFailureCode.INVALID_REQUEST,
                    str(exc),
                    step=step,
                    trace=partial,
                )
            except mujoco.FatalError as exc:
                partial = SimulationTrace(
                    times_s=np.asarray(times, dtype=np.float64),
                    qpos=np.asarray(qpos, dtype=np.float64),
                    qvel=np.asarray(qvel, dtype=np.float64),
                    ctrl=np.asarray(controls, dtype=np.float64),
                    contacts=tuple(contacts),
                    actuator_force=np.asarray(actuator_forces, dtype=np.float64),
                    generalized_effort=np.asarray(generalized_efforts, dtype=np.float64),
                )
                return _failure_result(
                    request,
                    SimulationFailureCode.ENGINE_ERROR,
                    str(exc),
                    step=step,
                    trace=partial,
                )
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                partial = SimulationTrace(
                    times_s=np.asarray(times, dtype=np.float64),
                    qpos=np.asarray(qpos, dtype=np.float64),
                    qvel=np.asarray(qvel, dtype=np.float64),
                    ctrl=np.asarray(controls, dtype=np.float64),
                    contacts=tuple(contacts),
                    actuator_force=np.asarray(actuator_forces, dtype=np.float64),
                    generalized_effort=np.asarray(generalized_efforts, dtype=np.float64),
                )
                return _failure_result(
                    request,
                    SimulationFailureCode.NONFINITE_STATE,
                    "MuJoCo produced a non-finite state",
                    step=step,
                    trace=partial,
                )
            times.append(float(data.time))
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            controls.append(data.ctrl.copy())
            actuator_forces.append(data.actuator_force.copy())
            generalized_efforts.append(data.qfrc_actuator.copy())
            contacts.append(capture_contact_frame(model, data))
            terminal_target = target.qpos.copy()

        trace = SimulationTrace(
            times_s=np.asarray(times, dtype=np.float64),
            qpos=np.asarray(qpos, dtype=np.float64),
            qvel=np.asarray(qvel, dtype=np.float64),
            ctrl=np.asarray(controls, dtype=np.float64),
            contacts=tuple(contacts),
            actuator_force=np.asarray(actuator_forces, dtype=np.float64),
            generalized_effort=np.asarray(generalized_efforts, dtype=np.float64),
        )
        metrics = extract_metrics(
            times_s=trace.times_s,
            qpos=trace.qpos,
            qvel=trace.qvel,
            ctrl=trace.ctrl,
            actuator_force=trace.actuator_force,
            generalized_effort=trace.generalized_effort,
            contacts=trace.contacts,
            root_translation_adr=root_adr,
            actuated_qpos_adrs=_actuated_qpos_adrs(model),
            terminal_target_qpos=terminal_target,
        )
        if not metrics.finite:
            return _failure_result(
                request,
                SimulationFailureCode.NONFINITE_STATE,
                "copied simulation trace contains non-finite values",
                step=step_count,
                trace=trace,
            )
        return SimulationResult(
            status=SimulationStatus.COMPLETED,
            request_id=request.request_id,
            model_hash=model_source_hash(request),
            trace=trace,
            metrics=metrics,
            failure=None,
            diagnostics={
                "physics_hz": PHYSICS_HZ,
                "timestep_s": FIXED_TIMESTEP_S,
                "native_mujoco_version": mujoco.__version__,
                "model_source_format": model_source_format,
                "free_root_qpos_adr": root_adr,
                "task_success_evaluated": False,
                "qpos_writes_after_initialization": 0,
                "controller": (
                    "whole_body_standing_plus_inverse_dynamics_pd"
                    if request.config.standing is not None
                    else "inverse_dynamics_plus_bounded_pd"
                ),
                "standing_control": request.config.standing is not None,
                "execution_backend": "closed_loop_process_compatible",
                "process_id": os.getpid(),
            },
        )

    def simulate_batch(
        self,
        requests: Sequence[SimulationRequest],
        *,
        max_workers: int | None = None,
    ) -> list[SimulationResult]:
        """Run independent requests with the backend their controls permit.

        Open-loop controls use MuJoCo's native batched rollout.  Closed-loop
        controllers execute in spawned worker processes, never Python threads,
        so mutable engine/controller state cannot share an address space.
        """

        if not requests:
            return []
        workers = max_workers or min(len(requests), 5)
        if workers < 1:
            raise ValueError("max_workers must be at least one")
        results: list[SimulationResult | None] = [None] * len(requests)
        open_indices = [
            index for index, request in enumerate(requests)
            if request.precomputed_ctrl is not None
        ]
        closed_indices = [
            index for index, request in enumerate(requests)
            if request.precomputed_ctrl is None
        ]
        if open_indices:
            open_results = _simulate_precomputed_requests(
                tuple(requests[index] for index in open_indices),
                max_workers=min(workers, len(open_indices)),
            )
            for index, result in zip(open_indices, open_results, strict=True):
                results[index] = result
        if closed_indices:
            closed_requests = tuple(requests[index] for index in closed_indices)
            if any(request.cancel_check is not None for request in closed_requests):
                raise ValueError(
                    "closed-loop batch requests cannot carry process-local cancellation callbacks"
                )
            context = multiprocessing.get_context("spawn")
            try:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(closed_requests)),
                    mp_context=context,
                ) as pool:
                    closed_results = tuple(
                        pool.map(_simulate_closed_loop_process, closed_requests)
                    )
            except Exception as exc:
                closed_results = tuple(
                    _failure_result(
                        request,
                        SimulationFailureCode.INVALID_REQUEST,
                        f"closed-loop request could not execute in an isolated process: {exc}",
                    )
                    for request in closed_requests
                )
            for index, result in zip(closed_indices, closed_results, strict=True):
                results[index] = result
        if any(result is None for result in results):
            raise RuntimeError("simulation batch did not populate every result")
        return [result for result in results if result is not None]


def _simulate_closed_loop_process(request: SimulationRequest) -> SimulationResult:
    """Spawn-safe module entrypoint for one isolated closed-loop request."""

    return NativeMujocoRuntime().simulate(request)


def _precomputed_failure(
    request: SimulationRequest,
    code: SimulationFailureCode,
    message: str,
) -> SimulationResult:
    return _failure_result(request, code, message)


def _simulate_precomputed_requests(
    requests: Sequence[SimulationRequest],
    *,
    max_workers: int,
) -> list[SimulationResult]:
    """Execute open-loop controls through MuJoCo's native rollout backend."""

    if not requests:
        return []
    prepared: list[tuple[SimulationRequest, mujoco.MjModel, mujoco.MjData, np.ndarray, int]] = []
    failures: dict[int, SimulationResult] = {}
    state_spec = (
        mujoco.mjtState.mjSTATE_TIME.value
        | mujoco.mjtState.mjSTATE_PHYSICS.value
    )
    signature: tuple[int, int, int, int, int] | None = None
    nstep: int | None = None
    for index, request in enumerate(requests):
        try:
            request.config.validate()
            if (request.model_xml is None) == (request.model_mjz is None):
                raise ValueError(
                    "provide exactly one authoritative model source: model_xml or model_mjz"
                )
            if request.cancel_check is not None:
                raise ValueError("native rollout does not support state-dependent cancellation")
            if request.config.standing is not None:
                raise ValueError("precomputed rollout cannot run a state-dependent standing controller")
            model, source_format = load_model_source(request)
            model.opt.timestep = FIXED_TIMESTEP_S
            root_adr = _free_root_qpos_adr(model, request.config.free_root_joint_name)
            if root_adr is None:
                failures[index] = _precomputed_failure(
                    request,
                    SimulationFailureCode.MISSING_FREE_ROOT,
                    "model must identify one free humanoid root joint",
                )
                continue
            controls = np.asarray(request.precomputed_ctrl, dtype=np.float64)
            expected_steps = int(np.ceil(request.config.duration_s / FIXED_TIMESTEP_S))
            if controls.shape != (expected_steps, model.nu) or np.any(~np.isfinite(controls)):
                raise ValueError(
                    f"precomputed_ctrl must be finite with shape ({expected_steps}, {model.nu})"
                )
            data = mujoco.MjData(model)
            if request.initial_qpos is not None:
                initial_qpos = np.asarray(request.initial_qpos, dtype=np.float64)
                if initial_qpos.shape != (model.nq,) or np.any(~np.isfinite(initial_qpos)):
                    raise ValueError(f"initial_qpos must be finite with shape ({model.nq},)")
                data.qpos[:] = initial_qpos
            if request.initial_qvel is not None:
                initial_qvel = np.asarray(request.initial_qvel, dtype=np.float64)
                if initial_qvel.shape != (model.nv,) or np.any(~np.isfinite(initial_qvel)):
                    raise ValueError(f"initial_qvel must be finite with shape ({model.nv},)")
                data.qvel[:] = initial_qvel
            mujoco.mj_forward(model, data)
            current_signature = (model.nq, model.nv, model.na, model.nu, model.nsensordata)
            if signature is None:
                signature = current_signature
                nstep = expected_steps
            if current_signature != signature or expected_steps != nstep:
                raise ValueError(
                    "one native rollout batch requires identical model size signatures and step counts"
                )
            prepared.append((request, model, data, controls, root_adr))
            # Source format is deterministic per request and recovered below.
            del source_format
        except Exception as exc:
            failures[index] = _precomputed_failure(
                request, SimulationFailureCode.INVALID_REQUEST, str(exc)
            )
    if failures and len(failures) == len(requests):
        return [failures[index] for index in range(len(requests))]
    if failures:
        # Preserve a single native call contract rather than silently splitting
        # malformed and valid size signatures into unrelated rollouts.
        return [
            failures.get(index)
            or _precomputed_failure(
                request,
                SimulationFailureCode.INVALID_REQUEST,
                "native rollout batch contains an incompatible request",
            )
            for index, request in enumerate(requests)
        ]

    initial_states = []
    controls_batch = []
    for request, model, data, controls, _ in prepared:
        state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, state_spec)
        initial_states.append(state)
        controls_batch.append(controls)
    models = [item[1] for item in prepared]
    rollout_data = [mujoco.MjData(models[index % len(models)]) for index in range(max_workers)]
    try:
        states, _ = mj_rollout.rollout(
            models,
            rollout_data,
            np.asarray(initial_states, dtype=np.float64),
            np.asarray(controls_batch, dtype=np.float64),
            control_spec=mujoco.mjtState.mjSTATE_CTRL.value,
            nstep=nstep,
        )
    except Exception as exc:
        return [
            _precomputed_failure(request, SimulationFailureCode.ENGINE_ERROR, str(exc))
            for request in requests
        ]

    output: list[SimulationResult] = []
    for batch_index, (request, model, data, controls, root_adr) in enumerate(prepared):
        times = [float(data.time)]
        qpos = [data.qpos.copy()]
        qvel = [data.qvel.copy()]
        ctrl = [np.zeros(model.nu, dtype=np.float64)]
        actuator_force = [np.zeros(model.nu, dtype=np.float64)]
        generalized_effort = [np.zeros(model.nv, dtype=np.float64)]
        contacts = [capture_contact_frame(model, data)]
        for step_index, state in enumerate(states[batch_index]):
            mujoco.mj_setState(model, data, state, state_spec)
            data.ctrl[:] = controls[step_index]
            mujoco.mj_forward(model, data)
            times.append(float(data.time))
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            ctrl.append(data.ctrl.copy())
            actuator_force.append(data.actuator_force.copy())
            generalized_effort.append(data.qfrc_actuator.copy())
            contacts.append(capture_contact_frame(model, data))
        trace = SimulationTrace(
            times_s=np.asarray(times, dtype=np.float64),
            qpos=np.asarray(qpos, dtype=np.float64),
            qvel=np.asarray(qvel, dtype=np.float64),
            ctrl=np.asarray(ctrl, dtype=np.float64),
            contacts=tuple(contacts),
            actuator_force=np.asarray(actuator_force, dtype=np.float64),
            generalized_effort=np.asarray(generalized_effort, dtype=np.float64),
        )
        metrics = extract_metrics(
            times_s=trace.times_s,
            qpos=trace.qpos,
            qvel=trace.qvel,
            ctrl=trace.ctrl,
            actuator_force=trace.actuator_force,
            generalized_effort=trace.generalized_effort,
            contacts=trace.contacts,
            root_translation_adr=root_adr,
            actuated_qpos_adrs=_actuated_qpos_adrs(model),
            terminal_target_qpos=trace.qpos[-1],
        )
        if not metrics.finite:
            output.append(
                _failure_result(
                    request,
                    SimulationFailureCode.NONFINITE_STATE,
                    "native rollout produced a non-finite trace",
                    trace=trace,
                )
            )
            continue
        output.append(
            SimulationResult(
                status=SimulationStatus.COMPLETED,
                request_id=request.request_id,
                model_hash=model_source_hash(request),
                trace=trace,
                metrics=metrics,
                failure=None,
                diagnostics={
                    "physics_hz": PHYSICS_HZ,
                    "timestep_s": FIXED_TIMESTEP_S,
                    "native_mujoco_version": mujoco.__version__,
                    "model_source_format": "mjz" if request.model_mjz is not None else "xml",
                    "free_root_qpos_adr": root_adr,
                    "task_success_evaluated": False,
                    "qpos_writes_after_initialization": 0,
                    "controller": "precomputed_open_loop",
                    "standing_control": False,
                    "execution_backend": "native_mujoco_rollout",
                    "rollout_batch_size": len(prepared),
                    "rollout_threads": max_workers,
                    "process_id": os.getpid(),
                },
            )
        )
    return output
