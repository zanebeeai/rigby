from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from pydantic import ValidationError

from .artifacts import ContentAddressedArtifactStore
from .config import RuntimeSettings
from .contracts import (
    CandidateTrajectoryV1,
    GateResultV1,
    MotionProgramV2,
    ReproducibilityManifestV1,
    RigAssetManifestV1,
    SceneManifestV2,
    SimulationResultV1,
    SolverConfigV1,
)
from .errors import ArtifactIntegrityError, FailureCode, JobLeaseError, RigbyV2Error
from .hashing import canonical_json
from .jobs import JobStore
from .observability import configure_json_logging, log_event
from .postgres_jobs import PostgresJobStore
from .records import FailureRecordV1, JobRecord, JobState
from .rigging.exporter import export_trace_to_artifact
from .simulation import (
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    StandingControlConfig,
)
from .simulation import SimulationRequest as NativeSimulationRequest
from .simulation import SimulationStatus as NativeSimulationStatus


@dataclass(slots=True)
class WorkerDependencies:
    jobs: JobStore
    artifacts: ContentAddressedArtifactStore
    settings: RuntimeSettings
    runtime: NativeMujocoRuntime


def _trace_bytes(result: Any) -> bytes:
    contact_payload = [
        {
            "time_s": frame.time_s,
            "contacts": [asdict(contact) for contact in frame.contacts],
        }
        for frame in result.trace.contacts
    ]
    contact_bytes = canonical_json(contact_payload).encode("utf-8")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        times_s=result.trace.times_s,
        qpos=result.trace.qpos,
        qvel=result.trace.qvel,
        ctrl=result.trace.ctrl,
        contacts_json=np.frombuffer(contact_bytes, dtype=np.uint8),
    )
    return buffer.getvalue()


def _joint_widths(joint_type: int) -> tuple[int, int]:
    kind = mujoco.mjtJoint(joint_type)
    if kind is mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if kind is mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _map_generalized_state(
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map generalized state by named joint, preserving target-only object state."""

    positions = np.broadcast_to(
        target_model.qpos0, (len(source_qpos), target_model.nq)
    ).copy()
    velocities = np.zeros((len(source_qvel), target_model.nv), dtype=np.float64)
    for source_joint in range(source_model.njnt):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, source_joint)
        if name is None:
            raise ValueError("Source rig contains an unnamed joint")
        target_joint = mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if target_joint < 0:
            raise ValueError(f"Compiled scene is missing rig joint {name!r}")
        source_type = int(source_model.jnt_type[source_joint])
        target_type = int(target_model.jnt_type[target_joint])
        if source_type != target_type:
            raise ValueError(f"Compiled scene changed the type of rig joint {name!r}")
        qpos_width, qvel_width = _joint_widths(source_type)
        source_qpos_adr = int(source_model.jnt_qposadr[source_joint])
        target_qpos_adr = int(target_model.jnt_qposadr[target_joint])
        source_qvel_adr = int(source_model.jnt_dofadr[source_joint])
        target_qvel_adr = int(target_model.jnt_dofadr[target_joint])
        positions[:, target_qpos_adr : target_qpos_adr + qpos_width] = source_qpos[
            :, source_qpos_adr : source_qpos_adr + qpos_width
        ]
        velocities[:, target_qvel_adr : target_qvel_adr + qvel_width] = source_qvel[
            :, source_qvel_adr : source_qvel_adr + qvel_width
        ]
    return positions, velocities


def _project_qpos_to_rig(
    scene_model: mujoco.MjModel,
    rig_model: mujoco.MjModel,
    scene_qpos: np.ndarray,
) -> np.ndarray:
    projected = np.broadcast_to(rig_model.qpos0, (len(scene_qpos), rig_model.nq)).copy()
    for rig_joint in range(rig_model.njnt):
        name = mujoco.mj_id2name(rig_model, mujoco.mjtObj.mjOBJ_JOINT, rig_joint)
        if name is None:
            raise ValueError("Rig contains an unnamed joint")
        scene_joint = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if scene_joint < 0:
            raise ValueError(f"Compiled scene is missing rig joint {name!r}")
        rig_type = int(rig_model.jnt_type[rig_joint])
        if rig_type != int(scene_model.jnt_type[scene_joint]):
            raise ValueError(f"Compiled scene changed the type of rig joint {name!r}")
        qpos_width, _ = _joint_widths(rig_type)
        rig_adr = int(rig_model.jnt_qposadr[rig_joint])
        scene_adr = int(scene_model.jnt_qposadr[scene_joint])
        projected[:, rig_adr : rig_adr + qpos_width] = scene_qpos[
            :, scene_adr : scene_adr + qpos_width
        ]
    return projected


def _canonical_standing_config(model: mujoco.MjModel) -> StandingControlConfig:
    prefixes = (
        "spine_",
        "chest_",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    correction_joints = tuple(
        name
        for joint_id in range(model.njnt)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        and name.startswith(prefixes)
    )
    return StandingControlConfig(
        support_foot_bodies=("left_foot", "right_foot"),
        pelvis_body="pelvis",
        torso_body="torso",
        correction_joint_names=correction_joints,
        max_generalized_correction=30.0,
    )


_INTEGRATORS = {
    mujoco.mjtIntegrator.mjINT_EULER: "Euler",
    mujoco.mjtIntegrator.mjINT_IMPLICIT: "implicit",
    mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "implicitfast",
    mujoco.mjtIntegrator.mjINT_RK4: "RK4",
}
_SOLVERS = {
    mujoco.mjtSolver.mjSOL_PGS: "PGS",
    mujoco.mjtSolver.mjSOL_CG: "CG",
    mujoco.mjtSolver.mjSOL_NEWTON: "Newton",
}


def _actual_solver_config(model: mujoco.MjModel) -> SolverConfigV1:
    """Read the authoritative options that the native runtime will execute."""

    try:
        integrator = _INTEGRATORS[mujoco.mjtIntegrator(int(model.opt.integrator))]
        solver = _SOLVERS[mujoco.mjtSolver(int(model.opt.solver))]
    except (KeyError, ValueError) as error:
        raise ValueError(
            "compiled model uses an unsupported MuJoCo solver option"
        ) from error
    return SolverConfigV1(
        # NativeMujocoRuntime authoritatively fixes this value before stepping.
        timestep_s=1.0 / 240.0,
        integrator=integrator,  # type: ignore[arg-type]
        solver=solver,  # type: ignore[arg-type]
        iterations=int(model.opt.iterations),
        tolerance=float(model.opt.tolerance),
    )


def _assert_requested_solver_is_authoritative(
    requested: SolverConfigV1, actual: SolverConfigV1
) -> None:
    if requested != actual:
        raise ValueError(
            "submitted solver configuration does not match the compiled MuJoCo model; "
            f"requested={requested.canonical_json()}, actual={actual.canonical_json()}"
        )


class SimulationWorker:
    def __init__(self, dependencies: WorkerDependencies) -> None:
        self.dependencies = dependencies

    def _load_contract(self, reference: Any, contract: type[Any]) -> Any:
        payload = self.dependencies.artifacts.read_bytes(reference)
        return contract.model_validate_json(payload)

    def _failure(
        self,
        record: JobRecord,
        *,
        code: FailureCode,
        message: str,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> JobRecord:
        failure = FailureRecordV1(
            job_id=record.job.job_id,
            code=code,
            message=message,
            stage="simulation",
            retryable=retryable,
            details=details or {},
        )
        log_event(
            logging.getLogger("rigby_v2.worker"),
            "simulation_job_failed",
            level=logging.ERROR,
            job_id=record.job.job_id,
            failure_code=code.value,
            retryable=retryable,
        )
        return self.dependencies.jobs.fail(
            record.job.job_id,
            self.dependencies.settings.worker_id,
            failure,
        )

    def execute(self, record: JobRecord) -> JobRecord:
        job = record.job
        artifacts = self.dependencies.artifacts
        current = self.dependencies.jobs.get(job.job_id)
        if current.cancel_requested or current.state is JobState.CANCELLED:
            return current
        log_event(
            logging.getLogger("rigby_v2.worker"),
            "simulation_job_started",
            job_id=job.job_id,
            attempt=record.attempts,
            worker_id=self.dependencies.settings.worker_id,
        )
        try:
            try:
                snapshot = self.dependencies.settings.reproducibility_snapshot()
            except OSError as error:
                raise RigbyV2Error(
                    FailureCode.INVALID_CONTRACT,
                    "dependency lock cannot be hashed; refusing an unreproducible simulation",
                    details={"expected": "uv.lock", "error": str(error)},
                ) from error
            lock_hash = snapshot["dependency_lock"]["sha256"]
            if not isinstance(lock_hash, str):
                raise RigbyV2Error(
                    FailureCode.INVALID_CONTRACT,
                    "dependency lock is missing; refusing an unreproducible simulation",
                    details={"expected": "uv.lock"},
                )
            # Reading every referenced input verifies that no hash-only or
            # missing input can be smuggled into a durable job.
            program = self._load_contract(job.program_artifact, MotionProgramV2)
            scene = self._load_contract(job.scene_artifact, SceneManifestV2)
            rig = self._load_contract(job.rig_artifact, RigAssetManifestV1)
            candidate = self._load_contract(
                job.candidate_artifact, CandidateTrajectoryV1
            )
            rig_xml = artifacts.read_bytes(rig.mjcf).decode("utf-8")
            rig_model = mujoco.MjModel.from_xml_string(rig_xml)
            model_mjz: bytes | None = None
            if scene.compiled_mjz is not None:
                assert scene.compiled_mjcf is not None
                model_mjz = artifacts.read_bytes(scene.compiled_mjz)
                model_xml = artifacts.read_bytes(scene.compiled_mjcf).decode("utf-8")
                model = mujoco.MjSpec.from_zip(io.BytesIO(model_mjz)).compile()
            else:
                model_xml = rig_xml
                model = rig_model
            actual_solver = _actual_solver_config(model)
            _assert_requested_solver_is_authoritative(job.solver, actual_solver)
            if scene.rig_asset_hash != job.rig_hash:
                raise ValueError(
                    "Scene manifest does not bind the submitted rig artifact"
                )
            if program.rig_id != rig.rig_id or program.scene_id != scene.scene_id:
                raise ValueError(
                    "Motion program rig/scene identifiers do not match the job"
                )
            if candidate.rig_id != rig.rig_id:
                raise ValueError(
                    f"Candidate rig {candidate.rig_id!r} does not match {rig.rig_id!r}"
                )
            if (
                candidate.program_hash is not None
                and candidate.program_hash != job.program_hash
            ):
                raise ValueError(
                    "Candidate trajectory does not bind the submitted program"
                )
            if candidate.rig_hash is not None and candidate.rig_hash != job.rig_hash:
                raise ValueError("Candidate trajectory does not bind the submitted rig")
            qpos = np.asarray(
                [sample.qpos for sample in candidate.samples], dtype=np.float64
            )
            qvel = np.asarray(
                [sample.qvel for sample in candidate.samples], dtype=np.float64
            )
            qacc = (
                np.asarray(
                    [sample.qacc for sample in candidate.samples], dtype=np.float64
                )
                if candidate.samples[0].qacc
                else None
            )
            times = np.asarray(
                [sample.time_s for sample in candidate.samples], dtype=np.float64
            )
            if qpos.shape[1] == rig_model.nq and qvel.shape[1] == rig_model.nv:
                source_qpos = qpos
                qpos, qvel = _map_generalized_state(rig_model, model, source_qpos, qvel)
                if qacc is not None:
                    _, qacc = _map_generalized_state(
                        rig_model, model, source_qpos, qacc
                    )
            elif qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
                raise ValueError(
                    "Candidate generalized-state dimensions do not match the rig or scene: "
                    f"qpos {qpos.shape[1]}/{model.nq}, qvel {qvel.shape[1]}/{model.nv}"
                )
            trajectory = LinearKeyframeTrajectory(
                times_s=times, qpos=qpos, qvel=qvel, qacc=qacc
            )
            requested_duration = float(times[-1])
            self.dependencies.jobs.heartbeat(
                job.job_id,
                self.dependencies.settings.worker_id,
                lease_seconds=max(
                    self.dependencies.settings.lease_seconds,
                    requested_duration * 10.0 + 30.0,
                ),
            )
            native = self.dependencies.runtime.simulate(
                NativeSimulationRequest(
                    request_id=job.job_id,
                    model_xml=model_xml if model_mjz is None else None,
                    model_mjz=model_mjz,
                    trajectory=trajectory,
                    config=SimulationConfig(
                        duration_s=requested_duration,
                        free_root_joint_name="pelvis_free",
                        standing=_canonical_standing_config(model),
                    ),
                    initial_qpos=qpos[0],
                    initial_qvel=qvel[0],
                    cancel_check=lambda: (
                        self.dependencies.jobs.get(job.job_id).cancel_requested
                    ),
                )
            )
            if native.status is not NativeSimulationStatus.COMPLETED:
                assert native.failure is not None
                if native.failure.code.value == "cancelled":
                    return self.dependencies.jobs.get(job.job_id)
                return self._failure(
                    record,
                    code=FailureCode.SIMULATION_FAILED,
                    message=native.failure.message,
                    details={
                        "native_code": native.failure.code.value,
                        "step": native.failure.step,
                        "diagnostics": native.diagnostics,
                    },
                    retryable=native.failure.code.value == "engine_error",
                )
            assert native.metrics is not None
            trace = artifacts.put_bytes(
                _trace_bytes(native),
                media_type="application/vnd.rigby.mujoco-trace+npz",
                filename=f"{job.job_id}-trace.npz",
            )
            exported = None
            if rig.visual_asset is not None:
                export_qpos = (
                    native.trace.qpos
                    if model is rig_model
                    else _project_qpos_to_rig(model, rig_model, native.trace.qpos)
                )
                exported = export_trace_to_artifact(
                    artifacts=artifacts,
                    source_glb_ref=rig.visual_asset,
                    model_xml=rig_xml,
                    rig=rig,
                    times_s=native.trace.times_s,
                    qpos=export_qpos,
                    filename=f"{job.job_id}-animation.glb",
                ).reference
            gates = (
                GateResultV1(
                    gate="finite_trace", passed=native.metrics.finite, measured=True
                ),
                GateResultV1(
                    gate="fixed_physics_rate",
                    passed=native.diagnostics["physics_hz"] == 240,
                    measured=int(native.diagnostics["physics_hz"]),
                    threshold=240,
                ),
            )
            result = SimulationResultV1(
                job_id=job.job_id,
                success=False,
                outcome="simulation_completed_uncertified",
                trace=trace,
                export=exported,
                gates=gates,
                metrics={
                    key: value
                    for key, value in asdict(native.metrics).items()
                    if value is not None
                },
                reproducibility=ReproducibilityManifestV1(
                    os=f"{snapshot['platform']['system']} {snapshot['platform']['release']}",
                    python_version=str(snapshot["platform"]["python"]),
                    mujoco_version=str(native.diagnostics["native_mujoco_version"]),
                    model_hash=native.model_hash,
                    solver=actual_solver,
                    seeds=(job.seed,),
                    requested_capture=job.capture,
                    cameras=scene.cameras,
                    dependency_lock_hash=lock_hash,
                ),
            )
            completed = self.dependencies.jobs.complete(
                job.job_id,
                self.dependencies.settings.worker_id,
                result,
            )
            log_event(
                logging.getLogger("rigby_v2.worker"),
                "simulation_job_completed",
                job_id=job.job_id,
                outcome=result.outcome,
                trace_sha256=result.trace.sha256,
                certified=result.success,
            )
            return completed
        except (ValidationError, ValueError, UnicodeDecodeError) as exc:
            return self._failure(
                record,
                code=FailureCode.INVALID_CONTRACT,
                message=str(exc),
            )
        except (ArtifactIntegrityError, FileNotFoundError) as exc:
            return self._failure(
                record,
                code=FailureCode.ARTIFACT_INTEGRITY,
                message=str(exc),
            )
        except RigbyV2Error as exc:
            return self._failure(
                record,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except (mujoco.FatalError, RuntimeError) as exc:
            if isinstance(exc, JobLeaseError):
                current = self.dependencies.jobs.get(job.job_id)
                if current.state is JobState.CANCELLED:
                    return current
                raise
            return self._failure(
                record,
                code=FailureCode.SIMULATION_FAILED,
                message=str(exc),
                retryable=True,
            )

    def run_once(self) -> JobRecord | None:
        self.dependencies.jobs.recover()
        record = self.dependencies.jobs.lease(
            self.dependencies.settings.worker_id,
            lease_seconds=self.dependencies.settings.lease_seconds,
        )
        if record is not None:
            self._pause_after_lease_for_restart_acceptance(record)
        return self.execute(record) if record is not None else None

    @staticmethod
    def _pause_after_lease_for_restart_acceptance(record: JobRecord) -> None:
        """Expose a deterministic process-kill point only in the opt-in acceptance run."""

        if os.environ.get("RIGBY_V2_RESTART_ACCEPTANCE") != "1":
            return
        marker_value = os.environ.get("RIGBY_V2_PAUSE_AFTER_LEASE_FILE")
        if not marker_value:
            return
        marker = Path(marker_value)
        if not marker.is_absolute():
            raise ValueError("RIGBY_V2_PAUSE_AFTER_LEASE_FILE must be absolute")
        marker.write_text(record.job.job_id, encoding="utf-8")
        log_event(
            logging.getLogger("rigby_v2.worker"),
            "restart_acceptance_paused_after_lease",
            job_id=record.job.job_id,
            marker=str(marker),
        )
        while marker.exists():
            time.sleep(0.05)

    def run_forever(self, *, idle_seconds: float = 0.25) -> None:
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be positive")
        while True:
            if self.run_once() is None:
                time.sleep(idle_seconds)


def create_worker(settings: RuntimeSettings | None = None) -> SimulationWorker:
    resolved = settings or RuntimeSettings.from_env()
    return SimulationWorker(
        WorkerDependencies(
            jobs=PostgresJobStore(resolved.database_url),
            artifacts=ContentAddressedArtifactStore(resolved.artifact_root),
            settings=resolved,
            runtime=NativeMujocoRuntime(),
        )
    )


def run() -> None:
    configure_json_logging()
    worker = create_worker()
    worker.dependencies.jobs.recover()
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        return
