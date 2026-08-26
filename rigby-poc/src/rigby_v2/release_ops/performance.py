"""Five-candidate local performance harness with phase-level timing."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.evidence import EvidenceRenderConfig, MujocoEvidenceRenderer
from rigby_v2.scenes import compile_scene
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    canonical_standing_config,
)


Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CandidateBenchmarkSpec:
    candidate_id: str
    object_pack: str
    duration_s: float = 10.0


@dataclass(frozen=True, slots=True)
class CandidateTiming:
    candidate_id: str
    duration_s: float
    phases_s: Mapping[str, float]
    total_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases_s", MappingProxyType(dict(self.phases_s)))


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    candidates: tuple[CandidateTiming, ...]
    p95_candidate_s: float
    total_wall_s: float
    target_p95_s: float
    vlm_included: bool = False
    workload_kind: str = "native_reference"

    @property
    def passed(self) -> bool:
        return (
            len(self.candidates) == 5
            and all(item.duration_s == 10.0 for item in self.candidates)
            and self.p95_candidate_s <= self.target_p95_s
            and not self.vlm_included
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "p95_candidate_s": self.p95_candidate_s,
            "total_wall_s": self.total_wall_s,
            "target_p95_s": self.target_p95_s,
            "vlm_included": self.vlm_included,
            "workload_kind": self.workload_kind,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "duration_s": item.duration_s,
                    "phases_s": dict(item.phases_s),
                    "total_s": item.total_s,
                }
                for item in self.candidates
            ],
        }


class PhaseTimer:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.phases: dict[str, float] = {}

    def measure(self, phase: str, action: Callable[[], object]) -> object:
        if not phase or phase in self.phases or phase == "vlm":
            raise ValueError("phase names must be unique, nonempty, and exclude VLM")
        start = self._clock()
        result = action()
        elapsed = self._clock() - start
        if elapsed < 0:
            raise ValueError("benchmark clock moved backwards")
        self.phases[phase] = elapsed
        return result


CandidateWorkload = Callable[[CandidateBenchmarkSpec, PhaseTimer], None]


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def run_performance_harness(
    specs: tuple[CandidateBenchmarkSpec, ...],
    workload: CandidateWorkload,
    *,
    clock: Clock = time.perf_counter,
    target_p95_s: float = 90.0,
    workload_kind: str = "native_reference",
) -> PerformanceReport:
    if len(specs) != 5 or len({item.candidate_id for item in specs}) != 5:
        raise ValueError("performance release gate requires exactly five unique candidates")
    if any(item.duration_s != 10.0 for item in specs):
        raise ValueError("every performance candidate must represent exactly 10 seconds")
    if target_p95_s <= 0:
        raise ValueError("performance target must be positive")
    candidates: list[CandidateTiming] = []
    for spec in specs:
        timer = PhaseTimer(clock)
        workload(spec, timer)
        if not timer.phases:
            raise ValueError("candidate workload did not record phase timings")
        total = sum(timer.phases.values())
        candidates.append(
            CandidateTiming(
                candidate_id=spec.candidate_id,
                duration_s=spec.duration_s,
                phases_s=timer.phases,
                total_s=total,
            )
        )
    totals = [candidate.total_s for candidate in candidates]
    return PerformanceReport(
        candidates=tuple(candidates),
        p95_candidate_s=_p95(totals),
        total_wall_s=sum(totals),
        target_p95_s=target_p95_s,
        vlm_included=False,
        workload_kind=workload_kind,
    )


REFERENCE_SPECS = tuple(
    CandidateBenchmarkSpec(f"candidate-{index + 1}", pack)
    for index, pack in enumerate(
        ("grasp_place_block", "drawer", "lever_button", "hand_tool", "container_lid")
    )
)


def local_reference_workload(spec: CandidateBenchmarkSpec, timer: PhaseTimer) -> None:
    state: dict[str, object] = {}

    def compile_model() -> None:
        compiled = compile_scene(spec.object_pack)
        state["compiled"] = compiled
        state["model"] = compiled.load_model()

    timer.measure("scene_compile", compile_model)
    model = state["model"]
    assert isinstance(model, mujoco.MjModel)

    def prepare() -> None:
        state["data"] = mujoco.MjData(model)
        state["sample_times"] = []
        state["sample_qpos"] = []

    timer.measure("trajectory_prepare", prepare)
    data = state["data"]
    assert isinstance(data, mujoco.MjData)

    def simulate() -> None:
        steps = round(spec.duration_s / model.opt.timestep)
        stride = max(1, round((1 / 30) / model.opt.timestep))
        sample_times = state["sample_times"]
        sample_qpos = state["sample_qpos"]
        assert isinstance(sample_times, list) and isinstance(sample_qpos, list)
        for step in range(steps + 1):
            if step % stride == 0:
                sample_times.append(float(data.time))
                sample_qpos.append(data.qpos.copy())
            if step < steps:
                mujoco.mj_step(model, data)

    timer.measure("native_simulation", simulate)

    def evidence() -> None:
        qpos = np.asarray(state["sample_qpos"], dtype=np.float32)
        times = np.asarray(state["sample_times"], dtype=np.float32)
        state["evidence"] = {
            "orbit": qpos[:, : min(12, qpos.shape[1])],
            "egocentric": qpos[:, : min(20, qpos.shape[1])],
            "task_closeup": qpos,
            "timestamps": times,
        }

    timer.measure("evidence_generation", evidence)

    def finalize() -> None:
        output = io.BytesIO()
        evidence_arrays = state["evidence"]
        assert isinstance(evidence_arrays, dict)
        np.savez_compressed(output, **evidence_arrays)
        state["artifact_sha256"] = hashlib.sha256(output.getvalue()).hexdigest()

    timer.measure("artifact_finalize", finalize)


def run_local_reference_benchmark(*, clock: Clock = time.perf_counter) -> PerformanceReport:
    return run_performance_harness(
        REFERENCE_SPECS,
        local_reference_workload,
        clock=clock,
    )


def run_production_renderer_benchmark(
    artifact_root: Path,
    *,
    clock: Clock = time.perf_counter,
    render_config: EvidenceRenderConfig | None = None,
) -> PerformanceReport:
    """Measure the real 240 Hz runtime plus audited three-camera renderer."""

    store = ContentAddressedArtifactStore(artifact_root)
    renderer = MujocoEvidenceRenderer(store, render_config)
    runtime = NativeMujocoRuntime()

    def workload(spec: CandidateBenchmarkSpec, timer: PhaseTimer) -> None:
        state: dict[str, object] = {}

        def compile_model() -> None:
            compiled = compile_scene(spec.object_pack)
            state["compiled"] = compiled
            state["model"] = compiled.load_model()

        timer.measure("scene_compile", compile_model)
        model = state["model"]
        assert isinstance(model, mujoco.MjModel)

        def prepare() -> None:
            # Exercise a representative articulated trajectory rather than a
            # stationary qpos0 render.  The alternating arm and hand pose keeps
            # the workload deterministic and limit-safe while forcing the
            # controller, trace interpolation, skin export inputs, and every
            # evidence camera to process changing state.
            rest = model.qpos0.copy()
            pose_a = rest.copy()
            pose_b = rest.copy()
            side = "left" if int(spec.candidate_id.rsplit("-", 1)[-1]) % 2 else "right"
            signed = 1.0 if side == "left" else -1.0
            targets = {
                f"{side}_shoulder_flex": 0.32,
                f"{side}_shoulder_abduct": 0.24 * signed,
                f"{side}_elbow_flex": 0.58,
                f"{side}_wrist_flex": 0.16,
                f"{side}_index_mcp": 0.42,
                f"{side}_middle_mcp": 0.36,
            }
            for joint_name, value in targets.items():
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id < 0:
                    raise RuntimeError(f"reference rig is missing joint {joint_name!r}")
                address = int(model.jnt_qposadr[joint_id])
                pose_a[address] = value
                pose_b[address] = value * (0.55 if "mcp" not in joint_name else 0.82)
            qpos = np.stack((rest, pose_a, pose_b, pose_a, rest))
            qvel = np.zeros((5, model.nv), dtype=np.float64)
            key_times = np.linspace(0.0, spec.duration_s, num=5, dtype=np.float64)
            state["request"] = SimulationRequest(
                model_xml=None,
                model_mjz=state["compiled"].mjz_bytes,  # type: ignore[union-attr]
                trajectory=LinearKeyframeTrajectory(
                    times_s=key_times,
                    qpos=qpos,
                    qvel=qvel,
                    qacc=qvel,
                ),
                config=SimulationConfig(
                    duration_s=spec.duration_s,
                    free_root_joint_name="pelvis_free",
                    standing=canonical_standing_config(model),
                ),
                initial_qpos=qpos[0],
                initial_qvel=qvel[0],
                request_id=spec.candidate_id,
            )

        timer.measure("trajectory_prepare", prepare)

        def simulate() -> None:
            result = runtime.simulate(state["request"])  # type: ignore[arg-type]
            if not result.completed:
                raise RuntimeError(f"reference simulation failed: {result.failure}")
            state["simulation"] = result

        timer.measure("native_simulation", simulate)

        def evidence() -> None:
            simulation = state["simulation"]
            state["evidence"] = renderer.render(
                model,
                simulation.trace,  # type: ignore[union-attr]
                candidate_id=spec.candidate_id,
                anonymous_id=f"reference-{spec.candidate_id}",
                metrics={"duration_s": spec.duration_s},
            )

        timer.measure("evidence_generation", evidence)

        def finalize() -> None:
            evidence_bundle = state["evidence"]
            references = [evidence_bundle.trace]  # type: ignore[union-attr]
            for camera in evidence_bundle.cameras:  # type: ignore[union-attr]
                references.extend((camera.raw_frames, camera.video))
            state["verified_artifact_bytes"] = sum(
                len(store.read_bytes(reference, verify=True)) for reference in references
            )

        timer.measure("artifact_finalize", finalize)

    return run_performance_harness(
        REFERENCE_SPECS,
        workload,
        clock=clock,
        workload_kind="articulated_keyframes_three_camera_h264",
    )
