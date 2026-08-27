"""Calibrated, deterministic survivor re-evaluation under bounded variation."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from rigby_v2.simulation.controller import ControlTarget, PDGains, TargetProvider
from rigby_v2.simulation.runtime import SimulationRequest, model_source_hash

from .engine import CertificationEngine
from .models import CandidateCertificationRequest, CertificationResult


@dataclass(frozen=True)
class RobustnessVariation:
    variation_id: str
    friction_scale: float = 1.0
    object_mass_scale: float = 1.0
    controller_gain_scale: float = 1.0
    initial_root_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_timing_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.variation_id:
            raise ValueError("variation_id is required")
        if not 0.8 <= self.friction_scale <= 1.2:
            raise ValueError("friction variation must remain within +/-20%")
        if not 0.8 <= self.object_mass_scale <= 1.2:
            raise ValueError("object mass variation must remain within +/-20%")
        if not 0.8 <= self.controller_gain_scale <= 1.2:
            raise ValueError("controller gain variation must remain within +/-20%")
        if not 0.95 <= self.phase_timing_scale <= 1.05:
            raise ValueError("phase timing variation must remain within +/-5%")
        if np.linalg.norm(self.initial_root_offset_m) > 0.02:
            raise ValueError("initial root variation must remain within 2 cm")


@dataclass(frozen=True)
class RobustnessCertificationResult:
    baseline: CertificationResult
    variations: tuple[tuple[RobustnessVariation, CertificationResult], ...]

    @property
    def robust(self) -> bool:
        return self.baseline.certified and all(
            result.certified for _, result in self.variations
        )


@dataclass(frozen=True)
class _PhaseTimingWarp(TargetProvider):
    provider: TargetProvider
    duration_s: float
    scale: float

    def sample(self, time_s: float) -> ControlTarget:
        # Endpoint-preserving monotone warp; derivative remains positive for
        # the admitted +/-5% calibration range.
        phase = 2.0 * np.pi * np.clip(time_s, 0.0, self.duration_s) / self.duration_s
        warped = time_s + (self.scale - 1.0) * self.duration_s * np.sin(phase) / (
            2.0 * np.pi
        )
        return self.provider.sample(float(np.clip(warped, 0.0, self.duration_s)))


def calibrated_variations() -> tuple[RobustnessVariation, ...]:
    return (
        RobustnessVariation("low-friction", friction_scale=0.9),
        RobustnessVariation("high-friction", friction_scale=1.1),
        RobustnessVariation("heavy-object", object_mass_scale=1.1),
        RobustnessVariation(
            "low-gain-fast-phase",
            controller_gain_scale=0.9,
            phase_timing_scale=0.98,
        ),
        RobustnessVariation(
            "initial-offset-slow-phase",
            initial_root_offset_m=(0.005, 0.0, 0.0),
            phase_timing_scale=1.02,
        ),
    )


def _mutated_mjz(request: SimulationRequest, variation: RobustnessVariation) -> bytes:
    if request.model_mjz is not None:
        spec = mujoco.MjSpec.from_zip(io.BytesIO(request.model_mjz))
    elif request.model_xml is not None:
        spec = mujoco.MjSpec.from_string(request.model_xml)
    else:
        raise ValueError("Simulation request has no model source")
    object_geom_count = 0
    for geom in spec.geoms:
        if not geom.name.startswith("obj__"):
            continue
        object_geom_count += 1
        friction = np.asarray(geom.friction, dtype=np.float64).copy()
        friction *= variation.friction_scale
        geom.friction = friction
        if variation.object_mass_scale != 1.0 and np.isfinite(geom.mass):
            geom.mass = float(geom.mass) * variation.object_mass_scale
    if variation.object_mass_scale != 1.0 and object_geom_count == 0:
        raise ValueError("Object-mass variation requires namespaced scene object geometry")
    spec.compile()
    stream = io.BytesIO()
    mujoco.MjSpec.to_zip(spec, stream)
    return stream.getvalue()


def vary_simulation_request(
    request: SimulationRequest, variation: RobustnessVariation
) -> SimulationRequest:
    gains = request.config.gains
    scale = variation.controller_gain_scale
    varied_gains = PDGains(
        kp=gains.kp * scale,
        kd=gains.kd * scale,
        max_pd_torque=gains.max_pd_torque,
        max_control=gains.max_control,
    )
    standing = request.config.standing
    if standing is not None and scale != 1.0:
        standing = replace(
            standing,
            com_kp=standing.com_kp * scale,
            com_kd=standing.com_kd * scale,
            pelvis_kp=standing.pelvis_kp * scale,
            pelvis_kd=standing.pelvis_kd * scale,
            torso_kp=standing.torso_kp * scale,
            torso_kd=standing.torso_kd * scale,
            foot_kp=standing.foot_kp * scale,
            foot_kd=standing.foot_kd * scale,
        )
    initial_qpos = (
        None
        if request.initial_qpos is None
        else np.asarray(request.initial_qpos, dtype=np.float64).copy()
    )
    if any(variation.initial_root_offset_m):
        if initial_qpos is None:
            raise ValueError("Initial-pose variation requires explicit initial_qpos")
        initial_qpos[:3] += np.asarray(variation.initial_root_offset_m)
    trajectory: TargetProvider = request.trajectory
    if variation.phase_timing_scale != 1.0:
        trajectory = _PhaseTimingWarp(
            request.trajectory,
            request.config.duration_s,
            variation.phase_timing_scale,
        )
    return replace(
        request,
        request_id=f"{request.request_id}:{variation.variation_id}",
        model_xml=None,
        model_mjz=_mutated_mjz(request, variation),
        trajectory=trajectory,
        config=replace(request.config, gains=varied_gains, standing=standing),
        initial_qpos=initial_qpos,
    )


def certify_robustness(
    request: CandidateCertificationRequest,
    *,
    engine: CertificationEngine | None = None,
    variations: tuple[RobustnessVariation, ...] | None = None,
    baseline: CertificationResult | None = None,
) -> RobustnessCertificationResult:
    evaluator = engine or CertificationEngine()
    baseline_result = baseline or evaluator.certify(request)
    if baseline_result.candidate_id != request.simulation.request_id:
        raise ValueError("robustness baseline does not bind the simulation request")
    selected = variations or calibrated_variations()
    varied_requests = tuple(
        (variation, vary_simulation_request(request.simulation, variation))
        for variation in selected
    )
    results = tuple(
        (
            variation,
            evaluator.certify(
                replace(
                    request,
                    simulation=simulation,
                    expected_model_hash=model_source_hash(simulation),
                )
            ),
        )
        for variation, simulation in varied_requests
    )
    return RobustnessCertificationResult(baseline=baseline_result, variations=results)
