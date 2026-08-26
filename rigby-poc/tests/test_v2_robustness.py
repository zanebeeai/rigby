from __future__ import annotations

import io
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationOutcome,
    CertificationPolicy,
    CertificationResult,
    RobustnessVariation,
    certify_robustness,
    vary_simulation_request,
)
from rigby_v2.rigging.manifest_builder import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    SimulationConfig,
    SimulationRequest,
    model_source_hash,
)


def _request(tmp_path: Path) -> tuple[SimulationRequest, mujoco.MjModel]:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "grasp_place_block",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    model = scene.compiled.load_model()
    return (
        SimulationRequest(
            model_xml=None,
            model_mjz=scene.compiled.mjz_bytes,
            trajectory=ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv)),
            config=SimulationConfig(duration_s=0.2, free_root_joint_name="pelvis_free"),
            initial_qpos=model.qpos0,
            request_id="candidate",
        ),
        model,
    )


def test_variations_change_physical_model_and_preserve_timing_endpoints(tmp_path: Path) -> None:
    request, baseline = _request(tmp_path)
    variation = RobustnessVariation(
        "combined",
        friction_scale=0.9,
        object_mass_scale=1.1,
        controller_gain_scale=0.9,
        initial_root_offset_m=(0.005, 0.0, 0.0),
        phase_timing_scale=1.02,
    )
    varied = vary_simulation_request(request, variation)
    assert varied.model_xml is None and varied.model_mjz is not None
    model = mujoco.MjSpec.from_zip(io.BytesIO(varied.model_mjz)).compile()
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "obj__block__body__box"
    )
    baseline_geom = mujoco.mj_name2id(
        baseline, mujoco.mjtObj.mjOBJ_GEOM, "obj__block__body__box"
    )
    assert model.geom_friction[object_geom, 0] == pytest.approx(
        baseline.geom_friction[baseline_geom, 0] * 0.9
    )
    assert model.body_mass[model.geom_bodyid[object_geom]] == pytest.approx(
        baseline.body_mass[baseline.geom_bodyid[baseline_geom]] * 1.1
    )
    assert np.asarray(varied.initial_qpos)[0] == pytest.approx(
        np.asarray(request.initial_qpos)[0] + 0.005
    )
    np.testing.assert_allclose(
        varied.trajectory.sample(0.0).qpos, request.trajectory.sample(0.0).qpos
    )
    np.testing.assert_allclose(
        varied.trajectory.sample(0.2).qpos, request.trajectory.sample(0.2).qpos
    )


def test_variation_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="friction"):
        RobustnessVariation("unsafe", friction_scale=0.5)
    with pytest.raises(ValueError, match="2 cm"):
        RobustnessVariation("unsafe", initial_root_offset_m=(0.03, 0.0, 0.0))


def test_robustness_rebinds_the_hash_of_each_mutated_model(tmp_path: Path) -> None:
    simulation, _ = _request(tmp_path)
    captured: list[CandidateCertificationRequest] = []

    class CaptureEngine:
        def certify(self, request: CandidateCertificationRequest) -> CertificationResult:
            captured.append(request)
            return CertificationResult(
                outcome=CertificationOutcome.CERTIFIED,
                candidate_id=request.simulation.request_id,
                violations=(),
                predicate_evaluations=(),
                simulation_runs=(),
            )

    certify_robustness(
        CandidateCertificationRequest(
            simulation=simulation,
            policy=CertificationPolicy(require_export_reimport=False),
            expected_model_hash=model_source_hash(simulation),
        ),
        engine=CaptureEngine(),  # type: ignore[arg-type]
        variations=(RobustnessVariation("low-friction", friction_scale=0.9),),
    )

    assert len(captured) == 2
    varied = captured[1]
    assert varied.expected_model_hash == model_source_hash(varied.simulation)
    assert varied.expected_model_hash != captured[0].expected_model_hash
