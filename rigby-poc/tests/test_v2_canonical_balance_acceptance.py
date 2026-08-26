from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_v2.rigging import load_canonical_human, xml_for_profile
from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    canonical_standing_config,
    repeat_replay,
)


PROFILES = ("small", "medium", "large")
FOOT_GEOMS = frozenset({"left_foot_collision", "right_foot_collision"})


def _request(
    profile: str,
    *,
    duration_s: float = 5.0,
    initial_qvel: np.ndarray | None = None,
) -> tuple[mujoco.MjModel, SimulationRequest]:
    xml = xml_for_profile(profile)
    model = mujoco.MjModel.from_xml_string(xml)
    request = SimulationRequest(
        model_xml=xml,
        trajectory=ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv)),
        config=SimulationConfig(
            duration_s=duration_s,
            standing=canonical_standing_config(model),
            free_root_joint_name="pelvis_free",
        ),
        # Deliberately leave initial_qpos unset: production must start at the
        # canonical model's own audited rest state, not a test-only pose.
        initial_qpos=None,
        initial_qvel=initial_qvel,
        request_id=f"canonical-balance-{profile}",
    )
    return model, request


def _body_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_name: str,
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data = mujoco.MjData(model)
    output = np.empty((len(qpos), 3), dtype=np.float64)
    for index, value in enumerate(qpos):
        data.qpos[:] = value
        mujoco.mj_forward(model, data)
        output[index] = data.xpos[body_id]
    return output


def _assert_certified_neutral(model: mujoco.MjModel, result) -> None:
    assert result.completed
    assert result.metrics is not None
    metrics = result.metrics
    assert metrics.finite
    assert metrics.root_translation_drift_m <= 0.05
    assert metrics.max_penetration_m <= 0.002
    assert metrics.max_speed_rad_or_m_s <= 20.0
    assert metrics.max_actuator_force <= 200.0
    assert metrics.max_generalized_effort <= 300.0
    assert metrics.max_joint_power <= 2_000.0
    acceleration = np.diff(result.trace.qvel, axis=0) / np.diff(
        result.trace.times_s
    )[:, None]
    assert float(np.max(np.abs(acceleration))) <= 500.0
    assert result.diagnostics["qpos_writes_after_initialization"] == 0
    assert model.nmocap == 0
    assert not any(
        model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD
        for index in range(model.neq)
    )
    # Free-root generalized effort must remain exactly absent.
    assert np.array_equal(
        result.trace.generalized_effort[:, :6],
        np.zeros((len(result.trace.times_s), 6)),
    )

    for frame in result.trace.contacts:
        for contact in frame.contacts:
            names = {contact.geom1_name, contact.geom2_name}
            if "ground" in names and contact.normal_force_n >= 0.05:
                assert names & FOOT_GEOMS

    for body_name, geom_name in (
        ("left_foot", "left_foot_collision"),
        ("right_foot", "right_foot_collision"),
    ):
        positions = _body_positions(model, result.trace.qpos, body_name)
        drift = np.linalg.norm(positions - positions[0], axis=1)
        assert float(np.max(drift)) <= 0.08
        contact_indices = [
            index
            for index, frame in enumerate(result.trace.contacts)
            if any(
                {contact.geom1_name, contact.geom2_name}
                == {"ground", geom_name}
                and contact.normal_force_n >= 0.05
                for contact in frame.contacts
            )
        ]
        assert contact_indices
        planted = positions[contact_indices, :2]
        slip = np.linalg.norm(planted - planted[0], axis=1)
        assert float(np.max(slip)) <= 0.025


@pytest.mark.parametrize("profile", PROFILES)
def test_canonical_rest_is_a_loaded_foot_contact_equilibrium(profile: str) -> None:
    model = load_canonical_human(profile)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert data.ncon == 8
    assert all(data.contact[index].dist > 0.0 for index in range(data.ncon))
    assert abs(float(data.qacc[2])) <= 0.05


@pytest.mark.parametrize("profile", PROFILES)
def test_production_controller_holds_every_profile_for_five_seconds(profile: str) -> None:
    model, request = _request(profile)
    result = NativeMujocoRuntime().simulate(request)

    _assert_certified_neutral(model, result)


def test_medium_neutral_holds_ten_seconds_and_replays_exactly_three_times() -> None:
    model, request = _request("medium", duration_s=10.0)
    report = repeat_replay(NativeMujocoRuntime(), request, runs=3)

    assert report.deterministic
    assert len(report.runs) == 3
    assert all(item.matches for item in report.comparisons)
    for run in report.runs:
        _assert_certified_neutral(model, run)


@pytest.mark.parametrize("profile", ("small", "large"))
def test_scaled_profiles_also_replay_exactly_three_times(profile: str) -> None:
    model, request = _request(profile)
    report = repeat_replay(NativeMujocoRuntime(), request, runs=3)

    assert report.deterministic
    assert len(report.runs) == 3
    for run in report.runs:
        _assert_certified_neutral(model, run)


def test_medium_recovers_from_a_physical_velocity_perturbation() -> None:
    base = load_canonical_human("medium")
    velocity = np.zeros(base.nv, dtype=np.float64)
    velocity[0] = 0.02  # 20 mm/s lateral impulse at the free pelvis
    velocity[4] = 0.02  # simultaneous pitch angular velocity
    model, request = _request("medium", initial_qvel=velocity)

    result = NativeMujocoRuntime().simulate(request)

    _assert_certified_neutral(model, result)
    assert np.linalg.norm(result.trace.qvel[-1, :6]) < 1e-3
