"""Measured local acceptance demonstrations for articulated object packs.

The routines in this module deliberately command only the canonical human's
actuated joints.  Object generalized coordinates remain at ``qpos0`` in every
controller target and may change only through MuJoCo contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationResult,
    ManifestStatePredicate,
    RobustnessVariation,
    calibrated_variations,
    policy_from_manifests,
    vary_simulation_request,
)
from rigby_core.contracts import ContactEdgeV2
from rigby_v2.evidence.renderer import trace_archive_bytes
from rigby_core.hashing import sha256_bytes
from rigby_v2.rigging import (
    stage_canonical_rig,
    validate_simulated_glb_round_trip,
)
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.scenes.compiler import CompiledScene, load_compiled_scene
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    SimulationResult,
    canonical_standing_config,
    model_source_hash,
    project_qpos_to_rig,
)
from rigby_core.simulation.controller import PDGains


PHYSICS_STEP_S = 1.0 / 240.0


@dataclass(frozen=True)
class GeometryChangeRecord:
    pack_id: str
    before: dict[str, object]
    after: dict[str, object]
    invariant: dict[str, object]


PACK_GEOMETRY_CHANGES = (
    GeometryChangeRecord(
        pack_id="lever_button",
        before={
            "object_position_m": (0.0, -0.72, 1.2),
            "panel_half_depth_m": 0.055,
            "lever_local_y_m": -0.07,
            "button_local_y_m": -0.07,
            "button_half_depth_m": 0.028,
            "neutral_self_overlap_m": {"button_panel": 0.013, "lever_panel": 0.007},
        },
        after={
            "object_position_m": (0.26, -0.58, 1.2),
            "panel_half_depth_m": 0.04,
            "lever_local_y_m": 0.07,
            "button_local_y_m": 0.10,
            "button_half_depth_m": 0.018,
            "full_button_stroke_clearance_m": 0.017,
            "button_surface_world_position_m": (0.40, -0.462, 1.22),
        },
        invariant={
            "button_range_m": (-0.025, 0.0),
            "button_success_threshold_m": -0.02,
            "lever_range_deg": (-45.0, 45.0),
            "lever_success_threshold_deg": 35.0,
        },
    ),
    GeometryChangeRecord(
        pack_id="drawer",
        before={
            "object_position_m": (0.0, -0.72, 0.72),
            "object_rotation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "handle_world_position_m": (0.0, -1.135, 0.74),
        },
        after={
            "object_position_m": (0.3, -0.905, 1.13),
            "object_rotation_wxyz": (0.0, 0.0, 0.0, 1.0),
            "handle_grasp_site_world_position_m": (0.3, -0.410, 1.15),
            "handle_center_world_position_m": (0.3, -0.435, 1.15),
            "handle_to_front_clearance_m": 0.087,
        },
        invariant={
            "drawer_range_m": (0.0, 0.30),
            "drawer_success_threshold_m": 0.22,
            "cabinet_half_size_m": (0.34, 0.28, 0.34),
        },
    ),
)


@dataclass(frozen=True)
class NeutralPackDiagnostic:
    pack_id: str
    joint_excursions: dict[str, float]
    object_object_contact_count: int
    maximum_object_object_force_n: float
    maximum_object_object_penetration_m: float
    predicate_satisfied: bool


@dataclass(frozen=True)
class ContactEvidence:
    first_time_s: float | None
    last_time_s: float | None
    sample_count: int
    geom_pairs: tuple[tuple[str, str], ...]
    maximum_normal_force_n: float
    maximum_penetration_m: float


@dataclass(frozen=True)
class PhysicalDemoEvidence:
    pack_id: str
    selected_predicate: str
    terminal_object_state: float
    maximum_object_state: float
    predicate_satisfied: bool
    contact: ContactEvidence
    qpos_writes_after_initialization: int
    object_actuator_count: int
    object_target_excursion: float
    trace_sha256: str
    simulation: SimulationResult


@dataclass(frozen=True)
class ButtonCertificationAttempt:
    certification: CertificationResult
    raw_evidence: PhysicalDemoEvidence
    request: CandidateCertificationRequest
    robustness_variations: tuple[
        tuple[RobustnessVariation, CertificationResult], ...
    ] = ()

    @property
    def robust(self) -> bool:
        return self.certification.certified and len(self.robustness_variations) == 5 and all(
            result.certified for _, result in self.robustness_variations
        )


def _joint_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"missing joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def _site_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if value < 0:
        raise ValueError(f"missing site {name!r}")
    return int(value)


def _joint_bindings(
    model: mujoco.MjModel, names: frozenset[str]
) -> tuple[tuple[str, int, int, float, float], ...]:
    bindings = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name not in names:
            continue
        lower, upper = model.jnt_range[joint_id]
        bindings.append(
            (
                name,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
                float(lower),
                float(upper),
            )
        )
    if len(bindings) != len(names):
        missing = names - {item[0] for item in bindings}
        raise ValueError(f"missing authored joints: {sorted(missing)}")
    return tuple(bindings)


def _solve_site_position(
    model: mujoco.MjModel,
    *,
    site_name: str,
    target_m: tuple[float, float, float],
    bindings: tuple[tuple[str, int, int, float, float], ...],
    reference: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    base = model.qpos0.copy()
    site = _site_id(model, site_name)
    target = np.asarray(target_m, dtype=np.float64)
    lower = np.asarray([item[3] for item in bindings])
    upper = np.asarray([item[4] for item in bindings])

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[:] = base
        for binding, value in zip(bindings, values, strict=True):
            data.qpos[binding[1]] = value
        mujoco.mj_forward(model, data)
        return np.concatenate(
            (80.0 * (data.site_xpos[site] - target), regularization * (values - reference))
        )

    solved = least_squares(
        residual,
        reference,
        bounds=(lower, upper),
        method="trf",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=1_000,
    )
    residual(solved.x)
    error = float(np.linalg.norm(data.site_xpos[site] - target))
    if error > 1e-5:
        raise RuntimeError(f"authored site target is infeasible by {error:.6g} m")
    qpos = base.copy()
    for binding, value in zip(bindings, solved.x, strict=True):
        qpos[binding[1]] = value
    return qpos, solved.x.copy()


def _quintic_trajectory(
    model: mujoco.MjModel,
    bindings: tuple[tuple[str, int, int, float, float], ...],
    anchors: tuple[tuple[float, np.ndarray], ...],
) -> LinearKeyframeTrajectory:
    duration = anchors[-1][0]
    times = np.arange(0.0, duration + PHYSICS_STEP_S * 0.5, PHYSICS_STEP_S)
    qpos = np.broadcast_to(model.qpos0, (len(times), model.nq)).copy()
    qvel = np.zeros((len(times), model.nv), dtype=np.float64)
    qacc = np.zeros((len(times), model.nv), dtype=np.float64)
    anchor_times = np.asarray([item[0] for item in anchors])
    for frame, time_s in enumerate(times):
        segment = int(np.searchsorted(anchor_times, time_s, side="right") - 1)
        segment = min(max(segment, 0), len(anchors) - 2)
        start_s, start = anchors[segment]
        end_s, end = anchors[segment + 1]
        u = float(np.clip((time_s - start_s) / (end_s - start_s), 0.0, 1.0))
        position_law = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        velocity_law = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / (end_s - start_s)
        acceleration_law = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (end_s - start_s) ** 2
        for _, qpos_adr, dof_adr, _, _ in bindings:
            delta = end[qpos_adr] - start[qpos_adr]
            qpos[frame, qpos_adr] = start[qpos_adr] + delta * position_law
            qvel[frame, dof_adr] = delta * velocity_law
            qacc[frame, dof_adr] = delta * acceleration_law
    return LinearKeyframeTrajectory(times, qpos, qvel, qacc)


def _button_request(compiled: CompiledScene, model: mujoco.MjModel) -> SimulationRequest:
    names = frozenset(
        {
            "left_shoulder_flex",
            "left_shoulder_abduct",
            "left_shoulder_twist",
            "left_elbow_flex",
            "left_forearm_twist",
        }
    )
    bindings = _joint_bindings(model, names)
    neutral = np.asarray([model.qpos0[item[1]] for item in bindings])
    setup, setup_values = _solve_site_position(
        model,
        site_name="left_index_tip",
        target_m=(0.4, -0.40, 1.22),
        bindings=bindings,
        reference=neutral,
        regularization=0.01,
    )
    pressed, _ = _solve_site_position(
        model,
        site_name="left_index_tip",
        target_m=(0.4, -0.486, 1.22),
        bindings=bindings,
        reference=setup_values,
        regularization=0.02,
    )
    trajectory = _quintic_trajectory(
        model,
        bindings,
        (
            (0.0, setup),
            (0.30, setup),
            (1.50, pressed),
            (1.90, pressed),
            (2.60, setup),
            (2.80, setup),
        ),
    )
    return SimulationRequest(
        model_xml=None,
        model_mjz=compiled.mjz_bytes,
        trajectory=trajectory,
        config=SimulationConfig(
            duration_s=2.80,
            gains=PDGains(kp=260.0, kd=26.0, max_pd_torque=60.0, max_control=60.0),
            free_root_joint_name="pelvis_free",
            standing=canonical_standing_config(model),
        ),
        initial_qpos=trajectory.qpos[0],
        initial_qvel=trajectory.qvel[0],
        request_id="acceptance-lever-button-press",
    )


def _contact_evidence(result: SimulationResult, token: str) -> ContactEvidence:
    selected = [
        (frame.time_s, contact)
        for frame in result.trace.contacts
        for contact in frame.contacts
        if token in contact.geom1_name + contact.geom2_name
        and (
            contact.geom1_name.startswith("left_")
            or contact.geom2_name.startswith("left_")
        )
    ]
    return ContactEvidence(
        first_time_s=None if not selected else selected[0][0],
        last_time_s=None if not selected else selected[-1][0],
        sample_count=len(selected),
        geom_pairs=tuple(
            sorted({(item.geom1_name, item.geom2_name) for _, item in selected})
        ),
        maximum_normal_force_n=max(
            (item.normal_force_n for _, item in selected), default=0.0
        ),
        maximum_penetration_m=max(
            (max(0.0, -item.distance_m) for _, item in selected), default=0.0
        ),
    )


def run_button_press_once() -> PhysicalDemoEvidence:
    compiled, model = load_compiled_scene("lever_button")
    request = _button_request(compiled, model)
    result = NativeMujocoRuntime().simulate(request)
    address = _joint_address(model, "obj__control_panel__button_slide")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    trajectory = request.trajectory
    assert isinstance(trajectory, LinearKeyframeTrajectory)
    target_excursion = float(np.ptp(trajectory.qpos[:, address]))
    terminal = float(result.trace.qpos[-1, address])
    return PhysicalDemoEvidence(
        pack_id="lever_button",
        selected_predicate="control_panel.button_pressed",
        terminal_object_state=terminal,
        maximum_object_state=float(np.min(result.trace.qpos[:, address])),
        predicate_satisfied=terminal <= -0.02,
        contact=_contact_evidence(result, "button"),
        qpos_writes_after_initialization=int(
            result.diagnostics["qpos_writes_after_initialization"]
        ),
        object_actuator_count=sum(
            "button" in name or "lever" in name for name in actuator_names
        ),
        object_target_excursion=target_excursion,
        trace_sha256=sha256_bytes(trace_archive_bytes(result.trace)),
        simulation=result,
    )


def certify_button_press(artifact_root: Path) -> ButtonCertificationAttempt:
    artifacts = ContentAddressedArtifactStore(artifact_root)
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "lever_button",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    model = scene.compiled.load_model()
    simulation = _button_request(scene.compiled, model)
    selected = next(
        item for item in scene.manifest.state_predicates if item.name == "button_pressed"
    )
    source_glb = artifacts.read_bytes(rig.manifest.visual_asset)
    rig_xml = artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")

    def export_reimport(result: SimulationResult):
        # glTF animation input accessors are float32.  Quantize only the time
        # coordinate before export so the validator compares the exact values
        # the format can carry; authoritative qpos samples remain untouched.
        gltf_times = np.asarray(result.trace.times_s, dtype=np.float32).astype(
            np.float64
        )
        return validate_simulated_glb_round_trip(
            source_glb=source_glb,
            model_xml=rig_xml,
            rig=rig.manifest,
            times_s=gltf_times,
            qpos=project_qpos_to_rig(model, mujoco.MjModel.from_xml_string(rig_xml), result.trace.qpos),
        )

    request = CandidateCertificationRequest(
        simulation=simulation,
        policy=policy_from_manifests(rig.manifest, scene.manifest),
        predicates=(ManifestStatePredicate(selected),),
        export_reimport=export_reimport,
        planned_contacts=(
            ContactEdgeV2(
                contact_id="button-press-contact",
                body_a="left_hand",
                body_b="obj__control_panel__button",
                start_s=1.17,
                end_s=1.94,
                max_slip_m=0.01,
                max_penetration_m=0.002,
            ),
        ),
        expected_model_hash=scene.compiled.mjz_sha256,
        rig_asset_hash=rig.reference.sha256,
        scene_rig_asset_hash=scene.manifest.rig_asset_hash,
        rig_coordinate_system=rig.manifest.coordinate_system,
        scene_coordinate_system=scene.manifest.coordinate_system,
    )
    certification = CertificationEngine().certify(request)
    raw = run_button_press_once()
    variation_results: list[tuple[RobustnessVariation, CertificationResult]] = []
    if certification.certified:
        contact_windows = {
            "low-gain-fast-phase": (1.175, 1.927),
            "initial-offset-slow-phase": (1.1625, 1.944),
        }
        for variation in calibrated_variations():
            varied_simulation = vary_simulation_request(simulation, variation)
            start_s, end_s = contact_windows.get(
                variation.variation_id, (1.17, 1.94)
            )
            varied_edge = request.planned_contacts[0].model_copy(
                update={"start_s": start_s, "end_s": end_s}
            )
            varied_request = replace(
                request,
                simulation=varied_simulation,
                planned_contacts=(varied_edge,),
                expected_model_hash=model_source_hash(varied_simulation),
            )
            variation_results.append(
                (variation, CertificationEngine().certify(varied_request))
            )
    return ButtonCertificationAttempt(
        certification,
        raw,
        request,
        tuple(variation_results),
    )


def neutral_pack_diagnostic(pack_id: str, duration_s: float = 1.25) -> NeutralPackDiagnostic:
    compiled, model = load_compiled_scene(pack_id)
    times = np.asarray([0.0, duration_s])
    qpos = np.stack((model.qpos0, model.qpos0))
    qvel = np.zeros((2, model.nv), dtype=np.float64)
    result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            model_xml=None,
            model_mjz=compiled.mjz_bytes,
            trajectory=LinearKeyframeTrajectory(times, qpos, qvel, qvel.copy()),
            config=SimulationConfig(
                duration_s=duration_s,
                free_root_joint_name="pelvis_free",
                standing=canonical_standing_config(model),
            ),
            initial_qpos=model.qpos0,
            request_id=f"neutral-{pack_id}",
        )
    )
    object_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): int(
            model.jnt_qposadr[joint_id]
        )
        for joint_id in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith("obj__")
    }
    excursions = {
        str(name): float(np.max(np.abs(result.trace.qpos[:, address] - model.qpos0[address])))
        for name, address in object_joints.items()
    }
    object_contacts = [
        contact
        for frame in result.trace.contacts
        for contact in frame.contacts
        if contact.geom1_name.startswith("obj__") and contact.geom2_name.startswith("obj__")
    ]
    predicate = (
        excursions.get("obj__drawer_unit__drawer_slide", 0.0) >= 0.22
        if pack_id == "drawer"
        else (
            excursions.get("obj__control_panel__button_slide", 0.0) >= 0.02
            or excursions.get("obj__control_panel__lever_hinge", 0.0) >= np.deg2rad(35.0)
        )
    )
    return NeutralPackDiagnostic(
        pack_id=pack_id,
        joint_excursions=excursions,
        object_object_contact_count=len(object_contacts),
        maximum_object_object_force_n=max(
            (item.normal_force_n for item in object_contacts), default=0.0
        ),
        maximum_object_object_penetration_m=max(
            (max(0.0, -item.distance_m) for item in object_contacts), default=0.0
        ),
        predicate_satisfied=predicate,
    )


def run_bounded_drawer_trial() -> PhysicalDemoEvidence:
    """Run the one bounded drawer trial; it is intentionally not certified.

    The legacy probe is retained to demonstrate why terminal state and contact
    gates are both required.  Its collision impulse can transiently push the
    slide beyond the success threshold, but the drawer returns nearly closed
    and penetration remains far outside certification tolerance.
    """

    compiled, model = load_compiled_scene("drawer")
    names = frozenset(
        {
            "left_shoulder_flex",
            "left_shoulder_abduct",
            "left_shoulder_twist",
            "left_elbow_flex",
            "left_forearm_twist",
            "left_wrist_flex",
            "left_wrist_deviation",
        }
    )
    bindings = _joint_bindings(model, names)
    neutral = np.asarray([model.qpos0[item[1]] for item in bindings])
    setup, setup_values = _solve_site_position(
        model,
        site_name="left_index_tip",
        target_m=(0.40, -0.59, 1.15),
        bindings=bindings,
        reference=neutral,
        regularization=0.01,
    )
    pulled, _ = _solve_site_position(
        model,
        site_name="left_index_tip",
        target_m=(0.40, -0.30, 1.15),
        bindings=bindings,
        reference=setup_values,
        regularization=0.05,
    )
    trajectory = _quintic_trajectory(
        model,
        bindings,
        (
            (0.0, setup),
            (0.30, setup),
            (2.30, pulled),
            (2.60, pulled),
            (2.90, pulled),
        ),
    )
    result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            model_xml=None,
            model_mjz=compiled.mjz_bytes,
            trajectory=trajectory,
            config=SimulationConfig(
                duration_s=2.90,
                gains=PDGains(260.0, 26.0, 60.0, 60.0),
                free_root_joint_name="pelvis_free",
                standing=canonical_standing_config(model),
            ),
            initial_qpos=trajectory.qpos[0],
            initial_qvel=trajectory.qvel[0],
            request_id="acceptance-drawer-bounded-trial",
        )
    )
    address = _joint_address(model, "obj__drawer_unit__drawer_slide")
    contact = _contact_evidence(result, "handle")
    terminal = float(result.trace.qpos[-1, address])
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    return PhysicalDemoEvidence(
        pack_id="drawer",
        selected_predicate="drawer_unit.opened",
        terminal_object_state=terminal,
        maximum_object_state=float(np.max(result.trace.qpos[:, address])),
        predicate_satisfied=terminal >= 0.22,
        contact=contact,
        qpos_writes_after_initialization=int(result.diagnostics["qpos_writes_after_initialization"]),
        object_actuator_count=sum("drawer" in name for name in actuator_names),
        object_target_excursion=float(np.ptp(trajectory.qpos[:, address])),
        trace_sha256=sha256_bytes(trace_archive_bytes(result.trace)),
        simulation=result,
    )
