"""Measure what a skill left behind, and say what the next one admits.

The neutral checker in ``rigby_core.skills.boundary`` compares a boundary
state with an initiation set; this module produces both from a body. The
joints come from the manifest's declared limits. The contact mode comes
from the same measurement the closure controller grips by: contact force on
the effector's members against the object, opposition across the measured
groups. An object on a support with no member touching it is resting; an
object in opposition is held; anything else is free. Nothing here reads a
plan or a commanded finger position.
"""

from __future__ import annotations

import mujoco
import numpy as np
from rigby_core.skills import BoundaryStateV1, ContactMode, InitiationSetV1, JointStateV1

from ..contact.closure import ClosureController
from ..contact.transfer import TransferStart
from ..contracts import EffectorV1, RobotAssetManifestV1
from ..grounding import ik
from ..grounding.workspace import WorkspaceFrame
from ..scenes.environment import SUPPORT_PREFIX


OBJECT_GEOM = "scene_block_geom"
OBJECT_NAME = "cube"


def arm_joint_names(model: mujoco.MjModel, effector: EffectorV1, frame: WorkspaceFrame) -> tuple[str, ...]:
    return ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))


def joint_states(model: mujoco.MjModel, manifest: RobotAssetManifestV1, arm_joints: tuple[str, ...], qpos: np.ndarray, qvel: np.ndarray) -> tuple[JointStateV1, ...]:
    dofs = {dof.joint: dof for dof in manifest.dofs}
    states = []
    for name in arm_joints:
        dof = dofs[name]
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        states.append(JointStateV1(name=dof.name, position=float(qpos[int(model.jnt_qposadr[joint])]), velocity=float(qvel[int(model.jnt_dofadr[joint])]),
                                   minimum=float(dof.minimum), maximum=float(dof.maximum), velocity_limit=float(dof.velocity_limit)))
    return tuple(states)


def contact_state(model: mujoco.MjModel, manifest: RobotAssetManifestV1, effector: EffectorV1, data: mujoco.MjData) -> tuple[ContactMode, dict[str, str], dict[str, str], dict]:
    """The contact mode with the object, what is held by which effector, and
    what rests on which support, from contact forces alone."""

    closure = ClosureController(model, manifest, effector, object_geoms=frozenset({OBJECT_GEOM}))
    forces, penetration, peak = closure.observe(data)
    contacted = tuple(sorted(body for body, force in forces.items() if force >= closure.contact_force_n))
    opposition = closure._opposition_satisfied(contacted)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    supports = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if object_geom not in (int(contact.geom1), int(contact.geom2)) or float(contact.dist) > 0.0:
            continue
        other = int(contact.geom2 if int(contact.geom1) == object_geom else contact.geom1)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        if name.startswith(SUPPORT_PREFIX):
            supports.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or name)
    resource = f"effector:{effector.chain_id}"
    resting = {OBJECT_NAME: sorted(supports)[0]} if supports else {}
    if opposition:
        mode, held = ContactMode.HOLDING, {OBJECT_NAME: resource}
    else:
        mode, held = ContactMode.FREE, {}
    return mode, held, resting, {"contacted_members": contacted, "opposition": opposition, "peak_force_n": float(peak), "penetration_m": float(penetration), "supports": sorted(set(supports))}


def measure_boundary(model: mujoco.MjModel, manifest: RobotAssetManifestV1, effector: EffectorV1, frame: WorkspaceFrame, start: TransferStart, *,
                     belief_age_s: float, owned: tuple[str, ...] = ()) -> tuple[BoundaryStateV1, dict]:
    """The boundary state at ``start`` for ``effector``, and the raw contact measurement it came from."""

    data = mujoco.MjData(model)
    if start.state is not None:
        from rigby_core.simulation.recording import STATE_SPEC

        mujoco.mj_setState(model, data, np.asarray(start.state, dtype=float), STATE_SPEC)
    else:
        data.qpos[:] = start.qpos
        data.qvel[:] = start.qvel
        data.time = start.time_s
    mujoco.mj_forward(model, data)
    arm = arm_joint_names(model, effector, frame)
    mode, held, resting, raw = contact_state(model, manifest, effector, data)
    boundary = BoundaryStateV1(time_s=float(data.time), joints=joint_states(model, manifest, arm, np.array(data.qpos), np.array(data.qvel)),
                               contact_mode=mode, held=held, resting_on=resting, belief_age_s=float(belief_age_s), owned=tuple(owned), manipulator=f"effector:{effector.chain_id}")
    return boundary, raw


def initiation_for(skill_id: str, mode: ContactMode, effector: EffectorV1, *, requires_object: bool, requires_resting: bool = False, max_belief_age_s: float = 5.0,
                   limit_margin_fraction: float = 0.02, speed_fraction: float = 0.05) -> InitiationSetV1:
    """What a transfer-family skill admits at its start: joints inside their
    limits by a margin, the body still, the contact mode it begins in, the
    object held by this effector if it begins holding, the object resting on
    a support if it begins by acquiring it, a belief no older than the
    limit, and this effector and the object not owned elsewhere."""

    resource = f"effector:{effector.chain_id}"
    return InitiationSetV1(skill_id=skill_id, limit_margin_fraction=limit_margin_fraction, speed_fraction=speed_fraction, contact_mode=mode,
                           required_held={OBJECT_NAME: resource} if requires_object else {}, forbidden_held=True,
                           required_resting=() if (requires_object or not requires_resting) else (OBJECT_NAME,), max_belief_age_s=max_belief_age_s,
                           required_resources=(resource, f"object:{OBJECT_NAME}"))
