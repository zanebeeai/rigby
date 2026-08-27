"""Turn an uploaded model into a measured ``RobotMorphologyV1``.

The output of this module is the only body-specific thing in the system. Every
magnitude-neutral term in a schema program -- ``distal``, ``fast``, ``wide`` --
resolves through ``RobotScaleV1``, and every Figure and Ground role resolves
through the derived sites. Get this wrong and nothing downstream can be right;
get it right and the semantic layer never has to know what it is driving.

Sites are *derived*, never authored. That is the point of difference from the v2
rig manifest, whose ``semantic_sites`` are written by hand for one humanoid. Each
site here carries a ``derivation`` string naming the rule that produced it, so a
site in the wrong place can be traced to the rule that put it there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from rigby_v2.contracts import Vec3

from ..contracts import (
    DirectionV1,
    EffectorKind,
    EffectorV1,
    JointKind,
    JointRole,
    KinematicChainV1,
    MorphologyClass,
    RobotJointV1,
    RobotMorphologyV1,
    RobotScaleV1,
    RobotSiteV1,
    SiteSemantic,
)
from ..errors import GeneralFailureCode, MorphologyError
from ..ingest.loader import LoadedModel
from . import envelope, frames, measure
from .graph import EffectorCluster, KinematicGraph


REACH_SAMPLES = 4096
REACH_SEED = 20260824

# Only evidence available for a camera. A lens has no mechanical signature, so
# unlike the gripper test this one genuinely cannot be behavioural -- and the
# site records that its derivation was a name hint rather than a measurement.
_SENSOR_NAME_HINTS = ("camera", "cam", "sensor", "lidar", "depth", "rgb", "imu")

_JOINT_KINDS = {
    int(mujoco.mjtJoint.mjJNT_HINGE): JointKind.HINGE,
    int(mujoco.mjtJoint.mjJNT_SLIDE): JointKind.SLIDE,
    int(mujoco.mjtJoint.mjJNT_BALL): JointKind.BALL,
    int(mujoco.mjtJoint.mjJNT_FREE): JointKind.FREE,
}


@dataclass(frozen=True, slots=True)
class ChainMeasurement:
    cluster: EffectorCluster
    chain_id: str
    bodies: tuple[int, ...]
    joints: tuple[int, ...]
    tip_body: int
    closure: measure.ClosureEvidence
    kind: EffectorKind


def _vec3(values: np.ndarray) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _looks_like_a_sensor(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SENSOR_NAME_HINTS)


def _classify_effector(
    graph: KinematicGraph, cluster: EffectorCluster, closure: measure.ClosureEvidence
) -> EffectorKind:
    """Decide what the thing on the end of this chain is.

    Order matters. Closure is checked first and wins outright, because a moving
    jaw is a gripper whatever it is called. Only a cluster that demonstrably does
    not close can fall through to the naming hint for sensors.
    """

    if closure.closes:
        # Count digits, not member bodies. A digit is a part with a surface that
        # can press on something; a massless link marking the tool centre point
        # is a frame. Counting frames made a two-finger hand with a grasp-target
        # marker between the jaws come out as a three-fingered one.
        digits = sum(
            1
            for body in cluster.member_bodies
            if graph.collidable_geoms_of_body(body)
        )
        return EffectorKind.PARALLEL_JAW if digits == 2 else EffectorKind.MULTIFINGER
    if all(
        _looks_like_a_sensor(graph.body_names[body]) for body in cluster.member_bodies
    ):
        return EffectorKind.SENSOR
    return EffectorKind.TOOL_TIP


def _chain_root(graph: KinematicGraph, chain: ChainMeasurement) -> int:
    """The first body on the chain that actually moves.

    Everything above it is welded to the world, so it, not the base link, is the
    point the chain's reachable set is centred on.
    """

    for body in chain.bodies:
        if graph.joints_of_body(body):
            return body
    return chain.bodies[-1]


def _chain_identity(graph: KinematicGraph, cluster: EffectorCluster) -> str:
    """A stable id derived from the model, never from a lookup table."""

    return graph.body_names[cluster.attach_body]


def _measure_chains(
    graph: KinematicGraph, base_qpos: np.ndarray
) -> tuple[ChainMeasurement, ...]:
    measured: list[ChainMeasurement] = []
    for cluster in graph.clusters:
        bodies = graph.chain_bodies(cluster)
        # A leaf reached without passing a single actuated joint is a coordinate
        # frame, not a chain. ROS-Industrial descriptions are full of them -- the
        # KUKA package hangs `base`, `flange` and `tool0` off fixed joints purely
        # so other tools have something to attach to -- and treating one as a
        # chain produces a chain with no joints, which is not a thing that can be
        # measured or moved.
        if not graph.joints_on_path(bodies):
            continue
        closure = measure.measure_closure(
            graph,
            cluster.member_bodies,
            cluster.interior_joints,
            base_qpos=base_qpos,
        )
        kind = _classify_effector(graph, cluster, closure)
        measured.append(
            ChainMeasurement(
                cluster=cluster,
                chain_id=_chain_identity(graph, cluster),
                bodies=bodies,
                joints=graph.joints_on_path(bodies),
                tip_body=cluster.member_bodies[0],
                closure=closure,
                kind=kind,
            )
        )
    return tuple(measured)


def _derive_sites(
    graph: KinematicGraph,
    chains: tuple[ChainMeasurement, ...],
    base_qpos: np.ndarray,
    up: np.ndarray,
) -> tuple[RobotSiteV1, ...]:
    model = graph.model
    data = mujoco.MjData(model)
    data.qpos[:] = base_qpos
    mujoco.mj_kinematics(model, data)

    sites: list[RobotSiteV1] = [
        RobotSiteV1(
            name="robot_base",
            body=graph.body_names[graph.base_body],
            semantic=SiteSemantic.BASE,
            position_m=_vec3(np.zeros(3)),
            derivation="base.root_body_origin.v1",
        )
    ]

    for chain in chains:
        prefix = chain.chain_id
        attach_name = graph.body_names[chain.cluster.attach_body]

        if chain.kind is EffectorKind.SENSOR:
            body = chain.cluster.member_bodies[0]
            sites.append(
                RobotSiteV1(
                    name=f"{prefix}_gaze",
                    body=graph.body_names[body],
                    semantic=SiteSemantic.GAZE,
                    position_m=_vec3(np.zeros(3)),
                    derivation="gaze.sensor_name_hint.v1",
                )
            )
            continue

        # The tip goes at the far end of the distal link along the direction the
        # link actually points, not at its origin -- on most robots the origin
        # sits at the joint, which is the wrong end entirely.
        tip_body = chain.tip_body
        axis = _distal_axis(graph, chain, data)
        local = measure.farthest_geom_point(graph, tip_body, axis, base_qpos)
        sites.append(
            RobotSiteV1(
                name=f"{prefix}_tip",
                body=graph.body_names[tip_body],
                semantic=SiteSemantic.TIP,
                position_m=_vec3(local),
                derivation="tip.distal_geom_extreme.v1",
            )
        )

        if chain.kind in (EffectorKind.PARALLEL_JAW, EffectorKind.MULTIFINGER):
            centre = _grasp_centre(graph, chain, base_qpos)
            sites.append(
                RobotSiteV1(
                    name=f"{prefix}_grasp_center",
                    body=attach_name,
                    semantic=SiteSemantic.GRASP_CENTER,
                    position_m=_vec3(centre),
                    derivation="grasp_center.half_closed_member_centroid.v1",
                )
            )
            for index, member in enumerate(chain.cluster.member_bodies):
                member_axis = _distal_axis(graph, chain, data)
                contact_local = measure.farthest_geom_point(
                    graph, member, member_axis, base_qpos
                )
                sites.append(
                    RobotSiteV1(
                        name=f"{prefix}_contact_{index}",
                        body=graph.body_names[member],
                        semantic=SiteSemantic.CONTACT,
                        position_m=_vec3(contact_local),
                        derivation="contact.member_distal_extreme.v1",
                    )
                )

        if len(chain.bodies) >= 2:
            sites.append(
                RobotSiteV1(
                    name=f"{prefix}_wrist",
                    body=graph.body_names[chain.bodies[-1]],
                    semantic=SiteSemantic.JOINT,
                    position_m=_vec3(np.zeros(3)),
                    derivation="wrist.last_chain_body_origin.v1",
                )
            )

    return tuple(sites)


def _distal_axis(
    graph: KinematicGraph, chain: ChainMeasurement, data: mujoco.MjData
) -> np.ndarray:
    """Which way the end of this chain points, in world coordinates at rest."""

    attach = chain.cluster.attach_body
    parent = graph.parent(attach)
    direction = np.array(data.xpos[attach], dtype=float) - np.array(
        data.xpos[parent], dtype=float
    )
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        matrix = np.array(data.xmat[attach], dtype=float).reshape(3, 3)
        return matrix[:, 2]
    return direction / norm


def _grasp_centre(
    graph: KinematicGraph, chain: ChainMeasurement, base_qpos: np.ndarray
) -> np.ndarray:
    """Centroid of the closing members at half stroke, in the attach body frame.

    Half stroke rather than fully open or fully closed: that is where an object
    of typical size for this gripper would sit.
    """

    model = graph.model
    data = mujoco.MjData(model)
    data.qpos[:] = base_qpos
    for joint in chain.cluster.interior_joints:
        low, high = measure.joint_range(model, joint)
        data.qpos[int(model.jnt_qposadr[joint])] = (low + high) / 2.0
    mujoco.mj_kinematics(model, data)

    points = [
        np.array(data.xpos[member], dtype=float)
        for member in chain.cluster.member_bodies
    ]
    centroid = np.mean(points, axis=0)
    attach = chain.cluster.attach_body
    attach_pos = np.array(data.xpos[attach], dtype=float)
    attach_mat = np.array(data.xmat[attach], dtype=float).reshape(3, 3)
    return attach_mat.T @ (centroid - attach_pos)


def _joint_roles(
    graph: KinematicGraph,
    chains: tuple[ChainMeasurement, ...],
    base_qpos: np.ndarray,
    reach_radius: float,
) -> dict[int, tuple[JointRole, measure.JointMotion]]:
    grip_joints = {
        joint
        for chain in chains
        if chain.closure.closes
        for joint in chain.cluster.interior_joints
    }

    owning_chain: dict[int, ChainMeasurement] = {}
    for chain in chains:
        for joint in chain.joints:
            owning_chain.setdefault(joint, chain)
        for joint in chain.cluster.interior_joints:
            owning_chain.setdefault(joint, chain)

    roles: dict[int, tuple[JointRole, measure.JointMotion]] = {}
    live_by_chain: dict[str, list[int]] = {}

    for joint in graph.actuated_joints:
        chain = owning_chain.get(joint)
        tip_body = chain.tip_body if chain else graph.base_body
        motion = measure.measure_joint_motion(
            graph,
            joint,
            tip_body,
            base_qpos=base_qpos,
            context_joints=chain.joints if chain else (),
            seed=REACH_SEED,
        )

        if joint in grip_joints:
            roles[joint] = (JointRole.GRIP, motion)
            continue

        low, high = measure.joint_range(graph.model, joint)
        if high - low < 1e-6 or (
            motion.translation_m < 1e-5 and motion.rotation_rad < 1e-4
        ):
            roles[joint] = (JointRole.IMMOBILE, motion)
            continue

        if motion.translation_m >= measure.MAJOR_TRANSLATION_FRACTION * reach_radius:
            roles[joint] = (JointRole.MAJOR_POSITION, motion)
        elif motion.rotation_rad >= measure.ORIENT_MIN_ROTATION_RAD:
            roles[joint] = (JointRole.WRIST_ORIENT, motion)
        else:
            roles[joint] = (JointRole.IMMOBILE, motion)

        if chain is not None and roles[joint][0] in (
            JointRole.MAJOR_POSITION,
            JointRole.WRIST_ORIENT,
        ):
            live_by_chain.setdefault(chain.chain_id, []).append(joint)

    # Redundancy is a property of a whole chain rather than of any one joint, so
    # it is resolved only after every joint has a tentative role -- and it
    # overrides that role, because a joint that adds nothing to the reachable
    # pose set is redundant no matter how far it happens to swing the tip.
    for chain in chains:
        candidates = tuple(live_by_chain.get(chain.chain_id, ()))
        redundant = measure.redundant_joints(
            graph,
            chain.tip_body,
            candidates,
            length_scale=reach_radius,
            seed=REACH_SEED,
        )
        for joint in redundant:
            roles[joint] = (JointRole.REDUNDANT, roles[joint][1])

    return roles


def _build_scale(
    graph: KinematicGraph,
    chains: tuple[ChainMeasurement, ...],
    roles: dict[int, tuple[JointRole, measure.JointMotion]],
    samples: dict[int, np.ndarray],
    base_position: np.ndarray,
) -> RobotScaleV1:
    model = graph.model

    all_tips = np.concatenate([values for values in samples.values()], axis=0)
    offsets = np.linalg.norm(all_tips - base_position, axis=1)
    reach_radius = float(offsets.max())

    segment_lengths: list[float] = []
    data = mujoco.MjData(model)
    data.qpos[:] = measure.neutral_qpos(model)
    mujoco.mj_kinematics(model, data)
    for chain in chains:
        for parent, child in zip(chain.bodies, chain.bodies[1:]):
            length = float(
                np.linalg.norm(
                    np.array(data.xpos[child]) - np.array(data.xpos[parent])
                )
            )
            if length > 1e-6:
                segment_lengths.append(length)
    characteristic = float(np.median(segment_lengths)) if segment_lengths else reach_radius

    # Tip speed a positioning joint produces at its velocity limit, taken at half
    # of it: "neutral" should be a comfortable pace, not the maximum the hardware
    # can survive.
    speeds: list[float] = []
    for joint, (role, motion) in roles.items():
        if role is not JointRole.MAJOR_POSITION:
            continue
        low, high = measure.joint_range(model, joint)
        span = high - low
        if span < 1e-9:
            continue
        moment_arm = motion.translation_m / span
        speeds.append(_velocity_limit(model, joint) * moment_arm)
    neutral_speed = 0.5 * float(np.median(speeds)) if speeds else 0.1

    efforts = [
        float(abs(model.jnt_actfrcrange[joint][1]))
        for joint, (role, _) in roles.items()
        if role is JointRole.MAJOR_POSITION and model.jnt_actfrclimited[joint]
    ]
    payload = (
        max(0.01, min(efforts) / (9.81 * max(reach_radius, 1e-6))) if efforts else 0.01
    )

    base_geoms = graph.geoms_of_body(graph.base_body)
    footprint = 0.05
    if base_geoms:
        footprint = max(
            0.01, 2.0 * float(max(model.geom_size[index][0] for index in base_geoms))
        )

    return RobotScaleV1(
        reach_radius_m=round(reach_radius, 6),
        characteristic_length_m=round(min(characteristic, reach_radius), 6),
        neutral_speed_mps=round(max(neutral_speed, 1e-4), 6),
        base_footprint_m=round(footprint, 6),
        payload_kg=round(payload, 6),
        total_mass_kg=round(float(model.body_mass.sum()), 6),
        workspace_centroid_m=_vec3(all_tips.mean(axis=0)),
        reach_samples=int(all_tips.shape[0]),
    )


_VELOCITY_LIMITS: dict[int, dict[int, float]] = {}


def register_velocity_limits(model: mujoco.MjModel, limits: dict[int, float]) -> None:
    """Attach URDF velocity limits, which MuJoCo does not carry on a joint."""

    _VELOCITY_LIMITS[id(model)] = dict(limits)


def _velocity_limit(model: mujoco.MjModel, joint: int) -> float:
    registered = _VELOCITY_LIMITS.get(id(model), {})
    if joint in registered and registered[joint] > 0.0:
        return registered[joint]
    # No declared limit: assume the joint can traverse its whole range in one
    # second. Recorded rather than hidden -- the manifest says which joints had a
    # declared limit and which were assumed.
    low, high = measure.joint_range(model, joint)
    return max(1e-3, (high - low))


def _self_collision_pairs(
    graph: KinematicGraph, chains: tuple[ChainMeasurement, ...]
) -> tuple[tuple[str, str], ...]:
    """Body pairs that could plausibly touch, excluding parent-child neighbours.

    Adjacent links are always in contact at the joint; listing them would drown
    the real signal. What is left is the set worth guarding with a collision
    objective during IK.
    """

    model = graph.model
    moving = [
        body
        for body in range(1, model.nbody)
        if graph.collidable_geoms_of_body(body)
    ]
    adjacency = {
        (min(body, graph.parent(body)), max(body, graph.parent(body)))
        for body in range(1, model.nbody)
    }

    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(moving):
        for second in moving[index + 1 :]:
            if (min(first, second), max(first, second)) in adjacency:
                continue
            pairs.append(
                tuple(sorted((graph.body_names[first], graph.body_names[second])))
            )
    return tuple(sorted(set(pairs)))




def analyze(loaded: LoadedModel, *, velocity_limits: dict[str, float] | None = None) -> RobotMorphologyV1:
    """Measure a compiled model and describe what kind of robot it is."""

    graph = KinematicGraph(loaded.model)
    model = loaded.model
    base_qpos = measure.neutral_qpos(model)

    if velocity_limits:
        by_index = {}
        for index, name in enumerate(graph.joint_names):
            if name in velocity_limits:
                by_index[index] = float(velocity_limits[name])
        register_velocity_limits(model, by_index)

    chains = _measure_chains(graph, base_qpos)
    if not chains:
        raise MorphologyError(
            GeneralFailureCode.NO_EFFECTOR,
            "No chain terminates in anything that could act on the world",
        )

    data = mujoco.MjData(model)
    data.qpos[:] = base_qpos
    mujoco.mj_kinematics(model, data)
    base_position = np.array(data.xpos[graph.base_body], dtype=float)

    up = frames.gravity_up(np.array(model.opt.gravity, dtype=float))

    # Sites are derived before the workspace is sampled, because the envelope has
    # to be measured at the point the chain is actually steered by. Sampling the
    # tip body instead and then aiming at a grasp centre puts the two a whole
    # palm-length apart, and the robot's own rest position reads as unreachable.
    sites = _derive_sites(graph, chains, base_qpos, up)
    probes = _steering_probes(graph, chains, sites)
    chain_joints = {chain.chain_id: chain.joints for chain in chains}
    samples = measure.sample_reach(
        graph, probes, chain_joints, samples=REACH_SAMPLES, seed=REACH_SEED
    )

    all_tips = np.concatenate(list(samples.values()), axis=0)
    reach_probe = float(np.linalg.norm(all_tips - base_position, axis=1).max())

    # Symmetry is measured before the frame, because a mirror plane is the
    # strongest evidence there is about which way a two-armed robot faces.
    closing_chains = [chain for chain in chains if chain.closure.closes]
    symmetry = None
    if len(closing_chains) == 2:
        first, second = closing_chains
        length = min(len(first.bodies), len(second.bodies))
        symmetry = frames.measure_symmetry(
            left_chain_id=first.chain_id,
            right_chain_id=second.chain_id,
            left_positions=np.array(
                [data.xpos[body] for body in first.bodies[-length:]], dtype=float
            ),
            right_positions=np.array(
                [data.xpos[body] for body in second.bodies[-length:]], dtype=float
            ),
            up=up,
            scale_m=reach_probe,
        )

    intrinsic = frames.build_intrinsic_frame(
        gravity=np.array(model.opt.gravity, dtype=float),
        tip_samples=all_tips,
        base_position=base_position,
        mirror_normal=(
            np.array(
                [
                    symmetry.plane_normal.x,
                    symmetry.plane_normal.y,
                    symmetry.plane_normal.z,
                ],
                dtype=float,
            )
            if symmetry is not None
            else None
        ),
    )

    roles = _joint_roles(graph, chains, base_qpos, reach_probe)
    scale = _build_scale(graph, chains, roles, samples, base_position)
    sites_by_body: dict[str, list[RobotSiteV1]] = {}
    for site in sites:
        sites_by_body.setdefault(site.body, []).append(site)

    effort_floor = measure.effort_floor(
        model, {joint: _velocity_limit(model, joint) for joint in graph.actuated_joints}
    )
    joints = tuple(
        RobotJointV1(
            name=graph.joint_names[joint],
            kind=_JOINT_KINDS[int(model.jnt_type[joint])],
            body=graph.body_names[int(model.jnt_bodyid[joint])],
            axis=_axis_direction(model, joint),
            minimum=measure.joint_range(model, joint)[0],
            maximum=measure.joint_range(model, joint)[1],
            velocity_limit=_velocity_limit(model, joint),
            effort_limit=_effort_limit(model, joint, effort_floor),
            effort_declared=_effort_declared(model, joint),
            role=roles[joint][0],
            tip_translation_m=round(roles[joint][1].translation_m, 6),
            tip_rotation_rad=round(roles[joint][1].rotation_rad, 6),
        )
        for joint in graph.actuated_joints
    )

    fallback_front = np.array(
        [intrinsic.front.x, intrinsic.front.y, intrinsic.front.z], dtype=float
    )
    kinematic_chains = tuple(
        _build_chain(
            graph,
            chain,
            roles,
            samples[chain.chain_id],
            data,
            up=up,
            fallback_front=fallback_front,
        )
        for chain in chains
    )

    effectors = tuple(
        _build_effector(graph, chain, sites_by_body) for chain in chains
    )

    grasping = [effector for effector in effectors if effector.can_grasp]
    positioning = max((chain.positioning_dof for chain in kinematic_chains), default=0)
    if len(grasping) >= 2:
        morphology_class = MorphologyClass.FIXED_BASE_BIMANUAL
    elif positioning < 2 and grasping:
        # Nothing to position with, but something to hold with.
        morphology_class = MorphologyClass.DEXTEROUS_EFFECTOR
    else:
        morphology_class = MorphologyClass.FIXED_BASE_ARM

    return RobotMorphologyV1(
        robot_id=loaded.robot_id,
        morphology_class=morphology_class,
        base_body=graph.body_names[graph.base_body],
        joints=joints,
        chains=kinematic_chains,
        effectors=effectors,
        sites=sites,
        intrinsic_frame=intrinsic,
        scale=scale,
        symmetry=symmetry,
        self_collision_pairs=_self_collision_pairs(graph, chains),
    )


def _steering_probes(
    graph: KinematicGraph,
    chains: tuple[ChainMeasurement, ...],
    sites: tuple[RobotSiteV1, ...],
) -> dict[str, tuple[int, np.ndarray]]:
    """The point on each chain whose reachable set defines that chain's workspace.

    A grasping chain is steered by its grasp centre, a rigid one by its tip, a
    sensor by its gaze origin -- the same choice grounding makes, so the envelope
    and the targets are measured against the same point.
    """

    by_name = {site.name: site for site in sites}
    wanted_order = {
        EffectorKind.PARALLEL_JAW: SiteSemantic.GRASP_CENTER,
        EffectorKind.MULTIFINGER: SiteSemantic.GRASP_CENTER,
        EffectorKind.SENSOR: SiteSemantic.GAZE,
        EffectorKind.TOOL_TIP: SiteSemantic.TIP,
    }

    probes: dict[str, tuple[int, np.ndarray]] = {}
    for chain in chains:
        wanted = wanted_order[chain.kind]
        chosen = next(
            (
                site
                for site in sites
                if site.name.startswith(f"{chain.chain_id}_")
                and site.semantic is wanted
            ),
            None,
        )
        if chosen is None:
            chosen = next(
                site
                for site in sites
                if site.name.startswith(f"{chain.chain_id}_")
            )
        body = mujoco.mj_name2id(graph.model, mujoco.mjtObj.mjOBJ_BODY, chosen.body)
        local = np.array(
            [chosen.position_m.x, chosen.position_m.y, chosen.position_m.z],
            dtype=float,
        )
        probes[chain.chain_id] = (body, local)
    return probes


def _build_chain(
    graph: KinematicGraph,
    chain: ChainMeasurement,
    roles: dict[int, tuple[JointRole, measure.JointMotion]],
    tip_samples: np.ndarray,
    data: mujoco.MjData,
    *,
    up: np.ndarray,
    fallback_front: np.ndarray,
) -> KinematicChainV1:
    root_body = _chain_root(graph, chain)
    origin = np.array(data.xpos[root_body], dtype=float)
    centroid = tip_samples.mean(axis=0)

    # The envelope is measured in the same frame grounding will read it in, so
    # the two can never disagree about which way "out" is.
    out, side = envelope.working_frame(centroid, origin, up, fallback_front)
    grid = envelope.build_envelope(tip_samples, origin, out, side, up)
    inner = envelope.build_inner_envelope(tip_samples, origin, out, side, up)

    return KinematicChainV1(
        chain_id=chain.chain_id,
        bodies=tuple(graph.body_names[body] for body in chain.bodies),
        joints=tuple(graph.joint_names[joint] for joint in chain.joints),
        tip_body=graph.body_names[chain.bodies[-1]],
        root_body=graph.body_names[root_body],
        workspace_centroid_m=_vec3(centroid),
        working_direction=DirectionV1(
            x=float(out[0]), y=float(out[1]), z=float(out[2])
        ),
        reach_envelope_m=envelope.flatten(grid),
        reach_inner_m=envelope.flatten(inner),
        envelope_azimuth_bins=envelope.AZIMUTH_BINS,
        envelope_elevation_bins=envelope.ELEVATION_BINS,
        reach_radius_m=round(
            float(np.linalg.norm(tip_samples - origin, axis=1).max()), 6
        ),
        positioning_dof=sum(
            1 for joint in chain.joints if roles[joint][0] is JointRole.MAJOR_POSITION
        ),
        orienting_dof=sum(
            1 for joint in chain.joints if roles[joint][0] is JointRole.WRIST_ORIENT
        ),
    )



def _effort_limit(
    model: mujoco.MjModel, joint: int, floor: np.ndarray
) -> float:
    """The joint's torque limit: what it declares, or what it must be able to hold.

    A declared limit is taken as given -- it is the installer's statement about
    the hardware and outranks anything derivable. When the source declares none,
    this returns the measured lower bound instead, and ``effort_declared`` is
    false so that nothing downstream mistakes it for a limit.
    """

    if _effort_declared(model, joint):
        return max(1e-3, float(abs(model.jnt_actfrcrange[joint][1])))

    dof = int(model.jnt_dofadr[joint])
    return max(1e-3, float(floor[dof]) * measure.EFFORT_DYNAMIC_MARGIN)


def _effort_declared(model: mujoco.MjModel, joint: int) -> bool:
    """Whether the source actually stated a torque limit for this joint."""

    return bool(model.jnt_actfrclimited[joint]) and (
        float(abs(model.jnt_actfrcrange[joint][1])) > 0.0
    )

def _axis_direction(model: mujoco.MjModel, joint: int) -> DirectionV1:
    axis = np.array(model.jnt_axis[joint], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    axis = axis / norm
    return DirectionV1(x=float(axis[0]), y=float(axis[1]), z=float(axis[2]))


def _build_effector(
    graph: KinematicGraph,
    chain: ChainMeasurement,
    sites_by_body: dict[str, list[RobotSiteV1]],
) -> EffectorV1:
    member_names = tuple(
        graph.body_names[body] for body in chain.cluster.member_bodies
    )
    attach_name = graph.body_names[chain.cluster.attach_body]

    # Which joints move each member, read off the tree. Everything below the
    # attachment point and on the way to that member -- so a multi-phalange
    # finger contributes all of its joints and a single-slide jaw contributes
    # one, without either case being special-cased.
    interior = set(chain.cluster.interior_joints)
    member_joints: list[tuple[str, ...]] = []
    for body in chain.cluster.member_bodies:
        driving = [
            graph.joint_names[joint]
            for ancestor in graph.path_from_base(body)
            for joint in graph.joints_of_body(ancestor)
            if joint in interior
        ]
        seen: set[str] = set()
        ordered = tuple(
            name for name in driving if not (name in seen or seen.add(name))
        )
        member_joints.append(ordered)
    member_joint_names = tuple(member_joints)

    # Which end of each member's travel extends it. Swept one member at a time
    # against the body it hangs off, so a finger that straightens and a jaw that
    # slides open are the same measurement rather than two special cases.
    extends_upper: list[bool] = []
    probe = mujoco.MjData(graph.model)
    attach_body = chain.cluster.attach_body
    for body, joint_names in zip(chain.cluster.member_bodies, member_joint_names):
        joints = [
            mujoco.mj_name2id(graph.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        reach: list[float] = []
        for end in (0, 1):
            probe.qpos[:] = graph.model.qpos0
            for joint in joints:
                if joint < 0:
                    continue
                limits = measure.joint_range(graph.model, joint)
                probe.qpos[graph.model.jnt_qposadr[joint]] = limits[end]
            mujoco.mj_kinematics(graph.model, probe)
            reach.append(
                float(
                    np.linalg.norm(
                        np.asarray(probe.xpos[body]) - np.asarray(probe.xpos[attach_body])
                    )
                )
            )
        extends_upper.append(reach[1] > reach[0])
    member_extension = tuple(extends_upper)

    relevant = set(member_names) | {attach_name}
    site_names = tuple(
        sorted(
            site.name
            for body in relevant
            for site in sites_by_body.get(body, ())
        )
    )
    if not site_names:
        site_names = ("robot_base",)

    if chain.kind in (EffectorKind.PARALLEL_JAW, EffectorKind.MULTIFINGER):
        groups = tuple(
            tuple(member_names[index] for index in group)
            for group in chain.closure.opposition_groups
        )
        aperture = max(chain.closure.open_distance_m, 1e-4)
        return EffectorV1(
            name=f"{chain.chain_id}_effector",
            kind=chain.kind,
            chain_id=chain.chain_id,
            tip_body=graph.body_names[chain.tip_body],
            member_bodies=member_names,
            grip_joints=tuple(
                graph.joint_names[joint] for joint in chain.cluster.interior_joints
            ),
            opposition_groups=groups,
            member_joints=member_joint_names,
            member_extends_toward_upper=member_extension,
            max_aperture_m=round(aperture, 6),
            closes_toward_upper=bool(chain.closure.drive_to_upper),
            site_names=site_names,
        )

    return EffectorV1(
        name=f"{chain.chain_id}_effector",
        kind=chain.kind,
        chain_id=chain.chain_id,
        tip_body=graph.body_names[chain.tip_body],
        member_bodies=(),
        site_names=site_names,
    )
