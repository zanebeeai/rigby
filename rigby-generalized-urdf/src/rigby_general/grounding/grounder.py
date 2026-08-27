"""Turn a body-neutral schema program into an ordinary ``MotionProgramV2``.

This is the only place in the system that knows both languages. Above it,
everything is ordinals and roles; below it, everything is the certified v2
pipeline, unchanged -- compile, simulate, certify, render, judge.

Two rules govern the whole module:

**No model call.** Grounding is arithmetic over measurements. The same schema
program and the same robot must produce byte-identical output every time, or a
primitive certified yesterday describes a motion that no longer exists.

**Fail closed.** A term that will not resolve inside this robot's measured limits
raises :class:`GroundingError` naming the schema and the measurement it broke. It
is never quietly clamped: a shortened reach that still compiles is a motion
nobody asked for, presented as if they had.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from rigby_v2.hashing import content_hash
from rigby_v2.contracts import (
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Vec3,
)

from ..contracts import (
    EffectorKind,
    FrameSource,
    RobotAssetManifestV1,
    SiteSemantic,
)
from ..errors import GeneralFailureCode, GroundingError, RigbyGeneralError
from ..schema.inventory import Requirement, SchemaInventory, capabilities_of
from ..schema.program import (
    BindingRole,
    BoundaryCondition,
    Conformation,
    Contour,
    Deixis,
    MotionSchemaProgramV1,
    PathVector,
    ReferenceFrame,
    SchemaKind,
    SegmentV1,
    Stative,
)
from . import ik, magnitudes
from .workspace import WorkspaceFrame, build_workspace_frame


# How a conformation places its target relative to the chain's working direction.
# Elevation is measured up from horizontal; azimuth around the up axis.
_CONFORMATION_PLACEMENT: dict[Conformation, tuple[float, float]] = {
    Conformation.POINT: (0.0, 0.0),
    Conformation.SURFACE: (0.0, -35.0),
    Conformation.VOLUME: (0.0, -10.0),
    Conformation.LINE: (0.0, 0.0),
    Conformation.AXIS: (0.0, 15.0),
    Conformation.BOUNDARY: (0.0, 0.0),
}

_LINE_HALF_SWEEP_DEG = 38.0
_ARC_APEX_LIFT = 0.28
_OSCILLATION_HALF_SWEEP_DEG = 22.0
_TOWARD_ARRIVAL = 0.55
_AWAY_DEPARTURE = 1.35
_SUBDIVISIONS_PER_SPAN = 5

_PHASE_FOR_BOUNDARY: dict[BoundaryCondition, PhaseKind] = {
    BoundaryCondition.CONTACT_MADE: PhaseKind.ATTACK,
    BoundaryCondition.CONTACT_BROKEN: PhaseKind.RELEASE,
    BoundaryCondition.DIRECTION_REVERSED: PhaseKind.ACTION,
    BoundaryCondition.ORIENTATION_SET: PhaseKind.SETUP,
    BoundaryCondition.DWELL: PhaseKind.HOLD,
    BoundaryCondition.TERMINUS: PhaseKind.ACTION,
}


@dataclass(frozen=True, slots=True)
class GroundedProgram:
    program: MotionProgramV2
    figure_sites: tuple[str, ...]
    waypoint_count: int
    measured_speed_mps: float


@dataclass(frozen=True, slots=True)
class _Binding:
    frame: WorkspaceFrame
    owner: str


def _vec3(values: np.ndarray) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def figure_site_for(manifest: RobotAssetManifestV1, chain_id: str) -> str:
    """Which site an effector is steered by.

    A grasping effector is steered by its grasp centre, not by a fingertip. The
    v2 compiler selects IK joints by walking up the body tree from the target
    site, so aiming at a fingertip enlists the finger joints in the reach and the
    gripper flaps its way across the workspace. The grasp centre sits on the palm
    and stops the walk above the closure joints.
    """

    morphology = manifest.morphology
    effector = next(
        (item for item in morphology.effectors if item.chain_id == chain_id), None
    )
    if effector is None:  # pragma: no cover - defensive
        raise KeyError(f"no effector on chain {chain_id!r}")

    wanted = (
        SiteSemantic.GRASP_CENTER
        if effector.kind in (EffectorKind.PARALLEL_JAW, EffectorKind.MULTIFINGER)
        else SiteSemantic.GAZE
        if effector.kind is EffectorKind.SENSOR
        else SiteSemantic.TIP
    )
    for name in effector.site_names:
        if morphology.site(name).semantic is wanted:
            return name
    return effector.site_names[0]


def _ordered_chains(manifest: RobotAssetManifestV1) -> tuple:
    """Chains in a stable order, grasping ones first.

    ``PRIMARY_EFFECTOR`` has to mean the same thing on every ingest of a robot,
    so the order is derived and sorted rather than discovered.
    """

    morphology = manifest.morphology
    graspable = {
        effector.chain_id
        for effector in morphology.effectors
        if effector.can_grasp
    }
    sensors = {
        effector.chain_id
        for effector in morphology.effectors
        if effector.kind is EffectorKind.SENSOR
    }
    return tuple(
        sorted(
            morphology.chains,
            key=lambda chain: (
                0 if chain.chain_id in graspable else 2 if chain.chain_id in sensors else 1,
                chain.chain_id,
            ),
        )
    )


def _bind_role(
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    role: BindingRole,
    *,
    cache: dict[str, WorkspaceFrame],
) -> _Binding:
    chains = _ordered_chains(manifest)
    morphology = manifest.morphology

    if role is BindingRole.PRIMARY_EFFECTOR:
        chain = chains[0]
    elif role is BindingRole.SECONDARY_EFFECTOR:
        if len(chains) < 2:
            raise RigbyGeneralError(
                GeneralFailureCode.UNAFFORDED_SCHEMA,
                f"{morphology.robot_id} has one chain, so it has no second effector",
                details={"role": role.value, "chains": len(chains)},
            )
        chain = chains[1]
    elif role is BindingRole.SENSOR:
        sensor = next(
            (
                effector
                for effector in morphology.effectors
                if effector.kind is EffectorKind.SENSOR
            ),
            None,
        )
        if sensor is None:
            raise RigbyGeneralError(
                GeneralFailureCode.UNAFFORDED_SCHEMA,
                f"{morphology.robot_id} carries no sensor to aim",
                details={"role": role.value},
            )
        chain = next(
            item for item in morphology.chains if item.chain_id == sensor.chain_id
        )
    else:  # pragma: no cover - guarded by the caller
        raise RigbyGeneralError(
            GeneralFailureCode.INVENTED_BINDING,
            f"role {role.value!r} cannot act as a Figure",
            details={"role": role.value},
        )

    site = figure_site_for(manifest, chain.chain_id)
    if chain.chain_id not in cache:
        cache[chain.chain_id] = build_workspace_frame(
            model, morphology, chain, figure_site=site
        )
    return _Binding(frame=cache[chain.chain_id], owner=f"chain_{chain.chain_id}")


def _ground_anchor(
    segment: SegmentV1, binding: _Binding, manifest: RobotAssetManifestV1
) -> np.ndarray:
    """Where the Ground sits, in world coordinates.

    Talmy's asymmetry made concrete: the Ground supplies the reference and the
    Figure is located against it. On a kinematic tree the base is the ultimate
    Ground, which is why ``BASE`` resolves to the chain's own root rather than to
    anything the schema had to name.
    """

    frame = binding.frame
    role = segment.ground.role

    if role in (BindingRole.BASE, BindingRole.SELF):
        return frame.origin
    if role is BindingRole.WORLD_GROUND:
        return np.array([frame.origin[0], frame.origin[1], 0.0])
    if role is BindingRole.GRAVITY_AXIS:
        return frame.origin
    if role in (BindingRole.FRONT_AXIS, BindingRole.LATERAL_AXIS):
        intrinsic = manifest.morphology.intrinsic_frame
        if intrinsic.source is not FrameSource.OPERATOR_CONFIRMED:
            raise GroundingError(
                f"{manifest.rig_id}: a {role.value} reference needs a confirmed "
                "front direction; nobody has said which way this robot faces",
                schema_key=segment.motion_schema.canonical_key,
                measurement="intrinsic_frame.source",
            )
        return frame.origin
    if role in (
        BindingRole.TARGET_OBJECT,
        BindingRole.SECONDARY_OBJECT,
        BindingRole.SUPPORT_SURFACE,
    ):
        raise GroundingError(
            f"{manifest.rig_id}: {role.value} needs a scene, and none was supplied",
            schema_key=segment.motion_schema.canonical_key,
            measurement="scene",
        )
    if role is BindingRole.SECONDARY_EFFECTOR:
        return frame.origin
    return frame.origin  # pragma: no cover - every role is covered above


def _waypoints(
    segment: SegmentV1,
    binding: _Binding,
    inventory: SchemaInventory,
    manner: magnitudes.ResolvedManner,
) -> list[np.ndarray]:
    """The point sequence a schema describes, in world coordinates.

    Tversky and Lee's finding shapes this directly: a route is a chain of
    segments whose meaning sits at the reorientations, with the straight stretches
    carrying no information. So a schema produces only the points where something
    changes -- start, any reversal or apex, and the terminus -- and interpolation
    fills the rest.
    """

    frame = binding.frame
    motion = segment.motion_schema

    if motion.kind is SchemaKind.STATIVE:
        return _stative_waypoints(segment, frame, manner)

    assert motion.vector and motion.conformation and motion.contour and motion.deixis
    base_fraction = inventory.remove_fraction(segment.region.remove.value)
    fraction = magnitudes.resolve_radius_fraction(
        base_fraction, amplitude_scale=manner.amplitude_scale
    )
    azimuth, elevation = _CONFORMATION_PLACEMENT[motion.conformation]

    if motion.conformation is Conformation.BOUNDARY:
        fraction = min(0.92, max(fraction, 0.88))

    if motion.deixis is Deixis.HITHER:
        azimuth += 180.0
    elif motion.deixis is Deixis.THITHER:
        azimuth += 0.0

    anchor = frame.point(
        radius_fraction=fraction, azimuth_deg=azimuth, elevation_deg=elevation
    )

    if motion.contour is Contour.OSCILLATING:
        return _oscillation_waypoints(frame, fraction, azimuth, elevation, manner, motion)
    if motion.contour is Contour.CIRCULAR:
        return _circle_waypoints(frame, fraction, elevation, manner)
    if motion.conformation is Conformation.LINE:
        return _line_waypoints(frame, fraction, azimuth, elevation, motion)

    start, end = _endpoints_for_vector(frame, motion.vector, anchor, fraction, azimuth, elevation)

    if motion.contour is Contour.ARCED:
        midpoint = (start + end) / 2.0
        apex = midpoint + frame.up * (_ARC_APEX_LIFT * frame.reach_m * manner.amplitude_scale)
        return [start, frame.clamp(apex), end]
    return [start, end]


def _endpoints_for_vector(
    frame: WorkspaceFrame,
    vector: PathVector,
    anchor: np.ndarray,
    fraction: float,
    azimuth: float,
    elevation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the Figure starts and stops, per Talmy's arrival/departure sense."""

    if vector is PathVector.TO:
        return frame.home, anchor
    if vector is PathVector.FROM:
        return anchor, frame.home
    if vector is PathVector.TOWARD:
        partial = frame.point(
            radius_fraction=fraction / max(_TOWARD_ARRIVAL, 1e-6) * _TOWARD_ARRIVAL,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
        )
        return frame.home, frame.clamp(partial)
    if vector is PathVector.AWAY:
        departed = frame.point(
            radius_fraction=min(0.92, fraction * _AWAY_DEPARTURE),
            azimuth_deg=azimuth,
            elevation_deg=elevation,
        )
        return anchor, frame.clamp(departed)
    return frame.home, anchor


def _line_waypoints(
    frame: WorkspaceFrame,
    fraction: float,
    azimuth: float,
    elevation: float,
    motion,
) -> list[np.ndarray]:
    left = frame.point(
        radius_fraction=fraction,
        azimuth_deg=azimuth - _LINE_HALF_SWEEP_DEG,
        elevation_deg=elevation,
    )
    right = frame.point(
        radius_fraction=fraction,
        azimuth_deg=azimuth + _LINE_HALF_SWEEP_DEG,
        elevation_deg=elevation,
    )
    if motion.contour is Contour.ARCED:
        apex = (left + right) / 2.0 + frame.up * (_ARC_APEX_LIFT * frame.reach_m * 0.6)
        return [left, frame.clamp(apex), right]
    return [left, right]


def _circle_waypoints(
    frame: WorkspaceFrame,
    fraction: float,
    elevation: float,
    manner: magnitudes.ResolvedManner,
) -> list[np.ndarray]:
    """Trace an actual circle, in a plane the arm can work in.

    The circle lies in the plane spanned by the chain's lateral and up axes --
    a vertical loop facing the way the arm works -- centred on a point at the
    requested remove. Its radius is a fraction of the distance to that centre, so
    the whole loop stays inside the reachable set at every bearing rather than
    swinging out past the envelope on one side.
    """

    centre_fraction = max(0.3, fraction)
    centre = frame.point(radius_fraction=centre_fraction, elevation_deg=elevation)
    radius = 0.3 * centre_fraction * frame.directional_reach(0.0, elevation)
    radius *= manner.amplitude_scale

    steps = 12
    points: list[np.ndarray] = [frame.home]
    for index in range(steps + 1):
        angle = 2.0 * math.pi * index / steps
        offset = radius * (math.cos(angle) * frame.side + math.sin(angle) * frame.up)
        points.append(frame.clamp(centre + offset))
    return points


def _oscillation_waypoints(
    frame: WorkspaceFrame,
    fraction: float,
    azimuth: float,
    elevation: float,
    manner: magnitudes.ResolvedManner,
    motion,
) -> list[np.ndarray]:
    """Out, then back and forth with explicit reversals, then hold the last pose.

    The reversals are the point. An oscillation that never turns around is a
    reach, and the certification gates measure reversal count precisely so a
    candidate cannot quietly become one.
    """

    sweep = _OSCILLATION_HALF_SWEEP_DEG * manner.amplitude_scale
    along_line = motion.conformation is Conformation.LINE

    points: list[np.ndarray] = [frame.home]
    centre = frame.point(
        radius_fraction=fraction, azimuth_deg=azimuth, elevation_deg=elevation
    )
    points.append(centre)
    for cycle in range(manner.cycles):
        for sign in (1.0, -1.0):
            if along_line:
                offset = frame.point(
                    radius_fraction=fraction,
                    azimuth_deg=azimuth + sign * sweep,
                    elevation_deg=elevation,
                )
            else:
                offset = frame.point(
                    radius_fraction=fraction,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation + sign * sweep,
                )
            points.append(frame.clamp(offset))
    points.append(centre)
    return points


def _stative_waypoints(
    segment: SegmentV1, frame: WorkspaceFrame, manner: magnitudes.ResolvedManner
) -> list[np.ndarray]:
    stative = segment.motion_schema.stative
    if stative is Stative.HOLD:
        return [frame.home, frame.home]
    if stative is Stative.ORIENT:
        # Turning in place: the site stays put and the joints below it rotate.
        return [frame.home, frame.home]
    return [frame.home, frame.home]


def ground(
    schema_program: MotionSchemaProgramV1,
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    *,
    scene_id: str = "empty",
    seed: int = 0,
    contact_scene: bool = False,
    duration_scale: float = 1.0,
) -> GroundedProgram:
    """Ground a body-neutral program against one robot. No model call.

    ``duration_scale`` stretches every segment uniformly. The analytic velocity
    margin below gets the timing close, but the closed-loop response overshoots
    its reference by an amount no formula predicts exactly, so the bake measures
    the real overshoot and re-grounds with a scale that accounts for it. The
    motion is unchanged -- only its pace -- so a stretched primitive is still the
    primitive that was asked for.
    """

    if duration_scale <= 0.0:
        raise ValueError("duration_scale must be positive")

    capabilities = capabilities_of(manifest.morphology, contact_scene=contact_scene)
    cache: dict[str, WorkspaceFrame] = {}

    phases: list[MotionPhaseV2] = []
    # One track per chain, not one per segment. Separate tracks meet at phase
    # boundaries that the compiler's retimer maps through independently, and the
    # tiny mismatch there reads to the controller as a step in the reference --
    # producing a velocity spike that no amount of slowing the motion down
    # removes, because it is a discontinuity rather than a pace. Accumulating
    # into a single continuous series per chain makes the boundary disappear.
    keyframes_by_chain: dict[str, list[MotionKeyframeV2]] = {}
    owner_by_chain: dict[str, str] = {}
    site_by_chain: dict[str, str] = {}
    figure_sites: list[str] = []
    cursor = 0.0
    total_waypoints = 0
    fastest = 0.0
    rest_by_chain: dict[str, np.ndarray] = {}

    for index, segment in enumerate(schema_program.segments):
        entry = _entry_for(inventory, segment)
        missing = entry.requires - capabilities
        if missing:
            raise RigbyGeneralError(
                GeneralFailureCode.UNAFFORDED_SCHEMA,
                f"{manifest.rig_id} cannot be asked for {entry.entry_id!r}: it "
                f"lacks {sorted(item.value for item in missing)}",
                details={
                    "schema": entry.canonical_key,
                    "missing": sorted(item.value for item in missing),
                },
            )
        if segment.frame is ReferenceFrame.INTRINSIC and (
            Requirement.CONFIRMED_FRAME not in capabilities
        ):
            raise GroundingError(
                f"{manifest.rig_id}: an intrinsic-frame schema needs a confirmed "
                "front direction; nobody has said which way this robot faces",
                schema_key=segment.motion_schema.canonical_key,
                measurement="intrinsic_frame.source",
            )

        binding = _bind_role(manifest, model, segment.figure.role, cache=cache)
        _ground_anchor(segment, binding, manifest)

        resolved = magnitudes.resolve_manner(
            segment.manner,
            neutral_speed_mps=manifest.morphology.scale.neutral_speed_mps,
        )
        fastest = max(fastest, resolved.speed_mps)

        points = _waypoints(segment, binding, inventory, resolved)
        for point in points:
            if not binding.frame.contains(point, margin=1.02):
                azimuth, elevation = binding.frame.bearing_of(point)
                raise GroundingError(
                    f"{manifest.rig_id}: {entry.entry_id!r} at "
                    f"{segment.region.remove.value} lands "
                    f"{float(np.linalg.norm(point - binding.frame.origin)):.3f} m "
                    f"from the chain root, past a measured reach of "
                    f"{binding.frame.directional_reach(azimuth, elevation):.3f} m "
                    f"along that bearing",
                    schema_key=segment.motion_schema.canonical_key,
                    measurement="chain.reach_envelope_m",
                )

        # Every segment has to begin where the robot actually is. A path that
        # sweeps a line, traces a circle, or departs *from* somewhere does not
        # start at rest, and a trajectory whose first sample sits half a metre
        # from the arm's real pose is not a motion -- it is a teleport, which the
        # controller then chases at whatever speed it can manage. Prepending the
        # current position turns that jump into an approach.
        current_position = _chain_position(
            model, binding, rest_by_chain, manifest
        )
        if float(np.linalg.norm(points[0] - current_position)) > 1e-4:
            points = [current_position, *points]

        dense = ik.densify(points, per_span=_SUBDIVISIONS_PER_SPAN)
        joint_names = _solver_joints(manifest, model, binding)
        # Seeding from where the previous segment left this chain is what keeps
        # the trajectory in one piece. Re-seeding from rest lets the solver pick
        # a different elbow configuration for the same Cartesian point, and the
        # arm snaps between them at the segment boundary -- which the compiler
        # then correctly rejects as an infinite joint velocity.
        ik_seed = rest_by_chain.get(
            binding.frame.chain_id, np.asarray(manifest.rest_qpos, dtype=float)
        )
        try:
            solution = ik.solve_site_path(
                model,
                binding.frame.figure_site,
                joint_names,
                np.asarray(dense),
                seed_qpos=ik_seed,
            )
        except ik.IkFailure as error:
            raise GroundingError(
                f"{manifest.rig_id}: {entry.entry_id!r} at "
                f"{segment.region.remove.value} is not solvable -- {error}",
                schema_key=segment.motion_schema.canonical_key,
                measurement="inverse_kinematics.residual_m",
            ) from error

        length = magnitudes.path_length([tuple(point) for point in dense])
        duration = magnitudes.duration_for_path(length, speed_mps=resolved.speed_mps)

        keyframes, duration = _joint_keyframes(
            model,
            manifest,
            solution,
            joint_names,
            cursor,
            duration * duration_scale,
            resolved,
        )
        chain_id = binding.frame.chain_id
        existing = keyframes_by_chain.setdefault(chain_id, [])
        owner_by_chain[chain_id] = binding.owner
        site_by_chain[chain_id] = binding.frame.figure_site
        if existing and keyframes and keyframes[0].time_s <= existing[-1].time_s:
            # The previous segment already ended in this configuration, because
            # this segment's inverse kinematics was warm-started from it.
            keyframes = keyframes[1:]
        existing.extend(keyframes)
        figure_sites.append(binding.frame.figure_site)
        total_waypoints += len(dense)
        rest_by_chain[binding.frame.chain_id] = solution.qpos[-1]

        phases.append(
            MotionPhaseV2(
                phase_id=f"phase_{index}",
                kind=_PHASE_FOR_BOUNDARY[segment.boundary],
                start_s=round(cursor, 6),
                end_s=round(cursor + duration, 6),
                energy=_energy_for(resolved),
            )
        )
        cursor += duration

    recovery = _recovery_keyframes(
        cursor,
        keyframes_by_chain,
        np.asarray(manifest.rest_qpos, dtype=float),
        model,
        manifest,
    )
    if recovery is not None:
        phase, chain_id, extra = recovery
        phases.append(phase)
        keyframes_by_chain[chain_id].extend(extra)
        cursor = phase.end_s

    # Monotonicity is enforced once over the whole assembled series rather than
    # inside each piece. Segments and the recovery are timed independently, so a
    # collision can only appear where two of them meet -- exactly the seam that a
    # per-piece check cannot see.
    tracks = [
        MotionTrackV2(
            track_id=f"chain_{chain_id}",
            target=site_by_chain[chain_id],
            owner=owner_by_chain[chain_id],
            interpolation=InterpolationKind.QUINTIC,
            keyframes=tuple(_strictly_increasing(frames)),
        )
        for chain_id, frames in sorted(keyframes_by_chain.items())
    ]
    cursor = max(
        (track.keyframes[-1].time_s for track in tracks), default=cursor
    )
    phases = _clip_phases(phases, cursor)

    # A deterministic identity, not a fresh UUID. Grounding the same schema
    # against the same robot must produce the same program every time, or a
    # primitive could never be content-addressed and a library entry could never
    # be shown to be the thing that was certified.
    identity = content_hash(
        {
            "schema": schema_program.role_normalized_hash(),
            "rig": manifest.rig_id,
            "manifest": manifest.content_hash(),
            "scene": scene_id,
            "seed": seed,
            "inventory": inventory.sha256,
            "duration_scale": duration_scale,
        }
    )

    program = MotionProgramV2(
        program_id=f"grounded-{identity[:32]}",
        source_text=schema_program.source_text,
        duration_s=round(cursor, 6),
        rig_id=manifest.rig_id,
        scene_id=scene_id,
        seed=seed,
        phases=tuple(phases),
        tracks=tuple(tracks),
        metadata={
            "schema_program_id": schema_program.program_id,
            "schema_keys": list(schema_program.canonical_keys),
            "role_normalized_hash": schema_program.role_normalized_hash(),
            "inventory_sha256": inventory.sha256,
            "duration_scale": duration_scale,
            "grounded_against": {
                "reach_radius_m": manifest.morphology.scale.reach_radius_m,
                "neutral_speed_mps": manifest.morphology.scale.neutral_speed_mps,
                "characteristic_length_m": (
                    manifest.morphology.scale.characteristic_length_m
                ),
            },
        },
    )

    return GroundedProgram(
        program=program,
        figure_sites=tuple(figure_sites),
        waypoint_count=total_waypoints,
        measured_speed_mps=fastest,
    )


def _strictly_increasing(
    frames: list[MotionKeyframeV2], *, minimum_gap: float = 1e-3
) -> list[MotionKeyframeV2]:
    ordered: list[MotionKeyframeV2] = []
    for frame in frames:
        if ordered and frame.time_s <= ordered[-1].time_s:
            frame = frame.model_copy(
                update={"time_s": round(ordered[-1].time_s + minimum_gap, 6)}
            )
        ordered.append(frame)
    return ordered


def _clip_phases(
    phases: list[MotionPhaseV2], duration: float
) -> list[MotionPhaseV2]:
    """Keep the phase schedule inside the duration the keyframes actually span."""

    clipped: list[MotionPhaseV2] = []
    for phase in phases:
        end = min(phase.end_s, duration)
        if end <= phase.start_s:
            end = min(duration, phase.start_s + 1e-3)
        clipped.append(phase.model_copy(update={"end_s": round(end, 6)}))
    return clipped


def _chain_position(
    model,
    binding: _Binding,
    rest_by_chain: dict[str, np.ndarray],
    manifest: RobotAssetManifestV1,
) -> np.ndarray:
    """Where this chain's steered site is right now, in world coordinates."""

    import mujoco

    qpos = rest_by_chain.get(binding.frame.chain_id)
    if qpos is None:
        return binding.frame.home

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, binding.frame.figure_site
    )
    return np.array(data.site_xpos[site], dtype=float)


def _entry_for(inventory: SchemaInventory, segment: SegmentV1):
    key = (
        f"{segment.motion_schema.canonical_key}"
        f"|{segment.figure.role.value}->{segment.ground.role.value}"
    )
    try:
        return inventory.by_binding_key(key)
    except KeyError as error:
        raise RigbyGeneralError(
            GeneralFailureCode.INVENTED_BINDING,
            f"binding {key!r} is not in the sealed inventory",
            details={"binding": key},
        ) from error


def _solver_joints(
    manifest: RobotAssetManifestV1, model, binding: _Binding
) -> tuple[str, ...]:
    """Joints the solver may move, with every closure joint held out.

    A gripper that helps the arm reach is a gripper that opens and closes its way
    across the workspace. Closure belongs to contact schemas, not to travel.
    """

    closure = frozenset(
        name
        for effector in manifest.morphology.effectors
        for name in effector.grip_joints
    )
    joints = ik.chain_joint_names(
        model, binding.frame.figure_site, exclude=closure
    )
    if not joints:  # pragma: no cover - a chain always has joints above its site
        raise GroundingError(
            f"{manifest.rig_id}: no articulated joints above "
            f"{binding.frame.figure_site!r}",
            measurement="chain.joints",
        )
    return joints


# Two smooth time laws sit between an authored keyframe span and the joint
# velocity that finally comes out, and their peaks multiply.
#
# The first is the quintic interpolation between keyframes, which peaks at 1.875
# times its own average rate. The second is the compiler's phase retimer, an
# endpoint-flat monotone law that peaks at very nearly the same figure. Sizing a
# segment on the average rate would therefore put its midpoint some three and a
# half times over the limit while the mean looked entirely comfortable.
#
# The margin above the product is deliberate: exceeding a velocity limit does not
# produce a slightly worse motion, it produces a rejected candidate, so the cost
# of being generous here is a fractionally slower demo and the cost of being tight
# is a failed bake.
_QUINTIC_PEAK_FACTOR = 1.875
_RETIMING_PEAK_FACTOR = 1.9
# Plan to half the limit rather than right up to it. A tracking controller has
# to be able to *correct* an error, and correcting means briefly moving faster
# than the plan -- if the plan already sits at the ceiling, every correction is a
# violation. Measured closed-loop overshoot across the zoo runs about 1.8 times
# the planned peak, so half leaves the simulated motion comfortably inside the
# hardware limit rather than fractionally outside it.
_VELOCITY_HEADROOM = 0.5
_VELOCITY_PEAK_FACTOR = (
    _QUINTIC_PEAK_FACTOR * _RETIMING_PEAK_FACTOR / _VELOCITY_HEADROOM
)


def _joint_keyframes(
    model,
    manifest: RobotAssetManifestV1,
    solution: "ik.IkSolution",
    joint_names: tuple[str, ...],
    start_s: float,
    requested_duration: float,
    manner: magnitudes.ResolvedManner,
) -> tuple[list[MotionKeyframeV2], float]:
    """Lay the solved configurations out in time, and return the duration used.

    Two things decide the timing, and they are not the same kind of thing.

    Manner asks for a pace, and that is a *preference* expressed in Cartesian
    terms -- how fast the effector should travel. Joint velocity limits are a
    *fact* about the hardware, and Cartesian speed does not bound them: near the
    base axis the tip can crawl while the base yaw races, because the moment arm
    has collapsed. So the segment is stretched until the fastest joint is inside
    its limit, and the stretch is recorded rather than hidden. Honouring a
    requested pace by exceeding a velocity limit would only move the failure
    downstream, where it surfaces as a rejected candidate instead of a slower one.

    Spacing within the segment is proportional to joint travel rather than even,
    so a span the arm barely moves through does not get the same time as one it
    sweeps right across.
    """

    import mujoco

    addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in joint_names
    ]
    values = solution.qpos[:, addresses]
    limits = {dof.name: dof.velocity_limit for dof in manifest.dofs}

    spans = [
        float(np.linalg.norm(later - earlier))
        for earlier, later in zip(values, values[1:])
    ]
    total = sum(spans)
    if total < 1e-9:
        fractions = [index / max(1, len(values) - 1) for index in range(len(values))]
    else:
        fractions = [0.0]
        running = 0.0
        for span in spans:
            running += span
            fractions.append(running / total)

    # Timing is derived from the joint travel itself rather than from a separate
    # Cartesian speed estimate, and that is what makes Manner mean anything.
    #
    # The first version computed a preferred duration from path length and pace,
    # then raised it to whatever the velocity limits demanded. The floor won every
    # time -- "slowly", "at a normal pace" and "as fast as you can" all produced
    # the identical 22.06 s motion, which quietly deleted one of Talmy's slots.
    #
    # So the floor becomes the *reference* instead. ``floor_at_limit`` is the
    # fastest this segment can be performed at all; neutral sits a fixed margin
    # slower than that, and each speed ordinal is a step either side. The full
    # range spans about four to one, the top of it lands just at the hardware
    # limit, and nothing can ask for more than the robot has.
    floor_at_limit = magnitudes.MIN_SEGMENT_DURATION_S
    for index in range(len(values) - 1):
        share = fractions[index + 1] - fractions[index]
        if share <= 1e-9:
            continue
        for column, name in enumerate(joint_names):
            travel = abs(float(values[index + 1][column] - values[index][column]))
            limit = limits.get(name, 1.0)
            if travel < 1e-9 or limit <= 0.0:
                continue
            needed = _VELOCITY_PEAK_FACTOR * _VELOCITY_HEADROOM * travel / (limit * share)
            floor_at_limit = max(floor_at_limit, needed)

    neutral = floor_at_limit / _VELOCITY_HEADROOM
    paced = neutral / (magnitudes.SPEED_STEP ** manner.speed_ordinal)
    duration = max(magnitudes.MIN_SEGMENT_DURATION_S, floor_at_limit, paced)
    duration = min(duration, magnitudes.MAX_SEGMENT_DURATION_S * 8.0)

    keyframes: list[MotionKeyframeV2] = []
    for index, fraction in enumerate(fractions):
        is_endpoint = index in (0, len(fractions) - 1)
        keyframes.append(
            MotionKeyframeV2(
                time_s=round(start_s + fraction * duration, 6),
                joint_values={
                    name: float(values[index][column])
                    for column, name in enumerate(joint_names)
                },
                hard=manner.hard_endpoints and is_endpoint,
            )
        )

    for index in range(1, len(keyframes)):
        if keyframes[index].time_s <= keyframes[index - 1].time_s:
            keyframes[index] = keyframes[index].model_copy(
                update={"time_s": round(keyframes[index - 1].time_s + 1e-3, 6)}
            )
    return keyframes, duration


def _energy_for(manner: magnitudes.ResolvedManner) -> float:
    """Phase energy, kept below the compiler's energetic-retiming threshold at
    neutral effort.

    ``profile_for_phase`` switches to a more aggressive time law at 0.55, so a
    neutral request must land under it -- otherwise asking for nothing in
    particular would silently buy a faster, tighter motion than was asked for."""

    return float(min(1.0, max(0.0, 0.45 * manner.effort_scale)))


def _recovery_keyframes(
    cursor: float,
    keyframes_by_chain: dict[str, list[MotionKeyframeV2]],
    rest_qpos: np.ndarray,
    model,
    manifest: RobotAssetManifestV1,
) -> tuple[MotionPhaseV2, str, list[MotionKeyframeV2]] | None:
    """Retrace the outbound path back to rest.

    Not decoration. A primitive that ends wherever the last segment left it
    cannot be composed with the next one, because its start state depends on what
    ran before it. Returning to rest is what makes each primitive independently
    reusable, which is the entire point of a library.

    It retraces rather than cutting straight home, and that is the whole trick. A
    single span from the far end of a reach back to rest is one enormous quintic
    interpolation, and the arm cannot follow it: measured on a desktop arm, the
    outbound motion tracked to 7 mm and the return to 26 mm, nearly three times
    the tolerance, with the error sitting almost entirely in that one span. The
    outbound path is already known to be feasible, waypoint by waypoint. Walking
    back along it is feasible for exactly the same reason, and needs no separate
    argument.
    """

    import mujoco

    if not keyframes_by_chain:
        return None
    chain_id = sorted(keyframes_by_chain)[-1]
    frames = keyframes_by_chain[chain_id]
    if len(frames) < 2 or not frames[-1].joint_values:
        return None

    resting: dict[str, float] = {}
    for name in frames[-1].joint_values:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        resting[name] = float(rest_qpos[int(model.jnt_qposadr[joint_id])])

    # Reverse the outbound configurations, then finish at rest exactly. The
    # outbound path may have started somewhere other than rest -- a ``from``
    # schema does -- so the final keyframe is stated rather than assumed.
    returning = [dict(frame.joint_values) for frame in reversed(frames[:-1])]
    returning.append(resting)

    limits = {dof.name: dof.velocity_limit for dof in manifest.dofs}
    spans: list[float] = []
    previous = dict(frames[-1].joint_values)
    for values in returning:
        spans.append(
            max(
                (abs(values.get(name, 0.0) - previous.get(name, 0.0)) for name in values),
                default=0.0,
            )
        )
        previous = values
    if sum(spans) < 1e-6:
        return None

    duration = 0.0
    for span, values in zip(spans, returning):
        slowest = min((limits.get(name, 1.0) for name in values), default=1.0)
        duration += max(0.05, _VELOCITY_PEAK_FACTOR * span / max(slowest, 1e-3))

    total = sum(spans)
    extra: list[MotionKeyframeV2] = []
    running = 0.0
    for span, values in zip(spans, returning):
        running += span
        extra.append(
            MotionKeyframeV2(
                time_s=round(cursor + duration * (running / total), 6),
                joint_values=values,
            )
        )

    for index in range(1, len(extra)):
        if extra[index].time_s <= extra[index - 1].time_s:
            extra[index] = extra[index].model_copy(
                update={"time_s": round(extra[index - 1].time_s + 1e-3, 6)}
            )

    phase = MotionPhaseV2(
        phase_id="phase_recovery",
        kind=PhaseKind.RECOVERY,
        start_s=round(cursor, 6),
        end_s=extra[-1].time_s,
        energy=0.3,
    )
    return phase, chain_id, extra
