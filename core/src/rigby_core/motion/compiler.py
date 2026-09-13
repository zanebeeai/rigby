from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

import mujoco
import numpy as np

from rigby_core.contracts import (
    CoordinateFrame,
    InterpolationKind,
    MotionProgramV2,
    MotionTrackV2,
    Quaternion,
    QuaternionMeaning,
    QuaternionOrder,
    RigAssetManifestV1,
    TrackOwnership,
)
from rigby_core.hashing import content_hash

from .accents import MotionAccentSpec, accent_fraction
from .errors import MotionCompilationError, MotionFailureReason
from .ownership import build_joint_series, resolve_joint_sample
from .refinement import (
    CollisionDistanceObjective,
    SiteOrientationObjective,
    SitePositionObjective,
    refine_joint_window,
)
from .rotations import slerp, squad
from .timing import PhaseRetimer, QuinticSegment, TimingProfile
from .trajectory import CandidateTrajectoryV1, ContactPlateauV1


@dataclass(frozen=True)
class _DofBinding:
    name: str
    qpos_adr: int
    dof_adr: int
    minimum: float
    maximum: float
    velocity_limit: float


def _task_components(track: MotionTrackV2) -> tuple[bool, bool, bool]:
    has_joint = any(keyframe.joint_values for keyframe in track.keyframes)
    has_position = any(keyframe.position is not None for keyframe in track.keyframes)
    has_rotation = any(keyframe.rotation is not None for keyframe in track.keyframes)
    return has_joint, has_position, has_rotation


def _wxyz(quaternion: Quaternion) -> np.ndarray:
    values = np.asarray(quaternion.values, dtype=np.float64)
    if quaternion.convention.order is QuaternionOrder.XYZW:
        return values[[3, 0, 1, 2]]
    return values.copy()


def _multiply_quaternion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _accent_rotation(
    rotation: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Apply a bounded fraction of the segment's world-space rotation."""

    if fraction == 0.0:
        return rotation
    if float(np.dot(start, end)) < 0.0:
        end = -end
    inverse_start = start.copy()
    inverse_start[1:] *= -1.0
    relative = _multiply_quaternion(end, inverse_start)
    vector_length = float(np.linalg.norm(relative[1:]))
    if vector_length < 1e-12:
        return rotation
    half_angle = float(np.arctan2(vector_length, relative[0]))
    axis = relative[1:] / vector_length
    extra_half_angle = half_angle * fraction
    extra = np.asarray(
        [np.cos(extra_half_angle), *(axis * np.sin(extra_half_angle))],
        dtype=np.float64,
    )
    result = _multiply_quaternion(extra, rotation)
    return result / np.linalg.norm(result)


def _task_track_sample(
    track: MotionTrackV2,
    authored_time_s: float,
    accent: MotionAccentSpec | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    times = np.asarray([item.time_s for item in track.keyframes], dtype=np.float64)
    if len(times) == 1:
        left = right = 0
        progress = 0.0
    else:
        right = int(np.searchsorted(times, authored_time_s, side="right"))
        right = min(max(right, 1), len(times) - 1)
        left = right - 1
        progress = float(
            np.clip(
                (authored_time_s - times[left]) / (times[right] - times[left]),
                0.0,
                1.0,
            )
        )
    position: np.ndarray | None = None
    if track.keyframes[left].position is not None:
        start = track.keyframes[left].position
        end = track.keyframes[right].position
        if end is None:
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Task-space position track {track.track_id!r} has a missing component",
            )
        start_value = np.asarray([start.x, start.y, start.z], dtype=np.float64)
        end_value = np.asarray([end.x, end.y, end.z], dtype=np.float64)
        if left == right:
            position = start_value
        else:
            segment = QuinticSegment(start_value, end_value, times[right] - times[left])
            position = segment.sample(authored_time_s - times[left]).position
            if accent is not None:
                position = position + (end_value - start_value) * accent_fraction(
                    progress, accent
                )
    rotation: np.ndarray | None = None
    if track.keyframes[left].rotation is not None:
        start_rotation = track.keyframes[left].rotation
        end_rotation = track.keyframes[right].rotation
        if end_rotation is None:
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Task-space rotation track {track.track_id!r} has a missing component",
            )
        if left == right:
            rotation = _wxyz(start_rotation)
        elif track.interpolation is InterpolationKind.SQUAD:
            previous = track.keyframes[max(0, left - 1)].rotation or start_rotation
            following = track.keyframes[min(len(track.keyframes) - 1, right + 1)].rotation or end_rotation
            rotation = _wxyz(
                squad(previous, start_rotation, end_rotation, following, progress)
            )
        else:
            rotation = _wxyz(slerp(start_rotation, end_rotation, progress))
        if accent is not None and left != right:
            rotation = _accent_rotation(
                rotation,
                _wxyz(start_rotation),
                _wxyz(end_rotation),
                accent_fraction(progress, accent),
            )
    return position, rotation


def _task_joint_chain(
    model: mujoco.MjModel,
    site_id: int,
    owner: str | None = None,
) -> tuple[str, ...]:
    body_id = int(model.site_bodyid[site_id])
    names: list[str] = []
    while body_id > 0:
        first_joint = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        for joint_id in range(first_joint, first_joint + joint_count):
            if model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                names.append(name)
        body_id = int(model.body_parentid[body_id])
    names.reverse()
    if owner is not None:
        side = next(
            (prefix for prefix in ("left_", "right_") if owner.startswith(prefix)),
            None,
        )
        if side is not None:
            names = [name for name in names if name.startswith(side)]
    if not names:
        raise MotionCompilationError(
            MotionFailureReason.IK_INFEASIBLE,
            "Task-space target has no articulated scalar-joint chain",
        )
    return tuple(names)


def _active_task_track(
    tracks: Iterable[MotionTrackV2], time_s: float, target: str
) -> MotionTrackV2 | None:
    active = [
        track
        for track in tracks
        if track.target == target
        and track.keyframes[0].time_s - 1e-12 <= time_s <= track.keyframes[-1].time_s + 1e-12
    ]
    if not active:
        return None
    if any(track.ownership is TrackOwnership.ADDITIVE for track in active):
        raise MotionCompilationError(
            MotionFailureReason.UNSUPPORTED_TRACK,
            f"Task-space target {target!r} does not support ambiguous additive pose composition",
        )
    priority = max(track.priority for track in active)
    winners = [track for track in active if track.priority == priority]
    if len(winners) != 1:
        raise MotionCompilationError(
            MotionFailureReason.UNRESOLVED_TRACK_OWNERSHIP,
            f"Task-space tracks tie for target {target!r}",
        )
    return winners[0]


def _refine_task_tracks(
    candidate: CandidateTrajectoryV1,
    model: mujoco.MjModel,
    task_tracks: tuple[MotionTrackV2, ...],
    retimer: PhaseRetimer,
    collision_objectives: tuple[CollisionDistanceObjective, ...],
    accents: dict[str, MotionAccentSpec],
    planned_contacts: tuple,
    allowed_contact_pairs: tuple[tuple[str, str], ...],
    protected_times: set[float],
) -> CandidateTrajectoryV1:
    if not task_tracks and not collision_objectives:
        return candidate
    positions: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    rotations: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    selected_joints: set[str] = set()
    targets = sorted({track.target for track in task_tracks})
    site_ids: dict[str, int] = {}
    for target in targets:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target)
        if site_id < 0:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_MODEL,
                f"Task-space target {target!r} is not a MuJoCo site",
            )
        site_ids[target] = site_id
        owners = {track.owner for track in task_tracks if track.target == target}
        for owner in owners:
            selected_joints.update(_task_joint_chain(model, site_id, owner))
    for frame_index, time_s in enumerate(candidate.times_s):
        authored = retimer.sample(float(time_s)).authored_time_s
        for target in targets:
            track = _active_task_track(task_tracks, authored, target)
            if track is None:
                continue
            accent = accents.get(track.track_id)
            if any(abs(authored - anchor) <= 1e-9 for anchor in protected_times):
                accent = None
            position, rotation = _task_track_sample(track, authored, accent)
            if position is not None:
                positions[target].append((frame_index, position))
            if rotation is not None:
                rotations[target].append((frame_index, rotation))
    site_objectives = tuple(
        SitePositionObjective(
            site_name=target,
            frame_indices=tuple(index for index, _ in values),
            targets_m=np.asarray([value for _, value in values]),
            weight=20.0,
            tolerance_m=0.005,
        )
        for target, values in positions.items()
        if values
    )
    orientation_objectives = tuple(
        SiteOrientationObjective(
            site_name=target,
            frame_indices=tuple(index for index, _ in values),
            targets_wxyz=np.asarray([value for _, value in values]),
            weight=15.0,
            tolerance_rad=0.04,
        )
        for target, values in rotations.items()
        if values
    )
    if collision_objectives and not selected_joints:
        raise MotionCompilationError(
            MotionFailureReason.UNSUPPORTED_TRACK,
            "Collision refinement requires at least one task-space articulated chain",
        )
    refined = refine_joint_window(
        model,
        candidate,
        sorted(selected_joints),
        0,
        len(candidate.times_s) - 1,
        site_objectives=site_objectives,
        orientation_objectives=orientation_objectives,
        collision_objectives=collision_objectives,
        temporal_weight=1e-4,
        # Frames without an active task objective must remain close to the
        # already-compiled joint trajectory.  A near-zero deviation term lets
        # temporal smoothing propagate a later release pose backward through
        # the whole motion window, creating contacts before the authored
        # approach.  This weight is still four orders below the task-space
        # residual, but dominates smoothing on unconstrained frames.
        deviation_weight=1e-3,
        max_nfev=300,
        anchor_window_endpoints=False,
    )
    automatic = _automatic_collision_objectives(
        refined,
        model,
        tuple(site_ids.values()),
        planned_contacts,
        allowed_contact_pairs,
    )
    if not automatic:
        return refined
    return refine_joint_window(
        model,
        refined,
        sorted(selected_joints),
        0,
        len(refined.times_s) - 1,
        site_objectives=site_objectives,
        orientation_objectives=orientation_objectives,
        collision_objectives=tuple(collision_objectives) + automatic,
        temporal_weight=1e-4,
        deviation_weight=1e-3,
        max_nfev=300,
        anchor_window_endpoints=False,
    )


def _geom_matches_selector(model: mujoco.MjModel, geom_id: int, selector: str) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name == selector:
        return True
    body_id = int(model.geom_bodyid[geom_id])
    while body_id >= 0:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == selector:
            return True
        if body_id == 0:
            break
        body_id = int(model.body_parentid[body_id])
    return False


def _pair_is_declared_allowed(
    model: mujoco.MjModel,
    first: int,
    second: int,
    allowed_contact_pairs: tuple[tuple[str, str], ...],
) -> bool:
    selectors = list(allowed_contact_pairs)
    return any(
        (
            _geom_matches_selector(model, first, left)
            and _geom_matches_selector(model, second, right)
        )
        or (
            _geom_matches_selector(model, first, right)
            and _geom_matches_selector(model, second, left)
        )
        for left, right in selectors
    )


def _automatic_collision_objectives(
    candidate: CandidateTrajectoryV1,
    model: mujoco.MjModel,
    task_site_ids: tuple[int, ...],
    planned_contacts: tuple,
    allowed_contact_pairs: tuple[tuple[str, str], ...],
) -> tuple[CollisionDistanceObjective, ...]:
    """Discover a small deterministic set of near task-chain collision pairs."""

    relevant_bodies: set[int] = set()
    for site_id in task_site_ids:
        body_id = int(model.site_bodyid[site_id])
        while body_id > 0:
            relevant_bodies.add(body_id)
            body_id = int(model.body_parentid[body_id])
    relevant_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in relevant_bodies
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    ]
    excluded_bodies = {
        tuple(sorted((int(signature) >> 16, int(signature) & 0xFFFF)))
        for signature in model.exclude_signature
    }

    def descendant_of(body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == ancestor_id

    def excluded_subtrees(first_body: int, second_body: int) -> bool:
        # MJCF excludes apply to both named body subtrees, not merely the two
        # exact body IDs.  Treating them as exact pairs creates impossible
        # avoidance objectives for intentionally overlapping torso/head and
        # hand/digit geometry that MuJoCo itself filters from contact.
        return any(
            (
                descendant_of(first_body, left)
                and descendant_of(second_body, right)
            )
            or (
                descendant_of(first_body, right)
                and descendant_of(second_body, left)
            )
            for left, right in excluded_bodies
        )
    stride = max(1, int(np.ceil(len(candidate.times_s) / 24.0)))
    sampled_frames = tuple(range(0, len(candidate.times_s), stride))
    if sampled_frames[-1] != len(candidate.times_s) - 1:
        sampled_frames += (len(candidate.times_s) - 1,)
    data = mujoco.MjData(model)
    near: list[tuple[float, str, str, tuple[int, ...]]] = []
    visited_pairs: set[tuple[int, int]] = set()
    for relevant_geom in relevant_geoms:
        for other_geom in range(model.ngeom):
            pair = tuple(sorted((relevant_geom, other_geom)))
            if relevant_geom == other_geom or pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            first, second = pair
            first_body = int(model.geom_bodyid[first])
            second_body = int(model.geom_bodyid[second])
            second_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second)
            if not second_name or first_body == second_body:
                continue
            if int(model.body_parentid[first_body]) == second_body or int(model.body_parentid[second_body]) == first_body:
                continue
            if excluded_subtrees(first_body, second_body):
                continue
            if not (
                (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
            ):
                continue
            if _pair_is_declared_allowed(model, first, second, allowed_contact_pairs):
                continue
            close_frames: list[int] = []
            minimum = float("inf")
            for frame in sampled_frames:
                authored_contact = any(
                    edge.start_s - 1e-9 <= candidate.times_s[frame] <= edge.end_s + 1e-9
                    and _pair_is_declared_allowed(
                        model,
                        first,
                        second,
                        ((edge.body_a, edge.body_b),),
                    )
                    for edge in planned_contacts
                )
                if authored_contact:
                    continue
                data.qpos[:] = candidate.qpos[frame]
                mujoco.mj_forward(model, data)
                distance = float(
                    mujoco.mj_geomDistance(model, data, first, second, 0.04, None)
                )
                minimum = min(minimum, distance)
                if distance < 0.015:
                    close_frames.append(frame)
            if close_frames:
                first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first)
                assert first_name is not None
                near.append((minimum, first_name, second_name, tuple(close_frames)))
    near.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        CollisionDistanceObjective(first, second, frames, minimum_distance_m=0.002)
        for _, first, second, frames in near[:12]
    )


def _dof_bindings(
    model: mujoco.MjModel, rig: RigAssetManifestV1
) -> dict[str, _DofBinding]:
    bindings: dict[str, _DofBinding] = {}
    for dof in rig.dofs:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        if joint_id < 0:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_MODEL,
                f"Rig DOF {dof.name!r} references missing joint {dof.joint!r}",
            )
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise MotionCompilationError(
                MotionFailureReason.INVALID_MODEL,
                f"Authored DOF {dof.name!r} must map to one scalar MuJoCo joint",
            )
        minimum, maximum = dof.minimum, dof.maximum
        if bool(model.jnt_limited[joint_id]):
            minimum = max(minimum, float(model.jnt_range[joint_id, 0]))
            maximum = min(maximum, float(model.jnt_range[joint_id, 1]))
        if maximum <= minimum:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_MODEL,
                f"Rig and MuJoCo limits do not overlap for {dof.name!r}",
            )
        bindings[dof.name] = _DofBinding(
            name=dof.name,
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            dof_adr=int(model.jnt_dofadr[joint_id]),
            minimum=minimum,
            maximum=maximum,
            velocity_limit=dof.velocity_limit,
        )
    return bindings


def _quaternion_addresses(model: mujoco.MjModel) -> tuple[int, ...]:
    addresses: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        address = int(model.jnt_qposadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            addresses.append(address + 3)
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            addresses.append(address)
    return tuple(addresses)


def _sample_times(duration_s: float, sample_hz: int) -> np.ndarray:
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    steps = int(np.ceil(duration_s * sample_hz))
    times = np.arange(steps + 1, dtype=np.float64) / float(sample_hz)
    times[-1] = duration_s
    if len(times) > 2 and times[-1] <= times[-2]:
        times = times[:-1]
        times[-1] = duration_s
    return times


def compile_motion_program(
    program: MotionProgramV2,
    model: mujoco.MjModel,
    rig: RigAssetManifestV1,
    *,
    sample_hz: int = 240,
    timing_profile: TimingProfile | None = None,
    collision_objectives: tuple[CollisionDistanceObjective, ...] = (),
    accents: tuple[MotionAccentSpec, ...] = (),
    allowed_contact_pairs: tuple[tuple[str, str], ...] = (),
) -> CandidateTrajectoryV1:
    """Compile joint and world-task-space tracks into generalized targets."""

    if program.rig_id != rig.rig_id:
        raise MotionCompilationError(
            MotionFailureReason.INVALID_MODEL,
            f"Program rig {program.rig_id!r} does not match {rig.rig_id!r}",
        )
    bindings = _dof_bindings(model, rig)
    accent_by_track = {accent.track_id: accent for accent in accents}
    if len(accent_by_track) != len(accents):
        raise ValueError("motion accent track IDs must be unique")
    track_ids = {track.track_id for track in program.tracks}
    unknown_accents = sorted(set(accent_by_track) - track_ids)
    if unknown_accents:
        raise MotionCompilationError(
            MotionFailureReason.UNSUPPORTED_TRACK,
            "Motion accent references an unknown track",
            details={"track_ids": unknown_accents},
        )
    joint_tracks: list[MotionTrackV2] = []
    task_tracks: list[MotionTrackV2] = []
    for track in program.tracks:
        has_joint, has_position, has_rotation = _task_components(track)
        if has_joint and (has_position or has_rotation):
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Track {track.track_id!r} must declare either joint or task-space targets",
            )
        if has_joint:
            joint_tracks.append(track)
            continue
        if track.interpolation is InterpolationKind.BOUNDED_QUINTIC:
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Bounded quintic track {track.track_id!r} requires joint-value targets",
            )
        if not (has_position or has_rotation):
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Track {track.track_id!r} contains no compilable target",
            )
        if has_position and not all(
            keyframe.position is not None for keyframe in track.keyframes
        ):
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Task-space position track {track.track_id!r} must specify every keyframe",
            )
        if has_rotation:
            if not all(keyframe.rotation is not None for keyframe in track.keyframes):
                raise MotionCompilationError(
                    MotionFailureReason.UNSUPPORTED_TRACK,
                    f"Task-space rotation track {track.track_id!r} must specify every keyframe",
                )
            if track.interpolation not in {
                InterpolationKind.SLERP,
                InterpolationKind.SQUAD,
            }:
                raise MotionCompilationError(
                    MotionFailureReason.UNSUPPORTED_TRACK,
                    f"Task-space rotation track {track.track_id!r} requires SLERP or SQUAD",
                )
            for keyframe in track.keyframes:
                assert keyframe.rotation is not None
                convention = keyframe.rotation.convention
                if (
                    convention.frame is not CoordinateFrame.WORLD
                    or convention.meaning is not QuaternionMeaning.ABSOLUTE
                ):
                    raise MotionCompilationError(
                        MotionFailureReason.UNSUPPORTED_TRACK,
                        f"Task-space rotation track {track.track_id!r} requires world-absolute quaternions",
                    )
        task_tracks.append(track)

    series_by_joint = build_joint_series(tuple(joint_tracks), program.duration_s)
    unknown = sorted(set(series_by_joint) - set(bindings))
    if unknown:
        raise MotionCompilationError(
            MotionFailureReason.UNKNOWN_DOF,
            "Motion program references DOFs absent from the rig manifest",
            details={"unknown_dofs": unknown},
        )
    for joint_name, series in series_by_joint.items():
        binding = bindings[joint_name]
        for item in series:
            if item.ownership is TrackOwnership.ADDITIVE:
                continue
            if np.any(item.values < binding.minimum) or np.any(item.values > binding.maximum):
                raise MotionCompilationError(
                    MotionFailureReason.JOINT_LIMIT_VIOLATION,
                    f"Track {item.track.track_id!r} exceeds limits for {joint_name!r}",
                    details={
                        "joint": joint_name,
                        "minimum": binding.minimum,
                        "maximum": binding.maximum,
                    },
                )

    protected = {
        keyframe.time_s
        for track in program.tracks
        for keyframe in track.keyframes
        if keyframe.hard
    }
    for contact in program.contacts:
        protected.update((contact.start_s, contact.end_s))
    retimer = PhaseRetimer(
        program.phases,
        program.duration_s,
        protected_times_s=protected,
        profile=timing_profile,
    )
    times = _sample_times(program.duration_s, sample_hz)
    rest_qpos = np.asarray(rig.rest_qpos, dtype=np.float64)
    if rest_qpos.shape != (model.nq,) or np.any(~np.isfinite(rest_qpos)):
        raise MotionCompilationError(
            MotionFailureReason.INVALID_MODEL,
            f"Rig rest_qpos must be finite with shape ({model.nq},)",
        )
    qpos = np.repeat(rest_qpos[None, :], len(times), axis=0)
    qvel = np.zeros((len(times), model.nv), dtype=np.float64)
    qacc = np.zeros((len(times), model.nv), dtype=np.float64)
    for sample_index, time_s in enumerate(times):
        retimed = retimer.sample(float(time_s))
        for joint_name, series in series_by_joint.items():
            binding = bindings[joint_name]
            neutral = float(rest_qpos[binding.qpos_adr])
            sample = resolve_joint_sample(series, retimed.authored_time_s, neutral)
            position = sample.position
            velocity = sample.velocity * retimed.first_derivative
            acceleration = (
                sample.acceleration * retimed.first_derivative**2
                + sample.velocity * retimed.second_derivative
            )
            if position < binding.minimum - 1e-9 or position > binding.maximum + 1e-9:
                raise MotionCompilationError(
                    MotionFailureReason.JOINT_LIMIT_VIOLATION,
                    f"Composed tracks exceed limits for {joint_name!r}",
                    details={"time_s": float(time_s), "value": position},
                )
            if abs(velocity) > binding.velocity_limit + 1e-9:
                raise MotionCompilationError(
                    MotionFailureReason.JOINT_LIMIT_VIOLATION,
                    f"Composed tracks exceed velocity limits for {joint_name!r}",
                    details={
                        "time_s": float(time_s),
                        "velocity": velocity,
                        "limit": binding.velocity_limit,
                    },
                )
            qpos[sample_index, binding.qpos_adr] = position
            qvel[sample_index, binding.dof_adr] = velocity
            qacc[sample_index, binding.dof_adr] = acceleration

    # Joint accents modify the actual generalized trajectory.  A deterministic
    # global scale keeps every accent within authored joint and velocity bounds.
    accent_delta = np.zeros_like(qpos)
    affected: set[str] = set()
    protected_tolerance = 0.5 / float(sample_hz) + 1e-12
    joint_tracks_by_id = {track.track_id: track for track in joint_tracks}
    for track_id, accent in accent_by_track.items():
        track = joint_tracks_by_id.get(track_id)
        if track is None:
            continue
        key_times = np.asarray([key.time_s for key in track.keyframes], dtype=np.float64)
        for sample_index, time_s in enumerate(times):
            authored = retimer.sample(float(time_s)).authored_time_s
            if any(abs(authored - anchor) <= protected_tolerance for anchor in protected):
                continue
            right = min(max(int(np.searchsorted(key_times, authored, side="right")), 1), len(key_times) - 1)
            left = right - 1
            span = key_times[right] - key_times[left]
            progress = float(np.clip((authored - key_times[left]) / span, 0.0, 1.0))
            for name in set(track.keyframes[left].joint_values) & set(track.keyframes[right].joint_values):
                affected.add(name)
                binding = bindings[name]
                travel = (
                    track.keyframes[right].joint_values[name]
                    - track.keyframes[left].joint_values[name]
                )
                accent_delta[sample_index, binding.qpos_adr] += travel * accent_fraction(progress, accent)
    if affected:
        def feasible(scale: float) -> bool:
            proposed = qpos + scale * accent_delta
            for name in affected:
                binding = bindings[name]
                values = proposed[:, binding.qpos_adr]
                velocity = np.gradient(values, times, edge_order=2)
                if (
                    np.any(values < binding.minimum - 1e-9)
                    or np.any(values > binding.maximum + 1e-9)
                    or np.any(np.abs(velocity) > binding.velocity_limit + 1e-9)
                ):
                    return False
            return True

        low, high = 0.0, 1.0
        for _ in range(30):
            middle = (low + high) * 0.5
            if feasible(middle):
                low = middle
            else:
                high = middle
        if low <= 1e-6:
            raise MotionCompilationError(
                MotionFailureReason.JOINT_LIMIT_VIOLATION,
                "Authored motion accent has no feasible bounded magnitude",
            )
        qpos += low * accent_delta
        for name in affected:
            binding = bindings[name]
            qvel[:, binding.dof_adr] = np.gradient(
                qpos[:, binding.qpos_adr], times, edge_order=2
            )
            qacc[:, binding.dof_adr] = np.gradient(
                qvel[:, binding.dof_adr], times, edge_order=2
            )

    identity = {
        "program_hash": program.content_hash(),
        "rig_hash": rig.content_hash(),
        "sample_hz": sample_hz,
        "times_s": times.tolist(),
        "qpos": qpos.tolist(),
    }
    candidate_id = f"candidate-{content_hash(identity)[:20]}"
    candidate = CandidateTrajectoryV1(
        candidate_id=candidate_id,
        program_hash=program.content_hash(),
        rig_hash=rig.content_hash(),
        times_s=times,
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        quaternion_qpos_adrs=_quaternion_addresses(model),
        contact_plateaus=tuple(
            ContactPlateauV1(contact.contact_id, contact.start_s, contact.end_s)
            for contact in program.contacts
        ),
    )
    candidate = _refine_task_tracks(
        candidate,
        model,
        tuple(task_tracks),
        retimer,
        collision_objectives,
        accent_by_track,
        program.contacts,
        allowed_contact_pairs,
        protected,
    )
    for binding in bindings.values():
        position = candidate.qpos[:, binding.qpos_adr]
        velocity = candidate.qvel[:, binding.dof_adr]
        if np.any(position < binding.minimum - 1e-8) or np.any(
            position > binding.maximum + 1e-8
        ):
            raise MotionCompilationError(
                MotionFailureReason.JOINT_LIMIT_VIOLATION,
                f"Task-space refinement exceeds limits for {binding.name!r}",
            )
        if np.any(np.abs(velocity) > binding.velocity_limit + 1e-8):
            raise MotionCompilationError(
                MotionFailureReason.JOINT_LIMIT_VIOLATION,
                f"Task-space refinement exceeds velocity limit for {binding.name!r}",
            )
    return candidate
