from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from rigby_core.hashing import content_hash

from .errors import MotionCompilationError, MotionFailureReason
from .trajectory import CandidateTrajectoryV1


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SitePositionObjective:
    site_name: str
    frame_indices: tuple[int, ...]
    targets_m: npt.ArrayLike
    weight: float = 1.0
    tolerance_m: float = 0.003

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets_m, dtype=np.float64).copy()
        if targets.shape != (len(self.frame_indices), 3) or np.any(~np.isfinite(targets)):
            raise ValueError("site targets must have shape (frame_count, 3) and be finite")
        if self.weight <= 0.0 or self.tolerance_m <= 0.0:
            raise ValueError("site objective weight and tolerance must be positive")
        targets.setflags(write=False)
        object.__setattr__(self, "targets_m", targets)


@dataclass(frozen=True)
class PinchDistanceObjective:
    first_site: str
    second_site: str
    frame_indices: tuple[int, ...]
    target_distance_m: float | npt.ArrayLike
    weight: float = 1.0
    tolerance_m: float = 0.002

    def __post_init__(self) -> None:
        target = np.broadcast_to(
            np.asarray(self.target_distance_m, dtype=np.float64),
            (len(self.frame_indices),),
        ).copy()
        if np.any(~np.isfinite(target)) or np.any(target < 0.0):
            raise ValueError("pinch targets must be finite nonnegative distances")
        if self.weight <= 0.0 or self.tolerance_m <= 0.0:
            raise ValueError("pinch objective weight and tolerance must be positive")
        target.setflags(write=False)
        object.__setattr__(self, "target_distance_m", target)


@dataclass(frozen=True)
class SiteOrientationObjective:
    site_name: str
    frame_indices: tuple[int, ...]
    targets_wxyz: npt.ArrayLike
    weight: float = 1.0
    tolerance_rad: float = 0.03

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets_wxyz, dtype=np.float64).copy()
        if targets.shape != (len(self.frame_indices), 4) or np.any(~np.isfinite(targets)):
            raise ValueError("orientation targets must have shape (frame_count, 4)")
        norms = np.linalg.norm(targets, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("orientation targets must be normalized WXYZ quaternions")
        if self.weight <= 0.0 or self.tolerance_rad <= 0.0:
            raise ValueError("orientation objective weight and tolerance must be positive")
        targets /= norms[:, None]
        targets.setflags(write=False)
        object.__setattr__(self, "targets_wxyz", targets)


@dataclass(frozen=True)
class CollisionDistanceObjective:
    first_geom: str
    second_geom: str
    frame_indices: tuple[int, ...]
    minimum_distance_m: float = 0.002
    weight: float = 1.0
    tolerance_m: float = 0.0005

    def __post_init__(self) -> None:
        if self.first_geom == self.second_geom:
            raise ValueError("collision objective requires two distinct geoms")
        if self.minimum_distance_m < 0.0 or self.weight <= 0.0 or self.tolerance_m <= 0.0:
            raise ValueError("collision objective bounds must be nonnegative/positive")


@dataclass(frozen=True)
class HardJointAnchor:
    frame_index: int
    joint_name: str
    value: float


@dataclass(frozen=True)
class _JointBinding:
    name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int
    minimum: float
    maximum: float


def _joint_bindings(model: mujoco.MjModel, joint_names: Iterable[str]) -> tuple[_JointBinding, ...]:
    bindings: list[_JointBinding] = []
    seen: set[str] = set()
    for name in joint_names:
        if name in seen:
            raise ValueError(f"joint {name!r} was selected more than once")
        seen.add(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise MotionCompilationError(
                MotionFailureReason.UNKNOWN_DOF,
                f"Refinement references missing MuJoCo joint {name!r}",
            )
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise MotionCompilationError(
                MotionFailureReason.INVALID_MODEL,
                f"Refinement joint {name!r} is not scalar",
            )
        if bool(model.jnt_limited[joint_id]):
            minimum, maximum = map(float, model.jnt_range[joint_id])
        else:
            minimum, maximum = -np.inf, np.inf
        bindings.append(
            _JointBinding(
                name=name,
                joint_id=joint_id,
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                dof_adr=int(model.jnt_dofadr[joint_id]),
                minimum=minimum,
                maximum=maximum,
            )
        )
    if not bindings:
        raise ValueError("at least one refinement joint is required")
    return tuple(bindings)


def _site_id(model: mujoco.MjModel, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if identifier < 0:
        raise MotionCompilationError(
            MotionFailureReason.INVALID_MODEL,
            f"Refinement references missing MuJoCo site {name!r}",
        )
    return identifier


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if identifier < 0:
        raise MotionCompilationError(
            MotionFailureReason.INVALID_MODEL,
            f"Refinement references missing MuJoCo geom {name!r}",
        )
    return identifier


def _matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest world-space SO(3) logarithm from current to target.

    The traditional half cross-sum is only a small-angle approximation and
    becomes exactly zero at 180 degrees.  That singularity let a grossly
    inverted hand orientation pass the hard task-space gate.  A normalized
    relative quaternion retains a finite pi-radian residual while preserving
    the same local sign/Jacobian convention around the solution.
    """

    current_quaternion = np.empty(4, dtype=np.float64)
    target_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(current_quaternion, current.reshape(9))
    mujoco.mju_mat2Quat(target_quaternion, target.reshape(9))
    inverse_current = current_quaternion.copy()
    inverse_current[1:] *= -1.0
    relative = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(relative, target_quaternion, inverse_current)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, max(0.0, float(relative[0])))
    return relative[1:] * (angle / vector_norm)


def refine_joint_window(
    model: mujoco.MjModel,
    candidate: CandidateTrajectoryV1,
    joint_names: Iterable[str],
    start_index: int,
    end_index: int,
    *,
    site_objectives: Iterable[SitePositionObjective] = (),
    orientation_objectives: Iterable[SiteOrientationObjective] = (),
    pinch_objectives: Iterable[PinchDistanceObjective] = (),
    collision_objectives: Iterable[CollisionDistanceObjective] = (),
    hard_anchors: Iterable[HardJointAnchor] = (),
    temporal_weight: float = 0.02,
    deviation_weight: float = 1e-4,
    max_nfev: int = 200,
    anchor_window_endpoints: bool = True,
) -> CandidateTrajectoryV1:
    """Jointly refine an arm/wrist/hand window using MuJoCo site Jacobians."""

    if not 0 <= start_index < end_index < len(candidate.times_s):
        raise ValueError("refinement window indices are invalid")
    if temporal_weight < 0.0 or deviation_weight < 0.0 or max_nfev <= 0:
        raise ValueError("refinement weights must be nonnegative and max_nfev positive")
    bindings = _joint_bindings(model, joint_names)
    sites = tuple(site_objectives)
    orientations = tuple(orientation_objectives)
    pinches = tuple(pinch_objectives)
    collisions = tuple(collision_objectives)
    anchors = tuple(hard_anchors)
    for objective in sites:
        _site_id(model, objective.site_name)
    for objective in orientations:
        _site_id(model, objective.site_name)
    for objective in pinches:
        _site_id(model, objective.first_site)
        _site_id(model, objective.second_site)
    for objective in collisions:
        _geom_id(model, objective.first_geom)
        _geom_id(model, objective.second_geom)
    for frame_index in [
        frame
        for objective in (*sites, *orientations, *pinches, *collisions)
        for frame in objective.frame_indices
    ]:
        if not start_index <= frame_index <= end_index:
            raise ValueError("objective frame lies outside the refinement window")

    frame_count = end_index - start_index + 1
    base = np.column_stack(
        [candidate.qpos[start_index : end_index + 1, binding.qpos_adr] for binding in bindings]
    )
    values = base.copy()
    fixed = np.zeros_like(values, dtype=bool)
    # Window endpoints are hard positional anchors to the unmodified motion.
    if anchor_window_endpoints:
        fixed[0, :] = True
        fixed[-1, :] = True
    binding_index = {binding.name: index for index, binding in enumerate(bindings)}
    for anchor in anchors:
        if anchor.joint_name not in binding_index:
            raise ValueError(f"anchor joint {anchor.joint_name!r} is not selected")
        if not start_index <= anchor.frame_index <= end_index:
            raise ValueError("hard anchor lies outside the refinement window")
        local_frame = anchor.frame_index - start_index
        joint_index = binding_index[anchor.joint_name]
        binding = bindings[joint_index]
        if not binding.minimum <= anchor.value <= binding.maximum:
            raise MotionCompilationError(
                MotionFailureReason.JOINT_LIMIT_VIOLATION,
                f"Hard anchor for {anchor.joint_name!r} exceeds its joint limits",
            )
        values[local_frame, joint_index] = anchor.value
        fixed[local_frame, joint_index] = True

    for frame, joint in zip(*np.nonzero(fixed)):
        binding = bindings[joint]
        if not binding.minimum <= values[frame, joint] <= binding.maximum:
            raise MotionCompilationError(
                MotionFailureReason.JOINT_LIMIT_VIOLATION,
                f"Fixed refinement value for {binding.name!r} exceeds its joint limits",
            )

    variable_cells = [
        (frame, joint)
        for frame in range(frame_count)
        for joint in range(len(bindings))
        if not fixed[frame, joint]
    ]
    variable_column = {cell: index for index, cell in enumerate(variable_cells)}
    x0 = np.asarray([values[cell] for cell in variable_cells], dtype=np.float64)
    lower = np.asarray([bindings[joint].minimum for _, joint in variable_cells], dtype=np.float64)
    upper = np.asarray([bindings[joint].maximum for _, joint in variable_cells], dtype=np.float64)
    x0 = np.clip(x0, lower, upper)
    data = mujoco.MjData(model)
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    def matrix_for(x: np.ndarray) -> np.ndarray:
        matrix = values.copy()
        for column, cell in enumerate(variable_cells):
            matrix[cell] = x[column]
        return matrix

    def forward_site(
        matrix: np.ndarray, local_frame: int, site: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        global_frame = start_index + local_frame
        data.qpos[:] = candidate.qpos[global_frame]
        for joint_index, binding in enumerate(bindings):
            data.qpos[binding.qpos_adr] = matrix[local_frame, joint_index]
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(
            model, data, jacobian_position, jacobian_rotation, site
        )
        return (
            data.site_xpos[site].copy(),
            jacobian_position.copy(),
            data.site_xmat[site].reshape(3, 3).copy(),
            jacobian_rotation.copy(),
        )

    def forward_collision(
        matrix: np.ndarray,
        local_frame: int,
        first_geom: int,
        second_geom: int,
        distance_limit: float,
    ) -> tuple[float, np.ndarray]:
        global_frame = start_index + local_frame
        data.qpos[:] = candidate.qpos[global_frame]
        for joint_index, binding in enumerate(bindings):
            data.qpos[binding.qpos_adr] = matrix[local_frame, joint_index]
        mujoco.mj_forward(model, data)
        fromto = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                first_geom,
                second_geom,
                max(1.0, distance_limit + 0.1),
                fromto,
            )
        )
        delta = fromto[3:] - fromto[:3]
        length = float(np.linalg.norm(delta))
        direction = delta / length if length > 1e-12 else np.zeros(3)
        first_body = int(model.geom_bodyid[first_geom])
        second_body = int(model.geom_bodyid[second_geom])
        first_jacobian = np.zeros((3, model.nv), dtype=np.float64)
        second_jacobian = np.zeros((3, model.nv), dtype=np.float64)
        rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, first_jacobian, rotation, fromto[:3], first_body)
        mujoco.mj_jac(model, data, second_jacobian, rotation, fromto[3:], second_body)
        gradient = direction @ (second_jacobian - first_jacobian)
        return distance, gradient

    def residual_and_jacobian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = matrix_for(x)
        residuals: list[float] = []
        rows: list[np.ndarray] = []

        if deviation_weight > 0.0:
            weight = float(np.sqrt(deviation_weight))
            for column, (frame, joint) in enumerate(variable_cells):
                residuals.append(weight * (matrix[frame, joint] - base[frame, joint]))
                row = np.zeros(len(variable_cells), dtype=np.float64)
                row[column] = weight
                rows.append(row)

        if temporal_weight > 0.0 and frame_count >= 3:
            weight = float(np.sqrt(temporal_weight))
            for frame in range(1, frame_count - 1):
                for joint in range(len(bindings)):
                    residuals.append(
                        weight
                        * (matrix[frame - 1, joint] - 2.0 * matrix[frame, joint] + matrix[frame + 1, joint])
                    )
                    row = np.zeros(len(variable_cells), dtype=np.float64)
                    for cell, coefficient in (
                        ((frame - 1, joint), weight),
                        ((frame, joint), -2.0 * weight),
                        ((frame + 1, joint), weight),
                    ):
                        if cell in variable_column:
                            row[variable_column[cell]] += coefficient
                    rows.append(row)

        for objective in sites:
            site = _site_id(model, objective.site_name)
            targets = np.asarray(objective.targets_m)
            for target_index, global_frame in enumerate(objective.frame_indices):
                local_frame = global_frame - start_index
                position, site_jacobian, _, _ = forward_site(matrix, local_frame, site)
                difference = objective.weight * (position - targets[target_index])
                for axis in range(3):
                    residuals.append(float(difference[axis]))
                    row = np.zeros(len(variable_cells), dtype=np.float64)
                    for joint_index, binding in enumerate(bindings):
                        cell = (local_frame, joint_index)
                        if cell in variable_column:
                            row[variable_column[cell]] = (
                                objective.weight * site_jacobian[axis, binding.dof_adr]
                            )
                    rows.append(row)

        for objective in orientations:
            site = _site_id(model, objective.site_name)
            targets = np.asarray(objective.targets_wxyz)
            for target_index, global_frame in enumerate(objective.frame_indices):
                local_frame = global_frame - start_index
                _, _, rotation, rotational_jacobian = forward_site(
                    matrix, local_frame, site
                )
                target_rotation = _matrix_from_wxyz(targets[target_index])
                difference = objective.weight * _orientation_error(
                    rotation, target_rotation
                )
                for axis in range(3):
                    residuals.append(float(difference[axis]))
                    row = np.zeros(len(variable_cells), dtype=np.float64)
                    for joint_index, binding in enumerate(bindings):
                        cell = (local_frame, joint_index)
                        if cell in variable_column:
                            row[variable_column[cell]] = (
                                -objective.weight
                                * rotational_jacobian[axis, binding.dof_adr]
                            )
                    rows.append(row)

        for objective in pinches:
            first_site = _site_id(model, objective.first_site)
            second_site = _site_id(model, objective.second_site)
            targets = np.asarray(objective.target_distance_m)
            for target_index, global_frame in enumerate(objective.frame_indices):
                local_frame = global_frame - start_index
                first_position, first_jacobian, _, _ = forward_site(
                    matrix, local_frame, first_site
                )
                second_position, second_jacobian, _, _ = forward_site(
                    matrix, local_frame, second_site
                )
                delta = first_position - second_position
                distance = float(np.linalg.norm(delta))
                direction = delta / distance if distance > 1e-12 else np.zeros(3)
                residuals.append(objective.weight * (distance - targets[target_index]))
                row = np.zeros(len(variable_cells), dtype=np.float64)
                for joint_index, binding in enumerate(bindings):
                    cell = (local_frame, joint_index)
                    if cell in variable_column:
                        derivative = float(
                            direction
                            @ (
                                first_jacobian[:, binding.dof_adr]
                                - second_jacobian[:, binding.dof_adr]
                            )
                        )
                        row[variable_column[cell]] = objective.weight * derivative
                rows.append(row)
        for objective in collisions:
            first_geom = _geom_id(model, objective.first_geom)
            second_geom = _geom_id(model, objective.second_geom)
            for global_frame in objective.frame_indices:
                local_frame = global_frame - start_index
                distance, distance_jacobian = forward_collision(
                    matrix,
                    local_frame,
                    first_geom,
                    second_geom,
                    objective.minimum_distance_m,
                )
                deficit = objective.minimum_distance_m - distance
                if deficit <= 0.0:
                    continue
                residuals.append(objective.weight * deficit)
                row = np.zeros(len(variable_cells), dtype=np.float64)
                for joint_index, binding in enumerate(bindings):
                    cell = (local_frame, joint_index)
                    if cell in variable_column:
                        row[variable_column[cell]] = (
                            -objective.weight
                            * distance_jacobian[binding.dof_adr]
                        )
                rows.append(row)
        if not residuals:
            return np.empty(0, dtype=np.float64), np.empty((0, len(variable_cells)))
        return np.asarray(residuals), np.vstack(rows)

    if variable_cells and (
        sites
        or orientations
        or pinches
        or collisions
        or temporal_weight > 0.0
        or deviation_weight > 0.0
    ):
        optimization = least_squares(
            lambda x: residual_and_jacobian(x)[0],
            x0,
            jac=lambda x: residual_and_jacobian(x)[1],
            bounds=(lower, upper),
            method="trf",
            # Task-space windows produce a tall, structured Jacobian.  The
            # exact dense SVD solver scales quadratically and can exhaust
            # memory even for a slow 1 Hz grasp program.  LSMR solves the same
            # bounded least-squares objective iteratively without materializing
            # the augmented dense SVD and remains deterministic for fixed data.
            tr_solver="lsmr",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        optimized = matrix_for(optimization.x)
    else:
        optimized = values

    violations: list[dict[str, float | int | str]] = []
    for objective in sites:
        site = _site_id(model, objective.site_name)
        targets = np.asarray(objective.targets_m)
        for target_index, global_frame in enumerate(objective.frame_indices):
            position, _, _, _ = forward_site(
                optimized, global_frame - start_index, site
            )
            error = float(np.linalg.norm(position - targets[target_index]))
            if error > objective.tolerance_m:
                violations.append(
                    {"objective": objective.site_name, "frame": global_frame, "error_m": error}
                )
    for objective in pinches:
        first_site = _site_id(model, objective.first_site)
        second_site = _site_id(model, objective.second_site)
        targets = np.asarray(objective.target_distance_m)
        for target_index, global_frame in enumerate(objective.frame_indices):
            first, _, _, _ = forward_site(
                optimized, global_frame - start_index, first_site
            )
            second, _, _, _ = forward_site(
                optimized, global_frame - start_index, second_site
            )
            error = abs(float(np.linalg.norm(first - second)) - float(targets[target_index]))
            if error > objective.tolerance_m:
                violations.append(
                    {
                        "objective": f"{objective.first_site}:{objective.second_site}",
                        "frame": global_frame,
                        "error_m": error,
                    }
                )
    for objective in orientations:
        site = _site_id(model, objective.site_name)
        targets = np.asarray(objective.targets_wxyz)
        for target_index, global_frame in enumerate(objective.frame_indices):
            _, _, rotation, _ = forward_site(
                optimized, global_frame - start_index, site
            )
            error = float(
                np.linalg.norm(
                    _orientation_error(
                        rotation, _matrix_from_wxyz(targets[target_index])
                    )
                )
            )
            if error > objective.tolerance_rad:
                violations.append(
                    {
                        "objective": f"{objective.site_name}:orientation",
                        "frame": global_frame,
                        "error_rad": error,
                    }
                )
    for objective in collisions:
        first_geom = _geom_id(model, objective.first_geom)
        second_geom = _geom_id(model, objective.second_geom)
        for global_frame in objective.frame_indices:
            distance, _ = forward_collision(
                optimized,
                global_frame - start_index,
                first_geom,
                second_geom,
                objective.minimum_distance_m,
            )
            deficit = objective.minimum_distance_m - distance
            if deficit > objective.tolerance_m:
                violations.append(
                    {
                        "objective": f"{objective.first_geom}:{objective.second_geom}:collision",
                        "frame": global_frame,
                        "error_m": deficit,
                    }
                )
    if violations:
        raise MotionCompilationError(
            MotionFailureReason.IK_INFEASIBLE,
            "Joint window cannot satisfy its task-space/collision tolerances within limits",
            details={"violations": violations},
        )

    qpos = candidate.qpos.copy()
    for local_frame in range(frame_count):
        global_frame = start_index + local_frame
        for joint_index, binding in enumerate(bindings):
            qpos[global_frame, binding.qpos_adr] = optimized[local_frame, joint_index]
    qvel = candidate.qvel.copy()
    qacc = candidate.qacc.copy()
    edge_order = 2 if len(candidate.times_s) >= 3 else 1
    for binding in bindings:
        velocity = np.gradient(
            qpos[:, binding.qpos_adr], candidate.times_s, edge_order=edge_order
        )
        acceleration = np.gradient(velocity, candidate.times_s, edge_order=edge_order)
        qvel[:, binding.dof_adr] = velocity
        qacc[:, binding.dof_adr] = acceleration
    candidate_id = "candidate-" + content_hash(
        {
            "base": candidate.content_hash(),
            "window": [start_index, end_index],
            "joints": [binding.name for binding in bindings],
            "qpos": qpos.tolist(),
        }
    )[:20]
    return CandidateTrajectoryV1(
        candidate_id=candidate_id,
        program_hash=candidate.program_hash,
        rig_hash=candidate.rig_hash,
        times_s=candidate.times_s,
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        quaternion_qpos_adrs=candidate.quaternion_qpos_adrs,
        contact_plateaus=candidate.contact_plateaus,
    )
