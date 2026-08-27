from __future__ import annotations

import math

import numpy as np

from rigby_v2.contracts import Quaternion, QuaternionConvention, QuaternionOrder


def _to_wxyz(quaternion: Quaternion) -> np.ndarray:
    values = np.asarray(quaternion.values, dtype=np.float64)
    if quaternion.convention.order is QuaternionOrder.WXYZ:
        return values.copy()
    return values[[3, 0, 1, 2]].copy()


def _from_wxyz(values: np.ndarray, convention: QuaternionConvention) -> Quaternion:
    normalized = values / np.linalg.norm(values)
    if convention.order is QuaternionOrder.XYZW:
        normalized = normalized[[1, 2, 3, 0]]
    return Quaternion(values=tuple(float(value) for value in normalized), convention=convention)


def _require_same_convention(*quaternions: Quaternion) -> QuaternionConvention:
    if not quaternions:
        raise ValueError("at least one quaternion is required")
    convention = quaternions[0].convention
    if any(quaternion.convention != convention for quaternion in quaternions[1:]):
        raise ValueError("quaternion interpolation requires an identical explicit convention")
    return convention


def slerp_wxyz(start: np.ndarray, end: np.ndarray, progress: float) -> np.ndarray:
    """Shortest-path SLERP for internal, explicitly MuJoCo-ordered values."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if start.shape != (4,) or end.shape != (4,):
        raise ValueError("SLERP inputs must contain four WXYZ components")
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    progress = float(np.clip(progress, 0.0, 1.0))
    if dot > 0.9995:
        result = start + progress * (end - start)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sine = math.sin(angle)
    result = (
        math.sin((1.0 - progress) * angle) / sine * start
        + math.sin(progress * angle) / sine * end
    )
    return result / np.linalg.norm(result)


def slerp(start: Quaternion, end: Quaternion, progress: float) -> Quaternion:
    convention = _require_same_convention(start, end)
    return _from_wxyz(slerp_wxyz(_to_wxyz(start), _to_wxyz(end), progress), convention)


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _inverse(value: np.ndarray) -> np.ndarray:
    conjugate = value.copy()
    conjugate[1:] *= -1.0
    return conjugate / float(np.dot(value, value))


def _log(value: np.ndarray) -> np.ndarray:
    value = value / np.linalg.norm(value)
    vector = value[1:]
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = math.atan2(length, float(value[0]))
    return vector * (angle / length)


def _exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.asarray([1.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    scale = math.sin(angle) / angle
    return np.asarray([math.cos(angle), *(vector * scale)], dtype=np.float64)


def _aligned(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    return -value if np.dot(reference, value) < 0.0 else value


def _squad_tangent(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> np.ndarray:
    previous = _aligned(current, previous)
    following = _aligned(current, following)
    inverse = _inverse(current)
    tangent_vector = -0.25 * (
        _log(_multiply(inverse, previous)) + _log(_multiply(inverse, following))
    )
    tangent = _multiply(current, _exp(tangent_vector))
    return tangent / np.linalg.norm(tangent)


def squad(
    previous: Quaternion,
    start: Quaternion,
    end: Quaternion,
    following: Quaternion,
    progress: float,
) -> Quaternion:
    """Spherical cubic interpolation through ``start`` and ``end``."""

    convention = _require_same_convention(previous, start, end, following)
    q_previous, q_start, q_end, q_following = (
        _to_wxyz(value) for value in (previous, start, end, following)
    )
    start_tangent = _squad_tangent(q_previous, q_start, q_end)
    end_tangent = _squad_tangent(q_start, q_end, q_following)
    progress = float(np.clip(progress, 0.0, 1.0))
    primary = slerp_wxyz(q_start, q_end, progress)
    tangent = slerp_wxyz(start_tangent, end_tangent, progress)
    result = slerp_wxyz(primary, tangent, 2.0 * progress * (1.0 - progress))
    return _from_wxyz(result, convention)
