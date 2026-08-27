"""Repeat-run comparison for native MuJoCo traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .runtime import NativeMujocoRuntime, SimulationRequest, SimulationResult, SimulationStatus


@dataclass(frozen=True)
class ReplayTolerance:
    qpos_atol: float = 1e-10
    qvel_atol: float = 1e-10
    ctrl_atol: float = 1e-10
    contact_atol: float = 1e-9


@dataclass(frozen=True)
class ReplayComparison:
    matches: bool
    reason: str | None
    max_qpos_error: float
    max_qvel_error: float
    max_ctrl_error: float
    contact_topology_matches: bool


@dataclass(frozen=True)
class ReplayReport:
    deterministic: bool
    runs: tuple[SimulationResult, ...]
    comparisons: tuple[ReplayComparison, ...]


def _max_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float(np.max(np.abs(first - second))) if first.size else 0.0


def _compare_contacts(
    first: SimulationResult,
    second: SimulationResult,
    atol: float,
) -> bool:
    if len(first.trace.contacts) != len(second.trace.contacts):
        return False
    for first_frame, second_frame in zip(first.trace.contacts, second.trace.contacts, strict=True):
        if len(first_frame.contacts) != len(second_frame.contacts):
            return False
        for left, right in zip(first_frame.contacts, second_frame.contacts, strict=True):
            if (left.geom1_name, left.geom2_name) != (right.geom1_name, right.geom2_name):
                return False
            numeric_left = np.asarray(
                (left.distance_m, left.normal_force_n, *left.position_m), dtype=np.float64
            )
            numeric_right = np.asarray(
                (right.distance_m, right.normal_force_n, *right.position_m), dtype=np.float64
            )
            if not np.allclose(numeric_left, numeric_right, rtol=0.0, atol=atol):
                return False
    return True


def compare_replays(
    first: SimulationResult,
    second: SimulationResult,
    tolerance: ReplayTolerance | None = None,
) -> ReplayComparison:
    limits = tolerance or ReplayTolerance()
    qpos_error = _max_error(first.trace.qpos, second.trace.qpos)
    qvel_error = _max_error(first.trace.qvel, second.trace.qvel)
    ctrl_error = _max_error(first.trace.ctrl, second.trace.ctrl)
    contacts_match = _compare_contacts(first, second, limits.contact_atol)
    reason: str | None = None
    if first.status is not SimulationStatus.COMPLETED or second.status is not SimulationStatus.COMPLETED:
        reason = "one or both simulations did not complete"
    elif first.model_hash != second.model_hash:
        reason = "model hashes differ"
    elif not np.array_equal(first.trace.times_s, second.trace.times_s):
        reason = "sample times differ"
    elif qpos_error > limits.qpos_atol:
        reason = "qpos traces diverged"
    elif qvel_error > limits.qvel_atol:
        reason = "qvel traces diverged"
    elif ctrl_error > limits.ctrl_atol:
        reason = "control traces diverged"
    elif not contacts_match:
        reason = "contact traces diverged"
    return ReplayComparison(
        matches=reason is None,
        reason=reason,
        max_qpos_error=qpos_error,
        max_qvel_error=qvel_error,
        max_ctrl_error=ctrl_error,
        contact_topology_matches=contacts_match,
    )


def repeat_replay(
    runtime: NativeMujocoRuntime,
    request: SimulationRequest,
    *,
    runs: int = 3,
    tolerance: ReplayTolerance | None = None,
) -> ReplayReport:
    if runs < 2:
        raise ValueError("repeat replay requires at least two runs")
    results = tuple(runtime.simulate(request) for _ in range(runs))
    baseline = results[0]
    comparisons = tuple(
        compare_replays(baseline, result, tolerance) for result in results[1:]
    )
    return ReplayReport(
        deterministic=all(comparison.matches for comparison in comparisons),
        runs=results,
        comparisons=comparisons,
    )

