from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from rigby_v2.hashing import content_hash
from rigby_v2.contracts import (
    CandidateContactPlateauV1 as PersistedContactPlateauV1,
    CandidateTrajectoryV1 as PersistedCandidateTrajectoryV1,
    TrajectorySampleV1,
)
from rigby_v2.simulation.controller import ControlTarget

from .rotations import slerp_wxyz


FloatArray = npt.NDArray[np.float64]


def _immutable_array(value: npt.ArrayLike, dimensions: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ContactPlateauV1:
    contact_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class CandidateTrajectoryV1:
    """Runtime-ready generalized target trajectory produced by the compiler."""

    candidate_id: str
    program_hash: str
    rig_hash: str
    times_s: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    qacc: FloatArray
    quaternion_qpos_adrs: tuple[int, ...] = ()
    contact_plateaus: tuple[ContactPlateauV1, ...] = ()
    schema_version: str = "candidate_trajectory.v1"

    def __post_init__(self) -> None:
        times = _immutable_array(self.times_s, 1, "times_s")
        qpos = _immutable_array(self.qpos, 2, "qpos")
        qvel = _immutable_array(self.qvel, 2, "qvel")
        qacc = _immutable_array(self.qacc, 2, "qacc")
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing with at least two samples")
        if qpos.shape[0] != len(times) or qvel.shape[0] != len(times) or qacc.shape != qvel.shape:
            raise ValueError("trajectory arrays must share their sample dimension")
        for address in self.quaternion_qpos_adrs:
            if address < 0 or address + 4 > qpos.shape[1]:
                raise ValueError("quaternion qpos address is outside qpos")
            norms = np.linalg.norm(qpos[:, address : address + 4], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
                raise ValueError("trajectory quaternions must remain normalized")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "qvel", qvel)
        object.__setattr__(self, "qacc", qacc)

    def sample(self, time_s: float) -> ControlTarget:
        time_s = float(np.clip(time_s, self.times_s[0], self.times_s[-1]))
        right = int(np.searchsorted(self.times_s, time_s, side="right"))
        right = min(max(right, 1), len(self.times_s) - 1)
        left = right - 1
        span = float(self.times_s[right] - self.times_s[left])
        alpha = (time_s - float(self.times_s[left])) / span
        qpos = (1.0 - alpha) * self.qpos[left] + alpha * self.qpos[right]
        for address in self.quaternion_qpos_adrs:
            qpos[address : address + 4] = slerp_wxyz(
                self.qpos[left, address : address + 4],
                self.qpos[right, address : address + 4],
                alpha,
            )
        return ControlTarget(
            qpos=qpos,
            qvel=(1.0 - alpha) * self.qvel[left] + alpha * self.qvel[right],
            qacc=(1.0 - alpha) * self.qacc[left] + alpha * self.qacc[right],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "program_hash": self.program_hash,
            "rig_hash": self.rig_hash,
            "times_s": self.times_s.tolist(),
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "qacc": self.qacc.tolist(),
            "quaternion_qpos_adrs": self.quaternion_qpos_adrs,
            "contact_plateaus": [plateau.__dict__ for plateau in self.contact_plateaus],
        }

    def content_hash(self) -> str:
        return content_hash(self.payload())

    def to_contract(self, *, rig_id: str) -> PersistedCandidateTrajectoryV1:
        """Freeze the runtime trajectory into the durable queue contract."""

        return PersistedCandidateTrajectoryV1(
            candidate_id=self.candidate_id,
            rig_id=rig_id,
            program_hash=self.program_hash,
            rig_hash=self.rig_hash,
            quaternion_qpos_adrs=self.quaternion_qpos_adrs,
            contact_plateaus=tuple(
                PersistedContactPlateauV1(
                    contact_id=value.contact_id,
                    start_s=value.start_s,
                    end_s=value.end_s,
                )
                for value in self.contact_plateaus
            ),
            samples=tuple(
                TrajectorySampleV1(
                    time_s=float(self.times_s[index]),
                    qpos=tuple(map(float, self.qpos[index])),
                    qvel=tuple(map(float, self.qvel[index])),
                    qacc=tuple(map(float, self.qacc[index])),
                )
                for index in range(len(self.times_s))
            ),
        )
