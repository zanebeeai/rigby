"""Physical evidence independent of a particular robot or controller.

Capture does not advance or recompute live physics. Integration state includes
warmstart accelerations, actuator state, controls and user inputs. Derived
contacts/sensors after mj_step describe the preceding integration interval;
they are retained as such, rather than relabelled as a fresh current observation.
See https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility
and the adjacent discussion of consistency in mjData.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np


STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
INPUT_SPEC = int(mujoco.mjtState.mjSTATE_USER) & ~int(mujoco.mjtState.mjSTATE_CTRL)


def _require_self_contained_physics(model: mujoco.MjModel) -> None:
    if model.nplugin:
        raise ValueError("Replay recording does not yet support model plugins")
    for name in ("control", "passive", "sensor", "act_gain", "act_bias", "act_dyn", "contactfilter"):
        if getattr(mujoco, "get_mjcb_" + name)() is not None:
            raise ValueError(f"Replay recording cannot capture external mjcb_{name}")


@dataclass(frozen=True)
class PhysicsRecord:
    arrays: dict[str, np.ndarray]

    def content_hash(self) -> str:
        """Hash typed array contents, excluding archive timestamps and run IDs."""
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays.items()):
            array = np.ascontiguousarray(value)
            header = json.dumps([name, array.dtype.str, array.shape], separators=(",", ":"))
            digest.update(header.encode("utf-8") + b"\0")
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to_bytes(self) -> bytes:
        stream = io.BytesIO()
        np.savez_compressed(stream, **self.arrays)
        return stream.getvalue()

    @classmethod
    def from_bytes(cls, value: bytes) -> PhysicsRecord:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        return cls(arrays)


class _Column:
    """Rows of one shape kept in numpy chunks, so a recording's memory is its
    data and not a Python object per sample: a crawler's trial of six
    hundred seconds logs three and a half million contacts, and a list of
    thirteen Python floats for each of them was a memory error at finish.
    Reads by index and slice, and a write to the last row, behave as the
    list did.
    """

    CHUNK = 16384

    def __init__(self, shape: tuple[int, ...], dtype: type) -> None:
        self.shape = shape
        self.dtype = dtype
        self.chunks: list[np.ndarray] = []
        self.buffer = np.empty((self.CHUNK, *shape), dtype=dtype)
        self.fill = 0

    def append(self, row: object) -> None:
        if self.fill == self.CHUNK:
            self.chunks.append(self.buffer)
            self.buffer = np.empty((self.CHUNK, *self.shape), dtype=self.dtype)
            self.fill = 0
        self.buffer[self.fill] = row
        self.fill += 1

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self.chunks) + self.fill

    def _locate(self, index: int) -> tuple[np.ndarray, int]:
        count = len(self)
        position = index + count if index < 0 else index
        if not 0 <= position < count:
            raise IndexError(index)
        for chunk in self.chunks:
            if position < len(chunk):
                return chunk, position
            position -= len(chunk)
        return self.buffer, position

    def __getitem__(self, index: int | slice) -> np.ndarray:
        if isinstance(index, slice):
            return self.array()[index]
        chunk, position = self._locate(index)
        return chunk[position]

    def __setitem__(self, index: int, value: object) -> None:
        chunk, position = self._locate(index)
        chunk[position] = value

    def array(self) -> np.ndarray:
        """Every row as one array; the rows are kept as that one chunk afterwards, so a record and its recorder share them."""
        parts = [*self.chunks, self.buffer[:self.fill]]
        whole = np.concatenate(parts, axis=0) if len(self) else np.empty((0, *self.shape), dtype=self.dtype)
        self.chunks = [whole] if len(whole) else []
        self.buffer = np.empty((self.CHUNK, *self.shape), dtype=self.dtype)
        self.fill = 0
        return whole


class PhysicsRecorder:
    """Observe samples immediately before commands are applied to mj_step.

    The final sample has a computed command but no following step. Callers must
    retain a sample after the last integrated interval, including failed runs.
    Plugins and external callbacks need their own pinned implementation/state;
    this first recorder deliberately refuses them instead of implying support.
    """

    def __init__(self, model: mujoco.MjModel):
        _require_self_contained_physics(model)
        self.model = model
        self.rows: dict[str, _Column] = {
            "time_s": _Column((), np.float64),
            "control_time_s": _Column((), np.float64),
            "state": _Column((mujoco.mj_stateSize(model, STATE_SPEC),), np.float64),
            "qpos": _Column((model.nq,), np.float64),
            "qvel": _Column((model.nv,), np.float64),
            "action": _Column((model.nu,), np.float64),
            "demand": _Column((model.nu,), np.float64),
            "user_input": _Column((mujoco.mj_stateSize(model, INPUT_SPEC),), np.float64),
            "sensordata": _Column((model.nsensordata,), np.float64),
            "qfrc_actuator": _Column((model.nv,), np.float64),
            "contact_sample_time_s": _Column((), np.float64),
            "contact_offsets": _Column((), np.int64),
            "contact_geom": _Column((2,), np.int64),
            "contact_geometry": _Column((13,), np.float64),
            "contact_wrench": _Column((6,), np.float64),
        }
        self.rows["contact_offsets"].append(0)

    def capture(
        self, data: mujoco.MjData, action: np.ndarray, *,
        control_time_s: float, demand: np.ndarray | None = None,
    ) -> None:
        model = self.model
        action = np.asarray(action, dtype=np.float64)
        demand = action if demand is None else np.asarray(demand, dtype=np.float64)
        if action.shape != (model.nu,) or demand.shape != (model.nu,):
            raise ValueError("Action/demand dimensions must match the model")
        if self.rows["time_s"] and data.time <= self.rows["time_s"][-1]:
            raise ValueError("Recorded simulation time must strictly increase")
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        rows = self.rows
        rows["time_s"].append(float(data.time))
        rows["control_time_s"].append(float(control_time_s))
        rows["state"].append(state)
        user_input = np.empty(mujoco.mj_stateSize(model, INPUT_SPEC), dtype=np.float64)
        mujoco.mj_getState(model, data, user_input, INPUT_SPEC)
        rows["user_input"].append(user_input)
        for name, value in (
            ("qpos", data.qpos), ("qvel", data.qvel), ("action", action),
            ("demand", demand), ("sensordata", data.sensordata),
            ("qfrc_actuator", data.qfrc_actuator),
        ):
            rows[name].append(np.array(value, dtype=np.float64, copy=True))
        rows["contact_sample_time_s"].append(
            float(data.time) if len(rows["time_s"]) == 1 else float(data.time - model.opt.timestep)
        )
        wrench = np.zeros(6, dtype=np.float64)
        geometry = np.empty(13, dtype=np.float64)
        for i in range(data.ncon):
            contact = data.contact[i]
            mujoco.mj_contactForce(model, data, i, wrench)
            rows["contact_geom"].append((int(contact.geom1), int(contact.geom2)))
            geometry[0] = float(contact.dist)
            geometry[1:4] = contact.pos
            geometry[4:13] = contact.frame
            rows["contact_geometry"].append(geometry)
            rows["contact_wrench"].append(wrench)
        rows["contact_offsets"].append(len(rows["contact_geom"]))

    def amend_last_action(self, action: np.ndarray, *, demand: np.ndarray | None = None) -> None:
        """Replace the command on the final sample.

        The final sample of a run carries a computed command that no step
        applied. When a second run continues from exactly that state and
        time, the command actually applied from it is the second run's
        first, and the record must say so for the recorded controls to
        replay to the recorded states across the boundary.
        """

        if not self.rows["time_s"]:
            raise ValueError("Nothing recorded to amend")
        model = self.model
        action = np.asarray(action, dtype=np.float64)
        demand = action if demand is None else np.asarray(demand, dtype=np.float64)
        if action.shape != (model.nu,) or demand.shape != (model.nu,):
            raise ValueError("Action/demand dimensions must match the model")
        self.rows["action"][-1] = np.array(action, dtype=np.float64, copy=True)
        self.rows["demand"][-1] = np.array(demand, dtype=np.float64, copy=True)

    def finish(self) -> PhysicsRecord:
        if not self.rows["time_s"]:
            raise ValueError("An evidence record needs at least the initial state")
        # one column at a time: the peak is the data plus one column's copy, not twice the data
        arrays = {key: column.array() for key, column in self.rows.items()}
        for key, width in (("contact_geom", 2), ("contact_geometry", 13), ("contact_wrench", 6)):
            arrays[key] = arrays[key].reshape(-1, width)
        arrays["state_spec"] = np.asarray(STATE_SPEC, dtype=np.int64)
        arrays["input_spec"] = np.asarray(INPUT_SPEC, dtype=np.int64)
        arrays["timestep_s"] = np.asarray(self.model.opt.timestep, dtype=np.float64)
        return PhysicsRecord(arrays)


def replay_physics(
    model: mujoco.MjModel, record: PhysicsRecord, *,
    controller: Callable[[mujoco.MjData, int], np.ndarray] | None = None,
) -> dict[str, object]:
    """Integrate from one initial state; never reset to intermediate samples.

    The default replays recorded controls. A caller-supplied controller may
    instead recompute every command, which is separately checked for equality.
    Exact equality is the declared same-version/platform baseline criterion.
    This does not evaluate task success or cross-platform robustness.
    """
    arrays = record.arrays
    _require_self_contained_physics(model)
    states, actions = arrays["state"], arrays["action"]
    if int(arrays["state_spec"]) != STATE_SPEC or states.ndim != 2 or len(states) < 1:
        raise ValueError("Unsupported or empty integration-state record")
    if states.shape[1] != mujoco.mj_stateSize(model, STATE_SPEC):
        raise ValueError("Integration-state dimensions do not match the model")
    if actions.shape != (len(states), model.nu):
        raise ValueError("Recorded action dimensions do not match the model")
    if int(arrays["input_spec"]) != INPUT_SPEC or arrays["user_input"].shape != (
        len(states), mujoco.mj_stateSize(model, INPUT_SPEC)
    ):
        raise ValueError("Unsupported user-input schedule")
    if float(arrays["timestep_s"]) != float(model.opt.timestep):
        raise ValueError("Recorded timestep does not match the model")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        return {"agrees": False, "reason": "nonfinite_record", "samples": len(states)}
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, states[0], STATE_SPEC)
    # mj_step performs the forward pipeline itself. Calling mj_forward here
    # would replace saved warmstart values before the first recorded step.
    current = np.empty(states.shape[1], dtype=np.float64)
    max_error = 0.0
    first_state_difference = None
    first_action_difference = None
    for i in range(len(states)):
        # Reapply declared external forces/mocap/equality/user inputs, never
        # robot/object qpos, velocity, actuator state, time or warmstart values.
        mujoco.mj_setState(model, data, arrays["user_input"][i], INPUT_SPEC)
        mujoco.mj_getState(model, data, current, STATE_SPEC)
        error = float(np.max(np.abs(current - states[i]), initial=0.0))
        max_error = max(max_error, error)
        if not np.array_equal(current, states[i]) and first_state_difference is None:
            first_state_difference = i
        command = actions[i] if controller is None else np.asarray(controller(data, i))
        if command.shape != (model.nu,) or not np.array_equal(command, actions[i]):
            if first_action_difference is None:
                first_action_difference = i
        if i + 1 < len(states):
            data.ctrl[:] = command
            mujoco.mj_step(model, data)
    return {
        "agrees": first_state_difference is None and first_action_difference is None,
        "mode": "recorded_controls" if controller is None else "recomputed_controller",
        "samples": len(states), "integrated_steps": len(states) - 1,
        "tolerance": {"absolute": 0.0, "relative": 0.0},
        "max_state_error": max_error,
        "first_state_difference": first_state_difference,
        "first_action_difference": first_action_difference,
    }
