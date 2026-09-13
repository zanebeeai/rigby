"""Declared sensor packets and a process boundary for trusted acting policies.

This prevents accidental API / inherited-Python-state leakage. It is **not** an
OS sandbox: same-user code could deliberately open files, use the network, or
inspect other processes. Sensor adapters and declarations are trusted. A numeric
value cannot establish that its source was a physical sensor rather than an
oracle; the adapter must be audited separately.

All persisted inputs are immutable, finite, extra-forbidding JSON contracts.
Evaluator truth and a fully observed diagnostic baseline have distinct types and
cannot be projected or sent to an acting worker. The worker starts a fresh Python
interpreter, imports an explicitly named policy, and receives only JSON sensor
packets. It never receives a simulator object or a pickled Python closure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import (
    BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator,
    model_validator,
)

from .hashing import canonical_json


MAX_WIRE_BYTES = 32 * 1024 * 1024
MAX_SENSOR_VALUES = 4 * 1024 * 1024
Number = StrictInt | StrictFloat
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z")
_PRIVILEGED_NAMES = frozenset({
    "object_poses", "success_labels", "simulator_state", "ground_truth",
    "evaluator", "model", "data", "mjmodel", "mjdata",
})


class ObservationBoundaryError(ValueError):
    """An input is not a complete, declared sensor observation."""


class PolicyWorkerError(RuntimeError):
    """A worker failed, returned an invalid action, or violated its protocol."""


class PolicyWorkerTimeout(PolicyWorkerError):
    """The worker exceeded its deadline and has been stopped and reaped."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True,
        allow_inf_nan=False, serialize_by_alias=True, validate_by_name=True,
    )

    def canonical_json(self) -> str:
        return canonical_json(self)

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Expected a nonempty portable identifier")
    return value


class SensorChannel(_FrozenModel):
    """A declared sensor channel; values are flattened in row-major order.

    Positions / velocities are one-dimensional, grouped by physical unit, with
    an optional entity name for each element. RGB is H x W x 3 uint8; depth is
    H x W meters. Contact channels contain nonnegative measured normal force in
    newtons or binary 0/1 contact readings. No contact-pair object IDs or poses
    are included in readings. ``frame`` names the declared sensor frame.
    """

    name: str
    kind: Literal["joint_position", "joint_velocity", "rgb", "depth", "contact"]
    shape: tuple[StrictInt, ...]
    unit: Literal[
        "radian", "meter", "radian_per_second", "meter_per_second",
        "uint8", "newton", "binary",
    ]
    entities: tuple[str, ...] = ()
    frame: str | None = None

    @field_validator("name")
    @classmethod
    def channel_name(cls, value: str) -> str:
        _identifier(value)
        if value.casefold() in _PRIVILEGED_NAMES:
            raise ValueError("Privileged fields cannot be declared sensor channels")
        return value

    @field_validator("entities")
    @classmethod
    def entity_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _identifier(value)
        if len(set(values)) != len(values):
            raise ValueError("Duplicate entity names")
        return values

    @field_validator("frame")
    @classmethod
    def frame_name(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else None

    @model_validator(mode="after")
    def dimensions_and_units(self) -> Self:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError("Sensor dimensions must be positive integers")
        if math.prod(self.shape) > MAX_SENSOR_VALUES:
            raise ValueError("Sensor channel exceeds the bounded packet size")
        allowed = {
            "joint_position": (1, {"radian", "meter"}),
            "joint_velocity": (1, {"radian_per_second", "meter_per_second"}),
            "rgb": (3, {"uint8"}),
            "depth": (2, {"meter"}),
            "contact": (1, {"newton", "binary"}),
        }
        rank, units = allowed[self.kind]
        if len(self.shape) != rank or self.unit not in units:
            raise ValueError("Sensor shape or unit does not match its declared kind")
        if self.kind == "rgb" and self.shape[-1] != 3:
            raise ValueError("RGB channels require exactly three color components")
        if self.entities and (
            self.kind in {"rgb", "depth"} or len(self.entities) != self.shape[0]
        ):
            raise ValueError("Entities must name each scalar non-image sensor")
        return self


class SensorDeclaration(_FrozenModel):
    schema_version: Literal["rigby.sensor-declaration/1"] = Field(
        default="rigby.sensor-declaration/1", alias="schema",
    )
    channels: tuple[SensorChannel, ...]

    @model_validator(mode="after")
    def unique_channels(self) -> Self:
        names = [channel.name for channel in self.channels]
        if not names or len(names) != len(set(names)):
            raise ValueError("Declare at least one channel with unique names")
        if sum(math.prod(channel.shape) for channel in self.channels) > MAX_SENSOR_VALUES:
            raise ValueError("Sensor declaration exceeds the bounded packet size")
        return self


class SensorReading(_FrozenModel):
    channel: str
    values: tuple[Number, ...]

    @field_validator("channel")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value)


class PolicyObservation(_FrozenModel):
    schema_version: Literal["rigby.policy-observation/1"] = Field(
        default="rigby.policy-observation/1", alias="schema",
    )
    access: Literal["declared_sensors"] = "declared_sensors"
    sequence: StrictInt = Field(ge=0)
    time_s: StrictFloat = Field(ge=0)
    declaration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readings: tuple[SensorReading, ...]


class ObjectPose(_FrozenModel):
    object_id: str
    position_xyz: tuple[Number, Number, Number]
    orientation_wxyz: tuple[Number, Number, Number, Number]

    @field_validator("object_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("orientation_wxyz")
    @classmethod
    def unit_quaternion(cls, value: tuple[Number, Number, Number, Number]):
        if not math.isclose(sum(x * x for x in value), 1.0, abs_tol=1e-6, rel_tol=0.0):
            raise ValueError("Evaluator pose requires a unit wxyz quaternion")
        return value


class PredicateValue(_FrozenModel):
    name: str
    satisfied: bool

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value)


class EvaluatorTruth(_FrozenModel):
    """Privileged state for scoring; never a policy observation."""

    schema_version: Literal["rigby.evaluator-truth/1"] = Field(
        default="rigby.evaluator-truth/1", alias="schema",
    )
    access: Literal["evaluator_only"] = "evaluator_only"
    time_s: StrictFloat = Field(ge=0)
    object_poses: tuple[ObjectPose, ...] = ()
    success_labels: tuple[PredicateValue, ...] = ()
    simulator_state: tuple[Number, ...] = ()

    @model_validator(mode="after")
    def unique_entities(self) -> Self:
        for values in (
            [pose.object_id for pose in self.object_poses],
            [predicate.name for predicate in self.success_labels],
        ):
            if len(values) != len(set(values)):
                raise ValueError("Evaluator truth contains duplicate identifiers")
        return self


class FullyObservedDiagnostic(_FrozenModel):
    """Explicitly privileged baseline input, rejected by PolicyWorker.act()."""

    schema_version: Literal["rigby.fully-observed-diagnostic/1"] = Field(
        default="rigby.fully-observed-diagnostic/1", alias="schema",
    )
    access: Literal["fully_observed_diagnostic"] = "fully_observed_diagnostic"
    observation: PolicyObservation
    evaluator_truth: EvaluatorTruth


class PolicyAction(_FrozenModel):
    values: tuple[Number, ...]


def _check_storage(value: object) -> None:
    """Reject Python validation bypasses instead of serializing away extras."""
    if isinstance(value, BaseModel):
        if type(value) not in {
            SensorChannel, SensorDeclaration, SensorReading, PolicyObservation, PolicyAction,
        }:
            raise ObservationBoundaryError("Unexpected model type at the sensor boundary")
        if set(value.__dict__) != set(type(value).model_fields) or value.__pydantic_extra__:
            raise ObservationBoundaryError("Undeclared model storage at the sensor boundary")
        if value.__pydantic_private__:
            raise ObservationBoundaryError("Private model storage cannot cross the sensor boundary")
        for item in value.__dict__.values():
            _check_storage(item)
    elif isinstance(value, tuple):
        for item in value:
            _check_storage(item)
    elif value is not None and type(value) not in {str, int, float, bool}:
        raise ObservationBoundaryError("Only immutable declared JSON values may cross the boundary")


class ObservationProjector:
    """Build complete packets, rejecting rather than dropping unexpected fields."""

    def __init__(self, declaration: SensorDeclaration):
        if type(declaration) is not SensorDeclaration:
            raise ObservationBoundaryError("Expected an explicit SensorDeclaration")
        # Revalidate even model_copy/model_construct bypasses before trusting it.
        _check_storage(declaration)
        self.declaration = SensorDeclaration.model_validate_json(declaration.canonical_json())
        self.declaration_sha256 = self.declaration.content_hash()

    def project(
        self, *, sequence: int, time_s: float,
        readings: Mapping[str, Sequence[int | float]],
    ) -> PolicyObservation:
        if not isinstance(readings, Mapping):
            raise ObservationBoundaryError("Projection accepts sensor readings, never evaluator objects")
        expected = {channel.name for channel in self.declaration.channels}
        if set(readings) != expected:
            raise ObservationBoundaryError("Readings must contain exactly the declared channel names")
        result = PolicyObservation(
            sequence=sequence, time_s=time_s,
            declaration_sha256=self.declaration_sha256,
            readings=tuple(
                SensorReading(channel=channel.name, values=tuple(readings[channel.name]))
                for channel in self.declaration.channels
            ),
        )
        return self.validate(result)

    def validate(self, packet: PolicyObservation) -> PolicyObservation:
        if type(packet) is not PolicyObservation:
            raise ObservationBoundaryError("Only declared-sensor PolicyObservation packets are accepted")
        _check_storage(packet)
        return self.from_json(packet.canonical_json())

    def from_json(self, value: str | bytes) -> PolicyObservation:
        if len(value if isinstance(value, bytes) else value.encode("utf-8")) > MAX_WIRE_BYTES:
            raise ObservationBoundaryError("Observation exceeds the bounded wire size")
        # Reject duplicate JSON keys as well as ordinary unknown model fields.
        parsed = _load_json(value)
        packet = PolicyObservation.model_validate_json(_wire_json(parsed))
        if packet.declaration_sha256 != self.declaration_sha256:
            raise ObservationBoundaryError("Observation declaration digest does not match")
        if tuple(reading.channel for reading in packet.readings) != tuple(
            channel.name for channel in self.declaration.channels
        ):
            raise ObservationBoundaryError("Reading inventory/order does not match the declaration")
        for channel, reading in zip(self.declaration.channels, packet.readings, strict=True):
            if len(reading.values) != math.prod(channel.shape):
                raise ObservationBoundaryError(f"Wrong reading size for {channel.name}")
            if channel.kind == "rgb" and any(
                type(value) is not int or not 0 <= value <= 255 for value in reading.values
            ):
                raise ObservationBoundaryError("RGB readings must be uint8 integers")
            if channel.kind in {"depth", "contact"} and any(value < 0 for value in reading.values):
                raise ObservationBoundaryError("Depth and contact readings must be nonnegative")
            if channel.unit == "binary" and any(
                type(value) is not int or value not in (0, 1) for value in reading.values
            ):
                raise ObservationBoundaryError("Binary contacts must be integer 0 or 1")
        return packet


def _load_json(value: str | bytes) -> object:
    def unique_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ObservationBoundaryError("Duplicate JSON object key")
            result[key] = item
        return result

    def reject_constant(value):
        raise ObservationBoundaryError("Nonfinite JSON values are forbidden")

    return json.loads(value, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def _wire_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)


class PolicyWorker:
    """Persistent clean subprocess running ``module:function(packet)``.

    The callable must return PolicyAction with exactly ``action_size`` values.
    Import paths are explicit; parent globals, environment secrets, and closures
    are not transferred. Only declared packets are sent after startup. Sequence
    numbers must increase, and simulation time cannot decrease. On a deadline,
    process exit, protocol violation, or invalid action the worker is closed.

    Use as a context manager. ``timeout_s`` applies separately to startup and
    each action. Policy code must not independently fetch privileged data; this
    API boundary does not claim OS isolation or prevent same-user file access.
    """

    def __init__(
        self, *, policy: str, declaration: SensorDeclaration, action_size: int,
        policy_paths: Sequence[str | Path] = (), timeout_s: float = 10.0,
    ):
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", policy):
            raise ValueError("Policy must be an importable module:function name")
        if type(action_size) is not int or not 1 <= action_size <= MAX_SENSOR_VALUES:
            raise ValueError("action_size must be a positive bounded integer")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.projector = ObservationProjector(declaration)
        self.policy = policy
        self.action_size = action_size
        self.timeout_s = float(timeout_s)
        self.policy_paths = tuple(str(Path(path).resolve(strict=True)) for path in policy_paths)
        if any(not Path(path).is_dir() for path in self.policy_paths):
            raise ValueError("Policy paths must be existing directories")
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._responses: queue.Queue = queue.Queue()
        self._closed = False
        self._last_sequence = -1
        self._last_time = -1.0

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        if self._closed or self._process is not None:
            raise PolicyWorkerError("A worker can be started only once")
        self._temporary = tempfile.TemporaryDirectory(prefix="rigby-policy-")
        source_root = str(Path(__file__).resolve().parents[1])
        paths = list(dict.fromkeys((source_root, *self.policy_paths)))
        bootstrap = (
            "import json,runpy,sys; sys.path[:0]=json.loads(sys.argv[1]); "
            "runpy.run_module('rigby_core.observation_worker',run_name='__main__')"
        )
        # Explicitly preserve only Windows runtime / temporary-directory needs.
        # -I additionally ignores PYTHONPATH, PYTHONSTARTUP and user site paths.
        environment = {
            key: value for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP"}
        }
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-u", "-c", bootstrap, _wire_json(paths)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", cwd=self._temporary.name, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._reader = threading.Thread(target=self._read_responses, daemon=True)
            self._reader.start()
            deadline = time.monotonic() + self.timeout_s
            self._send({
                "operation": "initialize", "policy": self.policy,
                "declaration": json.loads(self.projector.declaration.canonical_json()),
                "action_size": self.action_size,
            }, deadline=deadline)
            response = self._receive(deadline=deadline)
            if response != {"status": "ready", "declaration_sha256": self.projector.declaration_sha256}:
                raise PolicyWorkerError(f"Worker did not initialize: {response}")
            return self
        except BaseException:
            self.close()
            raise

    def _read_responses(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(MAX_WIRE_BYTES + 1)
                if not line:
                    self._responses.put(None)
                    return
                if len(line.encode("utf-8")) > MAX_WIRE_BYTES or not line.endswith("\n"):
                    self._responses.put(PolicyWorkerError("Worker response exceeds wire limit"))
                    return
                self._responses.put(line)
        except (OSError, ValueError) as error:
            self._responses.put(PolicyWorkerError(f"Worker output closed: {error}"))

    def _send(self, value: object, *, deadline: float) -> None:
        assert self._process is not None and self._process.stdin is not None
        message = _wire_json(value) + "\n"
        if len(message.encode("utf-8")) > MAX_WIRE_BYTES:
            raise ObservationBoundaryError("Worker request exceeds the bounded wire size")
        # A dead or unresponsive reader can block a large pipe write. Bound the
        # write as well as the response wait; close() terminates the child and
        # joins this writer before releasing its streams.
        completed: queue.Queue = queue.Queue(maxsize=1)

        def write_message():
            try:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(message)
                self._process.stdin.flush()
                completed.put(None)
            except (OSError, ValueError) as error:
                completed.put(error)

        self._writer = threading.Thread(target=write_message, daemon=True)
        self._writer.start()
        try:
            failure = completed.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as error:
            raise PolicyWorkerTimeout("Policy input deadline exceeded; worker stopped") from error
        if failure is not None:
            raise PolicyWorkerError("Worker closed its input") from failure
        self._writer.join()

    def _receive(self, *, deadline: float) -> dict:
        try:
            item = self._responses.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as error:
            raise PolicyWorkerTimeout("Policy deadline exceeded; worker stopped") from error
        if item is None:
            raise PolicyWorkerError("Policy worker exited without a response")
        if isinstance(item, Exception):
            raise item
        try:
            value = _load_json(item)
        except (ValueError, TypeError) as error:
            raise PolicyWorkerError("Policy worker returned malformed JSON") from error
        if not isinstance(value, dict):
            raise PolicyWorkerError("Policy worker returned a non-object response")
        return value

    def act(self, observation: PolicyObservation) -> PolicyAction:
        if self._closed or self._process is None:
            raise PolicyWorkerError("Policy worker is not running")
        packet = self.projector.validate(observation)
        if packet.sequence <= self._last_sequence or packet.time_s < self._last_time:
            raise ObservationBoundaryError("Observation sequence/time is stale or moves backwards")
        try:
            deadline = time.monotonic() + self.timeout_s
            self._send({"operation": "act", "observation": json.loads(packet.canonical_json())}, deadline=deadline)
            response = self._receive(deadline=deadline)
            if set(response) != {"status", "sequence", "action"} or response["status"] != "action":
                raise PolicyWorkerError(f"Policy failed: {response}")
            if type(response["sequence"]) is not int or response["sequence"] != packet.sequence:
                raise PolicyWorkerError("Worker returned an action for the wrong observation")
            action = PolicyAction.model_validate_json(_wire_json(response["action"]))
            if len(action.values) != self.action_size:
                raise PolicyWorkerError("Policy action has the wrong size")
            self._last_sequence, self._last_time = packet.sequence, packet.time_s
            return action
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                # Do not write a potentially blocked pipe while recovering from
                # a hung policy. Termination is deterministic and always reaped.
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            else:
                process.wait()
            if self._writer is not None:
                self._writer.join(timeout=2.0)
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            if self._reader is not None:
                self._reader.join(timeout=2.0)
            if process.stdout is not None:
                process.stdout.close()
        if self._temporary is not None:
            self._temporary.cleanup()

    def __exit__(self, *_exception) -> None:
        self.close()
