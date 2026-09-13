"""Private JSON worker entry point; launched only through PolicyWorker."""

from __future__ import annotations

import contextlib
import importlib
import json
import sys

from .observations import (
    MAX_WIRE_BYTES, ObservationProjector, PolicyAction, SensorDeclaration,
    _check_storage, _load_json, _wire_json,
)


def _read() -> dict | None:
    line = sys.stdin.readline(MAX_WIRE_BYTES + 1)
    if not line:
        return None
    if not line.endswith("\n") or len(line.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("Worker request exceeds wire limit")
    value = _load_json(line)
    if not isinstance(value, dict):
        raise ValueError("Worker request must be an object")
    return value


def _reply(value: dict) -> None:
    line = _wire_json(value) + "\n"
    if len(line.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("Worker reply exceeds wire limit")
    sys.stdout.write(line)
    sys.stdout.flush()


def main() -> int:
    try:
        config = _read()
        if config is None or set(config) != {"operation", "policy", "declaration", "action_size"}:
            raise ValueError("Expected exact worker initialization fields")
        if config["operation"] != "initialize":
            raise ValueError("Worker must be initialized first")
        declaration = SensorDeclaration.model_validate_json(_wire_json(config["declaration"]))
        projector = ObservationProjector(declaration)
        action_size = config["action_size"]
        if type(action_size) is not int or action_size < 1:
            raise ValueError("Invalid action size")
        module_name, function_name = config["policy"].split(":")
        # A trusted policy's ordinary logging must not become a protocol packet.
        with contextlib.redirect_stdout(sys.stderr):
            policy = getattr(importlib.import_module(module_name), function_name)
        if not callable(policy):
            raise ValueError("Policy target is not callable")
        _reply({"status": "ready", "declaration_sha256": projector.declaration_sha256})
        last_sequence, last_time = -1, -1.0
        while (request := _read()) is not None:
            if set(request) != {"operation", "observation"} or request["operation"] != "act":
                raise ValueError("Expected exact acting request fields")
            packet = projector.from_json(_wire_json(request["observation"]))
            if packet.sequence <= last_sequence or packet.time_s < last_time:
                raise ValueError("Observation sequence/time moved backwards")
            with contextlib.redirect_stdout(sys.stderr):
                action = policy(packet)
            if type(action) is not PolicyAction:
                raise ValueError("Policy must return PolicyAction")
            _check_storage(action)
            action = PolicyAction.model_validate_json(action.canonical_json())
            if len(action.values) != action_size:
                raise ValueError("Policy action has the wrong size")
            _reply({
                "status": "action", "sequence": packet.sequence,
                "action": json.loads(action.canonical_json()),
            })
            last_sequence, last_time = packet.sequence, packet.time_s
        return 0
    except Exception as error:
        _reply({"status": "error", "error_type": type(error).__name__, "message": str(error)[:500]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
