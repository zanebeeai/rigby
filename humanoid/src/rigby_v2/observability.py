"""Small deterministic JSON logging boundary for durable v2 services."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from collections.abc import Mapping
from typing import Any


_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__)
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "query",
    "body",
    "prompt",
    "password",
    "secret",
    "credential",
    "database_url",
    "db_url",
    "api_key",
    "access_token",
    "refresh_token",
)
_VALUE_REDACTIONS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(postgres(?:ql)?://[^:/\s]+:)[^@\s]+@"),
    re.compile(r"(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)[^\s,;]+"),
)


def _sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_value(value: Any, *, key: object | None = None) -> Any:
    if key is not None and _sensitive_key(key):
        return _REDACTED
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_value(child, key=child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _VALUE_REDACTIONS:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        return redacted
    return value


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = _redact_value(record.getMessage())
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": _redact_value(getattr(record, "event", record.getMessage())),
            "message": message,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in {"event", "message"}:
                payload[key] = _redact_value(value, key=key)
        if record.exc_info:
            # Exception messages frequently contain rejected request values or
            # connection strings.  The type is sufficient for aggregation.
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def configure_json_logging(*, level: int = logging.INFO) -> None:
    root = logging.getLogger("rigby_v2")
    handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, "_rigby_v2_json_handler", False)
    ]
    if handlers:
        handler = handlers[0]
        for duplicate in handlers[1:]:
            root.removeHandler(duplicate)
    else:
        handler = logging.StreamHandler()
        handler._rigby_v2_json_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(JsonLogFormatter())
    handler.setLevel(level)
    root.setLevel(level)
    root.propagate = False


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    logger.log(level, event, extra={"event": event, **fields})
