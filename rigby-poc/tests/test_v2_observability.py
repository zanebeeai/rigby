from __future__ import annotations

import io
import json
import logging

from rigby_v2.observability import JsonLogFormatter, log_event


def test_structured_log_event_is_machine_readable_and_contextual() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("rigby_v2.test.observability")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    log_event(
        logger,
        "candidate_certified",
        job_id="job-1",
        candidate_id="candidate-2",
        trace_sha256="a" * 64,
    )

    payload = json.loads(stream.getvalue())
    assert payload["event"] == "candidate_certified"
    assert payload["job_id"] == "job-1"
    assert payload["candidate_id"] == "candidate-2"
    assert payload["trace_sha256"] == "a" * 64
    assert payload["timestamp"].endswith("+00:00")
