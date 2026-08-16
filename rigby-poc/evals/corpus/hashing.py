"""Canonical digests for corpus cases.

The motion digest deliberately uses the same canonical key set and byte encoding
as :func:`evals.review.motion_sha256`, so a corpus hash and a run-archive hash of
the same motion are the same string.  ``tests/test_corpus_format.py`` pins that
agreement.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from typing import Any

from rigby_poc.models import ClipResult

#: Keys of ``clip.json`` that define the motion, in the sense that changing any of
#: them changes what a viewer sees.  Provenance and metrics are excluded on purpose:
#: they move for reasons that are not motion changes.
CANONICAL_MOTION_KEYS = ("schema_version", "fps", "duration_s", "frames", "contacts")

#: Reserved ``expected.motion_sha256`` key for motion that is bit-identical on every
#: platform.  Platform-dependent cases use :func:`platform_key` instead.
PORTABLE_PLATFORM_KEY = "any"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def motion_sha256(clip: ClipResult) -> str:
    payload = clip.model_dump(mode="json")
    return sha256_value({key: payload[key] for key in CANONICAL_MOTION_KEYS})


def metrics_sha256(clip: ClipResult) -> str:
    return sha256_value(clip.model_dump(mode="json")["metrics"])


def observables_sha256(clip: ClipResult) -> str:
    payload = clip.model_dump(mode="json")
    return sha256_value(
        {
            "slider": payload["slider_observables"],
            "parametric": payload["parametric_observables"],
        }
    )


def platform_key() -> str:
    """Identify the (platform, architecture, MuJoCo version) this process runs on.

    MuJoCo guarantees determinism per platform and version, not across them, so any
    case whose motion passes through the physics solver has to record one hash per
    key of this shape.  03a contains no such case; the format carries the key so
    03b can add them without a schema change.
    """
    machine = platform.machine().lower() or "unknown"
    return f"{sys.platform}-{machine}|mujoco-{version('mujoco')}"
