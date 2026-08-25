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


def platform_key(solver_used: bool = True) -> str:
    """Identify the platform a hash was blessed on.

    **Not just a MuJoCo concern.**  03a assumed the physics solver was the only
    source of cross-platform nondeterminism and classified everything else
    ``portable``.  Windows CI disproved that: pure-Python/numpy compilation differs
    between arm64 and x86-64 in the last few ulps, because the two differ in libm
    transcendental implementations (``sin``/``cos``/``acos``) and in FMA
    contraction, and numpy dispatches to different SIMD kernels.  Measured on
    ``max_angular_jerk_rad_s3``: 705.8566226509565 on darwin-arm64 against
    705.8566226509532 on win32-amd64, 29 ulps -- amplified there because jerk is a
    third derivative and divides by ``dt`` three times.  Bit-identical floating
    point across instruction sets is not achievable without pinning the math
    library.  See plan 03 section 6.1.

    So every case is keyed by platform.  ``solver_used`` decides whether the MuJoCo
    version is part of the key: a case that never enters the solver must not have
    its hashes invalidated by an unrelated ``mujoco`` upgrade, which would turn
    re-blessing into a routine event and make a real drift indistinguishable from
    a dependency bump at review time.
    """
    machine = platform.machine().lower() or "unknown"
    base = f"{sys.platform}-{machine}"
    return f"{base}|mujoco-{version('mujoco')}" if solver_used else base
