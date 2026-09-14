"""Canonical digests for corpus cases.

The motion digest deliberately uses the same canonical key set and byte encoding
as :func:`evals.review.motion_sha256`, so a corpus hash and a run-archive hash of
the same motion are the same string.  ``tests/test_corpus_format.py`` pins that
agreement.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
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


#: Fallback when no CPU model can be read. A key that says ``unknown-cpu`` is
#: honest about what it does not know; silently dropping the field would put two
#: machines back on one key, which is the defect this names.
UNKNOWN_CPU = "unknown-cpu"


def _slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "-" for character in value.lower()
    )
    return "-".join(part for part in cleaned.split("-") if part) or UNKNOWN_CPU


@lru_cache(maxsize=1)
def cpu_model() -> str:
    """A slug for this machine's CPU model, read once.

    Read from the OS rather than from ``platform.processor()``, which returns
    ``'arm'`` on macOS and the empty string on many Linux builds -- neither
    distinguishes two machines, which is the whole point here.
    """

    try:
        if sys.platform == "darwin":
            output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            if output:
                return _slug(output)
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return _slug(line.split(":", 1)[1])
        elif sys.platform == "win32":
            output = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if output:
                return _slug(output)
    except (OSError, ValueError, subprocess.SubprocessError):
        # An unreadable CPU is a fallback, not a failure: the key degrades to one
        # that names less rather than refusing to produce a key at all.
        return _fallback_cpu()
    return _fallback_cpu()


def _fallback_cpu() -> str:
    processor = platform.processor()
    return _slug(processor) if processor else UNKNOWN_CPU


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

    **The key names the CPU, because ``<platform>-<machine>`` named two machines.**
    Measured 2026-09-13: checking out ``8e777f8``, whose own commit message records
    "verify 47 matched", and running ``evals.corpus verify`` on a second
    ``darwin-arm64`` machine gives **1 matched, 46 moved** -- same commit, with
    ``uv.lock`` and every ``pyproject.toml`` untouched since, the pinned versions
    installed, BLAS thread count making no difference to a digest, and two runs in
    separate processes agreeing with each other.  43 of those 46 are identical to
    the committed clip at slim precision; the difference is in the last bits.

    Two machines cannot share a key and both be right.  Whichever blessed last made
    the other red, and a reader could not tell that from a real drift -- the same
    defect described above between architectures, one level finer.  The CPU model
    is the machine-identifying attribute most directly tied to the causes named
    there: numpy's SIMD kernel dispatch and FMA contraction.  It is not *proven* to
    be the attribute that differs, which would need the other machine; it is the
    narrowest field that separates them.  If a mismatch survives it, the next
    candidates are the OS version (libm) and the numpy build, and they belong here
    for the same reason.

    Naming the CPU also makes blessing additive rather than territorial: a second
    developer's ``bless --write`` writes a new column beside the first's instead of
    overwriting it.  See ``docs/evidence/corpus-red-2026-09-13.md``.
    """
    machine = platform.machine().lower() or "unknown"
    base = f"{sys.platform}-{machine}-{cpu_model()}"
    return f"{base}|mujoco-{version('mujoco')}" if solver_used else base
