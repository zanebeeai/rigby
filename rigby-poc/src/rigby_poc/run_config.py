"""Plan 01 section 4.4 — the effective configuration a run actually used.

Model ids, reasoning effort, image handling, call budgets and prompt identities are read
from the environment at construction and only partially echoed into `routing`, so no
single record states what a run was configured with. Without one, a calibration result
cannot be attributed to a configuration, and prompt drift across git revisions is
invisible in the artifacts (section 1.1).

Two rules this module exists to keep:

- **Never dump `os.environ`.** Section 8.4: request payloads must never contain the API
  key, and a configuration snapshot is the obvious place for one to arrive by accident.
  The allowlist below is explicit, and a test asserts no value in the document looks like
  a credential.
- **Prompt hashes come from the prompt module, not from re-hashing text here.** Lane
  `judge` owns `GraderPrompt.record()`; duplicating the hashing would give two sources
  that can disagree, and the disagreement would be silent.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Any

from .compiler import PROJECT_ROOT

# Read for their *values*, which are configuration. Anything holding a credential is
# absent by construction rather than filtered afterwards.
ENVIRONMENT_ALLOWLIST = (
    "OPENAI_JUDGE_MODEL",
    "OPENAI_JUDGE_FALLBACK_MODEL",
    "OPENAI_JUDGE_REASONING_EFFORT",
    "OPENAI_JUDGE_IMAGE_DETAIL",
    "OPENAI_JUDGE_MAX_IMAGE_DIMENSION_PX",
    "OPENAI_JUDGE_REPAIR_MODEL",
    "OPENAI_PLANNER_MODEL",
    "OPENAI_REPAIR_MODEL",
    "RIGBY_GRADER_MODE",
)

# Substrings that must never appear in an allowlisted *name*. Belt and braces: the
# allowlist already excludes them, and this makes adding one to it fail loudly.
_CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _environment_snapshot() -> dict[str, str]:
    for name in ENVIRONMENT_ALLOWLIST:
        if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS):
            raise ValueError(f"{name} looks like a credential and must not be snapshotted")
    return {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}


def _system_prompts() -> dict[str, dict[str, Any]]:
    """The four module-constant prompts, hashed. Section 1.1: never recorded before."""
    from . import judge

    prompts = {
        "unary": judge.UNARY_SYSTEM_PROMPT,
        "pairwise": judge.PAIRWISE_SYSTEM_PROMPT,
        "repair": judge.REPAIR_SYSTEM_PROMPT,
        "five_way": judge.FIVE_WAY_SYSTEM_PROMPT,
    }
    return {
        name: {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "characters": len(text),
        }
        for name, text in prompts.items()
    }


def _grader_prompts() -> list[dict[str, Any]]:
    """Prompt identity for the split graders, from lane `judge`'s own record()."""
    try:
        from .judge_prompts import GRADER_SPECS, grader_prompt
    except ImportError:  # pragma: no cover - the split graders are not always present
        return []
    records: list[dict[str, Any]] = []
    for name in sorted(GRADER_SPECS):
        try:
            records.append(dict(grader_prompt(name).record()))
        except Exception as error:  # noqa: BLE001 - a prompt that will not assemble is a fact
            records.append({"grader": name, "error": f"{type(error).__name__}: {error}"})
    return records


def effective_configuration(
    *,
    run_id: str,
    prompt: str | None = None,
    provider: str | None = None,
    selection_mode: str | None = None,
    max_rounds: int | None = None,
    max_model_calls: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole configuration of one run, in one document."""
    from . import judge

    document: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "request": {
            "prompt": prompt,
            "provider": provider,
            "selection_mode": selection_mode,
            "max_rounds": max_rounds,
            "max_model_calls": max_model_calls,
        },
        "defaults": {
            "judge_model": judge.DEFAULT_JUDGE_MODEL,
            "judge_fallback_model": judge.DEFAULT_JUDGE_FALLBACK_MODEL,
            "judge_reasoning_effort": judge.DEFAULT_JUDGE_REASONING_EFFORT,
            "judge_image_detail": judge.DEFAULT_JUDGE_IMAGE_DETAIL,
            "judge_max_image_dimension_px": judge.DEFAULT_JUDGE_MAX_IMAGE_DIMENSION_PX,
        },
        "environment": _environment_snapshot(),
        "system_prompts": _system_prompts(),
        "grader_prompts": _grader_prompts(),
        "code": {
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "machine": platform.machine(),
        },
    }
    if extra:
        document["extra"] = extra
    return document
