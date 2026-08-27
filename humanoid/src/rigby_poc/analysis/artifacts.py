"""Load a stored result directory into the inputs :func:`analyze` expects.

``ResultStore.persist`` writes ``program.json`` and ``scene.json`` from
``request.program`` / ``request.scene`` — the **pre-override** pair. The clip on
disk was compiled from the post-override pair. Reading ``program.json``
directly therefore analyses a program the clip was never compiled from, which
is silently wrong rather than loudly wrong. This module reads ``request.json``
and applies the overrides, so callers get the effective pair.

``compiler.apply_overrides`` is imported inside the function rather than at
module scope: ``compiler`` imports this package for the moved helpers, so a
module-level import here would close an import cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import ClipResult, CompileRequest, MotionProgram, SceneManifest


def load_request(result_dir: Path) -> CompileRequest:
    return CompileRequest.model_validate_json(
        (result_dir / "request.json").read_text(encoding="utf-8")
    )


def load_clip(result_dir: Path) -> ClipResult:
    return ClipResult.model_validate_json(
        (result_dir / "clip.json").read_text(encoding="utf-8")
    )


def load_persisted_metrics(result_dir: Path) -> dict:
    return json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))


def effective_program(request: CompileRequest) -> tuple[SceneManifest, MotionProgram]:
    """Apply parameter overrides, yielding the pair the clip was compiled from."""

    from ..compiler import apply_overrides

    return apply_overrides(request)


def load_analysis_inputs(
    result_dir: Path,
) -> tuple[ClipResult, MotionProgram, SceneManifest]:
    """Return ``(clip, program, scene)`` ready to hand to :func:`analyze`."""

    scene, program = effective_program(load_request(result_dir))
    return load_clip(result_dir), program, scene
