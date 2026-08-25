"""Load corpus cases and recompile them.

Nothing in this module touches the network, starts a server, needs an API key, or
imports the planner.  ``tests/test_corpus_loads_offline.py`` enforces that.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rigby_poc.compiler import COMPILER_VERSION, compile_motion
from rigby_poc.models import (
    ClipResult,
    CompileRequest,
    MotionProgram,
    ParameterOverrides,
    SceneManifest,
)

from .hashing import (
    CANONICAL_MOTION_KEYS,
    PORTABLE_PLATFORM_KEY,
    canonical_bytes,
    metrics_sha256,
    motion_sha256,
    observables_sha256,
    platform_key,
)
from .models import (
    BlessEnvironment,
    CaseEntry,
    CorpusManifest,
    DeterminismClass,
    ExpectedResult,
    StoragePolicy,
)

CORPUS_ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "manifest.json"
CASES_DIRNAME = "cases"

SCENE_FILE = "scene.json"
PROGRAM_FILE = "program.json"
OVERRIDES_FILE = "overrides.json"
EXPECTED_FILE = "expected.json"
#: Reserved for 03b.  A case directory may carry one; the loader ignores it and the
#: determinism test still recompiles, so a stored clip can never become a second
#: source of truth.
SLIM_CLIP_FILE = "clip.slim.json.gz"

#: Decimal places quaternion and position components are rounded to before the slim
#: clip is written.  Plan 03 section 1.3 measured 1e-6 at 17 KB gzipped per case,
#: which is what this buys; the *hash* is always taken from a recompile, never from
#: the stored clip, so this rounding can never become the corpus's source of truth.
SLIM_CLIP_PRECISION = 6


class CorpusError(RuntimeError):
    """A corpus directory is missing, malformed, or inconsistent with its manifest."""


def _round_floats(value: Any, places: int) -> Any:
    """Round every float in a JSON-shaped structure, leaving ints alone."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [_round_floats(item, places) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item, places) for key, item in value.items()}
    return value


def slim_clip_payload(clip: ClipResult, places: int = SLIM_CLIP_PRECISION) -> dict[str, Any]:
    """The motion-defining half of a clip, rounded, ready to gzip.

    Only :data:`~evals.corpus.hashing.CANONICAL_MOTION_KEYS` are kept: metrics are
    recomputed by whoever reads this, and provenance is not motion.
    """
    payload = clip.model_dump(mode="json")
    return {key: _round_floats(payload[key], places) for key in CANONICAL_MOTION_KEYS}


def write_slim_clip(path: Path, clip: ClipResult) -> None:
    """Write ``clip.slim.json.gz``.

    Committed for MuJoCo cases only.  On a platform that has never blessed such a
    case there is no hash to assert against, and plan 03 section 6.1 leaves that a
    skip -- which means zero coverage there.  This file is what
    ``tests/test_corpus_determinism.py`` falls back to instead, comparing frames
    within a tolerance rather than skipping outright.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(slim_clip_payload(clip))
    # mtime=0 so the gzip container is byte-identical run to run; the corpus is
    # committed and a header timestamp would show up as a diff on every re-bless.
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as handle:
        handle.write(body)


def read_slim_clip(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CorpusError(f"missing slim clip: {path}")
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise CorpusError(f"missing corpus file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write pretty, sorted, newline-terminated JSON so diffs stay reviewable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class CorpusCase:
    """One loaded case: its manifest row and its three compile inputs."""

    entry: CaseEntry
    root: Path
    scene: SceneManifest
    program: MotionProgram
    overrides: ParameterOverrides
    expected: ExpectedResult

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def slim_clip_path(self) -> Path:
        return self.root / SLIM_CLIP_FILE

    def slim_clip(self) -> dict[str, Any] | None:
        """The committed clip, or ``None`` when this case stores programs only."""
        if self.entry.storage != StoragePolicy.PROGRAM_AND_CLIP:
            return None
        return read_slim_clip(self.slim_clip_path)

    def compile_request(self) -> CompileRequest:
        return CompileRequest(
            scene=self.scene,
            program=self.program,
            parameter_overrides=self.overrides,
            persist=False,
        )


def manifest_path(root: Path | None = None) -> Path:
    return (root or CORPUS_ROOT) / MANIFEST_NAME


def cases_root(root: Path | None = None) -> Path:
    return (root or CORPUS_ROOT) / CASES_DIRNAME


def load_manifest(root: Path | None = None) -> CorpusManifest:
    return CorpusManifest.model_validate(_read_json(manifest_path(root)))


def load_case_from_entry(entry: CaseEntry, root: Path | None = None) -> CorpusCase:
    directory = cases_root(root) / entry.id
    if not directory.is_dir():
        raise CorpusError(f"case {entry.id!r} has no directory at {directory}")
    expected = ExpectedResult.model_validate(_read_json(directory / EXPECTED_FILE))
    if expected.case_id != entry.id:
        raise CorpusError(
            f"case {entry.id!r} holds an expected.json for {expected.case_id!r}"
        )
    overrides_path = directory / OVERRIDES_FILE
    overrides = (
        ParameterOverrides.model_validate(_read_json(overrides_path))
        if overrides_path.is_file()
        else ParameterOverrides()
    )
    program = MotionProgram.model_validate(_read_json(directory / PROGRAM_FILE))
    if program.intent != entry.intent:
        raise CorpusError(
            f"case {entry.id!r} declares intent {entry.intent} but its program is "
            f"{program.intent}"
        )
    return CorpusCase(
        entry=entry,
        root=directory,
        scene=SceneManifest.model_validate(_read_json(directory / SCENE_FILE)),
        program=program,
        overrides=overrides,
        expected=expected,
    )


def load_case(case_id: str, root: Path | None = None) -> CorpusCase:
    manifest = load_manifest(root)
    entry = manifest.entry(case_id)
    if entry is None:
        raise CorpusError(f"unknown corpus case: {case_id!r}")
    return load_case_from_entry(entry, root)


def load_corpus(root: Path | None = None) -> list[CorpusCase]:
    manifest = load_manifest(root)
    cases = [load_case_from_entry(entry, root) for entry in manifest.cases]
    listed = {entry.id for entry in manifest.cases}
    directory = cases_root(root)
    if directory.is_dir():
        found = {item.name for item in directory.iterdir() if item.is_dir()}
        orphans = sorted(found - listed)
        if orphans:
            raise CorpusError(f"case directories missing from the manifest: {orphans}")
    return cases


def compile_inputs(
    scene: SceneManifest,
    program: MotionProgram,
    overrides: ParameterOverrides | None = None,
) -> ClipResult:
    return compile_motion(
        CompileRequest(
            scene=scene,
            program=program,
            parameter_overrides=overrides or ParameterOverrides(),
            persist=False,
        )
    )


def compile_case(case: CorpusCase) -> ClipResult:
    return compile_motion(case.compile_request())


def observe(
    case_id: str,
    clip: ClipResult,
    determinism_class: DeterminismClass = DeterminismClass.PLATFORM_DEPENDENT,
    *,
    solver_used: bool = False,
    now: str | None = None,
) -> ExpectedResult:
    """Build the ``expected.json`` a freshly compiled clip would be blessed to."""
    key = (
        PORTABLE_PLATFORM_KEY
        if determinism_class == DeterminismClass.PORTABLE
        else platform_key(solver_used)
    )
    return ExpectedResult(
        case_id=case_id,
        determinism_class=determinism_class,
        solver_used=solver_used,
        motion_sha256={key: motion_sha256(clip)},
        metrics_sha256={key: metrics_sha256(clip)},
        observables_sha256={key: observables_sha256(clip)},
        success=clip.success,
        structural_valid=bool(clip.metrics.get("structural_valid", False)),
        fps=clip.fps,
        frame_count=len(clip.frames),
        duration_s=clip.duration_s,
        contact_count=len(clip.contacts),
        environment=BlessEnvironment(
            platform_key=platform_key(solver_used),
            python_version=_python_version(),
            compiler_version=COMPILER_VERSION,
            blessed_at=now or datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
