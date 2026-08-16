"""Load corpus cases and recompile them.

Nothing in this module touches the network, starts a server, needs an API key, or
imports the planner.  ``tests/test_corpus_loads_offline.py`` enforces that.
"""

from __future__ import annotations

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
    PORTABLE_PLATFORM_KEY,
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


class CorpusError(RuntimeError):
    """A corpus directory is missing, malformed, or inconsistent with its manifest."""


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
    determinism_class: DeterminismClass,
    *,
    now: str | None = None,
) -> ExpectedResult:
    """Build the ``expected.json`` a freshly compiled clip would be blessed to."""
    key = (
        PORTABLE_PLATFORM_KEY
        if determinism_class == DeterminismClass.PORTABLE
        else platform_key()
    )
    return ExpectedResult(
        case_id=case_id,
        determinism_class=determinism_class,
        motion_sha256={key: motion_sha256(clip)},
        metrics_sha256=metrics_sha256(clip),
        observables_sha256=observables_sha256(clip),
        success=clip.success,
        structural_valid=bool(clip.metrics.get("structural_valid", False)),
        fps=clip.fps,
        frame_count=len(clip.frames),
        duration_s=clip.duration_s,
        contact_count=len(clip.contacts),
        environment=BlessEnvironment(
            platform_key=platform_key(),
            python_version=_python_version(),
            compiler_version=COMPILER_VERSION,
            blessed_at=now or datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
