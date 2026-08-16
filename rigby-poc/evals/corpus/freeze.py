"""Turn a live result -- or an offline prompt -- into a corpus case directory.

Freezing does the three things plan 03 section 3.5 asks for: it strips volatile
provenance, it pins the seed to a literal so the case never depends on an LLM
response id, and it emits a case directory with its expectations recorded.
"""

from __future__ import annotations

from pathlib import Path

from rigby_poc.compiler import COMPILER_VERSION
from rigby_poc.models import (
    BodyAction,
    CompileRequest,
    Intent,
    MotionProgram,
    ParameterOverrides,
    SceneManifest,
)

from .loader import (
    EXPECTED_FILE,
    OVERRIDES_FILE,
    PROGRAM_FILE,
    SCENE_FILE,
    CorpusCase,
    CorpusError,
    cases_root,
    compile_inputs,
    load_manifest,
    manifest_path,
    observe,
    write_json,
)
from .models import (
    CaseEntry,
    CorpusManifest,
    Coverage,
    CoverageAxis,
    DeterminismClass,
    Family,
    StoragePolicy,
)

#: Corpus programs carry this seed rather than one derived from a response id.
#: The compiler never consumes ``program.seed`` -- motion is bit-identical across
#: seeds, verified in ``tests/test_corpus_determinism.py`` -- so the literal is
#: provenance, not a control.
PINNED_SEED = 0


def pin_seed(program: MotionProgram, seed: int = PINNED_SEED) -> MotionProgram:
    """Replace the program's seed, and every sequence step's seed, with a literal."""
    return program.model_copy(
        update={
            "seed": seed,
            "steps": [pin_seed(step, seed) for step in program.steps],
        }
    )


def body_actions_of(program: MotionProgram) -> list[BodyAction]:
    actions: set[BodyAction] = {
        primitive.body.action
        for primitive in program.primitives
        if primitive.body is not None
    }
    for step in program.steps:
        actions.update(body_actions_of(step))
    return sorted(actions, key=lambda action: action.value)


def plan_offline(prompt: str, scene: SceneManifest) -> MotionProgram:
    """Plan a program with the rule planner.

    Imported lazily on purpose: the loader must never pull the planner -- and with
    it the OpenAI client -- into an offline corpus run.
    """
    from rigby_poc.models import PlanRequest
    from rigby_poc.planner import plan_motion

    return plan_motion(PlanRequest(text=prompt, scene=scene, provider="offline")).program


def load_live_result(
    result_root: Path,
) -> tuple[SceneManifest, MotionProgram, ParameterOverrides]:
    """Read the compile inputs out of a ``results/<id>/`` directory.

    ``request.json`` is preferred because ``store.py`` persists the *pre-override*
    program in ``program.json``; taking that program without its overrides would
    freeze a case that compiles to different motion than the run it came from.
    """
    request_path = result_root / "request.json"
    if request_path.is_file():
        request = CompileRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        return request.scene, request.program, request.parameter_overrides
    scene_path = result_root / SCENE_FILE
    program_path = result_root / PROGRAM_FILE
    if not (scene_path.is_file() and program_path.is_file()):
        raise CorpusError(f"{result_root} is not a result directory")
    return (
        SceneManifest.model_validate_json(scene_path.read_text(encoding="utf-8")),
        MotionProgram.model_validate_json(program_path.read_text(encoding="utf-8")),
        ParameterOverrides(),
    )


def rebuild_coverage(manifest: CorpusManifest) -> Coverage:
    """Derive the covered lists from the cases; leave the deferred reasons alone.

    A member that is now covered drops out of ``deferred`` automatically, so the two
    lists cannot contradict each other.  Nothing is ever *added* to ``deferred``:
    that is a human declaration, and an undeclared member is the failure signal the
    coverage test looks for.
    """
    intents = sorted({case.intent.value for case in manifest.cases})
    actions = sorted(
        {action.value for case in manifest.cases for action in case.body_actions}
    )
    return Coverage(
        intent=CoverageAxis(
            covered=intents,
            deferred={
                name: reason
                for name, reason in manifest.coverage.intent.deferred.items()
                if name not in intents
            },
        ),
        body_action=CoverageAxis(
            covered=actions,
            deferred={
                name: reason
                for name, reason in manifest.coverage.body_action.deferred.items()
                if name not in actions
            },
        ),
    )


def upsert_entry(manifest: CorpusManifest, entry: CaseEntry) -> CorpusManifest:
    cases = [case for case in manifest.cases if case.id != entry.id] + [entry]
    updated = manifest.model_copy(
        update={
            "cases": sorted(cases, key=lambda case: case.id),
            "compiler_version": COMPILER_VERSION,
        }
    )
    return updated.model_copy(update={"coverage": rebuild_coverage(updated)})


def load_or_seed_manifest(root: Path | None = None) -> CorpusManifest:
    if manifest_path(root).is_file():
        return load_manifest(root)
    return CorpusManifest(compiler_version=COMPILER_VERSION)


def freeze_case(
    *,
    case_id: str,
    family: Family,
    scene: SceneManifest,
    program: MotionProgram,
    overrides: ParameterOverrides | None = None,
    source_prompt: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    determinism_class: DeterminismClass = DeterminismClass.PORTABLE,
    root: Path | None = None,
    seed: int = PINNED_SEED,
) -> CorpusCase:
    """Write ``cases/<case_id>/`` and register it in the manifest."""
    overrides = overrides or ParameterOverrides()
    source_seed = program.seed if program.seed != seed else None
    pinned = pin_seed(program, seed)

    clip = compile_inputs(scene, pinned, overrides)
    expected = observe(case_id, clip, determinism_class)
    entry = CaseEntry(
        id=case_id,
        family=family,
        intent=pinned.intent,
        body_actions=body_actions_of(pinned),
        tags=sorted(tags or []),
        source_prompt=source_prompt or pinned.source_text,
        source_seed=source_seed,
        expected_structural_valid=expected.structural_valid,
        storage=StoragePolicy.PROGRAMS_ONLY,
        notes=notes,
    )

    directory = cases_root(root) / case_id
    write_json(directory / SCENE_FILE, scene.model_dump(mode="json"))
    write_json(directory / PROGRAM_FILE, pinned.model_dump(mode="json"))
    overrides_path = directory / OVERRIDES_FILE
    if any(value is not None for value in overrides.model_dump().values()):
        write_json(overrides_path, overrides.model_dump(mode="json"))
    elif overrides_path.is_file():
        overrides_path.unlink()
    write_json(directory / EXPECTED_FILE, expected.model_dump(mode="json"))

    manifest = upsert_entry(load_or_seed_manifest(root), entry)
    write_json(manifest_path(root), manifest.model_dump(mode="json"))

    return CorpusCase(
        entry=entry,
        root=directory,
        scene=scene,
        program=pinned,
        overrides=overrides,
        expected=expected,
    )


def freeze_from_prompt(
    prompt: str,
    *,
    case_id: str,
    family: Family,
    scene: SceneManifest | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    determinism_class: DeterminismClass = DeterminismClass.PORTABLE,
    root: Path | None = None,
    seed: int = PINNED_SEED,
) -> CorpusCase:
    from rigby_poc.models import default_scene

    resolved_scene = scene or default_scene()
    program = plan_offline(prompt, resolved_scene)
    if program.intent == Intent.UNSUPPORTED:
        raise CorpusError(
            f"the offline planner rejected {prompt!r}: {program.unsupported_reason}"
        )
    return freeze_case(
        case_id=case_id,
        family=family,
        scene=resolved_scene,
        program=program,
        source_prompt=prompt,
        tags=tags,
        notes=notes,
        determinism_class=determinism_class,
        root=root,
        seed=seed,
    )


def freeze_from_result(
    result_root: Path,
    *,
    case_id: str,
    family: Family,
    source_prompt: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    determinism_class: DeterminismClass = DeterminismClass.PORTABLE,
    root: Path | None = None,
    seed: int = PINNED_SEED,
) -> CorpusCase:
    scene, program, overrides = load_live_result(result_root)
    return freeze_case(
        case_id=case_id,
        family=family,
        scene=scene,
        program=program,
        overrides=overrides,
        source_prompt=source_prompt,
        tags=tags,
        notes=notes,
        determinism_class=determinism_class,
        root=root,
        seed=seed,
    )
