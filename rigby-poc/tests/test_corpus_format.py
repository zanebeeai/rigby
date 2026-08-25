"""The case format, the loader, and the bless / freeze round trips."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from evals.corpus import (
    PORTABLE_PLATFORM_KEY,
    CorpusError,
    DeterminismClass,
    ExpectedResult,
    Family,
    compile_case,
    load_case,
    load_corpus,
    load_manifest,
    motion_sha256,
    observe,
    platform_key,
)
from evals.corpus.bless import bless
from evals.corpus.freeze import freeze_from_prompt, freeze_from_result, pin_seed
from evals.corpus.loader import EXPECTED_FILE, MANIFEST_NAME, PROGRAM_FILE, OVERRIDES_FILE
from evals.corpus.verify import Verdict, compare_case, rebless
from evals.review import motion_sha256 as review_motion_sha256
from pydantic import ValidationError
from rigby_poc.models import (
    CompileRequest,
    Hand,
    ParameterOverrides,
    PlanRequest,
    default_scene,
)

SAMPLE_CASE_ID = "gesture-shaka-playful-right"


def test_the_corpus_hash_is_the_same_string_the_run_archive_records(
    tmp_path: Path,
) -> None:
    """One definition of "the motion", shared with ``evals/review.py``.

    If these ever diverge, a corpus hash and an acceptance-run hash of the same clip
    stop being comparable, and no error would say so.
    """
    case = load_case(SAMPLE_CASE_ID)
    clip = compile_case(case)
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(json.dumps(clip.model_dump(mode="json")), encoding="utf-8")
    assert motion_sha256(clip) == review_motion_sha256(clip_path)


def test_a_portable_case_records_exactly_one_hash() -> None:
    with pytest.raises(ValidationError):
        _expected(
            determinism_class=DeterminismClass.PORTABLE,
            motion_sha256={"darwin-arm64|mujoco-3.11.0": "a" * 64},
        )


def test_a_platform_dependent_case_cannot_claim_a_portable_hash() -> None:
    with pytest.raises(ValidationError):
        _expected(
            determinism_class=DeterminismClass.PLATFORM_DEPENDENT,
            motion_sha256={PORTABLE_PLATFORM_KEY: "a" * 64},
            metrics_sha256={PORTABLE_PLATFORM_KEY: "b" * 64},
            observables_sha256={PORTABLE_PLATFORM_KEY: "c" * 64},
        )


def test_a_platform_is_blessed_for_all_three_digests_or_for_none() -> None:
    """Partial blessing would silently stop asserting the two derived digests.

    ``metrics_sha256`` and ``observables_sha256`` were bare strings before 03b, so a
    naive migration that only keyed ``motion_sha256`` would leave the other two
    asserting cross-platform bit-equality of floats -- the exact claim Windows CI
    disproved.
    """
    with pytest.raises(ValidationError, match="blessed for all three digests"):
        _expected(
            determinism_class=DeterminismClass.PLATFORM_DEPENDENT,
            motion_sha256={"darwin-arm64": "a" * 64, "win32-amd64": "d" * 64},
            metrics_sha256={"darwin-arm64": "b" * 64},
            observables_sha256={"darwin-arm64": "c" * 64},
        )


def test_an_unblessed_platform_is_a_skip_not_a_failure() -> None:
    """03b's MuJoCo cases will be blessed on macOS before Windows, or the reverse.

    Whoever runs second must see a clear "no hash for this platform", never a
    failure that looks like a compiler regression.
    """
    foreign = "someotheros-riscv64|mujoco-0.0.0"
    recorded = _expected(
        determinism_class=DeterminismClass.PLATFORM_DEPENDENT,
        motion_sha256={foreign: "a" * 64},
        metrics_sha256={foreign: "b" * 64},
        observables_sha256={foreign: "c" * 64},
    )
    for name in ExpectedResult.DIGESTS:
        assert recorded.resolve(name, platform_key()) is None


def test_blessing_a_platform_dependent_case_keeps_other_platforms_hashes(
    tmp_path: Path,
) -> None:
    """Whoever blesses second must not delete the first developer's hash."""
    root = _corpus_copy(tmp_path)
    case = load_case(SAMPLE_CASE_ID, root)
    foreign = "someotheros-riscv64|mujoco-0.0.0"
    recorded = case.expected.model_copy(
        update={
            "determinism_class": DeterminismClass.PLATFORM_DEPENDENT,
            "motion_sha256": {foreign: "a" * 64},
            "metrics_sha256": {foreign: "b" * 64},
            "observables_sha256": {foreign: "c" * 64},
        }
    )
    case = replace(case, expected=recorded)

    clip = compile_case(case)
    solver_used = case.expected.solver_used
    observed = observe(
        case.id, clip, DeterminismClass.PLATFORM_DEPENDENT, solver_used=solver_used
    )
    merged = rebless(case, observed)

    # Every digest keeps the other platform's entry, not just the motion one.
    assert merged.motion_sha256[foreign] == "a" * 64
    assert merged.metrics_sha256[foreign] == "b" * 64
    assert merged.observables_sha256[foreign] == "c" * 64
    assert merged.motion_sha256[platform_key(solver_used)] == motion_sha256(clip)


def test_loader_rejects_a_case_directory_the_manifest_does_not_list(
    tmp_path: Path,
) -> None:
    root = _corpus_copy(tmp_path)
    (root / "cases" / "stowaway").mkdir()
    with pytest.raises(CorpusError, match="missing from the manifest"):
        load_corpus(root)


def test_loader_rejects_an_expected_file_belonging_to_another_case(
    tmp_path: Path,
) -> None:
    root = _corpus_copy(tmp_path)
    path = root / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["case_id"] = "strike-jab-left"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError, match="expected.json for"):
        load_case(SAMPLE_CASE_ID, root)


def test_loader_rejects_an_unknown_field(tmp_path: Path) -> None:
    root = _corpus_copy(tmp_path)
    path = root / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frame_count_typo"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_case(SAMPLE_CASE_ID, root)


def test_bless_is_a_dry_run_until_write_is_passed(tmp_path: Path) -> None:
    root = _corpus_copy(tmp_path)
    path = root / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    from evals.corpus.hashing import platform_key

    # This platform's key explicitly: on a platform nobody has blessed there is no
    # entry to rewrite, and corrupting a foreign column makes `bless` report an
    # unblessed platform instead of a moved hash.
    key = platform_key(bool(payload.get("solver_used")))
    payload["motion_sha256"] = {**payload["motion_sha256"], key: "b" * 64}
    payload["metrics_sha256"] = {**payload["metrics_sha256"], key: "b" * 64}
    payload["observables_sha256"] = {**payload["observables_sha256"], key: "b" * 64}
    payload["frame_count"] = 3
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    stale = path.read_text(encoding="utf-8")

    report = bless([SAMPLE_CASE_ID], root=root, write=False)
    assert report.dry_run
    assert [item.case_id for item in report.moved] == [SAMPLE_CASE_ID]
    assert path.read_text(encoding="utf-8") == stale

    rendered = report.render()
    assert "motion_sha256" in rendered
    assert "b" * 64 in rendered
    assert "frame_count" in rendered
    assert "why each hash moved" in rendered

    applied = bless([SAMPLE_CASE_ID], root=root, write=True)
    assert applied.written == [SAMPLE_CASE_ID]
    assert compare_case(load_case(SAMPLE_CASE_ID, root)).verdict is Verdict.MATCH


def test_bless_rejects_an_unknown_case_id(tmp_path: Path) -> None:
    root = _corpus_copy(tmp_path)
    with pytest.raises(SystemExit, match="unknown corpus case"):
        bless(["no-such-case"], root=root)


def test_freezing_a_prompt_produces_a_case_that_reproduces(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    case = freeze_from_prompt(
        "throw a right hook",
        case_id="strike-hook-right",
        family=Family.STRIKE,
        tags=["right", "hook"],
        root=root,
    )
    assert case.program.seed == 0
    assert (root / MANIFEST_NAME).is_file()
    assert compare_case(load_case("strike-hook-right", root)).verdict is Verdict.MATCH

    manifest = load_manifest(root)
    assert [entry.id for entry in manifest.cases] == ["strike-hook-right"]
    assert manifest.coverage.intent.covered == ["strike"]


def test_freezing_a_live_result_pins_the_seed_and_keeps_its_overrides(
    tmp_path: Path,
) -> None:
    """``store.py`` persists the pre-override program, so a case has to carry the
    overrides too or it compiles to different motion than the run it came from."""
    from rigby_poc.planner import plan_motion

    scene = default_scene()
    program = plan_motion(
        PlanRequest(text="throw a right jab", scene=scene, provider="offline")
    ).program.model_copy(update={"seed": 1_234_567})
    overrides = ParameterOverrides(hand=Hand.LEFT, easing=0.25)

    result_root = tmp_path / "results" / "000042-throw-a-right-jab"
    result_root.mkdir(parents=True)
    (result_root / "request.json").write_text(
        CompileRequest(
            scene=scene, program=program, parameter_overrides=overrides
        ).model_dump_json(),
        encoding="utf-8",
    )

    root = tmp_path / "corpus"
    case = freeze_from_result(
        result_root, case_id="strike-jab-frozen", family=Family.STRIKE, root=root
    )

    assert case.program.seed == 0
    assert case.entry.source_seed == 1_234_567
    assert (case.root / OVERRIDES_FILE).is_file()

    reloaded = load_case("strike-jab-frozen", root)
    assert reloaded.overrides.easing == 0.25
    assert reloaded.overrides.hand is Hand.LEFT
    assert compare_case(reloaded).verdict is Verdict.MATCH


def test_freezing_rejects_a_prompt_the_offline_planner_cannot_support(
    tmp_path: Path,
) -> None:
    with pytest.raises(CorpusError, match="offline planner rejected"):
        freeze_from_prompt(
            "recite the alphabet backwards in latin",
            case_id="nonsense",
            family=Family.GESTURE,
            root=tmp_path / "corpus",
        )


def test_pinning_a_seed_reaches_sequence_steps() -> None:
    from rigby_poc.planner import plan_motion

    program = plan_motion(
        PlanRequest(
            text="wave with your right hand, then pick up the block",
            scene=default_scene(),
            provider="offline",
        )
    ).program
    seeded = program.model_copy(
        update={"seed": 99, "steps": [step.model_copy(update={"seed": 99}) for step in program.steps]}
    )
    pinned = pin_seed(seeded)
    assert pinned.seed == 0
    assert pinned.steps
    assert all(step.seed == 0 for step in pinned.steps)


def test_case_programs_on_disk_round_trip_through_their_models() -> None:
    """The committed JSON is exactly what the model serialises, so a hand edit that
    changes formatting but not content does not show up as a spurious diff."""
    for case in load_corpus():
        raw = json.loads((case.root / PROGRAM_FILE).read_text(encoding="utf-8"))
        assert case.program.model_dump(mode="json") == raw


def _expected(**overrides: object) -> ExpectedResult:
    from evals.corpus.models import BlessEnvironment

    payload: dict[str, object] = {
        "case_id": "sample-case",
        "determinism_class": DeterminismClass.PORTABLE,
        "motion_sha256": {PORTABLE_PLATFORM_KEY: "a" * 64},
        "metrics_sha256": {PORTABLE_PLATFORM_KEY: "b" * 64},
        "observables_sha256": {PORTABLE_PLATFORM_KEY: "c" * 64},
        "success": True,
        "structural_valid": True,
        "fps": 30,
        "frame_count": 10,
        "duration_s": 0.33,
        "contact_count": 0,
        "environment": BlessEnvironment(
            platform_key="darwin-arm64|mujoco-3.11.0",
            python_version="3.12.0",
            compiler_version="rigby-compiler-0.4.0",
            blessed_at="2026-01-01T00:00:00+00:00",
        ),
    }
    payload.update(overrides)
    return ExpectedResult.model_validate(payload)


def _corpus_copy(tmp_path: Path) -> Path:
    from shutil import copytree

    from evals.corpus import CORPUS_ROOT

    root = tmp_path / "corpus"
    copytree(
        CORPUS_ROOT,
        root,
        ignore=lambda _, names: [name for name in names if name == "__pycache__"],
    )
    return root


def test_no_corpus_test_asserts_that_this_platform_is_blessed(tmp_path: Path) -> None:
    """The whole corpus tooling must work on a platform nobody has blessed.

    The first Windows CI run failed eight of these tests, and none of the eight was
    about MuJoCo or about floating point. Every one assumed the running machine had
    a column in ``expected.json``: the cross-process test compared a real hash
    against ``None``, two corruption helpers rewrote whatever keys happened to be
    there and so corrupted a *foreign* column, and the freeze round trip compared
    whole digest maps rather than this platform's entry.

    Simulating an unblessed platform is cheap -- rename every column to a foreign
    key -- and it is the only way to exercise that path from a machine that *is*
    blessed. Without this, the next such assumption is found by CI on somebody
    else's machine, which is how the previous eight were found.
    """
    root = _corpus_copy(tmp_path)
    for path in sorted((root / "cases").glob("*/expected.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name in ExpectedResult.DIGESTS:
            payload[name] = {
                f"foreignos-otherarch{'|mujoco-0.0.0' if '|' in key else ''}": value
                for key, value in payload[name].items()
            }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    cases = load_corpus(root)
    assert len(cases) == 47
    key = platform_key()
    for case in cases:
        for name in ExpectedResult.DIGESTS:
            assert case.expected.resolve(name, key) is None, (
                f"{case.id} still resolves a {name} for {key!r} after the rename"
            )
        # The committed clip is what an unblessed platform compares against, so it
        # has to be there for every case, not only the MuJoCo ones.
        assert case.slim_clip_path.is_file(), case.id

    comparison = compare_case(load_case(SAMPLE_CASE_ID, root))
    assert comparison.verdict is Verdict.TOLERANCE_MATCH, comparison.render()
    assert comparison.tolerance is not None
    assert comparison.tolerance.max_deviation == 0.0
