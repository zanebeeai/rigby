"""The equivalence harness: analysis output must match the compiler exactly.

Two independent assertions per case, and they fail for different reasons:

* against ``clip.metrics`` from the same run — proves the extraction is
  behaviour-preserving *today*, and can never go stale;
* against the frozen fixture — proves the numbers have not moved since 02a
  landed, and catches a compiler change that silently alters both sides.

The fixture is a stand-in for the golden corpus (plan 03). When that lands it
supersedes ``tests/fixtures/analysis_equivalence/``.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest

from rigby_poc import analysis
from rigby_poc.analysis.equivalence import owned_metric_keys, required_metric_keys
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    Hand,
    MotionProgram,
    ParameterOverrides,
    SceneManifest,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "analysis_equivalence"
CASE_IDS: list[str] = json.loads(
    (FIXTURE_DIR / "index.json").read_text(encoding="utf-8")
)["cases"]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _load_case(case_id: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def _compile_case(case: dict):
    scene = SceneManifest.model_validate(case["scene"])
    program = MotionProgram.model_validate(case["program"])
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    return scene, program, clip


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_analysis_matches_the_compiler_for_every_frozen_case(case_id: str) -> None:
    case = _load_case(case_id)
    scene, program, clip = _compile_case(case)

    assert clip.frames, f"{case_id}: compiled to zero frames"
    result = analysis.analyze(clip, program, scene)

    for key in sorted(result):
        assert key in clip.metrics, (
            f"{case_id}: analysis produced {key!r}, which the compiler does not emit"
        )
        assert _canonical(result[key]) == _canonical(clip.metrics[key]), (
            f"{case_id}: {key!r} differs between analysis and the compiler"
        )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_analysis_output_is_byte_identical_to_the_frozen_fixture(
    case_id: str,
) -> None:
    case = _load_case(case_id)
    scene, program, clip = _compile_case(case)

    result = analysis.analyze(clip, program, scene)

    assert _canonical(result) == _canonical(case["expected_metrics"]), (
        f"{case_id}: analysis output drifted from the frozen fixture. If a metric "
        "definition changed deliberately, regenerate with "
        "`uv run python tests/fixtures/analysis_equivalence/freeze.py`."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_analysis_owns_exactly_the_declared_key_set(case_id: str) -> None:
    case = _load_case(case_id)
    scene, program, clip = _compile_case(case)

    keys = set(analysis.analyze(clip, program, scene))

    unexpected = sorted(keys - owned_metric_keys(program))
    assert not unexpected, (
        f"{case_id}: analysis emitted undeclared keys {unexpected}; add them to "
        "rigby_poc.analysis.equivalence"
    )
    dropped = sorted(required_metric_keys(program) - keys)
    assert not dropped, (
        f"{case_id}: analysis silently stopped producing {dropped}"
    )


def test_the_fixture_covers_every_supported_intent() -> None:
    intents = {_load_case(case_id)["intent"] for case_id in CASE_IDS}

    assert intents == {
        "gesture",
        "grab",
        "strike",
        "composite",
        "full_body",
        "object_interaction",
        "sequence",
    }


def test_the_fixture_exercises_every_owned_metric_family() -> None:
    """Every family in the ownership ledger must be non-empty somewhere.

    Without this a family could quietly emit nothing for all fourteen cases and
    the equivalence assertions above would still pass.
    """

    from rigby_poc.analysis import equivalence

    families = {
        "safety": equivalence.SAFETY_KEYS,
        "angular": equivalence.ANGULAR_KEYS,
        "gesture_structure": equivalence.GESTURE_STRUCTURE_KEYS,
        "shake": equivalence.SHAKE_KEYS,
        "contact": equivalence.CONTACT_KEYS,
        "semantic_cycle": equivalence.SEMANTIC_KEYS,
        "parallel_forearm": equivalence.PARALLEL_FOREARM_KEYS,
    }
    seen: set[str] = set()
    for case_id in CASE_IDS:
        seen |= set(_load_case(case_id)["expected_metrics"])

    uncovered = sorted(name for name, keys in families.items() if not keys <= seen)
    assert not uncovered, f"no fixture case exercises: {uncovered}"


def test_analysis_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """``analyze`` must run with no server, no browser and no API key."""

    case = _load_case("composite_travel_foul")
    scene, program, clip = _compile_case(case)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("analysis opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert analysis.analyze(clip, program, scene)


def test_analysis_of_a_125_frame_clip_stays_cheap() -> None:
    """The fast tier depends on analysis being cheap enough to run everywhere.

    Plan 02 §5 targets 100 ms for a 125-frame clip. Measured on the heaviest
    path (composite travel signal, one full-hierarchy FK evaluation per frame)
    that is ~85 ms, so the target holds but with little headroom. The assertion
    below is a regression ceiling with room for slower CI hardware, not the
    target itself — a tight gate here would flake rather than inform.
    """

    case = _load_case("composite_travel_foul")
    scene, program, clip = _compile_case(case)
    trimmed = clip.model_copy(update={"frames": clip.frames[:125]})

    analysis.analyze(trimmed, program, scene)  # warm the rig cache
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        analysis.analyze(trimmed, program, scene)
        samples.append((time.perf_counter() - start) * 1000.0)
    elapsed_ms = min(samples)

    assert elapsed_ms < 300.0, f"analysis took {elapsed_ms:.1f} ms for 125 frames"


def test_a_stored_result_must_be_read_through_apply_overrides(tmp_path: Path) -> None:
    """``store.py`` persists the pre-override program, so reading it is a trap.

    ``artifacts.load_analysis_inputs`` applies the overrides; reading
    ``program.json`` directly analyses a program the clip was never compiled
    from.
    """

    from rigby_poc.analysis.artifacts import load_analysis_inputs
    from rigby_poc.store import ResultStore

    case = _load_case("gesture_shaka_right")
    scene = SceneManifest.model_validate(case["scene"])
    program = MotionProgram.model_validate(case["program"])
    assert program.hand == Hand.RIGHT
    request = CompileRequest(
        scene=scene,
        program=program,
        parameter_overrides=ParameterOverrides(hand=Hand.LEFT, duration_s=1.75),
    )
    clip = compile_motion(request)

    result_id = ResultStore(root=tmp_path).persist(request, clip)
    result_dir = tmp_path / result_id

    persisted = MotionProgram.model_validate_json(
        (result_dir / "program.json").read_text(encoding="utf-8")
    )
    loaded_clip, effective_program, effective_scene = load_analysis_inputs(result_dir)

    assert persisted.hand == Hand.RIGHT, "store no longer persists the raw program"
    assert effective_program.hand == Hand.LEFT

    from_effective = analysis.analyze(
        loaded_clip, effective_program, effective_scene
    )
    from_persisted = analysis.analyze(loaded_clip, persisted, effective_scene)

    for key in from_effective:
        assert _canonical(from_effective[key]) == _canonical(clip.metrics[key])
    assert _canonical(from_persisted) != _canonical(from_effective), (
        "analysing the persisted pre-override program agreed with the effective "
        "one; the trap is real but this case no longer demonstrates it"
    )
