"""The equivalence harness: analysis output must match the compiler exactly.

Three independent assertions per case, and they fail for different reasons:

* ``compiler_metrics`` — the whole of ``ClipResult.metrics`` against a baseline
  frozen *before* PR 02b began moving metric blocks out of ``compiler.py``.
  This is the gate on the extraction itself. Moving a block into
  ``rigby_poc.analysis`` must not change a single digit of what the compiler
  emits, and this assertion is the only one that can prove it, because the
  other two compare the analyzer against a compiler that has already been
  changed;
* against ``clip.metrics`` from the same run — proves the analyzer agrees with
  the compiler *today*, and can never go stale;
* against the frozen ``expected_metrics`` — pins the analysis layer's own
  output, which grows as each path is ported.

The fixture is a stand-in for the golden corpus (plan 03). When that lands it
supersedes ``tests/fixtures/analysis_equivalence/``.
"""

from __future__ import annotations

import json
import socket
import sys
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
sys.path.insert(0, str(FIXTURE_DIR))
from freeze import architecture_key  # noqa: E402

_INDEX = json.loads((FIXTURE_DIR / "index.json").read_text(encoding="utf-8"))
CASE_IDS: list[str] = _INDEX["cases"]
BLESSED_ARCHITECTURE: str = _INDEX["blessed_architecture"]

# Both frozen snapshots are exact floats, so they are statements about the
# machine that produced them and nothing more. Measured on this fixture,
# darwin-arm64 and a Windows x86-64 runner disagree by up to 56 ulps on
# max_angular_jerk_rad_s3 — different libm transcendentals, different FMA
# contraction, different numpy and scipy SIMD kernels. Nothing in the code under
# test is wrong when that happens.
#
# The alternative to skipping is a relative tolerance near 1e-13, and that costs
# more than it buys: this harness caught a deliberate 1e-15 perturbation of the
# compiler during 02b, which is exactly the size of error a bad extraction
# produces. A gate loose enough to be portable would not have caught it. So the
# comparisons stay exact and run only where they mean something; bless another
# architecture by running freeze.py on it.
#
# What still runs everywhere is the assertion that matters most on a foreign
# machine: analysis output against the compiler *in the same process*. Both
# sides then see the same libm, so an extraction that broke on Windows alone
# would still be caught on Windows.
exact_snapshot = pytest.mark.skipif(
    architecture_key() != BLESSED_ARCHITECTURE,
    reason=(
        f"snapshot blessed on {BLESSED_ARCHITECTURE}, running on "
        f"{architecture_key()}; exact float comparison is architecture-local"
    ),
)


# Metric keys the compiler gained after the baseline was frozen, with why.
# Each one is authoring intent that no post-hoc pass can invert out of a clip,
# so persisting it is what lets the analysis layer reproduce the metrics that
# compare achieved motion against what was commanded (plan 02 §1.4).
ADDED_COMPILER_KEYS: dict[str, str] = {
    "support_constraints": (
        "commanded ankle position per constrained frame; feeds "
        "max_support_foot_target_error_m, max_support_foot_slide_per_frame_m "
        "and support_contact_fraction"
    ),
    "climb_support_constraints": (
        "commanded per-limb climb support targets; feeds "
        "climb_support_target_max_error_m and climb_three_point_support_fraction"
    ),
}


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


@exact_snapshot
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_compiler_metrics_match_the_pre_extraction_baseline(case_id: str) -> None:
    """``compiler.py`` must emit the same numbers it did before the move.

    PR 02 is an extraction, not a redesign: every metric block that moves into
    ``rigby_poc.analysis`` has to leave ``ClipResult.metrics`` byte-identical.
    The baseline in each fixture file was frozen from the compiler before 02b
    touched it, and ``freeze.py`` refuses to overwrite it without an explicit
    ``--rebless-compiler``.

    A key may be *added* — 02b persists the commanded IK support targets, which
    are authoring intent no post-hoc pass can recover — but every addition has
    to be declared in ``ADDED_COMPILER_KEYS`` below, and nothing already frozen
    may move.
    """

    case = _load_case(case_id)
    baseline = case["compiler_metrics"]
    _, _, clip = _compile_case(case)

    for key in sorted(baseline):
        assert key in clip.metrics, f"{case_id}: the compiler stopped emitting {key!r}"
        assert _canonical(clip.metrics[key]) == _canonical(baseline[key]), (
            f"{case_id}: {key!r} moved during the extraction. This is a move, not "
            "a redesign — find the cause rather than re-blessing the baseline."
        )

    added = sorted(set(clip.metrics) - set(baseline))
    undeclared = [key for key in added if key not in ADDED_COMPILER_KEYS]
    assert not undeclared, (
        f"{case_id}: the compiler grew undeclared metric keys {undeclared}; add "
        "them to ADDED_COMPILER_KEYS with the reason they cannot be derived"
    )


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


@exact_snapshot
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
