"""Every corpus case must recompile to the motion it was blessed with.

This is the point of the corpus: it turns "compilation is deterministic" from a
claim into a test, and it makes a compiler change that alters motion fail loudly
instead of silently re-blessing itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from evals.corpus import compile_case, load_corpus, motion_sha256, platform_key
from evals.corpus.freeze import (
    body_actions_of,
    hand_shapes_of,
    object_actions_of,
    pin_seed,
    strike_types_of,
)
from evals.corpus.hashing import CANONICAL_MOTION_KEYS
from evals.corpus.loader import CorpusCase, compile_inputs
from evals.corpus.models import DeterminismClass, StoragePolicy
from evals.corpus.verify import (
    SlimClipShapeError,
    Verdict,
    compare_case,
    compare_slim_clip,
)
from rigby_poc import compiler

CASES = {case.id: case for case in load_corpus()}
CASE_IDS = sorted(CASES)

#: Fast cases used where a test compiles more than once; the point of those tests is
#: the comparison, not breadth, and breadth is covered case by case below.
CHEAP_CASE_IDS = ("gesture-shaka-playful-right", "strike-uppercut-right")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


#: Cases whose motion passes through the MuJoCo solver, so the MuJoCo version is
#: part of their platform key.  Derived rather than listed; the declaration itself
#: is checked by ``test_a_case_that_runs_the_solver_declares_itself``.
SOLVER_CASE_IDS = sorted(
    case_id for case_id, case in CASES.items() if case.expected.solver_used
)


def test_the_corpus_is_not_empty() -> None:
    assert len(CASE_IDS) == 47, CASE_IDS


def test_no_case_claims_to_be_bit_identical_across_platforms() -> None:
    """The 03a premise, disproved by Windows CI and pinned here.

    03a classified everything that avoided MuJoCo as ``portable``, meaning one hash
    under the reserved ``"any"`` key, asserted on every platform.  A Windows run on
    x86-64 against hashes blessed on darwin-arm64 failed 12 cases on last-few-ulp
    floating point -- libm ``sin``/``cos``/``acos`` and FMA contraction differ
    between the two instruction sets, amplified most in
    ``max_angular_jerk_rad_s3`` because a third derivative divides by ``dt`` three
    times.  Bit-identical floating point across ISAs is not achievable without
    pinning the math library, so ``portable`` is a claim no case can make.

    If a case is ever genuinely bit-identical everywhere, this test is what has to
    be changed to say so, deliberately.
    """
    claimed = sorted(
        case_id
        for case_id, case in CASES.items()
        if case.expected.determinism_class is DeterminismClass.PORTABLE
    )
    assert not claimed, claimed


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_the_platform_key_names_mujoco_only_when_the_solver_ran(case_id: str) -> None:
    """A ``mujoco`` upgrade must not invalidate a case that never enters it.

    If it did, re-blessing would become routine, and a routine re-bless is
    indistinguishable from a real drift at review time -- which is the whole value
    of the hash.
    """
    case = CASES[case_id]
    keys = set(case.expected.motion_sha256)
    named = {key for key in keys if "mujoco-" in key}
    assert named == (keys if case.expected.solver_used else set()), keys


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_recompiles_to_its_recorded_motion(case_id: str) -> None:
    comparison = compare_case(CASES[case_id])
    if comparison.verdict is Verdict.UNBLESSED_PLATFORM:
        pytest.skip(comparison.render())
    if comparison.tolerance is not None:
        # No hash is blessed for this platform, so the committed clip is all there
        # is to compare against.  Reported either way and never failed: a drift here
        # is a finding about MuJoCo's cross-platform behaviour or about the
        # PROVISIONAL tolerance, and neither is a reason to turn a first run on a
        # new platform red for everybody.  `python -m evals.corpus verify` is the
        # strict form and does exit non-zero on it.
        pytest.xfail(comparison.render())
    assert comparison.verdict is Verdict.MATCH, "\n" + comparison.render()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_manifest_row_agrees_with_the_case_it_points_at(case_id: str) -> None:
    case = CASES[case_id]
    assert case.entry.intent == case.program.intent
    assert case.entry.body_actions == body_actions_of(case.program)
    assert case.entry.object_actions == object_actions_of(case.program)
    assert case.entry.strike_types == strike_types_of(case.program)
    assert case.entry.hand_shapes == hand_shapes_of(case.program)
    assert case.entry.expected_structural_valid == case.expected.structural_valid
    assert case.expected.case_id == case_id


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_program_carries_a_pinned_seed(case_id: str) -> None:
    """A corpus program never inherits a seed derived from an LLM response id."""
    case = CASES[case_id]
    assert case.program == pin_seed(case.program, case.program.seed)
    assert case.program.seed == 0


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_a_case_that_runs_the_solver_declares_itself(case_id: str) -> None:
    """Whether a case enters the solver is observed, never trusted.

    Declaring a MuJoCo case ``portable`` records its hash under the reserved
    MuJoCo version out of its platform key, so a solver upgrade would silently
    leave a stale hash asserted rather than flagging the case for a re-bless.

    03b found four such cases while writing this: ``object-drop-right``,
    ``object-throw-far``, ``sequence-grab-then-wave`` and the throw-then-catch
    sequence all reach ``simulate_grasp`` without being named ``grab``.  A list of
    grasp-looking case ids would have missed every one, so this counts calls.
    """
    case = CASES[case_id]
    calls = 0
    real = compiler.simulate_grasp

    def counting(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    compiler.simulate_grasp = counting  # type: ignore[assignment]
    try:
        compile_case(case)
    finally:
        compiler.simulate_grasp = real  # type: ignore[assignment]

    assert case.expected.solver_used is (calls > 0), (
        f"{case_id} calls simulate_grasp {calls} time(s) but declares "
        f"solver_used={case.expected.solver_used}"
    )
    assert case.entry.storage is StoragePolicy.PROGRAM_AND_CLIP
    assert case.slim_clip_path.is_file()


@pytest.mark.parametrize("case_id", SOLVER_CASE_IDS)
def test_the_committed_slim_clip_matches_a_fresh_compile_exactly(case_id: str) -> None:
    """On a blessed platform the fallback comparator's deviation must be zero.

    This is the only part of the per-platform fallback that can be verified without
    a second machine.  It pins the comparator: if ``compare_slim_clip`` ever starts
    reporting drift here, the tolerance is masking a bug rather than absorbing
    solver noise, and the number a second platform reports would be meaningless.
    """
    case = CASES[case_id]
    report = compare_slim_clip(case, compile_case(case))
    assert report.leaves > 1000, report
    assert report.max_deviation == 0.0, (
        f"{case_id} drifted by {report.max_deviation:.3e} at {report.at} on the "
        f"platform that blessed it"
    )


@pytest.mark.parametrize("case_id", SOLVER_CASE_IDS)
def test_the_slim_clip_holds_only_motion(case_id: str) -> None:
    """A stored clip must not become a second source of truth for metrics."""
    case = CASES[case_id]
    stored = case.slim_clip()
    assert stored is not None
    assert set(stored) == set(CANONICAL_MOTION_KEYS)
    assert "metrics" not in stored


def test_a_slim_clip_that_changed_shape_is_not_absorbed_by_the_tolerance() -> None:
    """A dropped frame is a motion change, not solver drift."""
    case = CASES[SOLVER_CASE_IDS[0]]
    clip = compile_case(case)
    shortened = clip.model_copy(update={"frames": clip.frames[:-1]})
    with pytest.raises(SlimClipShapeError):
        compare_slim_clip(case, shortened)


@pytest.mark.parametrize("case_id", CHEAP_CASE_IDS)
def test_recompiling_in_the_same_process_is_idempotent(case_id: str) -> None:
    case = CASES[case_id]
    assert motion_sha256(compile_case(case)) == motion_sha256(compile_case(case))


@pytest.mark.parametrize("case_id", CHEAP_CASE_IDS)
def test_motion_does_not_depend_on_the_program_seed(case_id: str) -> None:
    """``program.seed`` is provenance, never a control.

    The compiler receives it and never consumes it, which is why a corpus can pin
    the seed to a literal without changing what the case compiles to. If this ever
    fails, seed pinning becomes lossy and the case format has to record the seed the
    motion was produced with.
    """
    case: CorpusCase = CASES[case_id]
    reseeded = pin_seed(case.program, 2_123_456_789)
    baseline = compile_inputs(case.scene, case.program, case.overrides)
    varied = compile_inputs(case.scene, reseeded, case.overrides)
    assert motion_sha256(baseline) == motion_sha256(varied)


_SUBPROCESS_PROBE = """
import json, sys
from evals.corpus import load_case, compile_case, motion_sha256
print(json.dumps({case_id: motion_sha256(compile_case(load_case(case_id)))
                  for case_id in sys.argv[1:]}))
"""


@pytest.mark.parametrize("hash_seed", ("0", "1", "524287"))
def test_hashes_survive_a_fresh_process_with_a_different_hash_seed(
    hash_seed: str,
) -> None:
    """Separate process, different ``PYTHONHASHSEED``, same motion.

    Set iteration order and ``hash()`` are the classic sources of cross-process
    drift. Pinning them here means a future compiler change that starts iterating a
    set into output is caught by this suite rather than by a developer whose CI
    disagrees with their laptop.
    """
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE, *CHEAP_CASE_IDS],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    for case_id in CHEAP_CASE_IDS:
        expected = CASES[case_id].expected
        key = platform_key(expected.solver_used)
        assert observed[case_id] == expected.resolve_motion_sha256(key)
