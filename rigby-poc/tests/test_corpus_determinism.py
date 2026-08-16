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
from evals.corpus.freeze import body_actions_of, pin_seed
from evals.corpus.loader import CorpusCase, compile_inputs
from evals.corpus.models import DeterminismClass, StoragePolicy
from evals.corpus.verify import Verdict, compare_case

CASES = {case.id: case for case in load_corpus()}
CASE_IDS = sorted(CASES)

#: Fast cases used where a test compiles more than once; the point of those tests is
#: the comparison, not breadth, and breadth is covered case by case below.
CHEAP_CASE_IDS = ("gesture-shaka-playful-right", "strike-uppercut-right")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_the_corpus_is_not_empty() -> None:
    assert len(CASE_IDS) == 12, CASE_IDS


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_recompiles_to_its_recorded_motion(case_id: str) -> None:
    comparison = compare_case(CASES[case_id])
    if comparison.verdict is Verdict.UNBLESSED_PLATFORM:
        pytest.skip(comparison.render())
    assert comparison.verdict is Verdict.MATCH, "\n" + comparison.render()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_manifest_row_agrees_with_the_case_it_points_at(case_id: str) -> None:
    case = CASES[case_id]
    assert case.entry.intent == case.program.intent
    assert case.entry.body_actions == body_actions_of(case.program)
    assert case.entry.expected_structural_valid == case.expected.structural_valid
    assert case.expected.case_id == case_id


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_program_carries_a_pinned_seed(case_id: str) -> None:
    """A corpus program never inherits a seed derived from an LLM response id."""
    case = CASES[case_id]
    assert case.program == pin_seed(case.program, case.program.seed)
    assert case.program.seed == 0


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_03a_cases_are_portable_and_store_no_clip(case_id: str) -> None:
    """03a deliberately contains no MuJoCo case, so every hash is platform-free."""
    case = CASES[case_id]
    assert case.expected.determinism_class is DeterminismClass.PORTABLE
    assert case.entry.storage is StoragePolicy.PROGRAMS_ONLY
    assert not list(case.root.glob("*.gz"))


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
    key = platform_key()
    for case_id in CHEAP_CASE_IDS:
        assert observed[case_id] == CASES[case_id].expected.resolve_motion_sha256(key)
