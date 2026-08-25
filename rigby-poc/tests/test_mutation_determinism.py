"""Mutations are pure: same input, byte-identical output. Plan 06 section 3.6.

A calibration suite that cannot be reproduced cannot be audited, and a mutation that
drifts between runs turns a detection threshold into a moving target.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.legacy import legacy_specs

#: Compiles corpus cases, so `medium` by input rather than by duration.
pytestmark = pytest.mark.medium

GESTURE_CASE = "gesture-hangten-shake-right"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clip():
    return compile_case(load_case(GESTURE_CASE))


@pytest.fixture(scope="module")
def specs():
    return legacy_specs()


def test_the_same_spec_twice_gives_byte_identical_output(clip, specs):
    for spec in specs:
        first = spec.apply(clip).model_dump(mode="json")
        second = spec.apply(clip).model_dump(mode="json")
        assert first == second, spec.id


def test_two_specs_do_not_interfere(clip, specs):
    """Applying B after A must give the same B as applying B alone.

    Shared mutable state between specs -- a cached frame list, a module-level
    accumulator -- would make a sweep's results depend on iteration order, which is
    invisible in any single run.
    """
    alone = {spec.id: spec.apply(clip).model_dump(mode="json") for spec in specs}
    for spec in specs:
        for other in specs:
            other.apply(clip)
        assert spec.apply(clip).model_dump(mode="json") == alone[spec.id], spec.id


_PROBE = """
import json, sys
from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.legacy import legacy_specs
from evals.corpus.hashing import sha256_value

clip = compile_case(load_case(sys.argv[1]))
print(json.dumps({
    spec.id: sha256_value(spec.apply(clip).model_dump(mode="json")["frames"])
    for spec in legacy_specs()
}))
"""


@pytest.mark.parametrize("hash_seed", ("0", "524287"))
def test_mutations_survive_a_fresh_process_with_a_different_hash_seed(
    clip, specs, hash_seed
) -> None:
    """Set iteration order is the classic source of cross-process drift.

    The mutation layer walks ``frame.bones``, which is a dict, so an implementation
    that started iterating a *set* into output would pass in-process and fail here.
    """
    from evals.corpus.hashing import sha256_value

    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, GESTURE_CASE],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    for spec in specs:
        expected = sha256_value(spec.apply(clip).model_dump(mode="json")["frames"])
        assert observed[spec.id] == expected, spec.id
