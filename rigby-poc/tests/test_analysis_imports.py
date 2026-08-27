"""The analysis package must stay independent of the compiler.

This is the property that makes the layer worth extracting: a check has to be
runnable against a stored clip, a mutated clip or a corpus fixture without
dragging in generation, physics, or a MuJoCo model. It is also the property
most easily lost by accident — one convenience import of ``compiler`` closes a
cycle, because ``compiler`` imports this package for the moved helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ``thresholds`` is allowed because reading a committed limit file is part of
# being runnable against a stored clip, not in tension with it: a check that
# cannot reach its own threshold is no more portable than one that needs the
# compiler. Its whole import closure is itself -- stdlib only, no numpy, no
# scipy, no rig, no I/O beyond one lru_cached JSON read -- which makes it a
# strictly cleaner dependency than ``primitives``, already allowed here.
# ``arm_plane`` is exempt for the same structural reason as ``thresholds``: it
# is pure numpy/scipy bend-plane math whose entire import closure is itself --
# no rig asset, no I/O, and by its own docstring it "must not import either"
# ``primitives`` or ``kinematics`` because it sits *below* both. Nothing in
# ``analysis`` imports it directly; it arrives transitively because the two
# already-allowed, load-bearing siblings (``kinematics`` for ``rig_kinematics``,
# ``primitives``) import its solver helpers at load time. Allowing it keeps
# this guard about its stated intent -- analysis must run without the
# compiler, physics, planner or store -- which a leaf math module does not
# threaten. Its constants are pinned against the analysis-derived frames by
# ``tests/test_arm_plane.py``, so the two layers cannot silently diverge.
ALLOWED_SIBLINGS = {
    "rigby_poc",
    "rigby_poc.arm_plane",
    "rigby_poc.models",
    "rigby_poc.kinematics",
    "rigby_poc.primitives",
    "rigby_poc.thresholds",
}


def _modules_after_importing(statement: str) -> set[str]:
    script = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('rigby_poc'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_importing_analysis_does_not_import_the_compiler() -> None:
    loaded = _modules_after_importing("import rigby_poc.analysis")

    siblings = {name for name in loaded if not name.startswith("rigby_poc.analysis")}

    assert siblings <= ALLOWED_SIBLINGS, (
        f"rigby_poc.analysis pulled in {sorted(siblings - ALLOWED_SIBLINGS)}; the "
        "analysis layer must not depend on the compiler, physics, planner or store"
    )


def test_importing_analysis_does_not_import_mujoco() -> None:
    """Checks must run where the physics engine is not installed or not wanted."""

    script = (
        "import sys\n"
        "import rigby_poc.analysis\n"
        "print('mujoco' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip().splitlines()[-1] == "False"


def test_the_compiler_still_imports_cleanly_on_top_of_analysis() -> None:
    loaded = _modules_after_importing("import rigby_poc.compiler")

    assert "rigby_poc.analysis" in loaded


def test_the_quality_shim_re_exports_the_moved_names() -> None:
    """``evals`` and older tests still import from ``rigby_poc.quality``."""

    import rigby_poc.quality as shim
    from rigby_poc.analysis import gesture

    for name in shim.__all__:
        assert getattr(shim, name) is getattr(gesture, name), name
