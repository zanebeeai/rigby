"""Every module must be reachable: imported by something, or runnable as a script.

A green suite does not detect code that nothing calls. Lane `judge` shipped
`llm_graders.py` with four graders that raised `AttributeError` on first call --
the extraction they depended on was half-finished -- and the suite stayed green
through it, because no test and no module referenced the file at all. It was
found only when someone sat down to write the tests.

That is the same defect as a gate that cannot fail, one level up: the code is
present, documented, and structurally unable to be observed. This test closes it
at the module level, which is the cheapest place to close it.

"Reachable" has three legitimate doors, not one, and a guard that recognises only
the first flags 19 innocent modules on this tree and gets an allowlist within a
week -- at which point it looks like coverage while being maintained by nobody.
A module is reachable if:

* something other than itself imports it -- **including a test**. Fourteen
  modules here are imported only by tests (`transcript` awaiting its 01b/01c
  wiring, `calibration_stats`, `thresholds`, eight CLI tools). Test-only is a
  door, not a smell;
* it is an executable script with an ``if __name__ == "__main__":`` guard.
  `evals/` holds five of those and they are legitimately not imported; or
* it is a console entry point declared in ``pyproject.toml``.

This is a reachability guard, not a coverage gate. A module with one trivial
test passes, and that is correct -- conflating the two is how it grows an
allowlist. Predicate independently modelled by lane `capture` against the same
tree, reaching the same three doors.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (PROJECT_ROOT / "src" / "rigby_poc", PROJECT_ROOT / "evals")
SCAN_ROOTS = (*SOURCE_ROOTS, PROJECT_ROOT / "tests")
SKIP_PARTS = frozenset({".venv", "node_modules", "__pycache__", ".git"})
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

#: Package plumbing is reachable by definition -- importing the package runs it.
PLUMBING = frozenset({"__init__", "__main__"})


def _console_script_modules() -> set[str]:
    """Modules named as console entry points in pyproject.toml.

    A console script is reachable from the command line without a ``__main__``
    guard, because the installer generates the wrapper.
    """

    import tomllib

    scripts = (
        tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        .get("project", {})
        .get("scripts", {})
    )
    return {target.split(":", 1)[0].split(".")[-1] for target in scripts.values()}


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not SKIP_PARTS.intersection(path.parts)
    ]


def _parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not expected
        return None


def _is_executable_script(tree: ast.AST) -> bool:
    """True if the module guards a body with ``__name__ == "__main__"``."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if isinstance(left, ast.Name) and left.id == "__name__":
            for comparator in node.test.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
                    return True
    return False


def _imported_names(tree: ast.AST) -> set[str]:
    """Every module name this file imports, by last dotted component."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Call):
            function = node.func
            label = getattr(function, "attr", None) or getattr(function, "id", None)
            if label in {"import_module", "__import__"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        names.update(argument.value.split("."))
    return names


@pytest.fixture(scope="module")
def trees() -> dict[Path, ast.AST]:
    parsed: dict[Path, ast.AST] = {}
    for root in SCAN_ROOTS:
        for path in _python_files(root):
            tree = _parse(path)
            if tree is not None:
                parsed[path] = tree
    return parsed


def test_every_module_is_imported_or_is_an_executable_script(
    trees: dict[Path, ast.AST],
) -> None:
    """The guard. See the module docstring for the incident that motivated it."""

    importers: dict[str, set[Path]] = {}
    for path, tree in trees.items():
        for name in _imported_names(tree):
            importers.setdefault(name, set()).add(path)

    console_scripts = _console_script_modules()

    unreachable: list[str] = []
    for root in SOURCE_ROOTS:
        for path in _python_files(root):
            stem = path.stem
            if stem in PLUMBING:
                continue
            tree = trees.get(path)
            if tree is None:
                continue
            if _is_executable_script(tree) or stem in console_scripts:
                continue
            if importers.get(stem, set()) - {path}:
                continue
            unreachable.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert not unreachable, (
        "these modules are imported by nothing and are not executable scripts, so "
        "nothing in the suite can observe them being broken:\n"
        + "\n".join(f"  {name}" for name in unreachable)
        + "\nAdd a test that imports the module, or give it a __main__ guard if it "
        "is genuinely a script."
    )


def test_the_known_entry_point_scripts_are_recognised_as_scripts(
    trees: dict[Path, ast.AST],
) -> None:
    """The exemption must be earned by a real guard, not assumed by directory.

    If one of these loses its ``__main__`` guard it stops being a script and the
    guard above should start demanding an importer.
    """

    for name in (
        "autonomous_goal_audit",
        "batch_flywheel",
        "complex_structural_sweep",
        "render_demo_gif",
        "rerank_existing",
    ):
        path = PROJECT_ROOT / "evals" / f"{name}.py"
        assert path.exists(), f"{name} moved; update this list deliberately"
        tree = trees[path]
        assert _is_executable_script(tree), f"{name} lost its __main__ guard"


def test_console_entry_points_are_recognised() -> None:
    """`rigby-humanoid = "rigby_poc.app:run"` makes `app` reachable from a shell."""

    assert "app" in _console_script_modules()


def test_the_script_detector_does_not_fire_on_a_plain_module(tmp_path: Path) -> None:
    plain = tmp_path / "plain.py"
    plain.write_text("VALUE = 1\n", encoding="utf-8")
    tree = _parse(plain)
    assert tree is not None
    assert not _is_executable_script(tree)


def test_the_script_detector_fires_on_a_guarded_module(tmp_path: Path) -> None:
    script = tmp_path / "script.py"
    script.write_text('if __name__ == "__main__":\n    pass\n', encoding="utf-8")
    tree = _parse(script)
    assert tree is not None
    assert _is_executable_script(tree)


def test_import_detection_covers_the_forms_this_repository_uses(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from .thresholds import value_of\n"
        "from importlib import import_module\n"
        "import_module('evals.capture')\n",
        encoding="utf-8",
    )
    tree = _parse(sample)
    assert tree is not None
    names = _imported_names(tree)
    assert {"json", "pathlib", "Path", "thresholds", "value_of", "evals", "capture"} <= names
