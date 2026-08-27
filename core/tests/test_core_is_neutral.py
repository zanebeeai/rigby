"""core/ must not import the product stacks built on top of it.

The whole point of extracting this tier was that ``rigby_general`` reached its
five neutral modules by depending on the entire humanoid distribution -- which
dragged playwright, fastapi, openai and a 30k-line rig stack behind them. That
is a structural property, not a style preference, and structural properties that
nothing asserts come back. A single ``from rigby_v2 import ...`` added here in
six months would silently restore the old coupling and every suite would stay
green, because both packages are installed in the same environment.

So: this test reads the AST, not the import machinery. An import guarded by
``TYPE_CHECKING`` or hidden inside a function body still counts -- those are
exactly the forms someone reaches for when they know the dependency is wrong.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "rigby_core"
FORBIDDEN = ("rigby_v2", "rigby_poc", "rigby_general")


def _sources() -> list[Path]:
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_the_package_root_is_where_we_think_it_is() -> None:
    """A path typo would make every assertion below vacuously true."""
    assert PACKAGE_ROOT.is_dir(), PACKAGE_ROOT
    assert _sources(), f"no sources under {PACKAGE_ROOT}"


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_module_imports_a_downstream_package(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            if name and name.split(".")[0] in FORBIDDEN:
                offenders.append(f"line {node.lineno}: {name}")
    assert not offenders, (
        f"{path.name} imports a package that depends on core:\n  "
        + "\n  ".join(offenders)
        + "\n\ncore/ is the bottom tier. If it needs something from a product "
        "stack, that something belongs in core/, not the import."
    )
