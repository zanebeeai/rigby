from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_pipeline_packages_ship_in_the_wheel() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

    assert {"src/rigby_poc", "evals"}.issubset(packages)
