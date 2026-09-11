"""`provider_status()["dotenv_scope"]` must not depend on the folder's name.

It did. The scope was computed as ``loaded_from.parent.name == "rigby-humanoid"``,
so renaming this directory to `humanoid` made a package-local `.env` report
"workspace" instead of "poc". Every suite stayed green through it, because
nothing referenced the field at all -- the same shape as the reachability defect
`test_module_reachability` exists to catch, one level down at the value.

A diagnostic that lies is worse than one that is absent: it is consulted exactly
when someone is trying to work out which `.env` a run picked up. So the scope is
derived from the paths `load_environment` actually searches, and this test pins
that by relocating the package rather than by asserting the current folder name
-- an assertion naming `humanoid` would reintroduce the same coupling it is here
to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_poc import planner

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _provider_status_with_package_root(tmp_path: Path, monkeypatch, name: str) -> dict:
    """Run `provider_status` as if the package lived in a folder called `name`."""
    package_root = tmp_path / name
    (package_root / "src" / "rigby_poc").mkdir(parents=True)
    fake_module = package_root / "src" / "rigby_poc" / "planner.py"
    fake_module.write_text("", encoding="utf-8")
    (package_root / ".env").write_text("OPENAI_PLANNER_MODEL=from-package\n", encoding="utf-8")

    monkeypatch.setattr(planner, "__file__", str(fake_module))
    monkeypatch.setattr(planner, "load_dotenv", lambda *a, **k: None)
    return planner.provider_status()


@pytest.mark.parametrize("folder", ["humanoid", "rigby-humanoid", "anything-at-all"])
def test_package_local_env_reports_poc_scope_under_any_folder_name(
    folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = _provider_status_with_package_root(tmp_path, monkeypatch, folder)
    assert status["dotenv_loaded"] is True
    assert status["dotenv_scope"] == "poc", (
        f"a package-local .env must report 'poc' scope even when the checkout "
        f"directory is named {folder!r}"
    )


def test_a_parent_env_reports_workspace_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other branch must still be reachable, or the test above proves nothing."""
    package_root = tmp_path / "humanoid"
    (package_root / "src" / "rigby_poc").mkdir(parents=True)
    fake_module = package_root / "src" / "rigby_poc" / "planner.py"
    fake_module.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_PLANNER_MODEL=from-workspace\n", encoding="utf-8")

    monkeypatch.setattr(planner, "__file__", str(fake_module))
    monkeypatch.setattr(planner, "load_dotenv", lambda *a, **k: None)
    status = planner.provider_status()
    assert status["dotenv_loaded"] is True
    assert status["dotenv_scope"] == "workspace"
