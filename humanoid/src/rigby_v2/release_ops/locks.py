"""Read-only dependency and build-lock auditing."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

from .models import CheckStatus, DiagnosticCheck, DiagnosticReport


_EXACT_DEPENDENCY = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^]]+\])?==(?P<version>[^;\s]+)"
)


from rigby_v2.config import resolve_lock_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dependency_locks(project_root: Path) -> DiagnosticReport:
    pyproject_path = project_root / "pyproject.toml"
    lock_path = resolve_lock_path(project_root)
    checks: list[DiagnosticCheck] = []
    if not pyproject_path.is_file() or not lock_path.is_file():
        missing = [str(path.name) for path in (pyproject_path, lock_path) if not path.is_file()]
        return DiagnosticReport(
            kind="dependency_lock_audit",
            checks=(
                DiagnosticCheck(
                    "lock_files_present",
                    CheckStatus.FAIL,
                    "required dependency metadata is missing",
                    details={"missing": missing},
                ),
            ),
        )
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    checks.append(
        DiagnosticCheck(
            "lock_files_present",
            CheckStatus.PASS,
            "pyproject.toml and uv.lock are present",
            details={"uv_lock_sha256": _sha256(lock_path)},
        )
    )

    requires_python = str(pyproject.get("project", {}).get("requires-python", ""))
    lock_python = str(lock.get("requires-python", ""))
    python_ok = requires_python == ">=3.12,<3.13" and lock_python == "==3.12.*"
    checks.append(
        DiagnosticCheck(
            "python_lock",
            CheckStatus.PASS if python_ok else CheckStatus.FAIL,
            "Python runtime range and lock agree" if python_ok else "Python runtime range is not release-locked to 3.12",
            details={"project": requires_python, "lock": lock_python},
        )
    )

    dependencies = pyproject.get("project", {}).get("dependencies", [])
    parsed: dict[str, str] = {}
    unpinned: list[str] = []
    for requirement in dependencies:
        match = _EXACT_DEPENDENCY.match(str(requirement))
        if match is None:
            unpinned.append(str(requirement))
        else:
            parsed[match.group("name").lower().replace("_", "-")] = match.group("version")
    checks.append(
        DiagnosticCheck(
            "runtime_dependencies_exact",
            CheckStatus.PASS if not unpinned else CheckStatus.FAIL,
            "all direct runtime dependencies use exact pins" if not unpinned else "direct runtime dependencies are not exactly pinned",
            details={"unpinned": unpinned, "count": len(dependencies)},
        )
    )
    locked_versions = {
        str(package["name"]).lower().replace("_", "-"): str(package["version"])
        for package in lock.get("package", [])
        if "name" in package and "version" in package
    }
    mismatches = {
        name: {"required": version, "locked": locked_versions.get(name)}
        for name, version in parsed.items()
        if locked_versions.get(name) != version
    }
    checks.append(
        DiagnosticCheck(
            "runtime_lock_matches",
            CheckStatus.PASS if not mismatches else CheckStatus.FAIL,
            "direct pins exactly match uv.lock" if not mismatches else "direct pins and uv.lock disagree",
            details={"mismatches": mismatches},
        )
    )

    build_requires = [str(value) for value in pyproject.get("build-system", {}).get("requires", [])]
    build_unpinned = [value for value in build_requires if _EXACT_DEPENDENCY.match(value) is None]
    checks.append(
        DiagnosticCheck(
            "build_backend_exact",
            CheckStatus.PASS if not build_unpinned else CheckStatus.FAIL,
            "build backend requirements use exact pins" if not build_unpinned else "build backend requirements are not exactly pinned",
            details={
                "requirements": build_requires,
                "unpinned": build_unpinned,
                "remediation": "pin hatchling exactly and regenerate uv.lock/build constraints",
            },
        )
    )
    return DiagnosticReport(kind="dependency_lock_audit", checks=tuple(checks))

