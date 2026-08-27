"""Offline wheel build and contents smoke verification."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


BuildRunner = Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[int, str]]


def _run(command: tuple[str, ...], cwd: Path, env: Mapping[str, str]) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        shell=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


@dataclass(frozen=True, slots=True)
class WheelSmokeReport:
    passed: bool
    wheel_name: str | None
    build_output: str
    required_members_present: bool
    entry_points_present: bool
    import_smoke_passed: bool
    blockers: tuple[str, ...]


def verify_wheel_smoke(
    project_root: Path,
    *,
    runner: BuildRunner = _run,
    uv_path: str | None = None,
) -> WheelSmokeReport:
    uv = uv_path or shutil.which("uv")
    if uv is None:
        return WheelSmokeReport(
            passed=False,
            wheel_name=None,
            build_output="uv was not found; offline wheel build was not attempted",
            required_members_present=False,
            entry_points_present=False,
            import_smoke_passed=False,
            blockers=("uv_missing",),
        )
    with tempfile.TemporaryDirectory(prefix="rigby-wheel-smoke-") as temporary:
        root = Path(temporary)
        output = root / "dist"
        output.mkdir()
        environment = dict(os.environ)
        environment.update(
            {
                "UV_OFFLINE": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        command = (uv, "build", "--wheel", "--offline", "--out-dir", str(output))
        code, build_output = runner(command, project_root, environment)
        wheels = sorted(output.glob("*.whl"))
        if code != 0 or len(wheels) != 1:
            return WheelSmokeReport(
                passed=False,
                wheel_name=wheels[0].name if len(wheels) == 1 else None,
                build_output=build_output,
                required_members_present=False,
                entry_points_present=False,
                import_smoke_passed=False,
                blockers=("offline_build_failed",),
            )
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            required = {
                "rigby_v2/__init__.py",
                "rigby_v2/app.py",
                "rigby_v2/worker.py",
                "rigby_v2/release_ops/__init__.py",
            }
            required_present = required <= names
            entry_names = [value for value in names if value.endswith(".dist-info/entry_points.txt")]
            entry_text = archive.read(entry_names[0]).decode("utf-8") if len(entry_names) == 1 else ""
            entry_points = "rigby-v2 = rigby_v2.app:run" in entry_text and (
                "rigby-v2-worker = rigby_v2.worker:run" in entry_text
            )
        smoke_env = dict(environment)
        smoke_env["PYTHONPATH"] = str(wheel)
        smoke_command = (
            sys.executable,
            "-c",
            "import rigby_v2.release_ops as release_ops; from rigby_v2.config import RuntimeSettings; print(release_ops.__file__); print(RuntimeSettings().physics_hz)",
        )
        smoke_code, smoke_output = runner(smoke_command, project_root, smoke_env)
        import_passed = (
            smoke_code == 0
            and wheel.name in smoke_output
            and smoke_output.strip().endswith("240")
        )
        blockers = []
        if not required_present:
            blockers.append("wheel_members_missing")
        if not entry_points:
            blockers.append("entry_points_missing")
        if not import_passed:
            blockers.append("wheel_import_failed")
        return WheelSmokeReport(
            passed=not blockers,
            wheel_name=wheel.name,
            build_output=build_output + "\n" + smoke_output,
            required_members_present=required_present,
            entry_points_present=entry_points,
            import_smoke_passed=import_passed,
            blockers=tuple(blockers),
        )
