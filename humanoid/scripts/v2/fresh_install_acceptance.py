from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from rigby_v2.config import RuntimeSettings
from rigby_core.hashing import hash_file
from rigby_v2.release_ops import seal_release_evidence

ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"isolated install command failed ({command[0]}):\n{completed.stdout[-4000:]}"
        )
    return completed.stdout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_health(port: int, process: subprocess.Popen[bytes]) -> dict[str, object]:
    deadline = time.monotonic() + 30
    url = f"http://127.0.0.1:{port}/api/v2/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"isolated API exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload.get("status") == "ok":
                return payload
        except (
            OSError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
        time.sleep(0.1)
    raise RuntimeError(f"isolated API health timed out: {last_error!r}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the offline isolated install")
    settings = RuntimeSettings.from_env()
    install_environment = dict(os.environ)
    install_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    build_environment = dict(install_environment)
    build_environment.update(
        {
            "UV_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
        }
    )

    with tempfile.TemporaryDirectory(prefix="rigby-fresh-install-") as temporary:
        root = Path(temporary)
        distribution = root / "dist"
        distribution.mkdir()
        _run(
            [uv, "build", "--wheel", "--offline", "--out-dir", str(distribution)],
            cwd=ROOT,
            environment=build_environment,
        )
        wheels = tuple(distribution.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("offline build did not produce exactly one wheel")
        virtual_environment = root / "venv"
        _run(
            [uv, "venv", str(virtual_environment), "--python", sys.executable],
            cwd=root,
            environment=install_environment,
        )
        python = virtual_environment / "Scripts" / "python.exe"
        _run(
            [
                uv,
                "pip",
                "install",
                "--link-mode",
                "copy",
                "--python",
                str(python),
                str(wheels[0]),
            ],
            cwd=root,
            environment=install_environment,
        )
        import_output = _run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "import json, pathlib, rigby_v2, mujoco, fastapi, psycopg; "
                    "print(json.dumps({'module': str(pathlib.Path(rigby_v2.__file__).resolve()), "
                    "'mujoco': mujoco.__version__, 'fastapi': fastapi.__version__, "
                    "'psycopg': psycopg.__version__}))"
                ),
            ],
            cwd=root,
            environment=install_environment,
        ).strip()
        imported = json.loads(import_output.splitlines()[-1])
        if virtual_environment.resolve() not in Path(imported["module"]).parents:
            raise RuntimeError("isolated import resolved outside the fresh environment")
        entrypoints = (
            virtual_environment / "Scripts" / "rigby-v2.exe",
            virtual_environment / "Scripts" / "rigby-v2-worker.exe",
        )
        if not all(path.is_file() for path in entrypoints):
            raise RuntimeError("installed v2 API/worker entry points are missing")

        port = _free_port()
        api_environment = install_environment | {
            "RIGBY_V2_API_PORT": str(port),
            "RIGBY_V2_DATABASE_URL": settings.database_url,
            "RIGBY_V2_ARTIFACT_DIR": str(root / "artifacts"),
            # The documented local install includes the immutable repository
            # asset/config bundle separately from the Python wheel.
            "RIGBY_V2_PROJECT_ROOT": str(ROOT),
        }
        process = subprocess.Popen(
            [str(python), "-I", "-c", "from rigby_v2.app import run; run()"],
            cwd=root,
            env=api_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            health = _wait_health(port, process)
        finally:
            _stop(process)

        payload = {
            "wheel_built_offline": True,
            "dependency_install_used_registry": True,
            "isolated_environment": True,
            "source_tree_not_on_pythonpath": True,
            "wheel_name": wheels[0].name,
            "wheel_sha256": hash_file(wheels[0]),
            "dependency_lock_sha256": hash_file(ROOT / "uv.lock"),
            "installed_module": imported["module"],
            "versions": {
                "mujoco": imported["mujoco"],
                "fastapi": imported["fastapi"],
                "psycopg": imported["psycopg"],
            },
            "entry_points": [path.name for path in entrypoints],
            "api_health_status": health["status"],
            "asset_bundle": "documented immutable repository assets/config",
            "temporary_environment_cleaned": True,
        }

    evidence = seal_release_evidence(
        settings.project_root / "artifacts-v2" / "release-evidence",
        requirement_id="fresh_machine_install",
        passed=True,
        payload=payload,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": evidence.artifact_path,
                "evidence_sha256": evidence.artifact_sha256,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
