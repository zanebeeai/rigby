"""Offline, read-only fresh-machine preflight checks."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlsplit

from rigby_v2.config import RuntimeSettings

from .locks import audit_dependency_locks
from .models import CheckStatus, DiagnosticCheck, DiagnosticReport


CommandRunner = Callable[[tuple[str, ...]], tuple[int, str]]
TcpProbe = Callable[[str, int, float], bool]


def _run(command: tuple[str, ...]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, f"{type(error).__name__}: {error}"
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode, output[0] if output else ""


def _tcp(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class PreflightProbes:
    which: Callable[[str], str | None] = shutil.which
    run_command: CommandRunner = _run
    tcp_probe: TcpProbe = _tcp
    package_version: Callable[[str], str] = importlib.metadata.version
    python_version: tuple[int, int, int] = sys.version_info[:3]


def _tool_check(
    check_id: str,
    executable: str,
    args: tuple[str, ...],
    probes: PreflightProbes,
) -> DiagnosticCheck:
    path = probes.which(executable)
    if path is None:
        return DiagnosticCheck(check_id, CheckStatus.FAIL, f"{executable} is not installed")
    code, output = probes.run_command((path, *args))
    return DiagnosticCheck(
        check_id,
        CheckStatus.PASS if code == 0 else CheckStatus.FAIL,
        f"{executable} responded" if code == 0 else f"{executable} probe failed",
        details={"path": path, "version": output},
    )


def _database_check(
    settings: RuntimeSettings,
    project_root: Path,
    probes: PreflightProbes,
) -> DiagnosticCheck:
    parsed = urlsplit(settings.database_url)
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
        return DiagnosticCheck(
            "postgres_runtime", CheckStatus.FAIL, "RIGBY_V2_DATABASE_URL is not a PostgreSQL URL"
        )
    port = parsed.port or 5432
    reachable = probes.tcp_probe(parsed.hostname, port, 0.5)
    docker = probes.which("docker")
    docker_ready = False
    docker_output = ""
    if docker is not None:
        code, docker_output = probes.run_command(
            (docker, "version", "--format", "{{.Server.Version}}")
        )
        docker_ready = code == 0 and (project_root / "compose.v2.yaml").is_file()
    passed = reachable or docker_ready
    return DiagnosticCheck(
        "postgres_runtime",
        CheckStatus.PASS if passed else CheckStatus.FAIL,
        "PostgreSQL is reachable" if reachable else (
            "Docker and the pinned local PostgreSQL compose file are available"
            if docker_ready
            else "neither configured PostgreSQL nor the local Docker runtime is ready"
        ),
        details={
            "host": parsed.hostname,
            "port": port,
            "reachable": reachable,
            "docker_ready": docker_ready,
            "docker_server": docker_output,
        },
    )


def run_fresh_machine_preflight(
    *,
    project_root: Path,
    env: Mapping[str, str] | None = None,
    probes: PreflightProbes | None = None,
) -> DiagnosticReport:
    """Inspect a machine without installing, downloading, starting, or writing."""

    probe = probes or PreflightProbes()
    values = dict(os.environ if env is None else env)
    values.setdefault("RIGBY_V2_PROJECT_ROOT", str(project_root))
    checks: list[DiagnosticCheck] = []
    python_ok = probe.python_version[:2] == (3, 12)
    checks.append(
        DiagnosticCheck(
            "python_3_12",
            CheckStatus.PASS if python_ok else CheckStatus.FAIL,
            "Python 3.12 is active" if python_ok else "Rigby v2 requires Python 3.12",
            details={"version": ".".join(map(str, probe.python_version))},
        )
    )
    try:
        mujoco_version = probe.package_version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        mujoco_version = None
    checks.append(
        DiagnosticCheck(
            "mujoco_3_11",
            CheckStatus.PASS if mujoco_version == "3.11.0" else CheckStatus.FAIL,
            "MuJoCo 3.11.0 is installed" if mujoco_version == "3.11.0" else "MuJoCo 3.11.0 is not installed exactly",
            details={"version": mujoco_version},
        )
    )
    checks.append(_tool_check("ffmpeg", "ffmpeg", ("-version",), probe))
    try:
        settings = RuntimeSettings.from_env(values)
        config_error = None
    except (ValueError, OSError) as error:
        settings = None
        config_error = str(error)
    checks.append(
        DiagnosticCheck(
            "runtime_config",
            CheckStatus.PASS if settings is not None else CheckStatus.FAIL,
            "runtime configuration validates" if settings is not None else "runtime configuration is invalid",
            details={"error": config_error},
        )
    )
    if settings is not None:
        checks.append(_database_check(settings, project_root, probe))
        artifact_parent = (
            settings.artifact_root if settings.artifact_root.exists() else settings.artifact_root.parent
        )
        checks.append(
            DiagnosticCheck(
                "artifact_location",
                CheckStatus.PASS
                if artifact_parent.is_dir() and os.access(artifact_parent, os.R_OK | os.W_OK)
                else CheckStatus.FAIL,
                "artifact location exists and is accessible"
                if artifact_parent.is_dir() and os.access(artifact_parent, os.R_OK | os.W_OK)
                else "artifact location parent is unavailable",
                details={"path": str(settings.artifact_root)},
            )
        )
    required_assets = (
        "assets/v2/canonical_human.xml",
        "assets/v2/rig_manifest.json",
        "assets/models/human-male.glb",
        "assets/v2/object_packs/grasp_place_block.json",
        "assets/v2/object_packs/drawer.json",
        "assets/v2/object_packs/lever_button.json",
        "assets/v2/object_packs/hand_tool.json",
        "assets/v2/object_packs/container_lid.json",
        "assets/v2/object_packs/two_handed_object.json",
        "assets/v2/datasets/registry.json",
        "compose.v2.yaml",
        "db/init/001_rigby_v2.sql",
    )
    missing = [value for value in required_assets if not (project_root / value).is_file()]
    checks.append(
        DiagnosticCheck(
            "required_assets",
            CheckStatus.PASS if not missing else CheckStatus.FAIL,
            "canonical rig, scene packs, dataset policies, and database schema are present"
            if not missing
            else "required release assets are missing",
            details={"missing": missing, "checked": len(required_assets)},
        )
    )
    lock_report = audit_dependency_locks(project_root)
    checks.extend(lock_report.checks)
    checks.append(
        DiagnosticCheck(
            "offline_read_only",
            CheckStatus.PASS,
            "preflight made no downloads, service starts, or filesystem writes",
        )
    )
    return DiagnosticReport(
        kind="fresh_machine_preflight",
        checks=tuple(checks),
        offline=True,
        read_only=True,
    )
