"""Local runtime configuration and base-tree identity.

Neither v2 tree in this repository is tracked by git, so a path dependency on
``rigby-poc`` has no commit to name. ``base_tree_fingerprint`` supplies the
missing identity by hashing the source it actually imported. Every baked
primitive and every audit report records it, so a library can always be traced to
the exact compiler, solver, and gate code that produced it.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_URL = "postgresql://rigby:rigby-local-only@127.0.0.1:54330/rigby_general"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BaseTreeIdentity:
    """A content fingerprint of the ``rigby_core`` source this process imported."""

    package: str
    root: Path
    file_count: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "file_count": self.file_count,
            "sha256": self.sha256,
        }


@lru_cache(maxsize=1)
def base_tree_fingerprint() -> BaseTreeIdentity:
    """Hash every ``.py`` file of the imported ``rigby_core`` package.

    The digest covers sorted ``(relative path, file digest)`` pairs, so it is
    stable across machines and changes if any source file, name, or path does.
    """

    import rigby_core

    module_file = getattr(rigby_core, "__file__", None)
    if module_file is None:  # pragma: no cover - namespace package
        raise RuntimeError("rigby_core does not expose a filesystem location")
    root = Path(module_file).resolve().parent

    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        count += 1
    if count == 0:  # pragma: no cover - defensive
        raise RuntimeError(f"no rigby_core sources found under {root}")

    return BaseTreeIdentity(
        package="rigby_core",
        root=root,
        file_count=count,
        sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class GeneralSettings:
    """Immutable local configuration shared by the API and the worker.

    Secrets are excluded from :meth:`reproducibility_snapshot` by construction --
    they are never stored on the dataclass in the first place.
    """

    project_root: Path = PROJECT_ROOT
    artifact_root: Path = PROJECT_ROOT / "artifacts-general"
    robot_root: Path = PROJECT_ROOT / "robots"
    database_url: str = DEFAULT_DATABASE_URL
    worker_id: str = "local-worker-1"
    api_port: int = 8020
    physics_hz: int = 240
    render_fps: int = 30
    lease_seconds: int = 60
    max_attempts: int = 3
    bake_budget_seconds: int = 1_200
    bake_max_attempts: int = 500

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.render_fps <= 0:
            raise ValueError("physics_hz and render_fps must be positive")
        if self.physics_hz % self.render_fps:
            raise ValueError("render_fps must be an integer divisor of physics_hz")
        if self.lease_seconds < 5:
            raise ValueError("lease_seconds must be at least 5")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("api_port must be a valid TCP port")
        if self.bake_budget_seconds <= 0:
            raise ValueError("bake_budget_seconds must be positive")
        if self.bake_max_attempts <= 0:
            raise ValueError("bake_max_attempts must be positive")
        if not self.worker_id.strip():
            raise ValueError("worker_id cannot be empty")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GeneralSettings":
        values = os.environ if env is None else env
        project_root = Path(
            values.get("RIGBY_GENERAL_PROJECT_ROOT", PROJECT_ROOT)
        ).resolve()

        def _rooted(key: str, default: str) -> Path:
            candidate = Path(values.get(key, default))
            if not candidate.is_absolute():
                candidate = project_root / candidate
            return candidate.resolve()

        return cls(
            project_root=project_root,
            artifact_root=_rooted("RIGBY_GENERAL_ARTIFACT_DIR", "artifacts-general"),
            robot_root=_rooted("RIGBY_GENERAL_ROBOT_DIR", "robots"),
            database_url=values.get("RIGBY_GENERAL_DATABASE_URL", DEFAULT_DATABASE_URL),
            worker_id=values.get("RIGBY_GENERAL_WORKER_ID", "local-worker-1"),
            api_port=int(values.get("RIGBY_GENERAL_API_PORT", "8020")),
            physics_hz=int(values.get("RIGBY_GENERAL_PHYSICS_HZ", "240")),
            render_fps=int(values.get("RIGBY_GENERAL_RENDER_FPS", "30")),
            lease_seconds=int(values.get("RIGBY_GENERAL_LEASE_SECONDS", "60")),
            max_attempts=int(values.get("RIGBY_GENERAL_MAX_ATTEMPTS", "3")),
            bake_budget_seconds=int(
                values.get("RIGBY_GENERAL_BAKE_BUDGET_SECONDS", "1200")
            ),
            bake_max_attempts=int(values.get("RIGBY_GENERAL_BAKE_MAX_ATTEMPTS", "500")),
        )

    @property
    def steps_per_frame(self) -> int:
        return self.physics_hz // self.render_fps

    def reproducibility_snapshot(self) -> dict[str, object]:
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "executable_bits": 64 if sys.maxsize > 2**32 else 32,
            },
            "dependencies": {
                name: _package_version(name)
                for name in ("mujoco", "numpy", "scipy", "pydantic", "fastapi")
            },
            "base_tree": base_tree_fingerprint().as_dict(),
            "runtime": {
                "physics_hz": self.physics_hz,
                "render_fps": self.render_fps,
                "steps_per_frame": self.steps_per_frame,
                "lease_seconds": self.lease_seconds,
                "max_attempts": self.max_attempts,
                "bake_budget_seconds": self.bake_budget_seconds,
                "bake_max_attempts": self.bake_max_attempts,
                "worker_id": self.worker_id,
            },
        }
