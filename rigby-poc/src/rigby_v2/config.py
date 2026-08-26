from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Immutable local runtime configuration.

    Secrets are intentionally excluded from :meth:`reproducibility_snapshot`.
    The same settings object is shared by the API and worker entrypoints so a
    submitted job records the exact timing and lease policy under which it ran.
    """

    project_root: Path = PROJECT_ROOT
    artifact_root: Path = PROJECT_ROOT / "artifacts-v2"
    database_url: str = "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby"
    worker_id: str = "local-worker-1"
    physics_hz: int = 240
    render_fps: int = 30
    lease_seconds: int = 60
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.render_fps <= 0:
            raise ValueError("physics_hz and render_fps must be positive")
        if self.physics_hz % self.render_fps:
            raise ValueError("render_fps must be an integer divisor of physics_hz")
        if self.lease_seconds < 5:
            raise ValueError("lease_seconds must be at least 5")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not self.worker_id.strip():
            raise ValueError("worker_id cannot be empty")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeSettings":
        values = os.environ if env is None else env
        project_root = Path(values.get("RIGBY_V2_PROJECT_ROOT", PROJECT_ROOT)).resolve()
        artifact_value = values.get("RIGBY_V2_ARTIFACT_DIR", "artifacts-v2")
        artifact_root = Path(artifact_value)
        if not artifact_root.is_absolute():
            artifact_root = project_root / artifact_root
        return cls(
            project_root=project_root,
            artifact_root=artifact_root.resolve(),
            database_url=values.get(
                "RIGBY_V2_DATABASE_URL",
                "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby",
            ),
            worker_id=values.get("RIGBY_V2_WORKER_ID", "local-worker-1"),
            physics_hz=int(values.get("RIGBY_V2_PHYSICS_HZ", "240")),
            render_fps=int(values.get("RIGBY_V2_RENDER_FPS", "30")),
            lease_seconds=int(values.get("RIGBY_V2_LEASE_SECONDS", "60")),
            max_attempts=int(values.get("RIGBY_V2_MAX_ATTEMPTS", "3")),
        )

    @property
    def steps_per_frame(self) -> int:
        return self.physics_hz // self.render_fps

    def reproducibility_snapshot(self) -> dict[str, object]:
        lock_path = self.project_root / "uv.lock"
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
            "dependency_lock": {
                "path": lock_path.name,
                "sha256": _sha256_file(lock_path),
            },
            "runtime": {
                "physics_hz": self.physics_hz,
                "render_fps": self.render_fps,
                "steps_per_frame": self.steps_per_frame,
                "lease_seconds": self.lease_seconds,
                "max_attempts": self.max_attempts,
                "worker_id": self.worker_id,
            },
        }

