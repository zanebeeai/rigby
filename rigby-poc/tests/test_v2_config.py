from __future__ import annotations

from pathlib import Path

import pytest

from rigby_v2.config import RuntimeSettings


def test_settings_are_local_and_reproducible(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_env(
        {
            "RIGBY_V2_PROJECT_ROOT": str(tmp_path),
            "RIGBY_V2_ARTIFACT_DIR": "artifacts",
            "RIGBY_V2_WORKER_ID": "worker-test",
            "RIGBY_V2_PHYSICS_HZ": "240",
            "RIGBY_V2_RENDER_FPS": "30",
        }
    )

    assert settings.artifact_root == (tmp_path / "artifacts").resolve()
    assert settings.steps_per_frame == 8
    snapshot = settings.reproducibility_snapshot()
    assert snapshot["runtime"]["worker_id"] == "worker-test"
    assert snapshot["dependencies"]["mujoco"] == "3.11.0"
    assert "database_url" not in str(snapshot)


def test_render_rate_must_divide_physics_rate() -> None:
    with pytest.raises(ValueError, match="integer divisor"):
        RuntimeSettings(physics_hz=240, render_fps=29)

