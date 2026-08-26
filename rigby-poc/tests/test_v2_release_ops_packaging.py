from __future__ import annotations

from pathlib import Path

from rigby_v2.release_ops import verify_wheel_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_existing_project_builds_offline_wheel_with_v2_entry_points_and_imports() -> None:
    report = verify_wheel_smoke(ROOT)
    assert report.passed, report.build_output
    assert report.wheel_name is not None and report.wheel_name.endswith(".whl")
    assert report.required_members_present
    assert report.entry_points_present
    assert report.import_smoke_passed

