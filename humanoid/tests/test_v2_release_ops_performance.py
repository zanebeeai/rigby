from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rigby_v2.release_ops import (
    REFERENCE_SPECS,
    CandidateBenchmarkSpec,
    run_local_reference_benchmark,
    run_performance_harness,
)

pytestmark = pytest.mark.fast


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_injectable_harness_calculates_phase_totals_and_interpolated_p95() -> None:
    clock = ManualClock()
    specs = tuple(
        CandidateBenchmarkSpec(f"candidate-{index}", f"pack-{index}")
        for index in range(1, 6)
    )

    def workload(spec: CandidateBenchmarkSpec, timer: object) -> None:
        index = int(spec.candidate_id.rsplit("-", 1)[1])
        timer.measure("compile", lambda: clock.advance(index * 0.2))  # type: ignore[attr-defined]
        timer.measure("simulation", lambda: clock.advance(index * 0.8))  # type: ignore[attr-defined]
        timer.measure("evidence", lambda: clock.advance(index * 0.1))  # type: ignore[attr-defined]

    report = run_performance_harness(specs, workload, clock=clock)
    assert [candidate.total_s for candidate in report.candidates] == pytest.approx(
        [1.1, 2.2, 3.3, 4.4, 5.5]
    )
    assert report.p95_candidate_s == pytest.approx(5.28)
    assert report.total_wall_s == pytest.approx(16.5)
    assert report.vlm_included is False
    assert report.passed
    assert report.to_dict()["p95_candidate_s"] == pytest.approx(5.28)


def test_harness_refuses_wrong_candidate_count_or_duration() -> None:
    def noop(_spec: CandidateBenchmarkSpec, timer: object) -> None:
        timer.measure("simulation", lambda: None)  # type: ignore[attr-defined]

    try:
        run_performance_harness(REFERENCE_SPECS[:4], noop)
    except ValueError as error:
        assert "exactly five" in str(error)
    else:
        raise AssertionError("four candidates must be rejected")
    bad = list(REFERENCE_SPECS)
    bad[0] = CandidateBenchmarkSpec("candidate-1", "pack", 9.0)
    try:
        run_performance_harness(tuple(bad), noop)
    except ValueError as error:
        assert "exactly 10 seconds" in str(error)
    else:
        raise AssertionError("non-ten-second candidate must be rejected")


def test_local_reference_benchmark_runs_five_real_ten_second_mujoco_candidates() -> None:
    report = run_local_reference_benchmark()
    assert len(report.candidates) == 5
    assert report.p95_candidate_s <= 90.0
    assert report.passed
    assert report.to_dict()["passed"] is True
    required_phases = {
        "scene_compile",
        "trajectory_prepare",
        "native_simulation",
        "evidence_generation",
        "artifact_finalize",
    }
    assert all(set(candidate.phases_s) == required_phases for candidate in report.candidates)


def test_sealed_production_renderer_measurement_passes_reference_laptop_gate() -> None:
    root = Path(__file__).resolve().parents[1] / "assets" / "v2" / "benchmark"
    report_path = root / "production_renderer_performance.json"
    expected = (root / "production_renderer_performance.json.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == expected
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["vlm_included"] is False
    assert report["workload_kind"] == "articulated_keyframes_three_camera_h264"
    assert report["p95_candidate_s"] <= 90.0
    assert len(report["candidates"]) == 5
    assert all(item["duration_s"] == 10.0 for item in report["candidates"])
    required = {
        "scene_compile",
        "trajectory_prepare",
        "native_simulation",
        "evidence_generation",
        "artifact_finalize",
    }
    assert all(set(item["phases_s"]) == required for item in report["candidates"])
