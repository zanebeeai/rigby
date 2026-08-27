from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rigby_v2.acceptance.production_drawer import (
    build_production_drawer_case,
    certify_production_drawer,
    run_production_drawer_once,
)
from rigby_v2.certification import CertificationOutcome
from rigby_v2.hashing import hash_file

pytestmark = pytest.mark.medium


ROOT = Path(__file__).resolve().parents[1]
FAILURE_EVIDENCE = ROOT / "assets" / "v2" / "acceptance" / "drawer_failure_attempts.v1.json"


def test_production_drawer_opens_by_measured_contact_without_hidden_control(tmp_path) -> None:
    case = build_production_drawer_case(tmp_path / "artifacts")
    _, diagnostics = run_production_drawer_once(case)

    assert diagnostics.production_compile_succeeded
    assert diagnostics.maximum_travel_m >= 0.22
    assert diagnostics.terminal_travel_m >= 0.22
    assert diagnostics.contact_samples > 0
    assert diagnostics.first_contact_s is not None
    assert diagnostics.last_contact_s is not None
    assert diagnostics.maximum_global_penetration_m <= 0.002
    assert diagnostics.object_actuator_count == 0
    assert diagnostics.object_target_excursion_m == 0.0
    assert diagnostics.qpos_writes_after_initialization == 0


@pytest.mark.skipif(
    os.environ.get("RIGBY_TEST_LONG_ACCEPTANCE") != "1",
    reason="three-repeat plus five-variation drawer acceptance is opt-in",
)
def test_production_drawer_is_repeatable_exportable_and_robust(tmp_path) -> None:
    result = certify_production_drawer(
        build_production_drawer_case(tmp_path / "artifacts")
    )

    assert result.baseline.outcome is CertificationOutcome.CERTIFIED
    assert len(result.baseline.simulation_runs) == 3
    assert result.baseline.violations == ()
    assert result.robust
    assert result.robustness is not None
    assert len(result.robustness.variations) == 5
    assert all(
        certification.outcome is CertificationOutcome.CERTIFIED
        for _, certification in result.robustness.variations
    )


def test_historical_drawer_failure_evidence_remains_hash_bound() -> None:
    evidence = json.loads(FAILURE_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "rigby.drawer_bounded_failure.v1"
    assert evidence["status"] == "frozen_failure"
    assert evidence["attempt_limit"] == 2
    assert len(evidence["attempts"]) == 2
    assert evidence["gate_contract"] == {
        "drawer_open_threshold_m": 0.22,
        "maximum_penetration_m": 0.002,
        "object_actuation_allowed": False,
        "post_initialization_qpos_writes_allowed": False,
    }
    assert evidence["certification"] == {
        "baseline_certified": False,
        "export_reimport_attempted": False,
        "robustness_variations_attempted": 0,
        "three_repeat_certification_attempted": False,
    }
    digest_path = FAILURE_EVIDENCE.with_suffix(FAILURE_EVIDENCE.suffix + ".sha256")
    assert digest_path.read_text(encoding="utf-8").strip() == hash_file(FAILURE_EVIDENCE)
