from __future__ import annotations

import numpy as np

from rigby_v2.acceptance import (
    PACK_GEOMETRY_CHANGES,
    certify_button_press,
    neutral_pack_diagnostic,
    run_bounded_drawer_trial,
    run_button_press_once,
)
from rigby_v2.certification import CertificationOutcome
import pytest

pytestmark = pytest.mark.medium


def test_repaired_articulated_packs_have_no_neutral_false_success() -> None:
    assert {item.pack_id for item in PACK_GEOMETRY_CHANGES} == {
        "drawer",
        "lever_button",
    }
    for pack_id in ("drawer", "lever_button"):
        diagnostic = neutral_pack_diagnostic(pack_id)
        assert not diagnostic.predicate_satisfied
        assert diagnostic.object_object_contact_count == 0
        assert diagnostic.maximum_object_object_force_n == 0.0
        assert diagnostic.maximum_object_object_penetration_m == 0.0
        assert max(diagnostic.joint_excursions.values()) == 0.0


def test_button_press_is_real_repeatable_contact() -> None:
    runs = tuple(run_button_press_once() for _ in range(3))
    assert all(item.predicate_satisfied for item in runs)
    assert all(item.terminal_object_state <= -0.02 for item in runs)
    assert all(item.contact.sample_count > 0 for item in runs)
    assert all(item.contact.maximum_penetration_m <= 0.002 for item in runs)
    assert all(item.qpos_writes_after_initialization == 0 for item in runs)
    assert all(item.object_actuator_count == 0 for item in runs)
    assert all(item.object_target_excursion == 0.0 for item in runs)
    assert len({item.trace_sha256 for item in runs}) == 1
    assert all(
        any("button_cap" in first_name + second_name for first_name, second_name in item.contact.geom_pairs)
        for item in runs
    )
    assert np.array_equal(
        runs[0].simulation.trace.qpos, runs[1].simulation.trace.qpos
    )


def test_button_press_runs_full_three_repeat_and_export_attempt(tmp_path) -> None:
    attempt = certify_button_press(tmp_path / "artifacts")
    assert len(attempt.certification.simulation_runs) == 3
    assert all(run.completed for run in attempt.certification.simulation_runs)
    assert attempt.raw_evidence.predicate_satisfied
    assert all(item.passed for item in attempt.certification.predicate_evaluations)
    assert attempt.certification.outcome is CertificationOutcome.CERTIFIED
    assert attempt.certification.violations == ()
    assert attempt.robust
    assert len(attempt.robustness_variations) == 5
    assert all(
        result.outcome is CertificationOutcome.CERTIFIED
        for _, result in attempt.robustness_variations
    )


def test_drawer_bounded_trial_is_recorded_as_measured_failure() -> None:
    result = run_bounded_drawer_trial()
    assert not result.predicate_satisfied
    assert result.terminal_object_state < 0.22
    assert result.maximum_object_state < 0.22
    assert result.contact.sample_count > 0
    assert result.contact.maximum_penetration_m > 0.002
    assert result.object_actuator_count == 0
    assert result.object_target_excursion == 0.0
    assert result.qpos_writes_after_initialization == 0
