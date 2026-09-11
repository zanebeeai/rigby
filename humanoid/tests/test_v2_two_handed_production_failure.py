from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_core.hashing import hash_file
from rigby_v2.scenes import load_object_pack

pytestmark = pytest.mark.fast


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "assets" / "v2" / "acceptance" / "two_handed_object_failure_attempts.v1.json"


def test_calibrated_forward_handles_remain_symmetric_realistic_and_declared() -> None:
    pack = load_object_pack("two_handed_object")
    object_spec = pack.objects[0]
    parts = {part.part_id: part for part in object_spec.parts}
    left = parts["left_forward_handle"]
    right = parts["right_forward_handle"]

    assert left.pos == pytest.approx((0.42, 0.24, 0.0))
    assert right.pos == pytest.approx((-0.42, 0.24, 0.0))
    assert left.mass_kg == right.mass_kg == pytest.approx(0.05)
    assert left.geoms[0].size == pytest.approx((0.015, 0.12))
    assert right.geoms[0].size == pytest.approx((0.015, 0.12))
    assert {site.name for site in object_spec.sites} >= {
        "left_forward_grasp",
        "right_forward_grasp",
    }


def test_two_attempt_ik_failure_is_sealed_without_false_certification() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "frozen_failure"
    assert evidence["attempt_limit"] == len(evidence["attempts"]) == 2
    assert all(item["outcome"] == "compile_infeasible" for item in evidence["attempts"])
    assert evidence["certification"] == {
        "baseline_simulation_attempted": False,
        "export_reimport_attempted": False,
        "robustness_variations_attempted": 0,
        "three_repeat_certification_attempted": False,
    }
    assert evidence["gate_contract"]["minimum_lift_m"] == 0.18
    assert evidence["gate_contract"]["maximum_penetration_m"] == 0.002
    assert evidence["model_invariants"] == {
        "equality_constraint_count": 0,
        "mocap_body_count": 0,
        "object_actuator_count": 0,
        "object_free_joint_count": 1,
    }
    digest = EVIDENCE.with_suffix(EVIDENCE.suffix + ".sha256")
    assert digest.read_text(encoding="utf-8").strip() == hash_file(EVIDENCE)
