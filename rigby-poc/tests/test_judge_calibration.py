from __future__ import annotations

from pathlib import Path

from evals.calibrate_judge import (
    _obvious_comparisons,
    _ordered_unary_records,
    calibration_summary,
    structural_prefilter_record,
)


def test_calibration_thresholds_require_all_three_metrics() -> None:
    unary = [
        {
            "accepted": False,
            "corruption": {
                "kind": (
                    "wrist_rotation"
                    if index < 12
                    else "wrong_joint_shake"
                    if index < 16
                    else "timing"
                    if index >= 24
                    else "fist_shape"
                )
            },
        }
        for index in range(28)
    ]
    pairwise = [{"base_won": True, "order_consistent": True} for _ in range(10)]
    summary = calibration_summary(unary, pairwise)
    assert summary["false_acceptance_rate"] == 0.0
    assert summary["obvious_pairwise_agreement"] == 1.0
    assert summary["ab_order_consistency"] == 1.0
    assert summary["passed"] is True

    unary[0]["accepted"] = True
    unary[1]["accepted"] = True
    assert calibration_summary(unary, pairwise)["passed"] is False
    unary[0]["accepted"] = False
    unary[1]["accepted"] = False
    pairwise[0]["order_consistent"] = False
    assert calibration_summary(unary, pairwise)["passed"] is True
    pairwise[1]["order_consistent"] = False
    assert calibration_summary(unary, pairwise)["passed"] is False


def test_false_acceptance_is_the_hybrid_acceptance_decision() -> None:
    unary = [
        {
            "vlm_accepted": True,
            "structural_valid": False,
            "accepted": False,
            "corruption": {
                "kind": (
                    "wrist_rotation"
                    if index < 12
                    else "wrong_joint_shake"
                    if index < 16
                    else "timing"
                    if index >= 24
                    else "fist_shape"
                )
            },
        }
        for index in range(28)
    ]
    pairwise = [{"base_won": True, "order_consistent": True} for _ in range(10)]
    summary = calibration_summary(unary, pairwise)
    assert summary["false_accepts"] == 0
    assert summary["passed"] is True


def test_structural_prefilter_records_a_zero_cost_hybrid_rejection() -> None:
    record = structural_prefilter_record(
        {
            "corruption": {"id": "wrist-flex", "kind": "wrist_rotation"},
            "result_id": "corrupt-result",
            "structural_valid": False,
            "structural_failures": ["wrist swing exceeds calibrated limit"],
        }
    )
    assert record["accepted"] is False
    assert record["vlm_accepted"] is None
    assert record["rejection_source"] == "deterministic_structural_prefilter"
    assert record["structural_failures"] == ["wrist swing exceeds calibrated limit"]


def test_calibration_cannot_pass_without_timing_corruptions() -> None:
    unary = [
        {"accepted": False, "corruption": {"kind": "wrist_rotation"}}
        for _ in range(28)
    ]
    pairwise = [{"base_won": True, "order_consistent": True} for _ in range(10)]
    summary = calibration_summary(unary, pairwise)
    assert summary["completed_timing_corruptions"] == 0
    assert summary["passed"] is False


def test_obvious_pairs_stratify_timing_fingers_and_wrists() -> None:
    available = []
    for index in range(4):
        available.append(
            ({"corruption": {"id": f"timing-{index}", "kind": "timing"}}, Path(f"t{index}"))
        )
    for kind in ("fist_shape", "open_middle_fingers"):
        for index, angle in enumerate((-0.24, -0.1, 0.1, 0.24)):
            available.append(
                (
                    {"corruption": {"id": f"{kind}-{index}", "kind": kind, "angle_rad": angle}},
                    Path(f"f-{kind}-{index}"),
                )
            )
    for index, angle in enumerate((-1.25, -0.85, 0.85, 1.25)):
        available.append(
            (
                {
                    "corruption": {
                        "id": f"wrist-{index}",
                        "kind": "wrist_rotation",
                        "angle_rad": angle,
                    }
                },
                Path(f"w{index}"),
            )
        )
    for index in range(4):
        available.append(
            (
                {
                    "corruption": {
                        "id": f"wrong-joint-{index}",
                        "kind": "wrong_joint_shake",
                    }
                },
                Path(f"j{index}"),
            )
        )
    selected = _obvious_comparisons(available, 10)
    kinds = [record["corruption"]["kind"] for record, _ in selected]
    assert kinds.count("timing") == 4
    assert kinds.count("wrong_joint_shake") == 2
    assert sum(kind in {"fist_shape", "open_middle_fingers"} for kind in kinds) == 2
    assert kinds.count("wrist_rotation") == 2


def test_unary_calibration_prioritizes_wrong_joint_then_timing() -> None:
    records = [
        {"corruption": {"id": "finger", "kind": "fist_shape"}},
        {"corruption": {"id": "wrist", "kind": "wrist_rotation"}},
        {"corruption": {"id": "timing", "kind": "timing"}},
        {"corruption": {"id": "wrong", "kind": "wrong_joint_shake"}},
    ]
    ordered = _ordered_unary_records(records)
    assert [record["corruption"]["kind"] for _, record in ordered] == [
        "wrong_joint_shake",
        "timing",
        "wrist_rotation",
        "fist_shape",
    ]
    assert [index for index, _ in ordered] == [4, 3, 2, 1]
