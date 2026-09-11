"""Deterministic construction and structural validation of calibration studies."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from rigby_v2.flywheel.schemas import (
    CalibrationRatingV1,
    DefectKind,
    HumanComparisonPairV1,
    RubricScoresV1,
)


SUPPORTED_ACTION_FAMILIES = (
    "communicative_gesture",
    "grasp_place",
    "tool_use",
    "articulated_object",
    "bimanual_multiphase",
)
MINIMUM_PAIR_COUNT = 200
MINIMUM_PAIRS_PER_FAMILY = 30
_BLIND_ID = re.compile(r"^blind_[0-9a-f]{20}$")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _blind(value: str) -> str:
    return "blind_" + _digest(value)[:20]


@dataclass(frozen=True, slots=True)
class PairGroundTruth:
    pair_id: str
    preferred_record_id: str
    critical_reject_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RaterAssignment:
    rater_id_hash: str
    presentation_order: str


@dataclass(frozen=True, slots=True)
class ComparisonAssignment:
    pair_id: str
    action_family: str
    left_record_id: str
    right_record_id: str
    defects: tuple[DefectKind, ...]
    raters: tuple[RaterAssignment, RaterAssignment, RaterAssignment]


@dataclass(frozen=True, slots=True)
class CalibrationStudyBlueprint:
    seed: int
    assignments: tuple[ComparisonAssignment, ...]
    ground_truth: Mapping[str, PairGroundTruth]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ground_truth", MappingProxyType(dict(self.ground_truth)))


@dataclass(frozen=True, slots=True)
class CalibrationStudy:
    seed: int
    pairs: tuple[HumanComparisonPairV1, ...]
    ground_truth: Mapping[str, PairGroundTruth]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ground_truth", MappingProxyType(dict(self.ground_truth)))


@dataclass(frozen=True, slots=True)
class StudyValidation:
    pair_count: int
    pair_counts_by_family: Mapping[str, int]
    rating_count: int
    presentation_counts: Mapping[str, int]
    defect_counts: Mapping[DefectKind, int]
    reversal_twin_groups: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pair_counts_by_family", MappingProxyType(dict(self.pair_counts_by_family))
        )
        object.__setattr__(
            self, "presentation_counts", MappingProxyType(dict(self.presentation_counts))
        )
        object.__setattr__(self, "defect_counts", MappingProxyType(dict(self.defect_counts)))


class CalibrationStudyError(ValueError):
    pass


def construct_calibration_blueprint(
    *,
    seed: int = 20260810,
    pairs_per_family: int = 40,
) -> CalibrationStudyBlueprint:
    if pairs_per_family < MINIMUM_PAIRS_PER_FAMILY:
        raise CalibrationStudyError(
            f"each supported family requires at least {MINIMUM_PAIRS_PER_FAMILY} pairs"
        )
    if pairs_per_family * len(SUPPORTED_ACTION_FAMILIES) < MINIMUM_PAIR_COUNT:
        raise CalibrationStudyError(f"study requires at least {MINIMUM_PAIR_COUNT} pairs")
    rng = random.Random(seed)
    assignments: list[ComparisonAssignment] = []
    truths: dict[str, PairGroundTruth] = {}
    defect_cycle = (
        DefectKind.REALISTIC,
        DefectKind.NEAR_TIE,
        DefectKind.LEFT_RIGHT,
        DefectKind.TIMING,
        DefectKind.MISSING_VIEW,
        DefectKind.HAND,
        DefectKind.CONTACT,
    )
    global_index = 0
    for family in SUPPORTED_ACTION_FAMILIES:
        for family_index in range(pairs_per_family):
            pair_id = f"cal-{family}-{family_index:03d}"
            if family_index < 8:
                twin_index = family_index // 2
                a = _blind(f"{seed}:{family}:reversal:{twin_index}:a")
                b = _blind(f"{seed}:{family}:reversal:{twin_index}:b")
                left, right = (a, b) if family_index % 2 == 0 else (b, a)
                defects = (DefectKind.ORDER_REVERSAL, DefectKind.REALISTIC)
                preferred = a
            else:
                a = _blind(f"{seed}:{family}:{family_index}:a")
                b = _blind(f"{seed}:{family}:{family_index}:b")
                left, right = (a, b)
                if rng.random() < 0.5:
                    left, right = right, left
                defect = defect_cycle[(family_index - 8) % len(defect_cycle)]
                defects = (defect,)
                preferred = a if rng.random() < 0.5 else b

            base_orders = (
                ["left_right", "right_left", "left_right"]
                if global_index % 2 == 0
                else ["right_left", "left_right", "right_left"]
            )
            rng.shuffle(base_orders)
            raters = tuple(
                RaterAssignment(
                    rater_id_hash=_digest(f"rater:{seed}:{global_index}:{rater_slot}"),
                    presentation_order=base_orders[rater_slot],
                )
                for rater_slot in range(3)
            )
            assignment = ComparisonAssignment(
                pair_id=pair_id,
                action_family=family,
                left_record_id=left,
                right_record_id=right,
                defects=defects,
                raters=raters,  # type: ignore[arg-type]
            )
            critical = (
                (b if preferred == a else a,)
                if any(
                    defect
                    in {
                        DefectKind.LEFT_RIGHT,
                        DefectKind.MISSING_VIEW,
                        DefectKind.HAND,
                        DefectKind.CONTACT,
                    }
                    for defect in defects
                )
                else ()
            )
            assignments.append(assignment)
            truths[pair_id] = PairGroundTruth(
                pair_id=pair_id,
                preferred_record_id=preferred,
                critical_reject_record_ids=critical,
            )
            global_index += 1
    return CalibrationStudyBlueprint(
        seed=seed,
        assignments=tuple(assignments),
        ground_truth=truths,
    )


def _verdict_for_record(
    assignment: ComparisonAssignment,
    order: str,
    record_id: str,
) -> str:
    screen_left = (
        assignment.left_record_id if order == "left_right" else assignment.right_record_id
    )
    return "left" if record_id == screen_left else "right"


def build_synthetic_completed_study(
    *,
    seed: int = 20260810,
    correct_preference_probability: float = 0.96,
    critical_false_accept_probability: float = 0.0,
) -> CalibrationStudy:
    """Build deterministic synthetic ratings without downloading dataset media."""

    if not 0.5 <= correct_preference_probability <= 1.0:
        raise ValueError("synthetic preference probability must be between 0.5 and 1")
    if not 0.0 <= critical_false_accept_probability <= 1.0:
        raise ValueError("critical false-accept probability must be between zero and one")
    blueprint = construct_calibration_blueprint(seed=seed)
    rng = random.Random(seed ^ 0x51A7C0DE)
    rubric = RubricScoresV1(
        semantic_fidelity=4.2,
        physical_plausibility=4.1,
        contact_quality=4.0,
        timing_energy=4.0,
        whole_body_quality=4.1,
        visual_clarity=4.3,
    )
    pairs: list[HumanComparisonPairV1] = []
    for assignment in blueprint.assignments:
        truth = blueprint.ground_truth[assignment.pair_id]
        other = (
            assignment.right_record_id
            if truth.preferred_record_id == assignment.left_record_id
            else assignment.left_record_id
        )
        ratings = []
        for rater in assignment.raters:
            error_probability = (
                critical_false_accept_probability
                if truth.critical_reject_record_ids
                else 1.0 - correct_preference_probability
            )
            selected = (
                other if rng.random() < error_probability else truth.preferred_record_id
            )
            ratings.append(
                CalibrationRatingV1(
                    rater_id_hash=rater.rater_id_hash,
                    presentation_order=rater.presentation_order,
                    verdict=_verdict_for_record(
                        assignment, rater.presentation_order, selected
                    ),
                    rubric=rubric,
                )
            )
        pairs.append(
            HumanComparisonPairV1(
                pair_id=assignment.pair_id,
                action_family=assignment.action_family,
                left_record_id=assignment.left_record_id,
                right_record_id=assignment.right_record_id,
                defects=assignment.defects,
                ratings=tuple(ratings),
            )
        )
    study = CalibrationStudy(seed=seed, pairs=tuple(pairs), ground_truth=blueprint.ground_truth)
    validate_calibration_study(study)
    return study


def selected_record(pair: HumanComparisonPairV1, rating: CalibrationRatingV1) -> str | None:
    if rating.verdict not in {"left", "right"}:
        return None
    screen_left = (
        pair.left_record_id
        if rating.presentation_order == "left_right"
        else pair.right_record_id
    )
    screen_right = (
        pair.right_record_id
        if rating.presentation_order == "left_right"
        else pair.left_record_id
    )
    return screen_left if rating.verdict == "left" else screen_right


def validate_calibration_study(study: CalibrationStudy) -> StudyValidation:
    if len(study.pairs) < MINIMUM_PAIR_COUNT:
        raise CalibrationStudyError(f"study requires at least {MINIMUM_PAIR_COUNT} pairs")
    pair_ids = [pair.pair_id for pair in study.pairs]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != set(study.ground_truth):
        raise CalibrationStudyError("pair IDs must be unique and exactly match ground truth")
    family_counts = {family: 0 for family in SUPPORTED_ACTION_FAMILIES}
    defect_counts = {defect: 0 for defect in DefectKind}
    presentation_counts = {"left_right": 0, "right_left": 0}
    reversal_groups: dict[tuple[str, str], list[HumanComparisonPairV1]] = {}
    for pair in study.pairs:
        if pair.action_family not in family_counts:
            raise CalibrationStudyError(f"unsupported action family: {pair.action_family}")
        family_counts[pair.action_family] += 1
        if not _BLIND_ID.fullmatch(pair.left_record_id) or not _BLIND_ID.fullmatch(
            pair.right_record_id
        ):
            raise CalibrationStudyError("comparison record IDs must be blinded")
        if len({rating.rater_id_hash for rating in pair.ratings}) != 3:
            raise CalibrationStudyError("each pair requires exactly three independent raters")
        for rating in pair.ratings:
            presentation_counts[rating.presentation_order] += 1
        for defect in pair.defects:
            defect_counts[defect] += 1
        if DefectKind.ORDER_REVERSAL in pair.defects:
            key = tuple(sorted((pair.left_record_id, pair.right_record_id)))
            reversal_groups.setdefault(key, []).append(pair)
    if any(value < MINIMUM_PAIRS_PER_FAMILY for value in family_counts.values()):
        raise CalibrationStudyError(
            f"each supported family requires at least {MINIMUM_PAIRS_PER_FAMILY} pairs"
        )
    if any(value == 0 for value in defect_counts.values()):
        raise CalibrationStudyError("study must cover every required defect class")
    if abs(presentation_counts["left_right"] - presentation_counts["right_left"]) > 1:
        raise CalibrationStudyError("presentation order must be globally balanced")
    if any(len(group) != 2 for group in reversal_groups.values()):
        raise CalibrationStudyError("order-reversal comparisons must occur in twin pairs")
    for group in reversal_groups.values():
        first, second = group
        if first.left_record_id != second.right_record_id or first.right_record_id != second.left_record_id:
            raise CalibrationStudyError("order-reversal twins must swap record sides")
    return StudyValidation(
        pair_count=len(study.pairs),
        pair_counts_by_family=family_counts,
        rating_count=sum(len(pair.ratings) for pair in study.pairs),
        presentation_counts=presentation_counts,
        defect_counts=defect_counts,
        reversal_twin_groups=len(reversal_groups),
    )
