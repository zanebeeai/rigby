from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from scipy.spatial.transform import Rotation

from rigby_poc.compiler import PROJECT_ROOT
from rigby_poc.models import (
    BonePose,
    ClipResult,
    CompileRequest,
    Failure,
    FailureCode,
    Hand,
    Quat,
)
from rigby_poc.quality import evaluate_gesture_structure, shake_joint_oscillation_metrics
from rigby_poc.store import ResultStore


CorruptionKind = Literal[
    "wrist_rotation",
    "wrong_joint_shake",
    "fist_shape",
    "open_middle_fingers",
    "timing",
]


@dataclass(frozen=True)
class CorruptionSpec:
    id: str
    kind: CorruptionKind
    axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angle_rad: float = 0.0
    mode: str | None = None


def corruption_specs() -> list[CorruptionSpec]:
    specs: list[CorruptionSpec] = []
    for axis_name, axis in (
        ("flex", (1.0, 0.0, 0.0)),
        ("twist", (0.0, 1.0, 0.0)),
        ("deviate", (0.0, 0.0, 1.0)),
    ):
        for index, angle in enumerate((-1.25, -0.85, 0.85, 1.25), start=1):
            specs.append(
                CorruptionSpec(
                    id=f"wrist-{axis_name}-{index}",
                    kind="wrist_rotation",
                    axis=axis,
                    angle_rad=angle,
                )
            )
    for corruption_id, axis, scale in (
        ("wrong-joint-flex-weak", (1.0, 0.0, 0.0), 0.75),
        ("wrong-joint-flex-full", (1.0, 0.0, 0.0), 1.0),
        ("wrong-joint-deviate-full", (0.0, 0.0, 1.0), 1.0),
        ("wrong-joint-deviate-strong", (0.0, 0.0, 1.0), 1.25),
    ):
        specs.append(
            CorruptionSpec(
                id=corruption_id,
                kind="wrong_joint_shake",
                axis=axis,
                angle_rad=scale,
                mode="move_forearm_oscillation_to_hand_joint",
            )
        )
    for index, angle in enumerate((-0.24, -0.10, 0.10, 0.24), start=1):
        specs.append(
            CorruptionSpec(
                id=f"fist-shape-{index}",
                kind="fist_shape",
                axis=(0.0, 0.0, 1.0),
                angle_rad=angle,
            )
        )
    for index, angle in enumerate((-0.24, -0.10, 0.10, 0.24), start=1):
        specs.append(
            CorruptionSpec(
                id=f"open-middle-{index}",
                kind="open_middle_fingers",
                axis=(1.0, 0.0, 0.0),
                angle_rad=angle,
            )
        )
    for corruption_id, mode in (
        ("timing-snap-present", "snap_present"),
        ("timing-stutter-present", "stutter_present"),
        ("timing-reverse-present", "reverse_present"),
        ("timing-remove-hold", "remove_hold"),
    ):
        specs.append(CorruptionSpec(id=corruption_id, kind="timing", mode=mode))
    return specs


def _quat(rotation: Rotation) -> Quat:
    value = rotation.as_quat()
    return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))


def _presentation_ranges(clip: ClipResult) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in clip.metrics.get("phase_ranges_s", []):
        if isinstance(item, dict) and item.get("kind") in {"present", "hold", "shake"}:
            ranges.append((float(item["start_s"]), float(item["end_s"])))
    if not ranges:
        ranges.append((0.0, clip.duration_s * 0.75))
    return ranges


def _ranges_for(clip: ClipResult, kind: str) -> list[tuple[float, float]]:
    return [
        (float(item["start_s"]), float(item["end_s"]))
        for item in clip.metrics.get("phase_ranges_s", [])
        if isinstance(item, dict) and item.get("kind") == kind
    ]


def _indices_in_ranges(frames: list, ranges: list[tuple[float, float]]) -> list[int]:
    return [
        index
        for index, frame in enumerate(frames)
        if any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in ranges)
    ]


def _copy_pose(frames: list, target_index: int, source_index: int) -> None:
    source = frames[source_index]
    target = frames[target_index]
    target.bones = {
        name: pose.model_copy(deep=True)
        for name, pose in source.bones.items()
    }
    target.objects = {
        name: transform.model_copy(deep=True)
        for name, transform in source.objects.items()
    }


def _corrupt_timing(frames: list, clip: ClipResult, mode: str | None) -> None:
    present = _indices_in_ranges(frames, _ranges_for(clip, "present"))
    hold = _indices_in_ranges(frames, _ranges_for(clip, "hold"))
    recover = _indices_in_ranges(frames, _ranges_for(clip, "recover"))
    if len(present) < 3:
        raise ValueError("timing corruption requires a sampled present phase")
    original = [frame.model_copy(deep=True) for frame in frames]
    if mode == "snap_present":
        source = present[0]
        for target in present[:-1]:
            _copy_pose(original, target, source)
            frames[target].bones = original[target].bones
            frames[target].objects = original[target].objects
    elif mode == "stutter_present":
        last = len(present) - 1
        for offset, target in enumerate(present):
            progress = offset / max(last, 1)
            stepped = min(3, int(progress * 4)) / 3
            source_offset = round(stepped * last)
            _copy_pose(frames, target, present[source_offset])
    elif mode == "reverse_present":
        for target, source in zip(present, reversed(present), strict=True):
            frames[target].bones = original[source].bones
            frames[target].objects = original[source].objects
    elif mode == "remove_hold":
        if not hold or not recover:
            raise ValueError("remove-hold corruption requires hold and recover phases")
        last = len(recover) - 1
        for offset, target in enumerate(hold):
            source_offset = round((offset / max(len(hold) - 1, 1)) * last)
            frames[target].bones = original[recover[source_offset]].bones
            frames[target].objects = original[recover[source_offset]].objects
    else:
        raise ValueError(f"unknown timing corruption mode: {mode}")


def _corrupt_wrong_joint_shake(
    frames: list,
    clip: ClipResult,
    hand: Hand,
    spec: CorruptionSpec,
) -> None:
    """Recreate the first-pilot bug without changing beats or endpoints.

    The corrected longitudinal lower-arm oscillation is removed and transferred
    to wrist flexion/deviation. Its magnitude stays inside the broad structural
    wrist limit on purpose: this is a visual-anatomy corruption the VLM must
    reject rather than an easy deterministic-gate failure.
    """
    indices = _indices_in_ranges(frames, _ranges_for(clip, "shake"))
    if len(indices) < 3:
        raise ValueError("wrong-joint corruption requires a shake phase")
    prefix = hand.value
    lower_bone = f"{prefix}LowerArm"
    hand_bone = f"{prefix}Hand"
    baseline = Rotation.from_quat(frames[indices[0]].bones[lower_bone].rotation.as_list())
    for index in indices:
        lower = Rotation.from_quat(frames[index].bones[lower_bone].rotation.as_list())
        signed_forearm_rotation = float((baseline.inv() * lower).as_rotvec()[1])
        frames[index].bones[lower_bone] = BonePose(
            rotation=_quat(
                lower * Rotation.from_rotvec([0.0, -signed_forearm_rotation, 0.0])
            )
        )
        hand_rotation = Rotation.from_quat(frames[index].bones[hand_bone].rotation.as_list())
        wrong_joint_rotation = Rotation.from_rotvec(
            [
                component * signed_forearm_rotation * spec.angle_rad
                for component in spec.axis
            ]
        )
        frames[index].bones[hand_bone] = BonePose(
            rotation=_quat(hand_rotation * wrong_joint_rotation)
        )


def corrupt_clip(base: ClipResult, hand: Hand, spec: CorruptionSpec) -> ClipResult:
    frames = [frame.model_copy(deep=True) for frame in base.frames]
    active_ranges = _presentation_ranges(base)
    if spec.kind == "timing":
        _corrupt_timing(frames, base, spec.mode)
    elif spec.kind == "wrong_joint_shake":
        _corrupt_wrong_joint_shake(frames, base, hand, spec)
    prefix = hand.value
    for frame in frames:
        if not any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in active_ranges):
            continue
        if spec.kind == "wrist_rotation":
            current = Rotation.from_quat(frame.bones[f"{prefix}Hand"].rotation.as_list())
            corruption = Rotation.from_rotvec([component * spec.angle_rad for component in spec.axis])
            frame.bones[f"{prefix}Hand"] = BonePose(rotation=_quat(current * corruption))
        elif spec.kind == "fist_shape":
            for finger, segments in (
                ("Thumb", ("Metacarpal", "Proximal", "Distal")),
                ("Little", ("Proximal", "Intermediate", "Distal")),
            ):
                for segment_index, segment in enumerate(segments):
                    angle = 1.15 * (1.0, 1.10, 0.82)[segment_index]
                    frame.bones[f"{prefix}{finger}{segment}"] = BonePose(
                        rotation=_quat(Rotation.from_rotvec([angle, 0.0, 0.0]))
                    )
        elif spec.kind == "open_middle_fingers":
            for finger in ("Index", "Middle", "Ring"):
                for segment in ("Proximal", "Intermediate", "Distal"):
                    frame.bones[f"{prefix}{finger}{segment}"] = BonePose(rotation=Quat())
        if spec.kind not in {"wrist_rotation", "wrong_joint_shake", "timing"} and spec.angle_rad:
            current = Rotation.from_quat(frame.bones[f"{prefix}Hand"].rotation.as_list())
            variation = Rotation.from_rotvec(
                [component * spec.angle_rad for component in spec.axis]
            )
            frame.bones[f"{prefix}Hand"] = BonePose(rotation=_quat(current * variation))

    structure = evaluate_gesture_structure(frames, hand, active_ranges)
    metrics = dict(base.metrics)
    metrics.update(structure)
    metrics.update(shake_joint_oscillation_metrics(frames, hand, _ranges_for(base, "shake")))
    metrics["deliberate_corruption"] = asdict(spec)
    return base.model_copy(
        update={
            "success": False,
            "frames": frames,
            "metrics": metrics,
            "failure": Failure(
                code=FailureCode.INVALID_PROGRAM,
                message=f"Deliberate VLM calibration corruption: {spec.id}",
                details={"corruption": asdict(spec)},
                recoverable=False,
            ),
        }
    )


def persist_corruption_suite(base_result_id: str, output_path: Path) -> Path:
    root = PROJECT_ROOT / "results" / base_result_id
    if not root.is_dir():
        raise FileNotFoundError(base_result_id)
    request = CompileRequest.model_validate_json((root / "request.json").read_text(encoding="utf-8"))
    base = ClipResult.model_validate_json((root / "clip.json").read_text(encoding="utf-8"))
    store = ResultStore()
    records = []
    for spec in corruption_specs():
        corrupted = corrupt_clip(base, request.program.hand, spec)
        result_id = store.persist(request, corrupted)
        records.append(
            {
                "corruption": asdict(spec),
                "result_id": result_id,
                "structural_valid": corrupted.metrics.get("structural_valid"),
                "structural_failures": corrupted.metrics.get("structural_failures"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "base_result_id": base_result_id,
                "prompt": request.program.source_text,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deliberate visual corruptions for VLM calibration")
    parser.add_argument("base_result_id")
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()
    print(persist_corruption_suite(arguments.base_result_id, arguments.output_path))


if __name__ == "__main__":
    main()
