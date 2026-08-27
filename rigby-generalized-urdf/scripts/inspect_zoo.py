"""Ingest every zoo robot and print what the analyser measured.

A developer-facing view of the same numbers the acceptance audit checks. Useful
when a threshold needs tuning, because it shows the measurement rather than just
the pass or fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_general.ingest.integrity import check_integrity
from rigby_general.ingest.loader import load_model, urdf_velocity_limits
from rigby_general.morphology.analyze import analyze
from rigby_general.morphology.frames import horizontal_angle


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


def inspect(directory: Path, *, verbose: bool) -> tuple[bool, str]:
    source = directory / "robot.urdf"
    truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))

    loaded = load_model(source, robot_id=directory.name)
    report = check_integrity(loaded)
    report.raise_for_status(directory.name)

    morphology = analyze(
        loaded, velocity_limits=urdf_velocity_limits(loaded.source_bytes)
    )

    produced_sites = {(site.semantic.value, site.body) for site in morphology.sites}
    expected_sites = {
        (item["semantic"], item["body"]) for item in truth["expected_sites"]
    }
    missing = expected_sites - produced_sites
    accuracy = (
        len(expected_sites & produced_sites) / len(expected_sites)
        if expected_sites
        else 1.0
    )

    expected_kinds = sorted(item["kind"] for item in truth["expected_effectors"])
    produced_kinds = sorted(effector.kind.value for effector in morphology.effectors)
    kinds_ok = expected_kinds == produced_kinds
    class_ok = morphology.morphology_class.value == truth["morphology_class"]

    scale = morphology.scale
    lines = [
        f"{directory.name:<18} class={morphology.morphology_class.value:<20}"
        f" reach={scale.reach_radius_m:.3f}m  charlen={scale.characteristic_length_m:.3f}m"
        f"  vneutral={scale.neutral_speed_mps:.3f}m/s  payload={scale.payload_kg:.2f}kg",
        f"{'':18} effectors={produced_kinds} expected={expected_kinds}"
        f" {'OK' if kinds_ok else 'MISMATCH'}",
        f"{'':18} sites {accuracy:.0%}"
        + (f"  MISSING={sorted(missing)}" if missing else "")
        + f"   class {'OK' if class_ok else 'MISMATCH'}",
        f"{'':18} front={horizontal_angle(morphology.intrinsic_frame.front):+.1f}deg"
        f" conf={morphology.intrinsic_frame.confidence:.2f}"
        f" evidence={list(morphology.intrinsic_frame.evidence)}"
        f" symmetry={'yes' if morphology.symmetry else 'no'}",
    ]

    if verbose:
        roles: dict[str, list[str]] = {}
        for joint in morphology.joints:
            roles.setdefault(joint.role.value, []).append(
                f"{joint.name}(t={joint.tip_translation_m:.3f},r={joint.tip_rotation_rad:.2f})"
            )
        for role, entries in sorted(roles.items()):
            lines.append(f"{'':18}   {role:<15} {entries}")
        for effector in morphology.effectors:
            lines.append(
                f"{'':18}   effector {effector.name}: members={list(effector.member_bodies)}"
                f" aperture={effector.max_aperture_m} groups={list(effector.opposition_groups)}"
            )

    ok = kinds_ok and class_ok and accuracy >= 0.95
    return ok, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ZOO_ROOT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    arguments = parser.parse_args()

    failures = 0
    for directory in sorted(arguments.root.iterdir()):
        if not (directory / "robot.urdf").is_file():
            continue
        if arguments.only and arguments.only not in directory.name:
            continue
        try:
            ok, text = inspect(directory, verbose=arguments.verbose)
        except Exception as error:  # noqa: BLE001 - developer tool, show everything
            failures += 1
            print(f"{directory.name:<18} FAILED {type(error).__name__}: {error}")
            continue
        failures += 0 if ok else 1
        print(text)
        print()

    print(f"{'ALL OK' if failures == 0 else f'{failures} robot(s) not matching'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
