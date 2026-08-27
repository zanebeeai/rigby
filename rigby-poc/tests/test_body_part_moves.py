"""The movement vocabulary and the range-of-motion envelope cannot disagree.

``config/body_parts.v1.json`` authors every move as a signed fraction of a DOF's
own ROM range, so an out-of-range pose is not rejected by a check -- it is
unnameable. These tests are what make that claim load-bearing rather than a
comment: they take every move in the catalog, compose it through the same
anatomical frames ``rom_checks`` decomposes with, and assert the result comes
back inside the committed bound.

The round trip matters more than either half. ``compose`` and ``decompose`` are
inverses over one bone's frame, so a move that survives the trip is stated in the
same axes the ROM document measures -- which is the property that would silently
break if the vocabulary were authored in degrees, or in some other basis, and
merely bounds-checked afterwards.
"""

from __future__ import annotations

import math

import pytest

from rigby_poc.analysis.anatomy.frame import all_frames, decompose
from rigby_poc.analysis.anatomy.rom import rom_limit
from rigby_poc.body_parts import (
    BodyPartError,
    body_part_catalog,
    move_angles_deg,
    move_rotations,
    moves_for,
    parts,
    sequence_angles_deg,
    sequence_names,
    sequence_rotations,
)

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast

_DOFS = ("flexion", "abduction", "twist")


def _all_moves():
    for part, moves in body_part_catalog()["move_sets"].items():
        for move in moves:
            yield part, move


def test_the_probe_saw_a_real_catalog() -> None:
    """The instrument test. Every assertion below is vacuous on an empty catalog."""
    catalog = body_part_catalog()
    assert len(catalog["parts"]) == 30
    assert sum(len(v) for v in catalog["move_sets"].values()) > 250
    assert len(list(_all_moves())) > 250


def test_every_rig_bone_belongs_to_exactly_one_part() -> None:
    """No bone is orphaned, and none is driven by two parts at once."""
    owners: dict[str, list[str]] = {}
    for part, spec in parts().items():
        for bone in spec["bones"]:
            owners.setdefault(bone, []).append(part)
    duplicated = {b: p for b, p in owners.items() if len(p) > 1}
    assert not duplicated, f"bones claimed by more than one part: {duplicated}"
    assert len(owners) == 52, f"{len(owners)} bones covered, the rig has 52"


def test_every_move_lands_inside_the_committed_rom_bound() -> None:
    """The load-bearing one.

    A move is a fraction of ``typical_deg``, so this is a tautology *if* the
    resolution code is right and a real failure if it is not -- an off-by-one on
    the sign convention, or scaling a negative fraction by the upper bound,
    puts a move outside the envelope while still looking like a fraction.
    """
    for part, move in _all_moves():
        for bone, angles in move_angles_deg(part, move).items():
            for dof, degrees in angles.items():
                low, high = rom_limit(bone, dof).typical_deg
                assert low - 1e-9 <= degrees <= high + 1e-9, (
                    f"{part}.{move} puts {bone}.{dof} at {degrees:.3f} deg, "
                    f"outside its typical range [{low}, {high}]"
                )


def test_every_move_survives_the_compose_decompose_round_trip() -> None:
    """The vocabulary is stated in the axes the ROM layer measures.

    Without this, a move could satisfy the numeric bound above while composing
    to a rotation that ``rom_checks`` reads as a different angle entirely.
    """
    frames = all_frames()
    for part, move in _all_moves():
        authored = move_angles_deg(part, move)
        rotations = move_rotations(part, move)
        for bone, angles in authored.items():
            measured = decompose(
                [
                    rotations[bone].x,
                    rotations[bone].y,
                    rotations[bone].z,
                    rotations[bone].w,
                ],
                frames[bone],
            )
            got = dict(zip(_DOFS, measured.as_degrees()))
            for dof, degrees in angles.items():
                assert got[dof] == pytest.approx(degrees, abs=1e-6), (
                    f"{part}.{move} authored {bone}.{dof}={degrees:.4f} deg but "
                    f"composes to {got[dof]:.4f} deg"
                )


def test_every_part_can_address_all_three_of_its_degrees_of_freedom() -> None:
    """A DOF the vocabulary cannot name exists in the envelope and nowhere else.

    This is the assertion that caught the shipped gap: clavicle, elbow, knee,
    ankle and toes each had a DOF with a ROM entry and no move that reached it.
    """
    for part in parts():
        reachable = {
            dof
            for move in moves_for(part).values()
            for dofs in move.values()
            for dof in dofs
        }
        assert reachable == set(_DOFS), (
            f"{part} never reaches {sorted(set(_DOFS) - reachable)}"
        )


def test_a_sequence_is_timed_and_interpolates_between_its_steps() -> None:
    """``grip`` is thumb-then-fingers, and the ordering is observable.

    The specific failure this encodes: with a single-keyframe grasp, index and
    middle reach a free object before the thumb and push it out of the hand.
    Halfway through the sequence the thumb must already be near opposition
    while the fingers are still mostly open.
    """
    thumb_mid = sequence_angles_deg("right_grip_power", 0.5)["rightThumbMetacarpal"]
    thumb_end = sequence_angles_deg("right_grip_power", 1.0)["rightThumbMetacarpal"]
    index_mid = sequence_angles_deg("right_grip_power", 0.5)["rightIndexProximal"]
    index_end = sequence_angles_deg("right_grip_power", 1.0)["rightIndexProximal"]

    assert thumb_mid["twist"] == pytest.approx(thumb_end["twist"], abs=1e-6), (
        "the thumb should have reached opposition by the midpoint"
    )
    assert index_mid["flexion"] < 0.5 * index_end["flexion"], (
        "the fingers should still be mostly open when the thumb anchors"
    )


def test_every_sequence_resolves_to_in_range_rotations_throughout() -> None:
    """Sampled continuously, not only at the authored steps.

    Interpolating between two in-range poses stays in range only because the
    blend is in angle space. Blending quaternions instead can leave the segment
    between two valid poses, which is why this samples the interior.
    """
    for name in sequence_names():
        for step in range(21):
            progress = step / 20.0
            for bone, angles in sequence_angles_deg(name, progress).items():
                for dof, degrees in angles.items():
                    low, high = rom_limit(bone, dof).typical_deg
                    assert low - 1e-9 <= degrees <= high + 1e-9, (
                        f"{name} at t={progress:.2f} puts {bone}.{dof} at "
                        f"{degrees:.3f}, outside [{low}, {high}]"
                    )
            assert sequence_rotations(name, progress)


def test_unknown_names_raise_rather_than_returning_nothing() -> None:
    with pytest.raises(BodyPartError):
        moves_for("no_such_part")
    with pytest.raises(BodyPartError):
        move_angles_deg("right_shoulder", "no_such_move")
    with pytest.raises(BodyPartError):
        sequence_angles_deg("no_such_sequence", 0.5)


def test_the_catalog_still_matches_the_rom_document_it_was_generated_from() -> None:
    """Drift guard. The envelope is copied into the catalog for inspection; if
    ``rom.v1.json`` moves and the catalog does not, every bound above is stale.
    """
    for part, spec in parts().items():
        for bone, envelope in spec["dof_envelope"].items():
            for dof, published in envelope.items():
                limit = rom_limit(bone, dof)
                assert tuple(published["typical_deg"]) == limit.typical_deg, (
                    f"{part}/{bone}.{dof} typical range is stale"
                )
                assert tuple(published["max_deg"]) == limit.max_deg
                assert published["enforced"] == limit.enforced


def test_moves_are_fractions_and_never_raw_degrees() -> None:
    """A value outside [-1, 1] means someone authored an angle by mistake."""
    for part, moves in body_part_catalog()["move_sets"].items():
        for move, spec in moves.items():
            for bone, dofs in spec.items():
                for dof, value in dofs.items():
                    assert -1.0 <= float(value) <= 1.0, (
                        f"{part}.{move}.{bone}.{dof} is {value}; "
                        "moves are fractions of the ROM range, not degrees"
                    )


def test_the_thumb_opposition_ladder_reaches_the_finger_it_names() -> None:
    """A move called ``oppose_little`` must put the thumb near the little finger.

    Written from a sign convention rather than measured, the ladder ran exactly
    backwards: every opposition move carried the thumb further from the fingers
    than a plain ``extend`` did, 17.0 cm at ``oppose_little`` against 10.4 cm.
    The rig has no anatomical-position reference for the thumb -- ``rom.v1.json``
    marks its bounds provisional and says so -- so nothing but a measurement can
    establish which way opposition goes. This is that measurement, kept as a
    test so the ladder cannot silently invert again.
    """
    import numpy as np

    from rigby_poc.body_parts import move_rotations
    from rigby_poc.kinematics import rig_kinematics
    from rigby_poc.models import BonePose

    kinematics = rig_kinematics()

    def tip_distances(move: str) -> dict[str, float]:
        pose = {
            name: BonePose(rotation=rotation)
            for name, rotation in move_rotations("right_thumb", move).items()
        }
        positions = kinematics.canonical_positions(pose)
        thumb = positions["rightThumbDistal"]
        return {
            finger: float(np.linalg.norm(thumb - positions[f"right{finger}Distal"]))
            for finger in ("Index", "Middle", "Ring", "Little")
        }

    extend = tip_distances("extend")
    for finger in ("Index", "Middle", "Ring", "Little"):
        opposed = tip_distances(f"oppose_{finger.lower()}")
        assert opposed[finger] < extend[finger], (
            f"oppose_{finger.lower()} moves the thumb AWAY from the {finger}"
        )

    # Abduction is the axis that chooses the finger, so reaching further round
    # the hand must be a monotonic progression, not four unrelated poses.
    reach = [tip_distances(f"oppose_{f}")["Little"] for f in ("index", "middle", "ring", "little")]
    assert reach == sorted(reach, reverse=True), (
        "the ladder does not close on the little finger in order"
    )
    assert tip_distances("oppose_little")["Little"] < tip_distances("oppose_index")["Little"]


def test_every_super_primitive_move_exists_in_the_vocabulary() -> None:
    """The schedule and the movement vocabulary must not drift apart.

    A step that demands thumb load while naming a move that does not exist is a
    disagreement that surfaces only as an unexplained failed grasp.
    """
    from rigby_poc.body_parts import moves_for
    from rigby_poc.super_primitives import catalog

    for name, spec in catalog()["super_primitives"].items():
        for step in spec["steps"]:
            for part, move in (step.get("moves") or {}).items():
                for side in ("left", "right"):
                    assert move in moves_for(f"{side}_{part}"), (
                        f"{name}.{step['label']} names {side}_{part}.{move}"
                    )
