"""``config/rom.v1.json`` must be complete, cited, and reproducible.

Plan 04 §3.3 and §5. The property that matters most here is not that the
numbers are right -- most of them are provisional and say so -- but that every
one of them is *attributable*. An uncited threshold is indistinguishable from a
guess, which is exactly how the current global velocity ceilings came to be
derived from six clips of one gesture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_poc.analysis.anatomy import authored_rom
from rigby_poc.analysis.anatomy.conventions import joint_class
from rigby_poc.analysis.anatomy.neutral import NO_NEUTRAL, rest_offset
from rigby_poc.analysis.anatomy.rom import (
    DOFS,
    REQUIRED_SOURCE_FIELDS,
    ROM_FILE,
    RomError,
    enforceability,
    rom_limit,
    rom_limits,
)
from rigby_poc.analysis.rig import canonical_bone_names

pytestmark = pytest.mark.medium

#: Bones ``compiler.py`` never assigns a rotation to on any path, so no prompt
#: can exercise their limits. Confirmed independently by lane `groundtruth` over
#: the corpus and here by reading every assignment site.
IMMOBILE = {"leftToes", "rightToes", "neck", "spine", "upperChest"}


def test_every_canonical_bone_has_every_dof() -> None:
    expected = {(bone, dof) for bone in canonical_bone_names() for dof in DOFS}

    assert set(rom_limits()) == expected
    assert len(rom_limits()) == 156


def test_every_entry_carries_a_source_with_its_kind_s_required_fields() -> None:
    for (bone, dof), limit in rom_limits().items():
        source = limit.source
        kind = source.get("kind")
        assert kind in REQUIRED_SOURCE_FIELDS, f"{bone}.{dof}: unknown source kind {kind!r}"
        missing = REQUIRED_SOURCE_FIELDS[kind] - set(source)
        assert not missing, f"{bone}.{dof}: source kind {kind!r} is missing {sorted(missing)}"
        assert source.get("reference_frame"), f"{bone}.{dof}: no reference_frame"


def test_no_entry_claims_to_be_measured() -> None:
    """Nothing in this table is derived from data yet, and it must not say it is.

    04b is report-only precisely because no ROM value here has been calibrated.
    A ``measured`` kind would require an n, and there is no n to give.
    """

    kinds = {limit.source["kind"] for limit in rom_limits().values()}

    assert "measured" not in kinds
    assert kinds <= {"external", "invariant", "provisional"}


def test_the_thumb_declares_that_no_reference_frame_exists() -> None:
    """Plan §6.1: a missing basis is a well-formed answer, not a missing value."""

    for bone in canonical_bone_names():
        if joint_class(bone) not in NO_NEUTRAL:
            continue
        assert rest_offset(bone) is None
        for dof in DOFS:
            limit = rom_limit(bone, dof)
            assert limit.rest_offset_deg is None
            assert limit.source["kind"] == "provisional"
            assert "none" in limit.source["reference_frame"]


def test_the_rig_s_own_rest_pose_is_inside_every_typical_band() -> None:
    """The cheapest possible check that the rest offsets are applied correctly.

    A bind pose is a plausible human posture, so a limit that excludes it is a
    limit on the wrong band. This caught a sign error in
    ``DofLimit.to_rest_relative`` that had put every bound on the far side of
    the rest pose -- 179 degrees out at the shoulder -- and then a neck offset
    of 40.9 degrees derived from treating the spine as straight at neutral,
    which it is not.
    """

    outside = [
        (bone, dof, limit.typical_deg, limit.rest_offset_deg)
        for (bone, dof), limit in rom_limits().items()
        if limit.band_of(0.0) != "within_typical"
    ]

    assert outside == []


def test_hinge_dofs_are_hard_asserted_and_hinge_twist_is_not() -> None:
    """Plan §3.3's sketch was wrong and this pins the correction.

    Forearm pronation and knee tibial rotation are real degrees of freedom with
    ranges near 80 and 30 degrees. The sketch hard-asserted both at +-5, which
    would have rejected every pronation the rig deliberately produces -- the rig
    declares ``forearm_twist_rad`` 1.35 rad itself.
    """

    for side in ("left", "right"):
        for bone in (f"{side}LowerArm", f"{side}LowerLeg"):
            assert rom_limit(bone, "abduction").hard_assert, bone
            twist = rom_limit(bone, "twist")
            assert not twist.hard_assert, bone
            # Range width, not the upper bound: the knee's range is asymmetric
            # and is negated-and-swapped on the right side, so the upper bound
            # is 35 on the left and 15 on the right for the same joint.
            assert twist.max_deg[1] - twist.max_deg[0] >= 40.0, bone
        assert not rom_limit(f"{side}LowerArm", "flexion").hard_assert


def test_a_hard_asserted_dof_has_no_separate_typical_band() -> None:
    for (bone, dof), limit in rom_limits().items():
        if limit.hard_assert:
            assert limit.typical_deg == limit.max_deg, f"{bone}.{dof}"


def test_structurally_immobile_bones_are_marked_mutation_only() -> None:
    """Plan §6.1e: a limit that can never fire must not look like one that can."""

    marked = {
        bone for bone in canonical_bone_names() if enforceability(bone) == "mutation_only"
    }

    assert marked == IMMOBILE


def test_the_committed_file_matches_the_authored_table() -> None:
    """The numbers are decided in ``authored_rom``; the JSON is its output.

    Without this, the two drift and the rationale in the module stops
    describing the value the loader actually reads.
    """

    assert Path(ROM_FILE).read_text(encoding="utf-8") == authored_rom.rom_json()


def test_the_committed_file_holds_only_plain_json_numbers() -> None:
    """Lane `infra`: a numpy scalar serialises differently across platforms."""

    document = json.loads(Path(ROM_FILE).read_text(encoding="utf-8"))

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif node is not None:
            assert type(node) in (bool, int, float, str), type(node)

    walk(document)


def test_an_unknown_limit_fails_loudly() -> None:
    with pytest.raises(RomError, match="rather than hardcoding"):
        rom_limit("leftLowerArm", "sideways")


def test_flexion_and_abduction_limits_are_identical_across_sides() -> None:
    """Plan §3.4: authored once, applied to both sides unchanged."""

    for bone in canonical_bone_names():
        if not bone.startswith("left"):
            continue
        mirror = "right" + bone[len("left") :]
        for dof in ("flexion", "abduction"):
            left, right = rom_limit(bone, dof), rom_limit(mirror, dof)
            assert left.typical_deg == right.typical_deg, f"{bone}.{dof}"
            assert left.max_deg == right.max_deg, f"{bone}.{dof}"


def test_twist_limits_are_negated_and_swapped_across_sides() -> None:
    """The half of §3.4's mirror rule that is real.

    04a measured that a mirrored pose preserves flexion and abduction and
    negates only twist. It follows that an asymmetric twist range authored for
    the left must be negated *and* swapped for the right, or the two sides
    permit different physical motions. Lane `groundtruth` measured the matching
    fact in the motion: a turn twists the two knees 9.0 degrees in opposite
    signs.
    """

    asymmetric = 0
    for bone in canonical_bone_names():
        if not bone.startswith("left"):
            continue
        mirror = "right" + bone[len("left") :]
        left, right = rom_limit(bone, "twist"), rom_limit(mirror, "twist")
        assert right.typical_deg == (-left.typical_deg[1], -left.typical_deg[0]), bone
        assert right.max_deg == (-left.max_deg[1], -left.max_deg[0]), bone
        if left.max_deg[0] != -left.max_deg[1]:
            asymmetric += 1

    # The knee is the one case that makes this test bite: a symmetric range
    # mirrors onto itself and the assertion above holds vacuously. Tibial
    # rotation is ~10 deg internal against ~30 external, so the knee is the only
    # joint in the table whose twist range is not symmetric.
    assert asymmetric == 1


def test_decomposition_is_batched_not_per_frame(monkeypatch) -> None:
    """A guard against the regression this file's own history contains.

    The first implementation called the scalar ``decompose`` once per bone per
    frame, building four ``Rotation`` objects each time. Measured at 2.9 ms per
    frame over 52 bones, which put ``fullbody-burpee-cycle`` at 1.75 s for a
    single report-only check. Batching per bone made it 11x faster.

    **This asserts the structural property, not a wall-clock bound.** A timing
    test is the wrong instrument here and lane `capture` sized the problem: the
    only cross-platform datapoint anyone has is a 4.2x whole-suite ratio between
    Windows CI and local macOS, which is itself confounded by runner hardware
    and dominated by startup rather than numpy. A bound loose enough to survive
    that is close to loose enough to miss the regression it exists for -- the
    usable window sits between roughly 640 ms and 1750 ms, which is not a window
    worth betting a guard on.

    Counting ``Rotation`` constructions has none of that exposure. Batched, the
    count is a function of the bone count alone; per-frame, it scales with the
    clip. Ten frames and a hundred frames must therefore cost the same.
    """

    from scipy.spatial.transform import Rotation

    from evals.corpus import load_corpus
    from evals.corpus.loader import compile_case
    from rigby_poc.analysis.anatomy.rom import rom_violations

    case = next(item for item in load_corpus() if item.entry.id == "fullbody-burpee-cycle")
    clip = compile_case(case)
    rom_violations(clip.frames[:2], fps=clip.fps)  # warm every cache

    calls = {"n": 0}
    original = Rotation.from_quat

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(Rotation, "from_quat", counting)

    calls["n"] = 0
    rom_violations(clip.frames[:10], fps=clip.fps)
    short = calls["n"]

    calls["n"] = 0
    rom_violations(clip.frames[:100], fps=clip.fps)
    long = calls["n"]

    assert short == long, (
        f"Rotation.from_quat was called {short} times for 10 frames and {long} "
        "for 100. The decomposition is running per frame rather than per bone; "
        "see rigby_poc.analysis.anatomy.frame.decompose_series."
    )
    assert long <= 4 * len(rom_limits()) / len(DOFS)


def test_an_unmeasured_dof_is_skipped_not_reported_as_within_range() -> None:
    """A DOF nothing measured must not claim a band.

    Report-only mode makes this easy to get wrong: every check returns
    ``status="pass"``, so the tempting shape is to emit ``within_typical`` for
    anything that produced no violation. That puts a value in the output which
    is present and not derived from what it claims to describe -- and an empty
    clip yielded 156 of them.

    Lanes `infra`, `analysis` and `groundtruth` each hit the same shape
    independently: ten never-written ``_base_metrics`` defaults, stale
    post-mutation metric reads, and the ``NOT_MEASURED`` versus ``PASSED``
    distinction in ``evals/corpus/gates.py``. This keeps 04b off that list.
    """

    from rigby_poc.analysis.anatomy.rom import rom_checks

    results = rom_checks([], fps=30.0)

    assert len(results) == 156
    assert {result.status for result in results} == {"skip"}
    assert all("nothing was measured" in result.detail for result in results)
    assert not any(
        isinstance(result.measured, dict) and result.measured.get("band") == "within_typical"
        for result in results
    )
