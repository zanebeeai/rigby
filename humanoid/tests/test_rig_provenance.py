"""Every anatomical verdict says which skeleton it was measured against.

Plan 04 §6.1d. Found with lane `capture` while reviewing 05. Two independent
parses of one GLB -- this process's and the browser's -- with nothing connecting
them, and a cache that holds the first for the process lifetime. 05's render
provenance covers the case where the asset changes and its manifest does not;
the case it cannot cover is both changing together, where capture passes while
this process still serves the old skeleton.

The fix is not invalidation. It is one skeleton per process, said out loud.
"""

from __future__ import annotations

import hashlib

import pytest

from evals.capture import humanoid_asset_sha256
from rigby_poc.analysis.anatomy import all_frames, bone_anatomical_frame
from rigby_poc.analysis.anatomy.rom import (
    RomError,
    assert_evidence_matches_the_analysed_rig,
    rom_violations,
)
from rigby_poc.kinematics import PROJECT_ROOT, rig_kinematics

pytestmark = pytest.mark.medium


def test_the_digest_is_of_the_bytes_actually_parsed() -> None:
    """Not of whatever is on disk when someone asks.

    The distinction is the whole point: re-reading returns the digest of the
    current file while the caller holds transforms parsed from the previous one,
    which agrees with the declared hash exactly when it should not.
    """

    asset = PROJECT_ROOT / "assets" / "models" / "human-male.glb"
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()

    assert rig_kinematics().asset_sha256 == expected


def test_the_parsed_digest_agrees_with_the_declared_one_today() -> None:
    """A sanity check, and deliberately not the assertion that matters.

    This compares a parse against a *declaration*. It is worth having because a
    disagreement today would mean the shipped asset and its manifest are already
    out of step -- but it is green in the failure case §6.1d exists for, which is
    why :func:`assert_evidence_matches_the_analysed_rig` compares parse to parse
    instead.
    """

    assert rig_kinematics().asset_sha256 == humanoid_asset_sha256()


def test_every_frame_carries_the_digest_and_they_all_agree() -> None:
    """One skeleton per process, stated by every object derived from it.

    If two ever disagreed, frames from two rigs would be coexisting in the
    module caches -- lane `capture`'s trap. They cannot today because nothing
    re-parses, and `conftest.py` plus a test now assert the rig cache is never
    cleared.
    """

    digests = {frame.asset_sha256 for frame in all_frames().values()}

    assert digests == {rig_kinematics().asset_sha256}
    assert len(all_frames()) == 52


def test_a_violation_record_names_the_skeleton_it_was_measured_against() -> None:
    from evals.corpus import load_corpus
    from evals.corpus.loader import compile_case

    case = next(item for item in load_corpus() if item.entry.id == "fullbody-dance")
    clip = compile_case(case)

    violations = rom_violations(clip.frames, fps=clip.fps)

    assert violations
    assert {item.asset_sha256 for item in violations} == {rig_kinematics().asset_sha256}
    assert all(len(item.asset_sha256) == 64 for item in violations)


def test_evidence_from_a_different_skeleton_fails_loudly() -> None:
    """The assertion 04c adds, and the case 05 cannot catch on its own."""

    assert_evidence_matches_the_analysed_rig(rig_kinematics().asset_sha256)

    with pytest.raises(RomError, match="different skeleton"):
        assert_evidence_matches_the_analysed_rig("0" * 64)


def test_the_frame_cache_does_not_outlive_a_rig_change_silently() -> None:
    """The coexistence trap, exercised rather than reasoned about.

    Clearing the derived caches without clearing the rig must not change any
    digest, because the rig is what carries it. If a future refactor keyed the
    frames on something other than the live rig, this would drift.
    """

    before = bone_anatomical_frame("leftLowerArm").asset_sha256
    bone_anatomical_frame.cache_clear()
    all_frames.cache_clear()
    after = bone_anatomical_frame("leftLowerArm").asset_sha256

    assert before == after == rig_kinematics().asset_sha256
