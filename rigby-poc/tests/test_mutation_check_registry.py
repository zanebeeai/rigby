"""The registry that keeps a mutation from targeting a check nothing emits.

This file exists because of a defect it can now detect. 06a declared its target ids
from plan 06 section 3.3 rather than from the analyzer, and all four --
``anatomy.wrist.swing``, ``anatomy.wrist.twist``, ``signal.min_jerk``,
``signal.sparc`` -- are names no check emits. A target no check emits produces the
same detection-matrix cell as "no detector exists": nothing fired, every case, every
severity. Plan 06 section 6.1 calls that the worse half of the unfalsifiable class,
because the first is visible as a gap and the second is visible as a *result*.
"""

from __future__ import annotations

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from evals.mutations.anatomy import rom_sweep
from evals.mutations.checks import (
    EMITTED_BY_CASES,
    FIXED_CHECK_IDS,
    known_check_ids,
    report_only_reason,
    require_known_targets,
    rom_check_ids,
    rom_detected,
    unknown_targets,
)
from evals.mutations.clipping import limb_through_torso_sweep
from evals.mutations.legacy import legacy_specs
from evals.mutations.signal import jitter_sweep
from evals.mutations.timing import freeze_sweep, snap_sweep
from rigby_poc.analysis import validate

#: Compiles the whole corpus.
pytestmark = pytest.mark.medium

#: The four ids 06a declared. Kept as a literal rather than deleted: the anti-
#: tautology check for this whole file is that these still do not resolve.
RETIRED_06A_TARGETS = (
    "anatomy.wrist.swing",
    "anatomy.wrist.twist",
    "signal.min_jerk",
    "signal.sparc",
)


@pytest.fixture(scope="module")
def emitted_over_corpus() -> dict[str, int]:
    """Every check id ``validate`` emits, and how many cases emit it."""
    counts: dict[str, int] = {}
    for case in load_corpus():
        clip = compile_case(case)
        for result in validate(clip.metrics, case.program):
            counts[result.id] = counts.get(result.id, 0) + 1
    return counts


def test_the_probe_saw_a_real_corpus(emitted_over_corpus: dict[str, int]) -> None:
    # The instrument test for this file. Every assertion below is over a collection
    # this test did not construct, and all of them pass vacuously on an empty one --
    # the shape that made four judge guards green against nothing.
    assert emitted_over_corpus, "the corpus produced no checks at all"
    assert len(load_corpus()) >= 47


def test_the_declared_registry_is_what_the_analyzer_emits(
    emitted_over_corpus: dict[str, int],
) -> None:
    # Pinned by equality, not by containment. A subset assertion would let the
    # declaration silently rot back into a plan-derived wish list, which is the
    # defect this module was written for.
    assert set(emitted_over_corpus) == FIXED_CHECK_IDS


def test_the_declared_per_check_case_counts_are_the_measured_ones(
    emitted_over_corpus: dict[str, int],
) -> None:
    # The counts are the honest denominator for any rate quoted against an axis, so
    # they are asserted rather than left as prose.
    assert emitted_over_corpus == dict(EMITTED_BY_CASES)


def test_only_the_contract_checks_reach_every_case(
    emitted_over_corpus: dict[str, int],
) -> None:
    total = len(load_corpus())
    universal = {cid for cid, n in emitted_over_corpus.items() if n == total}
    assert universal == {
        "contract.clip.joint_limit_violations",
        "contract.clip.non_finite_transforms",
        "contract.clip.rotational_discontinuities",
    }


def test_the_ids_06a_declared_are_still_not_emitted(
    emitted_over_corpus: dict[str, int],
) -> None:
    # The anti-tautology check. If any of these ever becomes a real id, the
    # correction in `legacy.py` needs revisiting rather than silently agreeing.
    for retired in RETIRED_06A_TARGETS:
        assert retired not in emitted_over_corpus
        assert retired not in known_check_ids()


def test_the_06a_declaration_would_be_refused_now() -> None:
    # Demonstrates the guard against the exact input that motivated it, rather than
    # against a synthetic bad id -- the difference between a guard shown red against
    # a real defect and one shown red against a strawman.
    from evals.mutations.family import MutationFamily, Tier
    from evals.mutations.spec import MutationSpec

    as_06a_had_it = MutationSpec(
        id="wrist_rotation",
        family=MutationFamily.ANATOMY,
        targets=("anatomy.wrist.swing", "anatomy.wrist.twist"),
        severity=1.0,
        tier=Tier.SEVERE,
    )
    assert unknown_targets([as_06a_had_it]) == {
        "wrist_rotation": ("anatomy.wrist.swing", "anatomy.wrist.twist")
    }
    with pytest.raises(ValueError, match="name checks nothing emits"):
        require_known_targets([as_06a_had_it])


def test_every_shipped_spec_targets_a_check_that_exists() -> None:
    shipped = [
        *legacy_specs(),
        *snap_sweep(),
        *freeze_sweep(),
        *jitter_sweep(),
        *limb_through_torso_sweep(),
        *rom_sweep("rightLowerLeg", "flexion"),
    ]
    assert len(shipped) > 28, "the sweep families did not reach the assertion"
    require_known_targets(shipped)


def test_rom_ids_come_from_the_committed_document() -> None:
    ids = rom_check_ids()
    assert len(ids) == 156
    assert "anatomy.rom.rightLowerLeg.flexion" in ids
    # Generated, not declared: they are absent from the fixed set on purpose.
    assert not (ids & FIXED_CHECK_IDS)


def test_rom_is_flagged_report_only_and_the_others_are_not() -> None:
    assert report_only_reason("anatomy.rom.rightLowerLeg.flexion")
    assert report_only_reason("contract.clip.rotational_discontinuities") == ""


def test_rom_detection_reads_the_band_and_refuses_an_unmeasured_bone() -> None:
    assert rom_detected({"bone": "b", "dof": "flexion", "band": "beyond_max"})
    assert rom_detected({"bone": "b", "dof": "flexion", "band": "beyond_typical"})
    assert not rom_detected({"bone": "b", "dof": "flexion", "band": "within_typical"})
    # Not-measured is neither detection nor cleanliness. Returning False here is the
    # conflation four lanes hit independently this push.
    with pytest.raises(ValueError, match="not measured"):
        rom_detected({"bone": "leftToes", "dof": "flexion"})
