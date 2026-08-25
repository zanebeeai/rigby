"""`MAX_SNAPSHOTS_PER_VIEW` must be a measurement of the corpus, not a guess about it.

Plan 05 section 3.6 derives capture's wall-clock deadline, and `pipeline.py`'s subprocess
budget in turn, from a bound on how many phase points an intent can sample. The bound was
first set at 64 from the plan, then at 32 from a seven-intent sample whose largest was 24.
The 47-case corpus that arrived with 03b breaches 32 on two cases -- `fullbody-burpee-cycle`
samples 60 -- and capture refuses them outright, which is the bound working as designed and
also a capture path that cannot run the corpus it is meant to render.

Twice now the constant has been set from too small a sample. This test is what stops a
third time: it fails when the corpus outgrows the bound, naming the case, rather than
waiting for someone to try to capture it.

Counting phase points rather than timing anything is deliberate. A count is invariant to
load, to platform and to CPU count -- see the measurement-axes rule in `docs/testing.md`.
"""

from __future__ import annotations

import pytest

from evals.capture import MAX_SNAPSHOTS_PER_VIEW, phase_sampling_points
from evals.corpus.loader import compile_case, load_corpus

pytestmark = pytest.mark.medium


@pytest.fixture(scope="module")
def sample_counts() -> list[tuple[int, str, str]]:
    """One corpus compile for the whole module. 41 compiles, not 82."""
    return _sample_counts()


def _sample_counts() -> list[tuple[int, str, str]]:
    counts: list[tuple[int, str, str]] = []
    for case in load_corpus():
        clip = compile_case(case)
        if not clip.success:
            # Known-bad cases exist to fail compilation; they are never captured.
            continue
        payload = {
            "clip": clip.model_dump(mode="json"),
            "program": case.program.model_dump(mode="json"),
        }
        counts.append((len(phase_sampling_points(payload)), case.id, str(case.entry.intent)))
    return sorted(counts, reverse=True)


def test_the_snapshot_bound_covers_every_corpus_case(
    sample_counts: list[tuple[int, str, str]],
) -> None:
    counts = sample_counts
    assert counts, "no corpus case compiled; the bound cannot be checked against nothing"
    worst, case_id, intent = counts[0]
    over = [row for row in counts if row[0] > MAX_SNAPSHOTS_PER_VIEW]
    assert not over, (
        f"{len(over)} corpus case(s) sample more phase points than capture will accept "
        f"(MAX_SNAPSHOTS_PER_VIEW={MAX_SNAPSHOTS_PER_VIEW}): "
        + ", ".join(f"{cid} ({pts} pts, {i})" for pts, cid, i in over)
        + ". Raise the bound *and* re-check the derived timeout budget in plan 05 §3.6."
    )
    # The bound is not merely satisfied, it has room. A bound sitting exactly on the
    # corpus maximum would be breached by the next case anyone blesses.
    assert MAX_SNAPSHOTS_PER_VIEW >= worst * 1.25, (
        f"the bound is {MAX_SNAPSHOTS_PER_VIEW} against a corpus maximum of {worst} "
        f"({case_id}, {intent}); less than 25% headroom means the next case breaks capture"
    )


def test_the_bound_is_not_wastefully_far_above_the_corpus(
    sample_counts: list[tuple[int, str, str]],
) -> None:
    """The other direction: the bound sizes a timeout budget, so slack is not free.

    At a bound of 96 and 1 s per snapshot, a five-candidate round gets a 20-minute
    subprocess backstop. Tripling the bound again would make that an hour, which is too
    loose to reap a wedged browser in any useful time.
    """
    worst = sample_counts[0][0]
    assert MAX_SNAPSHOTS_PER_VIEW <= worst * 4, (
        f"the bound is {MAX_SNAPSHOTS_PER_VIEW} against a corpus maximum of {worst}; "
        "that much slack inflates the derived subprocess budget for no coverage"
    )
