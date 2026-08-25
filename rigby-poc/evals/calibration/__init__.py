"""Calibration drivers: they consume ground truth, they never author it.

Plan 10 §5. The corpus (`evals.corpus`) and the mutation library
(`evals.mutations`) supply labels; this package drives graders across them and
scores the result through `evals.calibration_stats`. It writes no corpus case and
authors no mutation spec — if an arm has no data, it reports that it has no data
rather than inventing some.

Reassigned from lane `groundtruth` mid-push; the design decisions in
`replay.py` are theirs and are recorded as such.
"""

from __future__ import annotations

from .replay import (
    RecordedGrader,
    ReplayError,
    ReplayStore,
    graders_from_record,
    prompt_versions_agree,
)

__all__ = [
    "RecordedGrader",
    "ReplayError",
    "ReplayStore",
    "graders_from_record",
    "prompt_versions_agree",
]
