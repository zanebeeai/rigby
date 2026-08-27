"""A grader must never be dispatched blind, and a key pose must exist in both views.

`_key_pose_labels` is a declared list. Its default is `{presented_pose}`, which
only the `present` phase emits (`evals/capture.py:984`), so a full-body,
composite or sequence clip matched nothing at all. The consequence was not a
crash: `crossview` was dispatched with **zero images** and returned four claims
anyway, and the three `evidence="both"` graders silently ran on timelines alone --
half their declared evidence, on 17 of 47 corpus cases.

Invisible to the suite because every judge in it is a fake client
(`docs/testing.md`): a fake returns claims whatever the payload was, so an empty
payload and a full one produce identical records. Found by running a real model.
"""

from __future__ import annotations

import pytest

from rigby_poc.judge import _key_pose_labels, _paired_key_pose_labels

pytestmark = pytest.mark.fast


def _snap(index: int, view: str, label: str) -> dict[str, object]:
    return {"id": f"{index:02d}-{view}", "view": view, "label": label, "phase": "body"}


def _full_body_snapshots() -> list[dict[str, object]]:
    labels = ["dance_entry", "dance_beat_2", "dance_beat_4", "dance_exit"]
    return [
        _snap(i + 1, view, label)
        for i, label in enumerate(labels)
        for view in ("ego", "orbit")
    ]


def test_the_declared_default_matches_nothing_a_full_body_clip_emits() -> None:
    # The defect itself, pinned. If this ever stops being true the fallback below
    # is dead code and should go -- but it is true today and it is why the
    # fallback exists.
    snapshots = _full_body_snapshots()
    emitted = {str(s["label"]) for s in snapshots}
    assert _key_pose_labels({"intent": "full_body"}) == {"presented_pose"}
    assert not (_key_pose_labels({"intent": "full_body"}) & emitted)


def test_the_selection_falls_back_to_a_label_the_evidence_carries() -> None:
    snapshots = _full_body_snapshots()
    selected = _paired_key_pose_labels({"intent": "full_body"}, snapshots)
    assert selected, "a clip with paired snapshots must yield a key pose"
    assert selected <= {str(s["label"]) for s in snapshots}


def test_every_selected_label_appears_in_both_views() -> None:
    # `crossview` compares one moment across ego and orbit. A label present in
    # only one view is not a cross-view observation, so selecting it would give
    # the grader a payload it cannot answer from.
    snapshots = _full_body_snapshots()
    snapshots.append(_snap(9, "ego", "ego_only_pose"))
    selected = _paired_key_pose_labels({"intent": "full_body"}, snapshots)
    for label in selected:
        views = {str(s["view"]) for s in snapshots if str(s["label"]) == label}
        assert {"ego", "orbit"} <= views, f"{label} is not paired: {views}"
    assert "ego_only_pose" not in selected


def test_a_declared_label_is_honoured_when_the_evidence_carries_it() -> None:
    # The fallback must not override a real declared key pose -- that would
    # change what every gesture and strike clip is judged on.
    snapshots = [
        _snap(1, view, label)
        for label in ("present_start", "presented_pose")
        for view in ("ego", "orbit")
    ]
    assert _paired_key_pose_labels({"intent": "gesture"}, snapshots) == {"presented_pose"}


def test_the_selection_is_deterministic() -> None:
    snapshots = _full_body_snapshots()
    first = _paired_key_pose_labels({"intent": "full_body"}, snapshots)
    assert all(_paired_key_pose_labels({"intent": "full_body"}, snapshots) == first for _ in range(5))


def test_no_paired_snapshot_yields_no_selection_rather_than_a_guess() -> None:
    # Not-measured, not a wrong pick. The dispatch guard turns this into a raise
    # at the point of use rather than a blind call.
    single_view = [_snap(1, "ego", "only_ego")]
    assert _paired_key_pose_labels({"intent": "full_body"}, single_view) == set()
