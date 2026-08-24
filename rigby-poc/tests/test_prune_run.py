"""Plan 05 section 3.7 — the `ResultStore` half of the retention policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.prune_run import (
    PRUNABLE_FILES,
    RETAINED_FILES,
    RunNotTerminal,
    apply_prune,
    candidate_result_ids,
    plan_prune,
)


def _store(root: Path, result_ids: list[str]) -> None:
    for result_id in result_ids:
        folder = root / result_id
        folder.mkdir(parents=True)
        for name in RETAINED_FILES:
            (folder / name).write_text("{}", encoding="utf-8")
        (folder / "clip.json").write_text("x" * 4096, encoding="utf-8")
        (folder / "animation.glb").write_bytes(b"y" * 2048)


def _run(root: Path, *, status: str, winner: str | None, candidates: list[str]) -> Path:
    run_dir = root / "20260824T120000-deadbeef"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {"run_id": "20260824T120000-deadbeef", "status": status, "winner_result_id": winner}
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "flywheel-trace.json").write_text(
        json.dumps(
            {
                "winner_result_id": winner,
                "rounds": [
                    {"candidates": [{"result_id": result_id} for result_id in candidates]}
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


ALL = ["000001-a", "000002-b", "000003-c"]


def test_a_finished_run_prunes_every_candidate_except_the_winner(tmp_path: Path) -> None:
    store = tmp_path / "results"
    _store(store, ALL)
    run_dir = _run(tmp_path, status="completed", winner="000002-b", candidates=ALL)

    plan = plan_prune(run_dir, store_root=store)
    assert plan.winner_result_id == "000002-b"
    assert plan.reclaimable_bytes == 2 * (4096 + 2048)

    reclaimed = apply_prune(plan)
    assert reclaimed == 2 * (4096 + 2048)
    for result_id in ("000001-a", "000003-c"):
        for name in PRUNABLE_FILES:
            assert not (store / result_id / name).exists()
        # What the candidate *was* survives; only what exists to be replayed is dropped.
        for name in RETAINED_FILES:
            assert (store / result_id / name).is_file()
    for name in (*RETAINED_FILES, *PRUNABLE_FILES):
        assert (store / "000002-b" / name).is_file()


def test_a_run_that_is_still_going_is_refused(tmp_path: Path) -> None:
    """Pruning a live run would race the process still writing to it."""
    store = tmp_path / "results"
    _store(store, ALL)
    run_dir = _run(tmp_path, status="running", winner=None, candidates=ALL)
    with pytest.raises(RunNotTerminal, match="running"):
        plan_prune(run_dir, store_root=store)


def test_planning_deletes_nothing(tmp_path: Path) -> None:
    """The default path is a report; `--apply` is the only thing that removes a file."""
    store = tmp_path / "results"
    _store(store, ALL)
    run_dir = _run(tmp_path, status="completed", winner="000002-b", candidates=ALL)
    plan_prune(run_dir, store_root=store)
    for result_id in ALL:
        for name in PRUNABLE_FILES:
            assert (store / result_id / name).is_file()


def test_explicitly_protected_results_survive(tmp_path: Path) -> None:
    store = tmp_path / "results"
    _store(store, ALL)
    run_dir = _run(tmp_path, status="completed", winner="000002-b", candidates=ALL)
    plan = plan_prune(run_dir, store_root=store, protect=["000003-c"])
    assert plan.protected == ["000002-b", "000003-c"]
    apply_prune(plan)
    assert (store / "000003-c" / "clip.json").is_file()
    assert not (store / "000001-a" / "clip.json").exists()


def test_a_run_with_no_winner_still_prunes_its_losers(tmp_path: Path) -> None:
    """A run that accepted nothing is exactly the run whose candidates are dead weight."""
    store = tmp_path / "results"
    _store(store, ALL)
    run_dir = _run(tmp_path, status="no_acceptable_candidate", winner=None, candidates=ALL)
    plan = plan_prune(run_dir, store_root=store)
    assert plan.winner_result_id is None
    assert len(plan.prunable) == len(ALL) * len(PRUNABLE_FILES)


def test_a_result_id_cannot_point_outside_the_store(tmp_path: Path) -> None:
    """Trace content is data. A pruning tool must not be talked out of its own root."""
    store = tmp_path / "results"
    _store(store, ALL)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "clip.json").write_text("precious", encoding="utf-8")
    run_dir = _run(
        tmp_path, status="completed", winner="000002-b", candidates=[*ALL, "../elsewhere"]
    )
    plan = plan_prune(run_dir, store_root=store)
    apply_prune(plan)
    assert (outside / "clip.json").is_file()


def test_candidate_ids_are_collected_in_trace_order_without_duplicates() -> None:
    trace = {
        "rounds": [
            {"candidates": [{"result_id": "a"}, {"result_id": "b"}]},
            {"candidates": [{"result_id": "b"}, {"result_id": "c"}, {"not_a_result": 1}]},
        ]
    }
    assert candidate_result_ids(trace) == ["a", "b", "c"]
