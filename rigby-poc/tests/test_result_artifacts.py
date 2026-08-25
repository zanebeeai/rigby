from __future__ import annotations

import json
from pathlib import Path

from evals.glb import check_glb
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner
from rigby_poc.store import ResultStore

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


ROOT = Path(__file__).resolve().parents[1]


def test_representative_persisted_results_are_complete_and_roundtrip(
    tmp_path: Path,
) -> None:
    profile = json.loads((ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json").read_text(encoding="utf-8"))
    aliases = {source: canonical for canonical, source in profile["bone_map"].items()}
    scene = default_scene()
    store = ResultStore(tmp_path / "results")
    for prompt in (
        "Throw up a hang-ten sign with your right hand.",
        "Step over the hurdle with your right foot.",
    ):
        program = OfflinePlanner().plan(
            PlanRequest(text=prompt, scene=scene, provider="offline")
        ).program
        request = CompileRequest(scene=scene, program=program, persist=True)
        clip = compile_motion(request)
        assert clip.success, clip.failure
        store.persist(request, clip)

    result_dirs = sorted(
        path
        for path in (tmp_path / "results").glob(
            "[0-9][0-9][0-9][0-9][0-9][0-9]-*"
        )
        if path.is_dir()
    )
    assert len(result_dirs) == 2
    required = {"request.json", "scene.json", "program.json", "clip.json", "metrics.json", "provenance.json"}
    complete_provenance = 0
    successful_exports = 0
    for result_dir in result_dirs:
        assert required <= {path.name for path in result_dir.iterdir()}
        clip = json.loads((result_dir / "clip.json").read_text(encoding="utf-8"))
        animation = result_dir / "animation.glb"
        assert clip.get("success")
        assert animation.is_file(), f"{result_dir.name} is a successful clip without a GLB"
        check = check_glb(
            animation.read_bytes(), clip,
            translation_tolerance=0.0001, rotation_tolerance=0.0001,
            node_aliases=aliases,
        )
        assert check.valid
        assert check.compared_samples > 0
        transform_failures = [failure for failure in check.failures if "incomplete Rigby provenance" not in failure]
        assert transform_failures == []
        complete_provenance += int(check.has_provenance)
        successful_exports += 1
    assert successful_exports == 2
    assert complete_provenance == 2


@pytest.fixture(scope="module")
def compiled() -> tuple[CompileRequest, object]:
    """One compile, reused by every crash-safety case.

    These cases are about the write path, not about the clip, and a compile is
    the expensive part of this file.
    """

    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text="Throw up a hang-ten sign with your right hand.", scene=scene, provider="offline")
    ).program
    request = CompileRequest(scene=scene, program=program, persist=True)
    clip = compile_motion(request)
    assert clip.success, clip.failure
    return request, clip


def _sequences(root: Path) -> list[int]:
    return sorted(int(path.name[:6]) for path in root.iterdir() if path.is_dir())


def test_a_killed_run_does_not_poison_every_later_persist(
    tmp_path: Path, compiled: tuple[CompileRequest, object]
) -> None:
    """`persist` updates the index last, so a kill leaves an unindexed folder.

    `next_sequence` still points at that id and nothing but a successful persist
    advances it, so every later persist collided on the same id -- permanently,
    until a human deleted the directory. The worktree could not pass the suite
    in the meantime and the `FileExistsError` named neither the store nor the
    kill.
    """

    request, clip = compiled
    root = tmp_path / "results"
    store = ResultStore(root)
    first = store.persist(request, clip)
    assert first.startswith("000001-")

    orphan = root / f"000002-{ResultStore._slug(request.program.source_text)}"
    orphan.mkdir()
    (orphan / "request.json").write_text("{}", encoding="utf-8")

    second = store.persist(request, clip)
    assert second.startswith("000003-"), f"reused an id already on disk: {second}"
    assert (root / second).is_dir()


def test_the_orphaned_directory_is_left_exactly_as_the_kill_left_it(
    tmp_path: Path, compiled: tuple[CompileRequest, object]
) -> None:
    """`exist_ok=True` is the wrong fix and this is why.

    Merging into the dead run's directory would leave seven files describing two
    different runs, which is a silent corruption in place of a loud failure. A
    killed run's directory is of unknown completeness; whether to delete it is
    `prune_run.py`'s decision, not the write path's.
    """

    request, clip = compiled
    root = tmp_path / "results"
    store = ResultStore(root)
    store.persist(request, clip)

    orphan = root / f"000002-{ResultStore._slug(request.program.source_text)}"
    orphan.mkdir()
    (orphan / "request.json").write_text('{"killed": true}', encoding="utf-8")

    store.persist(request, clip)

    assert {path.name for path in orphan.iterdir()} == {"request.json"}
    assert json.loads((orphan / "request.json").read_text(encoding="utf-8")) == {"killed": True}


def test_an_orphan_from_a_different_prompt_does_not_duplicate_a_sequence_number(
    tmp_path: Path, compiled: tuple[CompileRequest, object]
) -> None:
    """The quieter half of the same bug.

    The directory name is the sequence number *and* a slug derived from the
    prompt, so an orphan left by a different prompt does not collide on `mkdir`
    at all. It silently produced two directories sharing one sequence number,
    with the index claiming that number for whichever wrote second. Healing has
    to key on the number, not on the whole name.
    """

    request, clip = compiled
    root = tmp_path / "results"
    store = ResultStore(root)
    store.persist(request, clip)

    (root / "000002-some-other-prompt-entirely").mkdir()

    third = store.persist(request, clip)
    assert not third.startswith("000002-")
    sequences = _sequences(root)
    assert len(sequences) == len(set(sequences)), f"two directories share a sequence: {sequences}"


def test_ids_stay_contiguous_when_nothing_was_orphaned(
    tmp_path: Path, compiled: tuple[CompileRequest, object]
) -> None:
    """Healing must not become a habit of skipping ids on every write."""

    request, clip = compiled
    store = ResultStore(tmp_path / "results")
    assert [store.persist(request, clip)[:6] for _ in range(3)] == ["000001", "000002", "000003"]
