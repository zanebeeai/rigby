"""Plan 01 section 5 — the compaction tool's tests.

The four the plan names are here: dimensions survive the transcode, both hashes are
recorded, compaction is idempotent, and a compacted run still loads. The rest exist
because compaction *deletes lossless originals*, and every gate that stands between a
running pipeline and that deletion deserves a test that fails when the gate is removed.

Fixtures are synthesised rather than rendered. A browser render is a `slow` dependency and
none of these assertions need real pixels -- they need an image with enough structure that
a lossy encoder does something non-degenerate with it, which a gradient plus mild noise
supplies. The one claim that *does* need real evidence is the size budget, and that is
reported as a measurement in the PR rather than pinned as a byte count here: libwebp's
output size is a property of the encoder version, so asserting it would pin the library,
not the behaviour. See `docs/testing.md` on preferring a structural relation.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from evals.compact_run import (
    DEFAULT_AGE_DAYS,
    CompactionPlan,
    RunTooYoung,
    apply_compaction,
    is_compacted,
    is_run_dir,
    plan_compaction,
    sweep,
    transcode_to_webp,
)
from evals.prune_run import RunNotTerminal


pytestmark = pytest.mark.fast

WIDTH, HEIGHT = 320, 180


def _png(seed: int, *, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """An RGBA PNG with structure, mirroring what capture writes.

    Capture emits RGBA with a fully opaque alpha plane -- measured 255/255 on 26 of 26
    real snapshots -- so the fixture does too. That detail is load-bearing: a lossy WebP
    save drops the channel, and a test built on a fixture without it would not exercise
    the mode change at all.
    """

    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 7 + seed * 13) % 256,
                (y * 5 + seed * 29) % 256,
                (x * y + seed) % 256,
                255,
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_run(
    root: Path,
    *,
    status: str = "completed",
    age_days: float = 30.0,
    snapshots: int = 3,
    candidates: int = 1,
    name: str = "20260101T000000-fixture",
) -> Path:
    """A run directory shaped like the real thing: run.json plus nested evidence dirs.

    `root` is the *parent* of the run, mirroring results/pipeline-runs/<run_id>/, so a
    sweep over `root` sees run directories exactly one level down as it does in production.
    """
    run_dir = root / name
    stamp = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": status,
                "created_at": stamp,
                "updated_at": stamp,
                "winner_result_id": "000001-fixture",
            }
        ),
        encoding="utf-8",
    )
    # A transcript must survive compaction untouched; section 3.7 keeps them indefinitely.
    (run_dir / "transcript.jsonl").write_text('{"kind":"run"}\n', encoding="utf-8")
    for candidate in range(1, candidates + 1):
        evidence = run_dir / "artifacts" / "round-1" / f"candidate-{candidate}-recipe"
        evidence.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index in range(1, snapshots + 1):
            data = _png(index + candidate * 100)
            name = f"{index:02d}-phase-orbit.png"
            (evidence / name).write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            records.append(
                {
                    "id": f"{index:02d}-orbit",
                    "label": "phase",
                    "view": "orbit",
                    "path": name,
                    "sha256": digest,
                    "pixel_sha256": digest,
                    "pose_sha256": hashlib.sha256(f"pose{index}".encode()).hexdigest(),
                    "width_px": WIDTH,
                    "height_px": HEIGHT,
                    "camera": {},
                }
            )
        (evidence / "evidence-manifest.json").write_text(
            json.dumps({"schema_version": "1.1", "result_id": "000001-fixture", "snapshots": records}),
            encoding="utf-8",
        )
    return run_dir


def _manifests(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.rglob("evidence-manifest.json"))
    ]


# -- the four the plan names ------------------------------------------------------------


def test_the_transcode_preserves_dimensions() -> None:
    png = _png(1)
    encoded, width, height = transcode_to_webp(png)
    assert (width, height) == (WIDTH, HEIGHT)
    # Read back off the encoded bytes, not off the source: the point is what the archive
    # contains, and a helper that returned its own input's size would assert nothing.
    written = Image.open(io.BytesIO(encoded))
    written.load()
    assert written.size == (WIDTH, HEIGHT)
    assert encoded[:4] == b"RIFF" and encoded[8:12] == b"WEBP"


def test_both_hashes_are_recorded(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    before = _manifests(run_dir)[0]["snapshots"]
    apply_compaction(plan_compaction(run_dir))
    after = _manifests(run_dir)[0]["snapshots"]
    for original, compacted in zip(before, after):
        assert compacted["sha256"] == original["sha256"]
        assert compacted["pixel_sha256"] == original["pixel_sha256"]
        assert compacted["webp_sha256"] != compacted["sha256"]
        archived = (
            next(iter(run_dir.rglob(compacted["compacted_path"])))
        ).read_bytes()
        assert hashlib.sha256(archived).hexdigest() == compacted["webp_sha256"]


def test_compaction_is_idempotent(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    apply_compaction(plan_compaction(run_dir))
    first = _manifests(run_dir)[0]
    webps = sorted(p.name for p in run_dir.rglob("*.webp"))

    second_plan = plan_compaction(run_dir)
    assert second_plan.snapshots == []
    assert len(second_plan.already_compacted) == 1
    assert apply_compaction(second_plan) == 0

    assert _manifests(run_dir)[0] == first
    assert sorted(p.name for p in run_dir.rglob("*.webp")) == webps


def test_a_compacted_run_still_loads(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    apply_compaction(plan_compaction(run_dir))
    manifest = _manifests(run_dir)[0]
    # Still a manifest: schema, result id and every snapshot record survive.
    assert manifest["schema_version"] == "1.1"
    assert manifest["result_id"] == "000001-fixture"
    assert len(manifest["snapshots"]) == 3
    assert is_compacted(manifest)
    # Section 3.7 keeps transcripts indefinitely; compaction must not have touched it.
    assert (run_dir / "transcript.jsonl").read_text(encoding="utf-8") == '{"kind":"run"}\n'
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "completed"


# -- the gates that stand in front of a deletion ----------------------------------------


def test_a_running_run_is_refused(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, status="running")
    with pytest.raises(RunNotTerminal):
        plan_compaction(run_dir)
    assert len(list(run_dir.rglob("*.png"))) == 3


def test_a_run_inside_the_retention_window_is_refused(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, age_days=DEFAULT_AGE_DAYS - 1.0)
    with pytest.raises(RunTooYoung):
        plan_compaction(run_dir)
    assert len(list(run_dir.rglob("*.png"))) == 3


def test_planning_alone_deletes_nothing(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    plan = plan_compaction(run_dir)
    assert len(plan.snapshots) == 3
    assert plan.reclaimable_bytes > 0
    assert len(list(run_dir.rglob("*.png"))) == 3
    assert list(run_dir.rglob("*.webp")) == []
    assert not is_compacted(_manifests(run_dir)[0])


def test_every_candidate_directory_is_compacted(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, candidates=3, snapshots=2)
    plan = plan_compaction(run_dir)
    assert len(plan.manifests) == 3
    assert len(plan.snapshots) == 6
    apply_compaction(plan)
    assert list(run_dir.rglob("*.png")) == []
    assert len(list(run_dir.rglob("*.webp"))) == 6
    assert all(is_compacted(manifest) for manifest in _manifests(run_dir))


def test_the_archive_is_smaller_than_what_it_replaced(tmp_path: Path) -> None:
    """A structural relation, not a byte count: libwebp's exact output is its own version's."""
    run_dir = _build_run(tmp_path)
    plan = plan_compaction(run_dir)
    png_bytes = plan.reclaimable_bytes
    apply_compaction(plan)
    webp_bytes = sum(path.stat().st_size for path in run_dir.rglob("*.webp"))
    assert webp_bytes < png_bytes


def test_a_snapshot_path_that_escapes_the_evidence_directory_is_skipped(tmp_path: Path) -> None:
    """The manifest is data. `prune_run.py` refuses to treat it as a path and so does this.

    The escape target must **exist** for this to test anything. An earlier version pointed
    at a path with no file behind it, so the plan skipped it on `is_file()` and the test
    passed with the containment check deleted -- green for a reason unrelated to the guard
    it was named after. Mutation testing is what surfaced that; the file below is the fix.
    """

    run_dir = _build_run(tmp_path, snapshots=1)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png(99))
    before = outside.read_bytes()

    manifest_path = next(iter(run_dir.rglob("evidence-manifest.json")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # ../../../.. walks candidate -> round-1 -> artifacts -> run_dir -> tmp_path.
    manifest["snapshots"][0]["path"] = "../../../../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (manifest_path.parent / "../../../../outside.png").resolve() == outside.resolve()
    plan = plan_compaction(run_dir)
    assert plan.snapshots == []
    apply_compaction(plan)
    assert outside.is_file() and outside.read_bytes() == before
    assert not (tmp_path / "outside.webp").exists()


def test_re_applying_a_stale_plan_is_a_no_op(tmp_path: Path) -> None:
    """The guard inside `apply_compaction`, which the planning-level skip hides.

    Planning a run twice already yields an empty second plan, so the idempotence guard in
    `apply_compaction` is unreachable by that route -- deleting it left the idempotence
    test green. The route that does reach it is a plan applied twice: the PNGs named in it
    are gone after the first pass, and without the guard the second read raises.
    """

    run_dir = _build_run(tmp_path)
    plan = plan_compaction(run_dir)
    apply_compaction(plan)
    manifest = _manifests(run_dir)[0]

    assert apply_compaction(plan) == 0
    assert _manifests(run_dir)[0] == manifest
    assert len(list(run_dir.rglob("*.webp"))) == 3


def test_the_plan_reports_an_empty_run_without_failing(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, snapshots=0)
    plan = plan_compaction(run_dir)
    assert isinstance(plan, CompactionPlan)
    assert plan.snapshots == []
    assert plan.reclaimable_bytes == 0
    assert "0.00 MiB" in plan.describe()


# -- the retention policy, swept across a tree ------------------------------------------


def test_the_sweep_skips_what_it_may_not_touch_and_says_why(tmp_path: Path) -> None:
    """A sweep that raised on the first live run would compact nothing on a real tree."""
    root = tmp_path / "pipeline-runs"
    root.mkdir()
    _build_run(root, name="old", age_days=30.0)
    _build_run(root, name="young", age_days=DEFAULT_AGE_DAYS - 1.0)
    _build_run(root, name="live", status="running", age_days=30.0)
    (root / "not-a-run").mkdir()

    entries = sweep(root)
    by_name = {entry.run_dir.name: entry for entry in entries}
    assert set(by_name) == {"old", "young", "live"}
    assert by_name["old"].plan is not None and by_name["old"].skipped is None
    assert by_name["young"].plan is None and "retention" in (by_name["young"].skipped or "")
    assert by_name["live"].plan is None and "not terminal" in (by_name["live"].skipped or "")


def test_the_sweep_compacts_only_the_eligible_run(tmp_path: Path) -> None:
    root = tmp_path / "pipeline-runs"
    root.mkdir()
    _build_run(root, name="old", age_days=30.0)
    _build_run(root, name="young", age_days=1.0)

    for entry in sweep(root):
        if entry.plan is not None:
            apply_compaction(entry.plan)

    assert list((root / "old").rglob("*.png")) == []
    assert len(list((root / "old").rglob("*.webp"))) == 3
    # The run inside the window keeps every lossless original.
    assert len(list((root / "young").rglob("*.png"))) == 3
    assert list((root / "young").rglob("*.webp")) == []


def test_a_run_directory_is_told_apart_from_a_root_of_them(tmp_path: Path) -> None:
    root = tmp_path / "pipeline-runs"
    root.mkdir()
    run_dir = _build_run(root, name="old", age_days=30.0)
    assert is_run_dir(run_dir)
    assert not is_run_dir(root)


def test_an_unreadable_run_does_not_stop_the_sweep(tmp_path: Path) -> None:
    root = tmp_path / "pipeline-runs"
    root.mkdir()
    _build_run(root, name="good", age_days=30.0)
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{not json", encoding="utf-8")

    entries = sweep(root)
    assert any(entry.plan is not None for entry in entries)
    assert any("unreadable" in (entry.skipped or "") for entry in entries)
