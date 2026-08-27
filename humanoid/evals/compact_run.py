"""Plan 01 section 3.7 — archive a finished run's evidence without voiding its contract.

`evals/prune_run.py` handles the `ResultStore` half of retention: a losing candidate keeps
the record of what it was and loses the two artifacts that exist to be replayed. This tool
handles the other half, the evidence PNGs, which are the bulk of a run on disk.

Two rules shape the whole module.

**The PNG digest is the contract and it never moves.** `judge.py` verifies every snapshot
against `sha256` before it will look at the pixels, and 05 section 3.3 deliberately emits
`pixel_sha256` under the same value. Compaction records the WebP digest *beside* those and
leaves them pointing at the bytes that were judged, so a compacted manifest still states
what the judge saw rather than what the archive kept.

**`payloads/images/` is out of scope, deliberately.** Those bytes are the ones actually
dispatched to a model, and definition-of-done item 3 is that any call can be replayed from
its `request.json`. A lossy transcode there would break replay for a saving that section
3.7 does not even claim -- the payload column is 1-3 MB against evidence's tens of MB.

Compaction destroys the lossless originals, so like `prune_run.py` it reports by default
and acts only under `--apply`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from rigby_poc.observability import retry_on_transient_lock
from rigby_poc.io_utils import atomic_write_json

from .prune_run import RunNotTerminal, TERMINAL_STATUSES


#: Section 3.7 names q88. Measured on 26 real 1600x900 snapshots: 385 KB -> 24.8 KB,
#: a 15.6x reduction. q80 reaches 23.5x and q95 only 8.2x; 88 is kept because the
#: archive's purpose is that a human can still see what the judge saw.
WEBP_QUALITY = 88
#: Pillow's WebP effort knob. 6 is the slowest and smallest; evidence is transcoded once
#: and read many times, so the asymmetry is worth it.
WEBP_METHOD = 6
COMPACTION_SCHEMA_VERSION = "1.0"
#: Section 3.7: "compact runs older than 7 days, keep transcripts indefinitely."
DEFAULT_AGE_DAYS = 7.0
EVIDENCE_MANIFEST = "evidence-manifest.json"


class RunTooYoung(RuntimeError):
    """The run is terminal but inside the retention window."""


@dataclass
class SnapshotPlan:
    """One PNG that would become a WebP."""

    manifest_path: Path
    snapshot_id: str
    png_path: Path
    webp_path: Path

    @property
    def png_bytes(self) -> int:
        return self.png_path.stat().st_size if self.png_path.is_file() else 0


@dataclass
class CompactionPlan:
    run_id: str
    run_dir: Path
    age_days: float
    manifests: list[Path] = field(default_factory=list)
    snapshots: list[SnapshotPlan] = field(default_factory=list)
    already_compacted: list[Path] = field(default_factory=list)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(snapshot.png_bytes for snapshot in self.snapshots)

    def describe(self) -> str:
        lines = [
            f"run {self.run_id}",
            f"  age: {self.age_days:.2f} days",
            f"  manifests: {len(self.manifests)} "
            f"({len(self.already_compacted)} already compacted)",
            f"  snapshots to transcode: {len(self.snapshots)}",
            f"  reclaimable: {self.reclaimable_bytes / 1_048_576:.2f} MiB",
        ]
        return "\n".join(lines)


def transcode_to_webp(
    png: bytes, *, quality: int = WEBP_QUALITY
) -> tuple[bytes, int, int]:
    """Return `(webp_bytes, width, height)` for one lossless capture.

    The dimensions come back out because section 5 requires the transcode to preserve
    them and the only honest way to assert that is to read them off the *encoded* result
    rather than off the source that went in.

    Capture writes RGBA and every pixel of the alpha plane is opaque -- measured 255/255
    on 26 of 26 snapshots -- so the channel carries no information. Pillow drops it on a
    lossy WebP save, which is why a round-tripped image comes back `RGB`. That is a mode
    change, not a content change, and the assertion below is on size for that reason.
    """

    source = Image.open(io.BytesIO(png))
    source.load()
    buffer = io.BytesIO()
    source.save(buffer, format="WEBP", quality=quality, method=WEBP_METHOD)
    encoded = buffer.getvalue()
    written = Image.open(io.BytesIO(encoded))
    written.load()
    if written.size != source.size:
        raise ValueError(
            f"WebP transcode changed dimensions {source.size} -> {written.size}"
        )
    return encoded, written.width, written.height


def _run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no run.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a run record")
    return value


def run_age_days(record: dict[str, Any], *, now: datetime | None = None) -> float:
    """How long the run has been in its terminal state, in days."""
    stamp = str(record.get("updated_at") or record.get("created_at") or "")
    if not stamp:
        raise ValueError("run record carries no timestamp")
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    return (reference - moment).total_seconds() / 86400.0


def is_compacted(manifest: dict[str, Any]) -> bool:
    """Whether this evidence manifest describes an archived, non-judgeable run."""
    return isinstance(manifest.get("compaction"), dict)


def find_manifests(run_dir: Path) -> list[Path]:
    """Every evidence manifest under a run, in a stable order."""
    return sorted(run_dir.rglob(EVIDENCE_MANIFEST))


def plan_compaction(
    run_dir: Path,
    *,
    older_than_days: float = DEFAULT_AGE_DAYS,
    now: datetime | None = None,
) -> CompactionPlan:
    """Decide what a finished run may archive, without touching anything."""
    record = _run_record(run_dir)
    status = str(record.get("status", ""))
    if status not in TERMINAL_STATUSES:
        raise RunNotTerminal(f"run {record.get('run_id')} is {status!r}, not terminal")
    age = run_age_days(record, now=now)
    if age < older_than_days:
        raise RunTooYoung(
            f"run {record.get('run_id')} is {age:.2f} days old; "
            f"retention keeps evidence for {older_than_days:g}"
        )

    plan = CompactionPlan(
        run_id=str(record.get("run_id", run_dir.name)), run_dir=run_dir, age_days=age
    )
    for manifest_path in find_manifests(run_dir):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            continue
        plan.manifests.append(manifest_path)
        if is_compacted(manifest):
            plan.already_compacted.append(manifest_path)
            continue
        for snapshot in manifest.get("snapshots", []):
            if not isinstance(snapshot, dict):
                continue
            name = str(snapshot.get("path", ""))
            # The manifest stores a bare filename by construction (05 section 3.3) and it
            # is data, so it is validated as a name rather than trusted as a path -- the
            # same containment rule `prune_run.py` applies before it deletes.
            if not name or Path(name).name != name:
                continue
            png_path = manifest_path.parent / name
            if not png_path.is_file():
                continue
            plan.snapshots.append(
                SnapshotPlan(
                    manifest_path=manifest_path,
                    snapshot_id=str(snapshot.get("id", name)),
                    png_path=png_path,
                    webp_path=png_path.with_suffix(".webp"),
                )
            )
    return plan


def apply_compaction(
    plan: CompactionPlan,
    *,
    quality: int = WEBP_QUALITY,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Transcode, rewrite each manifest, drop the PNGs. Returns bytes reclaimed.

    Ordering is deliberate: every snapshot in a manifest is transcoded and the manifest is
    rewritten *before* any PNG is unlinked. A crash midway therefore leaves the originals
    on disk beside a manifest that has not yet claimed they are gone, which the idempotence
    path then finishes. The reverse order would lose evidence to a partial write.
    """

    clock = now or (lambda: datetime.now(UTC))
    reclaimed = 0
    by_manifest: dict[Path, list[SnapshotPlan]] = {}
    for snapshot in plan.snapshots:
        by_manifest.setdefault(snapshot.manifest_path, []).append(snapshot)

    for manifest_path, snapshots in by_manifest.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if is_compacted(manifest):
            continue
        digests: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            png = snapshot.png_path.read_bytes()
            encoded, width, height = transcode_to_webp(png, quality=quality)
            retry_on_transient_lock(lambda: snapshot.webp_path.write_bytes(encoded))
            digests[snapshot.png_path.name] = {
                "compacted_path": snapshot.webp_path.name,
                "webp_sha256": hashlib.sha256(encoded).hexdigest(),
                "compacted_width_px": width,
                "compacted_height_px": height,
            }

        for snapshot in manifest.get("snapshots", []):
            if isinstance(snapshot, dict):
                extra = digests.get(str(snapshot.get("path", "")))
                if extra:
                    # `sha256` and `pixel_sha256` are untouched on purpose: they name the
                    # bytes the judge verified, and a compacted manifest that renamed them
                    # to the archive's digest would silently restate history.
                    snapshot.update(extra)
        manifest["compaction"] = {
            "schema_version": COMPACTION_SCHEMA_VERSION,
            "format": "webp",
            "quality": quality,
            "source_format": "png",
            "compacted_at": clock().isoformat(),
            "snapshots": len(digests),
            # Stated in the artifact rather than in the tool's output, so the caveat
            # travels with the number it qualifies.
            "judgeable": False,
        }
        atomic_write_json(manifest_path, manifest)

        for snapshot_plan in snapshots:
            if snapshot_plan.png_path.is_file():
                reclaimed += snapshot_plan.png_path.stat().st_size
                retry_on_transient_lock(
                    lambda: snapshot_plan.png_path.unlink(missing_ok=True)
                )
    return reclaimed


@dataclass
class SweepEntry:
    """One run the sweep considered, and what it decided."""

    run_dir: Path
    plan: CompactionPlan | None
    skipped: str | None = None


def is_run_dir(path: Path) -> bool:
    return (path / "run.json").is_file()


def sweep(
    root: Path,
    *,
    older_than_days: float = DEFAULT_AGE_DAYS,
    now: datetime | None = None,
) -> list[SweepEntry]:
    """Apply the retention policy across every run under `root`.

    A per-run tool with a seven-day default is not yet a policy -- somebody has to
    remember to invoke it, once per run, forever, which is the same as not having one.
    This is the operable half.

    A run that is not eligible is *reported*, never raised on: a sweep that aborts at the
    first still-running run would compact nothing on any tree that has a live run in it,
    which is every tree worth sweeping.
    """

    entries: list[SweepEntry] = []
    for child in sorted(root.iterdir() if root.is_dir() else []):
        if not child.is_dir() or not is_run_dir(child):
            continue
        try:
            entries.append(
                SweepEntry(
                    run_dir=child,
                    plan=plan_compaction(child, older_than_days=older_than_days, now=now),
                )
            )
        except (RunNotTerminal, RunTooYoung) as reason:
            entries.append(SweepEntry(run_dir=child, plan=None, skipped=str(reason)))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as reason:
            # A malformed run must not stop the sweep, but it must not be silent either.
            entries.append(SweepEntry(run_dir=child, plan=None, skipped=f"unreadable: {reason}"))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="one run directory, or a root holding many (results/pipeline-runs/)",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=DEFAULT_AGE_DAYS,
        help=f"retention window; default {DEFAULT_AGE_DAYS:g}",
    )
    parser.add_argument("--quality", type=int, default=WEBP_QUALITY)
    parser.add_argument("--apply", action="store_true", help="transcode, rather than report")
    arguments = parser.parse_args()

    if is_run_dir(arguments.run_dir):
        plans = [plan_compaction(arguments.run_dir, older_than_days=arguments.older_than_days)]
    else:
        entries = sweep(arguments.run_dir, older_than_days=arguments.older_than_days)
        for entry in entries:
            if entry.skipped:
                print(f"skip {entry.run_dir.name}: {entry.skipped}")
        plans = [entry.plan for entry in entries if entry.plan is not None]
        if not plans:
            print(f"no eligible runs under {arguments.run_dir}")
            return

    for plan in plans:
        print(plan.describe())
    total = sum(plan.reclaimable_bytes for plan in plans)
    if len(plans) > 1:
        print(f"\n{len(plans)} runs, {total / 1_048_576:.2f} MiB reclaimable")
    if not arguments.apply:
        print("\nreport only; pass --apply to transcode and drop the PNGs")
        return
    reclaimed = sum(apply_compaction(plan, quality=arguments.quality) for plan in plans)
    print(f"\nreclaimed {reclaimed / 1_048_576:.2f} MiB")


if __name__ == "__main__":
    main()
