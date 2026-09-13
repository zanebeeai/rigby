"""Package complete numeric evidence and make a labelled six-body preview."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.evidence_archive import pack_bundle

from .capture import ROOT, json_bytes
from .render import BANNER, HEIGHT, PREVIEW_LIMIT, WIDTH


def _source_matches_commit(source: dict) -> bool:
    names = list(source["package_source_sha256"])
    archive = subprocess.check_output(["git", "archive", "--format=tar", source["commit"], *names], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        return all(
            hashlib.sha256(tree.extractfile(name).read()).hexdigest() == digest
            for name, digest in source["package_source_sha256"].items()
        )


def publish_release(suite: Path, destination: Path) -> dict:
    index = json.loads((suite / "index.json").read_bytes())
    if not index.get("complete") or not index.get("cases"):
        raise ValueError("Publish only a completed evidence suite")
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    source_validation = {}
    for original in index["cases"]:
        label = original["robot_id"] + ("-actuation-loss" if original["fault"] else "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Nonportable release case identifier")
        physical = suite / label / "physical"
        manifest = verify_bundle(physical, original["sha256"])
        case = destination / label
        case.mkdir()
        archive = case / "physical.zip"
        archive_hash = pack_bundle(physical, archive, original["sha256"])
        source = json.loads((physical / "source.json").read_bytes())
        outcome = json.loads((physical / "outcome.json").read_bytes())
        source_key = hashlib.sha256(json_bytes(source["package_source_sha256"])).hexdigest()
        if source_key not in source_validation:
            source_validation[source_key] = _source_matches_commit(source)
        media_source = Path(original["media"]["bundle"].replace("\\", "/"))
        if not media_source.is_absolute():
            media_source = ROOT / media_source
        if not media_source.resolve().is_relative_to(suite.resolve()):
            raise ValueError("Presentation source is outside the supplied suite")
        media = verify_bundle(media_source, original["media"]["sha256"])
        if media["metadata"]["source_bundle_sha256"] != original["sha256"]:
            raise ValueError("Presentation belongs to a different physical episode")
        media_digest = write_bundle(
            case / "media", {name: (media_source / name).read_bytes() for name in media["files"]}, media["metadata"],
        )
        if media_digest != original["media"]["sha256"]:
            raise ValueError("Copying presentation evidence changed its digest")
        rows.append({
            "case": label, **manifest["metadata"], "physical_manifest_sha256": original["sha256"],
            "archive": archive.relative_to(destination).as_posix(), "archive_sha256": archive_hash,
            "archive_bytes": archive.stat().st_size,
            "media": (case / "media").relative_to(destination).as_posix(), "media_manifest_sha256": media_digest,
            "video_frames": media["metadata"]["frame_count"], "video_duration_s": media["metadata"]["playback_duration_s"],
            "replay": original["replay"], "source_commit": source["commit"],
            "refusal": outcome["refusal"],
            "package_sources_match_recorded_commit": source_validation[source_key],
            "package_source_map_sha256": source_key,
            "tracked_source_patch_sha256": hashlib.sha256(b"").hexdigest() if source_validation[source_key] else None,
        })
    result = {
        "schema": "rigby.evidence-release/1", "goal": "G01", "cases": rows,
        "provenance_note": "Package source bytes were independently compared with the recorded commit. Dirty flags also include untracked results; the empty source patch applies only when all archived package sources match that commit.",
        "scope": "Five successful canonical motions, one canonical compilation refusal, and one declared runtime diagnostic failure. No manipulation, locomotion or robustness claim.",
    }
    (destination / "index.json").write_bytes(json_bytes(result))
    make_preview(destination)
    return result


def make_preview(release: Path) -> dict:
    """Show independent timelines at normalized progress, with exact timestamps."""
    index = json.loads((release / "index.json").read_bytes())
    rows = [row for row in index["cases"] if not row["fault"]]
    if len(rows) != 6:
        raise ValueError("This research preview requires exactly six canonical bodies")
    images, maps, selected = [], [], []
    font, small = ImageFont.load_default(size=15), ImageFont.load_default(size=13)
    width, height = 2 * WIDTH, 64 + 3 * (HEIGHT + 42)
    frames, frame_sources = [], []
    try:
        for row in rows:
            media = release / row["media"]
            verify_bundle(media, row["media_manifest_sha256"])
            images.append(Image.open(media / "preview.gif"))
            mapping = json.loads((media / "frames.json").read_bytes())
            maps.append(mapping)
            selected.append(np.linspace(0, len(mapping) - 1, min(PREVIEW_LIMIT, len(mapping))).astype(int))
        for i in range(60):
            canvas = Image.new("RGB", (width, height), (17, 24, 39))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), "SIX BODIES | reach out as far as you can and then come back", font=font, fill="white")
            draw.text((12, 32), "GIF SUMMARY: independent timelines normalized to six seconds; exact simulation time per cell", font=small, fill=(191, 210, 233))
            sources = []
            for j, row in enumerate(rows):
                frame = round(i * (images[j].n_frames - 1) / 59)
                images[j].seek(frame)
                # Preserve the original global-view pixels; no geometric rescaling.
                tile = images[j].convert("RGB").crop((0, BANNER, WIDTH, BANNER + HEIGHT))
                x, y = (j % 2) * WIDTH, 64 + (j // 2) * (HEIGHT + 42)
                source_frame = int(selected[j][min(frame, len(selected[j]) - 1)])
                sample = maps[j][source_frame]
                status = row["outcome"].replace("pre_execution_refusal", "REFUSED before execution")
                tint = (100, 230, 165) if row["outcome"] == "success" else (255, 189, 100)
                draw.text((x + 8, y + 3), f"{row['robot_id']} | {status}", font=font, fill=tint)
                caption = f"sim t={sample['simulation_time_s']:.3f}s / {row['simulation_duration_s']:.3f}s"
                if row["outcome"] == "pre_execution_refusal":
                    caption += " | " + row["refusal"]["code"]
                draw.text((x + 8, y + 23), caption, font=small, fill=(191, 210, 233))
                canvas.paste(tile, (x, y + 42))
                sources.append({"case": row["case"], "media_sha256": row["media_manifest_sha256"], "source_video_frame": source_frame, "simulation_time_s": sample["simulation_time_s"]})
            frames.append(canvas)
            frame_sources.append({"preview_frame": i, "normalized_progress": i / 59, "sources": sources})
        initial_pixels = np.asarray(frames[0], dtype=np.int16)
        overview_frame = max(range(len(frames)), key=lambda i: int(np.abs(np.asarray(frames[i], dtype=np.int16) - initial_pixels).sum()))
        frames[overview_frame].save(release / "overview.png")
        frames[0].save(release / "six-body-preview.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    finally:
        for image in images:
            image.close()
    result = {
        "schema": "rigby.comparison-preview/1", "summary_only": True, "duration_s": 6.0,
        "timeline": "independent episodes normalized by progress, including an unexecuted refusal",
        "overview_frame": overview_frame,
        "overview_selection": "Largest pixel difference from the initial comparison frame; presentation selection only, all episodes and all preview frames retained.",
        "files": {name: hashlib.sha256((release / name).read_bytes()).hexdigest() for name in ("overview.png", "six-body-preview.gif")},
        "frames": frame_sources,
    }
    (release / "preview-manifest.json").write_bytes(json_bytes(result))
    return result
