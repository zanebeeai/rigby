"""D11: validate once, reuse in a new layout, invalidate after a physical-context change, repair.

Five panels on one clock, every one a sealed episode rendered from recorded
states (or a labelled slate where nothing ran): the jaw arm's validation
episode that promoted transfer_object; the same skill reused from the
persisted store in the mirrored layout; the retrieval refused under the
grown cube, with the differing dimension named; the revalidation episode
under the grown cube that issued the superseding certificate; and the
repaired skill reused again in the mirrored layout with the grown cube,
retrieved as version 2. The last panel is run here, from the persisted
store as the campaign left it; the others are the campaign's own clips.

    python any-robot/scripts/g11_d11_media.py --campaign docs/results/g11-store --local any-robot/results/g11-store --out docs/results/g11-d11
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rigby_core.skills import SkillStoreV1

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import BANNER, HEIGHT, PREVIEW_LIMIT, WIDTH, render_bundle
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import load_policy
from rigby_general.skills import TransferObjectSession, seal_tree_run
from rigby_general.skills.skill_store import EVIDENCE_SCHEMA, context_of, goal_for, run_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402
import g11_protocol as protocol  # noqa: E402
from g11_skill_store_campaign import ARGS, STORE, changed_world, episode_row, source_of  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FPS = 12
BACKGROUND = (17, 24, 39)
BODY = "zoo_jaw_arm"
LAYOUT = "mirrored"


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def slate(ffmpeg: str, destination: Path, lines: list[str], *, seconds: float = 3.0) -> dict:
    """A labelled still for something that, by design, did not run."""

    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    draw.text((12, 7), lines[0], fill=(255, 189, 100), font=font)
    for index, line in enumerate(lines[1:]):
        draw.text((12, 32 + 20 * index), line[:120], fill="white" if index < 2 else (191, 210, 233), font=font if index < 2 else small)
    draw.text((12, HEIGHT + BANNER - 24), "NO MOTION WAS EXECUTED | the store refused before any leaf ran | labelled slate, not a recording", fill=(170, 185, 205), font=small)
    frames = int(round(seconds * FPS))
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * WIDTH}x{HEIGHT + BANNER}", "-r", str(FPS), "-i", "pipe:0",
                                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], stdin=subprocess.PIPE)
    for _ in range(frames):
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait(timeout=120) != 0:
        raise RuntimeError("ffmpeg could not write the slate")
    canvas.save(destination.with_suffix(".png"))
    return {"video": destination.name, "frames": frames, "slate": True}


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], destination: Path) -> dict:
    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs, filters = [], []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(stream['duration'])):.3f},scale=trunc(iw*2/5/2)*2:trunc(ih*2/5/2)*2[v{index}]")
    layout = "|".join("0_0" if i == 0 else "+".join(f"w{j}" for j in range(i)) + "_0" for i in range(len(clips)))
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout={layout}[out]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    for _ in range(3):
        if subprocess.run(command, timeout=1200).returncode == 0:
            break
    else:
        raise RuntimeError("ffmpeg could not tile the clips")
    result = probe(ffprobe, destination)
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]), "width": int(result["width"]), "height": int(result["height"])}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d11-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=1200:-2", str(staging / "f%04d.png")], timeout=600)
        frames = sorted(staging.glob("f*.png"))[:PREVIEW_LIMIT]
        speed = real_duration_s / (len(frames) / 10.0)
        font = ImageFont.load_default(size=16)
        images = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 26), BACKGROUND)
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text((10, 5), f"{title} | GIF SUMMARY | approximately {speed:.1f}x speed | full video: {video.name}", fill=(255, 189, 100), font=font)
            images.append(canvas)
        images[0].save(destination, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    return {"gif": destination.name, "frames": len(images), "approximate_speed": round(speed, 2), "summary_of": video.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    promotion = json.loads((args.campaign / "promotion.json").read_bytes())
    reuse = json.loads((args.campaign / "reuse.json").read_bytes())
    invalidation = json.loads((args.campaign / "invalidation.json").read_bytes())
    geometry = invalidation["changes"]["geometry"]
    panels = []

    # 1. validate once: the first rendered validation success on the jaw arm
    validated = next(r for r in promotion["bodies"][BODY]["rows"] if r["skill_success"] and "media" in r)
    decision = promotion["bodies"][BODY]["decisions"]["transfer_object"]
    panels.append({"name": "validate", "title": "1 validate once", "video": args.campaign / validated["media"] / "episode.mp4", "frames": args.campaign / validated["media"] / "frames.json",
                   "text": f"validation episode {validated['episode_id']}: {validated['verdict']}; the set passed {decision['successes']}/{decision['episodes']} (threshold {decision['threshold']}); certificate {decision['certificate_id'][:12]} v1 {decision['status']}",
                   "source_media_sha256": validated["media_sha256"], "physical_sha256": validated["bundle"]["sha256"]})
    # 2. reuse in a new layout
    reused = next(r for r in reuse["runs"] if r["body"] == BODY and r["layout"] == LAYOUT and r["skill"] == "transfer_object")
    panels.append({"name": "reuse", "title": "2 reuse in the mirrored layout", "video": args.campaign / reused["episode"]["media"] / "episode.mp4", "frames": args.campaign / reused["episode"]["media"] / "frames.json",
                   "text": f"retrieved {reused['certificate_id'][:12]} v{reused['certificate_version']} (12 validation episodes not re-run); {reused['episode']['verdict']}",
                   "source_media_sha256": reused["episode"]["media_sha256"], "physical_sha256": reused["episode"]["bundle"]["sha256"]})
    # 3. the physical-context change: the retrieval refused, nothing ran
    refused = geometry["retrieval_under_new_context_before_revalidation"]
    best = refused["matches"][0] if refused["matches"] else None
    differing = ", ".join((d.get("dimension") or d.get("quantity") or d["kind"]) for d in best["verdict"]["differences"]) if best else "no certificate"
    slate_path = args.out / "refused" / "refused.mp4"
    slate_info = slate(ffmpeg, slate_path, [f"{BODY} | REFUSED | the cube grew to 35 mm: geometry changed", f"retrieval for transfer_object under the new context: {refused['reason'][:100]}",
                                             f"most similar certificate {best['certificate_id'][:12] if best else '-'}: {best['status'] if best else '-'}, similarity {best['verdict']['similarity']:.2f}, differs on {differing}" if best else "no certificate",
                                             f"{len(geometry['affected'])} certificate(s) invalidated by the geometry change; similarity is not validity"])
    panels.append({"name": "refused", "title": "3 invalidated: retrieval refused", "video": slate_path, "frames": None, "text": refused["reason"], "slate": True})
    # 4. repair: the revalidation episode under the grown cube
    revalidated = next(r for r in geometry["rows"] if r["skill_success"] and "media" in r)
    repair = geometry["decisions"]["transfer_object"]
    panels.append({"name": "revalidate", "title": "4 revalidate under the grown cube", "video": args.campaign / revalidated["media"] / "episode.mp4", "frames": args.campaign / revalidated["media"] / "frames.json",
                   "text": f"revalidation {revalidated['episode_id']}: {revalidated['verdict']}; the set {'passed' if repair['status'] == 'promoted' else 'failed'} {repair['successes']}/{repair['episodes']} (threshold {repair['threshold']}); certificate {repair['new_certificate_id'][:12]} v{repair['version']} {repair['status']}",
                   "source_media_sha256": revalidated["media_sha256"], "physical_sha256": revalidated["bundle"]["sha256"]})
    # 5. reuse after the repair: run here from the persisted store as the campaign left it
    store = SkillStoreV1.load(STORE)
    layouts_payload = json.loads((protocol.PROTOCOL / "layouts.json").read_bytes())
    changes = json.loads((protocol.PROTOCOL / "changes.json").read_bytes())
    environment = changed_world(changes["changes"]["geometry"], EnvironmentV1.model_validate(layouts_payload["layouts"][LAYOUT]["environment"]))
    policy = load_policy(g10.G09 / "policy.json")
    session = TransferObjectSession.open(BODY, source_of(BODY), environment, goal_for(environment), policy, configuration_name=protocol.CONFIGURATION, seed_label="d11-repaired")
    query = context_of(session)
    trace = store.retrieve("transfer_object", query)
    fifth = {"name": "repaired", "title": "5 reuse again, repaired", "retrieval": trace.model_dump(mode="json")}
    if trace.chosen is None:
        slate_two = slate(ffmpeg, args.out / "repaired" / "refused.mp4", [f"{BODY} | REFUSED after repair", trace.reason[:110], "the revalidation did not issue a certificate the mirrored layout with the grown cube is covered by"])
        fifth.update({"video": args.out / "repaired" / "refused.mp4", "frames": None, "text": trace.reason, "slate": True})
    else:
        certificate = store.certificate(trace.chosen)
        library = store.library(certificate.library_sha256)
        run = run_tree(session, library, "transfer_object", {**ARGS, "effector": session.effector.chain_id})
        episode = episode_row(run, episode_id="d11-repaired", body=BODY, draw=None, goal=goal_for(environment))
        bundle_dir = args.local / "d11" / "repaired" / "physical"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        sealed = seal_tree_run(bundle_dir, session=session, library=library, tree=run.tree, record=run.record, label="d11-repaired", caption=f"{BODY} | reuse after repair | {LAYOUT} + 35 mm cube | certificate v{certificate.version} | {run.record.verdict.value}",
                               runtime_calls=run.runtime.calls, goal="G11", protocol=EVIDENCE_SCHEMA["episode_protocol"],
                               task_extra={"demo": "D11", "layout": LAYOUT, "certificate_id": certificate.certificate_id, "certificate_version": certificate.version, "retrieval": trace.model_dump(mode="json")},
                               outcome_extra={"skill_success": episode["skill_success"], "oracle_placement": episode["oracle"], "false_completion": episode["false_completion"], "attempts": episode["attempts"]},
                               observation={"policy": "declared_sensors_with_oracle_labels_kept_apart", "inputs": ["joint encoders", "grasp point through the model", "contact force per opposition group", "cameras by ray visibility"], "vlm": False})
        media_dir = args.out / "repaired" / "media"
        if media_dir.exists():
            shutil.rmtree(media_dir)
        media = render_bundle(bundle_dir, media_dir, expected_digest=sealed["sha256"])
        fifth.update({"video": media_dir / "episode.mp4", "frames": media_dir / "frames.json", "text": f"retrieved {certificate.certificate_id[:12]} v{certificate.version}; {run.record.verdict.value}" + (f" | {run.record.root.reason}" if run.record.root.reason else ""),
                      "episode": {k: v for k, v in episode.items() if k not in ("leaf_calls", "verdict_trail", "events")}, "physical_sha256": sealed["sha256"], "source_media_sha256": media["sha256"], "physical_bundle": bundle_dir.as_posix()})
    panels.append(fifth)

    for panel in panels:
        if "physical_sha256" in panel and panel["name"] != "repaired":
            target = args.out / panel["name"]
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(panel["video"], target / "episode.mp4")
            if panel.get("frames"):
                shutil.copy2(panel["frames"], target / "frames.json")
            panel["video"] = target / "episode.mp4"
            panel["frames"] = target / "frames.json" if panel.get("frames") else None
    tiled = tile(ffmpeg, ffprobe, [p["video"] for p in panels], args.out / "d11-five-way.mp4")
    summary = gif_summary(ffmpeg, args.out / "d11-five-way.mp4", args.out / "d11-five-way-preview.gif", "D11: validate | reuse | refused | revalidate | reuse repaired", tiled["duration_s"])
    maps = [json.loads(Path(p["frames"]).read_bytes()) if p.get("frames") else None for p in panels]
    tile_frames = [{"frame": f, "playback_time_s": f / FPS, "panels": {p["name"]: (m[f] if m is not None and f < len(m) else ({"held": True, "recorded_sample": m[-1]["recorded_sample"], "simulation_time_s": m[-1]["simulation_time_s"]} if m is not None else {"slate": True}))
                                                                for p, m in zip(panels, maps)}} for f in range(tiled["frames"])]
    (args.out / "d11-five-way-frames.json").write_bytes(json_bytes(tile_frames))
    index = {"goal": "G11", "demo": "D11", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "body": BODY, "layout": LAYOUT, "store_version": store.version,
             "panels": [{k: (v.relative_to(args.out).as_posix() if isinstance(v, Path) else v) for k, v in p.items()} for p in panels],
             "tile": {**{k: v for k, v in tiled.items() if k != "frames"}, "frame_count": tiled["frames"], "preview": summary, "frames": "d11-five-way-frames.json", "layout": "left to right: validate once, reuse in the mirrored layout, retrieval refused after the cube grew, revalidate under the grown cube, reuse again with the superseding certificate; each clip at two fifths size; a shorter clip holds its final state"},
             "generation_calls": 0}
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps({p["name"]: p["text"][:90] for p in panels}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
