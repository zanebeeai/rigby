"""D12: missing skill, failed attempts, acquired execution, an uninterrupted held-out trial, reuse; the search budget on every frame.

One row of panels per problem, every panel a sealed trial rendered from
recorded states (or a labelled slate where the budget ran out and nothing
was acquired): the defaults failing in the problem's world, a searched
attempt that failed, the attempt that confirmed, one held-out trial run
uninterrupted with the settled vector, and the transfer tree reused from
the persisted store in the mirrored layout under it. A strip above each row
carries the search budget: attempts used of the ceiling, simulator-worker
minutes used, the held-out score, and what kind of thing was found.

    python any-robot/scripts/g12_d12_media.py --campaign docs/results/g12-acquisition --out docs/results/g12-d12
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

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import BANNER, HEIGHT, PREVIEW_LIMIT, WIDTH

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g12_protocol as protocol  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FPS = 12
BACKGROUND = (17, 24, 39)
STRIP = 30


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def slate(ffmpeg: str, destination: Path, lines: list[str], *, seconds: float = 3.0) -> None:
    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    draw.text((12, 7), lines[0], fill=(255, 189, 100), font=font)
    y = 32
    for line in lines[1:]:
        for start in range(0, len(line), 118):
            draw.text((12, y), line[start:start + 118], fill="white" if y < 80 else (191, 210, 233), font=font if y < 80 else small)
            y += 20 if y < 80 else 17
    draw.text((12, HEIGHT + BANNER - 24), "NO MOTION WAS EXECUTED | labelled slate, not a recording", fill=(170, 185, 205), font=small)
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


def strip_row(ffmpeg: str, ffprobe: str, clips: list[Path], destination: Path, title: str) -> dict:
    """Five clips scaled to two fifths, side by side on one clock, under a strip that names the budget."""

    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs, filters = [], []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(stream['duration'])):.3f},scale=trunc(iw*2/5/2)*2:trunc(ih*2/5/2)*2[v{index}]")
    layout = "|".join("0_0" if i == 0 else "+".join(f"w{j}" for j in range(i)) + "_0" for i in range(len(clips)))
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout={layout}[row]")
    text = title.replace("'", "’").replace(":", "\\:").replace("%", "%%")
    filters.append(f"[row]pad=iw:ih+{STRIP}:0:{STRIP}:color=0x111827,drawtext=text='{text}':x=10:y=8:fontsize=15:fontcolor=0xffbd64[out]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    result = subprocess.run(command, timeout=1200, capture_output=True, text=True)
    if result.returncode != 0:
        # drawtext may be unavailable in this ffmpeg build; the strip is then drawn by PIL on the summary only
        filters[-1] = f"[row]pad=iw:ih+{STRIP}:0:{STRIP}:color=0x111827[out]"
        command = [ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        if subprocess.run(command, timeout=1200).returncode != 0:
            raise RuntimeError("ffmpeg could not tile the clips")
        drawn = False
    else:
        drawn = True
    out = probe(ffprobe, destination)
    return {"video": destination.name, "frame_count": int(out["nb_read_frames"]), "duration_s": float(out["duration"]), "width": int(out["width"]), "height": int(out["height"]), "strip_drawn_in_video": drawn, "strip_text": title}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d12-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=1200:-2", str(staging / "f%04d.png")], timeout=600)
        frames = sorted(staging.glob("f*.png"))[:PREVIEW_LIMIT]
        speed = real_duration_s / (len(frames) / 10.0)
        font = ImageFont.load_default(size=15)
        images = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 26), BACKGROUND)
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text((10, 5), f"{title} | GIF SUMMARY | approximately {speed:.1f}x speed | full video: {video.name}"[:150], fill=(255, 189, 100), font=font)
            images.append(canvas)
        images[0].save(destination, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    return {"gif": destination.name, "frames": len(images), "approximate_speed": round(speed, 2), "summary_of": video.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"goal": "G12", "demo": "D12", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "problems": [], "generation_calls": 0}
    for problem_id in [p.problem_id for p in protocol.PROBLEMS]:
        report_path = args.campaign / problem_id / "problem.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_bytes())
        outcome = report["outcome"]
        budget = report["search"]["budget"]
        holdout = report["holdout"]
        target = args.out / problem_id
        target.mkdir(parents=True, exist_ok=True)
        shown = report.get("shown_attempts", {})
        panels = []

        def copy_panel(name: str, title: str, media_rel: str | None, text: str, source_root: Path) -> dict:
            panel = {"name": name, "title": title, "text": text}
            if media_rel is None:
                slate(ffmpeg, target / name / "slate.mp4", [title, text[:118], text[118:236]])
                panel.update({"video": target / name / "slate.mp4", "frames": None, "slate": True})
            else:
                (target / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / media_rel / "episode.mp4", target / name / "episode.mp4")
                shutil.copy2(source_root / media_rel / "frames.json", target / name / "frames.json")
                panel.update({"video": target / name / "episode.mp4", "frames": target / name / "frames.json", "source_media": media_rel})
            return panel

        problem_root = args.campaign / problem_id
        defaults = shown.get("defaults")
        panels.append(copy_panel("missing", f"1 missing: the defaults | {problem_id}", defaults["media"] if defaults else None,
                                 (f"attempt 0 at the defaults: {'certified' if defaults['episode']['certified'] else defaults['episode']['failed_gate']}" if defaults else "no attempt recorded"), problem_root))
        failed = shown.get("failed_search")
        panels.append(copy_panel("failed", f"2 a failed searched attempt | {problem_id}", failed["media"] if failed else None,
                                 (f"attempt {failed['attempt']}: {failed['episode']['failed_gate']}; " + ", ".join(f"{k.split('.')[-1]}={v:.3g}" for k, v in failed["parameters"].items() if abs(v - next(p.default for p in protocol.PARAMETERS if p.name == k)) > 1e-9)) if failed
                                 else ("no searched attempt failed: the defaults passed and no search was needed" if outcome["status"] == "instantiated" else "no searched attempt recorded"), problem_root))
        confirming = shown.get("confirming")
        panels.append(copy_panel("acquired", f"3 the confirming attempt | {problem_id}", confirming["media"] if confirming else None,
                                 (f"attempt {confirming['attempt']} certified the development and confirmation draws; moved: " + (", ".join(outcome["parameters_changed"]) or "nothing")) if confirming
                                 else f"BUDGET EXHAUSTED at {budget['attempts']} attempts, {budget['worker_minutes']:.1f} worker minutes: {outcome['limiting_capability'][:200]}", problem_root))
        first_success = next((r for r in holdout["rows"] if r["certified"] and "media" in r), None)
        first_any = next((r for r in holdout["rows"] if "media" in r), None)
        held = first_success or first_any
        panels.append(copy_panel("holdout", f"4 an uninterrupted held-out trial | {problem_id}", held["media"] if held else None,
                                 (f"held-out seed {held['draw']['seed']}: {'certified' if held['certified'] else held['failed_gate']}; the set scored {holdout['successes']}/{holdout['trials']} against {holdout['threshold']}") if held else "no held-out trial sealed", problem_root))
        reuse = report.get("reuse")
        panels.append(copy_panel("reuse", f"5 reuse from the store | {problem_id}", reuse["media"] if reuse and reuse.get("executed") else None,
                                 (f"transfer_object from the persisted store in the mirrored layout under certificate {reuse['certificate_id'][:12]}: {reuse['episode']['verdict']}") if reuse and reuse.get("executed")
                                 else ("not promoted: nothing to reuse" if outcome["status"] not in ("acquired", "instantiated") else "retrieval found no valid certificate"), problem_root))
        title = (f"D12 {problem_id} | {outcome['status']} ({outcome['kind'] or 'no kind'}) | search {budget['attempts']}/{budget['ceiling']['attempts']} attempts, {budget['worker_minutes']:.1f}/{budget['ceiling']['worker_minutes']:.0f} worker min | "
                 f"held-out {holdout['successes']}/{holdout['trials']} (threshold {holdout['threshold']})")
        row = strip_row(ffmpeg, ffprobe, [p["video"] for p in panels], target / f"{problem_id}-five-way.mp4", title)
        summary = gif_summary(ffmpeg, target / f"{problem_id}-five-way.mp4", target / f"{problem_id}-five-way-preview.gif", title, row["duration_s"])
        maps = [json.loads(Path(p["frames"]).read_bytes()) if p.get("frames") else None for p in panels]
        tile_frames = [{"frame": f, "playback_time_s": f / FPS, "panels": {p["name"]: (m[f] if m is not None and f < len(m) else ({"held": True, "recorded_sample": m[-1]["recorded_sample"], "simulation_time_s": m[-1]["simulation_time_s"]} if m is not None else {"slate": True}))
                                                                    for p, m in zip(panels, maps)}} for f in range(row["frame_count"])]
        (target / f"{problem_id}-five-way-frames.json").write_bytes(json_bytes(tile_frames))
        index["problems"].append({"problem_id": problem_id, "status": outcome["status"], "kind": outcome["kind"], "budget": budget, "holdout": {k: holdout[k] for k in ("trials", "successes", "threshold", "accepted")},
                                  "panels": [{k: (v.relative_to(args.out).as_posix() if isinstance(v, Path) else v) for k, v in p.items()} for p in panels],
                                  "tile": {**row, "preview": summary, "frames": f"{problem_id}/{problem_id}-five-way-frames.json", "video": f"{problem_id}/{row['video']}", "preview_gif": f"{problem_id}/{summary['gif']}"}})
        print(json.dumps({"problem": problem_id, "status": outcome["status"], "panels": [p["name"] + (" (slate)" if p.get("slate") else "") for p in panels]}), flush=True)
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
