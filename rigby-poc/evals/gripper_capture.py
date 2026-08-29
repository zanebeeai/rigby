"""Record a gripper clip to a video file you can watch outside the browser.

Drives the gripper viewer with Playwright, one screenshot per frame, and encodes
with the ffmpeg Playwright already ships -- there is no ffmpeg on PATH here, and
requiring one is what stopped this working the first time. A GIF is written too,
by PIL, so there is always something that plays anywhere.

Frames are SEEKED, not played. Screenshotting a running animation samples
whatever the render loop happened to reach and silently drops or repeats frames
under load; the viewer exposes a seek handle so each capture is the frame that
was asked for.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "milestones" / "video"

#: Playwright ships an ffmpeg for its own video recording. Borrow it rather than
#: making the user install one.
_BUNDLED = sorted(
    (Path.home() / "AppData/Local/ms-playwright").glob("ffmpeg-*/ffmpeg-*.exe"))


def _encoder() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return str(_BUNDLED[-1]) if _BUNDLED else None


def record(clip: str, base: str = "http://127.0.0.1:5173/static",
           out_dir: Path | None = None, width: int = 1280, height: int = 860,
           every: int = 1, last: int | None = None) -> list[Path]:
    """Walk the viewer through ``clip`` and write an MP4 and a GIF."""
    out_dir = out_dir or _OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Image.Image] = []

    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=1)
        page.goto(f"{base}/gripper.html?clip={clip}", wait_until="networkidle")
        page.wait_for_function("window.rigbyClip !== undefined", timeout=60000)
        total = int(page.evaluate("window.rigbyClip.frames"))
        # A run that finishes early spends its tail holding still. Recording
        # that is honest and dull, so the caller may stop at the interesting bit.
        if last is not None:
            total = min(total, last)
        fps = int(page.evaluate("window.rigbyClip.fps"))
        for index in range(0, total, every):
            page.evaluate(f"window.rigbyClip.seek({index})")
            shots.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
        browser.close()

    if not shots:
        raise RuntimeError(f"no frames captured for {clip}")
    written: list[Path] = []
    rate = max(fps // every, 1)

    encoder = _encoder()
    if encoder:
        # Playwright's ffmpeg is built for its own screen recording and nothing
        # else: one video encoder (VP8), one input decoder (MJPEG), and only the
        # piped image demuxer -- no image2 file sequence and no PNG at all. So
        # frames go in as JPEG on stdin and come out as WebM, which plays in
        # Edge, Chrome and VLC. A full ffmpeg on PATH takes H.264 instead.
        full = shutil.which("ffmpeg") is not None
        suffix, codec = (".mp4", "libx264") if full else (".webm", "libvpx")
        movie = out_dir / f"{clip}{suffix}"
        with tempfile.TemporaryDirectory() as scratch:
            # One file holding every frame back to back. image2pipe reads a
            # concatenated JPEG stream happily, and going through a file rather
            # than stdin sidesteps this build refusing "-i -".
            reel = Path(scratch) / "frames.mjpeg"
            with reel.open("wb") as handle:
                for shot in shots:
                    shot.save(handle, format="JPEG", quality=95)
            command = [encoder, "-y", "-f", "image2pipe", "-framerate",
                       str(rate), "-c:v", "mjpeg", "-i", str(reel),
                       "-c:v", codec, "-vf",
                       "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                       "-pix_fmt", "yuv420p"]
            command += ["-crf", "20"] if full else ["-b:v", "4M"]
            done = subprocess.run(command + [str(movie)], capture_output=True)
            if done.returncode != 0:
                tail = done.stderr.decode("utf-8", "replace").strip().splitlines()
                print(f"  encoding failed: {tail[-1] if tail else '?'}")
            else:
                written.append(movie)

    small = [s.resize((s.width // 2, s.height // 2), Image.LANCZOS)
             for s in shots]
    gif = out_dir / f"{clip}.gif"
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(1000 / rate), loop=0, optimize=True)
    written.append(gif)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", default=["gripper-run"])
    parser.add_argument("--base", default="http://127.0.0.1:5173/static")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--last", type=int, default=None,
                        help="stop after this many frames")
    args = parser.parse_args()
    if not _encoder():
        print("  no ffmpeg found -- writing GIF only")
    for clip in args.clips:
        for path in record(clip, base=args.base, every=args.every,
                           last=args.last):
            print(f"  {path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
