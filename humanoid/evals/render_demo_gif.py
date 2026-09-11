from __future__ import annotations

import argparse
import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from evals.capture import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    SAFE_RESULT_ID,
    _launch_capture_browser,
    _png_size,
    _result_payload,
)


HEADER_HEIGHT = 30
BACKGROUND = "#0b1117"
LABEL_COLOR = "#dbe6ed"
ACCENT_COLOR = "#63e6c3"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    windows_font = Path("C:/Windows/Fonts") / filename
    try:
        return ImageFont.truetype(str(windows_font), size=size)
    except OSError:
        return ImageFont.load_default()


def _capture_frame(page, target_url: str) -> bytes:
    page.goto(target_url, wait_until="domcontentloaded", timeout=45_000)
    page.wait_for_selector('body[data-capture-status="ready"]', timeout=45_000)
    state = page.evaluate("window.__RIGBY_CAPTURE__")
    if not isinstance(state, dict) or state.get("status") != "ready":
        raise RuntimeError(f"capture page failed: {state}")
    image = page.locator("#capture-app canvas").screenshot(
        type="png",
        animations="disabled",
        scale="css",
        timeout=45_000,
    )
    if _png_size(image) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        raise RuntimeError("capture page returned an unexpected canvas size")
    return image


def _compose_frame(
    ego_png: bytes,
    orbit_png: bytes,
    *,
    panel_width: int,
    time_s: float,
) -> Image.Image:
    panel_height = round(panel_width * CAPTURE_HEIGHT / CAPTURE_WIDTH)
    canvas = Image.new("RGB", (panel_width * 2, panel_height + HEADER_HEIGHT), BACKGROUND)
    resampling = Image.Resampling.LANCZOS
    for index, image_bytes in enumerate((ego_png, orbit_png)):
        with Image.open(io.BytesIO(image_bytes)) as source:
            panel = source.convert("RGB").resize((panel_width, panel_height), resampling)
        canvas.paste(panel, (index * panel_width, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    label_font = _font(14, bold=True)
    time_font = _font(12)
    draw.text((10, 6), "EGOCENTRIC", fill=ACCENT_COLOR, font=label_font)
    draw.text((panel_width + 10, 6), "ORBIT", fill=LABEL_COLOR, font=label_font)
    time_label = f"{time_s:0.2f}s"
    time_box = draw.textbbox((0, 0), time_label, font=time_font)
    draw.text(
        (panel_width * 2 - (time_box[2] - time_box[0]) - 10, 7),
        time_label,
        fill="#82919c",
        font=time_font,
    )
    return canvas


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = completed.stderr.strip().splitlines()
        raise RuntimeError(details[-1] if details else "ffmpeg failed")


def render_demo_gif(
    result_id: str,
    output_path: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    fps: int = 8,
    panel_width: int = 480,
    colors: int = 96,
) -> Path:
    if not SAFE_RESULT_ID.fullmatch(result_id):
        raise ValueError("result id contains unsupported characters")
    if not 2 <= fps <= 20:
        raise ValueError("fps must be between 2 and 20")
    if not 240 <= panel_width <= 800:
        raise ValueError("panel width must be between 240 and 800 pixels")
    if not 32 <= colors <= 256:
        raise ValueError("colors must be between 32 and 256")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the GIF")

    payload = _result_payload(base_url, result_id)
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    duration = float(clip.get("duration_s", 0.0))
    if duration <= 0.0:
        raise ValueError("result clip has no positive duration")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(2, math.ceil(duration * fps) + 1)
    sample_times = [min(duration, index / fps) for index in range(frame_count)]

    with tempfile.TemporaryDirectory(prefix="rigby-demo-") as temporary:
        temporary_path = Path(temporary)
        with sync_playwright() as playwright:
            browser, _channel = _launch_capture_browser(playwright.chromium)
            context = browser.new_context(
                viewport={"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT},
                device_scale_factor=1,
            )
            ego_page = context.new_page()
            orbit_page = context.new_page()
            try:
                for index, time_s in enumerate(sample_times):
                    common = {"result": result_id, "time": f"{time_s:.9f}"}
                    ego_url = (
                        f"{base_url.rstrip('/')}/capture.html?"
                        f"{urlencode({**common, 'view': 'ego'})}"
                    )
                    orbit_url = (
                        f"{base_url.rstrip('/')}/capture.html?"
                        f"{urlencode({**common, 'view': 'orbit'})}"
                    )
                    ego_png = _capture_frame(ego_page, ego_url)
                    orbit_png = _capture_frame(orbit_page, orbit_url)
                    composed = _compose_frame(
                        ego_png,
                        orbit_png,
                        panel_width=panel_width,
                        time_s=time_s,
                    )
                    composed.save(temporary_path / f"frame-{index:04d}.png", optimize=True)
            finally:
                context.close()
                browser.close()

        palette_path = temporary_path / "palette.png"
        input_pattern = temporary_path / "frame-%04d.png"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(input_pattern),
                "-vf",
                f"palettegen=max_colors={colors}:stats_mode=diff",
                str(palette_path),
            ]
        )
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(input_pattern),
                "-i",
                str(palette_path),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                "-loop",
                "0",
                str(output_path),
            ]
        )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side egocentric/orbit GIF from a saved Rigby result"
    )
    parser.add_argument("result_id")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--colors", type=int, default=96)
    arguments = parser.parse_args()
    path = render_demo_gif(
        arguments.result_id,
        arguments.output_path,
        base_url=arguments.base_url,
        fps=arguments.fps,
        panel_width=arguments.panel_width,
        colors=arguments.colors,
    )
    print(path)


if __name__ == "__main__":
    main()
