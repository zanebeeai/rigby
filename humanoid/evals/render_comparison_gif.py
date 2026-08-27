from __future__ import annotations

import argparse
import io
import math
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from evals.capture import CAPTURE_HEIGHT, CAPTURE_WIDTH, SAFE_RESULT_ID, _result_payload
from evals.render_demo_gif import (
    ACCENT_COLOR,
    BACKGROUND,
    HEADER_HEIGHT,
    LABEL_COLOR,
    _capture_frame,
    _font,
    _run_ffmpeg,
)


def _compose_pair(
    left_png: bytes,
    right_png: bytes,
    *,
    panel_width: int,
    time_s: float,
    left_label: str,
    right_label: str,
    crop: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    source_width, source_height = (crop[0], crop[1]) if crop else (CAPTURE_WIDTH, CAPTURE_HEIGHT)
    panel_height = round(panel_width * source_height / source_width)
    canvas = Image.new("RGB", (panel_width * 2, panel_height + HEADER_HEIGHT), BACKGROUND)
    resampling = Image.Resampling.LANCZOS
    for index, image_bytes in enumerate((left_png, right_png)):
        with Image.open(io.BytesIO(image_bytes)) as source:
            panel = source.convert("RGB")
            if crop:
                width, height, left, top = crop
                panel = panel.crop((left, top, left + width, top + height))
            panel = panel.resize((panel_width, panel_height), resampling)
        canvas.paste(panel, (index * panel_width, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    label_font = _font(14, bold=True)
    time_font = _font(12)
    draw.text((10, 6), left_label, fill=LABEL_COLOR, font=label_font)
    draw.text((panel_width + 10, 6), right_label, fill=ACCENT_COLOR, font=label_font)
    time_label = f"{time_s:0.2f}s"
    time_box = draw.textbbox((0, 0), time_label, font=time_font)
    draw.text(
        (panel_width * 2 - (time_box[2] - time_box[0]) - 10, 7),
        time_label,
        fill="#82919c",
        font=time_font,
    )
    return canvas


def render_comparison_gif(
    left_result_id: str,
    right_result_id: str,
    output_path: Path,
    *,
    left_base_url: str,
    right_base_url: str,
    view: str = "orbit",
    left_label: str = "BEFORE",
    right_label: str = "AFTER",
    fps: int = 8,
    panel_width: int = 480,
    colors: int = 96,
    sync: str = "proportional",
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Render two saved results side by side on one shared timeline.

    ``sync="proportional"`` samples each clip at the same fraction of its own
    duration, so corresponding phases stay aligned when the clips differ in
    length; ``sync="absolute"`` samples both at the same wall-clock time and
    the shorter clip holds its final pose past its own end. The header time is
    the longer clip's.
    """

    for result_id in (left_result_id, right_result_id):
        if not SAFE_RESULT_ID.fullmatch(result_id):
            raise ValueError("result id contains unsupported characters")
    if view not in {"ego", "orbit"}:
        raise ValueError("view must be ego or orbit")
    if sync not in {"proportional", "absolute"}:
        raise ValueError("sync must be proportional or absolute")
    if crop is not None:
        width, height, left, top = crop
        if not (0 < width and 0 < height):
            raise ValueError("crop region must have positive size")
        if left < 0 or top < 0 or left + width > CAPTURE_WIDTH or top + height > CAPTURE_HEIGHT:
            raise ValueError("crop region must lie inside the capture frame")
    if not 2 <= fps <= 20:
        raise ValueError("fps must be between 2 and 20")
    if not 240 <= panel_width <= 800:
        raise ValueError("panel width must be between 240 and 800 pixels")
    if not 32 <= colors <= 256:
        raise ValueError("colors must be between 32 and 256")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the GIF")

    durations = []
    for base_url, result_id in (
        (left_base_url, left_result_id),
        (right_base_url, right_result_id),
    ):
        payload = _result_payload(base_url, result_id)
        clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
        durations.append(float(clip.get("duration_s", 0.0)))
    duration = max(durations)
    if duration <= 0.0:
        raise ValueError("neither result clip has a positive duration")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(2, math.ceil(duration * fps) + 1)
    sample_times = [min(duration, index / fps) for index in range(frame_count)]

    with tempfile.TemporaryDirectory(prefix="rigby-comparison-") as temporary:
        temporary_path = Path(temporary)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(
                viewport={"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT},
                device_scale_factor=1,
            )
            left_page = context.new_page()
            right_page = context.new_page()
            try:
                for index, time_s in enumerate(sample_times):
                    fraction = time_s / duration
                    panels = []
                    for page, base_url, result_id, clip_duration in (
                        (left_page, left_base_url, left_result_id, durations[0]),
                        (right_page, right_base_url, right_result_id, durations[1]),
                    ):
                        if sync == "proportional":
                            clamped = fraction * clip_duration
                        else:
                            clamped = min(time_s, clip_duration)
                        query = urlencode(
                            {
                                "result": result_id,
                                "time": f"{clamped:.9f}",
                                "view": view,
                            }
                        )
                        panels.append(
                            _capture_frame(
                                page, f"{base_url.rstrip('/')}/capture.html?{query}"
                            )
                        )
                    composed = _compose_pair(
                        panels[0],
                        panels[1],
                        panel_width=panel_width,
                        time_s=time_s,
                        left_label=left_label,
                        right_label=right_label,
                        crop=crop,
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
        description=(
            "Render two saved Rigby results side by side (for example the same "
            "prompt compiled before and after a compiler change)"
        )
    )
    parser.add_argument("left_result_id")
    parser.add_argument("right_result_id")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--left-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--right-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--view", default="orbit", choices=("ego", "orbit"))
    parser.add_argument("--left-label", default="BEFORE")
    parser.add_argument("--right-label", default="AFTER")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--sync", default="proportional", choices=("proportional", "absolute"))
    parser.add_argument(
        "--crop",
        default=None,
        help="zoom both panels to a capture-space region, as WIDTH:HEIGHT:LEFT:TOP",
    )
    arguments = parser.parse_args()
    crop = None
    if arguments.crop:
        parts = arguments.crop.split(":")
        if len(parts) != 4:
            parser.error("--crop expects WIDTH:HEIGHT:LEFT:TOP")
        crop = tuple(int(part) for part in parts)
    path = render_comparison_gif(
        arguments.left_result_id,
        arguments.right_result_id,
        arguments.output_path,
        left_base_url=arguments.left_base_url,
        right_base_url=arguments.right_base_url,
        view=arguments.view,
        left_label=arguments.left_label,
        right_label=arguments.right_label,
        fps=arguments.fps,
        panel_width=arguments.panel_width,
        colors=arguments.colors,
        sync=arguments.sync,
        crop=crop,
    )
    print(path)


if __name__ == "__main__":
    main()
