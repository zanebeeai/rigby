"""D09: a visible grasp, a hidden slip and an occluded placement, with the monitor's verdicts on every frame.

Three recorded episodes, each rendered in full from its sealed physical
record and then annotated frame by frame with what the conditionals
decided under two sensor configurations, the sensors each verdict used
and how old their newest samples were, and the oracle's label from the
full state, which the monitor never reads:

- visible grasp: the jaw arm's scored transfer seen by the front camera
  and its contact sensor; opposition, held and moving-with-robot decided
  as they happen, and the overhead camera, which the hand stands under,
  reporting occlusion through the same instants;
- hidden slip: the jaw arm's unchecked return from the G08 pair, which
  set off with the cube in hand and dropped it on the way; overhead camera
  with contact against the low side camera alone: the contact sensor
  fails the hold the moment opposition is lost while the overhead camera
  is blind behind the hand, and the camera-only monitor says unknown while
  the hand hides the cube and fails only once it can see it on the
  platform;
- occluded placement: the long arm's placement with the hand hovering
  above the platform, so the overhead camera cannot see the cube through
  the dwell and stably-placed stays unknown with its re-observe fallback,
  while the front camera decides it.

The base frames are the G06 renderer's; the strip below them is the only
addition. No physics is replayed and no API or model call is made.

    python any-robot/scripts/g09_d09_media.py --out docs/results/g09-d09
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.skills import Decision, SampleQuality

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.sensing import Episode, EvidenceStreams, OracleLabeler, configuration, conditionals_for, decide, load_policy, policy_digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g09_calibrate as calibrate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / "assets/general/research-protocols/g09-conditionals-v1"
PREDICATES = ("reachable", "opposition_established", "held", "moving_with_robot", "stably_placed", "area_clear")
SHORT = {"reachable": "reachable", "opposition_established": "opposition", "held": "held", "moving_with_robot": "moving w/ robot", "stably_placed": "stably placed", "area_clear": "area clear"}
FPS = 12
PREVIEW_LIMIT = 60
STRIP = 132
COLOURS = {Decision.PASS: (52, 140, 92), Decision.FAIL: (170, 62, 62), Decision.UNKNOWN: (176, 120, 30)}
BACKGROUND = (17, 24, 39)

CASES = [
    {"name": "visible-grasp", "tree": "C:/Users/hocke/GitHub/rigby-g06/any-robot/results/g06-campaign", "relative": "zoo_jaw_arm/seed-000/physical",
     "configurations": ("front_contact", "overhead_contact"), "title": "VISIBLE GRASP | jaw arm, scored transfer seed 0",
     "caption": "front camera and contact decide the grasp as it happens; the overhead camera is hidden by the hand"},
    {"name": "hidden-slip", "tree": "C:/Users/hocke/GitHub/rigby-g08/docs/results/g08-d08", "relative": "contact-mode-change/before/physical",
     "configurations": ("overhead_contact", "side_vision_only"), "title": "HIDDEN SLIP | jaw arm, unchecked return with the cube in hand (G08 D08, before)",
     "caption": "the cube drops out during the return: contact fails the hold at once under a blind overhead camera; side camera alone is unknown until it sees the cube again"},
    {"name": "occluded-placement", "tree": "C:/Users/hocke/GitHub/rigby-g06/any-robot/results/g06-campaign", "relative": "zoo_long_arm/seed-000/physical",
     "configurations": ("overhead_contact", "front_contact"), "title": "OCCLUDED PLACEMENT | long arm, scored transfer seed 0",
     "caption": "the hand hovers over the platform: overhead stays unknown (re-observe); the front camera decides the placement"},
]


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def decode(ffmpeg: str, video: Path, width: int, height: int) -> list[Image.Image]:
    raw = subprocess.check_output([ffmpeg, "-v", "error", "-nostdin", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], timeout=600)
    count = len(raw) // (width * height * 3)
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)
    return [Image.fromarray(f.copy()) for f in frames]


def sensor_line(streams: EvidenceStreams, now_s: float, entity: str) -> str:
    parts = []
    for sensor_id, stream in streams.streams.items():
        sensor = stream.sensor
        if sensor.oracle or (sensor.entity and sensor.entity != entity) or sensor.kind.value in ("joint_encoders", "region"):
            continue
        hi = int(np.searchsorted(stream.times, now_s + 1e-9, side="right"))
        if hi == 0:
            parts.append(f"{sensor_id}: none yet")
            continue
        age = now_s - float(stream.times[hi - 1])
        quality = stream.quality[hi - 1]
        state = "ok" if quality is SampleQuality.VALID else quality.value.upper()
        parts.append(f"{sensor_id} {state} {age * 1000:.0f}ms")
    return " | ".join(parts)


def annotate(frame: Image.Image, *, title: str, caption: str, time_s: float, rows: list[dict], labels: dict[str, bool], policy_id: str, preview: str | None) -> Image.Image:
    canvas = Image.new("RGB", (frame.width, frame.height + STRIP), BACKGROUND)
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    if preview:
        draw.rectangle((0, 51, frame.width, 68), fill=BACKGROUND)
        draw.text((12, 51), preview, fill=(191, 210, 233), font=ImageFont.load_default(size=16))
    y = frame.height + 4
    draw.text((10, y), f"{title} | conditionals under policy {policy_id} | sim t={time_s:.3f}s", fill=(255, 220, 160), font=font)
    y += 16
    draw.text((10, y), caption, fill=(200, 210, 225), font=small)
    y += 15
    column = 150
    for row in rows:
        draw.text((10, y + 2), row["configuration"][:20], fill="white", font=font)
        for k, name in enumerate(PREDICATES):
            verdict = row["verdicts"][name]
            x = column + k * 134
            draw.rectangle((x, y, x + 128, y + 15), fill=COLOURS[verdict.decision])
            text = f"{SHORT[name]}: {verdict.decision.value.upper()}"
            if verdict.decision is Decision.UNKNOWN and verdict.reason:
                text = f"{SHORT[name]}: ?{verdict.reason.split(':')[0][:8]}"
            draw.text((x + 3, y + 1), text[:24], fill="white", font=small)
        y += 17
        draw.text((14, y), row["sensors"][:150], fill=(170, 185, 205), font=small)
        y += 14
    oracle = "  ".join(f"{SHORT[name]}={'1' if labels[name] else '0'}" for name in PREDICATES)
    draw.text((10, y + 2), f"oracle labels from the full state (never read by the monitor): {oracle}", fill=(150, 200, 255), font=small)
    return canvas


def overlay(case: dict, episode: Episode, media: Path, destination: Path, policy: dict, ffmpeg: str, ffprobe: str) -> dict:
    manifest = verify_bundle(media)
    frames_map = json.loads((media / "frames.json").read_bytes())
    stream = probe(ffprobe, media / "episode.mp4")
    width, height = int(stream["width"]), int(stream["height"])
    base = decode(ffmpeg, media / "episode.mp4", width, height)
    if len(base) != len(frames_map):
        raise RuntimeError("decoded frame count disagrees with the frame map")
    effector = episode.effectors[0]
    streams = {name: EvidenceStreams.build(episode, configuration(name, episode.effectors)) for name in case["configurations"]}
    conditionals = conditionals_for(episode, effector, policy)
    labeler = OracleLabeler(episode)
    records = []
    annotated: list[Image.Image] = []
    preview_indices = set(np.linspace(0, len(base) - 1, min(PREVIEW_LIMIT, len(base))).astype(int).tolist())
    previews = []
    speed = (len(base) / FPS) / (len(preview_indices) / 10)
    reach_label = None
    for index, (frame, entry) in enumerate(zip(base, frames_map)):
        now = float(entry["simulation_time_s"])
        rows = []
        record = {"frame": index, "simulation_time_s": now, "configurations": {}}
        for name in case["configurations"]:
            verdicts = {p: decide(conditionals[p], streams[name], now) for p in PREDICATES}
            rows.append({"configuration": name, "verdicts": verdicts, "sensors": sensor_line(streams[name], now, effector.chain_id)})
            record["configurations"][name] = {p: {"decision": v.decision.value, "reason": v.reason, "sensors": list(v.sensors_used)} for p, v in verdicts.items()}
        labels = {}
        for p in PREDICATES:
            if p == "reachable":
                if reach_label is None or index % FPS == 0:
                    reach_label = labeler.label(p, effector, now, conditionals[p].window.duration_s).value
                labels[p] = reach_label
            else:
                labels[p] = labeler.label(p, effector, now, conditionals[p].window.duration_s).value
        record["labels"] = labels
        records.append(record)
        annotated.append(annotate(frame, title=case["title"], caption=case["caption"], time_s=now, rows=rows, labels=labels, policy_id=policy["policy_id"], preview=None))
        if index in preview_indices:
            previews.append(annotate(frame, title=case["title"], caption=case["caption"], time_s=now, rows=rows, labels=labels, policy_id=policy["policy_id"],
                                     preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4"))
    with tempfile.TemporaryDirectory(prefix=".d09-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height + STRIP}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
        for image in annotated:
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait(timeout=600) != 0:
            raise RuntimeError("ffmpeg failed")
        encoded = probe(ffprobe, video)
        if int(encoded["nb_read_frames"]) != len(annotated) or abs(float(encoded["duration"]) - len(annotated) / FPS) > 0.01:
            raise RuntimeError("the annotated video lost frames or changed its playback duration")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        annotated[0].save(staging / "initial.png")
        annotated[-1].save(staging / "final.png")
        payloads = {name: (staging / name).read_bytes() for name in ("episode.mp4", "preview.gif", "initial.png", "final.png")}
        payloads["frames.json"] = (media / "frames.json").read_bytes()
        payloads["verdicts.json"] = json_bytes(records)
        payloads["encoding.json"] = json_bytes({"probe": encoded, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})
        tally = {name: {p: {d.value: sum(1 for r in records if r["configurations"][name][p]["decision"] == d.value) for d in Decision} for p in PREDICATES} for name in case["configurations"]}
        metadata = {"schema": "rigby.presentation-overlay/1", "source_media_sha256": hashlib.sha256((media / "manifest.json").read_bytes()).hexdigest(),
                    "source_bundle_sha256": manifest["metadata"]["source_bundle_sha256"], "source_trace_sha256": manifest["metadata"]["source_trace_sha256"],
                    "robot_id": manifest["metadata"]["robot_id"], "outcome": manifest["metadata"]["outcome"], "full_episode": True, "frame_count": len(annotated), "fps": FPS,
                    "simulation_duration_s": manifest["metadata"]["simulation_duration_s"], "playback_duration_s": len(annotated) / FPS, "preview_is_summary": True,
                    "refusal_slate": manifest["metadata"].get("refusal_slate", False), "physics_replay_performed_by_renderer": False,
                    "overlay": {"configurations": list(case["configurations"]), "predicates": list(PREDICATES), "policy_id": policy["policy_id"], "policy_digest": policy_digest(policy), "strip_px": STRIP},
                    "verdict_frames": tally, "case": case["name"], "caption": case["caption"]}
        digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "frames": len(annotated), "verdict_frames": tally, "simulation_duration_s": metadata["simulation_duration_s"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh destination")
    args.out.mkdir(parents=True)
    policy = load_policy(PROTOCOL / "policy.json")
    goal = calibrate.registered_goal()
    index = {"goal": "G09", "demo": "D09", "created_at_utc": datetime.now(timezone.utc).isoformat(), "policy_sha256": hashlib.sha256((PROTOCOL / "policy.json").read_bytes()).hexdigest(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "generation_calls": 0, "cases": []}
    for case in CASES:
        physical = Path(case["tree"]) / case["relative"]
        episode = Episode.load(physical, default_goal=goal)
        source = hashlib.sha256((physical / "manifest.json").read_bytes()).hexdigest()
        media = render_bundle(physical, args.out / case["name"] / "media", expected_digest=source)
        result = overlay(case, episode, args.out / case["name"] / "media", args.out / case["name"] / "overlay", policy, ffmpeg, ffprobe)
        shutil.copyfile(args.out / case["name"] / "overlay" / "episode.mp4", args.out / f"{case['name']}.mp4")
        shutil.copyfile(args.out / case["name"] / "overlay" / "preview.gif", args.out / f"{case['name']}-preview.gif")
        shutil.copyfile(args.out / case["name"] / "overlay" / "frames.json", args.out / f"{case['name']}-frames.json")
        entry = {**{k: v for k, v in case.items() if k != "tree"}, "physical_sha256": source, "trace_sha256": episode.trace_sha256, "outcome": episode.metadata.get("outcome"),
                 "failed_gate": episode.outcome.get("failed_gate"), "media_sha256": media["sha256"], "overlay": result, "video": f"{case['name']}.mp4", "preview": f"{case['name']}-preview.gif", "frames": f"{case['name']}-frames.json"}
        index["cases"].append(entry)
        print(json.dumps({"case": case["name"], "frames": result["frames"], "verdict_frames": result["verdict_frames"]}), flush=True)
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
