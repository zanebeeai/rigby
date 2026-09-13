"""D05: six synchronized bodies, and the dual-arm transition before and after.

Both pieces are assembled from recorded physical states, never from plans.
The six-body reel tiles the campaign's canonical full-duration episodes on one
common physics clock (a shorter episode holds its final state until the
longest ends), and the dual-arm comparison runs the exact same prompt on the
same body through the grounder as it stood before the self-collision guard
and as it stands now, records both executions, gates both, and shows them side
by side. No API or model calls.

    python any-robot/scripts/g05_d05_media.py --campaign docs/results/g05-campaign --out docs/results/g05-d05
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.contracts import MotionProgramV2
from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.motion.compiler import compile_motion_program
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics

from rigby_general.bake.runner import SAMPLE_HZ
from rigby_general.evidence.capture import json_bytes, source_provenance
from rigby_general.evidence.render import render_bundle
from rigby_general.gates.certify import GatePolicy, evaluate_gates, simulate
from rigby_general.gates.control import ControllerConfig
from rigby_general.grounding import ground
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.schema.inventory import afforded_entries, load_inventory


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROMPT = "reach out as far as you can and then come back"
BEFORE_COMMIT = "5a6c0b3bb17eb721da6ba6c144743c24ba872301"
"""The grounder as it stood with bounded curves but before the guard (PR #32)."""
FPS = 12
PREVIEW_LIMIT = 60


def pinned_module(commit: str, source: str, destination: Path, name: str):
    payload = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=REPO)
    destination.write_bytes(payload)
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bimanual_zoo_body() -> Path:
    """The public body with two grasping chains, found by structure."""

    for source in sorted((ROOT / "assets/general/zoo").glob("*/robot.urdf")):
        robot = ingest_robot(source, robot_id=source.parent.name)
        if len(robot.morphology.grasping_effectors) >= 2 and len(robot.morphology.chains) >= 2:
            return source
    raise RuntimeError("no bimanual body in the public zoo")


def record_program(robot, program: MotionProgramV2, destination: Path, *, label: str, caption: str, source_urdf: Path) -> dict:
    """Compile, execute on native physics with the recorder, gate, and seal."""

    model, manifest = robot.finalized.model, robot.manifest
    trajectory = compile_motion_program(program, model, manifest, sample_hz=SAMPLE_HZ)
    site = program.tracks[0].target
    recorder = PhysicsRecorder(model)
    trace = simulate(model, manifest, trajectory, site_name=site, recorder=recorder)
    record = recorder.finish()
    replay = replay_physics(model, record)
    if not replay["agrees"]:
        raise RuntimeError(f"{label}: recorded-control replay disagrees")
    violations = list(evaluate_gates(model, manifest, trace, GatePolicy()))
    status = "runtime_failure" if violations else "success"
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    outcome = {
        "status": status, "scope": "free-space motion; no object task", "refusal": None, "fault": None,
        "physical_steps": len(record.arrays["state"]) - 1,
        "violations": [asdict(v) for v in violations],
        "unexpected_contacts": [list(p) for p in trace.unexpected_contacts],
        "tracking_error_m": float(trace.tracking_error_m.max()),
        "reference_duration_s": float(program.duration_s),
        "actual_physics_duration_s": float(record.arrays["time_s"][-1]),
    }
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": robot.finalized.mjcf_xml.encode("utf-8"),
        "robot.urdf": source_urdf.read_bytes(), "robot.json": json_bytes(manifest.model_dump(mode="json")),
        "program.json": json_bytes(program.model_dump(mode="json")),
        "reference.json": json_bytes(trajectory.payload()),
        "world.json": json_bytes({"mode": "free_space_comparison", "object_count": 0, "timestep_s": model.opt.timestep,
                                  "gravity": model.opt.gravity.tolist(), "fault": None}),
        "task.json": json_bytes({"goal": "G05", "protocol": "rigby.d05-comparison/1", "prompt": PROMPT, "label": label,
                                 "gate_policy": asdict(GatePolicy()), "interventions": [], "retry_limit": 0}),
        "outcome.json": json_bytes(outcome), "trace.npz": record.to_bytes(),
        "controller.json": json_bytes({"class": "rigby_general.gates.control.ComputedTorqueController", "config": asdict(ControllerConfig())}),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "single recorded execution; the campaign trials carry the three-repeat certification"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    scale = robot.morphology.scale
    metadata = {
        "goal": "G05", "protocol": "rigby.d05-comparison/1", "robot_id": label, "rig_id": manifest.rig_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": status, "fault": False, "simulation_duration_s": float(record.arrays["time_s"][-1]),
        "reference_duration_s": float(program.duration_s), "reference_clock_matches_physics": True,
        "caption": caption, "trace_sha256": record.content_hash(),
        "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
        "camera": {"centre": [scale.workspace_centroid_m.x, scale.workspace_centroid_m.y, scale.workspace_centroid_m.z], "reach": scale.reach_radius_m},
        "task_site": site, "controller": "computed torque", "observation": "model + joint encoders",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, **metadata, "violations": outcome["violations"],
            "unexpected_contacts": outcome["unexpected_contacts"], "tracking_error_m": outcome["tracking_error_m"]}


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], layout: str, destination: Path) -> dict:
    """Tile clips on one clock; each shorter clip holds its final frame."""

    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs: list[str] = []
    filters: list[str] = []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        hold = max(0.0, longest - float(stream["duration"]))
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={hold:.3f}[v{index}]")
    chain = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(f"{chain}xstack=inputs={len(clips)}:layout={layout}[out]")
    subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                          "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], timeout=1200)
    result = probe(ffprobe, destination)
    expected = int(round(longest * FPS)) + 1
    if abs(int(result["nb_read_frames"]) - expected) > 2:
        raise RuntimeError(f"tiled video has {result['nb_read_frames']} frames; expected about {expected}")
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]),
            "width": int(result["width"]), "height": int(result["height"]), "inputs": [c.as_posix() for c in clips]}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d05-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=960:-1",
                               str(staging / "f%04d.png")], timeout=600)
        frames = sorted(staging.glob("f*.png"))[:PREVIEW_LIMIT]
        speed = real_duration_s / (len(frames) / 10.0)
        font = ImageFont.load_default(size=16)
        images = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 26), (17, 24, 39))
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text((10, 5), f"{title} | GIF SUMMARY | approximately {speed:.1f}x speed | full video: {video.name}", fill=(255, 189, 100), font=font)
            images.append(canvas)
        images[0].save(destination, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    return {"gif": destination.name, "frames": len(images), "approximate_speed": round(speed, 2), "summary_of": video.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh destination")
    args.out.mkdir(parents=True)
    index: dict = {"goal": "G05", "demo": "D05", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}

    # -- six synchronized bodies ---------------------------------------------
    summary = json.loads((args.campaign / "summary.json").read_bytes())
    bodies = [b["zoo_id"] for b in summary["bodies"]]
    clips = []
    per_body = []
    for zoo_id in bodies:
        media = args.campaign / zoo_id / "canonical" / "media"
        manifest = verify_bundle(media)
        clips.append(media / "episode.mp4")
        per_body.append({"zoo_id": zoo_id, "media_sha256": hashlib.sha256((media / "manifest.json").read_bytes()).hexdigest(),
                         "frames": manifest["metadata"]["frame_count"], "simulation_duration_s": manifest["metadata"]["simulation_duration_s"],
                         "outcome": manifest["metadata"]["outcome"]})
    if len(clips) != 6:
        raise SystemExit(f"D05 tiles six bodies; the campaign summary lists {len(clips)}")
    # xstack layout for a 3x2 grid of equal-size tiles, row-major in zoo order.
    layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
    six = tile(ffmpeg, ffprobe, clips, layout, args.out / "six-body-synchronized.mp4")
    six["bodies"] = per_body
    six["playback"] = "real time on the shared physics clock; a shorter episode holds its final recorded state"
    six["preview"] = gif_summary(ffmpeg, args.out / "six-body-synchronized.mp4", args.out / "six-body-preview.gif",
                                 "Six bodies, one prompt", six["duration_s"])
    index["six_body"] = six
    print(json.dumps({"six_body_frames": six["frames"], "duration_s": six["duration_s"]}), flush=True)

    # -- dual-arm transition: before and after --------------------------------
    source = bimanual_zoo_body()
    robot = ingest_robot(source, robot_id=source.parent.name)
    inventory = load_inventory()
    planner = OfflineSchemaPlanner(inventory)
    schema = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
    pinned_dir = args.out / "pinned"
    pinned_dir.mkdir()
    before_grounder = pinned_module(BEFORE_COMMIT, "any-robot/src/rigby_general/grounding/grounder.py",
                                    pinned_dir / "grounder-before.py", "rigby_general.grounding.d05_before")
    before = before_grounder.ground(schema, robot.manifest, robot.finalized.model, inventory).program
    after = ground(schema, robot.manifest, robot.finalized.model, inventory).program
    fixture = ROOT / "tests/fixtures/g05_composition/dual_arm_program_before_guard.json"
    frozen = MotionProgramV2.model_validate_json(fixture.read_bytes())
    comparison = {
        "body": source.parent.name, "before_commit": BEFORE_COMMIT,
        "before_matches_frozen_fixture": [t.keyframes for t in before.tracks] == [t.keyframes for t in frozen.tracks] and before.duration_s == frozen.duration_s,
        "before_duration_s": before.duration_s, "after_duration_s": after.duration_s,
    }
    for label, program, caption in (
        ("before", before, f"{source.parent.name} | BEFORE the guard: wrist folds onto forearm (self-collision)"),
        ("after", after, f"{source.parent.name} | AFTER the guard: return leg clear of the body"),
    ):
        physical = args.out / "dual-arm" / label / "physical"
        result = record_program(robot, program, physical, label=f"{source.parent.name}-{label}", caption=caption, source_urdf=source)
        media = render_bundle(physical, args.out / "dual-arm" / label / "media", expected_digest=result["sha256"])
        comparison[label] = {"physical_sha256": result["sha256"], "media_sha256": media["sha256"], "outcome": result["outcome"],
                             "violations": result["violations"], "unexpected_contacts": result["unexpected_contacts"],
                             "tracking_error_m": result["tracking_error_m"], "frames": media["frame_count"],
                             "simulation_duration_s": media["simulation_duration_s"]}
        print(json.dumps({"dual_arm": label, "outcome": result["outcome"], "violations": [v["code"] for v in result["violations"]]}), flush=True)
    if comparison["before"]["outcome"] != "runtime_failure" or comparison["after"]["outcome"] != "success":
        raise SystemExit("the before/after comparison did not reproduce the expected failure and success")
    side = tile(ffmpeg, ffprobe, [args.out / "dual-arm" / "before" / "media" / "episode.mp4", args.out / "dual-arm" / "after" / "media" / "episode.mp4"],
                "0_0|w0_0", args.out / "dual-arm-before-after.mp4")
    side["preview"] = gif_summary(ffmpeg, args.out / "dual-arm-before-after.mp4", args.out / "dual-arm-before-after-preview.gif",
                                  "Dual arm before/after", side["duration_s"])
    comparison["side_by_side"] = side
    index["dual_arm"] = comparison
    files = {p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "index.json"}
    index["files"] = files
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps({"before": comparison["before"]["outcome"], "after": comparison["after"]["outcome"],
                      "before_matches_fixture": comparison["before_matches_frozen_fixture"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
