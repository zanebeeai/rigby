"""D06: every gripper-bearing body's transfer in one fixed world, and the
repairs that made it possible, before and after, through identical physics.

Everything here is assembled from recorded physical states. The five-body
reel tiles the campaign's strict fixed-world canonical episodes -- the three
certified transfers, the multifinger hand's failure and the compact arm's
typed refusal slate -- on one physics clock, each shorter episode holding its
final recorded state. The enabled-set reel shows the three bodies the
independent map classed feasible and the primitive certified, side by side.
Each before/after pair shows one repair through identical physics. For the
facing objective and the finger standoff, the same body is run in the same
world through the same primitive with the repair switched off and then as
the code stands, both executions recorded, gated and tiled. For the turn
phase, the before side is the retained pilot run's own rendered episode of
the very seed or world that failed -- the jaw arm's first rendered failing
seed and the long arm's normalized world -- and the after side is that seed
or world run again at the current commit, recorded, sealed and rendered.
The restart seeds are shown where the campaign shows a body needed one. No
API or model calls.

    python any-robot/scripts/g06_d06_media.py --campaign docs/results/g06-campaign \
        --pilot docs/results/g06-campaign-pilot-1 --out docs/results/g06-d06
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

from PIL import Image, ImageDraw, ImageFont
from rigby_core.evidence import verify_bundle

from rigby_general.contact import transfer
from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.grounding import ik
from rigby_general.pipeline import ingest_robot


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g06_transfer_campaign as campaign  # noqa: E402

FPS = 12
PREVIEW_LIMIT = 60


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], layout: str, destination: Path, *, fill: str | None = None) -> dict:
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
    filters.append(f"{chain}xstack=inputs={len(clips)}:layout={layout}" + (f":fill={fill}" if fill else "") + "[out]")
    subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                          "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], timeout=1200)
    result = probe(ffprobe, destination)
    expected = int(round(longest * FPS)) + 1
    if abs(int(result["nb_read_frames"]) - expected) > 2:
        raise RuntimeError(f"tiled video has {result['nb_read_frames']} frames; expected about {expected}")
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]),
            "width": int(result["width"]), "height": int(result["height"]), "inputs": [c.as_posix() for c in clips]}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d06-gif-", dir=destination.parent) as temporary:
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


def canonical_media(campaign_dir: Path, zoo_id: str) -> tuple[Path, dict]:
    media = campaign_dir / zoo_id / "fixed-canonical" / "media"
    manifest = verify_bundle(media)
    return media, {"zoo_id": zoo_id, "media_sha256": hashlib.sha256((media / "manifest.json").read_bytes()).hexdigest(),
                   "frames": manifest["metadata"]["frame_count"], "simulation_duration_s": manifest["metadata"]["simulation_duration_s"],
                   "outcome": manifest["metadata"]["outcome"]}


def record_pair(out: Path, *, name: str, zoo_id: str, env, goal, before_caption: str, after_caption: str, switch_off, after_certifies: bool) -> dict:
    """Run the same body through the same world twice: once with a repair
    switched off, once as the code stands. Both recorded, sealed and rendered.

    The pair has to show the repair: the run without it must not certify,
    and the two runs must differ in outcome or in the gate that stopped
    them. Where the body certifies with the repair, that is required too."""

    source = REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf"
    robot = ingest_robot(source, robot_id=zoo_id)
    comparison = {"body": zoo_id, "repair": name}
    for label, caption in (("before", before_caption), ("after", after_caption)):
        restore = switch_off() if label == "before" else None
        try:
            result, recorder, scene, wall = campaign.run_one(robot, source, env, goal, record=True)
        finally:
            if restore is not None:
                restore()
        physical = out / name / label / "physical"
        sealed = campaign.seal(physical, label=f"{zoo_id}-{name}-{label}", robot=robot, source=source, scene=scene, env=env, goal=goal, result=result,
                               recorder=recorder, track="strict_fixed_world", trial={"kind": f"d06_{name}_{label}"}, caption=caption)
        media = render_bundle(physical, out / name / label / "media", expected_digest=sealed["sha256"])
        comparison[label] = {"physical_sha256": sealed["sha256"], "media_sha256": media["sha256"], "outcome": sealed["outcome"],
                             "certified": result.certified, "failed_gate": result.failed_gate, "violations": [v.code for v in result.violations],
                             "path_seed": result.path_seed, "robot_fixture_contacts": [list(p) for p in result.robot_fixture_contacts],
                             "lift_height_m": result.lift_height_m, "max_penetration_m": result.max_penetration_m,
                             "frames": media["frame_count"], "simulation_duration_s": media["simulation_duration_s"], "wall_seconds": wall}
        print(json.dumps({name: label, "outcome": sealed["outcome"], "gate": result.failed_gate, "seed": result.path_seed}), flush=True)
    before, after = comparison["before"], comparison["after"]
    if before["outcome"] == "success":
        raise SystemExit(f"{name}: the run without the repair certified; the pair shows nothing")
    if after_certifies and after["outcome"] != "success":
        raise SystemExit(f"{name}: the run with the repair was expected to certify and did not ({after['failed_gate']})")
    if (before["outcome"], before["failed_gate"]) == (after["outcome"], after["failed_gate"]):
        raise SystemExit(f"{name}: the two runs stopped at the same gate; the pair shows nothing")
    comparison["after_certifies"] = after_certifies
    return comparison


def record_after_pilot(out: Path, *, name: str, zoo_id: str, pilot_media: Path, pilot_row: dict, env, goal, before_caption: str, after_caption: str, trial: dict) -> dict:
    """The pilot's own rendered failure as the before side; the same world
    run again at the current commit as the after side."""

    manifest = verify_bundle(pilot_media)
    if manifest["metadata"]["outcome"] == "success":
        raise SystemExit(f"{name}: the pilot episode chosen as the before side certified")
    before = {"pilot": True, "media_sha256": hashlib.sha256((pilot_media / "manifest.json").read_bytes()).hexdigest(),
              "media": pilot_media.relative_to(REPO).as_posix(), "physical_sha256": pilot_row["bundle"]["sha256"], "bundle_retained": False,
              "outcome": manifest["metadata"]["outcome"], "certified": pilot_row["certified"], "failed_gate": pilot_row["failed_gate"],
              "violations": [v["code"] for v in pilot_row["violations"]], "path_seed": pilot_row["path_seed"],
              "lift_height_m": pilot_row["lift_height_m"], "max_penetration_m": pilot_row["max_penetration_m"],
              "frames": manifest["metadata"]["frame_count"], "simulation_duration_s": manifest["metadata"]["simulation_duration_s"], "caption": before_caption}
    source = REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf"
    robot = ingest_robot(source, robot_id=zoo_id)
    result, recorder, scene, wall = campaign.run_one(robot, source, env, goal, record=True)
    physical = out / name / "after" / "physical"
    sealed = campaign.seal(physical, label=f"{zoo_id}-{name}-after", robot=robot, source=source, scene=scene, env=env, goal=goal, result=result,
                           recorder=recorder, track=trial["track"], trial=trial, caption=after_caption)
    media = render_bundle(physical, out / name / "after" / "media", expected_digest=sealed["sha256"])
    after = {"physical_sha256": sealed["sha256"], "media_sha256": media["sha256"], "outcome": sealed["outcome"],
             "certified": result.certified, "failed_gate": result.failed_gate, "violations": [v.code for v in result.violations],
             "path_seed": result.path_seed, "robot_fixture_contacts": [list(p) for p in result.robot_fixture_contacts],
             "lift_height_m": result.lift_height_m, "max_penetration_m": result.max_penetration_m,
             "frames": media["frame_count"], "simulation_duration_s": media["simulation_duration_s"], "wall_seconds": wall}
    print(json.dumps({name: "after", "outcome": sealed["outcome"], "gate": result.failed_gate}), flush=True)
    if after["outcome"] != "success":
        raise SystemExit(f"{name}: the run with the repair was expected to certify and did not ({after['failed_gate']})")
    return {"body": zoo_id, "repair": name, "before": before, "after": after, "after_certifies": True, "before_is_pilot_episode": True}


def switch_off_facing():
    """Facing spent only in the position task's null space: the solver as it
    stood before the stacked task, gain and all."""

    previous = ik.STACKED_ITERATIONS
    ik.STACKED_ITERATIONS = 0

    def restore():
        ik.STACKED_ITERATIONS = previous
    return restore


def switch_off_standoff():
    """The grasp point placed at the object's centre regardless of how far
    the fingers reach past it."""

    previous = transfer.grasp_standoff_m
    transfer.grasp_standoff_m = lambda *args, **kwargs: 0.0

    def restore():
        transfer.grasp_standoff_m = previous
    return restore


def switch_off_restart_seeds():
    """Only the rest pose is offered as a seed."""

    previous = transfer.restart_seeds
    transfer.restart_seeds = lambda *args, **kwargs: previous(*args, **kwargs)[:1]

    def restore():
        transfer.restart_seeds = previous
    return restore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True, help="the retained first scored pass, whose rendered failures are the before side of the turn pairs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh destination")
    args.out.mkdir(parents=True)
    env, goal, feasibility, roster, registration = campaign.load_registration()
    index: dict = {"goal": "G06", "demo": "D06", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "registration_sha256": registration["registration_sha256"]}

    # -- every attempted body, one fixed world, one clock ----------------------
    summary = json.loads((args.campaign / "summary.json").read_bytes())
    attempted = [b["zoo_id"] for b in summary["bodies"] if b["feasibility_class"] != "unsupported_by_structure"]
    clips, per_body = [], []
    for zoo_id in attempted:
        media, row = canonical_media(args.campaign, zoo_id)
        clips.append(media / "episode.mp4")
        per_body.append(row)
    if len(clips) != 5:
        raise SystemExit(f"D06 tiles the five gripper-bearing bodies; the campaign attempted {len(clips)}")
    layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0"
    five = tile(ffmpeg, ffprobe, clips, layout, args.out / "five-body-fixed-world.mp4", fill="black")
    five["bodies"] = per_body
    five["playback"] = "real time on the shared physics clock; a shorter episode holds its final recorded state"
    five["preview"] = gif_summary(ffmpeg, args.out / "five-body-fixed-world.mp4", args.out / "five-body-fixed-world-preview.gif",
                                  "Five bodies, one fixed world", five["duration_s"])
    index["five_body"] = five
    print(json.dumps({"five_body_frames": five["frames"], "duration_s": five["duration_s"]}), flush=True)

    # -- the enabled set --------------------------------------------------------
    enabled = [b["zoo_id"] for b in summary["bodies"] if b["feasibility_class"] == "feasible" and b["fixed_canonical_certified"]]
    if len(enabled) < 3:
        raise SystemExit(f"G06.2 asks for at least three enabled configurations; the campaign certified {enabled}")
    rows = [canonical_media(args.campaign, zoo_id) for zoo_id in enabled]
    layout = "|".join("0_0" if i == 0 else "+".join(f"w{j}" for j in range(i)) + "_0" for i in range(len(enabled)))
    trio = tile(ffmpeg, ffprobe, [m / "episode.mp4" for m, _ in rows], layout, args.out / "enabled-set-synchronized.mp4")
    trio["bodies"] = [r for _, r in rows]
    trio["preview"] = gif_summary(ffmpeg, args.out / "enabled-set-synchronized.mp4", args.out / "enabled-set-synchronized-preview.gif",
                                  "Enabled set, one fixed world", trio["duration_s"])
    index["enabled_set"] = trio
    print(json.dumps({"enabled_set": enabled, "frames": trio["frames"]}), flush=True)

    # -- before and after each repair, identical physics -----------------------
    # The facing repair was made for the body whose hand arrived more than a
    # right angle from vertical: the multifinger hand, found by its effector
    # kind, which does not certify either way -- the pair shows the approach
    # it fixed, not a success. The standoff repair is shown on a certified
    # jaw body, where fingers longer than the cube is tall went into the bench
    # without it. The restart seeds are shown on any certified body whose
    # campaign path needed one; at the registered fixture none did.
    pairs = []
    certified = {b["zoo_id"]: b for b in summary["bodies"] if b.get("fixed_canonical_certified")}
    kinds = {}
    for zoo_id in attempted:
        robot = ingest_robot(REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf", robot_id=zoo_id)
        kinds[zoo_id] = robot.morphology.grasping_effectors[0].kind.value
    multifinger = [z for z, k in kinds.items() if k not in ("parallel_jaw",)]
    facing_body = multifinger[0] if multifinger else next(iter(certified))
    pairs.append(record_pair(args.out, name="facing", zoo_id=facing_body, env=env, goal=goal,
                             before_caption=f"{facing_body} | BEFORE: facing as a null-space afterthought; a finger goes through the bench, no grasp",
                             after_caption=f"{facing_body} | AFTER: facing solved with the point; hand vertical, cube gripped and lifted",
                             switch_off=switch_off_facing, after_certifies=facing_body in certified))
    jaw_body = next((z for z in certified if kinds[z] == "parallel_jaw"), next(iter(certified)))
    pairs.append(record_pair(args.out, name="standoff", zoo_id=jaw_body, env=env, goal=goal,
                             before_caption=f"{jaw_body} | BEFORE: grasp point at the cube's centre; fingers driven into the bench",
                             after_caption=f"{jaw_body} | AFTER: grasp point raised by the fingers' reach; fingers clear of the bench",
                             switch_off=switch_off_standoff, after_certifies=True))
    seeded = [zoo_id for zoo_id in certified
              if json.loads((args.campaign / zoo_id / "trials.json").read_bytes())["fixed"]["canonical"]["path_seed"] != "rest"]
    for zoo_id in seeded:
        pairs.append(record_pair(args.out, name="restart-seeds", zoo_id=zoo_id, env=env, goal=goal,
                                 before_caption=f"{zoo_id} | BEFORE: solved from the rest pose only; the arm folds through its base and is refused",
                                 after_caption=f"{zoo_id} | AFTER: solved from a turned seed; the path reaches out clear of the base",
                                 switch_off=switch_off_restart_seeds, after_certifies=True))
    index["restart_seeds_needed_by"] = seeded
    # The turn phase: the pilot's first rendered failing jaw seed, and the
    # long arm's normalized world, each against the same world at HEAD.
    pilot_summary = json.loads((args.pilot / "summary.json").read_bytes())
    index["pilot_commit"] = pilot_summary["provenance"]["commit"]
    turn_pairs = []
    for zoo_id in [b["zoo_id"] for b in pilot_summary["bodies"] if b["feasibility_class"] == "feasible"]:
        pilot_trials = json.loads((args.pilot / zoo_id / "trials.json").read_bytes())
        failing = next((r for r in pilot_trials["fixed"]["trials"] if not r["certified"] and "bundle" in r and r["phases"]), None)
        if failing is not None and kinds.get(zoo_id) == "parallel_jaw" and zoo_id in certified and not any(p["repair"] == "turn-seeded" for p in turn_pairs):
            draw = failing["draw"]
            turn_pairs.append(record_after_pilot(args.out, name="turn-seeded", zoo_id=zoo_id, pilot_media=args.pilot / zoo_id / "fixed" / f"seed-{failing['seed']:03d}" / "media",
                                                 pilot_row=failing, env=campaign.perturbed(env, draw), goal=goal,
                                                 before_caption=f"{zoo_id} | BEFORE the turn phase | seed {failing['seed']} | {failing['failed_gate']}",
                                                 after_caption=f"{zoo_id} | AFTER: hand turned vertical at the hover | seed {failing['seed']}",
                                                 trial={"kind": "d06_turn_after", "track": "strict_fixed_world", "seed": failing["seed"], "draw": draw}))
        normalized_row = pilot_trials["normalized"]
        if not normalized_row["certified"] and normalized_row["phases"] and zoo_id in certified and not any(p["repair"] == "turn-normalized" for p in turn_pairs):
            robot = ingest_robot(REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf", robot_id=zoo_id)
            scaled_env, scaled_goal, normalization = campaign.normalized(env, goal, robot, roster["normalization"]["reference_reach_m"], roster["normalization"]["reference_aperture_m"])
            turn_pairs.append(record_after_pilot(args.out, name="turn-normalized", zoo_id=zoo_id, pilot_media=args.pilot / zoo_id / "normalized" / "media",
                                                 pilot_row=normalized_row, env=scaled_env, goal=scaled_goal,
                                                 before_caption=f"{zoo_id} | BEFORE the turn phase | normalized world x{normalization['length_factor']:.2f} | {normalized_row['failed_gate']}",
                                                 after_caption=f"{zoo_id} | AFTER: hand turned vertical at the hover | normalized world x{normalization['length_factor']:.2f}",
                                                 trial={"kind": "d06_turn_after", "track": "capability_normalized", "normalization": normalization}))
    pairs.extend(turn_pairs)
    for pair in pairs:
        name = pair["repair"]
        before_clip = (REPO / pair["before"]["media"] / "episode.mp4") if pair.get("before_is_pilot_episode") else args.out / name / "before" / "media" / "episode.mp4"
        side = tile(ffmpeg, ffprobe, [before_clip, args.out / name / "after" / "media" / "episode.mp4"],
                    "0_0|w0_0", args.out / f"{name}-before-after.mp4")
        side["preview"] = gif_summary(ffmpeg, args.out / f"{name}-before-after.mp4", args.out / f"{name}-before-after-preview.gif",
                                      f"{pair['body']} {name} before/after", side["duration_s"])
        pair["side_by_side"] = side
    index["repairs"] = pairs

    files = {p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "index.json"}
    index["files"] = files
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps({"pairs": [(p["repair"], p["body"], p["before"]["failed_gate"]) for p in pairs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
