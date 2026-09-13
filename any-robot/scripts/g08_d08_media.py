"""D08: before and after transition repair, through identical physics.

Three pairs, each the same two skills on the same body in the same world,
left composed without the boundary check and right composed through it:

- a joint-limit approach: the reviewed dual-arm failure's keyframe, the
  left wrist inside the margin of the limit it folded to in the pre-guard
  return, followed by the transfer; without the check the transfer runs
  from there and the free-motion joint gate fires on the composition; with
  it the wrist is moved inside the margin, the boundary verified again,
  and the transfer certifies;
- carry to place: the acquisition cut part way through the carry, the
  cube still held and the arm still moving, followed by the placement;
  without the check the placement begins from a moving arm; with it the
  arm is braked along a velocity-matched quintic with the cube held, the
  boundary verified again, and the placement certifies;
- a contact-mode change: the acquisition, ending holding the cube,
  followed by a free-space return to rest, which begins free; without the
  check the return runs with the cube in hand and carries it away from
  the world's places; with it the composition is rejected before the
  return moves, and the placement -- a skill that changes what is held --
  is what must run between them, shown composed through both boundaries.

Every side is one continuous physics record, sealed and rendered in full,
and the two sides of a pair are tiled on one clock. No API or model calls.

    python any-robot/scripts/g08_d08_media.py --out docs/results/g08-d08
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
from rigby_core.simulation.recording import PhysicsRecorder

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.transitions import compose


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g08_transition_campaign as campaign  # noqa: E402

FPS = 12
PREVIEW_LIMIT = 60


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], layout: str, destination: Path) -> dict:
    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs, filters = [], []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(stream['duration'])):.3f}[v{index}]")
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout={layout}[out]")
    subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264",
                           "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], timeout=1200)
    result = probe(ffprobe, destination)
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]), "inputs": [c.as_posix() for c in clips]}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d08-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=960:-1", str(staging / "f%04d.png")], timeout=600)
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


def side(out: Path, body: campaign.Body, name: str, label: str, case: dict, *, validate: bool, caption: str, corpus_sha256: str) -> dict:
    result, recorder, wall = campaign.run_case(body, case, validate=validate)
    physical = out / name / label / "physical"
    sealed = campaign.seal(physical, body=body, case=case, result=result, recorder=recorder, caption=caption, label=f"{body.zoo_id}-{name}-{label}", corpus_sha256=corpus_sha256)
    media = render_bundle(physical, out / name / label / "media", expected_digest=sealed["sha256"])
    summary = result.summary()
    print(json.dumps({name: label, "status": sealed["outcome"], "success": result.composed_success, "rejected": result.rejected, "rejection": result.rejection,
                      "repairs": [r.kind.value for r in result.repairs], "gates": [g["code"] for g in result.gate_violations]}), flush=True)
    return {"physical_sha256": sealed["sha256"], "media_sha256": media["sha256"], "outcome": sealed["outcome"], "composed_success": result.composed_success,
            "validated": validate, "rejected": result.rejected, "rejection": result.rejection, "repairs": summary["repairs"], "verdicts": summary["verdicts"],
            "gate_violations": summary["gate_violations"], "second": summary["second"], "first": summary["first"], "boundary": summary["boundary"],
            "frames": media["frame_count"], "simulation_duration_s": media["simulation_duration_s"], "wall_seconds": wall}


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
    corpus, registration, env, goal = campaign.load_corpus()
    corpus_sha256 = registration["files"]["corpus.json"]
    index: dict = {"goal": "G08", "demo": "D08", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "registration_sha256": registration["registration_sha256"], "pairs": []}

    # 1. the joint-limit approach: the reviewed dual-arm failure.
    reviewed = next(c for c in corpus["injected_cases"] if c["case_id"].startswith("zoo_dual_arm-reviewed_dual_arm_failure"))
    dual = campaign.Body("zoo_dual_arm", env, goal)
    pair = {"name": "joint-limit-approach", "body": "zoo_dual_arm", "case": reviewed}
    pair["before"] = side(args.out, dual, "joint-limit-approach", "before", reviewed, validate=False, corpus_sha256=corpus_sha256,
                          caption="zoo_dual_arm | BEFORE: the wrist at its limit from the reviewed return, then the transfer, no boundary check")
    pair["after"] = side(args.out, dual, "joint-limit-approach", "after", reviewed, validate=True, corpus_sha256=corpus_sha256,
                         caption="zoo_dual_arm | AFTER: the wrist moved inside its margin, the boundary verified, then the transfer")
    index["pairs"].append(pair)

    # 2. carry to place: the carry cut, the cube held, the arm moving.
    jaw = campaign.Body("zoo_jaw_arm", env, goal)
    cut = {"case_id": "zoo_jaw_arm-carry-to-place", "zoo_id": "zoo_jaw_arm", "kind": "velocity_too_high",
           "first": {"skill": "acquire_carry_cut", "stop_fraction": 0.5}, "second": {"skill": "place"}, "expect": "repaired_then_reverified",
           "note": "the acquisition's carry is cut half way; the placement must not begin from a moving arm"}
    carry_body = _with_cut_carry(jaw)
    pair = {"name": "carry-to-place", "body": "zoo_jaw_arm", "case": cut}
    pair["before"] = side(args.out, carry_body, "carry-to-place", "before", cut, validate=False, corpus_sha256=corpus_sha256,
                          caption="zoo_jaw_arm | BEFORE: the carry cut mid-flight, the placement begun from a moving arm, no boundary check")
    pair["after"] = side(args.out, carry_body, "carry-to-place", "after", cut, validate=True, corpus_sha256=corpus_sha256,
                         caption="zoo_jaw_arm | AFTER: braked to rest with the cube held, the boundary verified, then the placement")
    index["pairs"].append(pair)

    # 3. a contact-mode change: holding into a free-space return.
    mode = {"case_id": "zoo_jaw_arm-contact-mode-change", "zoo_id": "zoo_jaw_arm", "kind": "holding_into_free",
            "first": {"skill": "acquire_carry"}, "second": {"skill": "return_home"}, "expect": "rejected",
            "note": "ends holding the cube; the return to rest begins free and would carry the cube away"}
    pair = {"name": "contact-mode-change", "body": "zoo_jaw_arm", "case": mode}
    pair["before"] = side(args.out, jaw, "contact-mode-change", "before", mode, validate=False, corpus_sha256=corpus_sha256,
                          caption="zoo_jaw_arm | BEFORE: the return to rest run while holding the cube, no boundary check")
    pair["after"] = side(args.out, jaw, "contact-mode-change", "after", mode, validate=True, corpus_sha256=corpus_sha256,
                         caption="zoo_jaw_arm | AFTER: rejected before the return moves; a contact change must run first")
    pair["inserted"] = _inserted(args.out, jaw, corpus_sha256)
    index["pairs"].append(pair)

    for pair in index["pairs"]:
        name = pair["name"]
        tiled = tile(ffmpeg, ffprobe, [args.out / name / "before" / "media" / "episode.mp4", args.out / name / "after" / "media" / "episode.mp4"], "0_0|w0_0", args.out / f"{name}-before-after.mp4")
        tiled["preview"] = gif_summary(ffmpeg, args.out / f"{name}-before-after.mp4", args.out / f"{name}-before-after-preview.gif", f"{pair['body']} {name} before/after", tiled["duration_s"])
        pair["side_by_side"] = tiled
    files = {p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "index.json"}
    index["files"] = files
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps({"pairs": [(p["name"], p["before"]["outcome"], p["after"]["outcome"]) for p in index["pairs"]]}))
    return 0


def _with_cut_carry(body: campaign.Body) -> campaign.Body:
    """A body whose ``acquire_carry_cut`` skill is the acquisition interrupted
    half way through its carry, the cube still held."""

    from rigby_general.contact.transfer import attempt_transfer
    from rigby_general.transitions.compose import Skill, SkillOutcome
    from rigby_general.transitions.boundary import initiation_for
    from rigby_core.skills import ContactMode

    original = body.skill

    def skill(spec: dict):
        if spec["skill"] != "acquire_carry_cut":
            return original(spec)
        fraction = float(spec.get("stop_fraction", 0.5))

        def run(resume, recorder, should_stop):
            carry_started: dict = {}

            def stopper(now: float) -> bool:
                return bool(should_stop and should_stop(now)) or ("at" in carry_started and now >= carry_started["at"])

            # First find when the carry begins and how long it lasts, dry.
            dry = attempt_transfer(body.manifest, body.scene, body.effector, body.frame, phase_range=("approach", "carry"))
            carry = next(p for p in dry.phases if p.name == "carry")
            carry_started["at"] = carry.start_s + fraction * (carry.end_s - carry.start_s)
            result = attempt_transfer(body.manifest, body.scene, body.effector, body.frame, recorder=recorder, resume=resume, should_stop=stopper, phase_range=("approach", "carry"))
            return SkillOutcome(skill_id="acquire_carry_cut", executed=result.executed, certified=result.executed and result.interrupted and result.opposition_achieved,
                                gate=None if result.interrupted else result.failed_gate, continuation=result.continuation(), phases=[p.name for p in result.phases],
                                violations=[v.code for v in result.violations if v.code != "interrupted"], physics_s=(result.final_time_s - float(result.times_s[0])) if result.executed else 0.0,
                                interrupted=result.interrupted, detail="the carry cut at %.0f%% with the cube held" % (100 * fraction),
                                measurements={"lift_height_m": result.lift_height_m, "hold_s": result.hold_s, "peak_force_n": result.peak_force_n, "max_penetration_m": result.max_penetration_m})
        return Skill(skill_id="acquire_carry_cut", initiation=initiation_for("acquire_carry_cut", ContactMode.FREE, body.effector, requires_object=False, requires_resting=True), run=run)

    body.skill = skill  # type: ignore[method-assign]
    return body


def _inserted(out: Path, body: campaign.Body, corpus_sha256: str) -> dict:
    """The acquisition, then the placement, then the return, composed through
    both boundaries on one record: the contact change inserted where the
    check refused the direct composition."""

    recorder = PhysicsRecorder(body.model)
    acquire, place, back = body.skill({"skill": "acquire_carry"}), body.skill({"skill": "place"}), body.skill({"skill": "return_home"})
    first = compose(body.model, body.manifest, body.effector, body.frame, acquire, place, recorder, guard=body.guard)
    second = None
    if first.composed_success and first.second is not None:
        second = compose(body.model, body.manifest, body.effector, body.frame, place, back, recorder, guard=body.guard, first_done=first.second)
    case = {"case_id": "zoo_jaw_arm-contact-mode-change-inserted", "zoo_id": "zoo_jaw_arm", "kind": "inserted_contact_change",
            "first": {"skill": "place"}, "second": {"skill": "return_home"}, "expect": "composed_success",
            "note": "acquire and carry, then place (the contact change), then the free return, each boundary checked on one record",
            "preceding_composition": first.summary()}
    result = second if second is not None else first
    physical = out / "contact-mode-change" / "inserted" / "physical"
    sealed = campaign.seal(physical, body=body, case=case, result=result, recorder=recorder, label="zoo_jaw_arm-contact-mode-change-inserted", corpus_sha256=corpus_sha256,
                           caption="zoo_jaw_arm | acquire and carry, then place (the contact change), then return: each boundary checked")
    media = render_bundle(physical, out / "contact-mode-change" / "inserted" / "media", expected_digest=sealed["sha256"])
    print(json.dumps({"contact-mode-change": "inserted", "acquire_to_place": first.composed_success, "place_to_return": None if second is None else second.composed_success}), flush=True)
    return {"physical_sha256": sealed["sha256"], "media_sha256": media["sha256"], "outcome": sealed["outcome"], "acquire_to_place_success": first.composed_success,
            "place_to_return_success": None if second is None else second.composed_success, "place_to_return_boundary": None if second is None else second.summary()["boundary"],
            "frames": media["frame_count"], "simulation_duration_s": media["simulation_duration_s"]}


if __name__ == "__main__":
    raise SystemExit(main())
