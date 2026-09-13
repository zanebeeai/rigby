"""Before and after the two G15 revisions, on the same seeds: the eight-centimetre layout against the twelve, and the belief-closed loop against the observation-closed one.

Two pairs, each run here in full and sealed segment by segment, then
rendered through the D15 renderer so both halves of a pair carry the same
banner, the same live tree and the same clock:

* ``pitch``: the long arm, five objects, one registered seed. Before: the
  v1 layout, neighbours eight centimetres apart, and the jaw open to its
  limit, where the arm's finger came down on the fourth cube while it
  took the first. After: the v3 layout and the jaw opened as wide as the
  cube needs, the same draw (jitter, mass, friction) under both.
* ``look``: the jaw arm, three objects, one registered seed, on the v1
  layout in both halves -- identical worlds, identical physics until the
  trees diverge. Before: the v1 library, the loop closed on the transfers'
  verdicts, which reported success after a placement knocked the first
  cube out of its cell (a false completion). After: the v2 library, every
  pass ending with a look, which sees the cube out of its cell, transfers
  it again and succeeds -- or fails honestly.
* ``closure``: the jaw arm, three objects, one registered seed, protocol
  v2 in both halves. Before: the compiler's soft grip-joint limits, which
  a lost hold's snap drove 1.3 cm past their range so the fingers crossed
  through one another, and the arm planner's guard counting that finger
  pair, so every later path and IK solution was refused and the chain died
  with the first object. After: limits that hold against the closure's
  force, and the guard leaving the closure's pairs to the closure; the
  chain goes on to the objects it can still place.

Both halves of a pair run under the campaign's own seed label, so the
sensors draw the same noise as the campaign's episode of that seed.

    python any-robot/scripts/g15_before_after.py --out docs/results/g15-before-after --local any-robot/results/g15-before-after
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.evidence.capture import json_bytes
from rigby_general.pipeline import ingest_robot
from rigby_general.sensing import load_policy
from rigby_general.contact import closure as closure_module
from rigby_general.grounding import grounder
from rigby_general.skills.clear_work_area import LAYOUT_V1, LAYOUT_V2, LAYOUT_V3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402
import g15_clearance_campaign as campaign  # noqa: E402
import g15_d15_media as media  # noqa: E402
import g15_protocol as protocol  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PAIRS = {
    "pitch": {"body": "zoo_long_arm", "objects": 5, "seed": 5001,
              "before": {"layout": LAYOUT_V1, "observe_each_pass": True, "opening_sized": False, "label": "v1 layout, 8 cm, jaw open to its limit"},
              "after": {"layout": LAYOUT_V3, "observe_each_pass": True, "opening_sized": True, "label": "v3 layout, jaw opened to the cube's width"},
              "what_changed": "the layout (positions every enabled body has transferred from and to alone, ten centimetres between neighbours) and the jaw's opening (sized to the object rather than the joint limit); the same draw, the same library"},
    "look": {"body": "zoo_jaw_arm", "objects": 3, "seed": 5030,
             "before": {"layout": LAYOUT_V1, "observe_each_pass": False, "opening_sized": False, "label": "v1 tree, loop closed on verdicts"}, "after": {"layout": LAYOUT_V1, "observe_each_pass": True, "opening_sized": False, "label": "v3 tree, every pass ends with a look"},
             "what_changed": "the tree only: every pass ends with a look at the area and the cells, and what the cameras see decides which placements stand; the same world, the same draw, identical physics until the trees diverge"},
    "closure": {"body": "zoo_jaw_arm", "objects": 3, "seed": 5002,
                "before": {"layout": LAYOUT_V2, "observe_each_pass": True, "closure_fix": False, "opening_sized": False, "label": "soft grip limits, guard counting the fingers"},
                "after": {"layout": LAYOUT_V2, "observe_each_pass": True, "closure_fix": True, "opening_sized": False, "label": "grip limits that hold, guard leaving the fingers to the closure"},
                "what_changed": "the closure only: grip-joint limits that hold against the closure's force (fingers cannot cross), and the arm planner's guard no longer counting the closure's own finger pair; the same world, the same draw, the same tree, identical physics until the first lost hold"},
}


def draw_for(corpus: dict, seed: int) -> dict:
    return next(d for d in corpus["nominal"] if d["seed"] == seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--pair", choices=tuple(PAIRS), default=None)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    corpus, registration = campaign.load_registration()
    policy = load_policy(g10.G09 / "policy.json")
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_bytes()) if index_path.exists() else {"goal": "G15", "created_at_utc": datetime.now(timezone.utc).isoformat(), "pairs": {}, "generation_calls": 0}
    index["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for name, spec in PAIRS.items():
        if args.pair and name != args.pair:
            continue
        body, count, seed = spec["body"], spec["objects"], spec["seed"]
        source = ROOT / "assets/general/zoo" / body / "robot.urdf"
        robot = ingest_robot(source, robot_id=body)
        draw = draw_for(corpus, seed)
        halves = {}
        for half in ("before", "after"):
            variant = spec[half]
            world = campaign.world_for(count, draw, layout=variant["layout"])
            episode_id = f"{name}-{half}-{body}-{count}-{seed}"
            # The campaign's own label seeds the sensors' noise streams: both halves draw the same noise as the campaign's episode of this seed,
            # so a pair's physics is identical until the trees or the worlds diverge.
            grounder.GUARD_LEAVES_CLOSURE_PAIRS = closure_module.CLOSURE_LIMITS_HOLD = variant.get("closure_fix", True)
            closure_module.OPENING_SIZED = variant.get("opening_sized", True)
            try:
                row = campaign.run_episode(body, source, robot, world, policy, flat=False, disturbance=None, seed_label=f"{body}-{count}-tree-{seed}", observe_each_pass=variant["observe_each_pass"])
            finally:
                grounder.GUARD_LEAVES_CLOSURE_PAIRS = closure_module.CLOSURE_LIMITS_HOLD = closure_module.OPENING_SIZED = True
            caption = f"{body} | {count} objects | {variant['label']} | seed {seed} | {row['verdict']}" + (f" | {row['root_reason']}" if row["root_reason"] else "")
            sealed = campaign.seal_episode(row, local=args.local / name, label=half, caption=caption,
                                           task_extra={"pair": name, "half": half, "seed": seed, "draw": draw, "objects": count, "layout": variant["layout"].as_json(), "observe_each_pass": variant["observe_each_pass"],
                                                       "closure_fix": variant.get("closure_fix", True), "opening_sized": variant.get("opening_sized", True),
                                                       "library_id": row["library_id"], "registration_sha256": registration["registration_sha256"], "sensor_configuration": protocol.CONFIGURATION})
            public = campaign.public_row(row)
            public.update({"episode_id": episode_id, "zoo_id": body, "objects": count, "seed": seed, "executor": "tree", "sealed": sealed, "half": half, "label": variant["label"], "layout": variant["layout"].name,
                           "observe_each_pass": variant["observe_each_pass"], "closure_fix": variant.get("closure_fix", True), "opening_sized": variant.get("opening_sized", True), "draw_sha256": hashlib.sha256(json_bytes(draw)).hexdigest()})
            rendered = media.render_episode(sealed["segments"], args.out / name / half, title=f"{body} | {count} objects | seed {seed} | {half.upper()}: {variant['label']}", ffmpeg=ffmpeg, ffprobe=ffprobe, total=count,
                                            disturbance=None)
            (args.out / name / f"{half}-row.json").write_bytes(json_bytes(public))
            halves[half] = {"episode_id": episode_id, "verdict": row["verdict"], "root_reason": row["root_reason"], "false_completion": row["false_completion"], "placed_by_oracle": row["placed_by_oracle"],
                            "placed_by_belief": row["placed_by_belief"], "physics_s": row["physics_s"], "passes": row["passes"], "looks": row["looks"], "placements_undone_by_look": row["placements_undone_by_look"],
                            "segments": len(sealed["segments"]), "label": variant["label"], "layout": variant["layout"].name, "observe_each_pass": variant["observe_each_pass"], "library_id": row["library_id"],
                            "closure_fix": variant.get("closure_fix", True), "opening_sized": variant.get("opening_sized", True),
                            "video": f"{name}/{half}/media/episode.mp4", "preview": f"{name}/{half}/media/preview.gif", "frames": f"{name}/{half}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"],
                            "row": f"{name}/{half}-row.json", "oracle": row["oracle"]}
            print(json.dumps({"pair": name, "half": half, "verdict": row["verdict"], "reason": row["root_reason"], "placed": f"{row['placed_by_oracle']}/{count}", "false_completion": row["false_completion"], "physics_s": round(row["physics_s"], 1)}), flush=True)
            del row
        pair = media.tile_pair(ffmpeg, ffprobe, args.out / halves["before"]["video"], args.out / halves["after"]["video"], args.out / name / "pair.mp4")
        summary = media.gif_summary(ffmpeg, args.out / name / "pair.mp4", args.out / name / "pair-preview.gif", f"G15 {name}: before (left) and after (right), {body}, seed {seed}", pair["duration_s"])
        index["pairs"][name] = {"body": body, "objects": count, "seed": seed, "what_changed": spec["what_changed"], "before": halves["before"], "after": halves["after"],
                                "pair": {**{k: v for k, v in pair.items()}, "video": f"{name}/pair.mp4", "preview": {**summary, "gif": f"{name}/pair-preview.gif"}, "layout": "left before, right after; a shorter clip holds its final state"}}
        index_path.write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
