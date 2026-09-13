"""Record every predeclared invalid request of the G05 roster as a refusal clip.

A refusal happens before any motion, so its evidence is the initial state and
the typed reason: a sealed bundle whose physical record is that single state,
rendered as a two-second slate labelled as a pre-execution refusal. Nothing is
executed, retried or re-drawn. No API or model calls.

    python any-robot/scripts/g05_refusal_clips.py --campaign docs/results/g05-campaign
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.evidence.composition import capture_prompt
from rigby_general.evidence.render import render_bundle


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ROSTER_DIR = ROOT / "assets/general/research-protocols/g05-composition-v1"


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    roster = json.loads((ROSTER_DIR / "roster.json").read_bytes())
    registration = json.loads((ROSTER_DIR / "registration.json").read_bytes())
    summary = json.loads((args.campaign / "summary.json").read_bytes())
    scored = {b["zoo_id"]: b for b in summary["bodies"]}
    index = {"goal": "G05", "created_at_utc": datetime.now(timezone.utc).isoformat(), "roster_sha256": registration["roster_sha256"], "clips": []}
    for body in roster["bodies"]:
        zoo_id = body["zoo_id"]
        source = REPO / body["source_urdf"]
        capability = ingest_capability_body(source)
        robot = capability.robot
        if robot.manifest.rig_id != scored[zoo_id]["rig_id"]:
            raise SystemExit(f"{zoo_id}: intake no longer matches the scored campaign")
        trials = json.loads((args.campaign / zoo_id / "trials.json").read_bytes())
        recorded = {c["case_id"]: c for c in trials["invalid_requests"]}
        for case in body["invalid_requests"]:
            if case["prompt"] is None:
                continue
            destination = args.campaign / zoo_id / "refusals" / case["case_id"]
            if destination.exists():
                raise SystemExit(f"{destination} exists; refusal clips are written once")
            result = capture_prompt(
                robot, case["prompt"], destination / "physical", label=f"{zoo_id}-{case['case_id']}",
                source_urdf=source, start_qpos=np.asarray(case["start_qpos"], dtype=float), goal="G05",
                protocol_reference={"roster_sha256": registration["roster_sha256"], "case_id": case["case_id"]},
                caption=f"{zoo_id} | INVALID REQUEST: {case['case_id']} | {case['reason'][:60]}",
            )
            if result["outcome"] != "pre_execution_refusal":
                raise SystemExit(f"{zoo_id}/{case['case_id']}: expected a pre-execution refusal, got {result['outcome']}")
            media = render_bundle(destination / "physical", destination / "media", expected_digest=result["sha256"])
            outcome = result["outcome_record"]
            row = {
                "zoo_id": zoo_id, "case_id": case["case_id"], "prompt": case["prompt"], "reason": case["reason"],
                "expected": case["expected"], "scored_record_correct": recorded[case["case_id"]]["correct_typed_refusal"],
                "refusal": outcome["refusal"], "refused_leaf_bakes": [
                    {"entry": f["entry_id"], "code": f["failure_code"], "gate": f["failed_gate"]} for f in outcome["refused_leaf_bakes"]
                ],
                "physical_sha256": result["sha256"], "media_sha256": media["sha256"],
                "frames": media["frame_count"], "refusal_slate": media["refusal_slate"],
                "video": (destination / "media" / "episode.mp4").relative_to(args.campaign).as_posix(),
                "preview": (destination / "media" / "preview.gif").relative_to(args.campaign).as_posix(),
                "frames_map": (destination / "media" / "frames.json").relative_to(args.campaign).as_posix(),
            }
            index["clips"].append(row)
            print(json.dumps({"body": zoo_id, "case": case["case_id"], "refusal": outcome["refusal"]["code"] if outcome["refusal"] else None, "frames": media["frame_count"]}), flush=True)
    (args.campaign / "refusals.json").write_bytes(json_bytes(index))
    print(json.dumps({"clips": len(index["clips"]), "sha256": hashlib.sha256((args.campaign / "refusals.json").read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
