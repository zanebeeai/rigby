"""One offline entry point for capture, integrity, physics replay and rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_core.evidence import verify_bundle
from rigby_core.evidence_archive import unpack_bundle

from .capture import ROOT, capture_canonical, json_bytes, replay_bundle
from .render import render_bundle
from .release import publish_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("robot_id")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--fault", action="store_true", help="Declared zero-actuator-gain diagnostic")
    for name in ("verify", "replay", "render"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        command.add_argument("--expected-sha256")
        if name == "render":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--fps", type=int, default=12)
    suite = commands.add_parser("suite")
    suite.add_argument("--out", type=Path, required=True)
    suite.add_argument("--render", action="store_true")
    suite.add_argument("--fault-robot", help="Public body for the injected diagnostic; defaults to the first listed body")
    release = commands.add_parser("release")
    release.add_argument("suite", type=Path)
    release.add_argument("--out", type=Path, required=True)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("--out", type=Path, required=True)
    unpack.add_argument("--archive-sha256", required=True)
    unpack.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture_canonical(args.robot_id, args.out, fault=args.fault)
    elif args.command == "verify":
        manifest = verify_bundle(args.bundle, args.expected_sha256)
        result = {"integrity_valid": True, "files": len(manifest["files"]), "metadata": manifest["metadata"]}
    elif args.command == "replay":
        result = replay_bundle(args.bundle, args.expected_sha256)
    elif args.command == "render":
        result = render_bundle(args.bundle, args.out, expected_digest=args.expected_sha256, fps=args.fps)
    elif args.command == "release":
        result = publish_release(args.suite, args.out)
    elif args.command == "unpack":
        result = {"manifest_sha256": unpack_bundle(args.archive, args.out, archive_sha256=args.archive_sha256, expected_digest=args.expected_sha256)}
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        rows = []
        cases = [(path.parent.name, False) for path in sorted((ROOT / "any-robot/assets/general/zoo").glob("*/robot.urdf"))]
        if not cases:
            raise RuntimeError("The public development zoo is empty")
        cases.append((args.fault_robot or cases[0][0], True))
        for robot_id, fault in cases:
            label = robot_id + ("-actuation-loss" if fault else "")
            physical = args.out / label / "physical"
            row = capture_canonical(robot_id, physical, fault=fault)
            row["replay"] = replay_bundle(physical, row["sha256"])
            if not row["replay"]["agrees"]:
                raise RuntimeError(f"Saved and recomputed control replay disagrees for {label}")
            if args.render:
                row["media"] = render_bundle(physical, args.out / label / "media", expected_digest=row["sha256"])
            rows.append(row)
            print(json.dumps({"case": label, "outcome": row["outcome"], "physics_duration_s": row["simulation_duration_s"], "sha256": row["sha256"]}), flush=True)
            (args.out / "index.json").write_bytes(json_bytes({"goal": "G01", "complete": False, "cases": rows}))
        result = {"goal": "G01", "complete": True, "cases": rows}
        (args.out / "index.json").write_bytes(json_bytes(result))
    if args.command in {"suite", "release"}:
        print(json.dumps({"goal": "G01", "cases": len(result["cases"]), "out": args.out.as_posix()}, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 1 if result.get("agrees") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
