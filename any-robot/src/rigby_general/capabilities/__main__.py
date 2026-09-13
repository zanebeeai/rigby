"""Inspect a URDF or replay an archived body inspection, without API calls."""

import argparse
import json
from pathlib import Path

from .inspection import capture_inspection, render_inspection, verify_inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("source", type=Path)
    capture.add_argument("destination", type=Path)
    capture.add_argument("--title")
    render = commands.add_parser("render")
    render.add_argument("source", type=Path)
    render.add_argument("destination", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("source", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture_inspection(args.source, args.destination, title=args.title)
    elif args.command == "render":
        result = render_inspection(args.source, args.destination)
    else:
        result = verify_inspection(args.source)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
