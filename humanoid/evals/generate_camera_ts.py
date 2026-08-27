"""Generate ``frontend/src/generated/camera.ts`` from ``config/camera.v1.json``.

Plan 08 §3.3, G5. FOV 94.0 had six hand-maintained copies across Python and
TypeScript, 1600x900 had six, and the neutral gaze three. Values that must agree
across a language boundary reach the frontend by generation, not by retyping.

Run: ``uv run python -m evals.generate_camera_ts``          (write)
     ``uv run python -m evals.generate_camera_ts --check``  (fail if stale)

The ``--check`` form is what CI and the test suite use, so a hand-edit of the
generated file or a change to the JSON without regenerating fails loudly rather
than diverging silently -- which is the failure this whole plan exists to remove.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG = PROJECT_ROOT / "config" / "camera.v1.json"
GENERATED_TS = PROJECT_ROOT / "frontend" / "src" / "generated" / "camera.ts"

HEADER = """// GENERATED FILE -- DO NOT EDIT.
//
// Source: config/camera.v1.json
// Regenerate: uv run python -m evals.generate_camera_ts
//
// These values must agree across the Python/TypeScript boundary. Before this
// file they were retyped by hand in six places and were free to diverge.
"""


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "] as const"
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return json.dumps(value)


def render() -> str:
    document = json.loads(CAMERA_CONFIG.read_text(encoding="utf-8"))
    lines = [HEADER]
    for name, entry in document["camera"].items():
        unit = entry["unit"]
        lines.append(f"/** {unit} */")
        lines.append(f"export const {_camel(name)} = {_literal(entry['value'])};")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the generated file is missing or stale",
    )
    arguments = parser.parse_args(argv)
    expected = render()

    if arguments.check:
        if not GENERATED_TS.exists():
            print(f"{GENERATED_TS} is missing; run without --check", file=sys.stderr)
            return 1
        actual = GENERATED_TS.read_text(encoding="utf-8")
        if actual != expected:
            print(
                f"{GENERATED_TS.relative_to(PROJECT_ROOT)} is stale.\n"
                "Run: uv run python -m evals.generate_camera_ts",
                file=sys.stderr,
            )
            return 1
        return 0

    GENERATED_TS.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_TS.write_text(expected, encoding="utf-8")
    print(f"wrote {GENERATED_TS.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
