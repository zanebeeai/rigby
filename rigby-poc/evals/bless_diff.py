"""Report what a corpus bless changed, by value rather than by diff rendering.

``git diff --stat`` cannot answer the question a bless needs answered. Adding a
second platform's digest to a map rewrites the first entry's line -- ``"x": "a"``
becomes ``"x": "a",`` -- so a **pure addition** renders as deletions. The
Windows bless of a 47-case corpus reported *423 insertions, 282 deletions*, and
every one of those deletions was punctuation or provenance. "Insertions only",
the signal two lanes had agreed to trust, was structurally incapable of
confirming the property it stood for.

This parses both sides and compares by value, which is the property anyone
actually cares about: **did any pre-existing digest change?**

    uv run python -m evals.bless_diff                 # working tree vs HEAD
    uv run python -m evals.bless_diff --against <ref>

Exit code is the number of changed-or-lost digests, so zero means the bless was
purely additive no matter what the diff stat says.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = PROJECT_ROOT / "evals" / "corpus" / "cases"
DIGEST_FIELDS = ("motion_sha256", "metrics_sha256", "observables_sha256")


def _at_ref(path: Path, ref: str) -> dict | None:
    """The committed contents of ``path`` at ``ref``, or None if absent there."""

    relative = path.relative_to(PROJECT_ROOT.parent).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def compare(before: dict, after: dict) -> tuple[list[str], list[str]]:
    """``(changed, added)`` platform-keyed digests between two case documents."""

    changed: list[str] = []
    added: list[str] = []
    for field in DIGEST_FIELDS:
        old, new = before.get(field), after.get(field)
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        for key, value in old.items():
            if key not in new:
                changed.append(f"{field}[{key}] LOST")
            elif new[key] != value:
                changed.append(f"{field}[{key}] CHANGED")
        added.extend(f"{field}[{key}]" for key in new if key not in old)
    return changed, added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--against", default="HEAD", help="git ref to compare against")
    arguments = parser.parse_args(argv)

    cases = sorted(CASES.glob("*/expected.json"))
    if not cases:
        print(f"no corpus cases under {CASES}", file=sys.stderr)
        return 126

    compared = 0
    all_changed: list[str] = []
    added_total = 0
    new_files: list[str] = []
    for path in cases:
        before = _at_ref(path, arguments.against)
        if before is None:
            new_files.append(path.parent.name)
            continue
        after = json.loads(path.read_text(encoding="utf-8"))
        changed, added = compare(before, after)
        all_changed.extend(f"{path.parent.name}.{item}" for item in changed)
        added_total += len(added)
        compared += 1

    print(f"cases compared vs {arguments.against}: {compared}")
    if new_files:
        print(f"cases not present at {arguments.against}: {len(new_files)}")
    print(f"pre-existing digests preserved: {compared * len(DIGEST_FIELDS) - len(all_changed)}"
          f"/{compared * len(DIGEST_FIELDS)}")
    print(f"new platform digests added:     {added_total}")
    if all_changed:
        print("\nCHANGED OR LOST -- this bless is not purely additive:")
        for item in all_changed:
            print(f"  {item}")
    else:
        print("\nno pre-existing digest changed; the bless is purely additive.")
    return min(len(all_changed), 125)


if __name__ == "__main__":
    raise SystemExit(main())
