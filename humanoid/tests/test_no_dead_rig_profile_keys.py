"""Rig-profile blocks that nothing reads.

`test_no_orphan_thresholds` covers `config/thresholds.v1.json`. The rig profile
is the other config file the compiler reads, it is larger, and nothing has ever
checked it -- so a block can be added, described, and never consumed, which is
the dead-config defect one file over from where anyone looks.

**A ledger, not a gate.** Six blocks were dead when this ledger was written
(`required_bones` has since gained a reader and come off), and a strict guard
red on six at once acquires an allowlist within a week -- the shape this repository
learned from the hardcoded-threshold guard, which was red on 79 sites. So the
known-dead set is written down and may only ever SHRINK: a new dead block fails
the suite, and deleting one requires lowering the count in the same commit.

**Coverage is by ancestor, deliberately.** A key inside an iterated dict never
appears literally anywhere -- `grip_presets.crate_grip.segment_angles_rad`
entries are bone names read by iteration, and flagging them would have produced
9 false positives against 6 real findings. A block counts as read if it, or any
ancestor, is named in the source. Measured 2026-08-25: the flat form flags 15,
the ancestor form flags 6, and hand-checking all 6 confirmed none has a reader.

**Word boundaries matter here and the measurement showed why.** `rest_pose` is
mentioned four times outside the profile, and all four are *test function names*
containing the substring -- `test_the_rest_pose_clip_violates_nothing_at_all`.
A substring test would have called it read. It is not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES = PROJECT_ROOT / "config" / "rig_profiles"
SEARCH_ROOTS = ("src", "evals", "tests", "frontend/src", "config")
SUFFIXES = {".py", ".ts", ".json", ".yaml", ".yml"}

#: Blocks with no reader as of 2026-08-25, each with what it appears to describe.
#: This list may only ever get SHORTER. Deleting a block means removing its entry
#: here in the same commit; `test_the_dead_key_ledger_only_shrinks` asserts the
#: count, so adding an entry to silence a failure fails a different test.
#:
#: Not deleted here: removing blocks from a rig profile is a change to the rig
#: contract and belongs in a PR about the rig, not in the guard that found them.
KNOWN_DEAD: dict[str, str] = {
    "canonical_standard": "names the rig convention; no consumer",
    "clip_rotation_contract": "describes clip rotation order; no consumer",
    "end_effectors": "hand/foot terminal bones; no consumer",
    "rest_pose": "rest-pose reference; the four textual hits are test NAMES",
    "root_motion_source": "declares where root motion comes from; no consumer",
}
LEDGER_HIGH_WATER_MARK = 5  # == len(KNOWN_DEAD); required_bones gained a reader (test_v2_canonical_human.py) and came off


def _corpus() -> str:
    chunks = []
    for root in SEARCH_ROOTS:
        base = PROJECT_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            # This file names every dead block in its own ledger, so including
            # it would make each entry look read -- the guard reviving its own
            # findings. It caught exactly that on first run. `test_markers_complete`
            # and `test_corpus_compile_budget` both exclude themselves for the
            # same reason; a scanner is not exempt from what it scans for.
            if path.suffix not in SUFFIXES or "rig_profiles" in path.parts:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if any(part in {"node_modules", ".venv", "dist", "__pycache__"} for part in path.parts):
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _walk(node: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(path + (key,))
            found.extend(_walk(value, path + (key,)))
    return found


def _dead_blocks(profile: Path, corpus: str) -> list[str]:
    document = json.loads(profile.read_text(encoding="utf-8"))
    paths = _walk(document)
    assert paths, f"{profile.name} yielded no keys -- the scan found nothing to check"
    read = {part for path in paths for part in path
            if re.search(r"\b" + re.escape(part) + r"\b", corpus)}
    dead = [".".join(p) for p in paths if not any(part in read for part in p)]
    return [d for d in dead if not any(d.startswith(other + ".") for other in dead)]


def test_no_new_dead_block_appears_in_a_rig_profile() -> None:
    profiles = sorted(PROFILES.glob("*.json"))
    assert profiles, f"no rig profiles found under {PROFILES} -- the scan is looking at nothing"
    corpus = _corpus()
    assert len(corpus) > 100_000, f"source scan collected only {len(corpus)} chars; it is not reading the tree"

    new = sorted(
        block
        for profile in profiles
        for block in _dead_blocks(profile, corpus)
        if block not in KNOWN_DEAD
    )
    assert not new, (
        f"these rig-profile blocks have no reader and are not on the ledger: {new}. "
        "Wire them up or delete them; adding to KNOWN_DEAD to silence this fails "
        "test_the_dead_key_ledger_only_shrinks."
    )


def test_the_dead_key_ledger_only_shrinks() -> None:
    assert len(KNOWN_DEAD) <= LEDGER_HIGH_WATER_MARK, (
        f"the dead-key ledger grew to {len(KNOWN_DEAD)} from {LEDGER_HIGH_WATER_MARK}. "
        "It may only shrink."
    )


def test_every_ledger_entry_is_still_dead() -> None:
    """The ledger must not outlive what it describes.

    A block that gains a reader has to come off, or the ledger starts asserting
    the opposite of the truth -- the same staleness that made
    `SEEDED_SAFETY_FIELDS` need a live producer check.
    """

    corpus = _corpus()
    dead = {block for profile in sorted(PROFILES.glob("*.json")) for block in _dead_blocks(profile, corpus)}
    revived = sorted(key for key in KNOWN_DEAD if key not in dead)
    assert not revived, (
        f"these are on the dead-key ledger but now have readers: {revived}. "
        "Delete them from KNOWN_DEAD and lower LEDGER_HIGH_WATER_MARK."
    )


def test_the_scan_distinguishes_a_reader_from_a_name_that_merely_contains_it() -> None:
    """`rest_pose` is why this is a word-boundary match and not a substring one."""

    assert re.search(r"\brest_pose\b", "profile['rest_pose']")
    assert not re.search(r"\brest_pose\b", "def test_the_rest_pose_clip_violates_nothing() -> None:")
