"""The bless comparison must answer by value, not by diff rendering.

`git diff --stat` cannot answer "did any pre-existing digest change?". Adding a
second platform's digest to a map rewrites the first entry's line, so a pure
addition renders as deletions -- the Windows bless of 47 cases reported 423
insertions and 282 deletions with **zero** digests changed. "Insertions only",
the signal two lanes had agreed to trust, was structurally incapable of
confirming the property it stood for.
"""

from __future__ import annotations

import pytest

from evals.bless_diff import DIGEST_FIELDS, compare


#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _case(**digests) -> dict:
    return {field: dict(digests.get(field, {})) for field in DIGEST_FIELDS}


def test_a_second_platform_added_is_purely_additive() -> None:
    """The real case: 423 insertions, 282 deletions, nothing changed."""

    before = _case(motion_sha256={"darwin-arm64": "a"})
    after = _case(motion_sha256={"darwin-arm64": "a", "win32-amd64": "b"})
    changed, added = compare(before, after)
    assert changed == []
    assert added == ["motion_sha256[win32-amd64]"]


def test_a_changed_digest_is_caught() -> None:
    before = _case(motion_sha256={"darwin-arm64": "a"})
    after = _case(motion_sha256={"darwin-arm64": "DIFFERENT"})
    changed, _ = compare(before, after)
    assert changed == ["motion_sha256[darwin-arm64] CHANGED"]


def test_a_dropped_platform_is_caught() -> None:
    """The failure the old `bless --write` actually had: overwrite, not merge."""

    before = _case(motion_sha256={"darwin-arm64": "a"})
    after = _case(motion_sha256={"win32-amd64": "b"})
    changed, added = compare(before, after)
    assert changed == ["motion_sha256[darwin-arm64] LOST"]
    assert added == ["motion_sha256[win32-amd64]"]


def test_every_digest_field_is_compared() -> None:
    before = _case(**{field: {"darwin-arm64": "a"} for field in DIGEST_FIELDS})
    after = _case(**{field: {"darwin-arm64": "b"} for field in DIGEST_FIELDS})
    changed, _ = compare(before, after)
    assert len(changed) == len(DIGEST_FIELDS) == 3


def test_a_non_map_digest_is_skipped_not_crashed() -> None:
    """Pre-03c cases carried a bare string; the tool must survive a mixed tree."""

    before = {"motion_sha256": "a-bare-string"}
    after = {"motion_sha256": {"darwin-arm64": "a"}}
    assert compare(before, after) == ([], [])


def test_provenance_changes_are_not_reported_as_digest_changes() -> None:
    """`environment` being overwritten is a real issue and a different one.

    It is lane `groundtruth`'s to fix in the format; this tool answers only the
    digest question, and conflating the two would make its answer unusable.
    """

    before = {**_case(motion_sha256={"darwin-arm64": "a"}), "environment": {"platform_key": "darwin-arm64"}}
    after = {**_case(motion_sha256={"darwin-arm64": "a"}), "environment": {"platform_key": "win32-amd64"}}
    assert compare(before, after) == ([], [])
