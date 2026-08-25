"""The ``python -m evals.corpus`` commands, exercised through their entry point.

``bless`` is only useful if it is a single command with a readable diff (plan 03
section 6.2), so the diff's content is asserted, not just its exit code.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest
from evals.corpus import CORPUS_ROOT, PORTABLE_PLATFORM_KEY, load_manifest
from evals.corpus.__main__ import main
from evals.corpus.loader import EXPECTED_FILE
from evals.corpus.seed_cases import SEED_CASES_BY_ID

SAMPLE_CASE_ID = "strike-uppercut-right"


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    copytree(
        CORPUS_ROOT,
        root,
        ignore=lambda _, names: [name for name in names if name == "__pycache__"],
    )
    return root


def _break_case(root: Path, case_id: str) -> None:
    path = root / "cases" / case_id / EXPECTED_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Corrupt this platform's entry.  Adding the reserved portable key would be
    # rejected outright, since no case is portable after 03b.
    payload["motion_sha256"] = dict.fromkeys(payload["motion_sha256"], "f" * 64)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_list_prints_every_case_and_the_deferred_reasons(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(corpus_root), "list"]) == 0
    output = capsys.readouterr().out
    manifest = load_manifest(corpus_root)
    for entry in manifest.cases:
        assert entry.id in output
    assert "grab" in output
    # Every coverage axis is reported, including the three 03b added.  The corpus
    # currently has no deferrals at all, which is the point of 03b -- so asserting
    # that "deferred:" appears would now assert a hole exists.
    for name, axis in manifest.coverage.axes().items():
        assert f"{name}: {len(axis.covered)} covered, {len(axis.deferred)} deferred" in output


def test_list_prints_a_deferral_reason_when_there_is_one(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deferral reason must reach the reader; the live corpus has none to show."""
    path = corpus_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["coverage"]["intent"]["covered"].remove("gesture")
    manifest["coverage"]["intent"]["deferred"] = {"gesture": "a stated reason"}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert main(["--root", str(corpus_root), "list"]) == 0
    output = capsys.readouterr().out
    assert "gesture              deferred: a stated reason" in output


def test_verify_exits_zero_when_nothing_moved(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(corpus_root), "verify", "--case", SAMPLE_CASE_ID])
    assert code == 0
    assert "1 matched, 0 moved" in capsys.readouterr().out


def test_verify_exits_non_zero_when_a_hash_moved(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _break_case(corpus_root, SAMPLE_CASE_ID)
    code = main(["--root", str(corpus_root), "verify", "--case", SAMPLE_CASE_ID])
    assert code == 1
    assert "0 matched, 1 moved" in capsys.readouterr().out


def test_bless_shows_a_readable_diff_and_writes_nothing_by_default(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _break_case(corpus_root, SAMPLE_CASE_ID)
    path = corpus_root / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE
    before = path.read_text(encoding="utf-8")

    assert main(["--root", str(corpus_root), "bless", "--case", SAMPLE_CASE_ID]) == 0
    output = capsys.readouterr().out
    assert f"! {SAMPLE_CASE_ID}" in output
    assert "expected  " in output and "actual    " in output
    assert "f" * 64 in output
    assert "Dry run: nothing written" in output
    assert path.read_text(encoding="utf-8") == before

    assert (
        main(["--root", str(corpus_root), "bless", "--case", SAMPLE_CASE_ID, "--write"])
        == 0
    )
    assert "Wrote 1 expected.json" in capsys.readouterr().out
    assert main(["--root", str(corpus_root), "verify", "--case", SAMPLE_CASE_ID]) == 0


def test_freeze_from_seed_rebuilds_a_committed_case_byte_for_byte(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The recorded prompt must actually reproduce the committed program.

    Without this, ``seed_cases.py`` is a comment rather than provenance.
    """
    root = tmp_path / "corpus"
    assert main(["--root", str(root), "freeze", "--from-seed", SAMPLE_CASE_ID]) == 0
    capsys.readouterr()

    for name in ("program.json", "scene.json"):
        rebuilt = (root / "cases" / SAMPLE_CASE_ID / name).read_text(encoding="utf-8")
        committed = (CORPUS_ROOT / "cases" / SAMPLE_CASE_ID / name).read_text(
            encoding="utf-8"
        )
        assert rebuilt == committed, name

    rebuilt_expected = json.loads(
        (root / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE).read_text(encoding="utf-8")
    )
    committed_expected = json.loads(
        (CORPUS_ROOT / "cases" / SAMPLE_CASE_ID / EXPECTED_FILE).read_text(
            encoding="utf-8"
        )
    )
    for key in ("motion_sha256", "metrics_sha256", "observables_sha256", "frame_count"):
        assert rebuilt_expected[key] == committed_expected[key], key


def test_freeze_rejects_an_unknown_seed_id(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown seed case"):
        main(["--root", str(tmp_path), "freeze", "--from-seed", "not-a-case"])


def test_freeze_requires_an_id_and_family_without_a_seed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--id is required"):
        main(["--root", str(tmp_path), "freeze", "--prompt", "throw a right jab"])


def test_every_seed_case_id_is_a_valid_case_id() -> None:
    assert set(SEED_CASES_BY_ID) == {entry.id for entry in load_manifest().cases}
