"""What the corpus actually does on *this* machine, digest by digest.

``python -m evals.corpus verify`` answers "do the digests match what is
committed" with a yes or a no. When the answer is no it does not say whether the
motion changed or only its last bits did, and those are different facts with
different remedies: one is a re-bless, the other is a regression.

This probe records, per case, the freshly compiled digests beside the committed
ones *and* the maximum numeric deviation from the committed slim clip. Float
drift below the slim clip's rounding shows as a deviation of exactly zero with a
moved digest; a real motion change shows as a deviation you can read in radians
or metres, at a named path.

It exists because plan 14's regression check cannot be "corpus verify is green"
on a tree where it already is not. The check is instead "nothing moved relative
to a snapshot taken on this machine before the change", and this writes that
snapshot.

    python -m evals.probes.corpus_snapshot results/physics-verifier/before.json
    python -m evals.probes.corpus_snapshot results/physics-verifier/after.json
    python -m evals.probes.corpus_snapshot --compare before.json after.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections.abc import Iterator
from pathlib import Path

from evals.corpus.hashing import platform_key
from evals.corpus.loader import CorpusCase, compile_case, load_corpus, observe
from evals.corpus.verify import SlimClipShapeError, compare_slim_clip

SLIM_TOLERANCE = 1e-3


def iter_committed_clips(root: Path | None = None) -> Iterator[tuple[str, dict]]:
    """Yield ``(case_id, slim_clip)`` for every case that committed one.

    Reads the frames the corpus already stores rather than recompiling. A probe
    that answers a question by looping ``compile_case`` over 47 cases is the
    regression ``tests/test_corpus_compile_budget.py`` exists to catch.
    """

    for case in load_corpus(root):
        path = case.slim_clip_path
        if not path.is_file():
            continue
        with gzip.open(path, "rt") as handle:
            yield case.id, json.load(handle)


def _row(case: CorpusCase) -> dict:
    clip = compile_case(case)
    observed = observe(
        case.id,
        clip,
        case.expected.determinism_class,
        solver_used=case.expected.solver_used,
    )
    key = platform_key(case.expected.solver_used)
    row: dict = {
        "platform_key": key,
        "frame_count": observed.frame_count,
        "committed_frame_count": case.expected.frame_count,
        "contact_count": observed.contact_count,
        "success": observed.success,
        "structural_valid": observed.structural_valid,
    }
    for name in case.expected.DIGESTS:
        row[name] = observed.resolve(name, key)
        row[f"committed_{name}"] = case.expected.resolve(name, key)
    if case.slim_clip_path.is_file() and clip.frames:
        try:
            report = compare_slim_clip(case, clip)
            row["slim_max_deviation"] = report.max_deviation
            row["slim_at"] = report.at
        except SlimClipShapeError as exc:
            # A frame-count change is not a deviation, it is a different clip. Record
            # it as its own kind of move rather than letting the walk raise.
            row["slim_error"] = f"{type(exc).__name__}: {exc}"
    return row


def snapshot(root: Path | None = None) -> dict:
    cases = {case.id: _row(case) for case in load_corpus(root)}
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": cases,
    }


def render(report: dict) -> str:
    cases = report["cases"]
    moved = [
        case_id
        for case_id, row in cases.items()
        if row.get("motion_sha256") != row.get("committed_motion_sha256")
    ]
    shape = {c: r["slim_error"] for c, r in cases.items() if "slim_error" in r}
    over = {
        c: (r["slim_max_deviation"], r.get("slim_at", ""))
        for c, r in cases.items()
        if r.get("slim_max_deviation", 0.0) > SLIM_TOLERANCE
    }
    drift = [
        c
        for c, r in cases.items()
        if c in moved and c not in shape and c not in over and "slim_max_deviation" in r
    ]
    lines = [
        f"{len(cases)} cases on this machine",
        f"  motion digests differing from committed: {len(moved)}",
        f"  of those, identical to the committed clip at slim precision: {len(drift)}",
        f"  exceeding the {SLIM_TOLERANCE} slim tolerance: {len(over)}",
        f"  frame count or shape changed: {len(shape)}",
    ]
    for case_id, (deviation, at) in sorted(over.items()):
        lines.append(f"    ! {case_id}: max deviation {deviation:.4f} at {at}")
    for case_id, message in sorted(shape.items()):
        row = cases[case_id]
        lines.append(
            f"    ! {case_id}: frames {row['committed_frame_count']} -> {row['frame_count']} ({message})"
        )
    return "\n".join(lines)


def compare(before: dict, after: dict) -> tuple[str, int]:
    """Did anything move between two snapshots of the same machine?"""

    names = sorted(set(before["cases"]) | set(after["cases"]))
    changed: list[str] = []
    for case_id in names:
        left = before["cases"].get(case_id)
        right = after["cases"].get(case_id)
        if left is None or right is None:
            changed.append(f"  ! {case_id}: present in only one snapshot")
            continue
        for field in (
            "motion_sha256",
            "metrics_sha256",
            "observables_sha256",
            "frame_count",
        ):
            if left.get(field) != right.get(field):
                changed.append(
                    f"  ! {case_id}.{field}: {left.get(field)} -> {right.get(field)}"
                )
    if not changed:
        return f"{len(names)} cases, none moved between the two snapshots", 0
    return "\n".join([f"{len(changed)} field(s) moved:", *changed]), 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=None)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        before = json.loads(args.compare[0].read_text())
        after = json.loads(args.compare[1].read_text())
        message, code = compare(before, after)
        print(message)
        return code

    report = snapshot()
    print(render(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", newline="\n"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
