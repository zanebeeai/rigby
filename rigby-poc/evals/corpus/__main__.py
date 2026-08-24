"""Command line for the golden corpus.

    python -m evals.corpus list
    python -m evals.corpus verify [--case ID ...]
    python -m evals.corpus bless  [--case ID ...] [--write]
    python -m evals.corpus freeze --id ID --family FAMILY (--prompt TEXT | --result DIR)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bless import bless
from .freeze import freeze_from_prompt, freeze_from_result, freeze_from_seed
from .loader import load_corpus, load_manifest
from .models import DeterminismClass, Family
from .seed_cases import SEED_CASES_BY_ID
from .verify import Verdict, compare_case


def _corpus_root(value: str | None) -> Path | None:
    return Path(value).resolve() if value else None


def _require_id(args: argparse.Namespace) -> str:
    if not args.id:
        raise SystemExit("--id is required unless --from-seed is used")
    return str(args.id)


def _require_family(args: argparse.Namespace) -> Family:
    if not args.family:
        raise SystemExit("--family is required unless --from-seed is used")
    return Family(args.family)


def _command_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(_corpus_root(args.root))
    print(f"schema {manifest.schema_version}  blessed against {manifest.compiler_version}")
    print(f"storage policy: {manifest.storage_policy.value}")
    print()
    for entry in manifest.cases:
        actions = ",".join(action.value for action in entry.body_actions) or "-"
        gates = ",".join(gate.value for gate in entry.must_fail)
        suffix = f"  must fail: {gates}" if gates else ""
        print(
            f"  {entry.id:38s} {entry.family.value:19s} {entry.intent.value:19s} "
            f"{actions}{suffix}"
        )
    print()
    for axis_name, axis in manifest.coverage.axes().items():
        print(f"{axis_name}: {len(axis.covered)} covered, {len(axis.deferred)} deferred")
        for name, reason in sorted(axis.deferred.items()):
            print(f"    {name:20s} deferred: {reason}")
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    root = _corpus_root(args.root)
    cases = load_corpus(root)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.id in wanted]
    failures = 0
    skipped = 0
    for case in cases:
        comparison = compare_case(case)
        print(comparison.render())
        failures += int(comparison.verdict is Verdict.MISMATCH)
        skipped += int(comparison.verdict is Verdict.UNBLESSED_PLATFORM)
    print()
    print(f"{len(cases) - failures - skipped} matched, {failures} moved, {skipped} unblessed here")
    return 1 if failures else 0


def _command_bless(args: argparse.Namespace) -> int:
    report = bless(args.case, root=_corpus_root(args.root), write=args.write)
    print(report.render())
    return 0


def _command_freeze(args: argparse.Namespace) -> int:
    root = _corpus_root(args.root)
    determinism = DeterminismClass(args.determinism)
    if args.from_seed:
        seed = SEED_CASES_BY_ID.get(args.from_seed)
        if seed is None:
            raise SystemExit(
                f"unknown seed case {args.from_seed!r}; known: {sorted(SEED_CASES_BY_ID)}"
            )
        # The seed row carries the case's overrides, determinism class and storage
        # policy; --determinism is ignored here on purpose, since a flag defaulting
        # to `portable` would mis-bless a MuJoCo case.
        case = freeze_from_seed(seed, root=root)
    elif args.prompt:
        case = freeze_from_prompt(
            args.prompt,
            case_id=_require_id(args),
            family=_require_family(args),
            tags=args.tag,
            notes=args.notes,
            determinism_class=determinism,
            root=root,
        )
    else:
        case = freeze_from_result(
            Path(args.result).resolve(),
            case_id=_require_id(args),
            family=_require_family(args),
            source_prompt=args.source_prompt,
            tags=args.tag,
            notes=args.notes,
            determinism_class=determinism,
            root=root,
        )
    expected = case.expected
    print(f"froze {case.id} -> {case.root}")
    print(f"  intent           {case.entry.intent.value}")
    print(f"  body actions     {[action.value for action in case.entry.body_actions]}")
    print(f"  frames           {expected.frame_count} at {expected.fps} fps ({expected.duration_s:.3f} s)")
    print(f"  structural_valid {expected.structural_valid}")
    print(f"  motion_sha256    {expected.motion_sha256}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.corpus", description=__doc__)
    parser.add_argument("--root", help="corpus directory (defaults to evals/corpus)")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show the manifest and coverage")
    listing.set_defaults(handler=_command_list)

    verify = sub.add_parser("verify", help="recompile and compare; exits non-zero on a move")
    verify.add_argument("--case", action="append", help="restrict to one case id (repeatable)")
    verify.set_defaults(handler=_command_verify)

    blessing = sub.add_parser(
        "bless",
        help="recompile and print the diff; re-records expectations only with --write",
    )
    blessing.add_argument("--case", action="append", help="restrict to one case id (repeatable)")
    blessing.add_argument(
        "--write",
        action="store_true",
        help="apply the diff. Without it this is a dry run.",
    )
    blessing.set_defaults(handler=_command_bless)

    freeze = sub.add_parser("freeze", help="create a case from a prompt or a live result")
    freeze.add_argument("--id", help="case id, lowercase kebab-case")
    freeze.add_argument("--family", choices=[item.value for item in Family])
    source = freeze.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-seed",
        help="rebuild a committed case from its recorded prompt in seed_cases.py",
    )
    source.add_argument("--prompt", help="plan this prompt with the offline rule planner")
    source.add_argument("--result", help="path to a results/<id>/ directory")
    freeze.add_argument("--source-prompt", help="prompt to record when freezing a result")
    freeze.add_argument("--tag", action="append", default=[], help="repeatable tag")
    freeze.add_argument("--notes", help="one line of context for the manifest")
    freeze.add_argument(
        "--determinism",
        default=DeterminismClass.PORTABLE.value,
        choices=[item.value for item in DeterminismClass],
        help="platform_dependent for cases whose motion passes through MuJoCo",
    )
    freeze.set_defaults(handler=_command_freeze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
