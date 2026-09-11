"""The demo registry: add, validate, build.

One JSON file per demo under demos/registry/, one folder of media per demo
under demos/media/ (or a reference to media that already lives in the
repository), and one static page, demos/index.html, that plays all of them.
Files never collide across branches because every id carries the date, the
prompt slug and eight hex characters of the media digest, so three people
adding demos on three branches merge without a conflict.

    uv run python demos/tools/demo_tools.py add --title "..." --prompt "..." \
        --tier any-robot --embodiment zoo_long_arm --kind gif \
        --how "uv run python scripts/build_demos.py --robots zoo_long_arm" \
        results/media/zoo_long_arm-pick-up-the-block.gif
    uv run python demos/tools/demo_tools.py validate
    uv run python demos/tools/demo_tools.py build          # writes demos/index.html
    uv run python demos/tools/demo_tools.py build --check  # CI: committed page is current

`add` records who made the demo (from `git config user.name` and
`user.email`, or `--author`), the commit, the branch and whether the tree
was dirty, so a demo carries its provenance from the moment it exists.
Schema: demos/schema/demo.v1.md. No third-party dependency: the validator
is plain Python so the CI job needs nothing but the workspace interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "demos"
REGISTRY = DEMOS / "registry"
MEDIA = DEMOS / "media"
PAGE = DEMOS / "index.html"

SCHEMA = "rigby.demo/1"
TIERS = ("core", "humanoid", "any-robot")
KINDS = ("gif", "video", "image", "humanoid-bones-v1", "anyrobot-qpos-v1", "gripper-links-v1")
MEDIA_EXT = {".gif", ".webm", ".mp4", ".png", ".jpg", ".jpeg"}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ENTRY_BYTES = 24 * 1024 * 1024

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, default: str = "") -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "demo"


# --------------------------------------------------------------------------
# add


def _core_demos():
    """`rigby_core.demos` loaded by path, so this script needs no installed package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rigby_core_demos", ROOT / "core" / "src" / "rigby_core" / "demos.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def cmd_add(a: argparse.Namespace) -> int:
    demos = _core_demos()
    spec = demos.DemoSpec(
        title=a.title, prompt=a.prompt, tier=a.tier, embodiment=a.embodiment, kind=a.kind, how=a.how,
        files=[Path(f).resolve() for f in a.files], payload=Path(a.payload).resolve() if a.payload else None,
        who=a.author, produced_at=a.produced_at, outcome=a.outcome, notes=a.notes, tags=a.tag or [],
        role=a.role, keep_paths=a.keep_paths,
    )
    try:
        out = demos.register(spec, ROOT)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(out.relative_to(ROOT).as_posix())
    return 0


# --------------------------------------------------------------------------
# validate


def load_entries() -> list[tuple[Path, dict]]:
    return _core_demos().load_entries(ROOT)


def validate_entry(path: Path, e: dict, seen: set[str]) -> list[str]:
    errs = []
    rel = path.relative_to(ROOT).as_posix()

    def need(key, typ):
        if key not in e:
            errs.append(f"{rel}: missing '{key}'")
            return None
        if typ is not None and not isinstance(e[key], typ):
            errs.append(f"{rel}: '{key}' must be {typ.__name__}")
            return None
        return e[key]

    if need("schema", str) != SCHEMA:
        errs.append(f"{rel}: schema must be {SCHEMA!r}")
    did = need("id", str)
    if did and did != path.stem:
        errs.append(f"{rel}: id {did!r} does not match the filename")
    if did in seen:
        errs.append(f"{rel}: duplicate id {did!r}")
    seen.add(did)
    if did and not re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+-[0-9a-f]{8}", did):
        errs.append(f"{rel}: id must look like YYYY-MM-DD-<slug>-<8 hex>")
    for k in ("title", "prompt", "who", "produced_at"):
        v = need(k, str)
        if v is not None and not v.strip():
            errs.append(f"{rel}: '{k}' is empty")
    if need("tier", str) not in TIERS:
        errs.append(f"{rel}: tier must be one of {TIERS}")
    if need("kind", str) not in KINDS:
        errs.append(f"{rel}: kind must be one of {KINDS}")
    need("embodiment", str)
    src = need("source", dict) or {}
    for k in ("commit", "branch", "how"):
        if not isinstance(src.get(k), str) or not src.get(k):
            errs.append(f"{rel}: source.{k} is required")
    if "dirty" not in src:
        errs.append(f"{rel}: source.dirty is required")
    media = need("media", list) or []
    if not media and not e.get("payload"):
        errs.append(f"{rel}: a demo needs media or a payload")
    total = 0
    for m in media:
        p = ROOT / m.get("path", "")
        if not p.is_file():
            errs.append(f"{rel}: media missing on disk: {m.get('path')}")
            continue
        if p.suffix.lower() not in MEDIA_EXT:
            errs.append(f"{rel}: media type not allowed: {p.suffix}")
        size = p.stat().st_size
        total += size
        if size > MAX_FILE_BYTES:
            errs.append(f"{rel}: {m['path']} is {size} bytes, over the {MAX_FILE_BYTES} limit; put it on a release and link it")
        if m.get("bytes") != size:
            errs.append(f"{rel}: {m['path']} size changed ({m.get('bytes')} recorded, {size} on disk)")
        if m.get("sha256") != sha256_file(p):
            errs.append(f"{rel}: {m['path']} sha256 does not match the file")
    if total > MAX_ENTRY_BYTES:
        errs.append(f"{rel}: media totals {total} bytes, over {MAX_ENTRY_BYTES}")
    payload = e.get("payload")
    if payload is not None:
        if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
            errs.append(f"{rel}: payload must be an object with a path")
        elif not (ROOT / payload["path"]).is_file():
            errs.append(f"{rel}: payload missing on disk: {payload['path']}")
        elif e.get("kind") in ("gif", "video", "image"):
            errs.append(f"{rel}: a payload needs a pose kind, not {e.get('kind')!r}")
    return errs


def cmd_validate(a: argparse.Namespace) -> int:
    entries = load_entries()
    seen: set[str] = set()
    errs: list[str] = []
    for p, e in entries:
        errs += validate_entry(p, e, seen)
    # Nothing under demos/media may be orphaned: every file belongs to an entry.
    referenced = {m["path"] for _, e in entries for m in e.get("media", [])}
    referenced |= {e["payload"]["path"] for _, e in entries if isinstance(e.get("payload"), dict) and "path" in e["payload"]}
    if MEDIA.is_dir():
        for p in MEDIA.rglob("*"):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in referenced:
                errs.append(f"orphan media not referenced by any entry: {p.relative_to(ROOT).as_posix()}")
    for err in errs:
        print(err, file=sys.stderr)
    print(f"{len(entries)} entries, {len(errs)} problems", file=sys.stderr)
    return 1 if errs else 0


# --------------------------------------------------------------------------
# build

def cmd_build(a: argparse.Namespace) -> int:
    page = _core_demos().render_index(ROOT)
    if a.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
        if current != page:
            print("demos/index.html is out of date: run `uv run python demos/tools/demo_tools.py build` and commit it", file=sys.stderr)
            return 1
        print("demos/index.html is current", file=sys.stderr)
        return 0
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(PAGE.relative_to(ROOT).as_posix())
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="register a demo and copy its media in")
    p.add_argument("files", nargs="+", help="media files (gif, webm, mp4, png)")
    p.add_argument("--title", required=True)
    p.add_argument("--prompt", required=True, help="the plain-language prompt the demo answers")
    p.add_argument("--tier", required=True, choices=TIERS)
    p.add_argument("--embodiment", required=True, help="robot or rig id, e.g. mesh2motion-human-vrm1, zoo_long_arm, gripper")
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--how", required=True, help="the command that produced it")
    p.add_argument("--author", help="override the git identity")
    p.add_argument("--produced-at", help="ISO timestamp; default now")
    p.add_argument("--outcome-state", choices=("ok", "partial", "failed"))
    p.add_argument("--outcome-text")
    p.add_argument("--notes", default="")
    p.add_argument("--role", default="clip", help="media role: clip, still, comparison")
    p.add_argument("--tag", action="append")
    p.add_argument("--payload", help="repo-relative path of a pose payload (clip.json, viewer export, gripper_clip)")
    p.add_argument("--keep-paths", action="store_true", help="reference the files where they are instead of copying")
    p.set_defaults(func=cmd_add)
    v = sub.add_parser("validate", help="check every registry entry and its media")
    v.set_defaults(func=cmd_validate)
    b = sub.add_parser("build", help="render demos/index.html")
    b.add_argument("--check", action="store_true", help="fail if the committed page differs")
    b.set_defaults(func=cmd_build)
    a = ap.parse_args()
    if a.cmd == "add":
        a.outcome = {"state": a.outcome_state, "text": a.outcome_text} if a.outcome_state or a.outcome_text else None
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
