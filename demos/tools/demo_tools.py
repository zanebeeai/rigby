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
import datetime as dt
import hashlib
import html
import json
import re
import shutil
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

# Git identities seen on this repository, so `who` is a stable handle rather
# than whatever a machine's git config happens to say.
HANDLES = {
    "zanebeeai": "zanebeeai",
    "zane beeai": "zanebeeai",
    "tony pan": "tpypan",
    "tpypan": "tpypan",
    "angelo wei": "AngeloWhey",
    "angelowhey": "AngeloWhey",
}


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


def handle_for(name: str, email: str) -> str:
    key = name.strip().lower()
    if key in HANDLES:
        return HANDLES[key]
    local = email.split("@")[0].lower()
    return HANDLES.get(local, name.strip() or local)


# --------------------------------------------------------------------------
# add


def cmd_add(a: argparse.Namespace) -> int:
    files = [Path(f).resolve() for f in a.files]
    for f in files:
        if not f.is_file():
            print(f"not a file: {f}", file=sys.stderr)
            return 2
    name = a.author or git("config", "user.name")
    email = git("config", "user.email")
    who = handle_for(name, email)
    commit = git("rev-parse", "--short=10", "HEAD", default="unknown")
    branch = git("rev-parse", "--abbrev-ref", "HEAD", default="unknown")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    produced = a.produced_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    digest = hashlib.sha256()
    for f in files:
        digest.update(bytes.fromhex(sha256_file(f)))
    did = f"{produced[:10]}-{slug(a.prompt or a.title)}-{digest.hexdigest()[:8]}"
    media = []
    if a.keep_paths:
        for f in files:
            try:
                rel = f.relative_to(ROOT).as_posix()
            except ValueError:
                print(f"--keep-paths needs a file inside the repository: {f}", file=sys.stderr)
                return 2
            media.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256_file(f), "role": a.role})
    else:
        dest = MEDIA / did
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dest / f.name
            if target.resolve() != f:
                shutil.copyfile(f, target)
            media.append({"path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size, "sha256": sha256_file(target), "role": a.role})
    entry = {
        "schema": SCHEMA,
        "id": did,
        "title": a.title,
        "prompt": a.prompt,
        "tier": a.tier,
        "embodiment": a.embodiment,
        "kind": a.kind,
        "who": who,
        "produced_at": produced,
        "source": {"commit": commit, "branch": branch, "dirty": dirty, "how": a.how},
        "outcome": a.outcome,
        "notes": a.notes,
        "media": media,
        "payload": a.payload,
        "tags": a.tag or [],
    }
    REGISTRY.mkdir(parents=True, exist_ok=True)
    out = REGISTRY / f"{did}.json"
    out.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT).as_posix())
    return 0


# --------------------------------------------------------------------------
# validate


def load_entries() -> list[tuple[Path, dict]]:
    entries = []
    for p in sorted(REGISTRY.glob("*.json")):
        entries.append((p, json.loads(p.read_text(encoding="utf-8"))))
    return entries


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

CSS = r"""
:root { --ground:#eef0f2; --surface:#fff; --ink:#1b2128; --muted:#5a6470; --line:#d5dae0; --accent:#1f6f8b; --accent-ink:#fff;
  --humanoid:#1f6f8b; --anyrobot:#7a5c2e; --core:#4d5a68; --ok:#3e7d4f; --warn:#b8862b; --bad:#b2402e; --code:#f4f6f8; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --ground:#121619; --surface:#1a2026; --ink:#e5e9ed; --muted:#96a1ac; --line:#2a333c;
  --accent:#5faec9; --accent-ink:#0d1418; --humanoid:#5faec9; --anyrobot:#c9a061; --core:#9aa8b6; --ok:#6db47e; --warn:#d6a84a; --bad:#d9776a; --code:#10151a; } }
:root[data-theme="dark"] { --ground:#121619; --surface:#1a2026; --ink:#e5e9ed; --muted:#96a1ac; --line:#2a333c;
  --accent:#5faec9; --accent-ink:#0d1418; --humanoid:#5faec9; --anyrobot:#c9a061; --core:#9aa8b6; --ok:#6db47e; --warn:#d6a84a; --bad:#d9776a; --code:#10151a; }
* { box-sizing:border-box; } body { margin:0; background:var(--ground); color:var(--ink); font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif; font-size:15px; line-height:1.45; }
h1,h2,h3 { font-family:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif; margin:0; letter-spacing:-0.01em; text-wrap:balance; }
h1 { font-size:1.9rem; font-weight:800; } h2 { font-size:1.05rem; font-weight:700; }
code,.mono { font-family:"JetBrains Mono",ui-monospace,Consolas,monospace; font-size:0.84em; }
a { color:var(--accent); }
.app { display:grid; grid-template-columns:260px 1fr; min-height:100vh; }
.side { border-right:1px solid var(--line); padding:20px 18px; display:grid; gap:18px; align-content:start; position:sticky; top:0; height:100vh; overflow:auto; background:var(--surface); }
.side .eyebrow { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.side p { margin:0; color:var(--muted); font-size:0.9rem; }
fieldset { border:0; padding:0; margin:0; display:grid; gap:4px; } legend { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); font-weight:600; margin-bottom:6px; }
label.opt { display:flex; gap:8px; align-items:center; font-size:0.92rem; cursor:pointer; } label.opt span.n { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; font-size:0.8rem; }
input[type=search] { width:100%; padding:7px 10px; border:1px solid var(--line); background:var(--ground); color:var(--ink); font:inherit; }
.main { padding:20px 24px 60px; display:grid; gap:18px; align-content:start; }
.bar { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; }
.bar .count { color:var(--muted); font-variant-numeric:tabular-nums; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
.card { background:var(--surface); border:1px solid var(--line); display:grid; grid-template-rows:auto 1fr; overflow:hidden; }
.card .media { background:#000; aspect-ratio:16/10; display:grid; place-items:center; overflow:hidden; }
.card .media img,.card .media video { width:100%; height:100%; object-fit:contain; display:block; }
.card .media .placeholder { color:#9aa; font-size:0.85rem; padding:12px; text-align:center; }
.card .body { padding:12px 14px 14px; display:grid; gap:6px; align-content:start; }
.card .title { font-weight:700; font-size:1rem; }
.card .prompt { color:var(--muted); font-style:italic; font-size:0.9rem; overflow-wrap:anywhere; }
.card .meta { display:flex; gap:6px; flex-wrap:wrap; font-size:0.74rem; }
.pill { display:inline-block; padding:1px 8px; border-radius:999px; border:1px solid currentColor; font-weight:600; letter-spacing:0.03em; white-space:nowrap; }
.pill.humanoid{color:var(--humanoid)} .pill.any-robot{color:var(--anyrobot)} .pill.core{color:var(--core)} .pill.who{color:var(--ink)} .pill.kind{color:var(--muted)}
.card .prov { font-size:0.78rem; color:var(--muted); display:grid; grid-template-columns:auto 1fr; gap:1px 10px; margin-top:4px; }
.card .prov dt { font-weight:600; } .card .prov dd { margin:0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
.card .how { background:var(--code); border:1px solid var(--line); padding:6px 8px; font-size:0.74rem; overflow-x:auto; white-space:pre; margin:0; }
.card .outcome { font-size:0.85rem; } .card .outcome::before { content:""; display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--muted); margin-right:6px; }
.card .outcome.ok::before{background:var(--ok)} .card .outcome.partial::before{background:var(--warn)} .card .outcome.failed::before{background:var(--bad)}
.empty { color:var(--muted); padding:40px; text-align:center; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media (max-width: 760px) { .app { grid-template-columns:1fr; } .side { position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); } }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('registry').textContent);
const state = { q: '', who: new Set(), tier: new Set(), kind: new Set(), emb: new Set() };
const $ = (s, el=document) => el.querySelector(s);
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function facet(name, key, items){
  const counts = {}; for (const d of DATA) { const v = d[key] || '—'; counts[v] = (counts[v]||0)+1; }
  const set = state[name];
  const box = $('#f-'+name); box.innerHTML = '';
  for (const v of Object.keys(counts).sort()) {
    const id = 'f-'+name+'-'+v.replace(/[^a-z0-9]/gi,'_');
    const l = document.createElement('label'); l.className = 'opt';
    l.innerHTML = `<input type="checkbox" id="${id}" ${set.has(v)?'checked':''}> ${esc(v)} <span class="n">${counts[v]}</span>`;
    l.querySelector('input').addEventListener('change', e => { e.target.checked ? set.add(v) : set.delete(v); render(); });
    box.appendChild(l);
  }
}
function matches(d){
  if (state.who.size && !state.who.has(d.who)) return false;
  if (state.tier.size && !state.tier.has(d.tier)) return false;
  if (state.kind.size && !state.kind.has(d.kind)) return false;
  if (state.emb.size && !state.emb.has(d.embodiment)) return false;
  if (state.q) { const hay = [d.title,d.prompt,d.embodiment,d.who,d.notes,(d.tags||[]).join(' '),d.source.commit,d.source.branch].join(' ').toLowerCase(); if (!hay.includes(state.q)) return false; }
  return true;
}
function mediaEl(d){
  const m = (d.media||[]).find(x => x.role !== 'still') || (d.media||[])[0];
  if (!m) return `<div class="placeholder">${esc(d.kind)} payload, no rendered media yet<br><code>${esc(d.payload?.path||'')}</code></div>`;
  const src = '../' + m.path;
  if (/\.(webm|mp4)$/i.test(m.path)) return `<video controls preload="metadata" src="${esc(src)}"></video>`;
  return `<img loading="lazy" src="${esc(src)}" alt="${esc(d.title)}">`;
}
function card(d){
  const oc = d.outcome ? `<div class="outcome ${esc(d.outcome.state||'')}">${esc(d.outcome.text||'')}</div>` : '';
  return `<article class="card" id="${esc(d.id)}">
    <div class="media">${mediaEl(d)}</div>
    <div class="body">
      <div class="title">${esc(d.title)}</div>
      <div class="prompt">“${esc(d.prompt)}”</div>
      <div class="meta"><span class="pill ${esc(d.tier)}">${esc(d.tier)}</span><span class="pill who">${esc(d.who)}</span><span class="pill kind">${esc(d.kind)}</span><span class="pill kind">${esc(d.embodiment)}</span></div>
      ${oc}
      <dl class="prov"><dt>made</dt><dd>${esc(d.produced_at.slice(0,10))}</dd><dt>commit</dt><dd class="mono">${esc(d.source.commit)}${d.source.dirty?' (dirty tree)':''} on ${esc(d.source.branch)}</dd><dt>files</dt><dd>${(d.media||[]).map(m=>`<a href="../${esc(m.path)}">${esc(m.path.split('/').pop())}</a>`).join(', ')}</dd></dl>
      <pre class="how">${esc(d.source.how)}</pre>
      ${d.notes ? `<div class="prompt" style="font-style:normal">${esc(d.notes)}</div>` : ''}
    </div></article>`;
}
function render(){
  const rows = DATA.filter(matches).sort((a,b) => b.produced_at.localeCompare(a.produced_at));
  $('#count').textContent = `${rows.length} of ${DATA.length}`;
  $('#grid').innerHTML = rows.length ? rows.map(card).join('') : '<div class="empty">No demo matches those filters.</div>';
}
facet('who','who'); facet('tier','tier'); facet('kind','kind'); facet('emb','embodiment');
$('#q').addEventListener('input', e => { state.q = e.target.value.trim().toLowerCase(); render(); });
render();
if (location.hash) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
"""


def render_page(entries: list[dict]) -> str:
    data = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    who = sorted({e["who"] for e in entries})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rigby Demos</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style></head>
<body><div class="app">
<aside class="side">
  <div><div class="eyebrow">zanebeeai/rigby</div><h1>Rigby Demos</h1><p>{len(entries)} demos by {html.escape(", ".join(who))}. One file per demo under <code>demos/registry</code>; add one with <code>demo_tools.py add</code>.</p></div>
  <input type="search" id="q" placeholder="search prompt, title, embodiment, commit" aria-label="search">
  <fieldset><legend>Who</legend><div id="f-who"></div></fieldset>
  <fieldset><legend>Tier</legend><div id="f-tier"></div></fieldset>
  <fieldset><legend>Kind</legend><div id="f-kind"></div></fieldset>
  <fieldset><legend>Embodiment</legend><div id="f-emb"></div></fieldset>
  <p><a href="../docs/results/index.html">The results ledger</a> covers everything produced before this registry existed.</p>
</aside>
<main class="main">
  <div class="bar"><h2>Every demo, newest first</h2><span class="count" id="count"></span></div>
  <div class="grid" id="grid"></div>
</main></div>
<script id="registry" type="application/json">{data}</script>
<script>{JS}</script>
</body></html>
"""


def cmd_build(a: argparse.Namespace) -> int:
    entries = [e for _, e in load_entries()]
    page = render_page(entries)
    if a.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
        if current != page:
            print("demos/index.html is out of date: run `uv run python demos/tools/demo_tools.py build` and commit it", file=sys.stderr)
            return 1
        print("demos/index.html is current", file=sys.stderr)
        return 0
    PAGE.write_text(page, encoding="utf-8")
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
        a.payload = {"path": a.payload} if a.payload else None
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
