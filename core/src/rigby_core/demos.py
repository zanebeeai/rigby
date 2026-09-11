"""Register a demo in the shared registry (`demos/registry`), from anywhere.

The registry is one JSON file per demo whose id carries the date, the prompt
slug and eight hex of the media digest, so demos registered on different
branches never collide. This module is the one implementation of "write an
entry": the command line in `demos/tools/demo_tools.py` calls it, and so do
the humanoid and any-robot apps when a person presses "save as demo" on a
result, which is how a demo gets its author, commit, branch and command the
moment it exists rather than never.

Standard library only: the CI job that validates the registry runs it on a
bare interpreter, and neither product should need MuJoCo to write a record.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "rigby.demo/1"
TIERS = ("core", "humanoid", "any-robot")
KINDS = ("gif", "video", "image", "humanoid-bones-v1", "anyrobot-qpos-v1", "gripper-links-v1")
MEDIA_EXT = {".gif", ".webm", ".mp4", ".png", ".jpg", ".jpeg"}
POSE_KINDS = ("humanoid-bones-v1", "anyrobot-qpos-v1", "gripper-links-v1")

# Git identities seen on this repository, mapped to the GitHub handle so `who`
# is stable across machines.
HANDLES = {
    "zanebeeai": "zanebeeai",
    "zane beeai": "zanebeeai",
    "tony pan": "tpypan",
    "tpypan": "tpypan",
    "angelo wei": "AngeloWhey",
    "angelowhey": "AngeloWhey",
}


def find_repo_root(start: Path | None = None) -> Path:
    """The repository root: the nearest ancestor holding `demos/` and `.git`."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "demos").is_dir() and (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("no repository root with a demos/ directory above " + str(here))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "demo"


def _git(root: Path, *args: str, default: str = "") -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return default


def git_identity(root: Path) -> str:
    """The GitHub handle for the machine's git identity, or the name itself."""
    name = _git(root, "config", "user.name")
    email = _git(root, "config", "user.email")
    key = name.strip().lower()
    if key in HANDLES:
        return HANDLES[key]
    local = email.split("@")[0].lower()
    return HANDLES.get(local, name.strip() or local or "unknown")


@dataclass
class Provenance:
    commit: str
    branch: str
    dirty: bool

    @classmethod
    def read(cls, root: Path) -> "Provenance":
        return cls(
            commit=_git(root, "rev-parse", "--short=10", "HEAD", default="unknown"),
            branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD", default="unknown"),
            dirty=bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        )


@dataclass
class DemoSpec:
    title: str
    prompt: str
    tier: str
    embodiment: str
    kind: str
    how: str
    files: list[Path] = field(default_factory=list)
    payload: Path | None = None
    who: str | None = None
    produced_at: str | None = None
    outcome: dict | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    role: str = "clip"
    keep_paths: bool = False


def register(spec: DemoSpec, root: Path | None = None) -> Path:
    """Write the registry entry for `spec`, copying media in, and return its path.

    Raises ValueError for a spec the validator would reject, so an app can
    turn it into a 4xx instead of writing a broken entry.
    """
    root = root or find_repo_root()
    if spec.tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    if spec.kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not spec.files and spec.payload is None:
        raise ValueError("a demo needs media or a payload")
    if spec.payload is not None and spec.kind not in POSE_KINDS:
        raise ValueError(f"a payload needs a pose kind, not {spec.kind!r}")
    for f in spec.files:
        if not f.is_file():
            raise ValueError(f"not a file: {f}")
        if f.suffix.lower() not in MEDIA_EXT:
            raise ValueError(f"media type not allowed: {f.suffix}")
    for key in ("title", "prompt", "embodiment", "how"):
        if not str(getattr(spec, key)).strip():
            raise ValueError(f"{key} is required")

    produced = spec.produced_at or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    digest = hashlib.sha256()
    for f in [*spec.files, *([spec.payload] if spec.payload else [])]:
        digest.update(bytes.fromhex(sha256_file(f)))
    did = f"{produced[:10]}-{slug(spec.prompt)}-{digest.hexdigest()[:8]}"

    registry = root / "demos" / "registry"
    media_dir = root / "demos" / "media" / did
    registry.mkdir(parents=True, exist_ok=True)

    def place(src: Path) -> str:
        if spec.keep_paths:
            return src.resolve().relative_to(root).as_posix()
        media_dir.mkdir(parents=True, exist_ok=True)
        target = media_dir / src.name
        if target.resolve() != src.resolve():
            shutil.copyfile(src, target)
        return target.relative_to(root).as_posix()

    media = []
    for f in spec.files:
        rel = place(f)
        media.append({"path": rel, "bytes": (root / rel).stat().st_size, "sha256": sha256_file(root / rel), "role": spec.role})
    payload = {"path": place(spec.payload)} if spec.payload else None

    prov = Provenance.read(root)
    entry = {
        "schema": SCHEMA,
        "id": did,
        "title": spec.title,
        "prompt": spec.prompt,
        "tier": spec.tier,
        "embodiment": spec.embodiment,
        "kind": spec.kind,
        "who": spec.who or git_identity(root),
        "produced_at": produced,
        "source": {"commit": prov.commit, "branch": prov.branch, "dirty": prov.dirty, "how": spec.how},
        "outcome": spec.outcome,
        "notes": spec.notes,
        "media": media,
        "payload": payload,
        "tags": list(spec.tags),
    }
    out = registry / f"{did}.json"
    out.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


# --------------------------------------------------------------------------
# the viewer: one static page rendered from the registry


def load_entries(root: Path) -> list[tuple[Path, dict]]:
    registry = root / "demos" / "registry"
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(registry.glob("*.json"))]


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
    """The static viewer, from the registry entries."""
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




def render_index(root: Path) -> str:
    return render_page([e for _, e in load_entries(root)])


def write_index(root: Path) -> Path:
    """Regenerate demos/index.html; the apps call this after registering."""
    out = root / "demos" / "index.html"
    out.write_text(render_index(root), encoding="utf-8", newline="\n")
    return out
