"""Render the results summary page from manifest.json.

Writes `docs/results/index.html`, a single static page that reads the manifest
and lays out every result set in the order the work happened, with the media
inline: the any-robot zoo GIFs, the gripper recordings, the humanoid README
demos. Open it from the repository (the media paths are relative), or publish
it with the media beside it.

    uv run python docs/results/build_summary.py
    uv run python docs/results/build_summary.py --out /somewhere/index.html --media-prefix media/

`--media-prefix` rewrites where the page looks for media, for a copy published
outside the repository where `humanoid/docs/media` is not two levels up.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TIER = {
    "humanoid-runs-aug07-12": ("humanoid", "Zane"),
    "humanoid-release-evidence": ("humanoid", "Zane"),
    "mjco-sim-workspace": ("humanoid", "Zane"),
    "humanoid-demo-gifs": ("humanoid", "Zane, Tony"),
    "humanoid-calibration-evidence": ("humanoid", "Tony"),
    "any-robot-zoo-aug24-27": ("any-robot", "Zane"),
    "any-robot-zoo-demos": ("any-robot", "Zane"),
    "any-robot-environments-aug27-28": ("any-robot", "Zane"),
    "gripper-milestones": ("humanoid", "Angelo"),
}

SHORT = {
    "humanoid-runs-aug07-12": "Humanoid archive",
    "humanoid-release-evidence": "v2 release evidence",
    "mjco-sim-workspace": "MuJoCo workspace",
    "humanoid-demo-gifs": "README demos",
    "humanoid-calibration-evidence": "Grader calibration",
    "any-robot-zoo-aug24-27": "Zoo trials, first pass",
    "any-robot-zoo-demos": "Zoo demos",
    "any-robot-environments-aug27-28": "Environment trials",
    "gripper-milestones": "Gripper milestones",
}

# One-line outcome per set, the number that a reader should leave with.
OUTCOME = {
    "humanoid-runs-aug07-12": ("985 structurally valid of 1,149 judged", "ok"),
    "humanoid-release-evidence": ("12 of 12 release requirements pass", "ok"),
    "mjco-sim-workspace": ("parity not measured; superseded by the v2 runtime", "warn"),
    "humanoid-demo-gifs": ("5 demos and 1 before/after, all offline planner", "ok"),
    "humanoid-calibration-evidence": ("grader MCC +0.054: not a valid instrument", "bad"),
    "any-robot-zoo-aug24-27": ("31 of 151 traces accepted", "warn"),
    "any-robot-zoo-demos": ("13 clips on 7 zoo arms; 1 of 5 grasps certified", "warn"),
    "any-robot-environments-aug27-28": ("78 of 241 traces accepted", "warn"),
    "gripper-milestones": ("12 of 13 runs placed, worst penetration 0.34 mm", "ok"),
}

CONTRIBUTIONS = [
    {
        "who": "Tony Pan",
        "handle": "tpypan",
        "span": "7 Jul to 7 Sep",
        "commits": "205 on main",
        "lines": "+139k / -14.7k in humanoid, +514 in core",
        "what": (
            "The humanoid evaluation stack: the observability tracer, the analysis layer "
            "extracted from the compiler, the 47-case golden corpus and its bless tooling, "
            "the mutation families, the calibration statistics and the 10f campaign that "
            "measured the VLM grader against them, legs in strikes, the rig-derived arm "
            "calibration, the test-suite audit, the CI workflow, and in September the "
            "grasp solver that places the hand by orientation and carries by FK."
        ),
    },
    {
        "who": "Zane Beeai",
        "handle": "zanebeeai",
        "span": "9 Aug to 10 Sep",
        "commits": "55 on main",
        "lines": "+92.7k in humanoid (the v2 runtime), +31k in any-robot",
        "what": (
            "The original POC and the v2 certified motion runtime, then the any-robot tier: "
            "ingest and measured morphology, the grounder, the bake and its gates, the "
            "certified primitive library, contact, the development zoo and the bench robots, "
            "authored environments and the trials across them, Motion Studio for arbitrary "
            "robots, and the September measured-grasp work."
        ),
    },
    {
        "who": "Angelo Wei",
        "handle": "AngeloWhey",
        "span": "25 Aug to 10 Sep",
        "commits": "99 on grasp/auto-lift-on-main",
        "lines": "+33.3k / -258 in humanoid, 298 files",
        "what": (
            "Embodiment truthfulness: the render-contact check, the ROM-grounded body-part "
            "vocabulary, joint rate limits and thumb opposition; then the closed-loop decision "
            "layer driven by a VLM over super primitives, and the three-segment gripper under "
            "computed torque with one wrist camera as its only percept, through pick-and-place "
            "to the unfinished cabinet."
        ),
    },
]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.0f} kB"


def num(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def period_dates(p: str) -> tuple[dt.date, dt.date]:
    parts = p.replace(" to ", " ").split()
    a = dt.date.fromisoformat(parts[0])
    b = dt.date.fromisoformat(parts[-1]) if len(parts) > 1 else a
    return a, b


def facts_rows(e: dict) -> list[tuple[str, str]]:
    """Hand-picked, readable facts per set (the manifest has everything else)."""
    f = e.get("facts") or {}
    s = e.get("stats") or {}
    rows: list[tuple[str, str]] = []
    sid = e["id"]
    if sid == "humanoid-runs-aug07-12":
        pm = f.get("planner_models", {})
        rows += [
            ("Numbered runs", f"{num(f['numbered_runs'])}, ids {f['first_id'][:6]} to {f['last_id'][:6]}"),
            ("Planner", ", ".join(f"{k} ({num(v)})" for k, v in pm.items())),
            ("Compiler versions", ", ".join(f"{k} ({num(v)})" for k, v in f.get("compiler_versions", {}).items())),
            ("Verdicts", f"{num(f['accepted_or_structural_valid'])} valid, {num(f['rejected_or_structural_invalid'])} invalid, {num(f['runs_without_verdict'])} compile-only with no verdict"),
            ("With exported GLB", num(f["runs_with_glb"])),
            ("Acceptance runs", f"{len(f.get('acceptance_runs', []))} blinded gesture-review runs"),
            ("Autonomous goal audit", ", ".join(f"{k}: {v}" for k, v in (f.get("autonomous_goal_audit") or {}).get("gates", {}).items())),
            ("Named sets", ", ".join(f.get("named_sets", [])[:14]) + (" ..." if len(f.get("named_sets", [])) > 14 else "")),
        ]
    elif sid == "humanoid-release-evidence":
        rows += [
            ("Release requirements", ", ".join(f"{k} ({v})" for k, v in (f.get("release_evidence") or {}).items())),
            ("Top level", ", ".join(f.get("top_level", []))),
        ]
    elif sid == "mjco-sim-workspace":
        rows += [
            ("Experiment folders", f"{len(f.get('artifacts_v2', []))} under artifacts-v2, 12 to 18 Aug"),
            ("Docs", ", ".join(f.get("docs", []))),
            ("Python files in no commit", f"{num(f['python_files_in_no_commit'])} of {num(f['python_files_checked'])}"),
        ]
    elif sid == "humanoid-demo-gifs":
        rows += [
            ("Compiler commit", f"{f.get('compiler_commit')} (manifest generated {f.get('generated_on')})"),
            ("Demos", "; ".join(f"{d['file']}: “{d['prompt'][:60]}{'…' if len(d['prompt']) > 60 else ''}” ({d['planner_model']}, {d['duration_s']} s)" for d in f.get("demos", []))),
            ("Comparison", "; ".join(f"{c['file']}: “{c['prompt']}”" for c in f.get("comparisons", []))),
        ]
    elif sid == "humanoid-calibration-evidence":
        rows += [
            ("Golden corpus", "47 cases under humanoid/evals/corpus/cases, blessed per platform"),
            ("10f campaign", "26 Aug, gpt-5.6-luna with gpt-5.6-terra fallback, 1,334 clips scored, $9.41"),
            ("Accept rate, unmutated", "34/41 = 0.829 (majority class 1.000)"),
            ("Accept rate, mutated", "881/1288 = 0.684 (majority class 0.000)"),
            ("Matthews correlation", "+0.054; balanced accuracy 0.573 against 0.500 chance"),
            ("Verdict", "the grader is not a valid instrument on this corpus"),
        ]
    elif sid in ("any-robot-zoo-aug24-27", "any-robot-environments-aug27-28"):
        rows += [
            ("Traces", f"{num(f['traces'])}: " + ", ".join(f"{k} {v}" for k, v in f.get("kinds", {}).items())),
            ("Accepted", f"{num(f['accepted'])} accepted, {num(f['rejected'])} refused"),
            ("Refusals", ", ".join(f"{k} ({v})" for k, v in list(f.get("failure_codes", {}).items())[:8])),
            ("Robots", ", ".join(f"{k} ({v})" for k, v in f.get("robots", {}).items())),
            ("Code base tree", ", ".join(f.get("code_base_trees", {}).keys())),
        ]
    elif sid == "any-robot-zoo-demos":
        rows += [
            ("Clips", f"{len(f.get('clips', []))} across " + ", ".join(sorted({c['robot_id'] for c in f.get('clips', [])}))),
            ("Grasp report", f"{f.get('grasp_attempts')} attempts, {f.get('grasp_certified')} certified: " + "; ".join(f"{k}: {v}" for k, v in (f.get("grasp_by_robot") or {}).items())),
        ]
    elif sid == "gripper-milestones":
        rows += [
            ("Branch tip", f["tip"][:7] + " " + f["tip"].split(" ", 1)[1]),
            ("Recorded runs", "13 complete gripper_clip_v1 recordings under milestones/runs"),
            ("Sensing", "joint encoders, one wrist camera, contact inferred from tracking error; no force sensor, no object pose, no object size"),
        ]
    if s:
        rows.append(("On disk", f"{num(s['files'])} files, {fmt_bytes(s['bytes'])}, modified {s['oldest']} to {s['newest']}"))
    return rows


def media_items(e: dict, media_prefix: str) -> list[tuple[str, str, str]]:
    """(kind, src, caption) for each media file worth showing."""
    items = []
    for m in e.get("media", []):
        f = m["file"]
        name = Path(f).name
        if name.endswith(".json") or name.endswith(".txt") or name.endswith(".md"):
            continue
        if f.startswith("humanoid/docs/media/"):
            src = media_prefix + "humanoid-demo-gifs/" + name if media_prefix else "../../" + f
        else:
            src = (media_prefix + f[len("media/"):]) if media_prefix else f
        kind = "video" if name.endswith(".webm") else "image"
        caption = name.replace("videos--", "").replace("milestones--", "")
        items.append((kind, src, caption))
    return items


def timeline_svg(sets: list[dict]) -> str:
    start = dt.date(2026, 8, 7)
    end = dt.date(2026, 9, 10)
    span = (end - start).days
    w, left, right, row_h, top = 1000, 190, 20, 26, 28
    h = top + row_h * len(sets) + 30
    x = lambda d: left + (w - left - right) * (d - start).days / span
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="When each result set was produced" class="timeline">']
    # week ticks
    d = start
    while d <= end:
        xx = x(d)
        out.append(f'<line x1="{xx:.1f}" y1="{top - 6}" x2="{xx:.1f}" y2="{h - 26}" class="tick"/>')
        out.append(f'<text x="{xx:.1f}" y="{h - 10}" class="tick-label" text-anchor="middle">{d.day} {d.strftime("%b")}</text>')
        d += dt.timedelta(days=7)
    for i, e in enumerate(sets):
        a, b = period_dates(e["period"])
        y = top + i * row_h
        tier, who = TIER[e["id"]]
        x0, x1 = x(a), max(x(b), x(a) + 6)
        out.append(f'<text x="{left - 10}" y="{y + 14}" class="row-label" text-anchor="end">{esc(SHORT[e["id"]])}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y + 4}" width="{x1 - x0:.1f}" height="14" rx="2" class="bar bar-{tier}"><title>{esc(e["period"])}, {esc(who)}</title></rect>')
    out.append("</svg>")
    return "\n".join(out)


CSS = r"""
:root {
  --ground: #eef0f2; --surface: #ffffff; --ink: #1b2128; --muted: #5a6470; --line: #d5dae0;
  --accent: #1f6f8b; --accent-ink: #ffffff; --ok: #3e7d4f; --warn: #b8862b; --bad: #b2402e;
  --humanoid: #1f6f8b; --anyrobot: #7a5c2e; --code: #f4f6f8;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #121619; --surface: #1a2026; --ink: #e5e9ed; --muted: #96a1ac; --line: #2a333c;
    --accent: #5faec9; --accent-ink: #0d1418; --ok: #6db47e; --warn: #d6a84a; --bad: #d9776a;
    --humanoid: #5faec9; --anyrobot: #c9a061; --code: #10151a;
  }
}
:root[data-theme="dark"] {
  --ground: #121619; --surface: #1a2026; --ink: #e5e9ed; --muted: #96a1ac; --line: #2a333c;
  --accent: #5faec9; --accent-ink: #0d1418; --ok: #6db47e; --warn: #d6a84a; --bad: #d9776a;
  --humanoid: #5faec9; --anyrobot: #c9a061; --code: #10151a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif; font-size: 16px; line-height: 1.5; }
h1, h2, h3 { font-family: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif; text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: 2.4rem; font-weight: 800; line-height: 1.05; }
h2 { font-size: 1.5rem; font-weight: 700; }
h3 { font-size: 1.15rem; font-weight: 700; }
code, .mono, table.ledger td.num, .facts dd.mono { font-family: "JetBrains Mono", ui-monospace, Consolas, monospace; font-size: 0.86em; }
a { color: var(--accent); }
.page { max-width: 1120px; margin: 0 auto; padding: 32px 24px 80px; display: grid; gap: 40px; }
.masthead { display: grid; gap: 10px; border-bottom: 2px solid var(--ink); padding-bottom: 20px; }
.masthead p { max-width: 68ch; margin: 0; color: var(--muted); }
.stamp { font-family: "JetBrains Mono", monospace; font-size: 0.8rem; color: var(--muted); letter-spacing: 0.02em; }
.eyebrow { font-size: 0.74rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
section { display: grid; gap: 16px; }
.wide { overflow-x: auto; }
table.ledger { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
table.ledger th { text-align: left; font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); padding: 6px 10px; border-bottom: 1px solid var(--line); }
table.ledger td { padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.ledger td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.ledger tr:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.04em; border: 1px solid currentColor; line-height: 1.5; white-space: nowrap; }
.pill.humanoid { color: var(--humanoid); } .pill.any-robot { color: var(--anyrobot); }
.pill.tracked { color: var(--ok); } .pill.untracked { color: var(--warn); } .pill.branch { color: var(--muted); }
.state { display: inline-flex; align-items: center; gap: 6px; }
.state::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--muted); flex: none; }
.state.ok::before { background: var(--ok); } .state.warn::before { background: var(--warn); } .state.bad::before { background: var(--bad); }
.timeline { width: 100%; height: auto; display: block; }
.timeline .tick { stroke: var(--line); stroke-width: 1; }
.timeline .tick-label, .timeline .row-label { fill: var(--muted); font-family: "JetBrains Mono", monospace; font-size: 11px; }
.timeline .row-label { fill: var(--ink); font-family: "Source Sans 3", system-ui, sans-serif; font-size: 13px; }
.timeline .bar { fill: var(--accent); } .timeline .bar-any-robot { fill: var(--anyrobot); }
.legend { display: flex; gap: 18px; font-size: 0.85rem; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend .humanoid::before { background: var(--humanoid); } .legend .any-robot::before { background: var(--anyrobot); }
.people { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.person { background: var(--surface); border: 1px solid var(--line); padding: 18px 20px; display: grid; gap: 8px; align-content: start; }
.person .who { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.person .who span { color: var(--muted); font-size: 0.85rem; }
.person .nums { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 0.9rem; }
.person .nums dt { color: var(--muted); } .person .nums dd { margin: 0; font-variant-numeric: tabular-nums; }
.person p { margin: 4px 0 0; font-size: 0.95rem; }
.set { background: var(--surface); border: 1px solid var(--line); padding: 22px 24px; display: grid; gap: 16px; }
.set header { display: grid; gap: 8px; }
.set .meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.86rem; color: var(--muted); }
.set .meta .who { color: var(--ink); font-weight: 600; }
.facts { display: grid; grid-template-columns: minmax(150px, 190px) 1fr; gap: 6px 16px; margin: 0; font-size: 0.93rem; }
.facts dt { color: var(--muted); } .facts dd { margin: 0; overflow-wrap: anywhere; }
.prose { max-width: 72ch; margin: 0; font-size: 0.95rem; }
.prose + .prose { margin-top: -6px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.gallery figure { margin: 0; display: grid; gap: 6px; }
.gallery img, .gallery video { width: 100%; height: auto; display: block; background: #000; border: 1px solid var(--line); }
.gallery figcaption { font-family: "JetBrains Mono", monospace; font-size: 0.72rem; color: var(--muted); overflow-wrap: anywhere; }
.gallery figure.wide { grid-column: 1 / -1; }
pre { background: var(--code); border: 1px solid var(--line); padding: 12px 14px; overflow-x: auto; margin: 0; font-size: 0.82rem; line-height: 1.45; }
.resultstable { font-family: "JetBrains Mono", monospace; font-size: 0.78rem; white-space: pre; overflow-x: auto; background: var(--code); border: 1px solid var(--line); padding: 12px 14px; margin: 0; }
.standing { display: grid; gap: 10px; }
.standing li { max-width: 78ch; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 640px) { .facts { grid-template-columns: 1fr; } h1 { font-size: 1.9rem; } .page { padding: 20px 14px 60px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
"""


def render(manifest: dict, media_prefix: str) -> str:
    sets = [e for e in manifest["sets"] if e.get("present")]
    o = []
    o.append("<title>Rigby Results Ledger</title>")
    o.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
    o.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">')
    o.append(f"<style>{CSS}</style>")
    o.append('<main class="page">')
    # masthead
    total_files = sum((e.get("stats") or {}).get("files", 0) for e in sets)
    total_bytes = sum((e.get("stats") or {}).get("bytes", 0) for e in sets)
    o.append('<header class="masthead">')
    o.append('<div class="eyebrow">zanebeeai/rigby &middot; every result set, 7 Aug to 10 Sep 2026</div>')
    o.append("<h1>Rigby Results Ledger</h1>")
    o.append("<p>What the three of us produced, where each set lives, on which code it ran, and the recordings that show it. "
             "Six of the nine sets exist only as untracked residue on one machine; this page and the manifest behind it are the record.</p>")
    o.append(f'<div class="stamp">generated {esc(manifest["generated_on"])} from commit {esc(manifest["generated_from_commit"])} &middot; '
             f'{num(total_files)} files, {fmt_bytes(total_bytes)} on disk &middot; docs/results/manifest.json</div>')
    o.append("</header>")

    # ledger table
    o.append('<section><div class="eyebrow">Ledger</div><div class="wide"><table class="ledger">')
    o.append("<thead><tr><th>Period</th><th>Set</th><th>Tier</th><th>Who</th><th>Where</th><th>Files</th><th>Size</th><th>Outcome</th></tr></thead><tbody>")
    for e in sets:
        tier, who = TIER[e["id"]]
        st = e.get("stats") or {"files": e["facts"]["files"], "bytes": e["facts"]["bytes"]}
        outcome, state = OUTCOME[e["id"]]
        tracked = e["tracked"]
        tclass = "tracked" if tracked is True else "branch" if isinstance(tracked, str) else "untracked"
        tword = "tracked" if tracked is True else "branch only" if isinstance(tracked, str) else "untracked"
        o.append(f'<tr><td class="num">{esc(e["period"].replace("2026-", "").replace(" to ", " → "))}</td>'
                 f'<td><a href="#{esc(e["id"])}">{esc(SHORT[e["id"]])}</a></td>'
                 f'<td><span class="pill {tier}">{tier}</span></td><td>{esc(who)}</td>'
                 f'<td><code>{esc(e["location"])}</code> <span class="pill {tclass}">{tword}</span></td>'
                 f'<td class="num">{num(st["files"])}</td><td class="num">{fmt_bytes(st["bytes"])}</td>'
                 f'<td><span class="state {state}">{esc(outcome)}</span></td></tr>')
    o.append("</tbody></table></div></section>")

    # timeline
    o.append('<section><div class="eyebrow">When</div>')
    o.append(timeline_svg(sets))
    o.append('<div class="legend"><span class="humanoid">humanoid tier</span><span class="any-robot">any-robot tier</span></div></section>')

    # contributions
    o.append('<section><div class="eyebrow">Who did what</div><h2>Three lines of work, one repository</h2><div class="people">')
    for c in CONTRIBUTIONS:
        o.append('<article class="person">')
        o.append(f'<div class="who"><h3>{esc(c["who"])}</h3><span>{esc(c["handle"])} &middot; {esc(c["span"])}</span></div>')
        o.append(f'<dl class="nums"><dt>Commits</dt><dd>{esc(c["commits"])}</dd><dt>Lines</dt><dd>{esc(c["lines"])}</dd></dl>')
        o.append(f'<p>{esc(c["what"])}</p></article>')
    o.append("</div></section>")

    # sets
    o.append('<section><div class="eyebrow">The sets, in order</div>')
    for e in sets:
        tier, who = TIER[e["id"]]
        tracked = e["tracked"]
        tclass = "tracked" if tracked is True else "branch" if isinstance(tracked, str) else "untracked"
        tword = "tracked on main" if tracked is True else "on the branch only" if isinstance(tracked, str) else "untracked residue"
        outcome, state = OUTCOME[e["id"]]
        o.append(f'<article class="set" id="{esc(e["id"])}"><header>')
        o.append(f'<div class="meta"><span class="pill {tier}">{tier}</span><span class="pill {tclass}">{tword}</span>'
                 f'<span class="who">{esc(who)}</span><span>{esc(e["period"])}</span><span><code>{esc(e["location"])}</code></span></div>')
        o.append(f'<h2>{esc(e["title"])}</h2>')
        o.append(f'<div><span class="state {state}">{esc(outcome)}</span></div>')
        o.append("</header>")
        rows = facts_rows(e)
        if rows:
            o.append('<dl class="facts">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>")
        items = media_items(e, media_prefix)
        if items:
            o.append('<div class="gallery">')
            for kind, src, caption in items:
                wide = " wide" if kind == "video" else ""
                if kind == "video":
                    o.append(f'<figure class="wide"><video controls preload="metadata" src="{esc(src)}"></video><figcaption>{esc(caption)}</figcaption></figure>')
                else:
                    o.append(f'<figure{wide}><img loading="lazy" src="{esc(src)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>')
            o.append("</div>")
        if e["id"] == "gripper-milestones":
            o.append('<div class="resultstable">' + esc("\n".join(e["facts"]["results_table"])) + "</div>")
        o.append(f'<p class="prose"><strong>Produced by.</strong> {esc(e["produced_by"])}</p>')
        o.append(f'<p class="prose"><strong>Code.</strong> {esc(e["code"])}</p>')
        o.append("</article>")
    o.append("</section>")

    # standing
    o.append('<section><div class="eyebrow">Where things stand</div><h2>What the ledger says, read as one</h2><ul class="standing">')
    o.append("<li><strong>The humanoid pipeline works end to end</strong> on the offline planner: 985 structurally valid clips, five README demos, twelve release requirements passing. "
             "The live-model runs (347 with gpt-5.6) exist but are a minority of the archive.</li>")
    o.append("<li><strong>The VLM grader is not yet an instrument.</strong> Tony's 10f campaign put the number on it: Matthews correlation +0.054 over 1,334 clips. "
             "Every acceptance claim that leans on the grader inherits that.</li>")
    o.append("<li><strong>Arbitrary robots reach and sweep; most cannot yet grasp.</strong> 13 zoo clips replay the same words on seven bodies, but only 31 of 151 and then 78 of 241 traces were accepted, "
             "and the refusals are the honest kind: too wide, out of reach, not lifted. The September measured-grasp work targets exactly those.</li>")
    o.append("<li><strong>The gripper is the one embodiment with a physically valid grasp on record:</strong> 12 of 13 placements under computed torque with sub-millimetre penetration, "
             "9 of 12 with the wrist camera as the only percept. It lives on a branch, so this ledger carries its recordings until PR 19 lands.</li>")
    o.append("<li><strong>Nothing here can be regenerated byte-for-byte.</strong> The code moved on under every set; the manifest records which line of history each ran on so a re-run can be compared, not matched.</li>")
    o.append("</ul></section>")
    o.append("</main>")
    return "\n".join(o) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "index.html")
    ap.add_argument("--media-prefix", default="", help="prefix for media paths in a published copy, e.g. media/")
    a = ap.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(manifest, a.media_prefix), encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
