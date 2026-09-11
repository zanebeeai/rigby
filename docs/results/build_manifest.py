"""Build the results manifest: every place a result was ever written, what is
there, how it was produced, and copies of the media worth keeping.

Results in this repository are deliberately not committed (`*/results/`,
`artifacts-v2/`, `docs/media/` under any-robot are all ignored) because a
script regenerates them. What that policy lost was the *record*: which runs
exist, on which code, and which of them mattered. This script writes that
record, from the working copy as it stands, into two files beside it:

    docs/results/manifest.json          machine-readable, one entry per set
    docs/results/RESULTS-MANIFEST.md    the same, for people

and copies the substantial media (GIFs, WebM recordings, filmstrips, the
JSON that describes them) into `docs/results/media/<set>/`, each with its
SHA-256 recorded in the manifest so a copy can be checked against the
original.

The sets it knows about are listed in `SETS` below, in the order the work
happened. Some live in tracked paths, some in the untracked residue that the
27 Aug tier split left behind (`rigby-poc/`, `rigby-generalized-urdf/`,
`rigby-mjco-sim/`), and one is read from a git ref because it was never on
`main`. A set whose path is absent on this machine is reported as absent, not
skipped silently.

Run from the repository root:

    uv run python docs/results/build_manifest.py
    uv run python docs/results/build_manifest.py --no-copy   # manifest only
    uv run python docs/results/build_manifest.py --sets gripper-milestones   # refresh one set
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MEDIA_OUT = HERE / "media"

MEDIA_EXT = {".gif", ".webm", ".mp4", ".png", ".jpg", ".jpeg", ".glb"}
SKIP_DIRS = {".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", ".git"}


# --------------------------------------------------------------------------
# helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def walk(path: Path) -> dict:
    """Count files and bytes, find the date range and the media, in one pass."""
    files = 0
    size = 0
    oldest = None
    newest = None
    media: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            files += 1
            size += st.st_size
            day = dt.datetime.fromtimestamp(st.st_mtime, dt.UTC).date()
            oldest = day if oldest is None or day < oldest else oldest
            newest = day if newest is None or day > newest else newest
            if p.suffix.lower() in MEDIA_EXT:
                media.append({"path": p.relative_to(ROOT).as_posix(), "bytes": st.st_size})
    return {
        "files": files,
        "bytes": size,
        "oldest": oldest.isoformat() if oldest else None,
        "newest": newest.isoformat() if newest else None,
        "media_count": len(media),
        "media_by_ext": dict(collections.Counter(Path(m["path"]).suffix.lower() for m in media)),
        "_media": media,
    }


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def copy_media(set_id: str, src: Path, name: str | None = None, data: bytes | None = None) -> dict:
    """Copy one file into docs/results/media/<set_id>/ and describe it."""
    name = name or src.name
    dest = MEDIA_OUT / set_id / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = src.read_bytes()
    if not dest.exists() or dest.read_bytes() != data:
        dest.write_bytes(data)
    return {
        "file": dest.relative_to(HERE).as_posix(),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "source": src.as_posix() if data is None or src.exists() else str(src),
    }


# --------------------------------------------------------------------------
# per-set analysers. Each returns a dict of headline facts for the manifest.


def analyse_any_robot_traces(path: Path) -> dict:
    kinds = collections.Counter()
    accepted = collections.Counter()
    failures = collections.Counter()
    robots = collections.Counter()
    created = []
    base_trees = collections.Counter()
    n = 0
    for run in sorted(path.iterdir()):
        trace = load_json(run / "trace.json") if run.is_dir() else None
        if not isinstance(trace, dict):
            continue
        n += 1
        kinds[trace.get("kind")] += 1
        accepted[bool(trace.get("accepted"))] += 1
        failure = trace.get("failure") or {}
        if failure.get("code"):
            failures[failure["code"]] += 1
        robots[run.name.split("--")[0]] += 1
        if trace.get("created_at"):
            created.append(trace["created_at"][:10])
        base = (trace.get("provenance") or {}).get("base_tree") or {}
        if base.get("sha256"):
            base_trees[f"{base.get('package')}@{base['sha256'][:12]}"] += 1
    other = [p.name for p in path.iterdir() if p.is_file()]
    return {
        "traces": n,
        "kinds": dict(kinds),
        "accepted": accepted.get(True, 0),
        "rejected": accepted.get(False, 0),
        "failure_codes": dict(failures.most_common()),
        "robots": dict(robots.most_common()),
        "created": [min(created), max(created)] if created else None,
        "code_base_trees": dict(base_trees),
        "top_level_files": other,
    }


def analyse_humanoid_runs(path: Path) -> dict:
    numbered = [p for p in path.iterdir() if p.is_dir() and p.name[:6].isdigit()]
    named = sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name[:6].isdigit())
    planners = collections.Counter()
    compilers = collections.Counter()
    accepted = collections.Counter()
    with_glb = 0
    with_evidence = 0
    for run in numbered:
        prov = load_json(run / "provenance.json") or {}
        planners[f"{prov.get('planner_provider')}/{prov.get('planner_model')}"] += 1
        compilers[prov.get("compiler_version")] += 1
        metrics = load_json(run / "metrics.json") or {}
        # Early runs carry `structural_valid`; later ones `accepted`.
        verdict = metrics.get("accepted", metrics.get("structural_valid"))
        if verdict is not None:
            accepted[bool(verdict)] += 1
        if (run / "animation.glb").exists():
            with_glb += 1
        if (run / "evidence-manifest.json").exists():
            with_evidence += 1
    ids = sorted(p.name for p in numbered)
    acceptance = path / "acceptance-runs"
    audit = load_json(path / "autonomous-goal-audit.json") or {}
    return {
        "numbered_runs": len(numbered),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "planner_models": dict(planners.most_common()),
        "compiler_versions": dict(compilers.most_common()),
        "accepted_or_structural_valid": accepted.get(True, 0),
        "rejected_or_structural_invalid": accepted.get(False, 0),
        "runs_without_verdict": len(numbered) - accepted.get(True, 0) - accepted.get(False, 0),
        "runs_with_glb": with_glb,
        "runs_with_evidence_pngs": with_evidence,
        "acceptance_runs": sorted(p.name for p in acceptance.iterdir()) if acceptance.is_dir() else [],
        "autonomous_goal_audit": {
            "status": audit.get("status"),
            "gates": {g.get("gate"): g.get("status") for g in audit.get("gates", [])},
        }
        if audit
        else None,
        "named_sets": named,
    }


def analyse_artifacts_v2(path: Path) -> dict:
    manifest = load_json(path / "release-evidence" / "manifest.json") or []
    return {
        "top_level": sorted(p.name for p in path.iterdir()),
        "release_evidence": {m.get("requirement_id"): m.get("status") for m in manifest}
        if isinstance(manifest, list)
        else None,
        "benchmarks": sorted(p.name for p in (path / "benchmarks").iterdir())
        if (path / "benchmarks").is_dir()
        else [],
    }


def analyse_mjco_sim(path: Path) -> dict:
    art = path / "artifacts-v2"
    dirs = []
    if art.is_dir():
        for p in sorted(art.iterdir()):
            if p.name.startswith(".pytest"):
                continue
            day = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.UTC).date().isoformat()
            dirs.append(f"{p.name} ({day})")
    docs = sorted(p.name for p in (path / "docs").iterdir()) if (path / "docs").is_dir() else []
    # How much of this tree is in no commit at all.
    unknown = 0
    checked = 0
    for sub in ("src", "tests", "scripts", "visual"):
        for p in (path / sub).rglob("*.py") if (path / sub).is_dir() else []:
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            checked += 1
            blob = subprocess.run(
                ["git", "hash-object", str(p)], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            known = subprocess.run(["git", "cat-file", "-e", blob], cwd=ROOT).returncode == 0
            unknown += 0 if known else 1
    return {
        "artifacts_v2": dirs,
        "docs": docs,
        "python_files_checked": checked,
        "python_files_in_no_commit": unknown,
    }


def analyse_demo_media(path: Path) -> dict:
    manifest = load_json(path / "demo-manifest.json") or {}
    report = load_json(path / "grasp-report.json") or {}
    out = {"demo_manifest_keys": sorted(manifest.keys()) if manifest else None}
    if "demos" in manifest:
        out["demos"] = [
            {k: d.get(k) for k in ("file", "result_id", "prompt", "planner_model", "duration_s")}
            for d in manifest["demos"]
        ]
        out["comparisons"] = [
            {k: d.get(k) for k in ("file", "prompt")} for d in manifest.get("comparisons", [])
        ]
        out["compiler_commit"] = manifest.get("compiler_commit")
        out["generated_on"] = manifest.get("generated_on")
    if "clips" in manifest:
        out["clips"] = [
            {k: c.get(k) for k in ("gif", "robot_id", "prompt", "certified_primitives_used", "tracking_error_m")}
            for c in manifest["clips"]
        ]
    if "attempts" in report:
        attempts = report["attempts"]
        out["grasp_attempts"] = len(attempts)
        out["grasp_certified"] = sum(1 for a in attempts if a.get("certified"))
        out["grasp_by_robot"] = {
            a.get("robot_id"): ("certified" if a.get("certified") else ", ".join(a.get("violations", [])))
            for a in attempts
        }
    return out


def analyse_git_ref(ref: str, prefix: str) -> dict:
    listing = git("ls-tree", "-r", "-l", ref, "--", prefix)
    files = []
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        size = int(meta.split()[3])
        files.append({"path": path, "bytes": size})
    tip = git("log", "-1", "--format=%H %ad %s", "--date=short", ref).strip()
    return {"ref": ref, "tip": tip, "files": len(files), "bytes": sum(f["bytes"] for f in files), "_files": files}


# --------------------------------------------------------------------------
# the sets, in the order the work happened.

SETS = [
    {
        "id": "humanoid-runs-aug07-12",
        "title": "Humanoid result archive (rigby-poc/results)",
        "path": "rigby-poc/results",
        "tracked": False,
        "period": "2026-08-07 to 2026-08-12",
        "produced_by": (
            "The humanoid app (`uv run rigby-humanoid`, then `POST /api/v1/pipeline-runs`, "
            "or the CLI in `rigby_poc`) writes one immutable six-digit directory per compiled "
            "clip: request.json, scene.json, program.json, clip.json, metrics.json, "
            "provenance.json, animation.glb. `acceptance-runs/` holds the blinded gesture "
            "review runs; `autonomous-goal-audit.json` is the gate summary. See the README "
            "inside the directory."
        ),
        "code": (
            "The `rigby-poc` layout that became `humanoid/` on 27 Aug (commit 2d39970). "
            "provenance.json records `compiler_version` but not a commit; the compiler "
            "versions present date the runs to the 0.1.0 to 0.4.0 compilers of early Aug."
        ),
        "analyse": analyse_humanoid_runs,
        "media": None,
    },
    {
        "id": "humanoid-release-evidence",
        "title": "Humanoid v2 release evidence and benchmarks (rigby-poc/artifacts-v2)",
        "path": "rigby-poc/artifacts-v2",
        "tracked": False,
        "period": "2026-08-11 to 2026-08-12",
        "produced_by": (
            "`rigby_v2.release` and `rigby_v2.release_ops` (now under humanoid/src/rigby_v2): "
            "each requirement in `release-evidence/manifest.json` is a JSON artefact with its "
            "SHA-256. The `supported-live-model-authority-canary-*` runs are the 11 Aug live-model "
            "canaries (`run-config.json` names the pipeline id and case ids)."
        ),
        "code": "rigby_v2 as tracked by PR 9 (the v2 runtime), pre-rename.",
        "analyse": analyse_artifacts_v2,
        "media": None,
    },
    {
        "id": "mjco-sim-workspace",
        "title": "MuJoCo production workspace (rigby-mjco-sim)",
        "path": "rigby-mjco-sim",
        "tracked": False,
        "period": "2026-08-12 to 2026-08-18",
        "produced_by": (
            "A standalone workspace (own pyproject and uv.lock) whose README calls it 'the "
            "clean local-first Rigby production workspace'. `artifacts-v2/` holds the button, "
            "drawer, bimanual-grasp and calibration-readiness experiments of 13-17 Aug; "
            "`docs/` holds the skinned-avatar SDF probes and the visual-parity audit; "
            "`visual/generate_demo_media.py` renders the viewer traces."
        ),
        "code": (
            "Its `rigby_v2` package is the ancestor of what PR 9 tracked, but a large part of "
            "this tree exists in no commit (counted below). It is not in git history at all."
        ),
        "analyse": analyse_mjco_sim,
        "media": None,
    },
    {
        "id": "humanoid-demo-gifs",
        "title": "Humanoid README demo GIFs (humanoid/docs/media)",
        "path": "humanoid/docs/media",
        "tracked": True,
        "period": "2026-08-27",
        "produced_by": (
            "`cd humanoid; uv run python -m evals.render_demo_gif <result_id> docs/media/<name>.gif` "
            "against a running app on :8000 (8 FPS sample, 480 px ego and orbit panels, 96-colour "
            "GIF via FFmpeg). `render_comparison_gif.py` made the before/after strip. "
            "`demo-manifest.json` records the result id, prompt, planner and clip length per GIF."
        ),
        "code": "demo-manifest.json says compiler commit d0dbc86 (27 Aug, feat/strike-torso-cross merge).",
        "analyse": analyse_demo_media,
        "media": "in-place",
    },
    {
        "id": "humanoid-calibration-evidence",
        "title": "Judge calibration evidence (humanoid/docs/evidence, humanoid/evals/corpus)",
        "path": "humanoid/docs/evidence",
        "tracked": True,
        "period": "2026-08-16 to 2026-09-07",
        "produced_by": (
            "`10f-calibration-report.md` and `frozen-judge-calibration.json` come from the "
            "calibration campaign driver (`humanoid/evals`, PRs 4, 6, 8 and the 10f2 campaign "
            "merge 880735c). The golden corpus under `humanoid/evals/corpus/cases` is blessed by "
            "the corpus CLI; `expected.json` per case carries the digest of the blessed clip."
        ),
        "code": "Tracked on main; every change is in git history.",
        "analyse": None,
        "media": None,
    },
    {
        "id": "any-robot-zoo-aug24-27",
        "title": "Any-robot zoo trials, first pass (rigby-generalized-urdf/results)",
        "path": "rigby-generalized-urdf/results",
        "tracked": False,
        "period": "2026-08-24 to 2026-08-27",
        "produced_by": (
            "`uv run python scripts/run_trials.py` (every gripper against every authored "
            "world), the contact probe in `rigby_general.contact`, and plain prompts through "
            "`rigby_general.run`. One `trace.json` per `<robot>--<prompt>` directory."
        ),
        "code": (
            "The `rigby-poc-general` package before it became `any-robot/` (PR 10, PR 12). "
            "trace.json provenance hashes the rigby_v2 base tree it ran against (114 files)."
        ),
        "analyse": analyse_any_robot_traces,
        "media": None,
    },
    {
        "id": "any-robot-zoo-demos",
        "title": "Any-robot zoo demo GIFs and grasp report (rigby-generalized-urdf/docs/media)",
        "path": "rigby-generalized-urdf/docs/media",
        "tracked": False,
        "period": "2026-08-24 to 2026-08-27",
        "produced_by": (
            "`uv run python scripts/build_demos.py` renders the same plain-language prompts on "
            "several zoo robots and writes `demo-manifest.json` with the role-normalized schema "
            "hash beside each clip (one prompt, one meaning, many bodies). "
            "`uv run python scripts/grasp_report.py --render` writes `grasp-report.json`."
        ),
        "code": "rigby-poc-general at PR 10 to PR 12; the manifest does not record a commit.",
        "analyse": analyse_demo_media,
        "media": "copy",
    },
    {
        "id": "any-robot-environments-aug27-28",
        "title": "Any-robot environment trials and IRL robots (any-robot/results)",
        "path": "any-robot/results",
        "tracked": False,
        "period": "2026-08-27 to 2026-08-28",
        "produced_by": (
            "`uv run python scripts/run_trials.py` over the six authored worlds "
            "(bench_jig, desk_bench, far_pallet, pallet_cell, raised_shelf, work_table) after "
            "the tier split, plus contact probes and prompts on the bench robots "
            "(KUKA KR6/LWR, iiwa7, uHand2, EEZYbotARM MK1, Beetlebot, SO-101). "
            "`scripts/build_studio.py` and `scripts/export_viewer.py` render any of them."
        ),
        "code": (
            "any-robot on the feat/environments-and-contact line (PR 12, PR 14, and the six "
            "28 Aug commits merged as PR 21). trace.json provenance hashes the rigby_core base "
            "tree (19 files)."
        ),
        "analyse": analyse_any_robot_traces,
        "media": None,
    },
    {
        "id": "gripper-milestones",
        "title": "Gripper milestones and recordings (Angelo, branch grasp/auto-lift)",
        "ref": "origin/grasp/auto-lift",
        "prefix": "rigby-poc",
        "subpaths": ["rigby-poc/videos", "rigby-poc/milestones"],
        "tracked": "on the branch only",
        "period": "2026-08-27 to 2026-09-09",
        "produced_by": (
            "Each `milestones/runs/<name>.json` is a complete `gripper_clip_v1` recording "
            "(every frame's joint poses, link geometry, block pose, contact forces, penetration) "
            "from the torque-driven gripper controller in `rigby_poc.gripper` and "
            "`rigby_poc.closed_loop`, played by `frontend/public/static/gripper.html?clip=<name>` "
            "under `npm run dev --prefix frontend`. The `.webm` files are screen recordings of "
            "that viewer; the branch commits no recorder script, so the exact capture settings "
            "are Angelo's to state. `RESULTS.txt` is the placed/lift/penetration table."
        ),
        "code": "PR 19 (grasp/auto-lift); rebased copy on the new base is grasp/auto-lift-on-main.",
        "analyse": None,
        "media": "git",
    },
]


# --------------------------------------------------------------------------


def build(copy: bool, only: set[str] | None = None) -> dict:
    """Build every set, or only the named ones with the rest reused from the
    previous manifest.json (the humanoid archive and the mjco-sim workspace
    take minutes to walk, so a one-set refresh must not pay for them)."""
    previous = {}
    if only:
        old = load_json(HERE / "manifest.json") or {}
        previous = {e["id"]: e for e in old.get("sets", [])}
    entries = []
    for spec in SETS:
        if only and spec["id"] not in only:
            if spec["id"] in previous:
                entries.append(previous[spec["id"]])
                continue
        entry = {
            k: v
            for k, v in spec.items()
            if k not in ("analyse", "media", "path", "ref", "prefix", "subpaths")
        }
        media_files = []
        if "ref" in spec:
            entry["location"] = f"{spec['ref']}:{spec['prefix']}"
            try:
                info = analyse_git_ref(spec["ref"], spec["prefix"])
            except subprocess.CalledProcessError:
                entry["present"] = False
                entries.append(entry)
                continue
            files = [f for f in info.pop("_files") if any(f["path"].startswith(s) for s in spec["subpaths"])]
            entry["present"] = True
            entry["facts"] = {"tip": info["tip"], "files": len(files), "bytes": sum(f["bytes"] for f in files)}
            table = git_bytes("show", f"{spec['ref']}:{spec['prefix']}/milestones/RESULTS.txt").decode()
            entry["facts"]["results_table"] = table.strip().splitlines()
            if copy:
                for f in files:
                    p = Path(f["path"])
                    if p.suffix.lower() in MEDIA_EXT | {".txt", ".md"} and "runs/" not in f["path"]:
                        data = git_bytes("show", f"{spec['ref']}:{f['path']}")
                        # videos/README.md and milestones/README.md must not collide.
                        name = "--".join(p.relative_to(spec["prefix"]).parts)
                        media_files.append(copy_media(spec["id"], p, name, data))
        else:
            path = ROOT / spec["path"]
            entry["location"] = spec["path"]
            entry["present"] = path.is_dir()
            if not entry["present"]:
                entries.append(entry)
                continue
            stats = walk(path)
            media = stats.pop("_media")
            entry["stats"] = stats
            if spec["analyse"]:
                entry["facts"] = spec["analyse"](path)
            if spec["media"] == "in-place":
                media_files = [
                    {"file": m["path"], "bytes": m["bytes"], "sha256": sha256_file(ROOT / m["path"]), "source": m["path"]}
                    for m in media
                ]
            elif spec["media"] == "copy" and copy:
                for m in media:
                    media_files.append(copy_media(spec["id"], ROOT / m["path"]))
                for name in ("demo-manifest.json", "grasp-report.json"):
                    if (path / name).exists():
                        media_files.append(copy_media(spec["id"], path / name))
        entry["media"] = media_files
        entries.append(entry)
    return {
        "schema_version": "1.0",
        "generated_on": dt.date.today().isoformat(),
        "generated_from_commit": git("rev-parse", "--short", "HEAD").strip(),
        "sets": entries,
    }


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.0f} kB"


def tracked_word(e: dict) -> str:
    t = e["tracked"]
    return "tracked" if t is True else "untracked" if t is False else str(t)


def render_md(manifest: dict) -> str:
    out = []
    out.append("# Results manifest\n")
    out.append(
        f"Generated {manifest['generated_on']} from commit `{manifest['generated_from_commit']}` by "
        "`docs/results/build_manifest.py`. One section per place results were ever written, in the "
        "order the work happened. Locations marked *untracked* are the residue the 27 Aug tier split "
        "left in the checkout; they exist on the machine that ran the experiments and nowhere else, "
        "which is the reason this file exists. Media copied into `docs/results/media/` carries its "
        "SHA-256 so a copy can be checked against its source.\n"
    )
    out.append("| set | location | tracked | period | files | size | media |")
    out.append("|---|---|---|---|---|---|---|")
    for e in manifest["sets"]:
        if not e.get("present"):
            out.append(f"| {e['id']} | `{e['location']}` | {tracked_word(e)} | {e['period']} | absent | | |")
            continue
        st = e.get("stats") or {"files": e["facts"]["files"], "bytes": e["facts"]["bytes"], "media_count": len(e["media"])}
        out.append(
            f"| {e['id']} | `{e['location']}` | {tracked_word(e)} | {e['period']} | {st['files']} | "
            f"{fmt_bytes(st['bytes'])} | {st.get('media_count', len(e['media']))} |"
        )
    out.append("")
    for e in manifest["sets"]:
        out.append(f"## {e['title']}\n")
        out.append(f"- **Location:** `{e['location']}` ({tracked_word(e)})")
        out.append(f"- **Period:** {e['period']}")
        out.append(f"- **Produced by:** {e['produced_by']}")
        out.append(f"- **Code:** {e['code']}")
        if not e.get("present"):
            out.append("- **Status:** absent on this machine\n")
            continue
        if "stats" in e:
            s = e["stats"]
            out.append(
                f"- **On disk:** {s['files']} files, {fmt_bytes(s['bytes'])}, modified {s['oldest']} to {s['newest']}"
                + (f", media {s['media_by_ext']}" if s["media_count"] else "")
            )
        facts = e.get("facts") or {}
        if facts:
            out.append("- **Facts:**")
            for k, v in facts.items():
                if k == "results_table":
                    out.append("  - results table:\n")
                    out.append("    ```text")
                    out.extend("    " + line for line in v)
                    out.append("    ```")
                    continue
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                    if len(v) > 900:
                        v = v[:900] + " ..."
                out.append(f"  - {k}: {v}")
        if e.get("media"):
            out.append("- **Media:**\n")
            out.append("  | file | size | sha256 |")
            out.append("  |---|---|---|")
            for m in e["media"]:
                out.append(f"  | `{m['file']}` | {fmt_bytes(m['bytes'])} | `{m['sha256'][:16]}` |")
        out.append("")
    out.append("## Regenerating\n")
    out.append(
        "Every set above except the recordings can be regenerated from the tracked code: "
        "`run_trials.py`, `build_demos.py` and `grasp_report.py` in `any-robot/scripts`, "
        "`render_demo_gif.py` in `humanoid/evals`, and the humanoid app for the archive. "
        "Regenerated results will not be byte-identical to the originals where the code moved on; "
        "the manifest's `code` field says which line of history each set came from.\n"
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-copy", action="store_true", help="write the manifest without copying media")
    parser.add_argument("--sets", nargs="*", help="rebuild only these set ids; the rest are reused from manifest.json")
    args = parser.parse_args()
    manifest = build(copy=not args.no_copy, only=set(args.sets) if args.sets else None)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (HERE / "RESULTS-MANIFEST.md").write_text(render_md(manifest) + "\n", encoding="utf-8", newline="\n")
    for e in manifest["sets"]:
        state = "absent" if not e.get("present") else (f"{e['stats']['files']} files" if "stats" in e else f"{e['facts']['files']} files")
        print(f"{e['id']:36s} {state}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
